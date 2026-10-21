#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight for a NATIVE Windows TAO host and its venv (no WSL, no container).

Every check here exists because something shipped broken without it. The
expensive ones are deliberately NOT cheap queries:

* GPU health is probed with a real allocation plus a bf16 matmul, because
  ``torch.cuda.is_available()``, ``get_device_name()`` and ``nvidia-smi`` can all
  pass on a Windows box where a four-byte allocation segfaults. Query health is
  not allocation health.
* ``torch`` is checked for the EXACT ``2.13.0+cu130`` line and a cuDNN backend
  version of at least 92000, because that is the only line shipping cuDNN 9.20
  and the framework's attention gate is ``CUDNN_MIN_BACKEND_VERSION = 92000``.
  Below it the cuDNN backend silently disables itself and falls back to NATTEN,
  which has no Windows build at all.
* FFmpeg is checked for BOTH major lines. torchcodec is happy on 7.1
  (``avformat-61``); PyNvVideoCodec 2.2.0 links ``avformat-62``, i.e. 8.x. One
  alone breaks the other.
* DLL directory registration is checked as a ``.pth`` in site-packages, not as
  ``PATH``. Since Python 3.8, Windows extension modules do not resolve dependent
  DLLs from ``PATH``; ``os.add_dll_directory()`` has to run at interpreter
  startup, and only a ``.pth`` covers every process.

Usage
-----
    # from anywhere: re-execs itself inside the venv for the in-venv half
    python preflight_cosmos3_windows_host.py --venv C:\\cosmos-win\\venv-tao [--json]

    # already inside the venv
    <venv>\\Scripts\\python.exe preflight_cosmos3_windows_host.py

    # host layer only (no torch import, no CUDA context, safe next to a live job)
    python preflight_cosmos3_windows_host.py --host-only

Point ``--venv`` at the FULL five-action environment. A training-only venv has
neither ``cosmos_rl`` nor the five ``cosmos-framework-*.exe`` entrypoints, so it
fails ``entrypoints`` and four ``import:*`` checks by design, not by accident.

Exit status = number of FAIL checks (0 means go). WARN never fails the gate.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

REQUIRED_TORCH = "2.13.0+cu130"
REQUIRED_CUDNN = 92000
REQUIRED_PYTHON = (3, 12)
ENTRYPOINTS = (
    "cosmos-framework-train",
    "cosmos-framework-export",
    "cosmos-framework-evaluate",
    "cosmos-framework-inference",
    "cosmos-framework-inference-microservice",
)

# Where to hunt for FFmpeg. Override with --ffmpeg-root (repeatable) or the
# TAO_WIN_FFMPEG_ROOT environment variable (os.pathsep-separated).
FFMPEG_ROOTS = (
    r"C:\cosmos-win",
    r"C:\opt",
    r"C:\Program Files",
    "C:\\",
)
# Globs applied under each root. The first entry is the BtbN zip's real,
# unrenamed shape: ffmpeg-n8.1-latest-win64-gpl-shared-8.1\bin. The others catch
# hosts that renamed the directory or extracted the zip one level deeper.
FFMPEG_DIR_GLOBS = (
    "ffmpeg-n*-shared-*/bin",
    "ffmpeg*/bin",
    "ffmpeg*/ffmpeg*/bin",
)
# Last-resort literals, checked after the globs. The first two are the actual
# directory names on the validated host; the last two are the tidy names older
# revisions of this file assumed and that exist on no machine we have seen.
FFMPEG_FALLBACK_DIRS = (
    r"C:\cosmos-win\ffmpeg-n8.1-latest-win64-gpl-shared-8.1\bin",
    r"C:\cosmos-win\ffmpeg-n7.1-latest-win64-gpl-shared-7.1\bin",
    r"C:\ffmpeg8\bin",
    r"C:\ffmpeg\bin",
)


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def add(self, name: str, status: str, detail: str, fix: str = "") -> None:
        entry = {"check": name, "status": status, "detail": detail}
        if fix:
            entry["fix"] = fix
        self.checks.append(entry)

    @property
    def failures(self) -> int:
        return sum(1 for c in self.checks if c["status"] == FAIL)

    def emit(self, as_json: bool) -> int:
        if as_json:
            print(json.dumps({"checks": self.checks, "failures": self.failures}, indent=2))
        else:
            width = max((len(c["check"]) for c in self.checks), default=10)
            for c in self.checks:
                print(f"{c['status']:<4} {c['check']:<{width}}  {c['detail']}")
                if c.get("fix") and c["status"] != PASS:
                    print(f"       fix: {c['fix']}")
            print(f"\n{self.failures} failure(s)")
        return self.failures


# ---------------------------------------------------------------------------
# Host-level checks (do not need the venv)
# ---------------------------------------------------------------------------

def check_platform(report: Report) -> None:
    if os.name != "nt":
        report.add("os", FAIL, f"os.name={os.name!r}; this platform is native Windows only",
                   "Use tao-run-on-virtualenv on POSIX hosts.")
        return
    report.add("os", PASS, f"{platform.system()} {platform.release()} build {platform.version()}")
    machine = platform.machine()
    normalized = machine.lower()
    if normalized in ("amd64", "x86_64"):
        report.add("arch", PASS, machine)
    elif normalized in ("arm64", "aarch64"):
        # Deliberately WARN, not FAIL. "No CUDA torch on Windows ARM64" stopped
        # being true: NVIDIA ships CUDA-enabled win_arm64 wheels on
        # https://pypi.nvidia.com/nvtorch_oot_nightly (e.g.
        # torch-2.14.0.dev20260728+cu134-cp313-cp313-win_arm64.whl). What is
        # still true is that THIS recipe cannot run there unchanged.
        report.add(
            "arch", WARN,
            f"{machine} (Windows on ARM); this recipe is pinned to win_amd64 + cp312",
            "NVIDIA does publish CUDA-enabled win_arm64 torch wheels on "
            "https://pypi.nvidia.com/nvtorch_oot_nightly, so ARM64 is not a hard wall "
            "at the torch layer -- but it needs a DIFFERENT pin set (cp313 + cu134, not "
            "the cp312 + cu130 pins every check below assumes), and torchcodec, "
            "opencv-python-headless, pyarrow, triton and vLLM have no win_arm64 wheels "
            "today. NOTHING in this skill has been validated on ARM64. Treat a green "
            "run here as unproven.",
        )
    else:
        report.add(
            "arch", WARN, f"{machine} is neither x64 nor ARM64",
            "Every pinned wheel in this recipe is win_amd64 + cp312. Expect pip to fail "
            "to find distributions rather than to install something subtly wrong.",
        )


def check_long_paths(report: Report) -> None:
    """LongPathsEnabled=0 makes pip die mid-install at 262 characters."""
    try:
        import winreg  # noqa: PLC0415  (Windows-only import)

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
    except (ImportError, OSError) as exc:
        report.add("long_paths", WARN, f"could not read LongPathsEnabled: {exc}")
        return
    if int(value) == 1:
        report.add("long_paths", PASS, "LongPathsEnabled=1")
    else:
        report.add(
            "long_paths", FAIL, "LongPathsEnabled=0; MAX_PATH is 260",
            "Set HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem\\LongPathsEnabled=1 "
            "(DWORD) BEFORE any pip step, then start a new process.",
        )


def check_vc_runtime(report: Report) -> None:
    """Server Core ships no VC++ redistributable; torch's c10.dll needs it."""
    missing = []
    for dll in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"):
        try:
            ctypes.WinDLL(dll)
        except OSError:
            missing.append(dll)
    if missing:
        report.add(
            "vc_runtime", FAIL, f"missing: {', '.join(missing)}",
            "Install the x64 Visual C++ redistributable (vc_redist.x64.exe /quiet "
            "/norestart). Without it `import torch` dies with WinError 126 on c10.dll.",
        )
    else:
        report.add("vc_runtime", PASS, "vcruntime140 / msvcp140 resolve")


def check_driver(report: Report) -> None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        report.add("driver", FAIL, "nvidia-smi not on PATH",
                   "Install an NVIDIA driver at CUDA 13.0 or newer, WDDM mode.")
        return
    try:
        out = subprocess.run(
            [smi, "--query-gpu=driver_version,name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        report.add("driver", FAIL, f"nvidia-smi failed: {exc}")
        return
    if out.returncode != 0:
        report.add("driver", FAIL, f"nvidia-smi exit {out.returncode}: {out.stderr.strip()}")
        return
    gpus = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    report.add("driver", PASS, "; ".join(gpus))
    if len(gpus) > 1:
        report.add(
            "gpu_count", WARN,
            f"{len(gpus)} GPUs visible, but only ONE is usable",
            "Multi-GPU is impossible with the STOCK torch Windows wheels: gloo cannot run "
            "a collective on a CUDA tensor, and the stock wheels are built with "
            "USE_C10D_NCCL off, so init_process_group('nccl') raises regardless of what "
            "is installed on the box. The run dies with 0xC0000005 and no traceback. Pin "
            "a single device with --gpu-ids. (A community NCCL port for Windows does "
            "exist -- github.com/SystemPanic/nccl-windows -- and vLLM can load its "
            "nccl.dll directly via VLLM_NCCL_SO_PATH because it bypasses torch's process "
            "group. That path is UNTESTED here and does not make torch distributed work "
            "without rebuilding torch.) **Multi-node is not supported**.",
        )
    else:
        report.add("gpu_count", PASS, "1 GPU visible")


def _ffmpeg_candidate_dirs(hints: list[str], roots: list[str]) -> list[Path]:
    """Every plausible FFmpeg ``bin`` directory, most-specific first.

    Do NOT hardcode tidy names. BtbN's release zips -- the only Windows builds
    that actually ship the shared DLLs -- extract to directories whose name
    carries the version twice, e.g.::

        ffmpeg-n8.1-latest-win64-gpl-shared-8.1\\bin
        ffmpeg-n7.1-latest-win64-gpl-shared-7.1\\bin

    and nobody renames them. An earlier revision of this check looked only in
    ``C:\\ffmpeg8\\bin`` and ``C:\\ffmpeg\\bin``, so it reported
    ``found avformat majors none`` on a host where both lines were installed and
    working. Glob for the real shape under each root instead.
    """
    candidates: list[Path] = [Path(h) for h in hints if h]
    for root in roots:
        base = Path(root)
        if not base.is_dir():
            continue
        for pattern in FFMPEG_DIR_GLOBS:
            try:
                candidates.extend(sorted(base.glob(pattern)))
            except OSError:
                continue
    candidates += [Path(p) for p in FFMPEG_FALLBACK_DIRS]

    ordered: list[Path] = []
    seen: set[str] = set()
    for directory in candidates:
        key = str(directory).lower().rstrip("\\/")
        if key in seen:
            continue
        seen.add(key)
        ordered.append(directory)
    return ordered


def check_ffmpeg(report: Report, hints: list[str], roots: list[str] | None = None) -> None:
    """Both major lines must be present, as SHARED builds."""
    roots = list(roots) if roots else list(FFMPEG_ROOTS)
    candidates = _ffmpeg_candidate_dirs(hints, roots)
    found: dict[str, Path] = {}
    searched = 0
    for directory in candidates:
        if not directory.is_dir():
            continue
        searched += 1
        names = {p.name.lower() for p in directory.glob("avformat-*.dll")}
        for name in names:
            match = re.match(r"avformat-(\d+)\.dll", name)
            if match:
                found.setdefault(match.group(1), directory)
    if "62" in found and "61" in found:
        report.add("ffmpeg", PASS,
                   f"avformat-62 at {found['62']}, avformat-61 at {found['61']}")
        return
    detail = (f"found avformat majors {sorted(found) or 'none'} "
              f"in {searched} existing dir(s) under roots {roots}")
    report.add(
        "ffmpeg", FAIL, detail,
        "Install BOTH shared builds: 8.1 (avformat-62, needed by PyNvVideoCodec) and "
        "7.1 (avformat-61, needed by torchcodec). Register 8.1 first. The "
        "commonly-downloaded 'essentials' build ships only .exe files and no DLLs, so it "
        "cannot work. Extract BtbN's *-win64-gpl-shared-* zips under one of the roots "
        "above WITHOUT renaming them, or point at them with --ffmpeg-root / --ffmpeg-dir.",
    )


# ---------------------------------------------------------------------------
# In-venv checks
# ---------------------------------------------------------------------------

def check_python(report: Report) -> None:
    version = sys.version_info[:2]
    detail = f"{platform.python_version()} ({sys.executable})"
    if version == REQUIRED_PYTHON:
        report.add("python", PASS, detail)
    else:
        report.add("python", FAIL, detail,
                   "Every wheel in the stack is cp312. If the python.org MSI is blocked "
                   "by policy (exit 1625) use a python-build-standalone tarball.")
    if sys.maxsize <= 2 ** 32:
        report.add("python_bits", FAIL, "32-bit interpreter")
    else:
        report.add("python_bits", PASS, "64-bit")


def check_dll_directory_pth(report: Report) -> None:
    site_packages = sysconfig.get_paths().get("purelib", "")
    pths = list(Path(site_packages).glob("*.pth")) if site_packages else []
    registrars = [p for p in pths if "add_dll_directory" in _safe_read(p)]
    if registrars:
        report.add("dll_dir_pth", PASS,
                   ", ".join(p.name for p in registrars))
    else:
        report.add(
            "dll_dir_pth", FAIL, f"no .pth calling os.add_dll_directory in {site_packages}",
            "Since Python 3.8 Windows extension modules do NOT resolve dependent DLLs "
            "from PATH. Drop a ONE-LINE .pth in site-packages registering the CUDA 12 "
            "runtime, FFmpeg 8.1 and FFmpeg 7.1 directories, in that order.",
        )


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def check_torch(report: Report) -> None:
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # ImportError, OSError(WinError 126), ...
        report.add("torch", FAIL, f"{type(exc).__name__}: {exc}",
                   "WinError 126 on c10.dll means the VC++ redistributable is missing.")
        return
    if torch.__version__ == REQUIRED_TORCH:
        report.add("torch", PASS, torch.__version__)
    else:
        report.add(
            "torch", FAIL, f"{torch.__version__} (need exactly {REQUIRED_TORCH})",
            "2.13.0+cu130 is the only line shipping cuDNN 9.20. Below the framework's "
            "CUDNN_MIN_BACKEND_VERSION = 92000 gate the cuDNN attention backend "
            "disables itself and falls back to NATTEN, which has no Windows build.",
        )
    try:
        cudnn_version = torch.backends.cudnn.version()
    except Exception as exc:
        report.add("cudnn", WARN, f"could not query cuDNN: {exc}")
        return
    if cudnn_version and cudnn_version >= REQUIRED_CUDNN:
        report.add("cudnn", PASS, str(cudnn_version))
    else:
        report.add("cudnn", FAIL, f"{cudnn_version} (need >= {REQUIRED_CUDNN})")


def check_gpu_allocation(report: Report) -> None:
    """Allocate for real. Query health is not allocation health."""
    try:
        import torch  # noqa: PLC0415
    except Exception:
        report.add("gpu_alloc", FAIL, "torch is not importable, so no allocation probe ran")
        return
    if not torch.cuda.is_available():
        report.add("gpu_alloc", FAIL, "torch.cuda.is_available() is False")
        return
    try:
        device = torch.device("cuda:0")
        a = torch.randn(512, 512, device=device, dtype=torch.bfloat16)
        b = torch.randn(512, 512, device=device, dtype=torch.bfloat16)
        c = (a @ b).float().sum().item()
        free, total = torch.cuda.mem_get_info(device)
        del a, b
        torch.cuda.empty_cache()
    except Exception as exc:
        report.add(
            "gpu_alloc", FAIL, f"{type(exc).__name__}: {exc}",
            "is_available(), get_device_name() and nvidia-smi can all pass while a real "
            "allocation segfaults. This probe is the one that matters.",
        )
        return
    if c != c:  # NaN
        report.add("gpu_alloc", FAIL, "bf16 matmul returned NaN")
        return
    report.add("gpu_alloc", PASS,
               f"bf16 matmul OK on {torch.cuda.get_device_name(0)}; "
               f"{free // (1024 ** 3)} GiB free of {total // (1024 ** 3)} GiB")


def check_imports(report: Report) -> None:
    required = {
        "torchcodec": "video decoding for evaluate/inference (set video_decoder='torchcodec')",
        "PyNvVideoCodec": "NVDEC decoder; needs FFmpeg 8.x and cudart64_12.dll",
        "transformers": "generation path for evaluate and inference",
        "cosmos_framework": "training entrypoint package",
        "cosmos_rl": "export / evaluate / inference / microservice entrypoints",
        "nvidia_tao_core": "TAO integration layer",
        "cv2": "opencv-python-headless",
        "orjson": "TAO status serialization",
        "flask": "microservice app import",
        "pytest": "imported at CONFIG LOAD by tao_video_sft_test.py -- a runtime dep",
    }
    for module, why in required.items():
        try:
            __import__(module)
        except Exception as exc:
            report.add(f"import:{module}", FAIL, f"{type(exc).__name__}: {exc}", why)
        else:
            report.add(f"import:{module}", PASS, why)


def check_cudart12(report: Report) -> None:
    """PyNvVideoCodec_130.pyd is a CUDA 13 build that links cudart64_12.dll."""
    try:
        ctypes.WinDLL("cudart64_12.dll")
    except OSError:
        report.add(
            "cudart64_12", WARN, "cudart64_12.dll does not resolve",
            "pip install nvidia-cuda-runtime-cu12 and register its bin directory in the "
            "site-packages .pth. It is NOT redundant with torch's bundled CUDA 13 "
            "runtime: PyNvVideoCodec_130.pyd links the CUDA 12 runtime DLL.",
        )
    else:
        report.add("cudart64_12", PASS, "resolves")


def check_entrypoints(report: Report, venv: Path | None) -> None:
    scripts_dir = venv / "Scripts" if venv else Path(sys.executable).parent
    missing = [name for name in ENTRYPOINTS
               if not (scripts_dir / f"{name}.exe").is_file()]
    if missing:
        report.add("entrypoints", FAIL, f"missing from {scripts_dir}: {', '.join(missing)}",
                   "Four of the five come from cosmos-rl; see cosmos3-windows-x64-host-setup.md.")
    else:
        report.add("entrypoints", PASS, f"all five present in {scripts_dir}")


def check_tao_core_pwd_guard(report: Report) -> None:
    """slurm_handler.py imports POSIX-only `pwd` unconditionally upstream."""
    try:
        import nvidia_tao_core  # noqa: PLC0415
    except Exception as exc:
        report.add("tao_core_pwd_guard", FAIL, f"nvidia_tao_core not importable: {exc}")
        return
    root = Path(nvidia_tao_core.__file__).parent
    handler = root / "microservices/handlers/execution_handlers/slurm_handler.py"
    if not handler.is_file():
        report.add("tao_core_pwd_guard", WARN, f"{handler} not found; skipping")
        return
    text = _safe_read(handler)
    guarded = re.search(r"try:\s*\n\s*import pwd", text) or "ImportError" in text
    if "import pwd" not in text:
        report.add("tao_core_pwd_guard", PASS, "no `import pwd` in slurm_handler.py")
    elif guarded:
        report.add("tao_core_pwd_guard", PASS, "`import pwd` is guarded")
    else:
        report.add(
            "tao_core_pwd_guard", FAIL, f"unguarded `import pwd` in {handler}",
            "Apply the one-line try/except ImportError guard locally. The published "
            "commit carrying it is not reachable from public tao-core.",
        )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_host_checks(report: Report, ffmpeg_hints: list[str],
                    ffmpeg_roots: list[str] | None = None) -> None:
    check_platform(report)
    if os.name != "nt":
        return
    check_long_paths(report)
    check_vc_runtime(report)
    check_driver(report)
    check_ffmpeg(report, ffmpeg_hints, ffmpeg_roots)


def run_venv_checks(report: Report, venv: Path | None) -> None:
    check_python(report)
    check_dll_directory_pth(report)
    check_torch(report)
    check_gpu_allocation(report)
    check_cudart12(report)
    check_imports(report)
    check_entrypoints(report, venv)
    check_tao_core_pwd_guard(report)


def _venv_python(venv: Path) -> Path:
    return venv / "Scripts" / "python.exe"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", default=None,
                        help="Venv root. When given and we are not already inside it, "
                             "the in-venv half is re-executed with its interpreter.")
    parser.add_argument("--ffmpeg-dir", action="append", default=[],
                        help="Exact bin directory to search for FFmpeg DLLs (repeatable).")
    parser.add_argument("--ffmpeg-root", action="append", default=[],
                        help="Root to glob for ffmpeg-n*-shared-*/bin (repeatable). "
                             f"Default: {', '.join(FFMPEG_ROOTS)}. Also read from "
                             "TAO_WIN_FFMPEG_ROOT.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    parser.add_argument("--host-only", action="store_true",
                        help="Skip the in-venv half.")
    args = parser.parse_args(argv)

    ffmpeg_roots = list(args.ffmpeg_root)
    ffmpeg_roots += [r for r in os.environ.get("TAO_WIN_FFMPEG_ROOT", "").split(os.pathsep) if r]
    ffmpeg_roots += list(FFMPEG_ROOTS)

    report = Report()
    run_host_checks(report, args.ffmpeg_dir, ffmpeg_roots)
    if os.name != "nt" or args.host_only:
        return report.emit(args.json)

    venv = Path(args.venv).expanduser().resolve() if args.venv else None
    if venv is not None:
        interpreter = _venv_python(venv)
        if not interpreter.is_file():
            report.add("venv", FAIL, f"{interpreter} is missing",
                       "Windows venvs put the interpreter in Scripts\\python.exe, "
                       "not bin/python.")
            return report.emit(args.json)
        report.add("venv", PASS, str(venv))
        if Path(sys.executable).resolve() != interpreter.resolve():
            # Re-exec the in-venv half. A FILE, never `python -c`: an inline
            # snippet passed through PowerShell loses its quotes and semicolons.
            completed = subprocess.run(
                [str(interpreter), str(Path(__file__).resolve()), "--json",
                 *sum((["--ffmpeg-dir", d] for d in args.ffmpeg_dir), []),
                 "--in-venv"],
                capture_output=True, text=True,
            )
            try:
                inner = json.loads(completed.stdout or "{}")
            except ValueError:
                report.add("venv_probe", FAIL,
                           f"in-venv probe produced no JSON (exit {completed.returncode})",
                           (completed.stderr or "").strip()[-500:])
            else:
                report.checks.extend(inner.get("checks", []))
            return report.emit(args.json)
    run_venv_checks(report, venv)
    return report.emit(args.json)


if __name__ == "__main__":
    # --in-venv is an internal marker for the re-exec above; it only suppresses
    # a second host pass, which the outer process already reported.
    if "--in-venv" in sys.argv:
        sys.argv.remove("--in-venv")
        _report = Report()
        run_venv_checks(_report, None)
        raise SystemExit(_report.emit(True))
    raise SystemExit(main())
