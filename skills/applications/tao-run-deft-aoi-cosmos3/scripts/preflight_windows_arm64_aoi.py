#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Report readiness for a native Windows ARM64 Cosmos3 DEFT AOI run.

This preflight deliberately performs no package installation, model download,
dataset download, source modification, or training submission.  It is the
native replacement for the container-oriented discovery steps: it checks the
isolated interpreter, the actual CUDA allocation gate, required patched source
trees, local paths, and the parquet_io backend the mining handoff will use.

Exit status is zero when the inspection itself ran.  Consumers must inspect
``blocking_findings`` before authorising a preparation or GPU stage.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import socket
import subprocess
import sys
from typing import Any


REQUIRED_MODULES = (
    "yaml",
    "torch",
    "transformers",
    "accelerate",
    "hydra",
    "omegaconf",
    "peft",
    "safetensors",
    "torchcodec",
)


def _path(value: str | pathlib.Path) -> pathlib.Path:
    return pathlib.Path(value).expanduser().resolve()


def _run_json(python: pathlib.Path, source: str) -> dict[str, Any]:
    completed = subprocess.run(
        [str(python), "-c", source],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"target interpreter failed ({completed.returncode}): {completed.stderr.strip()}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"target interpreter returned invalid JSON: {completed.stdout!r}") from exc


def _interpreter_report(python: pathlib.Path) -> dict[str, Any]:
    source = """
import importlib.util, json, platform, sys
result = {"python": sys.executable, "version": platform.python_version(), "machine": platform.machine(),
          "modules": {name: importlib.util.find_spec(name) is not None for name in %r}}
try:
    import torch
    result["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda,
                       "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        result["torch"].update({"gpu": torch.cuda.get_device_name(0),
                                "memory_bytes": torch.cuda.get_device_properties(0).total_memory})
except Exception as exc:
    result["torch_error"] = repr(exc)
print(json.dumps(result))
""" % (REQUIRED_MODULES,)
    return _run_json(python, source)


def _source_check(root: pathlib.Path | None, checks: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(root) if root is not None else None,
        "present": root is not None and root.is_dir(),
        "checks": {},
    }
    if root is None or not root.is_dir():
        return report
    for name, fragments in checks.items():
        path = root / name
        result: dict[str, Any] = {"path": str(path), "present": path.is_file()}
        if path.is_file() and fragments:
            text = path.read_text(encoding="utf-8", errors="replace")
            result["fragments"] = {fragment: fragment in text for fragment in fragments}
        report["checks"][name] = result
    return report


def _port_state(port: int) -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        return "in_use" if probe.connect_ex(("127.0.0.1", port)) == 0 else "free"


def _allocation_gate(bank_root: pathlib.Path, python: pathlib.Path, gib: int) -> dict[str, Any]:
    script = bank_root / "skills/platform/tao-run-on-windows-venv/scripts/preflight_windows_arm64.py"
    if not script.is_file():
        return {"status": "missing", "script": str(script)}
    completed = subprocess.run(
        [str(python), str(script), "--allocation-gib", str(gib)],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "script": str(script),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[4])
    parser.add_argument("--venv", required=True, type=pathlib.Path)
    parser.add_argument("--workspace", required=True, type=pathlib.Path)
    parser.add_argument("--model-path", type=pathlib.Path)
    parser.add_argument("--framework-root", type=pathlib.Path)
    parser.add_argument("--cosmos-rl-root", type=pathlib.Path)
    parser.add_argument("--tao-core-root", type=pathlib.Path)
    parser.add_argument("--allocation-gib", type=int, default=72)
    parser.add_argument(
        "--minimum-free-gib",
        type=float,
        default=40.0,
        help=(
            "minimum free space on the VLM/output drive; 40 GiB is the native "
            "checkpoint floor after a prepared VLM is already present"
        ),
    )
    parser.add_argument("--skip-allocation", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bank_root = _path(args.bank_root)
    venv = _path(args.venv)
    python = venv / "Scripts/python.exe"
    workspace = _path(args.workspace)
    report: dict[str, Any] = {
        "platform": "windows-venv",
        "architecture": "ARM64",
        "native_mining_table_format": "parquet (parquet_io: pyarrow, else duckdb)",
        "paths": {
            "bank_root": str(bank_root),
            "venv": str(venv),
            "workspace": str(workspace),
            "model_path": str(_path(args.model_path)) if args.model_path else None,
        },
    }
    findings: list[str] = []
    if not python.is_file():
        findings.append(f"target interpreter is missing: {python}")
    else:
        try:
            interpreter = _interpreter_report(python)
            report["interpreter"] = interpreter
            if interpreter["machine"] != "ARM64":
                findings.append(f"target Python is not native ARM64: {interpreter['machine']}")
            torch = interpreter.get("torch", {})
            if not torch.get("cuda_available"):
                findings.append("target Python has no CUDA-capable torch")
            missing_modules = [
                name for name, present in interpreter["modules"].items() if not present
            ]
            if missing_modules:
                findings.append("target Python is missing modules: " + ", ".join(missing_modules))
        except RuntimeError as exc:
            findings.append(str(exc))
    if args.model_path:
        model_path = _path(args.model_path)
        model_files = list(model_path.rglob("*")) if model_path.is_dir() else []
        safetensors = [path for path in model_files if path.is_file() and path.suffix == ".safetensors"]
        report["prepared_model"] = {
            "path": str(model_path),
            "present": model_path.is_dir(),
            "has_config": (model_path / "config.json").is_file(),
            "has_safetensors_index": (model_path / "model.safetensors.index.json").is_file(),
            "safetensors_files": len(safetensors),
            "size_bytes": sum(path.stat().st_size for path in model_files if path.is_file()),
        }
        if not model_path.is_dir():
            findings.append(f"prepared VLM model directory is missing: {model_path}")
        elif not report["prepared_model"]["has_config"]:
            findings.append(f"prepared VLM model config is missing: {model_path / 'config.json'}")
        elif not report["prepared_model"]["has_safetensors_index"] or not safetensors:
            findings.append(f"prepared VLM model safetensors are incomplete: {model_path}")
    if not workspace.is_dir():
        findings.append(f"AOI workspace is not staged: {workspace}")
    if args.minimum_free_gib <= 0:
        findings.append("--minimum-free-gib must be positive")
    report["storage"] = {
        "venv_drive_free_bytes": shutil.disk_usage(venv.anchor or venv).free,
        "workspace_drive_free_bytes": shutil.disk_usage(workspace.anchor or workspace).free,
        "minimum_free_gib": args.minimum_free_gib,
    }
    free_gib = report["storage"]["workspace_drive_free_bytes"] / 1024**3
    report["storage"]["workspace_drive_free_gib"] = round(free_gib, 2)
    if free_gib < args.minimum_free_gib:
        findings.append(
            f"workspace drive has {free_gib:.2f} GiB free; at least "
            f"{args.minimum_free_gib:.2f} GiB is required for the native checkpoint/output path"
        )
    report["ports"] = {str(port): _port_state(port) for port in (29500, 29502)}
    if report["ports"]["29500"] != "free" or report["ports"]["29502"] != "free":
        findings.append("training ports 29500 and 29502 must be free before a Framework train job")
    framework = _source_check(
        _path(args.framework_root) if args.framework_root else None,
        {
            "cosmos_framework/utils/win_compat.py": (),
            "cosmos_framework/scripts/_train.py": ("win_compat", "link_dir"),
            "cosmos_framework/configs/base/reasoner/experiment/tao_aoi_pair_sft.py": ("aoi_pair_conversation",),
        },
    )
    cosmos_rl = _source_check(
        _path(args.cosmos_rl_root) if args.cosmos_rl_root else None,
        {"cosmos_rl/evaluation/base.py": ("torchcodec",)},
    )
    tao_core = _source_check(
        _path(args.tao_core_root) if args.tao_core_root else None,
        {"nvidia_tao_core/microservices/handlers/execution_handlers/slurm_handler.py": ("import pwd", "ImportError")},
    )
    report["sources"] = {"cosmos_framework": framework, "cosmos_rl": cosmos_rl, "tao_core": tao_core}
    for name, source in report["sources"].items():
        if not source["present"]:
            findings.append(f"required patched source checkout is missing: {name}")
            continue
        for relative, check in source["checks"].items():
            if not check["present"]:
                findings.append(f"{name} is missing required patched file: {relative}")
                continue
            missing_fragments = [
                fragment
                for fragment, present in check.get("fragments", {}).items()
                if not present
            ]
            if missing_fragments:
                findings.append(
                    f"{name} is missing required Windows patch markers in {relative}: "
                    + ", ".join(missing_fragments)
                )
    if not args.skip_allocation and python.is_file():
        report["allocation_gate"] = _allocation_gate(bank_root, python, args.allocation_gib)
        if report["allocation_gate"]["status"] != "passed":
            findings.append("native CUDA allocation gate did not pass")
    report["blocking_findings"] = findings
    report["ready_for_gpu_stage"] = not findings
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        _atomic_json(_path(args.output), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
