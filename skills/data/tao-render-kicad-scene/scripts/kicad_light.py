#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thin step recorder and replayer for KiCad SDK / kicad-cli / Blender scripts.

Not a workflow engine: each `step` runs one native command, and the receipt
keeps the argv template, hashes, timing and memory so `replay` can re-execute
the same commands into a new directory and compare the results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
import time

SCHEMA = "tao.kicad_light.receipt.v1"
SKILL_DIR = Path(__file__).resolve().parents[1]
TOOLS = ("kicad-python", "kicad-cli", "blender", "python")
KICAD_PINS = ("AppRun", "bin/kicad-cli")
# Replay comparison policy per suffix: how equality is judged and whether it gates.
TEXT_UUID = (".kicad_pcb", ".kicad_pro", ".kicad_prl")
UUID_RE = re.compile(rb'\((uuid|tstamp) "?[0-9a-fA-F-]{36}"?\)')
# Cycles CPU is not bit-exact between runs; anything beyond this is a real difference.
PNG_TOLERANCE = {"max_abs_diff_8bit": 2, "changed_pixel_fraction": 1e-3}
PASSING = ("identical", "equal_modulo_uuid", "equal_modulo_volatile", "pixel_identical", "pixel_within_tolerance",
           "array_identical")
VOLATILE_KEYS = {"uuid", "timing_seconds", "seconds", "board_sha256", "peak_rss_mb"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def checkout_root():
    for parent in SKILL_DIR.parents:
        if (parent / ".git").exists():
            return parent
    return None


def external(path, what):
    """Runtime, data and outputs never live inside the code checkout."""
    path = Path(path).expanduser().resolve()
    root = checkout_root()
    if root and (path == root or root in path.parents):
        raise SystemExit(f"{what} must stay outside the code checkout: {path}")
    return path


def runtime(args):
    appdir = args.appdir or os.environ.get("KICAD_APPDIR")
    blender = args.blender or os.environ.get("BLENDER_BIN")
    if not appdir or not blender:
        raise SystemExit("Set --appdir/--blender or KICAD_APPDIR/BLENDER_BIN")
    return external(appdir, "KiCad AppDir"), external(blender, "Blender binary")


def probe(appdir, blender):
    """Versions and hashes of the external runtimes actually used."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    run = lambda cmd: subprocess.run(list(map(str, cmd)), capture_output=True, text=True, env=env, timeout=120)
    code = "import pcbnew,sys;print(pcbnew.GetBuildVersion());print(sys.version.split()[0])"
    sdk = run([appdir / "AppRun", "python3.11", "-c", code])
    cli = run([appdir / "AppRun", "kicad-cli", "version"])
    bl = run([blender, "--version"])
    if sdk.returncode or cli.returncode or bl.returncode:
        raise SystemExit("Runtime probe failed:\n" + sdk.stderr + cli.stderr + bl.stderr)
    lines = sdk.stdout.strip().splitlines()
    build = re.search(r"build hash: (\w+)", bl.stdout)
    return {
        "kicad": {"appdir": str(appdir), "pcbnew_build": lines[-2], "bundled_python": lines[-1],
                  "kicad_cli_version": cli.stdout.strip().splitlines()[-1],
                  "sdk_path": "pcbnew SWIG bindings (legacy, headless); not the IPC client",
                  "pins_sha256": {p: sha(appdir / p) for p in KICAD_PINS}},
        "blender": {"binary": str(blender), "version": bl.stdout.splitlines()[0].strip(),
                    "build_hash": build.group(1) if build else None, "binary_sha256": sha(blender)},
        "host": {"node": platform.node(), "platform": platform.platform(), "cpus": os.cpu_count(),
                 "controller_python": sys.version.split()[0]},
    }


def snapshot(run_dir):
    return {str(p.relative_to(run_dir)): sha(p) for p in sorted(run_dir.rglob("*"))
            if p.is_file() and p.name != "receipt.json"}


def load_receipt(run_dir):
    path = run_dir / "receipt.json"
    return json.loads(path.read_text()) if path.exists() else None


def save_receipt(run_dir, receipt):
    (run_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def expand(token, run_dir):
    return token.replace("{run}", str(run_dir)).replace("{skill}", str(SKILL_DIR))


def execute_step(run_dir, receipt, name, tool, template, appdir, blender, threads, timeout):
    """Run one native command; record what it read, wrote and cost."""
    if any(s["name"] == name for s in receipt["steps"]):
        raise SystemExit(f"Step name already recorded: {name}")
    argv = [expand(t, run_dir) for t in template]
    prefix = {"kicad-python": [appdir / "AppRun", "python3.11"],
              "kicad-cli": [appdir / "AppRun", "kicad-cli"],
              "blender": [blender, "-b", "--factory-startup", "-t", str(threads), "--python-exit-code", "1"],
              "python": [sys.executable]}[tool]
    # Files named on the command line that already exist are inputs: hash before and after.
    inputs = {}
    for token, raw in zip(argv, template):
        path = Path(token)
        if path.is_file():
            inputs[raw] = sha(path)
    before = snapshot(run_dir)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="void",
               OMP_NUM_THREADS=str(threads), KICAD_APPDIR=str(appdir))
    log = run_dir / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started_utc, started = utc(), time.perf_counter()
    with log.open("w") as stream:
        proc = subprocess.Popen(list(map(str, prefix)) + argv, stdout=stream, stderr=subprocess.STDOUT, env=env)
        timed_out = threading.Event()
        killer = threading.Timer(timeout, lambda: (timed_out.set(), proc.kill()))
        killer.start()
        _, status, usage = os.wait4(proc.pid, 0)  # per-step CPU time and peak RSS of the process tree
        killer.cancel()
    returncode = os.waitstatus_to_exitcode(status)
    seconds = time.perf_counter() - started
    after = snapshot(run_dir)
    step = {
        "name": name, "tool": tool, "argv_template": template, "returncode": returncode,
        "started_utc": started_utc, "finished_utc": utc(), "seconds": round(seconds, 3),
        "cpu_seconds": round(usage.ru_utime + usage.ru_stime, 3),
        "peak_rss_mb": round(usage.ru_maxrss / 1024, 1),
        "threads": threads, "device": "CPU", "inputs_sha256": inputs,
        "outputs_sha256": {k: v for k, v in after.items() if before.get(k) != v and not k.startswith("logs/")},
        "log": str(log.relative_to(run_dir)),
    }
    changed = [raw for raw, digest in inputs.items()
               if Path(expand(raw, run_dir)).is_file() and sha(expand(raw, run_dir)) != digest]
    if timed_out.is_set():
        step["error"] = f"timeout after {timeout} s; process killed; see {step['log']}"
    elif changed:
        step["error"] = "input overwritten: " + ", ".join(changed)
    elif returncode:
        step["error"] = f"exit {returncode}; see {step['log']}"
    step["status"] = "error" if "error" in step else "ok"
    receipt["steps"].append(step)
    receipt["status"] = "failed" if "error" in step else "open"
    receipt["updated_utc"] = utc()
    save_receipt(run_dir, receipt)
    return step


def cmd_check(args):
    appdir, blender = runtime(args)
    print(json.dumps(probe(appdir, blender), indent=2))


def cmd_step(args):
    appdir, blender = runtime(args)
    run_dir = external(args.run, "Run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    receipt = load_receipt(run_dir)
    if receipt is None:
        receipt = {"schema_version": SCHEMA, "status": "open", "created_utc": utc(),
                   "runtime": probe(appdir, blender), "steps": [],
                   "skill_dir_token": "{skill}", "run_dir_token": "{run}"}
    template = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    step = execute_step(run_dir, receipt, args.name, args.tool, template, appdir, blender,
                        args.threads, args.timeout)
    print(json.dumps({k: step[k] for k in ("name", "status", "seconds", "outputs_sha256")}, indent=2))
    if step["status"] != "ok":
        print(step["error"], file=sys.stderr)
        sys.exit(1)


def cmd_finish(args):
    run_dir = external(args.run, "Run directory")
    receipt = load_receipt(run_dir)
    if receipt is None:
        raise SystemExit("No receipt.json in " + str(run_dir))
    ok = all(s["status"] == "ok" for s in receipt["steps"]) and receipt["steps"]
    receipt["status"] = "complete" if ok else "failed"
    receipt["artifacts_sha256"] = snapshot(run_dir)
    receipt["totals"] = {"steps": len(receipt["steps"]),
                         "wall_seconds": round(sum(s["seconds"] for s in receipt["steps"]), 3),
                         "cpu_seconds": round(sum(s["cpu_seconds"] for s in receipt["steps"]), 3),
                         "peak_rss_mb": max(s["peak_rss_mb"] for s in receipt["steps"]),
                         "renders": sum(1 for s in receipt["steps"] if s["tool"] == "blender"),
                         "gpu_used": False}
    receipt["updated_utc"] = utc()
    save_receipt(run_dir, receipt)
    print(json.dumps({"status": receipt["status"], "totals": receipt["totals"],
                      "receipt_sha256": sha(run_dir / "receipt.json")}, indent=2))
    sys.exit(0 if ok else 1)


def strip_volatile(value):
    if isinstance(value, dict):
        return {k: strip_volatile(v) for k, v in sorted(value.items()) if k not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [strip_volatile(v) for v in value]
    return value


def pixels(path):
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.int16)


def compare(name, old, new):
    """Return (verdict, gating, detail) for one replayed artifact."""
    suffix = Path(name).suffix
    if not new.is_file():
        return "missing", True, None
    if sha(old) == sha(new):
        return "identical", True, None
    if suffix in TEXT_UUID:
        same = UUID_RE.sub(b"(uuid)", old.read_bytes()) == UUID_RE.sub(b"(uuid)", new.read_bytes())
        return ("equal_modulo_uuid" if same else "differs"), True, None
    if suffix == ".json":
        same = strip_volatile(json.loads(old.read_text())) == strip_volatile(json.loads(new.read_text()))
        return ("equal_modulo_volatile" if same else "differs"), True, "ignored keys: " + ",".join(sorted(VOLATILE_KEYS))
    if suffix == ".png":
        a, b = pixels(old), pixels(new)
        if a is None:
            return "bytes_differ_no_pixel_reader", True, "install numpy+Pillow for the controller python"
        if a.shape != b.shape:
            return "differs", True, f"shape {a.shape} vs {b.shape}"
        delta = abs(a - b)
        changed = float((delta.max(axis=2) > 0).mean())
        detail = {"max_abs_diff_8bit": int(delta.max()), "changed_pixel_fraction": changed,
                  "tolerance": PNG_TOLERANCE}
        if delta.max() == 0:
            return "pixel_identical", True, detail
        near = delta.max() <= PNG_TOLERANCE["max_abs_diff_8bit"] and changed <= PNG_TOLERANCE["changed_pixel_fraction"]
        return ("pixel_within_tolerance" if near else "differs"), True, detail
    if suffix == ".npy":  # entity-ID masks: values decide, not container bytes
        try:
            import numpy as np
        except ImportError:
            return "bytes_differ_no_array_reader", True, "install numpy for the controller python"
        a, b = np.load(old, allow_pickle=False), np.load(new, allow_pickle=False)
        if a.dtype != b.dtype or a.shape != b.shape:
            return "differs", True, f"dtype/shape {a.dtype}{a.shape} vs {b.dtype}{b.shape}"
        changed = int((a != b).sum())
        return ("array_identical" if changed == 0 else "differs"), True, {"changed_elements": changed}
    # Binary scene containers (.blend, .glb) embed non-semantic bytes; report, do not gate.
    return "bytes_differ_not_gating", False, "open the replayed file to inspect; gating artifacts cover its content"


def cmd_replay(args):
    appdir, blender = runtime(args)
    source = Path(args.receipt).expanduser().resolve()
    original = json.loads(source.read_text())
    old_dir = source.parent
    run_dir = external(args.run, "Replay directory")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit("Replay needs a new empty directory; the original run is never overwritten")
    run_dir.mkdir(parents=True, exist_ok=True)
    receipt = {"schema_version": SCHEMA, "status": "open", "created_utc": utc(),
               "runtime": probe(appdir, blender), "steps": [],
               "replay_of": {"receipt": str(source), "receipt_sha256": sha(source)}}
    for step in original["steps"]:
        done = execute_step(run_dir, receipt, step["name"], step["tool"], step["argv_template"],
                            appdir, blender, step.get("threads", args.threads), args.timeout)
        if done["status"] != "ok":
            receipt["status"] = "replay_step_failed"
            save_receipt(run_dir, receipt)
            raise SystemExit(f"Replay step failed: {step['name']}: {done['error']}")
    rows, passed = [], True
    for step in original["steps"]:
        for name in sorted(step["outputs_sha256"]):
            verdict, gating, detail = compare(name, old_dir / name, run_dir / name)
            good = verdict in PASSING or not gating
            passed = passed and good
            rows.append({"artifact": name, "step": step["name"], "verdict": verdict, "gating": gating,
                         "pass": good, "detail": detail})
    drift = {k: (original["runtime"][k], receipt["runtime"][k]) for k in ("kicad", "blender")
             if strip_volatile(original["runtime"][k]) != strip_volatile(receipt["runtime"][k])}
    receipt["comparison"] = {"pass": passed, "artifacts": rows, "runtime_drift": drift,
                             "policy": "sha256, else UUID-stripped KiCad text, volatile-key-stripped JSON, "
                                       f"decoded PNG pixels within {PNG_TOLERANCE}, .npy arrays equal in "
                                       "dtype/shape/values; .blend/.glb bytes reported but not gating"}
    receipt["status"] = "complete" if passed else "replay_mismatch"
    receipt["artifacts_sha256"] = snapshot(run_dir)
    receipt["updated_utc"] = utc()
    save_receipt(run_dir, receipt)
    summary = {}
    for row in rows:
        summary[row["verdict"]] = summary.get(row["verdict"], 0) + 1
    print(json.dumps({"replay_pass": passed, "verdicts": summary, "runtime_drift": bool(drift),
                      "failed": [r for r in rows if not r["pass"]]}, indent=2))
    sys.exit(0 if passed else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appdir", help="extracted KiCad AppDir (or KICAD_APPDIR)")
    parser.add_argument("--blender", help="Blender binary (or BLENDER_BIN)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="print runtime versions/hashes").set_defaults(func=cmd_check)
    step = sub.add_parser("step", help="run and record one native command")
    step.add_argument("--run", required=True)
    step.add_argument("--name", required=True)
    step.add_argument("--tool", required=True, choices=TOOLS)
    step.add_argument("--threads", type=int, default=8)
    step.add_argument("--timeout", type=int, default=1800)
    step.add_argument("argv", nargs=argparse.REMAINDER, help="-- args; use {run} and {skill} tokens")
    step.set_defaults(func=cmd_step)
    finish = sub.add_parser("finish", help="seal the receipt with artifact hashes and totals")
    finish.add_argument("--run", required=True)
    finish.set_defaults(func=cmd_finish)
    replay = sub.add_parser("replay", help="re-execute a receipt into a new directory and compare")
    replay.add_argument("--receipt", required=True)
    replay.add_argument("--run", required=True)
    replay.add_argument("--threads", type=int, default=8)
    replay.add_argument("--timeout", type=int, default=1800)
    replay.set_defaults(func=cmd_replay)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
