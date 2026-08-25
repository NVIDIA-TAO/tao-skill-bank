#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the IAA CUDA framework gate and record a bounded compatibility choice."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import subprocess
import tempfile
from typing import Sequence


COMPAT_PATH = "/usr/local/cuda/compat/lib.real:/usr/local/cuda/lib64"
DRIVER_FAILURE_MARKERS = (
    "nvidia driver on your system is too old",
    "torch.cuda.is_available() is false",
)


def docker_probe_argv(
    image: str,
    gpu_ids: Sequence[int],
    probe_script: pathlib.Path,
    required_clis: Sequence[str],
    *,
    compatibility_path: str | None = None,
) -> list[str]:
    selector = ",".join(str(item) for item in gpu_ids)
    argv = [
        "docker", "run", "--rm", "--gpus", f'"device={selector}"',
        "--entrypoint", "python3", "-e", "PYTHONDONTWRITEBYTECODE=1",
    ]
    if compatibility_path is not None:
        argv.extend(["-e", f"LD_LIBRARY_PATH={compatibility_path}"])
    argv.extend([
        "-v", f"{probe_script.resolve()}:/probe/check_iaa_cuda_runtime.py:ro",
        image, "/probe/check_iaa_cuda_runtime.py", "--min-gpus", str(len(gpu_ids)),
    ])
    for cli in required_clis:
        argv.extend(["--require-cli", cli])
    return argv


def bundle_check_argv(image: str) -> list[str]:
    script = (
        "from pathlib import Path; "
        "p=Path('/usr/local/cuda/compat/lib.real'); "
        "assert p.is_dir() and any(p.glob('libcuda.so*')) and "
        "any(p.glob('libnvidia-ptxjitcompiler.so*'))"
    )
    return ["docker", "run", "--rm", "--entrypoint", "python3", image, "-c", script]


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), text=True, capture_output=True, check=False)


def driver_insufficient(completed: subprocess.CompletedProcess[str]) -> bool:
    text = f"{completed.stdout}\n{completed.stderr}".lower()
    return completed.returncode != 0 and all(item in text for item in DRIVER_FAILURE_MARKERS)


def _write_atomic(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(raw, 0o600)
        os.replace(raw, path)
    finally:
        if os.path.exists(raw):
            os.unlink(raw)


def run_gate(
    image: str,
    gpu_ids: Sequence[int],
    probe_script: pathlib.Path,
    required_clis: Sequence[str],
    output: pathlib.Path,
) -> dict:
    normal = _run(docker_probe_argv(image, gpu_ids, probe_script, required_clis))
    mode = "native"
    path = None
    if normal.returncode != 0:
        if not driver_insufficient(normal):
            raise RuntimeError("CUDA gate failed for a reason other than an insufficient driver")
        bundle = _run(bundle_check_argv(image))
        if bundle.returncode != 0:
            raise RuntimeError("CUDA driver is insufficient and the image has no verified compatibility bundle")
        compatible = _run(
            docker_probe_argv(
                image, gpu_ids, probe_script, required_clis,
                compatibility_path=COMPAT_PATH,
            )
        )
        if compatible.returncode != 0:
            raise RuntimeError("CUDA gate still fails with the verified image compatibility bundle")
        mode, path = "image_forward_compat", COMPAT_PATH
    payload = {
        "schema_version": 1,
        "workflow": "tao-run-deft-iaa",
        "image": image,
        "gpu_ids": list(gpu_ids),
        "required_clis": list(required_clis),
        "status": "PASS",
        "compatibility_mode": mode,
        "compatibility_path": path,
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    _write_atomic(output.resolve(), payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--probe-script", required=True, type=pathlib.Path)
    parser.add_argument("--require-cli", action="append", default=[])
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    gpu_ids = [int(item) for item in args.gpu_ids.split(",")]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or any(item < 0 for item in gpu_ids):
        raise ValueError("--gpu-ids must be a non-empty unique comma-separated integer list")
    result = run_gate(args.image, gpu_ids, args.probe_script, args.require_cli, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
