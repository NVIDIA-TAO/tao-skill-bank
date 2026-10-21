#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe the native ARM64 CUDA runtime without loading a Cosmos model."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from typing import Any


def gib(value: int) -> float:
    return round(value / 2**30, 2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate native Windows-on-ARM Torch and optionally probe CUDA capacity."
    )
    parser.add_argument(
        "--allocation-gib",
        type=int,
        default=0,
        help="Allocate, write, synchronize, and release this many GiB of CUDA memory.",
    )
    args = parser.parse_args()
    if args.allocation_gib < 0:
        parser.error("--allocation-gib must be zero or positive")

    report: dict[str, Any] = {
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "status": "FAIL",
        "checks": [],
    }
    if platform.machine().upper() != "ARM64":
        report["checks"].append("The selected interpreter is not native ARM64.")
        print(json.dumps(report, indent=2))
        return 1

    try:
        import torch
    except Exception as exc:  # pragma: no cover - structured hardware preflight output
        report["checks"].append(f"Torch import failed: {type(exc).__name__}: {exc}")
        print(json.dumps(report, indent=2))
        return 1

    report["torch"] = torch.__version__
    report["torch_cuda"] = torch.version.cuda
    if not torch.cuda.is_available():
        report["checks"].append("torch.cuda.is_available() is false")
        print(json.dumps(report, indent=2))
        return 1

    device = 0
    free_before, total = torch.cuda.mem_get_info(device)
    report.update(
        {
            "device": torch.cuda.get_device_name(device),
            "cuda_total_gib": gib(total),
            "cuda_free_before_gib": gib(free_before),
        }
    )
    if args.allocation_gib:
        buffer = None
        started = time.perf_counter()
        try:
            buffer = torch.empty(args.allocation_gib * 2**30, dtype=torch.uint8, device=device)
            buffer.zero_()
            torch.cuda.synchronize(device)
            free_after, _ = torch.cuda.mem_get_info(device)
            report["allocation_probe"] = {
                "gib": args.allocation_gib,
                "result": "PASS",
                "cuda_free_after_gib": gib(free_after),
                "elapsed_seconds": round(time.perf_counter() - started, 1),
            }
        except Exception as exc:  # pragma: no cover - hardware-dependent result
            report["allocation_probe"] = {
                "gib": args.allocation_gib,
                "result": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(json.dumps(report, indent=2))
            return 1
        finally:
            del buffer
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)

    report["checks"].append("Native ARM64 Torch CUDA probe passed.")
    report["status"] = "PASS"
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
