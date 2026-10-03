#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the CUDA framework and TAO entrypoints inside one PAS runtime.

GPU enumeration alone does not establish that the PyTorch/CUDA build in a TAO
image can initialize against the host driver.  This probe deliberately creates
and synchronizes one CUDA tensor on every requested visible device.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import shutil
import sys
from typing import Any, Callable


def probe_runtime(
    *,
    minimum_gpus: int,
    required_clis: list[str],
    torch_module: Any,
    which: Callable[[str], str | None] = shutil.which,
) -> dict[str, Any]:
    """Return runtime facts after real CUDA initialization and CLI checks."""
    if minimum_gpus < 1:
        raise ValueError("minimum_gpus must be at least 1")
    missing = [name for name in required_clis if not which(name)]
    if missing:
        raise RuntimeError("missing TAO CLI entrypoints: " + ", ".join(missing))
    if not torch_module.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    visible = int(torch_module.cuda.device_count())
    if visible < minimum_gpus:
        raise RuntimeError(
            f"visible CUDA devices {visible} are fewer than required {minimum_gpus}"
        )

    devices = []
    for index in range(minimum_gpus):
        try:
            properties = torch_module.cuda.get_device_properties(index)
            allocation = torch_module.empty(1, device=f"cuda:{index}")
            allocation.add_(1)
            torch_module.cuda.synchronize(index)
        except Exception as exc:  # driver/runtime errors vary across torch builds
            raise RuntimeError(
                f"CUDA framework initialization failed on visible device {index}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        devices.append(
            {
                "index": index,
                "name": str(properties.name),
                "compute_capability": f"{properties.major}.{properties.minor}",
                "memory_gb": round(int(properties.total_memory) / (1024**3), 2),
            }
        )

    return {
        "status": "PASS",
        "torch_version": str(torch_module.__version__),
        "cuda_build": str(torch_module.version.cuda),
        "visible_gpus": visible,
        "tested_gpus": devices,
        "required_clis": required_clis,
    }


def probe_clip_lora_contract(config_type: Any, lora_module: Any) -> dict[str, Any]:
    """Verify the CLIP schema plus adapter/checkpoint implementation contract."""
    peft = dataclasses.asdict(config_type().peft)
    required_targets = ["q_proj", "k_proj", "v_proj", "out_proj"]
    if peft.get("method") != "lora" or not isinstance(peft.get("enabled"), bool):
        raise RuntimeError("CLIP PEFT schema lacks the LoRA enable/method contract")
    for tower in ("vision", "text"):
        block = peft.get(tower, {})
        required_fields = (
            "mode",
            "target_modules",
            "num_last_blocks",
            "rank",
            "alpha",
            "dropout",
        )
        missing = [key for key in required_fields if key not in block]
        if missing or block.get("target_modules") != required_targets:
            raise RuntimeError(f"CLIP PEFT {tower} contract is incomplete")
    symbols = (
        "LoRALinear",
        "inject_lora",
        "merge_lora",
        "_register_lora_checkpoint_compatibility",
    )
    missing_symbols = [
        name
        for name in symbols
        if name != "LoRALinear" and not callable(getattr(lora_module, name, None))
    ]
    if not isinstance(getattr(lora_module, "LoRALinear", None), type):
        missing_symbols.append("LoRALinear")
    if missing_symbols:
        raise RuntimeError(
            "CLIP LoRA runtime lacks: "
            + ", ".join(sorted(set(missing_symbols)))
        )
    return {
        "schema": peft,
        "runtime_symbols": list(symbols),
        "checkpoint_behavior": "register-and-merge",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-gpus", type=int, required=True)
    parser.add_argument("--require-cli", action="append", default=[])
    parser.add_argument("--require-clip-lora", action="store_true")
    parser.add_argument("--image-ref")
    parser.add_argument("--output", type=pathlib.Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        import torch

        result = probe_runtime(
            minimum_gpus=args.min_gpus,
            required_clis=args.require_cli,
            torch_module=torch,
        )
        if args.require_clip_lora:
            if not args.image_ref or args.output is None:
                raise ValueError("--require-clip-lora requires --image-ref and --output")
            from nvidia_tao_pytorch.config.clip.default_config import CLIPExperimentConfig
            from nvidia_tao_pytorch.multimodal.clip.model import lora

            result["image_ref"] = args.image_ref
            result["clip_lora"] = probe_clip_lora_contract(CLIPExperimentConfig, lora)
    except Exception as exc:
        print(
            f"PAS_CUDA_PROBE=FAIL reason={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("PAS_CUDA_PROBE=PASS " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
