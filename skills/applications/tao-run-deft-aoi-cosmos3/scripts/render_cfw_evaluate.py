#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Render multi-task Cosmos Framework evaluation TOML."""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from render_cfw_sft import _atomic_text, dump_toml


def build_config(
    *,
    action: str,
    annotation_path: str,
    media_root: str,
    model_path: str,
    results_dir: str,
    num_gpus: int,
    batch_size: int,
) -> dict[str, Any]:
    if action != "evaluate":
        raise ValueError("this renderer only accepts action=evaluate")
    if num_gpus <= 0 or batch_size <= 0:
        raise ValueError("num_gpus and batch_size must be positive")
    config = {
        "num_gpus": num_gpus,
        "results_dir": results_dir,
        "task": {"type": ""},
        "dataset": {
            "annotation_path": annotation_path,
            "media_dir": media_root,
            "system_prompt": "",
        },
        "model": {
            "model_name": model_path,
            "dtype": "bfloat16",
            "max_length": 40960,
            "tp_size": 1,
            "enable_lora": False,
            "base_model_path": "",
            "config_file": "",
            "export_dir": "",
            "vit_checkpoint_path": "",
            "attn_implementation": "sdpa",
        },
        "evaluation": {
            "answer_type": "freeform",
            "num_processes": num_gpus,
            "skip_saved": False,
            "seed": 42,
            "limit": -1,
            "shard_strategy": "stride",
            "shard_id": 0,
            "batch_size": batch_size,
            "barrier_timeout_seconds": 14400,
        },
        "vision": {
            "num_frames": 1,
            "process_threads": 8,
            "dataloader_num_workers": 0,
            "dataloader_persistent_workers": False,
        },
        "generation": {
            "max_retries": 10,
            "max_tokens": 1024,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
        "metrics": {"names": []},
        "results": {
            "save_individual_results": True,
            "save_confusion_matrix": False,
            "save_metrics_summary": False,
        },
    }
    tomllib.loads(dump_toml(config))
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("evaluate",), default="evaluate")
    parser.add_argument("--annotation-path", required=True)
    parser.add_argument("--media-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        config = build_config(
            action=args.action,
            annotation_path=args.annotation_path,
            media_root=args.media_root,
            model_path=args.model_path,
            results_dir=args.results_dir,
            num_gpus=args.num_gpus,
            batch_size=args.batch_size,
        )
        _atomic_text(args.output.expanduser().resolve(), dump_toml(config))
    except (OSError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"render_cfw_evaluate: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
