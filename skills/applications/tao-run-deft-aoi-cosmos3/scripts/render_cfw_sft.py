#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Render the reviewed NVPAW Cosmos Framework full or smoke SFT profile."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import tempfile
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


FULL_GPUS = 8
MICRO_BATCH_PER_RANK = 4
GRAD_ACCUMULATION = 16


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_scalar(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(
            f"{key} = {_scalar(item)}" for key, item in value.items()
        ) + " }"
    raise TypeError(f"unsupported TOML value: {type(value).__name__}")


def dump_toml(value: dict[str, Any]) -> str:
    """Serialize a nested, null-free mapping without flattening its source spec."""

    lines: list[str] = []

    def emit(table: dict[str, Any], prefix: tuple[str, ...]) -> None:
        scalars = [
            (key, item)
            for key, item in table.items()
            if not isinstance(item, dict) or key == "lr_multipliers"
        ]
        children = [
            (key, item)
            for key, item in table.items()
            if isinstance(item, dict) and key != "lr_multipliers"
        ]
        if prefix:
            if lines and lines[-1] != "":
                lines.append("")
            lines.append("[" + ".".join(prefix) + "]")
        lines.extend(f"{key} = {_scalar(item)}" for key, item in scalars)
        for key, child in children:
            emit(child, (*prefix, key))

    emit(value, ())
    return "\n".join(lines).rstrip() + "\n"


def build_profile(
    profile: str,
    *,
    model_path: str,
    train_jsonl: str,
    media_root: str,
    index_path: str,
    expected_rows: int,
    expected_sha256: str,
    expected_image_items: int,
    run_name: str,
    results_dir: str,
    resume_checkpoint: str | None = None,
    num_gpus: int | None = None,
) -> dict[str, Any]:
    if profile not in {"full", "smoke"}:
        raise ValueError("profile must be full or smoke")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
    if min(expected_rows, expected_image_items) <= 0:
        raise ValueError("expected rows and image items must be positive")
    selected_gpus = FULL_GPUS if num_gpus is None else int(num_gpus)
    if selected_gpus <= 0:
        raise ValueError("num_gpus must be positive")
    if profile == "full" and selected_gpus != FULL_GPUS:
        raise ValueError("full profile is fixed to 8 GPUs")

    max_iter = 500 if profile == "full" else 5
    save_iter = 100 if profile == "full" else 5
    warm_up_steps = 5 if profile == "full" else 1
    config: dict[str, Any] = {
        "job": {
            "task": "vlm",
            "experiment": "nvpaw_omni_vlm_sft",
            "project": "deft_aoi",
            "group": "train",
            "name": run_name,
            "wandb_mode": "disabled",
        },
        "model": {
            "attn_implementation": "flash_attention_2",
            "precision": "bfloat16",
            "backbone": {
                "model_name": model_path,
                "safetensors_path": model_path,
            },
            "ema": {"enabled": False, "rate": 0.1, "iteration_shift": 0},
            "parallelism": {
                "data_parallel_shard_degree": selected_gpus,
                "data_parallel_replicate_degree": 1,
                "context_parallel_shard_degree": 1,
                "cfg_parallel_shard_degree": 1,
            },
            "compile": {"enabled": False, "compile_dynamic": True},
            "activation_checkpointing": {
                "mode": "full",
                "save_ops_regex": ["fmha"],
                "preserve_rng_state": True,
                "determinism_check": "default",
            },
        },
        "optimizer": {
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "fused": True,
            "keys_to_select": [],
            "lr": 1.0e-6,
            "lr_multipliers": {"merger": 20.0},
            "weight_decay": 0.05,
        },
        "scheduler": {
            "cycle_lengths": [max_iter],
            "f_max": [1.0],
            "f_min": [0.1],
            "f_start": [0.05],
            "verbosity_interval": 0,
            "warm_up_steps": [warm_up_steps],
        },
        "trainer": {
            "distributed_parallelism": "fsdp",
            "grad_accum_iter": GRAD_ACCUMULATION,
            "logging_iter": 1,
            "max_iter": max_iter,
            "callbacks": {
                "grad_clip": {"clip_norm": 1.0, "force_finite": False}
            },
        },
        "checkpoint": {
            "keys_to_skip_loading": [],
            "load_path": resume_checkpoint or "???",
            "save_iter": save_iter,
            "save_freq_in_epoch": 0,
            "dcp_async_mode_enabled": False,
        },
    }
    freeze = {
        "vision_encoder": True,
        "multimodal_projector": False,
        "language_model": False,
    }
    hydra_overrides = [
        f"dataloader_train.distributor.dataset.jsonl_path={train_jsonl}",
        f"dataloader_train.distributor.dataset.media_root={media_root}",
        f"dataloader_train.distributor.dataset.index_path={index_path}",
        f"dataloader_train.distributor.dataset.expected_rows={expected_rows}",
        f"dataloader_train.distributor.dataset.expected_sha256={expected_sha256}",
        f"dataloader_train.distributor.dataset.expected_image_items={expected_image_items}",
        f"dataloader_train.processor.resample_dataset.jsonl_path={train_jsonl}",
        f"dataloader_train.processor.resample_dataset.media_root={media_root}",
        f"dataloader_train.processor.resample_dataset.index_path={index_path}",
        f"dataloader_train.processor.resample_dataset.expected_rows={expected_rows}",
        f"dataloader_train.processor.resample_dataset.expected_sha256={expected_sha256}",
        f"dataloader_train.processor.resample_dataset.expected_image_items={expected_image_items}",
        f"dataloader_train.distributor.micro_batch_size={MICRO_BATCH_PER_RANK}",
        f"dataloader_train.batcher.batch_size={MICRO_BATCH_PER_RANK}",
        "model.config.freeze.freeze_vision_encoder=true",
        "model.config.freeze.freeze_mm_projector=false",
        "model.config.freeze.freeze_llm=false",
    ]
    descriptor = {
        "schema_version": 1,
        "profile": profile,
        "config": config,
        "freeze": freeze,
        "data": {
            "jsonl_path": train_jsonl,
            "media_root": media_root,
            "index_path": index_path,
            "expected_rows": expected_rows,
            "expected_sha256": expected_sha256,
            "expected_image_items": expected_image_items,
            "micro_batch_per_rank": MICRO_BATCH_PER_RANK,
            "gradient_accumulation": GRAD_ACCUMULATION,
            "global_batch": MICRO_BATCH_PER_RANK
            * selected_gpus
            * GRAD_ACCUMULATION,
        },
        "results_dir": results_dir,
        "hydra_overrides": hydra_overrides,
    }
    validate_profile(descriptor, profile=profile)
    return descriptor


def validate_profile(descriptor: dict[str, Any], *, profile: str) -> None:
    config = descriptor.get("config", {})
    model = config.get("model", {})
    optimizer = config.get("optimizer", {})
    trainer = config.get("trainer", {})
    checkpoint = config.get("checkpoint", {})
    scheduler = config.get("scheduler", {})
    parallelism = model.get("parallelism", {})
    if config.get("job", {}).get("experiment") != "nvpaw_omni_vlm_sft":
        raise ValueError("CFW profile must select nvpaw_omni_vlm_sft")
    if model.get("precision") != "bfloat16":
        raise ValueError("CFW profile must use BF16")
    if optimizer.get("keys_to_select") != []:
        raise ValueError("CFW profile must use full-parameter tuning")
    if optimizer.get("lr") != 1.0e-6 or optimizer.get("weight_decay") != 0.05:
        raise ValueError("CFW profile optimizer differs from the reviewed recipe")
    if optimizer.get("betas") != [0.9, 0.999] or optimizer.get("fused") is not True:
        raise ValueError("CFW profile must use fused AdamW betas 0.9/0.999")
    if optimizer.get("lr_multipliers") != {"merger": 20.0}:
        raise ValueError("CFW profile must use merger LR multiplier 20")
    if trainer.get("distributed_parallelism") != "fsdp":
        raise ValueError("CFW profile must use FSDP")
    if trainer.get("grad_accum_iter") != GRAD_ACCUMULATION:
        raise ValueError("CFW profile must use gradient accumulation 16")
    if parallelism.get("data_parallel_replicate_degree") != 1:
        raise ValueError("CFW profile is a one-node recipe")
    if profile == "full" and parallelism.get("data_parallel_shard_degree") != FULL_GPUS:
        raise ValueError("full CFW profile must shard over 8 GPUs")
    if model.get("activation_checkpointing", {}).get("mode") != "full":
        raise ValueError("CFW profile must use full activation checkpointing")
    if checkpoint.get("dcp_async_mode_enabled") is not False:
        raise ValueError("CFW profile must use synchronous DCP")
    if checkpoint.get("keys_to_skip_loading") != []:
        raise ValueError("CFW DCP resume must restore all keys")
    if descriptor.get("freeze") != {
        "vision_encoder": True,
        "multimodal_projector": False,
        "language_model": False,
    }:
        raise ValueError("CFW freeze policy differs from the reviewed recipe")
    data = descriptor.get("data", {})
    expected_global = (
        data.get("micro_batch_per_rank", 0)
        * parallelism.get("data_parallel_shard_degree", 0)
        * trainer.get("grad_accum_iter", 0)
    )
    if data.get("global_batch") != expected_global:
        raise ValueError("CFW global batch is inconsistent")
    if profile == "full":
        if trainer.get("max_iter") != 500 or checkpoint.get("save_iter") != 100:
            raise ValueError("full CFW profile must run 500 iterations and save every 100")
        if scheduler.get("cycle_lengths") != [500] or scheduler.get("warm_up_steps") != [5]:
            raise ValueError("full CFW scheduler differs from the reviewed recipe")
    rendered = dump_toml(config)
    tomllib.loads(rendered)


def _atomic_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("full", "smoke"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--media-root", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-image-items", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--descriptor-output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    try:
        descriptor = build_profile(
            args.profile,
            model_path=args.model_path,
            train_jsonl=args.train_jsonl,
            media_root=args.media_root,
            index_path=args.index_path,
            expected_rows=args.expected_rows,
            expected_sha256=args.expected_sha256,
            expected_image_items=args.expected_image_items,
            run_name=args.run_name,
            results_dir=args.results_dir,
            resume_checkpoint=args.resume_checkpoint,
            num_gpus=args.num_gpus,
        )
        _atomic_text(args.output.expanduser().resolve(), dump_toml(descriptor["config"]))
        _atomic_text(
            args.descriptor_output.expanduser().resolve(),
            json.dumps(descriptor, indent=2, sort_keys=True) + "\n",
        )
    except (OSError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"render_cfw_sft: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
