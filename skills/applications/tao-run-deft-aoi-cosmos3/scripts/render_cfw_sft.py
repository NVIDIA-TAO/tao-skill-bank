#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Render and validate a Cosmos Framework CR3 SFT profile."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import tempfile
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 in supported TAO host environments
    import tomli as tomllib


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROFILES = {
    "smoke": SKILL_ROOT / "references/cosmos_framework_sft_smoke.toml",
    "full": SKILL_ROOT / "references/cosmos_framework_sft_full.toml",
}
MARKERS = {
    "__TAO_CR3_MODEL__": "model_path",
    "__TAO_CR3_TRAIN_ANNOTATION__": "annotation_path",
    "__TAO_CR3_MEDIA_ROOT__": "media_root",
    "__TAO_CR3_RUN_NAME__": "run_name",
}
LORA_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def validate_config(config: dict[str, Any], *, profile: str | None = None) -> None:
    job = config.get("job", {})
    model = config.get("model", {})
    trainer = config.get("trainer", {})
    checkpoint = config.get("checkpoint", {})
    optimizer = config.get("optimizer", {})
    custom = config.get("custom", {})
    if job.get("task") != "vlm" or job.get("experiment") != "tao_cr3_aoi":
        raise ValueError("Framework CR3 config must select job.task=vlm and experiment=tao_cr3_aoi")
    if model.get("precision") != "bfloat16":
        raise ValueError("Framework CR3 config must use BF16")
    if not model.get("lora_enabled"):
        raise ValueError("Framework CR3 config must enable native VLM LoRA")
    if model.get("lora_target_modules") != LORA_TARGETS:
        raise ValueError("Framework CR3 config must use the reviewed language-side LoRA targets")
    if model.get("parallelism", {}).get("data_parallel_replicate_degree") != 1:
        raise ValueError("Framework CR3 config must remain single-node")
    if trainer.get("distributed_parallelism") != "fsdp":
        raise ValueError("Framework CR3 config must use Framework FSDP2")
    if optimizer.get("keys_to_select") != ["lora_"]:
        raise ValueError("Framework CR3 optimizer must select LoRA parameters only")
    if checkpoint.get("save_iter", 0) < 1 or checkpoint.get("dcp_async_mode_enabled") is not False:
        raise ValueError("Framework CR3 config must use a positive synchronous DCP save cadence")
    if checkpoint.get("keys_to_skip_loading"):
        raise ValueError("Framework CR3 resume must restore native LoRA keys")
    if not isinstance(custom.get("seed"), int):
        raise ValueError("Framework CR3 config must record an integer seed")
    if custom.get("fsdp_master_dtype") != "bfloat16":
        raise ValueError("Framework CR3 config must request a BF16 FSDP master dtype")
    if custom.get("model_max_length") != 40960:
        raise ValueError("Framework CR3 config must keep model_max_length=40960")
    if profile == "smoke" and not 1 <= int(trainer.get("max_iter", 0)) <= 50:
        raise ValueError("smoke profile must run between 1 and 50 optimizer steps")


def load_toml(path: pathlib.Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def render_profile(
    profile: str,
    *,
    model_path: str,
    annotation_path: str,
    media_root: str,
    run_name: str,
) -> str:
    template = PROFILES[profile].read_text(encoding="utf-8")
    values = locals()
    for marker, name in MARKERS.items():
        template = template.replace(json.dumps(marker), json.dumps(values[name]))
    unresolved = [marker for marker in MARKERS if marker in template]
    if unresolved:
        raise ValueError(f"unresolved Framework TOML markers: {unresolved}")
    parsed = tomllib.loads(template)
    validate_config(parsed, profile=profile)
    return template


def atomic_write(path: pathlib.Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--model-path", required=True, help="Compute-frame HF Qwen3-VL snapshot path")
    parser.add_argument("--annotation-path", required=True, help="Compute-frame Train JSON-array path")
    parser.add_argument("--media-root", required=True, help="Compute-frame base for relative image paths")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rendered = render_profile(
            args.profile,
            model_path=args.model_path,
            annotation_path=args.annotation_path,
            media_root=args.media_root,
            run_name=args.run_name,
        )
        atomic_write(args.output.expanduser().resolve(), rendered)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"render_cfw_sft: {exc}", file=os.sys.stderr)
        return 2
    print(args.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
