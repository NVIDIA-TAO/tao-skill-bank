#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create and validate a TAO Data Services image-embeddings spec."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

VALID_MODELS = {"CLIP", "SigLIP"}
TAO_CKPT_SUFFIXES = {".pth", ".ckpt"}


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required: install with `python3 -m pip install pyyaml`.")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required: install with `python3 -m pip install pyyaml`.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def absolute_path(raw: str, *, must_exist: bool, kind: str) -> str:
    path = Path(raw).expanduser().resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{kind} does not exist: {path}")
    return str(path)


def validate_config(config: dict[str, Any]) -> None:
    required = ("input_parquet", "output_parquet", "model", "model_path")
    missing = [k for k in required if not config.get(k)]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    src = Path(str(config["input_parquet"])).expanduser()
    if not src.is_absolute():
        raise ValueError(f"input_parquet must be absolute: {src}")
    if not src.is_file():
        raise FileNotFoundError(f"input_parquet does not exist or is not a file: {src}")

    out = Path(str(config["output_parquet"])).expanduser()
    if not out.is_absolute():
        raise ValueError(f"output_parquet must be absolute: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        probe = out.parent / ".tao_image_embeddings_write_test"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise PermissionError(f"output_parquet parent is not writable: {out.parent}") from exc

    model = str(config["model"])
    if model not in VALID_MODELS:
        raise ValueError(f"model must be one of {sorted(VALID_MODELS)}.")
    config["model"] = model

    # A TAO checkpoint cannot be rebuilt without its training spec.
    model_path = str(config["model_path"])
    if Path(model_path).suffix in TAO_CKPT_SUFFIXES and not config.get("model_config_path"):
        raise ValueError(
            "model_config_path is required when model_path is a TAO checkpoint "
            f"({Path(model_path).suffix})."
        )

    batch_size = int(config.get("batch_size", 64))
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    config["batch_size"] = batch_size
    config["model_config_path"] = str(config.get("model_config_path", ""))


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    skill_dir = Path(__file__).resolve().parents[1]
    default_template = skill_dir / "assets" / "default_image_embeddings.yaml"
    template = Path(args.template).expanduser().resolve() if args.template else default_template
    config = load_yaml(template)
    config.update({
        "input_parquet": absolute_path(args.input_parquet, must_exist=True, kind="input_parquet"),
        "output_parquet": absolute_path(args.output_parquet, must_exist=False, kind="output_parquet"),
        "model": args.model,
        "model_path": args.model_path,
        "batch_size": args.batch_size,
    })
    if args.model_config_path:
        config["model_config_path"] = absolute_path(
            args.model_config_path, must_exist=True, kind="model_config_path"
        )
    validate_config(config)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", required=True, help="Absolute path to the parquet of image filepaths.")
    parser.add_argument("--output-parquet", required=True, help="Absolute path for the embedding parquet.")
    parser.add_argument("--output-spec", required=True, help="Path where the generated YAML should be written.")
    parser.add_argument("--template", help="Optional YAML template. Defaults to assets/default_image_embeddings.yaml.")
    parser.add_argument("--model", default="SigLIP", choices=sorted(VALID_MODELS))
    parser.add_argument("--model-path", default="google/siglip-base-patch16-224",
                        help="HF id, local HF snapshot dir, or TAO .pth/.ckpt checkpoint.")
    parser.add_argument("--model-config-path", default=None,
                        help="TAO experiment spec. Required when --model-path is a TAO checkpoint.")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        config = build_config(args)
        output_spec = Path(args.output_spec).expanduser().resolve()
        write_yaml(output_spec, config)
        print(f"Wrote image-embeddings spec: {output_spec}")
        print(f"Encoder: {config['model']} @ {config['model_path']}")
        print(f"Expected output parquet: {config['output_parquet']}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
