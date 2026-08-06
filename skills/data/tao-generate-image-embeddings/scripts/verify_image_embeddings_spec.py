#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate an existing TAO Data Services image-embeddings spec."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prepare_image_embeddings_spec import load_yaml, validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Path to image-embeddings YAML.")
    return parser.parse_args()


def main() -> int:
    try:
        spec = Path(parse_args().spec).expanduser().resolve()
        config = load_yaml(spec)
        validate_config(config)
        print(f"OK: image-embeddings spec is valid: {spec}")
        print(f"Encoder: {config['model']} @ {config['model_path']}")
        print(f"Output parquet: {config['output_parquet']}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
