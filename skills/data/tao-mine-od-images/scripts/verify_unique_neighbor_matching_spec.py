#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate an existing TAO Data Services TMM unique-neighbor-matching spec."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prepare_unique_neighbor_matching_spec import load_yaml, validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Path to unique-neighbor-matching YAML.")
    return parser.parse_args()


def main() -> int:
    try:
        spec = Path(parse_args().spec).expanduser().resolve()
        config = load_yaml(spec)
        validate_config(config)
        print(f"OK: unique-neighbor-matching spec is valid: {spec}")
        print(f"Output directory: {config['output_dir']}")
        print(f"Expected artifacts: final_unique_files.parquet, summary.json")
        out_dir = Path(str(config["output_dir"])).expanduser().resolve()
        if spec.parent != out_dir:
            print(f"WARNING: the spec is outside output_dir ({spec.parent} vs {out_dir}). The "
                  "run does not copy it, so the mined set will carry no record of the budget, "
                  "allocation policy or rare-class list that selected it. Author it at "
                  f"{out_dir / spec.name} instead.", file=sys.stderr)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
