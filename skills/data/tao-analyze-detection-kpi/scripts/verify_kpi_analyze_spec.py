#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate an existing TAO Data Services detection KPI-analyze spec."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prepare_kpi_analyze_spec import load_yaml, validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Path to kpi-analyze YAML.")
    return parser.parse_args()


def main() -> int:
    try:
        spec = Path(parse_args().spec).expanduser().resolve()
        config = load_yaml(spec)
        validate_config(config)
        print(f"OK: kpi-analyze spec is valid: {spec}")
        print(f"Sources: {len(config['data']['kpi_sources'])} | format: {config['data']['input_format']}")
        print(f"Expected output: {config['results_dir']}/kpi_calc.csv")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
