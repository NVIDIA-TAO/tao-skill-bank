#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Print packaged TAO AutoML model support without scanning model folders."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from list_tao_models import build_automl_support, format_automl_text


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skill-bank",
        type=Path,
        default=Path(os.environ.get("TAO_SKILL_BANK_PATH", Path.home() / "tao-skills-external")),
        help="Path to the packaged TAO skill bank.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="text",
        help="Output format.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the support summary helper."""
    args = parse_args()
    support = build_automl_support(args.skill_bank)
    if args.format == "json":
        print(json.dumps(support, indent=2, sort_keys=True))
    else:
        print(format_automl_text(support))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
