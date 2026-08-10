#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute ``desired_unique_count`` for the miner.

The reference pipeline fixes the mining budget across every iteration: it takes
the weak-image count from the **first** iteration and multiplies it by a
constant. Later iterations reuse that same number even as their own weak-image
count shrinks, so the amount of data added per iteration stays roughly constant
instead of decaying as the model improves.

Pass ``--weak-parquet`` pointing at iteration 1's weak-images parquet on every
iteration to reproduce that behavior; pass the current iteration's parquet
instead to let the budget track the shrinking gap set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-parquet", required=True,
                        help="Weak-images parquet whose row count seeds the budget.")
    parser.add_argument("--multiplier", type=int, required=True,
                        help="Budget multiplier applied to the weak-image count.")
    parser.add_argument("--min-count", type=int, default=1,
                        help="Floor for the resulting budget.")
    parser.add_argument("--max-count", type=int, default=None,
                        help="Optional ceiling, e.g. the source-pool size.")
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args()

    try:
        if args.multiplier < 1:
            raise ValueError("--multiplier must be at least 1.")

        weak_parquet = Path(args.weak_parquet).expanduser().resolve()
        if not weak_parquet.is_file():
            raise FileNotFoundError(f"--weak-parquet does not exist: {weak_parquet}")

        weak_count = int(len(pd.read_parquet(weak_parquet)))
        if weak_count == 0:
            print("ERROR: weak-images parquet has zero rows — nothing to mine for.", file=sys.stderr)
            return 1

        budget = weak_count * args.multiplier
        clamped_reason = None
        if budget < args.min_count:
            clamped_reason = f"raised to --min-count {args.min_count}"
            budget = args.min_count
        if args.max_count is not None and budget > args.max_count:
            clamped_reason = f"clamped to --max-count {args.max_count}"
            budget = args.max_count

        report = {
            "weak_parquet": str(weak_parquet),
            "weak_count": weak_count,
            "multiplier": args.multiplier,
            "desired_unique_count": budget,
            "clamped": clamped_reason,
        }
        if args.report_json:
            report_path = Path(args.report_json).expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)

        detail = f" ({clamped_reason})" if clamped_reason else ""
        print(f"weak_count={weak_count} x multiplier={args.multiplier} -> {budget}{detail}",
              file=sys.stderr)
        # stdout carries only the number, so callers can capture it directly.
        print(budget)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
