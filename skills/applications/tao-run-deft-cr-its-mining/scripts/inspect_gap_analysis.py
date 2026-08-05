#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect completed gap-analysis output and persist its loop-control status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from workflow_common import absolute_path, atomic_write_json


def count_gap_rows(gaps_jsonl: Path) -> int:
    """Count valid JSON objects, treating an absent output as zero gaps."""
    if not gaps_jsonl.exists():
        return 0
    if not gaps_jsonl.is_file():
        raise FileNotFoundError(f"gaps JSONL is not a file: {gaps_jsonl}")

    count = 0
    with gaps_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{gaps_jsonl}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{gaps_jsonl}:{line_number}: expected a JSON object")
            count += 1
    return count


def write_status(gaps_jsonl: Path, status_json: Path) -> dict[str, Any]:
    """Write an atomic status record for KFP and agent loop control."""
    gaps_jsonl = absolute_path(gaps_jsonl)
    status_json = absolute_path(status_json)
    weak_sample_count = count_gap_rows(gaps_jsonl)
    payload = {
        "gaps_jsonl": str(gaps_jsonl),
        "has_gaps": weak_sample_count > 0,
        "weak_sample_count": weak_sample_count,
    }
    atomic_write_json(status_json, payload)
    return payload


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaps-jsonl", required=True, type=Path)
    parser.add_argument("--status-json", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    """Inspect one completed gap-analysis stage."""
    args = parse_args()
    status = write_status(args.gaps_jsonl, args.status_json)
    print(f"has_gaps: {str(status['has_gaps']).lower()}")
    print(f"weak_sample_count: {status['weak_sample_count']}")
    print(f"status_json: {absolute_path(args.status_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
