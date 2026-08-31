#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collapse record-level selected gaps into target-level mining queries."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from typing import Any


def route(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        raise ValueError("selected gaps are empty")
    targets: dict[str, dict[str, Any]] = {}
    task_counts: Counter[str] = Counter()
    for row in rows:
        if row.get("evaluation_role") != "proxy":
            raise ValueError("routing accepts only Proxy selected gaps")
        record_id = row.get("id")
        target_id = row.get("target_id")
        target_path = row.get("target_path")
        task_type = row.get("task_type")
        if not all(isinstance(value, str) and value for value in (record_id, target_id, target_path, task_type)):
            raise ValueError("every selected gap requires id, target_id, target_path, and task_type")
        task_counts[task_type] += 1
        target = targets.setdefault(
            target_id,
            {
                "filepath": target_path,
                "target_id": target_id,
                "record_ids": [],
                "task_types": [],
                "datasets": [],
                "mining_eligible": True,
            },
        )
        if target["filepath"] != target_path:
            raise ValueError(
                f"target_id {target_id!r} maps to conflicting target paths"
            )
        target["record_ids"].append(record_id)
        if task_type not in target["task_types"]:
            target["task_types"].append(task_type)
        dataset = str(row.get("dataset", "unknown"))
        if dataset not in target["datasets"]:
            target["datasets"].append(dataset)
    output = []
    for target in targets.values():
        target["record_ids"].sort()
        target["task_types"].sort()
        target["datasets"].sort()
        output.append(target)
    output.sort(key=lambda row: (row["filepath"], row["target_id"]))
    return output, {
        "schema_version": "nvpaw_routing_v1",
        "selected_records": len(rows),
        "unique_targets": len(output),
        "embedding_queries": len(output),
        "task_records": dict(sorted(task_counts.items())),
        "mining_eligible_records": len(rows),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-gaps", required=True, type=pathlib.Path)
    parser.add_argument("--output-json", required=True, type=pathlib.Path)
    parser.add_argument("--output-parquet", required=True, type=pathlib.Path)
    parser.add_argument("--summary", required=True, type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = pq.read_table(args.selected_gaps).to_pylist()
        targets, summary = route(rows)
        for path in (args.output_json, args.output_parquet, args.summary):
            path.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(targets, indent=2, sort_keys=True) + "\n")
        pq.write_table(pa.Table.from_pylist(targets), args.output_parquet)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    except (ImportError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"route_selected_gaps: {exc}", file=sys.stderr)
        return 2
    print(
        f"route_selected_gaps: records={summary['selected_records']} "
        f"targets={summary['unique_targets']} output={args.output_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
