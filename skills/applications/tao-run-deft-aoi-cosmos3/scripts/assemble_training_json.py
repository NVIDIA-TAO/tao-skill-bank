#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Assemble monotonic, real-mining-only NVPAW training JSONL."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from typing import Any

from validate_sharegpt import load_records, target_path


def _fingerprint(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def assemble(
    previous_path: pathlib.Path | None,
    mined_path: pathlib.Path,
    *,
    validation_paths: list[pathlib.Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mined = load_records(mined_path)
    if not mined:
        raise ValueError("the current iteration must contribute at least one mined record")
    previous = load_records(previous_path) if previous_path is not None else []
    evaluation_targets: dict[str, str] = {}
    for path in validation_paths:
        for index, record in enumerate(load_records(path)):
            evaluation_targets[target_path(record, context=f"{path}:{index}")] = str(path)

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_count = 0
    tasks: Counter[str] = Counter()
    provenance: list[dict[str, Any]] = []
    for source_kind, source_path, records in (
        ("previous_iteration", previous_path, previous),
        ("current_mining", mined_path, mined),
    ):
        if source_path is None:
            continue
        for index, record in enumerate(records):
            target = target_path(record, context=f"{source_path}:{index}")
            if target in evaluation_targets:
                raise ValueError(
                    f"train/evaluation leakage: target {target!r} also occurs in "
                    f"{evaluation_targets[target]}"
                )
            key = _fingerprint(record)
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            merged.append(record)
            tasks[str(record.get("task_type", "unknown"))] += 1
            provenance.append(
                {
                    "source_kind": source_kind,
                    "source": str(source_path),
                    "source_index": index,
                    "id": record.get("id"),
                }
            )
    if not merged:
        raise ValueError("real-mining assembly produced no training records")
    mined_fingerprints = {_fingerprint(record) for record in mined}
    if not any(_fingerprint(record) in mined_fingerprints for record in merged):
        raise ValueError("current mined records were all lost during assembly")
    return merged, {
        "schema_version": 1,
        "format": "jsonl",
        "annotation_profile": "nvpaw_multitask_v1",
        "training_source": "mined_real_samples_only",
        "previous_iteration": str(previous_path) if previous_path else None,
        "mined_input": str(mined_path),
        "validation_inputs": [str(path) for path in validation_paths],
        "previous_records": len(previous),
        "mined_records": len(mined),
        "output_records": len(merged),
        "duplicates_skipped": duplicate_count,
        "tasks": dict(sorted(tasks.items())),
        "provenance": provenance,
    }


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-jsonl", type=pathlib.Path)
    parser.add_argument("--mined-jsonl", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--summary", type=pathlib.Path)
    parser.add_argument("--validation-jsonl", action="append", default=[], type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        rows, summary = assemble(
            args.previous_jsonl,
            args.mined_jsonl,
            validation_paths=args.validation_jsonl,
        )
        _write_jsonl(args.output, rows)
        summary_path = args.summary or args.output.with_name("assemble_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"assemble_training_json: {exc}", file=sys.stderr)
        return 2
    print(f"assemble_training_json: wrote {len(rows)} real records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
