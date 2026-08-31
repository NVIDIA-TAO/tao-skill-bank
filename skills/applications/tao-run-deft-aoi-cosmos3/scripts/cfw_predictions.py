#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Normalize Cosmos Framework evaluate/inference rows for the exact evaluator."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import sys
import tempfile
from typing import Any


PREDICTION_FIELDS = ("raw_prediction", "prediction", "generated_text", "output")
PREDICTION_SCHEMA = ("id", "task_type", "message", "GT", "raw_prediction")


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item["text"]
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return ""


def normalize_prediction(source: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    row_id = source.get("id")
    if not isinstance(row_id, str) or not row_id:
        raise ValueError("source prediction row requires a non-empty id")
    task_type = source.get("task_type")
    if not isinstance(task_type, str) or not task_type:
        raise ValueError(f"source row {row_id!r} requires task_type")
    messages = source.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"source row {row_id!r} requires messages")
    prompt_messages: list[dict[str, Any]] = []
    ground_truth: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"source row {row_id!r} contains a non-object message")
        if message.get("role") == "assistant":
            value = _text(message.get("content"))
            if value:
                ground_truth.append(value)
        else:
            prompt_messages.append(copy.deepcopy(message))
    if not ground_truth:
        raise ValueError(f"source row {row_id!r} has no assistant ground truth")
    prediction: Any = None
    for field in PREDICTION_FIELDS:
        if field in raw and raw[field] is not None:
            prediction = raw[field]
            break
    if prediction is None:
        raise ValueError(f"Framework result for {row_id!r} has no prediction field")
    if not isinstance(prediction, str):
        prediction = json.dumps(prediction, ensure_ascii=False, separators=(",", ":"))
    return {
        "id": row_id,
        "task_type": task_type,
        "message": prompt_messages,
        "GT": "\n".join(ground_truth).strip(),
        "raw_prediction": prediction,
    }


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def validate_prediction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("normalized prediction JSONL is empty")
    seen: set[str] = set()
    expected = set(PREDICTION_SCHEMA)
    for index, row in enumerate(rows):
        if set(row) != expected:
            raise ValueError(
                f"normalized prediction row {index} must contain exactly {PREDICTION_SCHEMA}"
            )
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id or row_id in seen:
            raise ValueError(
                f"normalized prediction row {index} has a missing or duplicate id"
            )
        seen.add(row_id)
        if not isinstance(row.get("task_type"), str) or not row["task_type"]:
            raise ValueError(f"normalized prediction row {index} has invalid task_type")
        if not isinstance(row.get("message"), list):
            raise ValueError(f"normalized prediction row {index} has invalid message")
        if not isinstance(row.get("GT"), str) or not isinstance(
            row.get("raw_prediction"), str
        ):
            raise ValueError(
                f"normalized prediction row {index} requires string GT/raw_prediction"
            )
    return rows


def read_prediction_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return validate_prediction_rows(read_jsonl(path))


def normalize_files(
    source_path: pathlib.Path,
    raw_path: pathlib.Path,
    *,
    allow_single_positional: bool = False,
) -> list[dict[str, Any]]:
    source_rows = read_jsonl(source_path)
    raw_rows = read_jsonl(raw_path)
    source_by_id = {str(row.get("id", "")): row for row in source_rows}
    if (
        allow_single_positional
        and len(source_rows) == 1
        and len(raw_rows) == 1
        and not raw_rows[0].get("id")
    ):
        raw_rows[0] = {**raw_rows[0], "id": source_rows[0].get("id")}
    raw_by_id = {str(row.get("id", "")): row for row in raw_rows}
    if "" in source_by_id or len(source_by_id) != len(source_rows):
        raise ValueError("source JSONL has missing or duplicate IDs")
    if "" in raw_by_id or len(raw_by_id) != len(raw_rows):
        raise ValueError("Framework result JSONL has missing or duplicate IDs")
    missing = sorted(source_by_id.keys() - raw_by_id.keys())
    unknown = sorted(raw_by_id.keys() - source_by_id.keys())
    if missing or unknown:
        raise ValueError(f"prediction coverage mismatch: missing={missing[:10]}, unknown={unknown[:10]}")
    return validate_prediction_rows(
        [normalize_prediction(row, raw_by_id[row["id"]]) for row in source_rows]
    )


def _atomic_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--framework-results", type=pathlib.Path, required=True)
    parser.add_argument(
        "--action", choices=("evaluate", "inference"), default="evaluate"
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    try:
        rows = normalize_files(
            args.source.expanduser().resolve(strict=True),
            args.framework_results.expanduser().resolve(strict=True),
            allow_single_positional=args.action == "inference",
        )
        _atomic_jsonl(args.output.expanduser().resolve(), rows)
    except (OSError, ValueError) as exc:
        print(f"cfw_predictions: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
