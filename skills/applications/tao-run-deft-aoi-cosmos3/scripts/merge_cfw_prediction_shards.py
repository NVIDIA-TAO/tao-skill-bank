#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Merge distributed CFW prediction shards into canonical source order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import tempfile
from typing import Any

from cfw_predictions import (
    _atomic_jsonl,
    normalize_prediction,
    read_jsonl,
    validate_prediction_rows,
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def merge(
    *,
    source: pathlib.Path,
    shard_dir: pathlib.Path,
    expected_shards: int,
    output: pathlib.Path,
    summary_output: pathlib.Path,
) -> dict[str, Any]:
    if expected_shards <= 0:
        raise ValueError("expected_shards must be positive")
    source = source.expanduser().resolve(strict=True)
    shard_dir = shard_dir.expanduser().resolve(strict=True)
    expected_names = {
        f"predictions_rank{rank}.jsonl" for rank in range(expected_shards)
    }
    actual_paths = sorted(shard_dir.glob("predictions_rank*.jsonl"))
    actual_names = {path.name for path in actual_paths}
    if actual_names != expected_names:
        raise ValueError(
            "prediction shard set mismatch: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )

    source_rows = read_jsonl(source)
    source_by_id = {str(row.get("id", "")): row for row in source_rows}
    if "" in source_by_id or len(source_by_id) != len(source_rows):
        raise ValueError("source JSONL has missing or duplicate IDs")

    raw_rows = [row for path in actual_paths for row in read_jsonl(path)]
    raw_by_id = {str(row.get("id", "")): row for row in raw_rows}
    if "" in raw_by_id or len(raw_by_id) != len(raw_rows):
        raise ValueError("prediction shards have missing or duplicate IDs")
    missing = sorted(source_by_id.keys() - raw_by_id.keys())
    unknown = sorted(raw_by_id.keys() - source_by_id.keys())
    if missing or unknown:
        raise ValueError(
            f"prediction coverage mismatch: missing={missing[:10]}, unknown={unknown[:10]}"
        )

    normalized = validate_prediction_rows(
        [normalize_prediction(row, raw_by_id[row["id"]]) for row in source_rows]
    )
    output = output.expanduser().resolve()
    summary_output = summary_output.expanduser().resolve()
    _atomic_jsonl(output, normalized)
    payload = {
        "schema_version": 1,
        "state": "COMPLETE",
        "source": str(source),
        "shard_dir": str(shard_dir),
        "shards": expected_shards,
        "rows": len(normalized),
        "output": str(output),
        "output_sha256": _sha256(output),
        "missing_ids": 0,
        "unknown_ids": 0,
    }
    _atomic_json(summary_output, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--shard-dir", type=pathlib.Path, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--summary-output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = merge(
            source=args.source,
            shard_dir=args.shard_dir,
            expected_shards=args.expected_shards,
            output=args.output,
            summary_output=args.summary_output,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"merge_cfw_prediction_shards: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
