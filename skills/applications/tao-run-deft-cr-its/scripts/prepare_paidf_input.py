#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert VLM BCQ gaps to the generic PAIDF Cosmos Predict media JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from workflow_common import absolute_path, normalize_media_path, read_jsonl, write_jsonl


def gap_media_path(record: dict[str, Any], source: Path, index: int) -> str:
    """Return the required absolute source-video path for one gap."""
    value = record.get("original_video_path") or record.get("video_id")
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{source}: record {index} must include non-empty "
            "'original_video_path' or 'video_id'"
        )
    path = normalize_media_path(value)
    if not Path(path).is_absolute():
        raise ValueError(f"{source}: record {index} media path must be absolute: {value}")
    return path


def stable_gap_id(media_path: str, question: str) -> str:
    """Derive a deterministic id for one media/question weak sample."""
    payload = json.dumps([media_path, question], ensure_ascii=False, separators=(",", ":"))
    return "gap_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def gap_entries(gaps_path: Path) -> list[dict[str, str]]:
    """Return validated, deduplicated weak-sample records used around PAIDF."""
    rows: list[dict[str, str]] = []
    seen: dict[str, dict[str, str]] = {}
    for index, record in enumerate(read_jsonl(gaps_path), start=1):
        question = record.get("question")
        answer = record.get("ground_truth")
        if not isinstance(question, str) or not question:
            raise ValueError(f"{gaps_path}: record {index} is missing non-empty 'question'")
        if not isinstance(answer, str) or not answer:
            raise ValueError(f"{gaps_path}: record {index} is missing non-empty 'ground_truth'")
        media_path = gap_media_path(record, gaps_path, index)
        row = {
            "id": stable_gap_id(media_path, question),
            "media_path": media_path,
            "question": question,
            "answer": answer,
        }
        previous = seen.get(row["id"])
        if previous is None:
            seen[row["id"]] = row
            rows.append(row)
        elif previous != row:
            raise ValueError(f"{gaps_path}: conflicting duplicate gap id {row['id']}")
    return rows


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaps-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    """Write generic PAIDF input rows while retaining gap context indirectly by id."""
    args = parse_args()
    gaps_path = absolute_path(args.gaps_jsonl)
    output_path = absolute_path(args.output_jsonl)
    rows = [{"id": row["id"], "media_path": row["media_path"]} for row in gap_entries(gaps_path)]
    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} PAIDF input rows -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
