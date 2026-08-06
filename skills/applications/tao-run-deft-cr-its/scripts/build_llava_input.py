#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Join BCQ gaps with successful PAIDF videos for LLaVA conversion."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from prepare_paidf_input import gap_entries
from workflow_common import absolute_path, normalize_media_path, read_jsonl, write_jsonl


def generated_video_mappings(path: Path) -> dict[str, dict[str, str]]:
    """Read successful PAIDF rows keyed by workflow gap id."""
    mappings: dict[str, dict[str, str]] = {}
    required = ("id", "original_media_path", "generated_video_path")
    for index, record in enumerate(read_jsonl(path), start=1):
        for field in required:
            if not isinstance(record.get(field), str) or not record[field]:
                raise ValueError(f"{path}: record {index} is missing non-empty {field!r}")
        mapping = {
            "id": record["id"],
            "original_media_path": normalize_media_path(record["original_media_path"]),
            "generated_video_path": normalize_media_path(record["generated_video_path"]),
        }
        previous = mappings.get(mapping["id"])
        if previous is not None and previous != mapping:
            raise ValueError(f"{path}: conflicting duplicate generated-video id {mapping['id']}")
        mappings[mapping["id"]] = mapping
    return mappings


def build_records(gaps_path: Path, generated_videos_path: Path) -> tuple[list[dict[str, Any]], int]:
    """Build converter rows for successful generations and count failed gaps."""
    mappings = generated_video_mappings(generated_videos_path)
    output: list[dict[str, Any]] = []
    skipped = 0
    for gap in gap_entries(gaps_path):
        mapping = mappings.get(gap["id"])
        if mapping is None:
            skipped += 1
            continue
        if mapping["original_media_path"] != gap["media_path"]:
            raise ValueError(
                f"{generated_videos_path}: media path mismatch for {gap['id']}: "
                f"{mapping['original_media_path']!r} != {gap['media_path']!r}"
            )
        output.append(
            {
                "id": gap["id"],
                "video_path": mapping["generated_video_path"],
                "question": gap["question"],
                "answer": gap["answer"],
            }
        )
    return output, skipped


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaps-jsonl", required=True, type=Path)
    parser.add_argument("--generated-videos-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    """Write the JSONL consumed by TAO Data Services LLaVA conversion."""
    args = parse_args()
    output_path = absolute_path(args.output_jsonl)
    records, skipped = build_records(
        absolute_path(args.gaps_jsonl),
        absolute_path(args.generated_videos_jsonl),
    )
    write_jsonl(output_path, records)
    print(f"Wrote {len(records)} LLaVA converter rows -> {output_path}")
    print(f"Skipped gaps without generated videos: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
