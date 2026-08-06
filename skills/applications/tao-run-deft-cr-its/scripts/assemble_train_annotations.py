#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble per-iteration LLaVA training annotations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from workflow_common import absolute_path, load_json_array, normalize_media_path, write_json_array


def annotation_id(record: dict[str, Any], source: Path, index: int) -> str:
    """Return a required LLaVA annotation id."""
    value = record.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}: item {index} is missing non-empty 'id'")
    return value


def absolute_video(record: dict[str, Any], source: Path, media_dir: Path | None) -> dict[str, Any]:
    """Return a copied LLaVA record with an absolute video path."""
    video = record.get("video")
    if not isinstance(video, str) or not video:
        raise ValueError(f"{source}: annotation {record.get('id')!r} is missing non-empty 'video'")
    if Path(video).is_absolute():
        video_path = Path(normalize_media_path(video))
    else:
        if media_dir is None:
            raise ValueError(
                f"{source}: annotation {record.get('id')!r} has relative video path {video!r}; "
                "provide its media directory"
            )
        video_path = Path(normalize_media_path(str(media_dir / video)))
    copied = dict(record)
    copied["video"] = str(video_path)
    return copied


def assemble_annotations(
    previous_annotations: Path | None,
    current_annotations: list[Path],
    previous_media_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Merge prior/seed and current derived annotations, deduped by LLaVA id."""
    sources: list[tuple[Path, Path | None, bool]] = []
    if previous_annotations is not None:
        sources.append((previous_annotations, previous_media_dir, False))
    sources.extend((path, None, True) for path in current_annotations if path)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    current_count = 0
    for source, media_dir, is_current in sources:
        if not source.is_file():
            raise FileNotFoundError(f"annotation source does not exist: {source}")
        source_records = load_json_array(source)
        if is_current:
            current_count += len(source_records)
        for index, record in enumerate(source_records, start=1):
            record_id = annotation_id(record, source, index)
            if record_id in seen:
                continue
            records.append(absolute_video(record, source, media_dir))
            seen.add(record_id)
    if current_count == 0:
        raise RuntimeError("no new mined or generated annotations were available for this iteration")
    return records


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-annotations", type=Path)
    parser.add_argument("--previous-media-dir", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--current-annotations", action="append", default=[], type=Path)
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    records = assemble_annotations(
        absolute_path(args.previous_annotations) if args.previous_annotations else None,
        [absolute_path(path) for path in args.current_annotations],
        absolute_path(args.previous_media_dir) if args.previous_media_dir else None,
    )
    output_path = absolute_path(args.output_json)
    write_json_array(output_path, records)
    print(f"Wrote {len(records)} merged LLaVA records -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
