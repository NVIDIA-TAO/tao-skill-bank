#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble per-iteration LLaVA training annotations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from workflow_common import absolute_path, load_json_array, write_json_array


def annotation_id(record: dict[str, Any], source: Path, index: int) -> str:
    """Return a required LLaVA annotation id."""
    value = record.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}: item {index} is missing non-empty 'id'")
    return value


def assemble_annotations(previous_annotations: Path | None, mined_annotations: list[Path]) -> list[dict[str, Any]]:
    """Concatenate optional previous and mined annotation sources, deduped by LLaVA id."""
    sources = []
    if previous_annotations is not None:
        sources.append(previous_annotations)
    sources.extend(path for path in mined_annotations if path)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"annotation source does not exist: {source}")
        for index, record in enumerate(load_json_array(source), start=1):
            record_id = annotation_id(record, source, index)
            if record_id in seen:
                continue
            records.append(record)
            seen.add(record_id)
    return records


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-annotations", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--mined-annotations", action="append", default=[], type=Path)
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    records = assemble_annotations(
        absolute_path(args.previous_annotations) if args.previous_annotations else None,
        [absolute_path(path) for path in args.mined_annotations],
    )
    output_path = absolute_path(args.output_json)
    write_json_array(output_path, records)
    print(f"Wrote {len(records)} merged LLaVA records -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
