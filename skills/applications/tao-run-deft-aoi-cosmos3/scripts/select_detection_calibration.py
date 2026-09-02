#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Prepend empty/few-box Mining examples to routed candidates for calibration."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Iterable, Iterator
from typing import Any

from validate_sharegpt import prompt_and_response, resolve_image, target_path


DETECTION_TASKS = {
    "Component Detection",
    "Defect Detection",
    "Ref_based Defect Detection",
}


def _json_answer(text: str, *, context: str) -> Any:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            value = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{context}: detection answer is not JSON") from exc


def select_calibration(
    records: Iterable[dict[str, Any]],
    *,
    media_root: pathlib.Path,
    max_empty: int,
    max_few: int,
    max_boxes: int = 2,
    excluded_identities: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if min(max_empty, max_few) < 0 or max_boxes < 1:
        raise ValueError("calibration quotas must be non-negative and max_boxes positive")
    if max_empty + max_few <= 0:
        raise ValueError("at least one calibration quota must be positive")
    media_root = media_root.expanduser().resolve()
    excluded = {str(pathlib.Path(value).expanduser().resolve()) for value in (excluded_identities or set())}
    empty: list[dict[str, Any]] = []
    few: list[dict[str, Any]] = []
    seen: set[str] = set()
    examined_detection = 0
    excluded_many = 0
    excluded_previously_mined = 0
    for index, record in enumerate(records):
        task_type = record.get("task_type")
        if task_type not in DETECTION_TASKS:
            continue
        examined_detection += 1
        _, answer = prompt_and_response(record, context=f"calibration record[{index}]")
        boxes = _json_answer(answer, context=f"calibration record[{index}]")
        if not isinstance(boxes, list):
            raise ValueError(f"calibration record[{index}]: detection answer must be a list")
        if len(boxes) > max_boxes:
            excluded_many += 1
            continue
        filepath = target_path(record, context=f"calibration record[{index}]")
        identity = str(resolve_image(filepath, media_root))
        if identity in excluded:
            excluded_previously_mined += 1
            continue
        if identity in seen:
            continue
        destination = empty if not boxes else few
        quota = max_empty if not boxes else max_few
        if len(destination) >= quota:
            continue
        seen.add(identity)
        destination.append(
            {
                "filepath": filepath,
                "route_tier": "calibration",
                "route_tiers": ["calibration"],
                "routed_task_types": [str(task_type)],
                "calibration_box_count": len(boxes),
                "calibration_record_id": record.get("id"),
            }
        )
        if len(empty) >= max_empty and len(few) >= max_few:
            break
    selected = [*empty, *few]
    if not selected:
        raise ValueError("no empty or few-box Mining calibration examples were found")
    return selected, {
        "schema_version": "detection_calibration_v1",
        "policy": "empty_and_few_box_from_mining",
        "max_boxes": max_boxes,
        "requested_empty": max_empty,
        "requested_few_box": max_few,
        "selected_empty": len(empty),
        "selected_few_box": len(few),
        "selected_total": len(selected),
        "examined_detection_records": examined_detection,
        "excluded_many_box": excluded_many,
        "excluded_previously_mined": excluded_previously_mined,
    }


def _stream_records(path: pathlib.Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            yield value


def _routed_rows(path: pathlib.Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("pyarrow is required to merge routed candidates") from exc
    return pq.read_table(path).to_pylist()


def merge_candidates(
    calibration: list[dict[str, Any]],
    routed: list[dict[str, Any]],
    *,
    media_root: pathlib.Path,
) -> tuple[list[dict[str, Any]], int]:
    output: list[dict[str, Any]] = []
    indexes: dict[str, int] = {}
    duplicates = 0
    for row in [*calibration, *routed]:
        filepath = row.get("filepath")
        if not isinstance(filepath, str) or not filepath:
            raise ValueError("every calibration/routed candidate requires filepath")
        identity = str(resolve_image(filepath, media_root))
        tasks = row.get("routed_task_types") or row.get("source_task_types")
        if not isinstance(tasks, (list, tuple)) or not tasks:
            raise ValueError(f"candidate {filepath!r} requires routed_task_types")
        tier = str(row.get("route_tier") or "strict")
        if identity not in indexes:
            indexes[identity] = len(output)
            output.append(
                {
                    "filepath": filepath,
                    "route_tier": tier,
                    "route_tiers": sorted(set(row.get("route_tiers") or [tier])),
                    "routed_task_types": sorted(set(str(item) for item in tasks)),
                }
            )
            continue
        duplicates += 1
        existing = output[indexes[identity]]
        existing["route_tiers"] = sorted(set(existing["route_tiers"]) | {tier})
        existing["routed_task_types"] = sorted(
            set(existing["routed_task_types"]) | {str(item) for item in tasks}
        )
        if "calibration" in existing["route_tiers"]:
            existing["route_tier"] = "calibration"
    return output, duplicates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-annotations", required=True, type=pathlib.Path)
    parser.add_argument("--media-root", required=True, type=pathlib.Path)
    parser.add_argument("--routed-candidates", type=pathlib.Path)
    parser.add_argument("--max-empty", type=int, required=True)
    parser.add_argument("--max-few", type=int, required=True)
    parser.add_argument("--max-boxes", type=int, default=2)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--summary", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        calibration, summary = select_calibration(
            _stream_records(args.source_annotations),
            media_root=args.media_root,
            max_empty=args.max_empty,
            max_few=args.max_few,
            max_boxes=args.max_boxes,
        )
        merged, duplicates = merge_candidates(
            calibration,
            _routed_rows(args.routed_candidates),
            media_root=args.media_root,
        )
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ValueError("pyarrow is required to write calibration candidates") from exc
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(merged), args.output)
        summary.update(
            {
                "routed_input": str(args.routed_candidates) if args.routed_candidates else None,
                "routed_records": len(merged) - len(calibration) + duplicates,
                "combined_unique_candidates": len(merged),
                "duplicates_merged": duplicates,
                "output": str(args.output),
            }
        )
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"select_detection_calibration: {exc}", file=sys.stderr)
        return 2
    print(
        "select_detection_calibration: "
        f"calibration={summary['selected_total']} combined={summary['combined_unique_candidates']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
