#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Align mined paths to source annotations and preserve compatible QA fan-out."""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
from collections import Counter
from typing import Any, Iterable

from nvpaw_annotations import TASK_SPECS
from validate_sharegpt import (
    image_paths,
    load_records,
    prompt_and_response,
    resolve_image,
    target_path,
)


def _path_keys(path_text: str, media_root: pathlib.Path) -> set[str]:
    normalized = path_text.replace("\\", "/").rstrip("/")
    resolved = str(resolve_image(path_text, media_root))
    return {normalized, resolved, pathlib.PurePosixPath(normalized).name}


def _load_mined_rows(path: pathlib.Path, column: str) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("pyarrow is required to read the mined parquet") from exc
    rows = pq.read_table(path).to_pylist()
    normalized = [
        {**row, "filepath": str(row[column])}
        for row in rows
        if row.get(column) is not None and str(row[column]).strip()
    ]
    if not normalized:
        raise ValueError(f"{path}: no mined paths in column {column!r}")
    return normalized


def _task_list(value: Any, *, context: str) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{context}: routed_task_types must be a string list") from exc
    if not isinstance(value, (list, tuple)) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{context}: routed_task_types must be a non-empty string list")
    return sorted(set(value))


def _normalize_mined_rows(
    values: Iterable[str | dict[str, Any]], media_root: pathlib.Path
) -> tuple[list[dict[str, Any]], int]:
    rows = list(values)
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for index, value in enumerate(rows):
        if isinstance(value, str):
            row: dict[str, Any] = {
                "filepath": value,
                "route_tier": "image_only",
            }
        elif isinstance(value, dict):
            row = dict(value)
        else:
            raise ValueError(f"mined row[{index}] must be a filepath string or object")
        filepath = row.get("filepath")
        if not isinstance(filepath, str) or not filepath:
            raise ValueError(f"mined row[{index}]: filepath is required")
        tier = row.get("route_tier", "image_only")
        if tier not in {"image_only", "strict", "fallback", "calibration"}:
            raise ValueError(f"mined row[{index}]: unsupported route_tier {tier!r}")
        routed = row.get("routed_task_types")
        if routed is not None:
            routed = _task_list(routed, context=f"mined row[{index}]")
        key = str(resolve_image(filepath, media_root))
        if key not in merged:
            merged[key] = {
                "filepath": filepath,
                "route_tier": tier,
                "route_tiers": [tier],
                "routed_task_types": routed,
            }
            order.append(key)
            continue
        existing = merged[key]
        existing["route_tiers"] = sorted(set(existing["route_tiers"]) | {tier})
        if tier in {"strict", "calibration"}:
            existing["route_tier"] = "strict"
        if tier == "calibration":
            existing["route_tier"] = "calibration"
        if routed is not None:
            existing["routed_task_types"] = sorted(
                set(existing.get("routed_task_types") or []) | set(routed)
            )
    return [merged[key] for key in order], len(rows) - len(order)


def _source_index(
    records: list[dict[str, Any]],
    media_root: pathlib.Path,
) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    index: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for record_index, record in enumerate(records):
        if record.get("task_type") not in TASK_SPECS:
            continue
        prompt_and_response(record, context=f"source record[{record_index}]")
        target = target_path(record, context=f"source record[{record_index}]")
        for key in _path_keys(target, media_root):
            index.setdefault(key, []).append((record_index, record))
    return index


def _match(
    mined_path: str,
    *,
    media_root: pathlib.Path,
    index: dict[str, list[tuple[int, dict[str, Any]]]],
) -> tuple[list[tuple[int, dict[str, Any]]], str]:
    resolved_mined = str(resolve_image(mined_path, media_root))
    exact_hits = index.get(resolved_mined, [])
    if exact_hits:
        return sorted(exact_hits), "exact"
    candidates: dict[int, dict[str, Any]] = {}
    for key in _path_keys(mined_path, media_root):
        hits = index.get(key, [])
        for source_index, record in hits:
            candidates[source_index] = record
    if not candidates:
        raise ValueError(
            f"missing source match for mined path {mined_path!r}; "
            f"candidate_indexes={sorted(candidates)}"
        )
    resolved_targets = set()
    for record in candidates.values():
        resolved_targets.add(
            str(resolve_image(target_path(record, context="source match"), media_root))
        )
    if len(resolved_targets) != 1:
        raise ValueError(
            f"ambiguous source match for mined path {mined_path!r}; "
            f"candidate_indexes={sorted(candidates)}"
        )
    return sorted(candidates.items()), "name"


def _format_path(
    path_text: str,
    *,
    media_root: pathlib.Path,
    relative: bool,
) -> str:
    resolved = resolve_image(path_text, media_root)
    if not relative:
        return str(resolved)
    try:
        return resolved.relative_to(media_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"cannot emit relative path outside media root: {resolved}"
        ) from exc


def emit_records(
    mined_paths: Iterable[str | dict[str, Any]],
    source_records: list[dict[str, Any]],
    *,
    media_root: pathlib.Path,
    relative: bool,
    annotation_profile: str = "nvpaw_multitask_v1",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    media_root = media_root.expanduser().resolve()
    mined_rows, duplicates_skipped = _normalize_mined_rows(mined_paths, media_root)
    if annotation_profile != "nvpaw_multitask_v1":
        raise ValueError("Cosmos Framework mining emission requires nvpaw_multitask_v1")
    index = _source_index(source_records, media_root)
    output: list[dict[str, Any]] = []
    matches: Counter[str] = Counter()
    route_tiers: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    seen_targets: set[str] = set()
    for mined in mined_rows:
        mined_path = mined["filepath"]
        resolved_target = str(resolve_image(mined_path, media_root))
        if resolved_target in seen_targets:
            continue
        matched, match_mode = _match(
            mined_path, media_root=media_root, index=index
        )
        seen_targets.add(resolved_target)
        matches[match_mode] += 1
        route_tiers[mined["route_tier"]] += 1
        emitted_for_target = 0
        for source_index, source in matched:
            routed_tasks = mined.get("routed_task_types")
            if routed_tasks is not None and source.get("task_type") not in routed_tasks:
                continue
            record = copy.deepcopy(source)
            for message in record["messages"]:
                content = message.get("content") if isinstance(message, dict) else None
                if not isinstance(content, list):
                    continue
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        item["image"] = _format_path(
                            str(item["image"]), media_root=media_root, relative=relative
                        )
            output.append(record)
            tasks[str(source["task_type"])] += 1
            emitted_for_target += 1
        if emitted_for_target == 0:
            raise ValueError(
                f"mined path {mined_path!r} has no source record matching "
                f"routed_task_types={mined.get('routed_task_types')!r}"
            )
    if not output:
        raise ValueError("no unique mined records were emitted")
    return output, {
        "mode": annotation_profile,
        "source_records": len(source_records),
        "output_records": len(output),
        "embedding_queries": len(seen_targets),
        "duplicates_skipped": duplicates_skipped,
        "match_modes": dict(sorted(matches.items())),
        "route_tiers": dict(sorted(route_tiers.items())),
        "tasks": dict(sorted(tasks.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mined-parquet", required=True, type=pathlib.Path)
    parser.add_argument("--source-annotations", required=True, type=pathlib.Path)
    parser.add_argument("--media-root", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--summary", type=pathlib.Path)
    parser.add_argument("--filepath-column", default="filepath")
    parser.add_argument("--emit-relative", action="store_true")
    args = parser.parse_args(argv)
    try:
        mined_paths = _load_mined_rows(args.mined_parquet, args.filepath_column)
        records, summary = emit_records(
            mined_paths,
            load_records(args.source_annotations),
            media_root=args.media_root.expanduser().resolve(),
            relative=args.emit_relative,
            annotation_profile="nvpaw_multitask_v1",
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
        )
        summary_path = args.summary or args.output.with_name(
            "emit_mined_summary.json"
        )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"emit_mined_sharegpt: {exc}", file=sys.stderr)
        return 2
    print(
        f"emit_mined_sharegpt: wrote {len(records)} nvpaw_multitask_v1 JSONL records "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
