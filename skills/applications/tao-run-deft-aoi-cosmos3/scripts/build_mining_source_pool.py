#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Build the unique supported target-image pool from canonical Mining JSONL."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import sys
import tempfile
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq

from nvpaw_annotations import TASK_SPECS
from validate_sharegpt import target_path


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_parquet(path: pathlib.Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".parquet", dir=path.parent
    )
    os.close(descriptor)
    temporary = pathlib.Path(temporary_name)
    try:
        pq.write_table(pa.table({"filepath": values}), temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def _json_loader() -> Callable[[bytes], dict[str, Any]]:
    try:
        import orjson
    except ImportError:
        return json.loads
    return orjson.loads


def _read_unique_pool(path: pathlib.Path) -> list[str]:
    table = pq.read_table(path.expanduser().resolve(strict=True), columns=["filepath"])
    values = table.column("filepath").to_pylist()
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"reuse pool has invalid filepath values: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"reuse pool filepath values are not unique: {path}")
    return values


def build(
    *,
    annotations: pathlib.Path,
    media_root: pathlib.Path,
    output: pathlib.Path,
    summary_output: pathlib.Path,
    reuse_pool: pathlib.Path | None = None,
    delta_output: pathlib.Path | None = None,
) -> dict[str, Any]:
    annotations = annotations.expanduser().resolve(strict=True)
    media_root = media_root.expanduser().resolve()
    loads = _json_loader()
    ordered_targets: list[str] = []
    seen: set[str] = set()
    tasks: collections.Counter[str] = collections.Counter()
    unsupported: collections.Counter[str] = collections.Counter()
    raw_rows = 0
    supported_rows = 0
    annotation_digest = hashlib.sha256()
    with annotations.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            annotation_digest.update(line)
            if not line.strip():
                continue
            raw_rows += 1
            try:
                row = loads(line)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{annotations}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{annotations}:{line_number}: row must be an object")
            task = row.get("task_type")
            if task not in TASK_SPECS:
                unsupported[str(task)] += 1
                continue
            target = pathlib.Path(
                target_path(row, context=f"{annotations}:{line_number}")
            ).expanduser()
            resolved = str((target if target.is_absolute() else media_root / target).resolve())
            if resolved not in seen:
                seen.add(resolved)
                ordered_targets.append(resolved)
            tasks[str(task)] += 1
            supported_rows += 1
    if not ordered_targets:
        raise ValueError("Mining annotations contain no supported target images")

    output = output.expanduser().resolve()
    _atomic_parquet(output, ordered_targets)
    reused_targets = 0
    delta_targets = len(ordered_targets)
    delta_sha256: str | None = None
    resolved_reuse: str | None = None
    resolved_delta: str | None = None
    if reuse_pool is not None:
        if delta_output is None:
            raise ValueError("delta_output is required with reuse_pool")
        cached = _read_unique_pool(reuse_pool)
        cached_set = set(cached)
        if not cached_set.issubset(seen):
            raise ValueError(
                "reuse pool is not a subset of the current Mining target pool: "
                f"extra_cached={sorted(cached_set - seen)[:10]}"
            )
        delta = [value for value in ordered_targets if value not in cached_set]
        delta_output = delta_output.expanduser().resolve()
        _atomic_parquet(delta_output, delta)
        reused_targets = len(cached)
        delta_targets = len(delta)
        delta_sha256 = _sha256(delta_output)
        resolved_reuse = str(reuse_pool.expanduser().resolve(strict=True))
        resolved_delta = str(delta_output)

    payload = {
        "schema_version": 1,
        "annotations": str(annotations),
        "annotation_sha256": annotation_digest.hexdigest(),
        "media_root": str(media_root),
        "raw_rows": raw_rows,
        "supported_rows": supported_rows,
        "unsupported_rows": raw_rows - supported_rows,
        "unsupported_tasks": dict(sorted(unsupported.items())),
        "task_rows": dict(sorted(tasks.items())),
        "pool_size": len(ordered_targets),
        "output": str(output),
        "output_sha256": _sha256(output),
        "reuse_pool": resolved_reuse,
        "reused_targets": reused_targets,
        "delta_output": resolved_delta,
        "delta_sha256": delta_sha256,
        "delta_targets": delta_targets,
    }
    _atomic_json(summary_output.expanduser().resolve(), payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=pathlib.Path, required=True)
    parser.add_argument("--media-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--summary-output", type=pathlib.Path, required=True)
    parser.add_argument("--reuse-pool", type=pathlib.Path)
    parser.add_argument("--delta-output", type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        payload = build(
            annotations=args.annotations,
            media_root=args.media_root,
            output=args.output,
            summary_output=args.summary_output,
            reuse_pool=args.reuse_pool,
            delta_output=args.delta_output,
        )
    except (OSError, ValueError, pa.ArrowException) as exc:
        print(f"build_mining_source_pool: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
