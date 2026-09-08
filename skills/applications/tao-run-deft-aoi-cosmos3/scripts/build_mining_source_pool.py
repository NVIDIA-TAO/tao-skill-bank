#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Build the unique atomic-sample embedding pool from canonical Mining JSONL."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
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
from atomic_samples import (
    embedding_filepath,
    materialize_pair_asset,
    pair_asset_path,
    sample_from_record,
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_parquet(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".parquet", dir=path.parent
    )
    os.close(descriptor)
    temporary = pathlib.Path(temporary_name)
    try:
        pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
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


def _read_unique_pool(path: pathlib.Path) -> list[dict[str, Any]]:
    table = pq.read_table(path.expanduser().resolve(strict=True))
    if "filepath" not in table.column_names:
        raise ValueError(f"reuse pool has no filepath column: {path}")
    values = table.column("filepath").to_pylist()
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"reuse pool has invalid filepath values: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"reuse pool filepath values are not unique: {path}")
    rows = table.to_pylist()
    if "atomic_sample_id" in table.column_names:
        identities = [row.get("atomic_sample_id") for row in rows]
        if any(not isinstance(value, str) or not value for value in identities):
            raise ValueError(f"reuse pool has invalid atomic_sample_id values: {path}")
        if len(identities) != len(set(identities)):
            raise ValueError(f"reuse pool atomic_sample_id values are not unique: {path}")
    return rows


def build(
    *,
    annotations: pathlib.Path,
    media_root: pathlib.Path,
    output: pathlib.Path,
    summary_output: pathlib.Path,
    reuse_pool: pathlib.Path | None = None,
    delta_output: pathlib.Path | None = None,
    pair_assets_dir: pathlib.Path | None = None,
    pair_asset_workers: int = 1,
) -> dict[str, Any]:
    annotations = annotations.expanduser().resolve(strict=True)
    media_root = media_root.expanduser().resolve()
    if pair_asset_workers <= 0:
        raise ValueError("pair_asset_workers must be positive")
    loads = _json_loader()
    ordered_samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    tasks: collections.Counter[str] = collections.Counter()
    unsupported: collections.Counter[str] = collections.Counter()
    raw_rows = 0
    supported_rows = 0
    resolved_path_cache: dict[str, str] = {}
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
            sample = sample_from_record(
                row,
                media_root=media_root,
                context=f"{annotations}:{line_number}",
                resolved_path_cache=resolved_path_cache,
            )
            identity = str(sample["atomic_sample_id"])
            if identity not in seen:
                seen.add(identity)
                if sample["sample_kind"] == "reference_pair":
                    if pair_assets_dir is None:
                        raise ValueError(
                            "pair_assets_dir is required for reference-pair embedding"
                        )
                    sample["filepath"] = str(
                        pair_asset_path(pair_assets_dir, identity)
                    )
                else:
                    sample["filepath"] = embedding_filepath(
                        sample, pair_assets_dir=pair_assets_dir
                    )
                ordered_samples.append(sample)
            tasks[str(task)] += 1
            supported_rows += 1
    if not ordered_samples:
        raise ValueError("Mining annotations contain no supported atomic samples")
    reference_samples = [
        sample
        for sample in ordered_samples
        if sample["sample_kind"] == "reference_pair"
    ]
    if pair_asset_workers == 1:
        for sample in reference_samples:
            materialize_pair_asset(sample, pair_assets_dir=pair_assets_dir)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=pair_asset_workers
        ) as executor:
            for _ in executor.map(
                lambda sample: materialize_pair_asset(
                    sample, pair_assets_dir=pair_assets_dir
                ),
                reference_samples,
            ):
                pass

    output = output.expanduser().resolve()
    _atomic_parquet(output, ordered_samples)
    reused_targets = 0
    delta_targets = len(ordered_samples)
    delta_sha256: str | None = None
    resolved_reuse: str | None = None
    resolved_delta: str | None = None
    if reuse_pool is not None:
        if delta_output is None:
            raise ValueError("delta_output is required with reuse_pool")
        cached = _read_unique_pool(reuse_pool)
        current_by_filepath = {row["filepath"]: row for row in ordered_samples}
        if all("atomic_sample_id" in row for row in cached):
            cached_set = {row["atomic_sample_id"] for row in cached}
            current_set = {row["atomic_sample_id"] for row in ordered_samples}
            delta = [
                row for row in ordered_samples if row["atomic_sample_id"] not in cached_set
            ]
        else:
            cached_set = {row["filepath"] for row in cached}
            if any(
                current_by_filepath.get(path, {}).get("sample_kind") == "reference_pair"
                for path in cached_set
            ):
                raise ValueError("legacy reuse pools cannot identify reference pairs atomically")
            current_set = set(current_by_filepath)
            delta = [row for row in ordered_samples if row["filepath"] not in cached_set]
        if not cached_set.issubset(current_set):
            raise ValueError(
                "reuse pool is not a subset of the current Mining atomic pool: "
                f"extra_cached={sorted(cached_set - current_set)[:10]}"
            )
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
        "pool_size": len(ordered_samples),
        "pool_identity": "atomic_sample_id",
        "single_images": sum(
            row["sample_kind"] == "single_image" for row in ordered_samples
        ),
        "reference_pairs": sum(
            row["sample_kind"] == "reference_pair" for row in ordered_samples
        ),
        "pair_embedding_asset_schema": "nvpaw_reference_pair_embedding_v1",
        "pair_asset_workers": pair_asset_workers,
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
    parser.add_argument("--pair-assets-dir", type=pathlib.Path)
    parser.add_argument("--pair-asset-workers", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        payload = build(
            annotations=args.annotations,
            media_root=args.media_root,
            output=args.output,
            summary_output=args.summary_output,
            reuse_pool=args.reuse_pool,
            delta_output=args.delta_output,
            pair_assets_dir=args.pair_assets_dir,
            pair_asset_workers=args.pair_asset_workers,
        )
    except (OSError, ValueError, pa.ArrowException) as exc:
        print(f"build_mining_source_pool: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
