#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Merge an attested embedding cache and exact delta into a current source pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import tempfile
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


REQUIRED_COLUMNS = (
    "filepath",
    "embedding",
    "original_filepath",
    "embedding_filepath",
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _read_pool(path: pathlib.Path) -> list[str]:
    table = pq.read_table(path, columns=["filepath"])
    values = table.column("filepath").to_pylist()
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"current pool has invalid filepath values: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"current pool filepath values are not unique: {path}")
    return values


def _read_embeddings(path: pathlib.Path, dimension: int) -> tuple[pa.Table, list[str]]:
    table = pq.read_table(path)
    if set(table.column_names) != set(REQUIRED_COLUMNS):
        raise ValueError(
            f"embedding table columns must be exactly {list(REQUIRED_COLUMNS)}: "
            f"path={path} columns={table.column_names}"
        )
    table = table.select(REQUIRED_COLUMNS)
    if table.num_rows == 0:
        raise ValueError(f"embedding table is empty: {path}")
    embedding_type = table.schema.field("embedding").type
    if not (pa.types.is_list(embedding_type) or pa.types.is_large_list(embedding_type)):
        raise ValueError(f"embedding must be a list column: path={path} type={embedding_type}")
    if table.column("embedding").null_count:
        raise ValueError(f"embedding contains null rows: {path}")
    lengths = pc.list_value_length(table.column("embedding"))
    length_range = pc.min_max(lengths).as_py()
    if length_range["min"] != dimension or length_range["max"] != dimension:
        raise ValueError(
            f"embedding dimension must be exactly {dimension}: "
            f"path={path} observed={length_range}"
        )
    if pc.list_flatten(table.column("embedding")).null_count:
        raise ValueError(f"embedding vectors contain null values: {path}")

    paths = table.column("filepath").to_pylist()
    originals = table.column("original_filepath").to_pylist()
    derivatives = table.column("embedding_filepath").to_pylist()
    for label, values in (("filepath", paths), ("original_filepath", originals), ("embedding_filepath", derivatives)):
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"{label} contains invalid values: {path}")
    if len(paths) != len(set(paths)):
        raise ValueError(f"filepath values are not unique: {path}")
    if paths != originals:
        raise ValueError(f"original_filepath does not exactly match filepath: {path}")
    return table, paths


def merge(
    *,
    current_pool: pathlib.Path,
    cached_embeddings: pathlib.Path,
    delta_embeddings: pathlib.Path,
    output: pathlib.Path,
    summary_output: pathlib.Path,
    embedding_dimension: int = 768,
) -> dict[str, Any]:
    if embedding_dimension <= 0:
        raise ValueError("embedding_dimension must be positive")
    current_pool = current_pool.expanduser().resolve(strict=True)
    cached_embeddings = cached_embeddings.expanduser().resolve(strict=True)
    delta_embeddings = delta_embeddings.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    summary_output = summary_output.expanduser().resolve()

    current_paths = _read_pool(current_pool)
    cached_table, cached_paths = _read_embeddings(cached_embeddings, embedding_dimension)
    delta_table, delta_paths = _read_embeddings(delta_embeddings, embedding_dimension)
    cached_set = set(cached_paths)
    delta_set = set(delta_paths)
    overlap = cached_set & delta_set
    if overlap:
        raise ValueError(f"cached and delta filepath sets overlap: {sorted(overlap)[:10]}")
    observed = cached_set | delta_set
    expected = set(current_paths)
    if observed != expected:
        raise ValueError(
            "cached plus delta filepath set does not exactly match current pool: "
            f"missing={sorted(expected - observed)[:10]} extra={sorted(observed - expected)[:10]}"
        )

    combined = pa.concat_tables([cached_table, delta_table], promote_options="none")
    index_by_path = {
        path: index for index, path in enumerate(cached_paths + delta_paths)
    }
    ordered = combined.take(pa.array([index_by_path[path] for path in current_paths], type=pa.int64()))
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".parquet", dir=output.parent
    )
    os.close(descriptor)
    temporary = pathlib.Path(temporary_name)
    try:
        pq.write_table(ordered, temporary, compression="zstd")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    payload = {
        "schema_version": 1,
        "current_pool": str(current_pool),
        "current_pool_sha256": _sha256(current_pool),
        "cached_embeddings": str(cached_embeddings),
        "cached_embeddings_sha256": _sha256(cached_embeddings),
        "delta_embeddings": str(delta_embeddings),
        "delta_embeddings_sha256": _sha256(delta_embeddings),
        "output": str(output),
        "output_sha256": _sha256(output),
        "rows": ordered.num_rows,
        "cached_rows": cached_table.num_rows,
        "delta_rows": delta_table.num_rows,
        "embedding_dimension": embedding_dimension,
        "filepath_order": "current_pool",
        "exact_set_match": True,
    }
    _atomic_json(summary_output, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-pool", required=True, type=pathlib.Path)
    parser.add_argument("--cached-embeddings", required=True, type=pathlib.Path)
    parser.add_argument("--delta-embeddings", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--summary-output", required=True, type=pathlib.Path)
    parser.add_argument("--embedding-dimension", type=int, default=768)
    args = parser.parse_args(argv)
    try:
        payload = merge(
            current_pool=args.current_pool,
            cached_embeddings=args.cached_embeddings,
            delta_embeddings=args.delta_embeddings,
            output=args.output,
            summary_output=args.summary_output,
            embedding_dimension=args.embedding_dimension,
        )
    except (OSError, ValueError, pa.ArrowException) as exc:
        print(f"merge_source_embedding_cache: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
