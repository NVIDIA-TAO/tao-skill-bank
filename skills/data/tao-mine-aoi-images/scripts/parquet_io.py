# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parquet I/O with two interchangeable backends: pyarrow, else duckdb.

Every mining/DEFT artifact is a small flat table (strings, floats, and one
list<float32> embedding column). pyarrow is the reference backend, but it ships
no Windows ARM64 wheel (PyPI/conda-forge, checked 2026-09-06; Arrow issue
#47195, PR #48539 is nightly-first). duckdb does ship cp313 win_arm64 and
writes the same parquet files: the round-trip is pyarrow-readback-identical on
every artifact of a real run, embeddings included. Callers use rows/columns of
plain Python values and never see either library.
"""
from __future__ import annotations

import pathlib
from typing import Any, Iterable, Mapping, Sequence

try:  # reference backend
    import pyarrow as _pa
    import pyarrow.parquet as _pq
    BACKEND = "pyarrow"
except ImportError:  # Windows ARM64 (no pyarrow wheel) and any pyarrow-less env
    _pa = _pq = None
    try:
        import duckdb as _duckdb
        BACKEND = "duckdb"
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "parquet_io needs pyarrow or duckdb in the selected Python "
            "(pip install pyarrow  |  pip install duckdb)") from exc


def _py(v: Any) -> Any:
    """numpy scalars/arrays -> plain Python (duckdb binds only builtins)."""
    if hasattr(v, "tolist"):
        return v.tolist()
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            return v.item()
        except (ValueError, TypeError):
            return v
    return v


def _duck_type(values: Iterable[Any]) -> str:
    for v in values:
        if v is None:
            continue
        if isinstance(v, bool):
            return "BOOLEAN"
        if isinstance(v, int):
            return "BIGINT"
        if isinstance(v, float):
            return "DOUBLE"
        if isinstance(v, (list, tuple)):
            return "FLOAT[]"
        return "VARCHAR"
    return "VARCHAR"


def num_rows(path: str | pathlib.Path) -> int:
    path = str(path)
    if BACKEND == "pyarrow":
        return int(_pq.ParquetFile(path).metadata.num_rows)
    con = _duckdb.connect()
    try:
        return int(con.execute("SELECT count(*) FROM read_parquet(?)", [path]).fetchone()[0])
    finally:
        con.close()


def column_names(path: str | pathlib.Path) -> list[str]:
    path = str(path)
    if BACKEND == "pyarrow":
        return list(_pq.ParquetFile(path).schema_arrow.names)
    con = _duckdb.connect()
    try:
        return [r[0] for r in con.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [path]).fetchall()]
    finally:
        con.close()


def read_rows(path: str | pathlib.Path, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Whole table as a list of dicts (parquet order preserved)."""
    path = str(path)
    if BACKEND == "pyarrow":
        return _pq.read_table(path, columns=list(columns) if columns else None).to_pylist()
    con = _duckdb.connect()
    try:
        sel = ", ".join(f'"{c}"' for c in columns) if columns else "*"
        cur = con.execute(f"SELECT {sel} FROM read_parquet(?)", [path])
        names = [d[0] for d in cur.description]
        return [dict(zip(names, row)) for row in cur.fetchall()]
    finally:
        con.close()


def read_columns(path: str | pathlib.Path, columns: Sequence[str] | None = None) -> dict[str, list[Any]]:
    """Whole table as {column: [values]}."""
    rows = read_rows(path, columns)
    names = list(columns) if columns else (list(rows[0].keys()) if rows else column_names(path))
    return {c: [r.get(c) for r in rows] for c in names}


def write_columns(columns: Mapping[str, Sequence[Any]], path: str | pathlib.Path,
                  float32_lists: Sequence[str] = ("embedding",)) -> None:
    """Write {column: [values]} as parquet. Columns named in float32_lists are
    list<float32> (pyarrow) / FLOAT[] (duckdb) so both backends agree on the
    embedding dtype."""
    path = str(path)
    names = list(columns.keys())
    if not names:
        raise ValueError("cannot write a table with no columns")
    n = len(columns[names[0]])
    for c in names:
        if len(columns[c]) != n:
            raise ValueError(f"column {c!r} has {len(columns[c])} rows, expected {n}")
    if BACKEND == "pyarrow":
        arrays = {}
        for c in names:
            if c in float32_lists:
                arrays[c] = _pa.array([list(map(float, v)) if v is not None else None
                                       for v in columns[c]], type=_pa.list_(_pa.float32()))
            else:
                vals = [_py(v) for v in columns[c]]
                # an all-null/empty column would otherwise become type `null`,
                # which downstream readers reject; strings are the schema default
                arrays[c] = (_pa.array(vals, type=_pa.string()) if not any(v is not None for v in vals)
                             else _pa.array(vals))
        _pq.write_table(_pa.table(arrays), path)
        return
    con = _duckdb.connect()
    try:
        types = {c: ("FLOAT[]" if c in float32_lists else _duck_type(map(_py, columns[c]))) for c in names}
        con.execute("CREATE TABLE t (" + ", ".join(f'"{c}" {types[c]}' for c in names) + ")")
        if n:
            placeholders = ", ".join("?" for _ in names)
            def cell(c, i):
                v = _py(columns[c][i])
                if c in float32_lists and v is not None:
                    v = [float(x) for x in v]
                return v
            con.executemany(f"INSERT INTO t VALUES ({placeholders})",
                            [tuple(cell(c, i) for c in names) for i in range(n)])
        con.execute(f"COPY t TO '{pathlib.Path(path).as_posix()}' (FORMAT PARQUET)")
    finally:
        con.close()


def write_rows(rows: Sequence[Mapping[str, Any]], path: str | pathlib.Path,
               columns: Sequence[str] | None = None,
               float32_lists: Sequence[str] = ("embedding",)) -> None:
    """Write a list of dicts as parquet. Column order: `columns` if given, else
    first-row key order."""
    names = list(columns) if columns else (list(rows[0].keys()) if rows else [])
    write_columns({c: [r.get(c) for r in rows] for c in names}, path, float32_lists)
