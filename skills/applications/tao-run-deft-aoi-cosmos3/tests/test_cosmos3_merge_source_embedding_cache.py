# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import merge_source_embedding_cache  # noqa: E402


def _write_pool(path: pathlib.Path, values: list[str]) -> None:
    pq.write_table(pa.table({"filepath": values}), path)


def _write_embeddings(path: pathlib.Path, values: list[str], *, dim: int = 3) -> None:
    pq.write_table(
        pa.table(
            {
                "filepath": values,
                "embedding": [[float(index)] * dim for index in range(len(values))],
                "original_filepath": values,
                "embedding_filepath": [f"{value}.png" for value in values],
            }
        ),
        path,
    )


class Cosmos3MergeSourceEmbeddingCacheTests(unittest.TestCase):
    def test_merges_and_reorders_to_current_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            pool = root / "pool.parquet"
            cached = root / "cached.parquet"
            delta = root / "delta.parquet"
            output = root / "output.parquet"
            summary = root / "summary.json"
            _write_pool(pool, ["a", "new", "b"])
            _write_embeddings(cached, ["a", "b"])
            _write_embeddings(delta, ["new"])

            payload = merge_source_embedding_cache.merge(
                current_pool=pool,
                cached_embeddings=cached,
                delta_embeddings=delta,
                output=output,
                summary_output=summary,
                embedding_dimension=3,
            )

            self.assertEqual(payload["rows"], 3)
            self.assertEqual(payload["cached_rows"], 2)
            self.assertEqual(payload["delta_rows"], 1)
            self.assertEqual(
                pq.read_table(output).column("filepath").to_pylist(),
                ["a", "new", "b"],
            )

    def test_rejects_missing_or_overlapping_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            pool = root / "pool.parquet"
            cached = root / "cached.parquet"
            delta = root / "delta.parquet"
            _write_pool(pool, ["a", "b"])
            _write_embeddings(cached, ["a"])
            _write_embeddings(delta, ["a"])

            with self.assertRaisesRegex(ValueError, "overlap"):
                merge_source_embedding_cache.merge(
                    current_pool=pool,
                    cached_embeddings=cached,
                    delta_embeddings=delta,
                    output=root / "output.parquet",
                    summary_output=root / "summary.json",
                    embedding_dimension=3,
                )

    def test_rejects_wrong_embedding_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            pool = root / "pool.parquet"
            cached = root / "cached.parquet"
            delta = root / "delta.parquet"
            _write_pool(pool, ["a", "b"])
            _write_embeddings(cached, ["a"])
            _write_embeddings(delta, ["b"], dim=2)

            with self.assertRaisesRegex(ValueError, "dimension"):
                merge_source_embedding_cache.merge(
                    current_pool=pool,
                    cached_embeddings=cached,
                    delta_embeddings=delta,
                    output=root / "output.parquet",
                    summary_output=root / "summary.json",
                    embedding_dimension=3,
                )


if __name__ == "__main__":
    unittest.main()
