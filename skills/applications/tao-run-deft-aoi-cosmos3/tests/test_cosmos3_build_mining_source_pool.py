# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import build_mining_source_pool  # noqa: E402


def _row(row_id: str, task: str, paths: list[str]) -> dict:
    return {
        "id": row_id,
        "task_type": task,
        "messages": [
            {
                "role": "user",
                "content": [
                    *({"type": "image", "image": path} for path in paths),
                    {"type": "text", "text": "question"},
                ],
            },
            {"role": "assistant", "content": "answer"},
        ],
    }


class Cosmos3BuildMiningSourcePoolTests(unittest.TestCase):
    def test_builds_unique_targets_and_exact_delta_against_reuse_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            annotations = root / "mining.jsonl"
            pool = root / "source_pool.parquet"
            reuse = root / "reuse.parquet"
            delta = root / "delta.parquet"
            summary = root / "summary.json"
            rows = [
                _row("a", "Component Detection", ["images/a.png"]),
                _row(
                    "b",
                    "Ref_based Defect Detection",
                    ["images/golden.png", "images/b.png"],
                ),
                _row("c", "Component Classification", ["images/a.png"]),
                _row("ignored", "Unsupported Task", ["images/x.png"]),
            ]
            annotations.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n"
            )
            cached = str((root / "images/a.png").resolve())
            pq.write_table(pa.table({"filepath": [cached]}), reuse)

            payload = build_mining_source_pool.build(
                annotations=annotations,
                media_root=root,
                output=pool,
                summary_output=summary,
                reuse_pool=reuse,
                delta_output=delta,
            )

            self.assertEqual(payload["raw_rows"], 4)
            self.assertEqual(payload["supported_rows"], 3)
            self.assertEqual(payload["pool_size"], 2)
            self.assertEqual(payload["reused_targets"], 1)
            self.assertEqual(payload["delta_targets"], 1)
            self.assertEqual(
                pq.read_table(delta).column("filepath").to_pylist(),
                [str((root / "images/b.png").resolve())],
            )

    def test_rejects_reuse_pool_that_is_not_a_subset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            annotations = root / "mining.jsonl"
            annotations.write_text(
                json.dumps(_row("a", "Component Detection", ["a.png"])) + "\n"
            )
            reuse = root / "reuse.parquet"
            pq.write_table(pa.table({"filepath": ["missing.png"]}), reuse)

            with self.assertRaisesRegex(ValueError, "not a subset"):
                build_mining_source_pool.build(
                    annotations=annotations,
                    media_root=root,
                    output=root / "pool.parquet",
                    summary_output=root / "summary.json",
                    reuse_pool=reuse,
                    delta_output=root / "delta.parquet",
                )


if __name__ == "__main__":
    unittest.main()
