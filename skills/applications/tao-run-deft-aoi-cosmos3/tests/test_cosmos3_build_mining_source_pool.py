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
from PIL import Image


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
    def test_reference_rows_are_one_atomic_pair_embedding_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            annotations = root / "mining.jsonl"
            for name, color in (("golden.png", "white"), ("test.png", "black")):
                path = root / "images" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 6), color=color).save(path)
            annotations.write_text(
                json.dumps(
                    _row(
                        "pair-a",
                        "Ref_based Defect Detection",
                        ["images/golden.png", "images/test.png"],
                    )
                )
                + "\n"
            )

            payload = build_mining_source_pool.build(
                annotations=annotations,
                media_root=root,
                output=root / "source_pool.parquet",
                summary_output=root / "summary.json",
                pair_assets_dir=root / "pair-assets",
            )

            rows = pq.read_table(root / "source_pool.parquet").to_pylist()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sample_kind"], "reference_pair")
            self.assertEqual(
                rows[0]["image_paths"],
                [
                    str((root / "images/golden.png").resolve()),
                    str((root / "images/test.png").resolve()),
                ],
            )
            self.assertTrue(rows[0]["atomic_sample_id"].startswith("reference_pair:"))
            self.assertNotEqual(rows[0]["filepath"], rows[0]["image_paths"][0])
            self.assertNotEqual(rows[0]["filepath"], rows[0]["image_paths"][1])
            self.assertTrue(pathlib.Path(rows[0]["filepath"]).is_file())
            self.assertEqual(payload["reference_pairs"], 1)
            self.assertEqual(payload["single_images"], 0)

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
            for name in ("a.png", "golden.png", "b.png"):
                path = root / "images" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 6), color="white").save(path)
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
                pair_assets_dir=root / "pair-assets",
            )

            self.assertEqual(payload["raw_rows"], 4)
            self.assertEqual(payload["supported_rows"], 3)
            self.assertEqual(payload["pool_size"], 2)
            self.assertEqual(payload["reused_targets"], 1)
            self.assertEqual(payload["delta_targets"], 1)
            delta_rows = pq.read_table(delta).to_pylist()
            self.assertEqual(len(delta_rows), 1)
            self.assertEqual(delta_rows[0]["sample_kind"], "reference_pair")
            self.assertEqual(
                delta_rows[0]["image_paths"],
                [
                    str((root / "images/golden.png").resolve()),
                    str((root / "images/b.png").resolve()),
                ],
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
