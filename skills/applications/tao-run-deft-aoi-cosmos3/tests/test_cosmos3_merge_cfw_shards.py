# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import merge_cfw_prediction_shards  # noqa: E402


class Cosmos3MergeCfwShardsTests(unittest.TestCase):
    def test_merge_normalizes_to_source_order_and_records_complete_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "source.jsonl"
            shards = root / "shards"
            output = root / "predictions.jsonl"
            summary = root / "summary.json"
            shards.mkdir()
            source.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "id": row_id,
                            "task_type": "BCQ",
                            "messages": [
                                {"role": "user", "content": "question"},
                                {"role": "assistant", "content": answer},
                            ],
                        }
                    )
                    for row_id, answer in (("first", "A"), ("second", "B"))
                )
                + "\n"
            )
            (shards / "predictions_rank0.jsonl").write_text(
                json.dumps({"id": "second", "raw_prediction": "B"}) + "\n"
            )
            (shards / "predictions_rank1.jsonl").write_text(
                json.dumps({"id": "first", "raw_prediction": "A"}) + "\n"
            )

            payload = merge_cfw_prediction_shards.merge(
                source=source,
                shard_dir=shards,
                expected_shards=2,
                output=output,
                summary_output=summary,
            )

            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["id"] for row in rows], ["first", "second"])
            self.assertEqual(payload["state"], "COMPLETE")
            self.assertEqual(payload["rows"], 2)
            self.assertEqual(json.loads(summary.read_text()), payload)

    def test_merge_rejects_missing_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "source.jsonl"
            shards = root / "shards"
            shards.mkdir()
            source.write_text("{}\n")
            (shards / "predictions_rank0.jsonl").write_text("{}\n")

            with self.assertRaisesRegex(ValueError, "shard set mismatch"):
                merge_cfw_prediction_shards.merge(
                    source=source,
                    shard_dir=shards,
                    expected_shards=2,
                    output=root / "out.jsonl",
                    summary_output=root / "summary.json",
                )


if __name__ == "__main__":
    unittest.main()
