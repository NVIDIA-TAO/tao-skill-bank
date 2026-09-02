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

import assemble_training_json  # noqa: E402


def _row(record_id: str, task_type: str = "Component Classification") -> dict:
    images = [
        {
            "type": "image",
            "image": f"images/{record_id}.png",
            "min_pixels": 1,
            "max_pixels": 1,
        }
    ]
    if task_type.startswith("Ref_based"):
        images.insert(
            0,
            {
                "type": "image",
                "image": f"images/{record_id}-golden.png",
                "min_pixels": 1,
                "max_pixels": 1,
            },
        )
    return {
        "id": record_id,
        "task_type": task_type,
        "messages": [
            {
                "role": "user",
                "content": [
                    *images,
                    {"type": "text", "text": "classify"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "A"}]},
        ],
    }


def _write(path: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


class Cosmos3AssembleTrainingContractTests(unittest.TestCase):
    def test_cap_retains_previous_rows_and_fills_with_current_mining(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            previous = _write(root / "previous.jsonl", [_row(f"old{i}") for i in range(4)])
            mined = _write(root / "mined.jsonl", [_row(f"new{i}") for i in range(3)])

            rows, summary = assemble_training_json.assemble(
                previous,
                mined,
                validation_paths=[],
                max_rows=6,
                row_multiple=2,
            )

            self.assertEqual(
                [row["id"] for row in rows],
                ["new0", "new1", "old0", "old1", "old2", "old3"],
            )
            self.assertEqual(summary["output_records"], 6)
            self.assertEqual(summary["materialization_cap"], 6)
            self.assertEqual(summary["row_multiple"], 2)
            self.assertEqual(summary["records_truncated"], 1)
            self.assertEqual(summary["retained_previous_records"], 4)
            self.assertEqual(summary["selected_current_records"], 2)

    def test_cap_rejects_silent_previous_iteration_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            previous = _write(root / "previous.jsonl", [_row(f"old{i}") for i in range(4)])
            mined = _write(root / "mined.jsonl", [_row(f"new{i}") for i in range(3)])

            with self.assertRaisesRegex(ValueError, "retain all previous iteration records"):
                assemble_training_json.assemble(
                    previous,
                    mined,
                    validation_paths=[],
                    max_rows=5,
                    row_multiple=2,
                )

    def test_exact_epoch_materialization_requires_one_complete_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined = _write(root / "mined.jsonl", [_row("new0")])
            with self.assertRaisesRegex(ValueError, "complete global batch"):
                assemble_training_json.assemble(
                    None,
                    mined,
                    validation_paths=[],
                    max_rows=20_000,
                    row_multiple=512,
                )

    def test_cap_balances_current_mining_across_task_types(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined = _write(
                root / "mined.jsonl",
                [
                    *(_row(f"a{i}", "Component Detection") for i in range(6)),
                    *(_row(f"b{i}", "Defect Classification") for i in range(2)),
                    *(_row(f"c{i}", "Ref_based Defect Detection") for i in range(2)),
                ],
            )

            rows, summary = assemble_training_json.assemble(
                None,
                mined,
                validation_paths=[],
                max_rows=8,
                row_multiple=4,
            )

            task_counts = {
                task: sum(row["task_type"] == task for row in rows)
                for task in {row["task_type"] for row in rows}
            }
            self.assertEqual(
                task_counts,
                {
                    "Component Detection": 4,
                    "Defect Classification": 2,
                    "Ref_based Defect Detection": 2,
                },
            )
            self.assertEqual(
                summary["selection_policy"],
                "monotonic_current_fill_task_balanced_v1",
            )


if __name__ == "__main__":
    unittest.main()
