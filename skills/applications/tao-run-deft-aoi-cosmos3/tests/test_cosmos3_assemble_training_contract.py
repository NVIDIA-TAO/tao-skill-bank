# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import importlib
import pathlib
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import assemble_training_json  # noqa: E402
import commit_stage  # noqa: E402


def _row(
    record_id: str,
    task_type: str = "Component Classification",
    *,
    reference: str | None = None,
    target: str | None = None,
) -> dict:
    images = [
        {
            "type": "image",
            "image": target or f"images/{record_id}.png",
            "min_pixels": 1,
            "max_pixels": 1,
        }
    ]
    if task_type.startswith("Ref_based"):
        images.insert(
            0,
            {
                "type": "image",
                "image": reference or f"images/{record_id}-golden.png",
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
    def test_five_round_lineage_is_cumulative_batch_aligned_and_monotonic(self) -> None:
        supply_counts = (2_566, 4_094, 5_637, 7_165, 8_728)
        expected_counts = (2_304, 3_840, 5_376, 6_912, 8_448)
        expected_steps = (15, 25, 35, 45, 55)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            universe = [_row(f"row-{index}") for index in range(supply_counts[-1])]
            previous_path = None
            previous_fingerprints: set[str] = set()
            for iteration, (supply, expected, steps) in enumerate(
                zip(supply_counts, expected_counts, expected_steps, strict=True),
                start=1,
            ):
                iteration_dir = root / f"iter{iteration}"
                iteration_dir.mkdir()
                mined = _write(iteration_dir / "mined.jsonl", universe[:supply])
                rows, summary = assemble_training_json.assemble(
                    previous_path,
                    mined,
                    validation_paths=[],
                    max_rows=20_000,
                    row_multiple=768,
                )
                output = _write(iteration_dir / "train.jsonl", rows)
                summary = assemble_training_json.bind_summary(summary, output)
                fingerprints = {
                    json.dumps(row, sort_keys=True, separators=(",", ":"))
                    for row in rows
                }

                self.assertEqual(len(rows), expected)
                self.assertEqual(len(rows) % 768, 0)
                self.assertEqual(len(rows) // 768 * 5, steps)
                self.assertTrue(previous_fingerprints.issubset(fingerprints))
                self.assertTrue(summary["previous_fingerprints_subset"])
                self.assertEqual(
                    summary["retained_previous_records"],
                    summary["previous_records"],
                )
                self.assertGreater(summary["selected_current_records"], 0)
                previous_path = output
                previous_fingerprints = fingerprints

            runner = importlib.import_module("render_iteration_mining_runner")
            plan = runner.build_plan(
                selector_command=["python", "defect_detection_ablation.py"],
                previous_jsonl=previous_path,
                previous_sha256=assemble_training_json.sha256_file(previous_path),
                mined_jsonl=root / "iter6" / "data" / "mined.jsonl",
                current_quota_manifest=root / "iter6" / "data" / "quota.json",
                train_jsonl=root / "iter6" / "assemble_data" / "train.jsonl",
                assemble_summary=root / "iter6" / "assemble_data" / "summary.json",
                final_quota_manifest=root / "iter6" / "assemble_data" / "quota.json",
                media_root=root,
                max_rows=20_000,
                row_multiple=768,
                epochs=5,
                global_batch=768,
            )
            self.assertNotEqual(plan["selector"]["output"], plan["assembler"]["output"])
            self.assertEqual(plan["selector"]["output"], plan["assembler"]["mined_input"])
            self.assertEqual(plan["assembler"]["previous_sha256"], plan["previous_sha256"])

    def test_commit_gate_rejects_missing_previous_hash_drop_and_lineage_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            previous = _write(root / "iter1-train.jsonl", [_row("old")])
            mined = _write(root / "iter2-mined.jsonl", [_row("new")])
            rows, summary = assemble_training_json.assemble(
                previous,
                mined,
                validation_paths=[],
            )
            output = _write(root / "iter2-train.jsonl", rows)
            valid = assemble_training_json.bind_summary(summary, output)

            cases = {
                "previous training JSONL is required": (None, valid),
                "previous SHA-256": (
                    previous,
                    {**valid, "previous_sha256": "0" * 64},
                ),
                "retain every previous record": (
                    previous,
                    {**valid, "retained_previous_records": 0},
                ),
                "previous_fingerprints_subset": (
                    previous,
                    {
                        key: value
                        for key, value in valid.items()
                        if key != "previous_fingerprints_subset"
                    },
                ),
            }
            for message, (prior, payload) in cases.items():
                with self.subTest(message=message):
                    summary_path = root / "assemble_summary.json"
                    summary_path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        commit_stage.validate_assembly_lineage(
                            summary_path,
                            previous_training_jsonl=prior,
                            mined_jsonl=mined,
                            combined_training_jsonl=output,
                            require_previous=True,
                        )

    def test_pair_atomic_leakage_allows_same_test_with_a_different_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            benchmark = _write(
                root / "benchmark.jsonl",
                [
                    _row(
                        "benchmark-pair",
                        "Ref_based Defect Detection",
                        reference="images/benchmark-golden.png",
                        target="images/shared-test.png",
                    )
                ],
            )
            mined = _write(
                root / "mined.jsonl",
                [
                    _row(
                        "mined-pair",
                        "Ref_based Defect Detection",
                        reference="images/mining-golden.png",
                        target="images/shared-test.png",
                    )
                ],
            )

            rows, summary = assemble_training_json.assemble(
                None,
                mined,
                validation_paths=[benchmark],
            )

            self.assertEqual([row["id"] for row in rows], ["mined-pair"])
            self.assertEqual(summary["output_records"], 1)

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

    def test_repetition_blend_runs_after_dedup_and_retains_monotonic_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            previous = _write(
                root / "previous.jsonl",
                [_row("old-dd", "Defect Detection")],
            )
            mined_rows = [
                _row("old-dd", "Defect Detection"),
                _row("new-dd", "Defect Detection"),
                *(_row(f"new-component-{index}", "Component Detection") for index in range(8)),
            ]
            mined = _write(root / "mined.jsonl", mined_rows)

            rows, summary = assemble_training_json.assemble(
                previous,
                mined,
                validation_paths=[],
                max_rows=10,
                row_multiple=1,
                repetition_config={
                    "enabled": True,
                    "policy": "deficit_proportional",
                    "rep_min": 0.5,
                    "rep_max": 3.0,
                    "never_repeat_empty_gt": True,
                    "explicit_multipliers": {},
                },
                deficit_weights={
                    "Defect Detection": 1.0,
                    "Component Detection": 1.0,
                },
                repetition_seed=17,
            )

            self.assertEqual(len(rows), 10)
            self.assertIn("old-dd", {row["id"] for row in rows})
            self.assertIn("new-dd", {row["id"] for row in rows})
            self.assertEqual(sum(row["task_type"] == "Defect Detection" for row in rows), 5)
            self.assertEqual(summary["duplicates_skipped"], 1)
            self.assertEqual(
                summary["repetition_blend"]["schema_version"],
                "repetition_blend_manifest_v1",
            )


if __name__ == "__main__":
    unittest.main()
