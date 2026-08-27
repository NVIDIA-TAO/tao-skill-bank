#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Core contracts for the Framework-only Cosmos3 DEFT AOI loop."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

for module_name in ("deft_context", "validate_sharegpt"):
    sys.modules.pop(module_name, None)

import assemble_training_json  # noqa: E402
import deft_context  # noqa: E402
import validate_sharegpt  # noqa: E402
import validate_split_contract  # noqa: E402


def record(target: str, label: str, *, record_id: str | None = None) -> dict:
    value = {
        "images": [target],
        "conversations": [
            {"from": "human", "value": "Inspect this image."},
            {"from": "gpt", "value": label},
        ],
    }
    if record_id is not None:
        value["id"] = record_id
    return value


def write_json(path: pathlib.Path, payload: object) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


class BareAnnotationTests(unittest.TestCase):
    def test_exact_bare_labels_only(self) -> None:
        summary = validate_sharegpt.validate_records(
            [record("a.png", "OK")],
            media_root=pathlib.Path("/tmp"),
            require_files=False,
        )
        self.assertEqual(summary["mode"], "bare_okng")
        self.assertEqual(summary["labels"], {"OK": 1})
        with self.assertRaisesRegex(ValueError, "exactly OK or NG"):
            validate_sharegpt.validate_records(
                [record("b.png", "Final answer: NG")],
                media_root=pathlib.Path("/tmp"),
                require_files=False,
            )

    def test_single_image_and_safe_unique_evaluation_ids(self) -> None:
        valid = [
            record("a.png", "OK", record_id="benchmark-0001"),
            record("b.png", "NG", record_id="benchmark-0002"),
        ]
        summary = validate_sharegpt.validate_records(
            valid,
            media_root=pathlib.Path("/tmp"),
            require_files=False,
            require_id=True,
        )
        self.assertEqual(summary["unique_ids"], 2)
        invalid = record("a.png", "OK")
        invalid["images"].append("reference.png")
        with self.assertRaisesRegex(ValueError, "exactly one image"):
            validate_sharegpt.validate_records(
                [invalid], media_root=pathlib.Path("/tmp"), require_files=False
            )

    def test_split_roles_are_disjoint_and_benchmark_hash_is_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            paths = {
                role: write_json(
                    root / f"{role}.json",
                    [record(f"images/{role}.png", "OK", record_id=f"{role}-1")],
                )
                for role in ("proxy", "benchmark", "mining")
            }
            summary = validate_split_contract.validate(paths, media_root=root)
            self.assertEqual(summary["records"], {"proxy": 1, "benchmark": 1, "mining": 1})
            self.assertIn("benchmark_sha256", summary)
            paths["mining"] = write_json(
                root / "mining.json",
                [record("images/benchmark.png", "NG", record_id="mining-2")],
            )
            with self.assertRaisesRegex(ValueError, "target leakage"):
                validate_split_contract.validate(paths, media_root=root)


class MiningOnlyAssemblyTests(unittest.TestCase):
    def test_first_iteration_uses_mined_real_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined = write_json(
                root / "iter1/mined.json",
                [record("images/ok-1.png", "OK"), record("images/ng-1.png", "NG")],
            )
            merged, summary = assemble_training_json.assemble(
                None, mined, dedupe=True, validation_paths=[]
            )
            self.assertEqual(merged, json.loads(mined.read_text()))
            self.assertEqual(summary["mined_input"], str(mined))
            self.assertEqual(summary["output_records"], 2)

    def test_later_iterations_are_monotonic_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            previous = write_json(
                root / "iter1/train.json",
                [record("images/ok-1.png", "OK"), record("images/ng-1.png", "NG")],
            )
            mined = write_json(
                root / "iter2/mined.json",
                [record("images/ng-1.png", "NG"), record("images/ng-2.png", "NG")],
            )
            merged, summary = assemble_training_json.assemble(
                previous, mined, dedupe=True, validation_paths=[]
            )
            self.assertEqual(len(merged), 3)
            self.assertEqual(summary["duplicates_skipped"], 1)
            self.assertEqual(summary["unique_target_images"]["new_after_dedup"], 1)

    def test_evaluation_samples_cannot_enter_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mined = write_json(
                root / "mined.json", [record("images/benchmark.png", "NG")]
            )
            benchmark = write_json(
                root / "benchmark.json", [record("images/benchmark.png", "NG")]
            )
            with self.assertRaisesRegex(ValueError, "train/evaluation leakage"):
                assemble_training_json.assemble(
                    None, mined, dedupe=True, validation_paths=[benchmark]
                )


class StateRoutingTests(unittest.TestCase):
    def test_baseline_starts_with_benchmark_evaluation(self) -> None:
        self.assertEqual(
            deft_context._next_stage(
                {"status": "in_progress", "current_iteration": 0, "iterations": {}}
            ),
            ("baseline", "evaluate_benchmark"),
        )

    def test_failed_benchmark_routes_to_proxy_then_real_mining(self) -> None:
        state = {
            "status": "in_progress",
            "current_iteration": 0,
            "max_iterations": 5,
            "iterations": {
                "baseline": {
                    "stage_completed": "benchmark_metrics",
                    "metric_result": {"passed": False},
                }
            },
        }
        self.assertEqual(deft_context._next_stage(state), ("baseline", "evaluate_proxy"))
        state["iterations"]["baseline"]["stage_completed"] = "routing"
        self.assertEqual(deft_context._next_stage(state), ("baseline", "data_mining"))

    def test_completed_data_validation_routes_directly_to_training(self) -> None:
        state = {
            "status": "in_progress",
            "current_iteration": 1,
            "max_iterations": 5,
            "iterations": {"iter1": {"stage_completed": "validate_data"}},
        }
        self.assertEqual(deft_context._next_stage(state), ("iter1", "train"))


if __name__ == "__main__":
    unittest.main()
