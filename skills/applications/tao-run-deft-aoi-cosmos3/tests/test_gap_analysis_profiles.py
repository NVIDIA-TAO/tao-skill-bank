#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import pathlib
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from gap_analysis.config import PACKAGED_PROFILES, load_profile, validate_config  # noqa: E402
from gap_analysis.runner import run_selection  # noqa: E402
import replay_gap_analysis  # noqa: E402
import run_gap_analysis  # noqa: E402


def candidates() -> list[dict]:
    rows: list[dict] = []
    for task, count, dataset, base in (
        ("Task A", 8, "dataset-a", 0.10),
        ("Task B", 4, "dataset-b", 0.80),
    ):
        for index in range(count):
            rows.append(
                {
                    "id": f"{task[-1].lower()}{index}",
                    "target_id": f"target-{task[-1].lower()}{index}",
                    "target_path": f"images/{task[-1].lower()}{index}.png",
                    "image_paths": [f"images/{task[-1].lower()}{index}.png"],
                    "evaluation_role": "proxy",
                    "task_type": task,
                    "metric_family": "classification",
                    "reference_cohort": "single_target",
                    "dataset": dataset,
                    "sample_score": 1.0 - min(base + index * 0.01, 1.0),
                    "weakness_score": min(base + index * 0.01, 1.0),
                    "gap_type": "classification_error",
                    "parse_ok": True,
                    "prediction_json": "{}",
                    "ground_truth_json": "{}",
                    "metadata_json": "{}",
                }
            )
    return rows


class GapProfileTests(unittest.TestCase):
    def test_all_packaged_profiles_validate_and_unknown_keys_fail(self) -> None:
        for name in PACKAGED_PROFILES:
            with self.subTest(profile=name):
                config = load_profile(name)
                self.assertEqual(config["schema_version"], "gap_analysis_v1")
        invalid = load_profile("global_topk")
        invalid["surprise"] = True
        with self.assertRaisesRegex(ValueError, "unknown gap-analysis keys.*surprise"):
            validate_config(invalid)
        incompatible = load_profile("legacy_bare_okng")
        incompatible["scorer"] = {"name": "sample_metric_deficit"}
        with self.assertRaisesRegex(ValueError, "requires the binary_error scorer"):
            validate_config(incompatible)

    def test_equal_round_robin_balances_an_imbalanced_candidate_set(self) -> None:
        global_config = load_profile("global_topk")
        global_config.update({"budget": 4, "fraction_per_group": 1.0})
        global_selected, _ = run_selection(candidates(), global_config)

        balanced_config = load_profile("equal_task_round_robin")
        balanced_config.update(
            {"budget": 4, "fraction_per_group": 1.0, "min_per_group": 0}
        )
        balanced_selected, summary = run_selection(candidates(), balanced_config)
        global_counts = {task: sum(row["task_type"] == task for row in global_selected) for task in ("Task A", "Task B")}
        balanced_counts = {task: sum(row["task_type"] == task for row in balanced_selected) for task in ("Task A", "Task B")}
        self.assertNotEqual(global_counts, balanced_counts)
        self.assertEqual(balanced_counts, {"Task A": 2, "Task B": 2})
        self.assertEqual(summary["realized_budget"], 4)
        self.assertEqual(len({row["id"] for row in balanced_selected}), 4)
        self.assertIn("per_group_mean_weakness", summary)
        self.assertEqual(summary["duplicate_target_rate"], 0.0)
        self.assertIn("max_per_dataset", summary["caps"])

    def test_deficit_weighting_prioritizes_the_weaker_task(self) -> None:
        config = load_profile("deficit_weighted_round_robin")
        config.update(
            {"budget": 6, "fraction_per_group": 1.0, "min_per_group": 0}
        )
        selected, summary = run_selection(candidates(), config)
        counts = {task: sum(row["task_type"] == task for row in selected) for task in ("Task A", "Task B")}
        self.assertGreater(counts["Task B"], counts["Task A"])
        self.assertEqual(summary["requested_budget"], 6)

    def test_seeded_random_is_repeatable_and_seed_changes_selection(self) -> None:
        config = load_profile("random_control")
        config.update(
            {"budget": 4, "fraction_per_group": 1.0, "min_per_group": 0, "seed": 17}
        )
        first, first_summary = run_selection(candidates(), config)
        second, second_summary = run_selection(copy.deepcopy(candidates()), copy.deepcopy(config))
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        self.assertEqual(first_summary["selected_ids_sha256"], second_summary["selected_ids_sha256"])
        changed = copy.deepcopy(config)
        changed["seed"] = 23
        third, _ = run_selection(candidates(), changed)
        self.assertNotEqual([row["id"] for row in first], [row["id"] for row in third])

    def test_benchmark_and_missing_diversity_embeddings_fail_loudly(self) -> None:
        benchmark = candidates()
        benchmark[0]["evaluation_role"] = "benchmark"
        with self.assertRaisesRegex(ValueError, "only Proxy candidates"):
            run_selection(benchmark, load_profile("global_topk"))
        with self.assertRaisesRegex(ValueError, "embedding"):
            run_selection(candidates(), load_profile("hardness_diversity"))

    def test_zero_deficit_rows_are_not_routed_as_gaps(self) -> None:
        perfect = candidates()
        for row in perfect:
            row["sample_score"] = 1.0
            row["weakness_score"] = 0.0
            row["gap_type"] = "correct"
        selected, summary = run_selection(perfect, load_profile("global_topk"))
        self.assertEqual(selected, [])
        self.assertEqual(summary["eligible_candidates"], 0)
        self.assertEqual(summary["realized_budget"], 0)
        self.assertEqual(sum(summary["per_group_support"].values()), len(perfect))

    def test_missing_groups_uncertainty_and_budget_caps_are_explicit(self) -> None:
        missing = load_profile("equal_task_round_robin")
        missing["expected_groups"] = {
            "groups": ["task_type=Task A|metric_family=classification|reference_cohort=single_target", "task_type=Task C"]
        }
        with self.assertRaisesRegex(ValueError, "missing expected gap-analysis groups"):
            run_selection(candidates(), missing)

        uncertainty = load_profile("global_topk")
        uncertainty["scorer"] = {"name": "uncertainty"}
        with self.assertRaisesRegex(ValueError, "no uncertainty field"):
            run_selection(candidates(), uncertainty)

        capped = load_profile("global_topk")
        capped.update({"budget": 100, "max_per_dataset": 2})
        selected, summary = run_selection(candidates(), capped)
        self.assertEqual(len(selected), 4)
        self.assertEqual(summary["budget_shortfall"], 96)
        self.assertEqual(summary["per_dataset_selected"], {"dataset-a": 2, "dataset-b": 2})

    def test_profile_cli_and_offline_replay_emit_reproducible_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            candidate_path = root / "gap_candidates.parquet"
            pq.write_table(pa.Table.from_pylist(candidates()), candidate_path)
            one = root / "one"
            status = run_gap_analysis.main(
                [
                    "--candidates",
                    str(candidate_path),
                    "--profile",
                    "equal_task_round_robin",
                    "--budget",
                    "4",
                    "--output-dir",
                    str(one),
                ]
            )
            self.assertEqual(status, 0)
            self.assertEqual(pq.read_table(one / "selected_gaps.parquet").num_rows, 4)
            summary = json.loads((one / "gap_analysis_summary.json").read_text())
            self.assertEqual(summary["realized_budget"], 4)

            replay = root / "replay"
            status = replay_gap_analysis.main(
                [
                    "--candidates",
                    str(candidate_path),
                    "--profiles",
                    "global_topk,equal_task_round_robin,random_control",
                    "--seeds",
                    "17,23",
                    "--budget",
                    "4",
                    "--output-dir",
                    str(replay),
                ]
            )
            self.assertEqual(status, 0)
            report = json.loads((replay / "replay_summary.json").read_text())
            self.assertEqual(len(report["runs"]), 6)
            self.assertTrue(report["pairwise_overlap"])


if __name__ == "__main__":
    unittest.main()
