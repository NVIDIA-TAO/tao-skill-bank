# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pathlib
import sys
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import benchmark_cadence  # noqa: E402
import deft_context  # noqa: E402


def _state(
    label: str,
    completed: str,
    *,
    cadence: str = "final_and_best",
    current: int = 1,
    maximum: int = 5,
    proxy_is_best: bool = False,
) -> dict:
    phase = {"status": "in_progress", "stage_completed": completed}
    if completed == "proxy_rcca":
        phase["proxy_is_best_so_far"] = proxy_is_best
    if completed == "benchmark_metrics":
        phase["metric_result"] = {"passed": False}
    return {
        "version": 7,
        "status": "in_progress",
        "current_iteration": current,
        "max_iterations": maximum,
        "config": {"evaluation": {"benchmark_cadence": cadence}},
        "iterations": {label: phase},
    }


class Cosmos3BenchmarkCadenceContractTests(unittest.TestCase):
    def test_proxy_is_always_scored_immediately_after_training(self) -> None:
        self.assertEqual(
            deft_context._next_stage(_state("iter1", "train")),
            ("iter1", "evaluate_proxy"),
        )

    def test_final_and_best_skips_benchmark_for_non_best_non_final_iteration(self) -> None:
        self.assertEqual(
            deft_context._next_stage(_state("iter1", "proxy_rcca")),
            ("iter2", "routing"),
        )

    def test_final_and_best_scores_new_best_and_final_checkpoint(self) -> None:
        self.assertEqual(
            deft_context._next_stage(
                _state("iter2", "proxy_rcca", current=2, proxy_is_best=True)
            ),
            ("iter2", "evaluate_benchmark"),
        )
        self.assertEqual(
            deft_context._next_stage(
                _state("iter5", "proxy_rcca", current=5, proxy_is_best=False)
            ),
            ("iter5", "evaluate_benchmark"),
        )

    def test_every_scores_each_checkpoint_after_proxy(self) -> None:
        self.assertEqual(
            deft_context._next_stage(
                _state("iter1", "proxy_rcca", cadence="every")
            ),
            ("iter1", "evaluate_benchmark"),
        )

    def test_nonterminal_benchmark_continues_directly_to_next_iteration(self) -> None:
        self.assertEqual(
            deft_context._next_stage(_state("iter1", "benchmark_metrics")),
            ("iter2", "routing"),
        )

    def test_proxy_best_uses_full_deterministic_kpi_ranking(self) -> None:
        baseline = {
            "value": 0.5,
            "tie_breakers": {
                "minimum_f1": 0.3,
                "mean_f1": 0.5,
                "coverage_failures": 0.0,
            },
        }
        improved = {
            "value": 0.5,
            "tie_breakers": {
                "minimum_f1": 0.3,
                "mean_f1": 0.51,
                "coverage_failures": 0.0,
            },
        }

        is_best, evidence = benchmark_cadence.is_proxy_best_so_far(
            improved, [baseline]
        )

        self.assertTrue(is_best)
        self.assertEqual(evidence["comparison_count"], 1)
        self.assertEqual(evidence["ranking"], [0.5, 0.3, 0.51, -0.0])

if __name__ == "__main__":
    unittest.main()
