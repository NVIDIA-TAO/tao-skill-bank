# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import metric_contract  # noqa: E402
import record_replacement_benchmark_metric  # noqa: E402


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReplacementBenchmarkTrajectoryTests(unittest.TestCase):
    def test_attachment_is_atomic_idempotent_and_does_not_change_pipeline_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            evaluator = root / "calculate_f1_metrics.py"
            evaluator.write_text("# frozen evaluator\n", encoding="utf-8")
            benchmark = root / "benchmark.jsonl"
            benchmark.write_text('{"id":"row-1"}\n', encoding="utf-8")
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps(
                    {
                        "id": "row-1",
                        "task_type": "Component Classification",
                        "message": [],
                        "GT": "A",
                        "raw_prediction": "A",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            raw_report = root / "raw_f1_report.json"
            raw_report.write_text(
                json.dumps(
                    {
                        "source": str(benchmark.resolve()),
                        "alignment": {
                            "source_rows": 1,
                            "evaluated_source_rows": 1,
                            "prediction_rows": 1,
                            "missing_evaluated_predictions": 0,
                            "unknown_prediction_ids": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            required = ["cohort.task.BCQ", "cohort.task.DET"]
            contract = metric_contract.validate_contract(
                {
                    "name": "f1_cohort_balanced_v1",
                    "display_name": "Worst required cohort F1 attainment",
                    "operator": ">=",
                    "target": 1.0,
                    "unit": "",
                    "evaluator": {
                        "type": "artifact",
                        "producer": "scripts/exact_f1_adapter.py",
                        "path_template": str(root / "{iter_label}" / "metric_result.json"),
                    },
                    "constraints": [],
                    "tie_breakers": [],
                    "kpi_profile": "f1_cohort_balanced_v1",
                    "required_components": required,
                    "component_threshold": 0.6,
                }
            )
            metric_result = {
                "name": "f1_cohort_balanced_v1",
                "value": 0.5,
                "unit": "",
                "constraints": {
                    "missing_evaluated_predictions": 0,
                    "unknown_prediction_ids": 0,
                },
                "kpi_profile": "f1_cohort_balanced_v1",
                "required_components": required,
                "components": {
                    "cohort.task.BCQ": {
                        "f1": 0.7,
                        "threshold": 0.6,
                        "attainment": 1.0,
                        "passed": True,
                    },
                    "cohort.task.DET": {
                        "f1": 0.3,
                        "threshold": 0.6,
                        "attainment": 0.5,
                        "passed": False,
                    },
                },
                "component_threshold": 0.6,
                "minimum_f1": 0.3,
                "tie_breakers": {},
                "evaluator_path": str(evaluator.resolve()),
                "evaluator_sha256": sha256(evaluator),
                "raw_report_path": str(raw_report.resolve()),
                "raw_report_sha256": sha256(raw_report),
                "alignment": {
                    "source_rows": 1,
                    "evaluated_source_rows": 1,
                    "prediction_rows": 1,
                    "missing_evaluated_predictions": 0,
                    "unknown_prediction_ids": 0,
                },
            }
            metric_path = root / "metric_result.json"
            metric_path.write_text(json.dumps(metric_result), encoding="utf-8")
            state = {
                "version": 7,
                "current_iteration": 3,
                "status": "in_progress",
                "metric_contract": contract,
                "metric_contract_sha256": metric_contract.contract_sha256(contract),
                "config": {
                    "base_model": "/models/base",
                    "annotations": {"benchmark": str(benchmark.resolve())},
                    "annotation_sha256": {"benchmark": sha256(benchmark)},
                    "evaluation": {
                        "benchmark": {
                            "annotations": str(benchmark.resolve()),
                            "sha256": sha256(benchmark),
                        }
                    },
                    "kpi": {
                        "evaluator": str(evaluator.resolve()),
                        "evaluator_sha256": sha256(evaluator),
                    },
                },
                "iterations": {
                    "baseline": {
                        "status": "complete",
                        "stage_completed": "benchmark_metrics",
                        "benchmark_predictions_jsonl": "/old/predictions.jsonl",
                        "metric_result": {"minimum_f1": 0.1, "passed": False},
                    },
                    "iter3": {"status": "in_progress", "stage_completed": "validate_data"},
                },
                "events": [{"seq": 1, "iter": "iter3", "stage": "validate_data"}],
            }
            state_path = root / "deft_state.json"
            state_path.write_text(json.dumps(state), encoding="utf-8")
            progress_before = copy.deepcopy(
                {key: state[key] for key in ("current_iteration", "status", "iterations", "events")}
            )
            args = argparse.Namespace(
                state_path=state_path,
                iter_label="baseline",
                result_json=metric_path,
                benchmark_results=predictions,
                raw_f1_report=raw_report,
                benchmark_rows=1,
                training_spec=None,
            )

            first = record_replacement_benchmark_metric.commit(args)
            second = record_replacement_benchmark_metric.commit(args)

            updated = json.loads(state_path.read_text(encoding="utf-8"))
            progress_after = {
                key: updated[key]
                for key in ("current_iteration", "status", "iterations", "events")
            }
            self.assertEqual(progress_after, progress_before)
            self.assertEqual(first, second)
            trajectory = updated["benchmark_trajectory"]
            self.assertEqual(trajectory["cohort_rows"], 1)
            self.assertEqual(trajectory["cohort_sha256"], sha256(benchmark))
            self.assertEqual(list(trajectory["evaluations"]), ["baseline"])
            self.assertEqual(
                trajectory["evaluations"]["baseline"]["metric_result"]["minimum_f1"],
                0.3,
            )
            self.assertEqual(
                trajectory["evaluations"]["baseline"]["supersedes"]["benchmark_predictions_jsonl"],
                "/old/predictions.jsonl",
            )


if __name__ == "__main__":
    unittest.main()
