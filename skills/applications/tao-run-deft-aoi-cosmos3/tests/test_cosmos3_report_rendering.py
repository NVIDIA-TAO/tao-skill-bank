#!/usr/bin/env python3
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

import render_report  # noqa: E402


class CosmosReportRenderingTests(unittest.TestCase):
    def test_framework_report_renders_state_and_escapes_event_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary)
            state = {
                "version": 7,
                "status": "complete",
                "config": {
                    "kpi": {
                        "profile": "f1_cohort_balanced_v1",
                        "component_threshold": 0.8,
                        "evaluator": "/workspace/eval/calculate_f1_metrics.py",
                    },
                    "training": {"backend": "cosmos-framework"},
                },
                "iterations": {
                    "iter1": {
                        "status": "complete",
                        "stage_completed": "benchmark_metrics",
                        "best_ckpt_path": "/results/iter1/train/checkpoints/iter_000000500",
                        "metric_result": {"minimum_f1": 0.84, "passed": True},
                    }
                },
                "events": [
                    {
                        "seq": 1,
                        "iter": "iter1",
                        "stage": "benchmark_metrics",
                        "status": "ok",
                        "duration_sec": 12,
                        "summary": "<img src=x onerror=alert(1)>",
                    }
                ],
            }
            (results / "deft_state.json").write_text(json.dumps(state), encoding="utf-8")
            output = render_report.render(results)
            text = output.read_text(encoding="utf-8")
            self.assertIn("DEFT AOI · Cosmos Framework", text)
            self.assertIn("f1_cohort_balanced_v1", text)
            self.assertIn("minimum F1=0.84 · gate=PASS", text)
            self.assertIn("&lt;img src=x onerror=alert(1)&gt;", text)
            self.assertNotIn("<img src=x onerror=alert(1)>", text)

    def test_operator_overlap_exception_is_rendered_as_selection_bias_disclosure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary)
            state = {
                "version": 7,
                "status": "in_progress",
                "config": {"kpi": {}, "training": {"backend": "cosmos-framework"}},
                "iterations": {},
                "events": [],
                "operator_contract_changes": [
                    {
                        "schema": "deft_operator_contract_change_audit_v1",
                        "active_benchmark_rows": 20657,
                        "active_benchmark_sha256": "1a385f67" + "0" * 56,
                        "authorized_overlap_exception": {
                            "physical_target_overlap": 797,
                            "benchmark_rows_on_overlapping_targets": 2001,
                            "proxy_rows_on_overlapping_targets": 918,
                            "disclosure": "Known selection bias from approved cohort overlap.",
                        },
                    }
                ],
            }
            (results / "deft_state.json").write_text(json.dumps(state), encoding="utf-8")

            text = render_report.render(results).read_text(encoding="utf-8")

            self.assertIn("Known selection-bias disclosure", text)
            self.assertIn("797 physical targets", text)
            self.assertIn("2,001 Benchmark rows", text)
            self.assertIn("918 Proxy rows", text)
            self.assertIn("20,657", text)
            self.assertIn("1a385f67", text)

    def test_replacement_benchmark_trajectory_supersedes_iteration_cards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary)
            state = {
                "version": 7,
                "status": "in_progress",
                "config": {"kpi": {}, "training": {"backend": "cosmos-framework"}},
                "iterations": {
                    "baseline": {
                        "status": "complete",
                        "stage_completed": "benchmark_metrics",
                        "metric_result": {"minimum_f1": 0.12, "passed": False},
                    }
                },
                "events": [],
                "benchmark_trajectory": {
                    "cohort_rows": 20657,
                    "cohort_sha256": "1a385f67" + "0" * 56,
                    "evaluations": {
                        "baseline": {
                            "metric_result": {
                                "minimum_f1": 0.34,
                                "passed": False,
                                "components": {
                                    "non_reference_based.tasks.BCQ.macro_f1": {"f1": 0.61},
                                    "non_reference_based.tasks.DET.f1": {"f1": 0.34},
                                },
                            }
                        }
                    },
                },
            }
            (results / "deft_state.json").write_text(json.dumps(state), encoding="utf-8")

            text = render_report.render(results).read_text(encoding="utf-8")

            self.assertIn("Replacement Benchmark trajectory", text)
            self.assertIn("20,657 rows", text)
            self.assertIn("replacement cohort minimum F1=0.34", text)
            self.assertIn("non_reference_based.tasks.DET.f1: 0.34", text)
            self.assertNotIn("minimum F1=0.12", text)


if __name__ == "__main__":
    unittest.main()
