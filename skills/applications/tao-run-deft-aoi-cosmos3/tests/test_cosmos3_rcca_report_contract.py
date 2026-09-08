#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import commit_stage  # noqa: E402


class Cosmos3RccaReportContractTests(unittest.TestCase):
    def test_proxy_rcca_commit_records_current_artifact_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary) / "results"
            proxy_dir = results / "baseline/proxy_rcca"
            proxy_dir.mkdir(parents=True)
            state = {
                "version": 7,
                "status": "in_progress",
                "results_dir": str(results),
                "max_iterations": 1,
                "current_iteration": 0,
                "config": {
                    "evaluation": {"benchmark_cadence": "final_and_best"},
                    "kpi": {
                        "evaluator": str(results / "evaluator.py"),
                        "evaluator_sha256": "a" * 64,
                        "component_threshold": 0.6,
                    },
                },
                "metric_contract": {"required_components": ["component-a"]},
                "iterations": {
                    "baseline": {
                        "status": "in_progress",
                        "stage_completed": "evaluate_proxy",
                    }
                },
                "events": [],
            }
            (results / "deft_state.json").write_text(json.dumps(state), encoding="utf-8")
            summary = proxy_dir / "gaps_summary.json"
            summary.write_text('{"candidate_count":1}\n', encoding="utf-8")
            candidates = proxy_dir / "gap_candidates.parquet"
            selected = proxy_dir / "selected_gaps.parquet"
            table = pa.table({"id": ["sample-1"]})
            pq.write_table(table, candidates)
            pq.write_table(table, selected)
            report = proxy_dir / "RCCA_Report.md"
            report.write_text(
                "# Proxy RCCA\n\n"
                "## Executive Summary\nsummary\n\n"
                "## Failure Mode Analysis\nanalysis\n\n"
                "## Root Cause Analysis\ncause\n\n"
                "## Corrective Actions\nactions\n\n"
                "## Validation Plan\nplan\n",
                encoding="utf-8",
            )
            raw_f1 = proxy_dir / "raw_f1.json"
            raw_f1.write_text('{"proxy":true}\n', encoding="utf-8")
            metric_result = proxy_dir / "metric_result.json"
            metric_result.write_text(
                json.dumps(
                    {
                        "name": "f1_cohort_balanced_v1",
                        "value": 0.5,
                        "component_threshold": 0.6,
                        "required_components": ["component-a"],
                        "constraints": {
                            "missing_evaluated_predictions": 0,
                            "unknown_prediction_ids": 0,
                        },
                        "tie_breakers": {
                            "minimum_f1": 0.3,
                            "mean_f1": 0.5,
                            "coverage_failures": 0.0,
                        },
                        "evaluator_path": str(results / "evaluator.py"),
                        "evaluator_sha256": "a" * 64,
                        "raw_report_path": str(raw_f1.resolve()),
                        "raw_report_sha256": hashlib.sha256(
                            raw_f1.read_bytes()
                        ).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                commit_stage,
                "render_html_report",
                return_value=results / "DEFT_Loop_Report.html",
            ):
                rc = commit_stage.main(
                    [
                        "--results-dir", str(results),
                        "--iter-label", "baseline",
                        "--stage", "proxy_rcca",
                        "--proxy-gaps-summary", str(summary),
                        "--gap-candidates", str(candidates),
                        "--selected-gaps", str(selected),
                        "--rcca-report", str(report),
                        "--proxy-raw-f1-report", str(raw_f1),
                        "--proxy-metric-result", str(metric_result),
                        "--duration-sec", "1",
                        "--summary", "Proxy RCCA complete",
                    ]
                )
            self.assertEqual(rc, 0)
            phase = json.loads((results / "deft_state.json").read_text())["iterations"]["baseline"]
            self.assertEqual(phase["rcca_report"], str(report.resolve()))
            self.assertEqual(phase["gap_candidate_count"], 1)
            self.assertEqual(phase["selected_gap_count"], 1)
            self.assertTrue(phase["proxy_is_best_so_far"])
            self.assertEqual(
                phase["proxy_metric_result"]["evaluation_role"], "proxy"
            )


if __name__ == "__main__":
    unittest.main()
