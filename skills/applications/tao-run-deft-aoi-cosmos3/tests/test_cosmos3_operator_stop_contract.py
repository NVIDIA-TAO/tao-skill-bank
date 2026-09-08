#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import commit_stage  # noqa: E402


class Cosmos3OperatorStopContractTests(unittest.TestCase):
    def test_operator_can_stop_after_proxy_rcca_without_benchmark_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary) / "results"
            results.mkdir()
            final_report = results / "DEFT_Loop_Report.html"
            final_report.write_text("operator stop report\n", encoding="utf-8")
            state = {
                "version": 7,
                "status": "in_progress",
                "results_dir": str(results),
                "max_iterations": 5,
                "current_iteration": 1,
                "config": {"evaluation": {"benchmark_cadence": "final_and_best"}},
                "iterations": {
                    "baseline": {"status": "complete"},
                    "iter1": {
                        "status": "complete",
                        "stage_completed": "proxy_rcca",
                        "proxy_raw_f1_report": str(results / "proxy_raw_f1.json"),
                        "proxy_metric_result": {"value": 0.4, "passed": False},
                        "proxy_is_best_so_far": True,
                    },
                },
                "events": [],
            }
            (results / "deft_state.json").write_text(
                json.dumps(state), encoding="utf-8"
            )
            reason = "iteration 1 contains a confounded calibration intervention"
            with mock.patch.object(
                commit_stage,
                "render_html_report",
                return_value=final_report,
            ):
                rc = commit_stage.main(
                    [
                        "--results-dir", str(results),
                        "--iter-label", "iter1",
                        "--stage", "loop_stop",
                        "--summary", "operator stop",
                        "--duration-sec", "1",
                        "--stop-reason", "operator_requested",
                        "--operator-reason", reason,
                        "--final-report", str(final_report),
                    ]
                )
            self.assertEqual(rc, 0)
            updated = json.loads((results / "deft_state.json").read_text())
            self.assertEqual(updated["status"], "complete")
            self.assertEqual(updated["stop_reason"], "operator_requested")
            self.assertEqual(updated["operator_stop"]["after_stage"], "proxy_rcca")
            self.assertFalse(updated["operator_stop"]["benchmark_scored"])
            self.assertEqual(updated["operator_stop"]["reason"], reason)
            self.assertNotIn("metric_result", updated["iterations"]["iter1"])


if __name__ == "__main__":
    unittest.main()
