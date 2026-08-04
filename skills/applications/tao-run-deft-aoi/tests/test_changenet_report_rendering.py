#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
for module_name in ("audit_deft_run", "init_deft_state", "metric_contract", "render_report"):
    sys.modules.pop(module_name, None)
sys.path.insert(0, str(SCRIPTS))

import audit_deft_run  # noqa: E402
import init_deft_state  # noqa: E402
import render_report  # noqa: E402


class ReportRenderingTests(unittest.TestCase):
    def _state(self, results: pathlib.Path) -> dict:
        contract = {
            "name": "escape_cost",
            "display_name": "Weighted escape cost",
            "operator": "<=",
            "target": 0.02,
            "unit": "cost/board",
            "evaluator": {
                "type": "artifact",
                "producer": "test",
                "path_template": str(
                    results / "{iter_label}/evaluate/metric_result.json"
                ),
            },
            "constraints": [],
        }
        return {
            "version": 2,
            "started_at": "2026-08-04T00:00:00+00:00",
            "kpi_target": "Weighted escape cost <= 0.02 cost/board",
            "metric_contract": contract,
            "results_dir": str(results),
            "max_iterations": 2,
            "current_iteration": 1,
            "config": {
                "kpi_test_csv": str(results / "kpi.csv"),
                "training_csv": str(results / "training.csv"),
                "mining_filter": {"metric": "cosine", "min_similarity": 0.9},
            },
            "iterations": {
                "baseline": {
                    "status": "complete",
                    "stage_completed": "evaluate",
                    "best_ckpt_path": str(results / "baseline/train/model.ckpt"),
                    "threshold": 0.5,
                    "metric_result": {
                        "name": "escape_cost",
                        "value": 0.031,
                        "unit": "cost/board",
                        "constraints": {},
                    },
                },
                "iter1": {
                    "status": "complete",
                    "stage_completed": "evaluate",
                    "best_ckpt_path": "</div><script>alert('x')</script>",
                    "threshold": 0.47,
                    "mining_mined_count": 8,
                    "metric_result": {
                        "name": "escape_cost",
                        "value": 0.018,
                        "unit": "cost/board",
                        "constraints": {},
                    },
                },
            },
            "_completed_step_values": [],
            "_status_values": [],
        }

    def test_renders_release_style_template_and_escapes_disk_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary)
            (results / "deft_state.json").write_text(
                json.dumps(self._state(results)), encoding="utf-8"
            )
            (results / "loop_log.jsonl").write_text(
                json.dumps(
                    {
                        "seq": 1,
                        "ts": "2026-08-04T00:01:00Z",
                        "iter": "iter1",
                        "stage": "evaluate",
                        "status": "ok",
                        "summary": "done",
                        "duration_sec": 1,
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "seq": 2,
                        "ts": "2026-08-04T00:02:00Z",
                        "iter": "iter1",
                        "stage": "loop_stop",
                        "status": "ok",
                        "summary": "target met",
                        "duration_sec": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = render_report.render(results)
            text = output.read_text(encoding="utf-8")
            self.assertIn("DEFT Loop Final Report", text)
            self.assertIn("--nvidia-green: #76b900", text)
            self.assertIn("KPI MET", text)
            self.assertNotRegex(text, r"\{\{\s+[A-Z0-9_]+\s+\}\}")
            self.assertNotIn("</div><script>alert('x')</script>", text)
            self.assertIn("&lt;/div&gt;&lt;script&gt;alert", text)
            self.assertIsNone(audit_deft_run._completion_report_error(results))

    def test_partial_state_still_produces_a_complete_live_shell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary)
            state = self._state(results)
            state["iterations"] = {}
            state["current_iteration"] = 0
            (results / "deft_state.json").write_text(
                json.dumps(state), encoding="utf-8"
            )
            output = render_report.render(results)
            text = output.read_text(encoding="utf-8")
            self.assertIn("IN PROGRESS", text)
            self.assertIn("No completed evaluation yet", text)
            self.assertNotRegex(text, r"\{\{\s+[A-Z0-9_]+\s+\}\}")

    def test_inline_chart_json_cannot_close_the_script_element(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary)
            state = self._state(results)
            state["metric_contract"]["unit"] = "</script><script>alert(1)</script>"
            for phase in state["iterations"].values():
                phase["metric_result"]["unit"] = state["metric_contract"]["unit"]
            (results / "deft_state.json").write_text(
                json.dumps(state), encoding="utf-8"
            )
            text = render_report.render(results).read_text(encoding="utf-8")
            self.assertNotIn('const metricUnit = "</script>', text)
            self.assertIn(r"\u003c/script\u003e", text)

    def test_initialization_hook_writes_the_live_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            results = root / "results"
            rc = init_deft_state.main(
                [
                    "--results-dir",
                    str(results),
                    "--workspace",
                    str(root / "workspace"),
                    "--kpi-target",
                    "FAR <= 1% at recall=100%",
                    "--max-iterations",
                    "1",
                    "--num-gpus",
                    "1",
                    "--num-epochs",
                    "1",
                    "--num-sdg",
                    "2",
                    "--project",
                    "nvpcb",
                    "--step",
                    "1",
                    "--train-container",
                    "example/train:1",
                    "--ag-container",
                    "example/anomalygen:1",
                ]
            )
            self.assertEqual(rc, 0)
            report = results / "DEFT_Loop_Report.html"
            self.assertTrue(report.is_file())
            self.assertIn("IN PROGRESS", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
