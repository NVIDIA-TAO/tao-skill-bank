#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
for module_name in (
    "audit_deft_run",
    "commit_stage",
    "init_deft_state",
    "metric_contract",
    "render_report",
):
    sys.modules.pop(module_name, None)
sys.path.insert(0, str(SCRIPTS))

import audit_deft_run  # noqa: E402
import commit_stage  # noqa: E402
import init_deft_state  # noqa: E402
import render_report  # noqa: E402


class ReportRenderingTests(unittest.TestCase):
    def test_allocation_proof_is_summed_and_rejects_invalid_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            allocation = pathlib.Path(temporary) / "allocation.json"
            allocation.write_text(
                json.dumps({"bridge": 3, "missing": 2}), encoding="utf-8"
            )
            path, total = commit_stage._required_allocation(
                allocation.resolve(), "--anomalygen-allocation"
            )
            self.assertEqual(path, str(allocation.resolve()))
            self.assertEqual(total, 5)

            allocation.write_text(json.dumps({"bridge": True}), encoding="utf-8")
            with self.assertRaises(ValueError):
                commit_stage._required_allocation(
                    allocation.resolve(), "--anomalygen-allocation"
                )

    def test_stage_commit_requires_positive_measured_duration(self) -> None:
        base = [
            "--results-dir",
            "/tmp/deft-duration-contract",
            "--iter-label",
            "baseline",
            "--stage",
            "loop_stop",
            "--summary",
            "done",
        ]
        with self.assertRaises(SystemExit):
            commit_stage._parser().parse_args(base)
        self.assertEqual(commit_stage.main([*base, "--duration-sec", "0"]), 2)

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
                "num_gpus": 1,
                "gpu_model": "NVIDIA RTX PRO 6000 Blackwell (96 GB)",
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
            state = self._state(results)
            (results / "training.csv").write_text(
                "input_path,label\n" + "".join(f"base-{index}.png,PASS\n" for index in range(118)),
                encoding="utf-8",
            )
            (results / "kpi.csv").write_text(
                "input_path,label\na.png,PASS\nb.png,PASS\nc.png,NG-Missing\n",
                encoding="utf-8",
            )
            combined = results / "iter1-combined.csv"
            combined.write_text(
                "input_path,label\n" + "".join(f"train-{index}.png,PASS\n" for index in range(120)),
                encoding="utf-8",
            )
            mining_summary = results / "iter1-knn-summary.csv"
            mining_summary.write_text(
                "candidate_count,kept_count,rejected_count,similarity_threshold\n5,2,3,0.9\n",
                encoding="utf-8",
            )
            sdg_csv = results / "iter1-sdg.csv"
            sdg_csv.write_text(
                "image,label\n"
                + "".join(f"sdg-{index}.png,NG\n" for index in range(5)),
                encoding="utf-8",
            )
            state["iterations"]["iter1"]["combined_training_csv"] = str(combined)
            state["iterations"]["iter1"]["mining_summary"] = str(mining_summary)
            state["iterations"]["iter1"]["anomalygen_sdg_csv"] = str(sdg_csv)
            (results / "deft_state.json").write_text(json.dumps(state), encoding="utf-8")
            (results / "loop_log.jsonl").write_text(
                json.dumps(
                    {
                        "seq": 1,
                        "ts": "2026-08-04T00:01:00Z",
                        "iter": "iter1",
                        "stage": "evaluate",
                        "status": "ok",
                        "summary": "done",
                        "duration_sec": 120,
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
            self.assertIn("Run Configuration &amp; Outcome", text)
            self.assertIn("NVIDIA RTX PRO 6000 Blackwell (96 GB)", text)
            self.assertIn("1 iters × ~2m 0s = 2m 0s total time", text)
            self.assertIn("KNN Raw Mined", text)
            self.assertIn("SDG Generated", text)
            self.assertIn("New Unique Images (After Dedup)", text)
            self.assertIn(">118</td>", text)
            self.assertIn(">120</td>", text)
            self.assertIn(">+2</td>", text)
            self.assertNotRegex(text, r"\{\{\s+[A-Z0-9_]+\s+\}\}")
            self.assertNotIn("</div><script>alert('x')</script>", text)
            self.assertIn("&lt;/div&gt;&lt;script&gt;alert", text)
            self.assertIsNone(audit_deft_run._completion_report_error(results))

    def test_terminal_gap_has_no_informational_kpi_banner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary)
            state = self._state(results)
            state["iterations"]["iter1"]["metric_result"]["value"] = 0.025
            (results / "deft_state.json").write_text(
                json.dumps(state), encoding="utf-8"
            )
            (results / "loop_log.jsonl").write_text(
                json.dumps(
                    {
                        "seq": 1,
                        "ts": "2026-08-04T00:02:00Z",
                        "iter": "iter1",
                        "stage": "loop_stop",
                        "status": "ok",
                        "summary": "iteration budget reached",
                        "duration_sec": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            text = render_report.render(results).read_text(encoding="utf-8")
            self.assertIn("0.005 cost/board from target", text)
            self.assertNotIn("Best result so far", text)
            self.assertNotIn(">i</div>", text)
            self.assertNotIn("KPI MET", text)

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
                    "--gpu-model",
                    "NVIDIA RTX PRO 6000 Blackwell (96 GB)",
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
