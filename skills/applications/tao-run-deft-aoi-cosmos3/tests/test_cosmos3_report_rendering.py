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


if __name__ == "__main__":
    unittest.main()
