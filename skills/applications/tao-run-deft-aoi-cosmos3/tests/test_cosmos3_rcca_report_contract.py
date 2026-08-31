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
                        "--duration-sec", "1",
                        "--summary", "Proxy RCCA complete",
                    ]
                )
            self.assertEqual(rc, 0)
            phase = json.loads((results / "deft_state.json").read_text())["iterations"]["baseline"]
            self.assertEqual(phase["rcca_report"], str(report.resolve()))
            self.assertEqual(phase["gap_candidate_count"], 1)
            self.assertEqual(phase["selected_gap_count"], 1)


if __name__ == "__main__":
    unittest.main()
