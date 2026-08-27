#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for truthful metrics and mining-only split validation."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import analyze_gaps  # noqa: E402
import validate_split_contract  # noqa: E402


def _record(target: str, label: str) -> dict:
    return {
        "images": [target],
        "conversations": [
            {"from": "human", "value": "Inspect this component."},
            {"from": "gpt", "value": label},
        ],
    }


def _write_json(path: pathlib.Path, payload: object) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


class MetricTruthfulnessTests(unittest.TestCase):
    def test_unknown_predictions_count_against_accuracy(self) -> None:
        summary, *_ = analyze_gaps.analyze(
            [
                {"gt": "OK", "response": "OK"},
                {"gt": "NG", "response": "NG"},
                {"gt": "NG", "response": "Unable to classify"},
            ],
            kpi_metric="accuracy",
            evaluation_role="benchmark",
        )
        self.assertEqual(summary["parseable_samples"], 2)
        self.assertEqual(summary["unknown_samples"], 1)
        self.assertAlmostEqual(summary["metrics"]["accuracy"], 2 / 3)

    def test_unknown_count_is_written_to_metric_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            results = _write_json(
                root / "results.json",
                [
                    {"gt": "OK", "response": "OK"},
                    {"gt": "NG", "response": "not sure"},
                ],
            )
            output_dir = root / "metrics"
            self.assertEqual(
                analyze_gaps.main(
                    [
                        "--results-json", str(results),
                        "--output-dir", str(output_dir),
                        "--evaluation-role", "benchmark",
                        "--kpi-metric", "accuracy",
                    ]
                ),
                0,
            )
            result = json.loads((output_dir / "metric_result.json").read_text())
            self.assertEqual(result["unknown_samples"], 1)
            self.assertEqual(result["value"], 0.5)


class MinedRealSplitTests(unittest.TestCase):
    def test_validator_accepts_only_proxy_benchmark_and_mining_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            roles = {
                role: _write_json(root / f"{role}.json", [_record(f"images/{role}.png", "OK")])
                for role in ("proxy", "benchmark", "mining")
            }
            summary = validate_split_contract.validate(roles, media_root=root)
            self.assertEqual(set(summary["records"]), {"proxy", "benchmark", "mining"})
            self.assertEqual(summary["roles"]["mining"], "mining_pool")

    def test_bare_label_violation_names_the_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            roles = {
                role: _write_json(root / f"{role}.json", [_record(f"images/{role}.png", "OK")])
                for role in ("proxy", "benchmark", "mining")
            }
            roles["mining"] = _write_json(
                root / "mining.json", [_record("images/mining.png", "NOT OK")]
            )
            with self.assertRaisesRegex(ValueError, r"mining\.json\[0\].*exactly OK or NG"):
                validate_split_contract.validate(roles, media_root=root)


if __name__ == "__main__":
    unittest.main()
