# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pathlib
import sys
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import metric_contract  # noqa: E402


class Cosmos3MetricContractTests(unittest.TestCase):
    def test_component_mapping_order_is_not_semantic(self) -> None:
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
                    "path_template": "/results/{iter_label}/metric_result.json",
                },
                "constraints": [],
                "tie_breakers": [],
                "kpi_profile": "f1_cohort_balanced_v1",
                "required_components": required,
                "component_threshold": 0.6,
            }
        )
        result = {
            "name": "f1_cohort_balanced_v1",
            "value": 0.5,
            "unit": "",
            "constraints": {},
            "kpi_profile": "f1_cohort_balanced_v1",
            "required_components": required,
            "components": {
                "cohort.task.DET": {
                    "f1": 0.3,
                    "threshold": 0.6,
                    "attainment": 0.5,
                    "passed": False,
                },
                "cohort.task.BCQ": {
                    "f1": 0.6,
                    "threshold": 0.6,
                    "attainment": 1.0,
                    "passed": True,
                },
            },
            "component_threshold": 0.6,
            "minimum_f1": 0.3,
            "tie_breakers": {},
        }

        normalized = metric_contract.result_from_iteration(
            {"metric_result": result}, contract
        )

        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["value"], 0.5)
        self.assertEqual(list(normalized["components"]), required)


if __name__ == "__main__":
    unittest.main()
