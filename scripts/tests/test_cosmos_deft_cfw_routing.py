# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
RESOLVER_PATH = ROOT / "scripts" / "resolve_tao_model.py"
WORKFLOW_PATH = (
    ROOT
    / "skills"
    / "models"
    / "tao-finetune-cosmos-reason"
    / "scripts"
    / "cosmos_workflow.py"
)
sys.path.insert(0, str(WORKFLOW_PATH.parent))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resolver = _load("deft_cfw_resolver", RESOLVER_PATH)
workflow = _load("deft_cfw_workflow", WORKFLOW_PATH)


class DeftCfwRoutingTests(unittest.TestCase):
    def test_deft_aoi_actions_route_to_framework_in_both_resolvers(self) -> None:
        for action in ("train", "evaluate", "inference"):
            resolved = resolver.resolve_model(
                ROOT,
                "Cosmos3-Nano",
                action=action,
                workload="deft-aoi",
            )
            self.assertIsNotNone(resolved)
            assert resolved is not None
            self.assertEqual(resolved["selected_backend"], "cosmos-framework")
            self.assertIn("DEFT AOI", resolved["backend_selection_reason"])

            selected, reason = workflow.select_backend(
                model="Cosmos3-Nano",
                action=action,
                workload="deft-aoi",
            )
            self.assertEqual(selected, "cosmos-framework")
            self.assertIn("DEFT AOI", reason)

    def test_explicit_backend_wins_and_unrelated_defaults_do_not_move(self) -> None:
        resolved = resolver.resolve_model(
            ROOT,
            "Cosmos3-Nano",
            action="train",
            backend="cosmos-rl",
            workload="deft-aoi",
        )
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved["selected_backend"], "cosmos-rl")

        self.assertEqual(
            workflow.select_backend(
                model="Cosmos3-Nano", action="train", workload="training"
            )[0],
            "cosmos-rl",
        )
        self.assertEqual(
            workflow.select_backend(
                model="Cosmos3-Nano", action="train", workload="automl"
            )[0],
            "cosmos-rl",
        )


if __name__ == "__main__":
    unittest.main()
