# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "automl_vcn_slurm_v2.py"
SPEC = importlib.util.spec_from_file_location("automl_vcn_slurm_v2", MODULE_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_regressing_challenger_rolls_back_to_incumbent():
    incumbent = {"metric_value": 15.1, "checkpoint": "deft.pth"}
    challenger = {"metric_value": 25.0, "checkpoint": "automl.pth"}

    result = runner.select_incumbent_or_challenger(incumbent, challenger)

    assert result["selected"] == "incumbent"
    assert result["selected_artifact"]["checkpoint"] == "deft.pth"


def test_improving_challenger_is_promoted():
    incumbent = {"metric_value": 15.1, "checkpoint": "deft.pth"}
    challenger = {"metric_value": 11.0, "checkpoint": "automl.pth"}

    result = runner.select_incumbent_or_challenger(incumbent, challenger)

    assert result["selected"] == "challenger"
    assert result["selected_artifact"]["checkpoint"] == "automl.pth"


def test_missing_challenger_metric_rolls_back_to_incumbent():
    incumbent = {"metric_value": 15.1, "checkpoint": "deft.pth"}

    result = runner.select_incumbent_or_challenger(
        incumbent,
        {"metric_value": None, "checkpoint": "automl.pth"},
    )

    assert result["selected"] == "incumbent"
