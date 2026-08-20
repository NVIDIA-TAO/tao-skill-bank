# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def test_evaluation_planner_exposes_recipe_owned_model_length_override() -> None:
    source = (
        Path(__file__).parents[1] / "scripts" / "evaluation_workflow.py"
    ).read_text(encoding="utf-8")

    assert 'parser.add_argument("--model-max-length", type=int)' in source
    assert 'getattr(args, "model_max_length", None)' in source
    assert '"user" if requested_model_max_length is not None' in source
