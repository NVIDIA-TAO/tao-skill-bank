# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for IAA DEFT Docker GPU scoping."""

import sys
from pathlib import Path

import pytest
import yaml

IAA_SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "applications"
    / "tao-run-deft-iaa"
    / "scripts"
)
sys.path.insert(0, str(IAA_SCRIPTS))
import run_deft_container as container  # noqa: E402

IAA_ROOT = IAA_SCRIPTS.parent


def test_docker_gpu_args_use_exact_approved_devices():
    assert container._docker_gpu_args(  # noqa: SLF001
        {"gpu_ids": [0, 2], "num_gpus": 2}
    ) == ["--gpus", '"device=0,2"']


@pytest.mark.parametrize(
    "config",
    [
        {"gpu_ids": [], "num_gpus": 0},
        {"gpu_ids": [0, 0], "num_gpus": 2},
        {"gpu_ids": [0, 2], "num_gpus": 1},
        {"gpu_ids": [True], "num_gpus": 1},
    ],
)
def test_docker_gpu_args_reject_invalid_or_mismatched_state(config):
    with pytest.raises(ValueError):
        container._docker_gpu_args(config)  # noqa: SLF001


def test_mining_template_has_no_competing_runtime_defaults():
    mining = yaml.safe_load((IAA_ROOT / "specs/mining_spec.yaml").read_text())

    assert mining["topn"] == "???"
    assert mining["knn_metric"] == "???"


def test_embedding_specs_name_the_same_siglip2_model():
    models = {
        yaml.safe_load((IAA_ROOT / "specs" / name).read_text())["model"]
        for name in ("image_embed_spec.yaml", "text_embed_spec.yaml")
    }
    assert models == {"SigLIP2"}
