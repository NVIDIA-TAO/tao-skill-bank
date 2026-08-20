# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Cosmos backend and default-image compatibility."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

resolve_tao_image = importlib.import_module("resolve_tao_image")
resolve_tao_model = importlib.import_module("resolve_tao_model")


COSMOS_RL_IMAGE = "nvcr.io/nvstaging/tao/cosmos_rl:7.2.0-rc-241-multiarch"  # versions-key: images.tao_toolkit.cosmos_rl


def test_cosmos_nano_default_train_preserves_rl_image_contract():
    resolved = resolve_tao_image.resolve_image(
        ROOT, "nvidia/Cosmos3-Nano", "train"
    )

    assert resolved["backend"] == "cosmos-rl"
    assert resolved["image"] == COSMOS_RL_IMAGE
    assert resolved["source"] == "backend.container_image"
    assert resolved["resolved_from"] == "absolute"


def test_cosmos_nano_evaluate_uses_same_rl_image_contract():
    resolved = resolve_tao_image.resolve_image(
        ROOT, "nvidia/Cosmos3-Nano", "evaluate"
    )

    assert resolved["backend"] == "cosmos-rl"
    assert resolved["image"] == COSMOS_RL_IMAGE


def test_explicit_framework_stays_repository_derived():
    resolved = resolve_tao_image.resolve_image(
        ROOT,
        "nvidia/Cosmos3-Nano",
        "train",
        backend="cosmos-framework",
    )

    assert resolved["backend"] == "cosmos-framework"
    assert resolved["image"] is None
    assert resolved["build_required"] is True
    assert resolved["runtime_input"] == "image_tag"


def test_cosmos_edge_auto_routes_to_framework_for_supported_actions():
    for action in ("train", "evaluate", "inference", "inference_microservice"):
        resolved = resolve_tao_image.resolve_image(
            ROOT, "nvidia/Cosmos3-Edge", action
        )
        assert resolved["backend"] == "cosmos-framework"
        assert resolved["build_required"] is True


def test_skill_and_versions_yaml_share_one_cosmos_rl_pin():
    skill_info = resolve_tao_model.load_yaml(
        ROOT
        / "skills"
        / "models"
        / "tao-finetune-cosmos-reason"
        / "references"
        / "skill_info.yaml"
    )
    versions = yaml.safe_load((ROOT / "versions.yaml").read_text(encoding="utf-8"))

    assert skill_info["container_image"] == COSMOS_RL_IMAGE
    assert versions["images"]["tao_toolkit"]["cosmos_rl"] == COSMOS_RL_IMAGE
