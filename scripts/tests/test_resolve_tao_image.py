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
list_tao_models = importlib.import_module("list_tao_models")

COSMOS_SKILL = (
    ROOT / "skills" / "models" / "tao-finetune-cosmos-reason"
)
COSMOS_SKILL_INFO_PATH = COSMOS_SKILL / "references" / "skill_info.yaml"
COSMOS_SKILL_INFO = resolve_tao_model.load_yaml(COSMOS_SKILL_INFO_PATH)
COSMOS_BACKENDS = COSMOS_SKILL_INFO["backend_contracts"]
COSMOS_RL_IMAGE = COSMOS_BACKENDS["cosmos-rl"]["container_image"]
COSMOS_FRAMEWORK_IMAGE = COSMOS_BACKENDS["cosmos-framework"]["container_image"]


def test_cosmos_nano_default_train_preserves_rl_image_contract():
    resolved = resolve_tao_image.resolve_image(
        ROOT, "nvidia/Cosmos3-Nano", "train"
    )

    assert resolved["backend"] == "cosmos-rl"
    assert resolved["image"] == COSMOS_RL_IMAGE
    assert resolved["source"] == (
        "skill_info.backend_contracts.cosmos-rl.container_image"
    )
    assert resolved["resolved_from"] == "absolute"


def test_cosmos_nano_evaluate_uses_same_rl_image_contract():
    resolved = resolve_tao_image.resolve_image(
        ROOT, "nvidia/Cosmos3-Nano", "evaluate"
    )

    assert resolved["backend"] == "cosmos-rl"
    assert resolved["image"] == COSMOS_RL_IMAGE


def test_explicit_framework_uses_skill_owned_backend_image():
    resolved = resolve_tao_image.resolve_image(
        ROOT,
        "nvidia/Cosmos3-Nano",
        "train",
        backend="cosmos-framework",
    )

    assert resolved["backend"] == "cosmos-framework"
    assert resolved["image"] == COSMOS_FRAMEWORK_IMAGE
    assert resolved["source"] == (
        "skill_info.backend_contracts.cosmos-framework.container_image"
    )


def test_cosmos_edge_auto_routes_to_framework_for_supported_actions():
    for action in ("train", "evaluate", "inference", "inference_microservice"):
        resolved = resolve_tao_image.resolve_image(
            ROOT, "nvidia/Cosmos3-Edge", action
        )
        assert resolved["backend"] == "cosmos-framework"
        assert resolved["image"] == COSMOS_FRAMEWORK_IMAGE


def test_skill_info_images_are_stamped_from_versions_yaml():
    versions = yaml.safe_load((ROOT / "versions.yaml").read_text(encoding="utf-8"))

    assert "container_image" not in COSMOS_SKILL_INFO
    assert set(COSMOS_BACKENDS) == {"cosmos-framework", "cosmos-rl"}
    assert all(
        isinstance(declaration.get("container_image"), str)
        and declaration["container_image"].startswith("nvcr.io/")
        for declaration in COSMOS_BACKENDS.values()
    )
    assert (
        versions["images"]["tao_toolkit"]["cosmos_rl"] == COSMOS_RL_IMAGE
    )
    assert (
        versions["images"]["tao_toolkit"]["cosmos_framework"]
        == COSMOS_FRAMEWORK_IMAGE
    )
    for declaration in COSMOS_BACKENDS.values():
        contract = resolve_tao_model.load_yaml(COSMOS_SKILL / declaration["path"])
        assert "container_image" not in contract

    allowed_image_files = {
        COSMOS_SKILL_INFO_PATH,
        ROOT / "versions.yaml",
    }
    image_offenders: dict[str, list[str]] = {}
    for image in (COSMOS_RL_IMAGE, COSMOS_FRAMEWORK_IMAGE):
        offenders = []
        for path in ROOT.rglob("*"):
            if path in allowed_image_files or not path.is_file():
                continue
            if path.suffix not in {".json", ".md", ".py", ".toml", ".yaml", ".yml"}:
                continue
            if image in path.read_text(encoding="utf-8", errors="ignore"):
                offenders.append(str(path.relative_to(ROOT)))
        image_offenders[image] = offenders
    assert not any(image_offenders.values()), image_offenders


def test_cosmos_consumers_do_not_use_a_versions_image_key():
    legacy_key = "images.tao_toolkit." + "cosmos_rl"
    offenders = []
    for root in (
        ROOT / "skills" / "applications" / "tao-run-deft-aoi-cosmos3",
        ROOT / "skills" / "applications" / "tao-run-inference-service",
    ):
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".md", ".yaml", ".yml"}:
                if legacy_key in path.read_text(encoding="utf-8"):
                    offenders.append(str(path.relative_to(ROOT)))
    assert not offenders


def test_model_listing_reads_cosmos_backend_images_from_skill_info():
    capabilities = list_tao_models.load_backend_capabilities(
        ROOT, "tao-finetune-cosmos-reason"
    )

    implementations = capabilities["implementations"]
    assert implementations["cosmos-rl"]["container_image"] == COSMOS_RL_IMAGE
    assert (
        implementations["cosmos-framework"]["container_image"]
        == COSMOS_FRAMEWORK_IMAGE
    )
