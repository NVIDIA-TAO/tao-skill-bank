# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release-contract guards for Visual ChangeNet skill workflows."""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "skills" / "models" / "tao-train-visual-changenet"
DEFT_ROOT = REPO_ROOT / "skills" / "applications" / "tao-run-deft-aoi"
DINOV3_VARIANTS = {
    "vit_small_dinov3",
    "vit_small_plus_dinov3",
    "vit_base_dinov3",
    "vit_large_dinov3",
    "vit_huge_plus_dinov3",
    "vit_7b_dinov3",
}


def test_dinov3_workflows_document_every_supported_variant():
    model_reference = (MODEL_ROOT / "references" / "dinov3-backbones.md").read_text()
    deft_reference = (DEFT_ROOT / "references" / "visual-changenet.md").read_text()

    for variant in DINOV3_VARIANTS:
        assert variant in model_reference
        assert variant in deft_reference
    for required in ("freeze_backbone: true", "HF_TOKEN", "pretrained_backbone_path"):
        assert required in model_reference
        assert required in deft_reference


def test_far_checkpoint_recipe_uses_checkpointer_contract():
    tuning = (MODEL_ROOT / "references" / "tuning-parameters.md").read_text()
    for required in (
        "enable_topk: true",
        "monitor: val_far",
        "mode: min",
        "replace_periodic: false",
        "optim.monitor_name",
    ):
        assert required in tuning


def test_downstream_checkpoint_defaults_are_explicit_inputs():
    templates = (
        MODEL_ROOT / "references" / "spec_template_train.yaml",
        DEFT_ROOT / "references" / "baseline_spec.yaml",
    )
    for template in templates:
        spec = yaml.safe_load(template.read_text())
        assert all(spec[action]["checkpoint"] == "" for action in ("evaluate", "inference", "export"))
