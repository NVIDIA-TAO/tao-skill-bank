# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release-contract tests for the fixed CLIP PAS PA profile."""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).parents[2]
CLIP_SKILL = REPO_ROOT / "skills/models/tao-finetune-clip"


def _load_yaml(path: Path):
    """Load one checked-in YAML mapping."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_pas_pa_profile_is_fixed_plain_training():
    """Guard the model, method, objective, and no-AutoML boundaries."""
    profile = _load_yaml(
        CLIP_SKILL / "references/pas_pa_reproduction.yaml"
    )
    spec = profile["spec_overrides"]

    assert profile["automl_policy"] == "off"
    assert spec["model"]["type"] == "siglip2-so400m-patch16-256"
    assert spec["peft"]["enabled"] is True
    assert spec["peft"]["method"] == "lora"
    assert spec["train"]["num_epochs"] == 20
    assert spec["train"]["num_gpus"] == 8
    assert spec["train"]["triplet_loss_weight"] == 0.0
    assert spec["train"]["pa_loss_weight"] == 0.01232533396889774
    assert spec["train"]["pa_margin"] == 0.1976525764912367
    assert spec["train"]["pa_inverse_temperature"] == 1.9850472486080717
    assert spec["train"]["pa_top_ratio"] == 0.07580481977201999
    assert spec["train"]["checkpointer"]["monitor"] == (
        "val/pas/overall_mAP"
    )
    assert spec["dataset"]["train"]["include_attribute_metadata"] is True
    assert spec["dataset"]["val"]["metadata_match_mode"] == (
        "scalar_plus_accessories"
    )
    assert spec["evaluate"]["pas_ground_truth_mode"] == (
        "scalar_plus_accessories"
    )


def test_pas_pa_reference_pins_implementation_and_runtime_gate():
    """The human procedure must identify code, image, and source override."""
    reference = (
        CLIP_SKILL / "references/pas-pa-reproduction.md"
    ).read_text(encoding="utf-8")
    skill = (CLIP_SKILL / "SKILL.md").read_text(encoding="utf-8")

    assert "https://github.com/NVIDIA-TAO/tao-pytorch/pull/118" in reference
    assert "aed14d5c09ecfa2ad5a79d95930784f4a06658c0" in reference
    assert "sha256:0db93f4e531c12d01d833eb654255fdfff36ee0f46bee1c6be45be502e36c8e0" in reference
    assert "PYTHONPATH=/workspace/tao-pytorch" in reference
    assert "references/pas-pa-reproduction.md" in skill


def test_clip_contract_stages_optional_pas_pair_files():
    """PAS metadata files must be expressible by the model data contract."""
    info = _load_yaml(CLIP_SKILL / "references/skill_info.yaml")
    train_inputs = info["actions"]["train"]["inputs"]
    evaluate_inputs = info["actions"]["evaluate"]["inputs"]

    assert train_inputs[
        "dataset.train.datasets[0].train_pairs_file"
    ]["optional"] is True
    assert train_inputs[
        "dataset.val.datasets[0].attribute_pairs_file"
    ]["optional"] is True
    assert evaluate_inputs[
        "dataset.val.datasets[0].attribute_pairs_file"
    ]["optional"] is True
