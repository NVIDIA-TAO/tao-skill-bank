# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for mutually exclusive PAS routing metadata."""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _description(relative: str) -> str:
    text = (REPO_ROOT / relative / "SKILL.md").read_text(encoding="utf-8")
    return yaml.safe_load(text.split("---", 2)[1])["description"]


def test_pas_routing_contract_is_structural_and_competitors_are_bounded():
    pas = _description("skills/applications/tao-run-deft-pas")
    aoi = _description("skills/applications/tao-run-deft-aoi")
    automl = _description("skills/applications/tao-run-automl")
    clip = _description("skills/models/tao-finetune-clip")

    exact_bug_probe = (
        "Improve my SigLIP2 image retrieval model on my attribute-labelled "
        "dataset until it stops getting better."
    )
    assert exact_bug_probe not in pas
    for signal in (
        "image-text retrieval",
        "attribute-labelled",
        "weak-attribute or caption-pair mining",
        "repeated retraining",
        "validation plateau",
    ):
        assert signal in pas
    assert "Do not use for CLIP / SigLIP" in aoi
    assert "belong to tao-run-deft-pas" in automl
    assert "belongs to tao-run-deft-pas" in clip

    orchestration = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    for signal in (
        "image-text retrieval",
        "attribute-labelled",
        "evaluate, mine, retrain",
        "tao-run-deft-pas",
    ):
        assert signal in orchestration


def test_clip_and_pas_require_one_explicit_finetuning_method_choice():
    clip = (REPO_ROOT / "skills/models/tao-finetune-clip/SKILL.md").read_text()
    pas = (REPO_ROOT / "skills/applications/tao-run-deft-pas/SKILL.md").read_text()
    preflight = (
        REPO_ROOT
        / "skills/applications/tao-run-deft-pas/references/preflight.md"
    ).read_text()

    question = "Fine-tuning method: LoRA (default) or full-parameter SFT?"
    assert question in clip
    assert question in pas
    assert "`--finetuning-method` to `lora`" in pas
    assert "`--finetuning-method sft`" in pas
    assert "method=<full-parameter SFT | LoRA>" in preflight
    assert "| fine-tuning method | `LoRA`;" in preflight
    assert '--finetuning-method "$FINETUNING_METHOD"' in preflight


def test_pas_preflight_keeps_prerequisites_and_side_effect_gates_executable():
    root = REPO_ROOT / "skills/applications/tao-run-deft-pas"
    skill = (root / "SKILL.md").read_text(encoding="utf-8")
    preflight = (root / "references/preflight.md").read_text(encoding="utf-8")

    assert "Inspect a changed image's identity before approval" in skill
    assert "container-starting schema/capability probe only after approval" in skill
    for prerequisite in ("python3-venv", "python3-pip", "jsonschema"):
        assert prerequisite in preflight
    assert '--min-free-disk-gb workspace=256' in preflight
    assert "--defer-container-probes" in preflight
    assert "discover_pas_inputs.py" in preflight
    assert "required_user_prompt" in preflight
    assert "check_pas_credentials.py" in preflight
    assert "CREATION_TARGET_ARGS+=(--allow-missing-path workspace)" in preflight
    assert '[[ "$PLATFORM" == docker && -z "${DOCKER_HOST:-}" ]]' in preflight
    assert "run_pas_runtime_probe.py" in preflight
    assert "/probe/check_pas_cuda_runtime.py:ro" in preflight
    assert "/attestation:rw" in preflight
    assert 'echo "$NGC_KEY"' not in preflight
    assert "env | grep" not in preflight
    assert "printenv" in preflight and "Do not run" in preflight
