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
