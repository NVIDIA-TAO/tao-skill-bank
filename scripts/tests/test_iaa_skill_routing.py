# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for mutually exclusive IAA routing metadata."""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _description(relative: str) -> str:
    text = (REPO_ROOT / relative / "SKILL.md").read_text(encoding="utf-8")
    return yaml.safe_load(text.split("---", 2)[1])["description"]


def test_plain_language_iaa_signals_are_explicit_and_competitors_are_bounded():
    iaa = _description("skills/applications/tao-run-deft-iaa")
    aoi = _description("skills/applications/tao-run-deft-aoi")
    automl = _description("skills/applications/tao-run-automl")
    clip = _description("skills/models/tao-finetune-clip")

    for signal in ("SigLIP2", "image retrieval", "attribute-labelled", "stops getting better"):
        assert signal in iaa
    assert "Do not use for CLIP / SigLIP" in aoi
    assert "belong to tao-run-deft-iaa" in automl
    assert "belongs to tao-run-deft-iaa" in clip

    orchestration = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    for signal in ("SigLIP2 image-retrieval", "attribute-labelled", "tao-run-deft-iaa"):
        assert signal in orchestration
