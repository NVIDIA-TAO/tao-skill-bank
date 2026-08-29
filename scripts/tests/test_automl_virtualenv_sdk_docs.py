# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVBUG 6662975: no skill documented VirtualEnvSDK construction.

Three skills forward-referenced each other and the loop never closed, so
users constructed the SDK without ``work_dir=`` and every trial's checkpoints
landed in ``$HOME`` (~1.4 GB per NV-Tesseract Forecasting trial).

The full construction can only live under ``skills/applications/tao-run-automl/``
because validate-skills.sh check 4 bans the literal ``tao_sdk`` everywhere else;
the model skills must therefore *link* rather than inline it. These tests pin
both halves, including that trap.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
AUTOML = REPO / "skills" / "applications" / "tao-run-automl"
SKILL_MD = AUTOML / "SKILL.md"
RUNNER_CFG = AUTOML / "references" / "automl-runner-configuration.md"
PREFLIGHT = AUTOML / "references" / "automl-preflight-concepts.md"

MODEL_REFS = [
    REPO / "skills" / "models" / s / "references" / "automl.md"
    for s in (
        "tao-finetune-nv-tesseract-forecasting",
        "tao-finetune-nv-tesseract-ad-diffusion",
    )
]

SIZE_CEILING = 20000  # must track validate-skills.sh check 3b


def test_skill_md_stays_under_the_signer_cap() -> None:
    assert len(SKILL_MD.read_text()) <= SIZE_CEILING


def test_runner_construction_defines_the_venv_sdk() -> None:
    text = SKILL_MD.read_text()
    section = text.split("## Runner Construction", 1)[1].split("\n## ", 1)[0]
    assert "VirtualEnvSDK(venv_path=..., work_dir=...)" in section
    # The caution must name the default, or the reader has no reason to act.
    assert "~/.tao_sdk/virtualenv" in section


def test_runner_configuration_documents_the_real_signature() -> None:
    text = RUNNER_CFG.read_text()
    assert "### VirtualEnvSDK (containerless venv runs)" in text
    assert "VirtualEnvSDK(venv_path, work_dir=None, state_file=None)" in text
    assert "- VirtualEnvSDK (containerless venv runs)" in text, "missing Contents entry"


def test_preflight_no_longer_says_all_sdks_take_no_arguments() -> None:
    text = PREFLIGHT.read_text()
    assert "Construct the SDK with no arguments" not in text
    assert "`VirtualEnvSDK`" in text


@pytest.mark.parametrize("ref", MODEL_REFS, ids=lambda p: p.parents[1].name)
def test_model_refs_link_instead_of_forward_referencing(ref: Path) -> None:
    text = ref.read_text()
    assert "Use tao-run-automl to create the VirtualEnvSDK-backed AutoMLRunner" not in text
    assert "work_dir" in text
    assert "automl-runner-configuration.md" in text


@pytest.mark.parametrize("ref", MODEL_REFS, ids=lambda p: p.parents[1].name)
def test_model_refs_do_not_leak_sdk_symbols(ref: Path) -> None:
    """check 4 bans the literal ``tao_sdk`` outside tao-run-automl.

    This is the trap: the obvious way to document the default is to paste
    ``~/.tao_sdk/virtualenv``, which fails CI in a model skill.
    """
    assert re.search(r"tao_sdk", ref.read_text()) is None
