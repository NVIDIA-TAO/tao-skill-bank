# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVBUG 6664736: both NV-Tesseract skills documented a checkpoint path the
runner never creates.

``train.output_dir`` is a *declared output*, and ``_auto_suffix_output_dirs``
in ``tao_automl.runner`` skips declared outputs (``if dotted in
declared_outputs: continue``) -- it is the only code that can emit a ``rec_``
path component. The SDK instead routes the value to
``<work_dir>/jobs/<job-id>/results/<dotted_key_with_underscores>``. These tests
pin the docs to that reality.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]

SKILLS = {
    "tao-finetune-nv-tesseract-forecasting": "best_model.pt",
    "tao-finetune-nv-tesseract-ad-diffusion": "best_finetuned_model.pth",
}


def _automl_md(skill: str) -> Path:
    return REPO / "skills" / "models" / skill / "references" / "automl.md"


@pytest.mark.parametrize("skill", sorted(SKILLS))
def test_no_rec_n_checkpoint_path_is_documented(skill: str) -> None:
    text = _automl_md(skill).read_text()
    assert re.search(r"output_dir/rec_|rec_<N>/", text) is None, (
        f"{skill}: documents a rec_<N> checkpoint path, which the runner never "
        "creates for a declared output"
    )


@pytest.mark.parametrize("skill", sorted(SKILLS))
def test_sdk_routed_results_path_is_documented(skill: str) -> None:
    text = _automl_md(skill).read_text()
    assert "results/train_output_dir/" in text
    assert SKILLS[skill] in text, f"{skill}: checkpoint basename not documented"


@pytest.mark.parametrize("skill", sorted(SKILLS))
def test_documented_key_matches_declared_output(skill: str) -> None:
    """The doc path is derived from skill_info.yaml, so rename it and this fails.

    The SDK sanitizes each declared output key by splitting on dots and joining
    the tokens with ``_`` (``tao_sdk`` virtualenv ``_route_outputs``), so
    ``train.output_dir`` -> ``train_output_dir``.
    """
    info = yaml.safe_load(
        (REPO / "skills" / "models" / skill / "references" / "skill_info.yaml").read_text()
    )
    declared = set(info["actions"]["train"]["outputs"])
    assert declared == {"train.output_dir"}, f"{skill}: declared outputs changed"

    sanitized = {"_".join(re.sub(r"[^A-Za-z0-9_-]", "_", t) for t in k.split(".")) for k in declared}
    text = _automl_md(skill).read_text()
    for name in sanitized:
        assert f"results/{name}/" in text, (
            f"{skill}: declared output {name!r} is not the one documented"
        )
