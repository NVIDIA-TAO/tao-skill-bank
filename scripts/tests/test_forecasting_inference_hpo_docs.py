# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVBUG 6662977: forecasting inference HPO documented an unusable basic mode.

The "Basic mode" search-space table listed four parameters: two
(``hpo.seq_len``, ``hpo.use_cross_channel``) are in the schema's
``automl_disabled_parameters``, and two (``hpo.cross_channel_heads``,
``hpo.cross_channel_dropout``) do not exist in the schema at all. An entire
documented mode of a documented feature could not run as written.

The general invariant -- every documented search parameter is searchable per
the packaged schema -- is what these tests enforce, so the next table that
drifts from the schema fails here rather than at a user's first HPO run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SKILL = REPO / "skills" / "models" / "tao-finetune-nv-tesseract-forecasting"
DOC = SKILL / "references" / "automl.md"
SCHEMA = json.loads((SKILL / "schemas" / "inference_hpo.schema.json").read_text())

SEARCHABLE = set(SCHEMA["automl_default_parameters"])
DISABLED = set(SCHEMA["automl_disabled_parameters"])


def _inference_hpo_section(text: str) -> str:
    start = text.index("### Inference HPO")
    rest = text[start + len("### Inference HPO"):]
    nxt = re.search(r"\n### ", rest)
    return rest[: nxt.start()] if nxt else rest


def _documented_spec_keys(section: str) -> set[str]:
    """Spec keys from the markdown search-space tables (2nd column, backticked)."""
    keys = set()
    for line in section.splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 2 and (m := re.fullmatch(r"`([a-z_]+\.[a-z_]+)`", cells[1])):
            keys.add(m.group(1))
    return keys


def test_documented_search_space_is_searchable_per_schema() -> None:
    documented = _documented_spec_keys(_inference_hpo_section(DOC.read_text()))
    assert documented, "no search-space table found -- did the section move?"
    assert documented <= SEARCHABLE, (
        f"documented but not searchable: {sorted(documented - SEARCHABLE)}"
    )
    assert not (documented & DISABLED), (
        f"documented but schema-disabled: {sorted(documented & DISABLED)}"
    )


def test_unusable_basic_mode_is_not_advertised() -> None:
    text = DOC.read_text()
    assert "#### Basic mode" not in text
    assert "[Basic mode](#basic-mode-tune-model-parameters)" not in text
    assert "Two modes are supported" not in text


def test_darr_is_documented_as_the_only_mode() -> None:
    section = _inference_hpo_section(DOC.read_text())
    assert "DARR mode only" in section
    # The DARR table must still cover the full searchable set.
    assert _documented_spec_keys(section) == SEARCHABLE
