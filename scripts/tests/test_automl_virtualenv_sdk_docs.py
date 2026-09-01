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

import os
import re
import subprocess
import sys
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


def _preflight_block() -> str:
    """The fenced block the docs tell an operator to run as the preflight."""
    text = RUNNER_CFG.read_text()
    after = text.split("Verify construction during preflight", 1)[1]
    return after.split("```bash", 1)[1].split("```", 1)[0]


def _preflight_python_body() -> str:
    """The python the preflight block feeds to the interpreter."""
    block = _preflight_block()
    assert "<<'PY'" in block, (
        "preflight is not a python heredoc; an import-only one-liner cannot "
        "validate venv_path"
    )
    return block.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]


def test_preflight_check_constructs_rather_than_imports() -> None:
    """An import-only preflight reports OK on a venv that cannot run a trial.

    ``VirtualEnvSDK`` validates ``venv_path`` in its constructor, so
    ``python -c "from ... import VirtualEnvSDK; print('OK')"`` exits 0 even
    when the selected venv does not exist.
    """
    block = _preflight_block()
    assert "VirtualEnvSDK(" in block, "preflight never constructs the SDK"
    assert "venv_path=" in block, "preflight does not pass the selected venv_path"
    assert "work_dir=" in block, "preflight does not pass an explicit work_dir"


def test_documented_preflight_fails_on_a_missing_venv(tmp_path: Path) -> None:
    """Execute the documented command against a stub with the real contract.

    The stub raises the ``ValueError`` the real constructor raises for a
    missing ``venv_path``. An import-only command exits 0 under this stub; a
    constructing one exits non-zero. String assertions alone cannot show that.
    """
    pkg = tmp_path / "tao_sdk" / "platforms"
    pkg.mkdir(parents=True)
    (pkg.parent / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "virtualenv.py").write_text(
        "import os\n"
        "\n"
        "\n"
        "class VirtualEnvSDK:\n"
        "    def __init__(self, venv_path, work_dir=None, state_file=None):\n"
        "        if not os.path.isdir(venv_path):\n"
        "            raise ValueError(\n"
        "                f'Virtual environment does not exist: {venv_path}'\n"
        "            )\n"
    )

    proc = subprocess.run(
        [sys.executable, "-"],
        input=_preflight_python_body(),
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(tmp_path),
            "VENV_PATH": str(tmp_path / "not-a-venv"),
            "WORK_DIR": str(tmp_path / "jobs"),
        },
    )

    assert proc.returncode != 0, "documented preflight passed on a missing venv"
    assert "Virtual environment does not exist" in proc.stderr


def test_documented_preflight_passes_on_a_valid_venv(tmp_path: Path) -> None:
    """The same command must still succeed when the venv is real."""
    pkg = tmp_path / "tao_sdk" / "platforms"
    pkg.mkdir(parents=True)
    (pkg.parent / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "virtualenv.py").write_text(
        "import os\n"
        "\n"
        "\n"
        "class VirtualEnvSDK:\n"
        "    def __init__(self, venv_path, work_dir=None, state_file=None):\n"
        "        if not os.path.isdir(venv_path):\n"
        "            raise ValueError(\n"
        "                f'Virtual environment does not exist: {venv_path}'\n"
        "            )\n"
        "        os.makedirs(os.path.join(work_dir, 'jobs'), exist_ok=True)\n"
    )
    venv_path = tmp_path / "model-venv"
    venv_path.mkdir()

    proc = subprocess.run(
        [sys.executable, "-"],
        input=_preflight_python_body(),
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(tmp_path),
            "VENV_PATH": str(venv_path),
            "WORK_DIR": str(tmp_path / "jobs"),
        },
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK"
    # Construction creates <work_dir>/jobs/ - the caution's second reason.
    assert (tmp_path / "jobs" / "jobs").is_dir()
