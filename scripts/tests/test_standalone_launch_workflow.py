# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for shared helpers in standalone skill installs."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SETUP = ROOT / "skills/core/tao-setup/SKILL.md"
LAUNCH = ROOT / "skills/core/tao-launch-workflow/SKILL.md"
PREFLIGHT = ROOT / "skills/core/tao-launch-workflow/references/platform-preflight.md"


def _bank_check(path):
    text = path.read_text(encoding="utf-8")
    return re.search(r"```bash\n(export TAO_SKILL_BANK_PATH=.*?\ndone)", text, re.S).group(1)


def _make_bank(root, snippet):
    items = re.search(r"for item in (.*); do", snippet).group(1).split()
    for item in items:
        path = root / item
        if item in {"skills/models", "skills/platform"}:
            path.mkdir(parents=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()


@pytest.mark.parametrize("path", [SETUP, PREFLIGHT])
def test_standalone_check_accepts_complete_bank(path, tmp_path):
    """Both entry points accept a checkout containing the shared helpers."""
    snippet = _bank_check(path)
    bank = tmp_path / "tao-skill-bank"
    _make_bank(bank, snippet)
    env = {**os.environ, "HOME": str(tmp_path)}
    env.pop("TAO_SKILL_BANK_PATH", None)

    result = subprocess.run(["bash", "-c", snippet], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("path", [SETUP, PREFLIGHT])
def test_standalone_check_rejects_incomplete_bank(path, tmp_path):
    """An individual skill copy cannot masquerade as the full bank."""
    snippet = _bank_check(path)
    bank = tmp_path / "individual-skill"
    _make_bank(bank, snippet)
    (bank / "scripts/tao_job_record.py").unlink()

    result = subprocess.run(
        ["bash", "-c", snippet],
        env={**os.environ, "TAO_SKILL_BANK_PATH": str(bank)},
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "missing scripts/tao_job_record.py" in result.stderr


@pytest.mark.parametrize("path", [SETUP, PREFLIGHT])
def test_standalone_check_rejects_missing_checkout(path, tmp_path):
    """An unset bank path cannot silently proceed without a checkout."""
    env = {**os.environ, "HOME": str(tmp_path)}
    env.pop("TAO_SKILL_BANK_PATH", None)
    result = subprocess.run(
        ["bash", "-c", _bank_check(path)], env=env,
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "Incomplete TAO skill bank" in result.stderr


def test_packaged_clip_image_resolves_from_skill_info():
    """Image lookup uses the packaged model contract, not config.json."""
    launch = LAUNCH.read_text(encoding="utf-8")
    assert "references/skill_info.yaml" in launch
    assert "skills/models/<network>/config.json" not in launch
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/resolve_tao_image.py"),
         "--skill-bank", str(ROOT), "--model", "tao-finetune-clip",
         "--action", "inference", "--format", "json"],
        capture_output=True, text=True, check=True,
    )
    metadata = yaml.safe_load((ROOT / "skills/models/tao-finetune-clip/references/skill_info.yaml").read_text())
    assert json.loads(result.stdout)["image"] == metadata["container_image"]


@pytest.mark.parametrize("output,code,launches", [
    ("job-123", 0, True), ("", 0, False), ("", 2, False),
])
def test_job_record_must_precede_launch(tmp_path, output, code, launches):
    """The documented submit guard stops on failure or an empty ID."""
    launch = LAUNCH.read_text(encoding="utf-8")
    start = launch.index("  if ! JOB_ID=$(")
    end = launch.index("  # <native launch", start)
    guard = re.sub(r"<[^>]+>", "value", launch[start:end])
    bank = tmp_path / "bank"
    helper = bank / "scripts/tao_job_record.py"
    helper.parent.mkdir(parents=True)
    helper.write_text(f"#!/bin/sh\nprintf '%s' '{output}'\nexit {code}\n")
    helper.chmod(0o755)
    script = f'BANK="{bank}"\n{guard}\nprintf "LAUNCHED:%s\\n" "$JOB_ID"\n'

    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert ("LAUNCHED:job-123" in result.stdout) == launches
    assert (result.returncode == 0) == launches
