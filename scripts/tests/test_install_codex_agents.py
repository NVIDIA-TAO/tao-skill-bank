# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the one-shot Codex installer."""

import os
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALLER = REPO_ROOT / "scripts" / "install-codex-agents.sh"
SKILL_INSTALLER = (
    REPO_ROOT / "skills" / "core" / "tao-setup" / "scripts" / "install-codex-agents.sh"
)
PUBLIC_MARKETPLACE = "https://github.com/NVIDIA-TAO/tao-skill-bank.git"


def _write_fake_codex(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$CODEX_CALL_LOG"
case "$*" in
  "plugin marketplace list"|"plugin list") exit 0 ;;
  "plugin marketplace add "*|"plugin add "*) exit 0 ;;
  *) exit 2 ;;
esac
"""
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_installer_defaults_to_public_https_without_github_credentials(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_codex(fake_bin / "codex")
    call_log = tmp_path / "codex-calls.log"

    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CODEX_CALL_LOG": str(call_log),
    }
    result = subprocess.run(
        ["bash", str(INSTALLER)],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    calls = call_log.read_text().splitlines()
    assert f"plugin marketplace add {PUBLIC_MARKETPLACE}" in calls
    assert all("git@github.com" not in call for call in calls)
    assert PUBLIC_MARKETPLACE in result.stdout


def test_packaged_installer_matches_top_level_installer():
    assert SKILL_INSTALLER.read_bytes() == INSTALLER.read_bytes()


def test_readme_uses_canonical_public_marketplace_url():
    readme = (REPO_ROOT / "README.md").read_text()

    assert "git@github.com" not in readme
    assert "NVIDIA-TAO/tao-skills-bank" not in readme
    assert readme.count(PUBLIC_MARKETPLACE) >= 2
