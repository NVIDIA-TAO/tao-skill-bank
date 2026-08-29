# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the one-shot Codex installer."""

import os
import re
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALLER = REPO_ROOT / "scripts" / "install-codex-agents.sh"
SKILL_INSTALLER = (
    REPO_ROOT / "skills" / "core" / "tao-setup" / "scripts" / "install-codex-agents.sh"
)
README = REPO_ROOT / "README.md"
PUBLIC_MARKETPLACE = "https://github.com/NVIDIA-TAO/tao-skill-bank.git"


def _pinned_ref() -> str:
    """The release tag the installer registers the marketplace at by default."""
    match = re.search(
        r'^DEFAULT_MARKETPLACE_REF="([^"]+)"', INSTALLER.read_text(), re.MULTILINE
    )
    assert match, "the installer must pin a default marketplace ref"
    return match.group(1)


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
    assert (
        f"plugin marketplace add {PUBLIC_MARKETPLACE} --ref {_pinned_ref()}" in calls
    )
    assert all("git@github.com" not in call for call in calls)
    assert PUBLIC_MARKETPLACE in result.stdout


def test_installer_ref_can_be_cleared_to_track_the_default_branch(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_codex(fake_bin / "codex")
    call_log = tmp_path / "codex-calls.log"

    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CODEX_CALL_LOG": str(call_log),
        "TAO_SKILL_BANK_REF": "",
    }
    subprocess.run(
        ["bash", str(INSTALLER)],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    calls = call_log.read_text().splitlines()
    assert f"plugin marketplace add {PUBLIC_MARKETPLACE}" in calls
    assert all("--ref" not in call for call in calls)


def test_installer_pin_matches_the_readme_install_instructions():
    """The install docs and the installer must name the same release tag."""
    assert f"NVIDIA-TAO/tao-skill-bank@{_pinned_ref()}" in README.read_text()


def test_packaged_installer_matches_top_level_installer():
    assert SKILL_INSTALLER.read_bytes() == INSTALLER.read_bytes()
