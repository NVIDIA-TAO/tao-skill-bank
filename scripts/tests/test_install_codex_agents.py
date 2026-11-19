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


def _write_scripted_codex(
    path: Path, *, marketplaces: str = "", failing_adds=(), version: str = "0.154.0"
) -> None:
    """A fake codex whose version, `plugin marketplace list` and `plugin add` are scripted."""
    fail_cases = "".join(
        f'  "plugin add {plugin}") echo "plugin not found" >&2; exit 1 ;;\n'
        for plugin in failing_adds
    )
    path.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$CODEX_CALL_LOG"
case "$*" in
  "--version") echo "codex-cli {version}" ;;
  "plugin marketplace list") printf '%s' '{marketplaces}' ;;
{fail_cases}  "plugin marketplace add "*|"plugin marketplace remove "*|"plugin add "*|"plugin remove "*) exit 0 ;;
  *) exit 2 ;;
esac
"""
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run_installer(tmp_path, **codex_kwargs):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_scripted_codex(fake_bin / "codex", **codex_kwargs)
    call_log = tmp_path / "codex-calls.log"
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CODEX_CALL_LOG": str(call_log),
    }
    result = subprocess.run(
        ["bash", str(INSTALLER)], check=False, capture_output=True, env=env, text=True
    )
    return result, call_log.read_text().splitlines()


def test_installer_installs_the_current_plugin_name_first(tmp_path):
    result, calls = _run_installer(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "plugin add nvidia-tao-toolkit@tao-skill-bank" in calls
    assert "plugin add tao-skill-bank@tao-local-plugins" not in calls


def test_installer_falls_back_to_the_name_older_releases_ship(tmp_path):
    """The script is served from main but installs a release tag.

    Tags through 7.3.0 ship the plugin as ``tao-skill-bank`` in a marketplace
    named ``tao-local-plugins``, so the current name is not found there.
    """
    result, calls = _run_installer(
        tmp_path, failing_adds=("nvidia-tao-toolkit@tao-skill-bank",)
    )

    assert result.returncode == 0, result.stderr
    assert calls.index("plugin add nvidia-tao-toolkit@tao-skill-bank") < calls.index(
        "plugin add tao-skill-bank@tao-local-plugins"
    )
    assert "Installed plugin tao-skill-bank@tao-local-plugins" in result.stdout


def test_installer_reports_both_failures_when_no_name_installs(tmp_path):
    result, _ = _run_installer(
        tmp_path,
        failing_adds=("nvidia-tao-toolkit@tao-skill-bank", "tao-skill-bank@tao-local-plugins"),
    )

    assert result.returncode == 1
    assert "nvidia-tao-toolkit@tao-skill-bank: plugin not found" in result.stderr
    assert "tao-skill-bank@tao-local-plugins: plugin not found" in result.stderr
    assert "Upgrade Codex" not in result.stderr


def test_installer_points_an_old_codex_cli_at_the_upgrade(tmp_path):
    """Releases after 7.3.0 use the repository root as the plugin source.

    Codex CLIs before 0.142.0 register such a marketplace and then find no
    plugin in it, which reads like a broken release unless the script says why.
    """
    result, _ = _run_installer(
        tmp_path,
        failing_adds=("nvidia-tao-toolkit@tao-skill-bank", "tao-skill-bank@tao-local-plugins"),
        version="0.141.0",
    )

    assert result.returncode == 1
    assert "older than 0.142.0" in result.stderr
    assert "Upgrade Codex" in result.stderr


def test_installer_clears_both_generations_before_installing(tmp_path):
    """An older plugin left enabled beside the new one lists every skill twice."""
    listing = (
        "MARKETPLACE        ROOT\n"
        "tao-local-plugins  /home/user/.codex/.tmp/marketplaces/tao-local-plugins\n"
    )
    result, calls = _run_installer(tmp_path, marketplaces=listing)

    assert result.returncode == 0, result.stderr
    add_marketplace = next(i for i, c in enumerate(calls) if c.startswith("plugin marketplace add "))
    for removal in (
        "plugin remove tao-skill-bank@tao-local-plugins",
        "plugin remove nvidia-tao-toolkit@tao-skill-bank",
        "plugin marketplace remove tao-local-plugins",
    ):
        assert removal in calls and calls.index(removal) < add_marketplace, removal
    assert "plugin marketplace remove tao-skill-bank" not in calls


def test_installer_re_points_a_marketplace_registered_under_the_current_name(tmp_path):
    """Every re-run after 7.3.0 finds ``tao-skill-bank`` already registered."""
    listing = (
        "MARKETPLACE     ROOT\n"
        "tao-skill-bank  /home/user/.codex/.tmp/marketplaces/tao-skill-bank\n"
    )
    result, calls = _run_installer(tmp_path, marketplaces=listing)

    assert result.returncode == 0, result.stderr
    add_marketplace = next(i for i, c in enumerate(calls) if c.startswith("plugin marketplace add "))
    assert calls.index("plugin marketplace remove tao-skill-bank") < add_marketplace
    assert "plugin marketplace remove tao-local-plugins" not in calls


def test_installer_reads_the_identity_from_a_marketplace_root_with_a_space(tmp_path):
    """The marketplace root is everything after the name column, spaces included."""
    root = tmp_path / "my plugins" / "tao-skill-bank"
    root.mkdir(parents=True)
    (root / "AGENTS.md").write_text("# identity from the marketplace root\n")
    listing = f"MARKETPLACE     ROOT\ntao-skill-bank  {root}\n"
    result, _ = _run_installer(tmp_path, marketplaces=listing)

    assert result.returncode == 0, result.stderr
    installed = tmp_path / "home" / ".codex" / "AGENTS.md"
    assert installed.read_text() == "# identity from the marketplace root\n"


def test_installer_matches_marketplace_names_exactly(tmp_path):
    """A clone directory named tao-skill-bank is not the tao-skill-bank marketplace."""
    listing = (
        "MARKETPLACE   ROOT\n"
        "team-plugins  /home/user/src/tao-skill-bank\n"
    )
    result, calls = _run_installer(tmp_path, marketplaces=listing)

    assert result.returncode == 0, result.stderr
    assert not [c for c in calls if c.startswith("plugin marketplace remove ")]
