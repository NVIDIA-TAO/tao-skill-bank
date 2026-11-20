# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the one-shot Codex installer."""

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALLER = REPO_ROOT / "scripts" / "install-codex-agents.sh"
SKILL_INSTALLER = (
    REPO_ROOT / "skills" / "core" / "tao-setup" / "scripts" / "install-codex-agents.sh"
)
README = REPO_ROOT / "README.md"
HELPER = REPO_ROOT / "scripts" / "tao_agent_identity.py"
IDENTITY = REPO_ROOT / "AGENTS.md"
PUBLIC_MARKETPLACE = "https://github.com/NVIDIA-TAO/tao-skill-bank.git"
BEGIN = "<!-- BEGIN nvidia-tao-toolkit agent identity"
END = "<!-- END nvidia-tao-toolkit agent identity -->"


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
    path: Path,
    *,
    marketplaces: str = "",
    failing_adds=(),
    failing_removals=(),
    version: str = "0.154.0",
) -> None:
    """A fake codex whose version, `plugin marketplace list` and `plugin add` are scripted."""
    fail_cases = "".join(
        f'  "plugin add {plugin}") echo "plugin not found" >&2; exit 1 ;;\n'
        for plugin in failing_adds
    ) + "".join(
        f'  "plugin marketplace remove {name}") echo "not configured or installed" >&2; exit 1 ;;\n'
        for name in failing_removals
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


def _run_installer(tmp_path, *args, installer=INSTALLER, piped=False, cwd=None, codex_home=None,
                   **codex_kwargs):
    """Run the installer against a scripted codex: (result, the codex calls it made).

    ``piped`` feeds the script to bash on stdin, as ``curl ... | bash -s -- <args>`` does.
    """
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    _write_scripted_codex(fake_bin / "codex", **codex_kwargs)
    call_log = tmp_path / "codex-calls.log"
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CODEX_CALL_LOG": str(call_log),
    }
    if codex_home:
        env["CODEX_HOME"] = str(codex_home)
    command = ["bash", "-s", "--", *args] if piped else ["bash", str(installer), *args]
    result = subprocess.run(
        command, check=False, capture_output=True, env=env, text=True, cwd=cwd,
        input=installer.read_text() if piped else None,
    )
    calls = call_log.read_text().splitlines() if call_log.exists() else []
    return result, calls


def _snapshot(folder):
    """Every file below ``folder`` with its bytes, to show that a run changed nothing."""
    return {path.relative_to(folder).as_posix(): path.read_bytes()
            for path in sorted(folder.rglob("*")) if path.is_file()}


def _notes(result):
    return [line for line in result.stdout.splitlines() if "NOTE:" in line]


def _project(tmp_path, text="# My rules\n\n- keep answers short\n"):
    project = tmp_path / "my project"
    project.mkdir()
    if text is not None:
        (project / "AGENTS.md").write_text(text)
    return project


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
    """The marketplace root is everything after the name column, spaces included.

    This root is a release that predates the helper: the helper of the clone the
    script runs from adds the identity of the installed release.
    """
    root = tmp_path / "my plugins" / "tao-skill-bank"
    root.mkdir(parents=True)
    (root / "AGENTS.md").write_text("# identity from the marketplace root\n")
    listing = f"MARKETPLACE     ROOT\ntao-skill-bank  {root}\n"
    project = _project(tmp_path)
    result, _ = _run_installer(tmp_path, "--project", str(project), marketplaces=listing)

    assert result.returncode == 0, result.stderr
    added = (project / "AGENTS.md").read_text()
    assert "# identity from the marketplace root\n" in added
    assert IDENTITY.read_text().splitlines()[0] not in added
    assert f'TAO_SKILL_BANK_PATH="{root}" python3 "{HELPER}" add --project' in result.stdout


# ── step 3: the agent identity goes into a project, never into the Codex home ──
def test_installer_writes_nothing_into_the_codex_home(tmp_path):
    """Earlier versions copied the identity over ~/.codex/AGENTS.md and backed that file up."""
    result, _ = _run_installer(tmp_path)

    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "home").exists()   # the scripted codex writes nothing either
    assert "was not written anywhere" in result.stdout
    assert f'python3 "{HELPER}" add --project /path/to/your/project' in result.stdout
    assert not _notes(result)


def test_installer_leaves_the_users_global_instructions_alone(tmp_path):
    codex_home = tmp_path / "home" / ".codex"
    codex_home.mkdir(parents=True)
    (codex_home / "AGENTS.md").write_text("# My global rules\n\n- always answer in French\n")
    before = _snapshot(tmp_path / "home")
    result, _ = _run_installer(tmp_path)

    assert result.returncode == 0, result.stderr
    assert _snapshot(tmp_path / "home") == before
    assert not _notes(result)


def test_installer_honours_codex_home_and_writes_nothing_there(tmp_path):
    elsewhere = tmp_path / "codex home"
    elsewhere.mkdir()
    (elsewhere / "AGENTS.md").write_bytes(IDENTITY.read_bytes())
    result, _ = _run_installer(tmp_path, codex_home=elsewhere)

    assert result.returncode == 0, result.stderr
    assert [path.name for path in elsewhere.iterdir()] == ["AGENTS.md"]
    assert not (tmp_path / "home").exists()
    assert len(_notes(result)) == 1 and str(elsewhere / "AGENTS.md") in _notes(result)[0]


def test_project_option_adds_the_identity_block_to_that_project(tmp_path):
    own = "# My rules\n\n- keep answers short\n"
    project = _project(tmp_path, own)
    result, calls = _run_installer(tmp_path, "--project", str(project))

    assert result.returncode == 0, result.stderr
    assert "plugin add nvidia-tao-toolkit@tao-skill-bank" in calls
    added = (project / "AGENTS.md").read_text()
    identity = IDENTITY.read_text()
    assert added.startswith(own + "\n" + BEGIN)
    assert added.endswith(identity.rstrip("\n") + "\n" + END + "\n")
    assert sorted(path.name for path in project.iterdir()) == ["AGENTS.md"]
    assert not (tmp_path / "home").exists()
    assert "remove" in result.stdout

    # A second run updates the block in place: the file does not grow.
    again, _ = _run_installer(tmp_path / "again", f"--project={project}")
    assert again.returncode == 0, again.stderr
    assert (project / "AGENTS.md").read_text() == added


def test_project_option_creates_the_file_in_a_project_without_one(tmp_path):
    project = _project(tmp_path, None)
    result, _ = _run_installer(tmp_path, "--project", str(project))

    assert result.returncode == 0, result.stderr
    created = (project / "AGENTS.md").read_text()
    assert created.startswith(BEGIN) and created.endswith(END + "\n")


def test_options_are_checked_before_anything_is_installed(tmp_path):
    for number, (args, message) in enumerate((
        (("--project", str(tmp_path / "no such folder")), "is not an existing directory"),
        (("--project",), "--project needs a directory"),
        (("--project=",), "--project needs a directory"),
        (("--global",), "unknown argument: --global"),
    )):
        result, calls = _run_installer(tmp_path / str(number), *args)
        assert result.returncode == 1 and message in result.stderr, args
        assert calls == [], args
    usage, calls = _run_installer(tmp_path / "help", "--help")
    assert usage.returncode == 0 and "--project DIR" in usage.stdout and calls == []


def test_project_option_reports_a_file_the_helper_refuses(tmp_path):
    """The plugin is installed by then; the file with a broken block stays as it is."""
    broken = "# My rules\n\n" + BEGIN + " -->\nhalf a block\n"
    project = _project(tmp_path, broken)
    result, calls = _run_installer(tmp_path, "--project", str(project))

    assert result.returncode == 1
    assert "plugin add nvidia-tao-toolkit@tao-skill-bank" in calls
    assert "no END marker" in result.stderr
    assert "the plugin is installed, but the TAO agent identity was not added" in result.stderr
    assert (project / "AGENTS.md").read_text() == broken


def _old_copy_home(tmp_path, *, copy, backups=()):
    """A Codex home as an earlier version of the installer left it."""
    codex_home = tmp_path / "home" / ".codex"
    codex_home.mkdir(parents=True)
    if copy is not None:
        (codex_home / "AGENTS.md").write_bytes(copy)
    for stamp, text in backups:
        (codex_home / f"AGENTS.md.bak.{stamp}").write_text(text)
    return codex_home


def _old_identity(words="the identity of an older release"):
    """What an earlier release shipped as AGENTS.md: the same heading, another text."""
    return IDENTITY.read_text().splitlines()[0] + f"\n\n{words}\n"


def _noted(tmp_path):
    """Run the installer on a prepared home: the one NOTE line. Nothing in the home changed."""
    before = _snapshot(tmp_path / "home")
    result, _ = _run_installer(tmp_path)
    assert result.returncode == 0, result.stderr
    assert _snapshot(tmp_path / "home") == before
    notes = _notes(result)
    assert len(notes) == 1, result.stdout
    return notes[0]


def test_installer_says_how_to_restore_the_file_an_earlier_version_replaced(tmp_path):
    own = "# My global rules\n"
    codex_home = _old_copy_home(
        tmp_path, copy=IDENTITY.read_bytes(),
        backups=(("20260301101500", own), ("20260518090000", _old_identity())),
    )
    note = _noted(tmp_path)

    mine, target = codex_home / "AGENTS.md.bak.20260301101500", codex_home / "AGENTS.md"
    assert "is a copy of the TAO agent identity" in note
    assert "do not start like the identity: 1 of 2." in note
    assert f'mv "{mine}" "{target}"' in note and "AGENTS.md.bak.20260518090000" not in note
    assert " rm " not in note


def test_installer_does_not_offer_an_older_identity_copy_as_the_users_file(tmp_path):
    """Earlier versions backed up every file that differed, the identity of the release before too.

    Somebody with no global file of their own who installed at two releases has
    only such copies as backups. Moving one back would leave an older identity
    in every Codex session under the name of "your own file".
    """
    codex_home = _old_copy_home(
        tmp_path, copy=IDENTITY.read_bytes(),
        backups=(("20260301101500", _old_identity("as release one shipped it")),
                 ("20260518090000", _old_identity("as release two shipped it"))),
    )
    note = _noted(tmp_path)

    assert "Every backup beside it (2) starts like the identity too" in note
    assert "older copies of it, not a file of your own, and can be deleted" in note
    assert f'rm "{codex_home / "AGENTS.md"}"' in note
    assert "keep that part" in note
    assert "mv " not in note and "your own file from before" not in note
    assert "AGENTS.md.bak.2026" not in note


def test_installer_offers_the_newest_backup_that_is_the_users_own(tmp_path):
    """The user put a file of their own back between two runs of an earlier version."""
    codex_home = _old_copy_home(
        tmp_path, copy=_old_identity().encode(),
        backups=(("20260301101500", "# My first rules\n"),
                 ("20260402120000", _old_identity("as release one shipped it")),
                 ("20260518090000", "# My later rules\n"),
                 ("20260701080000", _old_identity("as release two shipped it"))),
    )
    note = _noted(tmp_path)

    newest_own = codex_home / "AGENTS.md.bak.20260518090000"
    assert "starts like an older copy" in note
    assert "do not start like the identity: 2 of 4." in note
    assert f'mv "{newest_own}" "{codex_home / "AGENTS.md"}"' in note
    assert "AGENTS.md.bak.20260301101500" not in note and "AGENTS.md.bak.20260701080000" not in note


def test_installer_says_how_to_remove_a_copy_that_replaced_nothing(tmp_path):
    codex_home = _old_copy_home(tmp_path, copy=IDENTITY.read_bytes())
    note = _noted(tmp_path)

    assert "No backup of an earlier file is there." in note
    assert f'rm "{codex_home / "AGENTS.md"}"' in note
    assert "(if you added text of your own to it, keep that part)" in note


def test_installer_recognises_the_copy_of_an_older_release(tmp_path):
    """The identity of another release differs from this one and starts with the same heading."""
    _old_copy_home(tmp_path, copy=_old_identity().encode())
    note = _noted(tmp_path)

    assert "starts like an older copy" in note and " rm " in note


def test_installer_names_backups_that_outlived_the_copy(tmp_path):
    """The user put a file of their own back and the backups are still there.

    The file in place is the user's, so the note has no command that would
    move a backup over it.
    """
    codex_home = _old_copy_home(
        tmp_path, copy=b"# My global rules, restored\n",
        backups=(("20260301101500", "# old\n"), ("20260518090000", _old_identity())),
    )
    note = _noted(tmp_path)

    assert note.count("2 backup(s)") == 1 and "do not start like the identity: 1 of 2." in note
    assert str(codex_home / "AGENTS.md.bak.20260301101500") in note
    assert "itself is not a copy of the identity" in note and "an earlier file of your own" in note
    assert "mv " not in note and " rm " not in note
    assert f"Deleting them ({codex_home / 'AGENTS.md'}.bak.<timestamp>) ends this note." in note


def test_installer_says_how_to_restore_a_backup_when_no_global_file_is_left(tmp_path):
    codex_home = _old_copy_home(
        tmp_path, copy=None,
        backups=(("20260301101500", "# My global rules\n"), ("20260518090000", _old_identity())),
    )
    note = _noted(tmp_path)

    mine = codex_home / "AGENTS.md.bak.20260301101500"
    assert "ends this note." in note
    assert note.endswith(f'restore it before you delete the rest: mv "{mine}" '
                         f'"{codex_home / "AGENTS.md"}"')


def test_installer_says_that_backups_which_are_all_identity_copies_can_be_deleted(tmp_path):
    for number, copy in enumerate((None, b"# My global rules, restored\n")):
        _old_copy_home(tmp_path / str(number), copy=copy,
                       backups=(("20260402120000", _old_identity("release one")),
                                ("20260701080000", _old_identity("release two"))))
        note = _noted(tmp_path / str(number))

        assert "2 backup(s)" in note
        assert "Every one of them starts like the TAO agent identity" in note
        assert "can be deleted" in note and "ends this note." in note
        assert "mv " not in note and "AGENTS.md.bak.2026" not in note


def test_installer_counts_only_backups_named_as_earlier_versions_named_them(tmp_path):
    """AGENTS.md.bak.<14 digits>. Other files with a similar name are the user's own business."""
    codex_home = tmp_path / "home" / ".codex"
    codex_home.mkdir(parents=True)
    (codex_home / "AGENTS.md").write_text("# My global rules\n")
    for name in ("AGENTS.md.bak", "AGENTS.md.bak.1", "AGENTS.md.bak.2024-notes",
                 "AGENTS.md.bak.2026030110150", "AGENTS.md.bak.202603011015000",
                 "AGENTS.md.bak.20260301101500.txt", "AGENTS.md.backup.20260301101500"):
        (codex_home / name).write_text("# mine\n")
    before = _snapshot(tmp_path / "home")
    result, _ = _run_installer(tmp_path)

    assert result.returncode == 0, result.stderr
    assert not _notes(result) and _snapshot(tmp_path / "home") == before

    (codex_home / "AGENTS.md.bak.20260301101500").write_text("# mine\n")
    result, _ = _run_installer(tmp_path / "again", codex_home=codex_home)
    assert result.returncode == 0, result.stderr
    notes = _notes(result)
    assert len(notes) == 1 and notes[0].count("1 backup(s)") == 1
    assert "do not start like the identity: 1 of 1." in notes[0]


def test_piped_installer_cannot_tell_backups_apart_without_an_identity(tmp_path):
    """Piped, with a release whose identity is not found, there is no heading to compare with."""
    codex_home = _old_copy_home(
        tmp_path, copy=None,
        backups=(("20260301101500", "# My global rules\n"), ("20260518090000", _old_identity())),
    )
    before = _snapshot(tmp_path / "home")
    result, _ = _run_installer(tmp_path, piped=True, cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert _snapshot(tmp_path / "home") == before
    notes = _notes(result)
    assert len(notes) == 1 and "2 backup(s)" in notes[0]
    assert "cannot tell a file of your own from an older copy of the identity" in notes[0]
    assert "look at a backup before you restore or delete it" in notes[0]
    assert "mv " not in notes[0] and str(codex_home / "AGENTS.md.bak.2026") not in notes[0]


def test_a_block_added_to_the_global_file_on_purpose_is_not_reported_as_an_old_copy(tmp_path):
    codex_home = tmp_path / "home" / ".codex"
    codex_home.mkdir(parents=True)
    added = subprocess.run(
        ["python3", str(HELPER), "add", "--file", str(codex_home / "AGENTS.md")],
        capture_output=True, text=True,
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path / "home")},
    )
    assert added.returncode == 0, added.stderr
    before = _snapshot(tmp_path / "home")
    result, _ = _run_installer(tmp_path)

    assert result.returncode == 0, result.stderr
    assert not _notes(result) and _snapshot(tmp_path / "home") == before


@pytest.mark.parametrize("installer", [INSTALLER, SKILL_INSTALLER], ids=["clone", "skill"])
def test_installer_uses_the_helper_of_a_release_that_ships_it(tmp_path, installer):
    """The helper of the installed release goes with its identity.

    The copy in the tao-setup skill has no helper beside it; the copy in a clone
    has one, which serves only a release that ships none.
    """
    root = tmp_path / "marketplaces" / "tao-skill-bank"
    (root / "scripts").mkdir(parents=True)
    (root / "AGENTS.md").write_text("# identity of the installed release\n")
    shutil.copy(HELPER, root / "scripts" / "tao_agent_identity.py")
    listing = f"MARKETPLACE     ROOT\ntao-skill-bank  {root}\n"
    project = _project(tmp_path)
    result, _ = _run_installer(
        tmp_path, "--project", str(project), installer=installer, marketplaces=listing
    )

    assert result.returncode == 0, result.stderr
    assert "# identity of the installed release\n" in (project / "AGENTS.md").read_text()
    helper = root / "scripts" / "tao_agent_identity.py"
    assert f'python3 "{helper}" add --project' in result.stdout
    assert "TAO_SKILL_BANK_PATH=" not in result.stdout


def test_installer_without_a_helper_says_so_and_touches_no_file(tmp_path):
    """A release that predates the helper, installed by the copy that has none beside it."""
    root = tmp_path / "marketplaces" / "tao-skill-bank"
    root.mkdir(parents=True)
    (root / "AGENTS.md").write_text("# identity of an older release\n")
    listing = f"MARKETPLACE     ROOT\ntao-skill-bank  {root}\n"
    result, _ = _run_installer(tmp_path, installer=SKILL_INSTALLER, marketplaces=listing)

    assert result.returncode == 0, result.stderr
    assert "does not ship scripts/tao_agent_identity.py" in result.stdout
    assert not (tmp_path / "home").exists()

    project = _project(tmp_path)
    before = _snapshot(project)
    asked, calls = _run_installer(
        tmp_path / "asked", "--project", str(project), installer=SKILL_INSTALLER,
        marketplaces=listing,
    )
    assert asked.returncode == 1
    assert "plugin add nvidia-tao-toolkit@tao-skill-bank" in calls
    assert "the plugin is installed, but the TAO agent identity was not added" in asked.stderr
    assert _snapshot(project) == before


def test_piped_installer_does_not_take_a_file_above_the_working_directory_for_the_identity(
        tmp_path):
    """Piped into bash the script has no folder; "../AGENTS.md" is then somebody's own file."""
    workspace = tmp_path / "workspace"
    (workspace / "current").mkdir(parents=True)
    (workspace / "AGENTS.md").write_text("# TAO Claw Agent\n\nnot the identity: my own notes\n")
    _old_copy_home(tmp_path, copy=b"# TAO Claw Agent\n\nmy own global notes\n")
    before = _snapshot(workspace), _snapshot(tmp_path / "home")
    result, calls = _run_installer(tmp_path, piped=True, cwd=workspace / "current")

    assert result.returncode == 0, result.stderr
    assert "plugin add nvidia-tao-toolkit@tao-skill-bank" in calls
    assert "does not ship scripts/tao_agent_identity.py" in result.stdout
    assert not _notes(result)
    assert (_snapshot(workspace), _snapshot(tmp_path / "home")) == before


def test_installer_matches_marketplace_names_exactly(tmp_path):
    """A clone directory named tao-skill-bank is not the tao-skill-bank marketplace."""
    listing = (
        "MARKETPLACE   ROOT\n"
        "team-plugins  /home/user/src/tao-skill-bank\n"
    )
    result, calls = _run_installer(tmp_path, marketplaces=listing)

    assert result.returncode == 0, result.stderr
    assert not [c for c in calls if c.startswith("plugin marketplace remove ")]


def test_installer_re_points_a_clone_that_changed_names_in_place(tmp_path):
    """Codex lists a path marketplace by its manifest name and removes it by its registered name.

    A clone registered while it shipped ``tao-local-plugins`` and then updated
    to a tree that ships ``tao-skill-bank`` is listed under the new name but
    can only be removed under the old one.
    """
    listing = "MARKETPLACE     ROOT\ntao-skill-bank  /home/user/src/tao-skill-bank\n"
    result, calls = _run_installer(
        tmp_path, marketplaces=listing, failing_removals=("tao-skill-bank",)
    )

    assert result.returncode == 0, result.stderr
    add_marketplace = next(i for i, c in enumerate(calls) if c.startswith("plugin marketplace add "))
    assert calls.index("plugin marketplace remove tao-local-plugins") < add_marketplace


def test_installer_stops_when_no_registration_can_be_removed(tmp_path):
    listing = "MARKETPLACE     ROOT\ntao-skill-bank  /home/user/src/tao-skill-bank\n"
    result, calls = _run_installer(
        tmp_path,
        marketplaces=listing,
        failing_removals=("tao-skill-bank", "tao-local-plugins"),
    )

    assert result.returncode == 1
    assert "could not remove the registered marketplace" in result.stderr
    assert not [c for c in calls if c.startswith("plugin marketplace add ")]
