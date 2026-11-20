# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/tao_agent_identity.py, the identity block in a project's AGENTS.md.

The helper is driven as a subprocess, the way a user or an agent runs it: with
the interpreter of the test run and, where there is one, with the system
interpreter /usr/bin/python3, which is the oldest Python the helper supports on
the machines this suite runs on and has no third-party package installed.
Every test works in temporary folders and with a temporary home folder.
"""

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "scripts" / "tao_agent_identity.py"
IDENTITY = (REPO_ROOT / "AGENTS.md").read_bytes()
SYSTEM_PYTHON = "/usr/bin/python3"

# The marker lines are a contract with the files in users' projects.
BEGIN_PREFIX = b"<!-- BEGIN nvidia-tao-toolkit agent identity"
BEGIN_LINE = (
    b'<!-- BEGIN nvidia-tao-toolkit agent identity: generated block. "tao_agent_identity.py add" '
    b'of the plugin updates it and "remove" deletes it; keep your own text outside the markers. '
    b"-->"
)
END_PREFIX = b"<!-- END nvidia-tao-toolkit agent identity"
END_LINE = END_PREFIX + b" -->"
HEADING = IDENTITY.split(b"\n")[0]
OWN = b"# My rules\n\n- answer in short sentences\n- never push to main\n"
# The default of Codex's project_doc_max_bytes, written out so that a wrong
# constant in the helper does not pass.
CODEX_LIMIT = 32768


def _block(identity=IDENTITY, newline=b"\n"):
    """The block the helper writes for ``identity``, without the line end after its END line."""
    lines = identity.replace(b"\r\n", b"\n").strip(b"\n").split(b"\n")
    return newline.join([BEGIN_LINE, *lines, END_LINE])


def _interpreters():
    found = [pytest.param(sys.executable, id="test-interpreter")]
    if os.access(SYSTEM_PYTHON, os.X_OK):
        probe = subprocess.run(
            [SYSTEM_PYTHON, "-c", "import sys; print(sys.version_info >= (3, 9))"],
            capture_output=True, text=True, check=False,
        )
        if probe.stdout.strip() == "True":
            found.append(pytest.param(SYSTEM_PYTHON, id="system-python3"))
    return found


def _listing(folder):
    """Everything below ``folder``: the bytes of a file, the target of a link, None for a folder."""
    found = {}
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder).as_posix()
        if path.is_symlink():
            found[relative] = "-> " + os.readlink(path)
        elif path.is_dir():
            found[relative] = None
        else:
            found[relative] = path.read_bytes()
    return found


class Sandbox:
    """A temporary home folder and project, and the helper run with one interpreter."""

    def __init__(self, python, root):
        self.python = python
        self.root = root
        self.home = root / "home"
        self.project = root / "project"
        self.home.mkdir()
        self.project.mkdir()
        self.target = self.project / "AGENTS.md"

    def environment(self, bank=None, **more):
        environment = {
            key: value for key, value in os.environ.items()
            if key not in ("TAO_SKILL_BANK_PATH", "CODEX_HOME", "PYTHONPATH", "PYTHONHOME",
                           "PYTHONIOENCODING")
        }
        environment["HOME"] = str(self.home)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        if bank is not None:
            environment["TAO_SKILL_BANK_PATH"] = str(bank)
        environment.update(more)
        return environment

    def run(self, *args, cwd=None, bank=None, helper=HELPER, flags=(), **more):
        return subprocess.run(
            [self.python, *flags, str(helper), *[str(arg) for arg in args]],
            cwd=cwd or self.project, env=self.environment(bank, **more), capture_output=True,
            text=True, check=False,
        )

    def ok(self, *args, **kwargs):
        result = self.run(*args, **kwargs)
        assert result.returncode == 0, result.stderr
        return result

    def refused(self, *args, **kwargs):
        """Run a command that must be refused: exit 2, a message, and not one byte changed."""
        before = _listing(self.root)
        result = self.run(*args, **kwargs)
        assert result.returncode == 2, (result.stdout, result.stderr)
        assert result.stderr.startswith("tao_agent_identity: ") and not result.stdout
        assert _listing(self.root) == before
        return result.stderr

    def bank(self, identity, name="bank"):
        """A folder that holds ``identity`` as its AGENTS.md, for TAO_SKILL_BANK_PATH."""
        folder = self.root / name
        folder.mkdir(exist_ok=True)
        (folder / "AGENTS.md").write_bytes(identity)
        return folder


@pytest.fixture(params=_interpreters())
def box(request, tmp_path):
    # Inside a work tree the default target would be a file of that tree. Stop before
    # anything is written, and loudly: a skip would let the whole file pass unseen.
    for folder in (tmp_path, *tmp_path.parents):
        if (folder / ".git").exists():
            pytest.fail(f"the temporary folder is inside a git work tree ({folder}); run the "
                        f"tests with a --basetemp outside one")
    return Sandbox(request.param, tmp_path)


# ─────────────────────────────── what runs the helper ───────────────────────────────
def test_helper_needs_only_the_standard_library_and_python_3_9():
    source = HELPER.read_text(encoding="utf-8")
    tree = ast.parse(source, feature_version=(3, 9))   # no syntax newer than Python 3.9
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "the helper imports nothing from the folder it sits in"
            imported.add(node.module.split(".")[0])
    assert imported
    if hasattr(sys, "stdlib_module_names"):              # Python 3.10 and later
        assert imported <= set(sys.stdlib_module_names), imported
    assert source.startswith("#!/usr/bin/env python3\n")
    assert os.access(HELPER, os.X_OK)


def test_helper_runs_without_site_packages(box):
    """-I and -S take the user's and the installation's packages away; nothing is missing."""
    result = box.ok("add", flags=("-I", "-S"))
    assert "Created" in result.stdout and box.target.read_bytes() == _block() + b"\n"


# ─────────────────────────────── add ───────────────────────────────
def test_add_creates_a_missing_file_with_only_the_block(box):
    result = box.ok("add")
    assert box.target.read_bytes() == _block() + b"\n"
    assert f"Created {box.target}" in result.stdout and not result.stderr
    assert sorted(path.name for path in box.project.iterdir()) == ["AGENTS.md"]
    # the two marker lines, as they are documented
    lines = box.target.read_bytes().split(b"\n")
    assert lines[0] == BEGIN_LINE and lines[0].startswith(BEGIN_PREFIX) and lines[-2] == END_LINE
    assert lines[1] == IDENTITY.split(b"\n")[0]


def test_add_appends_the_block_after_one_blank_line_and_keeps_every_existing_byte(box):
    box.target.write_bytes(OWN)
    result = box.ok("add")
    assert box.target.read_bytes() == OWN + b"\n" + _block() + b"\n"
    assert f"The {len(OWN)} bytes that were there are unchanged" in result.stdout
    assert sorted(path.name for path in box.project.iterdir()) == ["AGENTS.md"]   # no backup


def test_add_to_a_file_without_a_final_newline_adds_that_newline_and_nothing_else(box):
    own = OWN.rstrip(b"\n")
    box.target.write_bytes(own)
    result = box.ok("add")
    assert box.target.read_bytes() == own + b"\n" + b"\n" + _block() + b"\n"
    assert "did not end with a newline, so one was added" in result.stdout


def test_add_keeps_windows_line_endings_and_writes_the_block_with_them(box):
    own = OWN.replace(b"\n", b"\r\n")
    box.target.write_bytes(own)
    box.ok("add")
    written = box.target.read_bytes()
    assert written == own + b"\r\n" + _block(newline=b"\r\n") + b"\r\n"
    assert written.count(b"\n") == written.count(b"\r\n")   # not one bare LF in the file
    assert box.run("status").returncode == 0
    box.ok("remove")
    assert box.target.read_bytes() == own


def test_add_fills_an_empty_file_with_only_the_block(box):
    box.target.write_bytes(b"")
    result = box.ok("add")
    assert box.target.read_bytes() == _block() + b"\n" and "which was empty" in result.stdout


def test_add_updates_the_block_in_place_and_leaves_every_byte_outside_the_markers(box):
    before = b"# My rules\n\n- one\n\n\n"            # two blank lines of the user's own
    after = b"\n\n## Added later\n\n- two\n  \n"     # and text after the block
    old = box.bank(b"# The identity of an older release\n\nold text\n", "old-bank")
    box.target.write_bytes(before.rstrip(b"\n") + b"\n")
    box.ok("add", bank=old)
    with_old = box.target.read_bytes()
    assert b"old text" in with_old
    # the user moves on: text before the block changes, text after it is added
    box.target.write_bytes(before + with_old[with_old.index(BEGIN_PREFIX):].rstrip(b"\n") + after)
    assert box.run("status").returncode == 1

    result = box.ok("add")
    assert box.target.read_bytes() == before + _block() + after
    assert "Updated" in result.stdout and "outside the markers is unchanged" in result.stdout
    assert box.run("status").returncode == 0


def test_add_finds_a_block_whose_begin_line_another_version_worded_differently(box):
    other = BEGIN_PREFIX + b" (written by another version of the helper) -->"
    box.target.write_bytes(OWN + b"\n" + other + b"\nold text\n" + END_LINE + b"  \nafter\n")
    box.ok("add")
    assert box.target.read_bytes() == OWN + b"\n" + _block() + b"\nafter\n"


def test_add_is_idempotent_and_does_not_rewrite_a_current_file(box):
    box.target.write_bytes(OWN)
    box.ok("add")
    first, stamp = box.target.read_bytes(), box.target.stat().st_mtime_ns
    again = box.ok("add")
    assert "already the current identity. Nothing to do." in again.stdout
    assert box.target.read_bytes() == first and box.target.stat().st_mtime_ns == stamp
    for _ in range(3):
        box.ok("add")
    assert box.target.read_bytes() == first
    assert first.count(BEGIN_PREFIX) == 1 and first.count(END_LINE) == 1


def test_add_reads_the_identity_as_text_whatever_line_ends_the_bank_has(box):
    """A checkout on Windows may hold AGENTS.md with CRLF; the block follows the target file."""
    windows_bank = box.bank(IDENTITY.replace(b"\n", b"\r\n") + b"\r\n\r\n", "windows-bank")
    box.target.write_bytes(OWN)
    box.ok("add", bank=windows_bank)
    assert box.target.read_bytes() == OWN + b"\n" + _block() + b"\n"
    assert box.run("status", bank=windows_bank).returncode == 0


def test_add_keeps_the_mode_of_the_file_and_leaves_no_other_file_behind(box):
    box.target.write_bytes(OWN)
    box.target.chmod(0o600)
    box.ok("add")
    assert box.target.stat().st_mode & 0o777 == 0o600
    box.ok("remove")
    assert box.target.stat().st_mode & 0o777 == 0o600
    assert sorted(path.name for path in box.project.iterdir()) == ["AGENTS.md"]
    # a new file gets the mode any new file gets
    box.target.unlink()
    box.ok("add")
    umask = os.umask(0)
    os.umask(umask)
    assert box.target.stat().st_mode & 0o777 == 0o666 & ~umask


def test_a_changed_file_is_a_new_file_so_a_second_name_keeps_the_old_content(box):
    """The write goes to a temporary file that takes the place of the target in one step."""
    box.target.write_bytes(OWN)
    other = box.root / "second-name.md"
    try:
        os.link(box.target, other)
    except OSError as exc:
        pytest.skip(f"no hard links here: {exc}")
    box.ok("add")
    assert box.target.read_bytes() == OWN + b"\n" + _block() + b"\n"
    assert other.read_bytes() == OWN
    assert sorted(path.name for path in box.project.iterdir()) == ["AGENTS.md"]


def test_add_writes_through_a_link_and_keeps_the_link(box):
    real = box.root / "shared" / "instructions.md"
    real.parent.mkdir()
    real.write_bytes(OWN)
    box.target.symlink_to(real)
    result = box.ok("add")
    assert box.target.is_symlink() and real.read_bytes() == OWN + b"\n" + _block() + b"\n"
    assert f"(a link to {real.resolve()})" in result.stdout
    box.ok("remove")
    assert box.target.is_symlink() and real.read_bytes() == OWN


def test_a_byte_order_mark_stays_where_it_is(box):
    """An editor on Windows may put one in front of a file that began with the BEGIN line."""
    mark = b"\xef\xbb\xbf"
    box.ok("add")
    box.target.write_bytes(mark + box.target.read_bytes().replace(b"# TAO", b"# Old TAO", 1))
    assert box.run("status").returncode == 1
    box.ok("add")
    assert box.target.read_bytes() == mark + _block() + b"\n"
    box.ok("remove")
    assert not box.target.exists()


# ─────────────────────────────── remove ───────────────────────────────
def test_remove_gives_back_the_file_byte_for_byte(box):
    box.target.write_bytes(OWN)
    box.ok("add")
    result = box.ok("remove")
    assert box.target.read_bytes() == OWN
    assert "and the blank line before it" in result.stdout
    # and again and again
    for _ in range(2):
        box.ok("add")
        box.ok("remove")
    assert box.target.read_bytes() == OWN
    assert sorted(path.name for path in box.project.iterdir()) == ["AGENTS.md"]


def test_remove_deletes_a_file_that_holds_nothing_but_the_block(box):
    box.ok("add")
    result = box.ok("remove")
    assert not box.target.exists() and "held nothing else and was deleted" in result.stdout
    assert list(box.project.iterdir()) == []


def test_remove_keeps_the_newline_that_add_put_after_text_without_one(box):
    own = OWN.rstrip(b"\n")
    box.target.write_bytes(own)
    box.ok("add")
    box.ok("remove")
    assert box.target.read_bytes() == own + b"\n"


def test_text_the_user_adds_after_the_block_survives_update_and_remove(box):
    later = b"\n## Added after the block\n\n- keep this\n"
    old = box.bank(b"# The identity of an older release\n", "old-bank")
    box.target.write_bytes(OWN)
    box.ok("add", bank=old)
    box.target.write_bytes(box.target.read_bytes() + later)

    box.ok("add")                                             # update
    assert box.target.read_bytes() == OWN + b"\n" + _block() + b"\n" + later
    box.ok("remove")
    assert box.target.read_bytes() == OWN + later


def test_text_the_user_adds_after_a_block_that_began_the_file_survives(box):
    later = b"\n# My rules\n\n- written after the block was added\n"
    box.ok("add")
    box.target.write_bytes(box.target.read_bytes() + later)
    box.ok("add")
    assert box.target.read_bytes() == _block() + b"\n" + later
    box.ok("remove")
    assert box.target.read_bytes() == later     # the blank line is the user's and stays


def test_remove_takes_only_the_blank_line_directly_before_the_block(box):
    own = b"# My rules\n\n- one\n\n\n\n"              # ends with three blank lines
    box.target.write_bytes(own)
    box.ok("add")
    assert box.target.read_bytes() == own + b"\n" + _block() + b"\n"
    box.ok("remove")
    assert box.target.read_bytes() == own
    # a block that the user moved directly below a line of text has no blank line to lose
    box.target.write_bytes(b"- one\n" + _block() + b"\n- two\n")
    result = box.ok("remove")
    assert box.target.read_bytes() == b"- one\n- two\n"
    assert "blank line" not in result.stdout


def test_remove_has_nothing_to_do_without_a_block_or_a_file_and_needs_no_bank(box):
    empty = box.root / "no-bank"
    empty.mkdir()
    missing = box.ok("remove", bank=empty)
    assert "does not exist. Nothing to do." in missing.stdout and not box.target.exists()
    box.target.write_bytes(OWN)
    absent = box.ok("remove", bank=empty)
    assert "has no nvidia-tao-toolkit agent identity block. Nothing to do." in absent.stdout
    assert box.target.read_bytes() == OWN
    box.ok("add")
    box.ok("remove", bank=empty)                # the bank is gone; the block can still go
    assert box.target.read_bytes() == OWN


def test_remove_through_a_link_leaves_an_empty_file_and_not_a_broken_link(box):
    real = box.root / "shared.md"
    box.ok("add", "--file", real)
    box.target.symlink_to(real)
    result = box.ok("remove")
    assert box.target.is_symlink() and real.read_bytes() == b""
    assert "kept because a link points at it" in result.stdout


# ─────────────────────────────── status ───────────────────────────────
def test_status_says_whether_the_block_is_there_and_current(box):
    missing = box.run("status")
    assert missing.returncode == 1 and "does not exist" in missing.stdout
    box.target.write_bytes(OWN)
    absent = box.run("status")
    assert absent.returncode == 1
    assert "has no nvidia-tao-toolkit agent identity block" in absent.stdout
    old = box.bank(b"# The identity of an older release\n", "old-bank")
    box.ok("add", bank=old)
    outdated = box.run("status")
    assert outdated.returncode == 1 and "differs from the current identity" in outdated.stdout
    assert 'Run "add" to update it' in outdated.stdout
    box.ok("add")
    current = box.run("status")
    assert current.returncode == 0 and "it is the current identity" in current.stdout
    # status writes nothing
    stamp = box.target.stat().st_mtime_ns
    box.run("status")
    assert box.target.stat().st_mtime_ns == stamp


def test_status_notices_a_block_that_was_edited_by_hand(box):
    box.target.write_bytes(OWN)
    box.ok("add")
    box.target.write_bytes(box.target.read_bytes().replace(b"TAO Claw Agent", b"My Agent", 1))
    assert box.run("status").returncode == 1
    box.ok("add")
    assert box.target.read_bytes() == OWN + b"\n" + _block() + b"\n"


# ─────────────────────────────── refusals ───────────────────────────────
BROKEN_MARKERS = {
    "begin-without-end": (OWN + b"\n" + BEGIN_LINE + b"\nhalf a block\n",
                          "has a BEGIN marker (line 6) and no END marker"),
    "end-without-begin": (OWN + b"\nthe rest of a block\n" + END_LINE + b"\n",
                          "has an END marker (line 7) and no BEGIN marker"),
    "two-blocks": (_block() + b"\n\n" + OWN + b"\n" + _block() + b"\n", "has more than one"),
    "two-begins": (OWN + BEGIN_LINE + b"\none\n" + BEGIN_LINE + b"\ntwo\n" + END_LINE + b"\n",
                   "has more than one"),
    "end-before-begin": (END_LINE + b"\n" + OWN + BEGIN_LINE + b"\n",
                         "END marker (line 1) before its BEGIN marker"),
}


@pytest.mark.parametrize("command", ["add", "remove", "status"])
@pytest.mark.parametrize("case", sorted(BROKEN_MARKERS))
def test_marker_lines_that_do_not_make_one_block_are_refused(box, command, case):
    content, message = BROKEN_MARKERS[case]
    box.target.write_bytes(content)
    error = box.refused(command)
    assert message in error and "Nothing was written" in error and str(box.target) in error


@pytest.mark.parametrize("command", ["add", "remove", "status"])
def test_a_target_that_is_a_directory_is_refused(box, command):
    box.target.mkdir()
    assert "is a directory, not a file" in box.refused(command)
    assert "is a directory, not a file" in box.refused(command, "--file", box.project)
    assert list(box.target.iterdir()) == []


@pytest.mark.parametrize("command", ["add", "status"])
def test_a_bank_without_an_identity_is_refused(box, command):
    box.target.write_bytes(OWN)
    empty = box.root / "no-bank"
    empty.mkdir()
    error = box.refused(command, bank=empty)
    assert f"the bank has no AGENTS.md: {empty / 'AGENTS.md'} is missing" in error
    assert "TAO_SKILL_BANK_PATH" in error
    blank = box.bank(b"\n \n", "blank-bank")
    assert "is empty" in box.refused(command, bank=blank)
    marked = box.bank(b"# Identity\n" + END_LINE + b"\n", "marked-bank")
    assert "contains a marker line" in box.refused(command, bank=marked)


def test_a_target_that_is_the_identity_itself_is_refused(box):
    """Inside a clone of the repository the default target is the identity file of the bank."""
    clone = box.bank(IDENTITY, "clone")
    (clone / ".git").mkdir()
    (clone / "skills").mkdir()
    error = box.refused("add", cwd=clone / "skills", bank=clone)
    assert f"is the identity file of the bank in {clone}" in error and "--project" in error
    # and so is a whole-file copy, which earlier installers made in the Codex home
    (box.home / ".codex").mkdir()
    (box.home / ".codex" / "AGENTS.md").write_bytes(IDENTITY)
    error = box.refused("add", "--file", "~/.codex/AGENTS.md")
    assert "is itself a copy of the identity" in error and "the identity twice" in error


def test_the_identity_file_of_the_bank_the_helper_ships_in_is_refused(box):
    """Run in a clone while the variable names another bank: its AGENTS.md is no project file."""
    clone = box.bank(b"# The identity of this clone\n", "clone")
    (clone / ".git").mkdir()
    (clone / "scripts").mkdir()
    copy = clone / "scripts" / "tao_agent_identity.py"
    shutil.copy(HELPER, copy)
    other = box.bank(b"# The identity of another bank\n", "other-bank")
    error = box.refused("add", cwd=clone / "scripts", helper=copy, bank=other)
    assert f"is the identity file of the bank in {clone}" in error and "--project" in error


def test_a_whole_copy_of_the_identity_of_another_release_is_refused(box):
    """Earlier installers copied the identity of their release over ~/.codex/AGENTS.md.

    Such a copy differs from the identity of today and starts with the same heading. A
    block appended to it would give the agent two identities.
    """
    older = IDENTITY.replace(b"You help users", b"You assist users", 1)
    assert older != IDENTITY and older.startswith(HEADING + b"\n")
    (box.home / ".codex").mkdir()
    copy = box.home / ".codex" / "AGENTS.md"
    for content in (older, b"\xef\xbb\xbf\r\n" + older.replace(b"\n", b"\r\n"),
                    HEADING + b"\n\nplus rules of my own\n"):
        copy.write_bytes(content)
        for args in (("add",), ("add", "--dry-run")):
            error = box.refused(*args, "--file", "~/.codex/AGENTS.md")
            assert f"starts with the heading of the identity itself ({HEADING.decode()})" in error
            assert "the identity twice" in error and "restore your own file" in error
    assert box.run("status", "--file", copy).returncode == 1       # no block, and no refusal
    assert "Nothing to do" in box.ok("remove", "--file", copy).stdout
    # the same heading further down is a line of the user's own text
    box.target.write_bytes(OWN + b"\n" + HEADING + b"\n")
    box.ok("add")
    assert box.target.read_bytes() == OWN + b"\n" + HEADING + b"\n\n" + _block() + b"\n"
    # and a file that has the block is updated, whatever stands before it
    box.target.write_bytes(HEADING + b"\n\n" + _block(b"# An older identity\n") + b"\n")
    box.ok("add")
    assert box.target.read_bytes() == HEADING + b"\n\n" + _block() + b"\n"


def test_a_file_that_is_not_utf_8_text_is_refused(box):
    """Redirection in Windows PowerShell 5 writes UTF-16; UTF-8 bytes after it make no text."""
    text = "# My rules\r\n\r\n- one\r\n"
    for content in (text.encode("utf-16"), b"\xff\xfe" + text.encode("utf-16-le"),
                    b"\xfe\xff" + text.encode("utf-16-be"), text.encode("utf-16-le"),
                    b"\xff\xfe", b"# My rules\n\x00\n"):
        box.target.write_bytes(content)
        for args in (("add",), ("add", "--dry-run")):
            error = box.refused(*args)
            assert "is not UTF-8 text" in error and "Convert the file to UTF-8 first" in error
        assert box.run("status").returncode == 1
        assert "Nothing to do" in box.ok("remove").stdout and box.target.read_bytes() == content


def test_a_read_only_file_is_left_as_it_is(box):
    """The write replaces the file, which the folder allows; the mode of the file says no."""
    if os.geteuid() == 0:
        pytest.skip("the superuser may write every file")

    def read_only(content):
        if box.target.exists():
            box.target.chmod(0o644)
        box.target.write_bytes(content)
        box.target.chmod(0o444)

    old = _block(b"# The identity of an older release\n")
    cases = (
        (OWN, ("add",)),                                    # append
        (OWN + b"\n" + old + b"\n", ("add",)),              # update
        (OWN + b"\n" + old + b"\n", ("remove",)),
        (_block() + b"\n", ("remove",)),                    # would delete the file
    )
    for content, command in cases:
        read_only(content)
        for args in (command, (*command, "--dry-run")):
            error = box.refused(*args)
            assert "is read-only, so it was left as it is" in error and "chmod u+w" in error
            assert box.target.stat().st_mode & 0o777 == 0o444
    assert box.run("status").returncode == 0                # reading is not refused
    current = box.ok("add")                                 # and neither is having nothing to do
    assert "Nothing to do" in current.stdout and box.target.stat().st_mode & 0o777 == 0o444
    # a link to a read-only file is refused as well
    real = box.root / "shared.md"
    real.write_bytes(OWN)
    real.chmod(0o444)
    box.target.unlink()
    box.target.symlink_to(real)
    assert "is read-only" in box.refused("add") and real.read_bytes() == OWN


def test_targets_that_cannot_be_written_are_refused(box):
    assert "is not an existing folder" in box.refused("add", "--project", box.root / "nowhere")
    error = box.refused("add", "--file", box.root / "nowhere" / "AGENTS.md")
    assert f"the folder {box.root / 'nowhere'} does not exist" in error
    # an empty path is a variable that was not set, not a wish for the current directory
    for option in ("--project", "--file"):
        assert "an empty one was given" in box.refused("add", option, "")
    box.target.symlink_to(box.root / "gone.md")
    assert "is a symbolic link to" in box.refused("add") and box.target.is_symlink()
    assert "which does not exist" in box.refused("remove")


def test_a_current_directory_that_was_deleted_is_refused(box):
    gone = box.root / "gone"
    gone.mkdir()
    before = _listing(box.root)
    result = subprocess.run(
        ["sh", "-c", 'cd "$1" && rmdir "$1" && shift && exec "$@"', "sh", str(gone),
         box.python, str(HELPER), "add"],
        env=box.environment(), capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2, (result.stdout, result.stderr)
    assert "the current directory cannot be read" in result.stderr and "--project" in result.stderr
    assert "Traceback" not in result.stderr
    del before["gone"]
    assert _listing(box.root) == before


def test_a_path_that_the_terminal_cannot_show_does_not_stop_the_helper(box):
    """The message comes after the write; an encoding error there would report a failure."""
    project = box.root / "projet d'\u00e9t\u00e9"
    project.mkdir()
    (project / "AGENTS.md").write_bytes(OWN)
    ascii_only = {"PYTHONIOENCODING": "ascii", "cwd": box.home}
    added = box.ok("add", "--project", project, **ascii_only)
    assert (project / "AGENTS.md").read_bytes() == OWN + b"\n" + _block() + b"\n"
    assert "Appended" in added.stdout and "d'\\xe9t\\xe9" in added.stdout
    assert box.run("status", "--project", project, **ascii_only).returncode == 0
    removed = box.ok("remove", "--project", project, **ascii_only)
    assert (project / "AGENTS.md").read_bytes() == OWN and "Removed" in removed.stdout
    refused = box.refused("add", "--project", project / "nowhere", **ascii_only)
    assert "d'\\xe9t\\xe9" in refused and "Traceback" not in refused


def test_command_line_mistakes_change_nothing(box):
    box.target.write_bytes(OWN)
    for args in (("add", "--project", box.project, "--file", box.target), (), ("install",),
                 ("status", "--dry-run")):
        before = _listing(box.root)
        result = box.run(*args)
        assert result.returncode == 2 and "usage:" in result.stderr, args
        assert _listing(box.root) == before


# ─────────────────────────────── what a marker line is ───────────────────────────────
# Text that resembles the marker lines, with a line of the user's own between. None of
# it starts a line with the marker text and ends it with "-->", so none of it is a block.
NOT_MARKERS = {
    "mid-line": b"See " + BEGIN_LINE + b"\nmine\nand then " + END_LINE + b"\n",
    "indented": b"    " + BEGIN_LINE + b"\n    mine\n    " + END_LINE + b"\n",
    "one-space": b" " + BEGIN_LINE + b"\nmine\n " + END_LINE + b"\n",
    "tab": b"\t" + BEGIN_LINE + b"\nmine\n\t" + END_LINE + b"\n",
    "block-quote": b"> " + BEGIN_LINE + b"\n> mine\n> " + END_LINE + b"\n",
    "list-item": b"- " + BEGIN_LINE + b"\n  mine\n- " + END_LINE + b"\n",
    "inline-code": b"`" + BEGIN_LINE + b"`\nmine\n`" + END_LINE + b"`\n",
    "not-closed": BEGIN_PREFIX + b": a comment that goes on\nmine\n" + END_PREFIX + b"\n",
    "words-after": BEGIN_LINE + b" starts it\nmine\n" + END_LINE + b" ends it\n",
    "longer-word": BEGIN_PREFIX + b"2 -->\nmine\n" + END_PREFIX + b"2 -->\n",
    "another-plugin": (b"<!-- BEGIN nvidia-tao-toolkit-extras agent identity -->\nmine\n"
                       b"<!-- END nvidia-tao-toolkit-extras agent identity -->\n"),
    "lower-case": BEGIN_LINE.lower() + b"\nmine\n" + END_LINE.lower() + b"\n",
}


@pytest.mark.parametrize("case", sorted(NOT_MARKERS))
def test_text_that_only_resembles_a_marker_line_is_the_users_own(box, case):
    content = OWN + b"\n" + NOT_MARKERS[case] + b"\n- and a rule after it\n"
    box.target.write_bytes(content)
    assert box.run("status").returncode == 1
    assert "has no nvidia-tao-toolkit agent identity block" in box.ok("remove").stdout
    assert box.target.read_bytes() == content

    old = box.bank(b"# The identity of an older release\n", "old-bank")
    assert "Appended" in box.ok("add", bank=old).stdout
    assert "Updated" in box.ok("add").stdout
    assert box.target.read_bytes() == content + b"\n" + _block() + b"\n"
    assert box.run("status").returncode == 0
    box.ok("remove")
    assert box.target.read_bytes() == content


# OWN has four lines and a blank line follows it, so each of these opens on line 6.
FENCED_EXAMPLES = {
    "backticks": b"```html\n" + BEGIN_LINE + b"\nmy own example\n" + END_LINE + b"\n```\n",
    "tildes": b"~~~\n" + BEGIN_LINE + b"\nmy own example\n" + END_LINE + b"\n~~~~\n",
    "indented-fence": (b"   ````md\n" + BEGIN_LINE + b"\nmy own example\n" + END_LINE
                       + b"\n  ````\n"),
    "other-fence-inside": (b"````\n```\n~~~~~\n" + BEGIN_LINE + b"\nmy own example\n" + END_LINE
                           + b"\n````\n"),
    "left-open": b"```\nnever closed\n" + BEGIN_LINE + b"\nold text\n" + END_LINE + b"\n",
    "text-after-a-fence": (b"```\n``` does not close it\n" + BEGIN_LINE + b"\nmy own example\n"
                           + END_LINE + b"\n```\n"),
    "windows": (b"```\r\n" + BEGIN_LINE + b"\r\nmy own example\r\n" + END_LINE
                + b"\r\n```\r\n"),
}


@pytest.mark.parametrize("command", ["add", "remove", "status"])
@pytest.mark.parametrize("case", sorted(FENCED_EXAMPLES))
def test_marker_lines_in_a_fenced_code_block_are_not_taken_for_a_block(box, command, case):
    """An example of the block in the user's own words: updating or removing it would destroy it."""
    box.target.write_bytes(OWN + b"\n" + FENCED_EXAMPLES[case])
    error = box.refused(command)
    assert "inside the fenced code block that opens on line 6" in error
    assert "Nothing was written" in error and str(box.target) in error


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"], ids=["unix", "windows"])
def test_a_code_block_that_is_closed_before_the_block_does_not_matter(box, newline):
    code = (b"```bash\nmake test\n```\n\n~~~\n```\n~~~  \n\n```inline``` code is no fence\n"
            b"    ```\n\nfour spaces make that line code, not a fence\n")
    content = (OWN + b"\n" + code).replace(b"\n", newline)
    box.target.write_bytes(content)
    old = box.bank(b"# The identity of an older release\n", "old-bank")
    assert "Appended" in box.ok("add", bank=old).stdout
    assert "Updated" in box.ok("add").stdout
    assert box.target.read_bytes() == content + newline + _block(newline=newline) + newline
    box.ok("remove")
    assert box.target.read_bytes() == content


def test_add_does_not_append_to_a_file_that_ends_inside_an_open_code_block(box):
    """The block would be code there, and the next run would refuse to touch it."""
    box.target.write_bytes(OWN + b"```\nthe fence above is never closed\n")
    for args in (("add",), ("add", "--dry-run")):
        error = box.refused(*args)
        assert "ends inside a fenced code block that opens on line 5" in error
        assert "Close the code block first" in error
    box.target.write_bytes(OWN + b"```\nnow it is\n```\n")
    assert "Appended" in box.ok("add").stdout


def test_a_code_block_inside_the_identity_does_not_stop_update_and_remove(box):
    """Only what stands before the BEGIN line decides; the identity may hold code of its own."""
    fenced = box.bank(b"# Identity\n\n```bash\nmake\n```\n", "fenced-bank")
    left_open = box.bank(b"# Identity\n\n```\nnever closed\n", "open-bank")
    later = b"\n- a rule after the block\n"
    box.target.write_bytes(OWN)
    for bank in (fenced, left_open):
        assert box.ok("add", bank=bank).stdout.startswith(("Appended", "Updated"))
        box.target.write_bytes(box.target.read_bytes() + later)
        assert box.run("status", bank=bank).returncode == 0
        assert "Updated" in box.ok("add").stdout
        assert box.target.read_bytes() == OWN + b"\n" + _block() + b"\n" + later
        box.ok("remove")
        assert box.target.read_bytes() == OWN + later
        box.target.write_bytes(OWN)


# ─────────────────────────────── --dry-run ───────────────────────────────
def test_dry_run_prints_what_add_would_do_and_writes_nothing(box):
    created = box.ok("add", "--dry-run")
    assert f"Would create {box.target}" in created.stdout
    assert "Dry run: nothing was written." in created.stdout and not box.target.exists()

    box.target.write_bytes(OWN)
    before = _listing(box.root)
    appended = box.ok("add", "--dry-run")
    assert "Would append" in appended.stdout and f"the {len(OWN)} bytes" in appended.stdout
    assert f"would have {len(OWN + b'x' + _block() + b'x')} bytes" in appended.stdout
    assert _listing(box.root) == before

    old = box.bank(b"# The identity of an older release\n", "old-bank")
    box.ok("add", bank=old)
    before = _listing(box.root)
    updated = box.ok("add", "--dry-run")
    assert "Would update" in updated.stdout and _listing(box.root) == before

    box.ok("add")
    before = _listing(box.root)
    current = box.ok("add", "--dry-run")
    assert "Nothing to do." in current.stdout and _listing(box.root) == before


def test_dry_run_prints_what_remove_would_do_and_writes_nothing(box):
    box.target.write_bytes(OWN)
    box.ok("add")
    before = _listing(box.root)
    removed = box.ok("remove", "--dry-run")
    assert "Would remove" in removed.stdout and "Dry run: nothing was written." in removed.stdout
    assert _listing(box.root) == before

    box.target.write_bytes(_block() + b"\n")
    before = _listing(box.root)
    deleted = box.ok("remove", "--dry-run")
    assert f"and delete {box.target}" in deleted.stdout and _listing(box.root) == before


# ─────────────────────────────── which file ───────────────────────────────
def _work_tree(folder):
    """Make ``folder`` the top of a git work tree; a real one where git is installed."""
    folder.mkdir(parents=True, exist_ok=True)
    if shutil.which("git"):
        subprocess.run(["git", "init", "--quiet", str(folder)], check=True, capture_output=True)
    else:
        (folder / ".git").mkdir()
    assert (folder / ".git").exists()
    return folder


def test_default_target_is_the_top_of_the_git_work_tree_around_the_current_directory(box):
    top = _work_tree(box.root / "work tree")
    deep = top / "src" / "package"
    deep.mkdir(parents=True)
    box.ok("add", cwd=deep)
    assert (top / "AGENTS.md").read_bytes() == _block() + b"\n"
    assert list(deep.iterdir()) == [] and not (top / "src" / "AGENTS.md").exists()
    assert box.run("status", cwd=top / "src").returncode == 0
    box.ok("remove", cwd=deep)
    assert not (top / "AGENTS.md").exists()


def test_default_target_is_the_nearest_work_tree_and_a_git_file_counts(box):
    """A submodule or a linked work tree has a .git file, and it is a project of its own."""
    outer = _work_tree(box.root / "outer")
    inner = outer / "vendor" / "inner"
    (inner / "lib").mkdir(parents=True)
    (inner / ".git").write_text("gitdir: ../../.git/modules/inner\n")
    box.ok("add", cwd=inner / "lib")
    assert (inner / "AGENTS.md").exists() and not (outer / "AGENTS.md").exists()


def test_default_target_outside_a_work_tree_is_the_current_directory(box):
    folder = box.root / "notes" / "today"
    folder.mkdir(parents=True)
    box.ok("add", cwd=folder)
    assert (folder / "AGENTS.md").read_bytes() == _block() + b"\n"
    assert not (box.root / "notes" / "AGENTS.md").exists() and not box.target.exists()


def test_project_option_names_the_folder_and_does_not_look_for_a_work_tree(box):
    top = _work_tree(box.root / "work tree")
    part = top / "services" / "payments"
    part.mkdir(parents=True)
    box.ok("add", "--project", part, cwd=box.home)
    assert (part / "AGENTS.md").exists() and not (top / "AGENTS.md").exists()
    assert list(box.home.iterdir()) == []
    assert box.run("status", "--project", part, cwd=box.home).returncode == 0


def test_file_option_adds_the_block_to_a_global_file_and_writes_nothing_else_there(box):
    """~/.codex/AGENTS.md may be named on purpose. It gets the block, never a whole copy."""
    codex_home = box.home / ".codex"
    codex_home.mkdir()
    own = b"# My global rules\n\n- always answer in French\n"
    (codex_home / "AGENTS.md").write_bytes(own)
    (codex_home / "config.toml").write_bytes(b'model = "x"\n')

    box.ok("add", "--file", "~/.codex/AGENTS.md", cwd=box.project)
    assert (codex_home / "AGENTS.md").read_bytes() == own + b"\n" + _block() + b"\n"
    assert sorted(path.name for path in codex_home.iterdir()) == ["AGENTS.md", "config.toml"]
    assert not box.target.exists()                       # the project was not touched
    assert box.run("status", "--file", codex_home / "AGENTS.md").returncode == 0
    box.ok("remove", "--file", "~/.codex/AGENTS.md")
    assert (codex_home / "AGENTS.md").read_bytes() == own
    assert sorted(path.name for path in codex_home.iterdir()) == ["AGENTS.md", "config.toml"]


def test_file_option_does_not_create_a_missing_home_folder(box):
    error = box.refused("add", "--file", "~/.codex/AGENTS.md")
    assert f"the folder {box.home / '.codex'} does not exist" in error
    assert list(box.home.iterdir()) == []


# ─────────────────────────────── which identity ───────────────────────────────
def test_identity_comes_from_the_bank_the_helper_ships_in_unless_the_variable_names_another(box):
    shipped = box.bank(b"# The identity this copy of the helper ships with\n", "plugin-folder")
    (shipped / "scripts").mkdir()
    copy = shipped / "scripts" / "tao_agent_identity.py"
    shutil.copy(HELPER, copy)
    box.ok("add", helper=copy)
    assert b"this copy of the helper ships with" in box.target.read_bytes()
    assert box.run("status", helper=copy).returncode == 0
    assert box.run("status").returncode == 1               # the repository has another identity

    other = box.bank(b"# The identity of the bank the variable names\n", "other-bank")
    box.ok("add", helper=copy, bank=other)
    assert box.target.read_bytes() == _block(other.joinpath("AGENTS.md").read_bytes()) + b"\n"


# ─────────────────────────────── warnings ───────────────────────────────
def test_add_warns_when_the_file_is_longer_than_codex_reads(box):
    own = b"- a rule that fills a line of the user's own instructions\n" * 400
    box.target.write_bytes(own)
    assert len(own) < CODEX_LIMIT < len(own) + len(IDENTITY)
    result = box.ok("add")
    assert box.target.read_bytes() == own + b"\n" + _block() + b"\n"       # warned, not refused
    assert "WARNING" in result.stderr and f"at most {CODEX_LIMIT} bytes" in result.stderr
    assert str(len(box.target.read_bytes())) in result.stderr
    assert "WARNING" in box.run("status").stderr
    # a file within the limit gets no warning
    box.target.write_bytes(OWN)
    assert box.ok("add").stderr == ""


def test_add_warns_when_codex_reads_an_override_file_instead(box):
    (box.project / "AGENTS.override.md").write_bytes(b"# Temporary rules\n")
    result = box.ok("add")
    assert "WARNING" in result.stderr and "AGENTS.override.md exists" in result.stderr
    assert box.target.read_bytes() == _block() + b"\n"
    assert (box.project / "AGENTS.override.md").read_bytes() == b"# Temporary rules\n"
