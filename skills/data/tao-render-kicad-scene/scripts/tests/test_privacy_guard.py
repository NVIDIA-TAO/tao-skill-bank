# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Privacy guard against self-made files and a simulated denylist; no real data involved."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

GUARD = Path(__file__).resolve().parents[1] / "privacy_guard.py"


def run(cwd, *args, stdin=None):
    return subprocess.run([sys.executable, str(GUARD), *args], cwd=cwd, input=stdin, capture_output=True, text=True)


def git(cwd, *args):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid")
    return subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture()
def repo(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    (private / "secret.txt").write_text("pretend private reference bytes\n")
    work = tmp_path / "repo"
    work.mkdir()
    git(work, "init", "-q")
    denylist = tmp_path / "denylist.txt"
    built = run(work, "build", "--root", str(private), "--out", str(denylist))
    assert built.returncode == 0, built.stderr
    assert json.loads(built.stdout)["unique_sha256"] == 1
    return work, private, denylist


def staged(work, denylist, *extra):
    result = run(work, "check", "--denylist", str(denylist), "--staged", *extra)
    return result.returncode, [f["problem"] for f in json.loads(result.stdout)["findings"]] if result.stdout else []


def test_plain_code_passes(repo):
    work, _, denylist = repo
    (work / "ok.py").write_text("print('hello')\n")
    git(work, "add", "ok.py")
    assert staged(work, denylist) == (0, [])


def test_denylisted_content_is_blocked_even_when_renamed(repo):
    work, private, denylist = repo
    (work / "notes.txt").write_bytes((private / "secret.txt").read_bytes())
    git(work, "add", "notes.txt")
    code, problems = staged(work, denylist)
    assert code == 1 and "content SHA-256 is a private source file" in problems


def test_unknown_binary_fails_closed_unless_allowed(repo):
    work, _, denylist = repo
    (work / "render.png").write_bytes(b"\x89PNG\r\n\x1a\n\0\0\0self-made")
    git(work, "add", "render.png")
    code, problems = staged(work, denylist)
    assert code == 1 and any("binary/image payload" in p for p in problems)
    assert staged(work, denylist, "--allow", "render.png") == (0, [])


def test_restricted_name_and_embedded_image_are_blocked(repo):
    work, _, denylist = repo
    (work / "U1_SolderLight.txt").write_text("name only\n")
    # Assembled at run time so this test file itself does not trip the embedded-image rule.
    (work / "page.html").write_text('<img src="data:image/' + 'png;base64,AAAA">\n')
    git(work, "add", ".")
    code, problems = staged(work, denylist)
    assert code == 1 and "restricted file name" in problems and "embedded base64 image" in problems


def test_lfs_pointer_to_private_file_is_blocked(repo):
    work, private, denylist = repo
    oid = denylist.read_text().splitlines()[1]
    (work / "big.bin.pointer").write_text(f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize 3\n")
    git(work, "add", ".")
    code, problems = staged(work, denylist)
    assert code == 1 and "Git LFS oid is a private source file" in problems


def test_build_works_outside_a_repository_and_refuses_inside(repo, tmp_path):
    work, private, _ = repo
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    assert run(outside, "build", "--root", str(private), "--out", str(outside / "d.txt")).returncode == 0
    inside = run(work, "build", "--root", str(private), "--out", str(work / "d.txt"))
    assert inside.returncode != 0 and "outside the repository" in inside.stderr


def test_missing_or_empty_denylist_fails_closed(repo):
    work, _, denylist = repo
    assert run(work, "check", "--denylist", str(denylist) + ".absent", "--staged").returncode != 0
    denylist.write_text("# nothing\n")
    assert run(work, "check", "--denylist", str(denylist), "--staged").returncode != 0


def test_pre_push_scans_commits_and_hooks_install(repo):
    work, private, denylist = repo
    (work / "ok.py").write_text("x = 1\n")
    git(work, "add", ".")
    git(work, "commit", "-q", "-m", "clean")
    clean = git(work, "rev-parse", "HEAD")
    (work / "leak.txt").write_bytes((private / "secret.txt").read_bytes())
    git(work, "add", ".")
    git(work, "commit", "-q", "-m", "leak")
    leak = git(work, "rev-parse", "HEAD")
    zero = "0" * 40
    ok = run(work, "check", "--denylist", str(denylist), "--pre-push", stdin=f"refs/heads/x {clean} refs/heads/x {zero}\n")
    assert ok.returncode == 0, ok.stdout
    bad = run(work, "check", "--denylist", str(denylist), "--pre-push", stdin=f"refs/heads/x {leak} refs/heads/x {clean}\n")
    assert bad.returncode == 1
    installed = run(work, "install", "--denylist", str(denylist))
    assert installed.returncode == 0, installed.stderr
    hooks = Path(git(work, "rev-parse", "--git-common-dir"))
    hooks = (work / hooks if not hooks.is_absolute() else hooks) / "hooks"
    assert (hooks / "pre-commit").stat().st_mode & 0o111 and (hooks / "pre-push").exists()
