# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for PAS plugin-cache versioning."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/check_pas_plugin_version.py"


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*args], cwd=root, check=True, capture_output=True, text=True
    )


def _write_manifests(root: Path, version: str) -> None:
    documents = {
        ".claude-plugin/marketplace.json": {"metadata": {"version": version}},
        ".claude-plugin/plugin.json": {"version": version},
        ".codex-plugin/plugin.json": {"version": version},
    }
    for relative, document in documents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document), encoding="utf-8")


def _check(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--base-ref",
            "HEAD",
            "--repo-root",
            str(root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_pas_change_requires_a_strictly_newer_synchronized_plugin_version(tmp_path):
    _run(tmp_path, "git", "init", "-q")
    _run(tmp_path, "git", "config", "user.email", "pas-test@nvidia.com")
    _run(tmp_path, "git", "config", "user.name", "PAS Test")
    _write_manifests(tmp_path, "0.1.13")
    skill = tmp_path / "skills/applications/tao-run-deft-pas/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("version one\n", encoding="utf-8")
    _run(tmp_path, "git", "add", ".")
    _run(tmp_path, "git", "commit", "-qm", "base")

    skill.write_text("version two\n", encoding="utf-8")
    unchanged = _check(tmp_path)
    assert unchanged.returncode == 1
    assert "without increasing" in unchanged.stderr

    _write_manifests(tmp_path, "0.1.14")
    bumped = _check(tmp_path)
    assert bumped.returncode == 0, bumped.stderr
    assert "0.1.13 -> 0.1.14" in bumped.stdout


def test_pas_change_rejects_manifest_version_drift(tmp_path):
    _run(tmp_path, "git", "init", "-q")
    _run(tmp_path, "git", "config", "user.email", "pas-test@nvidia.com")
    _run(tmp_path, "git", "config", "user.name", "PAS Test")
    _write_manifests(tmp_path, "0.1.13")
    skill = tmp_path / "skills/applications/tao-run-deft-pas/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("version one\n", encoding="utf-8")
    _run(tmp_path, "git", "add", ".")
    _run(tmp_path, "git", "commit", "-qm", "base")

    skill.write_text("version two\n", encoding="utf-8")
    _write_manifests(tmp_path, "0.1.14")
    (tmp_path / ".codex-plugin/plugin.json").write_text(
        json.dumps({"version": "0.1.15"}), encoding="utf-8"
    )

    drifted = _check(tmp_path)
    assert drifted.returncode == 1
    assert "versions disagree" in drifted.stderr
