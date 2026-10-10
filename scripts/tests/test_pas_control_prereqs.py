# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for pre-approval PAS control-Python discovery."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO_ROOT
    / "skills/applications/tao-run-deft-pas/scripts/check_pas_control_prereqs.py"
)
SPEC = importlib.util.spec_from_file_location("check_pas_control_prereqs", SCRIPT)
assert SPEC and SPEC.loader
prereqs = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prereqs
SPEC.loader.exec_module(prereqs)


def _fake_python(path: Path, *, venv: bool, pip: bool) -> Path:
    path.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-c\" ]; then\n"
        "  case \"$2\" in\n"
        "    *sys.version_info*) printf '%s\\n' '3.12.0'; exit 0 ;;\n"
        f"    *) exit {0 if venv else 1} ;;\n"
        "  esac\n"
        "fi\n"
        f"if [ \"$1\" = \"-m\" ] && [ \"$2\" = \"pip\" ]; then exit {0 if pip else 1}; fi\n"
        "exit 1\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def test_checker_runs_without_site_packages_and_does_not_write(tmp_path):
    before = set(tmp_path.rglob("*"))
    completed = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--python", sys.executable],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["ready"] is True
    assert report["next_action"] == "continue-read-only-preflight"
    assert report["required_user_prompt"] is None
    assert set(tmp_path.rglob("*")) == before


def test_missing_venv_and_pip_are_reported_before_approval(tmp_path, monkeypatch):
    python = _fake_python(tmp_path / "python3", venv=False, pip=False)
    monkeypatch.setattr(prereqs, "_debian_family", lambda: True)

    report = prereqs.inspect(str(python))

    assert report["ready"] is False
    assert report["missing_packages"] == ["python3-venv", "python3-pip"]
    assert report["remediation_command"] == (
        "sudo apt install -y python3-venv python3-pip"
    )
    assert report["next_action"] == "request-host-prerequisites"
    assert "pip is already present" not in report["required_user_prompt"]
    assert "python3-venv python3-pip" in report["approval_summary"]


def test_missing_pip_is_not_misreported_as_present(tmp_path, monkeypatch):
    python = _fake_python(tmp_path / "python3", venv=True, pip=False)
    monkeypatch.setattr(prereqs, "_debian_family", lambda: True)

    report = prereqs.inspect(str(python))

    assert report["missing_packages"] == ["python3-pip"]
    statuses = {check["name"]: check["status"] for check in report["checks"]}
    assert statuses == {"python": "pass", "venv_ensurepip": "pass", "pip": "missing"}


def test_missing_python_is_a_value_free_structured_failure(monkeypatch):
    monkeypatch.setattr(prereqs, "_debian_family", lambda: True)

    report = prereqs.inspect("definitely-not-a-python-command")

    assert report["ready"] is False
    assert report["python_executable"] is None
    assert report["missing_packages"] == ["python3", "python3-venv", "python3-pip"]
    assert report["remediation_command"] == (
        "sudo apt install -y python3 python3-venv python3-pip"
    )


def test_python39_os_release_fallback_detects_ubuntu(tmp_path, monkeypatch):
    release = tmp_path / "os-release"
    release.write_text('ID="ubuntu"\nID_LIKE="debian"\n', encoding="utf-8")
    monkeypatch.delattr(prereqs.platform, "freedesktop_os_release")
    original_path = prereqs.pathlib.Path
    monkeypatch.setattr(
        prereqs.pathlib,
        "Path",
        lambda value: release if value == "/etc/os-release" else original_path(value),
    )

    assert prereqs._debian_family() is True
