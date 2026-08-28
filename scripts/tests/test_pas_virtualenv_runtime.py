# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for PAS's production virtualenv contract."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
PAS_SCRIPTS = REPO / "skills" / "applications" / "tao-run-deft-pas" / "scripts"
sys.path.insert(0, str(PAS_SCRIPTS))
import manage_pas_virtualenv as manager  # noqa: E402
import virtualenv_runtime as runtime  # noqa: E402


def _profile_shell(tmp_path: Path, profile: str, cli: str) -> Path:
    venv = tmp_path / profile
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    python = bin_dir / "python"
    os.symlink(sys.executable, python)
    entrypoint = bin_dir / cli
    entrypoint.write_text(f"#!{python}\n", encoding="utf-8")
    entrypoint.chmod(0o755)
    return venv


def test_manifest_is_schema_valid_and_incomplete_locks_fail_closed():
    manifest = runtime.load_runtime_manifest()
    assert manifest["python"]["major_minor"] == "3.12"
    assert manifest["cuda"] == {
        "torch_version": "2.11.0+cu130",
        "torch_cuda_build": "13.0",
        "pytorch_index_url": "https://download.pytorch.org/whl/cu130",
    }
    for profile in runtime.PROFILE_NAMES:
        status = runtime.lock_status(profile)
        assert status["declared_status"] == "generation_required"
        assert status["prebuilt_verification_available"] is True
        assert status["ready_to_install"] is False
        assert "complete transitive" in status["blocker"]


def test_install_refuses_incomplete_lock_before_creating_target(tmp_path, capsys):
    target = tmp_path / "must-not-exist"
    result = manager.main(
        [
            "install",
            "--profile",
            "pyt",
            "--virtualenv",
            str(target),
            "--approve-install",
        ]
    )
    assert result == 2
    assert not target.exists()
    assert "complete transitive" in capsys.readouterr().err


def test_install_builds_at_final_non_relocatable_path(tmp_path, monkeypatch):
    target = tmp_path / "pyt-runtime"
    commands = []
    monkeypatch.setattr(
        manager,
        "lock_status",
        lambda profile: {
            "ready_to_install": True,
            "blocker": None,
            "lock_file": str(tmp_path / "complete.lock"),
        },
    )
    monkeypatch.setattr(manager.shutil, "which", lambda name: f"/tools/{name}")
    monkeypatch.setattr(
        manager.subprocess,
        "run",
        lambda command, **kwargs: commands.append([str(item) for item in command]),
    )
    monkeypatch.setattr(
        manager,
        "validate_tao_virtualenv",
        lambda path, **kwargs: Path(path).resolve(),
    )
    assert manager.main(
        [
            "install",
            "--profile",
            "pyt",
            "--virtualenv",
            str(target),
            "--approve-install",
        ]
    ) == 0
    assert target.is_dir()
    assert commands[0][-1] == str(target)
    assert commands[1][0] == str(target / "bin" / "python")


def test_failed_install_removes_only_reserved_target(tmp_path, monkeypatch):
    target = tmp_path / "ds-runtime"
    monkeypatch.setattr(
        manager,
        "lock_status",
        lambda profile: {
            "ready_to_install": True,
            "blocker": None,
            "lock_file": str(tmp_path / "complete.lock"),
        },
    )
    monkeypatch.setattr(manager.shutil, "which", lambda name: f"/tools/{name}")

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(manager.subprocess, "run", fail)
    assert manager.main(
        [
            "install",
            "--profile",
            "ds",
            "--virtualenv",
            str(target),
            "--approve-install",
        ]
    ) == 2
    assert not target.exists()


def test_prebuilt_verification_is_independent_of_install_lock(tmp_path):
    with pytest.raises(ValueError, match="virtualenv is missing or invalid"):
        runtime.validate_tao_virtualenv(
            tmp_path / "untrusted-prebuilt",
            profile="pyt",
            probe_imports=True,
        )


def test_full_ds_profile_verification_checks_metadata_pip_and_real_cuda_probe(
    tmp_path, monkeypatch
):
    venv = _profile_shell(tmp_path, "ds", "embedding")
    calls: list[list[str]] = []
    facts = {
        "implementation": "cpython",
        "python_major_minor": "3.12",
        "machine": "x86_64",
        "glibc": "2.39",
        "prefix": str(venv.resolve()),
        "base_prefix": "/usr",
        "distributions": {},
        "entrypoints": {},
        "imports": [],
        "torch_version": "2.11.0+cu130",
        "torch_cuda_build": "13.0",
    }

    def completed(command, **kwargs):
        calls.append([str(item) for item in command])
        if "-c" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                "import warning\nPAS_VENV_CONTRACT="
                + json.dumps({"errors": [], "facts": facts}),
                "",
            )
        if command[-2:] == ["pip", "check"]:
            return subprocess.CompletedProcess(command, 0, "No broken requirements found.\n", "")
        return subprocess.CompletedProcess(command, 0, "PAS_CUDA_PROBE=PASS {}\n", "")

    monkeypatch.setattr(runtime.subprocess, "run", completed)
    assert runtime.validate_tao_virtualenv(
        venv,
        profile="ds",
        probe_imports=True,
        required_cli="embedding",
        minimum_gpus=2,
    ) == venv.resolve()
    assert len(calls) == 3
    assert calls[1][-3:] == ["-m", "pip", "check"]
    assert "check_pas_cuda_runtime.py" in calls[2][1]
    assert calls[2][-4:] == ["--min-gpus", "2", "--require-cli", "embedding"]
    assert "tmm" not in calls[2]


def test_console_script_must_be_bound_to_profile_python(tmp_path, monkeypatch):
    venv = _profile_shell(tmp_path, "pyt", "clip")
    clip = venv / "bin" / "clip"
    clip.write_text("#!/usr/bin/python3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not bound"):
        runtime.validate_tao_virtualenv(
            venv, profile="pyt", probe_imports=False
        )


def test_legacy_shared_environment_is_verified_as_both_profiles(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    observed = []

    def validate(path, *, profile, probe_imports, required_cli=None, minimum_gpus=None):
        observed.append((Path(path), profile, probe_imports))
        return Path(path).resolve()

    monkeypatch.setattr(runtime, "validate_tao_virtualenv", validate)
    profiles = runtime.resolve_virtualenv_profiles(
        platform="virtualenv",
        legacy=shared,
        pyt=None,
        ds=None,
        probe_imports=True,
    )
    assert profiles == {"pyt": shared.resolve(), "ds": shared.resolve()}
    assert observed == [(shared, "pyt", True), (shared, "ds", True)]
