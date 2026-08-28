#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`deft_exec --submit` must record a handle before it returns.

The blocking path keeps no handle at all: a session that dies mid-stage leaves a
container with no job id, no backend_ref and no naming convention, so nothing in
the bank can find it and it holds its GPU until someone notices by hand. Submit
mode opens the record first (record-then-launch), names the container after the
minted id, and returns that id.

These are argv/unit-level checks that need no Docker daemon. The live
submit -> await -> COMPLETE cycle is exercised by scripts/ci/platform_smoke.sh.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
MODULE = REPO / "skills/applications/tao-run-deft-aoi/scripts/deft_exec.py"


@pytest.fixture(scope="module")
def deft_exec():
    spec = importlib.util.spec_from_file_location("deft_exec_submit", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_detach_names_container_after_job_id(deft_exec):
    """Later verbs locate the job by name, so the name is not cosmetic."""
    out = deft_exec._with_detach(["docker", "run", "img:1", "train"], "arch-train-abc123")
    assert out[:5] == ["docker", "run", "--name", "arch-train-abc123", "-d"]
    assert out[5:] == ["img:1", "train"]


def test_detach_preserves_caller_options(deft_exec):
    """Injection goes after `run`; the caller's own flags survive untouched."""
    out = deft_exec._with_detach(
        ["docker", "run", "--gpus", "all", "-v", "/w:/w", "img:1"], "j-1"
    )
    assert out.index("--gpus") > out.index("-d")
    assert out[-1] == "img:1"


def test_detach_rejects_rm(deft_exec):
    """--rm deletes the container on exit, making the exit code unreadable.

    Without this guard the job can never reach a terminal state: `docker
    inspect` fails, status maps to UNKNOWN forever, and --await-job times out
    leaving the record stuck at RUNNING. Reproduced before the guard existed.
    """
    with pytest.raises(ValueError, match="--rm"):
        deft_exec._with_detach(["docker", "run", "--rm", "img:1"], "j-1")


@pytest.mark.parametrize("flag", ["-d", "--detach", "--name"])
def test_detach_rejects_conflicting_flags(deft_exec, flag):
    """Submit owns detachment and naming; a caller setting them is ambiguous."""
    command = ["docker", "run", flag, "x", "img:1"]
    with pytest.raises(ValueError, match=flag):
        deft_exec._with_detach(command, "j-1")


def test_detach_rejects_non_docker_command(deft_exec):
    """Submit cannot name or detach what it does not understand."""
    with pytest.raises(ValueError, match="docker run"):
        deft_exec._with_detach(["srun", "-n1", "train.sh"], "j-1")


@pytest.mark.parametrize(
    "native,code,expected",
    [
        ("created", 0, "PENDING"),
        ("restarting", 0, "PENDING"),
        ("running", 0, "RUNNING"),
        ("paused", 0, "RUNNING"),
    ],
)
def test_vocabulary_mapping_matches_the_contract(deft_exec, native, code, expected):
    """Native sub-states must fold into the fixed vocabulary, never leak raw."""
    assert deft_exec.DOCKER_STATE_VOCAB[native] == expected


def test_vocabulary_covers_only_non_terminal_states(deft_exec):
    """exited/dead are resolved by exit code at call time, not by the table."""
    assert set(deft_exec.DOCKER_STATE_VOCAB) == {
        "created", "restarting", "running", "paused"
    }
    assert not {"exited", "dead"} & set(deft_exec.DOCKER_STATE_VOCAB)


def test_bank_resolution_prefers_env(deft_exec, monkeypatch, tmp_path):
    monkeypatch.setenv("TAO_SKILL_BANK_PATH", str(tmp_path))
    assert deft_exec._bank() == tmp_path.resolve()


def test_bank_resolution_falls_back_to_repo(deft_exec, monkeypatch):
    """Without the env var the helper must still find scripts/ at the repo root."""
    monkeypatch.delenv("TAO_SKILL_BANK_PATH", raising=False)
    assert (deft_exec._bank() / "scripts" / "tao_job_record.py").is_file()


def test_submit_still_enforces_airgap(deft_exec, tmp_path):
    """Submit must not become a way around the execution policy."""
    state = tmp_path / "s.json"
    state.write_text(
        '{"version":4,"results_dir":"%s",'
        '"execution_policy":{"network_mode":"airgap"}}' % tmp_path
    )
    with pytest.raises(ValueError):
        deft_exec.submit(
            state, ["docker", "run", "--pull=always", "img:1"],
            action="t", image="img:1", network_arch="a",
            storage_tier="A", parent_job=None, platform="docker",
        )
