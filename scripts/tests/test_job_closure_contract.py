#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The canonical four-verb recipe must document how a job is CLOSED.

The recipe every platform skill was copied from used to end at
`mark --state RUNNING`. Nothing anywhere told an agent to write a terminal
state, so `--state COMPLETE` appeared zero times bank-wide and happy-path
records legitimately ended stuck at RUNNING — which makes "poll until terminal"
impossible and leaves a failed tier-C upload unrepresentable.

These tests pin the closure half of the contract in prose, in the same
grep-over-documentation style as test_platform_contract.py, so the omission
cannot recur silently.
"""

from __future__ import annotations

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
CONTRACT = REPO / "skills/core/tao-launch-workflow/SKILL.md"
RECORD = REPO / "scripts/tao_job_record.py"

TERMINAL_STATES = ("COMPLETE", "ERROR", "CANCELED")


@pytest.fixture(scope="module")
def contract() -> str:
    return CONTRACT.read_text(encoding="utf-8")


def test_contract_documents_marking_complete(contract):
    """The recipe must show the closing mark, not just open + RUNNING."""
    assert "--state COMPLETE" in contract, (
        "The four-verb contract never tells an agent to close a job. Without it "
        "no record reaches a terminal state and poll-to-terminal has no terminal."
    )


@pytest.mark.parametrize("state", TERMINAL_STATES)
def test_contract_names_every_terminal_state(contract, state):
    """All three terminal states need a documented writer."""
    assert f"--state {state}" in contract, f"no documented writer for {state}"


def test_contract_defines_complete_as_results_survived(contract):
    """COMPLETE must be defined by artifacts, not by process exit status."""
    assert "results_dir` is readable" in contract or "results survived" in contract, (
        "COMPLETE must be defined against surviving results; 'the backend exited "
        "0' lets a job whose results were discarded look successful."
    )


def test_contract_orders_upload_before_terminal_mark(contract):
    """Upload -> verify -> mark. A terminal record cannot be repaired."""
    upload = contract.find("upload returned 0")
    assert upload != -1, "tier-C upload is not tied to the terminal mark"
    window = contract[upload : upload + 600]
    assert "one-way" in window or "refuses any" in window, (
        "the contract must say the terminal mark is irreversible, which is why "
        "the upload has to be verified first"
    )


def test_terminal_immutability_is_real(tmp_path):
    """The prose claim above must match tao_job_record.py's actual behaviour."""
    import subprocess

    env = {"TAO_STATE_DIR": str(tmp_path), "PATH": "/usr/bin:/bin"}
    job = subprocess.run(
        [
            "python3", str(RECORD), "open", "--platform", "docker",
            "--image", "img:1", "--network-arch", "arch", "--action", "train",
            "--storage-tier", "A", "--results-dir", str(tmp_path / "r"),
        ],
        capture_output=True, text=True, env=env, check=True,
    ).stdout.strip()

    subprocess.run(
        ["python3", str(RECORD), "mark", job, "--state", "COMPLETE"],
        capture_output=True, text=True, env=env, check=True,
    )
    repair = subprocess.run(
        ["python3", str(RECORD), "mark", job, "--state", "ERROR"],
        capture_output=True, text=True, env=env,
    )
    assert repair.returncode != 0, (
        "a terminal record accepted a later transition; the documented "
        "upload-before-mark ordering would then be unnecessary"
    )
