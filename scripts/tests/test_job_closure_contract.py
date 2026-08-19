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


# ── Cancelling a finished job is a no-op, not a failure ────────────────────
# Found end-to-end. `--cancel` on a COMPLETE job printed
#   tao_job_record.py mark failed: record ... is terminal (COMPLETE);
#   refusing transition to CANCELED
# and exited 2. The guard is right -- a COMPLETE run whose results exist must
# never be relabelled CANCELED -- but the caller's intent, "make sure this is
# not running", was already satisfied. Reporting that as an error trains people
# to ignore cancel's exit code.

def _deft_exec():
    import importlib.util
    path = REPO / "skills/applications/tao-run-deft-aoi/scripts/deft_exec.py"
    spec = importlib.util.spec_from_file_location("deft_exec_cancel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cancelling_a_finished_job_succeeds_quietly(monkeypatch, capsys):
    module = _deft_exec()
    monkeypatch.setattr(module, "_record",
                        lambda *a: '{"terminal_state": "COMPLETE"}')

    def unreachable(*a, **k):
        raise AssertionError("resolved a backend for an already-finished job")

    monkeypatch.setattr(module, "_backend", unreachable)
    assert module.cancel_job("job-1") == 0
    assert "already finished (COMPLETE)" in capsys.readouterr().out


def test_cancel_tolerates_a_job_finishing_mid_flight(monkeypatch, capsys):
    """The check and the mark are not atomic; the race must not read as failure."""
    module = _deft_exec()
    seen = {"n": 0}

    def fake_record(*args):
        if args[0] == "show":
            seen["n"] += 1
            # Not terminal on the first look, terminal by the time we mark.
            return '{"terminal_state": null}' if seen["n"] == 1 else '{"terminal_state": "COMPLETE"}'
        raise ValueError("record is terminal (COMPLETE); refusing transition")

    monkeypatch.setattr(module, "_record", fake_record)
    monkeypatch.setattr(module, "_backend",
                        lambda *a, **k: (type("R", (), {"cancel": staticmethod(lambda *a, **k: True)}), "ref", {}))
    assert module.cancel_job("job-1") == 0
    assert "finished as COMPLETE while being cancelled" in capsys.readouterr().out


def test_cancelling_a_live_job_still_marks_the_record(monkeypatch):
    module = _deft_exec()
    marked = []

    def fake_record(*args):
        if args[0] == "show":
            return '{"terminal_state": null}'
        marked.append(args)
        return ""

    monkeypatch.setattr(module, "_record", fake_record)
    monkeypatch.setattr(module, "_backend",
                        lambda *a, **k: (type("R", (), {"cancel": staticmethod(lambda *a, **k: True)}), "ref", {}))
    assert module.cancel_job("job-1") == 0
    assert any("CANCELED" in a for a in marked[0]), "live job was not closed"
