# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess

import pytest


REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "skills/platform/tao-run-on-slurm/scripts/slurm_action.py"
SPEC = importlib.util.spec_from_file_location("slurm_action", SCRIPT)
assert SPEC and SPEC.loader
ACTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACTION)


def _completed(argv, rc=0, stdout=b"", stderr=b""):
    return subprocess.CompletedProcess(argv, rc, stdout, stderr)


def _args(**values):
    defaults = {
        "login": "user@login",
        "job_id": "iaa-job-1",
        "backend_ref": "12345",
    }
    defaults.update(values)
    return argparse.Namespace(**defaults)


def test_submit_delegates_to_hardened_submit_gate(tmp_path, monkeypatch):
    rendered = tmp_path / "job.sbatch"
    rendered.write_text("#!/usr/bin/env bash\ntrue\n", encoding="utf-8")
    captured = {}

    def submit_action(**kwargs):
        captured.update(kwargs)
        return {"backend_ref": "12345", "job_id": kwargs["job_id"]}

    monkeypatch.setattr(ACTION.submit_gate, "submit_action", submit_action)
    result = ACTION.submit(_args(
        rendered_script=rendered,
        remote_script=pathlib.Path("/lustre/run/job.sbatch"),
        request=pathlib.Path("/lustre/run/action.json"),
        job_binding=pathlib.Path("/lustre/run/binding.json"),
    ))

    assert result == {"backend_ref": "12345", "job_id": "iaa-job-1"}
    assert captured == {
        "login": "user@login",
        "job_id": "iaa-job-1",
        "rendered_script": rendered,
        "remote_script": pathlib.Path("/lustre/run/job.sbatch"),
        "request_path": pathlib.Path("/lustre/run/action.json"),
        "binding_path": pathlib.Path("/lustre/run/binding.json"),
    }


def test_transport_is_noninteractive_bounded_and_has_no_credential_arguments(monkeypatch):
    captured = {}

    def run(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        return _completed(argv)

    monkeypatch.setattr(ACTION.subprocess, "run", run)
    ACTION._ssh("user@login", "squeue -h -j 12345")  # noqa: SLF001

    assert captured["argv"] == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ConnectionAttempts=1",
        "user@login",
        "squeue -h -j 12345",
    ]
    assert captured["timeout"] == ACTION.SSH_OPERATION_TIMEOUT_SECONDS
    assert not any("TOKEN" in token or "PASSWORD" in token for token in captured["argv"])


def test_transport_timeout_is_classified_without_command_disclosure(monkeypatch):
    def run(argv, **_kwargs):
        raise subprocess.TimeoutExpired(argv, ACTION.SSH_OPERATION_TIMEOUT_SECONDS)

    monkeypatch.setattr(ACTION.subprocess, "run", run)
    with pytest.raises(TimeoutError, match="bounded SLURM SSH operation"):
        ACTION._ssh("user@login", "squeue -h -j 12345")  # noqa: SLF001


def test_status_proves_exact_ownership_and_maps_queue_state(monkeypatch):
    calls = []

    def ssh(login, command):
        calls.append((login, command))
        if "TAO_JOB_NAME=" in command:
            return _completed([], stdout=b"TAO_JOB_NAME=iaa-job-1\n")
        if "squeue -h -j" in command:
            return _completed([], stdout=b"RUNNING\n")
        pytest.fail(f"unexpected command: {command}")

    monkeypatch.setattr(ACTION, "_ssh", ssh)
    result = ACTION.status(_args())

    assert result == {
        "backend_ref": "12345",
        "job_id": "iaa-job-1",
        "native_state": "RUNNING",
        "status": "RUNNING",
    }
    assert "scontrol show job -o 12345" in calls[0][1]
    assert calls[1][1] == "squeue -h -j 12345 -o '%T' 2>/dev/null || true"


def test_status_classifies_transient_ssh_timeout_as_unknown(monkeypatch):
    monkeypatch.setattr(
        ACTION, "_assert_job_ownership",
        lambda *_args: (_ for _ in ()).throw(TimeoutError("bounded timeout")),
    )
    result = ACTION.status(_args())
    assert result == {
        "backend_ref": "12345",
        "job_id": "iaa-job-1",
        "native_state": "SSH_TIMEOUT",
        "status": "UNKNOWN",
        "classification": "transient_ssh_timeout",
    }


def test_complete_status_synchronizes_before_reporting_success(monkeypatch, tmp_path):
    monkeypatch.setattr(ACTION, "_assert_job_ownership", lambda *_args: None)
    monkeypatch.setattr(ACTION, "_native_state", lambda *_args, **_kwargs: "COMPLETED")
    captured = {}

    def sync(args):
        captured["request"] = args.request
        return {"outputs": ["/shared/result"], "log_path": "/shared/action.log"}

    monkeypatch.setattr(ACTION, "_sync_complete_action", sync)
    request = tmp_path / "action.json"
    result = ACTION.status(_args(
        request=request, remote_results=pathlib.Path("/lustre/run/results"),
        log_dir=pathlib.Path("/lustre/run/logs"),
    ))

    assert result["status"] == "COMPLETE"
    assert result["synchronized"]["outputs"] == ["/shared/result"]
    assert captured["request"] == request


def test_error_status_synchronizes_log_and_reports_native_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(ACTION, "_assert_job_ownership", lambda *_args: None)
    monkeypatch.setattr(ACTION, "_native_state", lambda *_args, **_kwargs: "FAILED")
    monkeypatch.setattr(ACTION, "_native_exit_code", lambda *_args: 122)
    monkeypatch.setattr(
        ACTION, "_sync_terminal_log",
        lambda _args: {"log_path": "/shared/action.log", "diagnostic_outputs": []},
    )
    result = ACTION.status(_args(
        request=tmp_path / "action.json",
        remote_results=pathlib.Path("/lustre/run/results"),
        log_dir=pathlib.Path("/lustre/run/logs"),
    ))
    assert result["status"] == "ERROR"
    assert result["native_exit_code"] == 122
    assert result["synchronized"] == {
        "log_path": "/shared/action.log", "diagnostic_outputs": [],
    }


def test_unstarted_cancel_uses_zero_elapsed_evidence_without_remote_log(monkeypatch, tmp_path):
    monkeypatch.setattr(ACTION, "_assert_job_ownership", lambda *_args: None)
    monkeypatch.setattr(ACTION, "_native_state", lambda *_args, **_kwargs: "CANCELLED")
    monkeypatch.setattr(ACTION, "_native_elapsed_raw", lambda *_args: 0)
    monkeypatch.setattr(
        ACTION, "_sync_terminal_log",
        lambda _args: pytest.fail("unstarted cancellation must not fetch absent logs"),
    )
    monkeypatch.setattr(
        ACTION, "_sync_unstarted_cancel",
        lambda _args: {
            "log_path": "/shared/action.log", "diagnostic_outputs": [],
            "canceled_before_start": True,
        },
    )

    result = ACTION.status(_args(
        request=tmp_path / "action.json",
        remote_results=pathlib.Path("/lustre/run/results"),
        log_dir=pathlib.Path("/lustre/run/logs"),
    ))

    assert result["status"] == "CANCELED"
    assert result["synchronized"]["canceled_before_start"] is True


def test_unstarted_cancel_writes_bounded_local_evidence(tmp_path):
    results = tmp_path / "results"
    log_path = results / "pool_embed.log"
    results.mkdir()
    payload = {
        "workflow": "tao-run-deft-iaa", "platform": "slurm",
        "results_dir": str(results), "log_path": str(log_path),
    }
    payload["request_sha256"] = ACTION._canonical_sha256(payload)  # noqa: SLF001
    request = tmp_path / "action.json"
    request.write_text(json.dumps(payload), encoding="utf-8")

    result = ACTION._sync_unstarted_cancel(_args(request=request))  # noqa: SLF001

    assert result == {
        "log_path": str(log_path), "diagnostic_outputs": [],
        "canceled_before_start": True,
    }
    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "SLURM_JOB_ID=12345", "SLURM_JOB_NAME=iaa-job-1",
        "NATIVE_STATE=CANCELLED", "ELAPSED_RAW=0",
        "DETAIL=job canceled before execution; native stdout/stderr were not created",
    ]


def test_status_reconciles_bounded_accounting_lag(monkeypatch):
    accounting = iter((b"", b"COMPLETED\n"))
    clock = [0.0]

    def ssh(_login, command):
        if "TAO_JOB_NAME=" in command:
            return _completed([], stdout=b"TAO_JOB_NAME=iaa-job-1\n")
        if "squeue -h -j" in command:
            return _completed([], stdout=b"")
        if "sacct -j" in command:
            return _completed([], stdout=next(accounting))
        pytest.fail(f"unexpected command: {command}")

    monkeypatch.setattr(ACTION, "_ssh", ssh)
    monkeypatch.setattr(ACTION.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(ACTION.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    assert ACTION.status(_args())["status"] == "COMPLETE"
    assert clock[0] == ACTION.ACCOUNTING_RECONCILE_INTERVAL_SECONDS


def test_status_accounting_lookup_has_finite_unknown_exit(monkeypatch):
    clock = [0.0]

    def ssh(_login, command):
        if "TAO_JOB_NAME=" in command:
            return _completed([], stdout=b"TAO_JOB_NAME=iaa-job-1\n")
        return _completed([], stdout=b"")

    monkeypatch.setattr(ACTION, "_ssh", ssh)
    monkeypatch.setattr(ACTION.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(ACTION.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    result = ACTION.status(_args())
    assert result["native_state"] == "UNKNOWN"
    assert result["status"] == "UNKNOWN"
    assert clock[0] == ACTION.ACCOUNTING_RECONCILE_TIMEOUT_SECONDS


def test_logs_reads_only_two_derived_owned_paths_and_redacts(monkeypatch):
    calls = []

    def ssh(_login, command):
        calls.append(command)
        if "TAO_JOB_NAME=" in command:
            return _completed([], stdout=b"TAO_JOB_NAME=iaa-job-1\n")
        if command.startswith("tail -n"):
            return _completed([], stdout=b"token=super-secret-value\nworkload output\n")
        pytest.fail(f"unexpected command: {command}")

    monkeypatch.setattr(ACTION, "_ssh", ssh)
    result = ACTION.logs(_args(log_dir=pathlib.Path("/lustre/run/logs"), tail=37))

    assert result["log_paths"] == [
        "/lustre/run/logs/iaa-job-1-12345.out",
        "/lustre/run/logs/iaa-job-1-12345.err",
    ]
    assert "super-secret-value" not in result["text"]
    assert "[REDACTED]" in result["text"]
    assert calls[-1] == (
        "tail -n 37 -- /lustre/run/logs/iaa-job-1-12345.out "
        "/lustre/run/logs/iaa-job-1-12345.err 2>/dev/null || true"
    )


def test_cancel_requires_confirmation_before_transport(monkeypatch):
    monkeypatch.setattr(ACTION, "_ssh", lambda *_args: pytest.fail("transport called"))
    with pytest.raises(ValueError, match="requires --confirm"):
        ACTION.cancel(_args(confirm=False, timeout=10))


def test_cancel_proves_ownership_then_waits_for_terminal_accounting(monkeypatch):
    queue = iter((b"RUNNING\n", b""))
    clock = [0.0]
    commands = []

    def ssh(_login, command):
        commands.append(command)
        if "TAO_JOB_NAME=" in command:
            return _completed([], stdout=b"TAO_JOB_NAME=iaa-job-1\n")
        if "squeue -h -j" in command:
            return _completed([], stdout=next(queue))
        if command == "scancel 12345":
            return _completed([])
        if "sacct -j" in command:
            return _completed([], stdout=b"CANCELLED+\n")
        pytest.fail(f"unexpected command: {command}")

    monkeypatch.setattr(ACTION, "_ssh", ssh)
    monkeypatch.setattr(ACTION.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(ACTION.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    result = ACTION.cancel(_args(confirm=True, timeout=10))
    assert result["status"] == "CANCELED"
    assert result["native_state"] == "CANCELLED"
    assert result["already_terminal"] is False
    assert commands.count("scancel 12345") == 1


def test_cancel_treats_an_already_terminal_job_as_idempotent(monkeypatch):
    commands = []

    def ssh(_login, command):
        commands.append(command)
        if "TAO_JOB_NAME=" in command:
            return _completed([], stdout=b"TAO_JOB_NAME=iaa-job-1\n")
        if "squeue -h -j" in command:
            return _completed([], stdout=b"")
        if "sacct -j" in command:
            return _completed([], stdout=b"COMPLETED\n")
        pytest.fail(f"unexpected command: {command}")

    monkeypatch.setattr(ACTION, "_ssh", ssh)
    result = ACTION.cancel(_args(confirm=True, timeout=10))

    assert result["already_terminal"] is True
    assert result["status"] == "COMPLETE"
    assert not any(command.startswith("scancel ") for command in commands)


def test_ownership_mismatch_blocks_logs_and_cancel(monkeypatch):
    calls = []

    def ssh(_login, command):
        calls.append(command)
        return _completed([], stdout=b"TAO_JOB_NAME=somebody-elses-job\n")

    monkeypatch.setattr(ACTION, "_ssh", ssh)
    with pytest.raises(ValueError, match="not owned"):
        ACTION.logs(_args(log_dir=pathlib.Path("/lustre/run/logs"), tail=20))
    with pytest.raises(ValueError, match="not owned"):
        ACTION.cancel(_args(confirm=True, timeout=10))
    assert not any(command.startswith("tail ") or command.startswith("scancel ") for command in calls)


@pytest.mark.parametrize(
    ("native", "expected"),
    [
        ("PENDING", "PENDING"),
        ("CONFIGURING", "PENDING"),
        ("COMPLETING", "RUNNING"),
        ("COMPLETED", "COMPLETE"),
        ("OUT_OF_MEMORY", "ERROR"),
        ("PREEMPTED", "CANCELED"),
        ("FUTURE_STATE", "UNKNOWN"),
    ],
)
def test_native_state_mapping(native, expected):
    assert ACTION.map_state(native) == expected
