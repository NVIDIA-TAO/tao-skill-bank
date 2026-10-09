# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for PAS's packaged Docker consumer and runtime probe."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
PAS_SCRIPTS = REPO / "skills/applications/tao-run-deft-pas/scripts"
sys.path.insert(0, str(PAS_SCRIPTS))
import run_deft_docker_action as consumer  # noqa: E402
import run_pas_runtime_probe as runtime_probe  # noqa: E402


@pytest.mark.parametrize("verb", ["submit", "status", "logs", "cancel"])
def test_documented_docker_verbs_use_workspace_interpreter(verb):
    reference = (
        REPO
        / "skills/applications/tao-run-deft-pas/references/platform-execution.md"
    ).read_text(encoding="utf-8")
    expected = (
        '"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \\\n'
        f'  "$SKILL_ROOT/scripts/run_deft_docker_action.py" {verb} '
        '--request "$ACTION_REQUEST"'
    )
    assert expected in reference
    system_python = f'python3 "$SKILL_ROOT/scripts/run_deft_docker_action.py" {verb}'
    assert system_python not in reference


def test_docker_consumer_rejects_dependency_incomplete_python_override(tmp_path):
    workspace_python = tmp_path / ".venv/bin/python"
    workspace_python.parent.mkdir(parents=True)
    workspace_python.symlink_to(sys.executable)
    incomplete = tmp_path / "python-without-jsonschema"
    incomplete.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = -c ]; then\n"
        "  case \"$2\" in *jsonschema*) exit 1 ;; *) exit 0 ;; esac\n"
        "fi\n"
        "printf '%s\\n' INCOMPLETE_INTERPRETER_SELECTED >&2\n"
        "exit 86\n",
        encoding="utf-8",
    )
    incomplete.chmod(0o755)
    completed = subprocess.run(
        [
            str(PAS_SCRIPTS / "deft_python.sh"),
            "--workspace",
            str(tmp_path),
            str(PAS_SCRIPTS / "run_deft_docker_action.py"),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "DEFT_PYTHON": str(incomplete)},
    )
    assert completed.returncode == 0, completed.stderr
    assert "INCOMPLETE_INTERPRETER_SELECTED" not in completed.stderr
    assert "{submit,status,logs,cancel}" in completed.stdout


def test_preapproval_runtime_probe_launcher_is_standard_library_only(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            str(PAS_SCRIPTS / "run_pas_runtime_probe.py"),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--approved" in completed.stdout


def test_forwarded_credentials_are_required_at_submit_point_of_use(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(ValueError, match="point of use: HF_TOKEN"):
        consumer._require_forwarded_environment({"forward_env": ["HF_TOKEN"]})  # noqa: SLF001


def test_docker_command_forwards_only_the_credential_name(tmp_path, monkeypatch):
    results = tmp_path / "workspace/results/run"
    results.mkdir(parents=True)
    (results / "deft_state.json").write_text(
        json.dumps(
            {
                "schema_version": "3",
                "workflow": "tao-run-deft-pas",
                "results_dir": str(results),
                "config": {"gpu_ids": [1, 3]},
            }
        )
    )
    monkeypatch.setenv("HF_TOKEN", "secret-value-must-not-enter-argv")
    monkeypatch.setattr(consumer.os, "getuid", lambda: 1000)
    monkeypatch.setattr(consumer.os, "getgid", lambda: 1000)
    monkeypatch.setattr(consumer.os, "getgroups", lambda: [1000, 44])
    request = {
        "results_dir": str(results),
        "mounts": [],
        "environment": {"HOME": "/tmp"},
        "forward_env": ["HF_TOKEN"],
        "workload_image": "registry.example/tao@sha256:" + "a" * 64,
        "spec_bundle": {
            "compute_shape": {"gpus": 2},
            "command": "clip",
            "args": ["train"],
        },
    }
    command = consumer._docker_command(request, "job-id")  # noqa: SLF001
    assert "HF_TOKEN" in command
    assert all("secret-value-must-not-enter-argv" not in item for item in command)
    assert '"device=1,3"' in command
    assert ["--shm-size=8g", "--user", "1000:1000"] == command[
        command.index("--shm-size=8g") : command.index("--shm-size=8g") + 3
    ]
    assert command[command.index("--group-add") + 1] == "44"
    assert "tao-job=job-id" in command
    assert "USER=tao" in command
    assert "LOGNAME=tao" in command


def test_writable_docker_launch_refuses_root(tmp_path, monkeypatch):
    results = tmp_path / "workspace/results/run"
    results.mkdir(parents=True)
    (results / "deft_state.json").write_text(
        json.dumps(
            {
                "schema_version": "3",
                "workflow": "tao-run-deft-pas",
                "results_dir": str(results),
                "config": {"gpu_ids": [0]},
            }
        )
    )
    monkeypatch.setattr(consumer.os, "getuid", lambda: 0)
    request = {
        "results_dir": str(results),
        "mounts": [],
        "environment": {},
        "forward_env": [],
        "workload_image": "registry.example/tao:test",
        "spec_bundle": {
            "compute_shape": {"gpus": 1},
            "command": "clip",
            "args": ["train"],
        },
    }
    with pytest.raises(ValueError, match="refusing.*UID 0"):
        consumer._docker_command(request, "job-id")  # noqa: SLF001


def test_submit_opens_and_binds_job_before_native_launch(tmp_path, monkeypatch):
    request_path = tmp_path / "action.json"
    job_path = tmp_path / "state/jobs/job.json"
    request = {
        "platform": "docker",
        "workload_image": "registry.example/tao:test",
        "forward_env": [],
    }
    events: list[str] = []
    monkeypatch.setattr(consumer, "_load", lambda _args: (request_path, request))
    monkeypatch.setattr(
        consumer.action,
        "reconcile_request",
        lambda _args: {"state": "NO_JOB_RECORD"},
    )
    monkeypatch.setattr(
        consumer,
        "_open_job",
        lambda _request: (events.append("open") or ("job", job_path)),
    )
    monkeypatch.setattr(
        consumer.action,
        "bind_job",
        lambda _args: events.append("bind"),
    )
    monkeypatch.setattr(
        consumer,
        "_docker_command",
        lambda _request, _job: ["docker", "run"],
    )
    monkeypatch.setattr(
        consumer,
        "_mark",
        lambda *_args, **_kwargs: events.append("mark-running"),
    )

    def fake_run(command, **_kwargs):
        if command[:3] == ["docker", "image", "inspect"]:
            events.append("inspect")
            return subprocess.CompletedProcess(command, 0, "[]", "")
        events.append("native-launch")
        return subprocess.CompletedProcess(command, 0, "container-id\n", "")

    monkeypatch.setattr(consumer, "_run", fake_run)
    result = consumer.submit(argparse.Namespace(request=request_path))
    assert result["state"] == "RUNNING"
    assert events == ["inspect", "open", "bind", "native-launch", "mark-running"]


def test_failed_native_submit_is_finalized_and_can_be_retried(tmp_path, monkeypatch):
    request_path = tmp_path / "action.json"
    job_path = tmp_path / "state/jobs/job.json"
    log_path = tmp_path / "submit.log"
    request = {
        "platform": "docker",
        "workload_image": "registry.example/tao:test",
        "forward_env": [],
        "log_path": str(log_path),
    }
    events: list[tuple[str, str | None]] = []
    monkeypatch.setattr(consumer, "_load", lambda _args: (request_path, request))
    monkeypatch.setattr(
        consumer.action,
        "reconcile_request",
        lambda _args: {"state": "NO_JOB_RECORD"},
    )
    monkeypatch.setattr(consumer, "_open_job", lambda _request: ("job", job_path))
    monkeypatch.setattr(consumer.action, "bind_job", lambda _args: None)
    monkeypatch.setattr(
        consumer,
        "_docker_command",
        lambda _request, _job: ["docker", "run"],
    )
    monkeypatch.setattr(
        consumer,
        "_mark",
        lambda _request, _job, state, **kwargs: events.append(
            (state, kwargs.get("backend_ref"))
        ),
    )
    monkeypatch.setattr(
        consumer.action,
        "finalize",
        lambda _args: events.append(("FINALIZED", None)),
    )

    def fake_run(command, **_kwargs):
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, "[]", "")
        return subprocess.CompletedProcess(command, 125, "", "daemon rejected launch")

    monkeypatch.setattr(consumer, "_run", fake_run)
    with pytest.raises(ValueError, match="daemon rejected launch"):
        consumer.submit(argparse.Namespace(request=request_path))
    assert events == [
        ("ERROR", "submit-error:docker:job"),
        ("FINALIZED", None),
    ]
    assert log_path.read_text().strip() == "daemon rejected launch"


def test_prelaunch_submit_error_has_an_honest_terminal_lineage():
    job = {
        "terminal_state": "ERROR",
        "backend_ref": "submit-error:docker:job",
        "transitions": [
            {
                "ts": "2026-10-07T00:00:00+00:00",
                "state": "PENDING",
            },
            {
                "ts": "2026-10-07T00:00:01+00:00",
                "state": "ERROR",
            },
        ],
    }
    consumer.action._validate_terminal_job(job)  # noqa: SLF001
    job["backend_ref"] = "docker:unproven"
    with pytest.raises(ValueError, match="PENDING, RUNNING"):
        consumer.action._validate_terminal_job(job)  # noqa: SLF001


def test_runtime_probe_uses_digest_image_and_exact_probe_mount(tmp_path):
    output = tmp_path / "attestations/pyt.json"
    output.parent.mkdir()
    args = argparse.Namespace(
        image="registry.example/tao-pyt:test",
        image_kind="pyt",
        finetuning_method="lora",
        min_gpus=1,
        gpu_ids="2",
        output=output,
    )
    digest_ref = "registry.example/tao-pyt@sha256:" + "a" * 64
    command = runtime_probe.build_command(args, digest_ref, "sha256:" + "a" * 64)
    probe_path = PAS_SCRIPTS / "check_pas_cuda_runtime.py"
    assert digest_ref in command
    assert f"{probe_path}:/probe/check_pas_cuda_runtime.py:ro" in command
    assert f"{output.parent}:/attestation:rw" in command
    assert "--require-clip-lora" in command
    assert "USER=tao" in command
    assert "LOGNAME=tao" in command


def test_failed_runtime_probe_cannot_leave_a_stale_passing_attestation(
    tmp_path, monkeypatch
):
    output = tmp_path / "attestation.json"
    output.write_text('{"status":"PASS"}\n')
    monkeypatch.setattr(
        runtime_probe,
        "resolve_local_digest",
        lambda _image: (
            "registry.example/tao-pyt@sha256:" + "a" * 64,
            "sha256:" + "a" * 64,
        ),
    )
    monkeypatch.setattr(
        runtime_probe,
        "build_command",
        lambda *_args: ["docker", "run"],
    )
    monkeypatch.setattr(
        runtime_probe.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, "", "failed"),
    )
    assert runtime_probe.main(
        [
            "--approved",
            "--image",
            "registry.example/tao-pyt:test",
            "--image-kind",
            "pyt",
            "--finetuning-method",
            "sft",
            "--min-gpus",
            "1",
            "--gpu-ids",
            "0",
            "--output",
            str(output),
        ]
    ) == 1
    assert not output.exists()
