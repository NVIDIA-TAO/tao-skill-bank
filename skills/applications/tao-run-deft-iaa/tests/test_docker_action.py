from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest


SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("docker_action", SCRIPTS / "docker_action.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

DIGEST = "a" * 64
REF = f"docker/job-1/evaluate/{DIGEST}"


def _container(*, status: str = "running", exit_code: int = 0, digest: str = DIGEST):
    return {
        "Config": {"Labels": {
            "tao-job": "job-1", "tao-action": "evaluate",
            "tao-request-sha256": digest,
        }},
        "State": {"Status": status, "ExitCode": exit_code},
    }


def _completed(argv, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(argv, returncode, stdout, stderr)


def test_submit_preserves_renderer_gpu_selector_and_returns_backend_ref(monkeypatch, capsys):
    request = {
        "workflow": MODULE.WORKFLOW, "platform": "docker", "name": "evaluate",
        "request_sha256": DIGEST,
    }
    monkeypatch.setattr(MODULE, "_load_bound", lambda _args: (request, {}, {"id": "job-1"}))
    rendered = ["docker", "run", "--gpus", '"device=2,5"', "image"]
    monkeypatch.setattr(MODULE, "_renderer", lambda _request: lambda request, job: rendered)
    inspections = iter([None, _container()])
    monkeypatch.setattr(MODULE, "_inspect", lambda _name: next(inspections))
    calls = []
    monkeypatch.setattr(
        MODULE, "_run",
        lambda argv, timeout: calls.append((argv, timeout)) or _completed(argv, stdout="cid\n"),
    )

    assert MODULE.submit(argparse.Namespace()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "backend_ref": REF, "native_state": "running",
        "reconciled": False, "status": "RUNNING",
    }
    assert calls == [(rendered, 900)]
    assert rendered[rendered.index("--gpus") + 1] == '"device=2,5"'
    assert "--gpus all" not in " ".join(rendered)


def test_submit_reconciles_only_exact_existing_container(monkeypatch, capsys):
    request = {
        "workflow": MODULE.WORKFLOW, "platform": "docker", "name": "evaluate",
        "request_sha256": DIGEST,
    }
    monkeypatch.setattr(MODULE, "_load_bound", lambda _args: (request, {}, {"id": "job-1"}))
    monkeypatch.setattr(MODULE, "_inspect", lambda _name: _container(status="exited"))
    monkeypatch.setattr(
        MODULE, "_run", lambda *_args, **_kwargs: pytest.fail("reconcile launched Docker"),
    )

    MODULE.submit(argparse.Namespace())
    payload = json.loads(capsys.readouterr().out)
    assert payload["reconciled"] is True
    assert payload["status"] == "COMPLETE"


def test_ownership_mismatch_fails_closed(monkeypatch):
    monkeypatch.setattr(MODULE, "_inspect", lambda _name: _container(digest="b" * 64))
    with pytest.raises(MODULE.DockerActionError, match="exact IAA request/job"):
        MODULE._owned_container(REF)


@pytest.mark.parametrize(
    ("native", "exit_code", "expected"),
    [
        ("created", 0, "PENDING"), ("running", 0, "RUNNING"),
        ("exited", 0, "COMPLETE"), ("exited", 7, "ERROR"),
        ("mystery", 0, "UNKNOWN"),
    ],
)
def test_status_uses_fixed_vocabulary(monkeypatch, capsys, native, exit_code, expected):
    monkeypatch.setattr(
        MODULE, "_inspect", lambda _name: _container(status=native, exit_code=exit_code),
    )
    MODULE.status(argparse.Namespace(backend_ref=REF))
    assert json.loads(capsys.readouterr().out)["status"] == expected


def test_cancel_requires_confirmation_and_stops_only_owned_container(monkeypatch, capsys):
    with pytest.raises(MODULE.DockerActionError, match="--confirm"):
        MODULE.cancel(argparse.Namespace(backend_ref=REF, confirm=False))
    monkeypatch.setattr(MODULE, "_inspect", lambda _name: _container())
    calls = []
    monkeypatch.setattr(
        MODULE, "_run",
        lambda argv, timeout: calls.append((argv, timeout)) or _completed(argv),
    )
    MODULE.cancel(argparse.Namespace(backend_ref=REF, confirm=True))
    assert calls == [(["docker", "stop", "--time", "30", "job-1"], 60)]
    assert json.loads(capsys.readouterr().out)["status"] == "CANCELED"


def test_logs_are_bounded_and_redacted(monkeypatch, capsys):
    monkeypatch.setattr(MODULE, "_inspect", lambda _name: _container())
    monkeypatch.setenv("HF_TOKEN", "secret-value")
    calls = []
    monkeypatch.setattr(
        MODULE, "_run",
        lambda argv, timeout: calls.append((argv, timeout))
        or _completed(argv, stdout="token=secret-value\n"),
    )
    MODULE.logs(argparse.Namespace(backend_ref=REF, tail=25))
    assert calls == [(["docker", "logs", "--tail", "25", "job-1"], 60)]
    output = capsys.readouterr().out
    assert "secret-value" not in output
    assert "[REDACTED]" in output or "<redacted>" in output
