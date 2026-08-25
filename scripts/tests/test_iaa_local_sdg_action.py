"""Focused lifecycle tests for Airflow-orchestrated local composite SDG."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "skills/applications/tao-run-deft-iaa/scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("iaa_local_sdg_action", SCRIPTS / "local_sdg_action.py")
LOCAL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(LOCAL)


def _fixture(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> tuple[pathlib.Path, dict]:
    stage = tmp_path / "results" / "iter_1" / "datagen"
    runtime = stage / ".tao-runtime" / "controller"
    runtime.mkdir(parents=True)
    config = tmp_path / "results" / "config" / "sdg_config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("schema_version: '1'\n")
    request_path = stage / "airflow_sdg.action.json"
    request = {
        "workflow": LOCAL.producer.WORKFLOW,
        "kind": LOCAL.producer.KIND,
        "platform": "docker",
        "orchestrator": "airflow",
        "request_sha256": "a" * 64,
        "run_id": "run-test",
        "paths": {
            "stage_dir": str(stage),
            "runtime_root": str(runtime),
            "config_path": str(config),
        },
    }
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(request))
    monkeypatch.setattr(LOCAL.producer, "load_request", lambda _: (request_path, request))
    return request_path, request


def test_submit_publishes_detached_owned_backend_ref(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    request_path, request = _fixture(tmp_path, monkeypatch)

    class Process:
        pid = 41

        def __init__(self, *args, **kwargs):
            LOCAL._write_status(request, state="RUNNING", backend_ref="pid:41:99")

        def poll(self):
            return None

    monkeypatch.setattr(LOCAL.subprocess, "Popen", Process)
    monkeypatch.setattr(LOCAL, "_owned_process", lambda pid, start: (pid, start) == (41, 99))

    assert LOCAL.submit(SimpleNamespace(request=request_path, job_record=tmp_path / "job.json")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"backend_ref": "pid:41:99", "reconciled": False, "status": "RUNNING"}


def test_submit_reconciles_matching_worker_without_duplicate_launch(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    request_path, request = _fixture(tmp_path, monkeypatch)
    LOCAL._write_status(request, state="RUNNING", backend_ref="pid:41:99")
    monkeypatch.setattr(LOCAL, "_owned_process", lambda pid, start: True)
    monkeypatch.setattr(
        LOCAL.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("matching owned worker must not be resubmitted"),
    )

    assert LOCAL.submit(SimpleNamespace(request=request_path, job_record=tmp_path / "job.json")) == 0
    assert json.loads(capsys.readouterr().out)["reconciled"] is True


def test_status_reports_unknown_when_running_pid_ownership_is_lost(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    request_path, request = _fixture(tmp_path, monkeypatch)
    LOCAL._write_status(request, state="RUNNING", backend_ref="pid:41:99")
    monkeypatch.setattr(LOCAL, "_owned_process", lambda pid, start: False)

    assert LOCAL.status(SimpleNamespace(request=request_path)) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "UNKNOWN"


def test_worker_commits_complete_only_after_runtime_success(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path, request = _fixture(tmp_path, monkeypatch)
    job_path = tmp_path / "job.json"
    job = {"id": "job-test"}
    monkeypatch.setattr(LOCAL.producer, "_load_job", lambda path, req: (job_path, job))
    monkeypatch.setattr(LOCAL.producer, "_bind_job", lambda *args: {"binding_sha256": "b" * 64})
    monkeypatch.setattr(LOCAL, "_proc_start", lambda pid: 99)
    called = []
    monkeypatch.setattr(LOCAL.runtime, "execute_sdg", lambda conf: called.append(conf) or {})

    assert LOCAL.worker(SimpleNamespace(request=request_path, job_record=job_path)) == 0
    evidence = LOCAL._load_status(request)
    assert evidence["state"] == "COMPLETE"
    assert called[0]["request"]["platform"] == "docker"


def test_cancel_stops_owned_group_then_owned_endpoints(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    request_path, request = _fixture(tmp_path, monkeypatch)
    LOCAL._write_status(request, state="RUNNING", backend_ref="pid:41:99")
    ownership = iter((True, False, False))
    monkeypatch.setattr(LOCAL, "_owned_process", lambda pid, start: next(ownership, False))
    killed = []
    monkeypatch.setattr(LOCAL.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    calls = []
    monkeypatch.setattr(
        LOCAL.subprocess,
        "run",
        lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(returncode=0, stdout="{}", stderr=""),
    )

    assert LOCAL.cancel(SimpleNamespace(request=request_path, confirm=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "CANCELED"
    assert killed and calls[0][1].endswith("manage_sdg_endpoints.py")
    assert "stop" in calls[0]


def test_cancel_requires_confirmation(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request_path, _ = _fixture(tmp_path, monkeypatch)
    with pytest.raises(LOCAL.LocalSdgError, match="requires --confirm"):
        LOCAL.cancel(SimpleNamespace(request=request_path, confirm=False))
