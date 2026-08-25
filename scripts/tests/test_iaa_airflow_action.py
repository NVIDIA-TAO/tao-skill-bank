# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "skills"
    / "applications"
    / "tao-run-deft-iaa"
    / "scripts"
    / "airflow_action.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("iaa_airflow_action", SCRIPT)
assert SPEC and SPEC.loader
AIRFLOW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AIRFLOW)


def _write_bound_action(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    request_path = tmp_path / "gap_analysis.action.json"
    binding_path = tmp_path / "gap_analysis.job-binding.json"
    job_path = tmp_path / "jobs" / "deft-iaa-gap-analysis-test.json"
    job_path.parent.mkdir()
    request = {
        "schema_version": "1",
        "workflow": AIRFLOW.WORKFLOW,
        "platform": "airflow",
        "record_image": "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-data-services",
        "spec_bundle": {
            "network_arch": "iaa-adapter",
            "action": "deft-iaa-gap-analysis-test-action",
        },
        "job_binding_path": str(binding_path),
        "environment": {"IAA_COMPUTE_FRAME": "airflow"},
        "forward_env": [],
        "mounts": [{"source": str(tmp_path), "target": "/results", "read_only": False}],
    }
    request["request_sha256"] = AIRFLOW._canonical_sha256(request, "request_sha256")
    request_path.write_text(json.dumps(request), encoding="utf-8")
    job = {
        "id": "deft-iaa-gap-analysis-test",
        "platform": "airflow",
        "image": request["record_image"],
        "network_arch": "iaa-adapter",
        "action": request["spec_bundle"]["action"],
        "results_dir": "/shared/runs/test/iter_1/gaps",
        "backend_ref": None,
        "terminal_state": None,
        "transitions": [{"state": "PENDING"}],
    }
    job_path.write_text(json.dumps(job), encoding="utf-8")
    binding = {
        "schema_version": "1",
        "workflow": AIRFLOW.WORKFLOW,
        "platform": "airflow",
        "request_path": str(request_path),
        "request_sha256": request["request_sha256"],
        "job_record_path": str(job_path),
        "job_id": job["id"],
        "job_identity_sha256": "1" * 64,
        "results_scope": job["results_dir"],
        "staging_receipt_sha256": "2" * 64,
        "bound_at": "2026-08-22T00:00:00+00:00",
    }
    binding["binding_sha256"] = AIRFLOW._canonical_sha256(binding, "binding_sha256")
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    return request_path, binding_path, job_path


def _stub_bound_producer(
    monkeypatch: pytest.MonkeyPatch,
    paths: tuple[pathlib.Path, pathlib.Path, pathlib.Path],
) -> tuple[dict, dict, dict]:
    payloads = tuple(json.loads(path.read_text()) for path in paths)
    request, binding, job = payloads
    monkeypatch.setattr(
        AIRFLOW,
        "load_bound_action_for_submit",
        lambda request_path, binding_path, job_path: (request, binding, job),
    )
    return request, binding, job


@pytest.fixture(autouse=True)
def _airflow_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIRFLOW_BASE_URL", "https://airflow.example.test")
    monkeypatch.setenv("AIRFLOW_API_TOKEN", "test-airflow-token")
    monkeypatch.delenv("AIRFLOW_USERNAME", raising=False)
    monkeypatch.delenv("AIRFLOW_PASSWORD", raising=False)
    monkeypatch.delenv("TAO_IAA_AIRFLOW_DAG_ID", raising=False)
    monkeypatch.setenv("TAO_IAA_AIRFLOW_SHARED_ROOT", "/airflow/shared")


def test_airflow_is_application_scoped() -> None:
    assert not (ROOT / "skills" / "platform" / "tao-run-on-airflow").exists()
    assert SCRIPT.is_file()


@pytest.mark.parametrize(
    ("native", "expected"),
    [
        (None, "PENDING"),
        ("queued", "PENDING"),
        ("up_for_retry", "PENDING"),
        ("running", "RUNNING"),
        ("success", "COMPLETE"),
        ("failed", "ERROR"),
        ("removed", "CANCELED"),
        ("future_state", "UNKNOWN"),
    ],
)
def test_native_state_mapping(native: str | None, expected: str) -> None:
    assert AIRFLOW.map_state(native) == expected


def test_remote_plain_http_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIRFLOW_BASE_URL", "http://airflow.example.test:8080")
    with pytest.raises(AIRFLOW.AirflowContractError, match="requires HTTPS"):
        AIRFLOW._base_url()


def test_preflight_requires_exact_unpaused_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        AIRFLOW.AirflowClient,
        "dag",
        lambda self, dag_id: {"dag_id": dag_id, "is_paused": False, "tags": []},
    )
    with pytest.raises(AIRFLOW.AirflowContractError, match="contract tag"):
        AIRFLOW.preflight(argparse.Namespace(pool=["iaa-cpu:1"]))


def test_preflight_validates_explicit_pool_capacity(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        AIRFLOW.AirflowClient,
        "dag",
        lambda self, dag_id: {
            "dag_id": dag_id,
            "is_paused": False,
            "tags": [AIRFLOW.CONTRACT],
        },
    )
    pools = {
        "iaa-cpu": {"name": "iaa-cpu", "slots": 4, "open_slots": 3},
        "iaa-images": {"name": "iaa-images", "slots": 3, "open_slots": 0},
    }
    monkeypatch.setattr(
        AIRFLOW.AirflowClient,
        "pool",
        lambda self, name: pools[name],
    )
    assert AIRFLOW.preflight(argparse.Namespace(
        pool=["iaa-cpu:1", "iaa-images:3"],
    )) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["pools"][1] == {
        "name": "iaa-images",
        "minimum_slots": 3,
        "total_slots": 3,
        "open_slots": 0,
    }


def test_preflight_rejects_undersized_or_duplicate_pools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        AIRFLOW.AirflowClient,
        "pool",
        lambda self, name: {"name": name, "slots": 1, "open_slots": 1},
    )
    client = AIRFLOW.AirflowClient()
    with pytest.raises(AIRFLOW.AirflowContractError, match="at least 3 required"):
        AIRFLOW.validate_pools(client, ["iaa-images:3"])
    with pytest.raises(AIRFLOW.AirflowContractError, match="duplicate"):
        AIRFLOW.validate_pools(client, ["iaa-cpu:1", "iaa-cpu:1"])


def test_submit_uses_bound_job_id_and_never_places_token_in_conf(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request, binding, job = _write_bound_action(tmp_path)
    _, _, job_payload = _stub_bound_producer(monkeypatch, (request, binding, job))
    calls: list[tuple[str, str, object]] = []

    def fake_request(self, method, path, payload=None, **kwargs):
        calls.append((method, path, payload))
        if method == "GET":
            return {
                "dag_id": AIRFLOW.DEFAULT_DAG_ID,
                "is_paused": False,
                "tags": [{"name": AIRFLOW.CONTRACT}],
            }
        assert method == "POST"
        return {"state": "queued"}

    monkeypatch.setattr(AIRFLOW.AirflowClient, "_request", fake_request)
    assert AIRFLOW.submit(argparse.Namespace(
        request=request, job_binding=binding, job_record=job,
        mount=["/airflow/shared/run-1:/results:rw"],
    )) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["backend_ref"] == (
        f"{AIRFLOW.DEFAULT_DAG_ID}/deft-iaa-gap-analysis-test"
    )
    post = [call for call in calls if call[0] == "POST"]
    assert len(post) == 1
    serialized = json.dumps(post[0][2], sort_keys=True)
    assert "test-airflow-token" not in serialized
    assert post[0][2]["dag_run_id"] == "deft-iaa-gap-analysis-test"
    assert post[0][2]["logical_date"] is None
    assert post[0][2]["conf"]["job_identity"]["id"] == job_payload["id"]


def test_submit_reconciles_only_an_identical_conflict(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request_path, binding_path, job_path = _write_bound_action(tmp_path)
    request, binding, job = _stub_bound_producer(
        monkeypatch, (request_path, binding_path, job_path)
    )
    job_identity = {
        field: job.get(field) for field in AIRFLOW.JOB_IDENTITY_FIELDS
    }

    def fake_request(self, method, path, payload=None, **kwargs):
        if method == "GET" and path.endswith(AIRFLOW.DEFAULT_DAG_ID):
            return {
                "dag_id": AIRFLOW.DEFAULT_DAG_ID,
                "is_paused": False,
                "tags": [AIRFLOW.CONTRACT],
            }
        if method == "POST":
            raise AIRFLOW.AirflowApiError("conflict", 409)
        return {
            "state": "running",
            "conf": {
                "contract": AIRFLOW.CONTRACT,
                "job_id": "deft-iaa-gap-analysis-test",
                    "request_sha256": request["request_sha256"],
                    "binding_sha256": binding["binding_sha256"],
                    "job_identity": job_identity,
                },
        }

    monkeypatch.setattr(AIRFLOW.AirflowClient, "_request", fake_request)
    assert AIRFLOW.submit(argparse.Namespace(
        request=request_path, job_binding=binding_path, job_record=job_path,
        mount=["/airflow/shared/run-1:/results:rw"],
    )) == 0
    assert json.loads(capsys.readouterr().out)["reconciled"] is True


def test_submit_rejects_conflict_with_different_request(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, binding_path, job_path = _write_bound_action(tmp_path)
    _stub_bound_producer(monkeypatch, (request_path, binding_path, job_path))

    def fake_request(self, method, path, payload=None, **kwargs):
        if method == "GET" and path.endswith(AIRFLOW.DEFAULT_DAG_ID):
            return {
                "dag_id": AIRFLOW.DEFAULT_DAG_ID,
                "is_paused": False,
                "tags": [AIRFLOW.CONTRACT],
            }
        if method == "POST":
            raise AIRFLOW.AirflowApiError("conflict", 409)
        return {"state": "running", "conf": {"request_sha256": "0" * 64}}

    monkeypatch.setattr(AIRFLOW.AirflowClient, "_request", fake_request)
    with pytest.raises(AIRFLOW.AirflowContractError, match="does not match"):
        AIRFLOW.submit(argparse.Namespace(
            request=request_path, job_binding=binding_path, job_record=job_path,
            mount=["/airflow/shared/run-1:/results:rw"],
        ))


def test_request_containing_process_credential_is_rejected(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, binding_path, job_path = _write_bound_action(tmp_path)
    request = json.loads(request_path.read_text())
    request["environment"]["BAD"] = "literal-secret"
    request["request_sha256"] = AIRFLOW._canonical_sha256(request, "request_sha256")
    request_path.write_text(json.dumps(request), encoding="utf-8")
    binding = json.loads(binding_path.read_text())
    binding["request_sha256"] = request["request_sha256"]
    binding["binding_sha256"] = AIRFLOW._canonical_sha256(binding, "binding_sha256")
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "literal-secret")
    _stub_bound_producer(monkeypatch, (request_path, binding_path, job_path))
    with pytest.raises(AIRFLOW.AirflowContractError, match="credential value"):
        AIRFLOW._load_bound_action(request_path, binding_path, job_path)


def test_cancel_requires_confirmation() -> None:
    with pytest.raises(AIRFLOW.AirflowContractError, match="--confirm"):
        AIRFLOW.cancel(argparse.Namespace(
            backend_ref=f"{AIRFLOW.DEFAULT_DAG_ID}/run-1", confirm=False,
        ))


def test_logs_uses_airflow_v2_pagination(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths: list[str] = []

    def fake_request(self, method, path, payload=None, **kwargs):
        paths.append(path)
        return {"task_instances": []}

    monkeypatch.setattr(AIRFLOW.AirflowClient, "_request", fake_request)
    backend_ref = f"{AIRFLOW.DEFAULT_DAG_ID}/run-1"
    assert AIRFLOW.logs(argparse.Namespace(
        backend_ref=backend_ref, tail=17,
    )) == 0
    assert paths == [
        "/api/v2/dags/tao_deft_iaa_action_v1/dagRuns/run-1/"
        "taskInstances?page_limit=17&page_offset=0"
    ]
    assert json.loads(capsys.readouterr().out)["task_instances"] == []


def test_mount_translation_preserves_target_mode_and_shared_root(
    tmp_path: pathlib.Path,
) -> None:
    request_path, _, _ = _write_bound_action(tmp_path)
    request = json.loads(request_path.read_text())
    assert AIRFLOW._resolved_mounts(
        request, ["/airflow/shared/actions/test:/results:rw"]
    )[0]["source"] == "/airflow/shared/actions/test"
    with pytest.raises(AIRFLOW.AirflowContractError, match="mode differs"):
        AIRFLOW._resolved_mounts(
            request, ["/airflow/shared/actions/test:/results:ro"]
        )
    with pytest.raises(AIRFLOW.AirflowContractError, match="child"):
        AIRFLOW._resolved_mounts(request, ["/tmp/test:/results:rw"])
