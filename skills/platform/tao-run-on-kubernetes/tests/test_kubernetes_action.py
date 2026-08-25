from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest
import yaml


SCRIPT_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location("kubernetes_action", SCRIPT_DIR / "kubernetes_action.py")
assert SPEC and SPEC.loader
ACTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACTION)


def _fixture(tmp_path: pathlib.Path, *, forward: bool = False):
    source = tmp_path / "results"
    source.mkdir()
    request = {
        "schema_version": "1", "workflow": "test-workflow", "platform": "kubernetes",
        "name": "train", "gpu_ids": [2, 4],
        "workload_image": "registry.example/tao:1@sha256:" + "a" * 64,
        "spec_bundle": {
            "mode": "args", "command": "python3", "args": ["train.py", "--epochs", "1"],
            "image": "registry.example/tao:1@sha256:" + "a" * 64,
            "compute_shape": {"gpus": 2, "nodes": 1},
        },
        "mounts": [{"source": str(source), "target": "/results", "read_only": False}],
        "fresh_outputs": [str(source / "model.pth")],
        "environment": {"HOME": "/tmp"},
        "forward_env": ["HF_TOKEN"] if forward else [],
    }
    request["request_sha256"] = ACTION._canonical_sha256(request)
    request_path = tmp_path / "action.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    staging_path = tmp_path / "staging.json"
    staging_path.write_text(json.dumps({
        "schema_version": "1",
        "sources": [{"source": str(source), "sub_path": "jobs/job-1/results"}],
    }), encoding="utf-8")
    args = argparse.Namespace(
        request=request_path, staging_map=staging_path, job_id="Job_Record_1",
        namespace="tao", pvc_claim="tao-results",
        credential_secret="job-record-1-creds" if forward else None,
        image_pull_secret=None, ttl_seconds=3600, shm_size="16Gi",
        request_timeout_s=30, tail=200, confirm=False,
    )
    return request, args


def _owned_job(request: dict, args: argparse.Namespace, *, status=None, uid="uid-1") -> dict:
    name = ACTION.renderer.kubernetes_job_name(args.job_id)
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {
            "name": name, "namespace": args.namespace, "uid": uid,
            "annotations": {
                ACTION.JOB_ID_ANNOTATION: args.job_id,
                ACTION.REQUEST_ANNOTATION: request["request_sha256"],
                ACTION.MANAGED_ANNOTATION: ACTION.MANAGED_BY,
            },
        },
        "spec": {"template": {"spec": {
            "restartPolicy": "Never", "imagePullSecrets": [], "volumes": [],
            "containers": [],
        }}},
        "status": status or {},
    }


def test_load_request_rejects_digest_drift_and_credential_value(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, args = _fixture(tmp_path, forward=True)
    monkeypatch.setenv("HF_TOKEN", "credential-value-123")
    assert ACTION.load_request(args.request)["request_sha256"] == request["request_sha256"]

    request["environment"]["BAD"] = "credential-value-123"
    request["request_sha256"] = ACTION._canonical_sha256(request)
    args.request.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ACTION.ContractError, match="credential value"):
        ACTION.load_request(args.request)

    request["environment"].pop("BAD")
    args.request.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ACTION.ContractError, match="signed Kubernetes"):
        ACTION.load_request(args.request)


def test_submit_preserves_gpu_count_and_projects_only_secret_name(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, args = _fixture(tmp_path, forward=True)
    monkeypatch.setenv("HF_TOKEN", "credential-value-123")
    calls: list[tuple[list[str], str | None]] = []

    def fake_json(argv, *, timeout_s, stdin=None, check=True):
        calls.append((argv, stdin))
        if argv[:3] == ["get", "nodes", "-o"]:
            return {"items": [{"status": {"allocatable": {"nvidia.com/gpu": "8"}}}]}
        manifest = yaml.safe_load(stdin)
        manifest["metadata"]["uid"] = "uid-applied"
        return manifest

    def fake_run(argv, *, timeout_s, stdin=None, check=True):
        calls.append((argv, stdin))
        manifest = yaml.safe_load(stdin)
        manifest["metadata"]["uid"] = "uid-applied"
        return subprocess.CompletedProcess(argv, 0, json.dumps(manifest), "")

    monkeypatch.setattr(ACTION, "_kubectl_json", fake_json)
    monkeypatch.setattr(ACTION, "_get_job", lambda identity, timeout: None)
    monkeypatch.setattr(ACTION, "_run", fake_run)
    result = ACTION.submit(args)
    assert result["state"] == "RUNNING"
    assert result["resumed"] is False
    manifests = [yaml.safe_load(body) for argv, body in calls if body]
    assert len(manifests) == 2
    container = manifests[-1]["spec"]["template"]["spec"]["containers"][0]
    assert container["resources"] == {"limits": {"nvidia.com/gpu": "2"}}
    assert container["env"][-1] == {
        "name": "HF_TOKEN",
        "valueFrom": {"secretKeyRef": {"name": "job-record-1-creds", "key": "HF_TOKEN"}},
    }
    serialized = json.dumps(manifests)
    assert "credential-value-123" not in serialized
    assert "--gpus all" not in serialized


def test_submit_create_race_never_mutates_foreign_job(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, args = _fixture(tmp_path)
    manifest_text, _ = ACTION._render_manifest(args, request)
    desired = yaml.safe_load(manifest_text)
    foreign = json.loads(json.dumps(desired))
    foreign["metadata"]["annotations"][ACTION.REQUEST_ANNOTATION] = "f" * 64
    responses = iter([None, foreign])
    monkeypatch.setattr(ACTION, "_gpu_preflight", lambda request, timeout: None)
    monkeypatch.setattr(ACTION, "_kubectl_json", lambda *a, **k: desired)
    monkeypatch.setattr(ACTION, "_get_job", lambda identity, timeout: next(responses))
    calls = []
    monkeypatch.setattr(
        ACTION, "_run",
        lambda argv, **kwargs: calls.append(argv) or subprocess.CompletedProcess(argv, 1, "", "AlreadyExists"),
    )
    with pytest.raises(ACTION.ContractError, match="foreign or differently bound"):
        ACTION.submit(args)
    assert len(calls) == 1
    assert calls[0][:3] == ["kubectl", "create", "-f"]


def test_submit_reconciles_only_exact_owned_workload(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, args = _fixture(tmp_path)
    manifest_text, identity = ACTION._render_manifest(args, request)
    desired = yaml.safe_load(manifest_text)
    desired["metadata"]["uid"] = "uid-existing"
    monkeypatch.setattr(ACTION, "_gpu_preflight", lambda request, timeout: None)
    monkeypatch.setattr(ACTION, "_kubectl_json", lambda *a, **k: desired)
    monkeypatch.setattr(ACTION, "_get_job", lambda identity, timeout: desired)
    result = ACTION.submit(args)
    assert result == {
        "state": "PENDING", "backend_ref": f"tao/{identity['name']}",
        "resumed": True, "uid": "uid-existing",
    }

    foreign = json.loads(json.dumps(desired))
    foreign["metadata"]["annotations"][ACTION.REQUEST_ANNOTATION] = "f" * 64
    monkeypatch.setattr(ACTION, "_get_job", lambda identity, timeout: foreign)
    with pytest.raises(ACTION.ContractError, match="foreign or differently bound"):
        ACTION.submit(args)


@pytest.mark.parametrize(
    ("native", "expected"),
    [
        ({}, "PENDING"),
        ({"active": 1}, "RUNNING"),
        ({"conditions": [{"type": "Complete", "status": "True"}]}, "COMPLETE"),
        ({"conditions": [{"type": "Failed", "status": "True"}]}, "ERROR"),
    ],
)
def test_status_maps_only_exact_owned_job(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, native: dict, expected: str,
) -> None:
    request, args = _fixture(tmp_path)
    monkeypatch.setattr(ACTION, "_get_job", lambda identity, timeout: _owned_job(request, args, status=native))
    assert ACTION.status(args)["state"] == expected


def test_logs_are_bounded_and_redacted(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, args = _fixture(tmp_path, forward=True)
    args.tail = 17
    monkeypatch.setenv("HF_TOKEN", "credential-value-123")
    monkeypatch.setattr(ACTION, "_get_job", lambda identity, timeout: _owned_job(request, args))
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "token=credential-value-123\n", "")

    monkeypatch.setattr(ACTION, "_run", fake_run)
    assert "credential-value-123" not in ACTION.logs(args)
    assert calls[0][-2:] == ["--tail", "17"]


def test_cancel_requires_confirmation_and_proves_absence(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, args = _fixture(tmp_path)
    with pytest.raises(ACTION.ContractError, match="--confirm"):
        ACTION.cancel(args)
    args.confirm = True
    responses = iter([_owned_job(request, args), None])
    monkeypatch.setattr(ACTION, "_get_job", lambda identity, timeout: next(responses))
    calls = []
    monkeypatch.setattr(
        ACTION, "_run",
        lambda argv, **kwargs: calls.append(argv) or subprocess.CompletedProcess(argv, 0, "", ""),
    )
    assert ACTION.cancel(args) == {"state": "CANCELED", "deleted_uid": "uid-1"}
    assert calls[0][:3] == ["kubectl", "delete", "job"]
    assert "--cascade=foreground" in calls[0]


def test_cancel_absent_is_unknown_not_false_success(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, args = _fixture(tmp_path)
    args.confirm = True
    monkeypatch.setattr(ACTION, "_get_job", lambda identity, timeout: None)
    assert ACTION.cancel(args)["state"] == "UNKNOWN"
