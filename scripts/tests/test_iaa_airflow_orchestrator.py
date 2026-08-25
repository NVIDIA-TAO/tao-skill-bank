"""Deterministic tests for Airflow-to-compute composition in IAA."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "skills" / "applications" / "tao-run-deft-iaa" / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "iaa_airflow_orchestrator", SCRIPTS / "airflow_orchestrator.py"
)
ORCH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ORCH)


def _write_json(path: pathlib.Path, payload: dict) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _consumer(path: pathlib.Path) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """#!/usr/bin/env python3
import json, pathlib, sys
verb = sys.argv[1]
if verb == 'submit':
    if '--output' in sys.argv:
        output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('complete\\n')
    print(json.dumps({'backend_ref': 'native-123', 'status': 'RUNNING'}))
elif verb == 'status':
    print(json.dumps({'status': 'COMPLETE'}))
elif verb == 'logs':
    print(json.dumps({'status': 'COMPLETE', 'message': 'bounded'}))
elif verb == 'cancel':
    print(json.dumps({'status': 'CANCELED'}))
else:
    raise SystemExit(2)
"""
    )
    return path


def _fixture(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str = "docker",
    kind: str = "action",
) -> tuple[pathlib.Path, dict, pathlib.Path]:
    shared = tmp_path / "shared"
    shared.mkdir()
    monkeypatch.setenv("TAO_IAA_AIRFLOW_SHARED_ROOT", str(shared))
    request = _write_json(
        shared / "request.json",
        {"schema_version": "1", "workflow": ORCH.WORKFLOW, "platform": platform},
    )
    job = {
        "schema_version": 1,
        "id": "job-123",
        "platform": platform,
        "image": "example.invalid/image:1",
        "network_arch": "clip",
        "action": "evaluate",
        "results_dir": str(shared / "results"),
        "storage_tier": "A",
        "upload_excludes": [],
        "submitted_at": "2026-08-23T00:00:00+00:00",
        "backend_ref": None,
        "terminal_state": None,
        "transitions": [{"state": "PENDING"}],
    }
    job_path = _write_json(shared / "job.json", job)
    binding_path = _write_json(shared / "binding.json", {"binding": "test"})
    consumer = _consumer(
        shared / "runtime" / ORCH.ALLOWED_CONSUMERS[(platform, kind)]
    )
    output = shared / "result.txt"
    commands = {
        verb: [
            sys.executable,
            str(consumer),
            verb,
            *(["--output", str(output)] if verb == "submit" else []),
            *(["--backend-ref", "{backend_ref}"] if verb != "submit" else []),
            *(["--confirm"] if verb == "cancel" else []),
        ]
        for verb in ("submit", "status", "logs", "cancel")
    }
    if platform == "brev" and kind == "action":
        commands["submit"].extend(["--json", "--reconcile"])
        commands["cancel"].append("--json")
    plan = _write_json(
        shared / "plan.json",
        {
            "commands": commands,
            "expected_outputs": [str(output)],
            "poll_interval_s": 1,
            "deadline_s": 30,
            "unknown_status_limit": 3,
            "retain_on_failure": True,
            "forward_env": [],
        },
    )
    envelope_path = shared / "orchestration.json"
    args = SimpleNamespace(
        compute_platform=platform,
        compute_kind=kind,
        compute_request=request,
        job_record=job_path,
        job_binding=binding_path,
        consumer_plan=plan,
        output=envelope_path,
    )
    assert ORCH.prepare(args) == 0
    return envelope_path, json.loads(envelope_path.read_text()), output


@pytest.mark.parametrize(
    ("platform", "kind"),
    [
        (platform, kind)
        for platform in ORCH.COMPUTE_PLATFORMS
        for kind in ORCH.COMPUTE_KINDS
    ],
)
def test_complete_platform_kind_matrix_is_admitted(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    kind: str,
) -> None:
    envelope_path, envelope, _ = _fixture(tmp_path, monkeypatch, platform, kind)

    assert envelope["orchestrator"] == "airflow"
    assert envelope["compute_platform"] == platform
    assert envelope["compute_kind"] == kind
    assert ORCH.load_envelope(envelope_path)[1] == envelope


def test_execute_is_resumable_and_does_not_resubmit_committed_work(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope_path, envelope, output = _fixture(tmp_path, monkeypatch)
    conf = ORCH._conf(envelope_path, envelope)

    first = ORCH.execute_conf(conf)
    second = ORCH.execute_conf(conf)

    assert output.read_text() == "complete\n"
    assert first["status"] == second["status"] == "COMPLETE"
    assert first["compute_backend_ref"] == "native-123"
    log = pathlib.Path(envelope["log_path"]).read_text()
    assert log.count("OPERATION=submit") == 1


def test_prepare_reuses_byte_identical_envelope(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    envelope_path, envelope, _ = _fixture(tmp_path, monkeypatch)
    capsys.readouterr()
    args = SimpleNamespace(
        compute_platform=envelope["compute_platform"],
        compute_kind=envelope["compute_kind"],
        compute_request=pathlib.Path(envelope["compute_request_path"]),
        job_record=pathlib.Path(envelope["job_record_path"]),
        job_binding=pathlib.Path(envelope["job_binding_path"]),
        consumer_plan=envelope_path.with_name("plan.json"),
        output=envelope_path,
    )

    assert ORCH.prepare(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "reused"
    assert json.loads(envelope_path.read_text()) == envelope


def test_changed_job_transitions_do_not_break_immutable_identity(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope_path, envelope, _ = _fixture(tmp_path, monkeypatch)
    job_path = pathlib.Path(envelope["job_record_path"])
    job = json.loads(job_path.read_text())
    job["backend_ref"] = "native-123"
    job["transitions"].append({"state": "RUNNING"})
    _write_json(job_path, job)

    assert ORCH.load_envelope(envelope_path)[1]["job_id"] == "job-123"


def test_changed_job_identity_fails_closed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope_path, envelope, _ = _fixture(tmp_path, monkeypatch)
    job_path = pathlib.Path(envelope["job_record_path"])
    job = json.loads(job_path.read_text())
    job["results_dir"] += "-different"
    _write_json(job_path, job)

    with pytest.raises(ORCH.OrchestrationError, match="job identity differs"):
        ORCH.load_envelope(envelope_path)


def test_gpu_all_scope_widening_is_rejected(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope_path, envelope, _ = _fixture(tmp_path, monkeypatch)
    envelope["commands"]["submit"].extend(["--gpus", "all"])
    envelope["envelope_sha256"] = ORCH._canonical_sha256(envelope, "envelope_sha256")
    _write_json(envelope_path, envelope)

    with pytest.raises(ORCH.OrchestrationError, match="widened"):
        ORCH.load_envelope(envelope_path)


def test_credential_value_is_rejected_from_plan(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "not-a-real-secret-value")
    envelope_path, envelope, _ = _fixture(tmp_path, monkeypatch)
    envelope["commands"]["submit"].append("not-a-real-secret-value")
    envelope["envelope_sha256"] = ORCH._canonical_sha256(envelope, "envelope_sha256")
    _write_json(envelope_path, envelope)

    with pytest.raises(ORCH.OrchestrationError, match="credential value"):
        ORCH.load_envelope(envelope_path)


def test_request_digest_drift_fails_before_dispatch(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope_path, envelope, _ = _fixture(tmp_path, monkeypatch)
    request = pathlib.Path(envelope["compute_request_path"])
    request.write_text(request.read_text() + " ")

    with pytest.raises(ORCH.OrchestrationError, match="request file digest differs"):
        ORCH.load_envelope(envelope_path)


def test_consumer_digest_drift_fails_before_dispatch(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope_path, envelope, _ = _fixture(tmp_path, monkeypatch)
    consumer = pathlib.Path(envelope["commands"]["submit"][1])
    consumer.write_text(consumer.read_text() + "\n# changed after approval\n")

    with pytest.raises(ORCH.OrchestrationError, match="consumer digest differs"):
        ORCH.load_envelope(envelope_path)


def test_schema_matches_runtime_required_fields() -> None:
    schema = json.loads(
        (
            ROOT
            / "skills/applications/tao-run-deft-iaa/references/airflow-orchestration-request.schema.json"
        ).read_text()
    )
    assert set(schema["required"]) == ORCH.EXPECTED_FIELDS


def test_every_matrix_cell_has_a_packaged_consumer() -> None:
    roots = {
        "docker_action.py": SCRIPTS / "docker_action.py",
        "local_sdg_action.py": SCRIPTS / "local_sdg_action.py",
        "slurm_action.py": ROOT / "skills/platform/tao-run-on-slurm/scripts/slurm_action.py",
        "slurm_sdg_action.py": ROOT / "skills/platform/tao-run-on-slurm/scripts/slurm_sdg_action.py",
        "kubernetes_action.py": ROOT / "skills/platform/tao-run-on-kubernetes/scripts/kubernetes_action.py",
        "kubernetes_sdg_action.py": ROOT / "skills/platform/tao-run-on-kubernetes/scripts/kubernetes_sdg_action.py",
        "brev_action.py": ROOT / "skills/platform/tao-run-on-brev/scripts/brev_action.py",
        "brev_sdg_action.py": ROOT / "skills/platform/tao-run-on-brev/scripts/brev_sdg_action.py",
        "virtualenv_runner.py": ROOT / "skills/platform/tao-run-on-virtualenv/references/virtualenv_runner.py",
    }
    assert set(ORCH.ALLOWED_CONSUMERS.values()) == set(roots)
    assert all(path.is_file() and path.stat().st_size > 0 for path in roots.values())
