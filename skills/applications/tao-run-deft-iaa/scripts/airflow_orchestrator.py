#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose the IAA Airflow DAG with one existing compute-platform consumer.

The inner request and job record remain native to the selected compute
platform.  Airflow receives only a digest-bound orchestration envelope and
executes the selected consumer's four verbs without reconstructing a shell
command or duplicating DEFT state.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
from typing import Any, Iterable, Sequence

try:
    import airflow_action as airflow
except ModuleNotFoundError:  # The staged DAG worker needs only execute_conf().
    airflow = None


WORKFLOW = "tao-run-deft-iaa"
KIND = "airflow_compute_orchestration"
CONTRACT = "tao-deft-iaa-airflow-orchestration-v1"
COMPUTE_PLATFORMS = ("docker", "slurm", "kubernetes", "brev", "virtualenv")
COMPUTE_KINDS = ("action", "sdg")
TERMINAL = frozenset({"COMPLETE", "ERROR", "CANCELED"})
SHA256 = re.compile(r"[0-9a-f]{64}")
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.~-]{0,249}")
SAFE_ENV = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
ALLOWED_ENV = frozenset({
    "HF_TOKEN", "NGC_KEY", "BREV_API_TOKEN", "KUBECONFIG",
    "SLURM_JWT", "SSH_AUTH_SOCK",
})
SECRET_ENV_NAMES = (
    "AIRFLOW_API_TOKEN", "AIRFLOW_PASSWORD", "BREV_API_TOKEN", "HF_TOKEN", "NGC_KEY",
)
JOB_IDENTITY_FIELDS = (
    "schema_version", "id", "platform", "image", "network_arch", "action",
    "results_dir", "storage_tier", "upload_excludes", "submitted_at",
)
ALLOWED_CONSUMERS = {
    ("docker", "action"): "docker_action.py",
    ("docker", "sdg"): "local_sdg_action.py",
    ("slurm", "action"): "slurm_action.py",
    ("slurm", "sdg"): "slurm_sdg_action.py",
    ("kubernetes", "action"): "kubernetes_action.py",
    ("kubernetes", "sdg"): "kubernetes_sdg_action.py",
    ("brev", "action"): "brev_action.py",
    ("brev", "sdg"): "brev_sdg_action.py",
    ("virtualenv", "action"): "virtualenv_runner.py",
    ("virtualenv", "sdg"): "local_sdg_action.py",
}
EXPECTED_FIELDS = {
    "schema_version", "workflow", "kind", "contract", "orchestrator",
    "compute_platform", "compute_kind", "orchestration_id", "job_id",
    "compute_request_path", "compute_request_sha256", "job_record_path",
    "job_identity_sha256", "job_binding_path", "job_binding_sha256",
    "commands", "consumer_sha256", "expected_outputs", "poll_interval_s", "deadline_s",
    "unknown_status_limit", "retain_on_failure", "forward_env",
    "receipt_path", "log_path", "created_at", "envelope_sha256",
}


class OrchestrationError(ValueError):
    """One immutable orchestration input or backend result is invalid."""


def _canonical_sha256(payload: dict[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _job_identity_sha256(job: dict[str, Any]) -> str:
    identity = {field: job.get(field) for field in JOB_IDENTITY_FIELDS}
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _secret_values() -> tuple[str, ...]:
    names = set(SECRET_ENV_NAMES) | set(ALLOWED_ENV)
    return tuple(value for name in names if (value := os.environ.get(name)))


def _reject_secret_material(payload: Any, label: str) -> None:
    secrets = _secret_values()
    if any(secret in text for text in _walk_strings(payload) for secret in secrets):
        raise OrchestrationError(f"{label} contains a credential value")


def _redact(text: str) -> str:
    for value in _secret_values():
        text = text.replace(value, "[REDACTED]")
    return re.sub(
        r"(?i)(api[_-]?key|token|password)\s*[=:]\s*\S+",
        r"\1=<redacted>",
        text,
    )


def _shared_path(value: Any, field: str, *, exists: bool = False) -> pathlib.Path:
    if not isinstance(value, str) or not value.startswith("/") or "\x00" in value:
        raise OrchestrationError(f"{field} must be an absolute path")
    path = pathlib.Path(value)
    if path == pathlib.Path("/") or pathlib.Path(os.path.abspath(path)) != path:
        raise OrchestrationError(f"{field} must be normalized and non-root")
    raw_root = os.environ.get("TAO_IAA_AIRFLOW_SHARED_ROOT", "")
    if not raw_root.startswith("/"):
        raise OrchestrationError("TAO_IAA_AIRFLOW_SHARED_ROOT must be absolute")
    root = pathlib.Path(raw_root)
    if root == pathlib.Path("/") or pathlib.Path(os.path.abspath(root)) != root:
        raise OrchestrationError("Airflow shared root must be normalized and non-root")
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise OrchestrationError(f"{field} must be under Airflow shared root {root}") from exc
    if path.resolve(strict=False) != path:
        raise OrchestrationError(f"{field} must not traverse a symlink")
    if exists and (not path.is_file() or path.is_symlink() or path.stat().st_size == 0):
        raise OrchestrationError(f"{field} must be a non-empty regular file")
    return path


def _load_json(path: pathlib.Path, field: str) -> dict[str, Any]:
    _shared_path(str(path), field, exists=True)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OrchestrationError(f"{field} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise OrchestrationError(f"{field} root must be an object")
    return payload


def _validate_command(
    command: Any, *, platform: str, kind: str, verb: str
) -> list[str]:
    if (
        not isinstance(command, list)
        or len(command) < 3
        or any(not isinstance(item, str) or not item or "\x00" in item for item in command)
    ):
        raise OrchestrationError(f"commands.{verb} must be a non-empty argv array")
    if pathlib.Path(command[0]).name not in {"python", "python3", "python3.12"}:
        raise OrchestrationError(f"commands.{verb} must invoke Python directly")
    consumer = _shared_path(command[1], f"commands.{verb} consumer", exists=True)
    expected = ALLOWED_CONSUMERS[(platform, kind)]
    if consumer.name != expected:
        raise OrchestrationError(
            f"commands.{verb} must use {expected} for {platform}/{kind}"
        )
    expected_verb = "cancel" if verb == "cancel" else verb
    if command[2] != expected_verb:
        raise OrchestrationError(f"commands.{verb} does not invoke its matching verb")
    if verb == "cancel" and "--confirm" not in command:
        raise OrchestrationError("commands.cancel must preserve explicit owned-work confirmation")
    if platform == "brev" and kind == "action":
        if verb == "submit" and not {"--json", "--reconcile"}.issubset(command):
            raise OrchestrationError(
                "Airflow-orchestrated Brev submit requires --json --reconcile"
            )
        if verb == "cancel" and "--json" not in command:
            raise OrchestrationError("Airflow-orchestrated Brev cancel requires --json")
    lowered = [item.lower() for item in command]
    if any(
        lowered[index] == "--gpus" and lowered[index + 1] == "all"
        for index in range(len(lowered) - 1)
    ):
        raise OrchestrationError("explicit GPU selection was widened to --gpus all")
    sensitive_flags = {"--token", "--password", "--api-key", "--api_key", "-p"}
    if any(item in sensitive_flags for item in lowered):
        raise OrchestrationError(f"commands.{verb} carries a credential flag")
    _reject_secret_material(command, f"commands.{verb}")
    return list(command)


def validate_envelope(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != EXPECTED_FIELDS:
        raise OrchestrationError("Airflow orchestration envelope has missing or unexpected fields")
    fixed = {
        "schema_version": "1", "workflow": WORKFLOW, "kind": KIND,
        "contract": CONTRACT, "orchestrator": "airflow",
    }
    for field, expected in fixed.items():
        if payload.get(field) != expected:
            raise OrchestrationError(f"envelope.{field} must be {expected!r}")
    platform = payload.get("compute_platform")
    kind = payload.get("compute_kind")
    if platform not in COMPUTE_PLATFORMS or kind not in COMPUTE_KINDS:
        raise OrchestrationError("compute platform or kind is unsupported")
    if not isinstance(payload.get("envelope_sha256"), str) or not SHA256.fullmatch(
        payload["envelope_sha256"]
    ) or payload["envelope_sha256"] != _canonical_sha256(payload, "envelope_sha256"):
        raise OrchestrationError("envelope digest is missing or invalid")
    for field in ("orchestration_id", "job_id"):
        if not isinstance(payload[field], str) or SAFE_NAME.fullmatch(payload[field]) is None:
            raise OrchestrationError(f"envelope.{field} is invalid")
    request_path = _shared_path(
        payload.get("compute_request_path"), "compute_request_path", exists=True
    )
    request = _load_json(request_path, "compute request")
    if request.get("workflow") != WORKFLOW or request.get("platform") != platform:
        raise OrchestrationError("compute request workflow/platform differs from the envelope")
    request_digest = payload.get("compute_request_sha256")
    if not isinstance(request_digest, str) or not SHA256.fullmatch(request_digest):
        raise OrchestrationError("compute_request_sha256 is invalid")
    if _file_sha256(request_path) != request_digest:
        raise OrchestrationError("compute request file digest differs from the envelope")
    job_path = _shared_path(payload.get("job_record_path"), "job_record_path", exists=True)
    job = _load_json(job_path, "compute job record")
    if job.get("id") != payload["job_id"] or job.get("platform") != platform:
        raise OrchestrationError("compute job identity/platform differs from the envelope")
    if _job_identity_sha256(job) != payload.get("job_identity_sha256"):
        raise OrchestrationError("compute job identity differs from the envelope")
    binding_path_value = payload.get("job_binding_path")
    binding_digest = payload.get("job_binding_sha256")
    if kind == "action" and binding_path_value is None:
        raise OrchestrationError("ordinary compute actions require an immutable job binding")
    if binding_path_value is None:
        if binding_digest is not None:
            raise OrchestrationError("job binding path/digest must both be null or present")
    else:
        binding_path = _shared_path(binding_path_value, "job_binding_path", exists=True)
        if _file_sha256(binding_path) != binding_digest:
            raise OrchestrationError("job binding digest differs from the envelope")
    commands = payload.get("commands")
    if not isinstance(commands, dict) or set(commands) != {"submit", "status", "logs", "cancel"}:
        raise OrchestrationError("commands must define exactly submit/status/logs/cancel")
    for verb in commands:
        commands[verb] = _validate_command(
            commands[verb], platform=platform, kind=kind, verb=verb
        )
    consumers = {command[1] for command in commands.values()}
    if len(consumers) != 1:
        raise OrchestrationError("all four verbs must use the same staged consumer")
    consumer_digest = payload.get("consumer_sha256")
    if not isinstance(consumer_digest, str) or not SHA256.fullmatch(consumer_digest):
        raise OrchestrationError("consumer_sha256 is invalid")
    if _file_sha256(pathlib.Path(next(iter(consumers)))) != consumer_digest:
        raise OrchestrationError("staged consumer digest differs from the envelope")
    outputs = payload.get("expected_outputs")
    if not isinstance(outputs, list) or any(not isinstance(item, str) for item in outputs):
        raise OrchestrationError("expected_outputs must be an array of paths")
    for index, value in enumerate(outputs):
        _shared_path(value, f"expected_outputs[{index}]")
    for field, minimum, maximum in (
        ("poll_interval_s", 1, 3600), ("deadline_s", 30, 604800),
        ("unknown_status_limit", 1, 20),
    ):
        value = payload.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
            raise OrchestrationError(f"{field} must be in [{minimum}, {maximum}]")
    if payload["poll_interval_s"] >= payload["deadline_s"]:
        raise OrchestrationError("poll_interval_s must be less than deadline_s")
    if not isinstance(payload.get("retain_on_failure"), bool):
        raise OrchestrationError("retain_on_failure must be boolean")
    forward = payload.get("forward_env")
    if (
        not isinstance(forward, list) or len(forward) != len(set(forward))
        or any(not isinstance(name, str) or SAFE_ENV.fullmatch(name) is None or name not in ALLOWED_ENV for name in forward)
    ):
        raise OrchestrationError("forward_env contains an unsupported variable name")
    _shared_path(payload.get("receipt_path"), "receipt_path")
    _shared_path(payload.get("log_path"), "log_path")
    try:
        timestamp = dt.datetime.fromisoformat(payload["created_at"].replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise OrchestrationError("created_at must be ISO-8601") from exc
    if timestamp.utcoffset() is None:
        raise OrchestrationError("created_at must include a timezone")
    _reject_secret_material(payload, "orchestration envelope")
    return payload


def load_envelope(path: pathlib.Path) -> tuple[pathlib.Path, dict[str, Any]]:
    lexical = _shared_path(str(path), "orchestration envelope", exists=True)
    return lexical, validate_envelope(_load_json(lexical, "orchestration envelope"))


def _plan(path: pathlib.Path) -> dict[str, Any]:
    return _load_json(_shared_path(str(path), "consumer plan", exists=True), "consumer plan")


def prepare(args: argparse.Namespace) -> int:
    request_path = _shared_path(str(args.compute_request), "--compute-request", exists=True)
    request = _load_json(request_path, "compute request")
    platform = args.compute_platform
    if request.get("workflow") != WORKFLOW or request.get("platform") != platform:
        raise OrchestrationError("compute request does not match --compute-platform")
    job_path = _shared_path(str(args.job_record), "--job-record", exists=True)
    job = _load_json(job_path, "compute job record")
    if job.get("platform") != platform or not isinstance(job.get("id"), str):
        raise OrchestrationError("job record does not match the compute platform")
    plan = _plan(args.consumer_plan)
    if set(plan) != {
        "commands", "expected_outputs", "poll_interval_s", "deadline_s",
        "unknown_status_limit", "retain_on_failure", "forward_env",
    }:
        raise OrchestrationError("consumer plan has missing or unexpected fields")
    binding_path = None
    binding_digest = None
    if args.job_binding is not None:
        binding = _shared_path(str(args.job_binding), "--job-binding", exists=True)
        binding_path = str(binding)
        binding_digest = _file_sha256(binding)
    identity = {
        "platform": platform, "kind": args.compute_kind,
        "request_sha256": _file_sha256(request_path), "job_id": job["id"],
    }
    orchestration_id = "iaa-airflow-" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    output = _shared_path(str(args.output), "--output")
    receipt = output.with_name(output.stem + ".receipt.json")
    log = output.with_name(output.stem + ".log")
    existing_payload = (
        _load_json(output, "existing orchestration envelope")
        if output.exists()
        else None
    )
    created_at = (
        existing_payload.get("created_at")
        if isinstance(existing_payload, dict)
        else dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    )
    payload = {
        "schema_version": "1", "workflow": WORKFLOW, "kind": KIND,
        "contract": CONTRACT, "orchestrator": "airflow",
        "compute_platform": platform, "compute_kind": args.compute_kind,
        "orchestration_id": orchestration_id, "job_id": job["id"],
        "compute_request_path": str(request_path),
        "compute_request_sha256": _file_sha256(request_path),
        "job_record_path": str(job_path), "job_identity_sha256": _job_identity_sha256(job),
        "job_binding_path": binding_path, "job_binding_sha256": binding_digest,
        "commands": plan["commands"],
        "consumer_sha256": _file_sha256(
            _shared_path(
                plan["commands"]["submit"][1], "consumer plan submit script", exists=True
            )
        ),
        "expected_outputs": plan["expected_outputs"],
        "poll_interval_s": plan["poll_interval_s"], "deadline_s": plan["deadline_s"],
        "unknown_status_limit": plan["unknown_status_limit"],
        "retain_on_failure": plan["retain_on_failure"],
        "forward_env": plan["forward_env"], "receipt_path": str(receipt),
        "log_path": str(log),
        "created_at": created_at,
        "envelope_sha256": "0" * 64,
    }
    payload["envelope_sha256"] = _canonical_sha256(payload, "envelope_sha256")
    payload = validate_envelope(payload)
    if existing_payload is not None:
        if existing_payload != payload:
            raise OrchestrationError("refusing to replace a different orchestration envelope")
        disposition = "reused"
    else:
        _atomic_json(output, payload)
        disposition = "created"
    print(json.dumps({"status": disposition, "envelope": str(output), "orchestration_id": orchestration_id}, sort_keys=True))
    return 0


def _conf(envelope_path: pathlib.Path, envelope: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract": CONTRACT, "kind": KIND,
        "orchestration_id": envelope["orchestration_id"],
        "envelope_path": str(envelope_path),
        "envelope_sha256": envelope["envelope_sha256"],
    }


def submit(args: argparse.Namespace) -> int:
    if airflow is None:
        raise OrchestrationError("Airflow client module is unavailable")
    path, envelope = load_envelope(args.envelope)
    dag_id = airflow._dag_id()
    client = airflow.AirflowClient()
    airflow.validate_dag(client, dag_id)
    conf = _conf(path, envelope)
    run_id = envelope["orchestration_id"]
    endpoint = f"/api/v2/dags/{urllib.parse.quote(dag_id, safe='')}/dagRuns"
    try:
        response = client._request("POST", endpoint, {"dag_run_id": run_id, "logical_date": None, "conf": conf})
        reconciled = False
    except airflow.AirflowApiError as exc:
        if exc.status != 409:
            raise
        response = client.dag_run(dag_id, run_id)
        if response.get("conf") != conf:
            raise OrchestrationError("existing DAG run differs from the immutable orchestration envelope")
        reconciled = True
    native = response.get("state") if isinstance(response, dict) else None
    print(json.dumps({
        "backend_ref": airflow._backend_ref(dag_id, run_id),
        "status": airflow.map_state(native), "native_state": native,
        "compute_platform": envelope["compute_platform"], "reconciled": reconciled,
    }, sort_keys=True))
    return 0


def _receipt(envelope: dict[str, Any]) -> dict[str, Any] | None:
    path = pathlib.Path(envelope["receipt_path"])
    if not path.exists():
        return None
    payload = _load_json(path, "orchestration receipt")
    if (
        payload.get("envelope_sha256") != envelope["envelope_sha256"]
        or payload.get("compute_platform") != envelope["compute_platform"]
        or payload.get("job_id") != envelope["job_id"]
    ):
        raise OrchestrationError("orchestration receipt ownership differs from the envelope")
    return payload


def status(args: argparse.Namespace) -> int:
    if airflow is None:
        raise OrchestrationError("Airflow client module is unavailable")
    _, envelope = load_envelope(args.envelope)
    dag_id, run_id = airflow._parse_backend_ref(args.backend_ref)
    if run_id != envelope["orchestration_id"]:
        raise OrchestrationError("backend_ref differs from the orchestration envelope")
    dag = airflow.AirflowClient().dag_run(dag_id, run_id)
    dag_state = airflow.map_state(dag.get("state"))
    receipt = _receipt(envelope)
    compute_state = receipt.get("status") if receipt else "PENDING"
    if dag_state == "COMPLETE" and compute_state == "COMPLETE":
        combined = "COMPLETE"
    elif dag_state in {"ERROR", "CANCELED", "UNKNOWN"}:
        combined = dag_state
    elif compute_state in {"ERROR", "CANCELED", "UNKNOWN"}:
        combined = compute_state
    elif dag_state == "RUNNING" or compute_state == "RUNNING":
        combined = "RUNNING"
    else:
        combined = "PENDING"
    print(json.dumps({
        "backend_ref": args.backend_ref, "status": combined,
        "airflow_state": dag_state, "compute_state": compute_state,
        "compute_platform": envelope["compute_platform"],
        "compute_backend_ref": receipt.get("compute_backend_ref") if receipt else None,
    }, sort_keys=True))
    return 0


def logs(args: argparse.Namespace) -> int:
    _, envelope = load_envelope(args.envelope)
    path = pathlib.Path(envelope["log_path"])
    if not path.exists():
        print("orchestration log is not available yet")
        return 0
    if not path.is_file() or path.is_symlink():
        raise OrchestrationError("orchestration log is unsafe")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    print(_redact("\n".join(lines[-args.tail :])))
    return 0


def cancel(args: argparse.Namespace) -> int:
    if airflow is None:
        raise OrchestrationError("Airflow client module is unavailable")
    if not args.confirm:
        raise OrchestrationError("cancel requires --confirm")
    _, envelope = load_envelope(args.envelope)
    receipt = _receipt(envelope)
    if receipt and receipt.get("status") not in TERMINAL:
        command = _substitute_backend_ref(envelope["commands"]["cancel"], receipt.get("compute_backend_ref"))
        completed = _run_command(command, envelope, "cancel")
        state = _status_from_output(completed.stdout, "cancel")
        if state not in {"CANCELED", "COMPLETE"}:
            raise OrchestrationError("compute cancellation was not confirmed; Airflow run retained")
    dag_id, run_id = airflow._parse_backend_ref(args.backend_ref)
    if run_id != envelope["orchestration_id"]:
        raise OrchestrationError("backend_ref differs from the orchestration envelope")
    client = airflow.AirflowClient()
    current = airflow.map_state(client.dag_run(dag_id, run_id).get("state"))
    if current not in airflow.TERMINAL_STATES:
        endpoint = (
            f"/api/v2/dags/{urllib.parse.quote(dag_id, safe='')}/dagRuns/"
            f"{urllib.parse.quote(run_id, safe='')}"
        )
        client._request("DELETE", endpoint)
    print(json.dumps({"backend_ref": args.backend_ref, "status": "CANCELED", "compute_canceled_first": bool(receipt)}, sort_keys=True))
    return 0


def _substitute_backend_ref(command: list[str], backend_ref: Any) -> list[str]:
    if not isinstance(backend_ref, str) or not backend_ref:
        if any("{backend_ref}" in item for item in command):
            raise OrchestrationError("consumer command requires a missing compute backend_ref")
        return list(command)
    return [item.replace("{backend_ref}", backend_ref) for item in command]


def _run_command(command: list[str], envelope: dict[str, Any], operation: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    for name in envelope["forward_env"]:
        if not env.get(name):
            raise OrchestrationError(f"approved environment variable is absent: {name}")
    completed = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
    log_path = pathlib.Path(envelope["log_path"])
    block = (
        f"OPERATION={operation}\nCOMMAND={json.dumps(command)}\n"
        f"STDOUT={completed.stdout or ''}\nSTDERR={completed.stderr or ''}\n"
        f"EXIT_CODE={completed.returncode}\n"
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(_redact(block))
        handle.flush()
        os.fsync(handle.fileno())
    if completed.returncode != 0:
        raise RuntimeError(f"{operation} consumer exited {completed.returncode}; inspect {log_path}")
    return completed


def _json_result(output: str, operation: str) -> dict[str, Any]:
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    for row in reversed(rows):
        try:
            payload = json.loads(row)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise OrchestrationError(f"{operation} consumer returned no JSON object")


def _status_from_output(output: str, operation: str) -> str:
    payload = _json_result(output, operation)
    value = payload.get("status", payload.get("state"))
    if isinstance(value, str):
        value = value.upper()
    aliases = {"SUCCESS": "COMPLETE", "FAILED": "ERROR", "CANCELLED": "CANCELED"}
    value = aliases.get(value, value)
    if value not in {"PENDING", "RUNNING", "COMPLETE", "ERROR", "CANCELED", "UNKNOWN"}:
        raise OrchestrationError(f"{operation} consumer returned an invalid status")
    return value


def _cancel_owned(
    envelope: dict[str, Any], receipt: dict[str, Any], *, reason: str
) -> None:
    command = _substitute_backend_ref(
        envelope["commands"]["cancel"], receipt.get("compute_backend_ref")
    )
    state = _status_from_output(_run_command(command, envelope, "cancel").stdout, "cancel")
    if state not in {"CANCELED", "COMPLETE"}:
        raise RuntimeError(f"compute cancellation after {reason} was not confirmed")
    receipt["status"] = "CANCELED" if state == "CANCELED" else "COMPLETE"
    receipt["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    _atomic_json(pathlib.Path(envelope["receipt_path"]), receipt)


def execute_conf(conf: Any) -> dict[str, Any]:
    expected = {"contract", "kind", "orchestration_id", "envelope_path", "envelope_sha256"}
    if not isinstance(conf, dict) or set(conf) != expected or conf.get("contract") != CONTRACT or conf.get("kind") != KIND:
        raise OrchestrationError("Airflow orchestration conf has an invalid shape")
    path, envelope = load_envelope(pathlib.Path(str(conf.get("envelope_path"))))
    if (
        conf.get("orchestration_id") != envelope["orchestration_id"]
        or conf.get("envelope_sha256") != envelope["envelope_sha256"]
    ):
        raise OrchestrationError("Airflow conf differs from the immutable envelope")
    receipt = _receipt(envelope)
    if receipt is None:
        submitted_ns = time.time_ns()
        result = _json_result(_run_command(envelope["commands"]["submit"], envelope, "submit").stdout, "submit")
        backend_ref = result.get("backend_ref")
        if not isinstance(backend_ref, str) or not backend_ref:
            raise OrchestrationError("submit consumer did not return backend_ref")
        receipt = {
            "schema_version": "1", "workflow": WORKFLOW,
            "orchestrator": "airflow", "compute_platform": envelope["compute_platform"],
            "job_id": envelope["job_id"], "orchestration_id": envelope["orchestration_id"],
            "envelope_sha256": envelope["envelope_sha256"],
            "compute_backend_ref": backend_ref, "status": "RUNNING",
            "submitted_ns": submitted_ns,
            "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        _atomic_json(pathlib.Path(envelope["receipt_path"]), receipt)
    if receipt.get("status") == "COMPLETE":
        return receipt
    backend_ref = receipt.get("compute_backend_ref")
    deadline = time.monotonic() + envelope["deadline_s"]
    unknown_count = 0
    while time.monotonic() < deadline:
        command = _substitute_backend_ref(envelope["commands"]["status"], backend_ref)
        state = _status_from_output(_run_command(command, envelope, "status").stdout, "status")
        if state == "UNKNOWN":
            unknown_count += 1
            if unknown_count >= envelope["unknown_status_limit"]:
                receipt["status"] = "UNKNOWN"
                receipt["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
                _atomic_json(pathlib.Path(envelope["receipt_path"]), receipt)
                if not envelope["retain_on_failure"]:
                    _cancel_owned(envelope, receipt, reason="repeated UNKNOWN status")
                disposition = "retained for reconciliation" if envelope["retain_on_failure"] else "canceled by approved policy"
                raise RuntimeError(f"compute status remained UNKNOWN; owned workload {disposition}")
        else:
            unknown_count = 0
        if state in TERMINAL:
            receipt["status"] = state
            receipt["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            _atomic_json(pathlib.Path(envelope["receipt_path"]), receipt)
            if state != "COMPLETE":
                _run_command(_substitute_backend_ref(envelope["commands"]["logs"], backend_ref), envelope, "logs")
                if state == "ERROR" and not envelope["retain_on_failure"]:
                    _cancel_owned(envelope, receipt, reason="backend error")
                disposition = "retained" if envelope["retain_on_failure"] else "canceled"
                raise RuntimeError(f"compute backend terminated as {state}; owned workload {disposition} according to policy")
            started_ns = receipt.get("submitted_ns")
            if not isinstance(started_ns, int) or isinstance(started_ns, bool) or started_ns < 1:
                raise OrchestrationError("orchestration receipt lacks a valid submit time")
            for value in envelope["expected_outputs"]:
                output = pathlib.Path(value)
                if not output.is_file() or output.is_symlink() or output.stat().st_size == 0:
                    raise RuntimeError(f"compute output is missing, empty, or unsafe: {output}")
                if output.stat().st_mtime_ns < started_ns:
                    raise RuntimeError(f"compute output predates orchestration submission: {output}")
            return receipt
        time.sleep(envelope["poll_interval_s"])
    receipt["status"] = "UNKNOWN"
    receipt["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    _atomic_json(pathlib.Path(envelope["receipt_path"]), receipt)
    if not envelope["retain_on_failure"]:
        _cancel_owned(envelope, receipt, reason="deadline expiry")
    raise RuntimeError(
        "compute deadline exceeded; apply the approved retention/cancellation policy"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="verb", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--compute-platform", required=True, choices=COMPUTE_PLATFORMS)
    prepare_parser.add_argument("--compute-kind", required=True, choices=COMPUTE_KINDS)
    prepare_parser.add_argument("--compute-request", required=True, type=pathlib.Path)
    prepare_parser.add_argument("--job-record", required=True, type=pathlib.Path)
    prepare_parser.add_argument("--job-binding", type=pathlib.Path)
    prepare_parser.add_argument("--consumer-plan", required=True, type=pathlib.Path)
    prepare_parser.add_argument("--output", required=True, type=pathlib.Path)
    submit_parser = sub.add_parser("submit")
    submit_parser.add_argument("--envelope", required=True, type=pathlib.Path)
    for verb in ("status", "logs", "cancel"):
        child = sub.add_parser(verb)
        child.add_argument("--envelope", required=True, type=pathlib.Path)
        child.add_argument("--backend-ref", required=True)
        if verb == "logs":
            child.add_argument("--tail", type=int, default=200, choices=range(1, 10001))
        if verb == "cancel":
            child.add_argument("--confirm", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return {"prepare": prepare, "submit": submit, "status": status, "logs": logs, "cancel": cancel}[args.verb](args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"airflow orchestrator failed: {_redact(str(exc))}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
