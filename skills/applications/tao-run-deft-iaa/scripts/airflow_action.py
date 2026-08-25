#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IAA-scoped Airflow REST consumer for signed DEFT action requests.

Airflow is intentionally not a bank-wide platform.  This client accepts only
the versioned ``tao-run-deft-iaa`` DAG contract and the immutable action/job
binding emitted by this application.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable, Sequence

from run_deft_action import JOB_IDENTITY_FIELDS, load_bound_action_for_submit


WORKFLOW = "tao-run-deft-iaa"
CONTRACT = "tao-deft-iaa-action-v1"
DEFAULT_DAG_ID = "tao_deft_iaa_action_v1"
SAFE_DAG_ID = re.compile(r"[A-Za-z0-9_.-]{1,250}")
SAFE_RUN_ID = re.compile(r"[A-Za-z0-9_.~-]{1,250}")
SHA256 = re.compile(r"[0-9a-f]{64}")
TERMINAL_STATES = frozenset({"COMPLETE", "ERROR", "CANCELED"})
PENDING_STATES = frozenset(
    {None, "queued", "scheduled", "deferred", "up_for_retry", "up_for_reschedule", "restarting"}
)
SECRET_ENV_NAMES = (
    "AIRFLOW_API_TOKEN",
    "AIRFLOW_PASSWORD",
    "BREV_API_TOKEN",
    "HF_TOKEN",
    "NGC_KEY",
)


class AirflowContractError(ValueError):
    """A deterministic request, response, or configuration failure."""


class AirflowApiError(RuntimeError):
    """A bounded Airflow API failure with no response body disclosure."""

    def __init__(self, operation: str, status: int | None = None):
        detail = f" (HTTP {status})" if status is not None else ""
        super().__init__(f"Airflow {operation} failed{detail}")
        self.status = status


def _canonical_sha256(payload: dict[str, Any], digest_field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(digest_field, None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _regular_json(path: pathlib.Path, label: str) -> tuple[pathlib.Path, dict[str, Any]]:
    lexical = pathlib.Path(os.path.abspath(path.expanduser()))
    if (
        not lexical.is_file()
        or lexical.is_symlink()
        or lexical.resolve(strict=False) != lexical
        or lexical.stat().st_size == 0
    ):
        raise AirflowContractError(f"{label} must be a non-empty regular non-symlink file")
    try:
        payload = json.loads(lexical.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AirflowContractError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise AirflowContractError(f"{label} JSON root must be an object")
    return lexical, payload


def _secret_values() -> tuple[str, ...]:
    return tuple(
        value for name in SECRET_ENV_NAMES if (value := os.environ.get(name))
    )


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


def _reject_secret_material(payload: dict[str, Any], label: str) -> None:
    secrets = _secret_values()
    if not secrets:
        return
    if any(secret in text for text in _walk_strings(payload) for secret in secrets):
        raise AirflowContractError(f"{label} contains a credential value")


def _load_bound_action(
    request_path: pathlib.Path,
    binding_path: pathlib.Path,
    job_record_path: pathlib.Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        request, binding, job = load_bound_action_for_submit(
            request_path, binding_path, job_record_path,
        )
    except (OSError, ValueError) as exc:
        raise AirflowContractError(f"invalid bound IAA action: {exc}") from exc
    if request.get("platform") != "airflow":
        raise AirflowContractError("Airflow accepts only IAA requests with platform=airflow")
    _reject_secret_material(request, "action request")
    _reject_secret_material(binding, "job binding")
    _reject_secret_material(job, "job record")
    return request, binding, job


def _base_url() -> str:
    raw = os.environ.get("AIRFLOW_BASE_URL", "").strip().rstrip("/")
    if not raw:
        raise AirflowContractError("AIRFLOW_BASE_URL is required")
    parsed = urllib.parse.urlsplit(raw)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise AirflowContractError(
            "AIRFLOW_BASE_URL must be a credential-free HTTP(S) origin"
        )
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise AirflowContractError("remote Airflow requires HTTPS")
    return raw


def _dag_id() -> str:
    value = os.environ.get("TAO_IAA_AIRFLOW_DAG_ID", DEFAULT_DAG_ID).strip()
    if SAFE_DAG_ID.fullmatch(value) is None:
        raise AirflowContractError("TAO_IAA_AIRFLOW_DAG_ID is invalid")
    return value


def _shared_root() -> pathlib.PurePosixPath:
    raw = os.environ.get("TAO_IAA_AIRFLOW_SHARED_ROOT", "").strip()
    if not raw.startswith("/") or any(character in raw for character in ("\x00", "\n", "\r")):
        raise AirflowContractError("TAO_IAA_AIRFLOW_SHARED_ROOT must be an absolute path")
    path = pathlib.PurePosixPath(raw)
    if path == pathlib.PurePosixPath("/") or str(path) != raw or ".." in path.parts:
        raise AirflowContractError(
            "TAO_IAA_AIRFLOW_SHARED_ROOT must be normalized, non-root, and traversal-free"
        )
    return path


def _resolved_mounts(request: dict[str, Any], specifications: list[str]) -> list[dict[str, Any]]:
    declared = request.get("mounts")
    if not isinstance(declared, list) or not declared:
        raise AirflowContractError("action request must declare mounts")
    if len(specifications) != len(declared):
        raise AirflowContractError("repeat --mount exactly once per request mount, in order")
    root = _shared_root()
    resolved: list[dict[str, Any]] = []
    for index, (expected, raw) in enumerate(zip(declared, specifications, strict=True)):
        if not isinstance(expected, dict):
            raise AirflowContractError("action request contains an invalid mount")
        fields = raw.rsplit(":", 2)
        if len(fields) != 3 or fields[2] not in {"ro", "rw"}:
            raise AirflowContractError(
                "--mount must use COMPUTE_SOURCE:TARGET:ro|rw"
            )
        source_raw, target, mode = fields
        source = pathlib.PurePosixPath(source_raw)
        if (
            not source_raw.startswith("/")
            or source == pathlib.PurePosixPath("/")
            or str(source) != source_raw
            or ".." in source.parts
            or root not in source.parents
        ):
            raise AirflowContractError(
                f"resolved mount {index} source must be a normalized child of {root}"
            )
        if target != expected.get("target"):
            raise AirflowContractError(f"resolved mount {index} target differs from request")
        read_only = mode == "ro"
        if read_only is not expected.get("read_only"):
            raise AirflowContractError(f"resolved mount {index} mode differs from request")
        resolved.append({
            "source": source_raw,
            "target": target,
            "read_only": read_only,
            "declared_source_sha256": hashlib.sha256(
                str(expected.get("source", "")).encode("utf-8")
            ).hexdigest(),
        })
    return resolved


def _ssl_context() -> ssl.SSLContext:
    bundle = os.environ.get("AIRFLOW_CA_BUNDLE")
    if not bundle:
        return ssl.create_default_context()
    path = pathlib.Path(os.path.abspath(pathlib.Path(bundle).expanduser()))
    if not path.is_file() or path.is_symlink() or path.resolve(strict=False) != path:
        raise AirflowContractError("AIRFLOW_CA_BUNDLE must be a regular non-symlink file")
    return ssl.create_default_context(cafile=str(path))


class AirflowClient:
    """Small Airflow 3 REST client with environment-only authentication."""

    def __init__(self, *, timeout: int | None = None):
        self.base_url = _base_url()
        configured = os.environ.get("AIRFLOW_REQUEST_TIMEOUT", "30")
        try:
            self.timeout = timeout if timeout is not None else int(configured)
        except ValueError as exc:
            raise AirflowContractError("AIRFLOW_REQUEST_TIMEOUT must be an integer") from exc
        if not 1 <= self.timeout <= 300:
            raise AirflowContractError("AIRFLOW_REQUEST_TIMEOUT must be in [1, 300]")
        self.context = _ssl_context()
        self.token = self._token()

    def _token(self) -> str:
        token = os.environ.get("AIRFLOW_API_TOKEN")
        if token:
            return token
        local_all_admins = os.environ.get("TAO_IAA_AIRFLOW_LOCAL_ALL_ADMINS") == "1"
        if local_all_admins:
            parsed = urllib.parse.urlsplit(self.base_url)
            if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
                raise AirflowContractError(
                    "TAO_IAA_AIRFLOW_LOCAL_ALL_ADMINS is permitted only for loopback Airflow"
                )
            payload = self._request("GET", "/auth/token", auth=False)
            access = payload.get("access_token") if isinstance(payload, dict) else None
            if not isinstance(access, str) or not access:
                raise AirflowApiError("local all-admin token response validation")
            return access
        username = os.environ.get("AIRFLOW_USERNAME")
        password = os.environ.get("AIRFLOW_PASSWORD")
        if not username or not password:
            raise AirflowContractError(
                "set AIRFLOW_API_TOKEN or both AIRFLOW_USERNAME and AIRFLOW_PASSWORD"
            )
        payload = self._request(
            "POST", "/auth/token", {"username": username, "password": password}, auth=False
        )
        access = payload.get("access_token") if isinstance(payload, dict) else None
        if not isinstance(access, str) or not access:
            raise AirflowApiError("token response validation")
        return access

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        auth: bool = True,
        accept_text: bool = False,
    ) -> Any:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if auth:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(
            self.base_url + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout, context=self.context
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise AirflowApiError(f"{method} {path}", exc.code) from None
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            raise AirflowApiError(f"{method} {path}") from exc
        if accept_text:
            return raw.decode("utf-8", errors="replace")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AirflowApiError(f"{method} {path} response validation") from exc

    def dag(self, dag_id: str) -> dict[str, Any]:
        path = f"/api/v2/dags/{urllib.parse.quote(dag_id, safe='')}"
        payload = self._request("GET", path)
        if not isinstance(payload, dict):
            raise AirflowApiError("DAG response validation")
        return payload

    def dag_run(self, dag_id: str, run_id: str) -> dict[str, Any]:
        path = (
            f"/api/v2/dags/{urllib.parse.quote(dag_id, safe='')}/dagRuns/"
            f"{urllib.parse.quote(run_id, safe='')}"
        )
        payload = self._request("GET", path)
        if not isinstance(payload, dict):
            raise AirflowApiError("DAG-run response validation")
        return payload

    def pool(self, pool_name: str) -> dict[str, Any]:
        path = f"/api/v2/pools/{urllib.parse.quote(pool_name, safe='')}"
        payload = self._request("GET", path)
        if not isinstance(payload, dict):
            raise AirflowApiError("pool response validation")
        return payload


def _dag_tags(payload: dict[str, Any]) -> set[str]:
    tags = payload.get("tags")
    if not isinstance(tags, list):
        return set()
    return {
        item if isinstance(item, str) else item.get("name")
        for item in tags
        if isinstance(item, str) or isinstance(item, dict)
    } - {None}


def validate_dag(client: AirflowClient, dag_id: str) -> dict[str, Any]:
    payload = client.dag(dag_id)
    if payload.get("dag_id") != dag_id:
        raise AirflowContractError("Airflow returned a different DAG identity")
    if payload.get("is_paused") is True:
        raise AirflowContractError(f"required IAA DAG is paused: {dag_id}")
    if CONTRACT not in _dag_tags(payload):
        raise AirflowContractError(
            f"DAG {dag_id} does not advertise required contract tag {CONTRACT}"
        )
    return payload


def _pool_requirement(value: str) -> tuple[str, int]:
    fields = value.rsplit(":", 1)
    if len(fields) != 2 or SAFE_DAG_ID.fullmatch(fields[0]) is None:
        raise AirflowContractError("--pool must use NAME:MIN_SLOTS")
    try:
        minimum = int(fields[1])
    except ValueError as exc:
        raise AirflowContractError("--pool MIN_SLOTS must be an integer") from exc
    if not 1 <= minimum <= 4096:
        raise AirflowContractError("--pool MIN_SLOTS must be in [1, 4096]")
    return fields[0], minimum


def validate_pools(client: AirflowClient, requirements: Sequence[str]) -> list[dict[str, Any]]:
    if not requirements:
        raise AirflowContractError("preflight requires at least one --pool NAME:MIN_SLOTS")
    checked: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw in requirements:
        name, minimum = _pool_requirement(raw)
        if name in names:
            raise AirflowContractError(f"duplicate Airflow pool requirement: {name}")
        names.add(name)
        payload = client.pool(name)
        if payload.get("name") != name:
            raise AirflowContractError(f"Airflow returned a different pool identity for {name}")
        slots = payload.get("slots")
        if not isinstance(slots, int) or isinstance(slots, bool) or slots < minimum:
            raise AirflowContractError(
                f"Airflow pool {name} has {slots!r} total slots; at least {minimum} required"
            )
        open_slots = payload.get("open_slots")
        if open_slots is not None and (
            not isinstance(open_slots, int) or isinstance(open_slots, bool) or open_slots < 0
        ):
            raise AirflowContractError(f"Airflow pool {name} returned invalid open_slots")
        checked.append({
            "name": name,
            "minimum_slots": minimum,
            "total_slots": slots,
            "open_slots": open_slots,
        })
    return checked


def _backend_ref(dag_id: str, run_id: str) -> str:
    if SAFE_DAG_ID.fullmatch(dag_id) is None or SAFE_RUN_ID.fullmatch(run_id) is None:
        raise AirflowContractError("Airflow backend identity contains unsupported characters")
    return f"{dag_id}/{run_id}"


def _parse_backend_ref(value: str) -> tuple[str, str]:
    fields = value.split("/", 1)
    if len(fields) != 2:
        raise AirflowContractError("Airflow backend_ref must be <dag_id>/<dag_run_id>")
    dag_id, run_id = fields
    _backend_ref(dag_id, run_id)
    if dag_id != _dag_id():
        raise AirflowContractError("backend_ref DAG differs from the configured IAA DAG")
    return dag_id, run_id


def map_state(native: Any) -> str:
    normalized = native.lower() if isinstance(native, str) else native
    if normalized in PENDING_STATES:
        return "PENDING"
    if normalized == "running":
        return "RUNNING"
    if normalized == "success":
        return "COMPLETE"
    if normalized in {"failed", "upstream_failed"}:
        return "ERROR"
    if normalized in {"removed", "canceled", "cancelled"}:
        return "CANCELED"
    return "UNKNOWN"


def preflight(args: argparse.Namespace) -> int:
    dag_id = _dag_id()
    shared_root = _shared_root()
    client = AirflowClient()
    payload = validate_dag(client, dag_id)
    pools = validate_pools(client, args.pool)
    print(json.dumps({
        "status": "ok",
        "platform": "airflow",
        "dag_id": dag_id,
        "contract": CONTRACT,
        "paused": bool(payload.get("is_paused")),
        "shared_root": str(shared_root),
        "pools": pools,
    }, sort_keys=True))
    return 0


def submit(args: argparse.Namespace) -> int:
    request, binding, job = _load_bound_action(
        args.request, args.job_binding, args.job_record
    )
    dag_id = _dag_id()
    run_id = str(job["id"])
    backend_ref = _backend_ref(dag_id, run_id)
    client = AirflowClient()
    validate_dag(client, dag_id)
    mounts = _resolved_mounts(request, args.mount)
    job_identity = {field: job.get(field) for field in JOB_IDENTITY_FIELDS}
    conf = {
        "contract": CONTRACT,
        "job_id": run_id,
        "request_sha256": request["request_sha256"],
        "binding_sha256": binding["binding_sha256"],
        "job_identity": job_identity,
        "request": request,
        "resolved_mounts": mounts,
        "results_scope": binding["results_scope"],
        "staging_receipt_sha256": binding["staging_receipt_sha256"],
    }
    _reject_secret_material(conf, "Airflow DAG conf")
    path = f"/api/v2/dags/{urllib.parse.quote(dag_id, safe='')}/dagRuns"
    try:
        response = client._request(
            "POST", path, {"dag_run_id": run_id, "logical_date": None, "conf": conf}
        )
        reconciled = False
    except AirflowApiError as exc:
        if exc.status != 409:
            raise
        response = client.dag_run(dag_id, run_id)
        existing = response.get("conf")
        if (
            not isinstance(existing, dict)
            or existing.get("contract") != CONTRACT
            or existing.get("job_id") != run_id
            or existing.get("request_sha256") != request["request_sha256"]
            or existing.get("binding_sha256") != binding["binding_sha256"]
            or existing.get("job_identity") != job_identity
        ):
            raise AirflowContractError(
                "existing Airflow DAG run does not match the bound IAA action"
            )
        reconciled = True
    native = response.get("state") if isinstance(response, dict) else None
    print(json.dumps({
        "backend_ref": backend_ref,
        "status": map_state(native),
        "native_state": native,
        "reconciled": reconciled,
    }, sort_keys=True))
    return 0


def status(args: argparse.Namespace) -> int:
    dag_id, run_id = _parse_backend_ref(args.backend_ref)
    payload = AirflowClient().dag_run(dag_id, run_id)
    native = payload.get("state")
    print(json.dumps({
        "backend_ref": args.backend_ref,
        "status": map_state(native),
        "native_state": native,
    }, sort_keys=True))
    return 0


def _redact(text: str) -> str:
    for value in _secret_values():
        text = text.replace(value, "[REDACTED]")
    return text


def logs(args: argparse.Namespace) -> int:
    dag_id, run_id = _parse_backend_ref(args.backend_ref)
    client = AirflowClient()
    base = (
        f"/api/v2/dags/{urllib.parse.quote(dag_id, safe='')}/dagRuns/"
        f"{urllib.parse.quote(run_id, safe='')}/taskInstances"
    )
    payload = client._request("GET", f"{base}?page_limit={args.tail}&page_offset=0")
    instances = payload.get("task_instances") if isinstance(payload, dict) else None
    if not isinstance(instances, list):
        raise AirflowApiError("task-instance response validation")
    rows = []
    for item in instances[-args.tail :]:
        if not isinstance(item, dict):
            continue
        rows.append({
            "task_id": item.get("task_id"),
            "map_index": item.get("map_index", -1),
            "try_number": item.get("try_number"),
            "state": item.get("state"),
            "start_date": item.get("start_date"),
            "end_date": item.get("end_date"),
        })
    print(_redact(json.dumps({
        "backend_ref": args.backend_ref,
        "task_instances": rows,
        "note": "Canonical workload output is written to request.log_path on shared storage.",
    }, sort_keys=True)))
    return 0


def cancel(args: argparse.Namespace) -> int:
    if not args.confirm:
        raise AirflowContractError("cancel requires --confirm after approval")
    dag_id, run_id = _parse_backend_ref(args.backend_ref)
    client = AirflowClient()
    current = client.dag_run(dag_id, run_id)
    mapped = map_state(current.get("state"))
    if mapped in TERMINAL_STATES:
        raise AirflowContractError(f"cannot cancel terminal Airflow state {mapped}")
    path = (
        f"/api/v2/dags/{urllib.parse.quote(dag_id, safe='')}/dagRuns/"
        f"{urllib.parse.quote(run_id, safe='')}"
    )
    client._request("DELETE", path)
    print(json.dumps({
        "backend_ref": args.backend_ref,
        "status": "CANCELED",
        "native_state": "deleted",
    }, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="verb", required=True)
    preflight_parser = sub.add_parser("preflight")
    preflight_parser.add_argument(
        "--pool",
        action="append",
        required=True,
        help="repeat for every required Airflow pool as NAME:MIN_SLOTS",
    )
    submit_parser = sub.add_parser("submit")
    submit_parser.add_argument("--request", required=True, type=pathlib.Path)
    submit_parser.add_argument("--job-binding", required=True, type=pathlib.Path)
    submit_parser.add_argument("--job-record", required=True, type=pathlib.Path)
    submit_parser.add_argument(
        "--mount",
        action="append",
        default=[],
        help="repeat in request order as COMPUTE_SOURCE:TARGET:ro|rw",
    )
    for verb in ("status", "logs", "cancel"):
        child = sub.add_parser(verb)
        child.add_argument("--backend-ref", required=True)
        if verb == "logs":
            child.add_argument("--tail", type=int, default=200, choices=range(1, 1001))
        if verb == "cancel":
            child.add_argument("--confirm", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return {
            "preflight": preflight,
            "submit": submit,
            "status": status,
            "logs": logs,
            "cancel": cancel,
        }[args.verb](args)
    except (AirflowApiError, AirflowContractError, OSError) as exc:
        print(f"airflow action failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
