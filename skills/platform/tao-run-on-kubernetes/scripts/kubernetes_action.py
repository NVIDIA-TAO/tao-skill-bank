#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Four-verb Kubernetes consumer for one signed ordinary action request.

The existing renderer remains authoritative for argv, mounts, credentials, and
GPU shape.  This wrapper adds a deterministic native lifecycle without ever
reconstructing the workload command or broadening its resource request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Any, Sequence

import yaml

import render_action_job as renderer


MANAGED_BY = "tao-skill-bank"
REQUEST_ANNOTATION = "tao.nvidia.com/action-request-sha256"
MANAGED_ANNOTATION = "tao.nvidia.com/managed-by"
JOB_ID_ANNOTATION = "tao.nvidia.com/job-record-id"
SHA256 = re.compile(r"[0-9a-f]{64}")
SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|authorization|password|secret|token)\s*[=:]\s*\S+"
)
TERMINAL_CONDITIONS = {"Complete": "COMPLETE", "Failed": "ERROR"}


class ContractError(ValueError):
    """The request, Kubernetes object, or lifecycle response is unsafe."""


def _canonical_sha256(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("request_sha256", None)
    return hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
    ).hexdigest()


def _load_regular_json(path: pathlib.Path, label: str) -> dict[str, Any]:
    lexical = path.expanduser().absolute()
    if lexical != pathlib.Path(os.path.abspath(lexical)):
        raise ContractError(f"{label} path contains lexical traversal")
    if not lexical.is_file() or lexical.is_symlink() or lexical.resolve() != lexical:
        raise ContractError(f"{label} must be a regular non-symlink file")
    try:
        payload = json.loads(lexical.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"{label} JSON root must be an object")
    return payload


def load_request(path: pathlib.Path) -> dict[str, Any]:
    request = _load_regular_json(path, "action request")
    digest = request.get("request_sha256")
    if (
        request.get("schema_version") != renderer.REQUEST_SCHEMA_VERSION
        or request.get("platform") != "kubernetes"
        or not isinstance(digest, str)
        or SHA256.fullmatch(digest) is None
        or digest != _canonical_sha256(request)
    ):
        raise ContractError("action request is not a signed Kubernetes schema-v1 request")
    _reject_secret_values(request)
    return request


def _walk_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _secret_values(request: dict[str, Any]) -> tuple[str, ...]:
    names = set(request.get("forward_env") or [])
    names.update({"HF_TOKEN", "NGC_API_KEY", "NGC_KEY", "AWS_SECRET_ACCESS_KEY"})
    return tuple(
        value for name in names
        if isinstance(name, str) and len(value := os.environ.get(name, "")) >= 8
    )


def _reject_secret_values(payload: Any) -> None:
    secrets = _secret_values(payload if isinstance(payload, dict) else {})
    if any(secret in text for text in _walk_strings(payload) for secret in secrets):
        raise ContractError("request contains a credential value")


def _redact(text: str, request: dict[str, Any]) -> str:
    for value in _secret_values(request):
        text = text.replace(value, "[REDACTED]")
    return SECRET_PATTERN.sub(lambda match: match.group(1) + "=<redacted>", text)


def _timeout(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 300:
        raise ContractError("request timeout must be in [1, 300] seconds")
    return value


def _run(
    argv: list[str], *, timeout_s: int, stdin: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        argv, input=stdin, capture_output=True, text=True, check=False,
        timeout=_timeout(timeout_s),
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ContractError(f"kubectl command failed ({completed.returncode}): {detail[:500]}")
    return completed


def _kubectl_json(
    args: list[str], *, timeout_s: int, stdin: str | None = None,
    check: bool = True,
) -> dict[str, Any] | None:
    completed = _run(["kubectl", *args], timeout_s=timeout_s, stdin=stdin, check=check)
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ContractError("kubectl returned malformed JSON") from exc
    if not isinstance(payload, dict):
        raise ContractError("kubectl JSON root must be an object")
    return payload


def _identity(request: dict[str, Any], job_id: str, namespace: str) -> dict[str, str]:
    return {
        "name": renderer.kubernetes_job_name(job_id),
        "namespace": renderer._dns_name(namespace, "namespace", label_only=True),
        "job_id": job_id,
        "request_sha256": request["request_sha256"],
    }


def _owned(job: dict[str, Any], identity: dict[str, str]) -> bool:
    metadata = job.get("metadata") or {}
    annotations = metadata.get("annotations") or {}
    return (
        metadata.get("name") == identity["name"]
        and metadata.get("namespace") == identity["namespace"]
        and annotations.get(JOB_ID_ANNOTATION) == identity["job_id"]
        and annotations.get(REQUEST_ANNOTATION) == identity["request_sha256"]
        and annotations.get(MANAGED_ANNOTATION) == MANAGED_BY
    )


def _get_job(identity: dict[str, str], timeout_s: int) -> dict[str, Any] | None:
    completed = _run(
        ["kubectl", "get", "job", identity["name"], "-n", identity["namespace"],
         "--ignore-not-found", "-o", "json"],
        timeout_s=timeout_s, check=True,
    )
    if not completed.stdout.strip():
        return None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ContractError("kubectl returned malformed Job JSON") from exc
    if not isinstance(payload, dict):
        raise ContractError("kubectl Job JSON root must be an object")
    return payload


def _gpu_preflight(request: dict[str, Any], timeout_s: int) -> None:
    _, _, _, requested = renderer._command_bundle(request)
    gpu_ids = request.get("gpu_ids")
    if (
        not isinstance(gpu_ids, list)
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in gpu_ids)
        or len(gpu_ids) != len(set(gpu_ids))
        or len(gpu_ids) != requested
    ):
        raise ContractError("request gpu_ids must exactly bind compute_shape.gpus")
    if requested == 0:
        return
    nodes = _kubectl_json(["get", "nodes", "-o", "json"], timeout_s=timeout_s)
    rows = nodes.get("items") if nodes else None
    if not isinstance(rows, list):
        raise ContractError("Kubernetes node inventory is missing")
    capacities: list[int] = []
    for row in rows:
        raw = ((row.get("status") or {}).get("allocatable") or {}).get("nvidia.com/gpu", 0)
        try:
            capacities.append(int(raw))
        except (TypeError, ValueError) as exc:
            raise ContractError("Kubernetes GPU capacity is malformed") from exc
    largest = max(capacities, default=0)
    if largest < requested:
        raise ContractError(
            f"no Kubernetes node can fit {requested} GPUs; largest allocatable node has {largest}"
        )


def _render_manifest(args: argparse.Namespace, request: dict[str, Any]) -> tuple[str, dict[str, str]]:
    staging = _load_regular_json(args.staging_map, "staging map")
    identity = _identity(request, args.job_id, args.namespace)
    rendered = renderer.render_action_job(
        request, staging, job_id=args.job_id, namespace=args.namespace,
        pvc_claim=args.pvc_claim, credential_secret=args.credential_secret,
        image_pull_secret=args.image_pull_secret, ttl_seconds=args.ttl_seconds,
        shm_size=args.shm_size,
    )
    manifest = yaml.safe_load(rendered)
    if not isinstance(manifest, dict) or manifest.get("kind") != "Job":
        raise ContractError("renderer did not produce one Kubernetes Job")
    metadata = manifest.setdefault("metadata", {})
    annotations = metadata.setdefault("annotations", {})
    annotations.update({
        REQUEST_ANNOTATION: request["request_sha256"],
        MANAGED_ANNOTATION: MANAGED_BY,
    })
    if not _owned(manifest, identity):
        raise ContractError("rendered Job identity differs from the signed request")
    return yaml.safe_dump(manifest, sort_keys=True), identity


def _workload_identity(job: dict[str, Any]) -> Any:
    spec = job.get("spec") or {}
    template = spec.get("template") or {}
    pod_spec = template.get("spec") or {}
    return {
        "containers": pod_spec.get("containers"),
        "volumes": pod_spec.get("volumes"),
        "imagePullSecrets": pod_spec.get("imagePullSecrets"),
        "restartPolicy": pod_spec.get("restartPolicy"),
    }


def _resume_result(
    existing: dict[str, Any], desired: dict[str, Any], identity: dict[str, str],
) -> dict[str, Any]:
    if not _owned(existing, identity):
        raise ContractError("refusing to reuse a foreign or differently bound Kubernetes Job")
    if _workload_identity(existing) != _workload_identity(desired):
        raise ContractError("owned Kubernetes Job workload differs from the prepared request")
    return {
        "state": _native_state(existing),
        "backend_ref": f"{identity['namespace']}/{identity['name']}",
        "resumed": True,
        "uid": (existing.get("metadata") or {}).get("uid"),
    }


def _completed_json(completed: subprocess.CompletedProcess[str], label: str) -> dict[str, Any]:
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ContractError(f"{label} returned malformed JSON") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"{label} JSON root must be an object")
    return payload


def submit(args: argparse.Namespace) -> dict[str, Any]:
    request = load_request(args.request)
    _gpu_preflight(request, args.request_timeout_s)
    manifest_text, identity = _render_manifest(args, request)
    dry_run = _kubectl_json(
        ["apply", "--dry-run=server", "-f", "-", "-o", "json"],
        timeout_s=args.request_timeout_s, stdin=manifest_text,
    )
    if dry_run is None or not _owned(dry_run, identity):
        raise ContractError("server dry-run returned a different Job identity")
    existing = _get_job(identity, args.request_timeout_s)
    if existing is not None:
        return _resume_result(existing, dry_run, identity)
    # Create, rather than apply, so a same-name object appearing after the
    # absence check can never be mutated. Reconcile that race by ownership.
    created = _run(
        ["kubectl", "create", "-f", "-", "-o", "json"],
        timeout_s=args.request_timeout_s, stdin=manifest_text, check=False,
    )
    if created.returncode != 0:
        raced = _get_job(identity, args.request_timeout_s)
        if raced is None:
            raise ContractError(
                f"kubectl create failed ({created.returncode}) without a reconcilable Job"
            )
        return _resume_result(raced, dry_run, identity)
    applied = _completed_json(created, "kubectl create")
    if not _owned(applied, identity):
        raise ContractError("kubectl create returned a different or unowned Job")
    return {
        "state": "RUNNING", "backend_ref": f"{identity['namespace']}/{identity['name']}",
        "resumed": False, "uid": (applied.get("metadata") or {}).get("uid"),
    }


def _native_state(job: dict[str, Any]) -> str:
    status = job.get("status") or {}
    for condition in status.get("conditions") or []:
        if condition.get("status") == "True" and condition.get("type") in TERMINAL_CONDITIONS:
            return TERMINAL_CONDITIONS[condition["type"]]
    if int(status.get("failed", 0) or 0) > 0:
        return "ERROR"
    if int(status.get("active", 0) or 0) > 0:
        return "RUNNING"
    if int(status.get("succeeded", 0) or 0) > 0:
        return "COMPLETE"
    return "PENDING"


def status(args: argparse.Namespace) -> dict[str, Any]:
    request = load_request(args.request)
    identity = _identity(request, args.job_id, args.namespace)
    job = _get_job(identity, args.request_timeout_s)
    if job is None:
        return {"state": "UNKNOWN", "message": "Kubernetes Job is absent"}
    if not _owned(job, identity):
        raise ContractError("status discovered a foreign or differently bound Kubernetes Job")
    native = job.get("status") or {}
    return {
        "state": _native_state(job), "backend_ref": f"{identity['namespace']}/{identity['name']}",
        "active": int(native.get("active", 0) or 0),
        "succeeded": int(native.get("succeeded", 0) or 0),
        "failed": int(native.get("failed", 0) or 0),
        "uid": (job.get("metadata") or {}).get("uid"),
    }


def logs(args: argparse.Namespace) -> str:
    request = load_request(args.request)
    identity = _identity(request, args.job_id, args.namespace)
    job = _get_job(identity, args.request_timeout_s)
    if job is None or not _owned(job, identity):
        raise ContractError("logs require the exact owned Kubernetes Job")
    completed = _run(
        ["kubectl", "logs", f"job/{identity['name']}", "-n", identity["namespace"],
         "--all-containers=true", "--tail", str(args.tail)],
        timeout_s=args.request_timeout_s,
    )
    return _redact((completed.stdout or "") + (completed.stderr or ""), request)


def cancel(args: argparse.Namespace) -> dict[str, Any]:
    if not args.confirm:
        raise ContractError("cancel requires --confirm")
    request = load_request(args.request)
    identity = _identity(request, args.job_id, args.namespace)
    job = _get_job(identity, args.request_timeout_s)
    if job is None:
        return {"state": "UNKNOWN", "message": "Kubernetes Job is absent; cancellation is unproven"}
    if not _owned(job, identity):
        raise ContractError("refusing to cancel a foreign or differently bound Kubernetes Job")
    uid = (job.get("metadata") or {}).get("uid")
    _run(
        ["kubectl", "delete", "job", identity["name"], "-n", identity["namespace"],
         "--cascade=foreground", "--wait=true", f"--timeout={args.request_timeout_s}s"],
        timeout_s=args.request_timeout_s,
    )
    if _get_job(identity, args.request_timeout_s) is not None:
        raise ContractError("Kubernetes Job still exists after foreground cancellation")
    return {"state": "CANCELED", "deleted_uid": uid}


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--request", required=True, type=pathlib.Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--namespace", default=os.environ.get("TAO_K8S_NAMESPACE", "default"))
    parser.add_argument("--request-timeout-s", type=int, default=30)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="verb", required=True)
    submit_parser = commands.add_parser("submit")
    _common(submit_parser)
    submit_parser.add_argument("--staging-map", required=True, type=pathlib.Path)
    submit_parser.add_argument("--pvc-claim", required=True)
    submit_parser.add_argument("--credential-secret")
    submit_parser.add_argument("--image-pull-secret")
    submit_parser.add_argument("--ttl-seconds", type=int, default=3600)
    submit_parser.add_argument("--shm-size", default="16Gi")
    for verb in ("status", "logs", "cancel"):
        child = commands.add_parser(verb)
        _common(child)
        if verb == "logs":
            child.add_argument("--tail", type=int, choices=range(1, 1001), default=200)
        elif verb == "cancel":
            child.add_argument("--confirm", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.verb == "submit":
            result: Any = submit(args)
        elif args.verb == "status":
            result = status(args)
        elif args.verb == "logs":
            sys.stdout.write(logs(args))
            return 0
        else:
            result = cancel(args)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ContractError, renderer.RenderError, OSError, subprocess.TimeoutExpired, yaml.YAMLError) as exc:
        print(f"kubernetes action failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
