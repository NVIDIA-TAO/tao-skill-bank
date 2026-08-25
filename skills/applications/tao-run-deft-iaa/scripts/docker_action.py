#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Four-verb local Docker consumer for immutable IAA action requests."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Any, Sequence

from run_deft_action import load_bound_action_for_submit


REPO = pathlib.Path(__file__).resolve().parents[4]
DOCKER_SCRIPTS = REPO / "skills/platform/tao-run-on-docker/scripts"
sys.path.insert(0, str(DOCKER_SCRIPTS))
import render_iaa_adapter  # noqa: E402
import render_iaa_model_action  # noqa: E402


WORKFLOW = "tao-run-deft-iaa"
SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
SHA256 = re.compile(r"[0-9a-f]{64}")
SECRET_ENV_NAMES = (
    "AIRFLOW_API_TOKEN", "AIRFLOW_PASSWORD", "BREV_API_TOKEN", "HF_TOKEN", "NGC_KEY",
)


class DockerActionError(RuntimeError):
    """A bounded Docker operation or ownership check failed."""


def _run(argv: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise DockerActionError(f"Docker operation timed out after {timeout}s") from exc


def _load_bound(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        request, binding, job = load_bound_action_for_submit(
            args.request, args.job_binding, args.job_record,
        )
    except (OSError, ValueError) as exc:
        raise DockerActionError(f"invalid bound IAA action: {exc}") from exc
    if request.get("workflow") != WORKFLOW or request.get("platform") != "docker":
        raise DockerActionError("local Docker accepts only Docker IAA action requests")
    return request, binding, job


def _backend_ref(job_id: str, action: str, request_sha256: str) -> str:
    if (
        SAFE_COMPONENT.fullmatch(job_id) is None
        or SAFE_COMPONENT.fullmatch(action) is None
        or SHA256.fullmatch(request_sha256) is None
    ):
        raise DockerActionError("request cannot form a safe owned Docker backend reference")
    return f"docker/{job_id}/{action}/{request_sha256}"


def _parse_backend_ref(value: str) -> tuple[str, str, str]:
    parts = value.split("/")
    if len(parts) != 4 or parts[0] != "docker":
        raise DockerActionError("Docker backend_ref must be docker/<job>/<action>/<request-sha256>")
    _backend_ref(parts[1], parts[2], parts[3])
    return parts[1], parts[2], parts[3]


def _inspect(name: str) -> dict[str, Any] | None:
    result = _run(["docker", "inspect", name], timeout=30)
    if result.returncode != 0:
        lowered = result.stderr.lower()
        if "no such object" in lowered or "no such container" in lowered:
            return None
        raise DockerActionError("Docker inspect failed")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise DockerActionError("Docker inspect returned malformed JSON") from exc
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise DockerActionError("Docker inspect returned an unexpected object count")
    return payload[0]


def _owned_container(backend_ref: str, *, missing_ok: bool = False) -> dict[str, Any] | None:
    job_id, action, request_sha256 = _parse_backend_ref(backend_ref)
    payload = _inspect(job_id)
    if payload is None:
        if missing_ok:
            return None
        raise DockerActionError("owned Docker container is missing")
    labels = payload.get("Config", {}).get("Labels")
    expected = {
        "tao-job": job_id,
        "tao-action": action,
        "tao-request-sha256": request_sha256,
    }
    if not isinstance(labels, dict) or any(labels.get(key) != value for key, value in expected.items()):
        raise DockerActionError("Docker container labels do not match the exact IAA request/job")
    return payload


def _state(payload: dict[str, Any] | None) -> tuple[str, str | None]:
    if payload is None:
        return "UNKNOWN", None
    state = payload.get("State")
    if not isinstance(state, dict):
        return "UNKNOWN", None
    native = state.get("Status")
    if native in {"created", "restarting"}:
        return "PENDING", native
    if native in {"running", "paused"}:
        return "RUNNING", native
    if native in {"exited", "dead"}:
        return ("COMPLETE" if state.get("ExitCode") == 0 else "ERROR"), native
    if native == "removing":
        return "CANCELED", native
    return "UNKNOWN", native if isinstance(native, str) else None


def _renderer(request: dict[str, Any]):
    if request.get("name") in render_iaa_adapter.ADAPTER_ACTIONS:
        return render_iaa_adapter.render_argv
    return render_iaa_model_action.render_argv


def submit(args: argparse.Namespace) -> int:
    request, _binding, job = _load_bound(args)
    backend_ref = _backend_ref(job["id"], request["name"], request["request_sha256"])
    existing = _owned_container(backend_ref, missing_ok=True)
    reconciled = existing is not None
    if existing is None:
        try:
            argv = _renderer(request)(request, job["id"])
        except (OSError, ValueError) as exc:
            raise DockerActionError(f"Docker request rendering failed: {exc}") from exc
        if "--gpus" in argv:
            selector = argv[argv.index("--gpus") + 1]
            if selector == "all" or "device=" not in selector:
                raise DockerActionError("renderer did not preserve explicit Docker GPU IDs")
        result = _run(argv, timeout=900)
        if result.returncode != 0:
            raise DockerActionError("Docker submit failed; inspect the Docker daemon logs")
        existing = _owned_container(backend_ref)
    status_value, native = _state(existing)
    print(json.dumps({
        "backend_ref": backend_ref, "status": status_value,
        "native_state": native, "reconciled": reconciled,
    }, sort_keys=True))
    return 0


def status(args: argparse.Namespace) -> int:
    payload = _owned_container(args.backend_ref, missing_ok=True)
    status_value, native = _state(payload)
    print(json.dumps({
        "backend_ref": args.backend_ref, "status": status_value, "native_state": native,
    }, sort_keys=True))
    return 0


def _redact(text: str) -> str:
    for name in SECRET_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            text = text.replace(value, "[REDACTED]")
    return re.sub(
        r"(?i)(api[_-]?key|token|password)\s*[=:]\s*\S+", r"\1=<redacted>", text,
    )


def logs(args: argparse.Namespace) -> int:
    job_id, _action, _digest = _parse_backend_ref(args.backend_ref)
    _owned_container(args.backend_ref)
    result = _run(["docker", "logs", "--tail", str(args.tail), job_id], timeout=60)
    output = result.stdout + result.stderr
    if output:
        print(_redact(output), end="" if output.endswith("\n") else "\n")
    if result.returncode != 0:
        raise DockerActionError("Docker logs failed")
    return 0


def cancel(args: argparse.Namespace) -> int:
    if not args.confirm:
        raise DockerActionError("cancel requires --confirm after approval")
    job_id, _action, _digest = _parse_backend_ref(args.backend_ref)
    payload = _owned_container(args.backend_ref)
    mapped, native = _state(payload)
    if mapped in {"COMPLETE", "ERROR"}:
        raise DockerActionError(f"cannot cancel terminal Docker state {mapped}")
    result = _run(["docker", "stop", "--time", "30", job_id], timeout=60)
    if result.returncode != 0:
        raise DockerActionError("Docker cancel failed")
    print(json.dumps({
        "backend_ref": args.backend_ref, "status": "CANCELED", "native_state": native,
    }, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="verb", required=True)
    submit_parser = sub.add_parser("submit")
    submit_parser.add_argument("--request", required=True, type=pathlib.Path)
    submit_parser.add_argument("--job-binding", required=True, type=pathlib.Path)
    submit_parser.add_argument("--job-record", required=True, type=pathlib.Path)
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
        return {"submit": submit, "status": status, "logs": logs, "cancel": cancel}[args.verb](args)
    except (DockerActionError, OSError) as exc:
        print(f"docker_action: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
