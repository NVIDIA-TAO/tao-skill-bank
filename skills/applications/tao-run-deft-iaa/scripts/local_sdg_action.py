#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Four-verb local Docker/virtualenv consumer for one composite IAA SDG stage.

The worker is detached into its own process group so an Airflow task restart
can reconcile it and explicit cancellation can terminate only the exact owned
controller before stopping (never removing) its owned model containers.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

import airflow_dag_runtime as runtime
import airflow_sdg_action as producer


TERMINAL = frozenset({"COMPLETE", "ERROR", "CANCELED"})
BACKEND_REF = re.compile(r"pid:(?P<pid>[1-9][0-9]*):(?P<start>[1-9][0-9]*)")


class LocalSdgError(ValueError):
    pass


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


def _redact(text: str) -> str:
    for name in ("HF_TOKEN", "NGC_KEY", "AIRFLOW_API_TOKEN", "AIRFLOW_PASSWORD"):
        if value := os.environ.get(name):
            text = text.replace(value, "[REDACTED]")
    return re.sub(
        r"(?i)(api[_-]?key|token|password)\s*[=:]\s*\S+",
        r"\1=<redacted>",
        text,
    )


def _status_path(request: dict[str, Any]) -> pathlib.Path:
    stage = pathlib.Path(request["paths"]["stage_dir"])
    return stage / "local-sdg.status.json"


def _worker_log(request: dict[str, Any]) -> pathlib.Path:
    return pathlib.Path(request["paths"]["stage_dir"]) / "local-sdg.worker.log"


def _proc_start(pid: int) -> int | None:
    path = pathlib.Path(f"/proc/{pid}/stat")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    close = text.rfind(")")
    if close < 1:
        return None
    fields = text[close + 2 :].split()
    try:
        return int(fields[19])
    except (IndexError, ValueError):
        return None


def _backend_ref(pid: int, start: int) -> str:
    return f"pid:{pid}:{start}"


def _parse_backend_ref(value: str) -> tuple[int, int]:
    match = BACKEND_REF.fullmatch(value)
    if match is None:
        raise LocalSdgError("backend_ref must be pid:<pid>:<start-marker>")
    return int(match.group("pid")), int(match.group("start"))


def _owned_process(pid: int, start: int) -> bool:
    if _proc_start(pid) != start:
        return False
    try:
        command = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        return False
    return b"local_sdg_action.py" in command and b"_worker" in command


def _load_status(request: dict[str, Any]) -> dict[str, Any] | None:
    path = _status_path(request)
    if not path.exists():
        return None
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise LocalSdgError("local SDG status evidence is unsafe")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LocalSdgError("local SDG status evidence is malformed") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("request_sha256") != request["request_sha256"]
        or payload.get("platform") != request["platform"]
    ):
        raise LocalSdgError("local SDG status ownership differs from the request")
    return payload


def _write_status(
    request: dict[str, Any], *, state: str, backend_ref: str,
    message: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": "1", "workflow": producer.WORKFLOW,
        "kind": "local_sdg_action", "platform": request["platform"],
        "request_sha256": request["request_sha256"], "state": state,
        "backend_ref": backend_ref,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "message": _redact(message) if message else None,
    }
    _atomic_json(_status_path(request), payload)
    return payload


def _conf(
    request: dict[str, Any], binding: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    return {
        "contract": producer.airflow.CONTRACT,
        "kind": producer.KIND,
        "job_id": job["id"],
        "request_sha256": request["request_sha256"],
        "binding_sha256": binding["binding_sha256"],
        "request": request,
    }


def worker(args: argparse.Namespace) -> int:
    request_path, request = producer.load_request(args.request)
    job_path, job = producer._load_job(args.job_record, request)
    binding = producer._bind_job(request_path, request, job_path, job)
    pid = os.getpid()
    start = _proc_start(pid)
    if start is None:
        raise LocalSdgError("cannot bind the local SDG worker process identity")
    backend_ref = _backend_ref(pid, start)
    _write_status(request, state="RUNNING", backend_ref=backend_ref)
    try:
        runtime.execute_sdg(_conf(request, binding, job))
    except BaseException as exc:
        _write_status(
            request, state="ERROR", backend_ref=backend_ref,
            message=f"{type(exc).__name__}: {exc}",
        )
        raise
    _write_status(request, state="COMPLETE", backend_ref=backend_ref)
    return 0


def submit(args: argparse.Namespace) -> int:
    _, request = producer.load_request(args.request)
    if request["platform"] not in {"docker", "virtualenv"} or request.get("orchestrator") != "airflow":
        raise LocalSdgError("local SDG consumer requires Airflow-orchestrated docker/virtualenv")
    existing = _load_status(request)
    if existing is not None:
        backend_ref = existing.get("backend_ref")
        pid, start = _parse_backend_ref(str(backend_ref))
        state = existing.get("state")
        if state == "COMPLETE" or (state == "RUNNING" and _owned_process(pid, start)):
            print(json.dumps({"backend_ref": backend_ref, "status": state, "reconciled": True}, sort_keys=True))
            return 0
        raise LocalSdgError(
            "existing local SDG status is terminal failure or has lost process ownership; "
            "use the workflow's bounded corrected attempt"
        )
    log = _worker_log(request)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as handle:
        process = subprocess.Popen(
            [
                sys.executable, str(pathlib.Path(__file__).resolve()), "_worker",
                "--request", str(args.request), "--job-record", str(args.job_record),
            ],
            stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ),
        )
    deadline = time.monotonic() + 10
    evidence = None
    while time.monotonic() < deadline:
        evidence = _load_status(request)
        if evidence is not None:
            break
        if process.poll() is not None:
            raise LocalSdgError(
                f"local SDG worker exited during submit with {process.returncode}; inspect {log}"
            )
        time.sleep(0.05)
    if evidence is None:
        raise LocalSdgError("local SDG worker did not publish ownership within 10 seconds")
    print(json.dumps({
        "backend_ref": evidence["backend_ref"], "status": evidence["state"],
        "reconciled": False,
    }, sort_keys=True))
    return 0


def status(args: argparse.Namespace) -> int:
    _, request = producer.load_request(args.request)
    evidence = _load_status(request)
    if evidence is None:
        print(json.dumps({"status": "UNKNOWN", "native_state": "missing"}, sort_keys=True))
        return 0
    pid, start = _parse_backend_ref(str(evidence.get("backend_ref")))
    state = evidence.get("state")
    if state == "RUNNING" and not _owned_process(pid, start):
        state = "UNKNOWN"
    print(json.dumps({
        "backend_ref": evidence["backend_ref"], "status": state,
        "native_state": evidence.get("state"),
    }, sort_keys=True))
    return 0


def logs(args: argparse.Namespace) -> int:
    _, request = producer.load_request(args.request)
    paths = [_worker_log(request), pathlib.Path(request["paths"]["stage_dir"]) / "airflow_sdg.log"]
    rows: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        if not path.is_file() or path.is_symlink():
            raise LocalSdgError(f"local SDG log is unsafe: {path}")
        rows.extend(path.read_text(encoding="utf-8", errors="replace").splitlines())
    print(_redact("\n".join(rows[-args.tail :])))
    return 0


def cancel(args: argparse.Namespace) -> int:
    if not args.confirm:
        raise LocalSdgError("cancel requires --confirm")
    _, request = producer.load_request(args.request)
    evidence = _load_status(request)
    if evidence is None:
        print(json.dumps({"status": "UNKNOWN", "native_state": "missing"}, sort_keys=True))
        return 0
    pid, start = _parse_backend_ref(str(evidence.get("backend_ref")))
    state = evidence.get("state")
    if state == "RUNNING" and _owned_process(pid, start):
        os.killpg(pid, signal.SIGTERM)
        deadline = time.monotonic() + 30
        while _owned_process(pid, start) and time.monotonic() < deadline:
            time.sleep(0.1)
        if _owned_process(pid, start):
            os.killpg(pid, signal.SIGKILL)
            deadline = time.monotonic() + 5
            while _owned_process(pid, start) and time.monotonic() < deadline:
                time.sleep(0.1)
        if _owned_process(pid, start):
            raise LocalSdgError("owned local SDG process group did not terminate")
    config = pathlib.Path(request["paths"]["config_path"])
    runtime_root = pathlib.Path(request["paths"]["runtime_root"])
    stop = subprocess.run(
        [
            sys.executable, str(runtime_root / "manage_sdg_endpoints.py"), "stop",
            "--config", str(config), "--run-id", request["run_id"],
            "--platform", request["platform"],
        ],
        capture_output=True, text=True, check=False, env=dict(os.environ),
    )
    if stop.returncode != 0:
        raise LocalSdgError("owned endpoint stop failed: " + _redact(stop.stderr.strip()))
    _write_status(
        request, state="CANCELED", backend_ref=evidence["backend_ref"],
        message="owned controller stopped; owned endpoint containers retained",
    )
    print(json.dumps({"backend_ref": evidence["backend_ref"], "status": "CANCELED"}, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="verb", required=True)
    for verb in ("submit", "status", "logs", "cancel", "_worker"):
        child = sub.add_parser(verb)
        child.add_argument("--request", required=True, type=pathlib.Path)
        if verb in {"submit", "_worker"}:
            child.add_argument("--job-record", required=True, type=pathlib.Path)
        if verb == "logs":
            child.add_argument("--tail", type=int, default=200, choices=range(1, 10001))
        if verb == "cancel":
            child.add_argument("--confirm", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return {
            "submit": submit, "status": status, "logs": logs,
            "cancel": cancel, "_worker": worker,
        }[args.verb](args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"local sdg action failed: {_redact(str(exc))}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
