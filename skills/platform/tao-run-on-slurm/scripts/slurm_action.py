#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaged four-verb consumer for one ordinary SLURM action.

Submission remains owned by :mod:`slurm_submit_action`.  The other verbs prove
that the supplied numeric handle still has the exact signed job name before
reading state, reading its two deterministic logs, or cancelling it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import slurm_submit_action as submit_gate  # noqa: E402


SAFE_TOKEN = submit_gate.SAFE_TOKEN
SAFE_REMOTE_PATH = submit_gate.SAFE_REMOTE_PATH
TERMINAL_STATES = frozenset({
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL",
    "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT",
})
SECRET_RE = re.compile(
    r"(?i)(?:(?:api[_-]?key|token|password)\s*[=:]\s*)[^\s,;]+|"
    r"(?:authorization\s*:\s*bearer\s+)[^\s,;]+|"
    r"(?:hf_|nvapi-)[A-Za-z0-9_.-]{8,}"
)
ACCOUNTING_RECONCILE_TIMEOUT_SECONDS = 10.0
ACCOUNTING_RECONCILE_INTERVAL_SECONDS = 1.0
CANCEL_TIMEOUT_SECONDS = 30.0
CANCEL_INTERVAL_SECONDS = 1.0
SSH_OPERATION_TIMEOUT_SECONDS = 60.0
SYNC_OPERATION_TIMEOUT_SECONDS = 3600.0
MAX_SYNC_LOG_BYTES = 256 * 1024 * 1024


def _sanitize(text: str) -> str:
    return SECRET_RE.sub("[REDACTED]", text)


def _validate_identity(login: str, job_id: str, backend_ref: str) -> None:
    if SAFE_TOKEN.fullmatch(login) is None:
        raise ValueError("--login contains unsupported characters")
    if SAFE_TOKEN.fullmatch(job_id) is None:
        raise ValueError("--job-id contains unsupported characters")
    if not backend_ref.isdigit():
        raise ValueError("--backend-ref must be a numeric SLURM id")


def _ssh(login: str, command: str):
    try:
        return subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                "-o",
                "ConnectionAttempts=1",
                login,
                command,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=SSH_OPERATION_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"bounded SLURM SSH operation exceeded {SSH_OPERATION_TIMEOUT_SECONDS:g}s"
        ) from exc


def _require_ok(result, operation: str) -> bytes:
    if result.returncode != 0:
        detail = _sanitize(result.stderr.decode("utf-8", errors="replace").strip())
        raise ValueError(f"{operation} failed: {detail or 'no diagnostic output'}")
    return result.stdout


def _canonical_sha256(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("request_sha256", None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _local_request(path: pathlib.Path) -> tuple[pathlib.Path, dict[str, Any]]:
    resolved = pathlib.Path(os.path.abspath(path.expanduser()))
    if (
        not resolved.is_file()
        or resolved.is_symlink()
        or resolved.resolve() != resolved
        or resolved.stat().st_size == 0
    ):
        raise ValueError("--request must be one safe nonempty local action request")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("--request is not valid JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("workflow") != "tao-run-deft-iaa"
        or payload.get("platform") != "slurm"
        or payload.get("request_sha256") != _canonical_sha256(payload)
    ):
        raise ValueError("--request has invalid SLURM IAA ownership or digest")
    return resolved, payload


def _remote_path(value: pathlib.Path, label: str) -> pathlib.Path:
    raw = str(value)
    if (
        not value.is_absolute()
        or SAFE_REMOTE_PATH.fullmatch(raw) is None
        or value == pathlib.Path(value.anchor)
        or ".." in value.parts
    ):
        raise ValueError(f"{label} must be one safe non-root absolute remote path")
    return value


def _fetch_remote_file(
    login: str, remote: pathlib.Path, local: pathlib.Path, *, allow_empty: bool = False
) -> None:
    remote = _remote_path(remote, "remote synchronization source")
    local = pathlib.Path(os.path.abspath(local.expanduser()))
    if local == pathlib.Path(local.anchor) or local.resolve(strict=False) != local:
        raise ValueError("local synchronization target is unsafe")
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.parent.is_symlink() or local.parent.resolve() != local.parent:
        raise ValueError("local synchronization parent is unsafe")
    quoted = shlex.quote(str(remote))
    size_check = "test -f {0}; test ! -L {0}; stat -c %s -- {0}".format(quoted)
    size_raw = _require_ok(_ssh(login, size_check), "remote synchronization preflight")
    try:
        size = int(size_raw.decode("utf-8", errors="replace").strip())
    except ValueError as exc:
        raise ValueError("remote synchronization size is malformed") from exc
    if size < 0 or (size == 0 and not allow_empty):
        raise ValueError("remote synchronization source is empty")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=local.name + ".", suffix=".sync.tmp", dir=local.parent
    )
    os.close(descriptor)
    temporary = pathlib.Path(temporary_name)
    try:
        completed = subprocess.run(
            ["scp", "-q", "--", f"{login}:{remote}", str(temporary)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=SYNC_OPERATION_TIMEOUT_SECONDS,
        )
        _require_ok(completed, "remote synchronization copy")
        if temporary.is_symlink() or not temporary.is_file() or temporary.stat().st_size != size:
            raise ValueError("remote synchronization copy size or type differs")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, local)
        directory = os.open(local.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _capture_native_log(
    *, login: str, job_id: str, backend_ref: str, log_dir: pathlib.Path,
    destination: pathlib.Path,
) -> None:
    log_dir = _log_dir(log_dir)
    sources = (
        log_dir / f"{job_id}-{backend_ref}.out",
        log_dir / f"{job_id}-{backend_ref}.err",
    )
    destination = pathlib.Path(os.path.abspath(destination.expanduser()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="slurm-log-sync-", dir=destination.parent) as raw:
        temporary_dir = pathlib.Path(raw)
        local_parts: list[pathlib.Path] = []
        total = 0
        for index, source in enumerate(sources):
            target = temporary_dir / f"part-{index}.log"
            _fetch_remote_file(login, source, target, allow_empty=True)
            total += target.stat().st_size
            if total > MAX_SYNC_LOG_BYTES:
                raise ValueError("combined SLURM action log exceeds the bounded 256 MiB limit")
            local_parts.append(target)
        text = "".join(
            f"SLURM_LOG={source.name}\n"
            + part.read_text(encoding="utf-8", errors="replace")
            + "\n"
            for source, part in zip(sources, local_parts)
        )
        text = _sanitize(text)
        if not text.strip():
            raise ValueError("combined SLURM action log is empty")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=destination.name + ".", suffix=".sync.tmp", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _sync_complete_action(args: argparse.Namespace) -> dict[str, Any]:
    _request_path, request = _local_request(args.request)
    local_results = pathlib.Path(str(request.get("results_dir", "")))
    if not local_results.is_absolute() or local_results.resolve(strict=False) != local_results:
        raise ValueError("action request results_dir is unsafe")
    remote_results = _remote_path(args.remote_results, "--remote-results")
    synchronized: list[str] = []
    for raw in request.get("fresh_outputs", []):
        local = pathlib.Path(str(raw))
        if not local.is_absolute() or local.resolve(strict=False) != local:
            raise ValueError("action request fresh output path is unsafe")
        try:
            relative = local.relative_to(local_results)
        except ValueError as exc:
            raise ValueError("action request fresh output escapes results_dir") from exc
        _fetch_remote_file(args.login, remote_results / relative, local)
        synchronized.append(str(local))
    log_path = pathlib.Path(str(request.get("log_path", "")))
    try:
        log_path.relative_to(local_results)
    except ValueError as exc:
        raise ValueError("action request log path escapes results_dir") from exc
    _capture_native_log(
        login=args.login, job_id=args.job_id, backend_ref=args.backend_ref,
        log_dir=args.log_dir, destination=log_path,
    )
    return {"outputs": synchronized, "log_path": str(log_path)}


def _sync_terminal_log(args: argparse.Namespace) -> dict[str, Any]:
    _request_path, request = _local_request(args.request)
    local_results = pathlib.Path(str(request.get("results_dir", "")))
    log_path = pathlib.Path(str(request.get("log_path", "")))
    if (
        not local_results.is_absolute()
        or local_results.resolve(strict=False) != local_results
        or not log_path.is_absolute()
        or log_path.resolve(strict=False) != log_path
    ):
        raise ValueError("action request terminal log paths are unsafe")
    try:
        log_path.relative_to(local_results)
    except ValueError as exc:
        raise ValueError("action request log path escapes results_dir") from exc
    _capture_native_log(
        login=args.login, job_id=args.job_id, backend_ref=args.backend_ref,
        log_dir=args.log_dir, destination=log_path,
    )
    remote_results = _remote_path(args.remote_results, "--remote-results")
    diagnostics: list[str] = []
    for raw in request.get("fresh_outputs", []):
        local = pathlib.Path(str(raw))
        try:
            relative = local.relative_to(local_results)
        except ValueError as exc:
            raise ValueError("action request diagnostic output escapes results_dir") from exc
        remote = remote_results / relative
        quoted = shlex.quote(str(remote))
        probe = _require_ok(
            _ssh(
                args.login,
                f"if test -f {quoted} && test ! -L {quoted}; then stat -c %s -- {quoted}; "
                "else printf 'ABSENT\\n'; fi",
            ),
            "remote diagnostic output probe",
        ).decode("utf-8", errors="replace").strip()
        if probe == "ABSENT":
            continue
        try:
            size = int(probe)
        except ValueError as exc:
            raise ValueError("remote diagnostic output size is malformed") from exc
        if size <= 0:
            continue
        _fetch_remote_file(args.login, remote, local)
        diagnostics.append(str(local))
    return {"log_path": str(log_path), "diagnostic_outputs": diagnostics}


def _sync_unstarted_cancel(args: argparse.Namespace) -> dict[str, Any]:
    """Record authoritative zero-execution cancellation without native logs."""

    _request_path, request = _local_request(args.request)
    local_results = pathlib.Path(str(request.get("results_dir", "")))
    log_path = pathlib.Path(str(request.get("log_path", "")))
    if (
        not local_results.is_absolute()
        or local_results.resolve(strict=False) != local_results
        or not log_path.is_absolute()
        or log_path.resolve(strict=False) != log_path
    ):
        raise ValueError("action request canceled log paths are unsafe")
    try:
        log_path.relative_to(local_results)
    except ValueError as exc:
        raise ValueError("action request log path escapes results_dir") from exc
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.parent.is_symlink() or log_path.parent.resolve() != log_path.parent:
        raise ValueError("action request canceled log parent is unsafe")
    text = (
        f"SLURM_JOB_ID={args.backend_ref}\n"
        f"SLURM_JOB_NAME={args.job_id}\n"
        "NATIVE_STATE=CANCELLED\n"
        "ELAPSED_RAW=0\n"
        "DETAIL=job canceled before execution; native stdout/stderr were not created\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=log_path.name + ".", suffix=".sync.tmp", dir=log_path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, log_path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return {
        "log_path": str(log_path), "diagnostic_outputs": [],
        "canceled_before_start": True,
    }


def _assert_job_ownership(login: str, job_id: str, backend_ref: str) -> None:
    """Prove the scheduler handle still resolves to exactly ``job_id``."""

    _validate_identity(login, job_id, backend_ref)
    command = (
        "set -Eeuo pipefail; "
        f"name=$(scontrol show job -o {backend_ref} 2>/dev/null | "
        "sed -n 's/.* JobName=\\([^ ]*\\).*/\\1/p' | head -n1 || true); "
        f"if [ -z \"$name\" ]; then name=$(sacct -j {backend_ref} -X -n -o JobName%128 "
        "2>/dev/null | awk 'NF {print $1; exit}' || true); fi; "
        "printf 'TAO_JOB_NAME=%s\\n' \"$name\""
    )
    output = _require_ok(_ssh(login, command), "SLURM ownership query").decode(
        "utf-8", errors="replace"
    )
    names = re.findall(r"^TAO_JOB_NAME=([^\s]+)$", output, re.MULTILINE)
    if names != [job_id]:
        raise ValueError("SLURM handle is absent or not owned by the exact job name")


def _one_native_state(output: bytes, source: str) -> str | None:
    rows = [
        row.strip()
        for row in output.decode("utf-8", errors="replace").splitlines()
        if row.strip()
    ]
    if not rows:
        return None
    state = rows[0].split()[0].rstrip("+").upper()
    if not re.fullmatch(r"[A-Z_]+", state):
        raise ValueError(f"{source} returned an invalid SLURM state")
    return state


def _native_state(
    login: str,
    backend_ref: str,
    *,
    accounting_timeout_s: float = ACCOUNTING_RECONCILE_TIMEOUT_SECONDS,
) -> str:
    queued = _require_ok(
        _ssh(login, f"squeue -h -j {backend_ref} -o '%T' 2>/dev/null || true"),
        "SLURM queue status",
    )
    state = _one_native_state(queued, "squeue")
    if state is not None:
        return state

    deadline = time.monotonic() + accounting_timeout_s
    while True:
        accounting = _require_ok(
            _ssh(
                login,
                f"sacct -j {backend_ref} -X -n -o State%30 2>/dev/null || true",
            ),
            "SLURM accounting status",
        )
        state = _one_native_state(accounting, "sacct")
        if state is not None:
            return state
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "UNKNOWN"
        time.sleep(min(ACCOUNTING_RECONCILE_INTERVAL_SECONDS, remaining))


def _native_exit_code(login: str, backend_ref: str) -> int:
    output = _require_ok(
        _ssh(
            login,
            f"sacct -j {backend_ref} -X -n -o ExitCode -P 2>/dev/null || true",
        ),
        "SLURM accounting exit-code query",
    ).decode("utf-8", errors="replace")
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if not rows:
        raise ValueError("SLURM accounting returned no terminal exit code")
    match = re.fullmatch(r"([0-9]+):[0-9]+", rows[0])
    if match is None:
        raise ValueError("SLURM accounting returned a malformed exit code")
    return int(match.group(1))


def _native_elapsed_raw(login: str, backend_ref: str) -> int:
    output = _require_ok(
        _ssh(
            login,
            f"sacct -j {backend_ref} -X -n -o ElapsedRaw -P 2>/dev/null || true",
        ),
        "SLURM accounting elapsed query",
    ).decode("utf-8", errors="replace")
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if len(rows) != 1 or not rows[0].isdigit():
        raise ValueError("SLURM accounting returned malformed elapsed evidence")
    return int(rows[0])


def map_state(native_state: str) -> str:
    if native_state in {"PENDING", "CONFIGURING"}:
        return "PENDING"
    if native_state in {"RUNNING", "COMPLETING", "SUSPENDED", "STOPPED"}:
        return "RUNNING"
    if native_state == "COMPLETED":
        return "COMPLETE"
    if native_state in {"CANCELLED", "PREEMPTED", "REVOKED"}:
        return "CANCELED"
    if native_state in {
        "FAILED", "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY", "NODE_FAIL", "TIMEOUT",
    }:
        return "ERROR"
    return "UNKNOWN"


def submit(args: argparse.Namespace) -> dict[str, Any]:
    return submit_gate.submit_action(
        login=args.login,
        job_id=args.job_id,
        rendered_script=args.rendered_script,
        remote_script=args.remote_script,
        request_path=args.request,
        binding_path=args.job_binding,
    )


def status(args: argparse.Namespace) -> dict[str, Any]:
    try:
        _assert_job_ownership(args.login, args.job_id, args.backend_ref)
        native = _native_state(args.login, args.backend_ref)
    except TimeoutError:
        return {
            "backend_ref": args.backend_ref,
            "job_id": args.job_id,
            "native_state": "SSH_TIMEOUT",
            "status": "UNKNOWN",
            "classification": "transient_ssh_timeout",
        }
    payload = {
        "backend_ref": args.backend_ref,
        "job_id": args.job_id,
        "native_state": native,
        "status": map_state(native),
    }
    sync_values = (
        getattr(args, "request", None), getattr(args, "remote_results", None),
        getattr(args, "log_dir", None),
    )
    if any(value is not None for value in sync_values):
        if not all(value is not None for value in sync_values):
            raise ValueError(
                "terminal synchronization requires --request, --remote-results, and --log-dir"
            )
        if payload["status"] == "COMPLETE":
            payload["synchronized"] = _sync_complete_action(args)
        elif payload["status"] in {"ERROR", "CANCELED"}:
            if (
                payload["status"] == "CANCELED"
                and _native_elapsed_raw(args.login, args.backend_ref) == 0
            ):
                payload["synchronized"] = _sync_unstarted_cancel(args)
            else:
                payload["synchronized"] = _sync_terminal_log(args)
            if payload["status"] == "ERROR":
                payload["native_exit_code"] = _native_exit_code(
                    args.login, args.backend_ref
                )
    return payload


def _log_dir(value: pathlib.Path) -> pathlib.Path:
    raw = str(value)
    if (
        not value.is_absolute()
        or SAFE_REMOTE_PATH.fullmatch(raw) is None
        or value == pathlib.Path(value.anchor)
        or ".." in value.parts
    ):
        raise ValueError("--log-dir must be one safe non-root absolute remote path")
    return value


def logs(args: argparse.Namespace) -> dict[str, Any]:
    _assert_job_ownership(args.login, args.job_id, args.backend_ref)
    log_dir = _log_dir(args.log_dir)
    paths = [
        log_dir / f"{args.job_id}-{args.backend_ref}.out",
        log_dir / f"{args.job_id}-{args.backend_ref}.err",
    ]
    command = "tail -n {} -- {} 2>/dev/null || true".format(
        args.tail, " ".join(shlex.quote(str(path)) for path in paths)
    )
    text = _sanitize(
        _require_ok(_ssh(args.login, command), "SLURM log fetch").decode(
            "utf-8", errors="replace"
        )
    )
    return {
        "backend_ref": args.backend_ref,
        "job_id": args.job_id,
        "log_paths": [str(path) for path in paths],
        "text": text,
    }


def cancel(args: argparse.Namespace) -> dict[str, Any]:
    if not args.confirm:
        raise ValueError("cancel requires --confirm")
    _assert_job_ownership(args.login, args.job_id, args.backend_ref)
    native = _native_state(args.login, args.backend_ref)
    already_terminal = native in TERMINAL_STATES
    if native not in TERMINAL_STATES:
        _require_ok(
            _ssh(args.login, f"scancel {args.backend_ref}"),
            "SLURM cancellation",
        )
        deadline = time.monotonic() + args.timeout
        while native not in TERMINAL_STATES:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"SLURM cancellation did not terminalize within {args.timeout}s"
                )
            time.sleep(min(CANCEL_INTERVAL_SECONDS, remaining))
            native = _native_state(
                args.login,
                args.backend_ref,
                accounting_timeout_s=min(
                    ACCOUNTING_RECONCILE_TIMEOUT_SECONDS,
                    max(0.0, deadline - time.monotonic()),
                ),
            )
    return {
        "already_terminal": already_terminal,
        "backend_ref": args.backend_ref,
        "job_id": args.job_id,
        "native_state": native,
        "status": map_state(native),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="verb", required=True)

    submit_parser = sub.add_parser("submit")
    submit_parser.add_argument("--login", required=True)
    submit_parser.add_argument("--job-id", required=True)
    submit_parser.add_argument("--rendered-script", required=True, type=pathlib.Path)
    submit_parser.add_argument("--remote-script", required=True, type=pathlib.Path)
    submit_parser.add_argument("--request", type=pathlib.Path)
    submit_parser.add_argument("--job-binding", type=pathlib.Path)

    for verb in ("status", "logs", "cancel"):
        child = sub.add_parser(verb)
        child.add_argument("--login", required=True)
        child.add_argument("--job-id", required=True)
        child.add_argument("--backend-ref", required=True)
        if verb == "status":
            child.add_argument("--request", type=pathlib.Path)
            child.add_argument("--remote-results", type=pathlib.Path)
            child.add_argument("--log-dir", type=pathlib.Path)
        if verb == "logs":
            child.add_argument("--log-dir", required=True, type=pathlib.Path)
            child.add_argument("--tail", type=int, default=200, choices=range(1, 10001))
        if verb == "cancel":
            child.add_argument("--confirm", action="store_true")
            child.add_argument(
                "--timeout",
                type=int,
                default=int(CANCEL_TIMEOUT_SECONDS),
                choices=range(1, 301),
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = {
            "submit": submit,
            "status": status,
            "logs": logs,
            "cancel": cancel,
        }[args.verb](args)
    except (OSError, TimeoutError, ValueError) as exc:
        print(f"slurm_action[{args.verb}]: {_sanitize(str(exc))}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
