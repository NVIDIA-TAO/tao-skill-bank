#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate, stage, test, and submit one rendered SLURM action script."""

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
import time
from typing import Sequence


SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.@-]+$")
SAFE_REMOTE_PATH = re.compile(r"^/[A-Za-z0-9_./-]+$")
JOB_HANDLE = re.compile(r"^(?P<id>[0-9]+)(?:;[A-Za-z0-9_.-]+)?$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
CONTAINER_ENV_NAMES = (
    "NCCL_DEBUG",
    "LOGLEVEL",
    "NCCL_P2P_DISABLE",
    "NCCL_IB_DISABLE",
    "NCCL_SOCKET_IFNAME",
    "NCCL_IB_HCA",
    "NCCL_NET",
)
IAA_COMPUTE_FRAME_ENV = "IAA_COMPUTE_FRAME"
ADAPTER_ENVIRONMENT = {
    "HF_HOME": "/cache/huggingface",
    "HOME": "/tmp",
    IAA_COMPUTE_FRAME_ENV: "slurm",
    "PYTHONPATH": "/patches",
    "XDG_CACHE_HOME": "/cache",
}
ADAPTER_CONTAINER_ENV_NAMES = tuple(ADAPTER_ENVIRONMENT)
VISUALIZE_THREAD_CAPS = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
CONTAINER_ENV_RE = re.compile(r"--container-env(?:=|\s+)([^\s\\]+)")
SUBMIT_RECONCILE_TIMEOUT_SECONDS = 60.0
SUBMIT_RECONCILE_INTERVAL_SECONDS = 2.0
IAA_CLIP_TRAIN_JOB = re.compile(r"^clip-deft-iaa-train-[A-Za-z0-9-]+$")


def _shell_literal(literal: str) -> str:
    """Return a regex matching one optionally quoted, exact shell token."""

    escaped = re.escape(literal)
    return rf'(?:{escaped}|"{escaped}"|\'{escaped}\')'


IAA_CLIP_TRAIN_WRAPPED = re.compile(
    _shell_literal("/patches/run_clip_train_slurm.sh")
    + r"\s+"
    + _shell_literal("clip")
    + r"\s+"
    + _shell_literal("train")
)
JOB_IDENTITY_FIELDS = (
    "schema_version",
    "id",
    "platform",
    "image",
    "network_arch",
    "action",
    "results_dir",
    "storage_tier",
    "upload_excludes",
    "submitted_at",
)


def _run(
    argv: Sequence[str], *, input_bytes: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(argv),
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _require_ok(result: subprocess.CompletedProcess[bytes], operation: str) -> bytes:
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"{operation} failed: {detail or 'no diagnostic output'}")
    return result.stdout


def _ssh(login: str, command: str) -> subprocess.CompletedProcess[bytes]:
    return _run(["ssh", "-o", "BatchMode=yes", login, command])


def _canonical_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _remote_json(login: str, path: pathlib.Path, name: str) -> dict[str, object]:
    raw = str(path)
    if not path.is_absolute() or not SAFE_REMOTE_PATH.fullmatch(raw):
        raise ValueError(f"{name} must be one safe absolute remote path")
    if path == pathlib.Path(path.anchor) or ".." in path.parts:
        raise ValueError(f"{name} target is too broad or traversing")
    quoted = shlex.quote(raw)
    output = _require_ok(
        _ssh(
            login,
            f"set -Eeuo pipefail; test -f {quoted}; test ! -L {quoted}; "
            f"test -s {quoted}; cat -- {quoted}",
        ),
        f"{name} read",
    )
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} JSON root must be an object")
    return payload


def _local_json(path: pathlib.Path, name: str) -> dict[str, object]:
    resolved = pathlib.Path(os.path.abspath(path.expanduser()))
    if (
        not resolved.is_file()
        or resolved.is_symlink()
        or resolved.resolve() != resolved
        or resolved.stat().st_size == 0
    ):
        raise ValueError(f"{name} must be one safe nonempty local file")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} JSON root must be an object")
    return payload


def _validate_iaa_job_binding(
    *, login: str, job_id: str, request_path: pathlib.Path, binding_path: pathlib.Path
) -> tuple[dict[str, object], dict[str, str] | None]:
    """Prove one immutable IAA request/job binding before scheduler submit."""

    del login  # Control evidence is shared with the Airflow worker, not Lustre-mounted.
    request_path = pathlib.Path(os.path.abspath(request_path.expanduser()))
    binding_path = pathlib.Path(os.path.abspath(binding_path.expanduser()))
    request = _local_json(request_path, "IAA action request")
    binding = _local_json(binding_path, "IAA job binding")
    if request.get("schema_version") != "1":
        raise ValueError("IAA action request schema_version is unsupported")
    if binding.get("schema_version") != "1":
        raise ValueError("IAA job binding schema_version is unsupported")
    request_body = dict(request)
    request_digest = request_body.pop("request_sha256", None)
    if not isinstance(request_digest, str) or request_digest != _canonical_sha256(request_body):
        raise ValueError("IAA action request digest mismatch")
    binding_body = dict(binding)
    binding_digest = binding_body.pop("binding_sha256", None)
    if not isinstance(binding_digest, str) or binding_digest != _canonical_sha256(binding_body):
        raise ValueError("IAA job binding digest mismatch")
    job_record_value = binding.get("job_record_path")
    if not isinstance(job_record_value, str):
        raise ValueError("IAA job binding lacks job_record_path")
    job_record_path = pathlib.Path(job_record_value)
    job = _local_json(job_record_path, "IAA job record")
    if job.get("schema_version") != 1:
        raise ValueError("IAA job record schema_version is unsupported")
    expected_job_path = pathlib.Path(str(request.get("job_state_dir", ""))) / "jobs" / f"{job_id}.json"
    expected_identity = {
        field: job.get(field)
        for field in JOB_IDENTITY_FIELDS
    }
    transitions = job.get("transitions")
    expected = {
        "workflow": "tao-run-deft-iaa",
        "platform": "slurm",
        "request_path": str(request_path),
        "request_sha256": request_digest,
        "job_record_path": str(expected_job_path),
        "job_id": job_id,
        "job_identity_sha256": _canonical_sha256(expected_identity),
        "results_scope": job.get("results_dir"),
    }
    for field, value in expected.items():
        if binding.get(field) != value:
            raise ValueError(f"IAA job binding {field} does not match request/job ownership")
    if request.get("workflow") != "tao-run-deft-iaa" or request.get("platform") != "slurm":
        raise ValueError("IAA request workflow/platform is invalid for SLURM submit")
    if request.get("job_binding_path") != str(binding_path):
        raise ValueError("IAA request does not own the supplied job binding path")
    bundle = request.get("spec_bundle")
    if not isinstance(bundle, dict) or job.get("action") != bundle.get("action"):
        raise ValueError("IAA job action does not match the immutable request")
    if job.get("id") != job_id or job.get("platform") != "slurm":
        raise ValueError("IAA job record identity/platform is invalid")
    if (
        job.get("backend_ref") is not None
        or job.get("terminal_state") is not None
        or not isinstance(transitions, list)
        or len(transitions) != 1
        or not isinstance(transitions[0], dict)
        or transitions[0].get("state") != "PENDING"
    ):
        raise ValueError("IAA job binding must name one fresh PENDING job record")
    receipt = binding.get("staging_receipt_sha256")
    backend_sources = None
    if request.get("freshness_contract") == "remote-mirror-with-delete-before-submit":
        if not isinstance(receipt, str) or SHA256.fullmatch(receipt) is None:
            raise ValueError("IAA remote job binding lacks staged-absence receipt digest")
        receipt_path = pathlib.Path(str(request.get("staging_receipt_path", "")))
        staging = _local_json(receipt_path, "IAA staging receipt")
        staging_body = dict(staging)
        staging_digest = staging_body.pop("receipt_sha256", None)
        if staging_digest != receipt or staging_digest != _canonical_sha256(staging_body):
            raise ValueError("IAA staging receipt digest differs from the job binding")
        rows = staging.get("mount_map")
        if rows is not None:
            mounts = request.get("mounts")
            if not isinstance(mounts, list) or not mounts:
                raise ValueError("IAA request lacks mount rows for staged mapping")
            expected_sources = list(dict.fromkeys(str(row.get("source")) for row in mounts))
            if (
                not isinstance(rows, list)
                or [row.get("source") for row in rows if isinstance(row, dict)]
                != expected_sources
                or any(
                    not isinstance(row, dict)
                    or set(row) != {"source", "backend_source"}
                    or not isinstance(row.get("backend_source"), str)
                    or not row["backend_source"].startswith("/")
                    for row in rows
                )
            ):
                raise ValueError("IAA staging receipt mount_map is invalid")
            backend_sources = {
                str(row["source"]): str(row["backend_source"]) for row in rows
            }
    return request, backend_sources


def _container_mount_values(script_text: str) -> list[str]:
    """Extract literal Pyxis mount-list arguments from one rendered script."""

    normalized = re.sub(r"\\\s*\n", " ", script_text)
    try:
        tokens = shlex.split(normalized, comments=True, posix=True)
    except ValueError as exc:
        raise ValueError("rendered script shell tokens are malformed") from exc
    values: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--container-mounts":
            if index + 1 >= len(tokens):
                raise ValueError("Pyxis --container-mounts lacks a value")
            values.append(tokens[index + 1])
            index += 2
            continue
        if token.startswith("--container-mounts="):
            values.append(token.split("=", 1)[1])
        index += 1
    return values


def _validate_iaa_rendered_mounts(
    request: dict[str, object], rendered_script: pathlib.Path,
    backend_sources: dict[str, str] | None = None,
) -> None:
    """Bind every rendered Pyxis mount source, target, and mode to the request."""

    mounts = request.get("mounts")
    if not isinstance(mounts, list) or not mounts:
        raise ValueError("IAA action request mounts must be a non-empty list")
    expected: list[str] = []
    for row in mounts:
        if not isinstance(row, dict):
            raise ValueError("IAA action request mount row must be an object")
        source = row.get("source")
        target = row.get("target")
        read_only = row.get("read_only")
        if (
            not isinstance(source, str)
            or not source.startswith("/")
            or not isinstance(target, str)
            or not target.startswith("/")
            or not isinstance(read_only, bool)
            or any(character in source or character in target for character in ",:")
        ):
            raise ValueError("IAA action request mount row is not a literal absolute mapping")
        rendered_source = backend_sources.get(source, source) if backend_sources else source
        expected.append(f"{rendered_source}:{target}:{'ro' if read_only else 'rw'}")
    if len(expected) != len(set(expected)):
        raise ValueError("IAA action request contains duplicate mount mappings")
    values = _container_mount_values(rendered_script.read_text(encoding="utf-8"))
    if len(values) != 1:
        raise ValueError("typed IAA script must declare exactly one Pyxis container-mounts list")
    actual = values[0].split(",") if values[0] else []
    if actual != expected:
        raise ValueError(
            "rendered Pyxis mounts differ from immutable IAA request source/target/mode"
        )


def _validate_iaa_rendered_compute(
    request: dict[str, object], rendered_script: pathlib.Path
) -> None:
    """Bind Pyxis environment and scheduler GPU shape to the signed request."""

    bundle = request.get("spec_bundle")
    gpu_ids = request.get("gpu_ids")
    environment = request.get("environment")
    if not isinstance(bundle, dict) or not isinstance(gpu_ids, list):
        raise ValueError("IAA action request compute contract is malformed")
    compute = bundle.get("compute_shape")
    network_arch = bundle.get("network_arch")
    if not isinstance(compute, dict) or not isinstance(environment, dict):
        raise ValueError("IAA action request compute/environment contract is malformed")
    gpus = compute.get("gpus")
    if (
        not isinstance(gpus, int)
        or isinstance(gpus, bool)
        or gpus < 0
        or len(gpu_ids) != gpus
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in gpu_ids)
        or len(gpu_ids) != len(set(gpu_ids))
    ):
        raise ValueError("IAA action request GPU count/ids are inconsistent")
    is_adapter = network_arch == "iaa-adapter"
    if is_adapter != (gpus == 0):
        raise ValueError("IAA adapter/model network_arch disagrees with GPU compute shape")

    script_text = rendered_script.read_text(encoding="utf-8")
    gres = re.findall(r"^#SBATCH --gres=gpu:([0-9]+)\s*$", script_text, re.MULTILINE)
    if (gpus == 0 and gres) or (gpus > 0 and gres != [str(gpus)]):
        raise ValueError("rendered SLURM GPU request differs from immutable IAA compute shape")

    request_names: list[str] = []
    for name in sorted(environment):
        value = environment[name]
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Z][A-Z0-9_]*", name)
            or not isinstance(value, str)
            or not value
            or re.search(r"[\s#]", value)
        ):
            raise ValueError("IAA action request environment contains a nonliteral value")
        declarations = re.findall(
            rf"^export {re.escape(name)}=([^\s#]+)\s*$", script_text, re.MULTILINE
        )
        if declarations != [value]:
            raise ValueError(
                f"rendered IAA environment must declare exactly {name}={value}"
            )
        request_names.append(name)
    expected_env = (
        tuple(request_names)
        if is_adapter
        else tuple(dict.fromkeys((*CONTAINER_ENV_NAMES, *request_names)))
    )
    container_env = CONTAINER_ENV_RE.findall(script_text)
    if len(container_env) != 1 or tuple(container_env[0].split(",")) != expected_env:
        raise ValueError(
            "rendered Pyxis container-env differs from immutable IAA/platform allowlist"
        )
    known_request_names = {*ADAPTER_ENVIRONMENT, *VISUALIZE_THREAD_CAPS}
    for name in known_request_names - set(request_names):
        if re.search(rf"^export {re.escape(name)}=", script_text, re.MULTILINE):
            raise ValueError(f"rendered IAA environment contains unrequested {name}")


def _exact_job_ids(login: str, job_id: str) -> list[str]:
    quoted = shlex.quote(job_id)
    command = (
        "set -Eeuo pipefail; "
        f"squeue -h --name {quoted} -o '%i'; "
        f"sacct -X -n --name {quoted} -o JobIDRaw 2>/dev/null || true"
    )
    output = _require_ok(_ssh(login, command), "exact SLURM job-name query")
    ids = sorted(
        {
            line.strip().split(".", 1)[0]
            for line in output.decode("utf-8", errors="replace").splitlines()
            if line.strip().split(".", 1)[0].isdigit()
        }
    )
    return ids


def _reconcile_submitted_job(login: str, job_id: str) -> list[str]:
    """Poll bounded exact-name accounting after a lost/malformed sbatch reply."""

    deadline = time.monotonic() + SUBMIT_RECONCILE_TIMEOUT_SECONDS
    while True:
        ids = _exact_job_ids(login, job_id)
        if ids or time.monotonic() >= deadline:
            return ids
        time.sleep(
            min(
                SUBMIT_RECONCILE_INTERVAL_SECONDS,
                max(0.0, deadline - time.monotonic()),
            )
        )


def _validate_inputs(
    *, login: str, job_id: str, rendered_script: pathlib.Path, remote_script: pathlib.Path
) -> tuple[pathlib.Path, pathlib.Path, str]:
    if not SAFE_TOKEN.fullmatch(login):
        raise ValueError("login contains unsupported characters")
    if not SAFE_TOKEN.fullmatch(job_id):
        raise ValueError("job id contains unsupported characters")
    local = pathlib.Path(os.path.abspath(rendered_script.expanduser()))
    if not local.is_file() or local.is_symlink() or local.stat().st_size == 0:
        raise ValueError(f"rendered script must be one nonempty regular file: {local}")
    remote = pathlib.Path(str(remote_script))
    if not remote.is_absolute() or not SAFE_REMOTE_PATH.fullmatch(str(remote)):
        raise ValueError("remote script must be one safe absolute path")
    if remote == pathlib.Path(remote.anchor) or ".." in remote.parts:
        raise ValueError("remote script target is too broad or traversing")
    script_text = local.read_text(encoding="utf-8")
    if IAA_CLIP_TRAIN_JOB.fullmatch(job_id):
        # A single parent srun owns the allocation. The request-mounted wrapper
        # removes only that inherited one-task topology before TAO/Lightning
        # launches the approved GPUs. Enforce the application contract here so
        # a freehand render cannot consume an allocation and fail pre-workload.
        normalized_script = re.sub(r"\\\s*\n", " ", script_text)
        if not IAA_CLIP_TRAIN_WRAPPED.search(normalized_script):
            raise ValueError(
                "IAA clip train must prefix the exact argv with "
                "/patches/run_clip_train_slurm.sh"
            )
    container_env = CONTAINER_ENV_RE.findall(script_text)
    if "--container-image=" in script_text:
        if len(container_env) != 1:
            raise ValueError(
                "container launch must declare the one fixed Pyxis container-env allowlist"
            )
        declared_frame = re.findall(
            r"^export IAA_COMPUTE_FRAME=([^\s#]+)\s*$", script_text, re.MULTILINE
        )
        if declared_frame and "-deft-iaa-" not in job_id:
            if declared_frame != ["slurm"]:
                raise ValueError("IAA_COMPUTE_FRAME must be the fixed literal slurm")
            if "#SBATCH --gres" in script_text:
                raise ValueError("IAA adapter compute actions must not request GPUs")
            for name, value in ADAPTER_ENVIRONMENT.items():
                declarations = re.findall(
                    rf"^export {re.escape(name)}=([^\s#]+)\s*$",
                    script_text,
                    re.MULTILINE,
                )
                if declarations != [value]:
                    raise ValueError(
                        f"IAA adapter environment must declare exactly {name}={value}"
                    )
            visualize_finish = bool(
                re.search(
                    r"/iaa-runtime/run_iaa_compute\.py\s+visualize_finish(?:\s|$)",
                    script_text,
                )
            )
            if visualize_finish:
                for name, value in VISUALIZE_THREAD_CAPS.items():
                    declarations = re.findall(
                        rf"^export {re.escape(name)}=([^\s#]+)\s*$",
                        script_text,
                        re.MULTILINE,
                    )
                    if declarations != [value]:
                        raise ValueError(
                            f"visualize_finish environment must declare exactly "
                            f"{name}={value}"
                        )
                expected_env = (
                    *ADAPTER_CONTAINER_ENV_NAMES,
                    *VISUALIZE_THREAD_CAPS,
                )
            else:
                expected_env = ADAPTER_CONTAINER_ENV_NAMES
        else:
            expected_env = CONTAINER_ENV_NAMES
        if "-deft-iaa-" not in job_id and tuple(container_env[0].split(",")) != expected_env:
            raise ValueError(
                "Pyxis container-env differs from the fixed non-secret allowlist"
            )
    elif container_env:
        raise ValueError("container-env is forbidden without a container launch")
    digest = hashlib.sha256(local.read_bytes()).hexdigest()
    return local, remote, digest


def submit_action(
    *, login: str, job_id: str, rendered_script: pathlib.Path, remote_script: pathlib.Path,
    request_path: pathlib.Path | None = None,
    binding_path: pathlib.Path | None = None,
) -> dict[str, object]:
    local, remote, local_sha256 = _validate_inputs(
        login=login,
        job_id=job_id,
        rendered_script=rendered_script,
        remote_script=remote_script,
    )
    if "-deft-iaa-" in job_id:
        if request_path is None or binding_path is None:
            raise ValueError(
                "typed IAA SLURM submit requires --request and --job-binding"
            )
        request, backend_sources = _validate_iaa_job_binding(
            login=login,
            job_id=job_id,
            request_path=request_path,
            binding_path=binding_path,
        )
        _validate_iaa_rendered_mounts(request, local, backend_sources)
        _validate_iaa_rendered_compute(request, local)
    repo_root = pathlib.Path(__file__).resolve().parents[4]
    linter = repo_root / "scripts" / "redact_secrets.py"
    _require_ok(_run([sys.executable, str(linter), "lint", str(local)]), "secret lint")
    _require_ok(_run(["bash", "-n", str(local)]), "local shell syntax check")

    existing = _exact_job_ids(login, job_id)
    if existing:
        raise ValueError(
            f"exact SLURM job name already exists before submit: {job_id} -> {existing}"
        )

    remote_parent = shlex.quote(str(remote.parent))
    _require_ok(_ssh(login, f"mkdir -p -- {remote_parent}"), "remote script directory creation")
    temporary = pathlib.Path(f"{remote}.tmp.{local_sha256[:16]}")
    copy_target = f"{login}:{temporary}"
    _require_ok(_run(["scp", "-q", "--", str(local), copy_target]), "rendered script copy")

    quoted_temporary = shlex.quote(str(temporary))
    quoted_remote = shlex.quote(str(remote))
    verify = _require_ok(
        _ssh(
            login,
            "set -Eeuo pipefail; "
            f"test -s {quoted_temporary}; bash -n {quoted_temporary}; "
            f"sha256sum {quoted_temporary}",
        ),
        "remote staged-script validation",
    ).decode("utf-8", errors="replace").strip().split()
    if not verify or verify[0] != local_sha256:
        raise ValueError("remote staged-script SHA-256 does not match rendered input")

    promote = _require_ok(
        _ssh(
            login,
            "set -Eeuo pipefail; "
            f"mv -f -- {quoted_temporary} {quoted_remote}; "
            f"test -s {quoted_remote}; bash -n {quoted_remote}; sha256sum {quoted_remote}",
        ),
        "remote script promotion",
    ).decode("utf-8", errors="replace").strip().split()
    if not promote or promote[0] != local_sha256:
        raise ValueError("promoted remote-script SHA-256 does not match rendered input")

    _require_ok(_ssh(login, f"sbatch --test-only {quoted_remote}"), "sbatch test-only")
    existing = _exact_job_ids(login, job_id)
    if existing:
        raise ValueError(
            f"exact SLURM job name appeared before submit: {job_id} -> {existing}"
        )

    submitted = _ssh(login, f"sbatch --parsable {quoted_remote}")
    raw = submitted.stdout.decode("utf-8", errors="replace").strip()
    match = JOB_HANDLE.fullmatch(raw) if submitted.returncode == 0 else None
    reconciled = False
    if match is None:
        ids = _reconcile_submitted_job(login, job_id)
        if len(ids) != 1:
            detail = submitted.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(
                "ambiguous sbatch result and exact-name reconciliation did not find "
                f"one job: response={raw!r} ids={ids} diagnostic={detail!r}"
            )
        backend_ref = ids[0]
        reconciled = True
    else:
        backend_ref = match.group("id")
    return {
        "backend_ref": backend_ref,
        "job_id": job_id,
        "local_script": str(local),
        "remote_script": str(remote),
        "script_sha256": local_sha256,
        "reconciled": reconciled,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--login", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--rendered-script", required=True, type=pathlib.Path)
    parser.add_argument("--remote-script", required=True, type=pathlib.Path)
    parser.add_argument("--request", type=pathlib.Path)
    parser.add_argument("--job-binding", type=pathlib.Path)
    args = parser.parse_args()
    try:
        payload = submit_action(
            login=args.login,
            job_id=args.job_id,
            rendered_script=args.rendered_script,
            remote_script=args.remote_script,
            request_path=args.request,
            binding_path=args.job_binding,
        )
    except (OSError, ValueError) as exc:
        print(f"slurm_submit_action: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
