#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one canonical IAA action through Airflow with SLURM compute.

This is a narrow bridge between existing typed contracts.  It does not choose
workflow stages: the caller supplies the single action selected by the DEFT
audit.  It prepares the immutable action, stages its exact inputs, binds the
native job record, renders the signed SLURM shape, delegates the four verbs to
Airflow, synchronizes terminal evidence, and finalizes the action.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
BANK = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR))

import airflow_orchestrator as orchestrator  # noqa: E402
import command_contract  # noqa: E402
import run_deft_action as producer  # noqa: E402


WORKFLOW = "tao-run-deft-iaa"
SAFE_LOGIN = re.compile(r"^[A-Za-z0-9_.@-]+$")
SAFE_SLURM_TOKEN = re.compile(r"^[A-Za-z0-9_.-]+$")
SAFE_REMOTE = re.compile(r"^/[A-Za-z0-9_./-]+$")
TERMINAL = frozenset({"COMPLETE", "ERROR", "CANCELED"})
ADAPTER_ENVIRONMENT = {
    "HF_HOME": "/cache/huggingface",
    "HOME": "/tmp",
    "IAA_COMPUTE_FRAME": "slurm",
    "PYTHONPATH": "/patches",
    "XDG_CACHE_HOME": "/cache",
}
MODEL_CONTAINER_ENV = (
    "NCCL_DEBUG", "LOGLEVEL", "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE",
    "NCCL_SOCKET_IFNAME", "NCCL_IB_HCA", "NCCL_NET",
)
MODEL_PLATFORM_ENVIRONMENT = {
    "NCCL_IB_DISABLE": "1",
    "NCCL_NET": "Socket",
    "NCCL_P2P_DISABLE": "1",
}


class BridgeError(RuntimeError):
    """One bounded bridge operation failed."""


def _canonical_sha256(payload: dict[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _json(path: pathlib.Path, label: str) -> dict[str, Any]:
    path = pathlib.Path(os.path.abspath(path.expanduser()))
    if not path.is_file() or path.is_symlink() or path.resolve() != path:
        raise BridgeError(f"{label} is missing or unsafe: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BridgeError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise BridgeError(f"{label} root must be an object")
    return payload


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(raw, path)
    finally:
        try:
            os.unlink(raw)
        except FileNotFoundError:
            pass


def _run(
    argv: list[str], *, timeout: int = 300, operation: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise BridgeError(f"{operation} exceeded {timeout}s") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
        raise BridgeError(f"{operation} failed: {detail[-4000:]}")
    return completed


def _json_result(completed: subprocess.CompletedProcess[str], operation: str) -> dict[str, Any]:
    for row in reversed([line.strip() for line in completed.stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(row)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise BridgeError(f"{operation} returned no JSON object")


def _remote_path(value: pathlib.Path, label: str) -> pathlib.Path:
    if (
        not value.is_absolute()
        or SAFE_REMOTE.fullmatch(str(value)) is None
        or value == pathlib.Path(value.anchor)
        or ".." in value.parts
    ):
        raise BridgeError(f"{label} must be one safe non-root absolute path")
    return value


def _ssh(login: str, command: str, *, timeout: int = 120, operation: str) -> str:
    return _run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", login, command],
        timeout=timeout, operation=operation,
    ).stdout


def _workspace_mapping(
    source: pathlib.Path, local_workspace: pathlib.Path, remote_workspace: pathlib.Path
) -> pathlib.Path:
    source = pathlib.Path(os.path.abspath(source.expanduser()))
    try:
        relative = source.relative_to(local_workspace)
    except ValueError as exc:
        raise BridgeError(f"request mount source is outside the approved workspace: {source}") from exc
    return _remote_path(remote_workspace / relative, "mapped SLURM mount source")


def _stage_tree(
    *, source: pathlib.Path, login: str, target: pathlib.Path, receipt: pathlib.Path,
    request: pathlib.Path | None = None, snapshot_field: str | None = None,
    incremental_existing: bool = False,
) -> None:
    command = [
        sys.executable,
        str(BANK / "skills/platform/tao-run-on-slurm/scripts/slurm_stage_tree.py"),
        "--source", str(source), "--login", login, "--target", str(target),
        "--receipt", str(receipt),
    ]
    if request is not None and snapshot_field is not None:
        command.extend(["--action-request", str(request), "--snapshot-field", snapshot_field])
    if incremental_existing:
        command.append("--incremental-existing")
    _run(command, timeout=7200, operation=f"stage {source.name} to SLURM")


def _verify_remote_archives(
    *, login: str, state: dict[str, Any], local_workspace: pathlib.Path,
    remote_workspace: pathlib.Path,
) -> None:
    for field in ("images_archive", "metadata_archive"):
        local = pathlib.Path(state["config"][field])
        remote = _workspace_mapping(local, local_workspace, remote_workspace)
        digest = hashlib.sha256()
        with local.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        quoted = shlex.quote(str(remote))
        output = _ssh(
            login,
            f"set -Eeuo pipefail; test -f {quoted}; test ! -L {quoted}; sha256sum -- {quoted}",
            timeout=900,
            operation=f"verify remote {field}",
        ).strip().split()
        if not output or output[0] != digest.hexdigest():
            raise BridgeError(f"remote {field} differs from the approved local archive")


def _verify_backend_dataset(
    *, login: str, local: pathlib.Path, remote: pathlib.Path,
) -> None:
    if not local.is_dir() or local.is_symlink() or local.resolve() != local:
        raise BridgeError("reused controller dataset must be one safe existing directory")
    remote = _remote_path(remote, "--backend-dataset-root")
    required = (
        "train_pairs.json", "val_pairs.json", "test_pairs.json",
        "train_list.txt", "val_list.txt", "test_list.txt",
    )
    local_hashes: list[str] = []
    for name in required:
        path = local / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise BridgeError(f"reused controller dataset lacks {name}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        local_hashes.append(digest.hexdigest())
    quoted = shlex.quote(str(remote))
    names = " ".join(shlex.quote(name) for name in required)
    command = (
        "set -Eeuo pipefail; "
        f"test -d {quoted}; test ! -L {quoted}; "
        f"for child in images images_raw captions; do test -d {quoted}/$child; "
        f"test ! -L {quoted}/$child; done; "
        f"cd {quoted}; sha256sum -- {names}"
    )
    output = _ssh(login, command, timeout=1800, operation="verify reused SLURM dataset")
    remote_hashes = [line.split()[0] for line in output.splitlines() if line.strip()]
    if remote_hashes != local_hashes:
        raise BridgeError("backend dataset metadata differs from the approved controller dataset")


def _render(
    *, request: dict[str, Any], mount_map: dict[str, str], job_id: str,
    image: pathlib.Path, log_dir: pathlib.Path, account: str,
    cpu_partition: str, gpu_partition: str,
    cpu_time_minutes: int = 120, gpu_time_minutes: int = 240,
) -> str:
    bundle = request["spec_bundle"]
    adapter = bundle["network_arch"] == "iaa-adapter"
    gpus = int(bundle["compute_shape"]["gpus"])
    for value, name in (
        (cpu_time_minutes, "cpu_time_minutes"),
        (gpu_time_minutes, "gpu_time_minutes"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or not 10 <= value <= 240:
            raise BridgeError(f"{name} must be in [10, 240]")
    template_name = "cpu.sbatch.tmpl" if adapter else "singlenode.sbatch.tmpl"
    text = (BANK / "templates/slurm" / template_name).read_text(encoding="utf-8")
    mount_values = []
    for row in request["mounts"]:
        source = mount_map[row["source"]]
        mount_values.append(
            f"{source}:{row['target']}:{'ro' if row['read_only'] else 'rw'}"
        )
    command = [bundle["command"], *bundle["args"]]
    if request["name"] == "train":
        command.insert(0, "/patches/run_clip_train_slurm.sh")
    environment = request["environment"]
    if adapter:
        extra_names = sorted(
            name for name in environment if name not in ADAPTER_ENVIRONMENT
        )
        for name, expected in ADAPTER_ENVIRONMENT.items():
            if environment.get(name) != expected:
                raise BridgeError(f"adapter environment differs for {name}")
        extra_env = "\n".join(f"export {name}={environment[name]}" for name in extra_names)
        container_names = tuple(sorted(environment))
        partition = cpu_partition
    else:
        effective_environment = dict(MODEL_PLATFORM_ENVIRONMENT)
        effective_environment.update(environment)
        extra_env = "\n".join(
            f"export {name}={effective_environment[name]}"
            for name in sorted(effective_environment)
        )
        container_names = tuple(
            dict.fromkeys((*MODEL_CONTAINER_ENV, *sorted(environment)))
        )
        partition = gpu_partition
    replacements = {
        "JOB_NAME": job_id,
        "NUM_GPUS": str(gpus),
        "CPUS_PER_TASK": "16" if adapter else "32",
        "MEMORY": "64G",
        "TIME": (
            f"{cpu_time_minutes // 60:02d}:{cpu_time_minutes % 60:02d}:00"
            if adapter else
            f"{gpu_time_minutes // 60:02d}:{gpu_time_minutes % 60:02d}:00"
        ),
        "LOG_DIR": str(log_dir),
        "REQUEUE_DIRECTIVE": "#SBATCH --requeue",
        "SBATCH_EXTRA": f"#SBATCH --account={account}\n#SBATCH --partition={partition}",
        "ENV_FILE": "",
        "EXTRA_ENV": extra_env,
        "IMAGE": str(image),
        "CONTAINER_MOUNTS": ",".join(mount_values),
        "COMMAND": " ".join(shlex.quote(token) for token in command),
    }
    for key, value in replacements.items():
        text = text.replace(f"@@{key}@@", value)
    text = re.sub(
        r"--container-env=[^\s\\]+",
        "--container-env=" + ",".join(container_names),
        text,
        count=1,
    )
    if re.search(r"@@[A-Z0-9_]+@@", text):
        raise BridgeError("rendered SLURM template retains a placeholder")
    return text


def _write_text_atomic(path: pathlib.Path, text: str, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(raw, mode)
        os.replace(raw, path)
    finally:
        try:
            os.unlink(raw)
        except FileNotFoundError:
            pass


def _stage_consumer(shared_root: pathlib.Path) -> pathlib.Path:
    sources = [
        BANK / "skills/platform/tao-run-on-slurm/scripts/slurm_action.py",
        BANK / "skills/platform/tao-run-on-slurm/scripts/slurm_submit_action.py",
        BANK / "scripts/redact_secrets.py",
    ]
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.read_bytes())
    runtime = shared_root / "runtime" / f"slurm-action-{digest.hexdigest()[:16]}"
    consumer_dir = runtime / "skills/platform/tao-run-on-slurm/scripts"
    scripts_dir = runtime / "scripts"
    consumer_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    for source in sources[:2]:
        target = consumer_dir / source.name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise BridgeError(f"staged consumer conflicts with current bytes: {target}")
        if not target.exists():
            shutil.copy2(source, target)
    target = scripts_dir / sources[2].name
    if target.exists() and target.read_bytes() != sources[2].read_bytes():
        raise BridgeError(f"staged redactor conflicts with current bytes: {target}")
    if not target.exists():
        shutil.copy2(sources[2], target)
    return consumer_dir / "slurm_action.py"


def _open_job(
    *, request: dict[str, Any], backend_scope: pathlib.Path, env: dict[str, str]
) -> tuple[str, pathlib.Path]:
    bundle = request["spec_bundle"]
    command = [
        sys.executable, str(BANK / "scripts/tao_job_record.py"), "open",
        "--platform", "slurm", "--image", request["record_image"],
        "--network-arch", bundle["network_arch"], "--action", bundle["action"],
        "--storage-tier", "A", "--results-dir", str(backend_scope),
    ]
    for value in bundle["upload_excludes"]:
        command.extend(["--upload-exclude", value])
    completed = _run(command, operation="open SLURM job record", env=env)
    job_id = completed.stdout.strip()
    if not job_id or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", job_id) is None:
        raise BridgeError("job-record writer returned an invalid id")
    state_dir = pathlib.Path(env["TAO_STATE_DIR"])
    return job_id, state_dir / "jobs" / f"{job_id}.json"


def _orchestration_paths(
    runtime: pathlib.Path, job_id: str,
) -> tuple[pathlib.Path, pathlib.Path]:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", job_id) is None:
        raise BridgeError("job ID is unsafe for orchestration evidence paths")
    return (
        runtime / f"airflow-slurm-consumer-plan.{job_id}.json",
        runtime / f"airflow-slurm-orchestration.{job_id}.json",
    )


def _mark_job(
    *, job_id: str, state: str, env: dict[str, str], backend_ref: str | None = None,
    message: str | None = None, source: str = "poller",
) -> None:
    command = [
        sys.executable, str(BANK / "scripts/tao_job_record.py"), "mark", job_id,
        "--state", state, "--source", source,
    ]
    if backend_ref is not None:
        command.extend(["--backend-ref", backend_ref])
    if message:
        command.extend(["--message", message])
    _run(command, operation=f"mark job {state}", env=env)


def _prove_remote_job_absent(login: str, job_id: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", job_id) is None:
        raise BridgeError("pre-submit recovery job ID is unsafe")
    quoted = shlex.quote(job_id)
    output = _ssh(
        login,
        "set -Eeuo pipefail; "
        f"squeue -h --name {quoted} -o '%i|%j'; "
        f"sacct -X -n --name {quoted} -o 'JobIDRaw,JobName' -P 2>/dev/null || true",
        operation="prove failed Airflow submit created no SLURM job",
    )
    if output.strip():
        raise BridgeError(
            "Airflow submit failed without a backend receipt but an exact-name "
            "SLURM job exists; preserve the binding for operator reconciliation"
        )


def _fetch_dataset_tree(
    *, login: str, remote: pathlib.Path, local: pathlib.Path,
) -> None:
    if local.is_dir() and not local.is_symlink():
        return
    if local.exists() or local.is_symlink():
        raise BridgeError(f"dataset synchronization target is unsafe: {local}")
    remote = _remote_path(remote, "remote dataset root")
    local.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = pathlib.Path(
        tempfile.mkdtemp(prefix=local.name + ".", suffix=".sync.tmp", dir=local.parent)
    )
    try:
        _run(
            ["scp", "-q", "-r", "--", f"{login}:{remote}", str(temporary_parent)],
            timeout=7200, operation="synchronize rebuilt dataset",
        )
        candidate = temporary_parent / remote.name
        if not candidate.is_dir() or candidate.is_symlink():
            raise BridgeError("synchronized dataset root is missing or unsafe")
        os.replace(candidate, local)
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_materialization_outputs(results: pathlib.Path) -> tuple[pathlib.Path, ...]:
    return (
        results / "embeddings" / "source" / "source_pool.parquet",
        results / "iaa_splits" / "aug_pool_list.txt",
        results / "iaa_splits" / "aug_pool_pairs.json",
        results / "iaa_splits" / "eval_list.txt",
        results / "iaa_splits" / "eval_pairs.json",
        results / "iaa_splits" / "val_list.txt",
    )


def _safe_output_parent(path: pathlib.Path, results: pathlib.Path) -> None:
    try:
        relative = path.relative_to(results)
    except ValueError as exc:
        raise BridgeError("materialization output escapes results_dir") from exc
    cursor = results
    if cursor.is_symlink() or not cursor.is_dir():
        raise BridgeError("results_dir is missing or unsafe")
    for part in relative.parent.parts:
        cursor = cursor / part
        if cursor.exists() and cursor.is_symlink():
            raise BridgeError(f"materialization output parent is a symlink: {cursor}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.resolve() != path.parent:
        raise BridgeError(f"materialization output parent is unsafe: {path.parent}")


def _remote_file_evidence(login: str, path: pathlib.Path) -> tuple[int, str]:
    remote = _remote_path(path, "remote materialization output")
    quoted = shlex.quote(str(remote))
    output = _ssh(
        login,
        "set -Eeuo pipefail; "
        f"test -f {quoted}; test ! -L {quoted}; "
        f"stat -c '%s' -- {quoted}; sha256sum -- {quoted}",
        timeout=900,
        operation=f"verify remote materialization output {remote.name}",
    ).splitlines()
    if len(output) != 2 or not output[0].isdigit():
        raise BridgeError("remote materialization output evidence is malformed")
    digest = output[1].split()[0] if output[1].split() else ""
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise BridgeError("remote materialization output digest is malformed")
    size = int(output[0])
    if size <= 0:
        raise BridgeError(f"remote materialization output is empty: {remote}")
    return size, digest


def _remote_symlink_target_evidence(
    login: str, path: pathlib.Path,
) -> tuple[str, pathlib.Path, int, str]:
    remote = _remote_path(path, "remote checkpoint symlink")
    quoted = shlex.quote(str(remote))
    output = _ssh(
        login,
        "set -Eeuo pipefail; "
        f"test -L {quoted}; link=$(readlink -- {quoted}); "
        f"target=$(readlink -f -- {quoted}); test -f \"$target\"; test ! -L \"$target\"; "
        "printf '%s\\n' \"$(printf '%s' \"$link\" | base64 -w0)\"; "
        "printf '%s\\n' \"$(printf '%s' \"$target\" | base64 -w0)\"; "
        "stat -c '%s' -- \"$target\"; sha256sum -- \"$target\"",
        timeout=900, operation=f"verify remote checkpoint symlink {remote.name}",
    ).splitlines()
    if len(output) != 4 or not output[2].isdigit() or int(output[2]) <= 0:
        raise BridgeError("remote checkpoint symlink evidence is malformed")
    try:
        link = base64.b64decode(output[0], validate=True).decode("utf-8")
        resolved_text = base64.b64decode(output[1], validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise BridgeError("remote checkpoint symlink path evidence is malformed") from exc
    digest = output[3].split()[0] if output[3].split() else ""
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise BridgeError("remote checkpoint symlink digest is malformed")
    resolved = _remote_path(pathlib.Path(resolved_text), "remote checkpoint target")
    return link, resolved, int(output[2]), digest


def _copy_remote_file(login: str, remote: pathlib.Path, local: pathlib.Path) -> None:
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=local.name + ".", suffix=".sync.tmp", dir=local.parent
    )
    os.close(descriptor)
    os.unlink(temporary_raw)
    temporary = pathlib.Path(temporary_raw)
    try:
        _run(
            ["scp", "-q", "--", f"{login}:{remote}", str(temporary)],
            timeout=3600,
            operation=f"synchronize materialization output {local.name}",
        )
        if not temporary.is_file() or temporary.is_symlink():
            raise BridgeError(f"synchronized materialization output is unsafe: {local}")
        os.replace(temporary, local)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _remote_tree_evidence(
    login: str, remote: pathlib.Path,
) -> list[dict[str, Any]]:
    remote = _remote_path(remote, "remote visualization directory")
    quoted = shlex.quote(str(remote))
    script = (
        "set -Eeuo pipefail; "
        f"test -d {quoted}; test ! -L {quoted}; cd -- {quoted}; "
        "test -z \"$(find . -type l -print -quit)\"; "
        "while IFS= read -r -d '' rel; do "
        "size=$(stat -c '%s' -- \"$rel\"); "
        "digest=$(sha256sum -- \"$rel\" | cut -d' ' -f1); "
        "encoded=$(printf '%s' \"${rel#./}\" | base64 -w0); "
        "printf '%s|%s|%s\\n' \"$digest\" \"$size\" \"$encoded\"; "
        "done < <(find . -type f -print0 | LC_ALL=C sort -z)"
    )
    lines = _ssh(
        login, "bash -c " + shlex.quote(script), timeout=900,
        operation=f"verify remote visualization tree {remote.name}",
    ).splitlines()
    rows: list[dict[str, Any]] = []
    for line in lines:
        fields = line.split("|", 2)
        if (
            len(fields) != 3
            or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None
            or not fields[1].isdigit()
            or int(fields[1]) <= 0
        ):
            raise BridgeError("remote visualization tree evidence is malformed")
        try:
            relative_text = base64.b64decode(fields[2], validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise BridgeError("remote visualization path evidence is malformed") from exc
        relative = pathlib.PurePosixPath(relative_text)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise BridgeError("remote visualization tree contains an unsafe path")
        rows.append({
            "relative": relative.as_posix(), "size": int(fields[1]),
            "sha256": fields[0],
        })
    if not rows:
        raise BridgeError("remote visualization directory is empty")
    return rows


def _local_tree_evidence(root: pathlib.Path) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise BridgeError(f"synchronized visualization directory is unsafe: {root}")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise BridgeError(f"synchronized visualization directory contains a symlink: {root}")
    rows = [{
        "relative": path.relative_to(root).as_posix(),
        "size": path.stat().st_size, "sha256": _sha256_file(path),
    } for path in sorted(item for item in root.rglob("*") if item.is_file())]
    if not rows or any(row["size"] <= 0 for row in rows):
        raise BridgeError(f"synchronized visualization directory is empty: {root}")
    return rows


def _copy_remote_tree(
    login: str, remote: pathlib.Path, local: pathlib.Path,
    evidence: list[dict[str, Any]],
) -> str:
    _safe_output_parent(local, local.parents[2])
    disposition = "reused"
    if local.exists() or local.is_symlink():
        if local.is_symlink() or _local_tree_evidence(local) != evidence:
            raise BridgeError(f"existing controller visualization tree differs: {local}")
        return disposition
    temporary_parent = pathlib.Path(
        tempfile.mkdtemp(prefix=local.name + ".", suffix=".sync.tmp", dir=local.parent)
    )
    try:
        _run(
            ["scp", "-q", "-r", "--", f"{login}:{remote}", str(temporary_parent)],
            timeout=3600, operation=f"synchronize visualization tree {local.name}",
        )
        candidate = temporary_parent / remote.name
        if _local_tree_evidence(candidate) != evidence:
            raise BridgeError("synchronized visualization tree differs from remote evidence")
        os.replace(candidate, local)
        disposition = "fetched"
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)
    return disposition


def _remote_train_output_evidence(
    login: str, remote: pathlib.Path, *, allow_empty: bool = False,
) -> list[dict[str, Any]]:
    """Inventory only checkpoint and TensorBoard files needed after train."""
    remote = _remote_path(remote, "remote train directory")
    quoted = shlex.quote(str(remote))
    script = (
        "set -Eeuo pipefail; "
        f"test -d {quoted}; test ! -L {quoted}; cd -- {quoted}; "
        "while IFS= read -r -d '' rel; do "
        "test ! -L \"$rel\"; "
        "base=${rel##*/}; lower=$(printf '%s' \"$base\" | tr '[:upper:]' '[:lower:]'); "
        "case \"$lower\" in *latest*|*_pretrained.pth|*.tmp) continue;; esac; "
        "case \"$rel\" in ./best/*) continue;; esac; "
        "size=$(stat -c '%s' -- \"$rel\"); test \"$size\" -gt 0; "
        "digest=$(sha256sum -- \"$rel\" | cut -d' ' -f1); "
        "encoded=$(printf '%s' \"${rel#./}\" | base64 -w0); "
        "printf '%s|%s|%s\\n' \"$digest\" \"$size\" \"$encoded\"; "
        "done < <(find . -type f \\( -name '*.pth' -o -name '*.ckpt' "
        "-o -name '*.safetensors' -o -name 'events.out.tfevents*' \\) "
        "-print0 | LC_ALL=C sort -z)"
    )
    lines = _ssh(
        login, "bash -c " + shlex.quote(script), timeout=900,
        operation="inventory remote train side outputs",
    ).splitlines()
    rows: list[dict[str, Any]] = []
    for line in lines:
        fields = line.split("|", 2)
        if (
            len(fields) != 3
            or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None
            or not fields[1].isdigit()
            or int(fields[1]) <= 0
        ):
            raise BridgeError("remote train output evidence is malformed")
        try:
            relative_text = base64.b64decode(fields[2], validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise BridgeError("remote train output path evidence is malformed") from exc
        relative = pathlib.PurePosixPath(relative_text)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise BridgeError("remote train output contains an unsafe path")
        rows.append({
            "relative": relative.as_posix(), "size": int(fields[1]),
            "sha256": fields[0],
        })
    if not rows and not allow_empty:
        raise BridgeError("successful train produced no publishable checkpoint")
    checkpoint_rows = [
        row for row in rows
        if pathlib.PurePosixPath(row["relative"]).suffix.lower()
        in {".pth", ".ckpt", ".safetensors"}
    ]
    if rows and not checkpoint_rows:
        raise BridgeError("successful train side outputs contain no publishable checkpoint")
    return rows


def _synchronize_train_outputs(
    *, login: str, results: pathlib.Path, local_workspace: pathlib.Path,
    remote_workspace: pathlib.Path, label: str, request: dict[str, Any],
) -> pathlib.Path:
    if not re.fullmatch(r"iter[1-9][0-9]*", label):
        raise BridgeError("train output synchronization requires iterN")
    phase = results / f"iter_{int(label[4:])}" / "train"
    remote_phase = _workspace_mapping(phase, local_workspace, remote_workspace)
    evidence = _remote_train_output_evidence(login, remote_phase)
    rows: list[dict[str, Any]] = []
    for item in evidence:
        relative = pathlib.PurePosixPath(item["relative"])
        local = phase.joinpath(*relative.parts)
        remote = remote_phase.joinpath(*relative.parts)
        _safe_output_parent(local, results)
        disposition = "reused"
        if local.exists() or local.is_symlink():
            if (
                not local.is_file() or local.is_symlink()
                or local.stat().st_size != item["size"]
                or _sha256_file(local) != item["sha256"]
            ):
                raise BridgeError(f"existing controller train output differs: {local}")
        else:
            _copy_remote_file(login, remote, local)
            disposition = "fetched"
        if local.stat().st_size != item["size"] or _sha256_file(local) != item["sha256"]:
            raise BridgeError(f"synchronized train output differs: {local}")
        rows.append({
            **item, "local": str(local), "remote": str(remote),
            "disposition": disposition,
        })
    receipt_path = phase / "train.output-sync.json"
    receipt = {
        "schema_version": "1", "workflow": WORKFLOW,
        "kind": "airflow_slurm_train_output_sync", "name": "train",
        "label": label, "results_dir": str(results),
        "request_sha256": request["request_sha256"],
        "workload_status_sha256": _sha256_file(phase / "status.json"),
        "outputs": rows,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt, "receipt_sha256")
    if receipt_path.exists():
        existing = _json(receipt_path, "train output sync receipt")
        if (
            existing.get("workflow") != WORKFLOW
            or existing.get("kind") != receipt["kind"]
            or existing.get("label") != label
            or existing.get("request_sha256") != request["request_sha256"]
            or existing.get("workload_status_sha256") != receipt["workload_status_sha256"]
            or existing.get("receipt_sha256")
            != _canonical_sha256(existing, "receipt_sha256")
        ):
            raise BridgeError("existing train output sync receipt differs")
        normalized_existing = [
            {key: value for key, value in row.items() if key != "disposition"}
            for row in existing.get("outputs", []) if isinstance(row, dict)
        ]
        normalized_rows = [
            {key: value for key, value in row.items() if key != "disposition"}
            for row in rows
        ]
        if len(normalized_existing) != len(rows) or normalized_existing != normalized_rows:
            raise BridgeError("existing train output sync receipt artifacts differ")
        return receipt_path
    _atomic_json(receipt_path, receipt)
    return receipt_path


def _synchronize_publish_checkpoint_outputs(
    *, login: str, results: pathlib.Path, local_workspace: pathlib.Path,
    remote_workspace: pathlib.Path, label: str, request: dict[str, Any],
    recovered: bool,
) -> pathlib.Path:
    if not re.fullmatch(r"iter[1-9][0-9]*", label):
        raise BridgeError("checkpoint publication synchronization requires iterN")
    phase = results / f"iter_{int(label[4:])}"
    stage = phase / "train"
    status_path = stage / "publish-checkpoint.host.status.json"
    host_log = stage / "publish-checkpoint.host.log"
    outputs = (
        phase / "pretrained/model_state.pth",
        stage / "best/clip_best_val_t2i_mAP.json",
        stage / "best/clip_best_val_t2i_mAP.pth",
    )
    status = _json(status_path, "publish-checkpoint host status")
    if (
        status.get("workflow") != WORKFLOW
        or status.get("name") != "publish-checkpoint"
        or status.get("status") != "ok"
        or status.get("exit_code") != 0
        or status.get("log_path") != str(host_log)
        or status.get("fresh_outputs") != [str(path) for path in outputs]
    ):
        raise BridgeError("publish-checkpoint host status does not bind exact outputs")
    rows: list[dict[str, Any]] = []
    for role, local in (
        ("host_log", host_log), ("output", outputs[0]), ("output", outputs[1])
    ):
        _safe_output_parent(local, results)
        remote = _workspace_mapping(local, local_workspace, remote_workspace)
        size, digest = _remote_file_evidence(login, remote)
        disposition = "reused"
        if local.exists() or local.is_symlink():
            if (
                not local.is_file() or local.is_symlink()
                or local.stat().st_size != size or _sha256_file(local) != digest
            ):
                raise BridgeError(f"existing controller checkpoint publication differs: {local}")
        else:
            _copy_remote_file(login, remote, local)
            disposition = "fetched"
        if local.stat().st_size != size or _sha256_file(local) != digest:
            raise BridgeError(f"synchronized checkpoint publication differs: {local}")
        rows.append({
            "role": role, "local": str(local), "remote": str(remote),
            "size": size, "sha256": digest, "disposition": disposition,
        })
    metadata = _json(outputs[1], "best-checkpoint metadata")
    source_raw = metadata.get("selected_checkpoint")
    if not isinstance(source_raw, str) or not pathlib.Path(source_raw).is_absolute():
        raise BridgeError("best-checkpoint metadata lacks an absolute selected checkpoint")
    source = pathlib.Path(source_raw)
    try:
        source.relative_to(stage)
    except ValueError as exc:
        raise BridgeError("best-checkpoint source escapes the iteration train directory") from exc
    if (
        source.parent == stage / "best" or not source.is_file() or source.is_symlink()
        or source.stat().st_size <= 0
    ):
        raise BridgeError("best-checkpoint source is missing or unsafe on the controller")
    local_best = outputs[2]
    remote_best = _workspace_mapping(local_best, local_workspace, remote_workspace)
    expected_remote_source = _workspace_mapping(source, local_workspace, remote_workspace)
    link, remote_source, size, digest = _remote_symlink_target_evidence(
        login, remote_best
    )
    expected_link = os.path.relpath(expected_remote_source, remote_best.parent)
    local_link = os.path.relpath(source, local_best.parent)
    if (
        remote_source != expected_remote_source
        or link != expected_link or link != local_link
        or source.stat().st_size != size or _sha256_file(source) != digest
        or metadata.get("published_checkpoint") != str(local_best)
        or metadata.get("publish_mode") != "symlink"
    ):
        raise BridgeError("best-checkpoint symlink does not bind the selected raw checkpoint")
    _safe_output_parent(local_best, results)
    disposition = "reused"
    if local_best.exists() or local_best.is_symlink():
        if (
            not local_best.is_symlink()
            or os.readlink(local_best) != local_link
            or local_best.resolve() != source
        ):
            raise BridgeError("existing controller best-checkpoint symlink differs")
    else:
        os.symlink(local_link, local_best)
        disposition = "fetched"
    rows.append({
        "role": "output", "kind": "symlink", "local": str(local_best),
        "remote": str(remote_best), "target": local_link,
        "target_size": size, "target_sha256": digest,
        "disposition": disposition,
    })
    receipt_path = stage / "publish_checkpoint.output-sync.json"
    receipt = {
        "schema_version": "1", "workflow": WORKFLOW,
        "kind": "airflow_slurm_publish_checkpoint_output_sync",
        "name": "publish_checkpoint", "label": label,
        "results_dir": str(results), "request_sha256": request["request_sha256"],
        "host_status_sha256": _sha256_file(status_path),
        "recovered_after_terminal": recovered, "outputs": rows,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt, "receipt_sha256")
    if receipt_path.exists():
        existing = _json(receipt_path, "checkpoint publication output sync receipt")
        normalized = lambda values: [  # noqa: E731
            {key: value for key, value in row.items() if key != "disposition"}
            for row in values if isinstance(row, dict)
        ] if isinstance(values, list) else []
        if (
            existing.get("workflow") != WORKFLOW
            or existing.get("kind") != receipt["kind"]
            or existing.get("label") != label
            or existing.get("request_sha256") != request["request_sha256"]
            or existing.get("host_status_sha256") != receipt["host_status_sha256"]
            or existing.get("receipt_sha256")
            != _canonical_sha256(existing, "receipt_sha256")
            or len(existing.get("outputs", [])) != len(rows)
            or normalized(existing.get("outputs")) != normalized(rows)
        ):
            raise BridgeError("existing checkpoint publication output sync receipt differs")
        return receipt_path
    _atomic_json(receipt_path, receipt)
    return receipt_path


def _visualization_side_outputs(
    results: pathlib.Path, label: str, state: dict[str, Any], name: str,
) -> tuple[pathlib.Path, tuple[pathlib.Path, ...]]:
    if not re.fullmatch(r"iter[1-9][0-9]*", label):
        raise BridgeError("visualization output synchronization requires iterN")
    number = int(label[4:])
    phase = results / f"iter_{number}"
    config = state.get("config", {})
    if name == "visualize_prepare":
        status = phase / "visualization" / "visualize-prepare.host.status.json"
        expected: list[pathlib.Path] = []
        if config.get("visualize_embeddings") is True:
            expected.extend((
                phase / "embeddings" / "viz_weak" / "input.parquet",
                phase / "mining" / "mined_unique_images.parquet",
            ))
            if config.get("continual_dataset") is True and (
                number > 1 or bool(config.get("iaa_train_pairs_source_file"))
            ):
                expected.append(
                    phase / "embeddings" / "previous" / "prev_pool.parquet"
                )
        if config.get("visualize") is True:
            expected.append(phase / "visualization" / "samples")
    elif name == "visualize_finish":
        status = phase / "visualization" / "visualize-finish.host.status.json"
        expected = [phase / "visualization" / "tsne_plot.png"]
    else:
        raise BridgeError("visualization synchronization received an unsupported action")
    return status, tuple(expected)


def _synchronize_visualization_outputs(
    *, login: str, results: pathlib.Path, local_workspace: pathlib.Path,
    remote_workspace: pathlib.Path, label: str, state: dict[str, Any], name: str,
    recovered: bool,
) -> pathlib.Path:
    status_path, expected = _visualization_side_outputs(results, label, state, name)
    status = _json(status_path, f"{name} host status")
    expected_log = status_path.parent / f"{name.replace('_', '-')}.host.log"
    log_raw = status.get("log_path")
    if not isinstance(log_raw, str) or not pathlib.Path(log_raw).is_absolute():
        raise BridgeError(f"{name} host status has no absolute log_path")
    host_log = pathlib.Path(log_raw)
    if host_log != expected_log or host_log.resolve() != expected_log.resolve():
        raise BridgeError(f"{name} host status references a noncanonical log_path")
    if (
        status.get("workflow") != WORKFLOW
        or status.get("name") != name.replace("_", "-")
        or status.get("status") != "ok"
        or status.get("exit_code") != 0
        or status.get("fresh_outputs") != [str(path) for path in expected]
    ):
        raise BridgeError(f"{name} host status does not bind exact side outputs")
    rows: list[dict[str, Any]] = []
    remote_log = _workspace_mapping(host_log, local_workspace, remote_workspace)
    log_size, log_digest = _remote_file_evidence(login, remote_log)
    log_disposition = "reused"
    if host_log.exists() or host_log.is_symlink():
        if (
            not host_log.is_file() or host_log.is_symlink()
            or host_log.stat().st_size != log_size
            or _sha256_file(host_log) != log_digest
        ):
            raise BridgeError(f"existing controller visualization host log differs: {host_log}")
    else:
        _copy_remote_file(login, remote_log, host_log)
        log_disposition = "fetched"
    rows.append({
        "kind": "file", "role": "host_log", "local": str(host_log),
        "remote": str(remote_log), "size": log_size, "sha256": log_digest,
        "disposition": log_disposition,
    })
    for local in expected:
        _safe_output_parent(local, results)
        remote = _workspace_mapping(local, local_workspace, remote_workspace)
        if local.suffix:
            size, digest = _remote_file_evidence(login, remote)
            disposition = "reused"
            if local.exists() or local.is_symlink():
                if (
                    not local.is_file() or local.is_symlink()
                    or local.stat().st_size != size or _sha256_file(local) != digest
                ):
                    raise BridgeError(f"existing controller visualization output differs: {local}")
            else:
                _copy_remote_file(login, remote, local)
                disposition = "fetched"
            if local.stat().st_size != size or _sha256_file(local) != digest:
                raise BridgeError(f"synchronized visualization output differs: {local}")
            rows.append({
                "kind": "file", "local": str(local), "remote": str(remote),
                "size": size, "sha256": digest, "disposition": disposition,
            })
        else:
            evidence = _remote_tree_evidence(login, remote)
            disposition = _copy_remote_tree(login, remote, local, evidence)
            rows.append({
                "kind": "directory", "local": str(local), "remote": str(remote),
                "entries": evidence,
                "tree_sha256": hashlib.sha256(
                    json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "disposition": disposition,
            })
    receipt_path = status_path.parent / f"{name}.output-sync.json"
    receipt = {
        "schema_version": "1", "workflow": WORKFLOW,
        "kind": "airflow_slurm_visualization_output_sync", "name": name,
        "label": label, "results_dir": str(results),
        "recovered_after_terminal": recovered,
        "host_status_sha256": _sha256_file(status_path), "outputs": rows,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt, "receipt_sha256")
    if receipt_path.exists():
        existing = _json(receipt_path, "visualization output sync receipt")
        def same_artifacts(left: Any, right: Any) -> bool:
            if not isinstance(left, list) or not isinstance(right, list):
                return False
            return [
                {key: value for key, value in row.items() if key != "disposition"}
                for row in left if isinstance(row, dict)
            ] == [
                {key: value for key, value in row.items() if key != "disposition"}
                for row in right if isinstance(row, dict)
            ] and len(left) == len(right)

        valid_identity = (
            existing.get("workflow") != WORKFLOW
            or existing.get("kind") != receipt["kind"]
            or existing.get("name") != name or existing.get("label") != label
            or existing.get("host_status_sha256") != receipt["host_status_sha256"]
            or existing.get("receipt_sha256")
            != _canonical_sha256(existing, "receipt_sha256")
        )
        if valid_identity:
            raise BridgeError("existing visualization output sync receipt differs")
        if same_artifacts(existing.get("outputs"), rows):
            return receipt_path
        # Releases before the host-log fix synchronized every adapter output
        # but omitted the log referenced by the nested status. Preserve that
        # immutable receipt and add a narrow companion receipt after fetching
        # the exact remote log; do not rewrite prior evidence.
        if same_artifacts(existing.get("outputs"), rows[1:]):
            companion = status_path.parent / f"{name}.host-log-sync.json"
            companion_payload = {
                "schema_version": "1", "workflow": WORKFLOW,
                "kind": "airflow_slurm_visualization_host_log_sync",
                "name": name, "label": label, "results_dir": str(results),
                "prior_receipt_sha256": existing["receipt_sha256"],
                "host_status_sha256": receipt["host_status_sha256"],
                "host_log": rows[0],
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(
                    timespec="seconds"
                ),
            }
            companion_payload["receipt_sha256"] = _canonical_sha256(
                companion_payload, "receipt_sha256"
            )
            if companion.exists():
                prior = _json(companion, "visualization host-log sync receipt")
                if (
                    prior.get("workflow") != WORKFLOW
                    or prior.get("kind") != companion_payload["kind"]
                    or prior.get("name") != name or prior.get("label") != label
                    or prior.get("prior_receipt_sha256")
                    != companion_payload["prior_receipt_sha256"]
                    or prior.get("host_status_sha256")
                    != companion_payload["host_status_sha256"]
                    or not same_artifacts([prior.get("host_log")], [rows[0]])
                    or prior.get("receipt_sha256")
                    != _canonical_sha256(prior, "receipt_sha256")
                ):
                    raise BridgeError("existing visualization host-log sync receipt differs")
                return companion
            _atomic_json(companion, companion_payload)
            return companion
        raise BridgeError("existing visualization output sync receipt differs")
    _atomic_json(receipt_path, receipt)
    return receipt_path


def _synchronize_dataset_materialization(
    *, login: str, results: pathlib.Path, local_workspace: pathlib.Path,
    remote_workspace: pathlib.Path, recovered: bool,
) -> pathlib.Path:
    status_path = results / "dataset_setup" / "dataset-materialize.host.status.json"
    status = _json(status_path, "dataset materialization host status")
    expected = _dataset_materialization_outputs(results)
    if (
        status.get("workflow") != WORKFLOW
        or status.get("name") != "dataset-materialize"
        or status.get("status") != "ok"
        or status.get("exit_code") != 0
        or status.get("fresh_outputs") != [str(path) for path in expected]
    ):
        raise BridgeError("dataset materialization host status does not bind exact outputs")

    rows: list[dict[str, Any]] = []
    for local in expected:
        _safe_output_parent(local, results)
        remote = _workspace_mapping(local, local_workspace, remote_workspace)
        size, digest = _remote_file_evidence(login, remote)
        disposition = "reused"
        if local.exists() or local.is_symlink():
            if (
                not local.is_file()
                or local.is_symlink()
                or local.stat().st_size != size
                or _sha256_file(local) != digest
            ):
                raise BridgeError(
                    f"existing controller materialization output differs: {local}"
                )
        else:
            _copy_remote_file(login, remote, local)
            disposition = "fetched"
        if local.stat().st_size != size or _sha256_file(local) != digest:
            raise BridgeError(f"synchronized materialization output differs: {local}")
        rows.append({
            "local": str(local), "remote": str(remote), "size": size,
            "sha256": digest, "disposition": disposition,
        })

    receipt_path = results / "dataset_setup" / "dataset_materialize.output-sync.json"
    receipt = {
        "schema_version": "1", "workflow": WORKFLOW,
        "kind": "airflow_slurm_materialization_output_sync",
        "results_dir": str(results), "recovered_after_terminal": recovered,
        "host_status_sha256": _sha256_file(status_path), "outputs": rows,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt, "receipt_sha256")
    if receipt_path.exists():
        existing = _json(receipt_path, "materialization output synchronization receipt")
        existing_rows = existing.get("outputs")
        comparable_existing = [
            {key: row.get(key) for key in ("local", "remote", "size", "sha256")}
            for row in existing_rows
        ] if isinstance(existing_rows, list) else None
        comparable_current = [
            {key: row[key] for key in ("local", "remote", "size", "sha256")}
            for row in rows
        ]
        if (
            existing.get("workflow") != WORKFLOW
            or existing.get("kind") != receipt["kind"]
            or existing.get("results_dir") != str(results)
            or existing.get("host_status_sha256") != receipt["host_status_sha256"]
            or comparable_existing != comparable_current
            or existing.get("receipt_sha256")
            != _canonical_sha256(existing, "receipt_sha256")
        ):
            raise BridgeError("existing materialization output sync receipt differs")
        return receipt_path
    _atomic_json(receipt_path, receipt)
    return receipt_path


def _synchronize_gap_history(
    *, login: str, results: pathlib.Path, local_workspace: pathlib.Path,
    remote_workspace: pathlib.Path, label: str, request: dict[str, Any],
    recovered: bool,
) -> pathlib.Path:
    """Synchronize the run-wide ledger mutated by one remote gap action.

    The ledger is intentionally not a fresh output: each gap action extends the
    history produced by prior iterations.  The generic SLURM consumer therefore
    cannot copy it with the action's absent-before-run outputs.  This bridge
    handles that one bounded mutable side effect with digest-bound evidence.
    """

    if request.get("workflow") != WORKFLOW or request.get("name") != "gap_analysis":
        raise BridgeError("gap-history synchronization request identity differs")
    if request.get("label") != label:
        raise BridgeError("gap-history synchronization label differs")
    request_sha = request.get("request_sha256")
    if not isinstance(request_sha, str) or re.fullmatch(r"[0-9a-f]{64}", request_sha) is None:
        raise BridgeError("gap-history synchronization request digest is invalid")

    stage = command_contract.expected_stage_directory("gap_analysis", label, results)
    expected_gap = command_contract.expected_fresh_outputs(
        "gap_analysis", label, results
    )
    if len(expected_gap) != 1:
        raise BridgeError("gap-analysis fresh-output contract is not singular")
    gap_path = pathlib.Path(expected_gap[0])
    if (
        not gap_path.is_file()
        or gap_path.is_symlink()
        or gap_path.stat().st_size == 0
    ):
        raise BridgeError("gap-analysis output is missing before history synchronization")

    local = results / "caption_selection_history.json"
    _safe_output_parent(local, results)
    remote = _workspace_mapping(local, local_workspace, remote_workspace)
    size, digest = _remote_file_evidence(login, remote)
    receipt_path = stage / "gap_analysis.history-sync.json"

    if receipt_path.exists():
        receipt = _json(receipt_path, "gap-history synchronization receipt")
        current = receipt.get("current")
        if (
            receipt.get("workflow") != WORKFLOW
            or receipt.get("kind") != "airflow_slurm_gap_history_sync"
            or receipt.get("results_dir") != str(results)
            or receipt.get("label") != label
            or receipt.get("request_sha256") != request_sha
            or not isinstance(current, dict)
            or current.get("local") != str(local)
            or current.get("remote") != str(remote)
            or current.get("size") != size
            or current.get("sha256") != digest
            or receipt.get("receipt_sha256")
            != _canonical_sha256(receipt, "receipt_sha256")
        ):
            raise BridgeError("existing gap-history synchronization receipt differs")
        if (
            not local.is_file()
            or local.is_symlink()
            or local.stat().st_size != size
            or _sha256_file(local) != digest
        ):
            raise BridgeError("controller gap-history ledger differs from its receipt")
        _json(local, "caption selection history")
        return receipt_path

    previous: dict[str, Any] | None = None
    if local.exists() or local.is_symlink():
        if not local.is_file() or local.is_symlink() or local.stat().st_size == 0:
            raise BridgeError("existing controller gap-history ledger is unsafe")
        _json(local, "existing caption selection history")
        previous_digest = _sha256_file(local)
        if previous_digest == digest and local.stat().st_size == size:
            disposition = "reused"
        else:
            archive = (
                stage / ".tao-runtime" / "gap-history-sync"
                / "caption_selection_history.before.json"
            )
            archive.parent.mkdir(parents=True, exist_ok=True)
            if archive.exists() or archive.is_symlink():
                if (
                    not archive.is_file()
                    or archive.is_symlink()
                    or archive.stat().st_size != local.stat().st_size
                    or _sha256_file(archive) != previous_digest
                ):
                    raise BridgeError("existing archived gap-history ledger differs")
            else:
                shutil.copy2(local, archive)
            previous = {
                "path": str(archive), "size": archive.stat().st_size,
                "sha256": previous_digest,
            }
            _copy_remote_file(login, remote, local)
            disposition = "replaced_after_archive"
    else:
        _copy_remote_file(login, remote, local)
        disposition = "fetched"

    if (
        not local.is_file()
        or local.is_symlink()
        or local.stat().st_size != size
        or _sha256_file(local) != digest
    ):
        raise BridgeError("synchronized gap-history ledger differs")
    _json(local, "caption selection history")

    receipt = {
        "schema_version": "1", "workflow": WORKFLOW,
        "kind": "airflow_slurm_gap_history_sync", "results_dir": str(results),
        "label": label, "request_sha256": request_sha,
        "gap_output_sha256": _sha256_file(gap_path),
        "recovered_after_terminal": recovered, "previous": previous,
        "current": {
            "local": str(local), "remote": str(remote), "size": size,
            "sha256": digest, "disposition": disposition,
        },
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt, "receipt_sha256")
    _atomic_json(receipt_path, receipt)
    return receipt_path


def _mining_candidate_outputs(results: pathlib.Path, label: str) -> tuple[pathlib.Path, ...]:
    stage = command_contract.expected_stage_directory(
        "mining_postprocess", label, results
    )
    candidates = stage / "history_candidates"
    return (
        candidates / "mined_image_list.txt",
        candidates / "mined_pairs.json",
        candidates / "mined_dataset.json",
    )


def _synchronize_mining_candidates(
    *, login: str, results: pathlib.Path, local_workspace: pathlib.Path,
    remote_workspace: pathlib.Path, label: str, request: dict[str, Any],
    recovered: bool,
) -> pathlib.Path:
    """Synchronize the complete history-selection input contract."""

    if (
        request.get("workflow") != WORKFLOW
        or request.get("name") != "mining_postprocess"
        or request.get("label") != label
    ):
        raise BridgeError("mining-candidate synchronization request identity differs")
    request_sha = request.get("request_sha256")
    if not isinstance(request_sha, str) or re.fullmatch(r"[0-9a-f]{64}", request_sha) is None:
        raise BridgeError("mining-candidate synchronization request digest is invalid")
    expected = _mining_candidate_outputs(results, label)
    declared = request.get("fresh_outputs")
    if declared != [str(expected[1])]:
        raise BridgeError("mining postprocess does not bind its canonical candidate pairs")

    rows: list[dict[str, Any]] = []
    for local in expected:
        _safe_output_parent(local, results)
        remote = _workspace_mapping(local, local_workspace, remote_workspace)
        size, digest = _remote_file_evidence(login, remote)
        disposition = "reused"
        if local.exists() or local.is_symlink():
            if (
                not local.is_file()
                or local.is_symlink()
                or local.stat().st_size != size
                or _sha256_file(local) != digest
            ):
                raise BridgeError(f"existing controller mining candidate differs: {local}")
        else:
            _copy_remote_file(login, remote, local)
            disposition = "fetched"
        if local.stat().st_size != size or _sha256_file(local) != digest:
            raise BridgeError(f"synchronized mining candidate differs: {local}")
        if local.suffix == ".json":
            try:
                json.loads(local.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise BridgeError(f"synchronized mining candidate is invalid JSON: {local}") from exc
        rows.append({
            "local": str(local), "remote": str(remote), "size": size,
            "sha256": digest, "disposition": disposition,
        })

    stage = command_contract.expected_stage_directory(
        "mining_postprocess", label, results
    )
    receipt_path = stage / "mining_postprocess.candidates-sync.json"
    receipt = {
        "schema_version": "1", "workflow": WORKFLOW,
        "kind": "airflow_slurm_mining_candidate_sync",
        "results_dir": str(results), "label": label,
        "request_sha256": request_sha, "recovered_after_terminal": recovered,
        "outputs": rows,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt, "receipt_sha256")
    if receipt_path.exists():
        existing = _json(receipt_path, "mining-candidate synchronization receipt")
        comparable_existing = [
            {key: row.get(key) for key in ("local", "remote", "size", "sha256")}
            for row in existing.get("outputs", [])
        ] if isinstance(existing.get("outputs"), list) else None
        comparable_current = [
            {key: row[key] for key in ("local", "remote", "size", "sha256")}
            for row in rows
        ]
        if (
            existing.get("workflow") != WORKFLOW
            or existing.get("kind") != receipt["kind"]
            or existing.get("results_dir") != str(results)
            or existing.get("label") != label
            or existing.get("request_sha256") != request_sha
            or comparable_existing != comparable_current
            or existing.get("receipt_sha256")
            != _canonical_sha256(existing, "receipt_sha256")
        ):
            raise BridgeError("existing mining-candidate synchronization receipt differs")
        return receipt_path
    _atomic_json(receipt_path, receipt)
    return receipt_path


def _synchronize_history_host_status(
    *, login: str, results: pathlib.Path, local_workspace: pathlib.Path,
    remote_workspace: pathlib.Path, label: str, request: dict[str, Any],
    recovered: bool,
) -> pathlib.Path:
    """Synchronize the nested resume evidence required by finalization."""

    if (
        request.get("workflow") != WORKFLOW
        or request.get("name") != "history_select"
        or request.get("label") != label
    ):
        raise BridgeError("history host-status synchronization request identity differs")
    stage = command_contract.expected_stage_directory("history_select", label, results)
    local = stage / "history-select.host.status.json"
    _safe_output_parent(local, results)
    remote = _workspace_mapping(local, local_workspace, remote_workspace)
    size, digest = _remote_file_evidence(login, remote)
    disposition = "reused"
    if local.exists() or local.is_symlink():
        if (
            not local.is_file()
            or local.is_symlink()
            or local.stat().st_size != size
            or _sha256_file(local) != digest
        ):
            raise BridgeError("existing controller history host status differs")
    else:
        _copy_remote_file(login, remote, local)
        disposition = "fetched"
    if local.stat().st_size != size or _sha256_file(local) != digest:
        raise BridgeError("synchronized history host status differs")
    host = _json(local, "history-select host status")
    if (
        host.get("workflow") != WORKFLOW
        or host.get("name") != "history-select"
        or host.get("status") != "ok"
        or host.get("exit_code") != 0
        or not isinstance(host.get("resume"), bool)
    ):
        raise BridgeError("history-select host status lacks valid resume evidence")

    receipt_path = stage / "history_select.host-status-sync.json"
    receipt = {
        "schema_version": "1", "workflow": WORKFLOW,
        "kind": "airflow_slurm_history_host_status_sync",
        "results_dir": str(results), "label": label,
        "request_sha256": request.get("request_sha256"),
        "recovered_after_terminal": recovered,
        "output": {
            "local": str(local), "remote": str(remote), "size": size,
            "sha256": digest, "disposition": disposition,
        },
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt, "receipt_sha256")
    if receipt_path.exists():
        existing = _json(receipt_path, "history host-status synchronization receipt")
        comparable = ("local", "remote", "size", "sha256")
        if (
            existing.get("workflow") != WORKFLOW
            or existing.get("kind") != receipt["kind"]
            or existing.get("results_dir") != str(results)
            or existing.get("label") != label
            or existing.get("request_sha256") != request.get("request_sha256")
            or not isinstance(existing.get("output"), dict)
            or {key: existing["output"].get(key) for key in comparable}
            != {key: receipt["output"][key] for key in comparable}
            or existing.get("receipt_sha256")
            != _canonical_sha256(existing, "receipt_sha256")
        ):
            raise BridgeError("existing history host-status sync receipt differs")
        return receipt_path
    _atomic_json(receipt_path, receipt)
    return receipt_path


def _synchronize_mining_history(
    *, login: str, results: pathlib.Path, local_workspace: pathlib.Path,
    remote_workspace: pathlib.Path, label: str, request: dict[str, Any],
    recovered: bool,
) -> pathlib.Path:
    """Synchronize the run-wide history-selection ledger with provenance."""

    match = re.fullmatch(r"iter([1-9][0-9]*)", label)
    if (
        match is None
        or request.get("workflow") != WORKFLOW
        or request.get("name") != "history_select"
        or request.get("label") != label
    ):
        raise BridgeError("mining-history synchronization request identity differs")
    stage = command_contract.expected_stage_directory("history_select", label, results)
    local = results / "mining_selection_history.json"
    _safe_output_parent(local, results)
    remote = _workspace_mapping(local, local_workspace, remote_workspace)
    size, digest = _remote_file_evidence(login, remote)
    receipt_path = stage / "history_select.mining-history-sync.json"

    if receipt_path.exists():
        receipt = _json(receipt_path, "mining-history synchronization receipt")
        current = receipt.get("current")
        if (
            receipt.get("workflow") != WORKFLOW
            or receipt.get("kind") != "airflow_slurm_mining_history_sync"
            or receipt.get("results_dir") != str(results)
            or receipt.get("label") != label
            or receipt.get("request_sha256") != request.get("request_sha256")
            or not isinstance(current, dict)
            or current.get("local") != str(local)
            or current.get("remote") != str(remote)
            or current.get("size") != size
            or current.get("sha256") != digest
            or receipt.get("receipt_sha256")
            != _canonical_sha256(receipt, "receipt_sha256")
        ):
            raise BridgeError("existing mining-history synchronization receipt differs")
        if (
            not local.is_file() or local.is_symlink()
            or local.stat().st_size != size or _sha256_file(local) != digest
        ):
            raise BridgeError("controller mining-history ledger differs from its receipt")
        return receipt_path

    previous: dict[str, Any] | None = None
    if local.exists() or local.is_symlink():
        if not local.is_file() or local.is_symlink() or local.stat().st_size == 0:
            raise BridgeError("existing controller mining-history ledger is unsafe")
        _json(local, "existing mining selection history")
        previous_digest = _sha256_file(local)
        if previous_digest == digest and local.stat().st_size == size:
            disposition = "reused"
        else:
            archive = (
                stage / ".tao-runtime" / "history-ledger-sync"
                / "mining_selection_history.before.json"
            )
            archive.parent.mkdir(parents=True, exist_ok=True)
            if archive.exists() or archive.is_symlink():
                if (
                    not archive.is_file() or archive.is_symlink()
                    or archive.stat().st_size != local.stat().st_size
                    or _sha256_file(archive) != previous_digest
                ):
                    raise BridgeError("existing archived mining-history ledger differs")
            else:
                shutil.copy2(local, archive)
            previous = {
                "path": str(archive), "size": archive.stat().st_size,
                "sha256": previous_digest,
            }
            _copy_remote_file(login, remote, local)
            disposition = "replaced_after_archive"
    else:
        _copy_remote_file(login, remote, local)
        disposition = "fetched"
    if (
        not local.is_file() or local.is_symlink()
        or local.stat().st_size != size or _sha256_file(local) != digest
    ):
        raise BridgeError("synchronized mining-history ledger differs")
    history = _json(local, "mining selection history")
    rows = history.get("iterations")
    number = int(match.group(1))
    matches = [
        row for row in rows
        if isinstance(row, dict) and row.get("iteration") == number
    ] if isinstance(rows, list) else []
    if len(matches) != 1:
        raise BridgeError("mining-history ledger does not bind exactly one current iteration")

    receipt = {
        "schema_version": "1", "workflow": WORKFLOW,
        "kind": "airflow_slurm_mining_history_sync",
        "results_dir": str(results), "label": label,
        "request_sha256": request.get("request_sha256"),
        "recovered_after_terminal": recovered, "previous": previous,
        "current": {
            "local": str(local), "remote": str(remote), "size": size,
            "sha256": digest, "disposition": disposition,
        },
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt, "receipt_sha256")
    _atomic_json(receipt_path, receipt)
    return receipt_path


def recover_materialize_sync(args: argparse.Namespace) -> dict[str, Any]:
    if args.name != "dataset_materialize" or args.label != "baseline":
        raise BridgeError(
            "recover-materialize-sync is valid only for baseline dataset_materialize"
        )
    if SAFE_LOGIN.fullmatch(args.login) is None:
        raise BridgeError("--login contains unsupported characters")
    results = args.results_dir.resolve()
    state = _json(results / "deft_state.json", "DEFT state")
    if (
        state.get("workflow") != WORKFLOW
        or state.get("config", {}).get("platform") != "slurm"
        or state["config"].get("orchestrator") != "airflow"
    ):
        raise BridgeError("run is not an Airflow-orchestrated SLURM IAA workflow")
    stage = results / "dataset_setup"
    request = _json(stage / "dataset_materialize.action.json", "action request")
    status = _json(stage / "dataset_materialize.status.json", "platform status")
    binding = _json(stage / "dataset_materialize.job-binding.json", "job binding")
    if (
        request.get("workflow") != WORKFLOW
        or request.get("name") != "dataset_materialize"
        or status.get("workflow") != WORKFLOW
        or status.get("name") != "dataset_materialize"
        or status.get("status") != "ok"
        or status.get("backend_state") != "COMPLETE"
        or status.get("exit_code") != 0
        or status.get("request_sha256") != request.get("request_sha256")
        or binding.get("request_sha256") != request.get("request_sha256")
        or status.get("job_id") != binding.get("job_id")
    ):
        raise BridgeError("terminal dataset materialization evidence is inconsistent")
    state_dir_raw = os.environ.get("TAO_STATE_DIR")
    if not state_dir_raw or not pathlib.Path(state_dir_raw).is_absolute():
        raise BridgeError("TAO_STATE_DIR must identify the approved Airflow shared state root")
    job = _json(
        pathlib.Path(state_dir_raw) / "jobs" / f"{binding['job_id']}.json",
        "SLURM job record",
    )
    if (
        job.get("platform") != "slurm"
        or job.get("terminal_state") != "COMPLETE"
        or job.get("backend_ref") != status.get("backend_ref")
        or job.get("id") != binding.get("job_id")
    ):
        raise BridgeError("SLURM job record is not the bound successful terminal job")
    local_workspace = pathlib.Path(state["config"]["workspace"]).resolve()
    receipt = _synchronize_dataset_materialization(
        login=args.login, results=results, local_workspace=local_workspace,
        remote_workspace=_remote_path(args.remote_workspace, "--remote-workspace"),
        recovered=True,
    )
    return {
        "status": "COMPLETE", "operation": "recover-materialize-sync",
        "job_id": job["id"], "backend_ref": job["backend_ref"],
        "receipt": str(receipt),
    }


def recover_gap_history_sync(args: argparse.Namespace) -> dict[str, Any]:
    """Recover only a completed gap action's omitted mutable ledger."""

    if args.name != "gap_analysis" or not re.fullmatch(r"iter[1-9][0-9]*", args.label):
        raise BridgeError(
            "recover-gap-history-sync requires gap_analysis and an iterN label"
        )
    if SAFE_LOGIN.fullmatch(args.login) is None:
        raise BridgeError("--login contains unsupported characters")
    results = args.results_dir.resolve()
    state = _json(results / "deft_state.json", "DEFT state")
    if (
        state.get("workflow") != WORKFLOW
        or state.get("config", {}).get("platform") != "slurm"
        or state["config"].get("orchestrator") != "airflow"
    ):
        raise BridgeError("run is not an Airflow-orchestrated SLURM IAA workflow")
    stage = command_contract.expected_stage_directory(args.name, args.label, results)
    request = _json(stage / "gap_analysis.action.json", "action request")
    status = _json(stage / "gap_analysis.status.json", "platform status")
    binding = _json(stage / "gap_analysis.job-binding.json", "job binding")
    if (
        request.get("workflow") != WORKFLOW
        or request.get("name") != args.name
        or request.get("label") != args.label
        or status.get("workflow") != WORKFLOW
        or status.get("name") != args.name
        or status.get("status") != "ok"
        or status.get("backend_state") != "COMPLETE"
        or status.get("exit_code") != 0
        or status.get("request_sha256") != request.get("request_sha256")
        or binding.get("request_sha256") != request.get("request_sha256")
        or status.get("job_id") != binding.get("job_id")
    ):
        raise BridgeError("terminal gap-analysis evidence is inconsistent")
    state_dir_raw = os.environ.get("TAO_STATE_DIR")
    if not state_dir_raw or not pathlib.Path(state_dir_raw).is_absolute():
        raise BridgeError("TAO_STATE_DIR must identify the approved Airflow shared state root")
    job = _json(
        pathlib.Path(state_dir_raw) / "jobs" / f"{binding['job_id']}.json",
        "SLURM job record",
    )
    if (
        job.get("platform") != "slurm"
        or job.get("terminal_state") != "COMPLETE"
        or job.get("backend_ref") != status.get("backend_ref")
        or job.get("id") != binding.get("job_id")
    ):
        raise BridgeError("SLURM job record is not the bound successful terminal job")
    local_workspace = pathlib.Path(state["config"]["workspace"]).resolve()
    receipt = _synchronize_gap_history(
        login=args.login, results=results, local_workspace=local_workspace,
        remote_workspace=_remote_path(args.remote_workspace, "--remote-workspace"),
        label=args.label, request=request, recovered=True,
    )
    return {
        "status": "COMPLETE", "operation": "recover-gap-history-sync",
        "job_id": job["id"], "backend_ref": job["backend_ref"],
        "receipt": str(receipt),
    }


def recover_mining_candidates_sync(args: argparse.Namespace) -> dict[str, Any]:
    """Recover candidate artifacts omitted after a completed remote action."""

    if args.name != "mining_postprocess" or not re.fullmatch(r"iter[1-9][0-9]*", args.label):
        raise BridgeError(
            "recover-mining-candidates-sync requires mining_postprocess and iterN"
        )
    if SAFE_LOGIN.fullmatch(args.login) is None:
        raise BridgeError("--login contains unsupported characters")
    results = args.results_dir.resolve()
    state = _json(results / "deft_state.json", "DEFT state")
    if (
        state.get("workflow") != WORKFLOW
        or state.get("config", {}).get("platform") != "slurm"
        or state["config"].get("orchestrator") != "airflow"
    ):
        raise BridgeError("run is not an Airflow-orchestrated SLURM IAA workflow")
    stage = command_contract.expected_stage_directory(args.name, args.label, results)
    request = _json(stage / "mining_postprocess.action.json", "action request")
    status = _json(stage / "mining_postprocess.status.json", "platform status")
    binding = _json(stage / "mining_postprocess.job-binding.json", "job binding")
    if (
        request.get("workflow") != WORKFLOW
        or request.get("name") != args.name
        or request.get("label") != args.label
        or status.get("workflow") != WORKFLOW
        or status.get("name") != args.name
        or status.get("status") != "ok"
        or status.get("backend_state") != "COMPLETE"
        or status.get("exit_code") != 0
        or status.get("request_sha256") != request.get("request_sha256")
        or binding.get("request_sha256") != request.get("request_sha256")
        or status.get("job_id") != binding.get("job_id")
    ):
        raise BridgeError("terminal mining-postprocess evidence is inconsistent")
    state_dir_raw = os.environ.get("TAO_STATE_DIR")
    if not state_dir_raw or not pathlib.Path(state_dir_raw).is_absolute():
        raise BridgeError("TAO_STATE_DIR must identify the approved Airflow shared state root")
    job = _json(
        pathlib.Path(state_dir_raw) / "jobs" / f"{binding['job_id']}.json",
        "SLURM job record",
    )
    if (
        job.get("platform") != "slurm"
        or job.get("terminal_state") != "COMPLETE"
        or job.get("backend_ref") != status.get("backend_ref")
        or job.get("id") != binding.get("job_id")
    ):
        raise BridgeError("SLURM job record is not the bound successful terminal job")
    local_workspace = pathlib.Path(state["config"]["workspace"]).resolve()
    receipt = _synchronize_mining_candidates(
        login=args.login, results=results, local_workspace=local_workspace,
        remote_workspace=_remote_path(args.remote_workspace, "--remote-workspace"),
        label=args.label, request=request, recovered=True,
    )
    return {
        "status": "COMPLETE", "operation": "recover-mining-candidates-sync",
        "job_id": job["id"], "backend_ref": job["backend_ref"],
        "receipt": str(receipt),
    }


def recover_history_finalization(args: argparse.Namespace) -> dict[str, Any]:
    """Finalize native history-selection success after host status was omitted."""

    if args.name != "history_select" or not re.fullmatch(r"iter[1-9][0-9]*", args.label):
        raise BridgeError(
            "recover-history-finalization requires history_select and iterN"
        )
    if SAFE_LOGIN.fullmatch(args.login) is None:
        raise BridgeError("--login contains unsupported characters")
    results = args.results_dir.resolve()
    state = _json(results / "deft_state.json", "DEFT state")
    if (
        state.get("workflow") != WORKFLOW
        or state.get("config", {}).get("platform") != "slurm"
        or state["config"].get("orchestrator") != "airflow"
    ):
        raise BridgeError("run is not an Airflow-orchestrated SLURM IAA workflow")
    stage = command_contract.expected_stage_directory(args.name, args.label, results)
    status = _json(stage / "history_select.status.json", "platform status")
    request_path = pathlib.Path(str(status.get("request_path", "")))
    if request_path.parent != stage or request_path.name not in {
        "history_select.action.json", "history_select.attempt-2.action.json",
    }:
        raise BridgeError("history finalization status points to an unsafe request")
    request = _json(request_path, "action request")
    binding = _json(
        pathlib.Path(str(request.get("job_binding_path", ""))), "job binding"
    )
    if (
        request.get("workflow") != WORKFLOW
        or request.get("name") != args.name
        or request.get("label") != args.label
        or status.get("workflow") != WORKFLOW
        or status.get("name") != args.name
        or status.get("status") != "error"
        or status.get("backend_state") != "COMPLETE"
        or status.get("backend_exit_code") != 0
        or status.get("artifact_error")
        != "history-select host status is missing valid resume evidence"
        or status.get("request_sha256") != request.get("request_sha256")
        or binding.get("request_sha256") != request.get("request_sha256")
        or status.get("job_id") != binding.get("job_id")
    ):
        raise BridgeError("history finalization evidence is not the recoverable defect")
    state_dir_raw = os.environ.get("TAO_STATE_DIR")
    if not state_dir_raw or not pathlib.Path(state_dir_raw).is_absolute():
        raise BridgeError("TAO_STATE_DIR must identify the approved Airflow shared state root")
    job_path = pathlib.Path(state_dir_raw) / "jobs" / f"{binding['job_id']}.json"
    job = _json(job_path, "SLURM job record")
    if (
        job.get("platform") != "slurm"
        or job.get("terminal_state") != "COMPLETE"
        or job.get("backend_ref") != status.get("backend_ref")
        or job.get("id") != binding.get("job_id")
    ):
        raise BridgeError("SLURM job record is not the bound successful terminal job")
    local_workspace = pathlib.Path(state["config"]["workspace"]).resolve()
    receipt = _synchronize_history_host_status(
        login=args.login, results=results, local_workspace=local_workspace,
        remote_workspace=_remote_path(args.remote_workspace, "--remote-workspace"),
        label=args.label, request=request, recovered=True,
    )
    status_path, code = producer.finalize(argparse.Namespace(
        request=request_path, job_record=job_path, native_exit_code=0,
    ))
    if code != 0:
        raise BridgeError(f"history finalization recovery failed: {status_path}")
    return {
        "status": "COMPLETE", "operation": "recover-history-finalization",
        "job_id": job["id"], "backend_ref": job["backend_ref"],
        "receipt": str(receipt), "platform_status": str(status_path),
    }


def recover_mining_history_sync(args: argparse.Namespace) -> dict[str, Any]:
    """Recover the ledger from one finalized successful history action."""

    if args.name != "history_select" or not re.fullmatch(r"iter[1-9][0-9]*", args.label):
        raise BridgeError("recover-mining-history-sync requires history_select and iterN")
    if SAFE_LOGIN.fullmatch(args.login) is None:
        raise BridgeError("--login contains unsupported characters")
    results = args.results_dir.resolve()
    state = _json(results / "deft_state.json", "DEFT state")
    if (
        state.get("workflow") != WORKFLOW
        or state.get("config", {}).get("platform") != "slurm"
        or state["config"].get("orchestrator") != "airflow"
    ):
        raise BridgeError("run is not an Airflow-orchestrated SLURM IAA workflow")
    stage = command_contract.expected_stage_directory(args.name, args.label, results)
    status = _json(stage / "history_select.status.json", "platform status")
    request_path = pathlib.Path(str(status.get("request_path", "")))
    request = _json(request_path, "action request")
    binding = _json(pathlib.Path(str(request["job_binding_path"])), "job binding")
    if (
        request.get("workflow") != WORKFLOW
        or request.get("name") != args.name
        or request.get("label") != args.label
        or status.get("status") != "ok"
        or status.get("backend_state") != "COMPLETE"
        or status.get("exit_code") != 0
        or status.get("request_sha256") != request.get("request_sha256")
        or binding.get("request_sha256") != request.get("request_sha256")
        or status.get("job_id") != binding.get("job_id")
    ):
        raise BridgeError("terminal history-selection evidence is inconsistent")
    state_dir_raw = os.environ.get("TAO_STATE_DIR")
    if not state_dir_raw or not pathlib.Path(state_dir_raw).is_absolute():
        raise BridgeError("TAO_STATE_DIR must identify the approved Airflow shared state root")
    job = _json(
        pathlib.Path(state_dir_raw) / "jobs" / f"{binding['job_id']}.json",
        "SLURM job record",
    )
    if (
        job.get("platform") != "slurm"
        or job.get("terminal_state") != "COMPLETE"
        or job.get("backend_ref") != status.get("backend_ref")
        or job.get("id") != binding.get("job_id")
    ):
        raise BridgeError("SLURM job record is not the bound successful terminal job")
    local_workspace = pathlib.Path(state["config"]["workspace"]).resolve()
    receipt = _synchronize_mining_history(
        login=args.login, results=results, local_workspace=local_workspace,
        remote_workspace=_remote_path(args.remote_workspace, "--remote-workspace"),
        label=args.label, request=request, recovered=True,
    )
    return {
        "status": "COMPLETE", "operation": "recover-mining-history-sync",
        "job_id": job["id"], "backend_ref": job["backend_ref"],
        "receipt": str(receipt),
    }


def recover_visualization_host_log(args: argparse.Namespace) -> dict[str, Any]:
    """Recover one status-bound visualization log without rerunning compute."""

    if args.name not in {"visualize_prepare", "visualize_finish"} or not re.fullmatch(
        r"iter[1-9][0-9]*", args.label
    ):
        raise BridgeError(
            "recover-visualization-host-log requires a visualization action and iterN"
        )
    if SAFE_LOGIN.fullmatch(args.login) is None:
        raise BridgeError("--login contains unsupported characters")
    results = args.results_dir.resolve()
    state = _json(results / "deft_state.json", "DEFT state")
    if (
        state.get("workflow") != WORKFLOW
        or state.get("config", {}).get("platform") != "slurm"
        or state["config"].get("orchestrator") != "airflow"
    ):
        raise BridgeError("run is not an Airflow-orchestrated SLURM IAA workflow")
    stage = command_contract.expected_stage_directory(args.name, args.label, results)
    platform_status_path = stage / f"{args.name}.status.json"
    platform_status = _json(platform_status_path, "visualization platform status")
    request_path = pathlib.Path(str(platform_status.get("request_path", "")))
    request = _json(request_path, "visualization action request")
    binding = _json(pathlib.Path(str(request.get("job_binding_path", ""))), "job binding")
    host_status_path, expected = _visualization_side_outputs(
        results, args.label, state, args.name
    )
    host_status = _json(host_status_path, "visualization host status")
    expected_log = stage / f"{args.name.replace('_', '-')}.host.log"
    if (
        request.get("workflow") != WORKFLOW
        or request.get("name") != args.name
        or request.get("label") != args.label
        or platform_status.get("status") != "ok"
        or platform_status.get("backend_state") != "COMPLETE"
        or platform_status.get("backend_exit_code") != 0
        or platform_status.get("exit_code") != 0
        or platform_status.get("request_sha256") != request.get("request_sha256")
        or binding.get("request_sha256") != request.get("request_sha256")
        or platform_status.get("job_id") != binding.get("job_id")
        or host_status.get("workflow") != WORKFLOW
        or host_status.get("name") != args.name.replace("_", "-")
        or host_status.get("status") != "ok" or host_status.get("exit_code") != 0
        or host_status.get("fresh_outputs") != [str(path) for path in expected]
        or host_status.get("log_path") != str(expected_log)
    ):
        raise BridgeError("terminal visualization evidence is inconsistent")
    state_dir_raw = os.environ.get("TAO_STATE_DIR")
    if not state_dir_raw or not pathlib.Path(state_dir_raw).is_absolute():
        raise BridgeError("TAO_STATE_DIR must identify the approved Airflow shared state root")
    job = _json(
        pathlib.Path(state_dir_raw) / "jobs" / f"{binding['job_id']}.json",
        "SLURM job record",
    )
    if (
        job.get("platform") != "slurm" or job.get("id") != binding.get("job_id")
        or job.get("terminal_state") != "COMPLETE"
        or job.get("backend_ref") != platform_status.get("backend_ref")
    ):
        raise BridgeError("SLURM job record is not the bound successful terminal job")

    local_workspace = pathlib.Path(state["config"]["workspace"]).resolve()
    remote_workspace = _remote_path(args.remote_workspace, "--remote-workspace")
    remote_log = _workspace_mapping(expected_log, local_workspace, remote_workspace)
    quoted = shlex.quote(str(remote_log))
    probe = _ssh(
        args.login,
        "set -Eeuo pipefail; "
        f"if test -f {quoted} && test ! -L {quoted}; then printf 'PRESENT'; "
        f"elif test ! -e {quoted}; then printf 'MISSING'; else printf 'UNSAFE'; fi",
        operation="classify visualization host-log synchronization",
    ).strip()
    if probe == "PRESENT":
        receipt = _synchronize_visualization_outputs(
            login=args.login, results=results, local_workspace=local_workspace,
            remote_workspace=remote_workspace, label=args.label, state=state,
            name=args.name, recovered=True,
        )
        return {
            "status": "COMPLETE", "operation": "recover-visualization-host-log",
            "mode": "remote_exact", "job_id": job["id"],
            "backend_ref": job["backend_ref"], "receipt": str(receipt),
            "host_log": str(expected_log),
        }
    if probe != "MISSING":
        raise BridgeError("remote visualization host log is unsafe")

    # The old bridge could lose visualize_prepare's log when the subsequent
    # visualize_finish exact-tree stage mirrored the controller omission back
    # to SLURM. Recover only that proven ordering, using the immutable native
    # action log as the replacement evidence source.
    if args.name != "visualize_prepare":
        raise BridgeError("remote visualization host log is absent and unrecoverable")
    finish_status = _json(stage / "visualize_finish.status.json", "later visualization status")
    if (
        finish_status.get("status") != "ok"
        or finish_status.get("backend_state") != "COMPLETE"
        or finish_status.get("exit_code") != 0
        or not isinstance(finish_status.get("started_ns"), int)
        or finish_status["started_ns"] <= platform_status.get("started_ns", 0)
    ):
        raise BridgeError("missing prepare log is not followed by one successful finish action")
    sync_path = stage / f"{args.name}.output-sync.json"
    sync = _json(sync_path, "visualization output sync receipt")
    if (
        sync.get("workflow") != WORKFLOW
        or sync.get("kind") != "airflow_slurm_visualization_output_sync"
        or sync.get("name") != args.name or sync.get("label") != args.label
        or sync.get("host_status_sha256") != _sha256_file(host_status_path)
        or sync.get("receipt_sha256") != _canonical_sha256(sync, "receipt_sha256")
        or not isinstance(sync.get("outputs"), list) or not sync["outputs"]
    ):
        raise BridgeError("prior visualization output synchronization is inconsistent")
    action_log = pathlib.Path(str(platform_status.get("log_path", "")))
    try:
        action_log.relative_to(stage)
    except ValueError as exc:
        raise BridgeError("visualization platform log escapes its stage") from exc
    if not action_log.is_file() or action_log.is_symlink() or action_log.stat().st_size <= 0:
        raise BridgeError("visualization platform log is missing or unsafe")
    action_text = action_log.read_text(encoding="utf-8", errors="replace")
    marker = f"IAA_ADAPTER_COMPLETE operation={args.name} label={args.label}"
    if marker not in action_text or str(expected_log) not in action_text or str(host_status_path) not in action_text:
        raise BridgeError("visualization platform log does not prove adapter completion")
    if expected_log.exists() or expected_log.is_symlink():
        raise BridgeError("visualization host-log recovery target already exists")
    recovered_text = (
        "RECOVERED_FROM_IMMUTABLE_PLATFORM_ACTION_LOG\n"
        f"source_sha256={_sha256_file(action_log)}\n"
        + action_text
    )
    _write_text_atomic(expected_log, recovered_text)
    recovery_dir = stage / ".tao-runtime" / "visualize_prepare-host-log-recovery"
    recovery_dir.mkdir(parents=True, exist_ok=False)
    evidence = {
        "schema_version": "1", "workflow": WORKFLOW,
        "kind": "airflow_slurm_visualization_host_log_recovery",
        "name": args.name, "label": args.label, "job_id": job["id"],
        "backend_ref": job["backend_ref"], "remote_log": str(remote_log),
        "remote_disposition": "missing_after_later_exact_tree_stage",
        "platform_status_sha256": _sha256_file(platform_status_path),
        "host_status_sha256": _sha256_file(host_status_path),
        "output_sync_receipt_sha256": _sha256_file(sync_path),
        "source_action_log": str(action_log),
        "source_action_log_sha256": _sha256_file(action_log),
        "recovered_host_log": str(expected_log),
        "recovered_host_log_sha256": _sha256_file(expected_log),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    evidence["evidence_sha256"] = _canonical_sha256(evidence, "evidence_sha256")
    evidence_path = recovery_dir / "evidence.json"
    _atomic_json(evidence_path, evidence)
    return {
        "status": "COMPLETE", "operation": "recover-visualization-host-log",
        "mode": "platform_log_recovery", "job_id": job["id"],
        "backend_ref": job["backend_ref"], "evidence": str(evidence_path),
        "host_log": str(expected_log),
    }


def recover_monitoring(args: argparse.Namespace) -> dict[str, Any]:
    """Reconcile one native success after Airflow monitoring alone failed."""

    if SAFE_LOGIN.fullmatch(args.login) is None:
        raise BridgeError("--login contains unsupported characters")
    results = args.results_dir.resolve()
    state = _json(results / "deft_state.json", "DEFT state")
    if (
        state.get("workflow") != WORKFLOW
        or state.get("config", {}).get("platform") != "slurm"
        or state["config"].get("orchestrator") != "airflow"
    ):
        raise BridgeError("run is not an Airflow-orchestrated SLURM IAA workflow")
    stage = command_contract.expected_stage_directory(args.name, args.label, results)
    request_path = stage / f"{args.name}.action.json"
    request = _json(request_path, "action request")
    binding = _json(stage / f"{args.name}.job-binding.json", "job binding")
    if (
        request.get("workflow") != WORKFLOW
        or request.get("name") != args.name
        or request.get("label") != args.label
        or binding.get("request_sha256") != request.get("request_sha256")
    ):
        raise BridgeError("monitoring recovery request/binding identity differs")
    state_dir_raw = os.environ.get("TAO_STATE_DIR")
    if not state_dir_raw or not pathlib.Path(state_dir_raw).is_absolute():
        raise BridgeError("TAO_STATE_DIR must identify the approved Airflow shared state root")
    job_path = pathlib.Path(state_dir_raw) / "jobs" / f"{binding['job_id']}.json"
    job = _json(job_path, "SLURM job record")
    backend_ref = job.get("backend_ref")
    transitions = job.get("transitions")
    false_terminal = (
        job.get("terminal_state") == "ERROR"
        and job.get("terminal_write_by") == "poller"
        and isinstance(transitions, list)
        and len(transitions) == 3
        and transitions[-1].get("message") == "Airflow orchestration terminated as ERROR"
    )
    still_running = (
        job.get("terminal_state") is None
        and isinstance(transitions, list)
        and len(transitions) == 2
        and transitions[-1].get("state") == "RUNNING"
    )
    if (
        job.get("platform") != "slurm"
        or job.get("id") != binding.get("job_id")
        or not isinstance(backend_ref, str)
        or not backend_ref.isdigit()
        or not (false_terminal or still_running)
        or (stage / f"{args.name}.status.json").exists()
    ):
        raise BridgeError("job is not one recoverable Airflow-monitoring failure")

    runtime = pathlib.Path(request["platform_runtime_dir"])
    _plan_path, envelope_path = _orchestration_paths(runtime, job["id"])
    envelope = _json(envelope_path, "Airflow orchestration envelope")
    receipt = _json(
        pathlib.Path(envelope["receipt_path"]), "Airflow delegation receipt"
    )
    airflow_ref = f"tao_deft_iaa_action_v1/{envelope['orchestration_id']}"
    airflow_status = _json_result(_run(
        [sys.executable, str(SCRIPT_DIR / "airflow_orchestrator.py"), "status",
         "--envelope", str(envelope_path), "--backend-ref", airflow_ref],
        operation="verify failed Airflow monitoring run",
    ), "Airflow monitoring status")
    orchestration_log = pathlib.Path(envelope["log_path"])
    log_text = orchestration_log.read_text(encoding="utf-8")
    if (
        airflow_status.get("status") != "ERROR"
        or receipt.get("job_id") != job["id"]
        or receipt.get("compute_backend_ref") != backend_ref
        or receipt.get("status") != "RUNNING"
        or "slurm_action[status]: bounded SLURM SSH operation exceeded" not in log_text
        or "OPERATION=cancel" in log_text
    ):
        raise BridgeError("Airflow evidence is not a pure monitoring-timeout failure")

    local_workspace = pathlib.Path(state["config"]["workspace"]).resolve()
    remote_workspace = _remote_path(args.remote_workspace, "--remote-workspace")
    remote_results = _workspace_mapping(results, local_workspace, remote_workspace)
    log_dir = _remote_path(remote_workspace / "slurm-logs", "SLURM log directory")
    consumer = BANK / "skills/platform/tao-run-on-slurm/scripts/slurm_action.py"
    status_command = [
        sys.executable, str(consumer), "status", "--login", args.login,
        "--job-id", job["id"], "--backend-ref", backend_ref,
        "--request", str(request_path), "--remote-results", str(remote_results),
        "--log-dir", str(log_dir),
    ]
    deadline = time.monotonic() + args.deadline
    unknown = 0
    native: dict[str, Any] = {}
    while time.monotonic() < deadline:
        native = _json_result(_run(
            status_command, timeout=180,
            operation="recover native SLURM monitoring",
        ), "native monitoring recovery")
        native_status = str(native.get("status", "UNKNOWN")).upper()
        if native_status == "COMPLETE":
            break
        if native_status in {"ERROR", "CANCELED"}:
            raise BridgeError(
                f"native SLURM terminated as {native_status}; monitoring-only "
                "success recovery is forbidden"
            )
        unknown = unknown + 1 if native_status == "UNKNOWN" else 0
        if unknown >= 3:
            raise BridgeError("native SLURM status remained UNKNOWN during recovery")
        time.sleep(args.controller_poll_interval)
    else:
        raise BridgeError("native SLURM monitoring recovery exceeded its deadline")

    log_path = pathlib.Path(request["log_path"])
    outputs = [pathlib.Path(value) for value in request["fresh_outputs"]]
    if (
        not log_path.is_file() or log_path.is_symlink() or log_path.stat().st_size == 0
        or any(
            not path.is_file() or path.is_symlink() or path.stat().st_size == 0
            for path in outputs
        )
    ):
        raise BridgeError("native COMPLETE did not synchronize exact action artifacts")

    recovery_dir = runtime / "airflow-monitoring-recovery" / job["id"]
    archive = recovery_dir / "job-record.before.json"
    evidence_path = recovery_dir / "evidence.json"
    original_text = job_path.read_text(encoding="utf-8")
    _write_text_atomic(archive, original_text)
    corrected = dict(job)
    corrected_transitions = list(transitions[:-1] if false_terminal else transitions)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    corrected_transitions.append({
        "ts": now, "state": "COMPLETE",
        "message": f"native COMPLETE after Airflow timeout; evidence={evidence_path}",
        "source": "agent",
    })
    corrected["transitions"] = corrected_transitions
    corrected["terminal_state"] = "COMPLETE"
    corrected["terminal_write_by"] = "agent"
    corrected_text = json.dumps(corrected, indent=2, sort_keys=True) + "\n"
    evidence = {
        "schema_version": "1", "workflow": WORKFLOW,
        "kind": "airflow_slurm_monitoring_recovery", "name": args.name,
        "label": args.label, "job_id": job["id"], "backend_ref": backend_ref,
        "request_sha256": request["request_sha256"],
        "airflow_orchestration_id": envelope["orchestration_id"],
        "airflow_status": "ERROR", "native_status": "COMPLETE",
        "original_terminal_state": job.get("terminal_state"),
        "original_job_record_sha256": hashlib.sha256(original_text.encode()).hexdigest(),
        "corrected_job_record_sha256": hashlib.sha256(corrected_text.encode()).hexdigest(),
        "orchestration_log_sha256": _sha256_file(orchestration_log),
        "native_log_sha256": _sha256_file(log_path),
        "outputs": [
            {"path": str(path), "size": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in outputs
        ],
        "recovered_at": now,
    }
    evidence["evidence_sha256"] = _canonical_sha256(evidence, "evidence_sha256")
    _atomic_json(evidence_path, evidence)
    _write_text_atomic(job_path, corrected_text)
    status_path, code = producer.finalize(argparse.Namespace(
        request=request_path, job_record=job_path, native_exit_code=0,
    ))
    if code != 0:
        raise BridgeError(f"monitoring recovery finalization failed: {status_path}")
    return {
        "status": "COMPLETE", "operation": "recover-monitoring",
        "job_id": job["id"], "backend_ref": backend_ref,
        "evidence": str(evidence_path), "platform_status": str(status_path),
    }


def _prepare_action(args: argparse.Namespace) -> tuple[pathlib.Path, dict[str, Any]]:
    results = args.results_dir.resolve()
    state = _json(results / "deft_state.json", "DEFT state")
    if state.get("workflow") != WORKFLOW or state.get("config", {}).get("platform") != "slurm":
        raise BridgeError("run is not a canonical SLURM IAA workflow")
    if state["config"].get("orchestrator") != "airflow":
        raise BridgeError("run is not configured for Airflow orchestration")
    stage = command_contract.expected_stage_directory(args.name, args.label, results)
    outputs = command_contract.expected_fresh_outputs(args.name, args.label, results)
    image = command_contract.expected_image_kind(args.name)
    command = command_contract.expected_container_command(args.name, args.label, state["config"])
    return producer.prepare(argparse.Namespace(
        results_dir=results, image=image, stage_dir=stage, name=args.name,
        pass_hf_token=bool(state["config"].get("requires_hf_token")),
        fresh_output=outputs, command=command,
    ))


def _prepare_train_output_replay(
    args: argparse.Namespace,
) -> tuple[pathlib.Path, dict[str, Any]]:
    """Classify one historical checkpoint-sync loss and mint its only replay."""
    results = args.results_dir.resolve()
    shared_root = args.shared_root.resolve()
    try:
        results.relative_to(shared_root)
    except ValueError as exc:
        raise BridgeError("checkpoint-loss recovery results must be under shared root") from exc
    state = _json(results / "deft_state.json", "DEFT state")
    if (
        state.get("workflow") != WORKFLOW
        or state.get("config", {}).get("platform") != "slurm"
        or state["config"].get("orchestrator") != "airflow"
        or args.name != "train"
        or re.fullmatch(r"iter[1-9][0-9]*", args.label) is None
    ):
        raise BridgeError("recover-train-output-loss requires Airflow-over-SLURM iterN train")
    stage = command_contract.expected_stage_directory(args.name, args.label, results)
    outputs = command_contract.expected_fresh_outputs(args.name, args.label, results)
    image = command_contract.expected_image_kind(args.name)
    command = command_contract.expected_container_command(
        args.name, args.label, state["config"]
    )
    namespace = argparse.Namespace(
        results_dir=results, image=image, stage_dir=stage, name=args.name,
        pass_hf_token=bool(state["config"].get("requires_hf_token")),
        fresh_output=outputs, command=command,
        recovery_evidence=stage / ".tao-runtime/train.output-replay.evidence.json",
    )
    evidence_path = namespace.recovery_evidence
    if evidence_path.exists():
        return producer.train_output_replay(namespace)
    if producer._train_checkpoint_candidates(stage):  # noqa: SLF001
        raise BridgeError("checkpoint-loss recovery is forbidden while a checkpoint exists")
    if (stage / "train.output-sync.json").exists():
        raise BridgeError("checkpoint-loss recovery is forbidden after train output sync")
    local_workspace = pathlib.Path(state["config"]["workspace"]).resolve()
    remote_workspace = _remote_path(args.remote_workspace, "--remote-workspace")
    remote_stage = _workspace_mapping(stage, local_workspace, remote_workspace)
    remote_inventory = _remote_train_output_evidence(
        args.login, remote_stage, allow_empty=True
    )
    if remote_inventory:
        raise BridgeError("remote train checkpoints still exist; synchronize instead of replaying")

    state_dir_raw = os.environ.get("TAO_STATE_DIR")
    if not state_dir_raw or not pathlib.Path(state_dir_raw).is_absolute():
        raise BridgeError("TAO_STATE_DIR must identify the approved Airflow shared state root")
    state_dir = pathlib.Path(state_dir_raw).resolve()
    try:
        state_dir.relative_to(shared_root)
    except ValueError as exc:
        raise BridgeError("checkpoint-loss recovery TAO_STATE_DIR must be under shared root") from exc
    archive_root = stage / ".tao-runtime/train.output-replay.prior"
    paths = {
        "prior_train_request": stage / "train.attempt-2.action.json",
        "prior_train_log": stage / "train.attempt-2.log",
        "prior_train_platform_status_source": stage / "train.status.json",
        "prior_train_platform_status": archive_root / "train.status.json",
        "prior_train_workload_status_source": stage / "status.json",
        "prior_train_workload_status": archive_root / "status.json",
        "publisher_request": stage / "publish_checkpoint.action.json",
        "publisher_log": stage / "publish_checkpoint.log",
        "publisher_platform_status": stage / "publish_checkpoint.status.json",
    }
    train_status = _json(paths["prior_train_platform_status_source"], "train status")
    publisher_status = _json(paths["publisher_platform_status"], "publisher status")
    train_job_id = train_status.get("job_id")
    publisher_job_id = publisher_status.get("job_id")
    if (
        not isinstance(train_job_id, str) or SAFE_SLURM_TOKEN.fullmatch(train_job_id) is None
        or not isinstance(publisher_job_id, str)
        or SAFE_SLURM_TOKEN.fullmatch(publisher_job_id) is None
    ):
        raise BridgeError("checkpoint-loss recovery status has an unsafe job identity")
    paths["prior_train_job_record"] = state_dir / "jobs" / f"{train_job_id}.json"
    paths["publisher_job_record"] = state_dir / "jobs" / f"{publisher_job_id}.json"
    archive_roles = {
        "prior_train_platform_status", "prior_train_workload_status"
    }
    rows: list[dict[str, Any]] = []
    for role in (
        "prior_train_request", "prior_train_job_record", "prior_train_log",
        "prior_train_platform_status", "prior_train_workload_status",
        "publisher_request", "publisher_job_record", "publisher_log",
        "publisher_platform_status",
    ):
        source = paths[f"{role}_source"] if role in archive_roles else paths[role]
        if not source.is_file() or source.is_symlink() or source.stat().st_size <= 0:
            raise BridgeError(f"checkpoint-loss recovery {role} is missing or unsafe")
        rows.append({
            "role": role, "path": str(paths[role]),
            "size": source.stat().st_size, "sha256": _sha256_file(source),
        })
    payload = {
        "schema_version": "1", "workflow": WORKFLOW,
        "kind": "airflow_slurm_train_output_loss", "platform": "slurm",
        "name": "train", "label": args.label,
        "results_dir": str(results), "stage_dir": str(stage),
        "prior_train_attempt": 2, "publisher_attempt": 1,
        "controller_checkpoint_count": 0, "remote_checkpoint_count": 0,
        "remote_checkpoint_inventory_sha256": hashlib.sha256(b"").hexdigest(),
        "classifier": "successful-train-checkpoint-not-synchronized",
        "artifacts": rows,
        "classified_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    payload["evidence_sha256"] = _canonical_sha256(payload, "evidence_sha256")
    _atomic_json(evidence_path, payload)
    return producer.train_output_replay(namespace)


def run_action(args: argparse.Namespace) -> dict[str, Any]:
    if SAFE_LOGIN.fullmatch(args.login) is None:
        raise BridgeError("--login contains unsupported characters")
    for flag, value in (
        ("--account", args.account),
        ("--cpu-partition", args.cpu_partition),
        ("--gpu-partition", args.gpu_partition),
    ):
        if SAFE_SLURM_TOKEN.fullmatch(value) is None:
            raise BridgeError(f"{flag} contains unsupported characters")
    shared_root = args.shared_root.resolve()
    results = args.results_dir.resolve()
    request_path, request = (
        _prepare_train_output_replay(args)
        if getattr(args, "operation", "run") == "recover-train-output-loss"
        else _prepare_action(args)
    )
    state = _json(results / "deft_state.json", "DEFT state")
    local_workspace = pathlib.Path(state["config"]["workspace"]).resolve()
    try:
        results.relative_to(shared_root)
        request_path.relative_to(shared_root)
    except ValueError as exc:
        raise BridgeError("results and action request must be under the Airflow shared root") from exc
    state_dir_raw = os.environ.get("TAO_STATE_DIR")
    if not state_dir_raw or not pathlib.Path(state_dir_raw).is_absolute():
        raise BridgeError("TAO_STATE_DIR must identify the approved Airflow shared state root")
    state_dir = pathlib.Path(state_dir_raw).resolve()
    try:
        state_dir.relative_to(shared_root)
    except ValueError as exc:
        raise BridgeError("TAO_STATE_DIR must be under the Airflow shared root") from exc
    if pathlib.Path(request["job_state_dir"]).resolve() != state_dir:
        raise BridgeError(
            "prepared action job_state_dir does not match TAO_STATE_DIR; before native "
            "submission, run recover-bound-presubmit when a binding exists, then "
            "rebind-airflow-state with the approved shared TAO_STATE_DIR"
        )
    remote_workspace = _remote_path(args.remote_workspace, "--remote-workspace")
    backend_dataset_root = None
    if args.backend_dataset_root is not None:
        backend_dataset_root = _remote_path(
            args.backend_dataset_root, "--backend-dataset-root"
        )
        _verify_backend_dataset(
            login=args.login,
            local=pathlib.Path(state["config"]["dataset_root"]),
            remote=backend_dataset_root,
        )
    remote_results = _workspace_mapping(results, local_workspace, remote_workspace)
    runtime = pathlib.Path(request["platform_runtime_dir"])
    stage_receipt = runtime / "slurm-results-tree.staged.json"
    _stage_tree(
        source=results, login=args.login, target=remote_results, receipt=stage_receipt,
        incremental_existing=True,
    )
    for field in ("controller_snapshot", "patches_snapshot"):
        snapshot = request[field]
        source = pathlib.Path(snapshot["root"])
        target = _workspace_mapping(source, local_workspace, remote_workspace)
        _stage_tree(
            source=source, login=args.login, target=target,
            receipt=runtime / f"{field}.slurm-staged.json",
            request=request_path, snapshot_field=field,
        )
    if args.name == "dataset_rebuild":
        _verify_remote_archives(
            login=args.login, state=state, local_workspace=local_workspace,
            remote_workspace=remote_workspace,
        )
    mount_rows = []
    mount_map: dict[str, str] = {}
    for row in request["mounts"]:
        source = row["source"]
        if source not in mount_map:
            if (
                backend_dataset_root is not None
                and pathlib.Path(source) == pathlib.Path(state["config"]["dataset_root"])
            ):
                mount_map[source] = str(backend_dataset_root)
            else:
                mount_map[source] = str(
                    _workspace_mapping(pathlib.Path(source), local_workspace, remote_workspace)
                )
            mount_rows.append((source, mount_map[source]))
    if backend_dataset_root is not None and str(pathlib.Path(
        state["config"]["dataset_root"]
    )) not in mount_map:
        raise BridgeError(
            "--backend-dataset-root requires an existing request-owned dataset mount"
        )
    remote_absent = [
        _workspace_mapping(pathlib.Path(value), local_workspace, remote_workspace)
        for value in request["staging_absent_paths"]
    ]
    command = "set -Eeuo pipefail; " + " ".join(
        f"test ! -e {shlex.quote(str(path))};" for path in remote_absent
    )
    _ssh(args.login, command, operation="verify remote action outputs are absent")
    attest_args = argparse.Namespace(
        request=request_path,
        backend_scope=str(_workspace_mapping(
            pathlib.Path(request["stage_dir"]), local_workspace, remote_workspace,
        )),
        absent_path=request["staging_absent_paths"], mount_map=mount_rows,
    )
    producer.attest_staged(attest_args)
    env = dict(os.environ)
    state_dir = env.get("TAO_STATE_DIR")
    if not state_dir or not pathlib.Path(state_dir).is_absolute():
        raise BridgeError("TAO_STATE_DIR must identify the approved Airflow shared state root")
    job_id, job_record = _open_job(
        request=request, backend_scope=pathlib.Path(attest_args.backend_scope), env=env,
    )
    binding = producer.bind_job(
        argparse.Namespace(request=request_path, job_record=job_record)
    )
    log_dir = _remote_path(args.remote_workspace / "slurm-logs", "SLURM log directory")
    _ssh(
        args.login, f"mkdir -p -- {shlex.quote(str(log_dir))}",
        operation="create SLURM log directory",
    )
    image = _remote_path(
        args.ds_sqsh if request["image_kind"] == "ds" else args.pyt_sqsh,
        "selected SLURM image",
    )
    rendered = runtime / f"job_{job_id}.sbatch"
    _write_text_atomic(rendered, _render(
        request=request, mount_map=mount_map, job_id=job_id, image=image,
        log_dir=log_dir, account=args.account, cpu_partition=args.cpu_partition,
        gpu_partition=args.gpu_partition,
        cpu_time_minutes=args.cpu_time_minutes,
        gpu_time_minutes=args.gpu_time_minutes,
    ))
    remote_script = _workspace_mapping(rendered, local_workspace, remote_workspace)
    consumer = _stage_consumer(shared_root)
    interpreter = "/usr/bin/python3"
    plan_path, envelope_path = _orchestration_paths(runtime, job_id)
    commands = {
        "submit": [
            interpreter, str(consumer), "submit", "--login", args.login,
            "--job-id", job_id, "--rendered-script", str(rendered),
            "--remote-script", str(remote_script), "--request", str(request_path),
            "--job-binding", str(binding),
        ],
        "status": [
            interpreter, str(consumer), "status", "--login", args.login,
            "--job-id", job_id, "--backend-ref", "{backend_ref}",
            "--request", str(request_path), "--remote-results", str(remote_results),
            "--log-dir", str(log_dir),
        ],
        "logs": [
            interpreter, str(consumer), "logs", "--login", args.login,
            "--job-id", job_id, "--backend-ref", "{backend_ref}",
            "--log-dir", str(log_dir), "--tail", "500",
        ],
        "cancel": [
            interpreter, str(consumer), "cancel", "--login", args.login,
            "--job-id", job_id, "--backend-ref", "{backend_ref}", "--confirm",
        ],
    }
    _atomic_json(plan_path, {
        "commands": commands,
        "expected_outputs": request["fresh_outputs"],
        "poll_interval_s": args.compute_poll_interval,
        "deadline_s": args.deadline,
        "unknown_status_limit": 3,
        "retain_on_failure": True,
        "forward_env": request["forward_env"],
    })
    if orchestrator.prepare(argparse.Namespace(
        compute_platform="slurm", compute_kind="action",
        compute_request=request_path, job_record=job_record,
        job_binding=binding, consumer_plan=plan_path, output=envelope_path,
    )) != 0:
        raise BridgeError("Airflow envelope preparation failed")
    submitted = _json_result(_run(
        [sys.executable, str(SCRIPT_DIR / "airflow_orchestrator.py"), "submit",
         "--envelope", str(envelope_path)],
        operation="submit Airflow SLURM orchestration",
    ), "Airflow submit")
    airflow_ref = submitted.get("backend_ref")
    if not isinstance(airflow_ref, str) or not airflow_ref:
        raise BridgeError("Airflow submit returned no backend reference")
    envelope = _json(envelope_path, "Airflow orchestration envelope")
    receipt_path = pathlib.Path(envelope["receipt_path"])
    backend_ref: str | None = None
    running_marked = False
    deadline = time.monotonic() + args.deadline + 300
    last_status = "PENDING"
    last_status_payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status = _json_result(_run(
            [sys.executable, str(SCRIPT_DIR / "airflow_orchestrator.py"), "status",
             "--envelope", str(envelope_path), "--backend-ref", airflow_ref],
            operation="poll Airflow SLURM orchestration",
        ), "Airflow status")
        last_status_payload = status
        last_status = str(status.get("status", "UNKNOWN")).upper()
        if receipt_path.is_file():
            receipt = _json(receipt_path, "Airflow delegation receipt")
            candidate = receipt.get("compute_backend_ref")
            if isinstance(candidate, str) and candidate:
                backend_ref = candidate
                if not running_marked:
                    _mark_job(
                        job_id=job_id, state="RUNNING", env=env,
                        backend_ref=backend_ref, message="submitted by Airflow SLURM consumer",
                    )
                    running_marked = True
        if last_status in TERMINAL:
            break
        time.sleep(args.controller_poll_interval)
    if last_status != "COMPLETE":
        if running_marked:
            direct_command = [
                backend_ref if token == "{backend_ref}" else token
                for token in commands["status"]
            ]
            direct = _json_result(_run(
                direct_command, timeout=180,
                operation="reconcile native SLURM state after Airflow failure",
            ), "native SLURM reconciliation")
            direct_status = str(direct.get("status", "UNKNOWN")).upper()
            if direct_status in {"PENDING", "RUNNING", "UNKNOWN"}:
                raise BridgeError(
                    f"Airflow orchestration ended as {last_status} while native "
                    f"SLURM remains {direct_status}; job record preserved for "
                    "recover-monitoring"
                )
            if direct_status == "COMPLETE":
                raise BridgeError(
                    f"Airflow orchestration ended as {last_status} after native "
                    "SLURM completed and synchronized; job record preserved for "
                    "recover-monitoring"
                )
            _mark_job(
                job_id=job_id,
                state="ERROR" if direct_status == "ERROR" else "CANCELED",
                env=env,
                message=(
                    f"native SLURM terminated as {direct_status} after Airflow "
                    f"orchestration ended as {last_status}"
                ),
            )
            native_exit = direct.get("native_exit_code")
            if not isinstance(native_exit, int) or isinstance(native_exit, bool):
                native_exit = None
            status_path, _code = producer.finalize(argparse.Namespace(
                request=request_path, job_record=job_record,
                native_exit_code=native_exit,
            ))
            raise BridgeError(
                f"Airflow SLURM orchestration terminated as {last_status}; "
                f"failure evidence finalized at {status_path}"
            )
        _prove_remote_job_absent(args.login, job_id)
        _mark_job(
            job_id=job_id, state="CANCELED", env=env, source="agent",
            message=(
                f"Airflow orchestration ended as {last_status} before SLURM submit"
            ),
        )
        recovery = producer.recover_bound_presubmit(argparse.Namespace(
            request=request_path, job_record=job_record, login=args.login,
            confirm=True,
        ))
        raise BridgeError(
            f"Airflow SLURM orchestration terminated as {last_status} before "
            f"native submit; safe open boundary restored at {recovery}"
        )
    if backend_ref is None:
        receipt = _json(receipt_path, "Airflow delegation receipt")
        backend_ref = str(receipt.get("compute_backend_ref", ""))
        if not backend_ref:
            raise BridgeError("complete Airflow receipt lacks the SLURM backend reference")
        _mark_job(job_id=job_id, state="RUNNING", env=env, backend_ref=backend_ref)
    if args.name == "dataset_rebuild":
        local_dataset = pathlib.Path(state["config"]["dataset_root"])
        remote_dataset = _workspace_mapping(local_dataset, local_workspace, remote_workspace)
        _fetch_dataset_tree(login=args.login, remote=remote_dataset, local=local_dataset)
    if args.name == "dataset_materialize":
        _synchronize_dataset_materialization(
            login=args.login, results=results, local_workspace=local_workspace,
            remote_workspace=remote_workspace, recovered=False,
        )
    if args.name == "gap_analysis":
        _synchronize_gap_history(
            login=args.login, results=results, local_workspace=local_workspace,
            remote_workspace=remote_workspace, label=args.label,
            request=request, recovered=False,
        )
    if args.name == "mining_postprocess":
        _synchronize_mining_candidates(
            login=args.login, results=results, local_workspace=local_workspace,
            remote_workspace=remote_workspace, label=args.label,
            request=request, recovered=False,
        )
    if args.name == "history_select":
        _synchronize_history_host_status(
            login=args.login, results=results, local_workspace=local_workspace,
            remote_workspace=remote_workspace, label=args.label,
            request=request, recovered=False,
        )
        _synchronize_mining_history(
            login=args.login, results=results, local_workspace=local_workspace,
            remote_workspace=remote_workspace, label=args.label,
            request=request, recovered=False,
        )
    if args.name in {"visualize_prepare", "visualize_finish"}:
        _synchronize_visualization_outputs(
            login=args.login, results=results, local_workspace=local_workspace,
            remote_workspace=remote_workspace, label=args.label, state=state,
            name=args.name, recovered=False,
        )
    if args.name == "train":
        _synchronize_train_outputs(
            login=args.login, results=results,
            local_workspace=local_workspace,
            remote_workspace=remote_workspace, label=args.label,
            request=request,
        )
    if args.name == "publish_checkpoint":
        _synchronize_publish_checkpoint_outputs(
            login=args.login, results=results,
            local_workspace=local_workspace,
            remote_workspace=remote_workspace, label=args.label,
            request=request, recovered=False,
        )
    _mark_job(
        job_id=job_id, state="COMPLETE", env=env,
        message="SLURM and Airflow both completed; declared outputs synchronized",
    )
    status_path, code = producer.finalize(argparse.Namespace(
        request=request_path, job_record=job_record, native_exit_code=0,
    ))
    if code != 0:
        raise BridgeError(f"action finalization failed: {status_path}")
    return {
        "status": "COMPLETE", "name": args.name, "label": args.label,
        "request": str(request_path), "job_record": str(job_record),
        "slurm_backend_ref": backend_ref, "airflow_backend_ref": airflow_ref,
        "platform_status": str(status_path),
    }


def classify_visualize_output_loss(args: argparse.Namespace) -> dict[str, Any]:
    """Convert one proven composite-sync loss into the bounded retry shape."""
    if args.name != "visualize_prepare" or not re.fullmatch(
        r"iter[1-9][0-9]*", args.label
    ):
        raise BridgeError(
            "classify-visualize-output-loss requires visualize_prepare and iterN"
        )
    results = args.results_dir.resolve()
    state = _json(results / "deft_state.json", "DEFT state")
    if (
        state.get("workflow") != WORKFLOW
        or state.get("config", {}).get("platform") != "slurm"
        or state["config"].get("orchestrator") != "airflow"
    ):
        raise BridgeError("visualization output-loss recovery requires Airflow over SLURM")
    stage = command_contract.expected_stage_directory(args.name, args.label, results)
    status_path = stage / "visualize_prepare.status.json"
    request_path = stage / "visualize_prepare.action.json"
    binding_path = stage / "visualize_prepare.job-binding.json"
    host_status_path, expected = _visualization_side_outputs(
        results, args.label, state, args.name
    )
    status = _json(status_path, "successful visualization preparation status")
    request = _json(request_path, "visualization preparation request")
    binding = _json(binding_path, "visualization preparation job binding")
    host_status = _json(host_status_path, "visualization preparation host status")
    if (
        status.get("status") != "ok" or status.get("exit_code") != 0
        or status.get("backend_state") != "COMPLETE"
        or status.get("backend_exit_code") != 0
        or status.get("request_sha256") != request.get("request_sha256")
        or status.get("job_binding_sha256") != binding.get("binding_sha256")
        or request.get("attempt") != 1
        or host_status.get("status") != "ok" or host_status.get("exit_code") != 0
        or host_status.get("fresh_outputs") != [str(path) for path in expected]
    ):
        raise BridgeError("visualization output-loss recovery lacks exact successful attempt-1 evidence")
    if any(path.exists() or path.is_symlink() for path in expected):
        raise BridgeError("visualization output-loss recovery is forbidden while side outputs exist")
    if any((stage / name).exists() for name in (
        "visualize_prepare.attempt-1.status.json",
        "visualize_prepare.attempt-2.action.json",
    )):
        raise BridgeError("visualization output-loss recovery or retry already exists")

    local_workspace = pathlib.Path(state["config"]["workspace"]).resolve()
    remote_workspace = _remote_path(args.remote_workspace, "--remote-workspace")
    remote_expected = [
        _workspace_mapping(path, local_workspace, remote_workspace) for path in expected
    ]
    absence = _ssh(
        args.login,
        "set -Eeuo pipefail; " + " ".join(
            f"if test -e {shlex.quote(str(path))}; then printf '%s\\n' {shlex.quote(str(path))}; fi;"
            for path in remote_expected
        ),
        operation="prove visualization side outputs were lost before synchronization",
    )
    if absence.strip():
        raise BridgeError("remote visualization side outputs still exist; use synchronization, not retry")

    weak = results / f"iter_{int(args.label[4:])}" / "embeddings" / "viz_weak"
    weak_status_path = weak / "viz_weak_embed.status.json"
    weak_status = _json(weak_status_path, "downstream weak embedding failure")
    weak_log = pathlib.Path(str(weak_status.get("log_path", "")))
    missing_input = weak / "input.parquet"
    compute_alias = pathlib.PurePosixPath("/results") / missing_input.relative_to(results)
    weak_log_text = (
        weak_log.read_text(encoding="utf-8")
        if weak_log.is_file() and not weak_log.is_symlink() else ""
    )
    if (
        weak_status.get("workflow") != WORKFLOW
        or weak_status.get("name") != "viz_weak_embed"
        or weak_status.get("status") != "error"
        or weak_status.get("backend_state") != "ERROR"
        or not weak_log.is_file() or weak_log.is_symlink()
        or str(compute_alias) not in weak_log_text
        or "FileNotFoundError" not in weak_log_text
        or (weak / "embeddings.parquet").exists()
    ):
        raise BridgeError("downstream failure is not the exact missing visualization input defect")

    state_dir = pathlib.Path(str(request.get("job_state_dir", "")))
    job_id = status.get("job_id")
    if not state_dir.is_absolute() or not isinstance(job_id, str):
        raise BridgeError("visualization preparation job identity is invalid")
    job = _json(state_dir / "jobs" / f"{job_id}.json", "visualization job record")
    if (
        job.get("id") != job_id or job.get("terminal_state") != "COMPLETE"
        or job.get("backend_ref") != status.get("backend_ref")
        or binding.get("job_id") != job_id
        or binding.get("request_sha256") != request.get("request_sha256")
    ):
        raise BridgeError("visualization preparation job is not one terminal COMPLETE action")

    recovery_dir = stage / ".tao-runtime" / "visualize_prepare-output-loss-recovery"
    evidence_path = recovery_dir / "evidence.json"
    archived_status = recovery_dir / "successful-platform-status.json"
    if evidence_path.exists() or archived_status.exists():
        raise BridgeError("visualization output-loss recovery evidence already exists")
    recovery_dir.mkdir(parents=True, exist_ok=False)
    original_text = status_path.read_text(encoding="utf-8")
    _write_text_atomic(archived_status, original_text)
    classified = dict(status)
    classified.update({
        "status": "error", "exit_code": 3,
        "artifact_error": (
            "Airflow SLURM bridge did not synchronize visualize_prepare side outputs "
            "before the next exact-tree stage"
        ),
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    })
    classified_text = json.dumps(classified, indent=2, sort_keys=True) + "\n"
    evidence = {
        "schema_version": "1", "workflow": WORKFLOW,
        "kind": "airflow_slurm_visualize_prepare_output_loss", "label": args.label,
        "request_sha256": request["request_sha256"], "job_id": job_id,
        "backend_ref": status["backend_ref"],
        "successful_status_sha256": hashlib.sha256(original_text.encode()).hexdigest(),
        "classified_status_sha256": hashlib.sha256(classified_text.encode()).hexdigest(),
        "host_status_sha256": _sha256_file(host_status_path),
        "downstream_status_sha256": _sha256_file(weak_status_path),
        "downstream_log_sha256": _sha256_file(weak_log),
        "missing_local_outputs": [str(path) for path in expected],
        "missing_remote_outputs": [str(path) for path in remote_expected],
        "classified_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    evidence["evidence_sha256"] = _canonical_sha256(evidence, "evidence_sha256")
    _atomic_json(evidence_path, evidence)
    _write_text_atomic(status_path, classified_text)
    return {
        "status": "RECOVERABLE_ERROR", "operation": "classify-visualize-output-loss",
        "evidence": str(evidence_path), "platform_status": str(status_path),
    }


def recover_publish_checkpoint_sync(args: argparse.Namespace) -> dict[str, Any]:
    """Fetch one successful legacy publisher's exact nested outputs."""
    if args.name != "publish_checkpoint" or re.fullmatch(
        r"iter[1-9][0-9]*", args.label
    ) is None:
        raise BridgeError("recover-publish-checkpoint-sync requires iterN publish_checkpoint")
    results = args.results_dir.resolve()
    state = _json(results / "deft_state.json", "DEFT state")
    if (
        state.get("workflow") != WORKFLOW
        or state.get("config", {}).get("platform") != "slurm"
        or state["config"].get("orchestrator") != "airflow"
    ):
        raise BridgeError("checkpoint publication recovery requires Airflow over SLURM")
    stage = results / f"iter_{int(args.label[4:])}" / "train"
    platform_status_path = stage / "publish_checkpoint.status.json"
    platform_status = _json(platform_status_path, "publish-checkpoint platform status")
    request_raw = platform_status.get("request_path")
    if not isinstance(request_raw, str) or not pathlib.Path(request_raw).is_absolute():
        raise BridgeError("publish-checkpoint platform status lacks its request")
    request_path = pathlib.Path(request_raw)
    allowed_requests = {
        stage / "publish_checkpoint.action.json",
        stage / "publish_checkpoint.attempt-2.action.json",
    }
    if request_path not in allowed_requests:
        raise BridgeError("publish-checkpoint platform status names a noncanonical request")
    request_path, request = producer._load_launched_request(request_path)  # noqa: SLF001
    matches = producer._matching_job_records(request_path, request)  # noqa: SLF001
    if (
        platform_status.get("workflow") != WORKFLOW
        or platform_status.get("name") != "publish_checkpoint"
        or platform_status.get("platform") != "slurm"
        or platform_status.get("backend_state") != "COMPLETE"
        or platform_status.get("status") != "ok"
        or platform_status.get("exit_code") != 0
        or platform_status.get("request_sha256") != request["request_sha256"]
        or len(matches) != 1
        or matches[0][1].get("terminal_state") != "COMPLETE"
        or matches[0][1].get("backend_ref") != platform_status.get("backend_ref")
    ):
        raise BridgeError("checkpoint publication recovery lacks exact terminal success evidence")
    local_workspace = pathlib.Path(state["config"]["workspace"]).resolve()
    remote_workspace = _remote_path(args.remote_workspace, "--remote-workspace")
    receipt = _synchronize_publish_checkpoint_outputs(
        login=args.login, results=results, local_workspace=local_workspace,
        remote_workspace=remote_workspace, label=args.label, request=request,
        recovered=True,
    )
    return {
        "status": "COMPLETE", "operation": "recover-publish-checkpoint-sync",
        "request": str(request_path), "platform_status": str(platform_status_path),
        "receipt": str(receipt),
    }


def _bounded_integer(minimum: int, maximum: int, flag: str):
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{flag} must be an integer") from exc
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"{flag} must be between {minimum} and {maximum}"
            )
        return parsed

    return parse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation", nargs="?",
        choices=(
            "run", "recover-materialize-sync", "recover-gap-history-sync",
            "recover-mining-candidates-sync", "recover-history-finalization",
            "recover-mining-history-sync", "recover-visualization-host-log",
            "recover-monitoring",
            "classify-visualize-output-loss",
            "recover-train-output-loss",
            "recover-publish-checkpoint-sync",
        ),
        default="run",
    )
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--login", required=True)
    parser.add_argument("--remote-workspace", required=True, type=pathlib.Path)
    parser.add_argument("--shared-root", required=True, type=pathlib.Path)
    parser.add_argument("--pyt-sqsh", required=True, type=pathlib.Path)
    parser.add_argument("--ds-sqsh", required=True, type=pathlib.Path)
    parser.add_argument("--backend-dataset-root", type=pathlib.Path)
    parser.add_argument("--account", required=True)
    parser.add_argument("--cpu-partition", default="cpu_short")
    parser.add_argument("--gpu-partition", default="polar3")
    parser.add_argument(
        "--cpu-time-minutes",
        type=_bounded_integer(10, 240, "--cpu-time-minutes"), default=120,
    )
    parser.add_argument(
        "--gpu-time-minutes",
        type=_bounded_integer(10, 240, "--gpu-time-minutes"), default=240,
    )
    parser.add_argument(
        "--compute-poll-interval",
        type=_bounded_integer(5, 300, "--compute-poll-interval"), default=15,
        metavar="SECONDS",
    )
    parser.add_argument(
        "--controller-poll-interval",
        type=_bounded_integer(1, 300, "--controller-poll-interval"), default=10,
        metavar="SECONDS",
    )
    parser.add_argument(
        "--deadline", type=_bounded_integer(60, 86400, "--deadline"),
        default=21600, metavar="SECONDS",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.operation == "recover-materialize-sync":
            result = recover_materialize_sync(args)
        elif args.operation == "recover-gap-history-sync":
            result = recover_gap_history_sync(args)
        elif args.operation == "recover-mining-candidates-sync":
            result = recover_mining_candidates_sync(args)
        elif args.operation == "recover-history-finalization":
            result = recover_history_finalization(args)
        elif args.operation == "recover-mining-history-sync":
            result = recover_mining_history_sync(args)
        elif args.operation == "recover-visualization-host-log":
            result = recover_visualization_host_log(args)
        elif args.operation == "recover-monitoring":
            result = recover_monitoring(args)
        elif args.operation == "classify-visualize-output-loss":
            result = classify_visualize_output_loss(args)
        elif args.operation == "recover-publish-checkpoint-sync":
            result = recover_publish_checkpoint_sync(args)
        else:
            result = run_action(args)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (BridgeError, OSError, ValueError) as exc:
        print(f"airflow_slurm_action: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
