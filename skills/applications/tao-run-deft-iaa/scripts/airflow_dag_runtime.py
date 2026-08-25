#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed executor for the packaged IAA Airflow DAG.

The controller submits only signed JSON.  This module runs in the Airflow task
process, validates that JSON against shared-storage evidence, and executes the
bound work without reconstructing a shell command.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable


WORKFLOW = "tao-run-deft-iaa"
CONTRACT = "tao-deft-iaa-action-v1"
SDG_KIND = "airflow_sdg_action"
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
SAFE_ENV = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
SHA256 = re.compile(r"[0-9a-f]{64}")
SECRET_ENV_NAMES = (
    "AIRFLOW_API_TOKEN", "AIRFLOW_PASSWORD", "BREV_API_TOKEN", "HF_TOKEN", "NGC_KEY",
)
JOB_IDENTITY_FIELDS = (
    "schema_version", "id", "platform", "image", "network_arch", "action",
    "results_dir", "storage_tier", "upload_excludes", "submitted_at",
)
RUNTIME_ENVIRONMENT = {
    "TORCH_HOME": "/cache/torch",
    "TRITON_CACHE_DIR": "/cache/triton",
    "TORCHINDUCTOR_CACHE_DIR": "/cache/torchinductor",
    "MPLCONFIGDIR": "/cache/matplotlib",
}


class RuntimeContractError(ValueError):
    """The submitted request or its local evidence is inconsistent."""


def _canonical_sha256(payload: dict[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shared_root() -> pathlib.Path:
    raw = os.environ.get("TAO_IAA_AIRFLOW_SHARED_ROOT", "")
    if not raw.startswith("/"):
        raise RuntimeContractError("TAO_IAA_AIRFLOW_SHARED_ROOT must be absolute")
    path = pathlib.Path(raw)
    if path == pathlib.Path("/") or pathlib.Path(os.path.abspath(path)) != path:
        raise RuntimeContractError("Airflow shared root must be normalized and non-root")
    if not path.is_dir() or path.is_symlink() or path.resolve(strict=False) != path:
        raise RuntimeContractError("Airflow shared root is missing or unsafe")
    return path


def _absolute(value: Any, field: str, *, must_exist: bool = False) -> pathlib.Path:
    if not isinstance(value, str) or not value.startswith("/") or "\x00" in value:
        raise RuntimeContractError(f"{field} must be an absolute path")
    path = pathlib.Path(value)
    if path == pathlib.Path("/") or pathlib.Path(os.path.abspath(path)) != path:
        raise RuntimeContractError(f"{field} must be normalized and non-root")
    try:
        path.relative_to(_shared_root())
    except ValueError as exc:
        raise RuntimeContractError(f"{field} must be under the Airflow shared root") from exc
    if path.resolve(strict=False) != path:
        raise RuntimeContractError(f"{field} must not traverse a symlink")
    if must_exist and not path.exists():
        raise RuntimeContractError(f"{field} does not exist")
    return path


def _regular_json(path: pathlib.Path, field: str) -> dict[str, Any]:
    _absolute(str(path), field, must_exist=True)
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise RuntimeContractError(f"{field} must be a non-empty regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeContractError(f"{field} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeContractError(f"{field} JSON root must be an object")
    return payload


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
    return tuple(value for name in SECRET_ENV_NAMES if (value := os.environ.get(name)))


def _reject_secret_material(payload: Any, field: str) -> None:
    secrets = _secret_values()
    if any(secret in text for text in _walk_strings(payload) for secret in secrets):
        raise RuntimeContractError(f"{field} contains a credential value")


def _redact(text: str) -> str:
    for value in _secret_values():
        text = text.replace(value, "[REDACTED]")
    return text


def _atomic_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            if not text.endswith("\n"):
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


def _validate_digest(payload: dict[str, Any], field: str, label: str) -> None:
    digest = payload.get(field)
    if (
        not isinstance(digest, str) or SHA256.fullmatch(digest) is None
        or digest != _canonical_sha256(payload, field)
    ):
        raise RuntimeContractError(f"{label} digest is missing or invalid")


def _bound_standard_conf(conf: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected = {
        "contract", "job_id", "request_sha256", "binding_sha256", "request",
        "job_identity", "resolved_mounts", "results_scope", "staging_receipt_sha256",
    }
    if not isinstance(conf, dict) or set(conf) != expected or conf.get("contract") != CONTRACT:
        raise RuntimeContractError("ordinary Airflow conf has an invalid shape or contract")
    request = conf.get("request")
    if (
        not isinstance(request, dict) or request.get("workflow") != WORKFLOW
        or request.get("platform") != "airflow"
    ):
        raise RuntimeContractError("ordinary request does not bind IAA Airflow")
    _validate_digest(request, "request_sha256", "action request")
    if conf.get("request_sha256") != request["request_sha256"]:
        raise RuntimeContractError("conf request digest differs from embedded request")
    binding_path = _absolute(request.get("job_binding_path"), "job binding", must_exist=True)
    binding = _regular_json(binding_path, "job binding")
    _validate_digest(binding, "binding_sha256", "job binding")
    if conf.get("binding_sha256") != binding["binding_sha256"]:
        raise RuntimeContractError("conf binding digest differs from shared-storage binding")
    request_path = _absolute(binding.get("request_path"), "bound request", must_exist=True)
    if _regular_json(request_path, "bound request") != request:
        raise RuntimeContractError("embedded request differs from the bound request file")
    job = conf.get("job_identity")
    if not isinstance(job, dict) or set(job) != set(JOB_IDENTITY_FIELDS):
        raise RuntimeContractError("Airflow conf job identity has an invalid shape")
    identity_digest = hashlib.sha256(
        json.dumps(job, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if identity_digest != binding.get("job_identity_sha256"):
        raise RuntimeContractError("Airflow conf job identity differs from the bound job")
    bundle = request.get("spec_bundle")
    if not isinstance(bundle, dict):
        raise RuntimeContractError("request spec_bundle must be an object")
    expected_job = {
        "id": conf.get("job_id"), "platform": "airflow",
        "image": request.get("record_image"), "network_arch": bundle.get("network_arch"),
        "action": bundle.get("action"), "results_dir": conf.get("results_scope"),
    }
    for field, value in expected_job.items():
        if job.get(field) != value:
            raise RuntimeContractError(f"job record {field} differs from signed Airflow conf")
    if binding.get("job_id") != job.get("id") or binding.get("results_scope") != conf.get("results_scope"):
        raise RuntimeContractError("job binding identity or results scope differs")
    if binding.get("staging_receipt_sha256") != conf.get("staging_receipt_sha256"):
        raise RuntimeContractError("staging receipt digest differs from job binding")
    _reject_secret_material(conf, "Airflow conf")
    return request, binding, job


def _resolved_mounts(request: dict[str, Any], rows: Any) -> list[dict[str, Any]]:
    declared = request.get("mounts")
    if not isinstance(declared, list) or not isinstance(rows, list) or len(rows) != len(declared):
        raise RuntimeContractError("resolved mounts do not match the request cardinality")
    resolved: list[dict[str, Any]] = []
    targets: set[str] = set()
    for index, (expected, row) in enumerate(zip(declared, rows, strict=True)):
        if not isinstance(expected, dict) or not isinstance(row, dict):
            raise RuntimeContractError(f"mount {index} is invalid")
        if set(row) != {"source", "target", "read_only", "declared_source_sha256"}:
            raise RuntimeContractError(f"resolved mount {index} has an invalid shape")
        source = _absolute(row.get("source"), f"resolved mount {index} source", must_exist=True)
        target = row.get("target")
        if (
            not isinstance(target, str) or not target.startswith("/")
            or pathlib.PurePosixPath(target) == pathlib.PurePosixPath("/")
            or str(pathlib.PurePosixPath(target)) != target
        ):
            raise RuntimeContractError(f"resolved mount {index} target is invalid")
        if target in targets:
            raise RuntimeContractError(f"duplicate resolved mount target: {target}")
        targets.add(target)
        if target != expected.get("target") or row.get("read_only") is not expected.get("read_only"):
            raise RuntimeContractError(f"resolved mount {index} target or mode differs")
        declared_digest = hashlib.sha256(str(expected.get("source", "")).encode()).hexdigest()
        if row.get("declared_source_sha256") != declared_digest:
            raise RuntimeContractError(f"resolved mount {index} source binding differs")
        resolved.append({"source": str(source), "target": target, "read_only": row["read_only"]})
    return resolved


def _docker_inspect(name: str) -> dict[str, Any] | None:
    completed = subprocess.run(
        ["docker", "inspect", name], capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeContractError("Docker inspect returned malformed JSON") from exc
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise RuntimeContractError("Docker inspect returned an unexpected shape")
    return payload[0]


def _standard_docker_argv(
    request: dict[str, Any], job: dict[str, Any], mounts: list[dict[str, Any]],
) -> list[str]:
    job_id = job.get("id")
    if not isinstance(job_id, str) or SAFE_NAME.fullmatch(job_id) is None:
        raise RuntimeContractError("job ID cannot form a deterministic container name")
    bundle = request.get("spec_bundle")
    if not isinstance(bundle, dict) or bundle.get("mode") != "args":
        raise RuntimeContractError("Airflow IAA actions require an argv-mode spec bundle")
    command = bundle.get("command")
    args = bundle.get("args")
    if (
        not isinstance(command, str) or not command or not isinstance(args, list)
        or any(not isinstance(item, str) or "\x00" in item for item in args)
        or bundle.get("image") != request.get("workload_image")
    ):
        raise RuntimeContractError("Airflow workload image or argv is invalid")
    gpu_ids = request.get("gpu_ids")
    if (
        not isinstance(gpu_ids, list)
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in gpu_ids)
        or len(gpu_ids) != len(set(gpu_ids))
    ):
        raise RuntimeContractError("request gpu_ids must be distinct non-negative integers")
    shape = bundle.get("compute_shape")
    if not isinstance(shape, dict) or shape.get("nodes") != 1 or shape.get("gpus") != len(gpu_ids):
        raise RuntimeContractError("request GPU IDs differ from the signed compute shape")
    argv = [
        "docker", "run", "-d", "--name", job_id,
        "--label", f"tao-job={job_id}",
        "--label", f"tao-action={request.get('name')}",
        "--label", f"tao-request-sha256={request.get('request_sha256')}",
        "--label", "tao-platform=airflow",
    ]
    if gpu_ids:
        argv += ["--gpus", '"device=' + ",".join(str(item) for item in gpu_ids) + '"',
                 "--ipc=host", "--shm-size=8g"]
    uid, gid = os.getuid(), os.getgid()
    if uid == 0:
        raise RuntimeContractError("refusing writable Airflow Docker launch as UID 0")
    argv += ["--user", f"{uid}:{gid}"]
    for group_id in os.getgroups():
        if group_id != gid:
            argv += ["--group-add", str(group_id)]
    for row in mounts:
        mount = f"type=bind,src={row['source']},dst={row['target']}"
        if row["read_only"]:
            mount += ",readonly"
        argv += ["--mount", mount]
    environment = request.get("environment")
    if not isinstance(environment, dict):
        raise RuntimeContractError("request environment must be an object")
    effective_environment = dict(environment)
    if gpu_ids:
        effective_environment.update(RUNTIME_ENVIRONMENT)
    for name, value in sorted(effective_environment.items()):
        if SAFE_ENV.fullmatch(str(name)) is None or not isinstance(value, str) or "\x00" in value:
            raise RuntimeContractError(f"request environment entry is invalid: {name}")
        argv += ["-e", f"{name}={value}"]
    forward_env = request.get("forward_env")
    if forward_env not in ([], ["HF_TOKEN"]):
        raise RuntimeContractError("Airflow action may forward only HF_TOKEN")
    for name in forward_env:
        if not os.environ.get(name):
            raise RuntimeContractError(f"approved forwarded environment variable is absent: {name}")
        argv += ["-e", name]
    argv += [str(bundle["image"]), command, *args]
    if "--gpus" in argv and argv[argv.index("--gpus") + 1] == "all":
        raise AssertionError("explicit GPU IDs were widened to all")
    return argv


def _write_container_log(name: str, log_path: pathlib.Path, exit_code: int) -> None:
    completed = subprocess.run(
        ["docker", "logs", name], capture_output=True, text=True, check=False,
    )
    text = (completed.stdout or "") + (completed.stderr or "")
    text += f"\nAIRFLOW_DOCKER_EXIT_CODE={exit_code}\n"
    _atomic_text(log_path, _redact(text))


def execute_standard(conf: Any) -> dict[str, Any]:
    request, _, job = _bound_standard_conf(conf)
    mounts = _resolved_mounts(request, conf["resolved_mounts"])
    log_path = _absolute(request.get("log_path"), "action log")
    fresh_outputs = [_absolute(value, "fresh output") for value in request.get("fresh_outputs", [])]
    if not fresh_outputs:
        raise RuntimeContractError("ordinary action must declare fresh outputs")
    name = str(job["id"])
    existing = _docker_inspect(name)
    if existing is None:
        collisions = [
            path for path in [log_path, *fresh_outputs]
            if path.exists()
        ]
        if collisions:
            raise RuntimeContractError("fresh action paths were not absent before Airflow execution")
        argv = _standard_docker_argv(request, job, mounts)
        launched = subprocess.run(argv, capture_output=True, text=True, check=False)
        if launched.returncode != 0:
            _atomic_text(log_path, _redact((launched.stdout or "") + (launched.stderr or "")))
            raise RuntimeError(f"Docker launch failed for {name}; inspect {log_path}")
        existing = _docker_inspect(name)
    labels = ((existing or {}).get("Config") or {}).get("Labels") or {}
    if (
        labels.get("tao-job") != name
        or labels.get("tao-platform") != "airflow"
        or labels.get("tao-request-sha256") != request["request_sha256"]
    ):
        raise RuntimeContractError("existing container is not owned by this exact Airflow request")
    state = (existing or {}).get("State") or {}
    if state.get("Running"):
        timeout = int(os.environ.get("TAO_IAA_AIRFLOW_ACTION_TIMEOUT_S", "86400"))
        try:
            waited = subprocess.run(
                ["docker", "wait", name], capture_output=True, text=True,
                check=False, timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Docker action timed out after {timeout}s; container retained: {name}") from exc
        if waited.returncode != 0:
            raise RuntimeError(f"Docker wait failed for {name}")
        try:
            exit_code = int(waited.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError) as exc:
            raise RuntimeError("Docker wait returned an invalid exit code") from exc
    else:
        exit_code = int(state.get("ExitCode", 1))
    _write_container_log(name, log_path, exit_code)
    if exit_code != 0:
        raise RuntimeError(f"Airflow workload exited {exit_code}; retained {name}; inspect {log_path}")
    started_ns = request.get("started_ns")
    if not isinstance(started_ns, int) or isinstance(started_ns, bool) or started_ns < 1:
        raise RuntimeContractError("request started_ns is invalid")
    for output in fresh_outputs:
        if output.is_symlink() or not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"Airflow workload output is missing or empty: {output}")
        if output.stat().st_mtime_ns < started_ns:
            raise RuntimeError(f"Airflow workload output predates the signed request: {output}")
    return {
        "kind": "action", "job_id": name, "exit_code": 0,
        "fresh_outputs": [str(path) for path in fresh_outputs], "log_path": str(log_path),
    }


def _validate_sdg_conf(conf: Any) -> dict[str, Any]:
    expected = {"contract", "kind", "job_id", "request_sha256", "binding_sha256", "request"}
    if (
        not isinstance(conf, dict) or set(conf) != expected
        or conf.get("contract") != CONTRACT or conf.get("kind") != SDG_KIND
    ):
        raise RuntimeContractError("composite SDG conf has an invalid shape")
    request = conf.get("request")
    if (
        not isinstance(request, dict) or request.get("workflow") != WORKFLOW
        or request.get("platform") not in {"airflow", "docker", "virtualenv"}
        or request.get("kind") != SDG_KIND
    ):
        raise RuntimeContractError("composite request does not bind IAA Airflow SDG")
    if request.get("platform") != "airflow" and request.get("orchestrator") != "airflow":
        raise RuntimeContractError("local compute SDG request does not bind Airflow orchestration")
    _validate_digest(request, "request_sha256", "Airflow SDG request")
    if conf.get("request_sha256") != request["request_sha256"]:
        raise RuntimeContractError("composite conf request digest differs")
    binding = _regular_json(
        _absolute(request.get("job_binding_path"), "SDG job binding", must_exist=True),
        "SDG job binding",
    )
    _validate_digest(binding, "binding_sha256", "SDG job binding")
    if conf.get("binding_sha256") != binding["binding_sha256"] or binding.get("job_id") != conf.get("job_id"):
        raise RuntimeContractError("composite conf binding identity differs")
    request_path = _absolute(binding.get("request_path"), "bound SDG request", must_exist=True)
    if _regular_json(request_path, "bound SDG request") != request:
        raise RuntimeContractError("embedded SDG request differs from bound request")
    paths = request.get("paths")
    if not isinstance(paths, dict):
        raise RuntimeContractError("SDG paths must be an object")
    for field, value in paths.items():
        _absolute(value, f"SDG paths.{field}", must_exist=field not in {"stage_dir"})
    bindings = request.get("bindings")
    if not isinstance(bindings, dict):
        raise RuntimeContractError("SDG bindings must be an object")
    state_path = pathlib.Path(paths["results_dir"]) / "deft_state.json"
    if _file_sha256(state_path) != bindings.get("state_sha256"):
        raise RuntimeContractError("SDG state changed after request preparation")
    if _file_sha256(pathlib.Path(paths["config_path"])) != bindings.get("config_sha256"):
        raise RuntimeContractError("SDG config changed after request preparation")
    resources = request.get("resources")
    if not isinstance(resources, dict):
        raise RuntimeContractError("SDG resources must be an object")
    gpu_fields = ("image_edit_gpu_ids", "vlm_gpu_ids", "llm_gpu_ids", "tao_gpu_ids")
    gpu_sets = []
    for field in gpu_fields:
        values = resources.get(field)
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise RuntimeContractError(f"SDG resources.{field} is invalid")
        gpu_sets.append(set(values))
    if request.get("generation_nodes") == 1 and any(
        gpu_sets[left] & gpu_sets[right]
        for left in range(len(gpu_sets)) for right in range(left + 1, len(gpu_sets))
    ):
        raise RuntimeContractError("single-host Airflow SDG GPU selections overlap")
    if request.get("forward_env") not in ([], ["HF_TOKEN"]):
        raise RuntimeContractError("SDG may forward only HF_TOKEN")
    for name in request.get("forward_env", []):
        if not os.environ.get(name):
            raise RuntimeContractError(f"approved SDG environment variable is absent: {name}")
    _reject_secret_material(conf, "Airflow SDG conf")
    return request


def _run_logged(argv: list[str], log: pathlib.Path, env: dict[str, str]) -> None:
    started = time.monotonic()
    completed = subprocess.run(argv, capture_output=True, text=True, check=False, env=env)
    elapsed = time.monotonic() - started
    block = (
        f"COMMAND={json.dumps(argv)}\nELAPSED_SECONDS={elapsed:.3f}\n"
        + (completed.stdout or "") + (completed.stderr or "")
        + f"EXIT_CODE={completed.returncode}\n"
    )
    with log.open("a", encoding="utf-8") as handle:
        handle.write(_redact(block))
    if completed.returncode != 0:
        raise RuntimeError(f"Airflow SDG command exited {completed.returncode}; inspect {log}")


def execute_sdg(conf: Any) -> dict[str, Any]:
    request = _validate_sdg_conf(conf)
    paths = request["paths"]
    stage = _absolute(paths["stage_dir"], "SDG stage")
    stage.mkdir(parents=True, exist_ok=True)
    runtime = _absolute(paths["runtime_root"], "SDG runtime", must_exist=True)
    config = _absolute(paths["config_path"], "SDG config", must_exist=True)
    endpoint_manifest = stage / "endpoint_manifest.json"
    endpoint_pool = stage / "endpoint_pool.json"
    log = stage / "airflow_sdg.log"
    env = dict(os.environ)
    compute_platform = request["platform"]
    env["IAA_COMPUTE_FRAME"] = compute_platform
    common = [
        "--config", str(config), "--output-root", str(stage),
        "--mined-pairs", paths["mined_pairs"], "--dataset-root", paths["dataset_root"],
        "--gaps-parquet", paths["gaps_parquet"], "--eval-list", paths["eval_list"],
        "--eval-pairs", paths["eval_pairs"], "--attribute-vocab", paths["attribute_vocab"],
    ]
    start_argv = [
        sys.executable, str(runtime / "manage_sdg_endpoints.py"), "start",
        "--config", str(config), "--run-id", request["run_id"],
        "--cache-dir", paths["cache_dir"], "--output", str(endpoint_manifest),
        "--platform", compute_platform, "--service-host", "127.0.0.1",
        "--request-sha256", request["request_sha256"],
        "--image-edit-pool", str(endpoint_pool),
        "--gpu-identity-prefix", f"localhost/{request['run_id']}",
    ]
    _run_logged(start_argv, log, env)
    _run_logged(
        [sys.executable, str(runtime / "run_sdg_stage.py"), "prepare", *common], log, env,
    )
    _run_logged(
        [
            sys.executable, str(runtime / "run_sdg_stage.py"), "execute", *common,
            "--image-edit-endpoint-pool", str(endpoint_pool),
            "--execution-platform", compute_platform,
        ],
        log,
        env,
    )
    expected = [_absolute(value, "SDG expected output", must_exist=True)
                for value in request.get("expected_outputs", [])]
    if len(expected) != 6:
        raise RuntimeContractError("SDG request must declare six canonical outputs")
    for path in expected:
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise RuntimeError(f"Airflow SDG output is missing, empty, or unsafe: {path}")
    return {
        "kind": SDG_KIND, "job_id": conf["job_id"], "exit_code": 0,
        "expected_outputs": [str(path) for path in expected], "log_path": str(log),
    }


def _capture_dispatch_failure(conf: Any, exc: Exception) -> None:
    if not isinstance(conf, dict) or not isinstance(conf.get("request"), dict):
        return
    request = conf["request"]
    try:
        if request.get("kind") == SDG_KIND:
            paths = request.get("paths")
            if not isinstance(paths, dict):
                return
            path = _absolute(paths.get("stage_dir"), "SDG failure stage") / "airflow_sdg.log"
        else:
            path = _absolute(request.get("log_path"), "action failure log")
        line = _redact(f"AIRFLOW_DISPATCH_ERROR={type(exc).__name__}: {exc}")
        if path.exists():
            if not path.is_file() or path.is_symlink():
                return
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n" + line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        else:
            _atomic_text(path, line)
    except (OSError, RuntimeContractError, ValueError):
        return


def dispatch(conf: Any) -> dict[str, Any]:
    """Dispatch exactly one typed Airflow conf."""
    try:
        if isinstance(conf, dict) and conf.get("kind") == "airflow_compute_orchestration":
            from airflow_orchestrator import execute_conf

            return execute_conf(conf)
        if isinstance(conf, dict) and conf.get("kind") == SDG_KIND:
            return execute_sdg(conf)
        return execute_standard(conf)
    except Exception as exc:
        _capture_dispatch_failure(conf, exc)
        raise
