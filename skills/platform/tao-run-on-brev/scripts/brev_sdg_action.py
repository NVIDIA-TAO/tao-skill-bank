#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Four-verb Brev owner for one immutable composite IAA SDG action.

This adapter owns only Brev transport, controller lifecycle, and identity. The
staged shared endpoint manager owns endpoint semantics; the staged shared SDG
runtime owns prepare/augment/split/label/normalize semantics.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import ipaddress
import json
import os
import pathlib
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import secrets
from typing import Any, Sequence

import yaml

try:
    from brev_transport import run_remote
except ModuleNotFoundError:  # remote controller does not use launcher transport
    def run_remote(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("brev_transport is required for launcher verbs")


WORKFLOW = "tao-run-deft-iaa"
KIND = "brev_sdg_action"
NAME = "sdg_execute"
MARKER = "TAO_BREV_SDG="
TERMINAL = {"COMPLETE", "ERROR", "CANCELED"}
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}")
SAFE_ENV = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
SHA256 = re.compile(r"[0-9a-f]{64}")
SECRET_NAME = re.compile(r"(?:TOKEN|PASSWORD|PASSWD|SECRET|API_KEY)$", re.I)
SECRET_RE = re.compile(
    r"(?i)(authorization:\s*bearer\s+)\S+|"
    r"((?:token|password|secret|api[_-]?key)\s*[=:]\s*)\S+"
)
EXPECTED_KEYS = {
    "schema_version", "workflow", "kind", "platform", "name", "action_id",
    "run_id", "iteration", "attempt", "started_at", "started_ns", "local",
    "remote", "topology", "generation_nodes", "coordinator", "workers", "resources",
    "models", "limits", "bindings", "forward_env", "timeouts", "resume",
    "request_sha256",
}
MODEL_ROLES = ("image_edit", "vlm", "llm")
MODEL_KEYS = {"id", "revision", "backend", "port", "min_vram_mib"}
LIMIT_KEYS = {
    "startup_timeout_s", "request_timeout_s", "retry_interval_s",
    "image_edit_request_timeout_s", "verification_max_attempts",
    "max_samples_per_iteration",
}
SINGLE_HOST_GPU_IDS = {
    "image_edit": [0, 1, 2, 3], "vlm": [4], "llm": [5], "tao": [6, 7],
}
SINGLE_HOST_MIN_GPU_COUNT = 8
SINGLE_HOST_MIN_VRAM_MIB = 80000


class Cancelled(RuntimeError):
    """Raised after an explicit controller cancellation request."""


def _canonical_sha256(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("request_sha256", None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: pathlib.Path) -> str:
    """Hash the staged shared Python runtime without filesystem metadata."""
    if not root.is_dir() or root.is_symlink():
        raise ValueError("runtime_root must be a non-symlink directory")
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts
    )
    if not files:
        raise ValueError("runtime_root contains no regular files")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _absolute(value: Any, name: str) -> pathlib.Path:
    if not isinstance(value, str) or not value.startswith("/") or "\x00" in value:
        raise ValueError(f"{name} must be an absolute POSIX path")
    pure = pathlib.PurePosixPath(value)
    if pure == pathlib.PurePosixPath("/") or ".." in pure.parts or str(pure) != value.rstrip("/"):
        raise ValueError(f"{name} must be normalized, non-root, and not traverse '..'")
    return pathlib.Path(value)


def _under(path: pathlib.Path, root: pathlib.Path, name: str) -> pathlib.Path:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} must be under {root}") from exc
    return path


def _safe_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise ValueError(f"{name} contains unsupported characters")
    return value


def _worker_address(value: Any, name: str, *, allow_loopback: bool = False) -> str:
    """Accept a directly routable IPv4 address or DNS hostname, never a URL."""
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 253:
        raise ValueError(f"{name} must be a credential-free host")
    if any(char in value for char in "/:@?#[]"):
        raise ValueError(f"{name} must be a host without scheme, credentials, port, or path")
    lowered = value.lower()
    if not allow_loopback and (lowered == "localhost" or lowered.endswith(".localhost")):
        raise ValueError(f"{name} must be directly reachable from the coordinator")
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        labels = value.split(".")
        if any(
            not label or len(label) > 63
            or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label)
            for label in labels
        ):
            raise ValueError(f"{name} must be a valid IPv4 address or DNS hostname")
    else:
        unsafe = address.is_unspecified or address.is_link_local or address.is_multicast
        if address.version != 4 or unsafe or (address.is_loopback and not allow_loopback):
            raise ValueError(f"{name} must be a directly reachable non-loopback IPv4 address")
    return value


def _required_capacity(payload: dict[str, Any]) -> int:
    return sum(len(worker["gpu_ids"]) for worker in payload["workers"])


def validate_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != EXPECTED_KEYS:
        raise ValueError("SDG request has missing or unexpected fields")
    fixed = {
        "schema_version": "1", "workflow": WORKFLOW, "kind": KIND,
        "platform": "brev", "name": NAME,
    }
    for key, value in fixed.items():
        if payload.get(key) != value:
            raise ValueError(f"request.{key} must be {value!r}")
    if not isinstance(payload["request_sha256"], str) or not SHA256.fullmatch(payload["request_sha256"]):
        raise ValueError("request_sha256 must be a lowercase SHA-256")
    if _canonical_sha256(payload) != payload["request_sha256"]:
        raise ValueError("request_sha256 does not match immutable content")
    _safe_id(payload["action_id"], "action_id")
    _safe_id(payload["run_id"], "run_id")
    if not isinstance(payload["iteration"], int) or isinstance(payload["iteration"], bool) or payload["iteration"] < 1:
        raise ValueError("iteration must be a positive integer")
    if payload["attempt"] not in (1, 2):
        raise ValueError("attempt must be 1 or 2")
    if not isinstance(payload["started_at"], str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", payload["started_at"]
    ):
        raise ValueError("started_at must be a UTC ISO-8601 timestamp ending in Z")
    if not isinstance(payload["started_ns"], int) or isinstance(payload["started_ns"], bool) or payload["started_ns"] < 1:
        raise ValueError("started_ns must be a positive integer")

    local = payload["local"]
    if not isinstance(local, dict) or set(local) != {"results_dir", "stage_dir", "expected_outputs"}:
        raise ValueError("local must contain results_dir, stage_dir, and expected_outputs")
    local_results = _absolute(local["results_dir"], "local.results_dir")
    local_stage = _absolute(local["stage_dir"], "local.stage_dir")
    if local_stage != local_results / f"iter_{payload['iteration']}" / "datagen":
        raise ValueError("local.stage_dir must be the canonical iteration datagen directory")

    remote = payload["remote"]
    remote_keys = {
        "results_dir", "stage_dir", "dataset_root", "config_path",
        "config_sha256", "runtime_root", "runtime_sha256", "cache_dir",
        "controller_python",
        "mined_pairs", "eval_list", "attribute_vocab", "smoke_image",
        "gaps_parquet", "eval_pairs",
        "endpoint_pool_path", "expected_outputs",
    }
    if not isinstance(remote, dict) or set(remote) != remote_keys:
        raise ValueError("remote has missing or unexpected staged fields")
    results = _absolute(remote["results_dir"], "remote.results_dir")
    output = _absolute(remote["stage_dir"], "remote.stage_dir")
    if output != results / f"iter_{payload['iteration']}" / "datagen":
        raise ValueError("remote.stage_dir must be the canonical iteration datagen directory")
    dataset = _absolute(remote["dataset_root"], "remote.dataset_root")
    config = _absolute(remote["config_path"], "remote.config_path")
    if config != results / "config" / "sdg_config.yaml":
        raise ValueError("remote.config_path must be the run-owned sdg_config.yaml")
    runtime = _absolute(remote["runtime_root"], "remote.runtime_root")
    controller_python = _absolute(
        remote["controller_python"], "remote.controller_python"
    )
    if controller_python.name != "python" or controller_python.parent.name != "bin":
        raise ValueError("remote.controller_python must be an explicit virtualenv bin/python")
    if not controller_python.is_relative_to(runtime.parents[2]):
        raise ValueError("remote.controller_python must be scoped below remote_root")
    _absolute(remote["cache_dir"], "remote.cache_dir")
    _under(_absolute(remote["mined_pairs"], "remote.mined_pairs"), results, "remote.mined_pairs")
    _under(_absolute(remote["eval_list"], "remote.eval_list"), results, "remote.eval_list")
    _under(_absolute(remote["gaps_parquet"], "remote.gaps_parquet"), results, "remote.gaps_parquet")
    _under(_absolute(remote["eval_pairs"], "remote.eval_pairs"), results, "remote.eval_pairs")
    _under(_absolute(remote["attribute_vocab"], "remote.attribute_vocab"), dataset, "remote.attribute_vocab")
    _under(_absolute(remote["smoke_image"], "remote.smoke_image"), dataset, "remote.smoke_image")
    if _absolute(remote["endpoint_pool_path"], "remote.endpoint_pool_path") != output / "endpoint_pool.json":
        raise ValueError("remote.endpoint_pool_path must be the canonical stage endpoint_pool.json")
    for key in ("config_sha256", "runtime_sha256"):
        if not isinstance(remote[key], str) or not SHA256.fullmatch(remote[key]):
            raise ValueError(f"remote.{key} must be a lowercase SHA-256")
    if runtime in {results, dataset} or results in runtime.parents or dataset in runtime.parents:
        raise ValueError("runtime_root must be a separate staged read-only tree")

    generation_nodes = payload["generation_nodes"]
    if not isinstance(generation_nodes, int) or isinstance(generation_nodes, bool) or not 1 <= generation_nodes <= 32:
        raise ValueError("generation_nodes must be in [1, 32]")
    topology = payload["topology"]
    if topology not in {"single_host", "multi_host"}:
        raise ValueError("topology must be single_host or multi_host")
    if (generation_nodes == 1) != (topology == "single_host"):
        raise ValueError("generation_nodes=1 requires the canonical single_host topology")
    coordinator = payload["coordinator"]
    coordinator_keys = {"instance", "gpu_ids"}
    if topology == "single_host":
        coordinator_keys |= {"gpu_count", "gpu_memory_mib"}
    if not isinstance(coordinator, dict) or set(coordinator) != coordinator_keys:
        raise ValueError("coordinator has missing or unexpected topology fields")
    _safe_id(coordinator["instance"], "coordinator.instance")
    expected_coordinator_gpus = (
        {"vlm": [4], "llm": [5], "tao": [6, 7]}
        if topology == "single_host" else {"vlm": [0], "llm": [1]}
    )
    if coordinator["gpu_ids"] != expected_coordinator_gpus:
        raise ValueError("coordinator.gpu_ids do not match the signed topology")
    if topology == "single_host":
        memory = coordinator["gpu_memory_mib"]
        if coordinator["gpu_count"] != SINGLE_HOST_MIN_GPU_COUNT:
            raise ValueError("single-host Brev requires exactly eight visible GPUs")
        if (
            not isinstance(memory, list) or len(memory) != SINGLE_HOST_MIN_GPU_COUNT
            or any(not isinstance(value, int) or isinstance(value, bool)
                   or value < SINGLE_HOST_MIN_VRAM_MIB for value in memory)
        ):
            raise ValueError("single-host Brev requires eight GPUs with at least 80000 MiB each")
    workers = payload["workers"]
    if not isinstance(workers, list) or len(workers) != generation_nodes:
        raise ValueError("workers must contain exactly generation_nodes entries")
    instances = {coordinator["instance"]}
    urls: set[str] = set()
    worker_gpu_ids = SINGLE_HOST_GPU_IDS["image_edit"] if topology == "single_host" else list(range(8))
    for index, worker in enumerate(workers):
        if not isinstance(worker, dict) or set(worker) != {"id", "instance", "address", "gpu_ids", "ports"}:
            raise ValueError(f"workers[{index}] has missing or unexpected fields")
        _safe_id(worker["id"], f"workers[{index}].id")
        instance = _safe_id(worker["instance"], f"workers[{index}].instance")
        if topology == "single_host":
            if index != 0 or instance != coordinator["instance"]:
                raise ValueError("single-host worker must use the signed coordinator instance")
        elif instance in instances:
            raise ValueError("coordinator and worker instance identities must be distinct")
        instances.add(instance)
        address = _worker_address(
            worker["address"], f"workers[{index}].address",
            allow_loopback=topology == "single_host",
        )
        if topology == "single_host" and address != "127.0.0.1":
            raise ValueError("single-host worker address must be exactly 127.0.0.1")
        if worker["gpu_ids"] != worker_gpu_ids:
            raise ValueError(f"workers[{index}].gpu_ids do not match the signed topology")
        ports = worker["ports"]
        if not isinstance(ports, list) or len(ports) != len(worker_gpu_ids) or len(set(ports)) != len(worker_gpu_ids) or any(
            not isinstance(port, int) or isinstance(port, bool) or not 1024 <= port <= 65535
            for port in ports
        ):
            raise ValueError(f"workers[{index}].ports must match the explicit image-edit GPU count")
        for port in ports:
            url = f"http://{worker['address']}:{port}/v1"
            if url in urls:
                raise ValueError("worker endpoint URLs must be distinct")
            urls.add(url)
    models = payload["models"]
    if not isinstance(models, dict) or set(models) != set(MODEL_ROLES):
        raise ValueError("models must contain image_edit, vlm, and llm")
    model_ports: set[int] = set()
    for role in MODEL_ROLES:
        model = models[role]
        if not isinstance(model, dict) or set(model) != MODEL_KEYS:
            raise ValueError(f"models.{role} has missing or unexpected fields")
        if not isinstance(model["id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]{0,255}", model["id"]):
            raise ValueError(f"models.{role}.id is invalid")
        if not isinstance(model["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            raise ValueError(f"models.{role}.revision must be a pinned commit")
        expected_backend = "vllm-omni" if role == "image_edit" else "vllm"
        if model["backend"] != expected_backend:
            raise ValueError(f"models.{role}.backend must be {expected_backend}")
        port = model["port"]
        if not isinstance(port, int) or isinstance(port, bool) or not 1024 <= port <= 65535 or port in model_ports:
            raise ValueError(f"models.{role}.port must be unique and in [1024, 65535]")
        model_ports.add(port)
        if not isinstance(model["min_vram_mib"], int) or isinstance(model["min_vram_mib"], bool) or model["min_vram_mib"] < 1:
            raise ValueError(f"models.{role}.min_vram_mib must be positive")
    worker_slots = len(worker_gpu_ids)
    if models["image_edit"]["port"] + worker_slots - 1 > 65535:
        raise ValueError("models.image_edit.port must leave room for every worker port")
    expected_worker_ports = list(range(models["image_edit"]["port"], models["image_edit"]["port"] + worker_slots))
    if any(worker["ports"] != expected_worker_ports for worker in workers):
        raise ValueError("worker ports must be the approved contiguous image-edit range")
    limits = payload["limits"]
    if not isinstance(limits, dict) or set(limits) != LIMIT_KEYS:
        raise ValueError("limits has missing or unexpected fields")
    for key in LIMIT_KEYS:
        value = limits[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"limits.{key} must be a positive integer")
    if not 1 <= limits["verification_max_attempts"] <= 5:
        raise ValueError("limits.verification_max_attempts must be in [1, 5]")
    bindings = payload["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != {"state_sha256", "inventory_sha256"}:
        raise ValueError("bindings must contain state_sha256 and inventory_sha256")
    for key, digest in bindings.items():
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise ValueError(f"bindings.{key} must be a lowercase SHA-256")
    resources = payload["resources"]
    expected_resources = {
        "generation_nodes": generation_nodes, "gpus_per_worker": worker_slots,
        "capacity_per_worker": worker_slots, "coordinator_vlm_gpus": 1,
        "coordinator_llm_gpus": 1,
        **({"tao_gpus": 2, "host_gpu_count": 8, "host_min_vram_mib": 80000}
           if topology == "single_host" else {}),
    }
    if resources != expected_resources:
        raise ValueError("resources do not match the signed Brev topology")
    forward = payload["forward_env"]
    if not isinstance(forward, list) or len(set(forward)) != len(forward):
        raise ValueError("forward_env must be a unique variable-name list")
    for name in forward:
        if not isinstance(name, str) or not SAFE_ENV.fullmatch(name) or not SECRET_NAME.search(name):
            raise ValueError("forward_env may contain credential variable names only")
        if name in {"IMAGE_EDIT_API_KEY", "VLLM_API_KEY"}:
            raise ValueError("image-edit API authentication is generated ephemerally by the Brev adapter")
    timeouts = payload["timeouts"]
    if not isinstance(timeouts, dict) or set(timeouts) != {"controller_s", "worker_s", "readiness_s", "cancel_s"}:
        raise ValueError("timeouts must contain controller_s, worker_s, readiness_s, and cancel_s")
    if not isinstance(timeouts["controller_s"], int) or not 300 <= timeouts["controller_s"] <= 172800:
        raise ValueError("timeouts.controller_s must be in [300, 172800]")
    if not isinstance(timeouts["cancel_s"], int) or not 5 <= timeouts["cancel_s"] <= 120:
        raise ValueError("timeouts.cancel_s must be in [5, 120]")
    for key in ("worker_s", "readiness_s"):
        if not isinstance(timeouts[key], int) or isinstance(timeouts[key], bool) or not 30 <= timeouts[key] <= 3600:
            raise ValueError(f"timeouts.{key} must be in [30, 3600]")
    if payload["resume"] != {"max_controller_attempts": 2}:
        raise ValueError("resume.max_controller_attempts must be exactly 2")
    remote_outputs = [
        str(output / "dataset" / "sdg_manifest.json"),
        str(output / "dataset" / "sdg_pairs.json"),
        str(output / "dataset" / "sdg_image_list.txt"),
        str(output / "sdg_execution_manifest.json"),
    ]
    local_outputs = [
        str(local_stage / "dataset" / "sdg_manifest.json"),
        str(local_stage / "dataset" / "sdg_pairs.json"),
        str(local_stage / "dataset" / "sdg_image_list.txt"),
        str(local_stage / "sdg_execution_manifest.json"),
    ]
    if remote["expected_outputs"] != remote_outputs or local["expected_outputs"] != local_outputs:
        raise ValueError("local and remote expected_outputs must be the canonical four in order")
    return payload


def _load_json_file(path: pathlib.Path, name: str) -> tuple[dict[str, Any], str]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be an existing absolute non-symlink regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload, _file_sha256(path)


def _inventory_sha256(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("inventory_sha256", None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_inventory(payload: Any) -> dict[str, Any]:
    keys = {
        "schema_version", "platform", "status", "topology", "coordinator", "workers",
        "inventory_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != keys:
        raise ValueError("Brev inventory has missing or unexpected fields")
    if payload["schema_version"] != "1" or payload["platform"] != "brev" or payload["status"] != "resolved":
        raise ValueError("Brev inventory must be schema_version=1, platform=brev, status=resolved")
    topology = payload["topology"]
    if topology not in {"single_host", "multi_host"}:
        raise ValueError("Brev inventory topology must be single_host or multi_host")
    digest = payload["inventory_sha256"]
    if not isinstance(digest, str) or not SHA256.fullmatch(digest) or digest != _inventory_sha256(payload):
        raise ValueError("inventory_sha256 does not match canonical inventory content")
    coordinator = payload["coordinator"]
    coordinator_keys = {"instance"}
    if topology == "single_host":
        coordinator_keys |= {"gpu_count", "gpu_memory_mib"}
    if not isinstance(coordinator, dict) or set(coordinator) != coordinator_keys:
        raise ValueError("inventory.coordinator has missing or unexpected topology fields")
    coordinator_instance = _safe_id(coordinator["instance"], "inventory.coordinator.instance")
    if topology == "single_host":
        memory = coordinator["gpu_memory_mib"]
        if coordinator["gpu_count"] != SINGLE_HOST_MIN_GPU_COUNT:
            raise ValueError("single-host inventory requires exactly eight visible GPUs")
        if (
            not isinstance(memory, list) or len(memory) != SINGLE_HOST_MIN_GPU_COUNT
            or any(not isinstance(value, int) or isinstance(value, bool)
                   or value < SINGLE_HOST_MIN_VRAM_MIB for value in memory)
        ):
            raise ValueError("single-host inventory requires eight GPUs with at least 80000 MiB each")
    workers = payload["workers"]
    if not isinstance(workers, list) or not 1 <= len(workers) <= 32:
        raise ValueError("inventory.workers must contain 1 to 32 ordered workers")
    if topology == "single_host" and len(workers) != 1:
        raise ValueError("single-host inventory must contain exactly one local worker")
    ids: set[str] = set()
    instances = {coordinator_instance}
    addresses: set[str] = set()
    for index, worker in enumerate(workers):
        if not isinstance(worker, dict) or set(worker) != {"id", "instance", "address"}:
            raise ValueError(f"inventory.workers[{index}] must contain id, instance, and address")
        worker_id = _safe_id(worker["id"], f"inventory.workers[{index}].id")
        instance = _safe_id(worker["instance"], f"inventory.workers[{index}].instance")
        address = _worker_address(
            worker["address"], f"inventory.workers[{index}].address",
            allow_loopback=topology == "single_host",
        )
        if worker_id in ids:
            raise ValueError("inventory worker ids must be distinct")
        if topology == "single_host":
            if index != 0 or instance != coordinator_instance or address != "127.0.0.1":
                raise ValueError("single-host inventory worker must be the coordinator at 127.0.0.1")
        elif instance in instances:
            raise ValueError("inventory coordinator and worker instances must be distinct")
        if address in addresses:
            raise ValueError("inventory worker addresses must be distinct")
        ids.add(worker_id)
        instances.add(instance)
        addresses.add(address)
    return payload


def _remote_mirror(
    local: pathlib.Path, workspace: pathlib.Path, remote_root: pathlib.Path, name: str
) -> pathlib.Path:
    try:
        relative = local.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"{name} must be under state.config.workspace") from exc
    return remote_root / relative


def _configured_smoke_image(mined_pairs: pathlib.Path, dataset_root: pathlib.Path) -> pathlib.Path:
    pairs = json.loads(mined_pairs.read_text(encoding="utf-8"))
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("committed mined_pairs must contain at least one sample")
    first = pairs[0]
    if not isinstance(first, dict) or not isinstance(first.get("image_path"), str):
        raise ValueError("first committed mined pair lacks image_path")
    source = (dataset_root / first["image_path"]).resolve()
    try:
        source.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError("committed mined pair image_path escapes dataset root") from exc
    if not source.is_file() or source.is_symlink() or source.stat().st_size == 0:
        unique_name = pathlib.Path(str(first.get("unique_name", ""))).name
        fallback = dataset_root / "images" / unique_name
        if not unique_name or not fallback.is_file() or fallback.is_symlink() or fallback.stat().st_size == 0:
            raise ValueError("first committed mined pair has no readable dataset source image")
        source = fallback.resolve()
    return source


def _prepared_request(args: argparse.Namespace) -> dict[str, Any]:
    state, state_sha256 = _load_json_file(args.state, "--state")
    inventory, _ = _load_json_file(args.inventory, "--inventory")
    inventory = validate_inventory(inventory)
    if state.get("schema_version") != "3" or state.get("workflow") != WORKFLOW:
        raise ValueError("--state must be an initialized schema_version=3 IAA DEFT state")
    config = state.get("config")
    if not isinstance(config, dict) or config.get("platform") != "brev":
        raise ValueError("initialized state platform must be brev")
    results = _absolute(state.get("results_dir"), "state.results_dir").resolve()
    if args.state.parent.resolve() != results:
        raise ValueError("--state must be the canonical results_dir/deft_state.json")
    if not isinstance(args.iteration, int) or isinstance(args.iteration, bool) or args.iteration < 1:
        raise ValueError("--iteration must be positive")
    if args.iteration > state.get("max_iterations", 0):
        raise ValueError("--iteration exceeds the initialized max_iterations")
    if state.get("current_iteration") != args.iteration:
        raise ValueError("--iteration must equal the initialized current_iteration")
    label = f"iter{args.iteration}"
    phase = state.get("iterations", {}).get(label)
    if not isinstance(phase, dict):
        raise ValueError(f"state has no initialized {label} phase")
    completed = phase.get("stage_completed")
    if completed != "history_select":
        if completed in {"sdg", "visualize", "train", "evaluate", "gap_analysis"}:
            raise ValueError(f"{label}/sdg is already committed")
        raise ValueError(f"{label} must have committed history_select before SDG preparation")

    workspace = _absolute(config.get("workspace"), "state.config.workspace").resolve()
    dataset = _absolute(config.get("dataset_root"), "state.config.dataset_root").resolve()
    _remote_mirror(results, workspace, pathlib.Path("/remote"), "state.results_dir")
    _remote_mirror(dataset, workspace, pathlib.Path("/remote"), "state.config.dataset_root")
    remote_root = _absolute(str(args.remote_root), "--remote-root")
    controller_python = _absolute(
        str(args.remote_controller_python), "--remote-controller-python"
    )
    if controller_python.name != "python" or controller_python.parent.name != "bin":
        raise ValueError("--remote-controller-python must name an explicit virtualenv bin/python")
    if not controller_python.is_relative_to(remote_root):
        raise ValueError("--remote-controller-python must be scoped below --remote-root")
    remote_cache = _absolute(str(args.remote_cache), "--remote-cache")
    remote_results = _remote_mirror(results, workspace, remote_root, "state.results_dir")
    remote_dataset = _remote_mirror(dataset, workspace, remote_root, "state.config.dataset_root")

    config_path = _absolute(config.get("sdg_config"), "state.config.sdg_config").resolve()
    if config_path != results / "config" / "sdg_config.yaml" or not config_path.is_file() or config_path.is_symlink():
        raise ValueError("state.config.sdg_config must be the canonical existing run config")
    config_sha256 = _file_sha256(config_path)
    if config.get("sdg_config_sha256") != config_sha256:
        raise ValueError("state.config.sdg_config_sha256 does not match the run config")
    sdg_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(sdg_config, dict):
        raise ValueError("sdg_config root must be an object")
    endpoints = sdg_config.get("endpoints")
    generation = sdg_config.get("generation")
    configured_models = sdg_config.get("models")
    if not all(isinstance(item, dict) for item in (endpoints, generation, configured_models)):
        raise ValueError("sdg_config lacks endpoints, generation, or models")
    if endpoints.get("ownership") != "managed" or endpoints.get("reuse_requested") is not False:
        raise ValueError("Brev request preparation requires managed, non-reused endpoints")
    if not isinstance(config.get("requires_hf_token"), bool):
        raise ValueError("state.config.requires_hf_token must be boolean")
    generation_nodes = generation.get("generation_nodes")
    if generation_nodes != len(inventory["workers"]):
        raise ValueError("inventory must contain exactly the approved generation_nodes workers")
    topology = "single_host" if generation_nodes == 1 else "multi_host"
    if inventory["topology"] != topology:
        raise ValueError("inventory topology does not match the approved generation_nodes")
    expected_sdg_gpu_ids = (
        {role: SINGLE_HOST_GPU_IDS[role] for role in MODEL_ROLES}
        if topology == "single_host"
        else {"image_edit": list(range(8)), "vlm": [0], "llm": [1]}
    )
    slots_per_worker = len(expected_sdg_gpu_ids["image_edit"])
    if generation.get("gpus_per_generation_node") != slots_per_worker:
        raise ValueError("Brev generation GPU count does not match the selected topology")
    if endpoints.get("gpu_ids") != expected_sdg_gpu_ids:
        raise ValueError("Brev managed endpoint GPU roles do not match the selected topology")
    if topology == "single_host" and config.get("gpu_ids") != SINGLE_HOST_GPU_IDS["tao"]:
        raise ValueError("single-host Brev requires TAO gpu_ids=[6, 7]")
    approved_sdg = config.get("sdg")
    if not isinstance(approved_sdg, dict):
        raise ValueError("initialized state lacks approved SDG configuration")
    if (
        approved_sdg.get("endpoint_mode") != "managed"
        or approved_sdg.get("reuse_requested") is not False
        or approved_sdg.get("generation_nodes") != generation_nodes
        or approved_sdg.get("gpus_per_generation_node") != slots_per_worker
        or approved_sdg.get("gpu_ids") != endpoints.get("gpu_ids")
        or approved_sdg.get("models") != configured_models
    ):
        raise ValueError("state.config.sdg disagrees with the immutable sdg_config")

    models: dict[str, dict[str, Any]] = {}
    for role in MODEL_ROLES:
        source = configured_models.get(role)
        if not isinstance(source, dict):
            raise ValueError(f"sdg_config.models.{role} must be an object")
        models[role] = {key: source.get(key) for key in MODEL_KEYS}
    base_port = models["image_edit"]["port"]
    if not isinstance(base_port, int) or isinstance(base_port, bool) or base_port + slots_per_worker - 1 > 65535:
        raise ValueError("approved image-edit port must leave room for every worker service")
    worker_ports = list(range(base_port, base_port + slots_per_worker))

    limits = {
        "startup_timeout_s": endpoints.get("startup_timeout_s"),
        "request_timeout_s": endpoints.get("request_timeout_s"),
        "retry_interval_s": endpoints.get("retry_interval_s"),
        "image_edit_request_timeout_s": generation.get("image_edit_request_timeout_s"),
        "verification_max_attempts": generation.get("verification_max_attempts"),
        "max_samples_per_iteration": generation.get("max_samples_per_iteration"),
    }
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in limits.values()):
        raise ValueError("sdg_config contains invalid deployment or generation limits")
    startup = limits["startup_timeout_s"]
    if not 30 <= startup <= 3600:
        raise ValueError("sdg_config endpoints.startup_timeout_s must be in [30, 3600] for Brev")

    mined_pairs = _absolute(phase.get("mined_pairs"), f"state.iterations.{label}.mined_pairs").resolve()
    if mined_pairs != results / f"iter_{args.iteration}" / "mining" / "mined_pairs.json":
        raise ValueError("state mined_pairs path is not canonical")
    if not mined_pairs.is_file() or mined_pairs.is_symlink():
        raise ValueError("committed mined_pairs is unavailable")
    eval_list = results / "iaa_splits" / "eval_list.txt"
    eval_pairs = results / "iaa_splits" / "eval_pairs.json"
    gaps_parquet = results / f"iter_{args.iteration}" / "gaps" / "kpi_gaps.parquet"
    attribute_vocab = dataset / "attribute_vocab.json"
    for path, name in (
        (eval_list, "eval_list"), (eval_pairs, "eval_pairs"),
        (gaps_parquet, "gaps_parquet"), (attribute_vocab, "attribute_vocab"),
    ):
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise ValueError(f"canonical {name} is unavailable")
    smoke_image = _configured_smoke_image(mined_pairs, dataset)

    runtime_source = pathlib.Path(__file__).resolve().parents[3] / "applications" / "tao-run-deft-iaa" / "scripts"
    runtime_sha256 = _tree_sha256(runtime_source)
    remote_runtime = remote_root / ".tao" / "runtime" / f"iaa-deft-{runtime_sha256[:16]}"
    local_stage = results / f"iter_{args.iteration}" / "datagen"
    remote_stage = remote_results / f"iter_{args.iteration}" / "datagen"
    output_names = (
        "dataset/sdg_manifest.json", "dataset/sdg_pairs.json",
        "dataset/sdg_image_list.txt", "sdg_execution_manifest.json",
    )
    local_outputs = [str(local_stage / name) for name in output_names]
    remote_outputs = [str(remote_stage / name) for name in output_names]
    bindings = {
        "state_sha256": state_sha256,
        "inventory_sha256": inventory["inventory_sha256"],
    }
    identity_seed = {
        **bindings, "iteration": args.iteration, "remote_root": str(remote_root),
        "remote_cache": str(remote_cache),
    }
    identity_digest = hashlib.sha256(
        json.dumps(identity_seed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    try:
        state_started = dt.datetime.fromisoformat(str(state.get("started_at", "")).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("state.started_at must be a timezone-aware ISO timestamp") from exc
    if state_started.tzinfo is None:
        raise ValueError("state.started_at must be timezone-aware")
    started_utc = state_started.astimezone(dt.timezone.utc)
    started_at = started_utc.isoformat(timespec="microseconds").replace("+00:00", "Z")
    epoch_delta = started_utc - dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    started_ns = (
        (epoch_delta.days * 86400 + epoch_delta.seconds) * 1_000_000_000
        + epoch_delta.microseconds * 1000 + args.iteration
    )
    payload = {
        "schema_version": "1", "workflow": WORKFLOW, "kind": KIND,
        "platform": "brev", "name": NAME,
        "action_id": f"deft-iaa-sdg-{identity_digest[:16]}",
        "run_id": _safe_id(results.name, "run_id"), "iteration": args.iteration,
        "attempt": 1, "started_at": started_at, "started_ns": started_ns,
        "local": {
            "results_dir": str(results), "stage_dir": str(local_stage),
            "expected_outputs": local_outputs,
        },
        "remote": {
            "results_dir": str(remote_results), "stage_dir": str(remote_stage),
            "dataset_root": str(remote_dataset),
            "config_path": str(remote_results / "config" / "sdg_config.yaml"),
            "config_sha256": config_sha256, "runtime_root": str(remote_runtime),
            "runtime_sha256": runtime_sha256, "cache_dir": str(remote_cache),
            "controller_python": str(controller_python),
            "mined_pairs": str(_remote_mirror(mined_pairs, workspace, remote_root, "mined_pairs")),
            "eval_list": str(_remote_mirror(eval_list, workspace, remote_root, "eval_list")),
            "gaps_parquet": str(_remote_mirror(gaps_parquet, workspace, remote_root, "gaps_parquet")),
            "eval_pairs": str(_remote_mirror(eval_pairs, workspace, remote_root, "eval_pairs")),
            "attribute_vocab": str(_remote_mirror(attribute_vocab, workspace, remote_root, "attribute_vocab")),
            "smoke_image": str(_remote_mirror(smoke_image, workspace, remote_root, "smoke_image")),
            "endpoint_pool_path": str(remote_stage / "endpoint_pool.json"),
            "expected_outputs": remote_outputs,
        },
        "topology": topology, "generation_nodes": generation_nodes,
        "coordinator": {
            "instance": inventory["coordinator"]["instance"],
            "gpu_ids": (
                {"vlm": [4], "llm": [5], "tao": [6, 7]}
                if topology == "single_host" else {"vlm": [0], "llm": [1]}
            ),
            **({
                "gpu_count": inventory["coordinator"]["gpu_count"],
                "gpu_memory_mib": inventory["coordinator"]["gpu_memory_mib"],
            } if topology == "single_host" else {}),
        },
        "workers": [
            {**worker, "gpu_ids": expected_sdg_gpu_ids["image_edit"], "ports": worker_ports}
            for worker in inventory["workers"]
        ],
        "resources": {
            "generation_nodes": generation_nodes, "gpus_per_worker": slots_per_worker,
            "capacity_per_worker": slots_per_worker, "coordinator_vlm_gpus": 1,
            "coordinator_llm_gpus": 1,
            **({"tao_gpus": 2, "host_gpu_count": 8, "host_min_vram_mib": 80000}
               if topology == "single_host" else {}),
        },
        "models": models, "limits": limits, "bindings": bindings,
        "forward_env": ["HF_TOKEN"] if config.get("requires_hf_token") is True else [],
        "timeouts": {
            "controller_s": min(172800, max(300, startup + 4 * limits["image_edit_request_timeout_s"])),
            "worker_s": startup, "readiness_s": startup, "cancel_s": 30,
        },
        "resume": {"max_controller_attempts": 2},
    }
    payload["request_sha256"] = _canonical_sha256(payload)
    return validate_request(payload)


def prepare_request(args: argparse.Namespace) -> int:
    payload = _prepared_request(args)
    output = _absolute(str(args.output), "--output")
    if not output.parent.is_dir() or output.parent.is_symlink() or output.parent.resolve() != output.parent:
        raise ValueError("--output parent must be an existing non-symlink directory")
    if output.exists():
        existing, _ = _load_json_file(output, "--output")
        if existing != payload:
            raise ValueError("refusing to replace a different existing prepared request")
        disposition = "reused"
    else:
        _atomic_json(output, payload)
        disposition = "created"
    print(json.dumps({
        "status": disposition, "output": str(output),
        "action_id": payload["action_id"], "request_sha256": payload["request_sha256"],
        "generation_nodes": payload["generation_nodes"],
    }, sort_keys=True))
    return 0


def load_request(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError("--request must be an existing absolute non-symlink regular file")
    return validate_request(json.loads(path.read_text(encoding="utf-8")))


def validate_job_record(path: pathlib.Path, request: dict[str, Any], job_id: str) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError("--job-record must be an existing absolute non-symlink regular file")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("job record must be an object")
    expected = {"id": job_id, "platform": "brev", "action": request["action_id"]}
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"job record {key} does not match the composite request")
    return record


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _read_json(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"state path is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"state file is not an object: {path}")
    return value


def _state_paths(request: dict[str, Any]) -> dict[str, pathlib.Path]:
    root = (
        pathlib.Path(request["remote"]["stage_dir"])
        / "brev-controller"
        / request["action_id"]
    )
    return {
        "root": root, "request": root / "request.json",
        "controller": root / "brev_sdg_action.py", "status": root / "status.json",
        "progress": root / "progress.json", "pid": root / "controller.pid",
        "cancel": root / "cancel.request", "log": root / "controller.log",
        "endpoint_plan": root / "endpoint_plan.json",
        "endpoint_manifest": root / "endpoint_manifest.json",
        "endpoint_status": root / "endpoint_status.json",
        "runtime_log": root / "runtime.log",
        "pool_candidate": root / "endpoint_pool.candidate.json",
        "pool_readiness": root / "endpoint_pool.readiness.json",
    }


def _identity(request: dict[str, Any], job_id: str) -> dict[str, Any]:
    return {
        "kind": KIND, "job_id": job_id, "run_id": request["run_id"],
        "action_id": request["action_id"], "request_sha256": request["request_sha256"],
        "started_at": request["started_at"], "started_ns": request["started_ns"],
    }


def _validate_identity(value: dict[str, Any], identity: dict[str, Any]) -> None:
    for key, expected in identity.items():
        if value.get(key) != expected:
            raise RuntimeError(f"state identity mismatch for {key}")


def _pid_active(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def reconcile_local(request: dict[str, Any], job_id: str) -> dict[str, Any]:
    identity = _identity(request, job_id)
    paths = _state_paths(request)
    status = _read_json(paths["status"])
    if status is not None:
        _validate_identity(status, identity)
    pid = None
    if paths["pid"].exists():
        raw = paths["pid"].read_text(encoding="utf-8").strip()
        if not raw.isdigit():
            raise RuntimeError("controller pid file is malformed")
        pid = int(raw)
    if pid is not None and _pid_active(pid):
        try:
            command = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
        except OSError as exc:
            raise RuntimeError("cannot verify active controller ownership") from exc
        if not all(item in command for item in (str(paths["request"]), job_id, "_controller")):
            raise RuntimeError("controller pid belongs to a foreign process")
        return {**identity, "state": "ACTIVE", "pid": pid, "workflow_status": status.get("status") if status else "RUNNING"}
    if status is not None and status.get("status") in TERMINAL:
        return {**identity, "state": "TERMINAL", "pid": pid, "workflow_status": status["status"], "exit_code": status.get("exit_code")}
    progress = _read_json(paths["progress"])
    if progress is not None:
        _validate_identity(progress, identity)
        return {**identity, "state": "RESUMABLE", "pid": pid, "workflow_status": "ERROR"}
    return {**identity, "state": "NONE", "pid": pid, "workflow_status": None}


def _redact(text: str, secrets: Sequence[str] = ()) -> str:
    for secret in sorted((value for value in secrets if value), key=len, reverse=True):
        text = text.replace(secret, "<REDACTED>")
    return SECRET_RE.sub(lambda match: (match.group(1) or match.group(2)) + "<REDACTED>", text)


def _remote_json(instance: str, command: str, *, environment: dict[str, str] | None = None, timeout: int = 600) -> dict[str, Any]:
    completed = run_remote(instance, command, environment=environment, timeout=timeout)
    if completed.returncode != 0:
        stderr = completed.stderr.decode(errors="replace") if isinstance(completed.stderr, bytes) else (completed.stderr or "")
        raise RuntimeError(
            "Brev composite command failed: "
            + _redact(stderr.strip(), list((environment or {}).values()))
        )
    stdout = completed.stdout.decode(errors="replace") if isinstance(completed.stdout, bytes) else (completed.stdout or "")
    markers = [line[len(MARKER):] for line in stdout.splitlines() if line.startswith(MARKER)]
    if len(markers) != 1:
        raise RuntimeError("Brev composite command returned malformed evidence")
    value = json.loads(markers[0])
    if not isinstance(value, dict):
        raise RuntimeError("Brev composite evidence is not an object")
    return value


def _stage_file(instance: str, local: pathlib.Path, remote: pathlib.Path) -> str:
    """Promote one exact interface file through the Brev SSH alias."""
    if not local.is_absolute() or not local.is_file() or local.is_symlink():
        raise ValueError(f"staged input must be an absolute non-symlink file: {local}")
    digest = _file_sha256(local)
    temporary = remote.with_name(f".{remote.name}.{digest[:16]}.tmp")
    mkdir = run_remote(instance, shlex.join(["install", "-d", "-m", "700", str(remote.parent)]))
    if mkdir.returncode != 0:
        raise RuntimeError("Brev interface directory creation failed")
    copied = subprocess.run(
        ["scp", "-q", "--", str(local), f"{instance}:{temporary}"],
        capture_output=True, text=True, check=False,
    )
    if copied.returncode != 0:
        raise RuntimeError("Brev interface file copy failed: " + _redact(copied.stderr.strip()))
    command = (
        "set -eu; "
        + "test -f " + shlex.quote(str(temporary)) + "; "
        + "test \"$(sha256sum " + shlex.quote(str(temporary)) + " | cut -d' ' -f1)\" = " + shlex.quote(digest) + "; "
        + "if test -e " + shlex.quote(str(remote)) + "; then "
        + "test -f " + shlex.quote(str(remote)) + " && "
        + "test \"$(sha256sum " + shlex.quote(str(remote)) + " | cut -d' ' -f1)\" = " + shlex.quote(digest) + "; "
        + "rm -f -- " + shlex.quote(str(temporary)) + "; else "
        + "chmod 600 " + shlex.quote(str(temporary)) + " && mv -- "
        + shlex.quote(str(temporary)) + " " + shlex.quote(str(remote)) + "; fi"
    )
    promoted = run_remote(instance, command)
    if promoted.returncode != 0:
        raise RuntimeError("refusing to replace a different staged Brev interface file")
    return digest


def stage_interface(instance: str, request_path: pathlib.Path, request: dict[str, Any]) -> None:
    paths = _state_paths(request)
    _stage_file(instance, request_path, paths["request"])
    _stage_file(instance, pathlib.Path(__file__).resolve(), paths["controller"])


def stage_all_interfaces(request_path: pathlib.Path, request: dict[str, Any]) -> None:
    instances = list(dict.fromkeys([
        request["coordinator"]["instance"],
        *[item["instance"] for item in request["workers"]],
    ]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(instances)) as executor:
        futures = [executor.submit(stage_interface, instance, request_path, request) for instance in instances]
        for future in futures:
            future.result()


def _worker_paths(request: dict[str, Any], worker: dict[str, Any]) -> dict[str, pathlib.Path]:
    root = pathlib.Path(request["remote"]["stage_dir"]) / "brev-workers" / worker["id"]
    return {"root": root, "manifest": root / "manifest.json", "logs": root / "logs"}


def _worker_command(
    request: dict[str, Any], verb: str, job_id: str, worker_index: int,
    *, recreate_owned: bool = False,
) -> str:
    paths = _state_paths(request)
    argv = [
        request["remote"]["controller_python"], str(paths["controller"]),
        verb, "--request", str(paths["request"]),
        "--job-id", job_id, "--worker-index", str(worker_index),
    ]
    if recreate_owned:
        if verb != "_worker_start":
            raise ValueError("owned recreation is valid only for worker start")
        argv.append("--recreate-owned")
    return shlex.join(argv)


def _fanout_workers(
    request: dict[str, Any], job_id: str, verb: str,
    *, environment: dict[str, str] | None = None,
    recreate_owned: bool = False,
) -> list[dict[str, Any]]:
    def invoke(index: int, worker: dict[str, Any]) -> dict[str, Any]:
        return _remote_json(
            worker["instance"], _worker_command(
                request, verb, job_id, index, recreate_owned=recreate_owned
            ),
            environment=environment, timeout=request["timeouts"]["worker_s"],
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(request["workers"])) as executor:
        futures = {
            executor.submit(invoke, index, worker): index
            for index, worker in enumerate(request["workers"])
        }
        results: dict[int, dict[str, Any]] = {}
        errors = []
        for future, index in futures.items():
            try:
                results[index] = future.result()
            except Exception as exc:
                errors.append(f"worker[{index}]: {exc}")
        if errors:
            raise RuntimeError("Brev worker fanout failed: " + "; ".join(errors))
    return [results[index] for index in range(len(request["workers"]))]


def _sync_file(instance: str, remote: pathlib.Path, local: pathlib.Path) -> str:
    evidence = run_remote(instance, shlex.join(["sha256sum", str(remote)]))
    stdout = evidence.stdout.decode(errors="replace") if isinstance(evidence.stdout, bytes) else (evidence.stdout or "")
    fields = stdout.strip().split()
    if evidence.returncode != 0 or len(fields) != 2 or fields[1] != str(remote) or not SHA256.fullmatch(fields[0]):
        raise RuntimeError(f"remote output digest evidence is invalid for {remote}")
    digest = fields[0]
    local.parent.mkdir(parents=True, exist_ok=True)
    temporary = local.with_name(f".{local.name}.{uuid.uuid4().hex}.tmp")
    copied = subprocess.run(
        ["scp", "-q", "--", f"{instance}:{remote}", str(temporary)],
        capture_output=True, text=True, check=False,
    )
    try:
        if copied.returncode != 0 or not temporary.is_file() or temporary.is_symlink():
            raise RuntimeError(f"Brev output copy failed for {remote}: " + _redact(copied.stderr.strip()))
        if _file_sha256(temporary) != digest:
            raise RuntimeError(f"Brev output digest changed during copy: {remote}")
        os.chmod(temporary, 0o600)
        os.replace(temporary, local)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return digest


def _stage_json_replace(instance: str, remote: pathlib.Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n"
    with tempfile.NamedTemporaryFile(prefix="tao-brev-pool-", suffix=".json") as stream:
        stream.write(encoded)
        stream.flush()
        temporary = remote.with_name(f".{remote.name}.{uuid.uuid4().hex}.tmp")
        created = run_remote(instance, shlex.join(["install", "-d", "-m", "700", str(remote.parent)]))
        if created.returncode != 0:
            raise RuntimeError("Brev pool directory creation failed")
        copied = subprocess.run(
            ["scp", "-q", "--", stream.name, f"{instance}:{temporary}"],
            capture_output=True, text=True, check=False,
        )
        if copied.returncode != 0:
            raise RuntimeError("Brev endpoint-pool staging failed")
        promoted = run_remote(instance, shlex.join(["mv", "-f", str(temporary), str(remote)]))
        if promoted.returncode != 0:
            raise RuntimeError("Brev endpoint-pool promotion failed")


def _pool_candidate(request: dict[str, Any], workers: list[dict[str, Any]]) -> dict[str, Any]:
    endpoints = [endpoint for worker in workers for endpoint in worker.get("endpoints", [])]
    required = _required_capacity(request)
    if len(workers) != request["generation_nodes"] or len(endpoints) != required:
        raise RuntimeError("worker readiness did not produce every approved endpoint slot")
    models = {
        (worker.get("model", {}).get("id"), worker.get("model", {}).get("revision"))
        for worker in workers
    }
    if len(models) != 1 or None in next(iter(models)):
        raise RuntimeError("worker readiness model identities disagree")
    expected_model = request["models"]["image_edit"]
    if next(iter(models)) != (expected_model["id"], expected_model["revision"]):
        raise RuntimeError("worker readiness model identity differs from the signed request")
    identities = {item.get("id") for item in endpoints}
    urls = {item.get("url") for item in endpoints}
    if len(identities) != required or len(urls) != required or any(item.get("capacity") != 1 for item in endpoints):
        raise RuntimeError("worker endpoint identities, URLs, and unit capacities must be exact")
    model_id, revision = next(iter(models))
    return {
        "schema_version": "1", "platform": "brev",
        "model": {"id": model_id, "revision": revision},
        "required_capacity": required, "auth_env": "IMAGE_EDIT_API_KEY",
        "endpoints": endpoints, "created_at": request["started_at"],
        "request_sha256": request["request_sha256"],
    }


def sync_outputs(instance: str, request: dict[str, Any]) -> dict[str, Any]:
    records = []
    for remote, local in zip(
        request["remote"]["expected_outputs"], request["local"]["expected_outputs"]
    ):
        digest = _sync_file(instance, pathlib.Path(remote), pathlib.Path(local))
        records.append({"remote": remote, "local": local, "sha256": digest})
    return {"status": "synced", "outputs": records}


def _remote_command(request: dict[str, Any], verb: str, job_id: str, *extra: str) -> str:
    paths = _state_paths(request)
    return shlex.join([
        request["remote"]["controller_python"], str(paths["controller"]),
        verb, "--request", str(paths["request"]),
        "--job-id", job_id, *extra,
    ])


def remote_reconcile(instance: str, request: dict[str, Any], job_id: str) -> dict[str, Any]:
    return _remote_json(instance, _remote_command(request, "_reconcile", job_id))


def build_controller_start_command(request: dict[str, Any], job_id: str) -> str:
    paths = _state_paths(request)
    controller = _remote_command(request, "_controller", job_id)
    return (
        "umask 077; " + shlex.join(["mkdir", "-p", str(paths["root"])])
        + "; " + shlex.join(["rm", "-f", str(paths["cancel"])])
        + "; nohup " + controller + " >>" + shlex.quote(str(paths["log"]))
        + " 2>&1 </dev/null & printf '" + MARKER
        + "{\"state\":\"STARTED\",\"pid\":%s}\\n' \"$!\""
    )


def probe_controller_runtime(request: dict[str, Any]) -> list[dict[str, str]]:
    """Prove every host has the exact signed virtualenv and SDG dependencies."""
    executable = request["remote"]["controller_python"]
    probe = (
        "import json,sys,yaml,pandas,pyarrow; "
        "assert sys.prefix != sys.base_prefix; "
        "print(json.dumps({'status':'PASS','prefix':sys.prefix}))"
    )
    instances = list(dict.fromkeys([
        request["coordinator"]["instance"],
        *[item["instance"] for item in request["workers"]],
    ]))

    def run(instance: str) -> dict[str, str]:
        completed = run_remote(
            instance, shlex.join([executable, "-I", "-c", probe]), timeout=120
        )
        stdout = completed.stdout.decode(errors="replace") if isinstance(completed.stdout, bytes) else (completed.stdout or "")
        if completed.returncode != 0:
            raise RuntimeError(
                f"Brev SDG controller runtime probe failed on {instance}; "
                "provision the signed workspace virtualenv before submit"
            )
        lines = [line for line in stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise RuntimeError(f"Brev SDG controller runtime probe was malformed on {instance}")
        evidence = json.loads(lines[0])
        if evidence.get("status") != "PASS" or not isinstance(evidence.get("prefix"), str):
            raise RuntimeError(f"Brev SDG controller runtime probe did not pass on {instance}")
        return {"instance": instance, "status": "PASS", "runtime": executable}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(instances)) as executor:
        return list(executor.map(run, instances))


def submit(args: argparse.Namespace) -> int:
    request = load_request(args.request)
    job_id = _safe_id(args.job_id, "job_id")
    validate_job_record(args.job_record, request, job_id)
    if args.instance != request["coordinator"]["instance"]:
        raise ValueError("--instance must match signed coordinator.instance")
    runtime_evidence = probe_controller_runtime(request)
    stage_all_interfaces(args.request, request)
    evidence = remote_reconcile(args.instance, request, job_id)
    if evidence.get("state") == "ACTIVE":
        print(json.dumps(evidence, sort_keys=True))
        return 0
    if evidence.get("state") == "NONE" and args.resume:
        raise RuntimeError("--resume requires existing atomic progress")
    if evidence.get("state") in {"RESUMABLE", "TERMINAL"}:
        if not args.resume:
            raise RuntimeError("existing evidence requires explicit --resume")
        if evidence.get("workflow_status") == "COMPLETE":
            raise RuntimeError("completed composite action cannot be resubmitted")
    forwarded = {}
    for name in request["forward_env"]:
        if not os.environ.get(name):
            raise RuntimeError(f"approved forwarded variable is unset: {name}")
        forwarded[name] = os.environ[name]
    image_edit_key = secrets.token_urlsafe(32)
    worker_environment = {
        **forwarded, "IMAGE_EDIT_API_KEY": image_edit_key,
        "VLLM_API_KEY": image_edit_key,
    }
    try:
        worker_evidence = _fanout_workers(
            request, job_id, "_worker_start", environment=worker_environment,
            recreate_owned=args.resume,
        )
    except Exception:
        try:
            _fanout_workers(request, job_id, "_worker_stop")
        except Exception:
            pass
        raise
    candidate = _pool_candidate(request, worker_evidence)
    _stage_json_replace(
        args.instance, _state_paths(request)["pool_candidate"], candidate
    )
    # The shared manager authenticates both the image-edit pool and the local
    # VLM/LLM readiness probes. Keep one ephemeral action key, but expose both
    # variable names only in the remote controller process environment.
    forwarded["IMAGE_EDIT_API_KEY"] = image_edit_key
    forwarded["VLLM_API_KEY"] = image_edit_key
    started = _remote_json(
        args.instance, build_controller_start_command(request, job_id),
        environment=forwarded, timeout=600,
    )
    print(json.dumps({
        **_identity(request, job_id), **started,
        "workers": worker_evidence, "required_capacity": _required_capacity(request),
        "controller_runtime": runtime_evidence,
    }, sort_keys=True))
    return 0


def status(args: argparse.Namespace) -> int:
    request = load_request(args.request)
    job_id = _safe_id(args.job_id, "job_id")
    validate_job_record(args.job_record, request, job_id)
    if args.instance != request["coordinator"]["instance"]:
        raise ValueError("--instance must match signed coordinator.instance")
    evidence = remote_reconcile(args.instance, request, job_id)
    workers = _fanout_workers(request, job_id, "_worker_status")
    mapped = {
        "ACTIVE": "RUNNING", "RESUMABLE": "ERROR", "NONE": "UNKNOWN",
    }.get(evidence["state"], evidence.get("workflow_status", "UNKNOWN"))
    allowed_worker_states = {"READY", "STOPPED"} if mapped in {"COMPLETE", "CANCELED"} else {"READY"}
    if mapped in {"RUNNING", "COMPLETE", "CANCELED"} and any(
        worker.get("state") not in allowed_worker_states for worker in workers
    ):
        mapped = "ERROR"
    synchronization = sync_outputs(args.instance, request) if mapped == "COMPLETE" else None
    cleanup = _fanout_workers(request, job_id, "_worker_stop") if mapped in {"COMPLETE", "CANCELED"} else None
    print(json.dumps({
        "status": mapped, "native_state": evidence["state"],
        "exit_code": evidence.get("exit_code"), "synchronization": synchronization,
        "workers": workers, "worker_cleanup": cleanup,
    }, sort_keys=True))
    return 0


def logs(args: argparse.Namespace) -> int:
    request = load_request(args.request)
    job_id = _safe_id(args.job_id, "job_id")
    validate_job_record(args.job_record, request, job_id)
    if args.instance != request["coordinator"]["instance"]:
        raise ValueError("--instance must match signed coordinator.instance")
    completed = run_remote(args.instance, _remote_command(request, "_logs", job_id, "--tail", str(args.tail)))
    stdout = completed.stdout.decode(errors="replace") if isinstance(completed.stdout, bytes) else (completed.stdout or "")
    stderr = completed.stderr.decode(errors="replace") if isinstance(completed.stderr, bytes) else (completed.stderr or "")
    sys.stdout.write(_redact(stdout))
    sys.stderr.write(_redact(stderr))
    return int(completed.returncode)


def cancel(args: argparse.Namespace) -> int:
    if not args.confirm:
        raise ValueError("cancel requires --confirm after user approval")
    request = load_request(args.request)
    job_id = _safe_id(args.job_id, "job_id")
    validate_job_record(args.job_record, request, job_id)
    if args.instance != request["coordinator"]["instance"]:
        raise ValueError("--instance must match signed coordinator.instance")
    evidence = _remote_json(
        args.instance, _remote_command(request, "_cancel", job_id),
        timeout=request["timeouts"]["cancel_s"] + 10,
    )
    worker_cleanup = _fanout_workers(request, job_id, "_worker_stop")
    print(json.dumps({**evidence, "worker_cleanup": worker_cleanup}, sort_keys=True))
    return 0


def build_worker_helper_command(
    request: dict[str, Any], index: int, action: str, output: pathlib.Path,
    *, recreate_owned: bool = False,
) -> list[str]:
    worker = request["workers"][index]
    remote = request["remote"]
    argv = [
        remote["controller_python"],
        str(pathlib.Path(remote["runtime_root"]) / "manage_sdg_endpoints.py"),
        action, "--roles", "image_edit", "--platform", "brev",
        "--service-host", worker["address"],
        "--gpu-identity-prefix", worker["instance"],
        "--request-sha256", request["request_sha256"],
        "--config", remote["config_path"],
        "--run-id", f"{request['run_id']}-{request['action_id'][:24]}-img-{index:03d}",
        "--cache-dir", remote["cache_dir"], "--output", str(output),
    ]
    if recreate_owned:
        if action != "start":
            raise ValueError("--recreate-owned is valid only for start")
        argv.append("--recreate-owned")
    return argv


def _run_worker_helper(
    request: dict[str, Any], job_id: str, index: int, action: str,
    *, recreate_owned: bool = False,
) -> dict[str, Any]:
    worker = request["workers"][index]
    worker_capacity = len(worker["gpu_ids"])
    paths = _worker_paths(request, worker)
    paths["root"].mkdir(parents=True, exist_ok=True)
    output = paths["manifest"] if action == "start" else paths["root"] / f"{action}.json"
    if action == "start":
        if recreate_owned:
            ownership_path = paths["root"] / "resume-ownership.json"
            checked = subprocess.run(
                build_worker_helper_command(request, index, "status", ownership_path),
                capture_output=True, text=True, check=False,
            )
            ownership = _read_json(ownership_path) if checked.returncode == 0 else None
            containers = ownership.get("containers") if isinstance(ownership, dict) else None
            if (
                not isinstance(containers, dict) or len(containers) != worker_capacity
                or any(
                    not isinstance(item, dict) or item.get("owned") is not True
                    for item in containers.values()
                )
                or ownership.get("request_sha256", request["request_sha256"])
                != request["request_sha256"]
            ):
                raise RuntimeError(
                    f"resume recreation lacks exact signed ownership for {worker['id']}"
                )
        plan_path = paths["root"] / "plan.json"
        planned = subprocess.run(
            build_worker_helper_command(request, index, "plan", plan_path),
            capture_output=True, text=True, check=False,
        )
        if planned.returncode != 0:
            raise RuntimeError(f"shared endpoint manager plan failed for {worker['id']}")
        plan = _read_json(plan_path)
        commands = plan.get("commands") if isinstance(plan, dict) else None
        if not isinstance(commands, dict) or len(commands) != worker_capacity:
            raise RuntimeError(f"worker plan lacks every signed image-edit command for {worker['id']}")
        planned_gpus: set[int] = set()
        planned_ports: set[int] = set()
        for command in commands.values():
            if not isinstance(command, list) or "--gpus" not in command or "-p" not in command:
                raise RuntimeError(f"worker plan command is malformed for {worker['id']}")
            selector = command[command.index("--gpus") + 1].strip('"')
            if not selector.startswith("device=") or not selector[7:].isdigit():
                raise RuntimeError(f"worker plan widened GPU selection for {worker['id']}")
            planned_gpus.add(int(selector[7:]))
            binding = command[command.index("-p") + 1]
            parts = binding.split(":")
            if len(parts) != 3 or parts[0] != "0.0.0.0" or parts[1] != parts[2] or not parts[1].isdigit():
                raise RuntimeError(
                    f"worker plan is not directly reachable; Brev image services must bind 0.0.0.0"
                )
            planned_ports.add(int(parts[1]))
            secret_values = [os.environ.get(name) for name in ("IMAGE_EDIT_API_KEY", "VLLM_API_KEY")]
            if "all" in command or any(value and value in command for value in secret_values):
                raise RuntimeError(f"worker plan leaked a secret or widened GPU selection for {worker['id']}")
        if planned_gpus != set(worker["gpu_ids"]) or planned_ports != set(worker["ports"]):
            raise RuntimeError(f"worker plan GPU/port allocation differs from signed topology for {worker['id']}")
    result = subprocess.run(
        build_worker_helper_command(
            request, index, action, output, recreate_owned=recreate_owned
        ),
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"shared endpoint manager {action} failed for {worker['id']}: "
            + _redact((result.stderr or "").strip())
        )
    value = _read_json(output)
    if not value:
        raise RuntimeError(f"shared endpoint manager {action} wrote no evidence for {worker['id']}")
    if action != "start":
        state = "STOPPED" if action == "stop" else "ERROR"
        if action == "status":
            containers = value.get("containers")
            if isinstance(containers, dict) and len(containers) == worker_capacity:
                if all(
                    isinstance(item, dict) and item.get("owned") is True
                    and item.get("running") is True for item in containers.values()
                ):
                    state = "READY"
                elif all(
                    isinstance(item, dict) and item.get("owned") is True
                    and item.get("running") is False for item in containers.values()
                ):
                    state = "STOPPED"
        return {
            **_identity(request, job_id), "worker_id": worker["id"],
            "state": state, "manager": value,
        }
    pool = value.get("image_edit_endpoint_pool", value)
    if not isinstance(pool, dict):
        raise RuntimeError(f"shared endpoint manager emitted no pool for {worker['id']}")
    endpoints = pool.get("endpoints")
    model = pool.get("model")
    if (
        pool.get("schema_version") != "1" or pool.get("platform") != "brev"
        or pool.get("request_sha256") != request["request_sha256"]
        or pool.get("required_capacity") != worker_capacity or pool.get("auth_env") != "IMAGE_EDIT_API_KEY"
        or not isinstance(endpoints, list) or len(endpoints) != worker_capacity
        or not isinstance(model, dict) or set(model) != {"id", "revision"}
    ):
        raise RuntimeError(f"shared endpoint manager emitted an invalid pool for {worker['id']}")
    expected_urls = {
        f"http://{worker['address']}:{port}/v1" for port in worker["ports"]
    }
    if {item.get("url") for item in endpoints if isinstance(item, dict)} != expected_urls:
        raise RuntimeError(f"worker pool URLs do not match signed ports for {worker['id']}")
    expected_gpu_ids = {f"{worker['instance']}/gpu:{gpu_id}" for gpu_id in worker["gpu_ids"]}
    if {item.get("gpu_identity") for item in endpoints if isinstance(item, dict)} != expected_gpu_ids:
        raise RuntimeError(f"worker pool GPU identities do not match {worker['id']}")
    if any(
        not isinstance(item, dict)
        or set(item) != {"id", "url", "capacity", "gpu_identity", "owner"}
        or item.get("capacity") != 1
        or not isinstance(item.get("owner"), dict)
        or set(item["owner"]) != {"native_id", "name"}
        or not item["owner"].get("native_id") or not item["owner"].get("name")
        for item in endpoints
    ):
        raise RuntimeError(f"worker pool lacks exact unit-capacity ownership for {worker['id']}")
    return {
        **_identity(request, job_id), "state": "READY", "worker_id": worker["id"],
        "instance": worker["instance"], "capacity": worker_capacity, "model": model,
        "endpoints": endpoints,
        "manager_manifest": {"path": str(output), "sha256": _file_sha256(output)},
        "recovery": "recreated_exact_owned_for_ephemeral_auth" if recreate_owned else None,
    }


def _worker_start(
    request: dict[str, Any], job_id: str, index: int, *, recreate_owned: bool = False
) -> dict[str, Any]:
    if not os.environ.get("IMAGE_EDIT_API_KEY"):
        raise RuntimeError("ephemeral IMAGE_EDIT_API_KEY was not forwarded")
    return _run_worker_helper(
        request, job_id, index, "start", recreate_owned=recreate_owned
    )


def _worker_status(request: dict[str, Any], job_id: str, index: int) -> dict[str, Any]:
    return _run_worker_helper(request, job_id, index, "status")


def _worker_stop(request: dict[str, Any], job_id: str, index: int) -> dict[str, Any]:
    return _run_worker_helper(request, job_id, index, "stop")


class Controller:
    def __init__(self, request: dict[str, Any], job_id: str) -> None:
        self.request = request
        self.job_id = job_id
        self.identity = _identity(request, job_id)
        self.paths = _state_paths(request)
        self.current: subprocess.Popen[str] | None = None
        self.cancelled = False
        self.secrets = [os.environ.get(name, "") for name in request["forward_env"]]

    def _signal(self, _signum: int, _frame: Any) -> None:
        self.cancelled = True
        if self.current is not None and self.current.poll() is None:
            self.current.terminate()

    def _status(self, state: str, code: int | None, message: str) -> None:
        _atomic_json(self.paths["status"], {
            **self.identity, "status": state, "exit_code": code,
            "message": _redact(message, self.secrets), "updated_ns": time.time_ns(),
        })

    def _progress(self) -> dict[str, Any]:
        progress = _read_json(self.paths["progress"])
        if progress is None:
            progress = {**self.identity, "controller_attempt": 0, "runtime_complete": False}
        else:
            _validate_identity(progress, self.identity)
        progress["controller_attempt"] = int(progress.get("controller_attempt", 0)) + 1
        if progress["controller_attempt"] > self.request["resume"]["max_controller_attempts"]:
            raise RuntimeError("controller resume attempt budget exhausted")
        _atomic_json(self.paths["progress"], progress)
        return progress

    def _run(self, argv: list[str], log_path: pathlib.Path) -> None:
        with log_path.open("a", encoding="utf-8") as stream:
            self.current = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, text=True)
            try:
                code = self.current.wait(timeout=self.request["timeouts"]["controller_s"])
            except subprocess.TimeoutExpired as exc:
                self.current.terminate()
                try:
                    self.current.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.current.kill()
                    self.current.wait(timeout=10)
                raise RuntimeError(
                    f"child exceeded the {self.request['timeouts']['controller_s']} second deadline"
                ) from exc
            finally:
                self.current = None
        log_path.write_text(_redact(log_path.read_text(errors="replace"), self.secrets), encoding="utf-8")
        os.chmod(log_path, 0o600)
        if self.cancelled or self.paths["cancel"].exists():
            raise Cancelled("controller canceled during child execution")
        if code != 0:
            raise RuntimeError(f"child exited {code}; inspect {log_path}")

    def _helper(self, action: str, output: pathlib.Path) -> list[str]:
        remote = self.request["remote"]
        return [
            remote["controller_python"],
            str(pathlib.Path(remote["runtime_root"]) / "manage_sdg_endpoints.py"),
            action, "--roles", "vlm,llm", "--platform", "brev",
            "--service-host", "127.0.0.1",
            "--request-sha256", self.request["request_sha256"],
            "--config", remote["config_path"], "--run-id", self.request["run_id"] + "-aux",
            "--cache-dir", remote["cache_dir"], "--output", str(output),
        ]

    def _validate_staged_runtime(self) -> None:
        remote = self.request["remote"]
        config = pathlib.Path(remote["config_path"])
        if not config.is_file() or config.is_symlink() or _file_sha256(config) != remote["config_sha256"]:
            raise RuntimeError("staged SDG config digest mismatch")
        runtime = pathlib.Path(remote["runtime_root"])
        if _tree_sha256(runtime) != remote["runtime_sha256"]:
            raise RuntimeError("staged shared SDG runtime digest mismatch")
        for name in ("manage_sdg_endpoints.py", "run_sdg_stage.py"):
            if not (runtime / name).is_file():
                raise RuntimeError(f"staged shared runtime is missing {name}")

    def _validate_endpoint_manifest(self) -> None:
        value = _read_json(self.paths["endpoint_manifest"])
        if not value or value.get("status") != "success":
            raise RuntimeError("endpoint manager did not commit successful readiness evidence")
        if value.get("ownership") == "managed":
            containers = value.get("containers")
            if not isinstance(containers, dict) or set(containers) != {"vlm", "llm"}:
                raise RuntimeError("endpoint manifest lacks the two shared auxiliary roles")
            if any(not isinstance(item, dict) or item.get("owned") is not True for item in containers.values()):
                raise RuntimeError("endpoint manifest contains an unowned managed role")

    def _validate_endpoint_status(self) -> None:
        value = _read_json(self.paths["endpoint_status"])
        manifest = _read_json(self.paths["endpoint_manifest"])
        if not value or not manifest or value.get("ownership") != manifest.get("ownership"):
            raise RuntimeError("endpoint status does not match the readiness manifest")
        if value.get("ownership") == "managed":
            containers = value.get("containers")
            if not isinstance(containers, dict) or set(containers) != {"vlm", "llm"}:
                raise RuntimeError("endpoint status lacks the two shared auxiliary roles")
            if any(
                not isinstance(item, dict)
                or item.get("owned") is not True
                or item.get("running") is not True
                for item in containers.values()
            ):
                raise RuntimeError("endpoint status contains an unowned or stopped role")

    def _validate_aux_plan(self) -> None:
        plan = _read_json(self.paths["endpoint_plan"])
        commands = plan.get("commands") if isinstance(plan, dict) else None
        if not isinstance(commands, dict) or set(commands) != {"vlm", "llm"}:
            raise RuntimeError("auxiliary endpoint plan must contain exactly VLM and LLM")
        for role in ("vlm", "llm"):
            command = commands[role]
            expected_ids = self.request["coordinator"]["gpu_ids"][role]
            expected_selector = "device=" + ",".join(str(item) for item in expected_ids)
            expected_port = self.request["models"][role]["port"]
            if not isinstance(command, list) or "--gpus" not in command or "-p" not in command:
                raise RuntimeError(f"auxiliary {role} endpoint plan is malformed")
            selector = command[command.index("--gpus") + 1].strip('"')
            binding = command[command.index("-p") + 1]
            if selector != expected_selector or binding != f"127.0.0.1:{expected_port}:{expected_port}":
                raise RuntimeError(f"auxiliary {role} GPU/port allocation differs from signed topology")
            secret_values = [os.environ.get(name) for name in self.request["forward_env"]]
            if "all" in command or any(value and value in command for value in secret_values):
                raise RuntimeError(f"auxiliary {role} plan leaked a secret or widened GPU selection")

    def _validate_and_probe_pool(self) -> dict[str, Any]:
        candidate = _read_json(self.paths["pool_candidate"])
        required = _required_capacity(self.request)
        if not candidate or set(candidate) != {
            "schema_version", "platform", "model", "required_capacity", "auth_env",
            "endpoints", "created_at", "request_sha256",
        }:
            raise RuntimeError("endpoint-pool candidate has missing or unexpected fields")
        if (
            candidate["schema_version"] != "1"
            or candidate["platform"] != "brev"
            or candidate["required_capacity"] != required
            or candidate["auth_env"] != "IMAGE_EDIT_API_KEY"
            or candidate["request_sha256"] != self.request["request_sha256"]
            or candidate["created_at"] != self.request["started_at"]
        ):
            raise RuntimeError("endpoint-pool candidate identity/capacity is invalid")
        endpoints = candidate.get("endpoints")
        if not isinstance(endpoints, list) or len(endpoints) != required:
            raise RuntimeError("endpoint-pool candidate lacks every approved slot")
        expected = {
            (f"{worker['instance']}/gpu:{gpu_id}", f"http://{worker['address']}:{port}/v1")
            for worker in self.request["workers"]
            for gpu_id, port in zip(worker["gpu_ids"], worker["ports"])
        }
        actual = {
            (item.get("gpu_identity"), item.get("url"))
            for item in endpoints if isinstance(item, dict)
        }
        if actual != expected or any(
            not isinstance(item, dict) or set(item) != {"id", "url", "capacity", "gpu_identity", "owner"}
            or item.get("capacity") != 1
            or not isinstance(item.get("owner"), dict)
            or set(item["owner"]) != {"native_id", "name"}
            or not item["owner"].get("native_id") or not item["owner"].get("name")
            for item in endpoints
        ):
            raise RuntimeError("endpoint-pool slots do not match signed instance/GPU topology")
        key = os.environ.get("IMAGE_EDIT_API_KEY")
        if not key:
            raise RuntimeError("ephemeral IMAGE_EDIT_API_KEY is unavailable on coordinator")
        smoke = pathlib.Path(self.request["remote"]["smoke_image"])
        if not smoke.is_file() or smoke.is_symlink() or smoke.stat().st_size == 0:
            raise RuntimeError("coordinator smoke image is unavailable")
        readiness = []
        for item in endpoints:
            base = item["url"].rstrip("/")
            models_request = urllib.request.Request(base + "/models")
            models_request.add_header("Authorization", "Bearer " + key)
            try:
                with urllib.request.urlopen(models_request, timeout=30) as response:
                    models = json.loads(response.read().decode())
            except (OSError, ValueError, urllib.error.URLError) as exc:
                raise RuntimeError(
                    f"coordinator cannot directly reach {item['id']}; verify private Brev networking/firewall"
                ) from exc
            if candidate["model"]["id"] not in [
                row.get("id") for row in models.get("data", []) if isinstance(row, dict)
            ]:
                raise RuntimeError(f"endpoint {item['id']} serves the wrong model")
            boundary = "tao" + uuid.uuid4().hex
            body = (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{candidate['model']['id']}\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nReturn this image unchanged.\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"smoke.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n"
            ).encode() + smoke.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
            edit_request = urllib.request.Request(base + "/images/edits", data=body, method="POST")
            edit_request.add_header("Authorization", "Bearer " + key)
            edit_request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
            try:
                with urllib.request.urlopen(edit_request, timeout=self.request["timeouts"]["readiness_s"]) as response:
                    result = json.loads(response.read().decode())
            except (OSError, ValueError, urllib.error.URLError) as exc:
                raise RuntimeError(f"direct inference smoke failed for {item['id']}") from exc
            if not isinstance(result, dict) or not result.get("data"):
                raise RuntimeError(f"direct inference smoke returned no image for {item['id']}")
            readiness.append({"id": item["id"], "reachable_from": self.request["coordinator"]["instance"], "models_ok": True, "inference_ok": True})
        _atomic_json(self.paths["pool_readiness"], {
            **self.identity, "required_capacity": required, "ready_capacity": len(readiness),
            "control_host_relay": False, "endpoints": readiness,
        })
        _atomic_json(pathlib.Path(self.request["remote"]["endpoint_pool_path"]), candidate)
        return candidate

    def _outputs_complete(self) -> bool:
        return all(
            pathlib.Path(path).is_file() and not pathlib.Path(path).is_symlink()
            and pathlib.Path(path).stat().st_size > 0
            for path in self.request["remote"]["expected_outputs"]
        )

    def cleanup(self) -> None:
        helper = self._helper("stop", self.paths["endpoint_manifest"])
        completed = subprocess.run(helper, capture_output=True, text=True, check=False)
        cleanup_log = self.paths["root"] / "endpoint_stop.log"
        cleanup_log.write_text(_redact((completed.stdout or "") + (completed.stderr or ""), self.secrets), encoding="utf-8")
        os.chmod(cleanup_log, 0o600)
        if completed.returncode != 0:
            raise RuntimeError(f"owned endpoint stop failed; inspect {cleanup_log}")

    def run(self) -> int:
        self.paths["root"].mkdir(parents=True, exist_ok=True)
        self.paths["pid"].write_text(str(os.getpid()) + "\n", encoding="utf-8")
        os.chmod(self.paths["pid"], 0o600)
        signal.signal(signal.SIGTERM, self._signal)
        signal.signal(signal.SIGINT, self._signal)
        terminal, code, message = "ERROR", 1, "controller failed"
        endpoint_started = False
        try:
            progress = self._progress()
            self._status("RUNNING", None, "controller running")
            self._validate_staged_runtime()
            if progress.get("runtime_complete"):
                if not self._outputs_complete():
                    raise RuntimeError("runtime is journaled complete but canonical outputs are incomplete")
            else:
                remote = self.request["remote"]
                runtime = pathlib.Path(remote["runtime_root"]) / "run_sdg_stage.py"
                prepare = [
                    remote["controller_python"], str(runtime), "prepare",
                    "--config", remote["config_path"],
                    "--output-root", remote["stage_dir"],
                    "--mined-pairs", remote["mined_pairs"],
                    "--gaps-parquet", remote["gaps_parquet"],
                    "--attribute-vocab", remote["attribute_vocab"],
                    "--dataset-root", remote["dataset_root"],
                    "--eval-list", remote["eval_list"],
                    "--eval-pairs", remote["eval_pairs"],
                ]
                # The shared prepare operation is atomic and idempotent. Run it
                # before endpoint work so execute never sees an absent SDG plan.
                self._run(prepare, self.paths["runtime_log"])
                self._validate_and_probe_pool()
                self._run(self._helper("plan", self.paths["endpoint_plan"]), self.paths["runtime_log"])
                self._validate_aux_plan()
                self._run(self._helper("start", self.paths["endpoint_manifest"]), self.paths["runtime_log"])
                endpoint_started = True
                self._validate_endpoint_manifest()
                self._run(self._helper("status", self.paths["endpoint_status"]), self.paths["runtime_log"])
                self._validate_endpoint_status()
                command = [
                    remote["controller_python"], str(runtime), "execute", "--execution-platform", "brev",
                    "--config", remote["config_path"],
                    "--output-root", remote["stage_dir"],
                    "--mined-pairs", remote["mined_pairs"],
                    "--eval-list", remote["eval_list"],
                    "--attribute-vocab", remote["attribute_vocab"],
                    "--image-edit-endpoint-pool", remote["endpoint_pool_path"],
                ]
                self._run(command, self.paths["runtime_log"])
                if not self._outputs_complete():
                    raise RuntimeError("shared SDG runtime missed canonical outputs")
                progress["runtime_complete"] = True
                _atomic_json(self.paths["progress"], progress)
            terminal, code, message = "COMPLETE", 0, "composite SDG action completed"
        except Cancelled as exc:
            terminal, code, message = "CANCELED", 130, str(exc)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            terminal, code, message = "ERROR", 1, str(exc)
        finally:
            if endpoint_started or self.paths["endpoint_manifest"].is_file():
                try:
                    self.cleanup()
                except Exception as exc:
                    terminal, code = "ERROR", 1
                    message = f"{message}; cleanup failed: {exc}"
            self._status(terminal, code, message)
            try:
                self.paths["pid"].unlink()
            except FileNotFoundError:
                pass
        return code


def _internal_request(args: argparse.Namespace) -> dict[str, Any]:
    return load_request(args.request)


def internal_reconcile(args: argparse.Namespace) -> int:
    print(MARKER + json.dumps(reconcile_local(_internal_request(args), _safe_id(args.job_id, "job_id")), sort_keys=True))
    return 0


def internal_logs(args: argparse.Namespace) -> int:
    request = _internal_request(args)
    identity = _identity(request, _safe_id(args.job_id, "job_id"))
    status_value = _read_json(_state_paths(request)["status"])
    if status_value is not None:
        _validate_identity(status_value, identity)
    log = _state_paths(request)["log"]
    if not log.is_file() or log.is_symlink():
        raise RuntimeError("controller log is missing")
    print(_redact("\n".join(log.read_text(errors="replace").splitlines()[-args.tail:])))
    return 0


def internal_cancel(args: argparse.Namespace) -> int:
    request = _internal_request(args)
    job_id = _safe_id(args.job_id, "job_id")
    evidence = reconcile_local(request, job_id)
    if evidence["state"] == "TERMINAL":
        print(MARKER + json.dumps(evidence, sort_keys=True))
        return 0
    if evidence["state"] != "ACTIVE":
        raise RuntimeError("no owned active controller to cancel")
    paths = _state_paths(request)
    _atomic_json(paths["cancel"], {**_identity(request, job_id), "requested_ns": time.time_ns()})
    os.kill(int(evidence["pid"]), signal.SIGTERM)
    deadline = time.monotonic() + request["timeouts"]["cancel_s"]
    while time.monotonic() < deadline and _pid_active(int(evidence["pid"])):
        time.sleep(0.25)
    final = reconcile_local(request, job_id)
    if final.get("workflow_status") != "CANCELED":
        raise RuntimeError("controller did not reach CANCELED within cancel deadline")
    print(MARKER + json.dumps(final, sort_keys=True))
    return 0


def internal_controller(args: argparse.Namespace) -> int:
    return Controller(_internal_request(args), _safe_id(args.job_id, "job_id")).run()


def internal_worker_start(args: argparse.Namespace) -> int:
    request = _internal_request(args)
    if not 0 <= args.worker_index < len(request["workers"]):
        raise ValueError("--worker-index is outside the signed worker list")
    print(MARKER + json.dumps(_worker_start(
        request, _safe_id(args.job_id, "job_id"), args.worker_index,
        recreate_owned=args.recreate_owned,
    ), sort_keys=True))
    return 0


def internal_worker_status(args: argparse.Namespace) -> int:
    request = _internal_request(args)
    if not 0 <= args.worker_index < len(request["workers"]):
        raise ValueError("--worker-index is outside the signed worker list")
    print(MARKER + json.dumps(_worker_status(request, _safe_id(args.job_id, "job_id"), args.worker_index), sort_keys=True))
    return 0


def internal_worker_stop(args: argparse.Namespace) -> int:
    request = _internal_request(args)
    if not 0 <= args.worker_index < len(request["workers"]):
        raise ValueError("--worker-index is outside the signed worker list")
    print(MARKER + json.dumps(_worker_stop(request, _safe_id(args.job_id, "job_id"), args.worker_index), sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="verb", required=True)
    prepare = commands.add_parser(
        "prepare-request",
        help="derive one immutable Brev SDG request from initialized state and resolved inventory",
    )
    prepare.add_argument("--state", type=pathlib.Path, required=True)
    prepare.add_argument("--iteration", type=int, required=True)
    prepare.add_argument("--inventory", type=pathlib.Path, required=True)
    prepare.add_argument("--remote-root", type=pathlib.Path, required=True)
    prepare.add_argument("--remote-cache", type=pathlib.Path, required=True)
    prepare.add_argument("--remote-controller-python", type=pathlib.Path, required=True)
    prepare.add_argument("--output", type=pathlib.Path, required=True)
    submit_parser = commands.add_parser("submit")
    submit_parser.add_argument("--resume", action="store_true")
    for verb, child in [("submit", submit_parser)] + [
        (name, commands.add_parser(name)) for name in ("status", "logs", "cancel")
    ]:
        child.add_argument("--request", type=pathlib.Path, required=True)
        child.add_argument("--instance", required=True)
        child.add_argument("--job-id", required=True)
        child.add_argument("--job-record", type=pathlib.Path, required=True)
        if verb == "logs":
            child.add_argument("--tail", type=int, default=200)
        if verb == "cancel":
            child.add_argument("--confirm", action="store_true")
    for verb in ("_controller", "_reconcile", "_logs", "_cancel", "_worker_start", "_worker_status", "_worker_stop"):
        child = commands.add_parser(verb)
        child.add_argument("--request", type=pathlib.Path, required=True)
        child.add_argument("--job-id", required=True)
        if verb == "_logs":
            child.add_argument("--tail", type=int, default=200)
        if verb.startswith("_worker_"):
            child.add_argument("--worker-index", type=int, required=True)
        if verb == "_worker_start":
            child.add_argument("--recreate-owned", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    functions = {
        "prepare-request": prepare_request,
        "submit": submit, "status": status, "logs": logs, "cancel": cancel,
        "_controller": internal_controller, "_reconcile": internal_reconcile,
        "_logs": internal_logs, "_cancel": internal_cancel,
        "_worker_start": internal_worker_start,
        "_worker_status": internal_worker_status,
        "_worker_stop": internal_worker_stop,
    }
    try:
        if getattr(args, "tail", 1) < 1 or getattr(args, "tail", 1) > 100000:
            raise ValueError("--tail is outside the supported range")
        return functions[args.verb](args)
    except (
        OSError, RuntimeError, ValueError, json.JSONDecodeError,
        subprocess.SubprocessError, yaml.YAMLError,
    ) as exc:
        print(f"brev sdg action failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
