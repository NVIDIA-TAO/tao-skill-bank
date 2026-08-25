#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare and consume one signed composite IAA SDG Airflow action."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile
import urllib.parse
from typing import Any, Sequence

import yaml

import airflow_action as airflow


WORKFLOW = "tao-run-deft-iaa"
KIND = "airflow_sdg_action"
NAME = "sdg_execute"
MAX_IMAGE_WORKER_GPUS = 8
SHA256 = re.compile(r"[0-9a-f]{64}")
REVISION = re.compile(r"[0-9a-f]{40}")
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
PINNED_IMAGE = re.compile(r"[^\s@]+(?::[^\s@]+)?@sha256:[0-9a-f]{64}")
ROLES = ("image_edit", "vlm", "llm")
EXPECTED_KEYS = {
    "schema_version", "workflow", "kind", "platform", "name", "contract",
    "action_id", "run_id", "iteration", "attempt", "started_at", "started_ns",
    "generation_nodes", "images", "models", "resources", "paths", "limits",
    "bindings", "forward_env", "expected_outputs", "job_binding_path",
    "request_sha256",
}
COMPOSED_EXPECTED_KEYS = EXPECTED_KEYS | {"orchestrator"}


class ContractError(ValueError):
    pass


def _canonical_sha256(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("request_sha256", None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: pathlib.Path, field: str) -> pathlib.Path:
    lexical = pathlib.Path(os.path.abspath(path.expanduser()))
    if (
        not lexical.is_file()
        or lexical.is_symlink()
        or lexical.resolve(strict=False) != lexical
        or lexical.stat().st_size == 0
    ):
        raise ContractError(f"{field} must be a non-empty regular non-symlink file")
    return lexical


def _directory(path: pathlib.Path, field: str) -> pathlib.Path:
    lexical = pathlib.Path(os.path.abspath(path.expanduser()))
    if (
        not lexical.is_dir()
        or lexical.is_symlink()
        or lexical.resolve(strict=False) != lexical
    ):
        raise ContractError(f"{field} must be an existing regular non-symlink directory")
    return lexical


def _python_tree_sha256(root: pathlib.Path) -> str:
    files = sorted(
        path for path in root.rglob("*.py") if "__pycache__" not in path.parts
    )
    if not files:
        raise ContractError("staged Airflow controller has no IAA runtime Python files")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _absolute(value: Any, field: str) -> pathlib.Path:
    if not isinstance(value, str) or not value.startswith("/"):
        raise ContractError(f"{field} must be an absolute path")
    path = pathlib.Path(value)
    if path == pathlib.Path("/") or pathlib.Path(os.path.abspath(path)) != path:
        raise ContractError(f"{field} must be normalized, non-root, and traversal-free")
    return path


def _under(path: pathlib.Path, root: pathlib.Path, field: str) -> pathlib.Path:
    lexical = pathlib.Path(os.path.abspath(path.expanduser()))
    try:
        lexical.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"{field} must be visible under shared root {root}") from exc
    return lexical


def _pool_name(value: Any, field: str) -> str:
    if not isinstance(value, str) or SAFE_NAME.fullmatch(value) is None:
        raise ContractError(f"{field} must be a non-empty Airflow pool name")
    return value


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
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


@contextlib.contextmanager
def _exclusive_lock(path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(fd, "a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError("another process owns this Airflow SDG launch") from exc
        yield


def validate_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) not in {
        frozenset(EXPECTED_KEYS), frozenset(COMPOSED_EXPECTED_KEYS),
    }:
        raise ContractError("Airflow SDG request has missing or unexpected fields")
    fixed = {
        "schema_version": "1", "workflow": WORKFLOW, "kind": KIND,
        "name": NAME, "contract": airflow.CONTRACT,
    }
    for field, expected in fixed.items():
        if payload.get(field) != expected:
            raise ContractError(f"request.{field} must be {expected!r}")
    platform = payload.get("platform")
    orchestrator = payload.get("orchestrator")
    if not (
        (platform == "airflow" and orchestrator is None)
        or (platform in {"docker", "virtualenv"} and orchestrator == "airflow")
    ):
        raise ContractError(
            "request must be legacy platform=airflow or Airflow-orchestrated local "
            "compute on docker/virtualenv"
        )
    if (
        not isinstance(payload.get("request_sha256"), str)
        or SHA256.fullmatch(payload["request_sha256"]) is None
        or payload["request_sha256"] != _canonical_sha256(payload)
    ):
        raise ContractError("request_sha256 does not match immutable content")
    for field in ("action_id", "run_id"):
        if not isinstance(payload[field], str) or SAFE_NAME.fullmatch(payload[field]) is None:
            raise ContractError(f"request.{field} contains unsupported characters")
    try:
        timestamp = dt.datetime.fromisoformat(payload["started_at"].replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ContractError("started_at must be timezone-aware ISO-8601") from exc
    if timestamp.utcoffset() is None:
        raise ContractError("started_at must include a timezone")
    if (
        not isinstance(payload["started_ns"], int)
        or isinstance(payload["started_ns"], bool)
        or payload["started_ns"] < 1
    ):
        raise ContractError("started_ns must be a positive integer")
    if isinstance(payload["attempt"], bool) or payload["attempt"] not in (1, 2):
        raise ContractError("attempt must be 1 or 2")
    if (
        not isinstance(payload["iteration"], int)
        or isinstance(payload["iteration"], bool)
        or payload["iteration"] < 1
    ):
        raise ContractError("iteration must be positive")
    nodes = payload["generation_nodes"]
    if not isinstance(nodes, int) or isinstance(nodes, bool) or not 1 <= nodes <= 64:
        raise ContractError("generation_nodes must be in [1, 64]")
    images = payload["images"]
    if not isinstance(images, dict) or set(images) != {
        "augmentation", "auto_labeling", "image_edit", "text_serving", "controller",
    }:
        raise ContractError("images must bind all five SDG roles")
    for field, image in images.items():
        if field == "controller":
            if not isinstance(image, str) or not image or ":latest" in image:
                raise ContractError("images.controller must be an immutable versioned image")
        elif not isinstance(image, str) or PINNED_IMAGE.fullmatch(image) is None:
            raise ContractError(f"images.{field} must be digest pinned")
    models = payload["models"]
    if not isinstance(models, dict) or set(models) != set(ROLES):
        raise ContractError("models must bind image_edit, vlm, and llm")
    ports: set[int] = set()
    for role in ROLES:
        model = models[role]
        if not isinstance(model, dict) or set(model) != {
            "id", "revision", "backend", "port", "min_vram_mib",
        }:
            raise ContractError(f"models.{role} has an invalid shape")
        if not isinstance(model["id"], str) or not model["id"]:
            raise ContractError(f"models.{role}.id must be non-empty")
        if not isinstance(model["revision"], str) or REVISION.fullmatch(model["revision"]) is None:
            raise ContractError(f"models.{role}.revision must be immutable")
        if model["backend"] != ("vllm-omni" if role == "image_edit" else "vllm"):
            raise ContractError(f"models.{role}.backend is incompatible")
        port = model["port"]
        if not isinstance(port, int) or isinstance(port, bool) or not 1024 <= port <= 65535:
            raise ContractError(f"models.{role}.port is invalid")
        if port in ports:
            raise ContractError("model ports must be distinct")
        ports.add(port)
        if (
            not isinstance(model["min_vram_mib"], int)
            or isinstance(model["min_vram_mib"], bool)
            or model["min_vram_mib"] < 1
        ):
            raise ContractError(f"models.{role}.min_vram_mib must be positive")
    resources = payload["resources"]
    if not isinstance(resources, dict) or set(resources) != {
        "cpu_pool", "tao_gpu_pool", "image_worker_pool", "coordinator_pool",
        "gpus_per_image_worker", "image_worker_capacity", "vlm_gpus",
        "llm_gpus", "tao_gpus", "image_edit_gpu_ids", "vlm_gpu_ids",
        "llm_gpu_ids", "tao_gpu_ids",
    }:
        raise ContractError("resources has an invalid shape")
    for field in ("cpu_pool", "tao_gpu_pool", "image_worker_pool", "coordinator_pool"):
        _pool_name(resources[field], f"resources.{field}")
    gpu_lists: dict[str, list[int]] = {}
    for field in ("image_edit_gpu_ids", "vlm_gpu_ids", "llm_gpu_ids", "tao_gpu_ids"):
        values = resources[field]
        if (
            not isinstance(values, list) or not values
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values)
            or len(values) != len(set(values))
        ):
            raise ContractError(f"resources.{field} must contain distinct non-negative GPU IDs")
        gpu_lists[field] = values
    image_capacity = len(gpu_lists["image_edit_gpu_ids"])
    if not 1 <= image_capacity <= MAX_IMAGE_WORKER_GPUS:
        raise ContractError("resources.image_edit_gpu_ids must contain 1..8 GPUs")
    expected_resource_values = {
        "gpus_per_image_worker": image_capacity,
        "image_worker_capacity": image_capacity,
        "vlm_gpus": len(gpu_lists["vlm_gpu_ids"]),
        "llm_gpus": len(gpu_lists["llm_gpu_ids"]),
        "tao_gpus": len(gpu_lists["tao_gpu_ids"]),
    }
    for field, expected in expected_resource_values.items():
        if resources.get(field) != expected:
            raise ContractError(f"resources.{field} must be {expected}")
    if resources["vlm_gpus"] != 1 or resources["llm_gpus"] != 1:
        raise ContractError("Airflow VLM and LLM roles require exactly one GPU each")
    if nodes == 1:
        role_sets = [
            set(gpu_lists[field])
            for field in ("image_edit_gpu_ids", "vlm_gpu_ids", "llm_gpu_ids", "tao_gpu_ids")
        ]
        if any(role_sets[left] & role_sets[right]
               for left in range(len(role_sets)) for right in range(left + 1, len(role_sets))):
            raise ContractError("single-host Airflow GPU role selections must be disjoint")
    elif gpu_lists["image_edit_gpu_ids"] != list(range(MAX_IMAGE_WORKER_GPUS)):
        raise ContractError("distributed Airflow image workers require explicit GPU IDs 0..7")
    if models["image_edit"]["port"] + image_capacity - 1 > 65535:
        raise ContractError("image-edit port range exceeds 65535")
    root = pathlib.Path(str(airflow._shared_root()))
    paths = payload["paths"]
    expected_path_keys = {
        "results_dir", "stage_dir", "dataset_root", "config_path", "runtime_root",
        "cache_dir", "mined_pairs", "gaps_parquet", "eval_list", "eval_pairs",
        "attribute_vocab",
    }
    if not isinstance(paths, dict) or set(paths) != expected_path_keys:
        raise ContractError("paths has an invalid shape")
    normalized = {field: _under(_absolute(value, f"paths.{field}"), root, f"paths.{field}")
                  for field, value in paths.items()}
    stage = normalized["results_dir"] / f"iter_{payload['iteration']}" / "datagen"
    if normalized["stage_dir"] != stage:
        raise ContractError("paths.stage_dir is not the canonical iteration datagen path")
    if normalized["config_path"] != normalized["results_dir"] / "config" / "sdg_config.yaml":
        raise ContractError("paths.config_path is not canonical")
    expected_outputs = [
        str(stage / "dataset" / "sdg_manifest.json"),
        str(stage / "dataset" / "sdg_pairs.json"),
        str(stage / "dataset" / "sdg_image_list.txt"),
        str(stage / "sdg_execution_manifest.json"),
        str(stage / "endpoint_pool.json"),
        str(stage / "endpoint_manifest.json"),
    ]
    if payload["expected_outputs"] != expected_outputs:
        raise ContractError("expected_outputs must be the six canonical SDG artifacts")
    bindings = payload["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != {
        "state_sha256", "config_sha256", "runtime_sha256",
    } or any(not isinstance(value, str) or SHA256.fullmatch(value) is None
             for value in bindings.values()):
        raise ContractError("bindings must contain three SHA-256 digests")
    limits = payload["limits"]
    if not isinstance(limits, dict) or set(limits) != {
        "startup_timeout_s", "retry_interval_s", "request_timeout_s",
        "image_edit_request_timeout_s", "verification_max_attempts",
        "component_max_attempts", "max_samples_per_iteration", "dag_timeout_s",
    }:
        raise ContractError("limits has an invalid shape")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
           for value in limits.values()):
        raise ContractError("every SDG limit must be a positive integer")
    if limits["verification_max_attempts"] > 5 or limits["component_max_attempts"] > 2:
        raise ContractError("SDG retry limits exceed the bounded contract")
    if payload["forward_env"] not in ([], ["HF_TOKEN"]):
        raise ContractError("forward_env may contain only HF_TOKEN")
    binding = _absolute(payload["job_binding_path"], "job_binding_path")
    if binding != stage / "airflow-sdg.job-binding.json":
        raise ContractError("job_binding_path is not canonical")
    airflow._reject_secret_material(payload, "Airflow SDG request")
    return payload


def load_request(path: pathlib.Path) -> tuple[pathlib.Path, dict[str, Any]]:
    resolved, payload = airflow._regular_json(path, "Airflow SDG request")
    return resolved, validate_request(payload)


def prepare_request(args: argparse.Namespace) -> dict[str, Any]:
    state_path = _regular(args.deft_state, "--deft-state")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != "3"
        or state.get("workflow") != WORKFLOW
    ):
        raise ContractError("--deft-state is not initialized IAA schema-v3 state")
    results = _absolute(state.get("results_dir"), "state.results_dir")
    if state_path != results / "deft_state.json":
        raise ContractError("--deft-state is not the canonical state path")
    config_state = state.get("config")
    if not isinstance(config_state, dict):
        raise ContractError("initialized DEFT config is missing")
    compute_platform = config_state.get("platform")
    orchestrator = config_state.get("orchestrator")
    if not (
        (compute_platform == "airflow" and orchestrator is None)
        or (compute_platform in {"docker", "virtualenv"} and orchestrator == "airflow")
    ):
        raise ContractError(
            "Airflow local SDG preparation requires legacy platform=airflow or "
            "orchestrator=airflow with platform=docker|virtualenv; remote compute "
            "uses its canonical platform SDG request"
        )
    if state.get("gate_met") is True or state.get("loop_stop_reason") is not None:
        raise ContractError("stopped run cannot prepare SDG")
    iteration = args.iteration
    label = f"iter{iteration}"
    phase = (state.get("iterations") or {}).get(label)
    if (
        state.get("current_iteration") != iteration
        or not isinstance(phase, dict)
        or phase.get("status") != "in_progress"
        or phase.get("stage_completed") != "history_select"
    ):
        raise ContractError("current iteration must have committed history_select")
    config_path = _regular(args.sdg_config, "--sdg-config")
    if config_path != results / "config" / "sdg_config.yaml":
        raise ContractError("--sdg-config is not canonical")
    config_digest = _file_sha256(config_path)
    if (
        config_state.get("sdg_config") != str(config_path)
        or config_state.get("sdg_config_sha256") != config_digest
        or (config_state.get("spec_sha256") or {}).get("sdg_config.yaml") != config_digest
    ):
        raise ContractError("SDG config is not hash-bound by state")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != "1" or config.get("enabled") is not True:
        raise ContractError("SDG config must be enabled schema-v1")
    endpoints = config.get("endpoints")
    generation = config.get("generation")
    models = config.get("models")
    config_images = config.get("images")
    if not all(isinstance(value, dict) for value in (endpoints, generation, models, config_images)):
        raise ContractError("SDG config is incomplete")
    if set(models) != set(ROLES) or set(config_images) != {
        "augmentation", "auto_labeling", "image_edit_serving", "text_serving",
    }:
        raise ContractError("SDG config model/image roles are incomplete or unexpected")
    if endpoints.get("ownership") != "managed" or endpoints.get("reuse_requested") is not False:
        raise ContractError("Airflow managed SDG requires owned non-reused endpoints")
    nodes = generation.get("generation_nodes")
    endpoint_gpu_ids = endpoints.get("gpu_ids")
    gpus_per_node = generation.get("gpus_per_generation_node")
    tao_gpu_ids = config_state.get("gpu_ids")
    if (
        not isinstance(nodes, int)
        or isinstance(nodes, bool)
        or not 1 <= nodes <= 64
        or not isinstance(gpus_per_node, int)
        or isinstance(gpus_per_node, bool)
        or not 1 <= gpus_per_node <= MAX_IMAGE_WORKER_GPUS
        or not isinstance(endpoint_gpu_ids, dict)
        or set(endpoint_gpu_ids) != set(ROLES)
        or endpoint_gpu_ids.get("image_edit") is None
        or len(endpoint_gpu_ids["image_edit"]) != gpus_per_node
        or len(endpoint_gpu_ids.get("vlm") or []) != 1
        or len(endpoint_gpu_ids.get("llm") or []) != 1
        or not isinstance(tao_gpu_ids, list)
        or not tao_gpu_ids
    ):
        raise ContractError("Airflow SDG GPU topology is incomplete or inconsistent")
    if nodes == 1:
        role_sets = [set(endpoint_gpu_ids[role]) for role in ROLES] + [set(tao_gpu_ids)]
        if any(role_sets[left] & role_sets[right]
               for left in range(len(role_sets)) for right in range(left + 1, len(role_sets))):
            raise ContractError("single-host Airflow GPU role selections must be disjoint")
    elif endpoint_gpu_ids["image_edit"] != list(range(MAX_IMAGE_WORKER_GPUS)) or gpus_per_node != 8:
        raise ContractError("distributed Airflow requires N independent 8-GPU image workers")
    approved = config_state.get("sdg")
    if (
        not isinstance(approved, dict)
        or approved.get("endpoint_mode") != "managed"
        or approved.get("generation_nodes") != nodes
        or approved.get("gpus_per_generation_node") != gpus_per_node
        or approved.get("gpu_ids") != endpoint_gpu_ids
        or approved.get("models") != models
        or approved.get("images") != config_images
    ):
        raise ContractError("state.config.sdg differs from immutable SDG config")
    images = {
        "augmentation": config_images.get("augmentation"),
        "auto_labeling": config_images.get("auto_labeling"),
        "image_edit": config_images.get("image_edit_serving"),
        "text_serving": config_images.get("text_serving"),
        "controller": config_state.get("ds_image"),
    }
    prepared_models = {
        role: {field: models[role].get(field) for field in (
            "id", "revision", "backend", "port", "min_vram_mib",
        )}
        for role in ROLES
    }
    shared_root = pathlib.Path(str(airflow._shared_root()))
    results = _under(results, shared_root, "state.results_dir")
    dataset = _under(
        _absolute(config_state.get("dataset_root"), "state.config.dataset_root"),
        shared_root,
        "state.config.dataset_root",
    )
    stage = results / f"iter_{iteration}" / "datagen"
    runtime_root = _under(_directory(args.runtime_root, "--runtime-root"), shared_root, "--runtime-root")
    if runtime_root != stage / ".tao-runtime" / "controller":
        raise ContractError("--runtime-root is not the canonical staged controller path")
    runtime_digest = state.get(
        "active_runtime_sha256", config_state.get("iaa_deft_bundle_sha256")
    )
    if not isinstance(runtime_digest, str) or SHA256.fullmatch(runtime_digest) is None:
        raise ContractError("state does not bind the active runtime")
    if _python_tree_sha256(runtime_root / "iaa_deft") != runtime_digest:
        raise ContractError("staged Airflow IAA runtime differs from active state provenance")
    paths = {
        "results_dir": str(results),
        "stage_dir": str(stage),
        "dataset_root": str(dataset),
        "config_path": str(config_path),
        "runtime_root": str(runtime_root),
        "cache_dir": str(_under(_absolute(endpoints.get("cache_dir"), "cache_dir"), shared_root, "cache_dir")),
        "mined_pairs": str(_under(_absolute(phase.get("mined_pairs"), "mined_pairs"), shared_root, "mined_pairs")),
        "gaps_parquet": str(results / f"iter_{iteration}" / "gaps" / "kpi_gaps.parquet"),
        "eval_list": str(results / "iaa_splits" / "eval_list.txt"),
        "eval_pairs": str(results / "iaa_splits" / "eval_pairs.json"),
        "attribute_vocab": str(dataset / "attribute_vocab.json"),
    }
    for field in ("mined_pairs", "gaps_parquet", "eval_list", "eval_pairs", "attribute_vocab"):
        _regular(pathlib.Path(paths[field]), f"paths.{field}")
    started = dt.datetime.fromisoformat(str(state.get("started_at", "")).replace("Z", "+00:00"))
    if started.utcoffset() is None:
        raise ContractError("state.started_at must be timezone-aware")
    started = started.astimezone(dt.timezone.utc)
    epoch = started - dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    started_ns = ((epoch.days * 86400 + epoch.seconds) * 1_000_000_000
                  + epoch.microseconds * 1000 + iteration)
    identity = {
        "state_sha256": _file_sha256(state_path),
        "config_sha256": config_digest,
        "runtime_sha256": runtime_digest,
        "iteration": iteration,
        "resources": {
            "cpu_pool": args.cpu_pool,
            "tao_gpu_pool": args.tao_gpu_pool,
            "image_worker_pool": args.image_worker_pool,
            "coordinator_pool": args.coordinator_pool,
        },
    }
    action_id = "deft-iaa-sdg-" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    expected_outputs = [
        str(stage / "dataset" / "sdg_manifest.json"),
        str(stage / "dataset" / "sdg_pairs.json"),
        str(stage / "dataset" / "sdg_image_list.txt"),
        str(stage / "sdg_execution_manifest.json"),
        str(stage / "endpoint_pool.json"),
        str(stage / "endpoint_manifest.json"),
    ]
    payload = {
        "schema_version": "1", "workflow": WORKFLOW, "kind": KIND,
        "platform": compute_platform, "name": NAME, "contract": airflow.CONTRACT,
        "action_id": action_id, "run_id": results.name, "iteration": iteration,
        "attempt": 1,
        "started_at": started.isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "started_ns": started_ns, "generation_nodes": nodes, "images": images,
        "models": prepared_models,
        "resources": {
            "cpu_pool": _pool_name(args.cpu_pool, "--cpu-pool"),
            "tao_gpu_pool": _pool_name(args.tao_gpu_pool, "--tao-gpu-pool"),
            "image_worker_pool": _pool_name(args.image_worker_pool, "--image-worker-pool"),
            "coordinator_pool": _pool_name(args.coordinator_pool, "--coordinator-pool"),
            "gpus_per_image_worker": gpus_per_node,
            "image_worker_capacity": gpus_per_node,
            "vlm_gpus": 1, "llm_gpus": 1,
            "tao_gpus": len(tao_gpu_ids),
            "image_edit_gpu_ids": endpoint_gpu_ids["image_edit"],
            "vlm_gpu_ids": endpoint_gpu_ids["vlm"],
            "llm_gpu_ids": endpoint_gpu_ids["llm"],
            "tao_gpu_ids": tao_gpu_ids,
        },
        "paths": paths,
        "limits": {
            "startup_timeout_s": endpoints.get("startup_timeout_s"),
            "retry_interval_s": endpoints.get("retry_interval_s"),
            "request_timeout_s": endpoints.get("request_timeout_s"),
            "image_edit_request_timeout_s": generation.get("image_edit_request_timeout_s"),
            "verification_max_attempts": generation.get("verification_max_attempts"),
            "component_max_attempts": 2,
            "max_samples_per_iteration": generation.get("max_samples_per_iteration"),
            "dag_timeout_s": args.dag_timeout_s,
        },
        "bindings": {
            "state_sha256": identity["state_sha256"],
            "config_sha256": config_digest,
            "runtime_sha256": runtime_digest,
        },
        "forward_env": ["HF_TOKEN"] if config_state.get("requires_hf_token") else [],
        "expected_outputs": expected_outputs,
        "job_binding_path": str(stage / "airflow-sdg.job-binding.json"),
        "request_sha256": "0" * 64,
    }
    if orchestrator == "airflow":
        payload["orchestrator"] = "airflow"
    payload["request_sha256"] = _canonical_sha256(payload)
    payload = validate_request(payload)
    output = pathlib.Path(os.path.abspath(args.output.expanduser()))
    expected_output = stage / "airflow_sdg.action.json"
    if output != expected_output:
        raise ContractError(f"--output must be canonical {expected_output}")
    if output.exists():
        _, existing = load_request(output)
        if existing != payload:
            raise ContractError("refusing to replace a different existing Airflow SDG request")
        disposition = "reused"
    else:
        _atomic_json(output, payload)
        disposition = "created"
    return {"status": disposition, "request": str(output), "payload": payload}


def _load_job(path: pathlib.Path, request: dict[str, Any]) -> tuple[pathlib.Path, dict[str, Any]]:
    resolved, job = airflow._regular_json(path, "Airflow SDG job record")
    expected = {
        "platform": request["platform"],
        "image": request["images"]["controller"],
        "network_arch": "iaa-sdg",
        "action": request["action_id"],
        "results_dir": request["paths"]["stage_dir"],
    }
    for field, value in expected.items():
        if job.get(field) != value:
            raise ContractError(f"job record {field} differs from Airflow SDG request")
    transitions = job.get("transitions")
    if (
        job.get("backend_ref") is not None
        or job.get("terminal_state") is not None
        or not isinstance(transitions, list)
        or len(transitions) != 1
        or not isinstance(transitions[0], dict)
        or transitions[0].get("state") != "PENDING"
    ):
        raise ContractError("submit requires one PENDING unlaunched SDG job record")
    job_state_raw = os.environ.get("TAO_STATE_DIR")
    job_state = pathlib.Path(job_state_raw) if job_state_raw else pathlib.Path.home() / ".tao"
    job_state = pathlib.Path(os.path.abspath(job_state.expanduser()))
    expected_job_path = job_state / "jobs" / f"{job.get('id')}.json"
    if resolved != expected_job_path:
        raise ContractError("job record is outside the request-owned state directory")
    airflow._reject_secret_material(job, "Airflow SDG job record")
    return resolved, job


def _bind_job(
    request_path: pathlib.Path, request: dict[str, Any], job_path: pathlib.Path,
    job: dict[str, Any],
) -> dict[str, Any]:
    path = pathlib.Path(request["job_binding_path"])
    payload = {
        "schema_version": "1", "workflow": WORKFLOW, "platform": request["platform"],
        "request_path": str(request_path), "request_sha256": request["request_sha256"],
        "job_record_path": str(job_path), "job_id": job["id"],
        "job_identity_sha256": hashlib.sha256(
            json.dumps(job, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "results_scope": request["paths"]["stage_dir"],
        "staging_receipt_sha256": None,
        "bound_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    payload["binding_sha256"] = airflow._canonical_sha256(payload, "binding_sha256")
    if path.exists():
        _, existing = airflow._regular_json(path, "Airflow SDG job binding")
        if set(existing) != set(payload):
            raise ContractError("existing Airflow SDG job binding has an invalid shape")
        stable_fields = set(payload) - {"bound_at", "binding_sha256"}
        if any(existing.get(field) != payload[field] for field in stable_fields):
            raise ContractError("existing Airflow SDG job binding differs from request/job")
        digest = existing.get("binding_sha256")
        if digest != airflow._canonical_sha256(existing, "binding_sha256"):
            raise ContractError("existing Airflow SDG job binding digest is invalid")
        return existing
    _atomic_json(path, payload)
    return payload


def submit(args: argparse.Namespace) -> int:
    request_path, request = load_request(args.request)
    launch_lock = pathlib.Path(request["paths"]["stage_dir"]) / "airflow-sdg.launch.lock"
    with _exclusive_lock(launch_lock):
        job_path, job = _load_job(args.job_record, request)
        binding = _bind_job(request_path, request, job_path, job)
        client = airflow.AirflowClient()
        dag_id = airflow._dag_id()
        airflow.validate_dag(client, dag_id)
        resources = request["resources"]
        airflow.validate_pools(client, [
            f"{resources['cpu_pool']}:1",
            f"{resources['tao_gpu_pool']}:1",
            f"{resources['image_worker_pool']}:{request['generation_nodes']}",
            f"{resources['coordinator_pool']}:1",
        ])
        run_id = str(job["id"])
        conf = {
            "contract": airflow.CONTRACT,
            "kind": KIND,
            "job_id": run_id,
            "request_sha256": request["request_sha256"],
            "binding_sha256": binding["binding_sha256"],
            "request": request,
        }
        airflow._reject_secret_material(conf, "Airflow SDG DAG conf")
        path = f"/api/v2/dags/{urllib.parse.quote(dag_id, safe='')}/dagRuns"
        try:
            response = client._request(
                "POST", path,
                {"dag_run_id": run_id, "logical_date": None, "conf": conf},
            )
            reconciled = False
        except airflow.AirflowApiError as exc:
            if exc.status != 409:
                raise
            response = client.dag_run(dag_id, run_id)
            existing = response.get("conf")
            if not isinstance(existing, dict) or any(
                existing.get(field) != conf[field]
                for field in ("contract", "kind", "job_id", "request_sha256", "binding_sha256")
            ):
                raise ContractError("existing DAG run differs from the bound Airflow SDG request")
            reconciled = True
    native = response.get("state") if isinstance(response, dict) else None
    print(json.dumps({
        "backend_ref": airflow._backend_ref(dag_id, run_id),
        "status": airflow.map_state(native), "native_state": native,
        "reconciled": reconciled, "job_binding": request["job_binding_path"],
    }, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="verb", required=True)
    prepare = sub.add_parser("prepare-request")
    prepare.add_argument("--deft-state", required=True, type=pathlib.Path)
    prepare.add_argument("--sdg-config", required=True, type=pathlib.Path)
    prepare.add_argument("--iteration", required=True, type=int)
    prepare.add_argument("--runtime-root", required=True, type=pathlib.Path)
    prepare.add_argument("--cpu-pool", required=True)
    prepare.add_argument("--tao-gpu-pool", required=True)
    prepare.add_argument("--image-worker-pool", required=True)
    prepare.add_argument("--coordinator-pool", required=True)
    prepare.add_argument("--dag-timeout-s", type=int, default=14400)
    prepare.add_argument("--output", required=True, type=pathlib.Path)
    submit_parser = sub.add_parser("submit")
    submit_parser.add_argument("--request", required=True, type=pathlib.Path)
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
        if args.verb == "prepare-request":
            prepared = prepare_request(args)
            print(json.dumps({
                "status": prepared["status"], "request": prepared["request"],
                "action_id": prepared["payload"]["action_id"],
                "request_sha256": prepared["payload"]["request_sha256"],
                "generation_nodes": prepared["payload"]["generation_nodes"],
            }, sort_keys=True))
            return 0
        if args.verb == "submit":
            return submit(args)
        return {
            "status": airflow.status,
            "logs": airflow.logs,
            "cancel": airflow.cancel,
        }[args.verb](args)
    except (
        airflow.AirflowApiError, airflow.AirflowContractError,
        ContractError, OSError, ValueError, yaml.YAMLError,
    ) as exc:
        print(f"airflow SDG action failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
