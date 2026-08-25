#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes four-verb consumer for the composite IAA SDG action.

This is deliberately separate from distributed training.  It renders one
Indexed image-worker Job (eight TP=1 services per pod), one headless Service,
and one two-GPU coordinator Job.  Application logic remains in the staged
shared IAA runtime.
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
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any, Sequence

import yaml


WORKFLOW = "tao-run-deft-iaa"
KIND = "kubernetes_sdg_action"
NAME = "sdg_execute"
IMAGE_SERVICES_PER_WORKER = 8
SHA256 = re.compile(r"[0-9a-f]{64}")
REVISION = re.compile(r"[0-9a-f]{40}")
DNS_LABEL = re.compile(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
ROLES = ("image_edit", "vlm", "llm")
IMAGE_KEYS = {"augmentation", "auto_labeling", "image_edit", "text_serving", "controller"}
COMPONENT_SOURCE_KEYS = {"augmentation", "auto_labeling", "image_edit", "text_serving"}
EXPECTED_KEYS = {
    "schema_version", "workflow", "kind", "platform", "name", "action_id",
    "run_id", "iteration", "attempt", "started_at", "started_ns",
    "generation_nodes", "namespace", "pvc_claim", "pvc_mount",
    "service_account", "images", "component_sources", "models", "bindings",
    "paths", "limits", "forward_env",
    "expected_outputs", "request_sha256",
}
OWNED_LABELS = {
    "app.kubernetes.io/managed-by": "tao-skill-bank",
    "tao.nvidia.com/workflow": WORKFLOW,
    "tao.nvidia.com/action-kind": KIND,
}


class ContractError(ValueError):
    pass


def _canonical_sha256(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("request_sha256", None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _dns(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) > 63 or DNS_LABEL.fullmatch(value) is None:
        raise ContractError(f"{field} must be a Kubernetes DNS label")
    return value


def _absolute(value: Any, field: str) -> pathlib.PurePosixPath:
    if not isinstance(value, str) or not value.startswith("/") or "\x00" in value:
        raise ContractError(f"{field} must be an absolute POSIX path")
    path = pathlib.PurePosixPath(value)
    if path == pathlib.PurePosixPath("/") or ".." in path.parts or str(path) != value:
        raise ContractError(f"{field} must be normalized, non-root, and traversal-free")
    return path


def _exact_mapping(value: Any, field: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ContractError(f"{field} must contain exactly {sorted(keys)}")
    return value


def validate_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != EXPECTED_KEYS:
        raise ContractError("Kubernetes SDG request has missing or unexpected fields")
    fixed = {
        "schema_version": "1", "workflow": WORKFLOW, "kind": KIND,
        "platform": "kubernetes", "name": NAME,
    }
    for key, value in fixed.items():
        if payload.get(key) != value:
            raise ContractError(f"request.{key} must be {value!r}")
    if not isinstance(payload.get("request_sha256"), str) or SHA256.fullmatch(payload["request_sha256"]) is None:
        raise ContractError("request_sha256 must be lowercase SHA-256")
    if payload["request_sha256"] != _canonical_sha256(payload):
        raise ContractError("request_sha256 does not match immutable content")
    for key in ("action_id", "run_id"):
        if not isinstance(payload[key], str) or SAFE_ID.fullmatch(payload[key]) is None:
            raise ContractError(f"request.{key} contains unsupported characters")
    try:
        timestamp = dt.datetime.fromisoformat(payload["started_at"].replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ContractError("started_at must be timezone-aware ISO-8601") from exc
    if timestamp.utcoffset() is None:
        raise ContractError("started_at must include a timezone")
    if not isinstance(payload["started_ns"], int) or isinstance(payload["started_ns"], bool) or payload["started_ns"] < 1:
        raise ContractError("started_ns must be a positive integer")
    if not isinstance(payload["generation_nodes"], int) or isinstance(payload["generation_nodes"], bool) or not 1 <= payload["generation_nodes"] <= 64:
        raise ContractError("generation_nodes must be in [1, 64]")
    if not isinstance(payload["iteration"], int) or isinstance(payload["iteration"], bool) or payload["iteration"] < 1:
        raise ContractError("iteration must be positive")
    if payload["attempt"] not in (1, 2):
        raise ContractError("attempt must be 1 or 2")
    _dns(payload["namespace"], "namespace")
    _dns(payload["pvc_claim"], "pvc_claim")
    _dns(payload["service_account"], "service_account")
    mount = _absolute(payload["pvc_mount"], "pvc_mount")

    images = _exact_mapping(payload["images"], "images", IMAGE_KEYS)
    for key, image in images.items():
        if not isinstance(image, str) or re.fullmatch(r"[^\s@]+(?::[^\s@]+)?@sha256:[0-9a-f]{64}", image) is None:
            raise ContractError(f"images.{key} must be digest pinned")
    component_sources = _exact_mapping(
        payload["component_sources"], "component_sources", COMPONENT_SOURCE_KEYS,
    )
    for key, source in component_sources.items():
        if source != images[key]:
            raise ContractError(
                f"component_sources.{key} must bind the immutable deployed image"
            )
    bindings = _exact_mapping(
        payload["bindings"], "bindings",
        {"state_sha256", "config_sha256", "runtime_sha256"},
    )
    for key, digest in bindings.items():
        if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
            raise ContractError(f"bindings.{key} must be lowercase SHA-256")
    models = _exact_mapping(payload["models"], "models", set(ROLES))
    ports = set()
    for role in ROLES:
        model = _exact_mapping(
            models[role], f"models.{role}", {"id", "revision", "backend", "port"},
        )
        if not isinstance(model["id"], str) or not model["id"]:
            raise ContractError(f"models.{role}.id must be non-empty")
        if not isinstance(model["revision"], str) or REVISION.fullmatch(model["revision"]) is None:
            raise ContractError(f"models.{role}.revision must be immutable")
        expected_backend = "vllm-omni" if role == "image_edit" else "vllm"
        if model["backend"] != expected_backend:
            raise ContractError(f"models.{role}.backend must be {expected_backend}")
        port = model["port"]
        if not isinstance(port, int) or isinstance(port, bool) or not 1024 <= port <= 65535 or port in ports:
            raise ContractError(f"models.{role}.port must be unique and valid")
        ports.add(port)
    if models["image_edit"]["port"] + 7 > 65535:
        raise ContractError("image-edit base port must leave eight consecutive ports")

    paths = _exact_mapping(
        payload["paths"], "paths",
        {"results_dir", "stage_dir", "dataset_root", "config_path", "runtime_root", "cache_dir",
         "mined_pairs", "eval_list", "attribute_vocab"},
    )
    normalized = {key: _absolute(value, f"paths.{key}") for key, value in paths.items()}
    if normalized["stage_dir"] != normalized["results_dir"] / f"iter_{payload['iteration']}" / "datagen":
        raise ContractError("paths.stage_dir is not the canonical iteration datagen directory")
    for key, path in normalized.items():
        try:
            path.relative_to(mount)
        except ValueError as exc:
            raise ContractError(f"paths.{key} must be visible through the shared PVC mount") from exc
    limits = _exact_mapping(
        payload["limits"], "limits",
        {"startup_timeout_s", "retry_interval_s", "request_timeout_s", "component_max_attempts", "ttl_seconds"},
    )
    for key in ("startup_timeout_s", "retry_interval_s", "request_timeout_s", "ttl_seconds"):
        if not isinstance(limits[key], int) or isinstance(limits[key], bool) or limits[key] < 1:
            raise ContractError(f"limits.{key} must be positive")
    if limits["component_max_attempts"] not in (1, 2):
        raise ContractError("limits.component_max_attempts must be 1 or 2")
    if payload["forward_env"] not in ([], ["HF_TOKEN"]):
        raise ContractError("forward_env may contain only HF_TOKEN")
    expected_outputs = [
        str(normalized["stage_dir"] / "dataset" / "sdg_manifest.json"),
        str(normalized["stage_dir"] / "dataset" / "sdg_pairs.json"),
        str(normalized["stage_dir"] / "dataset" / "sdg_image_list.txt"),
        str(normalized["stage_dir"] / "sdg_execution_manifest.json"),
        str(normalized["stage_dir"] / "endpoint_pool.json"),
        str(normalized["stage_dir"] / "endpoint_manifest.json"),
    ]
    if payload["expected_outputs"] != expected_outputs:
        raise ContractError("expected_outputs must be the six canonical SDG artifacts")
    return payload


def load_request(path: pathlib.Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ContractError("--request must be a regular non-symlink file")
    return validate_request(json.loads(resolved.read_text()))


def sign_request(unsigned: dict[str, Any]) -> dict[str, Any]:
    if "request_sha256" in unsigned:
        raise ContractError("unsigned prepare input must omit request_sha256")
    payload = {**unsigned, "request_sha256": "0" * 64}
    payload["request_sha256"] = _canonical_sha256(payload)
    return validate_request(payload)


def _local_regular(path: pathlib.Path, field: str) -> pathlib.Path:
    lexical = path.expanduser().absolute()
    if lexical != pathlib.Path(os.path.abspath(lexical)):
        raise ContractError(f"{field} must not contain lexical traversal")
    if not lexical.is_file() or lexical.is_symlink() or lexical.resolve() != lexical:
        raise ContractError(f"{field} must be an absolute regular non-symlink file")
    return lexical


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _under_mount(path: pathlib.Path, mount: pathlib.Path, field: str) -> pathlib.Path:
    lexical = path.expanduser().absolute()
    if lexical != pathlib.Path(os.path.abspath(lexical)):
        raise ContractError(f"{field} must be normalized and traversal-free")
    try:
        lexical.relative_to(mount)
    except ValueError as exc:
        raise ContractError(f"{field} must be visible through --pvc-mount") from exc
    return lexical


def _pinned_image(value: Any, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(
        r"[^\s@]+(?::[^\s@]+)?@sha256:[0-9a-f]{64}", value,
    ) is None:
        raise ContractError(f"{field} must be one digest-pinned image")
    return value


def prepare_request(args: argparse.Namespace) -> dict[str, Any]:
    """Derive one byte-stable request from committed state and SDG config."""
    state_path = _local_regular(args.deft_state, "--deft-state")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (not isinstance(state, dict) or state.get("schema_version") != "3"
            or state.get("workflow") != WORKFLOW):
        raise ContractError("--deft-state is not initialized IAA DEFT schema-v3 state")
    results = pathlib.Path(str(state.get("results_dir", ""))).expanduser().absolute()
    if state_path != results / "deft_state.json":
        raise ContractError("--deft-state must be canonical results_dir/deft_state.json")
    config_state = state.get("config")
    if not isinstance(config_state, dict) or config_state.get("platform") != "kubernetes":
        raise ContractError("initialized DEFT state platform must be kubernetes")
    if state.get("gate_met") is True or state.get("loop_stop_reason") is not None:
        raise ContractError("run is already stopped and cannot prepare another SDG action")
    if (not isinstance(args.iteration, int) or isinstance(args.iteration, bool)
            or args.iteration < 1 or args.iteration > state.get("max_iterations", 0)):
        raise ContractError("--iteration is outside the initialized run budget")
    if state.get("current_iteration") != args.iteration:
        raise ContractError("--iteration must equal state.current_iteration")
    label = f"iter{args.iteration}"
    phase = (state.get("iterations") or {}).get(label)
    if not isinstance(phase, dict):
        raise ContractError(f"state has no initialized {label} phase")
    completed = phase.get("stage_completed")
    if completed != "history_select" or phase.get("status") != "in_progress":
        if completed in {"sdg", "visualize", "train", "evaluate", "gap_analysis"}:
            raise ContractError(f"{label}/sdg is already committed")
        raise ContractError(f"{label} must have committed history_select before SDG preparation")

    config_path = _local_regular(args.sdg_config, "--sdg-config")
    if config_path != results / "config" / "sdg_config.yaml":
        raise ContractError("--sdg-config must be canonical results_dir/config/sdg_config.yaml")
    if pathlib.Path(str(config_state.get("sdg_config", ""))).expanduser().absolute() != config_path:
        raise ContractError("state.config.sdg_config does not bind --sdg-config")
    config_sha256 = _file_sha256(config_path)
    recorded = config_state.get("sdg_config_sha256")
    specs = config_state.get("spec_sha256")
    if (recorded != config_sha256 or not isinstance(specs, dict)
            or specs.get("sdg_config.yaml") != config_sha256):
        raise ContractError("approved SDG config hashes do not match initialized state")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != "1" or config.get("enabled") is not True:
        raise ContractError("immutable SDG config must be enabled schema-v1")
    endpoints, generation = config.get("endpoints"), config.get("generation")
    models, config_images = config.get("models"), config.get("images")
    if not all(isinstance(item, dict) for item in (endpoints, generation, models, config_images)):
        raise ContractError("immutable SDG config lacks endpoint, generation, model, or image contracts")
    if endpoints.get("ownership") != "managed" or endpoints.get("reuse_requested") is not False:
        raise ContractError("Kubernetes preparation requires managed, non-reused endpoints")
    generation_nodes = generation.get("generation_nodes")
    if (not isinstance(generation_nodes, int) or isinstance(generation_nodes, bool)
            or not 1 <= generation_nodes <= 64
            or generation.get("gpus_per_generation_node") != IMAGE_SERVICES_PER_WORKER):
        raise ContractError("SDG config must bind 1..64 independent eight-GPU generation nodes")
    if endpoints.get("gpu_ids") != {
        "image_edit": list(range(8)), "vlm": [0], "llm": [1],
    }:
        raise ContractError("Kubernetes SDG GPU roles must be image_edit=0..7, vlm=[0], llm=[1]")
    approved = config_state.get("sdg")
    if (not isinstance(approved, dict) or approved.get("endpoint_mode") != "managed"
            or approved.get("reuse_requested") is not False
            or approved.get("generation_nodes") != generation_nodes
            or approved.get("gpus_per_generation_node") != 8
            or approved.get("gpu_ids") != endpoints.get("gpu_ids")
            or approved.get("models") != models or approved.get("images") != config_images):
        raise ContractError("state.config.sdg disagrees with immutable sdg_config")

    image_args = {
        "augmentation": args.augmentation_image,
        "auto_labeling": args.auto_labeling_image,
        "image_edit": args.image_edit_image,
        "text_serving": args.text_serving_image,
        "controller": args.controller_image,
    }
    images = {key: _pinned_image(value, f"--{key.replace('_', '-')}-image")
              for key, value in image_args.items()}
    component_sources = {
        "augmentation": config_images.get("augmentation"),
        "auto_labeling": config_images.get("auto_labeling"),
        "image_edit": config_images.get("image_edit_serving"),
        "text_serving": config_images.get("text_serving"),
    }
    for key, source in component_sources.items():
        _pinned_image(source, f"sdg_config.images.{key}")
        if images[key] != source:
            raise ContractError(f"--{key.replace('_', '-')}-image differs from immutable SDG config")

    prepared_models: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        source = models.get(role)
        if not isinstance(source, dict):
            raise ContractError(f"sdg_config.models.{role} is missing")
        prepared_models[role] = {
            key: source.get(key) for key in ("id", "revision", "backend", "port")
        }

    mount = pathlib.Path(str(_absolute(args.pvc_mount, "--pvc-mount")))
    results = _under_mount(results, mount, "state.results_dir")
    dataset = _under_mount(
        pathlib.Path(str(config_state.get("dataset_root", ""))), mount,
        "state.config.dataset_root",
    )
    stage = results / f"iter_{args.iteration}" / "datagen"
    runtime_root = _under_mount(args.runtime_root, mount, "--runtime-root")
    expected_runtime = stage / ".tao-runtime" / "controller"
    if runtime_root != expected_runtime:
        raise ContractError(f"--runtime-root must be canonical {expected_runtime}")
    cache_value = endpoints.get("cache_dir")
    cache_dir = _under_mount(pathlib.Path(str(cache_value)), mount, "sdg_config.endpoints.cache_dir")
    mined_pairs = _under_mount(
        pathlib.Path(str(phase.get("mined_pairs", ""))), mount,
        f"state.iterations.{label}.mined_pairs",
    )
    if mined_pairs != results / f"iter_{args.iteration}" / "mining" / "mined_pairs.json":
        raise ContractError("committed mined_pairs path is not canonical")
    eval_list = results / "iaa_splits" / "eval_list.txt"
    attribute_vocab = dataset / "attribute_vocab.json"
    for artifact, field in (
        (mined_pairs, "committed mined_pairs"),
        (eval_list, "canonical eval_list"),
        (attribute_vocab, "canonical attribute_vocab"),
    ):
        if (not artifact.is_file() or artifact.is_symlink()
                or artifact.stat().st_size == 0):
            raise ContractError(f"{field} must be an existing non-empty regular file")
    runtime_sha256 = state.get(
        "active_runtime_sha256", config_state.get("iaa_deft_bundle_sha256"),
    )
    if not isinstance(runtime_sha256, str) or SHA256.fullmatch(runtime_sha256) is None:
        raise ContractError("initialized state does not bind an active runtime SHA-256")
    if not isinstance(config_state.get("requires_hf_token"), bool):
        raise ContractError("state.config.requires_hf_token must be boolean")

    try:
        started = dt.datetime.fromisoformat(str(state.get("started_at", "")).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError("state.started_at must be timezone-aware ISO-8601") from exc
    if started.utcoffset() is None:
        raise ContractError("state.started_at must be timezone-aware ISO-8601")
    started_utc = started.astimezone(dt.timezone.utc)
    started_at = started_utc.isoformat(timespec="microseconds").replace("+00:00", "Z")
    epoch = started_utc - dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    started_ns = ((epoch.days * 86400 + epoch.seconds) * 1_000_000_000
                  + epoch.microseconds * 1000 + args.iteration)
    state_sha256 = _file_sha256(state_path)
    bindings = {
        "state_sha256": state_sha256, "config_sha256": config_sha256,
        "runtime_sha256": runtime_sha256,
    }
    identity = {
        **bindings, "iteration": args.iteration, "namespace": args.namespace,
        "pvc_claim": args.pvc_claim, "pvc_mount": str(mount),
        "service_account": args.service_account, "images": images,
        "runtime_root": str(runtime_root),
    }
    identity_sha = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    paths = {
        "results_dir": str(results), "stage_dir": str(stage),
        "dataset_root": str(dataset), "config_path": str(config_path),
        "runtime_root": str(runtime_root), "cache_dir": str(cache_dir),
        "mined_pairs": str(mined_pairs), "eval_list": str(eval_list),
        "attribute_vocab": str(attribute_vocab),
    }
    expected_outputs = [
        str(stage / "dataset" / "sdg_manifest.json"),
        str(stage / "dataset" / "sdg_pairs.json"),
        str(stage / "dataset" / "sdg_image_list.txt"),
        str(stage / "sdg_execution_manifest.json"),
        str(stage / "endpoint_pool.json"), str(stage / "endpoint_manifest.json"),
    ]
    payload = {
        "schema_version": "1", "workflow": WORKFLOW, "kind": KIND,
        "platform": "kubernetes", "name": NAME,
        "action_id": f"deft-iaa-sdg-{identity_sha[:16]}",
        "run_id": results.name, "iteration": args.iteration, "attempt": 1,
        "started_at": started_at, "started_ns": started_ns,
        "generation_nodes": generation_nodes, "namespace": args.namespace,
        "pvc_claim": args.pvc_claim, "pvc_mount": str(mount),
        "service_account": args.service_account, "images": images,
        "component_sources": component_sources, "models": prepared_models,
        "bindings": bindings, "paths": paths,
        "limits": {
            "startup_timeout_s": endpoints.get("startup_timeout_s"),
            "retry_interval_s": endpoints.get("retry_interval_s"),
            "request_timeout_s": endpoints.get("request_timeout_s"),
            "component_max_attempts": 2, "ttl_seconds": args.ttl_seconds,
        },
        "forward_env": ["HF_TOKEN"] if config_state["requires_hf_token"] else [],
        "expected_outputs": expected_outputs, "request_sha256": "0" * 64,
    }
    payload["request_sha256"] = _canonical_sha256(payload)
    payload = validate_request(payload)
    output = args.output.expanduser().absolute()
    if output != stage / "kubernetes_request.json":
        raise ContractError(f"--output must be canonical {stage / 'kubernetes_request.json'}")
    if output.exists():
        if output.is_symlink() or not output.is_file():
            raise ContractError("--output must not replace a symlink or special file")
        existing = validate_request(json.loads(output.read_text(encoding="utf-8")))
        if existing != payload:
            raise ContractError("refusing to replace a different existing prepared request")
        disposition = "reused"
    else:
        _atomic_json(output, payload)
        disposition = "created"
    return {"status": disposition, "output": str(output), "request": payload}


def _slug(request: dict[str, Any]) -> str:
    raw = f"tao-iaa-{request['run_id']}-{request['iteration']}-{request['action_id']}".lower()
    base = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    digest = request["request_sha256"][:8]
    return f"{base[:43].rstrip('-')}-{digest}"


def _labels(request: dict[str, Any], role: str) -> dict[str, str]:
    return {
        **OWNED_LABELS,
        "tao.nvidia.com/action": request["request_sha256"][:16],
        "tao.nvidia.com/run": re.sub(r"[^A-Za-z0-9_.-]", "-", request["run_id"])[:63],
        "tao.nvidia.com/role": role,
    }


def _secret_env(secret_name: str, name: str) -> dict[str, Any]:
    return {"name": name, "valueFrom": {"secretKeyRef": {"name": secret_name, "key": name}}}


def render_resources(request: dict[str, Any]) -> dict[str, Any]:
    request = validate_request(request)
    base = _slug(request)
    namespace, claim, mount = request["namespace"], request["pvc_claim"], request["pvc_mount"]
    worker_name, worker_service = f"{base}-edit", f"{base}-edit"
    coordinator_name, coordinator_service = f"{base}-coord", f"{base}-coord"
    secret_name = f"{base}-auth"
    volume = {"name": "workspace", "persistentVolumeClaim": {"claimName": claim}}
    volume_mount = {"name": "workspace", "mountPath": mount}
    image_model = request["models"]["image_edit"]
    worker_containers = []
    for ordinal in range(IMAGE_SERVICES_PER_WORKER):
        port = image_model["port"] + ordinal
        worker_containers.append({
            "name": f"image-edit-{ordinal}", "image": request["images"]["image_edit"],
            "command": ["vllm", "serve", image_model["id"]],
            "args": ["--omni", "--host", "0.0.0.0", "--port", str(port),
                     "--revision", image_model["revision"], "--served-model-name", image_model["id"],
                     "--tensor-parallel-size", "1"],
            "ports": [{"containerPort": port, "name": f"edit-{ordinal}"}],
            "env": [_secret_env(secret_name, "VLLM_API_KEY")],
            "resources": {"limits": {"nvidia.com/gpu": 1}, "requests": {"nvidia.com/gpu": 1}},
            "volumeMounts": [volume_mount, {"name": "dshm", "mountPath": "/dev/shm"}],
        })
    worker_labels = _labels(request, "image-worker")
    coordinator_labels = _labels(request, "coordinator")
    worker_service_object = {
        "apiVersion": "v1", "kind": "Service",
        "metadata": {"name": worker_service, "namespace": namespace, "labels": worker_labels},
        "spec": {"clusterIP": "None", "publishNotReadyAddresses": True,
                 "selector": worker_labels,
                 "ports": [{"name": f"edit-{i}", "port": image_model["port"] + i,
                            "targetPort": image_model["port"] + i} for i in range(8)]},
    }
    worker_job = {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": worker_name, "namespace": namespace, "labels": worker_labels,
                     "annotations": {"tao.nvidia.com/request-sha256": request["request_sha256"]}},
        "spec": {
            "completionMode": "Indexed", "completions": request["generation_nodes"],
            "parallelism": request["generation_nodes"], "backoffLimitPerIndex": 0,
            "ttlSecondsAfterFinished": request["limits"]["ttl_seconds"],
            "template": {"metadata": {"labels": worker_labels}, "spec": {
                "serviceAccountName": request["service_account"], "restartPolicy": "Never",
                "subdomain": worker_service, "containers": worker_containers,
                "volumes": [volume, {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}}],
            }},
        },
    }
    text_image = request["images"]["text_serving"]
    sidecars = []
    for role in ("vlm", "llm"):
        model = request["models"][role]
        sidecars.append({
            "name": role, "image": text_image,
            "restartPolicy": "Always",
            "args": ["--model", model["id"], "--host", "0.0.0.0", "--port", str(model["port"]),
                     "--revision", model["revision"], "--served-model-name", model["id"],
                     "--tensor-parallel-size", "1"],
            "resources": {"limits": {"nvidia.com/gpu": 1}, "requests": {"nvidia.com/gpu": 1}},
            "ports": [{"containerPort": model["port"], "name": role}],
            "env": [_secret_env(secret_name, "VLLM_API_KEY")],
            "volumeMounts": [volume_mount],
        })
    controller = {
        "name": "controller", "image": request["images"]["controller"],
        "command": ["python3", str(pathlib.PurePosixPath(request["paths"]["runtime_root"]) / "kubernetes_sdg_action.py")],
        "args": ["coordinator", "--request", str(pathlib.PurePosixPath(request["paths"]["stage_dir"]) / "kubernetes_request.json"),
                 "--worker-service", worker_service, "--worker-job", worker_name],
        "env": [_secret_env(secret_name, "IMAGE_EDIT_API_KEY")],
        "volumeMounts": [volume_mount],
    }
    if request["forward_env"]:
        controller["env"].append(_secret_env(secret_name, "HF_TOKEN"))
    coordinator_service_object = {
        "apiVersion": "v1", "kind": "Service",
        "metadata": {"name": coordinator_service, "namespace": namespace, "labels": coordinator_labels},
        "spec": {"selector": coordinator_labels, "ports": [
            {"name": role, "port": request["models"][role]["port"],
             "targetPort": request["models"][role]["port"]} for role in ("vlm", "llm")
        ]},
    }
    coordinator_job = {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": coordinator_name, "namespace": namespace, "labels": coordinator_labels,
                     "annotations": {"tao.nvidia.com/request-sha256": request["request_sha256"]}},
        "spec": {"backoffLimit": 0, "ttlSecondsAfterFinished": request["limits"]["ttl_seconds"],
                 "template": {"metadata": {"labels": coordinator_labels}, "spec": {
                     "serviceAccountName": request["service_account"], "restartPolicy": "Never",
                     "initContainers": sidecars, "containers": [controller], "volumes": [volume],
                 }}},
    }
    return {
        "apiVersion": "v1", "kind": "List",
        "items": [worker_service_object, coordinator_service_object, worker_job, coordinator_job],
        "_tao": {"base": base, "secret": secret_name, "worker_job": worker_name,
                 "coordinator_job": coordinator_name, "namespace": namespace},
    }


def _run(argv: list[str], *, stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(argv, input=stdin, capture_output=True, text=True, check=check)


def _kubectl_json(args: list[str], *, stdin: str | None = None) -> Any:
    result = _run(["kubectl", *args], stdin=stdin)
    return json.loads(result.stdout) if result.stdout.strip() else None


def _pvc_and_capacity_preflight(request: dict[str, Any]) -> None:
    namespace = request["namespace"]
    version = _kubectl_json(["version", "-o", "json"])
    server = version.get("serverVersion", {}) if isinstance(version, dict) else {}
    try:
        major, minor = int(str(server.get("major"))), int(re.sub(r"\D.*", "", str(server.get("minor"))))
    except ValueError as exc:
        raise ContractError("cannot determine Kubernetes server version") from exc
    if (major, minor) < (1, 29):
        raise ContractError("Kubernetes SDG requires v1.29+ native sidecars and Indexed Jobs")
    _kubectl_json(["get", "serviceaccount", request["service_account"], "-n", namespace, "-o", "json"])
    subject = f"system:serviceaccount:{namespace}:{request['service_account']}"
    for verb, resource in (("create", "jobs.batch"), ("get", "pods"), ("delete", "jobs.batch")):
        allowed = _run(["kubectl", "auth", "can-i", verb, resource, "-n", namespace,
                        "--as", subject])
        if allowed.stdout.strip() != "yes":
            raise ContractError(f"service account cannot {verb} {resource}")
    pvc = _kubectl_json(["get", "pvc", request["pvc_claim"], "-n", namespace, "-o", "json"])
    if pvc.get("status", {}).get("phase") != "Bound":
        raise ContractError("shared PVC is not Bound")
    modes = set(pvc.get("status", {}).get("accessModes") or pvc.get("spec", {}).get("accessModes") or [])
    if request["generation_nodes"] > 1 and "ReadWriteMany" not in modes:
        raise ContractError("multi-node SDG requires a ReadWriteMany shared PVC")
    nodes = _kubectl_json(["get", "nodes", "-o", "json"])
    capacities = [int(item.get("status", {}).get("allocatable", {}).get("nvidia.com/gpu", 0))
                  for item in nodes.get("items", [])]
    if len([value for value in capacities if value >= 8]) < request["generation_nodes"]:
        raise ContractError("cluster lacks one eight-GPU schedulable node per image worker")
    if sum(capacities) < request["generation_nodes"] * 8 + 2:
        raise ContractError("cluster allocatable GPU total cannot schedule workers plus coordinator")


def _secret_payload(request: dict[str, Any], secret_name: str) -> dict[str, Any]:
    token = secrets.token_urlsafe(48)
    data = {"VLLM_API_KEY": token, "IMAGE_EDIT_API_KEY": token,
            "VLM_API_KEY": token, "LLM_API_KEY": token}
    if request["forward_env"]:
        if "HF_TOKEN" not in os.environ:
            raise ContractError("HF_TOKEN environment variable is required")
        data["HF_TOKEN"] = os.environ["HF_TOKEN"]
    return {
        "apiVersion": "v1", "kind": "Secret",
        "metadata": {"name": secret_name, "namespace": request["namespace"],
                     "labels": _labels(request, "auth")},
        "type": "Opaque", "stringData": data,
    }


def _owned(resource: dict[str, Any], request: dict[str, Any]) -> bool:
    labels = resource.get("metadata", {}).get("labels") or {}
    return all(labels.get(key) == value for key, value in OWNED_LABELS.items()) and (
        labels.get("tao.nvidia.com/action") == request["request_sha256"][:16]
    )


def submit(request: dict[str, Any]) -> dict[str, Any]:
    _pvc_and_capacity_preflight(request)
    rendered = render_resources(request)
    identity = rendered.pop("_tao")
    selector = f"tao.nvidia.com/action={request['request_sha256'][:16]}"
    existing = _kubectl_json(["get", "job,service,secret", "-n", request["namespace"], "-l", selector, "-o", "json"])
    items = existing.get("items", []) if isinstance(existing, dict) else []
    if items:
        if any(not _owned(item, request) for item in items):
            raise ContractError("resume discovered mixed or foreign Kubernetes ownership")
        expected = {
            ("Job", identity["worker_job"]), ("Job", identity["coordinator_job"]),
            ("Service", identity["worker_job"]), ("Service", identity["coordinator_job"]),
            ("Secret", identity["secret"]),
        }
        actual = {(item.get("kind"), item.get("metadata", {}).get("name")) for item in items}
        if actual != expected:
            raise ContractError("resume discovered a partial Kubernetes SDG object set")
        return {"state": "RUNNING", "backend_ref": f"{request['namespace']}/{identity['base']}",
                "resumed": True, "native_ids": sorted(item["metadata"]["uid"] for item in items)}
    secret = _secret_payload(request, identity["secret"])
    # Secret material exists only in process memory and kubectl stdin.  It is
    # never an argv value, generated file, report, or log payload.
    _kubectl_json(["apply", "-f", "-", "-o", "json"], stdin=json.dumps(secret))
    try:
        applied = _kubectl_json(["apply", "-f", "-", "-o", "json"], stdin=json.dumps(rendered))
    except BaseException:
        _run(["kubectl", "delete", "secret", identity["secret"], "-n", request["namespace"], "--ignore-not-found=true"], check=False)
        raise
    applied_items = applied.get("items", []) if isinstance(applied, dict) else []
    return {"state": "RUNNING", "backend_ref": f"{request['namespace']}/{identity['base']}",
            "resumed": False, "native_ids": sorted(item["metadata"]["uid"] for item in applied_items),
            "workers": request["generation_nodes"], "worker_gpus": 8, "coordinator_gpus": 2}


def status(request: dict[str, Any]) -> dict[str, Any]:
    selector = f"tao.nvidia.com/action={request['request_sha256'][:16]}"
    payload = _kubectl_json(["get", "jobs", "-n", request["namespace"], "-l", selector, "-o", "json"])
    jobs = payload.get("items", []) if isinstance(payload, dict) else []
    if not jobs:
        return {"state": "UNKNOWN", "message": "owned Kubernetes Jobs are absent"}
    if any(not _owned(item, request) for item in jobs):
        raise ContractError("status discovered foreign Kubernetes ownership")
    jobs = [item for item in jobs if (item.get("metadata", {}).get("labels") or {}).get(
        "tao.nvidia.com/role"
    ) in {"image-worker", "coordinator"}]
    failed = sum(int(item.get("status", {}).get("failed", 0)) for item in jobs)
    active = sum(int(item.get("status", {}).get("active", 0)) for item in jobs)
    succeeded = sum(int(item.get("status", {}).get("succeeded", 0)) for item in jobs)
    stage = pathlib.Path(request["paths"]["stage_dir"])
    canonical = [stage / "endpoint_pool.json", stage / "endpoint_manifest.json",
                 *map(pathlib.Path, request["expected_outputs"][:4])]
    outputs_complete = all(
        path.is_file() and not path.is_symlink() and path.stat().st_size > 0
        for path in canonical
    )
    state = "ERROR" if failed else "COMPLETE" if succeeded >= 1 and outputs_complete else "RUNNING" if active else "PENDING"
    return {"state": state, "active": active, "succeeded": succeeded, "failed": failed,
            "expected_worker_pods": request["generation_nodes"], "outputs_complete": outputs_complete}


def logs(request: dict[str, Any], tail: int) -> str:
    selector = f"tao.nvidia.com/action={request['request_sha256'][:16]}"
    result = _run(["kubectl", "logs", "-n", request["namespace"], "-l", selector,
                   "--all-containers=true", "--tail", str(tail)], check=False)
    # Endpoint secrets are random and should never be printed by workloads;
    # redact common credential shapes defensively before returning logs.
    return re.sub(r"(?i)(api[_-]?key|token|password)\s*[=:]\s*\S+", r"\1=<redacted>", result.stdout + result.stderr)


def cancel(request: dict[str, Any], confirm: bool) -> dict[str, Any]:
    if not confirm:
        raise ContractError("cancel requires --confirm")
    selector = f"tao.nvidia.com/action={request['request_sha256'][:16]}"
    payload = _kubectl_json(["get", "job,service,secret", "-n", request["namespace"], "-l", selector, "-o", "json"])
    items = payload.get("items", []) if isinstance(payload, dict) else []
    if any(not _owned(item, request) for item in items):
        raise ContractError("refusing cleanup because ownership is mixed or foreign")
    native_ids = sorted(item.get("metadata", {}).get("uid", "") for item in items)
    _run(["kubectl", "delete", "job,service,secret", "-n", request["namespace"],
          "-l", selector, "--cascade=foreground", "--ignore-not-found=true"])
    return {"state": "CANCELED", "deleted_native_ids": native_ids}


def _wait_job(namespace: str, name: str, timeout_s: int, interval_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = _kubectl_json(["get", "job", name, "-n", namespace, "-o", "json"])
        status_payload = job.get("status", {})
        if int(status_payload.get("failed", 0)):
            raise ContractError(f"component Job {name} failed")
        if int(status_payload.get("succeeded", 0)) == 1:
            return
        time.sleep(interval_s)
    raise ContractError(f"component Job {name} readiness deadline exceeded")


def identity_worker_host(request: dict[str, Any]) -> str:
    identity = render_resources(request)["_tao"]
    return (
        f"{identity['worker_job']}-0.{identity['worker_job']}."
        f"{request['namespace']}.svc"
    )


def _component_native_argv(
    request: dict[str, Any], args: argparse.Namespace, coordinator_service: str,
) -> tuple[str, list[str], str, list[dict[str, Any]]]:
    runtime = pathlib.Path(request["paths"]["runtime_root"])
    sys.path.insert(0, str(runtime))
    try:
        from iaa_deft.sdg import build_component_command
    finally:
        sys.path.pop(0)
    config = yaml.safe_load(pathlib.Path(request["paths"]["config_path"]).read_text())
    endpoint_urls = {
        "image_edit": args.image_edit_url or (
            f"http://{identity_worker_host(request)}:{request['models']['image_edit']['port']}/v1"
        ),
        "vlm": f"http://{coordinator_service}.{request['namespace']}.svc:{request['models']['vlm']['port']}/v1",
        "llm": f"http://{coordinator_service}.{request['namespace']}.svc:{request['models']['llm']['port']}/v1",
    }
    if args.action == "augment" and (not args.image_edit_url or not args.image_edit_endpoint_id):
        raise ContractError("augment/component dispatch requires a bound image-edit endpoint")
    target = json.loads(args.target_attributes_json)
    docker_argv = build_component_command(
        config, args.action, input_root=args.input_root, output_root=args.output_root,
        source_key=args.source_key or "", attempt=args.attempt,
        target_attributes=target, endpoint_urls=endpoint_urls,
    )
    component = "auto_labeling" if args.action == "label" else "augmentation"
    image = config["images"][component]
    try:
        image_index = docker_argv.index(image)
    except ValueError as exc:
        raise ContractError("shared component command lacks its immutable image") from exc
    entrypoint = docker_argv[docker_argv.index("--entrypoint") + 1]
    native = [entrypoint, *docker_argv[image_index + 1:]]
    mount_root = pathlib.PurePosixPath(request["pvc_mount"])
    mounts = []
    targets = (
        ((args.input_root, "/input", True), (args.output_root, "/output", False))
        if component == "auto_labeling" else
        ((args.input_root, "/app/data/in", True), (args.output_root, "/app/data/out", False))
    )
    for source, destination, read_only in targets:
        source_path = pathlib.PurePosixPath(str(source))
        try:
            sub_path = source_path.relative_to(mount_root)
        except ValueError as exc:
            raise ContractError("component path is outside the shared PVC mount") from exc
        mounts.append({"name": "workspace", "mountPath": destination,
                       "subPath": str(sub_path), "readOnly": read_only})
    working = "/workspace" if component == "auto_labeling" else "/app"
    return request["images"][component], native, working, mounts


def component(request: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    rendered = render_resources(request)
    identity = rendered["_tao"]
    image, native, working, mounts = _component_native_argv(
        request, args, identity["coordinator_job"],
    )
    token = f"{args.action}-{args.source_key or 'stage'}-{args.attempt}"
    digest = hashlib.sha256(token.encode()).hexdigest()[:10]
    name = f"{identity['base']}-{args.action}-{digest}"[:63].rstrip("-")
    labels = _labels(request, "component")
    secret_name = identity["secret"]
    container = {
        "name": "component", "image": image, "command": [native[0]], "args": native[1:],
        "workingDir": working, "volumeMounts": mounts,
        "env": [_secret_env(secret_name, key) for key in
                ("IMAGE_EDIT_API_KEY", "VLM_API_KEY", "LLM_API_KEY")]
               + [{"name": "HOME", "value": "/tmp"}],
    }
    job = {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": name, "namespace": request["namespace"], "labels": labels,
                     "annotations": {"tao.nvidia.com/request-sha256": request["request_sha256"],
                                     "tao.nvidia.com/endpoint-id": args.image_edit_endpoint_id}},
        "spec": {"backoffLimit": 0, "ttlSecondsAfterFinished": request["limits"]["ttl_seconds"],
                 "template": {"metadata": {"labels": labels}, "spec": {
                     "serviceAccountName": request["service_account"], "restartPolicy": "Never",
                     "containers": [container], "volumes": [{"name": "workspace",
                         "persistentVolumeClaim": {"claimName": request["pvc_claim"]}}],
                 }}},
    }
    prior = _run(["kubectl", "get", "job", name, "-n", request["namespace"], "-o", "json"], check=False)
    if prior.returncode == 0:
        existing = json.loads(prior.stdout)
        if not _owned(existing, request):
            raise ContractError("component Job name belongs to another owner")
        if int(existing.get("status", {}).get("failed", 0)):
            _run(["kubectl", "delete", "job", name, "-n", request["namespace"],
                  "--cascade=foreground"])
            _kubectl_json(["apply", "-f", "-", "-o", "json"], stdin=json.dumps(job))
    else:
        _kubectl_json(["apply", "-f", "-", "-o", "json"], stdin=json.dumps(job))
    _wait_job(request["namespace"], name, request["limits"]["request_timeout_s"],
              request["limits"]["retry_interval_s"])
    return {"job": name, "owned": True, "action": args.action}


def _authorized_request(url: str, *, data: bytes | None = None, content_type: str | None = None) -> urllib.request.Request:
    request = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    if "IMAGE_EDIT_API_KEY" not in os.environ:
        raise ContractError("IMAGE_EDIT_API_KEY is absent in coordinator")
    request.add_header("Authorization", "Bearer " + os.environ["IMAGE_EDIT_API_KEY"])
    if content_type:
        request.add_header("Content-Type", content_type)
    return request


def _probe_model(url: str, model: str, role: str, timeout: int) -> None:
    request = _authorized_request(url.rstrip("/") + "/models")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode())
    ids = {item.get("id") for item in payload.get("data", []) if isinstance(item, dict)}
    if model not in ids:
        raise ContractError(f"endpoint {url} serves the wrong model")
    if role == "image_edit":
        boundary = "tao-sdg-readiness"
        image = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nReturn this image unchanged.\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"smoke.png\"\r\n"
            "Content-Type: image/png\r\n\r\n"
        ).encode() + image + f"\r\n--{boundary}--\r\n".encode()
        inference = _authorized_request(
            url.rstrip("/") + "/images/edits", data=body,
            content_type=f"multipart/form-data; boundary={boundary}",
        )
    else:
        body = json.dumps({
            "model": model, "messages": [{"role": "user", "content": "Reply READY."}],
            "max_tokens": 4, "temperature": 0,
        }).encode()
        inference = _authorized_request(
            url.rstrip("/") + "/chat/completions", data=body,
            content_type="application/json",
        )
    with urllib.request.urlopen(inference, timeout=timeout) as response:
        result = json.loads(response.read().decode())
    if role == "image_edit" and not result.get("data"):
        raise ContractError(f"image-edit inference smoke failed for {url}")
    if role != "image_edit" and not result.get("choices"):
        raise ContractError(f"{role} inference smoke failed for {url}")


def expected_endpoint_urls(
    request: dict[str, Any], worker_job: str, worker_service: str,
) -> list[tuple[int, int, str]]:
    return [
        (index, ordinal,
         f"http://{worker_job}-{index}.{worker_service}.{request['namespace']}.svc:"
         f"{request['models']['image_edit']['port'] + ordinal}/v1")
        for index in range(request["generation_nodes"])
        for ordinal in range(8)
    ]


def coordinator(request: dict[str, Any], worker_service: str, worker_job: str) -> dict[str, Any]:
    config_path = pathlib.Path(request["paths"]["config_path"])
    state_path = pathlib.Path(request["paths"]["results_dir"]) / "deft_state.json"
    if (_file_sha256(config_path) != request["bindings"]["config_sha256"]
            or _file_sha256(state_path) != request["bindings"]["state_sha256"]):
        raise ContractError("state or SDG config changed after Kubernetes request preparation")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state_runtime = state.get(
        "active_runtime_sha256", (state.get("config") or {}).get("iaa_deft_bundle_sha256"),
    )
    phase = (state.get("iterations") or {}).get(f"iter{request['iteration']}")
    if (state.get("current_iteration") != request["iteration"]
            or (state.get("config") or {}).get("platform") != "kubernetes"
            or not isinstance(phase, dict) or phase.get("status") != "in_progress"
            or phase.get("stage_completed") != "history_select"
            or state_runtime != request["bindings"]["runtime_sha256"]):
        raise ContractError("committed Kubernetes SDG state binding is no longer current")
    immutable = yaml.safe_load(config_path.read_text())
    if (
        immutable.get("generation", {}).get("generation_nodes") != request["generation_nodes"]
        or immutable.get("generation", {}).get("gpus_per_generation_node") != 8
    ):
        raise ContractError("immutable SDG topology disagrees with Kubernetes request")
    for role in ROLES:
        expected = immutable.get("models", {}).get(role, {})
        if any(request["models"][role].get(key) != expected.get(key) for key in ("id", "revision", "backend", "port")):
            raise ContractError(f"immutable {role} model disagrees with Kubernetes request")
    for component in ("augmentation", "auto_labeling"):
        if request["images"][component] != immutable.get("images", {}).get(component):
            raise ContractError(f"immutable {component} image disagrees with Kubernetes request")
    serving_images = {
        "image_edit": "image_edit_serving",
        "text_serving": "text_serving",
    }
    for request_key, config_key in serving_images.items():
        if request["images"][request_key] != immutable.get("images", {}).get(config_key):
            raise ContractError(
                f"immutable {config_key} image disagrees with Kubernetes request"
            )
    rendered = render_resources(request)
    identity = rendered["_tao"]
    if worker_service != identity["worker_job"] or worker_job != identity["worker_job"]:
        raise ContractError("coordinator worker identity disagrees with signed request")
    namespace = request["namespace"]
    selector = ",".join((
        f"tao.nvidia.com/action={request['request_sha256'][:16]}",
        "tao.nvidia.com/role=image-worker",
    ))
    deadline = time.monotonic() + request["limits"]["startup_timeout_s"]
    pods = []
    while time.monotonic() < deadline:
        payload = _kubectl_json(["get", "pods", "-n", namespace, "-l", selector, "-o", "json"])
        pods = payload.get("items", []) if isinstance(payload, dict) else []
        ready = [pod for pod in pods if pod.get("status", {}).get("phase") == "Running"
                 and all(item.get("ready") for item in pod.get("status", {}).get("containerStatuses", []))]
        if len(ready) == request["generation_nodes"]:
            pods = ready
            break
        time.sleep(request["limits"]["retry_interval_s"])
    if len(pods) != request["generation_nodes"]:
        raise ContractError("strict image-worker pod readiness deadline exceeded")
    indexed: dict[int, dict[str, Any]] = {}
    for pod in pods:
        if not _owned(pod, request):
            raise ContractError("worker discovery includes foreign ownership")
        raw_index = (pod.get("metadata", {}).get("labels") or {}).get("batch.kubernetes.io/job-completion-index")
        if raw_index is None or not str(raw_index).isdigit() or int(raw_index) in indexed:
            raise ContractError("worker pod lacks a unique completion index")
        indexed[int(raw_index)] = pod
    endpoints = []
    for index in range(request["generation_nodes"]):
        pod = indexed.get(index)
        if pod is None:
            raise ContractError("worker pool is partial")
        for _, ordinal, url in [row for row in expected_endpoint_urls(request, worker_job, worker_service) if row[0] == index]:
            _probe_model(url, request["models"]["image_edit"]["id"], "image_edit",
                         request["limits"]["request_timeout_s"])
            endpoints.append({
                "id": f"{worker_job}-{index}-gpu-{ordinal}", "url": url, "capacity": 1,
                "gpu_identity": f"{pod['metadata']['uid']}/gpu:{ordinal}",
                "owner": {"native_id": pod["metadata"]["uid"], "name": pod["metadata"]["name"]},
            })
    if len(endpoints) != request["generation_nodes"] * 8:
        raise ContractError("refusing to publish a partial endpoint pool")
    for role in ("vlm", "llm"):
        _probe_model(
            f"http://{identity['coordinator_job']}.{namespace}.svc:{request['models'][role]['port']}/v1",
            request["models"][role]["id"], role, request["limits"]["request_timeout_s"],
        )
    stage = pathlib.Path(request["paths"]["stage_dir"])
    pool = {
        "schema_version": "1", "platform": "kubernetes",
        "model": {"id": request["models"]["image_edit"]["id"],
                  "revision": request["models"]["image_edit"]["revision"]},
        "required_capacity": len(endpoints), "auth_env": "IMAGE_EDIT_API_KEY",
        "endpoints": endpoints, "created_at": request["started_at"],
        "request_sha256": request["request_sha256"],
    }
    endpoint_manifest = {
        "schema_version": "1", "platform": "kubernetes", "ownership": "kubernetes_job",
        "request_sha256": request["request_sha256"],
        "roles": {role: {"model": request["models"][role]["id"],
                         "revision": request["models"][role]["revision"], "ready": True}
                  for role in ("vlm", "llm")},
        "components": {role: request["images"][role]
                       for role in ("augmentation", "auto_labeling")},
        "image_edit_pool": {"path": str(stage / "endpoint_pool.json"),
                            "required_capacity": len(endpoints)},
    }
    # Publish only after every one of N*8 endpoints and both auxiliary roles
    # pass readiness; a partial pool is never observable.
    _atomic_json(stage / "endpoint_pool.json", pool)
    _atomic_json(stage / "endpoint_manifest.json", endpoint_manifest)
    runtime = pathlib.Path(request["paths"]["runtime_root"])
    command = [
        sys.executable, str(runtime / "run_sdg_stage.py"), "execute",
        "--execution-platform", "kubernetes", "--config", request["paths"]["config_path"],
        "--output-root", request["paths"]["stage_dir"],
        "--mined-pairs", request["paths"]["mined_pairs"],
        "--eval-list", request["paths"]["eval_list"],
        "--attribute-vocab", request["paths"]["attribute_vocab"],
        "--image-edit-endpoint-pool", str(stage / "endpoint_pool.json"),
        "--component-executor", str(runtime / "kubernetes_sdg_action.py"),
        "--component-executor-request", str(stage / "kubernetes_request.json"),
        "--component-executor-job-id", request["action_id"],
    ]
    completed = _run(command, check=False)
    if completed.returncode != 0:
        raise ContractError("shared SDG runtime failed; inspect canonical Kubernetes status evidence")
    if not all(pathlib.Path(path).is_file() for path in request["expected_outputs"]):
        raise ContractError("shared SDG runtime omitted canonical outputs")
    cleanup_selector = f"tao.nvidia.com/action={request['request_sha256'][:16]}"
    _run(["kubectl", "delete", "job", worker_job, "-n", namespace,
          "--cascade=foreground", "--ignore-not-found=true"])
    _run(["kubectl", "delete", "service", worker_service, "-n", namespace,
          "--ignore-not-found=true"])
    _run(["kubectl", "delete", "service", identity["coordinator_job"], "-n", namespace,
          "--ignore-not-found=true"])
    _run(["kubectl", "delete", "secret", identity["secret"], "-n", namespace,
          "--ignore-not-found=true"])
    _run(["kubectl", "delete", "job", "-n", namespace,
          "-l", f"tao.nvidia.com/action={request['request_sha256'][:16]},tao.nvidia.com/role=component",
          "--cascade=foreground", "--ignore-not-found=true"])
    _atomic_json(stage / "endpoint_cleanup.json", {
        "schema_version": "1", "platform": "kubernetes",
        "request_sha256": request["request_sha256"], "selector": cleanup_selector,
        "deleted": [worker_job, worker_service, identity["coordinator_job"],
                    identity["secret"], "owned-component-jobs"],
    })
    return {"status": "complete", "required_capacity": len(endpoints)}


def _atomic_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="verb", required=True)
    prepare = sub.add_parser("prepare-request")
    prepare.add_argument("--deft-state", required=True, type=pathlib.Path)
    prepare.add_argument("--sdg-config", required=True, type=pathlib.Path)
    prepare.add_argument("--iteration", required=True, type=int)
    prepare.add_argument("--namespace", required=True)
    prepare.add_argument("--pvc-claim", required=True)
    prepare.add_argument("--pvc-mount", required=True)
    prepare.add_argument("--service-account", required=True)
    prepare.add_argument("--runtime-root", required=True, type=pathlib.Path)
    prepare.add_argument("--augmentation-image", required=True)
    prepare.add_argument("--auto-labeling-image", required=True)
    prepare.add_argument("--image-edit-image", required=True)
    prepare.add_argument("--text-serving-image", required=True)
    prepare.add_argument("--controller-image", required=True)
    prepare.add_argument("--ttl-seconds", type=int, default=3600)
    prepare.add_argument("--output", required=True, type=pathlib.Path)
    for verb in ("submit", "status", "logs", "cancel"):
        child = sub.add_parser(verb)
        child.add_argument("--request", required=True, type=pathlib.Path)
        if verb == "logs":
            child.add_argument("--tail", type=int, default=200, choices=range(1, 10001))
        if verb == "cancel":
            child.add_argument("--confirm", action="store_true")
    # Internal coordinator entrypoint is rendered into the run-owned pod.  It
    # is not a public platform verb and intentionally does not share training.
    coordinator = sub.add_parser("coordinator", help=argparse.SUPPRESS)
    coordinator.add_argument("--request", required=True, type=pathlib.Path)
    coordinator.add_argument("--worker-service", required=True)
    coordinator.add_argument("--worker-job", required=True)
    component_parser = sub.add_parser("component", help=argparse.SUPPRESS)
    component_parser.add_argument("--request", required=True, type=pathlib.Path)
    component_parser.add_argument("--job-id", required=True)
    component_parser.add_argument("--action", required=True, choices=("preprocess", "augment", "split", "label"))
    component_parser.add_argument("--input-root", required=True, type=pathlib.Path)
    component_parser.add_argument("--output-root", required=True, type=pathlib.Path)
    component_parser.add_argument("--attempt", required=True, type=int)
    component_parser.add_argument("--source-key")
    component_parser.add_argument("--target-attributes-json", default="{}")
    component_parser.add_argument("--image-edit-url")
    component_parser.add_argument("--image-edit-endpoint-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.verb == "prepare-request":
            prepared = prepare_request(args)
            request = prepared["request"]
            print(json.dumps({
                "status": prepared["status"], "request": prepared["output"],
                "action_id": request["action_id"],
                "request_sha256": request["request_sha256"],
                "generation_nodes": request["generation_nodes"],
            }, sort_keys=True))
            return 0
        request = load_request(args.request)
        if args.verb == "submit":
            result = submit(request)
        elif args.verb == "status":
            result = status(request)
        elif args.verb == "logs":
            print(logs(request, args.tail), end="")
            return 0
        elif args.verb == "cancel":
            result = cancel(request, args.confirm)
        elif args.verb == "component":
            result = component(request, args)
        elif args.verb == "coordinator":
            result = coordinator(request, args.worker_service, args.worker_job)
        else:
            raise ContractError("unsupported action")
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"kubernetes_sdg_action[{args.verb}]: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
