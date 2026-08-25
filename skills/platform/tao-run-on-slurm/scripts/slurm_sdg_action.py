#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic request preparation plus four-verb SLURM IAA SDG consumer.

The immutable request describes data and model roles, never shell. Independent
eight-GPU image-worker allocations host one single-GPU image-edit service per
GPU, while a two-GPU coordinator hosts the VLM and LLM. The scheduler selects
physical GPUs; endpoint steps see only their step-local CUDA ordinals.
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
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any, Sequence

import yaml


WORKFLOW = "tao-run-deft-iaa"
KIND = "slurm_sdg_action"
NAME = "sdg_execute"
ROLES = ("image_edit", "vlm", "llm")
IMAGE_SERVICES_PER_NODE = 8
IMAGE_SERVICE_CPUS = 8
IMAGE_MASTER_PORT_BASE = 31000
IMAGE_MASTER_PORT_STRIDE = 100
WORKER_CLEANUP_TIMEOUT_CAP_S = 300
WORKER_OWNERSHIP_TIMEOUT_CAP_S = 60
SCHEDULER_QUERY_TIMEOUT_S = 10
WORKER_TERMINAL_STATES = {
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
    "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT",
}
ROLE_GPUS = {"image_edit": 1, "vlm": 1, "llm": 1}
SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]{0,127}$")
SAFE_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./-]{0,255}$")
SAFE_SOURCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
JOB_HANDLE = re.compile(r"^(?P<id>[0-9]+)(?:;[A-Za-z0-9_.-]+)?$")
SECRET_RE = re.compile(
    r"(?i)(?:(?:api[_-]?key|token|password)\s*[=:]\s*)[^\s,;]+|"
    r"(?:authorization\s*:\s*bearer\s+)[^\s,;]+|"
    r"(?:hf_|nvapi-)[A-Za-z0-9_.-]{8,}"
)
EXPECTED_KEYS = {
    "schema_version", "workflow", "kind", "platform", "name", "action_id",
    "started_at", "started_ns", "generation_nodes",
    "run_id", "iteration", "attempt", "results_dir", "stage_dir",
    "dataset_root", "config_path", "config_sha256", "runtime_root", "runtime_sha256",
    "cache_dir", "images", "component_sources",
    "models", "resources", "scheduler", "limits", "forward_env", "expected_outputs",
    "request_sha256",
}
POOL_REBIND_REPAIR_KIND = "unstarted-endpoint-pool-rebind"
POOL_REBIND_ERROR = "image-edit endpoint pool changed outside explicit unfinished resume"
SCHEDULER_RESCHEDULE_KIND = "pending-capacity-reschedule"
IMAGE_MASTER_PORT_REPAIR_KIND = "image-master-port-isolation"
IMAGE_KEYS = {"augmentation", "auto_labeling", "image_edit", "text_serving"}
EDITABLE_ATTRIBUTES = {
    "top outer color", "top outer type", "bottom color", "bottom type",
    "shoe color", "shoe type",
}


def _canonical_sha256(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("request_sha256", None)
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resume_sha256(payload: dict[str, Any]) -> str:
    """Bind data/config semantics while allowing one new scheduler attempt."""
    stable = dict(payload)
    for key in (
        "request_sha256", "action_id", "attempt", "retry", "repair", "reschedule",
        "launch_repair",
    ):
        stable.pop(key, None)
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: pathlib.Path, payload: Any) -> None:
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


def _absolute(value: Any, name: str) -> pathlib.Path:
    if not isinstance(value, str) or not value.startswith("/") or "\x00" in value:
        raise ValueError(f"{name} must be one absolute path")
    path = pathlib.PurePosixPath(value)
    if path == pathlib.PurePosixPath("/") or ".." in path.parts:
        raise ValueError(f"{name} must not be root or traverse '..'")
    return pathlib.Path(value)


def _under(path: pathlib.Path, root: pathlib.Path, name: str) -> pathlib.Path:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} must be under {root}") from exc
    return path


def _mapping(value: Any, name: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} must contain exactly {sorted(keys)}")
    return value


def load_request(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError("--request must be an absolute regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_request(payload)


def validate_request(payload: Any) -> dict[str, Any]:
    if (not isinstance(payload, dict)
            or set(payload) not in (
                EXPECTED_KEYS,
                EXPECTED_KEYS | {"retry"},
                EXPECTED_KEYS | {"retry", "repair"},
                EXPECTED_KEYS | {"retry", "repair", "reschedule"},
                EXPECTED_KEYS | {"retry", "repair", "reschedule", "launch_repair"},
            )):
        raise ValueError("SDG request has missing or unexpected fields")
    fixed = {
        "schema_version": "1", "workflow": WORKFLOW, "kind": KIND,
        "platform": "slurm", "name": NAME,
    }
    for key, value in fixed.items():
        if payload.get(key) != value:
            raise ValueError(f"request.{key} must be {value!r}")
    digest = payload.get("request_sha256")
    if not isinstance(digest, str) or not SHA256.fullmatch(digest):
        raise ValueError("request_sha256 must be a lowercase SHA-256")
    if _canonical_sha256(payload) != digest:
        raise ValueError("request_sha256 does not match immutable content")
    for key in ("action_id", "run_id"):
        if not isinstance(payload.get(key), str) or not SAFE_TOKEN.fullmatch(payload[key]):
            raise ValueError(f"request.{key} contains unsupported characters")
    try:
        started = dt.datetime.fromisoformat(payload["started_at"])
    except (TypeError, ValueError) as exc:
        raise ValueError("request.started_at must be one timezone-aware ISO timestamp") from exc
    if started.tzinfo is None:
        raise ValueError("request.started_at must include a timezone")
    if not isinstance(payload["started_ns"], int) or isinstance(payload["started_ns"], bool) or payload["started_ns"] < 1:
        raise ValueError("request.started_ns must be a positive integer")
    if (not isinstance(payload["generation_nodes"], int) or isinstance(payload["generation_nodes"], bool)
            or not 1 <= payload["generation_nodes"] <= 64):
        raise ValueError("request.generation_nodes must be in [1, 64]")
    if not isinstance(payload["iteration"], int) or isinstance(payload["iteration"], bool) or payload["iteration"] < 1:
        raise ValueError("request.iteration must be a positive integer")
    if payload["attempt"] not in (1, 2):
        raise ValueError("request.attempt must be 1 or 2")
    retry = payload.get("retry")
    if payload["attempt"] == 1 and retry is not None:
        raise ValueError("attempt 1 must not contain retry lineage")
    if payload["attempt"] == 2:
        retry_keys = {
            "job_id", "action_id", "backend_ref", "request_sha256",
            "job_record_sha256", "job_group_sha256", "native_states_sha256",
            "native_states", "terminal_evidence",
        }
        if not isinstance(retry, dict) or set(retry) not in (
            retry_keys, retry_keys | {"prior_partition", "new_partition"},
        ):
            raise ValueError(
                "retry must contain the legacy evidence fields and, when rebound, "
                "both prior_partition and new_partition"
            )
        for key in ("job_id", "action_id"):
            if not isinstance(retry[key], str) or not SAFE_TOKEN.fullmatch(retry[key]):
                raise ValueError(f"retry.{key} is invalid")
        if not str(retry["backend_ref"]).isdigit():
            raise ValueError("retry.backend_ref must be numeric")
        if "prior_partition" in retry:
            if any(
                not isinstance(retry[key], str)
                or not SAFE_TOKEN.fullmatch(retry[key])
                for key in ("prior_partition", "new_partition")
            ):
                raise ValueError("retry partition lineage contains unsupported characters")
            if retry["new_partition"] != payload.get("scheduler", {}).get("partition"):
                raise ValueError("retry.new_partition must equal request.scheduler.partition")
        for key in ("request_sha256", "job_record_sha256", "job_group_sha256",
                    "native_states_sha256"):
            if not isinstance(retry[key], str) or not SHA256.fullmatch(retry[key]):
                raise ValueError(f"retry.{key} must be a lowercase SHA-256")
        native_states = retry["native_states"]
        if (not isinstance(native_states, dict) or not native_states
                or any(not str(key).isdigit() or value not in WORKER_TERMINAL_STATES
                       for key, value in native_states.items())
                or hashlib.sha256(json.dumps(
                    native_states, sort_keys=True, separators=(",", ":")
                ).encode()).hexdigest() != retry["native_states_sha256"]):
            raise ValueError("retry.native_states is invalid or disagrees with its digest")
        terminal_evidence = retry["terminal_evidence"]
        if terminal_evidence is not None:
            terminal_evidence = _mapping(
                terminal_evidence, "retry.terminal_evidence",
                {"kind", "terminal_sha256", "cleanup_sha256",
                 "expected_outputs_sha256", "evidence_sha256"},
            )
            body = dict(terminal_evidence)
            evidence_sha256 = body.pop("evidence_sha256")
            outputs = terminal_evidence["expected_outputs_sha256"]
            if (
                terminal_evidence["kind"] != "coordinator-cleanup-failure"
                or any(
                    not isinstance(terminal_evidence[key], str)
                    or not SHA256.fullmatch(terminal_evidence[key])
                    for key in ("terminal_sha256", "cleanup_sha256", "evidence_sha256")
                )
                or not isinstance(outputs, dict)
                or set(outputs) != set(payload["expected_outputs"])
                or any(not isinstance(value, str) or not SHA256.fullmatch(value)
                       for value in outputs.values())
                or evidence_sha256 != hashlib.sha256(json.dumps(
                    body, sort_keys=True, separators=(",", ":")
                ).encode()).hexdigest()
            ):
                raise ValueError("retry.terminal_evidence is invalid or disagrees with its digest")
    repair = payload.get("repair")
    if repair is not None:
        if payload["attempt"] != 2 or retry is None:
            raise ValueError("pool-rebind repair requires authoritative attempt-2 retry lineage")
        repair = _mapping(
            repair, "repair",
            {
                "kind", "job_id", "action_id", "backend_ref", "request_sha256",
                "job_record_sha256", "job_group_sha256", "native_states_sha256",
                "native_states", "terminal_sha256", "cleanup_sha256",
                "execute_log_sha256", "progress_sha256", "runtime_rebind_sha256",
                "evidence_sha256",
            },
        )
        body = dict(repair)
        evidence_sha256 = body.pop("evidence_sha256")
        if repair["kind"] != POOL_REBIND_REPAIR_KIND:
            raise ValueError("repair.kind is not an allowlisted SDG repair")
        for key in ("job_id", "action_id"):
            if not isinstance(repair[key], str) or not SAFE_TOKEN.fullmatch(repair[key]):
                raise ValueError(f"repair.{key} is invalid")
        if not str(repair["backend_ref"]).isdigit():
            raise ValueError("repair.backend_ref must be numeric")
        for key in (
            "request_sha256", "job_record_sha256", "job_group_sha256",
            "native_states_sha256", "terminal_sha256", "cleanup_sha256",
            "execute_log_sha256", "progress_sha256", "evidence_sha256",
            "runtime_rebind_sha256",
        ):
            if not isinstance(repair[key], str) or not SHA256.fullmatch(repair[key]):
                raise ValueError(f"repair.{key} must be a lowercase SHA-256")
        native_states = repair["native_states"]
        if (
            not isinstance(native_states, dict)
            or not native_states
            or any(
                not str(key).isdigit() or value not in WORKER_TERMINAL_STATES
                for key, value in native_states.items()
            )
            or hashlib.sha256(json.dumps(
                native_states, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest() != repair["native_states_sha256"]
            or evidence_sha256 != hashlib.sha256(json.dumps(
                body, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
        ):
            raise ValueError("repair evidence is malformed or disagrees with its digest")
    reschedule = payload.get("reschedule")
    if reschedule is not None:
        if payload["attempt"] != 2 or retry is None or repair is None:
            raise ValueError(
                "scheduler reschedule requires authoritative attempt-2 repair lineage"
            )
        reschedule = _mapping(
            reschedule, "reschedule",
            {
                "kind", "job_id", "action_id", "backend_ref", "request_sha256",
                "job_record_sha256", "job_group_sha256",
                "native_accounting_sha256", "native_accounting",
                "prior_time_minutes", "new_time_minutes", "progress_sha256",
                "evidence_sha256",
            },
        )
        body = dict(reschedule)
        evidence_sha256 = body.pop("evidence_sha256")
        if reschedule["kind"] != SCHEDULER_RESCHEDULE_KIND:
            raise ValueError("reschedule.kind is not an allowlisted SDG reschedule")
        for key in ("job_id", "action_id"):
            if (
                not isinstance(reschedule[key], str)
                or not SAFE_TOKEN.fullmatch(reschedule[key])
            ):
                raise ValueError(f"reschedule.{key} is invalid")
        if not str(reschedule["backend_ref"]).isdigit():
            raise ValueError("reschedule.backend_ref must be numeric")
        for key in (
            "request_sha256", "job_record_sha256", "job_group_sha256",
            "native_accounting_sha256", "progress_sha256", "evidence_sha256",
        ):
            if (
                not isinstance(reschedule[key], str)
                or not SHA256.fullmatch(reschedule[key])
            ):
                raise ValueError(f"reschedule.{key} must be a lowercase SHA-256")
        accounting = reschedule["native_accounting"]
        if (
            not isinstance(accounting, dict)
            or not accounting
            or any(
                not str(native_id).isdigit()
                or not isinstance(row, dict)
                or set(row) != {"state", "elapsed_raw"}
                or row.get("state") != "CANCELLED"
                or row.get("elapsed_raw") != 0
                for native_id, row in accounting.items()
            )
            or hashlib.sha256(json.dumps(
                accounting, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest() != reschedule["native_accounting_sha256"]
            or not isinstance(reschedule["prior_time_minutes"], int)
            or not isinstance(reschedule["new_time_minutes"], int)
            or not 10 <= reschedule["new_time_minutes"] < reschedule["prior_time_minutes"]
        ):
            raise ValueError(
                "reschedule evidence is malformed or does not prove a shorter unstarted job"
            )
        if evidence_sha256 != hashlib.sha256(json.dumps(
            body, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest():
            raise ValueError("reschedule evidence digest disagrees with its content")
    launch_repair = payload.get("launch_repair")
    if launch_repair is not None:
        if (
            payload["attempt"] != 2
            or retry is None
            or repair is None
            or reschedule is None
        ):
            raise ValueError(
                "image master-port repair requires full attempt-2 reschedule lineage"
            )
        launch_repair = _mapping(
            launch_repair, "launch_repair",
            {
                "kind", "job_id", "action_id", "backend_ref", "request_sha256",
                "job_record_sha256", "job_group_sha256", "native_states_sha256",
                "native_states", "terminal_sha256", "cleanup_sha256",
                "failure_evidence", "descriptor_sha256", "progress_sha256",
                "evidence_sha256",
            },
        )
        body = dict(launch_repair)
        evidence_sha256 = body.pop("evidence_sha256")
        if launch_repair["kind"] != IMAGE_MASTER_PORT_REPAIR_KIND:
            raise ValueError("launch_repair.kind is not an allowlisted repair")
        for key in ("job_id", "action_id"):
            if (
                not isinstance(launch_repair[key], str)
                or not SAFE_TOKEN.fullmatch(launch_repair[key])
            ):
                raise ValueError(f"launch_repair.{key} is invalid")
        if not str(launch_repair["backend_ref"]).isdigit():
            raise ValueError("launch_repair.backend_ref must be numeric")
        for key in (
            "request_sha256", "job_record_sha256", "job_group_sha256",
            "native_states_sha256", "terminal_sha256", "cleanup_sha256",
            "progress_sha256", "evidence_sha256",
        ):
            if (
                not isinstance(launch_repair[key], str)
                or not SHA256.fullmatch(launch_repair[key])
            ):
                raise ValueError(f"launch_repair.{key} must be a lowercase SHA-256")
        native_states = launch_repair["native_states"]
        failures = launch_repair["failure_evidence"]
        descriptors = launch_repair["descriptor_sha256"]
        failed_ids = {
            str(item.get("native_id")) for item in failures if isinstance(item, dict)
        } if isinstance(failures, list) else set()
        if (
            not isinstance(native_states, dict)
            or not native_states
            or any(
                not str(native_id).isdigit() or state not in WORKER_TERMINAL_STATES
                for native_id, state in native_states.items()
            )
            or hashlib.sha256(json.dumps(
                native_states, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest() != launch_repair["native_states_sha256"]
            or not isinstance(failures, list)
            or not failures
            or any(
                not isinstance(item, dict)
                or set(item) != {
                    "worker_name", "native_id", "endpoint_id",
                    "worker_log_sha256", "endpoint_log_sha256",
                }
                or not SAFE_TOKEN.fullmatch(str(item.get("worker_name", "")))
                or not str(item.get("native_id", "")).isdigit()
                or not re.fullmatch(r"img-[0-9]{3}-gpu-[0-7]", str(item.get("endpoint_id", "")))
                or not SHA256.fullmatch(str(item.get("worker_log_sha256", "")))
                or not SHA256.fullmatch(str(item.get("endpoint_log_sha256", "")))
                for item in failures
            )
            or failed_ids != {
                native_id for native_id, state in native_states.items()
                if state == "FAILED"
            }
            or not isinstance(descriptors, dict)
            or any(
                not isinstance(path, str)
                or not path.startswith("/")
                or not SHA256.fullmatch(str(digest))
                for path, digest in descriptors.items()
            )
            or evidence_sha256 != hashlib.sha256(json.dumps(
                body, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
        ):
            raise ValueError(
                "image master-port repair evidence is malformed or disagrees with its digest"
            )

    results = _absolute(payload["results_dir"], "results_dir")
    stage = _absolute(payload["stage_dir"], "stage_dir")
    expected_stage = results / f"iter_{payload['iteration']}" / "datagen"
    if stage != expected_stage:
        raise ValueError(f"stage_dir must be {expected_stage}")
    config = _absolute(payload["config_path"], "config_path")
    if config != results / "config" / "sdg_config.yaml":
        raise ValueError("config_path must be the run-owned sdg_config.yaml")
    for key in ("config_sha256", "runtime_sha256"):
        if not isinstance(payload[key], str) or not SHA256.fullmatch(payload[key]):
            raise ValueError(f"request.{key} must be a lowercase SHA-256")
    shared_paths = {
        "results_dir": results, "stage_dir": stage, "config_path": config,
        "dataset_root": _absolute(payload["dataset_root"], "dataset_root"),
        "runtime_root": _absolute(payload["runtime_root"], "runtime_root"),
        "cache_dir": _absolute(payload["cache_dir"], "cache_dir"),
    }
    for name, path in shared_paths.items():
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(path)):
            raise ValueError(f"request.{name} contains unsupported shared-path characters")

    images = _mapping(payload["images"], "images", IMAGE_KEYS)
    for key, value in images.items():
        image = _absolute(value, f"images.{key}")
        if image.suffix != ".sqsh":
            raise ValueError(f"images.{key} must be a prevalidated .sqsh path")
    sources = _mapping(payload["component_sources"], "component_sources", IMAGE_KEYS)
    for key, value in sources.items():
        if (not isinstance(value, str) or not value or any(char.isspace() for char in value)
                or "@" in value.split("/", 1)[0]):
            raise ValueError(f"component_sources.{key} is invalid")

    models = _mapping(payload["models"], "models", set(ROLES))
    ports: set[int] = set()
    for role in ROLES:
        model = _mapping(
            models[role], f"models.{role}",
            {"id", "revision", "backend", "port", "tensor_parallel"},
        )
        if not isinstance(model["id"], str) or not SAFE_MODEL.fullmatch(model["id"]):
            raise ValueError(f"models.{role}.id is invalid")
        if not isinstance(model["revision"], str) or not REVISION.fullmatch(model["revision"]):
            raise ValueError(f"models.{role}.revision must be a 40-character commit")
        expected_backend = "vllm-omni" if role == "image_edit" else "vllm"
        if model["backend"] != expected_backend:
            raise ValueError(f"models.{role}.backend must be {expected_backend}")
        if model["tensor_parallel"] != 1:
            raise ValueError(f"models.{role}.tensor_parallel must be 1")
        port = model["port"]
        if not isinstance(port, int) or isinstance(port, bool) or not 1024 <= port <= 65535 or port in ports:
            raise ValueError(f"models.{role}.port must be unique and in [1024, 65535]")
        if role == "image_edit" and port + IMAGE_SERVICES_PER_NODE - 1 > 65535:
            raise ValueError("models.image_edit.port must leave room for eight service ports")
        ports.add(port)

    resources = _mapping(
        payload["resources"], "resources",
        {"coordinator_nodes", "coordinator_gpus", "image_worker_nodes", "image_worker_gpus",
         "image_worker_capacity", "image_worker_cpus_per_task",
         "coordinator_cpus_per_task", "time_minutes"},
    )
    required_resources = {
        "coordinator_nodes": 1, "coordinator_gpus": 2,
        "image_worker_nodes": 1, "image_worker_gpus": 8,
        "image_worker_capacity": 8,
    }
    for key, expected in required_resources.items():
        if resources.get(key) != expected:
            raise ValueError(f"resources.{key} must be {expected}")
    minimum_worker_cpus = IMAGE_SERVICES_PER_NODE * IMAGE_SERVICE_CPUS
    worker_cpus = resources["image_worker_cpus_per_task"]
    if (not isinstance(worker_cpus, int) or isinstance(worker_cpus, bool)
            or not minimum_worker_cpus <= worker_cpus <= 256):
        raise ValueError(
            "resources.image_worker_cpus_per_task must be in "
            f"[{minimum_worker_cpus}, 256]"
        )
    coordinator_cpus = resources["coordinator_cpus_per_task"]
    if (not isinstance(coordinator_cpus, int) or isinstance(coordinator_cpus, bool)
            or not 8 <= coordinator_cpus <= 256):
        raise ValueError("resources.coordinator_cpus_per_task must be in [8, 256]")
    if not isinstance(resources["time_minutes"], int) or not 10 <= resources["time_minutes"] <= 1440:
        raise ValueError("resources.time_minutes must be in [10, 1440]")
    if (
        reschedule is not None
        and resources["time_minutes"] != reschedule["new_time_minutes"]
    ):
        raise ValueError(
            "resources.time_minutes must equal reschedule.new_time_minutes"
        )
    scheduler = _mapping(payload["scheduler"], "scheduler", {"account", "partition"})
    for key, value in scheduler.items():
        if value is not None and (not isinstance(value, str) or not SAFE_TOKEN.fullmatch(value)):
            raise ValueError(f"scheduler.{key} contains unsupported characters")

    limits = _mapping(
        payload["limits"], "limits",
        {"startup_timeout_s", "retry_interval_s", "request_timeout_s", "image_edit_request_timeout_s",
         "verification_max_attempts", "component_max_attempts"},
    )
    for key in ("startup_timeout_s", "retry_interval_s", "request_timeout_s", "image_edit_request_timeout_s"):
        if not isinstance(limits[key], int) or isinstance(limits[key], bool) or limits[key] < 1:
            raise ValueError(f"limits.{key} must be a positive integer")
    if not 1 <= limits["verification_max_attempts"] <= 5:
        raise ValueError("limits.verification_max_attempts must be in [1, 5]")
    if limits["component_max_attempts"] not in (1, 2):
        raise ValueError("limits.component_max_attempts must be 1 or 2")
    if payload["forward_env"] not in ([], ["HF_TOKEN"]):
        raise ValueError("forward_env may contain only HF_TOKEN")
    expected = [
        str(stage / "dataset" / "sdg_manifest.json"),
        str(stage / "dataset" / "sdg_pairs.json"),
        str(stage / "dataset" / "sdg_image_list.txt"),
        str(stage / "sdg_execution_manifest.json"),
    ]
    if payload["expected_outputs"] != expected:
        raise ValueError("expected_outputs must be the four canonical SDG outputs in order")
    return payload


def _local_regular(path: pathlib.Path, name: str) -> pathlib.Path:
    lexical = path.expanduser().absolute()
    if lexical != pathlib.Path(os.path.abspath(lexical)):
        raise ValueError(f"{name} must not contain lexical traversal")
    if not lexical.is_file() or lexical.is_symlink() or lexical.resolve() != lexical:
        raise ValueError(f"{name} must be an absolute regular non-symlink file")
    return lexical


def _agent_cleanup_terminal_evidence(
    login: str, request: dict[str, Any], job_id: str, backend_ref: str,
    group: dict[str, Any], *, terminal_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Attest one cleanup-only backend failure after an agent terminalized its record."""

    stage = pathlib.Path(request["stage_dir"])
    terminal_path = terminal_path or stage / f"slurm_sdg_terminal.{job_id}.json"
    cleanup_path = stage / f"endpoint_cleanup.{job_id}.json"
    terminal = _remote_json_file(login, terminal_path, "coordinator terminal evidence")
    expected_terminal_keys = {
        "schema_version", "workflow", "kind", "status", "job_id", "action_id",
        "coordinator_native_id", "request_sha256", "resume_sha256", "attempt",
        "started_at", "started_ns", "worker_started_at", "finished_at", "error",
    }
    if (
        set(terminal) != expected_terminal_keys
        or terminal.get("schema_version") != "1"
        or terminal.get("workflow") != WORKFLOW
        or terminal.get("kind") != KIND
        or terminal.get("status") != "error"
        or terminal.get("job_id") != job_id
        or terminal.get("action_id") != request["action_id"]
        or str(terminal.get("coordinator_native_id", "")) != backend_ref
        or terminal.get("request_sha256") != request["request_sha256"]
        or terminal.get("resume_sha256") != _resume_sha256(request)
        or terminal.get("attempt") != request["attempt"]
        or terminal.get("started_at") != request["started_at"]
        or terminal.get("started_ns") != request["started_ns"]
        or terminal.get("error") != "owned image-worker cleanup did not complete"
    ):
        raise ValueError("agent-terminalized retry lacks exact coordinator cleanup-failure evidence")
    cleanup = _remote_json_file(login, cleanup_path, "endpoint cleanup evidence")
    if (
        set(cleanup) != {"schema_version", "job_id", "action_id", "request_sha256", "steps"}
        or cleanup.get("schema_version") != "1"
        or cleanup.get("job_id") != job_id
        or cleanup.get("action_id") != request["action_id"]
        or cleanup.get("request_sha256") != request["request_sha256"]
        or not isinstance(cleanup.get("steps"), list)
    ):
        raise ValueError("agent-terminalized retry lacks bound endpoint cleanup evidence")
    worker_steps = {
        str(step.get("native_id")): step
        for step in cleanup["steps"]
        if isinstance(step, dict) and step.get("role") == "image-worker"
    }
    expected_workers = {
        str(worker["native_id"]): worker["name"] for worker in group["image_workers"]
    }
    if (
        set(worker_steps) != set(expected_workers)
        or any(worker_steps[native_id].get("name") != name
               for native_id, name in expected_workers.items())
        or all(step.get("cleanup") == "canceled" for step in worker_steps.values())
    ):
        raise ValueError("agent-terminalized retry cleanup does not bind the failed worker group")
    outputs = {
        output: _remote_file_sha256(login, pathlib.Path(output), "canonical SDG output")
        for output in request["expected_outputs"]
    }
    body = {
        "kind": "coordinator-cleanup-failure",
        "terminal_sha256": hashlib.sha256(json.dumps(
            terminal, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "cleanup_sha256": hashlib.sha256(json.dumps(
            cleanup, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "expected_outputs_sha256": outputs,
    }
    body["evidence_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return body


def _retry_lineage(
    prior_request_path: pathlib.Path, job_record_path: pathlib.Path, login: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one authoritative failed attempt-1 chain and return its binding."""
    prior_path = _local_regular(prior_request_path, "--retry-from-request")
    prior = load_request(prior_path)
    if prior["attempt"] != 1 or prior.get("retry") is not None:
        raise ValueError("retry source must be an original attempt-1 request")
    record_file = _local_regular(job_record_path, "--retry-from-job-record")
    record = json.loads(record_file.read_text(encoding="utf-8"))
    job_id = record.get("id") if isinstance(record, dict) else None
    if not isinstance(job_id, str) or not SAFE_TOKEN.fullmatch(job_id):
        raise ValueError("retry job record has an invalid attempt-1 job ID")
    if record_file.name != f"{job_id}.json":
        raise ValueError("retry job-record filename must equal the attempt-1 job ID")
    transitions = record.get("transitions") if isinstance(record, dict) else None
    transition_states = [
        item.get("state") for item in transitions if isinstance(item, dict)
    ] if isinstance(transitions, list) else []
    if (record.get("schema_version") != 1 or record.get("id") != job_id
            or record.get("platform") != "slurm"
            or record.get("action") != prior["action_id"]
            or record.get("results_dir") != prior["stage_dir"]
            or not str(record.get("backend_ref", "")).isdigit()
            or record.get("terminal_state") != "ERROR"
            or record.get("err_class") != "ERR_INFRA" or record.get("redacted") is not True
            or record.get("terminal_write_by") not in {"backend-hook", "agent", "poller"}
            or not transition_states or transition_states[0] != "PENDING"
            or "RUNNING" not in transition_states or transition_states[-1] != "ERROR"):
        raise ValueError("retry job record does not prove terminal infrastructure ERROR")
    if not SAFE_TOKEN.fullmatch(login):
        raise ValueError("--retry-login contains unsupported characters")
    group = _remote_job_group(login, prior, job_id)
    group_sha256 = hashlib.sha256(
        json.dumps(group, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    record_backend_ref = str(record["backend_ref"])
    terminal_evidence = None
    if str(group["coordinator"]["native_id"]) != record_backend_ref:
        recovery = _load_duplicate_recovery(login, prior, job_id)
        if (
            recovery.get("record_backend_ref") != record_backend_ref
            or recovery.get("group_backend_ref") != str(group["coordinator"]["native_id"])
            or recovery.get("job_group_sha256") != group_sha256
        ):
            raise ValueError("duplicate-submit recovery does not bind retry ownership")
        native_states = dict(recovery["native_states"])
    else:
        members = [group["coordinator"], *group["image_workers"]]
        native_states = {}
        for member in members:
            native_id = str(member["native_id"])
            _assert_job_ownership(login, member["name"], native_id)
            native_states[native_id] = _native_state(login, native_id)
        if record.get("terminal_write_by") == "agent":
            terminal_evidence = _agent_cleanup_terminal_evidence(
                login, prior, job_id, record_backend_ref, group,
            )
    coordinator_state = native_states[record_backend_ref]
    failure_states = {"FAILED", "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY", "NODE_FAIL", "TIMEOUT"}
    if (coordinator_state not in failure_states
            or any(state not in WORKER_TERMINAL_STATES for state in native_states.values())):
        raise ValueError("retry native accounting is not one exact terminal failed job group")
    lineage = {
        "job_id": job_id, "action_id": prior["action_id"],
        "backend_ref": str(record["backend_ref"]),
        "request_sha256": prior["request_sha256"],
        "job_record_sha256": hashlib.sha256(record_file.read_bytes()).hexdigest(),
        "job_group_sha256": group_sha256,
        "native_states_sha256": hashlib.sha256(
            json.dumps(native_states, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "native_states": native_states,
        "terminal_evidence": terminal_evidence,
    }
    return prior, lineage


def _pool_rebind_repair_lineage(
    prior_request_path: pathlib.Path, job_record_path: pathlib.Path, login: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the one allowlisted repair for an unstarted attempt-2 pool rebind.

    The failed attempt must have reached endpoint readiness but no preprocessing,
    augmentation, labeling, or normalized output. This is a program repair, not a
    third controller retry, and it reuses attempt 2's original retry lineage.
    """
    prior_path = _local_regular(prior_request_path, "--repair-from-request")
    prior = load_request(prior_path)
    if prior["attempt"] != 2 or prior.get("repair") is not None:
        raise ValueError("pool-rebind repair source must be the original attempt-2 request")
    record_file = _local_regular(job_record_path, "--repair-from-job-record")
    record = json.loads(record_file.read_text(encoding="utf-8"))
    job_id = record.get("id") if isinstance(record, dict) else None
    transitions = record.get("transitions") if isinstance(record, dict) else None
    states = [
        item.get("state") for item in transitions if isinstance(item, dict)
    ] if isinstance(transitions, list) else []
    if (
        not isinstance(job_id, str)
        or not SAFE_TOKEN.fullmatch(job_id)
        or record_file.name != f"{job_id}.json"
        or record.get("schema_version") != 1
        or record.get("platform") != "slurm"
        or record.get("action") != prior["action_id"]
        or record.get("results_dir") != prior["stage_dir"]
        or record.get("retry_of") != prior["retry"]["job_id"]
        or not str(record.get("backend_ref", "")).isdigit()
        or record.get("terminal_state") != "ERROR"
        or record.get("redacted") is not True
        or record.get("terminal_write_by") not in {"backend-hook", "agent", "poller"}
        or not states
        or states[0] != "PENDING"
        or "RUNNING" not in states
        or states[-1] != "ERROR"
    ):
        raise ValueError("pool-rebind repair job record is not one terminal attempt-2 error")
    if not SAFE_TOKEN.fullmatch(login):
        raise ValueError("--repair-login contains unsupported characters")
    group = _remote_job_group(login, prior, job_id)
    backend_ref = str(record["backend_ref"])
    if str(group["coordinator"]["native_id"]) != backend_ref:
        raise ValueError("pool-rebind repair coordinator differs from its job record")
    native_states: dict[str, str] = {}
    for member in [group["coordinator"], *group["image_workers"]]:
        native_id = str(member["native_id"])
        _assert_job_ownership(login, member["name"], native_id)
        native_states[native_id] = _native_state(login, native_id)
    if (
        native_states[backend_ref] != "FAILED"
        or any(state not in WORKER_TERMINAL_STATES for state in native_states.values())
    ):
        raise ValueError("pool-rebind repair native job group is not terminal")

    stage = pathlib.Path(prior["stage_dir"])
    terminal_path = stage / f"slurm_sdg_terminal.{job_id}.json"
    cleanup_path = stage / f"endpoint_cleanup.{job_id}.json"
    execute_log_path = stage / "logs" / "shared-sdg-execute.log"
    progress_path = stage / "sdg_progress.json"
    terminal = _remote_json_file(login, terminal_path, "pool-rebind terminal evidence")
    cleanup = _remote_json_file(login, cleanup_path, "pool-rebind cleanup evidence")
    progress = _remote_json_file(login, progress_path, "pool-rebind progress evidence")
    if (
        terminal.get("schema_version") != "1"
        or terminal.get("status") != "error"
        or terminal.get("job_id") != job_id
        or terminal.get("action_id") != prior["action_id"]
        or terminal.get("request_sha256") != prior["request_sha256"]
        or terminal.get("attempt") != 2
        or not str(terminal.get("error", "")).startswith(
            "RuntimeError: shared SDG execute exited 2; inspect "
        )
    ):
        raise ValueError("pool-rebind terminal evidence does not bind the attempt-2 defect")
    worker_steps = [
        item for item in cleanup.get("steps", [])
        if isinstance(item, dict) and item.get("role") == "image-worker"
    ] if isinstance(cleanup, dict) else []
    expected_workers = {str(item["native_id"]) for item in group["image_workers"]}
    if (
        cleanup.get("schema_version") != "1"
        or cleanup.get("job_id") != job_id
        or cleanup.get("action_id") != prior["action_id"]
        or cleanup.get("request_sha256") != prior["request_sha256"]
        or {str(item.get("native_id")) for item in worker_steps} != expected_workers
        or any(item.get("owned") is not True or item.get("cleanup") != "canceled"
               for item in worker_steps)
    ):
        raise ValueError("pool-rebind cleanup does not prove all owned workers terminal")
    if (
        set(progress) != {
            "schema_version", "preprocessed", "augmentation", "split", "labeling",
            "command_attempts", "endpoint_attempts", "image_edit_endpoints",
            "image_edit_endpoint_pool", "image_edit_endpoint_history",
        }
        or progress.get("schema_version") != "1"
        or progress.get("preprocessed") is not False
        or progress.get("augmentation") != {}
        or progress.get("split") is not False
        or progress.get("labeling") != {}
        or progress.get("command_attempts") != {"preprocess:batch:1": 1}
        or progress.get("endpoint_attempts") != {}
        or progress.get("image_edit_endpoint_history") != []
        or not isinstance(progress.get("image_edit_endpoints"), list)
        or not isinstance(progress.get("image_edit_endpoint_pool"), dict)
    ):
        raise ValueError("pool-rebind repair requires exact zero-generation progress")
    execute_log = _require_ok(
        _ssh(login, f"test -s {shlex.quote(str(execute_log_path))} && "
                    f"test ! -L {shlex.quote(str(execute_log_path))} && "
                    f"cat -- {shlex.quote(str(execute_log_path))}"),
        "pool-rebind execute log",
    ).decode(errors="replace")
    lines = [line.strip() for line in execute_log.splitlines() if line.strip()]
    if not lines or lines[-1] != f"run_sdg_stage[execute]: {POOL_REBIND_ERROR}":
        raise ValueError("pool-rebind repair does not match the allowlisted execute error")
    absent = [
        *prior["expected_outputs"],
        str(stage / "accepted_crop_manifest.json"),
        str(stage / "auto_label_smoke_open_qa.json"),
    ]
    checks = " ".join(shlex.quote(path) for path in absent)
    _require_ok(
        _ssh(login, f"set -Eeuo pipefail; for p in {checks}; do test ! -e \"$p\"; done"),
        "pool-rebind zero-output proof",
    )
    group_sha256 = hashlib.sha256(json.dumps(
        group, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    body = {
        "kind": POOL_REBIND_REPAIR_KIND,
        "job_id": job_id,
        "action_id": prior["action_id"],
        "backend_ref": backend_ref,
        "request_sha256": prior["request_sha256"],
        "job_record_sha256": hashlib.sha256(record_file.read_bytes()).hexdigest(),
        "job_group_sha256": group_sha256,
        "native_states_sha256": hashlib.sha256(json.dumps(
            native_states, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "native_states": native_states,
        "terminal_sha256": _remote_file_sha256(login, terminal_path, "terminal evidence"),
        "cleanup_sha256": _remote_file_sha256(login, cleanup_path, "cleanup evidence"),
        "execute_log_sha256": hashlib.sha256(execute_log.encode()).hexdigest(),
        "progress_sha256": _remote_file_sha256(login, progress_path, "progress evidence"),
    }
    body["evidence_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return prior, body


def _native_accounting(login: str, backend_ref: str) -> dict[str, Any]:
    """Return the exact allocation state and elapsed seconds for one native job."""
    if not backend_ref.isdigit():
        raise ValueError("native accounting handle must be numeric")
    output = _require_ok(
        _ssh(
            login,
            f"sacct -j {backend_ref} -X -n -P -o State%40,ElapsedRaw 2>/dev/null",
        ),
        "SLURM reschedule accounting",
    ).decode(errors="replace").strip().splitlines()
    rows = [line for line in output if line.strip()]
    if len(rows) != 1:
        raise ValueError("SLURM reschedule accounting did not return one allocation")
    fields = rows[0].split("|")
    if len(fields) < 2:
        raise ValueError("SLURM reschedule accounting is malformed")
    state = fields[0].strip().split()[0].rstrip("+").upper()
    try:
        elapsed_raw = int(fields[1].strip())
    except ValueError as exc:
        raise ValueError("SLURM reschedule elapsed time is not numeric") from exc
    return {"state": state, "elapsed_raw": elapsed_raw}


def _scheduler_reschedule_lineage(
    prior_request_path: pathlib.Path, job_record_path: pathlib.Path, login: str,
    new_time_minutes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind one shorter resubmit of an attempt-2 repair that never executed."""
    prior_path = _local_regular(prior_request_path, "--reschedule-from-request")
    prior = load_request(prior_path)
    if (
        prior["attempt"] != 2
        or prior.get("retry") is None
        or prior.get("repair") is None
        or prior.get("reschedule") is not None
    ):
        raise ValueError(
            "scheduler reschedule source must be one unrepeated attempt-2 repair"
        )
    prior_minutes = prior["resources"]["time_minutes"]
    if (
        not isinstance(new_time_minutes, int)
        or isinstance(new_time_minutes, bool)
        or not 10 <= new_time_minutes < prior_minutes
    ):
        raise ValueError("scheduler reschedule must strictly shorten the prior walltime")

    record_file = _local_regular(job_record_path, "--reschedule-from-job-record")
    record = json.loads(record_file.read_text(encoding="utf-8"))
    job_id = record.get("id") if isinstance(record, dict) else None
    transitions = record.get("transitions") if isinstance(record, dict) else None
    states = [
        item.get("state") for item in transitions if isinstance(item, dict)
    ] if isinstance(transitions, list) else []
    terminal_state = record.get("terminal_state") if isinstance(record, dict) else None
    terminal_class_ok = (
        (terminal_state == "ERROR" and record.get("err_class") == "ERR_INFRA")
        or (terminal_state == "CANCELED" and record.get("err_class") is None)
    )
    if (
        not isinstance(job_id, str)
        or not SAFE_TOKEN.fullmatch(job_id)
        or record_file.name != f"{job_id}.json"
        or record.get("schema_version") != 1
        or record.get("platform") != "slurm"
        or record.get("action") != prior["action_id"]
        or record.get("results_dir") != prior["stage_dir"]
        or record.get("retry_of") != prior["repair"]["job_id"]
        or not str(record.get("backend_ref", "")).isdigit()
        or not terminal_class_ok
        or record.get("redacted") is not True
        or record.get("terminal_write_by") not in {"backend-hook", "agent", "poller"}
        or not states
        or states[0] != "PENDING"
        or "RUNNING" not in states
        or states[-1] != terminal_state
    ):
        raise ValueError(
            "scheduler reschedule job record is not one terminal repair allocation"
        )
    if not SAFE_TOKEN.fullmatch(login):
        raise ValueError("--reschedule-login contains unsupported characters")

    group = _remote_job_group(login, prior, job_id)
    backend_ref = str(record["backend_ref"])
    if str(group["coordinator"]["native_id"]) != backend_ref:
        raise ValueError("scheduler reschedule coordinator differs from its job record")
    accounting: dict[str, dict[str, Any]] = {}
    for member in [group["coordinator"], *group["image_workers"]]:
        native_id = str(member["native_id"])
        _assert_job_ownership(login, member["name"], native_id)
        accounting[native_id] = _native_accounting(login, native_id)
    if any(
        row != {"state": "CANCELLED", "elapsed_raw": 0}
        for row in accounting.values()
    ):
        raise ValueError(
            "scheduler reschedule requires every owned allocation canceled before execution"
        )

    stage = pathlib.Path(prior["stage_dir"])
    progress_path = stage / "sdg_progress.json"
    progress_sha256 = _remote_file_sha256(
        login, progress_path, "scheduler reschedule progress evidence"
    )
    if progress_sha256 != prior["repair"]["progress_sha256"]:
        raise ValueError("scheduler reschedule found changed SDG progress")
    absent = [
        *prior["expected_outputs"],
        str(stage / f"slurm_sdg_terminal.{job_id}.json"),
        str(stage / f"endpoint_cleanup.{job_id}.json"),
        *(
            str(stage / ".tao-runtime" / "image-workers" / f"{worker['name']}.json")
            for worker in group["image_workers"]
        ),
    ]
    checks = " ".join(shlex.quote(path) for path in absent)
    _require_ok(
        _ssh(login, f"set -Eeuo pipefail; for p in {checks}; do test ! -e \"$p\"; done"),
        "scheduler reschedule zero-execution proof",
    )

    group_sha256 = hashlib.sha256(json.dumps(
        group, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    accounting_sha256 = hashlib.sha256(json.dumps(
        accounting, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    body = {
        "kind": SCHEDULER_RESCHEDULE_KIND,
        "job_id": job_id,
        "action_id": prior["action_id"],
        "backend_ref": backend_ref,
        "request_sha256": prior["request_sha256"],
        "job_record_sha256": hashlib.sha256(record_file.read_bytes()).hexdigest(),
        "job_group_sha256": group_sha256,
        "native_accounting_sha256": accounting_sha256,
        "native_accounting": accounting,
        "prior_time_minutes": prior_minutes,
        "new_time_minutes": new_time_minutes,
        "progress_sha256": progress_sha256,
    }
    body["evidence_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return prior, body


def _image_master_port_repair_lineage(
    prior_request_path: pathlib.Path, job_record_path: pathlib.Path, login: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the one launch repair for vLLM-Omni same-node port collision."""
    prior_path = _local_regular(prior_request_path, "--launch-repair-from-request")
    prior = load_request(prior_path)
    if (
        prior["attempt"] != 2
        or prior.get("retry") is None
        or prior.get("repair") is None
        or prior.get("reschedule") is None
        or prior.get("launch_repair") is not None
    ):
        raise ValueError(
            "image master-port repair source must be one unrepaired attempt-2 reschedule"
        )
    record_file = _local_regular(
        job_record_path, "--launch-repair-from-job-record"
    )
    record = json.loads(record_file.read_text(encoding="utf-8"))
    job_id = record.get("id") if isinstance(record, dict) else None
    transitions = record.get("transitions") if isinstance(record, dict) else None
    states = [
        item.get("state") for item in transitions if isinstance(item, dict)
    ] if isinstance(transitions, list) else []
    terminal_state = record.get("terminal_state") if isinstance(record, dict) else None
    terminal_class_ok = (
        (terminal_state == "ERROR" and record.get("err_class") == "ERR_INFRA")
        or (terminal_state == "CANCELED" and record.get("err_class") is None)
    )
    if (
        not isinstance(job_id, str)
        or not SAFE_TOKEN.fullmatch(job_id)
        or record_file.name != f"{job_id}.json"
        or record.get("schema_version") != 1
        or record.get("platform") != "slurm"
        or record.get("action") != prior["action_id"]
        or record.get("results_dir") != prior["stage_dir"]
        or record.get("retry_of") != prior["reschedule"]["job_id"]
        or not str(record.get("backend_ref", "")).isdigit()
        or not terminal_class_ok
        or record.get("redacted") is not True
        or record.get("terminal_write_by") not in {"backend-hook", "agent", "poller"}
        or not states
        or states[0] != "PENDING"
        or "RUNNING" not in states
        or states[-1] != terminal_state
    ):
        raise ValueError(
            "image master-port repair job record is not one terminal rescheduled group"
        )
    if not SAFE_TOKEN.fullmatch(login):
        raise ValueError("--launch-repair-login contains unsupported characters")

    group = _remote_job_group(login, prior, job_id)
    backend_ref = str(record["backend_ref"])
    if str(group["coordinator"]["native_id"]) != backend_ref:
        raise ValueError("image master-port repair coordinator differs from its job record")
    native_states: dict[str, str] = {}
    for member in [group["coordinator"], *group["image_workers"]]:
        native_id = str(member["native_id"])
        _assert_job_ownership(login, member["name"], native_id)
        native_states[native_id] = _native_state(login, native_id)
    if (
        any(state not in WORKER_TERMINAL_STATES for state in native_states.values())
        or "FAILED" not in native_states.values()
    ):
        raise ValueError(
            "image master-port repair requires a terminal group with a failed worker"
        )

    stage = pathlib.Path(prior["stage_dir"])
    terminal_path = stage / f"slurm_sdg_terminal.{job_id}.json"
    cleanup_path = stage / f"endpoint_cleanup.{job_id}.json"
    terminal = _remote_json_file(login, terminal_path, "launch-repair terminal evidence")
    cleanup = _remote_json_file(login, cleanup_path, "launch-repair cleanup evidence")
    if (
        terminal.get("schema_version") != "1"
        or terminal.get("status") != "error"
        or terminal.get("job_id") != job_id
        or terminal.get("action_id") != prior["action_id"]
        or terminal.get("request_sha256") != prior["request_sha256"]
        or terminal.get("attempt") != 2
        or terminal.get("error") != "InterruptedError: worker received SIGTERM"
    ):
        raise ValueError("image master-port repair terminal evidence is incompatible")
    if (
        cleanup.get("schema_version") != "1"
        or cleanup.get("job_id") != job_id
        or cleanup.get("action_id") != prior["action_id"]
        or cleanup.get("request_sha256") != prior["request_sha256"]
        or not isinstance(cleanup.get("steps"), list)
    ):
        raise ValueError("image master-port repair cleanup evidence is incompatible")

    workers_by_native = {
        str(worker["native_id"]): worker for worker in group["image_workers"]
    }
    failure_evidence: list[dict[str, Any]] = []
    for native_id, state in native_states.items():
        if state != "FAILED":
            continue
        worker = workers_by_native.get(native_id)
        if worker is None:
            raise ValueError("image master-port repair failure is not an image worker")
        worker_log = stage / "slurm-logs" / f"{worker['name']}-{native_id}.err"
        worker_text = _require_ok(
            _ssh(
                login,
                f"test -s {shlex.quote(str(worker_log))} && "
                f"test ! -L {shlex.quote(str(worker_log))} && "
                f"cat -- {shlex.quote(str(worker_log))}",
            ),
            "image master-port worker log",
        ).decode(errors="replace")
        matches = re.findall(
            r"^slurm_sdg_action\[image-worker\]: "
            r"(img-[0-9]{3}-gpu-[0-7]) exited before readiness$",
            worker_text,
            re.MULTILINE,
        )
        if len(matches) != 1:
            raise ValueError("image master-port repair lacks one failed endpoint identity")
        endpoint_id = matches[0]
        expected_prefix = f"img-{int(worker['index']):03d}-gpu-"
        if not endpoint_id.startswith(expected_prefix):
            raise ValueError("failed endpoint does not belong to its exact image worker")
        gpu_id = int(endpoint_id.rsplit("-", 1)[1])
        endpoint_log = stage / "endpoint-logs" / worker["name"] / f"gpu-{gpu_id}.log"
        endpoint_text = _require_ok(
            _ssh(
                login,
                f"test -s {shlex.quote(str(endpoint_log))} && "
                f"test ! -L {shlex.quote(str(endpoint_log))} && "
                f"cat -- {shlex.quote(str(endpoint_log))}",
            ),
            "image master-port endpoint log",
        ).decode(errors="replace")
        if not all(
            marker in endpoint_text
            for marker in (
                "torch.distributed.DistNetworkError:",
                "EADDRINUSE",
                "port: 10787",
            )
        ):
            raise ValueError("failed endpoint is not the allowlisted master-port collision")
        failure_evidence.append({
            "worker_name": worker["name"],
            "native_id": native_id,
            "endpoint_id": endpoint_id,
            "worker_log_sha256": _remote_file_sha256(
                login, worker_log, "image master-port worker log"
            ),
            "endpoint_log_sha256": _remote_file_sha256(
                login, endpoint_log, "image master-port endpoint log"
            ),
        })
    if not failure_evidence:
        raise ValueError("image master-port repair found no allowlisted failed endpoint")

    progress_path = stage / "sdg_progress.json"
    progress_sha256 = _remote_file_sha256(
        login, progress_path, "image master-port progress evidence"
    )
    if progress_sha256 != prior["reschedule"]["progress_sha256"]:
        raise ValueError("image master-port repair found changed SDG progress")
    absent = [
        *prior["expected_outputs"],
        str(stage / "accepted_crop_manifest.json"),
        str(stage / "auto_label_smoke_open_qa.json"),
    ]
    checks = " ".join(shlex.quote(path) for path in absent)
    _require_ok(
        _ssh(login, f"set -Eeuo pipefail; for p in {checks}; do test ! -e \"$p\"; done"),
        "image master-port zero-generation proof",
    )
    descriptors: dict[str, str] = {}
    for worker in group["image_workers"]:
        descriptor = stage / ".tao-runtime" / "image-workers" / f"{worker['name']}.json"
        if _remote_exists(login, descriptor, "image-worker descriptor"):
            descriptors[str(descriptor)] = _remote_file_sha256(
                login, descriptor, "image-worker descriptor"
            )

    group_sha256 = hashlib.sha256(json.dumps(
        group, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    body = {
        "kind": IMAGE_MASTER_PORT_REPAIR_KIND,
        "job_id": job_id,
        "action_id": prior["action_id"],
        "backend_ref": backend_ref,
        "request_sha256": prior["request_sha256"],
        "job_record_sha256": hashlib.sha256(record_file.read_bytes()).hexdigest(),
        "job_group_sha256": group_sha256,
        "native_states_sha256": hashlib.sha256(json.dumps(
            native_states, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "native_states": native_states,
        "terminal_sha256": _remote_file_sha256(
            login, terminal_path, "launch-repair terminal evidence"
        ),
        "cleanup_sha256": _remote_file_sha256(
            login, cleanup_path, "launch-repair cleanup evidence"
        ),
        "failure_evidence": failure_evidence,
        "descriptor_sha256": descriptors,
        "progress_sha256": progress_sha256,
    }
    body["evidence_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return prior, body


def _runtime_rebind_evidence(
    state: dict[str, Any], results: pathlib.Path,
    prior_runtime_sha256: str, current_runtime_sha256: str,
) -> str:
    """Validate and bind the exact approved runtime transition used by repair."""
    lineage = state.get("runtime_lineage")
    if not isinstance(lineage, list) or not lineage:
        raise ValueError("pool-rebind repair runtime change lacks approved lineage")
    record = lineage[-1]
    expected_keys = {
        "schema_version", "sequence", "old_sha256", "new_sha256", "rebound_at",
        "reason", "evidence_path", "evidence_sha256", "plugin_base_version",
        "skill_version",
    }
    if (
        not isinstance(record, dict)
        or set(record) != expected_keys
        or record.get("schema_version") != "1"
        or record.get("sequence") != len(lineage)
        or record.get("old_sha256") != prior_runtime_sha256
        or record.get("new_sha256") != current_runtime_sha256
        or state.get("active_runtime_sha256") != current_runtime_sha256
    ):
        raise ValueError("pool-rebind repair runtime lineage does not bind old-to-new digests")
    evidence = pathlib.Path(str(record.get("evidence_path", "")))
    try:
        evidence.relative_to(results)
    except ValueError as exc:
        raise ValueError("pool-rebind repair runtime evidence escapes results_dir") from exc
    if not evidence.is_file() or evidence.is_symlink():
        raise ValueError("pool-rebind repair runtime evidence is missing or unsafe")
    evidence_sha256 = hashlib.sha256(evidence.read_bytes()).hexdigest()
    try:
        evidence_payload = json.loads(evidence.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("pool-rebind repair runtime evidence is not JSON") from exc
    if (
        evidence_sha256 != record.get("evidence_sha256")
        or not isinstance(evidence_payload, dict)
        or evidence_payload.get("result") != "PASS"
        or evidence_payload.get("runtime_sha256") != current_runtime_sha256
    ):
        raise ValueError("pool-rebind repair runtime evidence does not record validated PASS")
    return hashlib.sha256(json.dumps(
        record, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def _remote_sqsh(path: pathlib.Path, name: str) -> str:
    resolved = _absolute(str(path), name)
    if resolved.suffix != ".sqsh" or not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(resolved)):
        raise ValueError(f"{name} must be one safe absolute prepared .sqsh path")
    return str(resolved)


def prepare_request(args: argparse.Namespace) -> dict[str, Any]:
    """Deterministically author one signed request from committed run state."""
    state_path = _local_regular(args.deft_state, "--deft-state")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (not isinstance(state, dict) or state.get("schema_version") != "3"
            or state.get("workflow") != WORKFLOW):
        raise ValueError("--deft-state is not an initialized IAA DEFT schema-v3 state")
    results = _absolute(state.get("results_dir"), "state.results_dir")
    if state_path != results / "deft_state.json":
        raise ValueError("--deft-state must be the canonical state.results_dir/deft_state.json")
    config_state = state.get("config")
    if not isinstance(config_state, dict) or config_state.get("platform") != "slurm":
        raise ValueError("initialized DEFT state platform must be slurm")
    if state.get("gate_met") is True or state.get("loop_stop_reason") is not None:
        raise ValueError("run is already stopped and cannot prepare another SDG action")
    maximum = state.get("max_iterations")
    if (not isinstance(args.iteration, int) or isinstance(args.iteration, bool)
            or not isinstance(maximum, int) or not 1 <= args.iteration <= maximum):
        raise ValueError("--iteration is outside the initialized run budget")
    phase = (state.get("iterations") or {}).get(f"iter{args.iteration}")
    if not isinstance(phase, dict):
        raise ValueError("iteration is not initialized at a committed stage boundary")
    if phase.get("status") != "in_progress" or phase.get("stage_completed") != "history_select":
        raise ValueError(
            "SDG request requires history_select as the last committed stage; "
            "committed, failed, or out-of-order SDG work is not rerunnable"
        )

    config_path = _local_regular(args.sdg_config, "--sdg-config")
    if config_path != results / "config" / "sdg_config.yaml":
        raise ValueError("--sdg-config must be the immutable run config/sdg_config.yaml")
    if pathlib.Path(str(config_state.get("sdg_config", ""))).resolve() != config_path:
        raise ValueError("state.config.sdg_config does not bind --sdg-config")
    hashes = config_state.get("spec_sha256")
    expected_digest = hashes.get("sdg_config.yaml") if isinstance(hashes, dict) else None
    actual_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if expected_digest != actual_digest:
        raise ValueError("approved sdg_config.yaml digest does not match initialized state")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != "1" or config.get("enabled") is not True:
        raise ValueError("immutable SDG config is not an enabled schema-v1 object")
    endpoints = config.get("endpoints")
    generation = config.get("generation")
    models = config.get("models")
    config_images = config.get("images")
    if not all(isinstance(item, dict) for item in (endpoints, generation, models, config_images)):
        raise ValueError("immutable SDG config is missing endpoint, generation, or model contracts")
    if endpoints.get("ownership") != "managed" or endpoints.get("reuse_requested") is not False:
        raise ValueError("SLURM request preparation requires managed, non-reused endpoints")
    generation_nodes = generation.get("generation_nodes")
    if (not isinstance(generation_nodes, int) or isinstance(generation_nodes, bool)
            or not 1 <= generation_nodes <= 64
            or generation.get("gpus_per_generation_node") != IMAGE_SERVICES_PER_NODE):
        raise ValueError("SDG config must bind 1..64 independent eight-GPU generation nodes")

    local_stage = results / f"iter_{args.iteration}" / "datagen"
    backend_results_raw = getattr(args, "backend_results_dir", None)
    request_results = (
        _absolute(str(backend_results_raw), "--backend-results-dir")
        if backend_results_raw is not None else results
    )
    if request_results.name != results.name:
        raise ValueError("--backend-results-dir must preserve the approved run directory name")
    stage = request_results / f"iter_{args.iteration}" / "datagen"
    runtime_root = _absolute(str(args.runtime_root), "--runtime-root")
    _under(runtime_root, stage / ".tao-runtime", "--runtime-root")
    cache_dir = _absolute(str(args.cache_dir), "--cache-dir")
    backend_dataset_raw = getattr(args, "backend_dataset_root", None)
    dataset_root = _absolute(
        str(backend_dataset_raw) if backend_dataset_raw is not None
        else config_state.get("dataset_root"),
        "--backend-dataset-root" if backend_dataset_raw is not None
        else "state.config.dataset_root",
    )
    output = _absolute(str(args.output), "--output")
    controller_root = local_stage / ".tao-runtime" / "controller"
    _under(output, controller_root, "--output")
    if output.parent.resolve(strict=False) != output.parent:
        raise ValueError("--output parent must not traverse a symlink")
    if output.exists() and (not output.is_file() or output.is_symlink()):
        raise ValueError("--output must not replace a directory, special file, or symlink")
    run_id = results.name
    if not SAFE_TOKEN.fullmatch(run_id):
        raise ValueError("state results directory name cannot form a safe run identity")
    try:
        state_started = dt.datetime.fromisoformat(state["started_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("state.started_at must be one timezone-aware ISO timestamp") from exc
    if state_started.tzinfo is None:
        raise ValueError("state.started_at must include a timezone")
    started_ns = int(state_started.timestamp() * 1_000_000_000)
    identity_seed = json.dumps(
        {"run_id": run_id, "iteration": args.iteration, "sdg_config_sha256": actual_digest},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    action_id = "deft-iaa-sdg-" + hashlib.sha256(identity_seed).hexdigest()[:16]
    runtime_sha256 = state.get(
        "active_runtime_sha256", config_state.get("iaa_deft_bundle_sha256")
    )
    if not isinstance(runtime_sha256, str) or not SHA256.fullmatch(runtime_sha256):
        raise ValueError("initialized state does not bind one active IAA runtime SHA-256")

    prepared_models: dict[str, Any] = {}
    for role in ROLES:
        source = models.get(role)
        if not isinstance(source, dict):
            raise ValueError(f"SDG config model role is missing: {role}")
        prepared_models[role] = {
            "id": source.get("id"), "revision": source.get("revision"),
            "backend": source.get("backend"), "port": source.get("port"),
            "tensor_parallel": 1,
        }
    started_at = state_started.astimezone(dt.timezone.utc).isoformat()
    expected_outputs = [
        str(stage / "dataset" / "sdg_manifest.json"),
        str(stage / "dataset" / "sdg_pairs.json"),
        str(stage / "dataset" / "sdg_image_list.txt"),
        str(stage / "sdg_execution_manifest.json"),
    ]
    payload: dict[str, Any] = {
        "schema_version": "1", "workflow": WORKFLOW, "kind": KIND,
        "platform": "slurm", "name": NAME, "action_id": action_id,
        "started_at": started_at, "started_ns": started_ns,
        "generation_nodes": generation_nodes, "run_id": run_id,
        "iteration": args.iteration, "attempt": 1,
        "results_dir": str(request_results), "stage_dir": str(stage),
        "dataset_root": str(dataset_root),
        "config_path": str(request_results / "config" / "sdg_config.yaml"),
        "config_sha256": actual_digest, "runtime_root": str(runtime_root),
        "runtime_sha256": runtime_sha256, "cache_dir": str(cache_dir),
        "images": {
            "augmentation": _remote_sqsh(args.augmentation_image, "--augmentation-image"),
            "auto_labeling": _remote_sqsh(args.auto_labeling_image, "--auto-labeling-image"),
            "image_edit": _remote_sqsh(args.image_edit_image, "--image-edit-image"),
            "text_serving": _remote_sqsh(args.text_serving_image, "--text-serving-image"),
        },
        "component_sources": {
            "augmentation": config_images.get("augmentation"),
            "auto_labeling": config_images.get("auto_labeling"),
            "image_edit": config_images.get("image_edit_serving"),
            "text_serving": config_images.get("text_serving"),
        },
        "models": prepared_models,
        "resources": {
            "coordinator_nodes": 1, "coordinator_gpus": 2,
            "image_worker_nodes": 1, "image_worker_gpus": 8,
            "image_worker_capacity": 8,
            "image_worker_cpus_per_task": args.image_worker_cpus_per_task,
            "coordinator_cpus_per_task": args.coordinator_cpus_per_task,
            "time_minutes": args.time_minutes,
        },
        "scheduler": {"account": args.account, "partition": args.partition},
        "limits": {
            "startup_timeout_s": endpoints.get("startup_timeout_s"),
            "retry_interval_s": endpoints.get("retry_interval_s"),
            "request_timeout_s": endpoints.get("request_timeout_s"),
            "image_edit_request_timeout_s": generation.get("image_edit_request_timeout_s"),
            "verification_max_attempts": generation.get("verification_max_attempts"),
            "component_max_attempts": 2,
        },
        "forward_env": ["HF_TOKEN"] if config_state.get("requires_hf_token") is True else [],
        "expected_outputs": expected_outputs,
    }
    retry_inputs = (
        getattr(args, "retry_from_request", None),
        getattr(args, "retry_from_job_record", None),
        getattr(args, "retry_login", None),
    )
    repair_inputs = (
        getattr(args, "repair_from_request", None),
        getattr(args, "repair_from_job_record", None),
        getattr(args, "repair_login", None),
    )
    reschedule_inputs = (
        getattr(args, "reschedule_from_request", None),
        getattr(args, "reschedule_from_job_record", None),
        getattr(args, "reschedule_login", None),
    )
    launch_repair_inputs = (
        getattr(args, "launch_repair_from_request", None),
        getattr(args, "launch_repair_from_job_record", None),
        getattr(args, "launch_repair_login", None),
    )
    selected_recoveries = sum(
        any(item is not None for item in group)
        for group in (
            retry_inputs, repair_inputs, reschedule_inputs, launch_repair_inputs,
        )
    )
    if selected_recoveries > 1:
        raise ValueError(
            "retry, pool-rebind repair, scheduler reschedule, and launch repair "
            "are mutually exclusive"
        )
    if any(item is not None for item in launch_repair_inputs):
        if not all(item is not None for item in launch_repair_inputs):
            raise ValueError(
                "image master-port repair requires request, job-record, and login evidence"
            )
        prior, lineage = _image_master_port_repair_lineage(*launch_repair_inputs)
        if output == pathlib.Path(launch_repair_inputs[0]).absolute():
            raise ValueError("image master-port repair must use a distinct output path")
        if _resume_sha256(prior) != _resume_sha256(payload):
            raise ValueError(
                "image master-port repair changed signed workflow or resource semantics"
            )
        payload["attempt"] = 2
        payload["retry"] = prior["retry"]
        payload["repair"] = prior["repair"]
        payload["reschedule"] = prior["reschedule"]
        payload["launch_repair"] = lineage
        payload["action_id"] = "deft-iaa-sdg-" + hashlib.sha256(
            json.dumps(lineage, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
    elif any(item is not None for item in reschedule_inputs):
        if not all(item is not None for item in reschedule_inputs):
            raise ValueError(
                "scheduler reschedule requires request, job-record, and login evidence"
            )
        prior, lineage = _scheduler_reschedule_lineage(
            *reschedule_inputs, args.time_minutes
        )
        if output == pathlib.Path(reschedule_inputs[0]).absolute():
            raise ValueError("scheduler reschedule must use a distinct output path")
        comparable_prior = json.loads(json.dumps(prior))
        comparable_prior["resources"]["time_minutes"] = args.time_minutes
        if _resume_sha256(comparable_prior) != _resume_sha256(payload):
            raise ValueError(
                "scheduler reschedule changed more than the approved shorter walltime"
            )
        payload["attempt"] = 2
        payload["retry"] = prior["retry"]
        payload["repair"] = prior["repair"]
        payload["reschedule"] = lineage
        payload["action_id"] = "deft-iaa-sdg-" + hashlib.sha256(
            json.dumps(lineage, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
    elif any(item is not None for item in repair_inputs):
        if not all(item is not None for item in repair_inputs):
            raise ValueError(
                "pool-rebind repair requires request, job-record, and login evidence"
            )
        prior, lineage = _pool_rebind_repair_lineage(*repair_inputs)
        if output == pathlib.Path(repair_inputs[0]).absolute():
            raise ValueError("pool-rebind repair must use a distinct output path")
        comparable_prior = dict(prior)
        comparable_prior["runtime_sha256"] = payload["runtime_sha256"]
        if _resume_sha256(comparable_prior) != _resume_sha256(payload):
            raise ValueError("repair source does not match current committed workflow state")
        lineage["runtime_rebind_sha256"] = _runtime_rebind_evidence(
            state, results, prior["runtime_sha256"], payload["runtime_sha256"]
        )
        lineage.pop("evidence_sha256", None)
        lineage["evidence_sha256"] = hashlib.sha256(json.dumps(
            lineage, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        payload["attempt"] = 2
        payload["retry"] = prior["retry"]
        payload["repair"] = lineage
        payload["action_id"] = "deft-iaa-sdg-" + hashlib.sha256(
            json.dumps(lineage, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
    elif any(item is not None for item in retry_inputs):
        if not all(item is not None for item in retry_inputs):
            raise ValueError("attempt-2 preparation requires request, job-record, and login evidence")
        prior, lineage = _retry_lineage(*retry_inputs)
        if output == pathlib.Path(retry_inputs[0]).absolute():
            raise ValueError("attempt-2 request must use a distinct output path")
        comparable_prior = json.loads(json.dumps(prior))
        prior_partition = comparable_prior["scheduler"]["partition"]
        new_partition = payload["scheduler"]["partition"]
        comparable_prior["scheduler"]["partition"] = new_partition
        if _resume_sha256(comparable_prior) != _resume_sha256(payload):
            raise ValueError(
                "attempt-1 request does not match current committed workflow state "
                "apart from the selected retry partition"
            )
        lineage["prior_partition"] = prior_partition
        lineage["new_partition"] = new_partition
        payload["attempt"] = 2
        payload["action_id"] = "deft-iaa-sdg-" + hashlib.sha256(
            json.dumps(lineage, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        payload["retry"] = lineage
    payload["request_sha256"] = _canonical_sha256(payload)
    validate_request(payload)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        validate_request(existing)
        if existing != payload:
            raise ValueError("existing --output differs from the deterministic signed request")
        return {"status": "unchanged", "output": str(output), "request": payload}
    _atomic_json(output, payload)
    return {"status": "written", "output": str(output), "request": payload}


def _sanitize(text: str) -> str:
    return SECRET_RE.sub("[REDACTED]", text)


def _load_job_record(
    path: pathlib.Path, request: dict[str, Any], job_id: str, *, require_pending: bool = False,
) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError("--job-record must be an absolute regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("job record must be an object")
    if payload.get("id") != job_id:
        raise ValueError("job record id does not match --job-id")
    if payload.get("platform") != "slurm":
        raise ValueError("job record platform must be slurm")
    if payload.get("action") != request["action_id"]:
        raise ValueError("job record action does not match signed request action_id")
    if request["attempt"] == 2:
        parent = (
            request.get("launch_repair")
            or request.get("reschedule")
            or request.get("repair")
            or request["retry"]
        )
        transitions = payload.get("transitions")
        states = [
            item.get("state") for item in transitions if isinstance(item, dict)
        ] if isinstance(transitions, list) else []
        terminal = payload.get("terminal_state")
        if (payload.get("schema_version") != 1 or payload.get("redacted") is not True
                or payload.get("results_dir") != request["stage_dir"]
                or payload.get("retry_of") != parent["job_id"]
                or not states or states[0] != "PENDING"
                or states[-1] not in {
                    "PENDING", "RUNNING", "COMPLETE", "ERROR", "CANCELED", "UNKNOWN",
                }
                or terminal not in {None, "COMPLETE", "ERROR", "CANCELED"}
                or (terminal is not None and states[-1] != terminal)
                or (terminal is None and states[-1] in {"COMPLETE", "ERROR", "CANCELED"})):
            raise ValueError("attempt-2 job record lacks authoritative retry lineage")
        if require_pending and (
            states[-1] != "PENDING" or terminal is not None or payload.get("backend_ref") is not None
        ):
            raise ValueError("attempt-2 submit requires a fresh pending retry record")
        if payload.get("id") == parent["job_id"]:
            raise ValueError("attempt-2 must use a distinct job record")
    return payload


def _safe_remote(value: pathlib.Path, name: str) -> pathlib.Path:
    path = _absolute(str(value), name)
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(path)):
        raise ValueError(f"{name} contains unsupported remote-path characters")
    return path


def _run(
    argv: Sequence[str], *, input_bytes: bytes | None = None,
    timeout_s: int | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(argv), input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False, timeout=timeout_s,
    )


def _ssh(
    login: str, command: str, *, timeout_s: int | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return _run(["ssh", "-o", "BatchMode=yes", login, command], timeout_s=timeout_s)


def _require_ok(result: subprocess.CompletedProcess[bytes], operation: str) -> bytes:
    if result.returncode != 0:
        detail = _sanitize(result.stderr.decode(errors="replace").strip())
        raise ValueError(f"{operation} failed: {detail or 'no diagnostic output'}")
    return result.stdout


def _exact_job_ids(login: str, job_id: str) -> list[str]:
    quoted = shlex.quote(job_id)
    command = (
        "set -Eeuo pipefail; "
        f"squeue -h --name {quoted} -o '%i'; "
        f"sacct -X -n --name {quoted} -o JobIDRaw 2>/dev/null || true"
    )
    output = _require_ok(_ssh(login, command), "exact SLURM job-name query")
    return sorted({
        line.strip().split(".", 1)[0]
        for line in output.decode(errors="replace").splitlines()
        if line.strip().split(".", 1)[0].isdigit()
    })


def _stage_file(login: str, local: pathlib.Path, remote: pathlib.Path) -> str:
    if not local.is_file() or local.is_symlink() or local.stat().st_size == 0:
        raise ValueError(f"staged input must be a nonempty regular file: {local}")
    digest = hashlib.sha256(local.read_bytes()).hexdigest()
    temporary = pathlib.Path(f"{remote}.tmp.{digest[:16]}")
    _require_ok(_ssh(login, f"install -d -- {shlex.quote(str(remote.parent))}"), "remote directory creation")
    _require_ok(_run(["scp", "-q", "--", str(local), f"{login}:{temporary}"]), "remote file copy")
    command = (
        "set -Eeuo pipefail; "
        f"test -s {shlex.quote(str(temporary))}; "
        f"test \"$(sha256sum {shlex.quote(str(temporary))} | awk '{{print $1}}')\" = {shlex.quote(digest)}; "
        f"mv -f -- {shlex.quote(str(temporary))} {shlex.quote(str(remote))}; "
        f"sha256sum {shlex.quote(str(remote))}"
    )
    output = _require_ok(_ssh(login, command), "remote file promotion").decode().split()
    if not output or output[0] != digest:
        raise ValueError("promoted remote file digest mismatch")
    return digest


def _stage_shared_runtime(
    login: str,
    source: pathlib.Path,
    remote: pathlib.Path,
    expected_digest: str,
) -> dict[str, str]:
    """Stage the exact signed SDG executor tree and verify it remotely."""
    source = source.resolve()
    package = source / "iaa_deft"
    if (not source.is_dir() or source.is_symlink() or not package.is_dir()
            or package.is_symlink() or _python_tree_sha256(package) != expected_digest):
        raise ValueError("local shared SDG runtime disagrees with initialized state")
    files = [source / "run_sdg_stage.py"] + sorted(
        path for path in package.rglob("*.py") if "__pycache__" not in path.parts
    )
    if any(not path.is_file() or path.is_symlink() for path in files):
        raise ValueError("local shared SDG runtime contains an unsafe file")
    staged = {
        str(path.relative_to(source)): _stage_file(
            login, path, remote / path.relative_to(source)
        )
        for path in files
    }
    verifier = (
        "import hashlib,json,pathlib,sys;"
        "root=pathlib.Path(sys.argv[1]);expected=json.loads(sys.argv[2]);"
        "actual={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() "
        "for p in root.rglob('*.py') if '__pycache__' not in p.parts};"
        "sys.exit(0 if actual==expected else 2)"
    )
    command = (
        f"python3 -c {shlex.quote(verifier)} {shlex.quote(str(remote))} "
        f"{shlex.quote(json.dumps(staged, sort_keys=True, separators=(',', ':')))}"
    )
    _require_ok(_ssh(login, command), "remote shared-runtime verification")
    return staged


def _render(request: dict[str, Any], *, mode: str, job_id: str, worker: pathlib.Path,
            remote_request: pathlib.Path, auth_file: pathlib.Path,
            env_file: pathlib.Path | None, account: str | None, partition: str | None,
            worker_index: int | None = None, job_group: pathlib.Path | None = None,
            dependencies: Sequence[str] = (), base_job_id: str | None = None) -> str:
    for value, name in ((account, "account"), (partition, "partition")):
        if value is not None and not SAFE_TOKEN.fullmatch(value):
            raise ValueError(f"{name} contains unsupported characters")
    if mode not in {"image-worker", "coordinator"}:
        raise ValueError("render mode must be image-worker or coordinator")
    template_name = "slurm_sdg_image.sbatch.tmpl" if mode == "image-worker" else "slurm_sdg.sbatch.tmpl"
    template = pathlib.Path(__file__).resolve().parent.parent / "templates" / template_name
    text = template.read_text(encoding="utf-8")
    minutes = request["resources"]["time_minutes"]
    values = {
        "JOB_NAME": job_id,
        "BASE_JOB_ID": job_id if base_job_id is None else base_job_id,
        "CPUS_PER_TASK": str(request["resources"][
            "image_worker_cpus_per_task"
            if mode == "image-worker" else "coordinator_cpus_per_task"
        ]),
        "TIME": f"{minutes // 60:02d}:{minutes % 60:02d}:00",
        "LOG_DIR": str(pathlib.Path(request["stage_dir"]) / "slurm-logs"),
        "SBATCH_ACCOUNT": "" if account is None else f"#SBATCH --account={account}",
        "SBATCH_PARTITION": "" if partition is None else f"#SBATCH --partition={partition}",
        "ENV_FILE": "" if env_file is None else str(env_file),
        "AUTH_FILE": str(auth_file),
        "WORKER": str(worker),
        "REQUEST": str(remote_request),
        "WORKER_INDEX": "" if worker_index is None else str(worker_index),
        "JOB_GROUP": "" if job_group is None else str(job_group),
        "DEPENDENCY": "" if not dependencies else "#SBATCH --dependency=after:" + ":".join(dependencies),
    }
    for key, value in values.items():
        if any(character in value for character in ("\x00", "\n", "\r", '"')):
            raise ValueError(f"render value {key} contains unsupported characters")
        text = text.replace(f"@@{key}@@", value)
    leftovers = re.findall(r"@@[A-Z_]+@@", text)
    if leftovers:
        raise ValueError(f"unrendered template markers: {leftovers}")
    if "--gpus all" in text or "CUDA_VISIBLE_DEVICES" in text:
        raise AssertionError("SLURM SDG render widened or hard-coded GPU visibility")
    return text


def _stage_json(login: str, payload: Any, remote: pathlib.Path) -> str:
    with tempfile.TemporaryDirectory(prefix="tao-sdg-slurm-json-") as directory:
        local = pathlib.Path(directory) / "payload.json"
        local.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return _stage_file(login, local, remote)


def _ensure_remote_auth_file(
    login: str, auth_file: pathlib.Path, *, allow_create: bool = True,
) -> None:
    """Create one run-scoped secret remotely without returning or logging its value."""
    code = (
        "import os,pathlib,secrets,sys;"
        "p=pathlib.Path(sys.argv[1]);p.parent.mkdir(mode=0o700,parents=True,exist_ok=True);"
        "material=secrets.token_urlsafe(48);"
        "fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);"
        "os.write(fd,''.join('export '+name+'='+material+'\\n' for name in sys.argv[2:]).encode());"
        "os.close(fd)"
    )
    quoted_path = shlex.quote(str(auth_file))
    quoted_parent = shlex.quote(str(auth_file.parent))
    command = (
        "set -Eeuo pipefail; umask 077; "
        f"test -d {quoted_parent} && test ! -L {quoted_parent}; "
        f"test \"$(stat -c '%u' {quoted_parent})\" = \"$(id -u)\"; chmod 700 {quoted_parent}; "
        f"if [ ! -e {quoted_path} ] && [ {'1' if allow_create else '0'} = 1 ]; then "
        f"python3 -c {shlex.quote(code)} {quoted_path} "
        "IMAGE_EDIT_API_KEY VLLM_API_KEY VLM_API_KEY LLM_API_KEY OPENAI_API_KEY; fi; "
        f"test -f {quoted_path} && test ! -L {quoted_path}; "
        f"test \"$(stat -c '%a' {quoted_path})\" = 600; "
        f"test \"$(stat -c '%u' {quoted_path})\" = \"$(id -u)\""
    )
    _require_ok(_ssh(login, command), "remote endpoint-auth creation")


def _ensure_remote_intent(
    login: str, path: pathlib.Path, payload: dict[str, Any], *, allow_create: bool = True,
) -> str:
    """Create or verify immutable pre-submit ownership evidence."""
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    digest = hashlib.sha256(data).hexdigest()
    code = (
        "import os,pathlib,stat,sys;"
        "p=pathlib.Path(sys.argv[1]);data=sys.argv[2].encode();create=sys.argv[3]=='1';"
        "p.parent.mkdir(mode=0o700,parents=True,exist_ok=True);"
        "\ndef valid():\n s=p.stat();return p.is_file() and not p.is_symlink() and s.st_uid==os.getuid() and stat.S_IMODE(s.st_mode)==0o600 and p.read_bytes()==data"
        "\nif not create:\n sys.exit(0 if valid() else 4)"
        "\ntry:\n fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.write(fd,data);os.close(fd)"
        "\nexcept FileExistsError:\n sys.exit(0 if valid() else 3)"
    )
    completed = _ssh(
        login,
        f"python3 -c {shlex.quote(code)} {shlex.quote(str(path))} "
        f"{shlex.quote(data.decode())} {'1' if allow_create else '0'}",
    )
    _require_ok(completed, "immutable SLURM submit-intent creation")
    return digest


def _verify_remote_submit_inputs(login: str, request: dict[str, Any]) -> None:
    """Reject missing shared inputs before any SDG staging or scheduler work."""

    code = """
import hashlib
import os
import pathlib
import sys

cache = pathlib.Path(sys.argv[1])
dataset = pathlib.Path(sys.argv[2])
config = pathlib.Path(sys.argv[3])
config_sha = sys.argv[4]
images = [pathlib.Path(value) for value in sys.argv[5:]]

if not cache.is_dir() or cache.is_symlink() or not os.access(cache, os.R_OK | os.X_OK):
    raise SystemExit("cache_dir must be an existing readable non-symlink directory")
if not dataset.is_dir() or dataset.is_symlink() or not os.access(dataset, os.R_OK | os.X_OK):
    raise SystemExit("dataset_root must be an existing readable non-symlink directory")
if not config.is_file() or config.is_symlink():
    raise SystemExit("config_path must be an existing regular non-symlink file")
if hashlib.sha256(config.read_bytes()).hexdigest() != config_sha:
    raise SystemExit("config_path digest differs from the signed request")
for image in images:
    if not image.is_file() or image.is_symlink():
        raise SystemExit(f"prepared image is missing or unsafe: {image}")
    with image.open("rb") as handle:
        if handle.read(4) != b"hsqs":
            raise SystemExit(f"prepared image lacks SquashFS magic: {image}")
""".strip()
    values = [
        request["cache_dir"], request["dataset_root"], request["config_path"],
        request["config_sha256"],
        *(request["images"][key] for key in sorted(request["images"])),
    ]
    command = "python3 -c {} {}".format(
        shlex.quote(code), " ".join(shlex.quote(value) for value in values),
    )
    _require_ok(_ssh(login, command), "remote immutable SDG input preflight")


def _acquire_remote_lock(login: str, path: pathlib.Path, token: str) -> None:
    """Acquire one remote request-scoped lease without a check/create race."""

    code = (
        "import os,pathlib,sys;"
        "p=pathlib.Path(sys.argv[1]);data=(sys.argv[2]+'\\n').encode();"
        "p.parent.mkdir(mode=0o700,parents=True,exist_ok=True);"
        "\ntry:\n fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.write(fd,data);os.close(fd)"
        "\nexcept FileExistsError:\n sys.exit(17)"
    )
    completed = _ssh(
        login,
        f"python3 -c {shlex.quote(code)} {shlex.quote(str(path))} {shlex.quote(token)}",
    )
    if completed.returncode == 17:
        raise ValueError("another packaged SDG submit/recovery invocation holds the request lock")
    _require_ok(completed, "remote SDG request-lock acquisition")


def _release_remote_lock(login: str, path: pathlib.Path, token: str) -> None:
    code = (
        "import pathlib,sys;"
        "p=pathlib.Path(sys.argv[1]);expected=sys.argv[2]+'\\n';"
        "\nif not p.is_file() or p.is_symlink() or p.read_text()!=expected: sys.exit(19)"
        "\np.unlink()"
    )
    _require_ok(
        _ssh(
            login,
            f"python3 -c {shlex.quote(code)} {shlex.quote(str(path))} {shlex.quote(token)}",
        ),
        "remote SDG request-lock release",
    )


def _submit_rendered(login: str, *, rendered: str, remote_script: pathlib.Path,
                     job_name: str, intent_path: pathlib.Path,
                     intent_binding: dict[str, Any], retry_interval_s: int = 1,
                     reconcile_timeout_s: int = 30) -> tuple[str, str, bool, str]:
    with tempfile.TemporaryDirectory(prefix="tao-sdg-slurm-") as directory:
        local_script = pathlib.Path(directory) / "job.sbatch"
        local_script.write_text(rendered, encoding="utf-8")
        repo = pathlib.Path(__file__).resolve().parents[4]
        _require_ok(
            _run([sys.executable, str(repo / "scripts/redact_secrets.py"), "lint", str(local_script)]),
            "secret lint",
        )
        _require_ok(_run(["bash", "-n", str(local_script)]), "local bash syntax")
        script_digest = hashlib.sha256(local_script.read_bytes()).hexdigest()
        intent = dict(intent_binding)
        intent.update({"schema_version": "1", "job_name": job_name, "script_sha256": script_digest})
        matches = _exact_job_ids(login, job_name)
        if len(matches) > 1:
            raise ValueError(f"ambiguous existing SLURM ownership for {job_name}")
        intent_digest = _ensure_remote_intent(
            login, intent_path, intent, allow_create=not matches,
        )
        if len(matches) == 1:
            _assert_job_ownership(login, job_name, matches[0])
            return matches[0], script_digest, True, intent_digest
        staged_digest = _stage_file(login, local_script, remote_script)
        if staged_digest != script_digest:
            raise ValueError("staged SLURM script digest changed after intent creation")
    quoted = shlex.quote(str(remote_script))
    _require_ok(_ssh(login, f"bash -n {quoted}; sbatch --test-only {quoted}"), "remote SLURM validation")
    matches = _exact_job_ids(login, job_name)
    if len(matches) > 1:
        raise ValueError(f"ambiguous SLURM ownership race for {job_name}")
    if len(matches) == 1:
        _assert_job_ownership(login, job_name, matches[0])
        return matches[0], script_digest, True, intent_digest
    completed = _ssh(login, f"sbatch --parsable {quoted}")
    raw = completed.stdout.decode(errors="replace").strip()
    match = JOB_HANDLE.fullmatch(raw) if completed.returncode == 0 else None
    if match is not None:
        return match.group("id"), script_digest, False, intent_digest
    deadline = time.monotonic() + reconcile_timeout_s
    while True:
        matches = _exact_job_ids(login, job_name)
        if len(matches) > 1:
            raise ValueError(f"ambiguous sbatch response for {job_name}")
        if len(matches) == 1:
            _assert_job_ownership(login, job_name, matches[0])
            return matches[0], script_digest, True, intent_digest
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError(f"lost sbatch response did not reconcile {job_name}")
        time.sleep(min(retry_interval_s, remaining))


def _submit_unlocked(args: argparse.Namespace) -> dict[str, Any]:
    request = load_request(args.request)
    if not SAFE_TOKEN.fullmatch(args.job_id):
        raise ValueError("--job-id contains unsupported characters")
    _load_job_record(args.job_record, request, args.job_id, require_pending=True)
    if not SAFE_TOKEN.fullmatch(args.login):
        raise ValueError("--login contains unsupported characters")
    _verify_remote_submit_inputs(args.login, request)
    signed_account = request["scheduler"]["account"]
    signed_partition = request["scheduler"]["partition"]
    if args.account is not None and args.account != signed_account:
        raise ValueError("--account does not match the signed scheduler account")
    if args.partition is not None and args.partition != signed_partition:
        raise ValueError("--partition does not match the signed scheduler partition")
    account = signed_account
    partition = signed_partition
    remote_script = _safe_remote(args.remote_script, "--remote-script")
    runtime_dir = pathlib.Path(request["stage_dir"]) / ".tao-runtime"
    remote_request = runtime_dir / f"sdg.action.{args.job_id}.json"
    remote_worker = runtime_dir / f"slurm_sdg_action.{args.job_id}.py"
    shared_source = (
        pathlib.Path(__file__).resolve().parents[3]
        / "applications" / "tao-run-deft-iaa" / "scripts"
    )
    runtime_files = _stage_shared_runtime(
        args.login,
        shared_source,
        pathlib.Path(request["runtime_root"]),
        request["runtime_sha256"],
    )
    request_digest = _stage_file(args.login, args.request, remote_request)
    worker_digest = _stage_file(args.login, pathlib.Path(__file__).resolve(), remote_worker)
    env_file = _safe_remote(args.env_file, "--env-file") if args.env_file else None
    auth_file = runtime_dir / f"endpoint-auth.{args.job_id}.env"
    # A stage can have one compatible retry.  Keep scheduler ownership evidence
    # immutable per minted job record so a late cleanup/write from another
    # attempt cannot replace the group used by status, logs, or cancel.
    image_owners_path = runtime_dir / f"image-owners.{args.job_id}.json"
    job_group_path = runtime_dir / f"slurm-job-group.{args.job_id}.json"
    expected_names = [
        *(f"{args.job_id}-img-{index:03d}" for index in range(request["generation_nodes"])),
        f"{args.job_id}-coord",
    ]
    existing = {name: _exact_job_ids(args.login, name) for name in expected_names}
    if any(len(matches) > 1 for matches in existing.values()):
        raise ValueError("ambiguous exact SLURM jobs exist before recovery")
    _ensure_remote_auth_file(
        args.login, auth_file, allow_create=not any(existing.values()),
    )
    submitted: list[dict[str, Any]] = []
    script_digests: dict[str, str] = {}
    intent_digests: dict[str, str] = {}
    intent_binding = {
        "request_sha256": request["request_sha256"],
        "action_id": request["action_id"], "attempt": request["attempt"],
        "job_id": args.job_id,
    }
    try:
        for index in range(request["generation_nodes"]):
            name = f"{args.job_id}-img-{index:03d}"
            script_path = pathlib.Path(f"{remote_script}.img-{index:03d}.sbatch")
            rendered = _render(
                request, mode="image-worker", job_id=name, worker=remote_worker,
                remote_request=remote_request, auth_file=auth_file, env_file=env_file,
                account=account, partition=partition, worker_index=index,
            )
            native_id, digest, reconciled, intent_digest = _submit_rendered(
                args.login, rendered=rendered, remote_script=script_path, job_name=name,
                intent_path=runtime_dir / f"submit-intent.{name}.json",
                intent_binding=intent_binding,
                retry_interval_s=request["limits"]["retry_interval_s"],
                reconcile_timeout_s=min(request["limits"]["startup_timeout_s"], 60),
            )
            submitted.append({
                "role": "image-worker", "index": index, "name": name,
                "native_id": native_id, "reconciled": reconciled,
            })
            script_digests[name] = digest
            intent_digests[name] = intent_digest
        _stage_json(args.login, {
            "schema_version": "1", "request_sha256": request["request_sha256"],
            "generation_nodes": request["generation_nodes"], "workers": submitted,
        }, image_owners_path)
        coordinator_name = f"{args.job_id}-coord"
        coordinator_script = pathlib.Path(f"{remote_script}.coord.sbatch")
        rendered = _render(
            request, mode="coordinator", job_id=coordinator_name, worker=remote_worker,
            remote_request=remote_request, auth_file=auth_file, env_file=env_file,
            account=account, partition=partition, job_group=image_owners_path,
            base_job_id=args.job_id,
        )
        backend_ref, digest, reconciled, intent_digest = _submit_rendered(
            args.login, rendered=rendered, remote_script=coordinator_script,
            job_name=coordinator_name,
            intent_path=runtime_dir / f"submit-intent.{coordinator_name}.json",
            intent_binding=intent_binding,
            retry_interval_s=request["limits"]["retry_interval_s"],
            reconcile_timeout_s=min(request["limits"]["startup_timeout_s"], 60),
        )
        coordinator = {
            "role": "coordinator", "name": coordinator_name,
            "native_id": backend_ref, "reconciled": reconciled,
        }
        script_digests[coordinator_name] = digest
        intent_digests[coordinator_name] = intent_digest
        group = {
            "schema_version": "1", "request_sha256": request["request_sha256"],
            "job_id": args.job_id, "coordinator": coordinator, "image_workers": submitted,
        }
        _stage_json(args.login, group, job_group_path)
    except BaseException:
        # Preserve the sidecar, immutable intents, and any exact jobs. A retry
        # with this same job record reconciles them; cancel is a separate,
        # ownership-checked public operation.
        raise
    return {
        "job_id": args.job_id, "backend_ref": backend_ref,
        "coordinator": coordinator, "image_workers": submitted,
        "request_sha256": request_digest, "worker_sha256": worker_digest,
        "runtime_files_sha256": runtime_files,
        "script_sha256": script_digests, "submit_intent_sha256": intent_digests,
        "job_group": str(job_group_path),
    }


def submit(args: argparse.Namespace) -> dict[str, Any]:
    """Serialize all submits for one immutable request/job record on Lustre."""

    request = load_request(args.request)
    if not SAFE_TOKEN.fullmatch(args.job_id) or not SAFE_TOKEN.fullmatch(args.login):
        raise ValueError("submit login/job id contains unsupported characters")
    lock_path = (
        pathlib.Path(request["stage_dir"]) / ".tao-runtime"
        / f"submit.{args.job_id}.lock"
    )
    token = f"{os.getpid()}-{time.time_ns()}-{request['request_sha256'][:16]}"
    _acquire_remote_lock(args.login, lock_path, token)
    try:
        return _submit_unlocked(args)
    finally:
        _release_remote_lock(args.login, lock_path, token)


def _native_state(login: str, backend_ref: str) -> str:
    if not backend_ref.isdigit():
        raise ValueError("--backend-ref must be a numeric SLURM id")
    queued = _require_ok(
        _ssh(login, f"squeue -h -j {backend_ref} -o '%T' 2>/dev/null || true"),
        "SLURM queue status",
    ).decode().strip().splitlines()
    if queued:
        return queued[0].strip().upper()
    accounting = _require_ok(
        _ssh(login, f"sacct -j {backend_ref} -X -n -o State%30 2>/dev/null || true"),
        "SLURM accounting status",
    ).decode().strip().splitlines()
    return accounting[0].split()[0].rstrip("+").upper() if accounting else "UNKNOWN"


def _assert_job_ownership(login: str, job_id: str, backend_ref: str) -> None:
    if not backend_ref.isdigit():
        raise ValueError("--backend-ref must be a numeric SLURM id")
    command = (
        "set -Eeuo pipefail; "
        f"name=$(scontrol show job -o {backend_ref} 2>/dev/null | "
        "sed -n 's/.* JobName=\\([^ ]*\\).*/\\1/p' | head -n1 || true); "
        f"if [ -z \"$name\" ]; then name=$(sacct -j {backend_ref} -X -n -o JobName%128 "
        "2>/dev/null | awk 'NF {print $1; exit}' || true); fi; "
        "printf 'TAO_JOB_NAME=%s\\n' \"$name\""
    )
    output = _require_ok(
        _ssh(login, command),
        "SLURM ownership query",
    ).decode(errors="replace")
    names = re.findall(r"^TAO_JOB_NAME=([^\s]+)$", output, re.MULTILINE)
    if names != [job_id]:
        raise ValueError("SLURM handle is absent or not owned by the signed job name")


def _remote_job_group(login: str, request: dict[str, Any], base_job_id: str) -> dict[str, Any]:
    if not SAFE_TOKEN.fullmatch(base_job_id):
        raise ValueError("job id contains unsupported characters")
    path = (
        pathlib.Path(request["stage_dir"]) / ".tao-runtime"
        / f"slurm-job-group.{base_job_id}.json"
    )
    completed = _ssh(login, f"test -s {shlex.quote(str(path))} && cat -- {shlex.quote(str(path))}")
    payload = json.loads(_require_ok(completed, "SLURM job-group fetch").decode())
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "request_sha256", "job_id", "coordinator", "image_workers",
    }:
        raise ValueError("SLURM job group has missing or unexpected fields")
    if (payload["schema_version"] != "1" or payload["request_sha256"] != request["request_sha256"]
            or payload["job_id"] != base_job_id):
        raise ValueError("SLURM job group belongs to another request")
    coordinator = payload["coordinator"]
    if (not isinstance(coordinator, dict) or set(coordinator) != {
            "role", "name", "native_id", "reconciled"}
            or coordinator.get("role") != "coordinator"
            or coordinator.get("name") != f"{base_job_id}-coord"
            or not str(coordinator.get("native_id", "")).isdigit()
            or not isinstance(coordinator.get("reconciled"), bool)):
        raise ValueError("SLURM coordinator ownership record is invalid")
    workers = payload["image_workers"]
    if not isinstance(workers, list) or len(workers) != request["generation_nodes"]:
        raise ValueError("SLURM job group has the wrong image-worker count")
    for index, worker in enumerate(workers):
        if (not isinstance(worker, dict) or set(worker) != {
                "role", "index", "name", "native_id", "reconciled"}
                or worker.get("role") != "image-worker" or worker.get("index") != index
                or worker.get("name") != f"{base_job_id}-img-{index:03d}"
                or not str(worker.get("native_id", "")).isdigit()):
            raise ValueError("SLURM image-worker ownership record is invalid")
    native_ids = [str(coordinator["native_id"])] + [str(worker["native_id"]) for worker in workers]
    if len(set(native_ids)) != len(native_ids):
        raise ValueError("SLURM job-group native IDs must be distinct")
    return payload


def _remote_json_file(login: str, path: pathlib.Path, name: str) -> dict[str, Any]:
    completed = _ssh(
        login,
        f"test -s {shlex.quote(str(path))} && test ! -L {shlex.quote(str(path))} "
        f"&& cat -- {shlex.quote(str(path))}",
    )
    try:
        payload = json.loads(_require_ok(completed, name).decode())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} root must be an object")
    return payload


def _remote_file_sha256(login: str, path: pathlib.Path, name: str) -> str:
    completed = _ssh(
        login,
        f"test -s {shlex.quote(str(path))} && test ! -L {shlex.quote(str(path))} "
        f"&& sha256sum -- {shlex.quote(str(path))}",
    )
    words = _require_ok(completed, name).decode(errors="replace").split()
    if not words or not SHA256.fullmatch(words[0]):
        raise ValueError(f"{name} returned an invalid SHA-256")
    return words[0]


def _remote_exists(
    login: str, path: pathlib.Path, name: str, *, nonempty_regular: bool = False,
) -> bool:
    """Return remote presence while distinguishing absence from transport failure."""

    quoted = shlex.quote(str(path))
    if nonempty_regular:
        command = (
            f"if [ ! -e {quoted} ] && [ ! -L {quoted} ]; then exit 1; fi; "
            f"test -s {quoted} && test ! -L {quoted} || exit 3"
        )
    else:
        command = f"[ -e {quoted} ] || [ -L {quoted} ]"
    completed = _ssh(login, command)
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    detail = _sanitize(completed.stderr.decode(errors="replace").strip())
    raise ValueError(
        f"{name} presence probe failed with status {completed.returncode}: "
        f"{detail or 'no diagnostic output'}"
    )


def _duplicate_recovery_path(request: dict[str, Any], job_id: str) -> pathlib.Path:
    return (
        pathlib.Path(request["stage_dir"]) / ".tao-runtime"
        / f"duplicate-submit-recovery.{job_id}.json"
    )


def _load_duplicate_recovery(
    login: str, request: dict[str, Any], job_id: str,
) -> dict[str, Any]:
    payload = _remote_json_file(
        login, _duplicate_recovery_path(request, job_id),
        "SLURM duplicate-submit recovery",
    )
    body = dict(payload)
    digest = body.pop("evidence_sha256", None)
    if digest != hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest():
        raise ValueError("SLURM duplicate-submit recovery digest mismatch")
    if (
        set(payload) != {
            "schema_version", "workflow", "kind", "request_sha256", "action_id",
            "attempt", "job_id", "job_record_sha256", "record_backend_ref",
            "job_group_sha256", "group_backend_ref", "worker_native_ids",
            "coordinator_native_ids", "native_states", "submit_intent_sha256",
            "quarantined_terminal", "recorded_at", "evidence_sha256",
        }
        or payload.get("schema_version") != "1"
        or payload.get("workflow") != WORKFLOW
        or payload.get("kind") != "slurm_sdg_duplicate_submit_recovery"
        or payload.get("request_sha256") != request["request_sha256"]
        or payload.get("action_id") != request["action_id"]
        or payload.get("job_id") != job_id
        or payload.get("attempt") != 1
    ):
        raise ValueError("SLURM duplicate-submit recovery identity is invalid")
    native_states = payload.get("native_states")
    coordinator_ids = payload.get("coordinator_native_ids")
    worker_ids = payload.get("worker_native_ids")
    record_backend_ref = payload.get("record_backend_ref")
    group_backend_ref = payload.get("group_backend_ref")
    expected_names = {
        *(f"{job_id}-img-{index:03d}" for index in range(request["generation_nodes"])),
        f"{job_id}-coord",
    }
    submit_intents = payload.get("submit_intent_sha256")
    if (
        not isinstance(payload.get("job_record_sha256"), str)
        or not SHA256.fullmatch(payload["job_record_sha256"])
        or not isinstance(payload.get("job_group_sha256"), str)
        or not SHA256.fullmatch(payload["job_group_sha256"])
        or not isinstance(record_backend_ref, str)
        or not record_backend_ref.isdigit()
        or not isinstance(group_backend_ref, str)
        or not group_backend_ref.isdigit()
        or not isinstance(native_states, dict)
        or len(native_states) != request["generation_nodes"] + 2
        or any(not str(key).isdigit() or value not in WORKER_TERMINAL_STATES
               for key, value in native_states.items())
        or not isinstance(coordinator_ids, list)
        or len(coordinator_ids) != 2
        or len(set(coordinator_ids)) != 2
        or any(not str(value).isdigit() for value in coordinator_ids)
        or record_backend_ref not in coordinator_ids
        or group_backend_ref not in coordinator_ids
        or record_backend_ref == group_backend_ref
        or not isinstance(worker_ids, list)
        or len(worker_ids) != request["generation_nodes"]
        or any(not str(value).isdigit() for value in worker_ids)
        or len(set(worker_ids)) != len(worker_ids)
        or set(worker_ids).intersection(coordinator_ids)
        or set(map(str, coordinator_ids + worker_ids)) != set(native_states)
        or not isinstance(submit_intents, dict)
        or set(submit_intents) != expected_names
        or any(not isinstance(value, str) or not SHA256.fullmatch(value)
               for value in submit_intents.values())
    ):
        raise ValueError("SLURM duplicate-submit recovery native inventory is invalid")
    try:
        recorded_at = dt.datetime.fromisoformat(payload["recorded_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("SLURM duplicate-submit recovery timestamp is invalid") from exc
    if recorded_at.tzinfo is None:
        raise ValueError("SLURM duplicate-submit recovery timestamp must include a timezone")
    quarantined = payload.get("quarantined_terminal")
    if quarantined is not None:
        terminal = pathlib.Path(request["stage_dir"]) / f"slurm_sdg_terminal.{job_id}.json"
        archive = (
            pathlib.Path(request["stage_dir"]) / ".tao-runtime"
            / f"duplicate-submit-evidence.{job_id}" / "terminal.json"
        )
        if (
            not isinstance(quarantined, dict)
            or set(quarantined) != {"original", "archive", "size", "sha256"}
            or quarantined.get("original") != str(terminal)
            or quarantined.get("archive") != str(archive)
            or not isinstance(quarantined.get("size"), int)
            or isinstance(quarantined.get("size"), bool)
            or quarantined["size"] < 1
            or not isinstance(quarantined.get("sha256"), str)
            or not SHA256.fullmatch(quarantined["sha256"])
        ):
            raise ValueError("SLURM duplicate terminal quarantine binding is invalid")
        command = (
            "set -Eeuo pipefail; "
            f"test -s {shlex.quote(str(archive))}; test ! -L {shlex.quote(str(archive))}; "
            f"stat -c '%s' {shlex.quote(str(archive))}; "
            f"sha256sum {shlex.quote(str(archive))}"
        )
        lines = _require_ok(
            _ssh(login, command), "duplicate terminal quarantine verification"
        ).decode().splitlines()
        if (
            len(lines) != 2
            or lines[0].strip() != str(quarantined["size"])
            or not lines[1].split()
            or lines[1].split()[0] != quarantined["sha256"]
        ):
            raise ValueError("SLURM duplicate terminal quarantine changed after recovery")
    return payload


def recover_duplicate_submit(args: argparse.Namespace) -> dict[str, Any]:
    """Terminalize one proven two-coordinator race without accepting its output."""

    if not args.confirm:
        raise ValueError("duplicate-submit recovery requires --confirm")
    request = load_request(args.request)
    if request["attempt"] != 1:
        raise ValueError("duplicate-submit recovery is restricted to attempt 1")
    if not SAFE_TOKEN.fullmatch(args.login) or not SAFE_TOKEN.fullmatch(args.job_id):
        raise ValueError("duplicate-submit recovery login/job id is invalid")
    record = _load_job_record(args.job_record, request, args.job_id)
    backend_ref = str(record.get("backend_ref", ""))
    if not backend_ref.isdigit() or record.get("terminal_state") is not None:
        raise ValueError("duplicate-submit recovery requires the active attempt-1 record")
    runtime_dir = pathlib.Path(request["stage_dir"]) / ".tao-runtime"
    lock_path = runtime_dir / f"submit.{args.job_id}.lock"
    token = f"recovery-{os.getpid()}-{time.time_ns()}-{request['request_sha256'][:16]}"
    _acquire_remote_lock(args.login, lock_path, token)
    try:
        evidence_path = _duplicate_recovery_path(request, args.job_id)
        if _remote_exists(
            args.login, evidence_path, "duplicate-submit recovery",
            nonempty_regular=True,
        ):
            return _load_duplicate_recovery(args.login, request, args.job_id)
        group = _remote_job_group(args.login, request, args.job_id)
        coordinator_name = f"{args.job_id}-coord"
        coordinator_ids = _exact_job_ids(args.login, coordinator_name)
        if (
            len(coordinator_ids) != 2
            or backend_ref not in coordinator_ids
            or str(group["coordinator"]["native_id"]) not in coordinator_ids
            or backend_ref == str(group["coordinator"]["native_id"])
        ):
            raise ValueError("duplicate-submit recovery lacks the exact two-coordinator race")
        worker_ids: list[str] = []
        expected_names = [
            *(f"{args.job_id}-img-{index:03d}" for index in range(request["generation_nodes"])),
            coordinator_name,
        ]
        intent_digests: dict[str, str] = {}
        for index, worker in enumerate(group["image_workers"]):
            ids = _exact_job_ids(args.login, worker["name"])
            if ids != [str(worker["native_id"])]:
                raise ValueError("duplicate-submit recovery worker ownership is ambiguous")
            worker_ids.extend(ids)
        for name in expected_names:
            intent_path = runtime_dir / f"submit-intent.{name}.json"
            intent = _remote_json_file(args.login, intent_path, "SLURM submit intent")
            if (
                set(intent) != {
                    "schema_version", "request_sha256", "action_id", "attempt",
                    "job_id", "job_name", "script_sha256",
                }
                or intent.get("schema_version") != "1"
                or intent.get("request_sha256") != request["request_sha256"]
                or intent.get("action_id") != request["action_id"]
                or intent.get("attempt") != 1
                or intent.get("job_id") != args.job_id
                or intent.get("job_name") != name
                or not isinstance(intent.get("script_sha256"), str)
                or not SHA256.fullmatch(intent["script_sha256"])
            ):
                raise ValueError("duplicate-submit recovery found an invalid submit intent")
            intent_digests[name] = hashlib.sha256(
                (json.dumps(intent, sort_keys=True, separators=(",", ":")) + "\n").encode()
            ).hexdigest()
        for native_id in [*worker_ids, *coordinator_ids]:
            name = coordinator_name if native_id in coordinator_ids else next(
                worker["name"] for worker in group["image_workers"]
                if str(worker["native_id"]) == native_id
            )
            _assert_job_ownership(args.login, name, native_id)
        native_states = {
            native_id: _native_state(args.login, native_id)
            for native_id in [*worker_ids, *coordinator_ids]
        }
        active = [
            native_id for native_id, state in native_states.items()
            if state not in WORKER_TERMINAL_STATES
        ]
        if active:
            _require_ok(
                _ssh(args.login, "scancel " + " ".join(active)),
                "duplicate-submit exact job cancellation",
            )
            deadline = time.monotonic() + 120
            while active and time.monotonic() < deadline:
                time.sleep(2)
                for native_id in list(active):
                    state = _native_state(args.login, native_id)
                    native_states[native_id] = state
                    if state in WORKER_TERMINAL_STATES:
                        active.remove(native_id)
            if active:
                raise ValueError("duplicate-submit jobs did not terminalize after cancellation")
        if any(state not in WORKER_TERMINAL_STATES for state in native_states.values()):
            raise ValueError("duplicate-submit recovery lacks terminal native accounting")
        for output in request["expected_outputs"]:
            if _remote_exists(
                args.login, pathlib.Path(output), "canonical SDG output",
            ):
                raise ValueError("duplicate-submit recovery is forbidden after SDG output exists")
        terminal_path = (
            pathlib.Path(request["stage_dir"])
            / f"slurm_sdg_terminal.{args.job_id}.json"
        )
        terminal_evidence = None
        if _remote_exists(args.login, terminal_path, "coordinator terminal evidence"):
            archive = (
                runtime_dir / f"duplicate-submit-evidence.{args.job_id}"
                / "terminal.json"
            )
            command = (
                "set -Eeuo pipefail; "
                f"install -d -- {shlex.quote(str(archive.parent))}; "
                f"test -f {shlex.quote(str(terminal_path))}; "
                f"test ! -L {shlex.quote(str(terminal_path))}; "
                f"mv -- {shlex.quote(str(terminal_path))} {shlex.quote(str(archive))}; "
                f"stat -c '%s' {shlex.quote(str(archive))}; "
                f"sha256sum {shlex.quote(str(archive))}"
            )
            lines = _require_ok(
                _ssh(args.login, command), "duplicate terminal evidence quarantine"
            ).decode().splitlines()
            if len(lines) != 2:
                raise ValueError("duplicate terminal quarantine evidence is malformed")
            terminal_evidence = {
                "original": str(terminal_path), "archive": str(archive),
                "size": int(lines[0].strip()), "sha256": lines[1].split()[0],
            }
        group_sha256 = hashlib.sha256(
            json.dumps(group, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        payload = {
            "schema_version": "1", "workflow": WORKFLOW,
            "kind": "slurm_sdg_duplicate_submit_recovery",
            "request_sha256": request["request_sha256"],
            "action_id": request["action_id"], "attempt": 1,
            "job_id": args.job_id,
            "job_record_sha256": hashlib.sha256(args.job_record.read_bytes()).hexdigest(),
            "record_backend_ref": backend_ref,
            "job_group_sha256": group_sha256,
            "group_backend_ref": str(group["coordinator"]["native_id"]),
            "worker_native_ids": worker_ids,
            "coordinator_native_ids": coordinator_ids,
            "native_states": native_states,
            "submit_intent_sha256": intent_digests,
            "quarantined_terminal": terminal_evidence,
            "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        payload["evidence_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        _stage_json(args.login, payload, evidence_path)
        _load_duplicate_recovery(args.login, request, args.job_id)
        auth_file = runtime_dir / f"endpoint-auth.{args.job_id}.env"
        _require_ok(
            _ssh(
                args.login,
                f"shred -u {shlex.quote(str(auth_file))} 2>/dev/null || "
                f"rm -f -- {shlex.quote(str(auth_file))}",
            ),
            "duplicate-submit endpoint-auth cleanup",
        )
        return payload
    finally:
        _release_remote_lock(args.login, lock_path, token)


def _cleanup_recovery_path(request: dict[str, Any], job_id: str) -> pathlib.Path:
    return (
        pathlib.Path(request["stage_dir"]) / ".tao-runtime"
        / f"cleanup-recovery.{job_id}.json"
    )


def _load_cleanup_recovery(
    login: str, request: dict[str, Any], job_id: str, group: dict[str, Any],
) -> dict[str, Any]:
    payload = _remote_json_file(
        login, _cleanup_recovery_path(request, job_id), "SLURM cleanup recovery",
    )
    body = dict(payload)
    digest = body.pop("evidence_sha256", None)
    if digest != hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest():
        raise ValueError("SLURM cleanup recovery digest mismatch")
    group_sha256 = hashlib.sha256(json.dumps(
        group, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    expected_ids = {
        str(group["coordinator"]["native_id"]),
        *(str(worker["native_id"]) for worker in group["image_workers"]),
    }
    native_states = payload.get("native_states")
    if (
        set(payload) != {
            "schema_version", "workflow", "kind", "request_sha256", "action_id",
            "attempt", "job_id", "job_record_sha256", "job_group_sha256",
            "coordinator_native_id", "native_states", "terminal_evidence",
            "archived_terminal", "archived_terminal_sha256",
            "recorded_at", "evidence_sha256",
        }
        or payload.get("schema_version") != "1"
        or payload.get("workflow") != WORKFLOW
        or payload.get("kind") != "slurm_sdg_cleanup_recovery"
        or payload.get("request_sha256") != request["request_sha256"]
        or payload.get("action_id") != request["action_id"]
        or payload.get("attempt") != request["attempt"]
        or payload.get("job_id") != job_id
        or payload.get("job_group_sha256") != group_sha256
        or str(payload.get("coordinator_native_id", ""))
            != str(group["coordinator"]["native_id"])
        or not isinstance(payload.get("job_record_sha256"), str)
        or not SHA256.fullmatch(payload["job_record_sha256"])
        or not isinstance(payload.get("archived_terminal_sha256"), str)
        or not SHA256.fullmatch(payload["archived_terminal_sha256"])
        or not isinstance(native_states, dict)
        or set(native_states) != expected_ids
        or any(state not in WORKER_TERMINAL_STATES for state in native_states.values())
        or native_states[str(group["coordinator"]["native_id"])] not in {
            "FAILED", "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY", "NODE_FAIL", "TIMEOUT",
        }
    ):
        raise ValueError("SLURM cleanup recovery identity or terminal inventory is invalid")
    expected_archive = (
        pathlib.Path(request["stage_dir"]) / ".tao-runtime"
        / f"cleanup-recovery-evidence.{job_id}" / "terminal.error.json"
    )
    if (
        payload.get("archived_terminal") != str(expected_archive)
        or _remote_file_sha256(
            login, expected_archive, "archived coordinator terminal evidence",
        ) != payload["archived_terminal_sha256"]
    ):
        raise ValueError("SLURM cleanup recovery archived terminal changed")
    expected_terminal = _agent_cleanup_terminal_evidence(
        login, request, job_id, str(group["coordinator"]["native_id"]), group,
        terminal_path=expected_archive,
    )
    if payload.get("terminal_evidence") != expected_terminal:
        raise ValueError("SLURM cleanup recovery terminal evidence changed")
    try:
        recorded_at = dt.datetime.fromisoformat(payload["recorded_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("SLURM cleanup recovery timestamp is invalid") from exc
    if recorded_at.tzinfo is None:
        raise ValueError("SLURM cleanup recovery timestamp must include a timezone")
    return payload


def _publish_cleanup_recovered_terminal(
    login: str, request: dict[str, Any], job_id: str, recovery: dict[str, Any],
) -> dict[str, Any]:
    terminal_path = pathlib.Path(request["stage_dir"]) / f"slurm_sdg_terminal.{job_id}.json"
    if _remote_exists(login, terminal_path, "coordinator terminal evidence", nonempty_regular=True):
        current = _remote_json_file(login, terminal_path, "coordinator terminal evidence")
        if (
            current.get("status") == "ok"
            and current.get("job_id") == job_id
            and current.get("action_id") == request["action_id"]
            and current.get("request_sha256") == request["request_sha256"]
            and current.get("attempt") == request["attempt"]
            and current.get("cleanup_recovery_sha256") == recovery["evidence_sha256"]
            and current.get("expected_outputs") == request["expected_outputs"]
        ):
            return current
        if current.get("status") != "error":
            raise ValueError("cleanup recovery refuses to replace unrelated terminal evidence")
    archived = _remote_json_file(
        login, pathlib.Path(recovery["archived_terminal"]),
        "archived coordinator terminal evidence",
    )
    recovered = dict(archived)
    recovered["status"] = "ok"
    recovered.pop("error", None)
    recovered["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    recovered["expected_outputs"] = request["expected_outputs"]
    recovered["cleanup_recovery_sha256"] = recovery["evidence_sha256"]
    _stage_json(login, recovered, terminal_path)
    return recovered


def recover_cleanup_failure(args: argparse.Namespace) -> dict[str, Any]:
    """Recover one completed SDG action whose exact worker cleanup timed out."""

    if not args.confirm:
        raise ValueError("cleanup recovery requires --confirm")
    request = load_request(args.request)
    if not SAFE_TOKEN.fullmatch(args.login) or not SAFE_TOKEN.fullmatch(args.job_id):
        raise ValueError("cleanup recovery login/job id is invalid")
    record = _load_job_record(args.job_record, request, args.job_id)
    backend_ref = str(record.get("backend_ref", ""))
    if (
        not backend_ref.isdigit()
        or record.get("terminal_state") is not None
        or record.get("terminal_write_by") is not None
    ):
        raise ValueError("cleanup recovery requires the still-active exact job record")
    group = _remote_job_group(args.login, request, args.job_id)
    if str(group["coordinator"]["native_id"]) != backend_ref:
        raise ValueError("cleanup recovery backend does not match the exact coordinator")
    path = _cleanup_recovery_path(request, args.job_id)
    lock_path = pathlib.Path(f"{path}.lock")
    token = f"cleanup-{os.getpid()}-{time.time_ns()}-{request['request_sha256'][:16]}"
    _acquire_remote_lock(args.login, lock_path, token)
    try:
        if _remote_exists(
            args.login, path, "SLURM cleanup recovery", nonempty_regular=True,
        ):
            recovery = _load_cleanup_recovery(args.login, request, args.job_id, group)
            _publish_cleanup_recovered_terminal(
                args.login, request, args.job_id, recovery,
            )
            return recovery
        terminal_evidence = _agent_cleanup_terminal_evidence(
            args.login, request, args.job_id, backend_ref, group,
        )
        members = [group["coordinator"], *group["image_workers"]]
        for member in members:
            _assert_job_ownership(args.login, member["name"], str(member["native_id"]))
        native_states = {
            str(member["native_id"]): _native_state(args.login, str(member["native_id"]))
            for member in members
        }
        if native_states[backend_ref] not in {
            "FAILED", "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY", "NODE_FAIL", "TIMEOUT",
        }:
            raise ValueError("cleanup recovery requires a terminal failed coordinator")
        pending = {
            str(worker["native_id"]): worker for worker in group["image_workers"]
            if native_states[str(worker["native_id"])] not in WORKER_TERMINAL_STATES
        }
        deadline = time.monotonic() + min(
            request["limits"]["startup_timeout_s"], WORKER_CLEANUP_TIMEOUT_CAP_S,
        )
        while pending and time.monotonic() < deadline:
            for native_id in list(pending):
                worker = pending[native_id]
                _assert_job_ownership(args.login, worker["name"], native_id)
                try:
                    _ssh(args.login, f"scancel {native_id}", timeout_s=SCHEDULER_QUERY_TIMEOUT_S)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(request["limits"]["retry_interval_s"], remaining))
            for native_id in list(pending):
                try:
                    native_states[native_id] = _native_state(args.login, native_id)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    continue
                if native_states[native_id] in WORKER_TERMINAL_STATES:
                    pending.pop(native_id)
        if pending:
            raise ValueError("cleanup recovery workers did not terminalize within the shared deadline")
        terminal_path = pathlib.Path(request["stage_dir"]) / f"slurm_sdg_terminal.{args.job_id}.json"
        archive = (
            pathlib.Path(request["stage_dir"]) / ".tao-runtime"
            / f"cleanup-recovery-evidence.{args.job_id}" / "terminal.error.json"
        )
        archive_command = (
            "set -Eeuo pipefail; "
            f"install -d -- {shlex.quote(str(archive.parent))}; "
            f"test -s {shlex.quote(str(terminal_path))}; test ! -L {shlex.quote(str(terminal_path))}; "
            f"if [ ! -e {shlex.quote(str(archive))} ] && [ ! -L {shlex.quote(str(archive))} ]; then "
            f"cp -- {shlex.quote(str(terminal_path))} {shlex.quote(str(archive))}; fi; "
            f"test -s {shlex.quote(str(archive))}; test ! -L {shlex.quote(str(archive))}; "
            f"sha256sum -- {shlex.quote(str(archive))}"
        )
        archive_words = _require_ok(
            _ssh(args.login, archive_command), "cleanup terminal evidence archive",
        ).decode(errors="replace").split()
        if not archive_words or not SHA256.fullmatch(archive_words[0]):
            raise ValueError("cleanup terminal evidence archive returned an invalid digest")
        payload = {
            "schema_version": "1", "workflow": WORKFLOW,
            "kind": "slurm_sdg_cleanup_recovery",
            "request_sha256": request["request_sha256"],
            "action_id": request["action_id"], "attempt": request["attempt"],
            "job_id": args.job_id,
            "job_record_sha256": hashlib.sha256(args.job_record.read_bytes()).hexdigest(),
            "job_group_sha256": hashlib.sha256(json.dumps(
                group, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest(),
            "coordinator_native_id": backend_ref,
            "native_states": native_states,
            "terminal_evidence": terminal_evidence,
            "archived_terminal": str(archive),
            "archived_terminal_sha256": archive_words[0],
            "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        payload["evidence_sha256"] = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        _stage_json(args.login, payload, path)
        recovery = _load_cleanup_recovery(args.login, request, args.job_id, group)
        _publish_cleanup_recovered_terminal(
            args.login, request, args.job_id, recovery,
        )
        return recovery
    finally:
        _release_remote_lock(args.login, lock_path, token)


def status(args: argparse.Namespace) -> dict[str, Any]:
    request = load_request(args.request)
    _load_job_record(args.job_record, request, args.job_id)
    group = _remote_job_group(args.login, request, args.job_id)
    coordinator = group["coordinator"]
    if coordinator["native_id"] != args.backend_ref:
        raise ValueError("--backend-ref is not the exact owned coordinator")
    _assert_job_ownership(args.login, coordinator["name"], args.backend_ref)
    for worker in group["image_workers"]:
        _assert_job_ownership(args.login, worker["name"], worker["native_id"])
    native = _native_state(args.login, args.backend_ref)
    recovered_cleanup = False
    if native in {"PENDING", "CONFIGURING"}:
        mapped = "PENDING"
    elif native in {"RUNNING", "COMPLETING", "SUSPENDED", "STOPPED"}:
        mapped = "RUNNING"
    elif native == "COMPLETED":
        terminal = pathlib.Path(request["stage_dir"]) / f"slurm_sdg_terminal.{args.job_id}.json"
        command = f"test -s {shlex.quote(str(terminal))} && cat -- {shlex.quote(str(terminal))}"
        completed = _ssh(args.login, command)
        if completed.returncode != 0:
            mapped = "ERROR"
        else:
            evidence = json.loads(completed.stdout.decode())
            mapped = "COMPLETE" if (
                evidence.get("status") == "ok"
                and evidence.get("request_sha256") == request["request_sha256"]
                and evidence.get("job_id") == args.job_id
                and evidence.get("action_id") == request["action_id"]
            ) else "ERROR"
    elif native in {"CANCELLED", "PREEMPTED", "REVOKED"}:
        mapped = "CANCELED"
    elif native in {"FAILED", "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY", "NODE_FAIL", "TIMEOUT"}:
        recovery_path = _cleanup_recovery_path(request, args.job_id)
        if _remote_exists(
            args.login, recovery_path, "SLURM cleanup recovery", nonempty_regular=True,
        ):
            _load_cleanup_recovery(args.login, request, args.job_id, group)
            mapped = "COMPLETE"
            recovered_cleanup = True
        else:
            mapped = "ERROR"
    else:
        mapped = "UNKNOWN"
    result = {
        "status": mapped, "native_state": native, "backend_ref": args.backend_ref,
        "recovered_cleanup": recovered_cleanup,
    }
    if mapped == "COMPLETE" and getattr(args, "local_results_dir", None) is not None:
        result["synchronized"] = _synchronize_controller_outputs(
            args.login, request, args.local_results_dir
        )
    return result


def _dataset_tree_evidence(root: pathlib.Path) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink() or root.resolve() != root:
        raise ValueError("synchronized SDG dataset directory is missing or unsafe")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("synchronized SDG dataset contains a symlink")
    rows = [
        {
            "relative": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    if not rows or any(row["size"] <= 0 for row in rows):
        raise ValueError("synchronized SDG dataset contains an empty file")
    return rows


def _synchronize_controller_dataset_tree(
    login: str, remote_dataset: pathlib.Path, local_results: pathlib.Path, *,
    remote_results: pathlib.Path,
) -> list[dict[str, Any]]:
    """Copy the complete normalized image-text dataset with digest evidence."""

    try:
        relative = remote_dataset.relative_to(remote_results)
    except ValueError as exc:
        raise ValueError("remote SDG dataset escapes signed results_dir") from exc
    local_dataset = local_results / relative
    quoted = shlex.quote(str(remote_dataset))
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
    lines = _require_ok(
        _ssh(login, "bash -c " + shlex.quote(script)),
        "remote SDG dataset tree preflight",
    ).decode().splitlines()
    expected: list[dict[str, Any]] = []
    for line in lines:
        fields = line.split("|", 2)
        if (
            len(fields) != 3 or SHA256.fullmatch(fields[0]) is None
            or not fields[1].isdigit() or int(fields[1]) <= 0
        ):
            raise ValueError("remote SDG dataset tree evidence is malformed")
        try:
            relative_text = base64.b64decode(fields[2], validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("remote SDG dataset path evidence is malformed") from exc
        relative_path = pathlib.PurePosixPath(relative_text)
        if (
            relative_path.is_absolute() or not relative_path.parts
            or ".." in relative_path.parts
        ):
            raise ValueError("remote SDG dataset contains an unsafe path")
        expected.append({
            "relative": relative_path.as_posix(), "size": int(fields[1]),
            "sha256": fields[0],
        })
    if not expected:
        raise ValueError("remote SDG dataset tree is empty")

    local_dataset.parent.mkdir(parents=True, exist_ok=True)
    if local_dataset.parent.is_symlink() or local_dataset.parent.resolve() != local_dataset.parent:
        raise ValueError("local SDG dataset parent is unsafe")
    temporary_parent = pathlib.Path(tempfile.mkdtemp(
        prefix=local_dataset.name + ".", suffix=".sync.tmp", dir=local_dataset.parent
    ))
    try:
        copied = _run(
            ["scp", "-q", "-r", "--", f"{login}:{remote_dataset}", str(temporary_parent)],
            timeout_s=7200,
        )
        _require_ok(copied, "remote SDG dataset tree copy")
        candidate = temporary_parent / remote_dataset.name
        if _dataset_tree_evidence(candidate) != expected:
            raise ValueError("copied SDG dataset tree differs from remote evidence")
        if local_dataset.exists() or local_dataset.is_symlink():
            if not local_dataset.is_dir() or local_dataset.is_symlink():
                raise ValueError("existing controller SDG dataset is unsafe")
            existing = _dataset_tree_evidence(local_dataset)
            expected_by_name = {row["relative"]: row for row in expected}
            for row in existing:
                if expected_by_name.get(row["relative"]) != row:
                    raise ValueError("existing controller SDG dataset differs from remote evidence")
            for row in expected:
                destination = local_dataset / row["relative"]
                if destination.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.parent.is_symlink():
                    raise ValueError("controller SDG dataset destination parent is unsafe")
                os.replace(candidate / row["relative"], destination)
        else:
            os.replace(candidate, local_dataset)
        if _dataset_tree_evidence(local_dataset) != expected:
            raise ValueError("controller SDG dataset differs after synchronization")
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)
    return [
        {
            "kind": "dataset_file", "local": str(local_dataset / row["relative"]),
            "remote": str(remote_dataset / row["relative"]),
            "size": row["size"], "sha256": row["sha256"],
        }
        for row in expected
    ]


def _synchronize_controller_outputs(
    login: str, request: dict[str, Any], local_results_dir: pathlib.Path,
) -> list[dict[str, Any]]:
    """Atomically mirror exact terminal SDG evidence to an Airflow controller."""

    local_results = pathlib.Path(os.path.abspath(local_results_dir.expanduser()))
    if (
        not local_results.is_dir() or local_results.is_symlink()
        or local_results.resolve() != local_results
        or local_results.name != pathlib.Path(request["results_dir"]).name
    ):
        raise ValueError("--local-results-dir is not the approved controller run")
    remote_results = pathlib.Path(request["results_dir"])
    remote_stage = pathlib.Path(request["stage_dir"])
    dataset_rows = _synchronize_controller_dataset_tree(
        login, remote_stage / "dataset", local_results,
        remote_results=remote_results,
    )
    remotes = [
        pathlib.Path(request["expected_outputs"][3]),
        remote_stage / "endpoint_pool.json",
        remote_stage / "endpoint_manifest.json",
        remote_stage / "status" / "sdg-normalize.slurm.status.json",
    ]
    rows: list[dict[str, Any]] = list(dataset_rows)

    def copy_one(remote: pathlib.Path) -> None:
        try:
            relative = remote.relative_to(remote_results)
        except ValueError as exc:
            raise ValueError("remote SDG output escapes signed results_dir") from exc
        local = local_results / relative
        if local.parent.exists() and (
            local.parent.is_symlink() or local.parent.resolve() != local.parent
        ):
            raise ValueError("local SDG synchronization parent is unsafe")
        local.parent.mkdir(parents=True, exist_ok=True)
        quoted = shlex.quote(str(remote))
        evidence = _require_ok(
            _ssh(
                login,
                "set -Eeuo pipefail; "
                f"test -s {quoted}; test ! -L {quoted}; "
                f"stat -c '%s' -- {quoted}; sha256sum -- {quoted}",
            ),
            f"remote SDG output preflight {remote.name}",
        ).decode().splitlines()
        if len(evidence) != 2 or not evidence[0].isdigit():
            raise ValueError("remote SDG output evidence is malformed")
        size = int(evidence[0])
        digest = evidence[1].split()[0] if evidence[1].split() else ""
        if size <= 0 or SHA256.fullmatch(digest) is None:
            raise ValueError("remote SDG output size or digest is invalid")
        descriptor, temporary_raw = tempfile.mkstemp(
            prefix=local.name + ".", suffix=".sync.tmp", dir=local.parent
        )
        os.close(descriptor)
        temporary = pathlib.Path(temporary_raw)
        try:
            copied = _run(
                ["scp", "-q", "--", f"{login}:{remote}", str(temporary)],
                timeout_s=3600,
            )
            _require_ok(copied, f"remote SDG output copy {remote.name}")
            if (
                not temporary.is_file() or temporary.is_symlink()
                or temporary.stat().st_size != size
                or hashlib.sha256(temporary.read_bytes()).hexdigest() != digest
            ):
                raise ValueError("copied SDG output differs from remote evidence")
            os.replace(temporary, local)
        finally:
            temporary.unlink(missing_ok=True)
        rows.append({
            "local": str(local), "remote": str(remote), "size": size,
            "sha256": digest,
        })
    for remote in remotes:
        copy_one(remote)

    local_stage = local_results / remote_stage.relative_to(remote_results)
    endpoint = json.loads((local_stage / "endpoint_manifest.json").read_text())
    status = json.loads(
        (local_stage / "status" / "sdg-normalize.slurm.status.json").read_text()
    )
    job_id = endpoint.get("job_id")
    if not isinstance(job_id, str) or SAFE_TOKEN.fullmatch(job_id) is None:
        raise ValueError("synchronized endpoint manifest has an invalid job id")
    extra_remotes = [
        remote_stage / "logs" / "sdg-normalize.slurm.log",
        remote_stage / "status" / "sdg-normalize.slurm.pre-action.json",
        remote_stage / ".tao-runtime" / f"sdg.action.{job_id}.json",
        remote_stage / f"slurm_sdg_terminal.{job_id}.json",
    ]
    if status.get("log_path") != str(extra_remotes[0]):
        raise ValueError("synchronized SDG status does not bind the canonical remote log")
    pre_action = status.get("pre_action")
    if not isinstance(pre_action, dict) or pre_action.get("path") != str(extra_remotes[1]):
        raise ValueError("synchronized SDG status does not bind the canonical pre-action")
    for remote in extra_remotes:
        copy_one(remote)
    return rows


def logs(args: argparse.Namespace) -> str:
    request = load_request(args.request)
    _load_job_record(args.job_record, request, args.job_id)
    group = _remote_job_group(args.login, request, args.job_id)
    coordinator = group["coordinator"]
    if coordinator["native_id"] != args.backend_ref:
        raise ValueError("--backend-ref is not the exact owned coordinator")
    _assert_job_ownership(args.login, coordinator["name"], args.backend_ref)
    log_dir = pathlib.Path(request["stage_dir"]) / "slurm-logs"
    paths = [
        log_dir / f"{coordinator['name']}-{args.backend_ref}.{suffix}" for suffix in ("out", "err")
    ]
    command = "tail -n {} -- {} 2>/dev/null || true".format(
        args.tail, " ".join(shlex.quote(str(path)) for path in paths)
    )
    return _sanitize(_require_ok(_ssh(args.login, command), "SLURM log fetch").decode(errors="replace"))


def cancel(args: argparse.Namespace) -> dict[str, Any]:
    request = load_request(args.request)
    if not args.confirm:
        raise ValueError("cancel requires --confirm")
    _load_job_record(args.job_record, request, args.job_id)
    group = _remote_job_group(args.login, request, args.job_id)
    owned = [group["coordinator"], *group["image_workers"]]
    if group["coordinator"]["native_id"] != args.backend_ref:
        raise ValueError("--backend-ref is not the exact owned coordinator")
    for record in owned:
        _assert_job_ownership(args.login, record["name"], record["native_id"])
    native_ids = [record["native_id"] for record in owned]
    _require_ok(_ssh(args.login, "scancel " + " ".join(native_ids)), "SLURM group cancel")
    auth_file = (
        pathlib.Path(request["stage_dir"]) / ".tao-runtime"
        / f"endpoint-auth.{args.job_id}.env"
    )
    _require_ok(
        _ssh(
            args.login,
            f"shred -u {shlex.quote(str(auth_file))} 2>/dev/null || rm -f -- {shlex.quote(str(auth_file))}",
        ),
        "SLURM endpoint-auth cleanup",
    )
    return {
        "status": "CANCELED", "backend_ref": args.backend_ref, "job_id": args.job_id,
        "canceled_native_ids": native_ids,
    }


def _endpoint_command(request: dict[str, Any], role: str, *, port: int | None = None) -> list[str]:
    model = request["models"][role]
    endpoint_port = model["port"] if port is None else port
    if role != "image_edit" and endpoint_port != model["port"]:
        raise ValueError("only image-edit replicas may override the signed base port")
    if role == "image_edit" and not model["port"] <= endpoint_port < model["port"] + IMAGE_SERVICES_PER_NODE:
        raise ValueError("image-edit replica port is outside the signed eight-port range")
    image = request["images"]["image_edit" if role == "image_edit" else "text_serving"]
    command = [
        "srun", "--exclusive", "--exact", "--nodes=1", "--ntasks=1",
        f"--gpus={ROLE_GPUS[role]}", f"--cpus-per-task={IMAGE_SERVICE_CPUS}",
        f"--container-image={image}",
        f"--container-mounts={request['cache_dir']}:/root/.cache/huggingface",
    ]
    forwarded = list(request["forward_env"])
    # Every managed vLLM service enforces the same launch-scoped credential.
    # Forward only the variable name; the value remains in the protected
    # endpoint-auth environment and never enters argv or persisted evidence.
    if "VLLM_API_KEY" not in forwarded:
        forwarded.append("VLLM_API_KEY")
    if role == "image_edit" and "MASTER_PORT" not in forwarded:
        # Pass only the environment name. image_worker assigns a distinct,
        # deterministic value to each same-node vLLM-Omni service.
        forwarded.append("MASTER_PORT")
    if forwarded:
        command.append("--container-env=" + ",".join(forwarded))
    if role == "image_edit":
        command += [
            "vllm", "serve", model["id"], "--omni", "--host", "0.0.0.0",
            "--port", str(endpoint_port), "--revision", model["revision"],
            "--served-model-name", model["id"], "--tensor-parallel-size", "1",
        ]
    else:
        command += [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model["id"], "--host", "127.0.0.1", "--port", str(model["port"]),
            "--revision", model["revision"], "--served-model-name", model["id"],
            "--tensor-parallel-size", "1",
        ]
    if "all" in command or any(item.startswith("CUDA_VISIBLE_DEVICES") for item in command):
        raise AssertionError("endpoint command widened scheduler GPU selection")
    return command


def _image_master_port(gpu_id: int) -> int:
    if not 0 <= gpu_id < IMAGE_SERVICES_PER_NODE:
        raise ValueError("image-edit GPU index is outside the per-node service range")
    return IMAGE_MASTER_PORT_BASE + gpu_id * IMAGE_MASTER_PORT_STRIDE


def _request_json(url: str, *, timeout: int, payload: dict[str, Any] | None = None,
                  auth_env: str | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if auth_env is not None:
        token = os.environ.get(auth_env)
        if not token:
            raise ValueError(f"required endpoint auth environment is absent: {auth_env}")
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(
        url, data=data, headers=headers,
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _probe_role(request: dict[str, Any], role: str, *, base_url: str | None = None) -> None:
    model = request["models"][role]
    base = base_url or f"http://127.0.0.1:{model['port']}/v1"
    auth_env = "IMAGE_EDIT_API_KEY" if role == "image_edit" else "VLLM_API_KEY"
    payload = _request_json(
        base + "/models", timeout=request["limits"]["request_timeout_s"], auth_env=auth_env,
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError(f"{role} readiness returned malformed model metadata")
    ids = {item.get("id") for item in payload["data"] if isinstance(item, dict)}
    if model["id"] not in ids:
        raise ValueError(f"{role} readiness reported the wrong model")
    if role != "image_edit":
        response = _request_json(
            base + "/chat/completions", timeout=request["limits"]["request_timeout_s"],
            payload={"model": model["id"], "messages": [{"role": "user", "content": "Reply OK"}], "max_tokens": 2},
            auth_env=auth_env,
        )
        if (not isinstance(response, dict)
                or not isinstance(response.get("choices"), list)
                or not response["choices"]):
            raise ValueError(f"{role} minimal inference returned no choices")


def _local_worker_state(native_id: str, *, timeout_s: int = SCHEDULER_QUERY_TIMEOUT_S) -> str:
    """Return one exact local SLURM allocation state without inspecting steps."""
    if not native_id.isdigit():
        raise ValueError("image-worker native ID must be numeric")
    try:
        queued = _run(
            ["squeue", "-h", "-j", native_id, "-o", "%T"],
            timeout_s=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("image-worker squeue query timed out") from exc
    if queued.returncode != 0:
        raise ValueError("image-worker squeue query failed")
    lines = queued.stdout.decode(errors="replace").strip().splitlines()
    if lines:
        return lines[0].strip().upper().rstrip("+|")
    try:
        accounting = _run(
            ["sacct", "-X", "-n", "-j", native_id, "-o", "State", "-P"],
            timeout_s=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("image-worker sacct query timed out") from exc
    if accounting.returncode != 0:
        raise ValueError("image-worker sacct query failed")
    lines = accounting.stdout.decode(errors="replace").strip().splitlines()
    return lines[0].split()[0].upper().rstrip("+|") if lines else "UNKNOWN"


def _poll_worker_terminations(request: dict[str, Any], records: list[dict[str, Any]]) -> None:
    """Reconcile every issued cancellation against one shared cleanup deadline."""
    timeout = min(request["limits"]["startup_timeout_s"], WORKER_CLEANUP_TIMEOUT_CAP_S)
    deadline = time.monotonic() + timeout
    pending = {
        str(record["native_id"]): record
        for record in records if record["cleanup"] in {"accepted", "uncertain"}
    }
    while pending:
        deadline_reached = False
        for native_id in list(pending):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                deadline_reached = True
                break
            record = pending[native_id]
            try:
                state = _local_worker_state(
                    native_id,
                    timeout_s=max(1, min(SCHEDULER_QUERY_TIMEOUT_S, int(remaining))),
                )
            except (OSError, ValueError) as exc:
                # Scheduler RPC latency is not proof that an accepted cancel
                # failed. Preserve the latest diagnostic and keep polling
                # under the one shared cleanup deadline.
                record["last_query_error"] = _sanitize(str(exc))
                continue
            record["native_state"] = state
            if state in WORKER_TERMINAL_STATES:
                record["cleanup"] = "canceled"
                record.pop("last_query_error", None)
                record.pop("last_cancel_error", None)
                pending.pop(native_id)
                continue
            if record["cleanup"] == "uncertain":
                try:
                    if _local_job_name(native_id) != record["name"]:
                        record["cleanup"] = "failed"
                        record["error"] = "image-worker ownership changed during cancel reconciliation"
                        pending.pop(native_id)
                        continue
                    canceled = _run(
                        ["scancel", native_id], timeout_s=SCHEDULER_QUERY_TIMEOUT_S,
                    )
                    if canceled.returncode == 0:
                        record["cleanup"] = "accepted"
                        record.pop("last_cancel_error", None)
                    else:
                        record["last_cancel_error"] = _sanitize(
                            canceled.stderr.decode(errors="replace")
                            or f"scancel returned {canceled.returncode}"
                        )
                except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
                    record["last_cancel_error"] = _sanitize(str(exc))
        if not pending:
            break
        remaining = deadline - time.monotonic()
        if deadline_reached or remaining <= 0:
            for native_id, record in pending.items():
                state = record.get("native_state", "UNKNOWN")
                record["cleanup"] = "failed"
                record["error"] = (
                    f"image-worker {native_id} cleanup deadline exceeded in state {state}"
                )
                if record.get("last_query_error"):
                    record["error"] += f"; last query: {record['last_query_error']}"
                if record.get("last_cancel_error"):
                    record["error"] += f"; last cancel: {record['last_cancel_error']}"
            break
        time.sleep(min(request["limits"]["retry_interval_s"], remaining))


def _cleanup_image_workers(
    request: dict[str, Any], workers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Cancel all exactly owned workers before polling any of them."""
    records: list[dict[str, Any]] = []
    for worker in workers:
        record = {
            "role": "image-worker", "native_id": worker["native_id"],
            "name": worker["name"], "owned": False, "cleanup": "unverified",
        }
        records.append(record)
    ownership_timeout = min(
        request["limits"]["startup_timeout_s"], WORKER_OWNERSHIP_TIMEOUT_CAP_S,
    )
    ownership_deadline = time.monotonic() + ownership_timeout
    pending = {str(record["native_id"]): record for record in records}
    while pending and time.monotonic() < ownership_deadline:
        for native_id in list(pending):
            record = pending[native_id]
            try:
                observed_name = _local_job_name(native_id)
            except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
                record["last_ownership_error"] = _sanitize(str(exc))
                continue
            if observed_name != record["name"]:
                record["error"] = "image-worker ownership name mismatch"
            else:
                record["owned"] = True
                record.pop("last_ownership_error", None)
            pending.pop(native_id)
        if pending:
            remaining = ownership_deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(request["limits"]["retry_interval_s"], remaining))
    for native_id, record in pending.items():
        record["error"] = f"image-worker {native_id} ownership deadline exceeded"
        if record.get("last_ownership_error"):
            record["error"] += f"; last query: {record['last_ownership_error']}"
    # Do not interleave polling with cancellation. Every worker whose exact
    # ownership was proven receives its cancel before terminal-state polling.
    for record in records:
        if not record["owned"]:
            continue
        try:
            canceled = _run(
                ["scancel", str(record["native_id"])],
                timeout_s=SCHEDULER_QUERY_TIMEOUT_S,
            )
            record["returncode"] = canceled.returncode
            record["cleanup"] = "accepted" if canceled.returncode == 0 else "uncertain"
            if canceled.returncode != 0:
                record["last_cancel_error"] = _sanitize(
                    canceled.stderr.decode(errors="replace")
                    or f"scancel returned {canceled.returncode}"
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            # A transport timeout is not proof that SLURM rejected the cancel.
            # Reconcile state and retry only while exact ownership still holds.
            record["cleanup"] = "uncertain"
            record["last_cancel_error"] = _sanitize(str(exc))
    _poll_worker_terminations(request, records)
    return records


def _wait_readiness(request: dict[str, Any], processes: dict[str, subprocess.Popen[Any]]) -> None:
    deadline = time.monotonic() + request["limits"]["startup_timeout_s"]
    pending = set(processes)
    last: dict[str, str] = {}
    while pending and time.monotonic() < deadline:
        for role in sorted(pending):
            if processes[role].poll() is not None:
                raise RuntimeError(f"{role} endpoint step exited before readiness")
            try:
                _probe_role(request, role)
            except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as exc:
                last[role] = _sanitize(str(exc))
            else:
                pending.remove(role)
        if pending:
            time.sleep(request["limits"]["retry_interval_s"])
    if pending:
        raise TimeoutError(f"endpoint readiness deadline exceeded: {last}")


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _component_command(request: dict[str, Any], action: str, *, source_key: str = "", attempt: int = 1,
                       target_attributes: dict[str, str] | None = None,
                       image_edit_url: str | None = None) -> list[str]:
    stage = request["stage_dir"]
    base = ["srun", "--overlap", "--exact", "--nodes=1", "--ntasks=1", "--cpus-per-task=4"]
    urls = {role: f"http://127.0.0.1:{request['models'][role]['port']}/v1" for role in ROLES}
    if action in {"preprocess", "augment", "split"}:
        command = base + [
            f"--container-image={request['images']['augmentation']}",
            f"--container-mounts={stage}/source_ids:/app/data/in:ro,{stage}:/app/data/out",
            "--container-workdir=/app", "env",
            "UV_CACHE_DIR=/app/data/out/.tao-runtime/uv-cache",
            "uv", "run", "--frozen", "--no-sync",
        ]
        if action == "preprocess":
            return command + ["python", "modules/data_processing/combine_panes.py", "/app/data/in", "/app/data/out/panes"]
        if action == "split":
            return command + [
                "python", "modules/data_processing/create_PAS_augmented_dataset.py",
                "--base-dir", "/app/data/out/panes", "--augmented-folders", "/app/data/out/accepted",
                "--output-dir", "/app/data/out/augmented_dataset", "--output-json", "augmented_data.json",
            ]
        if not SAFE_SOURCE.fullmatch(source_key):
            raise ValueError("augment source_key is unsafe")
        if image_edit_url is None:
            raise ValueError("augment requires one pool-bound image-edit URL")
        output = f"/app/data/out/augmentation/{source_key}/attempt_{attempt}"
        command.insert(
            6,
            "--container-env=IMAGE_EDIT_API_KEY,VLM_API_KEY,LLM_API_KEY,OPENAI_API_KEY",
        )
        command += [
            "modules/cli.py", "--config", "configs/config_image_edit_verification.yaml",
            f"data.0.inputs.rgb=/app/data/out/panes/{source_key}.jpg",
            f"data.0.output.video={output}/output.jpg", f"data.0.output.caption={output}/output.txt",
            f"data.0.output.metadata={output}/output_metadata.json", "pipeline.retry=0",
            f"pipeline.request_timeout={request['limits']['image_edit_request_timeout_s']}",
            f"endpoints.vlm.url={urls['vlm']}", f"endpoints.vlm.model={request['models']['vlm']['id']}",
            f"endpoints.llm.url={urls['llm']}", f"endpoints.llm.model={request['models']['llm']['id']}",
            f"endpoints.image_edit.url={image_edit_url}",
            f"endpoints.image_edit.model={request['models']['image_edit']['id']}",
        ]
        for attribute, value in sorted((target_attributes or {}).items()):
            if (attribute not in EDITABLE_ATTRIBUTES or not isinstance(value, str)
                    or not value.strip() or any(c in value for c in "[]\n\r")):
                raise ValueError("target attribute contains unsupported characters")
            command.append(f"captioning.llm.variables.{attribute.replace(' ', '_')}=[{value}]")
        return command
    if action == "label":
        if not SAFE_SOURCE.fullmatch(source_key):
            raise ValueError("label source_key is unsafe")
        return base + [
            # The pinned auto-labeling image keeps its managed Python below
            # /root.  Pyxis root remapping maps only this submitting job user
            # to container root; it grants no host privilege and preserves
            # host ownership for the explicitly mounted run directory.
            "--container-remap-root", "--no-container-mount-home",
            "--container-env=VLM_API_KEY,LLM_API_KEY,OPENAI_API_KEY",
            f"--container-image={request['images']['auto_labeling']}",
            f"--container-mounts={stage}/label_inputs:/input:ro,{stage}:/output",
            "--container-workdir=/workspace", "env",
            "UV_CACHE_DIR=/output/.tao-runtime/uv-cache",
            "uv", "run", "--frozen", "--no-sync", "python", "modules/cli.py",
            "--config", "configs/pipeline_example.yaml", "super_resolution.enabled=false",
            "detection_and_tracking.enabled=false", "vlm_json.enabled=false", "mcq_generation.enabled=true",
            "mcq_generation.mode=question-driven-vlm-llm",
            "mcq_generation.window_metadata_extraction.single_window=true",
            "mcq_generation.window_metadata_extraction.vlm_verify_enabled=false",
            "mcq_generation.window_metadata_extraction.question_bank_file=/workspace/cookbooks/person_attributes/question_bank.json",
            "mcq_generation.window_metadata_extraction.qd_vlm_scene_prompt_template_file=/workspace/cookbooks/person_attributes/prompts/mcq/question_driven_vlm_llm/vlm_scene_prompt_template.md",
            f"endpoints.vlm.url={urls['vlm']}", f"endpoints.vlm.model={request['models']['vlm']['id']}",
            f"endpoints.llm.url={urls['llm']}", f"endpoints.llm.model={request['models']['llm']['id']}",
            f"data.0.inputs.video_path=/input/{source_key}.jpg",
            f"data.0.output.out_dir=/output/labels/{source_key}",
        ]
    raise ValueError(f"unsupported component action: {action}")


def _load_endpoint_pool(request: dict[str, Any], path: pathlib.Path) -> dict[str, Any]:
    stage = pathlib.Path(request["stage_dir"])
    if path != stage / "endpoint_pool.json" or not path.is_file() or path.is_symlink():
        raise ValueError("image-edit pool must be the run-owned regular endpoint_pool.json")
    return _validate_endpoint_pool_payload(request, json.loads(path.read_text(encoding="utf-8")))


def _validate_endpoint_pool_payload(request: dict[str, Any], payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "platform", "model", "required_capacity", "auth_env",
        "endpoints", "created_at", "request_sha256",
    }:
        raise ValueError("endpoint pool has missing or unexpected fields")
    if payload["schema_version"] != "1" or payload["platform"] != "slurm":
        raise ValueError("endpoint pool identity is invalid")
    if not isinstance(payload["created_at"], str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", payload["created_at"]
    ):
        raise ValueError("endpoint pool created_at must be UTC ISO-8601")
    if payload["request_sha256"] != request["request_sha256"]:
        raise ValueError("endpoint pool belongs to another immutable request")
    model = payload["model"]
    expected_model = request["models"]["image_edit"]
    if model != {"id": expected_model["id"], "revision": expected_model["revision"]}:
        raise ValueError("endpoint pool image-edit model binding is invalid")
    maximum = request["generation_nodes"] * IMAGE_SERVICES_PER_NODE
    required = payload["required_capacity"]
    if (
        not isinstance(required, int) or isinstance(required, bool)
        or not IMAGE_SERVICES_PER_NODE <= required <= maximum
        or required % IMAGE_SERVICES_PER_NODE
        or payload["auth_env"] != "IMAGE_EDIT_API_KEY"
    ):
        raise ValueError("endpoint pool capacity or auth contract is invalid")
    endpoints = payload["endpoints"]
    if not isinstance(endpoints, list) or len(endpoints) != required:
        raise ValueError("endpoint pool service count must equal its active capacity")
    ids: list[str] = []
    worker_indices: list[int] = []
    urls: set[str] = set()
    native_ids: set[str] = set()
    worker_names: dict[int, str] = {}
    worker_native_ids: dict[int, str] = {}
    for position, endpoint in enumerate(endpoints):
        if not isinstance(endpoint, dict) or set(endpoint) != {
            "id", "url", "capacity", "gpu_identity", "owner",
        }:
            raise ValueError("endpoint pool service has missing or unexpected fields")
        owner = endpoint["owner"]
        if not isinstance(owner, dict) or set(owner) != {"native_id", "name"}:
            raise ValueError("endpoint pool owner is invalid")
        if (endpoint["capacity"] != 1 or not str(owner["native_id"]).isdigit()
                or not SAFE_TOKEN.fullmatch(str(owner["name"]))):
            raise ValueError("endpoint pool service capacity or ownership is invalid")
        url = endpoint["url"]
        if not isinstance(url, str) or not re.fullmatch(r"http://[A-Za-z0-9_.-]+:[0-9]{4,5}/v1", url):
            raise ValueError("endpoint pool URL is invalid")
        endpoint_match = re.fullmatch(r"img-(\d{3})-gpu-([0-7])", str(endpoint["id"]))
        if endpoint_match is None:
            raise ValueError("endpoint pool service ID is invalid")
        actual_node, actual_gpu = map(int, endpoint_match.groups())
        if actual_node >= request["generation_nodes"]:
            raise ValueError("endpoint pool worker index exceeds generation_nodes")
        _, expected_gpu = divmod(position, IMAGE_SERVICES_PER_NODE)
        if actual_gpu != expected_gpu:
            raise ValueError("endpoint pool service GPU ordering is invalid")
        if expected_gpu == 0:
            if worker_indices and actual_node <= worker_indices[-1]:
                raise ValueError("endpoint pool worker indices must be strictly ordered")
            worker_indices.append(actual_node)
        elif actual_node != worker_indices[-1]:
            raise ValueError("endpoint pool worker block is incomplete")
        ids.append(endpoint["id"])
        if (not isinstance(endpoint["gpu_identity"], str)
                or not endpoint["gpu_identity"].endswith(f"/gpu-{expected_gpu}")
                or not owner["name"].endswith(f"-img-{actual_node:03d}")):
            raise ValueError("endpoint pool GPU or worker ordering is invalid")
        prior_name = worker_names.setdefault(actual_node, owner["name"])
        prior_native = worker_native_ids.setdefault(actual_node, str(owner["native_id"]))
        if prior_name != owner["name"] or prior_native != str(owner["native_id"]):
            raise ValueError("endpoint pool services disagree on worker ownership")
        if url in urls:
            raise ValueError("endpoint pool URLs must be distinct")
        urls.add(url)
        native_ids.add(str(owner["native_id"]))
    expected_ids = [
        f"img-{node:03d}-gpu-{gpu}" for node in worker_indices
        for gpu in range(IMAGE_SERVICES_PER_NODE)
    ]
    if ids != expected_ids or len(native_ids) != len(worker_indices):
        raise ValueError("endpoint pool ordering or worker ownership is invalid")
    return payload


def component(args: argparse.Namespace) -> int:
    """Translate one shared-runtime component call into one fixed Pyxis step."""
    request = load_request(args.request)
    if not SAFE_TOKEN.fullmatch(args.job_id):
        raise ValueError("--job-id contains unsupported characters")
    stage = pathlib.Path(request["stage_dir"])
    input_root = _absolute(str(args.input_root), "--input-root")
    output_root = _absolute(str(args.output_root), "--output-root")
    if output_root != stage:
        raise ValueError("--output-root must equal the signed SDG stage directory")
    expected_input = stage / ("label_inputs" if args.action == "label" else "source_ids")
    if input_root != expected_input:
        raise ValueError(f"--input-root for {args.action} must be {expected_input}")
    source_key = args.source_key or ""
    if args.action in {"augment", "label"}:
        if not SAFE_SOURCE.fullmatch(source_key):
            raise ValueError(f"{args.action} requires a safe --source-key")
    elif source_key:
        raise ValueError(f"{args.action} does not accept --source-key")
    if not 1 <= args.attempt <= request["limits"]["verification_max_attempts"]:
        raise ValueError("--attempt exceeds the signed verification bound")
    try:
        target_attributes = json.loads(args.target_attributes_json)
    except json.JSONDecodeError as exc:
        raise ValueError("--target-attributes-json is invalid") from exc
    canonical = json.dumps(target_attributes, sort_keys=True, separators=(",", ":"))
    if canonical != args.target_attributes_json or not isinstance(target_attributes, dict):
        raise ValueError("--target-attributes-json must be one canonical object")
    if args.action != "augment" and target_attributes:
        raise ValueError("target attributes are valid only for augment")
    image_edit_url = None
    if args.action == "augment":
        pool = _load_endpoint_pool(request, pathlib.Path(request["stage_dir"]) / "endpoint_pool.json")
        matches = [
            endpoint for endpoint in pool["endpoints"]
            if endpoint["id"] == args.image_edit_endpoint_id
        ]
        if len(matches) != 1 or matches[0]["url"] != args.image_edit_url:
            raise ValueError("selected image-edit endpoint is not bound to the signed pool")
        image_edit_url = matches[0]["url"]
    elif args.image_edit_endpoint_id is not None or args.image_edit_url is not None:
        raise ValueError("image-edit endpoint selection is valid only for augment")
    command = _component_command(
        request, args.action, source_key=source_key, attempt=args.attempt,
        target_attributes=target_attributes, image_edit_url=image_edit_url,
    )
    return subprocess.run(command, check=False).returncode


def _resolve_runtime_python(request: dict[str, Any]) -> pathlib.Path:
    results = pathlib.Path(request["results_dir"])
    workspace = results.parent.parent
    probe = (
        "import pandas,numpy,pyarrow,PIL,yaml,matplotlib,sklearn,torch"
    )
    failures: list[str] = []
    for candidate in (workspace / ".venv/bin/python", workspace / ".venv/bin/python3"):
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            failures.append(f"{candidate}: absent or not executable")
            continue
        completed = _run([str(candidate), "-c", probe])
        if completed.returncode == 0:
            return candidate
        failures.append(f"{candidate}: dependency probe exited {completed.returncode}")
    raise ValueError(
        "signed workspace has no approved IAA runtime interpreter; " + "; ".join(failures)
    )


def _execute_shared(
    request: dict[str, Any], request_path: pathlib.Path, job_id: str,
    runtime_python: pathlib.Path,
) -> None:
    runtime = pathlib.Path(request["runtime_root"])
    runner = runtime / "run_sdg_stage.py"
    if not runner.is_file() or runner.is_symlink():
        raise ValueError(f"staged shared SDG runtime is missing or unsafe: {runner}")
    stage = pathlib.Path(request["stage_dir"])
    iteration = request["iteration"]
    results = pathlib.Path(request["results_dir"])
    dataset = pathlib.Path(request["dataset_root"])
    mined_pairs = results / f"iter_{iteration}" / "mining" / "mined_pairs.json"
    gaps = results / f"iter_{iteration}" / "gaps" / "kpi_gaps.parquet"
    eval_list = results / "iaa_splits" / "eval_list.txt"
    eval_pairs = results / "iaa_splits" / "eval_pairs.json"
    attribute_vocab = dataset / "attribute_vocab.json"
    prepare = [
        str(runtime_python), str(runner), "prepare",
        "--config", request["config_path"],
        "--output-root", request["stage_dir"],
        "--mined-pairs", str(mined_pairs),
        "--gaps-parquet", str(gaps),
        "--attribute-vocab", str(attribute_vocab),
        "--dataset-root", str(dataset),
        "--eval-list", str(eval_list),
        "--eval-pairs", str(eval_pairs),
    ]
    execute = [
        str(runtime_python), str(runner), "execute",
        "--config", request["config_path"],
        "--output-root", request["stage_dir"],
        "--mined-pairs", str(mined_pairs),
        "--eval-list", str(eval_list),
        "--attribute-vocab", str(attribute_vocab),
        "--component-executor", str(pathlib.Path(__file__).resolve()),
        "--component-executor-request", str(request_path),
        "--component-executor-job-id", job_id,
        "--execution-platform", "slurm",
        "--image-edit-endpoint-pool", str(stage / "endpoint_pool.json"),
    ]
    if request.get("repair", {}).get("kind") == POOL_REBIND_REPAIR_KIND:
        execute.append("--explicit-unstarted-pool-rebind")
    for operation, command in (("prepare", prepare), ("execute", execute)):
        log_path = stage / "logs" / f"shared-sdg-{operation}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            completed = subprocess.run(
                command, stdout=handle, stderr=subprocess.STDOUT, check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"shared SDG {operation} exited {completed.returncode}; inspect {log_path}"
            )
    missing = [path for path in request["expected_outputs"] if not pathlib.Path(path).is_file()]
    if missing:
        raise ValueError(f"shared SDG execute missed canonical outputs: {missing}")


def _slurm_identity(expected_name: str) -> tuple[str, str]:
    native_id = os.environ.get("SLURM_JOB_ID", "")
    native_name = os.environ.get("SLURM_JOB_NAME", "")
    if not native_id.isdigit() or native_name != expected_name:
        raise ValueError("runtime SLURM identity does not match the owned job")
    node = socket.getfqdn()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}", node):
        raise ValueError("SLURM node hostname is unsafe for endpoint publication")
    return native_id, node


def _python_tree_sha256(root: pathlib.Path) -> str:
    files = sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    if not files:
        raise ValueError(f"staged shared runtime contains no Python files: {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _verify_signed_inputs(request: dict[str, Any]) -> dict[str, Any]:
    config_path = pathlib.Path(request["config_path"])
    if (not config_path.is_file() or config_path.is_symlink()
            or hashlib.sha256(config_path.read_bytes()).hexdigest() != request["config_sha256"]):
        raise ValueError("signed immutable SDG config is missing, unsafe, or changed")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("images"), dict):
        raise ValueError("signed immutable SDG config is invalid")
    expected_sources = {
        "augmentation": config["images"].get("augmentation"),
        "auto_labeling": config["images"].get("auto_labeling"),
        "image_edit": config["images"].get("image_edit_serving"),
        "text_serving": config["images"].get("text_serving"),
    }
    if request["component_sources"] != expected_sources:
        raise ValueError("signed component source images disagree with immutable SDG config")
    runtime = pathlib.Path(request["runtime_root"])
    runtime_package = runtime / "iaa_deft"
    if (not runtime.is_dir() or runtime.is_symlink() or not runtime_package.is_dir()
            or runtime_package.is_symlink()
            or _python_tree_sha256(runtime_package) != request["runtime_sha256"]):
        raise ValueError("staged shared runtime digest disagrees with initialized state")
    for role, raw in request["images"].items():
        path = pathlib.Path(raw)
        if not path.is_file() or path.is_symlink() or path.stat().st_size < 4:
            raise ValueError(f"prepared runtime SQSH is missing or invalid: {role}")
        with path.open("rb") as handle:
            magic = handle.read(4)
        if magic != b"hsqs":
            raise ValueError(f"prepared runtime SQSH is missing or invalid: {role}")
    return config


def _component_evidence(request: dict[str, Any], immutable_config: dict[str, Any]) -> dict[str, Any]:
    return {
        component: {
            "image": immutable_config["images"][component],
            "runtime_image": request["images"][component], "present": True,
            "conversion": {
                "source_image": immutable_config["images"][component],
                "sqsh_path": request["images"][component],
                "sqsh_magic_verified": True,
            },
        }
        for component in ("augmentation", "auto_labeling")
    }


def image_worker(args: argparse.Namespace) -> int:
    request = load_request(args.request)
    _verify_signed_inputs(request)
    if not 0 <= args.worker_index < request["generation_nodes"]:
        raise ValueError("--worker-index is outside generation_nodes")
    expected_name = f"{args.job_id.rsplit('-img-', 1)[0]}-img-{args.worker_index:03d}"
    if args.job_id != expected_name:
        raise ValueError("image-worker job name does not match its signed index")
    native_id, node = _slurm_identity(args.job_id)
    stage = pathlib.Path(request["stage_dir"])
    processes: dict[str, subprocess.Popen[Any]] = {}
    logs: dict[str, Any] = {}
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError("image worker stopped")))
    try:
        base_port = request["models"]["image_edit"]["port"]
        api_ports = list(range(base_port, base_port + IMAGE_SERVICES_PER_NODE))
        master_ports = [_image_master_port(gpu_id) for gpu_id in range(IMAGE_SERVICES_PER_NODE)]
        occupied = [
            port for port in [*api_ports, *master_ports] if not _port_available(port)
        ]
        if occupied:
            raise RuntimeError(f"image-worker port conflict: {occupied}")
        for gpu_id in range(IMAGE_SERVICES_PER_NODE):
            endpoint_id = f"img-{args.worker_index:03d}-gpu-{gpu_id}"
            log_path = stage / "endpoint-logs" / args.job_id / f"gpu-{gpu_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            logs[endpoint_id] = handle
            environment = dict(os.environ)
            environment["MASTER_PORT"] = str(_image_master_port(gpu_id))
            processes[endpoint_id] = subprocess.Popen(
                _endpoint_command(request, "image_edit", port=base_port + gpu_id),
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                env=environment,
            )
        deadline = time.monotonic() + request["limits"]["startup_timeout_s"]
        pending = set(processes)
        last: dict[str, str] = {}
        while pending and time.monotonic() < deadline:
            for endpoint_id in sorted(pending):
                gpu_id = int(endpoint_id.rsplit("-", 1)[1])
                if processes[endpoint_id].poll() is not None:
                    raise RuntimeError(f"{endpoint_id} exited before readiness")
                try:
                    _probe_role(
                        request, "image_edit",
                        base_url=f"http://127.0.0.1:{base_port + gpu_id}/v1",
                    )
                except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as exc:
                    last[endpoint_id] = _sanitize(str(exc))
                else:
                    pending.remove(endpoint_id)
            if pending:
                time.sleep(request["limits"]["retry_interval_s"])
        if pending:
            raise TimeoutError(f"image-worker readiness deadline exceeded: {last}")
        endpoints = [
            {
                "id": f"img-{args.worker_index:03d}-gpu-{gpu_id}",
                "url": f"http://{node}:{base_port + gpu_id}/v1", "capacity": 1,
                "gpu_identity": f"{node}/gpu-{gpu_id}",
                "owner": {"native_id": native_id, "name": args.job_id},
            }
            for gpu_id in range(IMAGE_SERVICES_PER_NODE)
        ]
        descriptor = stage / ".tao-runtime" / "image-workers" / f"{args.job_id}.json"
        _atomic_json(descriptor, {
            "schema_version": "1", "request_sha256": request["request_sha256"],
            "model": {
                "id": request["models"]["image_edit"]["id"],
                "revision": request["models"]["image_edit"]["revision"],
            },
            "worker_index": args.worker_index, "endpoints": endpoints,
        })
        while True:
            stopped = [name for name, process in processes.items() if process.poll() is not None]
            if stopped:
                raise RuntimeError(f"image-edit services exited unexpectedly: {stopped}")
            time.sleep(request["limits"]["retry_interval_s"])
    finally:
        for process in processes.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=30)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except OSError:
                        pass
        for handle in logs.values():
            handle.close()
        signal.signal(signal.SIGTERM, previous_sigterm)


def _load_image_owners(request: dict[str, Any], path: pathlib.Path, base_job_id: str) -> list[dict[str, Any]]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError("--job-group must be one absolute regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "request_sha256", "generation_nodes", "workers",
    }:
        raise ValueError("image owner manifest has missing or unexpected fields")
    if (payload["schema_version"] != "1" or payload["request_sha256"] != request["request_sha256"]
            or payload["generation_nodes"] != request["generation_nodes"]):
        raise ValueError("image owner manifest is incompatible with the request")
    workers = payload["workers"]
    if not isinstance(workers, list) or len(workers) != request["generation_nodes"]:
        raise ValueError("image owner manifest has the wrong worker count")
    native_ids: set[str] = set()
    for index, worker in enumerate(workers):
        if not isinstance(worker, dict) or set(worker) != {
            "role", "index", "name", "native_id", "reconciled",
        }:
            raise ValueError("image owner record is invalid")
        if (worker["role"] != "image-worker" or worker["index"] != index
                or worker["name"] != f"{base_job_id}-img-{index:03d}"
                or not str(worker["native_id"]).isdigit()
                or not isinstance(worker["reconciled"], bool)):
            raise ValueError("image owner record does not match its exact job")
        native_ids.add(str(worker["native_id"]))
    if len(native_ids) != len(workers):
        raise ValueError("image worker native IDs must be distinct")
    return workers


def _local_job_name(native_id: str) -> str:
    completed = _run(
        ["scontrol", "show", "job", "-o", native_id],
        timeout_s=SCHEDULER_QUERY_TIMEOUT_S,
    )
    output = _require_ok(completed, "coordinator image-worker ownership query").decode(errors="replace")
    matches = re.findall(r"(?:^| )JobName=([^ ]+)", output)
    if len(matches) != 1:
        raise ValueError("SLURM image-worker ownership response is ambiguous")
    return matches[0]


def _build_endpoint_pool(request: dict[str, Any], workers: list[dict[str, Any]]) -> dict[str, Any]:
    stage = pathlib.Path(request["stage_dir"])
    deadline = time.monotonic() + request["limits"]["startup_timeout_s"]
    settle_deadline: float | None = None
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    while time.monotonic() < deadline:
        selected = []
        for worker in workers:
            path = stage / ".tao-runtime" / "image-workers" / f"{worker['name']}.json"
            if path.is_file() and not path.is_symlink():
                selected.append((worker, json.loads(path.read_text(encoding="utf-8"))))
        now = time.monotonic()
        if selected and settle_deadline is None:
            settle_deadline = min(
                deadline,
                now + min(60, max(10, 2 * request["limits"]["retry_interval_s"])),
            )
        if len(selected) == len(workers) or (
            selected and settle_deadline is not None and now >= settle_deadline
        ):
            break
        time.sleep(request["limits"]["retry_interval_s"])
    if not selected:
        raise TimeoutError("no image-worker descriptor became ready before the startup deadline")
    endpoints: list[dict[str, Any]] = []
    for worker, descriptor in selected:
        if _local_job_name(str(worker["native_id"])) != worker["name"]:
            raise ValueError("image-worker native ID is not owned by its exact job name")
        expected_model = {
            "id": request["models"]["image_edit"]["id"],
            "revision": request["models"]["image_edit"]["revision"],
        }
        if (not isinstance(descriptor, dict) or set(descriptor) != {
                "schema_version", "request_sha256", "model", "worker_index", "endpoints"}
                or descriptor["schema_version"] != "1"
                or descriptor["request_sha256"] != request["request_sha256"]
                or descriptor["model"] != expected_model
                or descriptor["worker_index"] != worker["index"]
                or not isinstance(descriptor["endpoints"], list)
                or len(descriptor["endpoints"]) != IMAGE_SERVICES_PER_NODE):
            raise ValueError("image-worker descriptor is incompatible")
        endpoints.extend(descriptor["endpoints"])
    pool = {
        "schema_version": "1", "platform": "slurm",
        "model": {
            "id": request["models"]["image_edit"]["id"],
            "revision": request["models"]["image_edit"]["revision"],
        },
        "required_capacity": len(endpoints),
        "auth_env": "IMAGE_EDIT_API_KEY", "endpoints": endpoints,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "request_sha256": request["request_sha256"],
    }
    path = stage / "endpoint_pool.json"
    validated = _validate_endpoint_pool_payload(request, pool)
    for endpoint in validated["endpoints"]:
        _probe_role(request, "image_edit", base_url=endpoint["url"])
    _atomic_json(path, validated)
    return validated


def coordinator(args: argparse.Namespace) -> int:
    request = load_request(args.request)
    if not SAFE_TOKEN.fullmatch(args.job_id):
        raise ValueError("--job-id contains unsupported characters")
    coordinator_name = f"{args.job_id}-coord"
    coordinator_native_id, _ = _slurm_identity(coordinator_name)
    workers = _load_image_owners(request, args.job_group, args.job_id)
    immutable_config = _verify_signed_inputs(request)
    runtime_python = _resolve_runtime_python(request)
    stage = pathlib.Path(request["stage_dir"])
    stage.mkdir(parents=True, exist_ok=True)
    terminal = stage / f"slurm_sdg_terminal.{args.job_id}.json"
    if terminal.is_file():
        prior = json.loads(terminal.read_text())
        if prior.get("status") == "ok":
            if (prior.get("request_sha256") == request["request_sha256"]
                    and prior.get("job_id") == args.job_id
                    and prior.get("action_id") == request["action_id"]
                    and all(pathlib.Path(path).is_file() for path in request["expected_outputs"])):
                return 0
            raise ValueError("successful terminal evidence belongs to another request or is incomplete")
        raise ValueError("terminal error evidence is already recorded for this exact job")
    processes: dict[str, subprocess.Popen[Any]] = {}
    logs: dict[str, Any] = {}
    cleanup: list[dict[str, Any]] = []
    completed_successfully = False
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError("worker received SIGTERM")))
    try:
        occupied = [
            f"{role}:{request['models'][role]['port']}"
            for role in ("vlm", "llm") if not _port_available(request["models"][role]["port"])
        ]
        if occupied:
            raise RuntimeError(f"allocation-local endpoint port conflict: {occupied}")
        for role in ("vlm", "llm"):
            log_path = stage / "endpoint-logs" / args.job_id / f"{role}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            logs[role] = handle
            processes[role] = subprocess.Popen(
                _endpoint_command(request, role), stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        _wait_readiness(request, processes)
        pool = _build_endpoint_pool(request, workers)
        _atomic_json(stage / "endpoint_manifest.json", {
            "schema_version": "1", "ownership": "slurm_job", "job_id": args.job_id,
            "action_id": request["action_id"],
            "request_sha256": request["request_sha256"],
            "config_sha256": request["config_sha256"],
            "runtime_sha256": request["runtime_sha256"],
            "resume_sha256": _resume_sha256(request), "attempt": request["attempt"],
            "roles": {
                role: {"model": request["models"][role]["id"], "revision": request["models"][role]["revision"],
                       "port": request["models"][role]["port"], "gpus": ROLE_GPUS[role], "ready": True}
                for role in ("vlm", "llm")
            },
            "image_edit_pool": {
                "path": str(stage / "endpoint_pool.json"),
                "required_capacity": pool["required_capacity"],
                "requested_capacity": request["generation_nodes"] * IMAGE_SERVICES_PER_NODE,
                "active_nodes": pool["required_capacity"] // IMAGE_SERVICES_PER_NODE,
                "requested_nodes": request["generation_nodes"],
            },
            "components": _component_evidence(request, immutable_config),
        })
        _execute_shared(request, args.request.resolve(), args.job_id, runtime_python)
        completed_successfully = True
        return 0
    except BaseException as exc:
        _atomic_json(terminal, {
            "schema_version": "1", "workflow": WORKFLOW, "kind": KIND, "status": "error",
            "job_id": args.job_id, "action_id": request["action_id"],
            "coordinator_native_id": coordinator_native_id,
            "request_sha256": request["request_sha256"],
            "resume_sha256": _resume_sha256(request), "attempt": request["attempt"],
            "started_at": request["started_at"], "started_ns": request["started_ns"],
            "worker_started_at": started, "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "error": _sanitize(f"{type(exc).__name__}: {exc}"),
        })
        raise
    finally:
        for role, process in processes.items():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=30)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except OSError:
                        pass
            cleanup.append({"role": role, "returncode": process.poll(), "owned": True})
        for handle in logs.values():
            handle.close()
        worker_records = _cleanup_image_workers(request, workers)
        cleanup.extend(worker_records)
        worker_cleanup_failed = any(
            record["cleanup"] != "canceled" for record in worker_records
        )
        _atomic_json(stage / f"endpoint_cleanup.{args.job_id}.json", {
            "schema_version": "1", "job_id": args.job_id,
            "action_id": request["action_id"],
            "request_sha256": request["request_sha256"], "steps": cleanup,
        })
        auth_file = stage / ".tao-runtime" / f"endpoint-auth.{args.job_id}.env"
        try:
            if auth_file.is_file() and not auth_file.is_symlink():
                completed = _run(["shred", "-u", str(auth_file)])
                if completed.returncode != 0 and auth_file.exists():
                    auth_file.unlink()
        except OSError:
            pass
        for path in (stage / "endpoint-logs" / args.job_id).glob("*.log"):
            try:
                clean = _sanitize(path.read_text(errors="replace"))
                path.write_text(clean, encoding="utf-8")
            except OSError:
                pass
        signal.signal(signal.SIGTERM, previous_sigterm)
        if completed_successfully:
            status = "error" if worker_cleanup_failed else "ok"
            evidence = {
                "schema_version": "1", "workflow": WORKFLOW, "kind": KIND,
                "status": status, "job_id": args.job_id,
                "action_id": request["action_id"],
                "coordinator_native_id": coordinator_native_id,
                "request_sha256": request["request_sha256"],
                "resume_sha256": _resume_sha256(request), "attempt": request["attempt"],
                "started_at": request["started_at"], "started_ns": request["started_ns"],
                "worker_started_at": started,
                "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            if status == "ok":
                evidence["expected_outputs"] = request["expected_outputs"]
            else:
                evidence["error"] = "owned image-worker cleanup did not complete"
            _atomic_json(terminal, evidence)
            if worker_cleanup_failed:
                raise RuntimeError("owned image-worker cleanup did not complete")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="verb", required=True)
    prepare_parser = sub.add_parser("prepare-request")
    prepare_parser.add_argument("--deft-state", required=True, type=pathlib.Path)
    prepare_parser.add_argument("--sdg-config", required=True, type=pathlib.Path)
    prepare_parser.add_argument("--iteration", required=True, type=int)
    prepare_parser.add_argument("--runtime-root", required=True, type=pathlib.Path)
    prepare_parser.add_argument("--cache-dir", required=True, type=pathlib.Path)
    prepare_parser.add_argument("--backend-results-dir", type=pathlib.Path)
    prepare_parser.add_argument("--backend-dataset-root", type=pathlib.Path)
    prepare_parser.add_argument("--augmentation-image", required=True, type=pathlib.Path)
    prepare_parser.add_argument("--auto-labeling-image", required=True, type=pathlib.Path)
    prepare_parser.add_argument("--image-edit-image", required=True, type=pathlib.Path)
    prepare_parser.add_argument("--text-serving-image", required=True, type=pathlib.Path)
    prepare_parser.add_argument("--account")
    prepare_parser.add_argument("--partition")
    prepare_parser.add_argument("--image-worker-cpus-per-task", type=int, default=64)
    # 60 fits common two-GPU SLURM CPU/GPU limits while retaining ample CPU
    # for the coordinator's concurrent component workers.
    prepare_parser.add_argument("--coordinator-cpus-per-task", type=int, default=60)
    prepare_parser.add_argument("--time-minutes", type=int, default=240)
    prepare_parser.add_argument("--retry-from-request", type=pathlib.Path)
    prepare_parser.add_argument("--retry-from-job-record", type=pathlib.Path)
    prepare_parser.add_argument("--retry-login")
    prepare_parser.add_argument("--repair-from-request", type=pathlib.Path)
    prepare_parser.add_argument("--repair-from-job-record", type=pathlib.Path)
    prepare_parser.add_argument("--repair-login")
    prepare_parser.add_argument("--reschedule-from-request", type=pathlib.Path)
    prepare_parser.add_argument("--reschedule-from-job-record", type=pathlib.Path)
    prepare_parser.add_argument("--reschedule-login")
    prepare_parser.add_argument("--launch-repair-from-request", type=pathlib.Path)
    prepare_parser.add_argument("--launch-repair-from-job-record", type=pathlib.Path)
    prepare_parser.add_argument("--launch-repair-login")
    prepare_parser.add_argument("--output", required=True, type=pathlib.Path)
    submit_parser = sub.add_parser("submit")
    submit_parser.add_argument("--request", required=True, type=pathlib.Path)
    submit_parser.add_argument("--login", required=True)
    submit_parser.add_argument("--job-id", required=True)
    submit_parser.add_argument("--job-record", required=True, type=pathlib.Path)
    submit_parser.add_argument("--remote-script", required=True, type=pathlib.Path)
    submit_parser.add_argument("--env-file", type=pathlib.Path)
    submit_parser.add_argument("--account")
    submit_parser.add_argument("--partition")
    recovery_parser = sub.add_parser("recover-duplicate-submit")
    recovery_parser.add_argument("--request", required=True, type=pathlib.Path)
    recovery_parser.add_argument("--login", required=True)
    recovery_parser.add_argument("--job-id", required=True)
    recovery_parser.add_argument("--job-record", required=True, type=pathlib.Path)
    recovery_parser.add_argument("--confirm", action="store_true")
    cleanup_recovery_parser = sub.add_parser("recover-cleanup-failure")
    cleanup_recovery_parser.add_argument("--request", required=True, type=pathlib.Path)
    cleanup_recovery_parser.add_argument("--login", required=True)
    cleanup_recovery_parser.add_argument("--job-id", required=True)
    cleanup_recovery_parser.add_argument("--job-record", required=True, type=pathlib.Path)
    cleanup_recovery_parser.add_argument("--confirm", action="store_true")
    for verb in ("status", "logs", "cancel"):
        child = sub.add_parser(verb)
        child.add_argument("--request", required=True, type=pathlib.Path)
        child.add_argument("--login", required=True)
        child.add_argument("--backend-ref", required=True)
        child.add_argument("--job-id", required=True)
        child.add_argument("--job-record", required=True, type=pathlib.Path)
        if verb == "logs":
            child.add_argument("--tail", type=int, default=200, choices=range(1, 10001))
        if verb == "status":
            child.add_argument("--local-results-dir", type=pathlib.Path)
        if verb == "cancel":
            child.add_argument("--confirm", action="store_true")
    image_parser = sub.add_parser("image-worker")
    image_parser.add_argument("--request", required=True, type=pathlib.Path)
    image_parser.add_argument("--job-id", required=True)
    image_parser.add_argument("--worker-index", required=True, type=int)
    coordinator_parser = sub.add_parser("coordinator")
    coordinator_parser.add_argument("--request", required=True, type=pathlib.Path)
    coordinator_parser.add_argument("--job-id", required=True)
    coordinator_parser.add_argument("--job-group", required=True, type=pathlib.Path)
    component_parser = sub.add_parser("component")
    component_parser.add_argument("--request", required=True, type=pathlib.Path)
    component_parser.add_argument("--job-id", required=True)
    component_parser.add_argument(
        "--action", required=True, choices=("preprocess", "augment", "split", "label"),
    )
    component_parser.add_argument("--input-root", required=True, type=pathlib.Path)
    component_parser.add_argument("--output-root", required=True, type=pathlib.Path)
    component_parser.add_argument("--source-key")
    component_parser.add_argument("--attempt", required=True, type=int)
    component_parser.add_argument("--target-attributes-json", default="{}")
    component_parser.add_argument("--image-edit-endpoint-id")
    component_parser.add_argument("--image-edit-url")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.verb == "prepare-request":
            result = prepare_request(args)
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.verb in {"image-worker", "coordinator", "component"}:
            return {
                "image-worker": image_worker, "coordinator": coordinator, "component": component,
            }[args.verb](args)
        result = {
            "submit": submit,
            "recover-duplicate-submit": recover_duplicate_submit,
            "recover-cleanup-failure": recover_cleanup_failure,
            "status": status,
            "logs": logs,
            "cancel": cancel,
        }[args.verb](args)
        print(result if isinstance(result, str) else json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"slurm_sdg_action[{args.verb}]: {_sanitize(str(exc))}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
