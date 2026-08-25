# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight, start, validate, and stop run-scoped local IAA model endpoints."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re
import shutil
import socket
import subprocess
import sys
from typing import Any

import yaml

from iaa_deft.sdg import (
    ROLES,
    atomic_json,
    build_endpoint_command,
    container_name,
    endpoint_url,
    port_available,
    readiness_probe,
    validate_config,
    validate_gpu_inventory,
    validate_image_edit_endpoint_pool,
    wait_until_ready,
)


WORKFLOW_LABEL = "com.nvidia.tao.workflow=tao-run-deft-iaa"
WORKFLOW_COMPONENTS = ("augmentation", "auto_labeling")
IMAGE_EDIT_SHM_BYTES = 16 * 1024**3
MIN_MODEL_CACHE_CAPACITY_BYTES = 150 * 1024**3


def _selected_roles(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ROLES
    roles = tuple(item.strip() for item in value.split(",") if item.strip())
    if not roles or len(set(roles)) != len(roles) or any(item not in ROLES for item in roles):
        raise ValueError("--roles must be a unique comma-separated subset of image_edit,vlm,llm")
    return roles


def _instances(config: dict[str, Any], run_id: str, roles: tuple[str, ...]) -> list[dict[str, Any]]:
    result = []
    for role in roles:
        if role == "image_edit":
            for ordinal, gpu_id in enumerate(config["endpoints"]["gpu_ids"][role]):
                result.append({
                    "key": f"image_edit_gpu_{gpu_id}", "role": role, "gpu_id": gpu_id,
                    "ordinal": ordinal, "port": config["models"][role]["port"] + ordinal,
                    "name": container_name(run_id, role, gpu_id),
                })
        else:
            result.append({
                "key": role, "role": role, "gpu_id": None, "ordinal": None,
                "port": config["models"][role]["port"], "name": container_name(run_id, role),
            })
    return result


def _instance_url(config: dict[str, Any], instance: dict[str, Any], service_host: str) -> str:
    return endpoint_url(
        config, instance["role"], ordinal=instance["ordinal"], service_host=service_host,
    )


def _request_digest(config: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _cached_revision_bytes(
    config: dict[str, Any], cache_dir: pathlib.Path, roles: tuple[str, ...]
) -> int:
    """Count reusable files from only the configured immutable model revisions."""
    hub = (cache_dir / "hub").resolve()
    seen: set[tuple[int, int]] = set()
    total = 0
    for role in roles:
        model = config["models"][role]
        snapshot = hub / ("models--" + model["id"].replace("/", "--")) / "snapshots" / model["revision"]
        if not snapshot.is_dir():
            continue
        for candidate in snapshot.rglob("*"):
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(hub)
                stat = resolved.stat()
            except (FileNotFoundError, OSError, ValueError):
                continue
            if not resolved.is_file():
                continue
            identity = (stat.st_dev, stat.st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            total += stat.st_size
    return total


def _exact_revision_receipts(
    config: dict[str, Any], cache_dir: pathlib.Path, roles: tuple[str, ...]
) -> dict[str, dict[str, Any]] | None:
    """Return read-only receipts only when every selected immutable revision exists."""
    hub = (cache_dir / "hub").resolve()
    receipts: dict[str, dict[str, Any]] = {}
    for role in roles:
        model = config["models"][role]
        snapshot = (
            hub / ("models--" + model["id"].replace("/", "--"))
            / "snapshots" / model["revision"]
        )
        if not snapshot.is_dir() or snapshot.is_symlink():
            return None
        seen: set[tuple[int, int]] = set()
        files = 0
        total = 0
        for candidate in snapshot.rglob("*"):
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(hub)
                stat = resolved.stat()
            except (FileNotFoundError, OSError, ValueError):
                return None
            if not resolved.is_file():
                continue
            identity = (stat.st_dev, stat.st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            files += 1
            total += stat.st_size
        if files == 0 or total == 0:
            return None
        receipts[role] = {
            "model": model["id"], "revision": model["revision"],
            "snapshot": str(snapshot), "file_count": files, "bytes": total,
        }
    return receipts


def _instance_matches_immutable_config(
    inspect: dict[str, Any], config: dict[str, Any], run_id: str,
    instance: dict[str, Any],
) -> bool:
    """Bind one live service to the approved image/model/revision/GPU/port tuple."""
    role = instance["role"]
    if not _owned(inspect, run_id, role):
        return False
    state = inspect.get("State", {})
    health = state.get("Health", {}).get("Status")
    if (
        not state.get("Running") or state.get("Paused") or state.get("Restarting")
        or state.get("Dead") or state.get("OOMKilled")
        or state.get("Error") not in (None, "")
        or health not in (None, "", "healthy")
    ):
        return False
    model = config["models"][role]
    image_key = "image_edit_serving" if role == "image_edit" else "text_serving"
    if inspect.get("Config", {}).get("Image") != config["images"][image_key]:
        return False
    command = inspect.get("Config", {}).get("Cmd") or []
    pairs = {
        "--revision": str(model["revision"]),
        "--served-model-name": str(model["id"]),
        "--port": str(instance["port"]),
    }
    for flag, value in pairs.items():
        if flag not in command or command.index(flag) + 1 >= len(command):
            return False
        if command[command.index(flag) + 1] != value:
            return False
    if role == "image_edit":
        if str(model["id"]) not in command:
            return False
        expected_gpu_ids = [str(instance["gpu_id"])]
    else:
        if "--model" not in command or command.index("--model") + 1 >= len(command):
            return False
        if command[command.index("--model") + 1] != model["id"]:
            return False
        expected_gpu_ids = [str(item) for item in config["endpoints"]["gpu_ids"][role]]
    requests = inspect.get("HostConfig", {}).get("DeviceRequests") or []
    actual_gpu_ids = [
        str(item) for request in requests for item in (request.get("DeviceIDs") or [])
    ]
    if actual_gpu_ids != expected_gpu_ids:
        return False
    binding = (inspect.get("HostConfig", {}).get("PortBindings") or {}).get(
        f"{instance['port']}/tcp"
    )
    return isinstance(binding, list) and len(binding) == 1 and (
        binding[0].get("HostIp") == "127.0.0.1"
        and binding[0].get("HostPort") == str(instance["port"])
    )


def _healthy_reuse_snapshot(
    config: dict[str, Any], run_id: str, cache_dir: pathlib.Path,
    roles: tuple[str, ...], *, service_host: str = "127.0.0.1",
) -> dict[str, Any] | None:
    """Prove exact, currently healthy reuse without mutating Docker or the cache."""
    receipts = _exact_revision_receipts(config, cache_dir, roles)
    if receipts is None:
        return None
    containers: dict[str, dict[str, Any]] = {}
    for instance in _instances(config, run_id, roles):
        inspect = _inspect(instance["name"])
        if inspect is None or not _instance_matches_immutable_config(
            inspect, config, run_id, instance,
        ):
            return None
        role = instance["role"]
        try:
            probe = readiness_probe(
                config, role,
                base_url=_instance_url(config, instance, service_host),
            )
        except (OSError, ValueError, KeyError):
            return None
        if probe.get("models_ok") is not True:
            return None
        if role in {"vlm", "llm"} and probe.get("inference_ok") is not True:
            return None
        containers[instance["key"]] = {
            "name": instance["name"], "id": str(inspect.get("Id", ""))[:12],
            "owned": True, "created_this_call": False,
            "url": _instance_url(config, instance, service_host),
            "model": config["models"][role]["id"], "probe": probe,
            "log_command": ["docker", "logs", "--tail", "200", instance["name"]],
        }
    return {
        "disposition": "reuse_no_acquisition",
        "cache_receipts": receipts,
        "containers": containers,
    }


def _endpoint_pool(
    config: dict[str, Any], instances: list[dict[str, Any]], containers: dict[str, dict[str, Any]],
    *, platform: str, service_host: str, request_sha256: str, gpu_identity_prefix: str,
) -> dict[str, Any]:
    image_instances = [item for item in instances if item["role"] == "image_edit"]
    endpoints = []
    endpoint_prefix = re.sub(r"[^A-Za-z0-9._:-]+", "-", gpu_identity_prefix).strip("-")
    if not endpoint_prefix:
        raise ValueError("GPU identity prefix does not produce a usable endpoint ID")
    for item in image_instances:
        record = containers[item["key"]]
        endpoints.append({
            "id": f"{endpoint_prefix}-gpu-{item['gpu_id']}",
            "url": _instance_url(config, item, service_host), "capacity": 1,
            "gpu_identity": f"{gpu_identity_prefix}/gpu:{item['gpu_id']}",
            "owner": {"native_id": record.get("id") or item["name"], "name": item["name"]},
        })
    return validate_image_edit_endpoint_pool({
        "schema_version": "1", "platform": platform,
        "model": {
            "id": config["models"]["image_edit"]["id"],
            "revision": config["models"]["image_edit"]["revision"],
        },
        "required_capacity": len(endpoints),
        "auth_env": "IMAGE_EDIT_API_KEY" if platform in {"brev", "slurm", "kubernetes"} else None,
        "endpoints": endpoints,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "request_sha256": request_sha256,
    })


def _run(argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, check=check)


def _load(path: pathlib.Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    return validate_config(payload)


def _inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,compute_cap,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    result = _run(command)
    rows = []
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 5:
            raise ValueError(f"unexpected nvidia-smi row: {line!r}")
        rows.append({
            "index": int(parts[0]),
            "name": parts[1],
            "compute_capability": float(parts[2]),
            "memory_total_mib": int(parts[3]),
            "memory_free_mib": int(parts[4]),
        })
    return rows


def _inspect(name: str) -> dict[str, Any] | None:
    result = _run(["docker", "inspect", name], check=False)
    if result.returncode != 0:
        return None
    payload = json.loads(result.stdout)
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"docker inspect returned malformed data for {name}")
    return payload[0]


def _inspect_image(name: str) -> dict[str, Any] | None:
    result = _run(["docker", "image", "inspect", name], check=False)
    if result.returncode != 0:
        return None
    payload = json.loads(result.stdout)
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"docker image inspect returned malformed data for {name}")
    return payload[0]


def _component_image_record(config: dict[str, Any], component: str) -> dict[str, Any]:
    image = config["images"][component]
    inspect = _inspect_image(image)
    return {
        "component": component,
        "image": image,
        "present": inspect is not None,
        "image_id": inspect.get("Id", "") if inspect else "",
    }


def component_status(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "components": {name: _component_image_record(config, name) for name in WORKFLOW_COMPONENTS},
    }


def _require_component_images(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = component_status(config)["components"]
    missing = [name for name, record in records.items() if not record["present"]]
    if missing:
        raise ValueError(
            "approved component images are missing for " + ", ".join(missing)
            + "; pull the approved prebuilt images before endpoint validation/start"
        )
    return records


def _owned(inspect: dict[str, Any], run_id: str, role: str) -> bool:
    labels = inspect.get("Config", {}).get("Labels") or {}
    return (
        labels.get("com.nvidia.tao.workflow") == "tao-run-deft-iaa"
        and labels.get("com.nvidia.tao.run") == run_id
        and labels.get("com.nvidia.tao.role") == role
    )


def _recoverable_gpu_parse_failure(
    inspect: dict[str, Any], run_id: str, role: str, expected_gpu_ids: list[int]
) -> bool:
    """Recognize only our deterministic, never-started malformed Docker create."""
    if not _owned(inspect, run_id, role):
        return False
    state = inspect.get("State", {})
    started_at = state.get("StartedAt", "")
    parser_error = "cannot set both Count and DeviceIDs on device request"
    error_lines = str(state.get("Error", "")).splitlines()
    if (
        state.get("Status") != "created"
        or state.get("Running")
        or started_at not in ("", "0001-01-01T00:00:00Z")
        or not 1 <= len(error_lines) <= 2
        or any(line != parser_error for line in error_lines)
    ):
        return False
    requests = inspect.get("HostConfig", {}).get("DeviceRequests") or []
    requested_ids = [str(item) for request in requests for item in (request.get("DeviceIDs") or [])]
    return requested_ids != [str(item) for item in expected_gpu_ids]


def _failed_create_evidence(inspect: dict[str, Any], name: str) -> dict[str, Any]:
    """Capture diagnostic evidence without serializing environment or secrets."""
    state = inspect.get("State", {})
    requests = inspect.get("HostConfig", {}).get("DeviceRequests") or []
    logs = _run(["docker", "logs", "--tail", "200", name], check=False)
    # A never-started container should have no logs. Do not propagate unexpected
    # content because serving processes can echo credentials from their environment.
    return {
        "name": name,
        "status": state.get("Status", ""),
        "error": state.get("Error", ""),
        "device_requests": [
            {"Count": item.get("Count"), "DeviceIDs": item.get("DeviceIDs") or []}
            for item in requests
        ],
        "log_capture": "empty" if not (logs.stdout or logs.stderr).strip() else "redacted_nonempty",
        "disposition": "removed_and_recreated",
    }


def _stopped_endpoint_evidence(inspect: dict[str, Any], name: str, role: str) -> dict[str, Any]:
    """Capture a stopped owned endpoint without exposing logs or environment."""
    state = inspect.get("State", {})
    logs = _run(["docker", "logs", "--tail", "200", name], check=False)
    return {
        "name": name, "role": role, "status": state.get("Status", ""),
        "exit_code": state.get("ExitCode"), "oom_killed": state.get("OOMKilled"),
        "error": state.get("Error", ""), "finished_at": state.get("FinishedAt", ""),
        "log_capture": "empty" if not (logs.stdout or logs.stderr).strip() else "redacted_nonempty",
        "log_command": ["docker", "logs", "--tail", "200", name],
        "disposition": "restarted_and_reprobed",
    }


def _requires_shared_memory_recreate(inspect: dict[str, Any], role: str) -> bool:
    """Identify a stopped image-edit endpoint created with unsafe Docker defaults."""
    return (
        role == "image_edit"
        and not inspect.get("State", {}).get("Running")
        and int(inspect.get("HostConfig", {}).get("ShmSize") or 0) < IMAGE_EDIT_SHM_BYTES
    )


def _exact_owned_image_worker(
    inspect: dict[str, Any], run_id: str, gpu_id: int,
) -> bool:
    if not _owned(inspect, run_id, "image_edit"):
        return False
    requests = inspect.get("HostConfig", {}).get("DeviceRequests") or []
    device_ids = [str(item) for request in requests for item in (request.get("DeviceIDs") or [])]
    return device_ids == [str(gpu_id)]


def preflight(
    config: dict[str, Any], run_id: str, cache_dir: pathlib.Path,
    roles: tuple[str, ...] = ROLES, *, platform: str = "docker",
) -> dict:
    if not shutil.which("docker") or not shutil.which("nvidia-smi"):
        raise ValueError("managed endpoints require docker and nvidia-smi")
    remote_image_worker = platform in {"brev", "slurm", "kubernetes"} and "image_edit" in roles
    if remote_image_worker:
        missing = [name for name in ("VLLM_API_KEY", "IMAGE_EDIT_API_KEY") if name not in os.environ]
        if missing:
            raise ValueError("remote image-edit publishing requires environment variables: " + ", ".join(missing))
    publish_host = "0.0.0.0" if remote_image_worker else "127.0.0.1"
    info = _run(["docker", "info", "--format", "{{json .Runtimes}}"])
    runtimes = json.loads(info.stdout)
    if "nvidia" not in runtimes:
        raise ValueError("Docker NVIDIA runtime is unavailable")
    disk_probe = cache_dir
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    free_bytes = shutil.disk_usage(disk_probe).free
    reusable_bytes = _cached_revision_bytes(config, cache_dir, roles)
    usable_capacity = free_bytes + reusable_bytes
    healthy_reuse = None
    if platform in {"host", "docker", "virtualenv"}:
        healthy_reuse = _healthy_reuse_snapshot(config, run_id, cache_dir, roles)
    if healthy_reuse is not None:
        return {
            "ownership": "managed", "run_id": run_id,
            "cache_dir": str(cache_dir.resolve()),
            "cache_capacity": {
                "free_bytes": free_bytes,
                "reusable_revision_bytes": reusable_bytes,
                "usable_bytes": usable_capacity,
                "required_bytes": MIN_MODEL_CACHE_CAPACITY_BYTES,
                "acquisition_gate": "not_applicable",
                "disposition": "reuse_no_acquisition",
            },
            "gpu_inventory": [], "commands": {},
            "component_images": component_status(config)["components"],
            **healthy_reuse,
        }
    if usable_capacity < MIN_MODEL_CACHE_CAPACITY_BYTES:
        raise ValueError(
            "model cache has only "
            f"{usable_capacity // 1024**3} GiB usable capacity "
            f"({free_bytes // 1024**3} GiB free plus "
            f"{reusable_bytes // 1024**3} GiB in exact reusable revisions); "
            "require at least 150 GiB"
        )
    inventory = _inventory()
    conflicts: list[str] = []
    active_roles: set[str] = set()
    instances = _instances(config, run_id, roles)
    for instance in instances:
        role, name = instance["role"], instance["name"]
        existing = _inspect(name)
        if existing is not None:
            if not _owned(existing, run_id, role):
                conflicts.append(f"container name {name} belongs to another owner")
            elif existing.get("State", {}).get("Running"):
                active_roles.add(role)
            continue
        port = instance["port"]
        if not port_available(port):
            conflicts.append(f"127.0.0.1:{port} is already in use for managed role {role}")
    if conflicts:
        raise ValueError("; ".join(conflicts))
    # Validate the complete immutable allocation. Active roles only relax free
    # memory checks when every selected instance for that role is already live.
    fully_active = {
        role for role in roles
        if all(
            item["role"] != role or (
                (found := _inspect(item["name"])) is not None
                and _owned(found, run_id, role)
                and found.get("State", {}).get("Running")
            )
            for item in instances
        )
    }
    validate_gpu_inventory(config, inventory, fully_active, roles)
    return {
        "ownership": "managed",
        "run_id": run_id,
        "cache_dir": str(cache_dir.resolve()),
        "cache_capacity": {
            "free_bytes": free_bytes,
            "reusable_revision_bytes": reusable_bytes,
            "usable_bytes": usable_capacity,
            "required_bytes": MIN_MODEL_CACHE_CAPACITY_BYTES,
        },
        "gpu_inventory": inventory,
        "commands": {
            item["key"]: build_endpoint_command(
                config, item["role"], run_id, cache_dir,
                image_edit_gpu_id=item["gpu_id"], image_edit_ordinal=item["ordinal"],
                publish_host=publish_host, authenticated=remote_image_worker,
            ) if item["role"] == "image_edit" else build_endpoint_command(
                config, item["role"], run_id, cache_dir,
            )
            for item in instances
        },
        "component_images": component_status(config)["components"],
    }


def start(
    config: dict[str, Any], run_id: str, cache_dir: pathlib.Path,
    roles: tuple[str, ...] = ROLES, *, platform: str = "docker",
    service_host: str = "127.0.0.1", request_sha256: str | None = None,
    image_edit_pool: pathlib.Path | None = None, gpu_identity_prefix: str = "local",
    recreate_owned: bool = False,
) -> dict:
    if "image_edit" in roles and platform in {"host", "docker", "virtualenv"}:
        expected = len(config["endpoints"]["gpu_ids"]["image_edit"])
        if (
            config["generation"]["generation_nodes"] != 1
            or config["generation"]["gpus_per_generation_node"] != expected
        ):
            raise ValueError(
                "local image-edit pool topology must be one node with "
                "gpus_per_generation_node equal to explicit image-edit GPU IDs"
            )
    report = preflight(config, run_id, cache_dir, roles, platform=platform)
    report["component_images"] = _require_component_images(config)
    if report.get("disposition") == "reuse_no_acquisition":
        containers = report["containers"]
        instances = _instances(config, run_id, roles)
        if "image_edit" in roles:
            pool = _endpoint_pool(
                config, instances, containers, platform=platform,
                service_host=service_host,
                request_sha256=request_sha256 or _request_digest(config),
                gpu_identity_prefix=gpu_identity_prefix,
            )
            report["image_edit_endpoint_pool"] = pool
            if image_edit_pool is not None:
                atomic_json(image_edit_pool.resolve(), pool)
        return report
    cache_dir.mkdir(parents=True, exist_ok=True)
    containers: dict[str, dict[str, Any]] = {}
    recoveries: list[dict[str, Any]] = []
    restarts: list[dict[str, Any]] = []
    instances = _instances(config, run_id, roles)
    for instance in instances:
        role, name = instance["role"], instance["name"]
        existing = _inspect(name)
        gpu_ids = [instance["gpu_id"]] if role == "image_edit" else config["endpoints"]["gpu_ids"][role]
        if recreate_owned and role == "image_edit" and existing is not None:
            if not _exact_owned_image_worker(existing, run_id, instance["gpu_id"]):
                raise ValueError(f"refusing to recreate non-owned or GPU-mismatched container {name}")
            state = existing.get("State", {})
            recoveries.append({
                "name": name, "role": role, "gpu_id": instance["gpu_id"],
                "previous_status": state.get("Status", ""),
                "previous_running": bool(state.get("Running")),
                "disposition": "removed_and_recreated_for_run_scoped_auth_rotation",
            })
            if state.get("Running"):
                _run(["docker", "stop", "--time", "30", name])
            _run(["docker", "rm", name])
            existing = None
        if existing is not None and _recoverable_gpu_parse_failure(existing, run_id, role, gpu_ids):
            recoveries.append(_failed_create_evidence(existing, name))
            _run(["docker", "rm", name])
            existing = None
        if existing is not None and _owned(existing, run_id, role) and _requires_shared_memory_recreate(existing, role):
            evidence = _stopped_endpoint_evidence(existing, name, role)
            evidence["disposition"] = "removed_and_recreated_with_required_shared_memory"
            evidence["previous_shm_bytes"] = int(existing.get("HostConfig", {}).get("ShmSize") or 0)
            recoveries.append(evidence)
            _run(["docker", "rm", name])
            existing = None
        if existing is None:
            result = _run(build_endpoint_command(
                config, role, run_id, cache_dir,
                image_edit_gpu_id=instance["gpu_id"], image_edit_ordinal=instance["ordinal"],
                publish_host=("0.0.0.0" if platform in {"brev", "slurm", "kubernetes"} else "127.0.0.1"),
                authenticated=platform in {"brev", "slurm", "kubernetes"},
            ) if role == "image_edit" else build_endpoint_command(config, role, run_id, cache_dir))
            created = True
            container_id = result.stdout.strip()
        else:
            if not _owned(existing, run_id, role):
                raise ValueError(f"refusing to replace non-owned container {name}")
            created = False
            container_id = existing.get("Id", "")
            if not existing.get("State", {}).get("Running"):
                restarts.append(_stopped_endpoint_evidence(existing, name, role))
                _run(["docker", "start", name])
        probe = wait_until_ready(
            lambda role=role, instance=instance: readiness_probe(
                config, role, base_url=_instance_url(config, instance, "127.0.0.1")
            ),
            config["endpoints"]["startup_timeout_s"],
            config["endpoints"]["retry_interval_s"],
        )
        containers[instance["key"]] = {
            "name": name,
            "id": container_id[:12],
            "owned": True,
            "created_this_call": created,
            "url": _instance_url(config, instance, service_host),
            "model": config["models"][role]["id"],
            "probe": probe,
            "log_command": ["docker", "logs", "--tail", "200", name],
        }
    result = {**report, "containers": containers, "recoveries": recoveries, "restarts": restarts}
    if recreate_owned:
        result["auth_rotation"] = {
            "request_sha256": request_sha256,
            "recreated": [
                item["name"] for item in instances if item["role"] == "image_edit"
            ],
            "status": "complete",
        }
    if "image_edit" in roles:
        pool = _endpoint_pool(
            config, instances, containers, platform=platform, service_host=service_host,
            request_sha256=request_sha256 or _request_digest(config),
            gpu_identity_prefix=gpu_identity_prefix,
        )
        result["image_edit_endpoint_pool"] = pool
        if image_edit_pool is not None:
            atomic_json(image_edit_pool.resolve(), pool)
    return result


def repair_created(config: dict[str, Any], run_id: str) -> dict:
    """Remove only exact run-owned never-started GPU-parser failed creates."""
    recoveries = []
    for role in ROLES:
        name = container_name(run_id, role)
        existing = _inspect(name)
        if existing is None:
            continue
        if not _owned(existing, run_id, role):
            raise ValueError(f"refusing to replace non-owned container {name}")
        if _recoverable_gpu_parse_failure(
            existing, run_id, role, config["endpoints"]["gpu_ids"][role]
        ):
            evidence = _failed_create_evidence(existing, name)
            evidence["disposition"] = "removed_for_runtime_rebind"
            recoveries.append(evidence)
            _run(["docker", "rm", name])
    return {"ownership": "managed", "recoveries": recoveries}


def validate_external(config: dict[str, Any]) -> dict:
    if config["endpoints"]["ownership"] != "external":
        raise ValueError("validate-external requires ownership=external")
    return {
        "ownership": "external",
        "containers": {},
        "component_images": _require_component_images(config),
        "probes": {
            role: wait_until_ready(
                lambda role=role: readiness_probe(config, role),
                config["endpoints"]["startup_timeout_s"],
                config["endpoints"]["retry_interval_s"],
            )
            for role in ROLES
        },
    }


def status(config: dict[str, Any], run_id: str, roles: tuple[str, ...] = ROLES) -> dict:
    if config["endpoints"]["ownership"] == "external":
        return validate_external(config)
    containers = {}
    for instance in _instances(config, run_id, roles):
        role, name = instance["role"], instance["name"]
        inspect = _inspect(name)
        containers[instance["key"]] = {
            "name": name,
            "exists": inspect is not None,
            "owned": bool(inspect and _owned(inspect, run_id, role)),
            "running": bool(inspect and inspect.get("State", {}).get("Running")),
            "status": inspect.get("State", {}).get("Status") if inspect else "missing",
        }
    return {"ownership": "managed", "containers": containers, "component_images": component_status(config)["components"]}


def _all_managed_roles_running(
    config: dict[str, Any], run_id: str, roles: tuple[str, ...] = ROLES,
) -> bool:
    if config["endpoints"]["ownership"] != "managed":
        return False
    for instance in _instances(config, run_id, roles):
        role = instance["role"]
        inspect = _inspect(instance["name"])
        if inspect is None or not _owned(inspect, run_id, role) or not inspect.get("State", {}).get("Running"):
            return False
    return True


def _prior_managed_readiness(
    previous: dict[str, Any], config: dict[str, Any], run_id: str,
    roles: tuple[str, ...] = ROLES,
) -> bool:
    if previous.get("ownership") != "managed":
        return False
    containers = previous.get("containers")
    if not isinstance(containers, dict):
        return False
    legacy = all(role in containers for role in roles)
    for instance in _instances(config, run_id, roles):
        role = instance["role"]
        # Read-only compatibility for readiness evidence written before
        # image-edit replication. It can authorize a controlled restart/stop,
        # but new starts always materialize GPU-qualified workers.
        record = containers.get(role) if legacy else containers.get(instance["key"])
        probe = record.get("probe") if isinstance(record, dict) else None
        if (
            not isinstance(record, dict)
            or record.get("name") != (container_name(run_id, role) if legacy else instance["name"])
            or record.get("owned") is not True
            or record.get("model") != config["models"][role]["id"]
            or not isinstance(probe, dict)
            or probe.get("models_ok") is not True
            or (role in {"vlm", "llm"} and probe.get("inference_ok") is not True)
        ):
            return False
    return True


def _clean_owned_shutdown_evidence(
    config: dict[str, Any], run_id: str, roles: tuple[str, ...] = ROLES,
) -> dict[str, Any] | None:
    """Return sanitized evidence only when every managed role stopped cleanly."""
    if config["endpoints"]["ownership"] != "managed":
        return None
    inspections = {item["key"]: _inspect(item["name"]) for item in _instances(config, run_id, roles)}
    return _clean_owned_shutdown_from_inspections(config, run_id, inspections, roles)


def _clean_owned_shutdown_from_inspections(
    config: dict[str, Any], run_id: str, inspections: dict[str, dict[str, Any] | None],
    roles: tuple[str, ...] = ROLES,
) -> dict[str, Any] | None:
    """Validate one atomic-enough inspect snapshot and return sanitized evidence."""
    if config["endpoints"]["ownership"] != "managed":
        return None
    containers: dict[str, dict[str, Any]] = {}
    legacy = all(role in inspections for role in roles)
    for instance in _instances(config, run_id, roles):
        role, name = instance["role"], instance["name"]
        key = role if legacy else instance["key"]
        name = container_name(run_id, role) if legacy else name
        inspect = inspections.get(key)
        if inspect is None or not _owned(inspect, run_id, role):
            return None
        state = inspect.get("State", {})
        if (
            state.get("Running")
            or state.get("Status") != "exited"
            or state.get("ExitCode") != 0
            or state.get("OOMKilled") is True
            or state.get("Error") not in (None, "")
        ):
            return None
        containers[key] = {
            "name": name,
            "owned": True,
            "status": "exited",
            "exit_code": 0,
            "oom_killed": False,
            "finished_at": state.get("FinishedAt", ""),
        }
    return {"state": "intentionally_stopped", "containers": containers}


def _intentional_shutdown_resume(
    previous: dict[str, Any], config: dict[str, Any], run_id: str,
    roles: tuple[str, ...] = ROLES,
) -> dict[str, Any] | None:
    """Recognize persisted or legacy clean shutdowns without weakening crash bounds."""
    if not _prior_managed_readiness(previous, config, run_id, roles):
        return None
    previous_containers = previous.get("containers")
    legacy_previous = isinstance(previous_containers, dict) and all(
        role in previous_containers for role in roles
    )
    current = _clean_owned_shutdown_evidence(config, run_id, roles)
    if current is not None and legacy_previous and not all(
        role in current.get("containers", {}) for role in roles
    ):
        migrated = {}
        for instance in _instances(config, run_id, roles):
            if instance["role"] in migrated:
                continue
            record = current["containers"].get(instance["key"])
            if not isinstance(record, dict):
                current = None
                break
            migrated[instance["role"]] = {
                **record, "name": container_name(run_id, instance["role"]),
            }
        if current is not None:
            current = {"state": "intentionally_stopped", "containers": migrated}
    if current is None and legacy_previous:
        legacy_inspections = {
            role: _inspect(container_name(run_id, role)) for role in roles
        }
        current = _clean_owned_shutdown_from_inspections(
            config, run_id, legacy_inspections, roles,
        )
    if current is None:
        return None
    recorded = previous.get("lifecycle")
    if recorded is None:
        # Backward compatibility for releases whose stop command did not persist
        # lifecycle evidence. Exact ownership plus clean exit of every role is the
        # narrow signature of the old explicit owned-only stop operation.
        return current
    if not isinstance(recorded, dict) or recorded.get("state") != "intentionally_stopped":
        return None
    expected = recorded.get("containers")
    if not isinstance(expected, dict):
        return None
    legacy = all(role in expected for role in roles)
    for instance in _instances(config, run_id, roles):
        key = instance["role"] if legacy else instance["key"]
        old = expected.get(key)
        now = current["containers"].get(key)
        if not isinstance(old, dict) or old.get("name") != now["name"] or old.get("owned") is not True:
            return None
    return current


def _container_matches_immutable_config(
    inspect: dict[str, Any], config: dict[str, Any], run_id: str, role: str
) -> bool:
    """Bind a stopped container to the immutable model, image, port, and GPUs."""
    if not _owned(inspect, run_id, role):
        return False
    model = config["models"][role]
    image_key = "image_edit_serving" if role == "image_edit" else "text_serving"
    if inspect.get("Config", {}).get("Image") != config["images"][image_key]:
        return False
    command = inspect.get("Config", {}).get("Cmd") or []
    pairs = {
        "--revision": str(model["revision"]),
        "--served-model-name": str(model["id"]),
        "--port": str(model["port"]),
    }
    for flag, value in pairs.items():
        if (
            flag not in command
            or command.index(flag) + 1 >= len(command)
            or command[command.index(flag) + 1] != value
        ):
            return False
    if role == "image_edit":
        if str(model["id"]) not in command:
            return False
    elif (
        "--model" not in command
        or command.index("--model") + 1 >= len(command)
        or command[command.index("--model") + 1] != model["id"]
    ):
        return False
    requests = inspect.get("HostConfig", {}).get("DeviceRequests") or []
    actual_gpu_ids = [str(item) for request in requests for item in (request.get("DeviceIDs") or [])]
    return actual_gpu_ids == [str(item) for item in config["endpoints"]["gpu_ids"][role]]


def _committed_sdg_execution_receipt(
    receipt_path: pathlib.Path, manifest_path: pathlib.Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Validate independently durable evidence that these endpoints completed SDG."""
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("workflow") != "tao-run-deft-iaa":
        raise ValueError("execution receipt workflow does not match tao-run-deft-iaa")
    sdg = receipt.get("config", {}).get("sdg")
    if not isinstance(sdg, dict) or sdg.get("endpoint_mode") != "managed":
        raise ValueError("execution receipt does not bind managed SDG configuration")
    expected = {
        "models": config["models"],
        "gpu_ids": config["endpoints"]["gpu_ids"],
        "images": config["images"],
    }
    for key, value in expected.items():
        if sdg.get(key) != value:
            raise ValueError(f"execution receipt SDG {key} does not match immutable config")
    matches = []
    committed_or_later = {"sdg", "visualize", "train", "evaluate", "gap_analysis"}
    for iter_label, record in (receipt.get("iterations") or {}).items():
        if (
            not isinstance(record, dict)
            or record.get("endpoint_manifest") != str(manifest_path.resolve())
            or record.get("stage_completed") not in committed_or_later
            or record.get("status") not in {"in_progress", "complete"}
        ):
            continue
        execution_path = pathlib.Path(str(record.get("sdg_execution_manifest", ""))).resolve()
        status_path = pathlib.Path(str(record.get("sdg_status", ""))).resolve()
        if (
            not execution_path.is_relative_to(receipt_path.parent.resolve())
            or not status_path.is_relative_to(receipt_path.parent.resolve())
        ):
            continue
        if not execution_path.is_file() or not status_path.is_file():
            continue
        execution = json.loads(execution_path.read_text())
        status = json.loads(status_path.read_text())
        if (
            int(execution.get("selected_sources", 0)) > 0
            and int(execution.get("accepted_sources", 0)) > 0
            and status.get("status") == "ok"
            and status.get("exit_code") == 0
        ):
            matches.append({
                "iteration": iter_label,
                "sdg_execution_manifest": str(execution_path.resolve()),
                "sdg_status": str(status_path.resolve()),
            })
    if not matches:
        raise ValueError(
            "execution receipt has no committed successful SDG stage bound to this endpoint manifest"
        )
    return {"path": str(receipt_path.resolve()), "matches": matches}


def recover_overwritten_stop(
    config: dict[str, Any], run_id: str, output: pathlib.Path, receipt_path: pathlib.Path
) -> dict[str, Any]:
    """Recover only the old-helper clean-stop manifest-overwrite signature."""
    if config["endpoints"]["ownership"] != "managed":
        raise ValueError("restart-budget recovery requires managed endpoints")
    if not output.is_file():
        raise ValueError("restart-budget recovery requires the overwritten endpoint manifest")
    previous_bytes = output.read_bytes()
    previous = json.loads(previous_bytes)
    expected_error = f"endpoint restart budget exhausted; inspect {output}"
    if (
        previous.get("schema_version") != "1"
        or previous.get("status") != "error"
        or previous.get("action") != "start"
        or previous.get("error") != expected_error
        or previous.get("restart_count") != 3
    ):
        raise ValueError("manifest is not the exact old-helper restart-budget overwrite signature")
    inspections = {role: _inspect(container_name(run_id, role)) for role in ROLES}
    lifecycle = _clean_owned_shutdown_from_inspections(config, run_id, inspections)
    if lifecycle is None:
        raise ValueError("restart-budget recovery requires every exact owned role to be cleanly stopped")
    if any(
        inspect is None or not _container_matches_immutable_config(inspect, config, run_id, role)
        for role, inspect in inspections.items()
    ):
        raise ValueError("stopped endpoint identity does not match immutable model/image/GPU configuration")
    receipt = _committed_sdg_execution_receipt(receipt_path.resolve(), output.resolve(), config)
    evidence_path = output.with_name(f"{output.stem}.restart-budget-error.json")
    if evidence_path.exists() and json.loads(evidence_path.read_text()) != previous:
        raise ValueError(
            f"refusing to replace different recovery evidence at {evidence_path}"
        )
    if not evidence_path.exists():
        atomic_json(evidence_path, previous)
    containers = {
        role: {
            "name": container_name(run_id, role), "owned": True,
            "model": config["models"][role]["id"],
            "probe": {
                "models_ok": True, "inference_ok": True,
                "evidence": "committed_sdg_execution_receipt",
            },
        }
        for role in ROLES
    }
    return {
        "schema_version": "1", "status": "success", "attempt": int(previous.get("attempt", 1)),
        "restart_count": 2, "ownership": "managed", "containers": containers,
        "lifecycle": lifecycle,
        "recovery": {
            "action": "recover-overwritten-stop",
            "source_error_manifest": str(evidence_path.resolve()),
            "source_error_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            "execution_receipt": receipt,
            "disposition": "intentional_shutdown_restored_without_starting_endpoints",
        },
    }


def stop(config: dict[str, Any], run_id: str, roles: tuple[str, ...] = ROLES) -> dict:
    if config["endpoints"]["ownership"] == "external":
        return {"ownership": "external", "stopped": [], "note": "user-managed endpoints were not changed"}
    stopped = []
    for instance in _instances(config, run_id, roles):
        role, name = instance["role"], instance["name"]
        inspect = _inspect(name)
        if inspect is None:
            continue
        if not _owned(inspect, run_id, role):
            raise ValueError(f"refusing to stop non-owned container {name}")
        if inspect.get("State", {}).get("Running"):
            _run(["docker", "stop", "--time", "30", name])
            stopped.append(name)
    # A pre-replication helper used the unqualified image-edit name. Preserve
    # owned-only stop compatibility, but never touch a foreign legacy service.
    if "image_edit" in roles:
        legacy_name = container_name(run_id, "image_edit")
        legacy = _inspect(legacy_name)
        if legacy is not None:
            if not _owned(legacy, run_id, "image_edit"):
                raise ValueError(f"refusing to stop non-owned container {legacy_name}")
            if legacy.get("State", {}).get("Running"):
                _run(["docker", "stop", "--time", "30", legacy_name])
                stopped.append(legacy_name)
    return {"ownership": "managed", "stopped": stopped, "removed": [], "note": "containers remain for resumability and diagnosis"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "plan", "component-status", "repair-created", "recover-overwritten-stop",
            "start", "status", "validate-external", "stop",
        ),
    )
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cache-dir", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--execution-receipt", type=pathlib.Path)
    parser.add_argument("--roles", help="unique comma-separated role subset")
    parser.add_argument(
        "--platform", default="docker",
        choices=(
            "host", "docker", "virtualenv", "brev", "slurm", "kubernetes", "airflow",
        ),
    )
    parser.add_argument("--service-host", default="127.0.0.1")
    parser.add_argument("--request-sha256")
    parser.add_argument("--gpu-identity-prefix")
    parser.add_argument("--image-edit-pool", type=pathlib.Path)
    parser.add_argument("--recreate-owned", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    attempt = 1
    restart_count = 0
    try:
        config = _load(args.config)
        roles = _selected_roles(args.roles)
        if not re.fullmatch(r"[A-Za-z0-9.-]+", args.service_host):
            raise ValueError("--service-host must be a hostname or IPv4 address without a scheme or port")
        remote_platform = args.platform in {"brev", "slurm", "kubernetes"}
        if remote_platform and args.service_host in {"127.0.0.1", "localhost"} and "image_edit" in roles:
            raise ValueError("remote image-edit workers require an explicit reachable --service-host")
        if args.request_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", args.request_sha256):
            raise ValueError("--request-sha256 must be a lowercase SHA-256 digest")
        if remote_platform and args.request_sha256 is None:
            raise ValueError("remote endpoint lifecycle requires --request-sha256")
        prefix = args.gpu_identity_prefix or f"{socket.gethostname()}/{args.run_id}"
        if not re.fullmatch(r"[A-Za-z0-9._:-]+(?:/[A-Za-z0-9._:-]+)*", prefix):
            raise ValueError("--gpu-identity-prefix contains unsupported characters")
        if remote_platform and args.gpu_identity_prefix is None and "image_edit" in roles:
            raise ValueError("remote image-edit workers require --gpu-identity-prefix")
        if args.recreate_owned and not (
            args.action == "start" and remote_platform and roles == ("image_edit",)
            and args.request_sha256 is not None
        ):
            raise ValueError(
                "--recreate-owned is valid only for signed remote start --roles image_edit"
            )
        cache_dir = (args.cache_dir or pathlib.Path(config["endpoints"].get("cache_dir") or "~/.cache/huggingface")).expanduser().resolve()
        bounded = args.action in {"start", "validate-external"} and args.output is not None
        previous = json.loads(args.output.read_text()) if args.output and args.output.is_file() else None
        recreate_owned = args.recreate_owned
        if (
            recreate_owned and isinstance(previous, dict)
            and previous.get("status") == "success"
            and previous.get("auth_rotation", {}).get("status") == "complete"
            and previous.get("auth_rotation", {}).get("request_sha256") == args.request_sha256
        ):
            # One auth rotation is permitted for each signed request. A retry
            # can reuse/restart those exact workers but cannot rotate them again.
            recreate_owned = False
        if args.action == "recover-overwritten-stop":
            if args.output is None or args.execution_receipt is None:
                raise ValueError("recover-overwritten-stop requires --output and --execution-receipt")
            report = recover_overwritten_stop(config, args.run_id, args.output.resolve(), args.execution_receipt)
            atomic_json(args.output.resolve(), report)
            print(json.dumps(report, sort_keys=True))
            return 0
        intentional_shutdown = None
        if bounded and isinstance(previous, dict):
            if previous.get("status") == "success":
                if (
                    not recreate_owned
                    and (args.action != "start" or _all_managed_roles_running(config, args.run_id, roles))
                ):
                    print(json.dumps(previous, sort_keys=True))
                    return 0
                if not _prior_managed_readiness(previous, config, args.run_id, roles):
                    raise ValueError("successful endpoint manifest lacks prior owned readiness evidence")
                restart_count = int(previous.get("restart_count", 0))
                attempt = int(previous.get("attempt", 1))
                if not recreate_owned:
                    intentional_shutdown = _intentional_shutdown_resume(previous, config, args.run_id, roles)
                    if intentional_shutdown is None:
                        restart_count += 1
                        if restart_count > 2:
                            raise ValueError(f"endpoint restart budget exhausted; inspect {args.output}")
            else:
                attempt = int(previous.get("attempt", 0)) + 1
                if attempt > 2:
                    raise ValueError(f"endpoint launch attempt budget exhausted; inspect {args.output}")
        if args.action == "plan":
            report = preflight(
                config, args.run_id, cache_dir, roles, platform=args.platform,
            ) if config["endpoints"]["ownership"] == "managed" else {
                "ownership": "external",
                "urls": {role: endpoint_url(config, role) for role in ROLES},
                "component_images": component_status(config)["components"],
            }
        elif args.action == "component-status":
            report = component_status(config)
        elif args.action == "start":
            report = start(
                config, args.run_id, cache_dir, roles, platform=args.platform,
                service_host=args.service_host, request_sha256=args.request_sha256,
                image_edit_pool=args.image_edit_pool, gpu_identity_prefix=prefix,
                recreate_owned=recreate_owned,
            )
        elif args.action == "repair-created":
            report = repair_created(config, args.run_id)
        elif args.action == "status":
            report = status(config, args.run_id, roles)
        elif args.action == "validate-external":
            report = validate_external(config)
        else:
            report = stop(config, args.run_id, roles)
            if (
                config["endpoints"]["ownership"] == "managed"
                and args.output
                and isinstance(previous, dict)
                and previous.get("status") == "success"
            ):
                if not _prior_managed_readiness(previous, config, args.run_id, roles):
                    raise ValueError("refusing to attach shutdown evidence to an invalid endpoint manifest")
                lifecycle = _clean_owned_shutdown_evidence(config, args.run_id, roles)
                if lifecycle is None:
                    raise ValueError("owned endpoint shutdown did not leave every managed role cleanly stopped")
                report = {**previous, "lifecycle": lifecycle, "shutdown": report}
        if bounded:
            report = {
                **report, "schema_version": "1", "status": "success", "attempt": attempt,
                "restart_count": restart_count,
                "lifecycle": {"state": "running"},
            }
        if args.output:
            atomic_json(args.output.resolve(), report)
        print(json.dumps(report, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, yaml.YAMLError, subprocess.SubprocessError) as exc:
        if args.output and args.action in {"start", "validate-external"}:
            atomic_json(args.output.resolve(), {
                "schema_version": "1", "status": "error", "attempt": attempt,
                "restart_count": restart_count, "action": args.action, "error": str(exc),
            })
        print(f"manage_sdg_endpoints[{args.action}]: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
