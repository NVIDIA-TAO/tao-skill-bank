# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight, start, validate, and stop run-scoped local IAA model endpoints."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
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
    wait_until_ready,
)


WORKFLOW_LABEL = "com.nvidia.tao.workflow=tao-run-deft-iaa"
WORKFLOW_COMPONENTS = ("augmentation", "auto_labeling")


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


def preflight(config: dict[str, Any], run_id: str, cache_dir: pathlib.Path) -> dict:
    if not shutil.which("docker") or not shutil.which("nvidia-smi"):
        raise ValueError("managed endpoints require docker and nvidia-smi")
    info = _run(["docker", "info", "--format", "{{json .Runtimes}}"])
    runtimes = json.loads(info.stdout)
    if "nvidia" not in runtimes:
        raise ValueError("Docker NVIDIA runtime is unavailable")
    disk_probe = cache_dir
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    free_bytes = shutil.disk_usage(disk_probe).free
    if free_bytes < 150 * 1024**3:
        raise ValueError(f"model cache has only {free_bytes // 1024**3} GiB free; require at least 150 GiB")
    inventory = _inventory()
    conflicts: list[str] = []
    active_roles: set[str] = set()
    for role in ROLES:
        name = container_name(run_id, role)
        existing = _inspect(name)
        if existing is not None:
            if not _owned(existing, run_id, role):
                conflicts.append(f"container name {name} belongs to another owner")
            elif existing.get("State", {}).get("Running"):
                active_roles.add(role)
            continue
        port = config["models"][role]["port"]
        if not port_available(port):
            conflicts.append(f"127.0.0.1:{port} is already in use for managed role {role}")
    if conflicts:
        raise ValueError("; ".join(conflicts))
    validate_gpu_inventory(config, inventory, active_roles)
    return {
        "ownership": "managed",
        "run_id": run_id,
        "cache_dir": str(cache_dir.resolve()),
        "gpu_inventory": inventory,
        "commands": {role: build_endpoint_command(config, role, run_id, cache_dir) for role in ROLES},
        "component_images": component_status(config)["components"],
    }


def start(config: dict[str, Any], run_id: str, cache_dir: pathlib.Path) -> dict:
    report = preflight(config, run_id, cache_dir)
    report["component_images"] = _require_component_images(config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    containers: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        name = container_name(run_id, role)
        existing = _inspect(name)
        if existing is None:
            result = _run(build_endpoint_command(config, role, run_id, cache_dir))
            created = True
            container_id = result.stdout.strip()
        else:
            if not _owned(existing, run_id, role):
                raise ValueError(f"refusing to replace non-owned container {name}")
            created = False
            container_id = existing.get("Id", "")
            if not existing.get("State", {}).get("Running"):
                _run(["docker", "start", name])
        probe = wait_until_ready(
            lambda role=role: readiness_probe(config, role),
            config["endpoints"]["startup_timeout_s"],
            config["endpoints"]["retry_interval_s"],
        )
        containers[role] = {
            "name": name,
            "id": container_id[:12],
            "owned": True,
            "created_this_call": created,
            "url": endpoint_url(config, role),
            "model": config["models"][role]["id"],
            "probe": probe,
            "log_command": ["docker", "logs", "--tail", "200", name],
        }
    return {**report, "containers": containers}


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


def status(config: dict[str, Any], run_id: str) -> dict:
    if config["endpoints"]["ownership"] == "external":
        return validate_external(config)
    containers = {}
    for role in ROLES:
        name = container_name(run_id, role)
        inspect = _inspect(name)
        containers[role] = {
            "name": name,
            "exists": inspect is not None,
            "owned": bool(inspect and _owned(inspect, run_id, role)),
            "running": bool(inspect and inspect.get("State", {}).get("Running")),
            "status": inspect.get("State", {}).get("Status") if inspect else "missing",
        }
    return {"ownership": "managed", "containers": containers, "component_images": component_status(config)["components"]}


def stop(config: dict[str, Any], run_id: str) -> dict:
    if config["endpoints"]["ownership"] == "external":
        return {"ownership": "external", "stopped": [], "note": "user-managed endpoints were not changed"}
    stopped = []
    for role in ROLES:
        name = container_name(run_id, role)
        inspect = _inspect(name)
        if inspect is None:
            continue
        if not _owned(inspect, run_id, role):
            raise ValueError(f"refusing to stop non-owned container {name}")
        if inspect.get("State", {}).get("Running"):
            _run(["docker", "stop", "--time", "30", name])
            stopped.append(name)
    return {"ownership": "managed", "stopped": stopped, "removed": [], "note": "containers remain for resumability and diagnosis"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("plan", "component-status", "start", "status", "validate-external", "stop"),
    )
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cache-dir", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    attempt = 1
    try:
        config = _load(args.config)
        cache_dir = (args.cache_dir or pathlib.Path(config["endpoints"].get("cache_dir") or "~/.cache/huggingface")).expanduser().resolve()
        bounded = args.action in {"start", "validate-external"} and args.output is not None
        if bounded and args.output.is_file():
            previous = json.loads(args.output.read_text())
            if previous.get("status") == "success":
                print(json.dumps(previous, sort_keys=True))
                return 0
            attempt = int(previous.get("attempt", 0)) + 1
            if attempt > 2:
                raise ValueError(f"endpoint launch attempt budget exhausted; inspect {args.output}")
        if args.action == "plan":
            report = preflight(config, args.run_id, cache_dir) if config["endpoints"]["ownership"] == "managed" else {
                "ownership": "external",
                "urls": {role: endpoint_url(config, role) for role in ROLES},
                "component_images": component_status(config)["components"],
            }
        elif args.action == "component-status":
            report = component_status(config)
        elif args.action == "start":
            report = start(config, args.run_id, cache_dir)
        elif args.action == "status":
            report = status(config, args.run_id)
        elif args.action == "validate-external":
            report = validate_external(config)
        else:
            report = stop(config, args.run_id)
        if bounded:
            report = {**report, "schema_version": "1", "status": "success", "attempt": attempt}
        if args.output:
            atomic_json(args.output.resolve(), report)
        print(json.dumps(report, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, yaml.YAMLError, subprocess.SubprocessError) as exc:
        if args.output and args.action in {"start", "validate-external"}:
            atomic_json(args.output.resolve(), {
                "schema_version": "1", "status": "error", "attempt": attempt,
                "action": args.action, "error": str(exc),
            })
        print(f"manage_sdg_endpoints[{args.action}]: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
