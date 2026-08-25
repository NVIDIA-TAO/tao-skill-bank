# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic contracts for local IAA generation and normalization."""

from __future__ import annotations

import hashlib
import datetime
import json
import math
import os
import pathlib
import re
import shutil
import socket
import tempfile
import time
import urllib.error
import urllib.request
import urllib.parse
from typing import Any, Callable, Iterable
from collections import Counter


ROLES = ("image_edit", "vlm", "llm")
BACKENDS = {"image_edit": "vllm-omni", "vlm": "vllm", "llm": "vllm"}
QUERY_LEVELS = ("easy", "medium", "hard")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VECTOR_ATTRIBUTES = (
    "top outer color",
    "top outer type",
    "bottom color",
    "bottom type",
    "shoe color",
    "shoe type",
    "viewpoint",
)
EDITABLE_ATTRIBUTES = {
    "top outer color", "top outer type", "bottom color", "bottom type",
}
TEXT_WIDTH = {"easy": 4, "medium": 6, "hard": 7}


def validate_normalized_dataset(
    manifest_path: pathlib.Path,
    pairs_path: pathlib.Path,
    image_list_path: pathlib.Path,
) -> dict[str, int]:
    """Validate the complete normalized dataset consumed by CLIP training."""

    dataset = manifest_path.parent
    if (
        not dataset.is_dir() or dataset.is_symlink() or dataset.resolve() != dataset
        or pairs_path != dataset / "sdg_pairs.json"
        or image_list_path != dataset / "sdg_image_list.txt"
    ):
        raise ValueError("normalized SDG dataset paths are noncanonical or unsafe")
    if any(path.is_symlink() for path in dataset.rglob("*")):
        raise ValueError("normalized SDG dataset contains a symlink")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    names = [
        line.strip() for line in image_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not isinstance(manifest, dict) or not isinstance(pairs, list) or not pairs:
        raise ValueError("normalized SDG manifest or pairs are malformed")
    if (
        manifest.get("schema_version") != "1"
        or manifest.get("dataset_format_version") != 3
        or manifest.get("rejected_samples_included") != 0
        or manifest.get("num_pairs") != len(pairs)
        or len(names) != len(pairs) or len(set(names)) != len(names)
    ):
        raise ValueError("normalized SDG dataset counts or policy disagree")
    path_contract = {
        "image_dir": ("dataset", "images"),
        "caption_dir": ("dataset", "captions"),
        "image_list_file": ("dataset", "sdg_image_list.txt"),
        "pairs_file": ("dataset", "sdg_pairs.json"),
        "attribute_vocab_file": ("dataset", "attribute_vocab.json"),
    }
    for field, suffix in path_contract.items():
        value = manifest.get(field)
        path = pathlib.PurePosixPath(value) if isinstance(value, str) else None
        if path is None or not path.is_absolute() or tuple(path.parts[-2:]) != suffix:
            raise ValueError(f"normalized SDG manifest has a noncanonical {field}")
    image_root = dataset / "images"
    caption_root = dataset / "captions"
    vocab = dataset / "attribute_vocab.json"
    if (
        not image_root.is_dir() or image_root.is_symlink()
        or not caption_root.is_dir() or caption_root.is_symlink()
        or not vocab.is_file() or vocab.is_symlink() or vocab.stat().st_size <= 0
    ):
        raise ValueError("normalized SDG image, caption, or vocabulary artifacts are missing")
    image_files = {
        path.name: path for path in image_root.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    caption_files = {
        path.name: path for path in caption_root.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    expected_captions = {pathlib.Path(name).with_suffix(".txt").name for name in names}
    if set(image_files) != set(names) or set(caption_files) != expected_captions:
        raise ValueError("normalized SDG dataset file set disagrees with the image list")
    required = {
        "unique_name", "caption", "image_path", "query_type", "source_split",
        "is_augmented", "verification_metadata_sha256",
    }
    for index, (name, row) in enumerate(zip(names, pairs)):
        if (
            not isinstance(row, dict) or not required.issubset(row)
            or row.get("unique_name") != name
            or pathlib.Path(name).name != name
            or pathlib.Path(name).suffix.lower() not in IMAGE_SUFFIXES
            or row.get("image_path") != f"images/{name}"
            or row.get("query_type") not in QUERY_LEVELS
            or row.get("source_split") != "train"
            or row.get("is_augmented") is not True
            or not isinstance(row.get("caption"), str) or not row["caption"].strip()
            or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("verification_metadata_sha256", "")))
            or image_files[name].stat().st_size <= 0
        ):
            raise ValueError(f"normalized SDG pair {index} is invalid")
        caption_path = caption_root / pathlib.Path(name).with_suffix(".txt")
        if caption_path.stat().st_size <= 0 or caption_path.read_text(encoding="utf-8").strip() != row["caption"].strip():
            raise ValueError(f"normalized SDG caption {index} disagrees with its pair")
    sources = manifest.get("accepted_provenance")
    if (
        not isinstance(sources, list) or not sources
        or manifest.get("num_source_images") != len(sources)
    ):
        raise ValueError("normalized SDG accepted provenance is missing or inconsistent")
    return {"pairs": len(pairs), "sources": len(sources), "images": len(image_files)}


def atomic_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_gpu_ids(value: Any, role: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"endpoints.gpu_ids.{role} must be a non-empty list")
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value):
        raise ValueError(f"endpoints.gpu_ids.{role} must contain non-negative integers")
    if len(set(value)) != len(value):
        raise ValueError(f"endpoints.gpu_ids.{role} contains duplicate IDs")
    return list(value)


def validate_config(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != "1":
        raise ValueError("SDG config must be a schema-v1 object")
    images = payload.get("images")
    models = payload.get("models")
    endpoints = payload.get("endpoints")
    generation = payload.get("generation")
    if not all(isinstance(item, dict) for item in (images, models, endpoints, generation)):
        raise ValueError("SDG config requires images, models, endpoints, and generation objects")
    for name, image in images.items():
        if (
            not isinstance(image, str)
            or not re.fullmatch(r"[^\s@]+(?:[:][^\s@]+)?@sha256:[0-9a-f]{64}", image)
        ):
            raise ValueError(f"images.{name} must be pinned by sha256 digest")
    ownership = endpoints.get("ownership")
    if ownership not in {"managed", "external"}:
        raise ValueError("endpoints.ownership must be managed or external")
    reuse_requested = endpoints.get("reuse_requested", False)
    if not isinstance(reuse_requested, bool):
        raise ValueError("endpoints.reuse_requested must be boolean")
    if ownership == "external" and reuse_requested is not True:
        raise ValueError("external endpoints require explicit user-requested reuse evidence")
    if ownership == "managed" and reuse_requested:
        raise ValueError("managed endpoints cannot claim external reuse")
    forward_hf_token = endpoints.get("forward_hf_token", False)
    if not isinstance(forward_hf_token, bool):
        raise ValueError("endpoints.forward_hf_token must be boolean")
    gpu_map = endpoints.get("gpu_ids")
    urls = endpoints.get("external_urls")
    if not isinstance(gpu_map, dict) or not isinstance(urls, dict):
        raise ValueError("endpoints.gpu_ids and endpoints.external_urls must be objects")
    ports: set[int] = set()
    for role in ROLES:
        model = models.get(role)
        if not isinstance(model, dict):
            raise ValueError(f"models.{role} must be an object")
        for field in ("id", "revision", "backend"):
            if not isinstance(model.get(field), str) or not model[field].strip():
                raise ValueError(f"models.{role}.{field} must be non-empty")
        if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            raise ValueError(f"models.{role}.revision must be an immutable 40-character commit")
        if model["backend"] != BACKENDS[role]:
            raise ValueError(f"models.{role}.backend must be {BACKENDS[role]}")
        port = model.get("port")
        if not isinstance(port, int) or isinstance(port, bool) or not 1024 <= port <= 65535:
            raise ValueError(f"models.{role}.port must be in [1024, 65535]")
        if port in ports:
            raise ValueError(f"models.{role}.port duplicates another role: {port}")
        ports.add(port)
        if ownership == "managed":
            parse_gpu_ids(gpu_map.get(role), role)
            if urls.get(role):
                raise ValueError(f"external URL is not valid for managed role {role}")
        else:
            url = urls.get(role)
            if not isinstance(url, str) or not re.fullmatch(r"https?://[^\s]+", url):
                raise ValueError(f"external_urls.{role} must be an HTTP(S) URL")
            parsed = urllib.parse.urlsplit(url)
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError(f"external_urls.{role} must not contain credentials, query, or fragment")
            if parsed.hostname not in {"127.0.0.1", "localhost", "::1", "host.docker.internal"}:
                raise ValueError(f"external_urls.{role} must identify a local endpoint host")
            if gpu_map.get(role):
                raise ValueError(f"external role {role} must not claim local GPU IDs")
    for key in ("startup_timeout_s", "request_timeout_s", "retry_interval_s"):
        if not isinstance(endpoints.get(key), int) or endpoints[key] < 1:
            raise ValueError(f"endpoints.{key} must be an integer >= 1")
    retries = generation.get("verification_max_attempts")
    if not isinstance(retries, int) or isinstance(retries, bool) or not 1 <= retries <= 5:
        raise ValueError("generation.verification_max_attempts must be in [1, 5]")
    budget = generation.get("max_samples_per_iteration")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError("generation.max_samples_per_iteration must be >= 1")
    image_edit_timeout = generation.get("image_edit_request_timeout_s", 600)
    if not isinstance(image_edit_timeout, int) or isinstance(image_edit_timeout, bool) or image_edit_timeout < 1:
        raise ValueError("generation.image_edit_request_timeout_s must be an integer >= 1")
    max_in_flight = generation.get("max_in_flight")
    for field in ("generation_nodes", "gpus_per_generation_node"):
        value = generation.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"generation.{field} must be a positive integer")
    last_image_port = (
        models["image_edit"]["port"] + generation["gpus_per_generation_node"] - 1
    )
    if last_image_port > 65535:
        raise ValueError(
            "image-edit base port plus gpus_per_generation_node exceeds 65535"
        )
    auxiliary_ports = {models[role]["port"] for role in ("vlm", "llm")}
    image_ports = set(range(models["image_edit"]["port"], last_image_port + 1))
    if image_ports & auxiliary_ports:
        raise ValueError(
            "image-edit service port range must not overlap the VLM or LLM port"
        )
    if max_in_flight is not None:
        if not isinstance(max_in_flight, int) or isinstance(max_in_flight, bool) or max_in_flight < 1:
            raise ValueError("generation.max_in_flight must be an integer >= 1")
        if ownership == "managed":
            worker_count = len(parse_gpu_ids(gpu_map.get("image_edit"), "image_edit"))
            if max_in_flight > worker_count:
                raise ValueError(
                    "generation.max_in_flight cannot exceed the managed image-edit worker count"
                )
    return payload


def validate_image_edit_endpoint_pool(payload: Any) -> dict[str, Any]:
    """Validate a runtime-resolved one-slot-per-GPU endpoint pool."""
    if not isinstance(payload, dict) or payload.get("schema_version") != "1":
        raise ValueError("image-edit endpoint pool must be a schema-v1 object")
    expected_top = {
        "schema_version", "platform", "model", "required_capacity", "auth_env",
        "endpoints", "created_at", "request_sha256",
    }
    if set(payload) != expected_top:
        raise ValueError("image-edit endpoint pool has missing or unexpected top-level fields")
    platform = payload.get("platform")
    if platform not in {
        "host", "docker", "slurm", "brev", "virtualenv", "kubernetes", "airflow",
    }:
        raise ValueError("image-edit endpoint pool platform is invalid")
    model = payload.get("model")
    if not isinstance(model, dict) or set(model) != {"id", "revision"}:
        raise ValueError("image-edit endpoint pool model binding is invalid")
    if not all(isinstance(model.get(key), str) and model[key] for key in ("id", "revision")):
        raise ValueError("image-edit endpoint pool model fields must be non-empty")
    auth_env = payload.get("auth_env")
    if auth_env not in {None, "IMAGE_EDIT_API_KEY"}:
        raise ValueError("image-edit endpoint pool auth_env is invalid")
    created_at = payload.get("created_at")
    if not isinstance(created_at, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)", created_at
    ):
        raise ValueError("image-edit endpoint pool created_at must be UTC ISO-8601")
    try:
        timestamp = datetime.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("image-edit endpoint pool created_at must be UTC ISO-8601") from exc
    if timestamp.utcoffset() != datetime.timedelta(0):
        raise ValueError("image-edit endpoint pool created_at must use UTC")
    created_at = timestamp.isoformat().replace("+00:00", "Z")
    request_sha256 = payload.get("request_sha256")
    if not isinstance(request_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", request_sha256):
        raise ValueError("image-edit endpoint pool request_sha256 is invalid")
    endpoints = payload.get("endpoints")
    if not isinstance(endpoints, list) or not endpoints:
        raise ValueError("image-edit endpoint pool must contain endpoints")
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    urls: set[str] = set()
    gpu_identities: set[str] = set()
    for index, item in enumerate(endpoints):
        if not isinstance(item, dict):
            raise ValueError(f"image-edit endpoint pool entry {index} must be an object")
        endpoint_id = item.get("id")
        url = item.get("url")
        capacity = item.get("capacity")
        gpu_identity = item.get("gpu_identity")
        owner = item.get("owner")
        if set(item) != {"id", "url", "capacity", "gpu_identity", "owner"}:
            raise ValueError(f"image-edit endpoint pool entry {index} has unexpected fields")
        if not isinstance(endpoint_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", endpoint_id):
            raise ValueError(f"image-edit endpoint pool entry {index} has an invalid ID")
        if endpoint_id in ids:
            raise ValueError("image-edit endpoint pool contains a duplicate ID")
        ids.add(endpoint_id)
        if not isinstance(url, str) or not re.fullmatch(r"https?://[^\s]+", url):
            raise ValueError(f"image-edit endpoint pool entry {index} URL must be HTTP(S)")
        parsed = urllib.parse.urlsplit(url)
        if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("image-edit endpoint pool URLs must not contain credentials, query, or fragment")
        normalized_url = url.rstrip("/")
        if normalized_url in urls:
            raise ValueError("image-edit endpoint pool contains a duplicate URL")
        urls.add(normalized_url)
        if capacity != 1 or isinstance(capacity, bool):
            raise ValueError("image-edit endpoint pool capacity must be exactly 1 per endpoint")
        if not isinstance(gpu_identity, str) or not gpu_identity.strip():
            raise ValueError("image-edit endpoint pool requires a non-empty gpu_identity")
        if gpu_identity in gpu_identities:
            raise ValueError("image-edit endpoint pool contains a duplicate gpu_identity")
        gpu_identities.add(gpu_identity)
        if not isinstance(owner, dict) or set(owner) != {"native_id", "name"}:
            raise ValueError("image-edit endpoint pool owner binding is invalid")
        if any(
            not isinstance(owner.get(key), str)
            or not owner[key]
            or len(owner[key]) > 256
            or any(char in owner[key] for char in "\n\r\0")
            for key in ("native_id", "name")
        ):
            raise ValueError("image-edit endpoint pool owner fields must be sanitized")
        result.append({
            "id": endpoint_id, "url": normalized_url, "capacity": 1,
            "gpu_identity": gpu_identity,
            "owner": {"native_id": owner["native_id"], "name": owner["name"]},
        })
    required_capacity = payload.get("required_capacity")
    if (
        not isinstance(required_capacity, int) or isinstance(required_capacity, bool)
        or required_capacity != len(result)
    ):
        raise ValueError("image-edit endpoint pool required_capacity must equal endpoint count")
    return {
        "schema_version": "1", "platform": platform,
        "model": {"id": model["id"], "revision": model["revision"]},
        "required_capacity": required_capacity, "auth_env": auth_env,
        "endpoints": result, "created_at": created_at,
        "request_sha256": request_sha256,
    }


# The workflow permits the initial controller call plus one evidence-based
# correction. Endpoint-pool lineage must never create a third implicit retry.
MAX_CONTROLLER_POOL_SNAPSHOTS = 2


def _pool_snapshot(binding: dict[str, Any], entries: Any) -> dict[str, Any]:
    """Validate and reduce one endpoint pool to durable resume provenance."""
    if not isinstance(entries, list):
        raise ValueError("resumable image-edit endpoint snapshot must be a list")
    payload = dict(binding)
    payload["endpoints"] = entries
    payload["required_capacity"] = sum(
        item.get("capacity", 0) if isinstance(item, dict) else 0 for item in entries
    )
    validated = validate_image_edit_endpoint_pool(payload)
    if validated["required_capacity"] != binding["required_capacity"]:
        raise ValueError("resumable image-edit endpoint snapshot capacity changed")
    return {
        "platform": validated["platform"],
        "model": validated["model"],
        "required_capacity": validated["required_capacity"],
        "request_sha256": validated["request_sha256"],
        "endpoints": validated["endpoints"],
    }


def _validate_pool_snapshot(binding: dict[str, Any], snapshot: Any) -> dict[str, Any]:
    if not isinstance(snapshot, dict) or set(snapshot) != {
        "platform", "model", "required_capacity", "request_sha256", "endpoints",
    }:
        raise ValueError("resumable image-edit endpoint history is malformed")
    validated = _pool_snapshot(binding, snapshot["endpoints"])
    if snapshot != validated:
        raise ValueError("resumable image-edit endpoint history identity changed")
    return validated


def bind_resumable_endpoint_pool(
    progress: dict[str, Any], selected: list[dict[str, Any]],
    endpoint_entries: list[dict[str, Any]], binding: dict[str, Any] | None,
    *, allow_unstarted_rebind: bool = False,
) -> None:
    """Bind a validated pool during explicit unfinished-work resume.

    Completed per-sample records are deliberately untouched. Only pool
    snapshots and the current scheduling pool change.
    """
    prior_endpoint_entries = progress.get("image_edit_endpoints")
    history = progress.get("image_edit_endpoint_history", [])
    if not isinstance(history, list):
        raise ValueError("resumable image-edit endpoint history must be a list")
    if binding is None:
        if history:
            raise ValueError("endpoint history requires a validated runtime pool")
        if prior_endpoint_entries is not None and prior_endpoint_entries != endpoint_entries:
            raise ValueError("image-edit endpoint pool changed outside explicit platform resume")
    else:
        history = [_validate_pool_snapshot(binding, item) for item in history]
        if prior_endpoint_entries is not None:
            prior_snapshot = _pool_snapshot(binding, prior_endpoint_entries)
            if prior_endpoint_entries != endpoint_entries:
                selected_keys = {str(item.get("source_key", "")) for item in selected}
                terminal_augmentation = {
                    key for key, value in progress.get("augmentation", {}).items()
                    if isinstance(value, dict) and value.get("status") in {"accepted", "rejected"}
                }
                accepted_keys = {
                    key for key, value in progress.get("augmentation", {}).items()
                    if isinstance(value, dict) and value.get("status") == "accepted"
                }
                unfinished = (
                    selected_keys - terminal_augmentation
                    or not progress.get("split", False)
                    or accepted_keys - {
                        key for key, value in progress.get("labeling", {}).items()
                        if value == "accepted"
                    }
                )
                if not unfinished:
                    raise ValueError(
                        "image-edit endpoint pool changed outside explicit unfinished resume"
                    )
                if not progress.get("augmentation") and not allow_unstarted_rebind:
                    raise ValueError(
                        "image-edit endpoint pool changed outside explicit unfinished resume"
                    )
                if prior_snapshot not in history:
                    history.append(prior_snapshot)
                if len(history) + 1 > MAX_CONTROLLER_POOL_SNAPSHOTS:
                    raise ValueError("image-edit endpoint pool resume attempt budget exhausted")
        progress["image_edit_endpoint_pool"] = _pool_snapshot(binding, endpoint_entries)
        progress["image_edit_endpoint_history"] = history
    progress["image_edit_endpoints"] = endpoint_entries


def container_name(run_id: str, role: str, gpu_id: int | None = None) -> str:
    if role not in ROLES:
        raise ValueError(f"unknown model role: {role}")
    slug = re.sub(r"[^a-z0-9-]+", "-", run_id.lower()).strip("-")
    if not slug:
        raise ValueError("run ID does not contain a usable character")
    base = f"tao-deft-iaa-{slug[:32]}-{role.replace('_', '-')}"
    if gpu_id is not None:
        if role != "image_edit" or not isinstance(gpu_id, int) or isinstance(gpu_id, bool) or gpu_id < 0:
            raise ValueError("a GPU-qualified container name requires a non-negative image-edit GPU ID")
        return f"{base}-gpu-{gpu_id}"
    return base


def endpoint_url(
    config: dict[str, Any], role: str, *, ordinal: int | None = None,
    service_host: str = "127.0.0.1",
) -> str:
    validate_config(config)
    if config["endpoints"]["ownership"] == "external":
        return config["endpoints"]["external_urls"][role].rstrip("/")
    port = config["models"][role]["port"] + ((ordinal or 0) if role == "image_edit" else 0)
    return f"http://{service_host}:{port}/v1"


def build_endpoint_command(
    config: dict[str, Any], role: str, run_id: str, cache_dir: pathlib.Path,
    *, image_edit_gpu_id: int | None = None, image_edit_ordinal: int | None = None,
    publish_host: str = "127.0.0.1", authenticated: bool = False,
) -> list[str]:
    validate_config(config)
    if config["endpoints"]["ownership"] != "managed":
        raise ValueError("external endpoints are validated, never started")
    model = config["models"][role]
    gpu_ids = parse_gpu_ids(config["endpoints"]["gpu_ids"][role], role)
    replicated_worker = image_edit_gpu_id is not None
    port = model["port"]
    name = container_name(run_id, role)
    if image_edit_gpu_id is not None or image_edit_ordinal is not None:
        if role != "image_edit" or image_edit_gpu_id not in gpu_ids:
            raise ValueError("image-edit instance GPU must be explicitly configured")
        if not isinstance(image_edit_ordinal, int) or isinstance(image_edit_ordinal, bool) or image_edit_ordinal < 0:
            raise ValueError("image-edit instance ordinal must be a non-negative integer")
        gpu_ids = [image_edit_gpu_id]
        port += image_edit_ordinal
        if port > 65535:
            raise ValueError("image-edit instance port exceeds 65535")
        name = container_name(run_id, role, image_edit_gpu_id)
    image_key = "image_edit_serving" if role == "image_edit" else "text_serving"
    argv = [
        "docker", "run", "-d", "--name", name,
        "--label", "com.nvidia.tao.workflow=tao-run-deft-iaa",
        "--label", f"com.nvidia.tao.run={run_id}",
        "--label", f"com.nvidia.tao.role={role}",
        # Docker's --gpus parser requires the comma-separated device selector to
        # remain one quoted value. Without the literal quotes it interprets the
        # trailing IDs as a device count (for example Count=3, DeviceIDs=["0"]).
        "--gpus", f'"device={",".join(str(item) for item in gpu_ids)}"',
        "-p", f"{publish_host}:{port}:{port}",
        "-v", f"{cache_dir.resolve()}:/root/.cache/huggingface",
    ]
    # The image-edit server uses multiple diffusion worker processes. Docker's
    # 64 MiB default /dev/shm can terminate those workers with SIGBUS after a
    # few otherwise successful requests, so give this managed endpoint an
    # explicit, bounded shared-memory allocation.
    if role == "image_edit":
        if publish_host != "127.0.0.1":
            if not authenticated:
                raise ValueError("non-loopback image-edit publishing requires API authentication")
            argv += ["-e", "VLLM_API_KEY"]
        argv += ["--shm-size", "16g"]
    if config["endpoints"].get("forward_hf_token", False):
        if not os.environ.get("HF_TOKEN"):
            raise ValueError(
                "approved HF_TOKEN forwarding requires HF_TOKEN in the process environment"
            )
        argv += ["-e", "HF_TOKEN"]
    argv.append(config["images"][image_key])
    if role == "image_edit":
        argv += [
            "vllm", "serve", model["id"], "--omni", "--host", "0.0.0.0",
            "--port", str(port), "--revision", model["revision"],
            "--served-model-name", model["id"], "--tensor-parallel-size",
            "1" if replicated_worker else str(len(gpu_ids)),
        ]
    else:
        argv += [
            "--model", model["id"], "--host", "0.0.0.0", "--port", str(model["port"]),
            "--revision", model["revision"], "--served-model-name", model["id"],
            "--tensor-parallel-size", str(len(gpu_ids)),
        ]
    if "all" in argv:
        raise AssertionError("explicit GPU selection was widened")
    return argv


def build_component_command(
    config: dict[str, Any], action: str, *, input_root: pathlib.Path,
    output_root: pathlib.Path, source_key: str = "", attempt: int = 1,
    target_attributes: dict[str, str] | None = None,
    image_edit_url: str | None = None,
    endpoint_urls: dict[str, str] | None = None,
) -> list[str]:
    """Build one PAIDF component call without shell interpolation."""
    validate_config(config)
    uid_gid = f"{os.getuid()}:{os.getgid()}"
    base = [
        "docker", "run", "--rm", "--user", uid_gid,
        # The pinned component images install their Python runtimes below
        # /home/nvidia, which is group-traversable but not world-traversable.
        # Retain the caller UID for host-owned outputs while granting only the
        # component-image groups needed to execute those runtimes.
        "--group-add", "10000", "--group-add", "1000", "--network", "host",
        "-e", "HOME=/tmp",
    ]
    for name in ("VLM_API_KEY", "LLM_API_KEY", "IMAGE_EDIT_API_KEY"):
        if os.environ.get(name):
            base += ["-e", name]
    augmentation = base + [
        "-v", f"{input_root.resolve()}:/app/data/in:ro",
        "-v", f"{output_root.resolve()}:/app/data/out", "-w", "/app",
        "--entrypoint", "uv",
    ]
    auto_labeling = base + [
        "-v", f"{input_root.resolve()}:/input:ro",
        "-v", f"{output_root.resolve()}:/output", "-w", "/workspace",
        "--entrypoint", "uv",
    ]
    urls = {role: endpoint_url(config, role) for role in ROLES}
    if endpoint_urls is not None:
        if not isinstance(endpoint_urls, dict) or set(endpoint_urls) != set(ROLES):
            raise ValueError("endpoint URL overrides must contain image_edit, vlm, and llm")
        for role, value in endpoint_urls.items():
            normalized = value.rstrip("/") if isinstance(value, str) else ""
            parsed = urllib.parse.urlsplit(normalized)
            if (
                not re.fullmatch(r"https?://[^\s]+", normalized)
                or not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment
            ):
                raise ValueError(f"{role} endpoint URL override is invalid")
            urls[role] = normalized
    if image_edit_url is not None:
        normalized_image_edit_url = image_edit_url.rstrip("/")
        parsed_override = urllib.parse.urlsplit(normalized_image_edit_url)
        if (
            not re.fullmatch(r"https?://[^\s]+", normalized_image_edit_url)
            or not parsed_override.hostname or parsed_override.username or parsed_override.password
            or parsed_override.query or parsed_override.fragment
        ):
            raise ValueError("image-edit URL override is invalid or contains credential material")
        urls["image_edit"] = normalized_image_edit_url
    models = config["models"]
    if action == "preprocess":
        return augmentation + [
            config["images"]["augmentation"], "run", "--frozen", "--no-sync", "python",
            "modules/data_processing/combine_panes.py", "/app/data/in", "/app/data/out/panes",
        ]
    if action == "augment":
        if not source_key or not re.fullmatch(r"[A-Za-z0-9_.-]+", source_key):
            raise ValueError("augment requires a filesystem-safe source key")
        maximum = config["generation"]["verification_max_attempts"]
        if not isinstance(attempt, int) or not 1 <= attempt <= maximum:
            raise ValueError(f"attempt must be within the approved [1, {maximum}] bound")
        out = f"/app/data/out/augmentation/{source_key}/attempt_{attempt}"
        argv = augmentation + [
            config["images"]["augmentation"], "run", "--frozen", "--no-sync", "modules/cli.py",
            "--config", "configs/config_image_edit_verification.yaml",
            f"data.0.inputs.rgb=/app/data/out/panes/{source_key}.jpg",
            f"data.0.output.video={out}/output.jpg",
            f"data.0.output.caption={out}/output.txt",
            f"data.0.output.metadata={out}/output_metadata.json",
            f"pipeline.retry=0",
            f"pipeline.request_timeout={config['generation'].get('image_edit_request_timeout_s', 600)}",
            f"endpoints.vlm.url={urls['vlm']}", f"endpoints.vlm.model={models['vlm']['id']}",
            f"endpoints.llm.url={urls['llm']}", f"endpoints.llm.model={models['llm']['id']}",
            f"endpoints.image_edit.url={urls['image_edit']}",
            f"endpoints.image_edit.model={models['image_edit']['id']}",
        ]
        for attribute, value in sorted((target_attributes or {}).items()):
            variable = attribute.strip().replace(" ", "_")
            if variable not in {item.replace(" ", "_") for item in VECTOR_ATTRIBUTES if item != "viewpoint"}:
                raise ValueError(f"unsupported image-edit target attribute: {attribute}")
            if not str(value).strip() or any(char in str(value) for char in "[]\n\r"):
                raise ValueError(f"unsafe image-edit target value for {attribute}")
            argv.append(f"captioning.llm.variables.{variable}=[{value}]")
        return argv
    if action == "split":
        return augmentation + [
            config["images"]["augmentation"], "run", "--frozen", "--no-sync", "python",
            "modules/data_processing/create_PAS_augmented_dataset.py",
            "--base-dir", "/app/data/out/panes", "--augmented-folders", "/app/data/out/accepted",
            "--output-dir", "/app/data/out/augmented_dataset", "--output-json", "augmented_data.json",
        ]
    if action == "label":
        if not source_key or not re.fullmatch(r"[A-Za-z0-9_.-]+", source_key):
            raise ValueError("label requires a filesystem-safe source key")
        return auto_labeling + [
            config["images"]["auto_labeling"], "run", "--frozen", "--no-sync", "python", "modules/cli.py",
            "--config", "configs/pipeline_example.yaml",
            "super_resolution.enabled=false", "detection_and_tracking.enabled=false",
            "vlm_json.enabled=false", "mcq_generation.enabled=true",
            "mcq_generation.mode=question-driven-vlm-llm",
            "mcq_generation.window_metadata_extraction.single_window=true",
            "mcq_generation.window_metadata_extraction.vlm_verify_enabled=false",
            "mcq_generation.window_metadata_extraction.question_bank_file=/workspace/cookbooks/person_attributes/question_bank.json",
            "mcq_generation.window_metadata_extraction.qd_vlm_scene_prompt_template_file=/workspace/cookbooks/person_attributes/prompts/mcq/question_driven_vlm_llm/vlm_scene_prompt_template.md",
            f"data.0.inputs.video_path=/input/{source_key}.jpg",
            f"data.0.output.out_dir=/output/labels/{source_key}",
            f"endpoints.vlm.url={urls['vlm']}", f"endpoints.vlm.model={models['vlm']['id']}",
            f"endpoints.llm.url={urls['llm']}", f"endpoints.llm.model={models['llm']['id']}",
        ]
    raise ValueError(f"unknown component action: {action}")


def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def validate_gpu_inventory(
    config: dict[str, Any], inventory: list[dict[str, Any]], active_roles: set[str] | None = None,
    selected_roles: Iterable[str] | None = None,
) -> None:
    validate_config(config)
    if config["endpoints"]["ownership"] == "external":
        return
    available = {int(item["index"]): item for item in inventory}
    active_roles = active_roles or set()
    selected = tuple(selected_roles) if selected_roles is not None else ROLES
    if not selected or len(set(selected)) != len(selected) or any(role not in ROLES for role in selected):
        raise ValueError("selected GPU inventory roles are invalid")
    claimed: dict[int, int] = {}
    inactive_claimed: dict[int, int] = {}
    for role in selected:
        model = config["models"][role]
        gpu_ids = parse_gpu_ids(config["endpoints"]["gpu_ids"][role], role)
        # Image editing is replicated as one TP=1 service per selected GPU, so
        # every device must independently satisfy the model's material VRAM
        # requirement. Text roles remain one tensor-parallel service.
        per_gpu = int(model["min_vram_mib"]) if role == "image_edit" else (
            (int(model["min_vram_mib"]) + len(gpu_ids) - 1) // len(gpu_ids)
        )
        for gpu_id in gpu_ids:
            if gpu_id not in available:
                raise ValueError(f"{role} requests missing GPU {gpu_id}")
            claimed[gpu_id] = claimed.get(gpu_id, 0) + per_gpu
            if role not in active_roles:
                inactive_claimed[gpu_id] = inactive_claimed.get(gpu_id, 0) + per_gpu
    for gpu_id, required in claimed.items():
        item = available[gpu_id]
        total = int(item.get("memory_total_mib", item["memory_free_mib"]))
        free = int(item["memory_free_mib"])
        if total < required:
            raise ValueError(f"GPU {gpu_id} has {total} MiB total; endpoint allocation requires {required} MiB")
        new_required = inactive_claimed.get(gpu_id, 0)
        if free < new_required:
            raise ValueError(f"GPU {gpu_id} has {free} MiB free; new endpoint allocation requires {new_required} MiB")
        capability = float(item["compute_capability"])
        if capability < 8.0:
            raise ValueError(f"GPU {gpu_id} compute capability {capability:g} is unsupported; require >= 8.0")


def _json_request(url: str, method: str = "GET", payload: dict | None = None, timeout: int = 30) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def readiness_probe(
    config: dict[str, Any], role: str, request: Callable[..., Any] = _json_request,
    *, base_url: str | None = None,
) -> dict:
    base = base_url or endpoint_url(config, role)
    model_id = config["models"][role]["id"]
    models = request(base + "/models", timeout=config["endpoints"]["request_timeout_s"])
    ids = {item.get("id") for item in models.get("data", []) if isinstance(item, dict)}
    if model_id not in ids:
        raise ValueError(f"{role} endpoint does not serve approved model {model_id!r}; reported {sorted(ids)}")
    if role in {"vlm", "llm"}:
        response = request(
            base + "/chat/completions", method="POST", timeout=config["endpoints"]["request_timeout_s"],
            payload={"model": model_id, "messages": [{"role": "user", "content": "Reply with READY."}], "max_tokens": 8, "temperature": 0},
        )
        if not response.get("choices"):
            raise ValueError(f"{role} minimal inference returned no choices")
    return {"role": role, "model": model_id, "base_url": base, "models_ok": True, "inference_ok": role != "image_edit"}


def wait_until_ready(
    probe: Callable[[], Any], timeout_s: int, interval_s: float, sleep: Callable[[float], None] = time.sleep,
) -> Any:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while True:
        try:
            return probe()
        except (OSError, ValueError, KeyError, urllib.error.URLError) as exc:
            last_error = exc
        if time.monotonic() >= deadline:
            raise TimeoutError(f"endpoint readiness deadline exceeded: {last_error}")
        sleep(min(interval_s, max(0.0, deadline - time.monotonic())))


def verification_passed(metadata: dict[str, Any]) -> bool:
    block = metadata.get("attribute_verification")
    return isinstance(block, dict) and block.get("passed") is True


def residual_attribute_assignments(
    weak_vectors: Iterable[Any], mined_vectors: Iterable[Any], vocab_payload: dict[str, Any],
    count: int, scale_factor: float,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Deterministically sample the positive weak-minus-mined joint distribution."""
    if count < 1 or not 0 < scale_factor <= 1:
        raise ValueError("residual assignment count must be positive and scale_factor in (0, 1]")
    attributes = vocab_payload.get("attributes")
    id_to_value = vocab_payload.get("id_to_value")
    if not isinstance(attributes, list) or not isinstance(id_to_value, dict):
        raise ValueError("attribute vocabulary must contain attributes and id_to_value")

    def vector(value: Any) -> tuple[int, ...]:
        if hasattr(value, "as_py"):
            value = value.as_py()
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, (list, tuple)) or len(value) < len(attributes):
            raise ValueError("attribute vector is shorter than the vocabulary")
        return tuple(int(item) for item in value[: len(attributes)])

    weak = Counter(vector(value) for value in weak_vectors)
    mined = Counter(vector(value) for value in mined_vectors)
    if not weak:
        raise ValueError("weak gap evidence contains no attribute vectors")
    desired_total = max(len(mined), 1) * (1.0 + scale_factor)
    residual: Counter[tuple[int, ...]] = Counter()
    weak_total = sum(weak.values())
    for item, occurrences in sorted(weak.items()):
        desired = max(1, math.ceil(desired_total * occurrences / weak_total))
        remaining = max(0, desired - mined.get(item, 0))
        if remaining:
            residual[item] = remaining
    if not residual:
        # The weak set is fully covered at the requested scale. Keep the most
        # frequent weak combination so the stage remains finite and useful.
        best = sorted(weak.items(), key=lambda item: (-item[1], item[0]))[0][0]
        residual[best] = 1
    schedule = [item for item, occurrences in sorted(residual.items()) for _ in range(occurrences)]
    assignments: list[dict[str, str]] = []
    for index in range(count):
        item = schedule[index % len(schedule)]
        assignment: dict[str, str] = {}
        for attr, value_id in zip(attributes, item):
            normalized_attr = str(attr).strip().replace("_", " ")
            if normalized_attr not in EDITABLE_ATTRIBUTES:
                continue
            values = id_to_value.get(attr, id_to_value.get(normalized_attr))
            if not isinstance(values, list) or value_id < 0 or value_id >= len(values):
                continue
            value = str(values[value_id]).strip()
            if value and value not in {"__missing__", "<missing>", "not visible"}:
                assignment[normalized_attr] = value
        assignments.append(assignment)
    evidence = {
        "weak_rows": sum(weak.values()), "mined_rows": sum(mined.values()),
        "residual_combinations": len(residual), "schedule_size": len(schedule),
        "scale_factor": scale_factor,
    }
    return assignments, evidence


def accepted_augmentations(root: pathlib.Path, max_attempts: int) -> tuple[list[dict], list[dict]]:
    accepted: list[dict] = []
    rejected: list[dict] = []
    for metadata_path in sorted(root.glob("*/attempt_*/output_metadata.json")):
        match = re.fullmatch(r"attempt_([1-9][0-9]*)", metadata_path.parent.name)
        if not match:
            continue
        attempt = int(match.group(1))
        if attempt > max_attempts:
            raise ValueError(f"augmentation attempt exceeds approved bound: {metadata_path}")
        payload = json.loads(metadata_path.read_text())
        image_path = metadata_path.parent / "output.jpg"
        record = {
            "source_key": metadata_path.parent.parent.name,
            "attempt": attempt,
            "image": str(image_path.resolve()),
            "metadata": str(metadata_path.resolve()),
            "metadata_sha256": sha256(metadata_path),
        }
        if verification_passed(payload) and image_path.is_file() and image_path.stat().st_size:
            accepted.append(record)
        else:
            rejected.append(record)
    by_source: dict[str, list[dict]] = {}
    for item in accepted:
        by_source.setdefault(item["source_key"], []).append(item)
    duplicate = [key for key, rows in by_source.items() if len(rows) > 1]
    if duplicate:
        raise ValueError(f"multiple accepted attempts for source(s): {duplicate[:3]}")
    return accepted, rejected


def _normalize_text(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).lower()).split())


def _load_vocab(path: pathlib.Path) -> tuple[dict[str, dict[str, int]], dict[str, dict[int, str]]]:
    payload = json.loads(path.read_text())
    if payload.get("attributes") != list(VECTOR_ATTRIBUTES):
        raise ValueError("attribute vocabulary does not use the canonical IAA vector order")
    value_to_id = payload.get("value_to_id")
    if not isinstance(value_to_id, dict):
        raise ValueError("attribute vocabulary lacks value_to_id")
    forward: dict[str, dict[str, int]] = {}
    reverse: dict[str, dict[int, str]] = {}
    for attr in VECTOR_ATTRIBUTES:
        mapping = value_to_id.get(attr)
        if not isinstance(mapping, dict):
            raise ValueError(f"attribute vocabulary lacks {attr!r}")
        forward[attr] = {_normalize_text(label): int(idx) for label, idx in mapping.items()}
        reverse[attr] = {int(idx): _normalize_text(label) for label, idx in mapping.items()}
    return forward, reverse


def _queries(payload: Any, source: pathlib.Path) -> dict[str, list[str]]:
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        items = payload["items"]
        if len(items) != 9 or any(not isinstance(item, dict) for item in items):
            raise ValueError(f"{source} must contain exactly nine open-QA items")
        answers = [str(item.get("answer", "")).strip() for item in items]
        if any(not answer for answer in answers):
            raise ValueError(f"{source} contains an empty open-QA answer")
        return {
            level: answers[index * 3 : index * 3 + 3]
            for index, level in enumerate(QUERY_LEVELS)
        }
    if isinstance(payload, dict) and isinstance(payload.get("queries"), dict):
        payload = payload["queries"]
    if isinstance(payload, dict) and isinstance(payload.get("open_qa"), dict):
        payload = payload["open_qa"]
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a query-level object")
    result: dict[str, list[str]] = {}
    for level in QUERY_LEVELS:
        rows = payload.get(level)
        if rows is None:
            rows = [value for key, value in sorted(payload.items()) if str(key).startswith(level + "_")]
        if not isinstance(rows, list):
            raise ValueError(f"{source} lacks {level} captions")
        values = [str(item.get("answer", item.get("caption", "")) if isinstance(item, dict) else item).strip() for item in rows]
        values = [value for value in values if value]
        if len(values) != 3:
            raise ValueError(f"{source} must contain exactly three non-empty {level} captions")
        result[level] = values
    return result


def normalize_generated_pairs(
    accepted_manifest: pathlib.Path,
    labels_root: pathlib.Path,
    output_root: pathlib.Path,
    vocab_path: pathlib.Path,
    eval_names: set[str],
    caption_policy: str = "all",
) -> pathlib.Path:
    """Normalize accepted images and open-QA captions into the mining contract."""
    if caption_policy not in {"all", *QUERY_LEVELS}:
        raise ValueError("caption_policy must be all, easy, medium, or hard")
    manifest = json.loads(accepted_manifest.read_text())
    records = manifest.get("accepted")
    if not isinstance(records, list) or not records:
        raise ValueError("accepted manifest must contain at least one record")
    forward, reverse = _load_vocab(vocab_path)
    final_root = output_root
    final_manifest = final_root / "sdg_manifest.json"
    if output_root.exists() and any(output_root.iterdir()):
        if final_manifest.is_file():
            existing = json.loads(final_manifest.read_text())
            for field in ("image_list_file", "pairs_file"):
                if not pathlib.Path(existing.get(field, "")).is_file():
                    raise ValueError(f"incomplete normalized output: missing {field}")
            return final_manifest
        raise ValueError(f"partial normalized output requires recovery, not overwrite: {output_root}")
    build_root = final_root.with_name(final_root.name + ".building")
    if build_root.exists():
        shutil.rmtree(build_root)
    image_root = build_root / "images"
    caption_root = build_root / "captions"
    image_root.mkdir(parents=True)
    caption_root.mkdir(parents=True)
    levels = QUERY_LEVELS if caption_policy == "all" else (caption_policy,)
    pairs: list[dict] = []
    names: list[str] = []
    provenance: list[dict] = []
    for record in sorted(records, key=lambda item: (item["source_key"], item["attempt"])):
        source_image = pathlib.Path(record["image"]).resolve()
        if source_image.name in eval_names:
            raise ValueError(f"evaluation image entered generated training data: {source_image.name}")
        metadata_path = pathlib.Path(record["metadata"]).resolve()
        metadata = json.loads(metadata_path.read_text())
        if not verification_passed(metadata):
            raise ValueError(f"rejected image is present in accepted manifest: {source_image}")
        source_vector = record.get("source_attribute_values")
        if (
            not isinstance(source_vector, list)
            or len(source_vector) != len(VECTOR_ATTRIBUTES)
            or any(not isinstance(value, int) or isinstance(value, bool) for value in source_vector)
        ):
            raise ValueError(
                f"accepted record lacks a canonical source attribute vector: {record['source_key']}"
            )
        vector = list(source_vector)
        for index, attr in enumerate(VECTOR_ATTRIBUTES):
            if vector[index] not in reverse[attr]:
                raise ValueError(f"source attribute ID {vector[index]} is not in vocabulary for {attr}")

        raw_selections = metadata.get("selections")
        if not isinstance(raw_selections, dict) or not raw_selections:
            raise ValueError("accepted metadata lacks generated selections")
        selections: dict[str, Any] = {}
        for key, value in raw_selections.items():
            canonical = " ".join(str(key).strip().replace("_", " ").split())
            if canonical in selections and selections[canonical] != value:
                raise ValueError(f"accepted metadata has conflicting selections for {canonical}")
            selections[canonical] = value

        verification = metadata.get("attribute_verification", {}).get("details", {})
        results = verification.get("results")
        if not isinstance(results, list) or not results:
            raise ValueError("accepted metadata lacks per-attribute verification results")
        verified: set[str] = set()
        for result in results:
            if not isinstance(result, dict) or result.get("passed") is not True:
                raise ValueError("accepted metadata contains an unverified generated selection")
            attr = " ".join(str(result.get("variable", "")).strip().replace("_", " ").split())
            if attr not in VECTOR_ATTRIBUTES:
                raise ValueError(f"verification result names an unknown vector attribute: {attr!r}")
            if attr not in selections or _normalize_text(selections[attr]) != _normalize_text(result.get("value")):
                raise ValueError(f"verification result does not bind selection {attr!r}")
            verified.add(attr)
        raw_targets = record.get("target_attributes")
        if not isinstance(raw_targets, dict) or not raw_targets:
            raise ValueError(f"accepted record lacks target attributes: {record['source_key']}")
        targets = {
            " ".join(str(key).strip().replace("_", " ").split()): value
            for key, value in raw_targets.items()
        }
        unknown_targets = sorted(set(targets) - set(VECTOR_ATTRIBUTES))
        if unknown_targets:
            raise ValueError(f"accepted record names unknown target attributes: {unknown_targets}")
        unsupported_targets = sorted(set(targets) - EDITABLE_ATTRIBUTES)
        if unsupported_targets:
            raise ValueError(f"accepted record targets unsupported edit attributes: {unsupported_targets}")
        missing_verification = sorted(set(targets) - verified)
        if missing_verification:
            raise ValueError(f"target attributes lack verification evidence: {missing_verification}")
        for attr, value in targets.items():
            if attr not in selections or _normalize_text(selections[attr]) != _normalize_text(value):
                raise ValueError(f"generated selection does not match target attribute {attr!r}")
        for attr in sorted(verified):
            value = _normalize_text(selections[attr])
            if value not in forward[attr]:
                raise ValueError(f"value {selections[attr]!r} is not in vocabulary for {attr}")
            vector[VECTOR_ATTRIBUTES.index(attr)] = forward[attr][value]
        label_path = labels_root / record["source_key"] / "task" / "open_qa.json"
        queries = _queries(json.loads(label_path.read_text()), label_path)
        stem = f"{record['source_key']}__attempt_{record['attempt']}"
        for level in levels:
            for index, caption in enumerate(queries[level]):
                name = f"{stem}__{level}_{index}.jpg"
                output_image = image_root / name
                output_caption = caption_root / pathlib.Path(name).with_suffix(".txt")
                shutil.copyfile(source_image, output_image)
                output_caption.write_text(caption + "\n", encoding="utf-8")
                text_values = [
                    value if idx < TEXT_WIDTH[level] and reverse[attr].get(value) not in {"missing", "not visible"} else -1
                    for idx, (attr, value) in enumerate(zip(VECTOR_ATTRIBUTES, vector))
                ]
                names.append(name)
                pairs.append({
                    "unique_name": name,
                    "caption": caption,
                    "image_path": f"images/{name}",
                    "dataset": "IAA_SDG",
                    "query_type": level,
                    "person_id": record["source_key"],
                    "person_key": record["source_key"],
                    "source_split": "train",
                    "source_collection": "IAA_SDG",
                    "is_augmented": True,
                    "generation_attempt": record["attempt"],
                    "verification_metadata_sha256": record["metadata_sha256"],
                    "source_unique_name": record.get("source_unique_name", record["source_key"]),
                    "image_attr_values": vector,
                    "text_attr_values": text_values,
                    "verified_generated_attributes": sorted(verified),
                })
        provenance.append({
            "source_key": record["source_key"], "source_unique_name": record.get("source_unique_name", record["source_key"]),
            "attempt": record["attempt"], "metadata_sha256": record["metadata_sha256"],
        })
    image_list = build_root / "sdg_image_list.txt"
    pairs_path = build_root / "sdg_pairs.json"
    image_list.write_text("\n".join(names) + "\n", encoding="utf-8")
    atomic_json(pairs_path, pairs)
    shutil.copyfile(vocab_path, build_root / "attribute_vocab.json")
    build_manifest = build_root / "sdg_manifest.json"
    atomic_json(build_manifest, {
        "schema_version": "1",
        "dataset_format_version": 3,
        "caption_policy": caption_policy,
        "image_dir": str((final_root / "images").resolve()),
        "caption_dir": str((final_root / "captions").resolve()),
        "image_list_file": str((final_root / "sdg_image_list.txt").resolve()),
        "pairs_file": str((final_root / "sdg_pairs.json").resolve()),
        "attribute_vocab_file": str((final_root / "attribute_vocab.json").resolve()),
        "num_source_images": len(records),
        "num_pairs": len(pairs),
        "accepted_provenance": provenance,
        "rejected_samples_included": 0,
    })
    if final_root.exists():
        try:
            final_root.rmdir()
        except OSError:
            raise ValueError(f"normalized output root became non-empty during commit: {final_root}")
    os.replace(build_root, final_root)
    return final_manifest
