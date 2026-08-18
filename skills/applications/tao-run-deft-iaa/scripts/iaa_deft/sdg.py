# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic contracts for local IAA generation and normalization."""

from __future__ import annotations

import hashlib
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
TEXT_WIDTH = {"easy": 4, "medium": 6, "hard": 7}
REQUIRED_SELECTIONS = set(VECTOR_ATTRIBUTES) | {"accessories"}


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
        if not isinstance(image, str) or not image.strip() or image.endswith(":latest"):
            raise ValueError(f"images.{name} must be a pinned non-latest image")
    ownership = endpoints.get("ownership")
    if ownership not in {"managed", "external"}:
        raise ValueError("endpoints.ownership must be managed or external")
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
    return payload


def container_name(run_id: str, role: str) -> str:
    if role not in ROLES:
        raise ValueError(f"unknown model role: {role}")
    slug = re.sub(r"[^a-z0-9-]+", "-", run_id.lower()).strip("-")
    if not slug:
        raise ValueError("run ID does not contain a usable character")
    return f"tao-deft-iaa-{slug[:32]}-{role.replace('_', '-')}"


def endpoint_url(config: dict[str, Any], role: str) -> str:
    validate_config(config)
    if config["endpoints"]["ownership"] == "external":
        return config["endpoints"]["external_urls"][role].rstrip("/")
    return f"http://127.0.0.1:{config['models'][role]['port']}/v1"


def build_endpoint_command(
    config: dict[str, Any], role: str, run_id: str, cache_dir: pathlib.Path
) -> list[str]:
    validate_config(config)
    if config["endpoints"]["ownership"] != "managed":
        raise ValueError("external endpoints are validated, never started")
    model = config["models"][role]
    gpu_ids = parse_gpu_ids(config["endpoints"]["gpu_ids"][role], role)
    image_key = "image_edit_serving" if role == "image_edit" else "text_serving"
    argv = [
        "docker", "run", "-d", "--name", container_name(run_id, role),
        "--label", "com.nvidia.tao.workflow=tao-run-deft-iaa",
        "--label", f"com.nvidia.tao.run={run_id}",
        "--label", f"com.nvidia.tao.role={role}",
        "--gpus", "device=" + ",".join(str(item) for item in gpu_ids),
        "-p", f"127.0.0.1:{model['port']}:{model['port']}",
        "-v", f"{cache_dir.resolve()}:/root/.cache/huggingface",
    ]
    if os.environ.get("HF_TOKEN"):
        argv += ["-e", "HF_TOKEN"]
    argv.append(config["images"][image_key])
    if role == "image_edit":
        argv += [
            "vllm", "serve", model["id"], "--omni", "--host", "0.0.0.0",
            "--port", str(model["port"]), "--revision", model["revision"],
            "--served-model-name", model["id"], "--tensor-parallel-size", str(len(gpu_ids)),
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
) -> list[str]:
    """Build one PAIDF component call without shell interpolation."""
    validate_config(config)
    uid_gid = f"{os.getuid()}:{os.getgid()}"
    base = [
        "docker", "run", "--rm", "--user", uid_gid, "--network", "host",
        "-e", "HOME=/tmp",
    ]
    for name in ("VLM_API_KEY", "LLM_API_KEY", "IMAGE_EDIT_API_KEY"):
        if os.environ.get(name):
            base += ["-e", name]
    augmentation = base + [
        "-v", f"{input_root.resolve()}:/app/data/in:ro",
        "-v", f"{output_root.resolve()}:/app/data/out", "-w", "/app",
    ]
    auto_labeling = base + [
        "-v", f"{input_root.resolve()}:/input:ro",
        "-v", f"{output_root.resolve()}:/output", "-w", "/workspace",
    ]
    urls = {role: endpoint_url(config, role) for role in ROLES}
    models = config["models"]
    if action == "preprocess":
        return augmentation + [
            config["images"]["augmentation"], "uv", "run", "--frozen", "--no-sync", "python",
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
            config["images"]["augmentation"], "uv", "run", "--frozen", "--no-sync", "modules/cli.py",
            "--config", "configs/config_image_edit_verification.yaml",
            f"data.0.inputs.rgb=/app/data/out/panes/{source_key}.jpg",
            f"data.0.output.video={out}/output.jpg",
            f"data.0.output.caption={out}/output.txt",
            f"data.0.output.metadata={out}/output_metadata.json",
            f"pipeline.retry=0",
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
            config["images"]["augmentation"], "uv", "run", "--frozen", "--no-sync", "python",
            "modules/data_processing/create_PAS_augmented_dataset.py",
            "--base-dir", "/app/data/out/panes", "--augmented-folders", "/app/data/out/accepted",
            "--output-dir", "/app/data/out/augmented_dataset", "--output-json", "augmented_data.json",
        ]
    if action == "label":
        if not source_key or not re.fullmatch(r"[A-Za-z0-9_.-]+", source_key):
            raise ValueError("label requires a filesystem-safe source key")
        return auto_labeling + [
            config["images"]["auto_labeling"], "uv", "run", "--frozen", "--no-sync", "python", "modules/cli.py",
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
    config: dict[str, Any], inventory: list[dict[str, Any]], active_roles: set[str] | None = None
) -> None:
    validate_config(config)
    if config["endpoints"]["ownership"] == "external":
        return
    available = {int(item["index"]): item for item in inventory}
    active_roles = active_roles or set()
    claimed: dict[int, int] = {}
    inactive_claimed: dict[int, int] = {}
    for role in ROLES:
        model = config["models"][role]
        gpu_ids = parse_gpu_ids(config["endpoints"]["gpu_ids"][role], role)
        per_gpu = (int(model["min_vram_mib"]) + len(gpu_ids) - 1) // len(gpu_ids)
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


def readiness_probe(config: dict[str, Any], role: str, request: Callable[..., Any] = _json_request) -> dict:
    base = endpoint_url(config, role)
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
            if normalized_attr == "viewpoint":
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
        selections = metadata.get("selections")
        if not isinstance(selections, dict) or not REQUIRED_SELECTIONS.issubset(selections):
            missing = sorted(REQUIRED_SELECTIONS - set(selections or {}))
            raise ValueError(f"accepted metadata lacks selections: {missing}")
        vector: list[int] = []
        for attr in VECTOR_ATTRIBUTES:
            value = _normalize_text(selections[attr])
            if value not in forward[attr]:
                raise ValueError(f"value {selections[attr]!r} is not in vocabulary for {attr}")
            vector.append(forward[attr][value])
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
