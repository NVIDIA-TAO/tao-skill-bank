#!/usr/bin/env python3
"""Pure validation and provenance primitives for Cosmos TAO workflows.

This module deliberately has no machine- or user-specific defaults.  Every
filesystem location in its output originates in a runtime request.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


URI_RE = re.compile(r"^(?:hf_model://|https?://|s3://|ngc://|hf://)")
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
AETC_TASKS = {
    "bcq",
    "mcq",
    "bcq_openended",
    "mcq_openended",
    "open_qa",
    "scene_description",
    "video_summarization",
    "temporal_localization",
    "temporal_description",
    "causal_linkage",
}
ACCURACY_TASKS = {"bcq", "mcq"}
MEDIA_FIELDS = ("video", "video_id", "image", "image_id", "media", "media_path")


class WorkflowError(ValueError):
    """A deterministic, actionable request or parity failure."""


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def path_identity(value: str, *, required: bool = True) -> dict[str, Any]:
    """Preserve a supplied path and add non-destructive normalization details."""
    if not value:
        if required:
            raise WorkflowError("required runtime path is missing")
        return {"original": "", "expanded": "", "resolved": None, "exists": False}
    expanded = str(Path(value).expanduser())
    path = Path(expanded)
    exists = path.exists()
    return {
        "original": value,
        "expanded": expanded,
        "resolved": str(path.resolve()) if exists else None,
        "exists": exists,
        "kind": "directory" if path.is_dir() else "file" if path.is_file() else "missing",
    }


def is_model_uri(value: str) -> bool:
    return bool(URI_RE.match(value) or MODEL_ID_RE.fullmatch(value))


def _file_inventory(root: Path, names: Iterable[str]) -> list[dict[str, Any]]:
    inventory = []
    for name in names:
        path = root / name
        if path.is_file():
            inventory.append({"path": name, "size": path.stat().st_size, "sha256": sha256_file(path)})
    return inventory


def inspect_model(value: str, revision: str = "", prepared: str = "") -> dict[str, Any]:
    if not value:
        raise WorkflowError("base_model_path_or_uri is required for every Cosmos training request")
    supplied = path_identity(value)
    result: dict[str, Any] = {
        "supplied": supplied,
        "revision": revision or None,
        "prepared_checkpoint": path_identity(prepared, required=False),
    }
    if supplied["exists"]:
        root = Path(supplied["resolved"])
        if not root.is_dir():
            raise WorkflowError(f"base model must be a directory or supported URI: {value}")
        config_path = root / "config.json"
        if not config_path.is_file():
            raise WorkflowError(f"base model is missing config.json: {value}")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise WorkflowError(f"base model config.json is invalid: {value}: {exc}") from exc
        weight_files = sorted(p.name for p in root.glob("*.safetensors"))
        index = root / "model.safetensors.index.json"
        if not weight_files and not index.is_file():
            raise WorkflowError(f"base model contains no safetensors weights or index: {value}")
        indexed_weight_files: list[str] = []
        if index.is_file():
            try:
                index_payload = json.loads(index.read_text(encoding="utf-8"))
                weight_map = index_payload.get("weight_map", {})
            except json.JSONDecodeError as exc:
                raise WorkflowError(f"model safetensors index is invalid: {index}: {exc}") from exc
            if not isinstance(weight_map, dict) or not weight_map:
                raise WorkflowError(f"model safetensors index has no weight_map: {index}")
            indexed_weight_files = sorted(set(weight_map.values()))
            for relative in indexed_weight_files:
                if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
                    raise WorkflowError(f"model safetensors index contains an unsafe weight path: {relative!r}")
                if not (root / relative).is_file():
                    raise WorkflowError(f"model safetensors index references a missing weight file: {relative}")
        important = [
            "config.json", "generation_config.json", "model.safetensors.index.json",
            "tokenizer.json", "tokenizer_config.json", "processor_config.json",
            "preprocessor_config.json", "chat_template.json",
        ] + weight_files + indexed_weight_files
        inventory = _file_inventory(root, important)
        result.update(
            {
                "source_type": "local",
                "format": config.get("model_type", "unknown"),
                "config": {"model_type": config.get("model_type"), "architectures": config.get("architectures")},
                "files": inventory,
                "fingerprint": stable_hash(inventory),
            }
        )
    elif is_model_uri(value):
        if not revision:
            raise WorkflowError(
                "base_model_revision is required for a model URI/identifier so a clean run is immutable"
            )
        result.update(
            {
                "source_type": "uri",
                "format": "unresolved",
                "fingerprint": stable_hash({"uri": value, "revision": revision}),
            }
        )
    else:
        raise WorkflowError(f"base model path is inaccessible and is not a supported URI: {value}")

    if prepared:
        prepared_id = result["prepared_checkpoint"]
        if not prepared_id["exists"] or prepared_id["kind"] != "directory":
            raise WorkflowError(f"prepared checkpoint is inaccessible: {prepared}")
        prepared_result = inspect_model(prepared)
        result["prepared_checkpoint"].update(
            {
                "format": prepared_result["format"],
                "fingerprint": prepared_result["fingerprint"],
                "files": prepared_result["files"],
            }
        )
    return result


def load_annotation(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"cannot read annotation {path}: {exc}") from exc
    if isinstance(payload, list):
        records, metadata = payload, {}
    elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
        records, metadata = payload["items"], payload.get("metadata", {}) or {}
    else:
        raise WorkflowError(f"annotation must be a JSON array or an object containing an items array: {path}")
    if not records:
        raise WorkflowError(f"annotation contains zero records: {path}")
    if not all(isinstance(record, dict) for record in records):
        raise WorkflowError(f"annotation contains a non-object record: {path}")
    return records, metadata if isinstance(metadata, dict) else {}


def _record_media(record: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for field in MEDIA_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value:
            values.append(value)
        elif isinstance(value, list):
            values.extend(item for item in value if isinstance(item, str) and item)
    return values


def _record_key(record: Mapping[str, Any]) -> str:
    identity = {
        "id": record.get("id") or record.get("sample_id") or record.get("video_id"),
        "media": _record_media(record),
        "question": record.get("question"),
        "conversations": record.get("conversations"),
    }
    return stable_hash(identity)


def _record_task(record: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    value = record.get("task") or record.get("task_type") or metadata.get("task") or ""
    return str(value).strip().casefold().replace("-", "_").replace(" ", "_")


def inspect_dataset(
    *,
    workload: str,
    annotations: Sequence[str],
    media_roots: Sequence[str],
    selected_tasks: Sequence[str] = (),
    verify_media_content: bool = True,
) -> dict[str, Any]:
    if not annotations:
        raise WorkflowError("at least one runtime annotation path is required")
    if not media_roots:
        raise WorkflowError("at least one runtime media root is required")
    if len(media_roots) not in {1, len(annotations)}:
        raise WorkflowError("supply one shared media root or one media root per annotation")
    workload = workload.casefold()
    requested_tasks = {item.casefold().replace("-", "_") for item in selected_tasks}
    unknown = requested_tasks - AETC_TASKS
    if unknown:
        raise WorkflowError(f"unsupported AETC task selection: {sorted(unknown)}")

    annotation_ids = [path_identity(value) for value in annotations]
    media_ids = [path_identity(value) for value in media_roots]
    for item in annotation_ids:
        if not item["exists"] or item["kind"] != "file":
            raise WorkflowError(f"annotation path is inaccessible: {item['original']}")
    for item in media_ids:
        if not item["exists"] or item["kind"] != "directory":
            raise WorkflowError(f"media root is inaccessible: {item['original']}")

    record_keys: list[str] = []
    task_counts: dict[str, int] = {}
    manifest_entries: list[dict[str, Any]] = []
    media_entries: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    schema_errors: list[str] = []
    for annotation_index, annotation_id in enumerate(annotation_ids):
        annotation_path = Path(annotation_id["resolved"])
        root_id = media_ids[0 if len(media_ids) == 1 else annotation_index]
        root = Path(root_id["resolved"])
        records, metadata = load_annotation(annotation_path)
        manifest_entries.append(
            {
                "original": annotation_id["original"],
                "resolved": annotation_id["resolved"],
                "sha256": sha256_file(annotation_path),
                "count": len(records),
                "metadata": metadata,
            }
        )
        for index, record in enumerate(records):
            task = _record_task(record, metadata)
            if workload == "wts":
                conversations = record.get("conversations")
                if not isinstance(conversations, list) or len(conversations) < 2:
                    schema_errors.append(f"{annotation_path}:{index}: WTS record needs >=2 conversations")
                if not _record_media(record):
                    schema_errors.append(f"{annotation_path}:{index}: WTS record has no media field")
            elif workload == "aetc":
                if task not in AETC_TASKS:
                    schema_errors.append(f"{annotation_path}:{index}: missing/unsupported AETC task {task!r}")
                if requested_tasks and task not in requested_tasks:
                    continue
                if not _record_media(record):
                    schema_errors.append(f"{annotation_path}:{index}: AETC record has no media field")
            else:
                raise WorkflowError(f"unsupported dataset workload: {workload}")
            record_keys.append(_record_key(record))
            task_counts[task or workload] = task_counts.get(task or workload, 0) + 1
            for relative in _record_media(record):
                candidate = Path(relative)
                media_path = candidate if candidate.is_absolute() else root / candidate
                resolved = str(media_path.resolve()) if media_path.exists() else str(media_path)
                if not media_path.is_file():
                    missing.append({"annotation": str(annotation_path), "record": index, "media": relative})
                    continue
                if resolved not in media_entries:
                    entry = {"path": resolved, "size": media_path.stat().st_size}
                    if verify_media_content:
                        entry["sha256"] = sha256_file(media_path)
                    media_entries[resolved] = entry
    if schema_errors:
        raise WorkflowError("dataset schema validation failed: " + "; ".join(schema_errors[:20]))
    if missing:
        raise WorkflowError("referenced media is missing: " + json.dumps(missing[:20], sort_keys=True))
    if not record_keys:
        raise WorkflowError("task selection produced zero records")
    duplicate_count = len(record_keys) - len(set(record_keys))
    if duplicate_count:
        raise WorkflowError(f"dataset contains {duplicate_count} duplicate logical records")
    media_manifest = sorted(media_entries.values(), key=lambda item: item["path"])
    return {
        "workload": workload,
        "annotations": annotation_ids,
        "media_roots": media_ids,
        "annotation_manifest": manifest_entries,
        "record_count": len(record_keys),
        "record_keys_sha256": stable_hash(sorted(record_keys)),
        "record_key_set": sorted(record_keys),
        "duplicate_records": duplicate_count,
        "missing_media": 0,
        "media_count": len(media_manifest),
        "media_manifest": media_manifest,
        "media_fingerprint": stable_hash(media_manifest),
        "dataset_fingerprint": stable_hash(
            {"records": sorted(record_keys), "media": media_manifest, "tasks": task_counts}
        ),
        "tasks": task_counts,
        "metric_coverage": {
            "accuracy_tasks": sorted(set(task_counts) & ACCURACY_TASKS),
            "excluded_tasks": sorted(set(task_counts) - ACCURACY_TASKS),
            "aggregate": "example_weighted_over_accuracy_defined_tasks",
        },
    }


def assert_no_overlap(train: Mapping[str, Any], validation: Mapping[str, Any]) -> None:
    overlap = set(train["record_key_set"]) & set(validation["record_key_set"])
    if overlap:
        raise WorkflowError(f"train/validation overlap contains {len(overlap)} logical records")


def model_parity(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    same = left.get("fingerprint") == right.get("fingerprint")
    return {"status": "equivalent" if same else "invalid_mismatch", "left": left.get("fingerprint"), "right": right.get("fingerprint")}


def dataset_parity(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    same = left.get("dataset_fingerprint") == right.get("dataset_fingerprint")
    return {"status": "equivalent" if same else "invalid_mismatch", "left": left.get("dataset_fingerprint"), "right": right.get("dataset_fingerprint")}


def optimization_parity(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "training_mode", "epochs", "effective_global_batch", "optimizer", "learning_rate",
        "scheduler", "warmup", "weight_decay", "gradient_clip", "precision", "seed",
        "sequence_length", "frames", "system_prompt", "validation_frequency_epochs",
        "checkpoint_frequency_epochs", "lora",
    )
    differences = {key: {"left": left.get(key), "right": right.get(key)} for key in keys if left.get(key) != right.get(key)}
    return {"status": "equivalent" if not differences else "invalid_mismatch", "differences": differences}


def validate_provenance(
    provenance: Mapping[str, Any], expected_commits: Mapping[str, str],
    expected_trees: Mapping[str, str] | None = None,
) -> None:
    if not provenance:
        raise WorkflowError("image provenance is missing")
    repositories = provenance.get("repositories")
    if not isinstance(repositories, Mapping):
        raise WorkflowError("image provenance has no repository manifest")
    for name, commit in expected_commits.items():
        actual = repositories.get(name, {})
        actual_commit = actual.get("commit") if isinstance(actual, Mapping) else None
        if actual_commit != commit:
            raise WorkflowError(f"image source mismatch for {name}: expected {commit}, found {actual_commit}")
        if expected_trees is not None and actual.get("tree") != expected_trees.get(name):
            raise WorkflowError(
                f"image tree mismatch for {name}: expected {expected_trees.get(name)}, found {actual.get('tree')}"
            )
        if actual.get("dirty"):
            raise WorkflowError(f"image provenance reports dirty source for {name}")


def validate_metadata(metadata: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "experiment_id", "dataset", "training_mode", "backend", "tao_job_id",
        "slurm", "image", "repositories", "config", "paths", "dataset_fingerprints", "model",
        "launch_command", "stdout", "stderr", "results_dir", "checkpoint_dir", "timestamps",
        "scheduler", "child_process", "terminal_tao_status", "metrics", "artifacts",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise WorkflowError(f"SLURM metadata is incomplete; missing: {missing}")
    slurm_required = {
        "job_id", "submission_host", "cluster", "partition", "account", "qos", "reservation",
        "requested_resources", "allocated_resources", "node_list", "master_address", "master_port",
        "requeue", "exclusive", "time_limit", "timeout",
    }
    slurm = metadata.get("slurm")
    if not isinstance(slurm, Mapping) or slurm_required - set(slurm):
        raise WorkflowError(f"SLURM metadata is incomplete; missing slurm fields: {sorted(slurm_required - set(slurm or {}))}")
    if metadata.get("child_process", {}).get("exit_code") not in {None, 0} and metadata.get("terminal_tao_status") == "SUCCESS":
        raise WorkflowError("nonzero child-process exit code cannot have terminal TAO SUCCESS")
    if metadata.get("slurm", {}).get("requeue") and metadata.get("child_process", {}).get("exit_code") not in {None, 0}:
        raise WorkflowError("requeue cannot hide a child-process failure")
    if metadata.get("scheduler", {}).get("state") == "COMPLETED" and metadata.get("child_process", {}).get("exit_code") is None:
        raise WorkflowError("scheduler COMPLETED is invalid without a captured child-process exit code")
    if metadata.get("terminal_tao_status") == "SUCCESS":
        if metadata.get("scheduler", {}).get("state") != "COMPLETED" or metadata.get("child_process", {}).get("exit_code") != 0:
            raise WorkflowError("TAO SUCCESS requires scheduler COMPLETED and child-process exit code zero")


def selected_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Return only reproducibility settings; credentials are never persisted."""
    allow = {
        "PYTHONUNBUFFERED", "PYTHONHASHSEED", "NCCL_DEBUG", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        "PYTORCH_CUDA_ALLOC_CONF", "NVIDIA_DRIVER_CAPABILITIES", "CUDA_FORWARD_COMPAT",
    }
    return {key: environment[key] for key in sorted(allow & set(environment))}
