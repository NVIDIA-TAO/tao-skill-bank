#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build reproducible Cosmos3 TAO plans from runtime-only inputs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tomllib
import yaml
from cosmos_common import (
    WorkflowError,
    assert_no_overlap,
    dataset_parity,
    inspect_dataset,
    inspect_model,
    materialize_dataset,
    model_parity,
    optimization_parity,
    path_identity,
    planned_path_identity,
    selected_environment,
    sha256_file,
    stable_hash,
    validate_metadata,
    validate_provenance,
)

SKILL_DIR = Path(__file__).resolve().parents[1]
REFERENCES = SKILL_DIR / "references"
BACKEND_FILES = {
    "cosmos-framework": REFERENCES / "cosmos-framework-backend.yaml",
    "cosmos-rl": REFERENCES / "cosmos-rl-backend.yaml",
}
ALIASES = {
    "framework": "cosmos-framework", "cosmos_framework": "cosmos-framework",
    "cosmos-framework": "cosmos-framework", "rl": "cosmos-rl",
    "cosmos_rl": "cosmos-rl", "cosmos-rl": "cosmos-rl",
}
SUPPORTED_ACTIONS = {
    "train", "export", "evaluate", "inference", "inference_microservice", "quantize",
}
PLAN_ARTIFACT_SCHEMA_VERSION = 1
_PLAN_ARTIFACT_TRANSIENT_ARGS = {"verb", "format", "plan_artifact", "render_output"}


def resolve_model_name(
    requested: str,
    base_model_path_or_uri: str,
    inspected_model: Mapping[str, Any] | None = None,
) -> str:
    """Resolve Nano versus Edge from explicit input or public checkpoint identity."""
    if requested and requested.casefold() != "auto":
        model_tier(requested)
        return requested
    normalized = base_model_path_or_uri.casefold().replace("_", "-")
    if "cosmos3-edge" in normalized:
        return "nvidia/Cosmos3-Edge"
    if "cosmos3-nano" in normalized:
        return "nvidia/Cosmos3-Nano"
    path = Path(base_model_path_or_uri).expanduser()
    config_path = path / "config.json"
    if config_path.is_file():
        try:
            model_type = str(json.loads(config_path.read_text(encoding="utf-8")).get("model_type", ""))
        except json.JSONDecodeError as exc:
            raise WorkflowError(f"base model config.json is invalid: {config_path}: {exc}") from exc
        if model_type == "cosmos3_edge":
            return "nvidia/Cosmos3-Edge"
        if model_type in {"qwen3_vl", "cosmos3_omni"}:
            return "nvidia/Cosmos3-Nano"
    if inspected_model:
        model_type = str(inspected_model.get("config", {}).get("model_type") or inspected_model.get("format") or "")
        if model_type == "cosmos3_edge":
            return "nvidia/Cosmos3-Edge"
        if model_type in {"qwen3_vl", "cosmos3_omni"}:
            return "nvidia/Cosmos3-Nano"
    raise WorkflowError("model tier is ambiguous; supply Cosmos3-Nano/Edge or a recognizable public checkpoint")


def resolve_model_profile(
    args: argparse.Namespace,
    tier: str,
    backend: str,
    train_dataset: Mapping[str, Any],
    validation_dataset: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve model-aware runtime policy without modifying checkpoint files."""
    defaults = {
        # Cosmos-RL's native video-conversation hook uses 81,920 as the Qwen
        # video pixel budget. Resolve it in the shared frontend so Framework
        # and RL produce the same visual-token geometry instead of allowing
        # Framework to process the full source resolution.
        "nano": {
            "frames": 8,
            "sequence_length": 40960,
            "attention_implementation": "cosmos",
            "video_max_pixels": 81920,
        },
        "edge": {
            "frames": 6,
            "sequence_length": 16000,
            "attention_implementation": "flash_attention_2",
            "video_max_pixels": None,
        },
    }[tier]
    daft_only_values = {
        "fps": args.fps,
        "min_frames": args.min_frames,
        "max_frames": args.max_frames,
        "video_start": args.video_start,
        "video_end": args.video_end,
        "video_resized_height": args.video_resized_height,
        "video_resized_width": args.video_resized_width,
        "video_min_pixels": args.video_min_pixels,
        "video_total_pixels": args.video_total_pixels,
    }
    selected_daft_only = sorted(
        name for name, value in daft_only_values.items() if value is not None
    )
    if backend != "cosmos-rl" and selected_daft_only:
        raise WorkflowError(
            "DAFT vision options apply only to the cosmos-rl backend: "
            f"{selected_daft_only}"
        )
    if args.fps is not None and args.frames:
        raise WorkflowError("fps and frames/nframes are mutually exclusive")
    if args.fps is not None and args.fps <= 0:
        raise WorkflowError("fps must be positive")
    if (args.min_frames is not None or args.max_frames is not None) and args.fps is None:
        raise WorkflowError("min_frames and max_frames require fps sampling")
    if args.min_frames is not None and args.min_frames < 1:
        raise WorkflowError("min_frames must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        raise WorkflowError("max_frames must be positive")
    if (
        args.min_frames is not None
        and args.max_frames is not None
        and args.min_frames > args.max_frames
    ):
        raise WorkflowError("min_frames must not exceed max_frames")
    if (args.video_start is not None and args.video_start < 0) or (
        args.video_end is not None and args.video_end < 0
    ):
        raise WorkflowError("video_start and video_end must be nonnegative")
    if (
        args.video_start is not None
        and args.video_end is not None
        and args.video_start >= args.video_end
    ):
        raise WorkflowError("video_start must be less than video_end")
    if (args.video_resized_height is None) != (args.video_resized_width is None):
        raise WorkflowError("video_resized_height and video_resized_width must be set together")
    for name in ("video_resized_height", "video_resized_width", "video_min_pixels", "video_total_pixels"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise WorkflowError(f"{name} must be positive")
    if (
        args.video_min_pixels is not None
        and args.video_max_pixels
        and args.video_min_pixels > args.video_max_pixels
    ):
        raise WorkflowError("video_min_pixels must not exceed video_max_pixels")

    frames = 0 if args.fps is not None else (args.frames or defaults["frames"])
    capacity_frames = args.max_frames or (768 if args.fps is not None else frames)
    sequence_length = args.sequence_length or defaults["sequence_length"]
    attention = args.attention_implementation if args.attention_implementation != "auto" else defaults["attention_implementation"]
    if (args.fps is None and frames < 1) or sequence_length < 1:
        raise WorkflowError("frames and sequence_length must be positive")
    resolution_profiles = [train_dataset["profile"]["resolution"], validation_dataset["profile"]["resolution"]]
    measured_widths = [item["median_width"] for item in resolution_profiles if item["median_width"]]
    measured_heights = [item["median_height"] for item in resolution_profiles if item["median_height"]]
    frame_width = args.video_resized_width or args.video_frame_width or int(max(measured_widths, default=1280))
    frame_height = args.video_resized_height or args.video_frame_height or int(max(measured_heights, default=720))
    if frame_width < 1 or frame_height < 1:
        raise WorkflowError("video frame width and height must be positive")
    pixels_per_frame = min(frame_width * frame_height, 1280 * 720) if tier == "edge" else frame_width * frame_height
    max_pixels = args.video_max_pixels or (
        capacity_frames * pixels_per_frame
        if tier == "edge"
        else defaults["video_max_pixels"]
    )
    if max_pixels < 0:
        raise WorkflowError("video_max_pixels must be nonnegative")
    profile = {
        "model_tier": tier,
        "source": "user" if any((args.frames, args.fps, args.sequence_length, args.video_max_pixels, args.video_frame_width, args.video_frame_height, *daft_only_values.values())) or args.attention_implementation != "auto" else "dataset_metadata" if measured_widths and measured_heights else "model_safe_default",
        "frames": frames,
        "capacity_frames": capacity_frames,
        "sampling_mode": "fps" if args.fps is not None else "nframes",
        "vision": _vision_config(args, resolved_frames=frames),
        "sequence_length": sequence_length,
        "attention_implementation": attention,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "max_video_pixels": max_pixels or None,
        "checkpoint_mutation": False,
        "dataset_profile_fingerprints": {
            "train": stable_hash(train_dataset["profile"]),
            "validation": stable_hash(validation_dataset["profile"]),
        },
        "selection_basis": ["model_tier", "dataset_resolution_metadata", "record_count", "media_reuse", "explicit_overrides"],
    }
    args.frames = frames
    args.sequence_length = sequence_length
    args.attention_implementation = attention
    args.video_max_pixels = max_pixels
    return profile


def _vision_config(
    args: argparse.Namespace,
    *,
    resolved_frames: int | None = None,
) -> dict[str, int | float]:
    """Return the native DAFT/Qwen video-element options for one plan."""
    vision: dict[str, int | float] = {}
    if args.fps is not None:
        vision["fps"] = args.fps
        for argument, field in (("min_frames", "min_frames"), ("max_frames", "max_frames")):
            value = getattr(args, argument)
            if value is not None:
                vision[field] = value
    else:
        vision["nframes"] = resolved_frames if resolved_frames is not None else args.frames
    for argument, field in (
        ("video_start", "video_start"),
        ("video_end", "video_end"),
        ("video_resized_height", "resized_height"),
        ("video_resized_width", "resized_width"),
        ("video_min_pixels", "min_pixels"),
        ("video_max_pixels", "max_pixels"),
        ("video_total_pixels", "total_pixels"),
    ):
        value = getattr(args, argument)
        if value not in (None, 0):
            vision[field] = value
    return vision


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise WorkflowError(f"expected YAML object: {path}")
    return value


def model_tier(model: str) -> str:
    normalized = model.casefold().replace("_", "-")
    if "nano" in normalized or normalized in {"cosmos3", "cosmos-reason", "cosmos reason 3"}:
        return "nano"
    if "edge" in normalized:
        return "edge"
    raise WorkflowError(f"unsupported Cosmos family: {model!r}")


def select_backend(*, model: str, action: str, backend: str = "auto", workload: str = "training", comparative: bool = False) -> tuple[str, str]:
    action = action.casefold()
    if action not in SUPPORTED_ACTIONS:
        raise WorkflowError(f"unsupported Cosmos action: {action}")
    selected = backend.casefold()
    if comparative and selected == "auto":
        raise WorkflowError("backend selection is required for every comparative run")
    if selected != "auto":
        try:
            selected = ALIASES[selected]
        except KeyError as exc:
            raise WorkflowError("backend must be cosmos-framework, cosmos-rl, or auto") from exc
    tier = model_tier(model)
    if selected == "auto":
        if tier == "edge":
            action_contract = load_yaml(BACKEND_FILES["cosmos-framework"]).get("actions", {}).get(action, {})
            if not action_contract.get("supported"):
                raise WorkflowError(
                    f"Cosmos3-Edge does not support {action}: {action_contract.get('reason', 'unsupported')}"
                )
            return "cosmos-framework", "Cosmos3-Edge uses the Framework-native model and checkpoint action route"
        if action == "export":
            return "cosmos-framework", "Framework DCP export is owned by Cosmos Framework"
        if action != "train" or workload in {"automl", "hpo"}:
            return "cosmos-rl", "the requested action/schema is native to Cosmos-RL"
        return "cosmos-rl", "plain Cosmos3-Nano SFT preserves the Cosmos-RL compatibility default"
    contract = load_yaml(BACKEND_FILES[selected])
    action_contract = contract.get("actions", {}).get(action, {})
    if not action_contract.get("supported"):
        raise WorkflowError(f"{selected} does not support {action}: {action_contract.get('reason', 'unsupported')}")
    if tier == "edge" and selected == "cosmos-rl":
        raise WorkflowError("Cosmos-RL does not support Cosmos3-Edge")
    return selected, "backend explicitly selected by the request"


def _toml_scalar(value: Any) -> str:
    if value is None:
        raise TypeError("TOML does not have a null scalar")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_scalar(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {type(value).__name__}")


def dump_toml(data: Mapping[str, Any]) -> str:
    lines: list[str] = []
    def emit(table: Mapping[str, Any], prefix: tuple[str, ...]) -> None:
        # Canonical ordering is part of the sealed-plan contract. Plan JSON is
        # persisted with sort_keys=True, so insertion-order TOML would change
        # bytes/checksums after a fresh session reload.
        scalars = [
            (key, table[key])
            for key in sorted(table)
            if not isinstance(table[key], Mapping) and table[key] is not None
        ]
        children = [
            (key, table[key])
            for key in sorted(table)
            if isinstance(table[key], Mapping)
        ]
        if prefix:
            if lines and lines[-1]:
                lines.append("")
            lines.append("[" + ".".join(prefix) + "]")
        lines.extend(f"{key} = {_toml_scalar(value)}" for key, value in scalars)
        for key, child in children:
            emit(child, (*prefix, key))
    emit(data, ())
    return "\n".join(lines).rstrip() + "\n"


def _annotation_args(args: argparse.Namespace, split: str) -> tuple[list[str], list[str]]:
    annotations = list(getattr(args, f"{split}_annotation"))
    media = list(getattr(args, f"{split}_media_root"))
    return annotations, media


def _paired_annotation_roots(
    annotations: Sequence[str], media_roots: Sequence[str]
) -> list[tuple[str, str]]:
    if len(media_roots) == 1:
        return [(annotation, media_roots[0]) for annotation in annotations]
    if len(media_roots) != len(annotations):
        raise WorkflowError("supply one shared media root or one media root per annotation")
    return list(zip(annotations, media_roots, strict=True))


def _needs_remote_inspection(args: argparse.Namespace) -> bool:
    if args.platform != "slurm":
        return False
    values = [
        args.base_model_path_or_uri,
        args.prepared_checkpoint_path,
        args.results_dir,
        args.checkpoint_dir,
        args.cache_dir,
        args.sqsh_cache_dir,
        args.sqsh_path,
        args.video_override_map,
        args.video_override_manifest,
        *args.train_annotation,
        *args.train_media_root,
        *args.validation_annotation,
        *args.validation_media_root,
    ]
    return any(value and "://" not in value and not Path(value).expanduser().exists() for value in values)


def _remote_inspection(args: argparse.Namespace) -> dict[str, Any]:
    """Inspect SLURM-frame inputs by streaming the checked-in helper over SSH.

    No source file or startup patch is created on the cluster.  The helper runs
    from stdin and returns only structured identities/fingerprints.
    """
    if not args.slurm_user or not args.slurm_host:
        raise WorkflowError("slurm_user and at least one slurm_host are required for remote input inspection")
    if not args.ssh_key_path:
        raise WorkflowError("ssh_key_path is required for remote input inspection")
    key = Path(args.ssh_key_path).expanduser()
    if not key.is_file():
        raise WorkflowError(f"ssh_key_path is inaccessible: {args.ssh_key_path}")

    remote_args = [
        "python3", "-", "inspect-inputs",
        "--base-model-path-or-uri", args.base_model_path_or_uri,
        "--dataset-family", args.dataset_family,
    ]
    for option, value in (
        ("--base-model-revision", args.base_model_revision),
        ("--prepared-checkpoint-path", args.prepared_checkpoint_path),
    ):
        if value:
            remote_args.extend([option, value])
    for option, values in (
        ("--train-annotation", args.train_annotation),
        ("--train-media-root", args.train_media_root),
        ("--validation-annotation", args.validation_annotation),
        ("--validation-media-root", args.validation_media_root),
        ("--task", args.task),
    ):
        for value in values:
            remote_args.extend([option, value])
    for label, value in (
        ("results_dir", args.results_dir),
        ("checkpoint_dir", args.checkpoint_dir),
        ("cache_dir", args.cache_dir),
        ("sqsh_cache_dir", args.sqsh_cache_dir),
        ("sqsh_path", args.sqsh_path),
    ):
        remote_args.extend(["--runtime-path", f"{label}={value}"])
    if args.fast_media_fingerprint:
        remote_args.append("--fast-media-fingerprint")
    if getattr(args, "fast_model_fingerprint", False):
        remote_args.append("--fast-model-fingerprint")

    source = Path(__file__).with_name("cosmos_common.py").read_text(encoding="utf-8")
    failures: list[str] = []
    for host in args.slurm_host:
        ssh = [
            "ssh", "-o", "BatchMode=yes", "-o", "PasswordAuthentication=no",
            "-o", "PreferredAuthentications=publickey", "-o", "ConnectTimeout=15",
            "-o", "StrictHostKeyChecking=yes", "-i", str(key), "-o", "IdentitiesOnly=yes",
            f"{args.slurm_user}@{host}", shlex.join(remote_args),
        ]
        try:
            result = subprocess.run(
                ssh,
                input=source,
                text=True,
                capture_output=True,
                check=False,
                timeout=3600,
            )
        except subprocess.TimeoutExpired:
            failures.append(f"{host}: remote input inspection timed out")
            continue
        if result.returncode:
            detail = (result.stderr or result.stdout).strip().splitlines()
            failures.append(f"{host}: {detail[-1] if detail else f'exit {result.returncode}'}")
            continue
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            failures.append(f"{host}: invalid remote inspection JSON: {exc}")
            continue
        if not isinstance(payload, dict) or payload.get("frame") != "target_compute":
            failures.append(f"{host}: incomplete remote inspection payload")
            continue
        payload["verified_host"] = host
        return payload
    raise WorkflowError("remote SLURM input inspection failed: " + "; ".join(failures))


def _ssh_command(args: argparse.Namespace, host: str, remote_command: str) -> list[str]:
    key = Path(args.ssh_key_path).expanduser()
    return [
        "ssh", "-o", "BatchMode=yes", "-o", "PasswordAuthentication=no",
        "-o", "PreferredAuthentications=publickey", "-o", "ConnectTimeout=15",
        "-o", "StrictHostKeyChecking=yes", "-i", str(key), "-o", "IdentitiesOnly=yes",
        f"{args.slurm_user}@{host}", remote_command,
    ]


def _remote_helper(
    args: argparse.Namespace,
    helper_args: Sequence[str],
    *,
    host: str,
    timeout: int = 3600,
) -> dict[str, Any]:
    source = Path(__file__).with_name("cosmos_common.py").read_text(encoding="utf-8")
    command = _ssh_command(args, host, shlex.join(["python3", "-", *helper_args]))
    result = subprocess.run(
        command, input=source, text=True, capture_output=True, check=False, timeout=timeout,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise WorkflowError(
            f"remote helper failed on {host}: {detail[-1] if detail else f'exit {result.returncode}'}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise WorkflowError(f"remote helper returned invalid JSON on {host}: {exc}") from exc
    if not isinstance(payload, dict):
        raise WorkflowError(f"remote helper returned a non-object on {host}")
    return payload


def _remote_materialize_dataset(
    args: argparse.Namespace,
    *,
    split: str,
    output_path: str,
    sample_limit: int,
    host: str,
) -> dict[str, Any]:
    annotations, _ = _annotation_args(args, split)
    helper_args = [
        "materialize-dataset", "--dataset-family", args.dataset_family,
        "--output-path", output_path, "--sample-limit", str(sample_limit),
    ]
    for annotation in annotations:
        helper_args.extend(["--annotation", annotation])
    for task in args.task:
        helper_args.extend(["--task", task])
    return _remote_helper(args, helper_args, host=host)


def _remote_write_text(
    args: argparse.Namespace,
    *,
    output_path: str,
    content: str,
    host: str,
) -> str:
    output = Path(output_path)
    script = """set -Eeuo pipefail
umask 077
parent=$1
target=$2
mkdir -p -- "$parent"
tmp=$(mktemp --tmpdir="$parent" ".${target##*/}.XXXXXX")
trap 'rm -f -- "$tmp"' EXIT
cat > "$tmp"
chmod 0640 "$tmp"
mv -f -- "$tmp" "$target"
trap - EXIT
sha256sum "$target"
"""
    remote = shlex.join(["bash", "-c", script, "tao-write", str(output.parent), str(output)])
    result = subprocess.run(
        _ssh_command(args, host, remote), input=content, text=True,
        capture_output=True, check=False, timeout=120,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise WorkflowError(
            f"remote config write failed on {host}: {detail[-1] if detail else f'exit {result.returncode}'}"
        )
    fields = result.stdout.strip().split()
    if not fields or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
        raise WorkflowError(f"remote config checksum is invalid on {host}")
    return fields[0]


def _remote_file_sha256(args: argparse.Namespace, *, path: str, host: str) -> str:
    remote = shlex.join(["sha256sum", "--", path])
    result = subprocess.run(
        _ssh_command(args, host, remote), text=True, capture_output=True,
        check=False, timeout=120,
    )
    if result.returncode:
        raise WorkflowError(f"required generated config is inaccessible on {host}: {path}")
    fields = result.stdout.strip().split()
    if not fields or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
        raise WorkflowError(f"generated config checksum is invalid on {host}: {path}")
    return fields[0]


def _remote_file_exists(args: argparse.Namespace, *, path: str, host: str) -> bool:
    result = subprocess.run(
        _ssh_command(args, host, shlex.join(["test", "-f", path])),
        text=True, capture_output=True, check=False, timeout=120,
    )
    return result.returncode == 0


def _mount_mapping(value: str) -> tuple[Path, Path]:
    parts = value.split(":")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise WorkflowError(f"container mount must be SOURCE:TARGET[:OPTIONS], got {value!r}")
    return Path(parts[0]).expanduser().resolve(), Path(parts[1])


def _containerize(args: argparse.Namespace, value: str) -> str:
    """Translate a host path through the longest explicit container mount."""
    if not value or "://" in value:
        return value
    source_path = Path(value).expanduser().resolve()
    matches: list[tuple[int, Path, Path]] = []
    for mount in args.container_mount:
        source, target = _mount_mapping(mount)
        try:
            relative = source_path.relative_to(source)
        except ValueError:
            continue
        matches.append((len(source.parts), target, relative))
    if matches:
        _, target, relative = max(matches, key=lambda item: item[0])
        return str(target / relative)
    if args.platform == "slurm":
        raise WorkflowError(f"runtime path is not covered by a container mount: {value}")
    return str(source_path)


def _align_container_runtime_paths(args: argparse.Namespace) -> None:
    """Bind generated/runtime paths to their actual explicit container mounts."""
    if args.platform != "slurm":
        return
    mappings = (
        ("container_spec_path", "write_spec", "/specs/train.toml"),
        ("container_results_dir", "results_dir", "/results"),
        ("container_checkpoint_dir", "checkpoint_dir", "/results/checkpoints"),
        ("container_cache_dir", "cache_dir", "/cache"),
    )
    for container_name, host_name, default in mappings:
        host_value = getattr(args, host_name)
        if not host_value:
            continue
        translated = _containerize(args, host_value)
        current = getattr(args, container_name)
        if current == default:
            setattr(args, container_name, translated)
        elif current != translated:
            raise WorkflowError(
                f"{container_name}={current!r} does not match the explicit mount mapping "
                f"for {host_name}={host_value!r}; expected {translated!r}"
            )


def _training_contract(args: argparse.Namespace) -> dict[str, Any]:
    lora: dict[str, Any] | None = None
    if args.training_mode == "peft":
        missing = [name for name, value in (("rank", args.lora_rank), ("alpha", args.lora_alpha)) if not value]
        if missing or not args.lora_target_modules:
            raise WorkflowError("PEFT requires lora rank, alpha, and at least one target module")
        lora = {
            "rank": args.lora_rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
            "target_modules": list(args.lora_target_modules), "bias": args.lora_bias,
            "use_rslora": args.lora_use_rslora, "modules_to_save": list(args.lora_modules_to_save),
            "precision": args.lora_precision,
        }
    elif any((args.lora_rank, args.lora_alpha, args.lora_target_modules, args.lora_modules_to_save)):
        raise WorkflowError("dense SFT must not include an active LoRA configuration")
    return {
        "training_mode": args.training_mode,
        "epochs": 1 if args.run_mode == "smoke" else args.epochs,
        "effective_global_batch": args.effective_global_batch,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "optimizer_epsilon": args.optimizer_epsilon,
        "scheduler": args.scheduler,
        "warmup": args.warmup,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "precision": args.precision,
        "seed": args.seed,
        "sequence_length": args.sequence_length,
        "frames": args.frames,
        "vision": _vision_config(args),
        "system_prompt": args.system_prompt,
        "train_response_mode": (
            "hybrid" if args.dataset_family == "task_aware_video_reasoning" else "answer"
        ),
        "train_sample_multiplier": (
            2 if args.dataset_family == "task_aware_video_reasoning" else 1
        ),
        "validation_frequency_epochs": 1,
        "checkpoint_frequency_epochs": 1,
        "lora": lora,
    }


def _framework_spec(args: argparse.Namespace, train_count: int, val_count: int, contract: Mapping[str, Any]) -> dict[str, Any]:
    world = args.nodes * args.gpus_per_node
    if args.effective_global_batch % world:
        raise WorkflowError("Framework effective global batch must be divisible by total GPUs")
    per_rank_effective_batch = args.effective_global_batch // world
    requested_per_forward_batch = args.framework_per_forward_batch
    if requested_per_forward_batch < 0:
        raise WorkflowError("Framework per-forward batch must be nonnegative")
    per_forward_batch = requested_per_forward_batch or (
        per_rank_effective_batch if model_tier(args.model) == "nano" else 1
    )
    forward_global_batch = world * per_forward_batch
    if args.effective_global_batch % forward_global_batch:
        raise WorkflowError(
            "Framework effective global batch must be divisible by total GPUs times per-forward batch"
        )
    grad_accum = args.effective_global_batch // forward_global_batch
    smoke_train = min(train_count, args.smoke_train_samples) if args.run_mode == "smoke" else train_count
    smoke_val = min(val_count, args.smoke_validation_samples) if args.run_mode == "smoke" else val_count
    exposed_train_samples = smoke_train * int(contract["train_sample_multiplier"])
    steps = math.ceil(exposed_train_samples / args.effective_global_batch)
    val_steps = math.ceil(smoke_val / world)
    epochs = contract["epochs"]
    spec: dict[str, Any] = {
        "job": {"task": "vlm", "experiment": ("tao_task_aware_video_reasoning_edge" if args.dataset_family == "task_aware_video_reasoning" else "tao_video_conversation_edge") if model_tier(args.model) == "edge" else ("tao_task_aware_video_reasoning" if args.dataset_family == "task_aware_video_reasoning" else "tao_video_conversation"), "project": "cosmos3_reasoner", "group": args.dataset_family, "name": args.experiment_id, "wandb_mode": "disabled"},
        "model": {
            "attn_implementation": args.attention_implementation, "precision": args.precision,
            "backbone": {"model_name": "${oc.env:VLM_SAFETENSORS_PATH}", "safetensors_path": "${oc.env:VLM_SAFETENSORS_PATH}"},
            "ema": {"enabled": False, "rate": 0.1, "iteration_shift": 0},
            "parallelism": {"data_parallel_shard_degree": args.gpus_per_node, "data_parallel_replicate_degree": args.nodes, "context_parallel_shard_degree": 1, "cfg_parallel_shard_degree": 1},
            "compile": {"enabled": False, "compile_dynamic": True},
            "activation_checkpointing": {"mode": "full", "save_ops_regex": ["fmha"], "preserve_rng_state": True, "determinism_check": "default"},
        },
        "optimizer": {"betas": [0.9, 0.999], "eps": args.optimizer_epsilon, "fused": True, "lr": args.learning_rate, "weight_decay": args.weight_decay, "keys_to_select": []},
        # The common Cosmos contract expresses warmup in epochs. Framework's
        # native scheduler expresses it in optimizer steps, so translate it
        # using the same steps-per-epoch value that drives max_iter. A real
        # warmup starts at zero, matching Cosmos-RL's native default; with no
        # warmup, start at the requested peak LR immediately.
        "scheduler": {
            "cycle_lengths": [steps * epochs],
            "f_max": [1.0],
            "f_min": [1.0 if args.scheduler == "constant" else 0.0],
            "f_start": [0.0 if args.warmup else 1.0],
            "verbosity_interval": 0,
            "warm_up_steps": [steps * args.warmup],
        },
        "trainer": {
            "distributed_parallelism": "fsdp", "grad_accum_iter": grad_accum, "logging_iter": 1,
            "max_iter": steps * epochs, "num_epochs": epochs, "steps_per_epoch": steps,
            "max_val_iter": val_steps, "run_validation": True, "validation_iter": steps,
            "validation_freq_in_epoch": 1, "run_validation_on_start": False,
            "callbacks": {"compile_tokenizer": {"compile_after_iterations": 3, "enabled": False}, "grad_clip": {"clip_norm": args.gradient_clip, "force_finite": False}, "tao": {"enabled": True, "experiment_name": args.experiment_id, "logging_interval": 1, "validation_heartbeat_interval": 50}},
        },
        "checkpoint": {
            "keys_to_skip_loading": [],
            "load_path": "???",
            "save_iter": steps,
            "save_freq_in_epoch": 1,
            "dcp_async_mode_enabled": bool(args.async_checkpoint),
        },
        "dataloader_train": {
            "max_samples_per_batch": per_forward_batch,
            "max_sequence_length": args.sequence_length,
        },
    }
    if args.training_mode == "peft":
        lora = contract["lora"]
        spec["model"].update({
            "lora_enabled": True, "lora_rank": lora["rank"], "lora_alpha": lora["alpha"],
            "lora_dropout": lora["dropout"], "lora_target_modules": ",".join(lora["target_modules"]),
            "lora_bias": lora["bias"], "lora_use_rslora": lora["use_rslora"],
            "lora_modules_to_save": ",".join(lora["modules_to_save"]), "lora_precision": lora["precision"],
        })
        spec["optimizer"]["keys_to_select"] = ["lora_"] + lora["modules_to_save"]
        spec["checkpoint"]["keys_to_skip_loading"] = ["optimizer", "scheduler"]
    return spec


def _rl_video_runtime(
    args: argparse.Namespace,
    train_data: Mapping[str, Any],
    val_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve one explicit, attestable Cosmos-RL video runtime profile."""
    requested = getattr(args, "rl_video_profile", "auto")
    if requested not in {"auto", "system-pyav", "pynv-device-rgbp"}:
        raise WorkflowError(f"unsupported Cosmos-RL video profile: {requested}")
    if requested == "auto":
        selected = (
            "pynv-device-rgbp"
            if args.dataset_family == "video_conversation"
            else "system-pyav"
        )
        reason = (
            "video_conversation defaults to the source-baked device-RGBP "
            "throughput profile"
            if selected == "pynv-device-rgbp"
            else "task-aware video defaults to the sparse software profile"
        )
    else:
        selected = requested
        reason = "explicit user selection"

    fast = selected == "pynv-device-rgbp"
    workers_arg = getattr(args, "rl_dataloader_num_workers", None)
    prefetch_arg = getattr(args, "rl_dataloader_prefetch_factor", None)
    workers = (1 if fast else 0) if workers_arg is None else workers_arg
    prefetch = (2 if fast else 1) if prefetch_arg is None else prefetch_arg
    if workers < 0:
        raise WorkflowError("rl_dataloader_num_workers must be nonnegative")
    if workers and prefetch <= 0:
        raise WorkflowError(
            "rl_dataloader_prefetch_factor must be positive when workers are enabled"
        )

    unique_media_capacity = max(
        int(train_data["profile"]["unique_media_count"]),
        int(val_data["profile"]["unique_media_count"]),
        1,
    )
    video_cache_override = getattr(args, "rl_video_cache_size", None)
    decoder_cache_override = getattr(args, "rl_video_decoder_cache_size", None)
    batch_threads_override = getattr(args, "rl_sft_batch_threads", 0)
    if video_cache_override is not None and video_cache_override < 0:
        raise WorkflowError(
            "Cosmos-RL video cache size must be nonnegative"
        )
    if decoder_cache_override is not None and decoder_cache_override < 1:
        raise WorkflowError(
            "Cosmos-RL decoder cache size must be positive"
        )
    if batch_threads_override < 0:
        raise WorkflowError(
            "Cosmos-RL video cache sizes and SFT batch threads must be nonnegative"
        )
    if not fast and (
        video_cache_override is not None or decoder_cache_override is not None
    ):
        raise WorkflowError(
            "Cosmos-RL PyNv cache overrides require --rl-video-profile pynv-device-rgbp"
        )
    # The verified repeated-video fast profile keeps one epoch's rank-local media working
    # set resident after each entry is first encountered.  Capacity is derived
    # from inspected data rather than a benchmark/path name, and population
    # remains strictly on demand inside training.
    video_cache_size = (
        unique_media_capacity
        if fast and video_cache_override is None
        else (video_cache_override if fast else 0)
    )
    decoder_cache_size = (
        unique_media_capacity
        if fast and decoder_cache_override is None
        else (decoder_cache_override if fast else 1)
    )
    # The video-conversation fast path preprocesses one 31-sample logical batch
    # with four ordered threads.  The processed-video cache is single-flight,
    # so duplicate media do not trigger duplicate decodes while tokenization
    # and vision preprocessing can overlap.  The sparse software profile keeps
    # its conservative serial default.
    batch_threads = batch_threads_override or (4 if fast else 1)
    if batch_threads < 1:
        raise WorkflowError("resolved Cosmos-RL SFT batch threads must be positive")

    return {
        "requested_profile": requested,
        "selected_profile": selected,
        "selection_reason": reason,
        "video_decoder": "pynvvideocodec" if fast else "torchvision",
        "implementation": (
            "pynv_device_rgbp_dlpack" if fast else "system_pyav_sparse"
        ),
        "frame_transfer": "device_rgbp" if fast else "host_rgb",
        "video_cache_size": video_cache_size,
        "video_cache_scope": "rank_local_processed_fetch_video_memory",
        "video_cache_population": "on_demand_during_training",
        "video_cache_persists_to_disk": False,
        "decoder_cache_size": decoder_cache_size,
        "decoder_cache_scope": "rank_local_pynv_native_sessions" if fast else "none",
        "sft_batch_threads": batch_threads,
        "dataloader_num_workers": workers,
        "dataloader_prefetch_factor": prefetch if workers else None,
        "unique_media_capacity_basis": unique_media_capacity,
        "dataset_prewarm": False,
        "capability_fallback": (
            "tao_system_pyav_sparse" if fast else "not_applicable"
        ),
        "capability_fallback_scope": (
            "nvdec_incompatible_stream_only" if fast else "not_applicable"
        ),
    }


def _framework_video_runtime(
    args: argparse.Namespace,
    train_data: Mapping[str, Any],
    val_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the native Framework on-demand TorchCodec throughput profile."""
    unique_media_capacity = max(
        int(train_data["profile"]["unique_media_count"]),
        int(val_data["profile"]["unique_media_count"]),
        1,
    )
    cache_override = getattr(args, "framework_video_cache_size", None)
    process_threads_override = getattr(args, "framework_sft_process_threads", 0)
    decoder_threads_override = getattr(args, "framework_video_decoder_threads", 0)
    worker_override = getattr(args, "framework_dataloader_num_workers", None)
    prefetch_override = getattr(args, "framework_dataloader_prefetch_factor", None)
    if cache_override is not None and cache_override < 0:
        raise WorkflowError(
            "Framework video cache override must be nonnegative"
        )
    if min(process_threads_override, decoder_threads_override) < 0:
        raise WorkflowError(
            "Framework video cache and thread overrides must be nonnegative"
        )
    workers = 1 if worker_override is None else worker_override
    prefetch = 2 if prefetch_override is None else prefetch_override
    if workers < 0 or prefetch < 1:
        raise WorkflowError(
            "Framework DataLoader workers must be nonnegative and prefetch must be positive"
        )
    return {
        "selected_profile": "torchcodec-cuda-on-demand",
        "selection_reason": (
            "Framework video supervision uses its source-baked CUDA TorchCodec "
            "profile"
        ),
        "video_decoder": "torchcodec",
        "implementation": "torchcodec_cuda_indexed",
        "decoder_device": "cuda",
        "decoder_device_binding": "explicit_local_rank",
        "frame_transfer": "cuda_uint8_to_host_pil",
        "video_cache_size": unique_media_capacity if cache_override is None else cache_override,
        "video_cache_scope": "rank_local_decoded_pil_frames",
        "video_cache_population": "on_demand_during_training",
        "video_cache_persists_to_disk": False,
        "sft_process_threads": process_threads_override or 8,
        "decoder_threads": decoder_threads_override or 1,
        "dataloader_num_workers": workers,
        "dataloader_prefetch_factor": prefetch if workers else None,
        "dataloader_multiprocessing_context": "spawn" if workers else None,
        "dataloader_persistent_workers": bool(workers),
        "unique_media_capacity_basis": unique_media_capacity,
        "dataset_prewarm": False,
        "actual_device_attestation": "first_successful_decode_per_rank",
        "actual_device_requirement": "resolved_device_equals_cuda_local_rank",
    }


def _rl_spec(args: argparse.Namespace, contract: Mapping[str, Any], prepared_model: str, train_annotations: Sequence[str], train_media: Sequence[str], val_annotations: Sequence[str], val_media: Sequence[str], cache_keys: Mapping[str, str], video_runtime: Mapping[str, Any]) -> dict[str, Any]:
    if len(train_media) != 1 or len(val_media) != 1:
        raise WorkflowError("Cosmos-RL requires one explicit shared media root per split when annotations are merged")
    train_manifest = train_annotations[0] if len(train_annotations) == 1 else "__TAO_TRAIN_MERGED_MANIFEST__"
    val_manifest = val_annotations[0] if len(val_annotations) == 1 else "__TAO_VALIDATION_MERGED_MANIFEST__"
    spec = load_yaml(REFERENCES / "spec_template_train.yaml")
    cache_mode = getattr(args, "rl_dataset_cache_mode", "direct")
    if cache_mode not in {"direct", "prewarm"}:
        raise WorkflowError(f"unsupported Cosmos-RL dataset cache mode: {cache_mode}")
    # Direct mode deliberately sends every sample through the training data
    # path on demand. Prewarm remains an explicit opt-in for workloads that
    # benefit from immutable processor-output reuse.
    use_dataset_cache = (
        args.dataset_family == "video_conversation" and cache_mode == "prewarm"
    )
    train_batch_per_replica = getattr(args, "rl_train_batch_per_replica", 0) or args.rl_mini_batch
    if train_batch_per_replica % args.rl_mini_batch:
        raise WorkflowError(
            "rl_train_batch_per_replica must be divisible by rl_mini_batch"
        )
    if args.minimum_lr_factor is not None and not 0.0 <= args.minimum_lr_factor <= 1.0:
        raise WorkflowError("minimum_lr_factor must be between zero and one")
    spec["train"].update({
        "resume": False, "epoch": contract["epochs"], "compile": False,
        # Cosmos-RL's SFT worker interprets this as the per-DP-worker batch,
        # despite the historical field name.  The global batch is therefore
        # this value times dp_shard_size (replicate size is fixed at one).
        "train_batch_per_replica": train_batch_per_replica, "output_dir": args.container_checkpoint_dir,
        "optm_lr": args.learning_rate,
        # Optimizer implementation is a training-mode contract, not a dataset
        # property: full dense SFT uses fused AdamW while PEFT keeps the native
        # foreach implementation.
        "optm_impl": "fused" if args.training_mode == "dense" else "foreach",
        "optm_weight_decay": args.weight_decay,
        "optm_min_lr_factor": (
            args.minimum_lr_factor
            if args.minimum_lr_factor is not None
            else (1.0 if args.scheduler == "constant" else 0.0)
        ),
        "epsilon": args.optimizer_epsilon,
        # Cosmos-RL names a constant schedule "none"; the common parity
        # contract and Framework continue to expose it as "constant".
        "optm_warmup_epochs": args.warmup,
        "optm_decay_type": "none" if args.scheduler == "constant" else args.scheduler,
        "optm_grad_norm_clip": args.gradient_clip, "param_dtype": args.precision,
        # Preserve Cosmos-RL's full reproducibility contract in addition to
        # the DataLoader-specific seed below.
        "seed": args.seed, "deterministic": True,
    })
    spec["train"]["ckpt"].update({"enable_checkpoint": True, "save_freq_in_epoch": 1, "save_mode": "async" if args.async_checkpoint else "sync", "max_keep": args.max_checkpoints})
    dataloader_num_workers = int(video_runtime["dataloader_num_workers"])
    dataloader_prefetch_factor = video_runtime["dataloader_prefetch_factor"]
    validation_freq_steps = getattr(args, "rl_validation_freq_steps", 0)
    if validation_freq_steps < 0:
        raise WorkflowError("rl_validation_freq_steps must be nonnegative")
    spec["train"]["train_policy"].update({
        "type": "sft", "mini_batch": args.rl_mini_batch,
        "dataloader_num_workers": dataloader_num_workers,
        "conversation_column_name": "conversations", "enable_dataset_cache": use_dataset_cache,
        "dataloader_shuffle": True, "dataloader_seed": args.seed,
        # A full epoch must consume the final partial per-rank batch.
        "dataloader_drop_last": False,
    })
    if dataloader_num_workers:
        spec["train"]["train_policy"]["dataloader_prefetch_factor"] = (
            dataloader_prefetch_factor
        )
    else:
        spec["train"]["train_policy"].pop("dataloader_prefetch_factor", None)
    prewarmed_cache = use_dataset_cache and args.run_mode == "full"
    if prewarmed_cache:
        missing_cache_keys = {"train", "validation"} - set(cache_keys)
        if missing_cache_keys:
            raise WorkflowError(
                f"full Cosmos-RL cache keys are missing: {sorted(missing_cache_keys)}"
            )
        spec["train"]["train_policy"].update(
            {
                "dataset_cache_dir": args.container_cache_dir,
                "dataset_cache_fingerprint": cache_keys["train"],
                "validation_dataset_cache_fingerprint": cache_keys["validation"],
                "require_complete_dataset_cache": True,
            }
        )
    else:
        for key in (
            "dataset_cache_dir",
            "dataset_cache_fingerprint",
            "validation_dataset_cache_fingerprint",
            "require_complete_dataset_cache",
        ):
            spec["train"]["train_policy"].pop(key, None)
    spec["validation"].update(
        {
            "enable": True,
            "freq": validation_freq_steps or 20,
            "freq_in_epoch": 1,
            "batch_size": args.validation_batch_size,
            "dataloader_num_workers": dataloader_num_workers,
        }
    )
    if dataloader_num_workers:
        spec["validation"]["dataloader_prefetch_factor"] = (
            dataloader_prefetch_factor
        )
    else:
        spec["validation"].pop("dataloader_prefetch_factor", None)
    spec["validation"].pop("enable_dataset_cache", None)
    spec["policy"].update({
        "model_name_or_path": prepared_model, "model_max_length": args.sequence_length,
        "model_gradient_checkpointing": True,
    })
    spec["policy"]["parallelism"].update({"dp_shard_size": args.nodes * args.gpus_per_node, "dp_replicate_size": 1, "pp_size": 1, "tp_size": 1})
    if args.training_mode == "peft":
        lora = contract["lora"]
        spec["policy"]["lora"] = {
            "r": lora["rank"], "lora_alpha": lora["alpha"],
            "lora_dropout": lora["dropout"],
            "target_modules": lora["target_modules"], "bias": lora["bias"],
            "use_rslora": lora["use_rslora"], "modules_to_save": lora["modules_to_save"],
            "adapter_dtype": lora["precision"],
        }
    else:
        spec["policy"].pop("lora", None)
    spec["logging"].update({"logger": ["console", "tao"], "experiment_name": args.experiment_id, "project_name": "cosmos-rl-tao"})
    spec["custom"].update({
        "train_dataset": {"annotation_path": train_manifest, "media_path": train_media[0], "media_root": train_media[0], "response_mode": "hybrid" if args.dataset_family == "task_aware_video_reasoning" else "answer"},
        "val_dataset": {"annotation_path": val_manifest, "media_path": val_media[0], "media_root": val_media[0], "response_mode": "answer"},
        "vision": {
            **_vision_config(args),
            "video_decoder": video_runtime["video_decoder"],
        },
        "video_decoder": video_runtime["video_decoder"],
        "video_cache_size": video_runtime["video_cache_size"],
        "video_decoder_cache_size": video_runtime["decoder_cache_size"],
        "system_prompt": args.system_prompt,
    })
    if use_dataset_cache:
        spec["custom"]["vision"]["cache_dir"] = args.container_cache_dir
    if args.video_override_map:
        spec["custom"]["video_override_map"] = _containerize(args, args.video_override_map)
    return spec


def _env(args: argparse.Namespace, backend: str, prepared_model: str, train_annotations: Sequence[str], train_media: Sequence[str], val_annotations: Sequence[str], val_media: Sequence[str], rl_video_runtime: Mapping[str, Any] | None = None, framework_video_runtime: Mapping[str, Any] | None = None) -> dict[str, str]:
    tao_job_id = args.tao_job_id or args.experiment_id
    status_path = str(Path(args.container_results_dir) / tao_job_id / "status.json")
    common = {
        "PYTHONUNBUFFERED": "1", "PYTHONHASHSEED": str(args.seed), "NCCL_DEBUG": args.nccl_debug,
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1", "PYTORCH_CUDA_ALLOC_CONF": args.cuda_allocator,
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,video",
        "TAO_DATALOADER_SEED": str(args.seed),
        "TAO_JOB_ID": tao_job_id,
        "TAO_RESULTS_ROOT": args.container_results_dir,
        "TAO_API_JOB_ID": tao_job_id,
        "TAO_API_RESULTS_DIR": args.container_results_dir,
        "TAO_STATUS_FILE": status_path,
    }
    if args.video_override_map:
        common["TAO_VIDEO_OVERRIDE_MAP"] = _containerize(args, args.video_override_map)
    if backend == "cosmos-framework":
        if not framework_video_runtime:
            raise WorkflowError("Cosmos Framework video runtime was not resolved")
        framework_train_media = list(train_media) * len(train_annotations) if len(train_media) == 1 else list(train_media)
        framework_val_media = list(val_media) * len(val_annotations) if len(val_media) == 1 else list(val_media)
        common.update({
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "VLM_SAFETENSORS_PATH": prepared_model,
            # Framework derives DCP paths as
            # $IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>/checkpoints.
            # Keep those artifacts under the caller's checkpoint root; TAO
            # status and logs remain explicitly routed to results above.
            "IMAGINAIRE_OUTPUT_ROOT": args.container_checkpoint_dir,
            "TAO_VIDEO_DATASET_FAMILY": args.dataset_family,
            "TAO_VIDEO_TRAIN_ANNOTATION": train_annotations[0],
            "TAO_VIDEO_TRAIN_ANNOTATIONS": json.dumps(list(train_annotations)),
            "TAO_VIDEO_TRAIN_MEDIA": train_media[0],
            "TAO_VIDEO_TRAIN_MEDIA_ROOTS": json.dumps(framework_train_media),
            "TAO_VIDEO_VAL_ANNOTATION": val_annotations[0],
            "TAO_VIDEO_VAL_ANNOTATIONS": json.dumps(list(val_annotations)),
            "TAO_VIDEO_VAL_MEDIA": val_media[0],
            "TAO_VIDEO_VAL_MEDIA_ROOTS": json.dumps(framework_val_media),
            "TAO_VIDEO_NUM_FRAMES": str(args.frames), "TAO_VIDEO_SYSTEM_PROMPT": args.system_prompt,
            "TAO_VIDEO_CACHE_SIZE": str(framework_video_runtime["video_cache_size"]),
            "TAO_FRAMEWORK_SFT_PROCESS_THREADS": str(framework_video_runtime["sft_process_threads"]),
            "TAO_FRAMEWORK_DATALOADER_NUM_WORKERS": str(framework_video_runtime.get("dataloader_num_workers", 1)),
            "TAO_VIDEO_DECODER_DEVICE": str(framework_video_runtime["decoder_device"]),
            "TAO_VIDEO_DECODER_THREADS": str(framework_video_runtime["decoder_threads"]),
        })
        framework_prefetch = framework_video_runtime.get("dataloader_prefetch_factor", 2)
        if framework_prefetch is not None:
            common["TAO_FRAMEWORK_DATALOADER_PREFETCH_FACTOR"] = str(
                framework_prefetch
            )
        if args.video_max_pixels:
            common["TAO_VIDEO_MAX_PIXELS"] = str(args.video_max_pixels)
        if args.run_mode == "smoke":
            train_limit = args.smoke_train_samples
            if args.dataset_family == "task_aware_video_reasoning":
                train_limit *= 2
            common.update({"TAO_VIDEO_TRAIN_LIMIT": str(train_limit), "TAO_VIDEO_VAL_LIMIT": str(args.smoke_validation_samples)})
    else:
        common["COSMOS_SFT_REQUIRE_VISUAL_GRADIENTS"] = "1"
        if (
            args.dataset_family == "video_conversation"
            and getattr(args, "rl_dataset_cache_mode", "direct") == "prewarm"
        ):
            common["COSMOS_CACHE"] = args.container_cache_dir
        if not rl_video_runtime:
            raise WorkflowError("Cosmos-RL video runtime was not resolved")
        common["FORCE_QWENVL_VIDEO_READER"] = str(
            rl_video_runtime["video_decoder"]
        )
        common["TAO_SFT_BATCH_THREADS"] = str(
            rl_video_runtime["sft_batch_threads"]
        )
        if rl_video_runtime["selected_profile"] == "pynv-device-rgbp":
            common["TAO_PYNV_FRAME_TRANSFER"] = "device_rgbp"
            common["TAO_PYNV_VIDEO_CACHE_SIZE"] = str(
                rl_video_runtime["video_cache_size"]
            )
            common["TAO_PYNV_DECODER_CACHE_SIZE"] = str(
                rl_video_runtime["decoder_cache_size"]
            )
    return common


def _command(args: argparse.Namespace, backend: str) -> str:
    if backend == "cosmos-framework":
        parts = [
            "/workspace/.venv/bin/torchrun", f"--nproc_per_node={args.gpus_per_node}",
            f"--nnodes={args.nodes}", "--node_rank=${SLURM_PROCID:-0}",
            "--master_addr=${MASTER_ADDR:-127.0.0.1}", "--master_port=${MASTER_PORT:-29500}",
            "-m", "cosmos_framework.scripts.train", f"--sft-toml={args.container_spec_path}", "--",
        ]
        return " ".join(parts)
    hook_name = "tao_vl_reason_daft_sft_example.py" if args.dataset_family == "task_aware_video_reasoning" else "tao_sft_example.py"
    hook_assignment = (
        "hook=\"$(/opt/venv/cosmos_rl/bin/python -c "
        "'import cosmos_rl; from pathlib import Path; "
        f"print(Path(cosmos_rl.__file__).parent / \"tools\" / \"custom_hooks\" / \"{hook_name}\")'"
        ")\""
    )
    if args.nodes == 1:
        return "\n".join([
            hook_assignment,
            'test -f "$hook"',
            f"cosmos-rl --config {shlex.quote(args.container_spec_path)} \"$hook\"",
        ])
    return "\n".join([
        hook_assignment, 'test -f "$hook"',
        'export COSMOS_CONTROLLER_HOST="$MASTER_ADDR:18082"', 'controller_pid=""',
        "launcher_dir=\"$(/opt/venv/cosmos_rl/bin/python -c 'import cosmos_rl; from pathlib import Path; print(Path(cosmos_rl.__file__).parent / \"launcher\")')\"",
        'if [[ "${SLURM_PROCID:-0}" == "0" ]]; then',
        f"  bash \"$launcher_dir/launch_controller.sh\" --port 18082 --config {shlex.quote(args.container_spec_path)} --script \"$hook\" &",
        '  controller_pid="$!"', "fi", "sleep 10", "set +e",
        f"bash \"$launcher_dir/launch_replica.sh\" --type policy --ngpus {args.gpus_per_node} --nnodes {args.nodes} --rdzv-endpoint \"$MASTER_ADDR:$MASTER_PORT\" --config {shlex.quote(args.container_spec_path)} --script \"$hook\"",
        'child_rc="$?"', "set -e", '[[ -z "$controller_pid" ]] || kill "$controller_pid" 2>/dev/null || true', 'exit "$child_rc"',
    ])


def _source_commits(args: argparse.Namespace, backend: str) -> dict[str, str]:
    required = {"cosmos-framework": args.cosmos_framework_commit} if backend == "cosmos-framework" else {"cosmos-rl-github": args.cosmos_rl_commit}
    required["cosmos-rl"] = args.tao_integration_commit
    required["nvidia-tao-daft"] = args.daft_commit
    required["tao-core"] = args.tao_core_commit
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise WorkflowError(f"repository commit inputs are required for clean image provenance: {missing}")
    return required


def _image_plan(args: argparse.Namespace, backend: str, commits: Mapping[str, str]) -> dict[str, Any]:
    dockerfile = "Dockerfile"
    integration = path_identity(args.tao_integration_repo)
    native_name = "cosmos-framework" if backend == "cosmos-framework" else "cosmos-rl-github"
    native_repo = path_identity(args.cosmos_framework_repo if backend == "cosmos-framework" else args.cosmos_rl_repo)
    daft_repo = path_identity(args.daft_repo)
    tao_core_repo = path_identity(args.tao_core_repo)
    image = args.image_tag
    if not image:
        raise WorkflowError("image_tag is required; old or historical image tags are never selected implicitly")
    if not args.build_context or not args.build_timestamp:
        raise WorkflowError("build_context and build_timestamp are required image build inputs")
    missing_trees = [
        name for name, value in (
            (native_name, args.native_tree), ("cosmos-rl", args.integration_tree),
            ("nvidia-tao-daft", args.daft_tree), ("tao-core", args.tao_core_tree),
        ) if not value
    ]
    if missing_trees:
        raise WorkflowError(f"repository tree inputs are required for clean image provenance: {missing_trees}")
    if backend == "cosmos-framework":
        if not args.cosmos_framework_base_image:
            raise WorkflowError(
                "cosmos_framework_base_image is required for the unified clean Framework build"
            )
        if not args.cosmos_framework_source_repository or not args.cosmos_framework_source_branch:
            raise WorkflowError(
                "cosmos_framework_source_repository and cosmos_framework_source_branch "
                "are required for the unified clean Framework build"
            )
        build_args = {
            "COSMOS_BACKEND": "cosmos-framework",
            "COSMOS_FRAMEWORK_BASE_IMAGE": args.cosmos_framework_base_image,
            "COSMOS_FRAMEWORK_REPO": args.cosmos_framework_source_repository,
            "COSMOS_FRAMEWORK_BRANCH": args.cosmos_framework_source_branch,
            "EXPECTED_FRAMEWORK_COMMIT": commits[native_name],
            "EXPECTED_FRAMEWORK_TREE": args.native_tree,
            "ACTIONS_COMMIT": commits["cosmos-rl"], "ACTIONS_TREE": args.integration_tree,
            "DAFT_COMMIT": commits["nvidia-tao-daft"], "DAFT_TREE": args.daft_tree,
            "TAO_CORE_COMMIT": commits["tao-core"], "TAO_CORE_TREE": args.tao_core_tree,
            "SOURCE_DIRTY": "0",
            "BUILD_TIMESTAMP": args.build_timestamp,
        }
        commands = []
    else:
        if not args.cosmos_rl_base_image:
            raise WorkflowError("cosmos_rl_base_image is required for the clean Cosmos-RL build")
        if not args.cosmos_rl_source_repository or not args.cosmos_rl_source_branch:
            raise WorkflowError(
                "cosmos_rl_source_repository and cosmos_rl_source_branch are required "
                "for the clean Cosmos-RL build"
            )
        build_args = {
            "COSMOS_BACKEND": "cosmos-rl",
            "COSMOS_RL_BUILD_MODE": "no-efa",
            "VLLM_BASE_IMAGE": args.cosmos_rl_base_image,
            "COSMOS_RL_GITHUB_REPO": args.cosmos_rl_source_repository,
            "COSMOS_RL_GITHUB_BRANCH": args.cosmos_rl_source_branch,
            "COSMOS_RL_COMMIT": commits[native_name],
            "COSMOS_RL_TREE": args.native_tree,
            "ACTIONS_COMMIT": commits["cosmos-rl"], "ACTIONS_TREE": args.integration_tree,
            "DAFT_COMMIT": commits["nvidia-tao-daft"], "DAFT_TREE": args.daft_tree,
            "TAO_CORE_COMMIT": commits["tao-core"], "TAO_CORE_TREE": args.tao_core_tree,
            "SOURCE_DIRTY": "0", "BUILD_TIMESTAMP": args.build_timestamp,
            "PYAV_WHEEL_SHA256": "f9a65d1f48b818323fb411e80358f89d77dec340b01d27c6b2dfbb9cbf4b779f",
        }
        commands = []
    command = ["docker", "build", "--pull", "-f", str(Path(integration["expanded"]) / dockerfile), "-t", image]
    source_repository = (
        args.cosmos_framework_source_repository
        if backend == "cosmos-framework"
        else args.cosmos_rl_source_repository
    )
    if source_repository.startswith(("ssh://", "git@")):
        if not args.ssh_key_path:
            raise WorkflowError("ssh_key_path is required for an SSH source repository")
        command[2:2] = ["--ssh", f"default={args.ssh_key_path}"]
    for key, value in build_args.items():
        command.extend(["--build-arg", f"{key}={value}"])
    command.append(args.build_context)
    commands.append(shlex.join(command))
    return {
        "tag": image, "dockerfile": dockerfile, "build_context": args.build_context,
        "native_repository": native_repo, "integration_repository": integration,
        "daft_repository": daft_repo, "tao_core_repository": tao_core_repo,
        "build_arguments": build_args, "clean_build_commands": commands,
        "required_commits": dict(commits),
        "required_trees": {
            native_name: args.native_tree, "cosmos-rl": args.integration_tree,
            "nvidia-tao-daft": args.daft_tree, "tao-core": args.tao_core_tree,
        },
        "provenance_path": "/opt/tao/image-provenance.json",
        "must_rebuild_after_source_change": True,
        "sqsh": {
            "target": args.sqsh_path,
            "reuse_allowed": False,
            "command": shlex.join(["enroot", "import", "--output", args.sqsh_path, f"dockerd://{image}"]) if args.sqsh_path else None,
            "verification": "record SHA256 and verify /opt/tao/image-provenance.json through Pyxis before launch",
        },
    }


def _model_preparation(args: argparse.Namespace, model: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    supplied_format = args.base_model_format
    detected = model.get("format")
    if supplied_format == "auto":
        tier = model_tier(args.model)
        if tier == "edge":
            supplied_format = "cosmos3_edge"
        elif model.get("source_type") == "uri":
            raise WorkflowError("base_model_format must be explicit for a model URI")
        else:
            supplied_format = "qwen3_vl" if detected == "qwen3_vl" else "cosmos3_omni" if detected == "cosmos3_omni" else "unknown"
    if args.prepared_checkpoint_path:
        prepared = model["prepared_checkpoint"]
        accepted = {"qwen3_vl"} if model_tier(args.model) == "nano" else {"cosmos3_edge", "nemotron_h", "nemotron_vl"}
        if prepared.get("format") not in accepted:
            raise WorkflowError(f"prepared_checkpoint_path has incompatible model_type={prepared.get('format')!r}")
        return args.prepared_checkpoint_path, {"required": False, "reason": "validated prepared checkpoint supplied", "output": prepared}
    if model.get("source_type") == "local" and supplied_format in {"qwen3_vl", "cosmos3_edge"}:
        return args.base_model_path_or_uri, {"required": False, "reason": f"base model is already {supplied_format}; no processor overlay is created", "output": model["supplied"]}
    output = str((Path(args.checkpoint_dir).expanduser() / "prepared" / model["fingerprint"][:16]).resolve())
    if supplied_format in {"qwen3_vl", "cosmos3_edge"}:
        command = " ".join([
            "docker run --rm --entrypoint python",
            "-e HF_TOKEN",
            f"-e HF_MODEL_ID={shlex.quote(args.base_model_path_or_uri)}",
            f"-e HF_MODEL_REVISION={shlex.quote(args.base_model_revision)}",
            f"-v {shlex.quote(str(Path(args.checkpoint_dir).expanduser().resolve()))}:/output",
            f"-v {shlex.quote(str(Path(args.cache_dir).expanduser().resolve()))}:/cache",
            shlex.quote(args.image_tag), "-c",
            shlex.quote(
                "import os; from huggingface_hub import snapshot_download; "
                "snapshot_download(os.environ['HF_MODEL_ID'], revision=os.environ['HF_MODEL_REVISION'], "
                f"local_dir='/output/prepared/{model['fingerprint'][:16]}', cache_dir='/cache/huggingface')"
            ),
        ])
        return output, {
            "required": True, "kind": "immutable_public_checkpoint_snapshot", "output": path_identity(output, required=False),
            "command": command, "provenance": "fingerprint model/tokenizer/processor after download; do not modify checkpoint files",
        }
    if supplied_format != "cosmos3_omni":
        raise WorkflowError(f"unsupported Cosmos3-Nano base checkpoint format: {supplied_format}")
    if not args.vlm_architecture_model_path_or_uri:
        raise WorkflowError("Cosmos3 Omni conversion requires vlm_architecture_model_path_or_uri")
    if ("://" in args.vlm_architecture_model_path_or_uri or not Path(args.vlm_architecture_model_path_or_uri).expanduser().exists()) and not args.vlm_architecture_model_revision:
        raise WorkflowError("immutable architecture-model revision is required for a URI/identifier")
    script = SKILL_DIR / "scripts" / "prepare_cosmos3_vlm_checkpoint.py"
    command = [
        "python", str(script), "--base-model-path-or-uri", args.base_model_path_or_uri,
        "--vlm-architecture-model-path-or-uri", args.vlm_architecture_model_path_or_uri,
        "--output-path", output, "--cache-dir", args.cache_dir,
        "--framework-image", args.image_tag,
        "--framework-image-digest", "<RESOLVE_AFTER_CLEAN_BUILD>",
    ]
    if args.base_model_revision:
        command.extend(["--base-model-revision", args.base_model_revision])
    if args.vlm_architecture_model_revision:
        command.extend(["--vlm-architecture-model-revision", args.vlm_architecture_model_revision])
    return output, {
        "required": True, "kind": "cosmos3_omni_to_exact_qwen3_vl", "output": path_identity(output, required=False),
        "command": shlex.join(command), "provenance": "tao_conversion_provenance.json plus exact tensor/config validation",
    }


def _preflight_contract(
    args: argparse.Namespace,
    backend: str,
    plan_image: Mapping[str, Any],
    prepared_model: str,
    representative_media: str,
    decoder_artifact: Mapping[str, Any] | None = None,
    rl_video_runtime: Mapping[str, Any] | None = None,
    framework_video_runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    decoder_artifact = decoder_artifact or {"enabled": False}
    python = "/workspace/.venv/bin/python" if backend == "cosmos-framework" else "/opt/venv/cosmos_rl/bin/python"
    imports = [
        "import torch",
        "assert torch.cuda.is_available(), 'TAO_PREFLIGHT_ASSERTION_FAILED:cuda_available'",
        (
            f"assert torch.cuda.device_count() == {args.gpus_per_node}, "
            "'TAO_PREFLIGHT_ASSERTION_FAILED:cuda_device_count'"
        ),
    ]
    if backend == "cosmos-framework":
        if not framework_video_runtime:
            raise WorkflowError("Cosmos Framework preflight has no resolved video runtime")
        imports.extend([
            "import cosmos_framework", "import inspect", "import os", "from cosmos_framework.callbacks.tao_status import TAOStatusCallback",
            "from cosmos_framework.data.generator.dataflow import ContiguousBatcher",
            "from cosmos_framework.data.generator.dataflow import CosmosDataLoader",
            "from cosmos_framework.configs.base.reasoner.experiment.tao_video_sft import VideoSFTProcessor",
            "import cosmos_framework.data.generator.dataflow.loader as framework_dataflow_loader",
            "from cosmos_framework.scripts.export_vlm_dcp import export_vlm_dcp",
            "import torchcodec",
            "assert 'max_tokens' in inspect.signature(ContiguousBatcher).parameters, 'TAO_PREFLIGHT_ASSERTION_FAILED:contiguous_batcher_max_tokens'",
            "assert ContiguousBatcher.preserves_source_order is True, 'TAO_PREFLIGHT_ASSERTION_FAILED:contiguous_batcher_source_order'",
            "loader_source=inspect.getsource(framework_dataflow_loader._DataflowIterableDataset); assert 'group[-1]' in loader_source and 'cursor_epoch' in loader_source, 'TAO_PREFLIGHT_ASSERTION_FAILED:cross_epoch_resume_cursor'",
            "assert os.environ.get('TAO_VIDEO_DECODER_DEVICE') == 'cuda'",
            f"assert os.environ.get('TAO_VIDEO_CACHE_SIZE') == {str(framework_video_runtime['video_cache_size'])!r}",
            f"assert os.environ.get('TAO_FRAMEWORK_SFT_PROCESS_THREADS') == {str(framework_video_runtime['sft_process_threads'])!r}",
            f"assert os.environ.get('TAO_FRAMEWORK_DATALOADER_NUM_WORKERS') == {str(framework_video_runtime['dataloader_num_workers'])!r}",
            f"assert os.environ.get('TAO_VIDEO_DECODER_THREADS') == {str(framework_video_runtime['decoder_threads'])!r}",
            "framework_loader_source=inspect.getsource(CosmosDataLoader.__init__); assert 'multiprocessing_context' in framework_loader_source, 'TAO_PREFLIGHT_ASSERTION_FAILED:framework_spawn_prefetch'",
            "assert '__getstate__' in VideoSFTProcessor.__dict__ and '__setstate__' in VideoSFTProcessor.__dict__, 'TAO_PREFLIGHT_ASSERTION_FAILED:framework_spawn_pickle'",
            f"from torchcodec.decoders import VideoDecoder; d=VideoDecoder({representative_media!r}, device='cuda'); frame=d.get_frames_at([0]).data; assert str(frame.device).startswith('cuda'), frame.device; assert len(d)>0",
        ])
    else:
        imports.extend([
            "import cosmos_rl", "import av", "import inspect", "import os",
            "from nvidia_tao_core.microservices.handlers import huggingface_inference_microservice_server",
            "from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed",
            "from cosmos_rl.dispatcher.data.packer.hf_vlm_data_packer import HFVLMDataPacker",
            "from cosmos_rl.policy.trainer.llm_trainer import sft_trainer as sft_trainer_module",
            "assert (getattr(Qwen3VLVisionPatchEmbed.forward, '_tao_linear_patch_embed', False) or getattr(Qwen3VLVisionPatchEmbed.forward, '_tao_channels_last_3d', False)), 'TAO_PREFLIGHT_ASSERTION_FAILED:qwen_patch_embed'",
            "assert os.environ.get('COSMOS_SFT_REQUIRE_VISUAL_GRADIENTS') == '1', 'TAO_PREFLIGHT_ASSERTION_FAILED:visual_gradient_env'",
            "vlm_collate_source=inspect.getsource(HFVLMDataPacker._collate_fn)",
            "assert ('batch[\"attention_mask\"]' in vlm_collate_source or ('batch[\"input_ids\"] = torch.tensor' in vlm_collate_source and 'computed_max_len - len(x[\"input_ids\"])' in vlm_collate_source and 'batch[\"logprob_masks\"] = torch.tensor' in vlm_collate_source and 'computed_max_len - len(x[\"logprob_masks\"])' in vlm_collate_source)), 'TAO_PREFLIGHT_ASSERTION_FAILED:vlm_padding_contract'",
            "sft_source=inspect.getsource(sft_trainer_module.SFTTrainer.step_training)",
            "assert ('_enforce_visual_gradient_contract' in sft_source or ('self.data_packer.sft_collate_fn(' in sft_source and 'output = self.forward_model(**batch)' in sft_source and 'loss.backward()' in sft_source)), 'TAO_PREFLIGHT_ASSERTION_FAILED:visual_gradient_contract'",
            "assert av.codec.Codec('h264', 'r').name == 'h264', 'TAO_PREFLIGHT_ASSERTION_FAILED:h264_software_name'",
            "assert av.codec.Codec('hevc', 'r').name == 'hevc', 'TAO_PREFLIGHT_ASSERTION_FAILED:hevc_software_name'",
            "from cosmos_rl.utils.runtime_dependency_contract import verify_deepep, verify_vllm_conv3d",
            "verify_deepep()", "verify_vllm_conv3d()",
            "import qwen_vl_utils.vision_process as vp",
        ])
        if not rl_video_runtime:
            raise WorkflowError("Cosmos-RL preflight has no resolved video runtime")
        if rl_video_runtime["selected_profile"] == "pynv-device-rgbp":
            imports.extend([
                "import inspect",
                "import PyNvVideoCodec as nvc",
                "from cuda.bindings import driver as cuda_driver",
                "from cosmos_rl.policy.worker.sft_worker import _dataloader_worker_kwargs",
                "from cosmos_rl.dispatcher.data.packer import hf_vlm_data_packer as packer_module",
                "import cosmos_rl.launcher.launch_all as launch_all_module",
                "from cosmos_rl.utils.pynv_video_reader import register_pynv_video_reader",
                "from cosmos_rl.utils.video_pixel_bounds import normalize_video_pixel_bounds",
                "import cosmos_rl.utils.pynv_video_reader as pynv_reader",
                "assert nvc.OutputColorType.RGBP is not None, 'TAO_PREFLIGHT_ASSERTION_FAILED:pynv_rgbp'",
                "assert cuda_driver is not None, 'TAO_PREFLIGHT_ASSERTION_FAILED:cuda_driver_binding'",
                "assert os.environ.get('FORCE_QWENVL_VIDEO_READER') == 'pynvvideocodec', 'TAO_PREFLIGHT_ASSERTION_FAILED:forced_pynv_reader'",
                "assert os.environ.get('TAO_PYNV_FRAME_TRANSFER') == 'device_rgbp', 'TAO_PREFLIGHT_ASSERTION_FAILED:device_rgbp_env'",
                "worker_source=inspect.getsource(vp._ensure_forced_video_reader)",
                "assert 'TAO_PYNV_DECODER_CACHE_SIZE' in worker_source, 'TAO_PREFLIGHT_ASSERTION_FAILED:worker_decoder_cache_forwarding'",
                "packer_source=inspect.getsource(packer_module.qwen_vl_process_vision_info)",
                "assert 'normalize_video_pixel_bounds(' in packer_source, 'TAO_PREFLIGHT_ASSERTION_FAILED:worker_pixel_bound_normalization'",
                "assert 'vision_process.fetch_video(' in packer_source, 'TAO_PREFLIGHT_ASSERTION_FAILED:processed_video_cache_binding'",
                "worker_kwargs=_dataloader_worker_kwargs(1, 2)",
                "assert worker_kwargs['persistent_workers'] is True, 'TAO_PREFLIGHT_ASSERTION_FAILED:persistent_workers'",
                (
                    "profile=register_pynv_video_reader("
                    f"cache_size={int(rl_video_runtime['video_cache_size'])},"
                    f"decoder_cache_size={int(rl_video_runtime['decoder_cache_size'])},"
                    "strict=True)"
                ),
                "assert profile['frame_transfer'] == 'device_rgbp', 'TAO_PREFLIGHT_ASSERTION_FAILED:registered_frame_transfer'",
                "assert profile['capability_fallback'] == 'tao_system_pyav_sparse', 'TAO_PREFLIGHT_ASSERTION_FAILED:capability_fallback'",
                "assert '_is_nvdec_capability_error' in inspect.getsource(pynv_reader), 'TAO_PREFLIGHT_ASSERTION_FAILED:capability_classifier'",
                "assert 'TAO_VIDEO_DECODER_CAPABILITY_FALLBACK_ATTESTATION' in inspect.getsource(pynv_reader), 'TAO_PREFLIGHT_ASSERTION_FAILED:capability_attestation'",
                "pixel_probe={'video':'/tmp/tao-pixel-bound-probe.mp4','max_pixels':81920}",
                "pixel_probe=normalize_video_pixel_bounds(pixel_probe,16,vp)",
                "assert pixel_probe.get('min_pixels') == pixel_probe['max_pixels'] == 81920, 'TAO_PREFLIGHT_ASSERTION_FAILED:pixel_bound_visibility'",
                "assert isinstance(pixel_probe['min_pixels'],int) and isinstance(pixel_probe['max_pixels'],int), 'TAO_PREFLIGHT_ASSERTION_FAILED:pixel_bound_type'",
                "assert 'controller_id == -1 or i == controller_id' not in inspect.getsource(launch_all_module), 'TAO_PREFLIGHT_ASSERTION_FAILED:all_child_failures_propagate'",
                "assert vp.get_video_reader_backend() == 'pynvvideocodec', 'TAO_PREFLIGHT_ASSERTION_FAILED:registered_qwen_backend'",
            ])
        else:
            imports.extend([
                "from cosmos_rl.utils.system_pyav_video_reader import _assert_software_video_decoders, register_system_pyav_video_reader",
                "assert _assert_software_video_decoders() == {'h264': 'h264', 'hevc': 'hevc'}",
                "assert os.environ.get('FORCE_QWENVL_VIDEO_READER') == 'torchvision'",
                "assert vp.get_video_reader_backend() == 'torchvision'",
                "register_system_pyav_video_reader()",
                f"c=av.open({representative_media!r}); frame=next(c.decode(video=0)); assert frame is not None; c.close()",
            ])
    if args.dataset_family == "task_aware_video_reasoning":
        imports.append("import nvidia_tao_daft")
    imports.extend([
        "p=torch.cuda.get_device_properties(0)",
        "assert p.total_memory >= 30 * 1024**3, 'TAO_PREFLIGHT_ASSERTION_FAILED:gpu_memory'",
        "import tempfile; f=tempfile.NamedTemporaryFile(delete=False); f.close(); torch.distributed.init_process_group('nccl', init_method='file://'+f.name, rank=0, world_size=1); torch.distributed.destroy_process_group()",
        "print({'gpu': p.name, 'capability': (p.major,p.minor), 'memory':p.total_memory, 'torch':torch.__version__, 'cuda':torch.version.cuda})",
    ])
    container_checks = [f"{python} -c {shlex.quote('; '.join(imports))}"]
    if decoder_artifact["enabled"]:
        validator = [
            python,
            "-m",
            "cosmos_rl.utils.validate_video_override_artifacts",
            *decoder_artifact["validation_arguments"],
        ]
        container_checks.append(shlex.join(validator))
    container_check = " && ".join(container_checks)
    path_values = [prepared_model, args.results_dir, args.checkpoint_dir, args.cache_dir, *args.train_annotation, *args.train_media_root, *args.validation_annotation, *args.validation_media_root]
    path_checks = " && ".join(f"test -r {shlex.quote(value)}" for value in path_values)
    host = "command -v docker >/dev/null && docker version >/dev/null"
    if args.platform == "slurm":
        host = "command -v ssh >/dev/null"
        allocation = " && ".join([
            "command -v enroot >/dev/null", "srun --help 2>&1 | grep -q -- --container-image",
            f"test -r {shlex.quote(args.sqsh_path)}", path_checks,
            f"df -Pk {shlex.quote(args.results_dir)} {shlex.quote(args.checkpoint_dir)}",
            "nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader",
        ])
        if backend == "cosmos-framework":
            container_env_names = [
                "TAO_VIDEO_DECODER_DEVICE",
                "TAO_VIDEO_CACHE_SIZE",
                "TAO_FRAMEWORK_SFT_PROCESS_THREADS",
                "TAO_FRAMEWORK_DATALOADER_NUM_WORKERS",
                "TAO_FRAMEWORK_DATALOADER_PREFETCH_FACTOR",
                "TAO_VIDEO_DECODER_THREADS",
            ]
        elif rl_video_runtime["selected_profile"] == "pynv-device-rgbp":
            container_env_names = [
                "COSMOS_SFT_REQUIRE_VISUAL_GRADIENTS",
                "FORCE_QWENVL_VIDEO_READER",
                "TAO_PYNV_FRAME_TRANSFER",
                "TAO_PYNV_VIDEO_CACHE_SIZE",
                "TAO_PYNV_DECODER_CACHE_SIZE",
                "TAO_SFT_BATCH_THREADS",
            ]
        else:
            container_env_names = [
                "COSMOS_SFT_REQUIRE_VISUAL_GRADIENTS",
                "FORCE_QWENVL_VIDEO_READER",
            ]
        container = " ".join([
            "srun", "--nodes=1", "--ntasks=1", f"--gpus={args.gpus_per_node}",
            "--no-container-remap-root", "--no-container-mount-home",
            f"--container-env={','.join(container_env_names)}",
            f"--container-image={shlex.quote(args.sqsh_path)}",
            "bash -lc", shlex.quote(container_check),
        ])
    else:
        allocation = path_checks
        container = f"docker run --rm --gpus all {shlex.quote(plan_image['tag'])} bash -lc {shlex.quote(container_check)}"
    return {
        "submission_host": host,
        "target_compute_node": allocation,
        "container_runtime": container,
        # The same assertions are also available without the outer runtime
        # wrapper so SLURM can execute them inside the one training container
        # startup. This avoids a second Pyxis/Enroot launch and keeps the
        # training command as the allocation's first container journey.
        "container_startup": container_check,
        "checks": [
            "host and scheduler tools", "credential presence without reading values", "build-source identity when an image build is required",
            "Pyxis/Enroot and SQSH readability", "container mounts/shared storage", "non-root Python imports",
            "GPU count/type/memory", "driver/CUDA/PyTorch", "NCCL initialization",
            "explicit selected Cosmos-RL video profile", "checksum-pinned software System PyAV image capability", "backward-safe Qwen3-VL PatchEmbed",
            "padding-aware VLM attention mask", "first-update visual-gradient contract",
            "DeepEP Python/extension ABI", "vLLM Qwen3-VL Conv3D dispatch guard",
            "fingerprinted decoder-artifact coverage", "384 GiB free result/checkpoint space",
        ],
    }


def _decoder_artifact_plan(
    args: argparse.Namespace,
    *,
    backend: str,
    model: Mapping[str, Any],
    model_profile: Mapping[str, Any],
    train_data: Mapping[str, Any],
    val_data: Mapping[str, Any],
    rl_video_runtime: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if args.video_override_max_macroblocks < 1 or args.video_override_workers < 1:
        raise WorkflowError(
            "video_override_max_macroblocks and video_override_workers must be positive"
        )
    supplied = (
        bool(args.video_override_map),
        bool(args.video_override_manifest),
        bool(args.video_override_fingerprint),
    )
    if any(supplied) and not all(supplied):
        raise WorkflowError(
            "video_override_map, video_override_manifest, and "
            "video_override_fingerprint must be supplied together"
        )
    if args.video_override_fingerprint and not re.fullmatch(
        r"[0-9a-f]{64}", args.video_override_fingerprint
    ):
        raise WorkflowError("video_override_fingerprint must be a lowercase SHA256 digest")

    dataset_fingerprint = stable_hash({
        "train": train_data["dataset_fingerprint"],
        "validation": val_data["dataset_fingerprint"],
    })
    processor_fingerprint = stable_hash({
        "revision": args.processor_revision,
        "profile": model_profile,
    })
    if backend == "cosmos-framework":
        if any(supplied):
            raise WorkflowError(
                "video override artifacts are owned by the Cosmos-RL backend and "
                "must not be supplied to cosmos-framework"
            )
        return {
            "required": False,
            "enabled": False,
            "path": None,
            "manifest": None,
            "sha256": None,
            "input_fingerprints": {
                "dataset": dataset_fingerprint,
                "model": model["fingerprint"],
                "processor": processor_fingerprint,
            },
            "policy": {
                "macroblock_scan": False,
                "force_all_validation_media": False,
                "forced_runtime_sources": [],
                "gpu_random_access_validation_required": False,
                "selection_basis": "framework_native_torchcodec_cuda_on_demand",
            },
            "preparation_module": None,
            "preparation_arguments": [],
            "preparation_command": None,
            "validation_module": None,
            "validation_arguments": [],
            "validation_command": None,
        }
    artifact_root = (
        Path(args.cache_dir).expanduser()
        / "video-overrides"
        / f"{dataset_fingerprint[:16]}-{args.tao_integration_commit[:12]}"
    )
    map_path = args.video_override_map or str(artifact_root / "video_override_map.json")
    manifest_path = args.video_override_manifest or str(artifact_root / "manifest.json")
    output_dir = str(artifact_root / "videos")

    preparation_arguments: list[str] = []
    for annotation, media_root in [
        *_paired_annotation_roots(args.train_annotation, args.train_media_root),
        *_paired_annotation_roots(args.validation_annotation, args.validation_media_root),
    ]:
        preparation_arguments.extend([
            "--annotation-media-root", _containerize(args, annotation),
            _containerize(args, media_root),
        ])
    hardware_decoder_profile = (
        backend == "cosmos-rl"
        and rl_video_runtime is not None
        and rl_video_runtime["selected_profile"] == "pynv-device-rgbp"
    )
    force_all_validation_media = False
    if force_all_validation_media:
        for annotation in args.validation_annotation:
            preparation_arguments.extend([
                "--force-annotation", _containerize(args, annotation)
            ])
    for video in args.video_override_force_video:
        preparation_arguments.extend(["--force-video", _containerize(args, video)])
    preparation_arguments.extend([
        "--output-dir", _containerize(args, output_dir),
        "--override-map", _containerize(args, map_path),
        "--manifest", _containerize(args, manifest_path),
        "--dataset-fingerprint", dataset_fingerprint,
        "--model-fingerprint", model["fingerprint"],
        "--processor-fingerprint", processor_fingerprint,
        "--max-macroblocks", str(args.video_override_max_macroblocks),
        "--workers", str(args.video_override_workers),
    ])

    validation_arguments = [
        "--override-map", _containerize(args, map_path),
        "--manifest", _containerize(args, manifest_path),
        "--artifact-fingerprint", args.video_override_fingerprint or "<ARTIFACT_FINGERPRINT>",
        "--dataset-fingerprint", dataset_fingerprint,
        "--model-fingerprint", model["fingerprint"],
        "--processor-fingerprint", processor_fingerprint,
        "--integration-commit", args.tao_integration_commit,
    ]
    if force_all_validation_media:
        for annotation in args.validation_annotation:
            validation_arguments.extend([
                "--require-covered-annotation", _containerize(args, annotation)
            ])

    python = "/opt/venv/cosmos_rl/bin/python"

    # RL starts directly from the selected manifests. Its source-baked reader
    # handles the narrow permanent NVDEC-capability exception on demand, so a
    # separately prepared compatibility artifact is optional. Framework was
    # returned above because it owns a separate native TorchCodec contract.
    artifact_required = False
    return {
        "required": artifact_required,
        "enabled": all(supplied),
        "path": args.video_override_map or None,
        "manifest": args.video_override_manifest or None,
        "sha256": args.video_override_fingerprint or None,
        "input_fingerprints": {
            "dataset": dataset_fingerprint,
            "model": model["fingerprint"],
            "processor": processor_fingerprint,
        },
        "policy": {
            "macroblock_scan": True,
            "force_all_validation_media": force_all_validation_media,
            "forced_runtime_sources": list(args.video_override_force_video),
            "gpu_random_access_validation_required": artifact_required,
            "selection_basis": (
                "optional_resolved_hardware_decoder_compatibility"
                if hardware_decoder_profile
                else "resolved_backend_data_contract"
            ),
        },
        "preparation_module": "cosmos_rl.utils.video_override_artifacts",
        "preparation_arguments": preparation_arguments,
        "preparation_command": shlex.join([
            python,
            "-m",
            "cosmos_rl.utils.video_override_artifacts",
            *preparation_arguments,
        ]),
        "validation_module": "cosmos_rl.utils.validate_video_override_artifacts",
        "validation_arguments": validation_arguments,
        "validation_command": shlex.join([
            python,
            "-m",
            "cosmos_rl.utils.validate_video_override_artifacts",
            *validation_arguments,
        ]),
    }


_SLURM_NODE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SLURM_UNHEALTHY_STATE = re.compile(
    r"(?:^|[+,_])(DOWN|DRAIN(?:ED|ING)?|FAIL(?:ED|ING)?|NOT_RESPONDING|POWER_DOWN)(?:$|[+,_])",
    re.IGNORECASE,
)
_SLURM_UNHEALTHY_COMMENT = re.compile(
    r"(?:network\s+diagnostics?|quarantin|do\s+not\s+schedule|unhealthy|"
    r"hardware\s+diagnostics?|pyxis|enroot|mount\s+failure|"
    r"(?:i/?o|io)[-_\s]*error|lustre[-_\s]*error|node[-_\s]*failure|"
    r"gpu[^\s]*(?:fail|xid)|xid\s*\d+)",
    re.IGNORECASE,
)


def _slurm_node_exclusion_contract(args: argparse.Namespace) -> dict[str, Any]:
    requested = list(dict.fromkeys(getattr(args, "exclude_node", []) or []))
    auto_filter = bool(getattr(args, "exclude_unhealthy_inventory_nodes", False))
    if not requested and not auto_filter:
        return {
            "requested": [],
            "validated": [],
            "retired_or_missing": [],
            "auto_excluded": [],
            "auto_exclusion_reasons": {},
            "inventory": None,
        }
    invalid = [value for value in requested if not _SLURM_NODE_NAME.fullmatch(value)]
    if invalid:
        raise WorkflowError(f"invalid SLURM node exclusion names: {invalid}")
    inventory_path = str(getattr(args, "slurm_node_inventory_file", "") or "")
    if not inventory_path:
        raise WorkflowError(
            "--slurm-node-inventory-file is required when explicit or automatic node exclusions are used; "
            "capture the live `scontrol show nodes -o` output immediately before planning"
        )
    path = Path(inventory_path).expanduser().resolve()
    if not path.is_file():
        raise WorkflowError(f"SLURM node inventory is inaccessible: {path}")
    text = path.read_text(encoding="utf-8")
    inventory_lines: dict[str, str] = {}
    for line in text.splitlines():
        match = re.search(r"(?:^|\s)NodeName=([A-Za-z0-9_.-]+)", line)
        if match:
            inventory_lines[match.group(1)] = line
    inventory = set(inventory_lines)
    if not inventory:
        inventory = {
            token
            for token in re.split(r"\s+", text.strip())
            if token and _SLURM_NODE_NAME.fullmatch(token)
        }
    if not inventory:
        raise WorkflowError(f"SLURM node inventory contains no node names: {path}")
    auto_excluded: list[str] = []
    auto_exclusion_reasons: dict[str, list[str]] = {}
    target_partitions = {
        value.strip()
        for value in str(getattr(args, "partition", "") or "").split(",")
        if value.strip()
    }
    if auto_filter:
        for node, line in inventory_lines.items():
            partitions_match = re.search(r"(?:^|\s)Partitions=([^\s]+)", line)
            if target_partitions and partitions_match:
                node_partitions = set(partitions_match.group(1).split(","))
                if target_partitions.isdisjoint(node_partitions):
                    continue
            gres_match = re.search(r"(?:^|\s)Gres=([^\s]+)", line)
            if gres_match and "gpu:" not in gres_match.group(1).lower():
                continue
            reasons: list[str] = []
            state_match = re.search(r"(?:^|\s)State=([^\s]+)", line)
            if state_match and _SLURM_UNHEALTHY_STATE.search(state_match.group(1)):
                reasons.append(f"scheduler_state={state_match.group(1)}")
            comment_match = re.search(r"(?:^|\s)Comment=(.*)$", line)
            if comment_match:
                comment = comment_match.group(1).strip()
                if (
                    comment
                    and comment.lower() not in {"(null)", "none", "n/a"}
                    and _SLURM_UNHEALTHY_COMMENT.search(comment)
                ):
                    reasons.append("scheduler_diagnostic_comment")
            if reasons:
                auto_excluded.append(node)
                auto_exclusion_reasons[node] = reasons
    validated = list(
        dict.fromkeys(
            [value for value in requested if value in inventory]
            + sorted(auto_excluded)
        )
    )
    retired = [value for value in requested if value not in inventory]
    return {
        "requested": requested,
        "validated": validated,
        "retired_or_missing": retired,
        "auto_excluded": sorted(auto_excluded),
        "auto_exclusion_reasons": auto_exclusion_reasons,
        "inventory": {"path": str(path), "sha256": sha256_file(path), "node_count": len(inventory)},
    }


def build_plan(
    args: argparse.Namespace,
    *,
    remote_inspection_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a training plan, optionally reusing a sealed prior inspection.

    ``remote_inspection_override`` exists only for the checked-in retry helper.
    It lets an infrastructure retry retain the original model/dataset evidence
    without repeating a multi-hour media inspection.  Ordinary callers never
    supply it and keep the existing inspection behavior.
    """
    remote_inspection = (
        copy.deepcopy(dict(remote_inspection_override))
        if remote_inspection_override is not None
        else (_remote_inspection(args) if _needs_remote_inspection(args) else None)
    )
    node_exclusions = _slurm_node_exclusion_contract(args)
    inspected_model = remote_inspection["model"] if remote_inspection else None
    args.model = resolve_model_name(args.model, args.base_model_path_or_uri, inspected_model)
    backend, reason = select_backend(model=args.model, action=args.action, backend=args.backend, workload=args.workload, comparative=args.comparative)
    if args.action != "train":
        raise WorkflowError("this planner currently materializes training; use the backend action contract for non-train actions")
    tier = model_tier(args.model)
    if tier == "edge" and backend != "cosmos-framework":
        raise WorkflowError("Cosmos3-Edge training requires Cosmos Framework")
    if args.run_mode == "full" and (args.train_sample_limit or args.validation_sample_limit):
        raise WorkflowError("full runs must not contain a smoke/subset sample limit")
    if args.async_checkpoint and args.nodes > 1:
        raise WorkflowError("asynchronous distributed checkpointing is disabled for multi-node Cosmos runs")
    if not args.results_dir or not args.checkpoint_dir or not args.cache_dir:
        raise WorkflowError("results_dir, checkpoint_dir, and cache_dir are required runtime paths")
    _align_container_runtime_paths(args)
    model = inspected_model or inspect_model(
        args.base_model_path_or_uri,
        args.base_model_revision,
        args.prepared_checkpoint_path,
    )
    prepared_model, model_preparation = _model_preparation(args, model)
    train_annotations, train_media = _annotation_args(args, "train")
    val_annotations, val_media = _annotation_args(args, "validation")
    if remote_inspection:
        train_data = remote_inspection["datasets"]["train"]
        val_data = remote_inspection["datasets"]["validation"]
    else:
        train_data = inspect_dataset(dataset_family=args.dataset_family, annotations=train_annotations, media_roots=train_media, selected_tasks=args.task, verify_media_content=not args.fast_media_fingerprint)
        val_data = inspect_dataset(dataset_family=args.dataset_family, annotations=val_annotations, media_roots=val_media, selected_tasks=args.task, verify_media_content=not args.fast_media_fingerprint)
    if train_data["dataset_family"] != val_data["dataset_family"]:
        raise WorkflowError("training and validation annotations resolve to different dataset families")
    args.dataset_family = train_data["dataset_family"]
    if args.dataset_family == "video_conversation" and (len(train_annotations) != 1 or len(val_annotations) != 1):
        raise WorkflowError("video_conversation requires exactly one annotation file per split")
    model_profile = resolve_model_profile(args, tier, backend, train_data, val_data)
    args.frames = model_profile["frames"]
    args.sequence_length = model_profile["sequence_length"]
    assert_no_overlap(train_data, val_data)
    total_gpus = args.nodes * args.gpus_per_node
    if min(train_data["record_count"], val_data["record_count"]) < total_gpus:
        raise WorkflowError("train and validation datasets must each contain at least one record per global GPU")
    args.smoke_train_samples = min(train_data["record_count"], max(args.smoke_train_samples, total_gpus))
    args.smoke_validation_samples = min(val_data["record_count"], max(args.smoke_validation_samples, total_gpus))
    contract = _training_contract(args)
    logical_train_records = (
        min(train_data["record_count"], args.smoke_train_samples)
        if args.run_mode == "smoke"
        else train_data["record_count"]
    )
    exposed_train_samples = logical_train_records * contract["train_sample_multiplier"]
    contract.update({
        "logical_train_records": logical_train_records,
        "exposed_train_samples": exposed_train_samples,
        "optimizer_updates": math.ceil(exposed_train_samples / args.effective_global_batch)
        * contract["epochs"],
    })
    commits = _source_commits(args, backend)
    image = _image_plan(args, backend, commits)
    if backend == "cosmos-framework" and getattr(args, "rl_video_profile", "auto") != "auto":
        raise WorkflowError("--rl-video-profile applies only to the cosmos-rl backend")
    if backend == "cosmos-framework" and any(
        getattr(args, name, value) != value
        for name, value in (
            ("rl_video_cache_size", None),
            ("rl_video_decoder_cache_size", None),
            ("rl_sft_batch_threads", 0),
            ("rl_dataloader_num_workers", None),
            ("rl_dataloader_prefetch_factor", None),
        )
    ):
        raise WorkflowError("Cosmos-RL video runtime overrides do not apply to cosmos-framework")
    if backend == "cosmos-rl" and any(
        getattr(args, name, 0)
        for name in (
            "framework_video_cache_size",
            "framework_sft_process_threads",
            "framework_video_decoder_threads",
            "framework_dataloader_num_workers",
            "framework_dataloader_prefetch_factor",
        )
    ):
        raise WorkflowError("Framework video runtime overrides do not apply to cosmos-rl")
    if backend == "cosmos-rl" and args.framework_per_forward_batch != 0:
        raise WorkflowError("Framework per-forward batch applies only to cosmos-framework")
    rl_video_runtime = (
        _rl_video_runtime(args, train_data, val_data)
        if backend == "cosmos-rl"
        else None
    )
    framework_video_runtime = (
        _framework_video_runtime(args, train_data, val_data)
        if backend == "cosmos-framework"
        else None
    )
    decoder_artifact = _decoder_artifact_plan(
        args,
        backend=backend,
        model=model,
        model_profile=model_profile,
        train_data=train_data,
        val_data=val_data,
        rl_video_runtime=rl_video_runtime,
    )
    processor_fingerprint = stable_hash({
        "revision": args.processor_revision,
        "profile": model_profile,
        "decoder_artifact": decoder_artifact,
        "rl_video_runtime": rl_video_runtime,
        "framework_video_runtime": framework_video_runtime,
    })
    cache_keys = {
        split: hashlib.sha256(
            (
                f"dataset={dataset['dataset_fingerprint']}\n"
                f"model={model['fingerprint']}\n"
                f"processor={processor_fingerprint}\n"
            ).encode()
        ).hexdigest()
        for split, dataset in (("train", train_data), ("validation", val_data))
    }
    prepared_model_container = _containerize(args, prepared_model)
    train_annotations_container = [_containerize(args, value) for value in train_annotations]
    train_media_container = [_containerize(args, value) for value in train_media]
    val_annotations_container = [_containerize(args, value) for value in val_annotations]
    val_media_container = [_containerize(args, value) for value in val_media]
    spec = _framework_spec(args, train_data["record_count"], val_data["record_count"], contract) if backend == "cosmos-framework" else _rl_spec(args, contract, prepared_model_container, train_annotations_container, train_media_container, val_annotations_container, val_media_container, cache_keys, rl_video_runtime)
    if backend == "cosmos-framework":
        contract.update({
            "per_forward_batch": spec["dataloader_train"]["max_samples_per_batch"],
            "gradient_accumulation": spec["trainer"]["grad_accum_iter"],
        })
    environment = _env(args, backend, prepared_model_container, train_annotations_container, train_media_container, val_annotations_container, val_media_container, rl_video_runtime, framework_video_runtime)
    command = _command(args, backend)
    if decoder_artifact["enabled"]:
        python = "/workspace/.venv/bin/python" if backend == "cosmos-framework" else "/opt/venv/cosmos_rl/bin/python"
        runtime_validation = [
            python,
            "-m",
            decoder_artifact["validation_module"],
            *decoder_artifact["validation_arguments"],
            "--skip-file-hashes",
        ]
        command = f"{shlex.join(runtime_validation)} &&\n{command}"
    remote_paths = remote_inspection.get("runtime_paths", {}) if remote_inspection else {}

    def runtime_path(label: str, value: str, *, required: bool = True) -> dict[str, Any]:
        if label in remote_paths:
            return remote_paths[label]
        if not value and not required:
            return path_identity(value, required=False)
        return planned_path_identity(value)

    cache_mode = getattr(args, "rl_dataset_cache_mode", "direct")
    cache_prewarm_required = (
        backend == "cosmos-rl"
        and args.dataset_family == "video_conversation"
        and cache_mode == "prewarm"
    )
    plan = {
        "schema_version": 2, "experiment_id": args.experiment_id, "model_name": args.model,
        "model": model, "action": args.action, "workflow": args.workload, "dataset_family": args.dataset_family, "backend": backend,
        "model_preparation": model_preparation, "prepared_model_container_path": prepared_model_container,
        "backend_selection_reason": reason, "backend_contract": str(BACKEND_FILES[backend]),
        "run_mode": args.run_mode, "training": contract, "processor_profile": model_profile,
        "decoder_artifact": decoder_artifact,
        "rl_video_runtime": rl_video_runtime,
        "framework_video_runtime": framework_video_runtime,
        "datasets": {"train": train_data, "validation": val_data},
        "input_frame": {
            "kind": "slurm_remote" if remote_inspection else "submission_host",
            "verified_host": remote_inspection.get("verified_host") if remote_inspection else None,
            "inspection_transport": "repository_helper_streamed_over_ssh" if remote_inspection else "local_filesystem",
        },
        "paths": {
            "results_dir": runtime_path("results_dir", args.results_dir), "checkpoint_dir": runtime_path("checkpoint_dir", args.checkpoint_dir),
            "cache_dir": runtime_path("cache_dir", args.cache_dir), "sqsh_cache_dir": runtime_path("sqsh_cache_dir", args.sqsh_cache_dir, required=args.platform == "slurm"),
            "ssh_key_path": path_identity(args.ssh_key_path, required=args.platform == "slurm"),
        },
        "image": image, "sqsh": runtime_path("sqsh_path", args.sqsh_path, required=args.platform == "slurm"),
        "compute": {"platform": args.platform, "nodes": args.nodes, "gpus_per_node": args.gpus_per_node, "total_gpus": total_gpus, "cpus_per_task": args.cpus_per_task},
        "slurm_node_exclusions": node_exclusions,
        "cache_prewarm": {"mode": cache_mode, "required": cache_prewarm_required, "keys": cache_keys if cache_prewarm_required else {}, "path": args.cache_dir if cache_prewarm_required else "", "dataset_fingerprints": {"train": train_data["dataset_fingerprint"], "validation": val_data["dataset_fingerprint"]}, "model_fingerprint": model["fingerprint"], "processor_fingerprint": processor_fingerprint, "completeness_required": cache_prewarm_required, "resumable": cache_prewarm_required, "selection_basis": {"media_reuse": train_data["profile"]["media_reuse_class"], "record_count": train_data["record_count"], "resolution_class": train_data["profile"]["resolution"]["class"]}},
        "spec": spec, "environment": environment, "command": command,
        "config_container_path": args.container_spec_path,
        "evaluation_contract": {
            "schema_version": 1,
            "source": "sealed_training_plan",
            "validation_dataset_fingerprint": val_data["dataset_fingerprint"],
            "validation_annotations": [
                item["original"] for item in val_data["annotations"]
            ],
            "validation_media_roots": [
                item["original"] for item in val_data["media_roots"]
            ],
            "system_prompt": contract["system_prompt"],
            "frames": model_profile["frames"],
            "vision": contract["vision"],
            "max_video_pixels": model_profile["max_video_pixels"],
            "precision": contract["precision"],
            "seed": contract["seed"],
            "batch_size": args.validation_batch_size,
            "task_profile": val_data["evaluation_profile"],
            "generation": {
                "max_tokens": None,
                "temperature": 0.0,
                "repetition_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
            },
            "checkpoint_selection": None,
            "required_evaluation_intake": [
                "results_dir",
                "checkpoint_selection",
                "generation.max_tokens",
                *val_data["evaluation_profile"]["requires_user_input"],
            ],
        },
        "diagnostic_subset": {
            "required": False,
            "policy": "opt_in_only",
            "requested": args.run_mode == "smoke",
            "train_samples": args.smoke_train_samples,
            "validation_samples": args.smoke_validation_samples,
        },
        "first_update_gate": {
            "backend": backend,
            "required": backend == "cosmos-rl",
            "execution": "in_process_before_first_optimizer_update",
            "criteria": [
                "padding_aware_attention_mask",
                "component_parameter_counts_present",
                "trainable_visual_gradients_present",
                "trainable_visual_gradient_norms_finite_and_nonzero",
            ],
            "status_keys": [
                "model/components/vision_encoder/grad_norm",
                "model/components/vision_projector/grad_norm",
                "model/components/language_model/grad_norm",
                "model/components/lm_head/grad_norm",
                "model/components/visual_gradient_contract",
            ],
        },
        "metric_contract": {"train": {"key": "train/avg_loss", "weight": "valid_labels", "requires": ["train/loss_numerator", "train/valid_label_count"]}, "validation": {"key": "val/avg_loss", "weight": "valid_labels", "requires": ["val/loss_numerator", "val/valid_label_count"]}, "accuracy": {"route": "shared repository evaluator", "aggregation": val_data["metric_coverage"]["aggregate"], "coverage": val_data["metric_coverage"]}},
    }
    representative_media = _containerize(args, train_data["media_manifest"][0]["path"])
    plan["preflight"] = _preflight_contract(
        args, backend, image, prepared_model, representative_media,
        decoder_artifact, rl_video_runtime, framework_video_runtime
    )
    return plan


def write_spec(
    args: argparse.Namespace,
    plan: dict[str, Any],
    *,
    allow_remote_write: bool = False,
) -> Path:
    if not args.write_spec:
        raise WorkflowError("write_spec is required so the submitted config can be fingerprinted")
    output = Path(args.write_spec).expanduser()
    spec = copy.deepcopy(plan["spec"])
    is_remote = plan.get("input_frame", {}).get("kind") == "slurm_remote"
    verified_host = str(plan.get("input_frame", {}).get("verified_host") or "")
    materializations: list[dict[str, Any]] = []

    def materialize(split: str, marker: str, limit: int = 0) -> str:
        target = output.with_name(f"{split}_{'smoke' if limit else 'merged'}.json")
        if is_remote:
            record = {
                "split": split,
                "original": str(target),
                "container": _containerize(args, str(target)),
                "sample_limit": limit,
                "materialized": False,
            }
            if allow_remote_write:
                if not verified_host:
                    raise WorkflowError("remote materialization has no verified SLURM login host")
                result = _remote_materialize_dataset(
                    args, split=split, output_path=str(target),
                    sample_limit=limit, host=verified_host,
                )
                record.update(
                    {
                        "materialized": True,
                        "sha256": result["sha256"],
                        "record_count": result["record_count"],
                    }
                )
            materializations.append(record)
        else:
            annotations, _ = _annotation_args(args, split)
            result = materialize_dataset(
                dataset_family=args.dataset_family,
                annotations=annotations,
                output_path=str(target),
                selected_tasks=args.task,
                sample_limit=limit,
            )
            materializations.append(
                {
                    "split": split,
                    "original": str(target),
                    "container": _containerize(args, str(target.resolve())),
                    "sample_limit": limit,
                    "materialized": True,
                    "sha256": result["sha256"],
                    "record_count": result["record_count"],
                }
            )
        return materializations[-1]["container"]

    if plan["backend"] == "cosmos-rl":
        for split, marker, key in (
            ("train", "__TAO_TRAIN_MERGED_MANIFEST__", "train_dataset"),
            ("validation", "__TAO_VALIDATION_MERGED_MANIFEST__", "val_dataset"),
        ):
            current = spec["custom"][key]["annotation_path"]
            smoke_limit = (args.smoke_train_samples if split == "train" else args.smoke_validation_samples) if args.run_mode == "smoke" else 0
            if current == marker or smoke_limit:
                spec["custom"][key]["annotation_path"] = materialize(split, marker, smoke_limit)
    encoded = dump_toml(spec)
    # Parse before any local or remote write so invalid TOML cannot cross the
    # launch boundary.
    tomllib.loads(encoded)
    expected_sha256 = hashlib.sha256(encoded.encode()).hexdigest()
    materialized = False
    resolved = str(output) if is_remote else str(output.resolve())
    if is_remote:
        if allow_remote_write:
            if not verified_host:
                raise WorkflowError("remote config materialization has no verified SLURM login host")
            actual_sha256 = _remote_write_text(
                args, output_path=str(output), content=encoded, host=verified_host,
            )
            if actual_sha256 != expected_sha256:
                raise WorkflowError(
                    f"remote config checksum mismatch: expected {expected_sha256}, found {actual_sha256}"
                )
            materialized = True
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
        materialized = True
    plan["config"] = {
        "original": args.write_spec,
        "resolved": resolved,
        "container": args.container_spec_path,
        "sha256": expected_sha256,
        "materialized": materialized,
        "frame": "target_compute" if is_remote else "submission_host",
    }
    plan["generated_artifacts"] = materializations
    plan["spec"] = spec
    return output


def verify_materialized_spec(args: argparse.Namespace, plan: Mapping[str, Any]) -> None:
    config = plan.get("config", {})
    expected = str(config.get("sha256") or "")
    if not expected:
        raise WorkflowError("generated config has no expected checksum")
    if plan.get("input_frame", {}).get("kind") == "slurm_remote":
        host = str(plan.get("input_frame", {}).get("verified_host") or "")
        if not host:
            raise WorkflowError("remote config verification has no verified SLURM login host")
        actual = _remote_file_sha256(args, path=args.write_spec, host=host)
    else:
        path = Path(args.write_spec).expanduser()
        if not path.is_file():
            raise WorkflowError(f"required generated config is inaccessible: {path}")
        actual = sha256_file(path)
    if actual != expected:
        raise WorkflowError(
            f"generated config is stale: expected SHA256 {expected}, found {actual}; rerun materialize"
        )


def _planner_request(args: argparse.Namespace) -> dict[str, Any]:
    """Return the resolved, credential-free request needed by later verbs."""
    request: dict[str, Any] = {}
    for name, value in vars(args).items():
        if name in _PLAN_ARTIFACT_TRANSIENT_ARGS:
            continue
        if isinstance(value, Path):
            value = str(value)
        elif isinstance(value, tuple):
            value = list(value)
        try:
            json.dumps(value)
        except TypeError as exc:
            raise WorkflowError(f"planner request field {name!r} is not JSON serializable") from exc
        request[name] = copy.deepcopy(value)
    return request


def _plan_artifact_sha256(plan: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(plan))
    payload.pop("plan_artifact", None)
    return stable_hash(payload)


def save_plan_artifact(
    args: argparse.Namespace,
    plan: dict[str, Any],
    artifact_path: str,
) -> Path:
    """Atomically persist the inspected plan for all post-review verbs."""
    if not artifact_path or "://" in artifact_path:
        raise WorkflowError("plan_artifact must be a local controller-side filesystem path")
    path = Path(artifact_path).expanduser().resolve()
    plan["planner_request"] = _planner_request(args)
    plan["plan_artifact"] = {
        "schema_version": PLAN_ARTIFACT_SCHEMA_VERSION,
        "path": str(path),
        "sha256": _plan_artifact_sha256(plan),
    }
    encoded = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _atomic_write_text(path: Path, content: str) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_plan_artifact(
    current_args: argparse.Namespace,
    artifact_path: str,
) -> tuple[argparse.Namespace, dict[str, Any]]:
    """Load a sealed plan and make its resolved request authoritative."""
    if not artifact_path or "://" in artifact_path:
        raise WorkflowError("plan_artifact must be a local controller-side filesystem path")
    path = Path(artifact_path).expanduser().resolve()
    if not path.is_file():
        raise WorkflowError(f"approved plan artifact is inaccessible: {path}")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise WorkflowError("approved plan artifact must contain a JSON object")
    artifact = plan.get("plan_artifact", {})
    if artifact.get("schema_version") != PLAN_ARTIFACT_SCHEMA_VERSION:
        raise WorkflowError("approved plan artifact has an unsupported schema version")
    expected = str(artifact.get("sha256") or "")
    actual = _plan_artifact_sha256(plan)
    if not expected or actual != expected:
        raise WorkflowError(
            f"approved plan artifact checksum mismatch: expected {expected or '<missing>'}, found {actual}"
        )
    request = plan.get("planner_request")
    if not isinstance(request, dict) or not request:
        raise WorkflowError("approved plan artifact has no resolved planner request")
    args = argparse.Namespace(**copy.deepcopy(request))
    args.verb = current_args.verb
    args.format = current_args.format
    args.plan_artifact = str(path)
    args.render_output = getattr(current_args, "render_output", "")
    return args, plan


def render_slurm(args: argparse.Namespace, plan: Mapping[str, Any]) -> str:
    if args.platform != "slurm":
        raise WorkflowError("SLURM script rendering requires platform=slurm")
    if not args.partition or not args.account or not args.sqsh_path:
        raise WorkflowError("SLURM partition, account, and SQSH path are required")
    if args.use_requeue:
        raise WorkflowError("requeue is disabled by default and is not validated for Cosmos training")
    if plan["decoder_artifact"]["required"] and not plan["decoder_artifact"]["enabled"]:
        raise WorkflowError(
            "Cosmos training requires a complete fingerprinted decoder compatibility "
            "artifact for the resolved hardware video profile"
        )
    try:
        timeout_hours, timeout_minutes, timeout_seconds = (
            int(value) for value in args.timeout.split(":")
        )
    except (ValueError, AttributeError) as exc:
        raise WorkflowError("child timeout must use HH:MM:SS format") from exc
    if timeout_hours < 0 or not 0 <= timeout_minutes < 60 or not 0 <= timeout_seconds < 60:
        raise WorkflowError("child timeout must use valid HH:MM:SS fields")
    child_timeout_seconds = timeout_hours * 3600 + timeout_minutes * 60 + timeout_seconds
    if child_timeout_seconds <= 0:
        raise WorkflowError("child timeout must be greater than zero")
    sqsh = Path(args.sqsh_path)
    if not args.container_mount:
        raise WorkflowError("at least one explicit container mount is required for SLURM")
    mount_args = f"--container-mounts={shlex.quote(','.join(args.container_mount))}"
    container_env_args = f"--container-env={','.join(sorted(plan['environment']))}"
    env_exports = "\n".join(f"export {key}={shlex.quote(value)}" for key, value in plan["environment"].items())
    native = plan["command"]
    container_startup = str(plan.get("preflight", {}).get("container_startup") or "").strip()
    startup_lines = []
    if container_startup:
        startup_lines = [
            "runtime_preflight_rc=0",
            f"{container_startup} || runtime_preflight_rc=$?",
            'if [[ "$runtime_preflight_rc" -ne 0 ]]; then',
            '  echo "Cosmos packaged runtime startup check failed with exit code $runtime_preflight_rc" >&2',
            '  exit "$runtime_preflight_rc"',
            "fi",
            'echo "TAO_COSMOS_PACKAGED_RUNTIME_STARTUP_OK"',
        ]
    wrapped = "\n".join([
        'export HOME="/tmp/tao-${TAO_JOB_ID:?TAO_JOB_ID must be set}-${SLURM_PROCID:-0}"',
        'mkdir -p -m 700 "$HOME"',
        "ulimit -n 65536",
        "ulimit -s unlimited",
        "ulimit -l unlimited 2>/dev/null || true",
        *startup_lines,
        native,
    ])
    step_cpu_value = '"$step_cpus_per_task"' if args.exclusive and args.nodes == 1 else str(args.cpus_per_task)
    srun = " ".join(filter(None, [
        "timeout", "--signal=TERM", "--kill-after=30s", f"{child_timeout_seconds}s",
        "srun", f"--nodes={args.nodes}", f"--ntasks={args.nodes}", "--ntasks-per-node=1",
        f"--gpus-per-node={args.gpus_per_node}", f"--cpus-per-task={step_cpu_value}",
        "--no-container-remap-root", "--no-container-mount-home",
        container_env_args,
        f"--container-image={shlex.quote(str(sqsh))}",
        mount_args, "bash -lc", shlex.quote(wrapped),
    ]))
    job_name = args.tao_job_id or args.experiment_id
    writable_runtime_dirs = list(dict.fromkeys(
        str(Path(value).expanduser())
        for value in (args.results_dir, args.checkpoint_dir, args.cache_dir)
    ))
    runtime_dir_setup = [
        f"mkdir -p -- {shlex.quote(value)}" for value in writable_runtime_dirs
    ]
    lines = [
        "#!/usr/bin/env bash", f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={args.partition}",
        f"#SBATCH --account={args.account}", f"#SBATCH --nodes={args.nodes}", f"#SBATCH --ntasks={args.nodes}",
        "#SBATCH --ntasks-per-node=1", f"#SBATCH --gpus-per-node={args.gpus_per_node}",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}", f"#SBATCH --time={args.time_limit}", "#SBATCH --no-requeue",
        f"#SBATCH --output={args.stdout_path}", f"#SBATCH --error={args.stderr_path}",
    ]
    validated_exclusions = plan.get("slurm_node_exclusions", {}).get("validated", [])
    if validated_exclusions:
        lines.append(f"#SBATCH --exclude={','.join(validated_exclusions)}")
    if args.qos:
        lines.append(f"#SBATCH --qos={args.qos}")
    if args.reservation:
        lines.append(f"#SBATCH --reservation={args.reservation}")
    if args.exclusive:
        lines.append("#SBATCH --exclusive")
    cpu_step_setup = []
    if args.exclusive and args.nodes == 1:
        cpu_step_setup = [
            f"requested_cpus_per_task={args.cpus_per_task}",
            'slurm_job_record="$(scontrol show job -o "${SLURM_JOB_ID:?SLURM_JOB_ID must be set}")"',
            'if [[ "$slurm_job_record" =~ NumCPUs=([0-9]+) ]]; then',
            '  step_cpus_per_task="${BASH_REMATCH[1]}"',
            "else",
            '  echo "Unable to resolve NumCPUs for exclusive SLURM allocation $SLURM_JOB_ID" >&2',
            "  exit 2",
            "fi",
            'if (( step_cpus_per_task < requested_cpus_per_task )); then',
            '  echo "Exclusive allocation exposes fewer CPUs than requested: requested=$requested_cpus_per_task allocated=$step_cpus_per_task" >&2',
            "  exit 2",
            "fi",
            'printf "TAO_SLURM_CPU_ALLOCATION requested=%s allocated=%s step=%s policy=allocated-exclusive-single-node\\n" "$requested_cpus_per_task" "$step_cpus_per_task" "$step_cpus_per_task"',
        ]
    lines.extend([
        "", "set -Eeuo pipefail",
        "export SLURM_EXPORT_ENV=ALL",
        *cpu_step_setup,
        *runtime_dir_setup,
        f"mkdir -p {shlex.quote(str(Path(args.results_dir).expanduser() / (args.tao_job_id or args.experiment_id)))}",
        f"export TAO_CHILD_EXIT_FILE={shlex.quote(str(Path(args.results_dir).expanduser() / (args.tao_job_id or args.experiment_id) / 'child_exit_code'))}",
        env_exports, 'export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"',
        f"export MASTER_PORT={args.master_port}",
        "child_rc=0", "set +e", srun, 'child_rc="$?"', "set -e",
        'printf "%s\\n" "$child_rc" > "${TAO_CHILD_EXIT_FILE:?TAO_CHILD_EXIT_FILE must be set}"',
        'if [[ "$child_rc" -ne 0 ]]; then echo "Cosmos child process failed with exit code $child_rc" >&2; fi',
        'exit "$child_rc"', "",
    ])
    script = "\n".join(lines)
    check = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True, check=False)
    if check.returncode:
        raise WorkflowError(f"generated Bash job is invalid: {check.stderr}")
    return script


def initial_metadata(args: argparse.Namespace, plan: Mapping[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": 1, "experiment_id": plan["experiment_id"], "dataset": plan["dataset_family"],
        "training_mode": plan["training"]["training_mode"], "backend": plan["backend"], "tao_job_id": args.tao_job_id,
        "slurm": {
            "job_id": None, "submission_host": socket.gethostname(), "cluster": args.cluster,
            "partition": args.partition, "account": args.account, "qos": args.qos or None,
            "reservation": args.reservation or None, "requested_resources": plan["compute"],
            "allocated_resources": {}, "node_list": [], "master_address": None, "master_port": args.master_port,
            "requeue": args.use_requeue, "exclusive": args.exclusive, "time_limit": args.time_limit, "timeout": args.timeout,
        },
        "image": {"tag": plan["image"]["tag"], "digest": None, "provenance": plan["image"]["provenance_path"], "sqsh_path": args.sqsh_path, "sqsh_sha256": sha256_file(Path(args.sqsh_path)) if Path(args.sqsh_path).is_file() else None},
        "repositories": {
            name: {"commit": commit, "tree": plan["image"]["required_trees"][name], "dirty": False}
            for name, commit in plan["image"]["required_commits"].items()
        }, "config": plan.get("config", {}),
        "paths": plan["paths"], "dataset_fingerprints": {split: value["dataset_fingerprint"] for split, value in plan["datasets"].items()},
        "model": {"identity": plan["model"]["supplied"], "revision": plan["model"]["revision"], "fingerprint": plan["model"]["fingerprint"], "prepared": plan["model"]["prepared_checkpoint"]},
        "launch_command": plan["command"], "environment": selected_environment(plan["environment"]),
        "stdout": args.stdout_path, "stderr": args.stderr_path, "results_dir": args.results_dir,
        "checkpoint_dir": args.checkpoint_dir, "timestamps": {"planned": now, "started": None, "finished": None},
        "scheduler": {"state": "PLANNED", "reason": None, "exit_code": None},
        "child_process": {"exit_code": None}, "terminal_tao_status": "PENDING",
        "metrics": {
            "average_training_loss": None,
            "average_validation_loss": None,
            "average_validation_accuracy": None,
        },
        "artifacts": {
            "status_file": str(Path(args.results_dir).expanduser() / (args.tao_job_id or args.experiment_id) / "status.json"),
            "child_exit_file": str(Path(args.results_dir).expanduser() / (args.tao_job_id or args.experiment_id) / "child_exit_code"),
        },
    }


def parity_report(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    if left.get("backend") == right.get("backend"):
        raise WorkflowError("paired parity requires one Cosmos-RL plan and one Cosmos Framework plan")
    checks = {
        "model": model_parity(left["model"], right["model"]),
        "train_dataset": dataset_parity(left["datasets"]["train"], right["datasets"]["train"]),
        "validation_dataset": dataset_parity(left["datasets"]["validation"], right["datasets"]["validation"]),
        "optimization": optimization_parity(left["training"], right["training"]),
    }
    decoder_keys = (
        "required", "enabled", "path", "manifest", "sha256",
        "input_fingerprints", "policy",
    )
    left_decoder = {
        key: left.get("decoder_artifact", {}).get(key) for key in decoder_keys
    }
    right_decoder = {
        key: right.get("decoder_artifact", {}).get(key) for key in decoder_keys
    }
    if left_decoder["enabled"] and right_decoder["enabled"]:
        decoder_equal = left_decoder == right_decoder
    elif not left_decoder["enabled"] and not right_decoder["enabled"]:
        # A backend-specific hardware compatibility gate may make the
        # preparation artifact mandatory for only one implementation. Before
        # either artifact exists, parity is still determined by the sealed
        # inputs and any explicit forced sources; launch validation remains
        # responsible for blocking a required-but-unprepared backend.
        decoder_equal = (
            left_decoder["input_fingerprints"]
            == right_decoder["input_fingerprints"]
            and left_decoder["policy"].get("forced_runtime_sources", [])
            == right_decoder["policy"].get("forced_runtime_sources", [])
        )
    else:
        decoder_equal = False
    checks["decoder_artifact"] = {
        "status": "equivalent" if decoder_equal else "invalid_mismatch",
        "left": left_decoder,
        "right": right_decoder,
    }
    evaluator_left = left.get("metric_contract", {}).get("accuracy", {})
    evaluator_right = right.get("metric_contract", {}).get("accuracy", {})
    evaluator_equal = evaluator_left == evaluator_right
    checks["evaluator"] = {
        "status": "equivalent" if evaluator_equal else "invalid_mismatch",
        "left": evaluator_left,
        "right": evaluator_right,
    }
    invalid = sorted(name for name, result in checks.items() if result["status"] == "invalid_mismatch")
    return {
        "schema_version": 1,
        "left_backend": left["backend"],
        "right_backend": right["backend"],
        "checks": checks,
        "invalid_mismatches": invalid,
        "launch_allowed": not invalid,
        "backend_syntax_differences": [
            "Framework shard/replica topology versus Cosmos-RL controller/policy topology",
            "Framework DCP versus Cosmos-RL epoch policy checkpoint representation",
        ],
    }


def finalize_metadata(
    metadata: dict[str, Any], *, child_exit_file: Path, status_file: Path,
    scheduler_state: str, scheduler_reason: str | None, scheduler_exit_code: str | None,
    allocated_nodes: Sequence[str] = (), job_id: str | None = None,
) -> dict[str, Any]:
    if not child_exit_file.is_file():
        raise WorkflowError("child-process exit-code file is missing; scheduler completion is not sufficient")
    try:
        child_exit = int(child_exit_file.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise WorkflowError("child-process exit-code file is invalid") from exc
    if not status_file.is_file():
        raise WorkflowError("TAO structured status file is missing")
    status_text = status_file.read_text(encoding="utf-8")
    try:
        status_payload = json.loads(status_text)
        records = status_payload if isinstance(status_payload, list) else status_payload.get("records", [status_payload])
    except json.JSONDecodeError:
        try:
            records = [json.loads(line) for line in status_text.splitlines() if line.strip()]
        except json.JSONDecodeError as exc:
            raise WorkflowError("TAO structured status is neither JSON nor JSONL") from exc
    if not records or not isinstance(records[-1], Mapping):
        raise WorkflowError("TAO structured status contains no terminal record")
    tao_terminal = str(records[-1].get("status", "")).upper()
    metadata["slurm"].update({"job_id": job_id or metadata["slurm"].get("job_id"), "node_list": list(allocated_nodes)})
    metadata["slurm"]["allocated_resources"] = {
        **metadata["slurm"].get("allocated_resources", {}),
        "nodes": len(allocated_nodes) if allocated_nodes else None,
    }
    metadata["scheduler"] = {"state": scheduler_state, "reason": scheduler_reason, "exit_code": scheduler_exit_code}
    metadata["child_process"] = {"exit_code": child_exit}
    metadata["terminal_tao_status"] = tao_terminal
    metadata["timestamps"]["finished"] = datetime.now(timezone.utc).isoformat()
    if child_exit != 0 or scheduler_state.upper() != "COMPLETED" or tao_terminal != "SUCCESS":
        metadata["terminal_tao_status"] = "FAILURE"
    validate_metadata(metadata)
    return metadata


def local_preflight(args: argparse.Namespace, plan: Mapping[str, Any], env: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if env is None else env
    errors: list[str] = []
    warnings: list[str] = []
    decoder_artifact = plan["decoder_artifact"]
    if decoder_artifact["required"] and not decoder_artifact["enabled"]:
        errors.append(
            "the resolved hardware video profile requires video_override_map, "
            "video_override_manifest, and video_override_fingerprint"
        )

    def check_repository(name: str, identity: Mapping[str, Any], commit: str, tree: str) -> None:
        if not identity.get("exists") or identity.get("kind") != "directory":
            errors.append(f"repository is inaccessible: {name}={identity.get('original')}")
            return
        root = str(identity["resolved"])
        head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
        actual_tree = subprocess.run(["git", "-C", root, "rev-parse", "HEAD^{tree}"], text=True, capture_output=True, check=False)
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=all"], text=True, capture_output=True, check=False)
        if head.returncode or actual_tree.returncode or dirty.returncode:
            errors.append(f"repository is not a readable Git checkout: {name}={identity.get('original')}")
        elif head.stdout.strip() != commit:
            errors.append(f"repository commit mismatch for {name}: expected {commit}, found {head.stdout.strip()}")
        elif actual_tree.stdout.strip() != tree:
            errors.append(f"repository tree mismatch for {name}: expected {tree}, found {actual_tree.stdout.strip()}")
        elif dirty.stdout.strip():
            errors.append(f"repository must be clean before image build: {name}")

    # A launch through an existing SQSH consumes packaged code only. Host
    # worktrees are neither mounted nor read by Pyxis, so their current state
    # must not block that user journey. Source commit/tree/cleanliness remains
    # mandatory when the planned SQSH is absent and an image build is needed.
    sqsh_exists = False
    if args.platform == "slurm" and args.sqsh_path.endswith(".sqsh"):
        verified_host = str(plan.get("input_frame", {}).get("verified_host") or "")
        if plan.get("input_frame", {}).get("kind") == "slurm_remote":
            sqsh_exists = bool(verified_host) and _remote_file_exists(
                args, path=args.sqsh_path, host=verified_host,
            )
        else:
            sqsh_exists = Path(args.sqsh_path).expanduser().is_file()

    image = plan["image"]
    repository_identities = {
        ("cosmos-framework" if plan["backend"] == "cosmos-framework" else "cosmos-rl-github"): image["native_repository"],
        "cosmos-rl": image["integration_repository"],
        "nvidia-tao-daft": image["daft_repository"],
        "tao-core": image["tao_core_repository"],
    }
    if args.platform != "slurm" or not sqsh_exists:
        for name, identity in repository_identities.items():
            check_repository(name, identity, image["required_commits"][name], image["required_trees"][name])
    for key, value in plan["paths"].items():
        if key in {"sqsh_cache_dir", "ssh_key_path"} and args.platform != "slurm":
            continue
        if key == "ssh_key_path":
            if not value["exists"]:
                errors.append(f"runtime path is inaccessible on submission host: {key}={value['original']}")
            continue
        if not value["exists"] and not value.get("parent_writable"):
            frame = "target SLURM frame" if args.platform == "slurm" else "submission host"
            errors.append(f"runtime path has no writable parent on {frame}: {key}={value['original']}")
    if args.platform == "slurm":
        for executable in ("ssh",):
            if shutil.which(executable) is None:
                errors.append(f"missing SLURM prerequisite: {executable}")
        if not args.slurm_user or not args.slurm_host:
            errors.append("slurm_user and at least one slurm_host are required")
        if not args.partition or not args.account:
            errors.append("partition and account are required")
        if not args.sqsh_path.endswith(".sqsh"):
            errors.append("sqsh_path must name a .sqsh artifact")
        elif not sqsh_exists:
            errors.append("new SQSH has not been created from the planned image")
        if not args.container_mount:
            errors.append("at least one explicit SLURM container mount is required")
    else:
        if shutil.which("docker") is None:
            errors.append("Docker CLI is missing")
    if plan["model"]["source_type"] == "uri" and not env.get("HF_TOKEN"):
        errors.append("missing credential environment variable HF_TOKEN for model retrieval")
    if plan["image"]["tag"].startswith("nvcr.io/") and not env.get("NGC_KEY"):
        warnings.append("NGC_KEY is unset; it is required only if this image tag must be pushed or pulled")
    if args.gpu_architecture and args.gpu_architecture.casefold() not in {"a100", "h100", "h200", "b200", "gb200"}:
        errors.append(f"unsupported or unvalidated GPU architecture: {args.gpu_architecture}")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "backend": plan["backend"]}


def add_arguments(parser: argparse.ArgumentParser, *, require_inputs: bool) -> None:
    parser.add_argument("--model", default="auto")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--action", choices=sorted(SUPPORTED_ACTIONS), default="train")
    parser.add_argument("--backend", choices=("auto", "cosmos-framework", "cosmos-rl"), default="auto")
    parser.add_argument("--comparative", action="store_true")
    parser.add_argument("--workload", choices=("training", "automl"), default="training")
    parser.add_argument("--dataset-family", choices=("auto", "video_conversation", "task_aware_video_reasoning"), default="auto")
    parser.add_argument("--platform", choices=("docker", "slurm"), default="slurm")
    parser.add_argument("--base-model-path-or-uri", default="")
    parser.add_argument("--base-model-revision", default="")
    parser.add_argument("--base-model-format", choices=("auto", "qwen3_vl", "cosmos3_omni", "cosmos3_edge"), default="auto")
    parser.add_argument("--prepared-checkpoint-path", default="")
    parser.add_argument("--vlm-architecture-model-path-or-uri", default="")
    parser.add_argument("--vlm-architecture-model-revision", default="")
    parser.add_argument("--train-annotation", action="append", default=[])
    parser.add_argument("--train-media-root", action="append", default=[])
    parser.add_argument("--validation-annotation", action="append", default=[])
    parser.add_argument("--validation-media-root", action="append", default=[])
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--training-mode", choices=("dense", "peft"), default="dense")
    parser.add_argument("--lora-rank", type=int, default=0); parser.add_argument("--lora-alpha", type=int, default=0)
    parser.add_argument("--lora-dropout", type=float, default=0.0); parser.add_argument("--lora-target-modules", action="append", default=[])
    parser.add_argument("--lora-bias", choices=("none", "all", "lora_only"), default="none"); parser.add_argument("--lora-use-rslora", action="store_true")
    parser.add_argument("--lora-modules-to-save", action="append", default=[]); parser.add_argument("--lora-precision", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--epochs", type=int, default=1); parser.add_argument("--effective-global-batch", type=int, default=8)
    parser.add_argument(
        "--framework-per-forward-batch",
        type=int,
        default=0,
        help="Framework samples packed per rank and forward; 0 selects the Nano per-rank effective batch and keeps Edge at 1.",
    )
    parser.add_argument(
        "--render-output",
        default="",
        help="Atomically write render-slurm output to this local controller path.",
    )
    parser.add_argument("--rl-mini-batch", type=int, default=1)
    parser.add_argument(
        "--rl-train-batch-per-replica",
        type=int,
        default=0,
        help="Explicit Cosmos-RL train_batch_per_replica; 0 preserves the mini-batch-derived default.",
    )
    parser.add_argument(
        "--rl-video-profile",
        choices=("auto", "system-pyav", "pynv-device-rgbp"),
        default="auto",
        help=(
            "Cosmos-RL video runtime. Auto selects device-RGBP for video "
            "conversation and sparse System-PyAV for task-aware data."
        ),
    )
    parser.add_argument(
        "--rl-video-cache-size",
        type=int,
        default=None,
        help="Rank-local processed-video LRU entries; omit to use the inspected unique-media capacity.",
    )
    parser.add_argument(
        "--rl-video-decoder-cache-size",
        type=int,
        default=None,
        help="Rank-local PyNv native decoder-session entries; omit to use the inspected unique-media capacity.",
    )
    parser.add_argument(
        "--rl-sft-batch-threads",
        type=int,
        default=0,
        help="In-process logical-batch preprocessing threads; 0 selects 4 for the PyNv video fast path and 1 otherwise.",
    )
    parser.add_argument("--rl-dataloader-num-workers", type=int, default=None)
    parser.add_argument("--rl-dataloader-prefetch-factor", type=int, default=None)
    parser.add_argument(
        "--rl-dataset-cache-mode",
        choices=("direct", "prewarm"),
        default="direct",
        help="Process samples on demand (direct) or require deterministic prewarmed dataset caches (prewarm).",
    )
    parser.add_argument("--rl-validation-freq-steps", type=int, default=0)
    parser.add_argument(
        "--framework-video-cache-size",
        type=int,
        default=None,
        help=(
            "Rank-local decoded-frame LRU entries; omit to size the validated "
            "on-demand cache from the inspected unique-media working set."
        ),
    )
    parser.add_argument(
        "--framework-sft-process-threads",
        type=int,
        default=0,
        help="Ordered in-process Framework SFT preprocessing threads; 0 selects 8.",
    )
    parser.add_argument(
        "--framework-video-decoder-threads",
        type=int,
        default=0,
        help="TorchCodec decoder threads per on-demand decode; 0 selects 1.",
    )
    parser.add_argument(
        "--framework-dataloader-num-workers",
        type=int,
        default=None,
        help="Persistent spawned Framework DataLoader workers per rank; omit to select 1.",
    )
    parser.add_argument(
        "--framework-dataloader-prefetch-factor",
        type=int,
        default=None,
        help="Prefetched batches per Framework worker; omit to select 2 and ignore when workers are zero.",
    )
    parser.add_argument("--validation-batch-size", type=int, default=1)
    parser.add_argument("--optimizer", default="AdamW"); parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--optimizer-epsilon", type=float, default=1e-8)
    parser.add_argument("--scheduler", default="linear"); parser.add_argument("--warmup", type=float, default=0)
    parser.add_argument("--minimum-lr-factor", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01); parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--precision", default="bfloat16"); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence-length", type=int, default=0); parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--min-frames", type=int, default=None); parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--video-start", type=float, default=None); parser.add_argument("--video-end", type=float, default=None)
    parser.add_argument("--video-resized-height", type=int, default=None); parser.add_argument("--video-resized-width", type=int, default=None)
    parser.add_argument("--video-min-pixels", type=int, default=None); parser.add_argument("--video-total-pixels", type=int, default=None)
    parser.add_argument("--video-max-pixels", type=int, default=0); parser.add_argument("--video-frame-width", type=int, default=0)
    parser.add_argument("--video-frame-height", type=int, default=0)
    parser.add_argument("--video-override-map", default="")
    parser.add_argument("--video-override-manifest", default="")
    parser.add_argument("--video-override-fingerprint", default="")
    parser.add_argument("--video-override-force-video", action="append", default=[])
    parser.add_argument("--video-override-max-macroblocks", type=int, default=8192)
    parser.add_argument("--video-override-workers", type=int, default=16)
    parser.add_argument("--system-prompt", default=""); parser.add_argument("--attention-implementation", default="auto")
    parser.add_argument("--processor-revision", default="packaged"); parser.add_argument("--run-mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--skip-smoke", action="store_true"); parser.add_argument("--smoke-train-samples", type=int, default=16)
    parser.add_argument("--smoke-validation-samples", type=int, default=8); parser.add_argument("--train-sample-limit", type=int, default=0)
    parser.add_argument("--validation-sample-limit", type=int, default=0); parser.add_argument("--fast-media-fingerprint", action="store_true")
    parser.add_argument(
        "--fast-model-fingerprint",
        action="store_true",
        help=(
            "Hash model metadata/index files but identify weight shards by path and size; "
            "use when the request forbids reading every model byte"
        ),
    )
    parser.add_argument("--async-checkpoint", action="store_true"); parser.add_argument("--max-checkpoints", type=int, default=2)
    parser.add_argument("--results-dir", default=""); parser.add_argument("--checkpoint-dir", default=""); parser.add_argument("--cache-dir", default="")
    parser.add_argument("--sqsh-cache-dir", default=""); parser.add_argument("--ssh-key-path", default="")
    parser.add_argument("--tao-integration-repo", default=""); parser.add_argument("--cosmos-framework-repo", default="")
    parser.add_argument("--cosmos-rl-repo", default=""); parser.add_argument("--daft-repo", default=""); parser.add_argument("--tao-core-repo", default=""); parser.add_argument("--build-context", default="")
    parser.add_argument("--native-context-path", default="cosmos-rl-github"); parser.add_argument("--integration-context-path", default="cosmos-rl")
    parser.add_argument("--daft-context-path", default="nvidia-tao-daft"); parser.add_argument("--tao-core-context-path", default="tao-core")
    parser.add_argument("--image-tag", default=""); parser.add_argument("--sqsh-path", default="")
    parser.add_argument("--cosmos-rl-source-repository", default="")
    parser.add_argument("--cosmos-rl-source-branch", default="")
    parser.add_argument("--cosmos-framework-source-repository", default="")
    parser.add_argument("--cosmos-framework-source-branch", default="")
    parser.add_argument("--cosmos-framework-base-image", default=""); parser.add_argument("--cosmos-rl-base-image", default="")
    parser.add_argument("--cosmos-framework-commit", default=""); parser.add_argument("--cosmos-rl-commit", default="")
    parser.add_argument("--tao-integration-commit", default=""); parser.add_argument("--native-tree", default="")
    parser.add_argument("--daft-commit", default=""); parser.add_argument("--tao-core-commit", default="")
    parser.add_argument("--integration-tree", default=""); parser.add_argument("--daft-tree", default="")
    parser.add_argument("--tao-core-tree", default=""); parser.add_argument("--build-timestamp", default="")
    parser.add_argument("--write-spec", default=""); parser.add_argument("--container-spec-path", default="/specs/train.toml")
    parser.add_argument("--container-results-dir", default="/results")
    parser.add_argument("--container-checkpoint-dir", default="/results/checkpoints"); parser.add_argument("--container-cache-dir", default="/cache")
    parser.add_argument("--nodes", type=int, default=1); parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--cpus-per-task", type=int, default=64); parser.add_argument("--gpu-architecture", default="")
    parser.add_argument("--slurm-user", default=""); parser.add_argument("--slurm-host", action="append", default=[])
    parser.add_argument("--partition", default=""); parser.add_argument("--account", default=""); parser.add_argument("--qos", default="")
    parser.add_argument("--reservation", default=""); parser.add_argument("--time-limit", default="04:00:00"); parser.add_argument("--timeout", default="04:15:00")
    parser.add_argument("--exclusive", action="store_true"); parser.add_argument("--use-requeue", action="store_true")
    parser.add_argument("--exclude-node", action="append", default=[])
    parser.add_argument(
        "--exclude-unhealthy-inventory-nodes",
        action="store_true",
        help=(
            "Exclude nodes whose live scontrol record is DOWN/DRAIN/FAIL/NOT_RESPONDING "
            "or carries a diagnostic/quarantine scheduler comment."
        ),
    )
    parser.add_argument("--slurm-node-inventory-file", default="")
    parser.add_argument("--container-mount", action="append", default=[]); parser.add_argument("--cluster", default="")
    parser.add_argument("--master-port", type=int, default=29500); parser.add_argument("--stdout-path", default="")
    parser.add_argument("--stderr-path", default=""); parser.add_argument("--tao-job-id", default="")
    parser.add_argument("--nccl-debug", default="INFO"); parser.add_argument("--cuda-allocator", default="expandable_segments:True")
    parser.add_argument(
        "--plan-artifact",
        default="",
        help=(
            "Local sealed plan written by the plan verb and reused by preflight, "
            "materialize, and render-slurm without repeating input inspection."
        ),
    )
    parser.add_argument("--format", choices=("json", "text"), default="json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="verb", required=True)
    for verb in ("resolve", "plan", "preflight", "materialize", "render-slurm"):
        child = subs.add_parser(verb); add_arguments(child, require_inputs=verb != "resolve")
    child = subs.add_parser("validate-metadata"); child.add_argument("path", type=Path)
    child = subs.add_parser("verify-provenance"); child.add_argument("--plan", type=Path, required=True); child.add_argument("--provenance", type=Path, required=True)
    child = subs.add_parser("parity"); child.add_argument("left", type=Path); child.add_argument("right", type=Path)
    child = subs.add_parser("finalize-metadata")
    child.add_argument("metadata", type=Path); child.add_argument("--child-exit-file", type=Path, required=True)
    child.add_argument("--status-file", type=Path, required=True); child.add_argument("--scheduler-state", required=True)
    child.add_argument("--scheduler-reason", default=""); child.add_argument("--scheduler-exit-code", default="")
    child.add_argument("--allocated-node", action="append", default=[]); child.add_argument("--job-id", default="")
    args = parser.parse_args(argv)
    if getattr(args, "experiment_id", None) is None:
        args.experiment_id = ""
    if args.verb not in {"validate-metadata", "verify-provenance", "parity", "finalize-metadata"} and not args.experiment_id:
        args.experiment_id = str(uuid.uuid4())
    return args


def _text(data: Mapping[str, Any]) -> str:
    if "ok" in data and "errors" in data:
        return "\n".join([f"Cosmos preflight: {'PASS' if data['ok'] else 'FAIL'}", *(f"- ERROR: {x}" for x in data["errors"]), *(f"- warning: {x}" for x in data["warnings"])])
    if "ok" in data:
        lines = [f"Cosmos materialization: {'PASS' if data['ok'] else 'FAIL'}"]
        config = data.get("config")
        if isinstance(config, Mapping):
            lines.extend(
                f"- config {key}: {config[key]}"
                for key in ("original", "resolved", "sha256")
                if config.get(key)
            )
        return "\n".join(lines)
    return "\n".join(["Cosmos launch plan:", f"- backend: {data['backend']}", f"- reason: {data['backend_selection_reason']}", f"- contract: {data['backend_contract']}"])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.verb == "validate-metadata":
            data = json.loads(args.path.read_text(encoding="utf-8")); validate_metadata(data); result: Any = {"ok": True}
        elif args.verb == "verify-provenance":
            plan = json.loads(args.plan.read_text(encoding="utf-8"))
            provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
            validate_provenance(provenance, plan["image"]["required_commits"], plan["image"]["required_trees"])
            result = {"ok": True, "source_manifest_sha256": provenance.get("source_manifest_sha256")}
        elif args.verb == "parity":
            left = json.loads(args.left.read_text(encoding="utf-8")); right = json.loads(args.right.read_text(encoding="utf-8"))
            result = parity_report(left, right)
            if not result["launch_allowed"]:
                raise WorkflowError(f"paired launch blocked by invalid mismatches: {result['invalid_mismatches']}")
        elif args.verb == "finalize-metadata":
            data = json.loads(args.metadata.read_text(encoding="utf-8"))
            result = finalize_metadata(
                data, child_exit_file=args.child_exit_file, status_file=args.status_file,
                scheduler_state=args.scheduler_state, scheduler_reason=args.scheduler_reason or None,
                scheduler_exit_code=args.scheduler_exit_code or None, allocated_nodes=args.allocated_node,
                job_id=args.job_id or None,
            )
            args.metadata.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        elif args.verb == "resolve":
            args.model = resolve_model_name(args.model, args.base_model_path_or_uri)
            backend, reason = select_backend(model=args.model, action=args.action, backend=args.backend, workload=args.workload, comparative=args.comparative)
            result = {"schema_version": 2, "model": args.model, "backend": backend, "backend_selection_reason": reason, "backend_contract": str(BACKEND_FILES[backend])}
        else:
            if args.plan_artifact and args.verb != "plan":
                args, plan = load_plan_artifact(args, args.plan_artifact)
            else:
                plan = build_plan(args)
            write_spec(args, plan, allow_remote_write=args.verb == "materialize")
            if args.verb == "preflight": result = local_preflight(args, plan)
            elif args.verb == "materialize":
                result = {
                    "ok": True,
                    "config": plan["config"],
                    "generated_artifacts": plan.get("generated_artifacts", []),
                    "approved_plan": plan.get("plan_artifact"),
                }
            elif args.verb == "render-slurm":
                verify_materialized_spec(args, plan)
                script = render_slurm(args, plan)
                if args.render_output:
                    rendered = _atomic_write_text(Path(args.render_output), script)
                    result = {
                        "ok": True,
                        "output": str(rendered),
                        "sha256": sha256_file(rendered),
                        "approved_plan": plan.get("plan_artifact"),
                        "node_exclusions": plan.get("slurm_node_exclusions", {}),
                    }
                else:
                    result = script
            else:
                metadata = initial_metadata(args, plan); validate_metadata(metadata); plan["initial_metadata"] = metadata; result = plan
                if args.plan_artifact:
                    save_plan_artifact(args, plan, args.plan_artifact)
    except (OSError, json.JSONDecodeError, tomllib.TOMLDecodeError, WorkflowError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 2
    if isinstance(result, str):
        print(result, end="")
    else:
        print(json.dumps(result, indent=2, sort_keys=True) if getattr(args, "format", "json") == "json" else _text(result))
    return 1 if isinstance(result, Mapping) and "ok" in result and not result["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
