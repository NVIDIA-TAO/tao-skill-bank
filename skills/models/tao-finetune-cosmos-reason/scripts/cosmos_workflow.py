#!/usr/bin/env python3
"""Build reproducible Cosmos3-Nano TAO plans from runtime-only inputs."""

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
import tomllib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from cosmos_common import (
    WorkflowError,
    assert_no_overlap,
    inspect_dataset,
    inspect_model,
    dataset_parity,
    model_parity,
    optimization_parity,
    path_identity,
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
SUPPORTED_ACTIONS = {"train", "evaluate", "inference", "quantize"}


def resolve_model_name(requested: str, base_model_path_or_uri: str) -> str:
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
    raise WorkflowError("model tier is ambiguous; supply Cosmos3-Nano/Edge or a recognizable public checkpoint")


def resolve_model_profile(args: argparse.Namespace, tier: str) -> dict[str, Any]:
    """Resolve model-aware runtime policy without modifying checkpoint files."""
    defaults = {
        "nano": {"frames": 8, "sequence_length": 40960, "attention_implementation": "cosmos"},
        "edge": {"frames": 6, "sequence_length": 16000, "attention_implementation": "flash_attention_2"},
    }[tier]
    frames = args.frames or defaults["frames"]
    sequence_length = args.sequence_length or defaults["sequence_length"]
    attention = args.attention_implementation if args.attention_implementation != "auto" else defaults["attention_implementation"]
    if frames < 1 or sequence_length < 1:
        raise WorkflowError("frames and sequence_length must be positive")
    if args.video_frame_width < 1 or args.video_frame_height < 1:
        raise WorkflowError("video frame width and height must be positive")
    max_pixels = args.video_max_pixels or (
        frames * args.video_frame_width * args.video_frame_height if tier == "edge" else 0
    )
    if max_pixels < 0:
        raise WorkflowError("video_max_pixels must be nonnegative")
    profile = {
        "model_tier": tier,
        "source": "user" if any((args.frames, args.sequence_length, args.video_max_pixels)) or args.attention_implementation != "auto" else "tao_skill_default",
        "frames": frames,
        "sequence_length": sequence_length,
        "attention_implementation": attention,
        "frame_width": args.video_frame_width,
        "frame_height": args.video_frame_height,
        "max_video_pixels": max_pixels or None,
        "checkpoint_mutation": False,
    }
    args.frames = frames
    args.sequence_length = sequence_length
    args.attention_implementation = attention
    args.video_max_pixels = max_pixels
    return profile


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


def select_backend(*, model: str, action: str, backend: str = "auto", workload: str = "wts", comparative: bool = False) -> tuple[str, str]:
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
            if action != "train":
                raise WorkflowError("Cosmos3-Edge non-train actions require an explicit exported-checkpoint adapter")
            return "cosmos-framework", "Cosmos3-Edge training is native only in Cosmos Framework"
        if action != "train" or workload in {"automl", "hpo"}:
            return "cosmos-rl", "the requested action/schema is native to Cosmos-RL"
        return "cosmos-framework", "plain Cosmos3-Nano SFT defaults to the native Cosmos Framework trainer"
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
        scalars = [(k, v) for k, v in table.items() if not isinstance(v, Mapping) and v is not None]
        children = [(k, v) for k, v in table.items() if isinstance(v, Mapping)]
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
        "scheduler": args.scheduler,
        "warmup": args.warmup,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "precision": args.precision,
        "seed": args.seed,
        "sequence_length": args.sequence_length,
        "frames": args.frames,
        "system_prompt": args.system_prompt,
        "validation_frequency_epochs": 1,
        "checkpoint_frequency_epochs": 1,
        "lora": lora,
    }


def _framework_spec(args: argparse.Namespace, train_count: int, val_count: int, contract: Mapping[str, Any]) -> dict[str, Any]:
    world = args.nodes * args.gpus_per_node
    if args.effective_global_batch % world:
        raise WorkflowError("Framework effective global batch must be divisible by total GPUs")
    grad_accum = args.effective_global_batch // world
    smoke_train = min(train_count, args.smoke_train_samples) if args.run_mode == "smoke" else train_count
    smoke_val = min(val_count, args.smoke_validation_samples) if args.run_mode == "smoke" else val_count
    steps = math.ceil(smoke_train / args.effective_global_batch)
    val_steps = math.ceil(smoke_val / world)
    epochs = contract["epochs"]
    spec: dict[str, Any] = {
        "job": {"task": "vlm", "experiment": ("aetc_daft_vlm_edge" if args.workload == "aetc" else "wts_vlm_edge") if model_tier(args.model) == "edge" else ("aetc_daft_vlm" if args.workload == "aetc" else "wts_vlm"), "project": "cosmos3_reasoner", "group": args.workload, "name": args.experiment_id, "wandb_mode": "disabled"},
        "model": {
            "attn_implementation": args.attention_implementation, "precision": args.precision,
            "backbone": {"model_name": "${oc.env:VLM_SAFETENSORS_PATH}", "safetensors_path": "${oc.env:VLM_SAFETENSORS_PATH}"},
            "ema": {"enabled": False, "rate": 0.1, "iteration_shift": 0},
            "parallelism": {"data_parallel_shard_degree": args.gpus_per_node, "data_parallel_replicate_degree": args.nodes, "context_parallel_shard_degree": 1, "cfg_parallel_shard_degree": 1},
            "compile": {"enabled": False, "compile_dynamic": True},
            "activation_checkpointing": {"mode": "full", "save_ops_regex": ["fmha"], "preserve_rng_state": True, "determinism_check": "default"},
        },
        "optimizer": {"betas": [0.9, 0.999], "eps": 1e-8, "fused": True, "lr": args.learning_rate, "weight_decay": args.weight_decay, "keys_to_select": [], "keys_to_exclude": []},
        "scheduler": {"cycle_lengths": [steps * epochs], "f_max": [1.0], "f_min": [0.0], "f_start": [1.0], "verbosity_interval": 0, "warm_up_steps": [args.warmup]},
        "trainer": {
            "distributed_parallelism": "fsdp", "grad_accum_iter": grad_accum, "logging_iter": 1,
            "max_iter": steps * epochs, "num_epochs": epochs, "steps_per_epoch": steps,
            "max_val_iter": val_steps, "run_validation": True, "validation_iter": steps,
            "validation_freq_in_epoch": 1, "run_validation_on_start": False,
            "callbacks": {"compile_tokenizer": {"compile_after_iterations": 3, "enabled": False}, "grad_clip": {"clip_norm": args.gradient_clip, "force_finite": False}, "tao": {"enabled": True, "experiment_name": args.experiment_id, "logging_interval": 1, "validation_heartbeat_interval": 50}},
        },
        "checkpoint": {"keys_to_skip_loading": [], "load_path": "???", "save_iter": steps, "save_freq_in_epoch": 1, "dcp_async_mode_enabled": args.nodes == 1 and args.async_checkpoint},
        "dataloader_train": {"max_samples_per_batch": 1, "max_sequence_length": args.sequence_length},
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


def _rl_spec(args: argparse.Namespace, contract: Mapping[str, Any], prepared_model: str, train_annotations: Sequence[str], train_media: Sequence[str], val_annotations: Sequence[str], val_media: Sequence[str], cache_keys: Mapping[str, str]) -> dict[str, Any]:
    if len(train_media) != 1 or len(val_media) != 1:
        raise WorkflowError("Cosmos-RL requires one explicit shared media root per split when annotations are merged")
    train_manifest = train_annotations[0] if len(train_annotations) == 1 else "__TAO_TRAIN_MERGED_MANIFEST__"
    val_manifest = val_annotations[0] if len(val_annotations) == 1 else "__TAO_VALIDATION_MERGED_MANIFEST__"
    spec = load_yaml(REFERENCES / "spec_template_train.yaml")
    spec["train"].update({
        "resume": False, "epoch": contract["epochs"], "compile": False,
        "train_batch_per_replica": args.effective_global_batch, "output_dir": args.container_checkpoint_dir,
        "optm_lr": args.learning_rate, "optm_impl": "foreach", "optm_weight_decay": args.weight_decay,
        "optm_warmup_epochs": args.warmup, "optm_decay_type": args.scheduler,
        "optm_grad_norm_clip": args.gradient_clip, "param_dtype": args.precision,
    })
    spec["train"]["ckpt"].update({"enable_checkpoint": True, "save_freq_in_epoch": 1, "save_mode": "async" if args.async_checkpoint else "sync", "max_keep": args.max_checkpoints})
    spec["train"]["train_policy"].update({
        "type": "sft", "mini_batch": args.rl_mini_batch, "dataloader_num_workers": 0,
        "conversation_column_name": "conversations", "enable_dataset_cache": True,
        "dataloader_shuffle": True, "dataloader_seed": args.seed,
        "dataset_cache_dir": args.container_cache_dir,
        "dataset_cache_fingerprint": cache_keys["train"],
        "validation_dataset_cache_fingerprint": cache_keys["validation"],
        "require_complete_dataset_cache": True,
    })
    spec["train"]["train_policy"].pop("dataloader_prefetch_factor", None)
    spec["validation"].update({"enable": True, "freq_in_epoch": 1, "batch_size": args.validation_batch_size, "dataloader_num_workers": 0, "enable_dataset_cache": True})
    spec["validation"].pop("dataloader_prefetch_factor", None)
    spec["policy"].update({
        "model_name_or_path": prepared_model, "model_max_length": args.sequence_length,
        "model_gradient_checkpointing": True,
    })
    spec["policy"]["parallelism"].update({"dp_shard_size": args.nodes * args.gpus_per_node, "dp_replicate_size": 1, "pp_size": 1, "tp_size": 1})
    if args.training_mode == "peft":
        lora = contract["lora"]
        spec["policy"]["lora"] = {
            "dim": lora["rank"], "alpha": lora["alpha"], "dropout": lora["dropout"],
            "target_modules": lora["target_modules"], "bias": lora["bias"],
            "use_rslora": lora["use_rslora"], "modules_to_save": lora["modules_to_save"],
            "adapter_dtype": lora["precision"],
        }
    else:
        spec["policy"].pop("lora", None)
    spec["logging"].update({"logger": ["console", "tao"], "experiment_name": args.experiment_id, "project_name": "cosmos-rl-tao"})
    spec["custom"].update({
        "train_dataset": {"annotation_path": train_manifest, "media_path": train_media[0], "media_root": train_media[0], "response_mode": "hybrid" if args.workload == "aetc" else "answer"},
        "val_dataset": {"annotation_path": val_manifest, "media_path": val_media[0], "media_root": val_media[0], "response_mode": "answer"},
        "vision": {"nframes": args.frames, "video_decoder": "pynvvideocodec", "cache_dir": args.container_cache_dir},
        "system_prompt": args.system_prompt,
    })
    return spec


def _env(args: argparse.Namespace, backend: str, prepared_model: str, train_annotations: Sequence[str], train_media: Sequence[str], val_annotations: Sequence[str], val_media: Sequence[str]) -> dict[str, str]:
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
    if backend == "cosmos-framework":
        common.update({
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "VLM_SAFETENSORS_PATH": prepared_model,
            "IMAGINAIRE_OUTPUT_ROOT": args.container_results_dir,
            "WTS_TRAIN_ANNOTATION": train_annotations[0] if args.workload == "wts" else "",
            "WTS_TRAIN_MEDIA": train_media[0] if args.workload == "wts" else "",
            "WTS_VAL_ANNOTATION": val_annotations[0] if args.workload == "wts" else "",
            "WTS_VAL_MEDIA": val_media[0] if args.workload == "wts" else "",
            "WTS_NUM_VIDEO_FRAMES": str(args.frames), "WTS_SYSTEM_PROMPT": args.system_prompt,
            "AETC_TRAIN_ANNOTATIONS": json.dumps(list(train_annotations)) if args.workload == "aetc" else "[]",
            "AETC_TRAIN_MEDIA": train_media[0] if args.workload == "aetc" else "",
            "AETC_VAL_ANNOTATIONS": json.dumps(list(val_annotations)) if args.workload == "aetc" else "[]",
            "AETC_VAL_MEDIA": val_media[0] if args.workload == "aetc" else "",
            "AETC_NUM_VIDEO_FRAMES": str(args.frames), "AETC_SYSTEM_PROMPT": args.system_prompt,
        })
        if args.video_max_pixels:
            common["WTS_VIDEO_MAX_PIXELS"] = str(args.video_max_pixels)
            common["AETC_VIDEO_MAX_PIXELS"] = str(args.video_max_pixels)
        if args.run_mode == "smoke":
            common.update({"WTS_TRAIN_LIMIT": str(args.smoke_train_samples), "WTS_VAL_LIMIT": str(args.smoke_validation_samples), "AETC_TRAIN_LIMIT": str(args.smoke_train_samples), "AETC_VAL_LIMIT": str(args.smoke_validation_samples)})
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
    hook = "/opt/cosmos_rl/tao_vl_reason_daft_sft_example.py" if args.workload == "aetc" else "/opt/cosmos_rl/tao_sft_example.py"
    if args.nodes == 1:
        return f"cosmos-rl --config {shlex.quote(args.container_spec_path)} {hook}"
    return "\n".join([
        'export COSMOS_CONTROLLER_HOST="$MASTER_ADDR:18082"', 'controller_pid=""',
        'if [[ "${SLURM_PROCID:-0}" == "0" ]]; then',
        f"  launch_controller.sh --port 18082 --config {shlex.quote(args.container_spec_path)} --script {hook} &",
        '  controller_pid="$!"', "fi", "sleep 10", "set +e",
        f"launch_replica.sh --type policy --ngpus {args.gpus_per_node} --nnodes {args.nodes} --rdzv-endpoint \"$MASTER_ADDR:$MASTER_PORT\" --config {shlex.quote(args.container_spec_path)} --script {hook}",
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
    dockerfile = "Dockerfile" if backend == "cosmos-framework" else "Dockerfile.cosmos_rl"
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
        if not args.cosmos_framework_base_tag:
            raise WorkflowError("cosmos_framework_base_tag is required for the clean two-stage Framework build")
        native_build_args = {
            "SOURCE_COMMIT": commits[native_name], "SOURCE_TREE": args.native_tree,
            "SOURCE_DIRTY": "0", "BUILD_TIMESTAMP": args.build_timestamp,
        }
        native_command = ["docker", "build", "--pull", "-f", str(Path(native_repo["expanded"]) / "Dockerfile"), "-t", args.cosmos_framework_base_tag]
        for key, value in native_build_args.items():
            native_command.extend(["--build-arg", f"{key}={value}"])
        native_command.append(native_repo["expanded"])
        build_args = {
            "COSMOS_FRAMEWORK_BASE_IMAGE": args.cosmos_framework_base_tag,
            "ACTIONS_COMMIT": commits["cosmos-rl"], "ACTIONS_TREE": args.integration_tree,
            "DAFT_COMMIT": commits["nvidia-tao-daft"], "DAFT_TREE": args.daft_tree,
            "TAO_CORE_COMMIT": commits["tao-core"], "TAO_CORE_TREE": args.tao_core_tree,
            "EXPECTED_FRAMEWORK_COMMIT": commits[native_name], "SOURCE_DIRTY": "0",
            "BUILD_TIMESTAMP": args.build_timestamp,
            "LOCAL_COSMOS_ACTIONS_PATH": args.integration_context_path,
            "LOCAL_TAO_DAFT_PATH": args.daft_context_path,
            "LOCAL_TAO_CORE_PATH": args.tao_core_context_path,
        }
        commands = [shlex.join(native_command)]
    else:
        if not args.cosmos_rl_base_image:
            raise WorkflowError("cosmos_rl_base_image is required for the clean Cosmos-RL build")
        build_args = {
            "COSMOS_RL_BASE_IMAGE": args.cosmos_rl_base_image,
            "COSMOS_RL_COMMIT": commits[native_name], "COSMOS_RL_TREE": args.native_tree,
            "ACTIONS_COMMIT": commits["cosmos-rl"], "ACTIONS_TREE": args.integration_tree,
            "DAFT_COMMIT": commits["nvidia-tao-daft"], "DAFT_TREE": args.daft_tree,
            "TAO_CORE_COMMIT": commits["tao-core"], "TAO_CORE_TREE": args.tao_core_tree,
            "SOURCE_DIRTY": "0", "BUILD_TIMESTAMP": args.build_timestamp,
            "LOCAL_COSMOS_RL_PATH": args.native_context_path,
            "LOCAL_COSMOS_ACTIONS_PATH": args.integration_context_path,
            "LOCAL_TAO_DAFT_PATH": args.daft_context_path,
            "LOCAL_TAO_CORE_PATH": args.tao_core_context_path,
        }
        commands = []
    command = ["docker", "build", "--pull", "-f", str(Path(integration["expanded"]) / dockerfile), "-t", image]
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
            shlex.quote(args.cosmos_framework_base_tag), "-c",
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
        "--framework-image", args.cosmos_framework_base_tag,
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


def _preflight_contract(args: argparse.Namespace, backend: str, plan_image: Mapping[str, Any], prepared_model: str, representative_media: str) -> dict[str, Any]:
    python = "/workspace/.venv/bin/python" if backend == "cosmos-framework" else "/opt/venv/cosmos_rl/bin/python"
    imports = ["import torch", "assert torch.cuda.is_available()", f"assert torch.cuda.device_count() == {args.gpus_per_node}"]
    if backend == "cosmos-framework":
        imports.extend([
            "import cosmos_framework", "from cosmos_framework.callbacks.tao_status import TAOStatusCallback",
            "from cosmos_framework.scripts.export_vlm_dcp import export_vlm_dcp",
            "import torchcodec",
            f"from torchcodec.decoders import VideoDecoder; d=VideoDecoder({representative_media!r}, device='cuda'); assert len(d)>0",
        ])
    else:
        imports.extend([
            "import cosmos_rl", "import PyNvVideoCodec", "import ctypes", "ctypes.CDLL('libnvcuvid.so.1')",
            "from cosmos_rl.utils.pynv_video_reader import register_pynv_video_reader",
            f"d=PyNvVideoCodec.SimpleDecoder({representative_media!r}, gpu_id=0, use_device_memory=False); assert len(d)>0",
        ])
    if args.workload == "aetc":
        imports.append("import nvidia_tao_daft")
    imports.extend([
        "p=torch.cuda.get_device_properties(0)",
        "assert p.total_memory >= 30 * 1024**3, p.total_memory",
        "import tempfile; f=tempfile.NamedTemporaryFile(delete=False); f.close(); torch.distributed.init_process_group('nccl', init_method='file://'+f.name, rank=0, world_size=1); torch.distributed.destroy_process_group()",
        "print({'gpu': p.name, 'capability': (p.major,p.minor), 'memory':p.total_memory, 'torch':torch.__version__, 'cuda':torch.version.cuda})",
    ])
    container_check = f"{python} -c {shlex.quote('; '.join(imports))}"
    path_values = [prepared_model, args.results_dir, args.checkpoint_dir, args.cache_dir, *args.train_annotation, *args.train_media_root, *args.validation_annotation, *args.validation_media_root]
    path_checks = " && ".join(f"test -r {shlex.quote(value)}" for value in path_values)
    host = "command -v docker >/dev/null && docker version >/dev/null"
    if args.platform == "slurm":
        host = "command -v ssh >/dev/null && command -v sbatch >/dev/null && command -v srun >/dev/null"
        allocation = " && ".join([
            "command -v enroot >/dev/null", "srun --help 2>&1 | grep -q -- --container-image",
            f"test -r {shlex.quote(args.sqsh_path)}", path_checks,
            f"df -Pk {shlex.quote(args.results_dir)} {shlex.quote(args.checkpoint_dir)}",
            "nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader",
        ])
        container = " ".join([
            "srun", "--nodes=1", "--ntasks=1", f"--gpus={args.gpus_per_node}",
            f"--container-image={shlex.quote(args.sqsh_path)}", "bash -lc", shlex.quote(container_check),
        ])
    else:
        allocation = path_checks
        container = f"docker run --rm --gpus all {shlex.quote(plan_image['tag'])} bash -lc {shlex.quote(container_check)}"
    return {
        "submission_host": host,
        "target_compute_node": allocation,
        "container_runtime": container,
        "checks": [
            "host and scheduler tools", "credential presence without reading values", "repository clean state",
            "Pyxis/Enroot and SQSH readability", "container mounts/shared storage", "non-root Python imports",
            "GPU count/type/memory", "driver/CUDA/PyTorch", "NCCL initialization", "video decoder/libnvcuvid",
            "free result/checkpoint space",
        ],
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    args.model = resolve_model_name(args.model, args.base_model_path_or_uri)
    backend, reason = select_backend(model=args.model, action=args.action, backend=args.backend, workload=args.workload, comparative=args.comparative)
    if args.action != "train":
        raise WorkflowError("this planner currently materializes training; use the backend action contract for non-train actions")
    tier = model_tier(args.model)
    if tier == "edge" and backend != "cosmos-framework":
        raise WorkflowError("Cosmos3-Edge training requires Cosmos Framework")
    model_profile = resolve_model_profile(args, tier)
    if args.run_mode == "full" and (args.train_sample_limit or args.validation_sample_limit):
        raise WorkflowError("full runs must not contain a smoke/subset sample limit")
    if args.async_checkpoint and args.nodes > 1:
        raise WorkflowError("asynchronous distributed checkpointing is disabled for multi-node Cosmos runs")
    if not args.results_dir or not args.checkpoint_dir or not args.cache_dir:
        raise WorkflowError("results_dir, checkpoint_dir, and cache_dir are required runtime paths")
    model = inspect_model(args.base_model_path_or_uri, args.base_model_revision, args.prepared_checkpoint_path)
    prepared_model, model_preparation = _model_preparation(args, model)
    train_annotations, train_media = _annotation_args(args, "train")
    val_annotations, val_media = _annotation_args(args, "validation")
    train_data = inspect_dataset(workload=args.workload, annotations=train_annotations, media_roots=train_media, selected_tasks=args.aetc_task, verify_media_content=not args.fast_media_fingerprint)
    val_data = inspect_dataset(workload=args.workload, annotations=val_annotations, media_roots=val_media, selected_tasks=args.aetc_task, verify_media_content=not args.fast_media_fingerprint)
    assert_no_overlap(train_data, val_data)
    total_gpus = args.nodes * args.gpus_per_node
    if min(train_data["record_count"], val_data["record_count"]) < total_gpus:
        raise WorkflowError("train and validation datasets must each contain at least one record per global GPU")
    contract = _training_contract(args)
    commits = _source_commits(args, backend)
    image = _image_plan(args, backend, commits)
    processor_fingerprint = stable_hash({"revision": args.processor_revision, "profile": model_profile})
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
    spec = _framework_spec(args, train_data["record_count"], val_data["record_count"], contract) if backend == "cosmos-framework" else _rl_spec(args, contract, prepared_model_container, train_annotations_container, train_media_container, val_annotations_container, val_media_container, cache_keys)
    environment = _env(args, backend, prepared_model_container, train_annotations_container, train_media_container, val_annotations_container, val_media_container)
    plan = {
        "schema_version": 2, "experiment_id": args.experiment_id, "model_name": args.model,
        "model": model, "action": args.action, "workload": args.workload, "backend": backend,
        "model_preparation": model_preparation, "prepared_model_container_path": prepared_model_container,
        "backend_selection_reason": reason, "backend_contract": str(BACKEND_FILES[backend]),
        "run_mode": args.run_mode, "training": contract, "processor_profile": model_profile,
        "datasets": {"train": train_data, "validation": val_data},
        "paths": {
            "results_dir": path_identity(args.results_dir), "checkpoint_dir": path_identity(args.checkpoint_dir),
            "cache_dir": path_identity(args.cache_dir), "sqsh_cache_dir": path_identity(args.sqsh_cache_dir, required=args.platform == "slurm"),
            "ssh_key_path": path_identity(args.ssh_key_path, required=args.platform == "slurm"),
        },
        "image": image, "sqsh": path_identity(args.sqsh_path, required=args.platform == "slurm"),
        "compute": {"platform": args.platform, "nodes": args.nodes, "gpus_per_node": args.gpus_per_node, "total_gpus": total_gpus, "cpus_per_task": args.cpus_per_task},
        "cache_prewarm": {"required": backend == "cosmos-rl", "keys": cache_keys, "path": args.cache_dir, "dataset_fingerprints": {"train": train_data["dataset_fingerprint"], "validation": val_data["dataset_fingerprint"]}, "model_fingerprint": model["fingerprint"], "processor_fingerprint": processor_fingerprint, "completeness_required": True, "resumable": True},
        "spec": spec, "environment": environment, "command": _command(args, backend),
        "config_container_path": args.container_spec_path,
        "smoke_gate": {"required": not args.skip_smoke and args.run_mode == "full", "train_samples": args.smoke_train_samples, "validation_samples": args.smoke_validation_samples, "criteria": ["child_exit_code=0", "terminal_status=SUCCESS", "finite_train_avg_loss", "finite_val_avg_loss", "checkpoint_event", "validation_accuracy_present"]},
        "metric_contract": {"train": {"key": "train/avg_loss", "weight": "valid_labels", "requires": ["train/loss_numerator", "train/valid_label_count"]}, "validation": {"key": "val/avg_loss", "weight": "valid_labels", "requires": ["val/loss_numerator", "val/valid_label_count"]}, "accuracy": {"route": "shared repository evaluator", "aggregation": val_data["metric_coverage"]["aggregate"], "coverage": val_data["metric_coverage"]}},
    }
    representative_media = _containerize(args, train_data["media_manifest"][0]["path"])
    plan["preflight"] = _preflight_contract(args, backend, image, prepared_model, representative_media)
    return plan


def write_spec(args: argparse.Namespace, plan: dict[str, Any]) -> Path:
    if not args.write_spec:
        raise WorkflowError("write_spec is required so the submitted config can be fingerprinted")
    output = Path(args.write_spec).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    spec = copy.deepcopy(plan["spec"])

    def materialize(split: str, marker: str, limit: int = 0) -> str:
        records: list[dict[str, Any]] = []
        for entry in plan["datasets"][split]["annotation_manifest"]:
            payload = json.loads(Path(entry["resolved"]).read_text(encoding="utf-8"))
            metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
            items = payload.get("items", []) if isinstance(payload, dict) else payload
            for item in items:
                copied = dict(item)
                if args.workload == "aetc" and not copied.get("task"):
                    copied["task"] = metadata.get("task")
                records.append(copied)
        if limit:
            records = records[:limit]
        target = output.with_name(f"{split}_{'smoke' if limit else 'merged'}.json")
        if args.workload == "aetc":
            payload: Any = {"format": "tao-vl-reason-v1.0", "metadata": {"task": "mixed"}, "items": records}
        else:
            payload = records
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return _containerize(args, str(target.resolve()))

    if plan["backend"] == "cosmos-rl":
        for split, marker, key in (
            ("train", "__TAO_TRAIN_MERGED_MANIFEST__", "train_dataset"),
            ("validation", "__TAO_VALIDATION_MERGED_MANIFEST__", "val_dataset"),
        ):
            current = spec["custom"][key]["annotation_path"]
            smoke_limit = (args.smoke_train_samples if split == "train" else args.smoke_validation_samples) if args.run_mode == "smoke" else 0
            if current == marker or smoke_limit:
                spec["custom"][key]["annotation_path"] = materialize(split, marker, smoke_limit)
    output.write_text(dump_toml(spec), encoding="utf-8")
    with output.open("rb") as stream:
        tomllib.load(stream)
    plan["config"] = {
        "original": args.write_spec,
        "resolved": str(output.resolve()),
        "container": args.container_spec_path,
        "sha256": sha256_file(output),
    }
    plan["spec"] = spec
    return output


def render_slurm(args: argparse.Namespace, plan: Mapping[str, Any]) -> str:
    if args.platform != "slurm":
        raise WorkflowError("SLURM script rendering requires platform=slurm")
    if not args.partition or not args.account or not args.sqsh_path:
        raise WorkflowError("SLURM partition, account, and SQSH path are required")
    if args.use_requeue:
        raise WorkflowError("requeue is disabled by default and is not validated for Cosmos training")
    sqsh = Path(args.sqsh_path)
    if not args.container_mount:
        raise WorkflowError("at least one explicit container mount is required for SLURM")
    mount_args = " ".join(f"--container-mounts={shlex.quote(value)}" for value in args.container_mount)
    env_exports = "\n".join(f"export {key}={shlex.quote(value)}" for key, value in plan["environment"].items())
    native = plan["command"]
    wrapped = "\n".join(["ulimit -n 65536", "ulimit -s unlimited", "ulimit -l unlimited 2>/dev/null || true", native])
    srun = " ".join(filter(None, [
        "srun", f"--nodes={args.nodes}", f"--ntasks={args.nodes}", "--ntasks-per-node=1",
        f"--gpus-per-node={args.gpus_per_node}", f"--cpus-per-task={args.cpus_per_task}",
        f"--container-image={shlex.quote(str(sqsh))}", mount_args, "bash -lc", shlex.quote(wrapped),
    ]))
    lines = [
        "#!/usr/bin/env bash", "set -Eeuo pipefail", f"#SBATCH --partition={args.partition}",
        f"#SBATCH --account={args.account}", f"#SBATCH --nodes={args.nodes}", f"#SBATCH --ntasks={args.nodes}",
        "#SBATCH --ntasks-per-node=1", f"#SBATCH --gpus-per-node={args.gpus_per_node}",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}", f"#SBATCH --time={args.time_limit}", "#SBATCH --no-requeue",
        f"#SBATCH --output={args.stdout_path}", f"#SBATCH --error={args.stderr_path}",
    ]
    if args.qos:
        lines.append(f"#SBATCH --qos={args.qos}")
    if args.reservation:
        lines.append(f"#SBATCH --reservation={args.reservation}")
    if args.exclusive:
        lines.append("#SBATCH --exclusive")
    lines.extend([
        "", f"mkdir -p {shlex.quote(str(Path(args.results_dir).expanduser() / (args.tao_job_id or args.experiment_id)))}",
        f"export TAO_CHILD_EXIT_FILE={shlex.quote(str(Path(args.results_dir).expanduser() / (args.tao_job_id or args.experiment_id) / 'child_exit_code'))}",
        env_exports, 'export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"',
        f"export MASTER_PORT={args.master_port}", "child_rc=0", "set +e", srun, 'child_rc="$?"', "set -e",
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
        "schema_version": 1, "experiment_id": plan["experiment_id"], "dataset": plan["workload"],
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
    status_payload = json.loads(status_file.read_text(encoding="utf-8"))
    records = status_payload if isinstance(status_payload, list) else status_payload.get("records", [status_payload])
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

    image = plan["image"]
    repository_identities = {
        ("cosmos-framework" if plan["backend"] == "cosmos-framework" else "cosmos-rl-github"): image["native_repository"],
        "cosmos-rl": image["integration_repository"],
        "nvidia-tao-daft": image["daft_repository"],
        "tao-core": image["tao_core_repository"],
    }
    for name, identity in repository_identities.items():
        check_repository(name, identity, image["required_commits"][name], image["required_trees"][name])
    for key, value in plan["paths"].items():
        if key in {"sqsh_cache_dir", "ssh_key_path"} and args.platform != "slurm":
            continue
        if not value["exists"]:
            errors.append(f"runtime path is inaccessible on submission host: {key}={value['original']}")
    if args.platform == "slurm":
        for executable in ("ssh", "sbatch", "srun"):
            if shutil.which(executable) is None:
                errors.append(f"missing SLURM prerequisite: {executable}")
        if not args.slurm_user or not args.slurm_host:
            errors.append("slurm_user and at least one slurm_host are required")
        if not args.partition or not args.account:
            errors.append("partition and account are required")
        if not args.sqsh_path.endswith(".sqsh"):
            errors.append("sqsh_path must name a .sqsh artifact")
        elif not Path(args.sqsh_path).is_file():
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
    parser.add_argument("--workload", choices=("wts", "aetc", "automl"), default="wts")
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
    parser.add_argument("--aetc-task", action="append", default=[])
    parser.add_argument("--training-mode", choices=("dense", "peft"), default="dense")
    parser.add_argument("--lora-rank", type=int, default=0); parser.add_argument("--lora-alpha", type=int, default=0)
    parser.add_argument("--lora-dropout", type=float, default=0.0); parser.add_argument("--lora-target-modules", action="append", default=[])
    parser.add_argument("--lora-bias", choices=("none", "all", "lora_only"), default="none"); parser.add_argument("--lora-use-rslora", action="store_true")
    parser.add_argument("--lora-modules-to-save", action="append", default=[]); parser.add_argument("--lora-precision", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--epochs", type=int, default=1); parser.add_argument("--effective-global-batch", type=int, default=8)
    parser.add_argument("--rl-mini-batch", type=int, default=1); parser.add_argument("--validation-batch-size", type=int, default=1)
    parser.add_argument("--optimizer", default="AdamW"); parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--scheduler", default="linear"); parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.01); parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--precision", default="bfloat16"); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence-length", type=int, default=0); parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--video-max-pixels", type=int, default=0); parser.add_argument("--video-frame-width", type=int, default=1280)
    parser.add_argument("--video-frame-height", type=int, default=720)
    parser.add_argument("--system-prompt", default=""); parser.add_argument("--attention-implementation", default="auto")
    parser.add_argument("--processor-revision", default="packaged"); parser.add_argument("--run-mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--skip-smoke", action="store_true"); parser.add_argument("--smoke-train-samples", type=int, default=16)
    parser.add_argument("--smoke-validation-samples", type=int, default=8); parser.add_argument("--train-sample-limit", type=int, default=0)
    parser.add_argument("--validation-sample-limit", type=int, default=0); parser.add_argument("--fast-media-fingerprint", action="store_true")
    parser.add_argument("--async-checkpoint", action="store_true"); parser.add_argument("--max-checkpoints", type=int, default=2)
    parser.add_argument("--results-dir", default=""); parser.add_argument("--checkpoint-dir", default=""); parser.add_argument("--cache-dir", default="")
    parser.add_argument("--sqsh-cache-dir", default=""); parser.add_argument("--ssh-key-path", default="")
    parser.add_argument("--tao-integration-repo", default=""); parser.add_argument("--cosmos-framework-repo", default="")
    parser.add_argument("--cosmos-rl-repo", default=""); parser.add_argument("--daft-repo", default=""); parser.add_argument("--tao-core-repo", default=""); parser.add_argument("--build-context", default="")
    parser.add_argument("--native-context-path", default="cosmos-rl-github"); parser.add_argument("--integration-context-path", default="cosmos-rl")
    parser.add_argument("--daft-context-path", default="nvidia-tao-daft"); parser.add_argument("--tao-core-context-path", default="tao-core")
    parser.add_argument("--image-tag", default=""); parser.add_argument("--sqsh-path", default="")
    parser.add_argument("--cosmos-framework-base-tag", default=""); parser.add_argument("--cosmos-rl-base-image", default="")
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
    parser.add_argument("--container-mount", action="append", default=[]); parser.add_argument("--cluster", default="")
    parser.add_argument("--master-port", type=int, default=29500); parser.add_argument("--stdout-path", default="")
    parser.add_argument("--stderr-path", default=""); parser.add_argument("--tao-job-id", default="")
    parser.add_argument("--nccl-debug", default="INFO"); parser.add_argument("--cuda-allocator", default="expandable_segments:True")
    parser.add_argument("--format", choices=("json", "text"), default="json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="verb", required=True)
    for verb in ("resolve", "plan", "preflight", "render-slurm"):
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
    if "ok" in data:
        return "\n".join([f"Cosmos preflight: {'PASS' if data['ok'] else 'FAIL'}", *(f"- ERROR: {x}" for x in data["errors"]), *(f"- warning: {x}" for x in data["warnings"])])
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
            plan = build_plan(args); write_spec(args, plan)
            if args.verb == "preflight": result = local_preflight(args, plan)
            elif args.verb == "render-slurm": result = render_slurm(args, plan)
            else:
                metadata = initial_metadata(args, plan); validate_metadata(metadata); plan["initial_metadata"] = metadata; result = plan
    except (OSError, json.JSONDecodeError, tomllib.TOMLDecodeError, WorkflowError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 2
    if isinstance(result, str):
        print(result, end="")
    else:
        print(json.dumps(result, indent=2, sort_keys=True) if getattr(args, "format", "json") == "json" else _text(result))
    return 1 if isinstance(result, Mapping) and "ok" in result and not result["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
