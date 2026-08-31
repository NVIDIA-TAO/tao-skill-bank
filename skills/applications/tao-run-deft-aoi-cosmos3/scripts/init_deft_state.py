#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Initialize the disk-backed Cosmos3 DEFT AOI run state."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile
from typing import Any

import yaml

from gap_analysis.config import load_profile, validate_config
from metric_contract import render_target, validate_contract
from nvpaw_annotations import TASK_SPECS
from render_report import render as render_html_report
from task_mining_router import MINING_ROUTER_MODES


WORKFLOW = "tao-run-deft-aoi-cosmos3"
STAGES = [
    "train",
    "evaluate_benchmark",
    "benchmark_metrics",
    "evaluate_proxy",
    "proxy_rcca",
    "routing",
    "data_mining",
    "assemble_data",
    "validate_data",
    "loop_stop",
]
STATUSES = ["pending", "in_progress", "complete", "failed"]


def validate_base_model(value: str) -> str:
    candidate = pathlib.Path(value).expanduser()
    if not candidate.exists():
        raise ValueError(
            "DEFT AOI requires a complete local Qwen3-VL snapshot; "
            f"base model path does not exist: {candidate}"
        )
    resolved = candidate.resolve()
    config = resolved / "config.json"
    required_metadata = (
        config,
        resolved / "preprocessor_config.json",
        resolved / "tokenizer_config.json",
        resolved / "tokenizer.json",
    )
    if not resolved.is_dir() or any(
        not path.is_file() or path.stat().st_size == 0 for path in required_metadata
    ):
        raise ValueError(
            "local --base-model must contain non-empty config, tokenizer, and processor files"
        )
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid local base-model config: {config}") from exc
    if payload.get("model_type") != "qwen3_vl":
        raise ValueError("local --base-model config.model_type must be qwen3_vl")
    index = resolved / "model.safetensors.index.json"
    if index.is_file():
        try:
            index_payload = json.loads(index.read_text(encoding="utf-8"))
            weight_map = index_payload["weight_map"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid local base-model safetensors index: {index}") from exc
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"local base-model safetensors index is empty: {index}")
        raw_shard_names = list(weight_map.values())
        if not all(isinstance(name, str) and name for name in raw_shard_names):
            raise ValueError(f"local base-model safetensors index has invalid shard names: {index}")
        shard_names = set(raw_shard_names)
        missing = [
            name
            for name in sorted(shard_names)
            if not (resolved / name).is_file() or (resolved / name).stat().st_size == 0
        ]
        if missing:
            raise ValueError(f"local base-model safetensors shards are missing/empty: {missing}")
    else:
        shards = list(resolved.glob("*.safetensors"))
        if not shards or any(path.stat().st_size == 0 for path in shards):
            raise ValueError(
                "local --base-model must be a complete Hugging Face safetensors directory"
            )
    return str(resolved)


def _absolute_executable(path: pathlib.Path | str) -> pathlib.Path:
    """Make an interpreter path absolute without resolving a venv symlink."""
    return pathlib.Path(os.path.abspath(pathlib.Path(path).expanduser()))


def _resolve_specs(
    workspace: pathlib.Path, args: argparse.Namespace
) -> dict[str, pathlib.Path]:
    """Resolve the train/proxy/benchmark spec paths.

    Two layouts are valid. A per-role evaluate spec
    (``evaluate_spec_proxy.toml`` / ``evaluate_spec_benchmark.toml``) is
    preferred: each file is concrete, already carrying its own
    ``annotation_path``, ``results_dir``, and ``save_folder``, so no job has to
    mutate a shared file at launch and a Proxy job cannot pick up the Benchmark
    annotation. A single ``evaluate_spec.toml`` is still accepted and is then
    materialized per stage. Explicit flags override both.
    """
    specs_dir = workspace / "specs"
    shared_evaluate = specs_dir / "evaluate_spec.toml"

    def evaluate_for(role: str, override: pathlib.Path | None) -> pathlib.Path:
        if override is not None:
            return override.expanduser().resolve()
        per_role = specs_dir / f"evaluate_spec_{role}.toml"
        return (per_role if per_role.is_file() else shared_evaluate).resolve()

    train = (
        args.train_spec.expanduser().resolve()
        if args.train_spec is not None
        else (specs_dir / "train_spec.toml").resolve()
    )
    return {
        "train": train,
        "proxy": evaluate_for("proxy", args.proxy_spec),
        "benchmark": evaluate_for("benchmark", args.benchmark_spec),
    }


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_model_manifest(value: str) -> dict[str, Any]:
    root = pathlib.Path(value)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    index = root / "model.safetensors.index.json"
    if index.is_file():
        index_payload = json.loads(index.read_text(encoding="utf-8"))
        shard_names = sorted(set(index_payload["weight_map"].values()))
    else:
        shard_names = sorted(path.name for path in root.glob("*.safetensors"))
    metadata_names = (
        "config.json",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tao_conversion_provenance.json",
    )
    metadata_hashes = {
        name: _sha256(root / name)
        for name in metadata_names
        if (root / name).is_file()
    }
    return {
        "schema_version": 1,
        "path": str(root),
        "model_type": config.get("model_type"),
        "weight_shards": len(shard_names),
        "weight_shard_bytes": {
            name: (root / name).stat().st_size for name in shard_names
        },
        "metadata_sha256": metadata_hashes,
    }


def _immutable_image_contract(value: str, *, name: str) -> dict[str, str]:
    match = re.fullmatch(r"(.+)@(sha256:[0-9a-f]{64})", value)
    if match is None:
        raise ValueError(
            f"{name} must include the runtime image digest as tag@sha256:<64 hex>"
        )
    return {
        "reference": match.group(1),
        "digest": match.group(2),
        "immutable": value,
    }


def _metric_contract(
    *,
    results_dir: pathlib.Path,
    threshold: float,
) -> dict[str, Any]:
    return validate_contract(
        {
            "name": "f1_cohort_balanced_v1",
            "display_name": "Worst required cohort F1 attainment",
            "operator": ">=",
            "target": 1.0,
            "unit": "",
            "kpi_profile": "f1_cohort_balanced_v1",
            "component_threshold": threshold,
            "required_components": [
                "non_reference_based.tasks.BCQ.macro_f1",
                "non_reference_based.tasks.MCQ.macro_f1",
                "non_reference_based.tasks.DET.f1",
                "reference_based.tasks.BCQ.macro_f1",
                "reference_based.tasks.DET.f1",
            ],
            "evaluator": {
                "type": "artifact",
                "producer": "scripts/exact_f1_adapter.py",
                "path_template": str(
                    results_dir
                    / "{iter_label}"
                    / "benchmark_metrics"
                    / "metric_result.json"
                ),
            },
            "constraints": [
                {
                    "name": name,
                    "display_name": name.replace("_", " ").title(),
                    "operator": "<=",
                    "target": 0,
                    "unit": "",
                }
                for name in (
                    "missing_evaluated_predictions",
                    "unknown_prediction_ids",
                )
            ],
            "tie_breakers": [
                {"name": "minimum_f1", "direction": "max"},
                {"name": "mean_f1", "direction": "max"},
                {"name": "coverage_failures", "direction": "min"},
            ],
        }
    )


def _dict_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _resolve_gap_analysis(args: argparse.Namespace, annotation_profile: str) -> dict[str, Any]:
    config_path = getattr(args, "gap_analysis_config", None)
    profile = getattr(args, "gap_analysis_profile", None)
    if config_path is not None:
        try:
            payload = yaml.safe_load(config_path.expanduser().read_text())
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid --gap-analysis-config: {exc}") from exc
        resolved = validate_config(payload)
        profile_name = "custom"
    else:
        profile_name = profile or "deficit_weighted_round_robin"
        resolved = load_profile(profile_name)
    budget = getattr(args, "gap_analysis_budget", None)
    seed = getattr(args, "gap_analysis_seed", None)
    if budget is not None:
        resolved["budget"] = budget
    if seed is not None:
        resolved["seed"] = seed
    resolved = validate_config(resolved)
    expected_builder = "multitask_v1"
    if resolved["candidate_builder"] != expected_builder:
        raise ValueError(
            f"gap profile {profile_name!r} candidate_builder="
            f"{resolved['candidate_builder']!r} is incompatible with "
            f"annotation profile {annotation_profile!r}"
        )
    return {
        "profile": profile_name,
        "resolved": resolved,
        "sha256": _dict_sha256(resolved),
    }


def build_state(args: argparse.Namespace) -> dict[str, Any]:
    workspace = args.workspace.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    annotations = {
        "proxy": (args.proxy_annotations or workspace / "annotations/proxy_kpi.jsonl")
        .expanduser()
        .resolve(),
        "benchmark": (
            args.benchmark_annotations
            or workspace / "annotations/benchmark.jsonl"
        )
        .expanduser()
        .resolve(),
        "mining": (
            args.mining_annotations
            or workspace / "annotations/mining.jsonl"
        )
        .expanduser()
        .resolve(),
    }
    missing = [f"{role}={path}" for role, path in annotations.items() if not path.is_file()]
    if missing:
        raise ValueError("annotation file(s) missing: " + ", ".join(missing))

    # Specs are staged before state is initialized. Checking them here keeps the
    # failure recoverable: state is written exactly once and must never be
    # hand-edited, so do not persist paths that are already known to be absent.
    specs = _resolve_specs(workspace, args)
    missing_specs = [
        f"{role}={path}" for role, path in specs.items() if not path.is_file()
    ]
    if missing_specs:
        raise ValueError(
            "spec file(s) missing: "
            + ", ".join(missing_specs)
            + ". Build them from the tao-finetune-cosmos-reason templates and "
            "stage them before initializing state."
        )
    annotation_profile = "nvpaw_multitask_v1"
    mining_router_mode = getattr(args, "mining_router_mode", "task_strict")
    prompt_variant = getattr(args, "prompt_variant", "official_v1")
    if prompt_variant != "official_v1":
        raise ValueError(f"unsupported prompt variant {prompt_variant!r}")
    kpi_profile = "f1_cohort_balanced_v1"
    gap_analysis = _resolve_gap_analysis(args, annotation_profile)
    contract = _metric_contract(
        results_dir=results_dir,
        threshold=args.kpi_threshold,
    )
    benchmark_hash = _sha256(annotations["benchmark"])
    media_root = (args.media_root or workspace).expanduser().resolve()
    base_model = validate_base_model(
        args.base_model or str(workspace / "models" / "Cosmos3-Nano-VLM")
    )
    base_model_manifest = _base_model_manifest(base_model)
    evaluator = (
        args.evaluator or workspace / "eval" / "calculate_f1_metrics.py"
    ).expanduser().resolve()
    if not evaluator.is_file() or evaluator.stat().st_size == 0:
        raise ValueError(f"exact NVPAW evaluator is missing or empty: {evaluator}")
    network_mode = getattr(args, "network_mode", None) or (
        "airgap" if os.environ.get("AIR_GAPPED") == "1" else "network-enabled"
    )
    network_source = getattr(args, "network_mode_source", None) or (
        "environment:AIR_GAPPED" if os.environ.get("AIR_GAPPED") == "1" else "default"
    )
    python_executable = _absolute_executable(
        getattr(args, "python_executable", None) or sys.executable
    )
    offline = network_mode == "airgap"
    return {
        "version": 7,
        "workflow": WORKFLOW,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "status": "in_progress",
        "kpi_target": render_target(contract),
        "metric_contract": contract,
        "metric_contract_sha256": _dict_sha256(contract),
        "results_dir": str(results_dir),
        "max_iterations": args.max_iterations,
        "current_iteration": 0,
        "execution_policy": {
            "network_mode": network_mode,
            "activation_source": network_source,
            "allow_package_install": not offline,
            "allow_remote_fetch": not offline,
            "allow_container_pull": not offline,
            "allow_registry_login": not offline,
            "python_launcher": "scripts/deft_python.sh",
            "python_executable": str(python_executable),
            "hf_offline": offline,
        },
        "config": {
            "workspace": str(workspace),
            "platform": args.platform,
            "model_skill": "tao-finetune-cosmos-reason",
            "base_model": base_model,
            "base_model_manifest": base_model_manifest,
            "automl_policy": "off",
            "annotation_mode": annotation_profile,
            "annotation_profile": annotation_profile,
            "prompt_variant": prompt_variant,
            "media_root": str(media_root),
            "task_scope": {
                "eligible_task_types": sorted(TASK_SPECS),
                "mining_unsupported_task_policy": "ignore_and_count",
                "other_roles_unsupported_task_policy": "reject",
            },
            "annotations": {role: str(path) for role, path in annotations.items()},
            "annotation_sha256": {
                role: _sha256(path) for role, path in annotations.items()
            },
            "kpi": {
                "profile": kpi_profile,
                "component_threshold": args.kpi_threshold,
                "evaluator": str(evaluator),
                "evaluator_sha256": _sha256(evaluator),
            },
            "gap_analysis": gap_analysis,
            "evaluation": {
                "proxy": {
                    "annotations": str(annotations["proxy"]),
                    "drives_rcca": True,
                    "drives_loop_stop": False,
                },
                "benchmark": {
                    "annotations": str(annotations["benchmark"]),
                    "sha256": benchmark_hash,
                    "drives_rcca": False,
                    "drives_loop_stop": True,
                },
            },
            "specs": {role: str(path) for role, path in specs.items()},
            "containers": {
                "cosmos_framework": _immutable_image_contract(
                    args.framework_container, name="Framework container"
                ),
                "data_services": _immutable_image_contract(
                    args.mining_container, name="Mining container"
                ),
            },
            "training": {
                "backend": "cosmos-framework",
                "profile": args.recipe_profile,
                "annotation_source": "mined_real_samples_only",
                "checkpoint_format": "framework_dcp",
                "dcp_async_mode_enabled": False,
                "num_gpus": args.num_gpus,
                "num_nodes": args.num_nodes,
                "gpu_model": args.gpu_model,
                "full_parameter": True,
                "precision": "bfloat16",
                "micro_batch_per_rank": 4,
                "gradient_accumulation": 16,
                "global_batch": 4 * args.num_gpus * 16,
                "optimizer": {
                    "name": "AdamW",
                    "fused": True,
                    "learning_rate": 1.0e-6,
                    "weight_decay": 0.05,
                    "betas": [0.9, 0.999],
                    "merger_lr_multiplier": 20.0,
                },
                "freeze": {
                    "vision_encoder": True,
                    "multimodal_projector": False,
                    "language_model": False,
                },
            },
            "mining": {
                "router_mode": mining_router_mode,
                "top_k_scope": (
                    "target" if mining_router_mode == "image_only" else "target_task"
                ),
                "top_k_per_target": args.top_k_per_target,
                "metric": "cosine",
                "min_similarity": args.min_similarity,
                "history_aware": {
                    "enabled": True,
                    "identity": "target_id",
                    "history_file": str(results_dir / "mining_history.json"),
                },
            },
        },
        "iterations": {},
        "events": [],
        "_completed_step_values": STAGES,
        "_status_values": STATUSES,
    }


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2)
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
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--workspace", required=True, type=pathlib.Path)
    parser.add_argument(
        "--network-mode",
        choices=("airgap", "network-enabled"),
        help="Immutable execution mode; defaults from AIR_GAPPED=1, otherwise network-enabled.",
    )
    parser.add_argument(
        "--network-mode-source",
        help="Human-readable source of the mode decision, such as ci-prompt or operator.",
    )
    parser.add_argument(
        "--python-executable",
        type=pathlib.Path,
        help="Dependency-complete Python selected during preflight; defaults to this interpreter.",
    )
    parser.add_argument("--platform", required=True)
    parser.add_argument("--max-iterations", required=True, type=int)
    parser.add_argument("--kpi-threshold", type=float, default=0.8)
    parser.add_argument("--prompt-variant", default="official_v1")
    parser.add_argument("--evaluator", type=pathlib.Path)
    gap_choice = parser.add_mutually_exclusive_group()
    gap_choice.add_argument("--gap-analysis-profile")
    gap_choice.add_argument("--gap-analysis-config", type=pathlib.Path)
    parser.add_argument("--gap-analysis-budget", type=int)
    parser.add_argument("--gap-analysis-seed", type=int)
    parser.add_argument(
        "--base-model",
        help=(
            "Complete local Qwen3-VL Hugging Face snapshot. Defaults to "
            "WORKSPACE/models/Cosmos3-Nano-VLM."
        ),
    )
    parser.add_argument("--media-root", type=pathlib.Path)
    parser.add_argument("--train-spec", type=pathlib.Path)
    parser.add_argument(
        "--proxy-spec",
        type=pathlib.Path,
        help="Proxy evaluate spec. Defaults to specs/evaluate_spec_proxy.toml, "
        "falling back to specs/evaluate_spec.toml.",
    )
    parser.add_argument(
        "--benchmark-spec",
        type=pathlib.Path,
        help="Benchmark evaluate spec. Defaults to "
        "specs/evaluate_spec_benchmark.toml, falling back to "
        "specs/evaluate_spec.toml.",
    )
    parser.add_argument("--proxy-annotations", type=pathlib.Path)
    parser.add_argument("--benchmark-annotations", type=pathlib.Path)
    parser.add_argument("--mining-annotations", type=pathlib.Path)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument(
        "--gpu-model",
        required=True,
        help=(
            "Exact accelerator model recorded by the selected platform's "
            "Preflight, including memory when available"
        ),
    )
    parser.add_argument("--recipe-profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--top-k-per-target", type=int, default=5)
    parser.add_argument("--min-similarity", type=float, default=0.9)
    parser.add_argument(
        "--mining-router-mode",
        choices=MINING_ROUTER_MODES,
        default="task_strict",
        help=(
            "Candidate routing policy over the same image embeddings. "
            "Task-aware modes require --annotation-profile nvpaw_multitask_v1."
        ),
    )
    parser.add_argument("--framework-container", required=True)
    parser.add_argument("--mining-container", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.gpu_model = args.gpu_model.strip()
    if not args.gpu_model:
        print("init_deft_state: --gpu-model must not be empty", file=sys.stderr)
        return 2
    if args.network_mode is not None and args.network_mode_source is None:
        args.network_mode_source = "cli:--network-mode"
    if args.network_mode is None and args.network_mode_source is not None:
        print(
            "init_deft_state: --network-mode-source requires --network-mode",
            file=sys.stderr,
        )
        return 2
    if os.environ.get("AIR_GAPPED") == "1" and args.network_mode == "network-enabled":
        print(
            "init_deft_state: AIR_GAPPED=1 cannot be overridden by --network-mode network-enabled",
            file=sys.stderr,
        )
        return 2
    if args.python_executable is not None:
        executable = _absolute_executable(args.python_executable)
        if not executable.is_file() or not os.access(executable, os.X_OK):
            print(
                f"init_deft_state: --python-executable must be executable: {executable}",
                file=sys.stderr,
            )
            return 2
        args.python_executable = executable
    positive = {
        "max_iterations": args.max_iterations,
        "num_gpus": args.num_gpus,
        "num_nodes": args.num_nodes,
        "top_k_per_target": args.top_k_per_target,
    }
    if args.gap_analysis_budget is not None:
        positive["gap_analysis_budget"] = args.gap_analysis_budget
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        print(f"init_deft_state: positive values required: {invalid}", file=sys.stderr)
        return 2
    if not 0.0 < args.kpi_threshold <= 1.0:
        print("init_deft_state: --kpi-threshold must be in (0, 1]", file=sys.stderr)
        return 2
    if not -1.0 <= args.min_similarity <= 1.0:
        print("init_deft_state: --min-similarity must be in [-1, 1]", file=sys.stderr)
        return 2
    if args.recipe_profile == "full" and (args.num_gpus != 8 or args.num_nodes != 1):
        print(
            "init_deft_state: full recipe requires exactly one 8-GPU node",
            file=sys.stderr,
        )
        return 2
    if not args.framework_container or not args.mining_container:
        print(
            "init_deft_state: both container images are required; resolve "
            "images.tao_toolkit.{cosmos_framework,data_services} from versions.yaml",
            file=sys.stderr,
        )
        return 2
    output = args.results_dir.expanduser().resolve() / "deft_state.json"
    if output.exists():
        print(
            f"init_deft_state: refusing to overwrite existing state: {output}",
            file=sys.stderr,
        )
        return 2
    try:
        state = build_state(args)
        output.parent.mkdir(parents=True, exist_ok=True)
        (output.parent / "resolved_gap_analysis.yaml").write_text(
            yaml.safe_dump(
                state["config"]["gap_analysis"]["resolved"], sort_keys=True
            )
        )
        _atomic_json(output, state)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"init_deft_state: {exc}", file=sys.stderr)
        return 2
    print(f"init_deft_state: wrote {output}", file=sys.stderr)
    try:
        render_html_report(args.results_dir)
    except Exception as exc:  # noqa: BLE001 - state initialization remains valid
        print(f"init_deft_state: report hook failed: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
