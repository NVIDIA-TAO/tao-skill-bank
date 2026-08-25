# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact platform-neutral TAO argv contracts for the IAA DEFT workflow."""

from __future__ import annotations

import hashlib
import json
import re
import pathlib
from typing import Any


_MODEL_COMMANDS = {
    "pool_embed",
    "target_embed",
    "viz_weak_embed",
    "viz_mined_embed",
    "viz_previous_embed",
    "train",
    "evaluate",
}

_ADAPTERS = {
    "dataset_rebuild", "dataset_materialize", "gap_analysis",
    "mining_postprocess", "history_select", "visualize_prepare",
    "visualize_finish", "eval_config", "train_config",
    "publish_checkpoint", "iteration_summary", "metric_parse", "report",
    "sdg_normalize_repair",
}


def command_sha256(command: list[str]) -> str:
    """Return a stable digest for an argv vector (without shell parsing)."""
    encoded = json.dumps(
        command, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _iteration_number(label: str) -> int:
    match = re.fullmatch(r"iter([1-9][0-9]*)", label)
    if not match:
        raise ValueError(f"container command requires iterN, got {label!r}")
    return int(match.group(1))


def expected_container_command(
    name: str, label: str, config: dict[str, Any]
) -> list[str]:
    """Build the one allowed argv vector for a named workflow launch."""
    if name in _ADAPTERS:
        if name == "sdg_normalize_repair":
            return [
                "python3", "/iaa-runtime/repair_sdg_normalize_freshness.py",
                "recompute", "--results-dir", "/results",
                "--iteration", str(_iteration_number(label)),
            ]
        return [
            "python3", "/iaa-runtime/run_iaa_compute.py", name,
            "--results-dir", "/results", "--label", label,
        ]
    if name == "pool_embed":
        if label != "baseline":
            raise ValueError("pool_embed is valid only for baseline")
        return [
            "embedding",
            "text_embeddings",
            "-e",
            "/specs/text_embed_spec.yaml",
            "input_parquet=/results/embeddings/source/source_pool.parquet",
            "output_parquet=/results/embeddings/source/embeddings.parquet",
        ]

    if name == "evaluate":
        phase = "zs" if label == "baseline" else f"iter_{_iteration_number(label)}"
        return ["clip", "evaluate", "-e", f"/results/{phase}/specs/eval_config.yaml"]

    number = _iteration_number(label)
    if name == "target_embed":
        return [
            "embedding",
            "text_embeddings",
            "-e",
            "/specs/text_embed_spec.yaml",
            f"input_parquet=/results/iter_{number}/gaps/kpi_gaps.parquet",
            f"output_parquet=/results/iter_{number}/embeddings/target/embeddings.parquet",
        ]
    if name == "knn":
        return [
            "tmm",
            "nearest_neighbors",
            "-e",
            "/specs/mining_spec.yaml",
            "source_parquet=/results/embeddings/source/embeddings.parquet",
            f"target_parquet=/results/iter_{number}/embeddings/target/embeddings.parquet",
            f"output_parquet=/results/iter_{number}/mining/mined_samples.parquet",
            f"topn={config.get('mining_topn')}",
            f"knn_metric={config.get('knn_metric')}",
        ]
    if name == "train":
        return ["clip", "train", "-e", f"/results/iter_{number}/specs/train_config.yaml"]
    if name == "viz_weak_embed":
        return [
            "embedding",
            "image_embeddings",
            "-e",
            "/specs/image_embed_spec.yaml",
            f"input_parquet=/results/iter_{number}/embeddings/viz_weak/input.parquet",
            f"output_parquet=/results/iter_{number}/embeddings/viz_weak/embeddings.parquet",
        ]
    if name == "viz_mined_embed":
        return [
            "embedding",
            "image_embeddings",
            "-e",
            "/specs/image_embed_spec.yaml",
            f"input_parquet=/results/iter_{number}/mining/mined_unique_images.parquet",
            f"output_parquet=/results/iter_{number}/embeddings/augmented/mined_embeddings.parquet",
        ]
    if name == "viz_previous_embed":
        return [
            "embedding",
            "image_embeddings",
            "-e",
            "/specs/image_embed_spec.yaml",
            f"input_parquet=/results/iter_{number}/embeddings/previous/prev_pool.parquet",
            f"output_parquet=/results/iter_{number}/embeddings/previous/embeddings.parquet",
        ]
    raise ValueError(f"unsupported IAA DEFT container command name: {name!r}")


def expected_image_kind(name: str) -> str:
    if name in _ADAPTERS:
        return "pyt" if name == "publish_checkpoint" else "ds"
    if name in {"train", "evaluate"}:
        return "pyt"
    if name in {
        "pool_embed",
        "target_embed",
        "knn",
        "viz_weak_embed",
        "viz_mined_embed",
        "viz_previous_embed",
    }:
        return "ds"
    raise ValueError(f"unsupported IAA DEFT container command name: {name!r}")


def expected_hf_forwarding(name: str, config: dict[str, Any]) -> bool:
    """Only model-loading stages receive the approved token environment."""
    return bool(config.get("requires_hf_token")) and name in _MODEL_COMMANDS


def expected_stage_directory(
    name: str, label: str, results_dir: pathlib.Path
) -> pathlib.Path:
    """Return the exact host directory that owns a container status."""
    if name in {"dataset_rebuild", "dataset_materialize"}:
        return results_dir / "dataset_setup"
    if name == "report":
        return results_dir
    if name == "pool_embed":
        if label != "baseline":
            raise ValueError("pool_embed is valid only for baseline")
        return results_dir / "embeddings" / "source"
    phase = (
        results_dir / "zs"
        if label == "baseline"
        else results_dir / f"iter_{_iteration_number(label)}"
    )
    if name == "evaluate":
        return phase / "evaluate"
    adapter_suffixes = {
        "gap_analysis": pathlib.Path("gaps"),
        "mining_postprocess": pathlib.Path("mining"),
        "history_select": pathlib.Path("mining"),
        "visualize_prepare": pathlib.Path("visualization"),
        "visualize_finish": pathlib.Path("visualization"),
        "eval_config": pathlib.Path("specs"),
        "train_config": pathlib.Path("specs"),
        "publish_checkpoint": pathlib.Path("train"),
        "iteration_summary": pathlib.Path("."),
        "metric_parse": pathlib.Path("evaluate"),
        "sdg_normalize_repair": pathlib.Path("datagen"),
    }
    if name in adapter_suffixes:
        return phase / adapter_suffixes[name]
    _iteration_number(label)
    suffixes = {
        "target_embed": pathlib.Path("embeddings/target"),
        "knn": pathlib.Path("mining"),
        "viz_weak_embed": pathlib.Path("embeddings/viz_weak"),
        "viz_mined_embed": pathlib.Path("embeddings/augmented"),
        "viz_previous_embed": pathlib.Path("embeddings/previous"),
        "train": pathlib.Path("train"),
    }
    suffix = suffixes.get(name)
    if suffix is None:
        raise ValueError(f"unsupported IAA DEFT container command name: {name!r}")
    return phase / suffix


def expected_fresh_outputs(
    name: str, label: str, results_dir: pathlib.Path
) -> list[pathlib.Path]:
    """Return the exact files whose recreation proves a container stage."""
    if name == "dataset_rebuild":
        return [results_dir / "dataset_setup" / "rebuild_verify.log"]
    if name == "dataset_materialize":
        return [results_dir / "dataset_setup" / "dataset-materialize.host.status.json"]
    if name == "report":
        return [results_dir / "DEFT_Loop_Report.html"]
    if name == "pool_embed":
        return [results_dir / "embeddings" / "source" / "embeddings.parquet"]
    phase = (
        results_dir / "zs"
        if label == "baseline"
        else results_dir / f"iter_{_iteration_number(label)}"
    )
    if name == "evaluate":
        return [
            phase / "evaluate" / "nvidia_pas_metrics.csv",
            phase / "evaluate" / "nvidia_pas_metrics_aggregate.csv",
            phase / "evaluate" / "nvidia_pas_metrics_weighted_aggregate.csv",
            phase / "evaluate" / "status.json",
        ]
    if name == "eval_config" and label == "baseline":
        return [phase / "specs" / "eval_config.yaml"]
    if name == "metric_parse" and label == "baseline":
        return [phase / "evaluate" / "metric_result.json"]
    number = _iteration_number(label)
    adapter_fixed = {
        "gap_analysis": [phase / "gaps" / "kpi_gaps.parquet"],
        "mining_postprocess": [phase / "mining" / "history_candidates" / "mined_pairs.json"],
        "history_select": [
            phase / "mining" / "mined_image_list.txt",
            phase / "mining" / "mined_pairs.json",
            phase / "mining" / "mined_dataset.json",
            phase / "mining" / "cumulative_mined_unique_names.json",
        ],
        "visualize_prepare": [phase / "visualization" / "visualize-prepare.host.status.json"],
        "visualize_finish": [phase / "visualization" / "visualize-finish.host.status.json"],
        "eval_config": [phase / "specs" / "eval_config.yaml"],
        "train_config": [phase / "specs" / "train_config.yaml"],
        "publish_checkpoint": [phase / "train" / "publish-checkpoint.host.status.json"],
        "iteration_summary": [phase / "iteration_summary.json"],
        "metric_parse": [phase / "evaluate" / "metric_result.json"],
        "sdg_normalize_repair": [
            phase / "datagen" / "dataset" / "sdg_manifest.json",
            phase / "datagen" / "dataset" / "sdg_pairs.json",
            phase / "datagen" / "dataset" / "sdg_image_list.txt",
        ],
    }
    if name in adapter_fixed:
        return adapter_fixed[name]
    fixed = {
        "target_embed": phase / "embeddings" / "target" / "embeddings.parquet",
        "knn": phase / "mining" / "mined_samples.parquet",
        "viz_weak_embed": phase / "embeddings" / "viz_weak" / "embeddings.parquet",
        "viz_mined_embed": phase / "embeddings" / "augmented" / "mined_embeddings.parquet",
        "viz_previous_embed": phase / "embeddings" / "previous" / "embeddings.parquet",
        "train": phase / "train" / "status.json",
    }
    if name not in fixed:
        raise ValueError(
            f"unsupported IAA DEFT container command name for iter{number}: {name!r}"
        )
    return [fixed[name]]
