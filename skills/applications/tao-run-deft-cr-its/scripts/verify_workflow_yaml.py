#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the high-level DEFT CR ITS workflow YAML."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from workflow_common import (
    DATA_GENERATION_MODES,
    MODALITY_CHOICES,
    absolute_path,
    data_generation_mode,
    existing_absolute_path,
    genai_enabled,
    load_yaml,
    mining_enabled,
    optional_mapping,
    optional_embedding_parquets,
    optional_bool,
    require_mapping,
    require_string,
)


HF_MODEL_PREFIX = "hf_model://"


def require_positive_int(section: dict[str, Any], dotted_key: str) -> int:
    """Return a required positive integer value from a section."""
    key = dotted_key.rsplit(".", 1)[-1]
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"missing required positive integer: {dotted_key}")
    return value


def validate_optional_checkpoint(mining: dict[str, Any], workspace: Path) -> str | None:
    """Validate the optional local checkpoint path or remote Hugging Face model id."""
    value = mining.get("cosmos_embed_checkpoint_path")
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError("mining.cosmos_embed_checkpoint_path must be a string or null")
    configured_path = Path(os.path.expanduser(value))
    if configured_path.is_absolute():
        return str(
            existing_absolute_path(
                value,
                workspace,
                "mining.cosmos_embed_checkpoint_path",
                "path",
            )
        )
    model_id = value[len(HF_MODEL_PREFIX) :] if value.startswith(HF_MODEL_PREFIX) else value
    if model_id.startswith(".") or "/" not in model_id:
        raise ValueError(
            "mining.cosmos_embed_checkpoint_path must be an absolute local path, "
            "a Hugging Face model id like nvidia/Cosmos-Embed1-224p, or null"
        )
    return value


def validate_cosmos_embed_template(path: Path) -> int:
    """Return the positive GPU count declared by a Cosmos Embed inference template."""
    template = load_yaml(path)
    inference = template.get("inference")
    if not isinstance(inference, dict):
        raise ValueError(f"{path}: missing required object 'inference'")
    num_gpus = inference.get("num_gpus")
    if not isinstance(num_gpus, int) or isinstance(num_gpus, bool) or num_gpus < 1:
        raise ValueError(f"{path}: inference.num_gpus must be a positive integer")
    return num_gpus


def validate_captioning_endpoint(value: str) -> str:
    """Validate the required OpenAI-compatible VLM captioning base URL."""
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "genai.vlm_captioning_endpoint must be an absolute http(s) base URL"
        )
    return value.rstrip("/")


def load_llava_annotations(annotation_path: Path) -> list[dict[str, Any]]:
    """Read a LLaVA annotation file and require a list of JSON objects."""
    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{annotation_path}: expected a JSON array of LLaVA annotation items")
    if not payload:
        raise ValueError(f"{annotation_path}: annotation file has no items")
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{annotation_path}: item {index} is not a JSON object")
    return payload


def resolve_llava_media_path(media_value: str, media_dir: Path) -> Path:
    """Resolve a LLaVA video field using the configured dataset media directory."""
    if os.path.isabs(media_value):
        return Path(os.path.normpath(os.path.expanduser(media_value)))
    return Path(os.path.normpath(os.path.join(str(media_dir), media_value)))


def validate_annotations_match_media_dir(
    annotation_path: Path,
    media_dir: Path,
    dataset_key: str,
) -> int:
    """Require every LLaVA video item to resolve to an existing file."""
    items = load_llava_annotations(annotation_path)
    missing: list[tuple[int, Path]] = []
    annotation_ids: set[str] = set()
    for index, item in enumerate(items, start=1):
        annotation_id = item.get("id")
        if not isinstance(annotation_id, str) or not annotation_id.strip():
            raise ValueError(f"{annotation_path}: item {index} is missing non-empty field 'id'")
        if annotation_id in annotation_ids:
            raise ValueError(f"{annotation_path}: duplicate annotation id: {annotation_id!r}")
        annotation_ids.add(annotation_id)
        media_value = item.get("video")
        if not isinstance(media_value, str) or not media_value.strip():
            raise ValueError(f"{annotation_path}: item {index} is missing non-empty field 'video'")
        media_path = resolve_llava_media_path(media_value, media_dir)
        if not media_path.is_file():
            missing.append((index, media_path))

    if missing:
        examples = ", ".join(f"item {index}: {path}" for index, path in missing[:5])
        extra = "" if len(missing) <= 5 else f", ... {len(missing) - 5} more"
        raise FileNotFoundError(
            f"{dataset_key} annotations are not compatible with {dataset_key}.media_dir; "
            f"{len(missing)} referenced media file(s) were not found after resolving relative "
            f"'video' values against {media_dir}: {examples}{extra}"
        )
    return len(items)


def validate_workflow_config(config: dict[str, Any], workspace: Path) -> dict[str, Any]:
    """Validate required workflow fields and return resolved values for logging."""
    if not workspace.is_dir():
        raise NotADirectoryError(f"workspace must be an existing directory: {workspace}")

    run = require_mapping(config, "run")
    run_name = run.get("name")
    if run_name is not None and (not isinstance(run_name, str) or not run_name.strip()):
        raise ValueError("run.name must be a non-empty string or null")
    max_iterations = require_positive_int(run, "run.max_iterations")

    kpi_dataset = require_mapping(config, "kpi_dataset")
    kpi_annotations = existing_absolute_path(
        require_string(kpi_dataset, "kpi_dataset.annotations_path"),
        workspace,
        "kpi_dataset.annotations_path",
        "file",
    )
    kpi_media_dir = existing_absolute_path(
        require_string(kpi_dataset, "kpi_dataset.media_dir"),
        workspace,
        "kpi_dataset.media_dir",
        "dir",
    )
    kpi_annotation_count = validate_annotations_match_media_dir(
        kpi_annotations,
        kpi_media_dir,
        "kpi_dataset",
    )

    mode = data_generation_mode(config)
    use_mining = mining_enabled(mode)
    use_genai = genai_enabled(mode)

    train_dataset = optional_mapping(config, "train_dataset")
    if use_mining and train_dataset is None:
        raise ValueError("train_dataset is required when data_generation.mode includes mining")
    train_annotations: Path | None = None
    train_media_dir: Path | None = None
    train_annotation_count: int | None = None
    if train_dataset is not None:
        train_annotations = existing_absolute_path(
            require_string(train_dataset, "train_dataset.annotations_path"),
            workspace,
            "train_dataset.annotations_path",
            "file",
        )
        train_media_dir = existing_absolute_path(
            require_string(train_dataset, "train_dataset.media_dir"),
            workspace,
            "train_dataset.media_dir",
            "dir",
        )
        train_annotation_count = validate_annotations_match_media_dir(
            train_annotations,
            train_media_dir,
            "train_dataset",
        )

    cosmos_reason = require_mapping(config, "cosmos_reason")
    baseline_model = existing_absolute_path(
        require_string(cosmos_reason, "cosmos_reason.baseline_model_path"),
        workspace,
        "cosmos_reason.baseline_model_path",
        "path",
    )
    base_evaluate_toml = existing_absolute_path(
        require_string(cosmos_reason, "cosmos_reason.base_evaluate_toml"),
        workspace,
        "cosmos_reason.base_evaluate_toml",
        "file",
    )
    base_train_toml = existing_absolute_path(
        require_string(cosmos_reason, "cosmos_reason.base_train_toml"),
        workspace,
        "cosmos_reason.base_train_toml",
        "file",
    )
    continual_model = optional_bool(cosmos_reason, "cosmos_reason.continual_model", False)

    embeddings_spec_template: Path | None = None
    embeddings_num_gpus: int | None = None
    embeddings_modality: str | None = None
    mining_spec_template: Path | None = None
    mine_unique_only: bool | None = None
    cosmos_embed_checkpoint: str | None = None
    embedding_parquets: dict[str, Path | None] = {"kpi": None, "train": None}
    if use_mining:
        mining = require_mapping(config, "mining")
        embeddings_spec_template = existing_absolute_path(
            require_string(mining, "mining.embeddings_spec_template"),
            workspace,
            "mining.embeddings_spec_template",
            "file",
        )
        embeddings_num_gpus = validate_cosmos_embed_template(embeddings_spec_template)
        embeddings_modality = require_string(mining, "mining.embeddings_modality")
        if embeddings_modality not in MODALITY_CHOICES:
            choices = ", ".join(sorted(MODALITY_CHOICES))
            raise ValueError(f"mining.embeddings_modality must be one of: {choices}")
        mining_spec_template = existing_absolute_path(
            require_string(mining, "mining.mining_spec_template"),
            workspace,
            "mining.mining_spec_template",
            "file",
        )
        mine_unique_only = optional_bool(mining, "mining.mine_unique_only", True)
        cosmos_embed_checkpoint = validate_optional_checkpoint(mining, workspace)
        embedding_parquets = optional_embedding_parquets(mining, workspace, embeddings_modality)

    vlm_captioning_endpoint: str | None = None
    paidf_num_gpus: int | None = None
    generation_settings: Path | None = None
    caption_prompt_file: Path | None = None
    if use_genai:
        genai = require_mapping(config, "genai")
        vlm_captioning_endpoint = validate_captioning_endpoint(
            require_string(genai, "genai.vlm_captioning_endpoint")
        )
        paidf_num_gpus = require_positive_int(genai, "genai.paidf_num_gpus")
        generation_settings_value = genai.get("generation_settings")
        if generation_settings_value not in (None, ""):
            if not isinstance(generation_settings_value, str):
                raise ValueError("genai.generation_settings must be a string or null")
            generation_settings = existing_absolute_path(
                generation_settings_value,
                workspace,
                "genai.generation_settings",
                "file",
            )
        caption_prompt_value = genai.get("caption_prompt_file")
        if caption_prompt_value not in (None, ""):
            if not isinstance(caption_prompt_value, str):
                raise ValueError("genai.caption_prompt_file must be a string or null")
            caption_prompt_file = existing_absolute_path(
                caption_prompt_value,
                workspace,
                "genai.caption_prompt_file",
                "file",
            )

    return {
        "workspace": str(workspace),
        "run_name": run_name,
        "max_iterations": max_iterations,
        "data_generation_mode": mode,
        "available_data_generation_modes": list(DATA_GENERATION_MODES),
        "kpi_annotations": str(kpi_annotations),
        "kpi_media_dir": str(kpi_media_dir),
        "kpi_annotation_count": kpi_annotation_count,
        "train_annotations": str(train_annotations) if train_annotations else None,
        "train_media_dir": str(train_media_dir) if train_media_dir else None,
        "train_annotation_count": train_annotation_count,
        "baseline_model": str(baseline_model),
        "base_evaluate_toml": str(base_evaluate_toml),
        "base_train_toml": str(base_train_toml),
        "continual_model": continual_model,
        "embeddings_spec_template": str(embeddings_spec_template) if embeddings_spec_template else None,
        "embeddings_num_gpus": embeddings_num_gpus,
        "embeddings_modality": embeddings_modality,
        "cosmos_embed_checkpoint": cosmos_embed_checkpoint,
        "kpi_embeddings_parquet": str(embedding_parquets["kpi"]) if embedding_parquets["kpi"] else None,
        "train_embeddings_parquet": str(embedding_parquets["train"]) if embedding_parquets["train"] else None,
        "mining_spec_template": str(mining_spec_template) if mining_spec_template else None,
        "mine_unique_only": mine_unique_only,
        "vlm_captioning_endpoint": vlm_captioning_endpoint,
        "paidf_num_gpus": paidf_num_gpus,
        "generation_settings": str(generation_settings) if generation_settings else None,
        "caption_prompt_file": str(caption_prompt_file) if caption_prompt_file else None,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-yaml", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    """Validate workflow.yaml and print resolved paths."""
    args = parse_args()
    workflow_yaml = absolute_path(args.workflow_yaml)
    workspace = absolute_path(args.workspace)
    try:
        if not workflow_yaml.is_file():
            raise FileNotFoundError(f"workflow YAML does not exist: {workflow_yaml}")
        resolved = validate_workflow_config(load_yaml(workflow_yaml), workspace)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("workflow.yaml is valid")
    for key, value in resolved.items():
        if value is not None:
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
