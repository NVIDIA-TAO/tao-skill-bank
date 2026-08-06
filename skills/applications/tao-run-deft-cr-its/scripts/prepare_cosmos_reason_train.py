#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare derived annotations and configuration for Cosmos Reason training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from workflow_common import (
    absolute_path,
    data_generation_mode,
    dump_toml,
    existing_absolute_path,
    genai_enabled,
    load_json_array,
    load_toml,
    load_yaml,
    mining_enabled,
    normalize_media_path,
    optional_mapping,
    path_in_workspace,
    require_mapping,
    require_string,
    write_json_array,
)


def build_llava_records(
    mined_neighbors_parquet: Path,
    train_embeddings_parquet: Path,
    train_lookup_parquet: Path,
) -> list[dict[str, Any]]:
    """Recover mined source modalities and return their LLaVA records."""
    neighbors = pd.read_parquet(mined_neighbors_parquet)
    source_col = "source_filepath" if "source_filepath" in neighbors.columns else "filepath"
    if source_col not in neighbors.columns:
        raise ValueError(f"{mined_neighbors_parquet}: missing source filepath column")

    embeddings = pd.read_parquet(train_embeddings_parquet)
    required_embedding_columns = {"filepath", "modality"}
    missing = required_embedding_columns - set(embeddings.columns)
    if missing:
        raise ValueError(f"{train_embeddings_parquet}: missing required columns: {sorted(missing)}")
    source_metadata = embeddings[["filepath", "modality"]].drop_duplicates()
    conflicts = source_metadata.groupby("filepath")["modality"].nunique()
    conflicting_paths = conflicts[conflicts > 1].index.astype(str).tolist()
    if conflicting_paths:
        raise ValueError(
            f"{train_embeddings_parquet}: source filepaths map to multiple modalities: "
            f"{conflicting_paths[:5]}"
        )

    mined = neighbors[[source_col]].drop_duplicates().rename(columns={source_col: "source_filepath"})
    if mined.empty:
        return []
    mined = mined.merge(
        source_metadata,
        left_on="source_filepath",
        right_on="filepath",
        how="left",
        validate="one_to_one",
    )
    missing_metadata = mined[mined["modality"].isna()]["source_filepath"].astype(str).tolist()
    if missing_metadata:
        raise ValueError(
            "mined source filepaths are absent from the train embeddings parquet: "
            f"{missing_metadata[:5]}"
        )
    unsupported = sorted(set(mined["modality"].astype(str)) - {"text", "video"})
    if unsupported:
        raise ValueError(f"unsupported source modalities in {train_embeddings_parquet}: {unsupported}")

    lookup = pd.read_parquet(train_lookup_parquet)
    required = {"filepath", "annotation_id", "video_path", "question", "answer"}
    missing = required - set(lookup.columns)
    if missing:
        raise ValueError(f"{train_lookup_parquet}: missing required columns: {sorted(missing)}")

    matched: list[pd.DataFrame] = []
    text_sources = mined[mined["modality"] == "text"]
    if not text_sources.empty:
        matched.append(
            text_sources.merge(lookup, left_on="source_filepath", right_on="filepath", how="inner")
        )
    video_sources = mined[mined["modality"] == "video"]
    if not video_sources.empty:
        matched.append(
            video_sources.merge(lookup, left_on="source_filepath", right_on="video_path", how="inner")
        )
    merged = pd.concat(matched, ignore_index=True) if matched else pd.DataFrame()
    if merged.empty:
        raise RuntimeError("no mined neighbors joined to train lookup rows")
    unmatched = sorted(
        set(mined["source_filepath"].astype(str)) - set(merged["source_filepath"].astype(str))
    )
    if unmatched:
        raise ValueError(f"mined source filepaths did not match the train lookup: {unmatched[:5]}")

    records: list[dict[str, Any]] = []
    for _, row in merged.drop_duplicates(subset=["annotation_id"]).iterrows():
        records.append(
            {
                "id": str(row["annotation_id"]),
                "video": str(row["video_path"]),
                "conversations": [
                    {"from": "human", "value": f"<video>\n{row['question']}"},
                    {"from": "gpt", "value": str(row["answer"])},
                ],
            }
        )
    return records


def annotation_id(record: dict[str, Any], source: Path, index: int) -> str:
    """Return a required LLaVA annotation id."""
    value = record.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}: item {index} is missing non-empty 'id'")
    return value


def absolute_video(
    record: dict[str, Any],
    source: Path,
    media_dir: Path | None,
) -> dict[str, Any]:
    """Return a copied LLaVA record with an absolute video path."""
    video = record.get("video")
    if not isinstance(video, str) or not video:
        raise ValueError(f"{source}: annotation {record.get('id')!r} is missing non-empty 'video'")
    if Path(video).is_absolute():
        video_path = Path(normalize_media_path(video))
    else:
        if media_dir is None:
            raise ValueError(
                f"{source}: annotation {record.get('id')!r} has relative video path {video!r}; "
                "provide its media directory"
            )
        video_path = Path(normalize_media_path(str(media_dir / video)))
    copied = dict(record)
    copied["video"] = str(video_path)
    return copied


def assemble_annotations(
    previous_annotations: Path | None,
    current_annotations: list[Path],
    previous_media_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Merge prior/seed and current derived annotations, deduplicated by LLaVA id."""
    sources: list[tuple[Path, Path | None, bool]] = []
    if previous_annotations is not None:
        sources.append((previous_annotations, previous_media_dir, False))
    sources.extend((path, None, True) for path in current_annotations)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    current_count = 0
    for source, media_dir, is_current in sources:
        if not source.is_file():
            raise FileNotFoundError(f"annotation source does not exist: {source}")
        source_records = load_json_array(source)
        if is_current:
            current_count += len(source_records)
        for index, record in enumerate(source_records, start=1):
            record_id = annotation_id(record, source, index)
            if record_id in seen:
                continue
            records.append(absolute_video(record, source, media_dir))
            seen.add(record_id)
    if current_count == 0:
        raise RuntimeError("no new mined or generated annotations were available for this iteration")
    return records


def patch_train_config(
    config: dict[str, Any],
    *,
    train_dir: Path,
    train_annotations: Path,
    train_media_dir: Path,
    val_annotations: Path,
    val_media_dir: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Patch run-specific fields in a Cosmos Reason training config."""
    config["results_dir"] = str(train_dir)
    config.setdefault("train", {})
    config["train"]["output_dir"] = str(train_dir)
    config.setdefault("policy", {})
    config["policy"]["model_name_or_path"] = str(checkpoint_path)
    config.setdefault("custom", {})
    config["custom"].setdefault("train_dataset", {})
    config["custom"]["train_dataset"]["annotation_path"] = str(train_annotations)
    config["custom"]["train_dataset"]["media_path"] = str(train_media_dir)
    config["custom"].setdefault("val_dataset", {})
    config["custom"]["val_dataset"]["annotation_path"] = str(val_annotations)
    config["custom"]["val_dataset"]["media_path"] = str(val_media_dir)
    return config


def training_checkpoint(
    config: dict[str, Any],
    workspace: Path,
    run_dir: Path,
    iteration: int,
) -> Path:
    """Select the baseline or previous evaluated checkpoint for this iteration."""
    cosmos_reason = require_mapping(config, "cosmos_reason")
    continual_model = cosmos_reason.get("continual_model", False)
    if not isinstance(continual_model, bool):
        raise ValueError("cosmos_reason.continual_model must be true or false")
    if iteration == 1 or not continual_model:
        return existing_absolute_path(
            require_string(cosmos_reason, "cosmos_reason.baseline_model_path"),
            workspace,
            "cosmos_reason.baseline_model_path",
            "path",
        )

    previous_evaluate_toml = run_dir / f"iter_{iteration - 1}" / "evaluate" / "specs" / "evaluate.toml"
    if not previous_evaluate_toml.is_file():
        raise FileNotFoundError(
            f"previous iteration evaluation config does not exist: {previous_evaluate_toml}"
        )
    previous_config = load_toml(previous_evaluate_toml)
    model = previous_config.get("model")
    checkpoint_value = model.get("model_name") if isinstance(model, dict) else None
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise ValueError(f"{previous_evaluate_toml}: missing model.model_name")
    checkpoint_path = absolute_path(checkpoint_value)
    path_in_workspace(checkpoint_path, run_dir, "previous iteration checkpoint")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"previous iteration checkpoint does not exist: {checkpoint_path}")
    return checkpoint_path


def generate_train_toml(
    workspace: Path,
    workflow_yaml: Path,
    run_dir: Path,
    *,
    iteration: int,
    train_annotations: Path,
    checkpoint_path: Path,
) -> Path:
    """Write the Cosmos Reason training TOML for one iteration."""
    config = load_yaml(workflow_yaml)
    kpi_dataset = require_mapping(config, "kpi_dataset")
    mode = data_generation_mode(config)
    train_dataset = optional_mapping(config, "train_dataset")
    cosmos_reason = require_mapping(config, "cosmos_reason")
    base_train_toml = existing_absolute_path(
        require_string(cosmos_reason, "cosmos_reason.base_train_toml"),
        workspace,
        "cosmos_reason.base_train_toml",
        "file",
    )
    if genai_enabled(mode):
        # Generated annotations use absolute paths under per-iteration PAIDF outputs,
        # while combined mode can also include absolute paths from the source dataset.
        train_media_dir = workspace
    else:
        if train_dataset is None:
            raise ValueError("train_dataset is required for mining-only training")
        train_media_dir = existing_absolute_path(
            require_string(train_dataset, "train_dataset.media_dir"),
            workspace,
            "train_dataset.media_dir",
            "dir",
        )
    val_annotations = existing_absolute_path(
        require_string(kpi_dataset, "kpi_dataset.annotations_path"),
        workspace,
        "kpi_dataset.annotations_path",
        "file",
    )
    val_media_dir = existing_absolute_path(
        require_string(kpi_dataset, "kpi_dataset.media_dir"),
        workspace,
        "kpi_dataset.media_dir",
        "dir",
    )
    train_dir = run_dir / f"iter_{iteration}" / "train"
    output_path = train_dir / "specs" / "train.toml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patched = patch_train_config(
        load_toml(base_train_toml),
        train_dir=train_dir,
        train_annotations=train_annotations,
        train_media_dir=train_media_dir,
        val_annotations=val_annotations,
        val_media_dir=val_media_dir,
        checkpoint_path=checkpoint_path,
    )
    output_path.write_text(dump_toml(patched), encoding="utf-8")
    return output_path


def prepare_training(
    workspace: Path,
    workflow_yaml: Path,
    run_dir: Path,
    iteration: int,
) -> dict[str, Path]:
    """Build current/accumulated annotations and the training TOML."""
    if iteration < 1:
        raise ValueError("iteration must be >= 1")
    iteration_dir = run_dir / f"iter_{iteration}"
    mined_annotations = iteration_dir / "mining" / "mined_train_annotations.json"
    generated_annotations = iteration_dir / "genai" / "generated_llava_annotations.json"
    train_annotations = iteration_dir / "train" / "train_annotations.json"

    config = load_yaml(workflow_yaml)
    mode = data_generation_mode(config)
    outputs: dict[str, Path] = {}
    current_annotations: list[Path] = []
    if mining_enabled(mode):
        mined_records = build_llava_records(
            iteration_dir / "mining" / "mined_neighbors.parquet",
            run_dir / "embedding_parquets" / "train" / "embeddings.parquet",
            run_dir / "cosmos_embed_output" / "train" / "lookup.parquet",
        )
        write_json_array(mined_annotations, mined_records)
        current_annotations.append(mined_annotations)
        outputs["mined_annotations"] = mined_annotations
    if genai_enabled(mode) and generated_annotations.is_file():
        current_annotations.append(generated_annotations)
        outputs["generated_annotations"] = generated_annotations

    previous_annotations: Path | None = None
    previous_media_dir: Path | None = None
    if iteration > 1:
        previous_annotations = run_dir / f"iter_{iteration - 1}" / "train" / "train_annotations.json"
    elif mode == "genai":
        train_dataset = optional_mapping(config, "train_dataset")
        if train_dataset is not None:
            previous_annotations = existing_absolute_path(
                require_string(train_dataset, "train_dataset.annotations_path"),
                workspace,
                "train_dataset.annotations_path",
                "file",
            )
            previous_media_dir = existing_absolute_path(
                require_string(train_dataset, "train_dataset.media_dir"),
                workspace,
                "train_dataset.media_dir",
                "dir",
            )
    assembled_records = assemble_annotations(
        previous_annotations,
        current_annotations,
        previous_media_dir,
    )
    write_json_array(train_annotations, assembled_records)

    checkpoint_path = training_checkpoint(config, workspace, run_dir, iteration)
    train_toml = generate_train_toml(
        workspace,
        workflow_yaml,
        run_dir,
        iteration=iteration,
        train_annotations=train_annotations,
        checkpoint_path=checkpoint_path,
    )
    outputs.update(
        {
            "train_annotations": train_annotations,
            "checkpoint": checkpoint_path,
            "toml": train_toml,
        }
    )
    return outputs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--workflow-yaml", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--iteration", required=True, type=int)
    return parser.parse_args()


def main() -> int:
    """Prepare all Cosmos Reason training inputs for one iteration."""
    args = parse_args()
    workspace = absolute_path(args.workspace)
    workflow_yaml = absolute_path(args.workflow_yaml)
    run_dir = absolute_path(args.run_dir)
    if not workspace.is_dir():
        raise NotADirectoryError(f"workspace does not exist: {workspace}")
    if not workflow_yaml.is_file():
        raise FileNotFoundError(f"workflow YAML does not exist: {workflow_yaml}")
    path_in_workspace(workflow_yaml, workspace, "workflow YAML")
    path_in_workspace(run_dir, workspace, "run directory")

    outputs = prepare_training(workspace, workflow_yaml, run_dir, args.iteration)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
