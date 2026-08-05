#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate patched Cosmos Reason train/evaluate TOMLs for workflow stages."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from workflow_common import (
    absolute_path,
    dump_toml,
    existing_absolute_path,
    find_results_json,
    load_toml,
    load_yaml,
    path_in_workspace,
    require_mapping,
    require_string,
)


def patch_evaluate_config(
    config: dict[str, Any],
    *,
    results_dir: Path,
    annotation_path: Path,
    media_dir: Path,
    checkpoint_path: str,
) -> dict[str, Any]:
    """Patch only run-specific fields in a Cosmos Reason evaluate config."""
    config["results_dir"] = str(results_dir)
    config.setdefault("dataset", {})
    config["dataset"]["annotation_path"] = str(annotation_path)
    config["dataset"]["media_dir"] = str(media_dir)
    config.setdefault("model", {})
    config["model"]["model_name"] = checkpoint_path
    return config


def patch_train_config(
    config: dict[str, Any],
    *,
    train_dir: Path,
    train_annotations: Path,
    train_media_dir: Path,
    val_annotations: Path,
    val_media_dir: Path,
    checkpoint_path: str,
) -> dict[str, Any]:
    """Patch only run-specific fields in a Cosmos Reason train config."""
    config["results_dir"] = str(train_dir)
    config.setdefault("train", {})
    config["train"]["output_dir"] = str(train_dir)
    config.setdefault("policy", {})
    config["policy"]["model_name_or_path"] = checkpoint_path
    config.setdefault("custom", {})
    config["custom"].setdefault("train_dataset", {})
    config["custom"]["train_dataset"]["annotation_path"] = str(train_annotations)
    config["custom"]["train_dataset"]["media_path"] = str(train_media_dir)
    config["custom"].setdefault("val_dataset", {})
    config["custom"]["val_dataset"]["annotation_path"] = str(val_annotations)
    config["custom"]["val_dataset"]["media_path"] = str(val_media_dir)
    return config


def generate_evaluate_toml(
    workspace: Path,
    workflow_yaml: Path,
    run_dir: Path,
    *,
    output_dir: Path,
    checkpoint_path: str | None,
) -> Path:
    """Generate an evaluate TOML for baseline or per-iteration evaluation."""
    config = load_yaml(workflow_yaml)
    kpi_dataset = require_mapping(config, "kpi_dataset")
    cosmos_reason = require_mapping(config, "cosmos_reason")
    annotation_path = existing_absolute_path(
        require_string(kpi_dataset, "kpi_dataset.annotations_path"),
        workspace,
        "kpi_dataset.annotations_path",
        "file",
    )
    media_dir = existing_absolute_path(
        require_string(kpi_dataset, "kpi_dataset.media_dir"),
        workspace,
        "kpi_dataset.media_dir",
        "dir",
    )
    base_evaluate_toml = existing_absolute_path(
        require_string(cosmos_reason, "cosmos_reason.base_evaluate_toml"),
        workspace,
        "cosmos_reason.base_evaluate_toml",
        "file",
    )
    if checkpoint_path is None:
        checkpoint_path = str(
            existing_absolute_path(
                require_string(cosmos_reason, "cosmos_reason.baseline_model_path"),
                workspace,
                "cosmos_reason.baseline_model_path",
                "path",
            )
        )
    path_in_workspace(output_dir, workspace, "evaluate output directory")
    output_path = output_dir / "specs" / "evaluate.toml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patched = patch_evaluate_config(
        load_toml(base_evaluate_toml),
        results_dir=output_dir,
        annotation_path=annotation_path,
        media_dir=media_dir,
        checkpoint_path=checkpoint_path,
    )
    output_path.write_text(dump_toml(patched), encoding="utf-8")
    path_in_workspace(output_path, run_dir, "generated evaluate TOML")
    return output_path


def generate_train_toml(
    workspace: Path,
    workflow_yaml: Path,
    run_dir: Path,
    *,
    iteration: int,
    train_annotations: Path,
    checkpoint_path: str,
) -> Path:
    """Generate a train TOML for one workflow iteration."""
    config = load_yaml(workflow_yaml)
    kpi_dataset = require_mapping(config, "kpi_dataset")
    train_dataset = require_mapping(config, "train_dataset")
    cosmos_reason = require_mapping(config, "cosmos_reason")
    base_train_toml = existing_absolute_path(
        require_string(cosmos_reason, "cosmos_reason.base_train_toml"),
        workspace,
        "cosmos_reason.base_train_toml",
        "file",
    )
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
    train_annotations = absolute_path(train_annotations)
    if not train_annotations.is_file():
        raise FileNotFoundError(f"train annotations do not exist: {train_annotations}")
    path_in_workspace(train_annotations, run_dir, "train annotations")
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


def latest_safetensors_checkpoint(train_dir: Path) -> Path:
    """Return the latest epoch checkpoint under a completed Cosmos Reason train dir."""
    timestamp_re = re.compile(r"^\d{14}$")
    run_dirs = [
        child for child in train_dir.iterdir()
        if child.is_dir() and timestamp_re.match(child.name)
    ]
    if not run_dirs:
        raise FileNotFoundError(f"no timestamped train run directories found under {train_dir}")
    latest_run = max(run_dirs, key=lambda path: path.name)
    safetensors_dir = latest_run / "safetensors"
    if not safetensors_dir.is_dir():
        raise FileNotFoundError(f"safetensors directory not found: {safetensors_dir}")
    epoch_re = re.compile(r"^epoch_(\d+)$")
    epochs: list[tuple[int, Path]] = []
    for child in safetensors_dir.iterdir():
        match = epoch_re.match(child.name)
        if child.is_dir() and match:
            epochs.append((int(match.group(1)), child))
    if not epochs:
        raise FileNotFoundError(f"no epoch_<N> checkpoints found under {safetensors_dir}")
    return max(epochs, key=lambda item: item[0])[1]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workspace", required=True, type=Path)
    common.add_argument("--workflow-yaml", required=True, type=Path)
    common.add_argument("--run-dir", required=True, type=Path)

    baseline = subparsers.add_parser("baseline-evaluate", parents=[common])
    baseline.add_argument(
        "--checkpoint-path",
        help="Checkpoint to evaluate. Defaults to cosmos_reason.baseline_model_path.",
    )

    evaluate = subparsers.add_parser("iteration-evaluate", parents=[common])
    evaluate.add_argument("--iteration", required=True, type=int)
    evaluate.add_argument("--checkpoint-path", required=True)

    train = subparsers.add_parser("iteration-train", parents=[common])
    train.add_argument("--iteration", required=True, type=int)
    train.add_argument("--train-annotations", required=True, type=Path)
    train.add_argument("--checkpoint-path", required=True)

    checkpoint = subparsers.add_parser("latest-checkpoint")
    checkpoint.add_argument("--train-dir", required=True, type=Path)

    results = subparsers.add_parser("find-results-json")
    results.add_argument("--evaluate-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    if args.command == "latest-checkpoint":
        print(latest_safetensors_checkpoint(absolute_path(args.train_dir)))
        return 0
    if args.command == "find-results-json":
        print(find_results_json(absolute_path(args.evaluate_dir)))
        return 0

    workspace = absolute_path(args.workspace)
    workflow_yaml = absolute_path(args.workflow_yaml)
    run_dir = absolute_path(args.run_dir)
    if not workspace.is_dir():
        raise NotADirectoryError(f"workspace does not exist: {workspace}")
    if not workflow_yaml.is_file():
        raise FileNotFoundError(f"workflow YAML does not exist: {workflow_yaml}")
    path_in_workspace(workflow_yaml, workspace, "workflow YAML")
    path_in_workspace(run_dir, workspace, "run directory")

    if args.command == "baseline-evaluate":
        output_dir = run_dir / "baseline" / "evaluate"
        output_path = generate_evaluate_toml(
            workspace,
            workflow_yaml,
            run_dir,
            output_dir=output_dir,
            checkpoint_path=args.checkpoint_path,
        )
    elif args.command == "iteration-evaluate":
        output_dir = run_dir / f"iter_{args.iteration}" / "evaluate"
        output_path = generate_evaluate_toml(
            workspace,
            workflow_yaml,
            run_dir,
            output_dir=output_dir,
            checkpoint_path=args.checkpoint_path,
        )
    else:
        output_path = generate_train_toml(
            workspace,
            workflow_yaml,
            run_dir,
            iteration=args.iteration,
            train_annotations=args.train_annotations,
            checkpoint_path=args.checkpoint_path,
        )
    print(f"toml: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
