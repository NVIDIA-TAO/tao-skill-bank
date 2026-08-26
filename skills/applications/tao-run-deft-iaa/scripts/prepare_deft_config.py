# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize one immutable, run-specific IAA DEFT configuration.

The canonical templates remain unchanged under ``specs/``. This command copies
them into ``RESULTS_DIR/config`` and applies only values approved in the
pre-flight summary.  All paths written to the DEFT config are absolute host
paths, so later tool calls do not depend on a previous ``cd`` or ``export``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
import tempfile
from typing import Any

import yaml

from metric_contract import validate_contract


SPEC_NAMES = (
    "deft_config.yaml",
    "tao_spec.yaml",
    "text_embed_spec.yaml",
    "image_embed_spec.yaml",
    "mining_spec.yaml",
)
PINNED_PYT_IMAGE = "nvcr.io/nvstaging/tao/tao-toolkit-pyt:7.2.0-rc-53-multiarch"  # versions-key: images.tao_toolkit.pyt
PINNED_DS_IMAGE = "nvcr.io/nvstaging/tao/tao-toolkit-ds:7.2.0-rc-52-multiarch"  # versions-key: images.tao_toolkit.data_services


def _bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def _gpu_ids(raw: str, count: int) -> list[int]:
    try:
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--gpu-ids must be a comma-separated integer list") from exc
    if len(values) != count or len(set(values)) != len(values) or any(v < 0 for v in values):
        raise ValueError(
            "--gpu-ids must contain exactly --num-gpus distinct non-negative IDs"
        )
    return values


def _workspace_child(path: pathlib.Path, workspace: pathlib.Path, name: str) -> pathlib.Path:
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"{name} must be under --workspace {workspace}: {resolved}") from exc
    if relative == pathlib.Path("."):
        raise ValueError(f"{name} must be a child of --workspace, not the workspace itself")
    return resolved


def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty template: {path}")
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be an object: {path}")
    return payload


def _existing_path(path: pathlib.Path, name: str, *, directory: bool) -> pathlib.Path:
    resolved = path.expanduser().resolve()
    valid = resolved.is_dir() if directory else resolved.is_file()
    if not valid or (not directory and resolved.stat().st_size == 0):
        kind = "directory" if directory else "non-empty file"
        raise ValueError(f"{name} must be an existing {kind}: {resolved}")
    return resolved


def _python_tree_sha256(root: pathlib.Path) -> str:
    files = sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    if not files:
        raise ValueError(f"bundled IAA runtime contains no Python files: {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_yaml(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _copy_atomic(source: pathlib.Path, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    skill_root = pathlib.Path(__file__).resolve().parent.parent
    templates = skill_root / "specs"
    runtime_dir = skill_root / "scripts" / "iaa_deft"
    runtime_sha256 = _python_tree_sha256(runtime_dir)
    workspace = args.workspace.expanduser().resolve()
    if not workspace.is_dir() or workspace == pathlib.Path(workspace.anchor):
        raise ValueError(f"--workspace must be an existing non-root directory: {workspace}")
    results_dir = _workspace_child(args.results_dir, workspace, "--results-dir")
    dataset_root = _workspace_child(args.dataset_root, workspace, "--dataset-root")
    images_archive = _existing_path(
        args.images_archive, "--images-archive", directory=False
    )
    metadata_archive = _existing_path(
        args.metadata_archive, "--metadata-archive", directory=False
    )
    checksums_file = (
        _existing_path(args.checksums_file, "--checksums-file", directory=False)
        if args.checksums_file is not None
        else None
    )
    if dataset_root.parent == workspace:
        raise ValueError(
            "--dataset-root must be nested below a workspace data directory "
            "(for example <workspace>/data/iaa_v31_tao_ft)"
        )
    if results_dir in dataset_root.parents or dataset_root in results_dir.parents:
        raise ValueError("--results-dir and --dataset-root must not contain one another")
    if (results_dir / "deft_state.json").exists():
        raise ValueError(
            "refusing to rewrite config for an initialized run; resume from "
            "deft_state.json or choose a fresh --results-dir"
        )
    config_dir = results_dir / "config"

    positive = {
        "max_iterations": args.max_iterations,
        "training_epochs": args.training_epochs,
        "num_gpus": args.num_gpus,
        "mining_topn": args.mining_topn,
        "target_query_count": args.target_query_count,
    }
    invalid = {key: value for key, value in positive.items() if value < 1}
    if invalid:
        raise ValueError(
            "positive integers required: "
            + ", ".join(f"{key}={value}" for key, value in invalid.items())
        )
    if not 0.0 <= args.replay_fraction <= 1.0:
        raise ValueError("--replay-fraction must be in [0, 1]")
    if args.knn_metric not in {"cosine", "euclidean"}:
        raise ValueError("--knn-metric must be cosine or euclidean")
    # ``gpu_ids`` is an allocation in the launcher/host namespace. Container
    # runtimes expose that allocation as a dense zero-based CUDA namespace, so
    # TAO must never receive the host ordinals directly.
    host_gpu_ids = _gpu_ids(args.gpu_ids, args.num_gpus)
    container_gpu_ids = list(range(args.num_gpus))
    metric_contract = validate_contract(
        {
            "metric_name": args.metric_name,
            "query_type": args.metric_query_type,
            "op": args.metric_op,
            "target": args.metric_target,
        }
    )

    deft = _load_yaml(templates / "deft_config.yaml")
    tao = _load_yaml(templates / "tao_spec.yaml")

    deft["experiment"].update(
        {
            "results_path": str(results_dir),
            "train_config": str(config_dir / "tao_spec.yaml"),
            "eval_config": str(config_dir / "tao_spec.yaml"),
            "visualize": args.visualize,
            "visualize_embeddings": args.visualize_embeddings,
        }
    )
    deft["iteration"].update({"start": 1, "end": args.max_iterations})
    deft["training"].update(
        {
            "continual_model": args.continual_model,
            "continual_dataset": args.continual_dataset,
        }
    )
    deft["mining"].update(
        {"topn": args.mining_topn, "knn_metric": args.knn_metric}
    )
    deft["mining"].setdefault("history_aware", {}).update(
        {"enabled": args.history_aware, "replay_fraction": args.replay_fraction}
    )
    deft["gap_analysis"].update(
        {
            "metric_name": args.metric_name,
            "target_query_count": args.target_query_count,
        }
    )
    iaa = deft["iaa"]
    iaa.update(
        {
            "pool_pairs_source_file": str(dataset_root / "train_pairs.json"),
            "eval_pairs_source_file": str(dataset_root / "val_pairs.json"),
            "train_image_dir": str(dataset_root / "images"),
            "train_caption_dir": str(dataset_root / "captions"),
            "source_image_dir": str(dataset_root / "images"),
            "source_caption_dir": str(dataset_root / "captions"),
            "eval_image_dir": str(dataset_root / "images"),
            "eval_caption_dir": str(dataset_root / "captions"),
        }
    )
    if iaa.get("train_pairs_source_file"):
        iaa["train_pairs_source_file"] = str(dataset_root / "train_pairs.json")

    tao["train"].update(
        {
            "num_epochs": args.training_epochs,
            "num_gpus": args.num_gpus,
            "gpu_ids": container_gpu_ids,
        }
    )
    tao["evaluate"].update(
        {"num_gpus": args.num_gpus, "gpu_ids": container_gpu_ids}
    )

    _atomic_yaml(config_dir / "deft_config.yaml", deft)
    _atomic_yaml(config_dir / "tao_spec.yaml", tao)
    for name in SPEC_NAMES[2:]:
        _copy_atomic(templates / name, config_dir / name)
    _atomic_json(
        config_dir / "approval.json",
        {
            "schema_version": "3",
            "workflow": "tao-run-deft-iaa",
            "workspace": str(workspace),
            "results_dir": str(results_dir),
            "dataset_root": str(dataset_root),
            "iaa_deft_bundle_sha256": runtime_sha256,
            "images_archive": str(images_archive),
            "metadata_archive": str(metadata_archive),
            "checksums_file": str(checksums_file) if checksums_file else None,
            "requires_hf_token": args.requires_hf_token,
            "max_iterations": args.max_iterations,
            "host_gpu_ids": host_gpu_ids,
            "container_gpu_ids": container_gpu_ids,
            "metric_contract": metric_contract,
            "pyt_image": PINNED_PYT_IMAGE,
            "ds_image": PINNED_DS_IMAGE,
        },
    )

    return {
        "config_dir": str(config_dir),
        "deft_config": str(config_dir / "deft_config.yaml"),
        "tao_spec": str(config_dir / "tao_spec.yaml"),
        "max_iterations": args.max_iterations,
        "training_epochs": args.training_epochs,
        "num_gpus": args.num_gpus,
        "gpu_ids": host_gpu_ids,
        "container_gpu_ids": container_gpu_ids,
        "requires_hf_token": args.requires_hf_token,
        "iaa_deft_bundle_sha256": runtime_sha256,
        "approval_manifest": str(config_dir / "approval.json"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=pathlib.Path)
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--dataset-root", required=True, type=pathlib.Path)
    parser.add_argument("--images-archive", required=True, type=pathlib.Path)
    parser.add_argument("--metadata-archive", required=True, type=pathlib.Path)
    parser.add_argument("--checksums-file", type=pathlib.Path)
    parser.add_argument(
        "--requires-hf-token",
        action="store_true",
        help="Approve forwarding the existing HF_TOKEN by environment name to model stages.",
    )
    parser.add_argument("--max-iterations", required=True, type=int)
    parser.add_argument("--training-epochs", default=1, type=int)
    parser.add_argument("--num-gpus", default=1, type=int)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--mining-topn", default=25, type=int)
    parser.add_argument("--knn-metric", default="cosine")
    parser.add_argument("--target-query-count", default=10000, type=int)
    parser.add_argument("--history-aware", default=True, type=_bool)
    parser.add_argument("--replay-fraction", default=0.20, type=float)
    parser.add_argument("--continual-dataset", default=True, type=_bool)
    parser.add_argument("--continual-model", default=False, type=_bool)
    parser.add_argument("--visualize", default=True, type=_bool)
    parser.add_argument("--visualize-embeddings", default=True, type=_bool)
    parser.add_argument("--metric-name", default="Rank-1")
    parser.add_argument("--metric-query-type", default="medium")
    parser.add_argument("--metric-op", default=">=", choices=(">=", ">", "<=", "<"))
    parser.add_argument("--metric-target", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = materialize(args)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(f"prepare_deft_config: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
