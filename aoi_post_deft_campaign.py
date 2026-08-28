#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the incumbent-preserving four-arm AOI post-DEFT campaign.

The launcher keeps site paths on the command line, writes the fully resolved
configuration before submitting work, and runs the two conservative warm-start
arms plus the two from-scratch-on-DEFT-data arms used by the 2026-08-14 study.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import importlib
import json
import os
import traceback
from pathlib import Path


WARM_PARAMS = [
    "train.num_epochs",
    "train.optim.lr",
    "train.optim.weight_decay",
    "train.optim.momentum",
    "dataset.classify.fpratio_sampling",
    "train.classify.cls_weight[1]",
]
WARM_RANGES = {
    "train.num_epochs": {"valid_min": 1, "valid_max": 5},
    "train.optim.lr": {"valid_min": 1e-8, "valid_max": 2e-6},
    "train.optim.weight_decay": {"valid_min": 0.0, "valid_max": 0.03},
    "train.optim.momentum": {"valid_min": 0.85, "valid_max": 0.95},
    "dataset.classify.fpratio_sampling": {"valid_min": 0.15, "valid_max": 0.45},
    "train.classify.cls_weight[1]": {"valid_min": 2.0, "valid_max": 15.0},
}
WARM_INITIAL = {
    "train.num_epochs": 1,
    "train.optim.lr": 2e-7,
    "train.optim.weight_decay": 0.01,
    "train.optim.momentum": 0.9,
    "dataset.classify.fpratio_sampling": 0.2,
    "train.classify.cls_weight[1]": 10.0,
}

SCRATCH_PARAMS = [
    "train.classify.cls_weight[1]",
    "dataset.classify.fpratio_sampling",
    "train.optim.lr",
    "train.optim.weight_decay",
    "train.optim.momentum",
    "model.classify.learnable_difference_modules",
    "model.classify.train_margin_euclid",
    "model.classify.eval_margin",
    "dataset.classify.augmentation_config.random_color.brightness",
    "dataset.classify.augmentation_config.random_color.contrast",
    "dataset.classify.augmentation_config.random_color.saturation",
    "dataset.classify.augmentation_config.random_color.hue",
    "dataset.classify.augmentation_config.random_rotate.rotate_probability",
    "dataset.classify.augmentation_config.random_flip.hflip_probability",
    "dataset.classify.augmentation_config.random_flip.vflip_probability",
]
SCRATCH_RANGES = {
    "train.classify.cls_weight[1]": {"valid_min": 2.0, "valid_max": 25.0},
    "dataset.classify.fpratio_sampling": {"valid_min": 0.15, "valid_max": 0.98},
    "train.optim.lr": {"valid_min": 5e-6, "valid_max": 2e-4},
    "train.optim.weight_decay": {"valid_min": 0.005, "valid_max": 1.0},
    "train.optim.momentum": {"valid_min": 0.1, "valid_max": 0.95},
    "model.classify.learnable_difference_modules": {"valid_min": 1, "valid_max": 4},
    "model.classify.train_margin_euclid": {"valid_min": 1.0, "valid_max": 3.0},
    "model.classify.eval_margin": {"valid_min": 0.2, "valid_max": 2.5},
    "dataset.classify.augmentation_config.random_color.brightness": {
        "valid_min": 0.1,
        "valid_max": 2.0,
    },
    "dataset.classify.augmentation_config.random_color.contrast": {
        "valid_min": 0.1,
        "valid_max": 2.0,
    },
    "dataset.classify.augmentation_config.random_color.saturation": {
        "valid_min": 0.1,
        "valid_max": 2.0,
    },
    "dataset.classify.augmentation_config.random_color.hue": {
        "valid_min": 1e-7,
        "valid_max": 0.3,
    },
    "dataset.classify.augmentation_config.random_rotate.rotate_probability": {
        "valid_min": 0.1,
        "valid_max": 0.8,
    },
    "dataset.classify.augmentation_config.random_flip.hflip_probability": {
        "valid_min": 0.3,
        "valid_max": 0.7,
    },
    "dataset.classify.augmentation_config.random_flip.vflip_probability": {
        "valid_min": 0.3,
        "valid_max": 0.7,
    },
}
SCRATCH50_INITIAL = {
    "train.classify.cls_weight[1]": 2.2,
    "dataset.classify.fpratio_sampling": 0.33,
    "train.optim.lr": 0.000014194559202602006,
    "train.optim.weight_decay": 0.055,
    "train.optim.momentum": 0.11,
    "model.classify.learnable_difference_modules": 1,
    "model.classify.train_margin_euclid": 2.0,
    "model.classify.eval_margin": 0.3,
    "dataset.classify.augmentation_config.random_color.brightness": 0.54,
    "dataset.classify.augmentation_config.random_color.contrast": 0.32757810179178865,
    "dataset.classify.augmentation_config.random_color.saturation": 0.54,
    "dataset.classify.augmentation_config.random_color.hue": 1e-7,
    "dataset.classify.augmentation_config.random_rotate.rotate_probability": 0.45,
    "dataset.classify.augmentation_config.random_flip.hflip_probability": 0.33,
    "dataset.classify.augmentation_config.random_flip.vflip_probability": 0.33,
}
SCRATCH100_INITIAL = {
    "train.classify.cls_weight[1]": 20.37832545008366,
    "dataset.classify.fpratio_sampling": 0.33,
    "train.optim.lr": 5.5e-6,
    "train.optim.weight_decay": 0.9,
    "train.optim.momentum": 0.5803693883720338,
    "model.classify.learnable_difference_modules": 4,
    "model.classify.train_margin_euclid": 2.0,
    "model.classify.eval_margin": 0.3,
    "dataset.classify.augmentation_config.random_color.brightness": 0.11,
    "dataset.classify.augmentation_config.random_color.contrast": 0.11,
    "dataset.classify.augmentation_config.random_color.saturation": 0.13702394749695712,
    "dataset.classify.augmentation_config.random_color.hue": 1e-7,
    "dataset.classify.augmentation_config.random_rotate.rotate_probability": 0.4233215549737651,
    "dataset.classify.augmentation_config.random_flip.hflip_probability": 0.33,
    "dataset.classify.augmentation_config.random_flip.vflip_probability": 0.63,
}


def _deep_merge(target: dict, override: dict) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def build_tasks(args: argparse.Namespace) -> list[dict]:
    common = {
        "algorithm": "bfbo",
        "incumbent": {
            "name": args.incumbent_name,
            "metric_value": args.incumbent_far,
            "checkpoint": args.incumbent_checkpoint,
            "metric_protocol": "validation threshold frozen and applied to KPI",
        },
        "max_regression": args.max_regression,
        "images_dir": args.images_dir,
        "seed": args.seed,
    }
    branches = [
        {
            "tag": "postdeft_warm_full_mix050",
            "mode": "warm",
            "freeze_backbone": False,
            "csv": args.mix50_csv,
            "sample_count": args.mix50_sample_count,
            "recommendations": args.warm_recommendations,
            "params": WARM_PARAMS,
            "ranges": WARM_RANGES,
            "initial": WARM_INITIAL,
            "interval_cap": 1,
        },
        {
            "tag": "postdeft_warm_head_mix050",
            "mode": "warm",
            "freeze_backbone": True,
            "csv": args.mix50_csv,
            "sample_count": args.mix50_sample_count,
            "recommendations": args.warm_recommendations,
            "params": WARM_PARAMS,
            "ranges": WARM_RANGES,
            "initial": WARM_INITIAL,
            "interval_cap": 1,
        },
        {
            "tag": "postdeft_scratch_mix050",
            "mode": "scratch",
            "freeze_backbone": False,
            "csv": args.mix50_csv,
            "sample_count": args.mix50_sample_count,
            "recommendations": args.scratch_recommendations,
            "params": SCRATCH_PARAMS,
            "ranges": SCRATCH_RANGES,
            "initial": SCRATCH50_INITIAL,
            "interval_cap": 10,
        },
        {
            "tag": "postdeft_scratch_mix100",
            "mode": "scratch",
            "freeze_backbone": False,
            "csv": args.mix100_csv,
            "sample_count": args.mix100_sample_count,
            "recommendations": args.scratch_recommendations,
            "params": SCRATCH_PARAMS,
            "ranges": SCRATCH_RANGES,
            "initial": SCRATCH100_INITIAL,
            "interval_cap": 10,
        },
    ]
    return [{**copy.deepcopy(common), **branch} for branch in branches]


def _task_spec(task: dict) -> dict:
    if task["mode"] == "warm":
        return {
            "train": {
                "num_epochs": 5,
                "validation_interval": 1,
                "checkpoint_interval": 1,
                "pretrained_model_path": task["incumbent"]["checkpoint"],
                "seed": task["seed"],
                "classify": {"loss": "ce", "cls_weight": [1.0, 10.0]},
                "optim": {
                    "optim": "adamw",
                    "policy": "linear",
                    "lr": 2e-7,
                    "weight_decay": 0.01,
                    "momentum": 0.9,
                },
            },
            "model": {
                "backbone": {"freeze_backbone": task["freeze_backbone"]},
                "classify": {
                    "learnable_difference_modules": 4,
                    "train_margin_euclid": 2.0,
                    "eval_margin": 0.3,
                },
            },
        }
    return {
        "train": {
            "num_epochs": 100,
            "validation_interval": 10,
            "checkpoint_interval": 10,
            "pretrained_model_path": None,
            "seed": task["seed"],
        },
        "model": {"backbone": {"freeze_backbone": False}},
    }


def run_branch(campaign_dir: str, task: dict) -> dict:
    campaign = Path(campaign_dir)
    tag = task["tag"]
    try:
        runner = importlib.import_module("automl_vcn_slurm_v2")
        runner.WORKSPACE = campaign / "branches" / tag
        runner.BASE_SPEC = copy.deepcopy(runner.BASE_SPEC)
        _deep_merge(runner.BASE_SPEC, _task_spec(task))
        runner.BASE_SPEC["dataset"]["classify"]["train_dataset"] = {
            "csv_path": task["csv"],
            "images_dir": task["images_dir"],
        }
        config = {
            "algorithm": task["algorithm"],
            "settings": {
                "num_recommendations": task["recommendations"],
                "train_sample_count": task["sample_count"],
                "metric": "far_pct",
                "run_baseline": False,
            },
            "metric": "far_pct",
            "direction": "minimize",
            "use_val_far_eval_fn": True,
            "hyperparameters": task["params"],
            "ranges": task["ranges"],
            "forced_initial_overrides": task["initial"],
            "interval_cap": task["interval_cap"],
            "incumbent": task["incumbent"],
            "max_regression": task["max_regression"],
        }
        result = runner.launch_one(
            tag,
            config,
            str(campaign / "search_summaries"),
        )
        outcome = {
            "tag": tag,
            "status": "complete",
            "best": (result or {}).get("best"),
            "final_evaluation": (result or {}).get("final_evaluation"),
            "promotion": (result or {}).get("promotion"),
        }
    except Exception as error:  # Preserve every branch result for recovery.
        outcome = {
            "tag": tag,
            "status": "error",
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    _atomic_json(campaign / "branch_summaries" / f"{tag}.json", outcome)
    return outcome


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--mix50-csv", required=True)
    parser.add_argument("--mix100-csv", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--incumbent-checkpoint", required=True)
    parser.add_argument("--incumbent-far", required=True, type=float)
    parser.add_argument("--incumbent-name", default="deft_winner")
    parser.add_argument("--mix50-sample-count", type=int, default=309)
    parser.add_argument("--mix100-sample-count", type=int, default=404)
    parser.add_argument("--warm-recommendations", type=int, default=8)
    parser.add_argument("--scratch-recommendations", type=int, default=16)
    parser.add_argument("--max-regression", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the resolved manifest without submitting any jobs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_workers < 1:
        raise ValueError("--max-workers must be positive")
    tasks = build_tasks(args)
    manifest = {
        "launcher": str(Path(__file__).resolve()),
        "pid": os.getpid(),
        "branches": len(tasks),
        "recommendations": sum(task["recommendations"] for task in tasks),
        "max_concurrent_branches": args.max_workers,
        "gpus_per_trial": 8,
        "nodes_per_trial": 1,
        "tasks": tasks,
    }
    _atomic_json(args.campaign_dir / "campaign_manifest.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    results: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(run_branch, str(args.campaign_dir), task): task
            for task in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            _atomic_json(args.campaign_dir / "campaign_progress.json", results)
    _atomic_json(args.campaign_dir / "campaign_complete.json", results)
    return 0 if all(item["status"] == "complete" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
