#!/usr/bin/env python3
"""Run one LAM SegFormer AutoML or fixed DEFT/control track on SLURM."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import yaml


LOCAL_ROOT = Path(
    "/localhome/local-rarunachalam/workspace/"
    "lam_segformer_bayes_deft_20260820_231724"
)
REMOTE_ROOT = Path(
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
    "lam_segformer_bayes_deft_20260820_231724"
)
SOURCE_DATA = Path(
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/lam_research"
)
SKILL_BANK = LOCAL_ROOT / "skill_bank_snapshot"
SKILL_DIR = SKILL_BANK / "skills/models/tao-train-segformer"
IMAGE = "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt"

BACKBONES = {
    "fan_base": {
        "type": "fan_base_16_p4_hybrid",
        "ptm": REMOTE_ROOT / "inputs/ptms/fan_base/fan_base_backbone_stripped.pth",
    },
    "fan_large": {
        "type": "fan_large_16_p4_hybrid",
        "ptm": REMOTE_ROOT / "inputs/ptms/fan_large/fan_large_backbone_stripped.pth",
    },
    "mit_b5": {
        "type": "mit_b5",
        "ptm": REMOTE_ROOT / "inputs/ptms/mit_b5/mit_b5_backbone_stripped.pth",
    },
}

DATASETS = {
    "original": {"root": SOURCE_DATA, "samples": 316},
    "mix50": {"root": REMOTE_ROOT / "datasets/deft_mix50", "samples": 474},
    "mix100": {"root": REMOTE_ROOT / "datasets/deft_mix100", "samples": 632},
    "deft25": {
        "root": REMOTE_ROOT / "datasets/deft_model_v1_mix25",
        "samples": 395,
        "per_backbone": True,
    },
}

SEARCH_PARAMETERS = [
    "dataset.segment.augmentation.random_color.brightness",
    "dataset.segment.augmentation.random_color.color_probability",
    "dataset.segment.augmentation.random_color.contrast",
    "dataset.segment.augmentation.random_color.hue",
    "dataset.segment.augmentation.random_color.saturation",
    "dataset.segment.augmentation.random_flip.hflip_probability",
    "train.optim.lr",
    "train.optim.weight_decay",
]

SEARCH_RANGES = {
    "dataset.segment.augmentation.random_color.brightness": {
        "valid_min": 0.1,
        "valid_max": 0.5,
    },
    "dataset.segment.augmentation.random_color.color_probability": {
        "valid_min": 0.3,
        "valid_max": 0.8,
    },
    "dataset.segment.augmentation.random_color.contrast": {
        "valid_min": 0.1,
        "valid_max": 0.5,
    },
    "dataset.segment.augmentation.random_color.hue": {
        "valid_min": 0.05,
        "valid_max": 0.2,
    },
    "dataset.segment.augmentation.random_color.saturation": {
        "valid_min": 0.1,
        "valid_max": 0.5,
    },
    "dataset.segment.augmentation.random_flip.hflip_probability": {
        "valid_min": 0.3,
        "valid_max": 0.7,
    },
    "train.optim.lr": {"valid_min": 1.0e-5, "valid_max": 5.0e-4},
    "train.optim.weight_decay": {"valid_min": 0.001, "valid_max": 0.05},
}

CONTROL_VALUES = {
    "dataset.segment.augmentation.random_color.brightness": 0.3,
    "dataset.segment.augmentation.random_color.color_probability": 0.5,
    "dataset.segment.augmentation.random_color.contrast": 0.3,
    "dataset.segment.augmentation.random_color.hue": 0.3,
    "dataset.segment.augmentation.random_color.saturation": 0.3,
    "dataset.segment.augmentation.random_flip.hflip_probability": 0.5,
    "train.optim.lr": 6.0e-5,
    "train.optim.weight_decay": 0.01,
}


def dataset_root(backbone: str, dataset: str) -> Path:
    data = DATASETS[dataset]
    root = Path(data["root"])
    return root / backbone if data.get("per_backbone") else root


def build_spec(backbone: str, dataset: str) -> dict:
    template = SKILL_DIR / "references/spec_template_train.yaml"
    spec = yaml.safe_load(template.read_text())
    model = BACKBONES[backbone]
    data = DATASETS[dataset]

    spec["model_name"] = f"lam_{backbone}_{dataset}"
    spec["encryption_key"] = "tlt_encode"
    spec["wandb"]["enable"] = False
    spec["model"]["backbone"].update(
        {
            "type": model["type"],
            "pretrained_backbone_path": str(model["ptm"]),
            "freeze_backbone": False,
        }
    )
    spec["dataset"]["segment"].update(
        {
            "root_dir": str(dataset_root(backbone, dataset)),
            "num_classes": 4,
            "img_size": 1024,
            "batch_size": 1,
            "workers": 8,
            "label_transform": "None",
            "palette": [
                {
                    "label_id": 0,
                    "mapping_class": "background",
                    "rgb": [0],
                    "seg_class": "background",
                },
                {
                    "label_id": 1,
                    "mapping_class": "mask_height_1",
                    "rgb": [85],
                    "seg_class": "mask_height_1",
                },
                {
                    "label_id": 2,
                    "mapping_class": "mask_height_2",
                    "rgb": [170],
                    "seg_class": "mask_height_2",
                },
                {
                    "label_id": 3,
                    "mapping_class": "trench_depth",
                    "rgb": [255],
                    "seg_class": "trench_depth",
                },
            ],
        }
    )
    # torchvision ColorJitter constrains hue to [0, 0.5]; the historical 0.3
    # remains valid, while the search uses the narrower reviewed [0.05, 0.2].
    spec["train"].update(
        {
            "num_gpus": 8,
            "gpu_ids": list(range(8)),
            "num_nodes": 1,
            "num_epochs": 20,
            "checkpoint_interval": 20,
            "validation_interval": 10,
            "use_distributed_sampler": True,
            "sync_batchnorm": True,
        }
    )
    spec["train"]["optim"].update(
        {"optim": "adamw", "policy": "linear", "lr": 6.0e-5, "weight_decay": 0.01}
    )
    spec["train"]["segment"].update(
        {"loss": "ce", "weights": [2.4690, 18.8720, 14.7437, 2.6614]}
    )
    spec["train"]["tensorboard"]["enabled"] = False
    return spec


def validate_bundle(backbone: str, dataset: str, output: Path) -> None:
    import jsonschema
    from tao_automl.runner import SkillContext

    spec = build_spec(backbone, dataset)
    model_schema = json.loads((SKILL_DIR / "schemas/train.schema.json").read_text())
    artifact_schema = json.loads(
        (
            SKILL_BANK
            / "skills/core/tao-artifacts/references/spec_bundle.schema.json"
        ).read_text()
    )
    # TAO's generated schema dialect intentionally uses dataclass types such
    # as ``bool`` and ``int`` rather than JSON-Schema spellings. Parse it and
    # exercise the runner's model runtime gate; use JSON Schema only for the
    # platform-neutral artifact bundle below.
    if not isinstance(model_schema.get("properties"), dict):
        raise TypeError("packaged train schema has no properties object")
    SkillContext(SKILL_DIR, "train").validate_runtime()
    bundle = {
        "network_arch": "segformer",
        "action": "train",
        "image": IMAGE,
        "mode": "config",
        "command": (
            "python3 /lustre/fsw/portfolios/edgeai/users/rarunachalam/"
            "lam_segformer_bayes_deft_20260820_231724/controller/"
            "segformer_entrypoint.py train -e {config_path}"
        ),
        "config_format": "yaml",
        "spec": spec,
        "declared_inputs": [
            {
                "spec_key": "dataset.segment.root_dir",
                "type": "folder",
                "uri": str(dataset_root(backbone, dataset)),
            },
            {
                "spec_key": "model.backbone.pretrained_backbone_path",
                "type": "file",
                "uri": str(BACKBONES[backbone]["ptm"]),
            },
        ],
        "declared_outputs": [{"spec_key": "results_dir", "type": "folder"}],
        "upload_excludes": ["inputs/"],
        "compute_shape": {"gpus": 8, "nodes": 1},
        "gpu_spec_key": "train.num_gpus",
    }
    jsonschema.validate(bundle, artifact_schema)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")


def pin_control_recommendation(rec) -> None:
    """Override the sampler's fixed-range inflation with exact control values."""
    rec.specs.update(CONTROL_VALUES)
    logging.info("pinned control recommendation %d to exact reviewed defaults", rec.id)


def run_track(kind: str, backbone: str, variant: str, retry_tag: str = "") -> dict:
    from tao_automl.runner import AutoMLRunner
    from tao_sdk.platforms.slurm import SlurmSDK

    if kind in {"automl", "deft_automl"}:
        dataset = "original" if kind == "automl" else "deft25"
        if variant not in {"bayesian", "bfbo", "bayesian_llm", "bfbo_llm"}:
            raise ValueError(variant)
        algorithm = "bfbo" if variant.startswith("bfbo") else "bayesian"
        llm_enabled = variant.endswith("_llm")
        recommendations = 3
        ranges = SEARCH_RANGES
    elif kind == "control":
        dataset = variant
        if dataset not in DATASETS:
            raise ValueError(dataset)
        algorithm = "bayesian"
        llm_enabled = False
        recommendations = 1
        ranges = {
            key: {"valid_min": value, "valid_max": value}
            for key, value in CONTROL_VALUES.items()
        }
    else:
        raise ValueError(kind)

    track_name = f"{kind}_{backbone}_{variant}"
    if retry_tag:
        track_name = f"{track_name}_{retry_tag}"
    workspace = LOCAL_ROOT / "workspaces" / track_name
    workspace.mkdir(parents=True, exist_ok=True)
    os.environ["TAO_SDK_STATE_DIR"] = str(LOCAL_ROOT / "sdk_state" / track_name)

    settings = {
        "algorithm": algorithm,
        "automl_max_recommendations": recommendations,
        "metric": "val_miou",
        "direction": "maximize",
        "train_sample_count": DATASETS[dataset]["samples"],
        "run_baseline": False,
        "run_final_evaluation": False,
        "automl_delete_intermediate_ckpt": False,
        "session_id": (
            f"lam_segformer_{backbone}_{dataset}_automl_v1"
            if kind in {"automl", "deft_automl"}
            else f"lam_segformer_{backbone}_{dataset}_control_v1{retry_tag}"
        ),
        "experiment_id": (
            f"lam_segformer_{backbone}_{dataset}_automl_v1"
            if kind in {"automl", "deft_automl"}
            else f"lam_segformer_{backbone}_{dataset}_control_v1{retry_tag}"
        ),
    }
    if llm_enabled:
        endpoint = os.environ.get("base_url") or os.environ.get("AUTOML_LLM_ENDPOINT")
        model = os.environ.get("model") or os.environ.get("AUTOML_LLM_MODEL")
        api_key = os.environ.get("AUTOML_LLM_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        if not endpoint or not model or not api_key:
            raise RuntimeError("LLM endpoint, model, or API key missing from sourced config")
        settings.update(
            {
                "enable_llm_range_narrowing": True,
                "llm_analysis_interval": 1,
                "llm_endpoint": endpoint,
                "llm_model": model,
                "llm_api_key": api_key,
            }
        )

    sdk = SlurmSDK(poll_interval=30, epoch_milestone_interval=5)
    runner = AutoMLRunner(sdk=sdk, skill_dir=str(SKILL_DIR), action="train")
    logging.info(
        "starting %s algorithm=%s recommendations=%d dataset=%s",
        track_name,
        algorithm,
        recommendations,
        dataset,
    )
    result = runner.run(
        image=IMAGE,
        spec_overrides=build_spec(backbone, dataset),
        automl_settings=settings,
        automl_hyperparameters=SEARCH_PARAMETERS,
        custom_param_ranges=ranges,
        on_recommendation=(pin_control_recommendation if kind == "control" else None),
        workspace_path=str(workspace),
        resume=False,
        gpu_count=8,
        num_nodes=1,
        account=os.environ.get("SLURM_ACCOUNT") or None,
        env_vars={
            "PYTHONPATH": "/usr/local/lib/python3.12/dist-packages",
        },
    )
    result_path = workspace / "track_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("automl", "deft_automl", "control"))
    parser.add_argument("--backbone", choices=tuple(BACKBONES))
    parser.add_argument("--variant")
    parser.add_argument("--retry-tag", default="")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.validate_only:
        validate_bundle(args.backbone, args.variant, args.output)
        print(f"VALID={args.output}")
        return
    result = run_track(args.kind, args.backbone, args.variant, args.retry_tag)
    best = result.get("best") or {}
    print(json.dumps({"best": best, "status": "complete"}, default=str))


if __name__ == "__main__":
    main()
