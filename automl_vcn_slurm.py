"""
AutoML runner for visual-changenet on SLURM.
Bayesian-family: bayesian, bohb, dehb (three parallel experiments).
Dataset: DEFT AOI workspace training_set.csv (210 rows).
KPI: FAR@100% recall (far_at_recall, minimize).

BLOCKERS before running:
  1. SSH must be unblocked: ssh <user>@<slurm-login-host>
  2. Set SLURM_BASE_RESULTS_DIR (Lustre results root, e.g. /lustre/<results-root>)
  3. Set LUSTRE_TRAIN_CSV, LUSTRE_VAL_CSV, LUSTRE_IMAGES_DIR (staged dataset paths on Lustre)
  4. Optionally set SLURM_PARTITION, SLURM_ACCOUNT

Usage:
  export SLURM_BASE_RESULTS_DIR=/lustre/...
  export LUSTRE_TRAIN_CSV=/lustre/.../training_set.csv
  export LUSTRE_VAL_CSV=/lustre/.../validation_set.csv
  export LUSTRE_IMAGES_DIR=/lustre/.../kpi/images
  python automl_vcn_slurm.py [--algorithm bayesian|bohb|dehb|all]
"""

import os, sys, argparse
from pathlib import Path

SB = Path(os.environ.get("TAO_SKILL_BANK_PATH", os.environ.get("TAO_SKILL_BANK_PATH", ".")))
SKILL_DIR = SB / "skills" / "models" / "tao-train-visual-changenet"
WORKSPACE = Path(os.environ.get("AOI_WORKSPACE", "./workspace"))

IMAGE = "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt"

# Required dataset paths on Lustre (must be set before running)
LUSTRE_TRAIN_CSV   = os.environ.get("LUSTRE_TRAIN_CSV", "")
LUSTRE_VAL_CSV     = os.environ.get("LUSTRE_VAL_CSV", "")
LUSTRE_IMAGES_DIR  = os.environ.get("LUSTRE_IMAGES_DIR", "")
RESULTS_ROOT       = os.environ.get("SLURM_BASE_RESULTS_DIR", "")
PARTITION          = os.environ.get("SLURM_PARTITION", "gpu")
ACCOUNT            = os.environ.get("SLURM_ACCOUNT", "")

for var, name in [(LUSTRE_TRAIN_CSV,"LUSTRE_TRAIN_CSV"), (LUSTRE_VAL_CSV,"LUSTRE_VAL_CSV"),
                  (LUSTRE_IMAGES_DIR,"LUSTRE_IMAGES_DIR"), (RESULTS_ROOT,"SLURM_BASE_RESULTS_DIR")]:
    if not var:
        sys.exit(f"ERROR: {name} is not set. Export it before running.")

# Spec overrides — concrete values (nested dicts, no dotted keys)
spec_overrides = {
    "task": "classify",
    "train": {
        "num_epochs": 10,
        "num_nodes": 1,
        "validation_interval": 1,
        "checkpoint_interval": 10,
        "pretrained_model_path": None,
        "resume_training_checkpoint_path": None,
        "classify": {"loss": "ce", "cls_weight": [1.0, 10.0]},
        "optim": {"optim": "adamw", "policy": "linear"},
    },
    "model": {
        "backbone": {
            "type": "c_radio_v2_vit_base_patch16_224",
            "pretrained_backbone_path": "/data/pretrained_models/C-RADIOv2_B.safetensors",
            "freeze_backbone": False,
        },
        "classify": {
            "difference_module": "learnable",
            "learnable_difference_modules": 4,
            "embedding_vectors": 5,
            "embed_dec": 30,
        },
        "decode_head": {"use_summary_token": True},
    },
    "dataset": {
        "classify": {
            "train_dataset":      {"csv_path": LUSTRE_TRAIN_CSV, "images_dir": LUSTRE_IMAGES_DIR},
            "validation_dataset": {"csv_path": LUSTRE_VAL_CSV,   "images_dir": LUSTRE_IMAGES_DIR},
            "test_dataset":       {"csv_path": LUSTRE_VAL_CSV,   "images_dir": LUSTRE_IMAGES_DIR},
            "infer_dataset":      {"csv_path": "",                "images_dir": ""},
            "batch_size": 16,
            "workers": 2,
            "fpratio_sampling": 0.2,
            "num_input": 1,
            "input_map": {"SolderLight": 0},
            "concat_type": "linear",
            "image_width": 224, "image_height": 224, "image_ext": ".jpg",
            "grid_map": {"x": 2, "y": 2},
            "num_classes": 2,
            "augmentation_config": {
                "augment": True,
                "rgb_input_mean": [0.485, 0.456, 0.406],
                "rgb_input_std":  [0.229, 0.224, 0.225],
                "random_flip": {"enable": True, "vflip_probability": 0.5, "hflip_probability": 0.5},
                "random_rotate": {"enable": True, "rotate_probability": 0.5, "angle_list": [90,180,270]},
                "random_color": {"enable": True, "brightness": 0.3, "contrast": 0.3, "saturation": 0.3, "hue": 0.3},
                "with_scale_random_crop": {"enable": True},
                "with_random_crop": True, "with_random_blur": True,
            },
        }
    },
    "encryption_key": "tlt_encode",
}

# Hyperparameters to search (from automl_default_parameters, classify subset)
automl_hyperparameters = [
    "train.optim.lr",
    "train.optim.weight_decay",
    "train.optim.momentum",
    "model.classify.train_margin_euclid",
    "model.classify.eval_margin",
    "model.classify.learnable_difference_modules",
    "dataset.classify.fpratio_sampling",
    "dataset.classify.augmentation_config.random_color.brightness",
    "dataset.classify.augmentation_config.random_color.contrast",
    "dataset.classify.augmentation_config.random_color.saturation",
    "dataset.classify.augmentation_config.random_rotate.rotate_probability",
]

# Custom ranges to keep within dataset-size constraints (210 rows / batch 16 → ~13 steps)
custom_param_ranges = {
    "train.optim.lr":         {"min": 5e-6, "max": 5e-4},
    "train.optim.weight_decay": {"min": 1e-4, "max": 0.1},
    "train.optim.momentum":   {"min": 0.85, "max": 0.99},
    "dataset.classify.fpratio_sampling": {"min": 0.1, "max": 0.5},
    "model.classify.train_margin_euclid": {"min": 0.5, "max": 4.0},
    "model.classify.eval_margin": {"min": 0.1, "max": 1.0},
    "model.classify.learnable_difference_modules": {"values": [2, 4, 8]},
    "dataset.classify.augmentation_config.random_color.brightness": {"min": 0.1, "max": 0.5},
    "dataset.classify.augmentation_config.random_color.contrast":   {"min": 0.1, "max": 0.5},
    "dataset.classify.augmentation_config.random_color.saturation": {"min": 0.1, "max": 0.5},
    "dataset.classify.augmentation_config.random_rotate.rotate_probability": {"min": 0.3, "max": 0.8},
}

# Bayesian-family algorithm configs
BAYESIAN_ALGOS = {
    "bayesian": {
        "algorithm": "bayesian",
        "settings": {
            "num_recommendations": 8,
            "train_sample_count": 210,
        },
    },
    "bohb": {
        "algorithm": "bohb",
        "settings": {
            "num_recommendations": 10,
            "max_epochs": 10,
            "reduction_factor": 3,
            "train_sample_count": 210,
        },
    },
    "dehb": {
        "algorithm": "dehb",
        "settings": {
            "num_recommendations": 10,
            "max_epochs": 10,
            "reduction_factor": 3,
            "train_sample_count": 210,
        },
    },
}

def launch_one(algo_name, algo_cfg):
    from tao_automl.runner import AutoMLRunner
    # Platform SDK import
    try:
        from tao_automl.platforms.slurm import SlurmSDK
    except ImportError:
        from tao_automl.platform.slurm import SlurmSDK

    sdk = SlurmSDK(
        user=os.environ["SLURM_USER"],
        hostname=os.environ["SLURM_HOSTNAME"].split(",")[0],
        partition=PARTITION,
        account=ACCOUNT or None,
        container_image=IMAGE,
        gpus_per_node=1,
        num_nodes=1,
        time_limit="02:00:00",
        extra_sbatch_flags=["--exclusive"] if not ACCOUNT else [],
    )

    workspace_dir = str(Path(RESULTS_ROOT) / f"automl_vcn_{algo_name}")
    runner = AutoMLRunner(
        skill_dir=str(SKILL_DIR),
        platform_sdk=sdk,
        workspace_dir=workspace_dir,
    )

    result = runner.run(
        automl_algorithm=algo_cfg["algorithm"],
        automl_settings=algo_cfg["settings"],
        spec_overrides=spec_overrides,
        automl_hyperparameters=automl_hyperparameters,
        custom_param_ranges=custom_param_ranges,
        metric="val_loss",
        direction="minimize",
    )

    print(f"\n=== {algo_name} DONE ===")
    print(f"Best metric (val_loss): {result.get('best', {}).get('metric')}")
    print(f"Best HPs: {result.get('best', {}).get('spec_overrides')}")
    print(f"Workspace: {workspace_dir}")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="all", choices=list(BAYESIAN_ALGOS)+["all"])
    args = parser.parse_args()

    algos = BAYESIAN_ALGOS if args.algorithm == "all" else {args.algorithm: BAYESIAN_ALGOS[args.algorithm]}
    for name, cfg in algos.items():
        print(f"\nLaunching AutoML [{name}] — {cfg['settings'].get('num_recommendations')} recs, metric=val_loss minimize")
        launch_one(name, cfg)

