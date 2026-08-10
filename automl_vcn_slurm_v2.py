"""
AutoML v2 runner for visual-changenet on SLURM.

What's new vs v1:
  - FAR@100% recall eval_fn (actual KPI, not val_loss proxy)
  - SQSH pre-check — skips conversion if .sqsh already exists on Lustre
  - BOHB + DEHB rerun with correct SQSH handling
  - Three LLM-guided variants:
      llm_range   : bayesian + enable_llm_range_narrowing
      hybrid      : hybrid algorithm (Bayesian + LLM-guided exploration)
      hybrid_llm  : hybrid + hybrid_enable_llm_range_narrowing
  - LLM config sourced from base_url / model / AUTOML_LLM_API_KEY in config.env

Usage:
  set -a; source ~/.tao/config.env; set +a
  python automl_vcn_slurm_v2.py --algorithm <name>|all_bayesian|all_llm|all
  python automl_vcn_slurm_v2.py --far-only   # just score the bayesian winner
"""

import os, sys, argparse, logging, json
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
SB         = Path(os.environ.get("TAO_SKILL_BANK_PATH",
                                  os.environ.get("TAO_SKILL_BANK_PATH", ".")))
SKILL_DIR  = SB / "skills" / "models" / "tao-train-visual-changenet"
WORKSPACE  = Path(os.environ.get("AOI_WORKSPACE", "./workspace"))
LOCAL_KPI  = WORKSPACE / "kpi" / "testing_set.csv"

IMAGE = "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt"

# ── Lustre paths (from env set by the v1 run) ─────────────────────────────────
LUSTRE_AOI_ROOT = (
    os.environ.get("LUSTRE_AOI_ROOT", "/lustre/<project>/users/<user>/aoi_automl")
)
LUSTRE_TRAIN_CSV  = os.environ.get("LUSTRE_TRAIN_CSV",
    f"{LUSTRE_AOI_ROOT}/train/training_set.csv")
LUSTRE_VAL_CSV    = os.environ.get("LUSTRE_VAL_CSV",
    f"{LUSTRE_AOI_ROOT}/train/validation_set.csv")
LUSTRE_KPI_CSV    = os.environ.get("LUSTRE_KPI_CSV",
    f"{LUSTRE_AOI_ROOT}/train/testing_set.csv")
LUSTRE_IMAGES_DIR = os.environ.get("LUSTRE_IMAGES_DIR",
    f"{LUSTRE_AOI_ROOT}/images")
LUSTRE_BACKBONE   = (f"{LUSTRE_AOI_ROOT}/backbone/c_radio_v2_b.safetensors")
RESULTS_ROOT      = os.environ.get("SLURM_BASE_RESULTS_DIR",
    f"{LUSTRE_AOI_ROOT}/results")
PARTITION         = os.environ.get("SLURM_PARTITION", os.environ.get("SLURM_PARTITION", ""))
ACCOUNT           = os.environ.get("SLURM_ACCOUNT",
    os.environ.get("SLURM_ACCOUNT", ""))

# Bayesian winner job id (retained artifact from v1 run)
BAYESIAN_WINNER_JOB = "33d13b05-af93-4638-8de1-7858d628e232"
BAYESIAN_V1_RESULTS = f"{RESULTS_ROOT}/results/{BAYESIAN_WINNER_JOB}"

# ── LLM config (from ~/.tao/config.env) ───────────────────────────────────────
LLM_ENDPOINT = os.environ.get("base_url", "")
LLM_MODEL    = os.environ.get("model", "")
LLM_API_KEY  = os.environ.get("AUTOML_LLM_API_KEY", "")

# ── Base spec (nested dicts, no dotted keys) ──────────────────────────────────
BASE_SPEC = {
    "encryption_key": "tlt_encode",
    "task": "classify",
    "train": {
        "num_epochs": 100,
        "num_nodes": 1,
        "validation_interval": 10,
        "checkpoint_interval": 10,  # clamped per-rec to min(10, rung epochs) by on_recommendation
        # In-training FAR selection: the patched nvidia_tao_pytorch (PYTHONPATH
        # overlay on Lustre) logs val_far every validation epoch; the topk
        # checkpointer keeps exactly the best-val_far checkpoint (model_best_*).
        "checkpointer": {"enable_topk": True, "replace_periodic": True,
                          "monitor": "val_far", "mode": "min", "save_top_k": 1},
        "pretrained_model_path": None,
        "resume_training_checkpoint_path": None,
        "classify": {"loss": "ce", "cls_weight": [1.0, 10.0]},
        "optim": {"optim": "adamw", "policy": "linear"},
    },
    "model": {
        "backbone": {
            "type": "c_radio_v2_vit_base_patch16_224",
            "pretrained_backbone_path": LUSTRE_BACKBONE,
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
            "train_dataset":      {"csv_path": LUSTRE_TRAIN_CSV,  "images_dir": LUSTRE_IMAGES_DIR},
            "validation_dataset": {"csv_path": LUSTRE_VAL_CSV,    "images_dir": LUSTRE_IMAGES_DIR},
            "test_dataset":       {"csv_path": LUSTRE_VAL_CSV,    "images_dir": LUSTRE_IMAGES_DIR},
            "infer_dataset":      {"csv_path": LUSTRE_KPI_CSV,    "images_dir": LUSTRE_IMAGES_DIR},
            "batch_size": 16, "workers": 2, "fpratio_sampling": 0.2,
            "num_input": 1, "input_map": {"SolderLight": 0},
            "concat_type": "linear", "image_width": 224, "image_height": 224,
            "image_ext": ".jpg", "grid_map": {"x": 2, "y": 2}, "num_classes": 2,
            "augmentation_config": {
                "augment": True,
                "rgb_input_mean": [0.485, 0.456, 0.406],
                "rgb_input_std":  [0.229, 0.224, 0.225],
                "random_flip":    {"enable": True, "vflip_probability": 0.5, "hflip_probability": 0.5},
                "random_rotate":  {"enable": True, "rotate_probability": 0.5, "angle_list": [90,180,270]},
                "random_color":   {"enable": True, "brightness": 0.3, "contrast": 0.3,
                                   "saturation": 0.3, "hue": 0.3},
                "with_scale_random_crop": {"enable": True},
                "with_random_crop": True, "with_random_blur": True,
            },
        }
    },
}

AUTOML_HPS = [
    "train.optim.lr", "train.optim.weight_decay", "train.optim.momentum",
    "model.classify.train_margin_euclid", "model.classify.eval_margin",
    "model.classify.learnable_difference_modules",
    "dataset.classify.fpratio_sampling",
    "dataset.classify.augmentation_config.random_color.brightness",
    "dataset.classify.augmentation_config.random_color.contrast",
    "dataset.classify.augmentation_config.random_color.saturation",
    "dataset.classify.augmentation_config.random_rotate.rotate_probability",
]

# Range override keys MUST be valid_min/valid_max/valid_options — the brain
# copies these verbatim into parameter_config; min/max/values are silently ignored.
CUSTOM_RANGES = {
    "train.optim.lr":                     {"valid_min": 5e-6, "valid_max": 5e-4},
    "train.optim.weight_decay":           {"valid_min": 1e-4, "valid_max": 0.1},
    "train.optim.momentum":               {"valid_min": 0.85, "valid_max": 0.99},
    "dataset.classify.fpratio_sampling":  {"valid_min": 0.1, "valid_max": 0.5},
    "model.classify.train_margin_euclid": {"valid_min": 0.5, "valid_max": 4.0},
    "model.classify.eval_margin":         {"valid_min": 0.1, "valid_max": 1.0},
    "model.classify.learnable_difference_modules": {"valid_min": 2, "valid_max": 8},
    "dataset.classify.augmentation_config.random_color.brightness": {"valid_min": 0.1, "valid_max": 0.5},
    "dataset.classify.augmentation_config.random_color.contrast":   {"valid_min": 0.1, "valid_max": 0.5},
    "dataset.classify.augmentation_config.random_color.saturation": {"valid_min": 0.1, "valid_max": 0.5},
    "dataset.classify.augmentation_config.random_rotate.rotate_probability": {"valid_min": 0.3, "valid_max": 0.8},
}

LLM_BASE_SETTINGS = {
    "llm_endpoint": LLM_ENDPOINT,
    "llm_model":    LLM_MODEL,
    "llm_api_key":  LLM_API_KEY,
}

def _bayesian_settings(**extra):
    return {"num_recommendations": 10, "train_sample_count": 210, "metric": "far_pct", **extra}

def _mf_settings(**extra):
    return {"num_recommendations": 10, "automl_max_epochs": 10, "automl_reduction_factor": 3,
            "train_sample_count": 210, "metric": "far_pct", **extra}

ALGO_CONFIGS = {
    # ── Pure Bayesian-based — all optimize FAR directly via eval_fn ───────────
    "bfbo":      {"algorithm": "bfbo",    "settings": _bayesian_settings(),
                  "metric": "far_pct", "direction": "minimize", "use_far_eval_fn": True},
    "bohb":      {"algorithm": "bohb",    "settings": _mf_settings(),
                  "metric": "far_pct", "direction": "minimize", "use_far_eval_fn": True},
    # ── LLM-guided variants — also optimize FAR directly ──────────────────────
    "bayesian_llm": {"algorithm": "bayesian", "settings": _bayesian_settings(
                        enable_llm_range_narrowing=True, **LLM_BASE_SETTINGS),
                     "metric": "far_pct", "direction": "minimize", "use_far_eval_fn": True},
    "bfbo_llm":  {"algorithm": "bfbo",    "settings": _bayesian_settings(
                        enable_llm_range_narrowing=True, **LLM_BASE_SETTINGS),
                  "metric": "far_pct", "direction": "minimize", "use_far_eval_fn": True},
    "bohb_llm":  {"algorithm": "bohb",    "settings": _mf_settings(
                        enable_llm_range_narrowing=True, **LLM_BASE_SETTINGS),
                  "metric": "far_pct", "direction": "minimize", "use_far_eval_fn": True},
    # ── Bayesian optimizing FAR directly via eval_fn ──────────────────────────
    "bayesian_far": {
        "algorithm": "bayesian",
        "settings": {
            "num_recommendations": 10, "train_sample_count": 210,
            "metric": "far_pct",
        },
        "metric": "far_pct", "direction": "minimize",
        "use_far_eval_fn": True,   # signals launch_one to attach eval_fn
    },

    # ── 40-rec runs ────────────────────────────────────────────────────────────
    # Run A: current 11-param space, schema-default bounds (ranges=None mirrors
    # what the 10-rec runs effectively searched), 40 recommendations.
    "bfbo_llm_40_default": {
        "algorithm": "bfbo",
        "settings": _bayesian_settings(
            num_recommendations=40,
            enable_llm_range_narrowing=True, **LLM_BASE_SETTINGS),
        "metric": "far_pct", "direction": "minimize", "use_far_eval_fn": True,
        "hyperparameters": AUTOML_HPS,
        "ranges": None,
    },
    # Run B: hand-picked space. Evidence: best FAR recs used fpratio≈0.93,
    # momentum≈0.19-0.32, weight_decay≈0.14-0.94, lr≈3e-5, ldm=3, moderate
    # color jitter. Margins dropped — unused under CE loss (contrastive-only).
    # NOTE: range keys must be valid_min/valid_max — the brain override loop
    # copies keys verbatim into parameter_config (min/max are silently ignored).
    "bfbo_llm_40_custom": {
        "algorithm": "bfbo",
        "settings": _bayesian_settings(
            num_recommendations=40,
            enable_llm_range_narrowing=True, **LLM_BASE_SETTINGS),
        "metric": "far_pct", "direction": "minimize", "use_far_eval_fn": True,
        "hyperparameters": [
            "dataset.classify.fpratio_sampling",
            "train.optim.lr",
            "train.optim.weight_decay",
            "train.optim.momentum",
            "model.classify.learnable_difference_modules",
            "dataset.classify.augmentation_config.random_color.brightness",
            "dataset.classify.augmentation_config.random_color.contrast",
            "dataset.classify.augmentation_config.random_color.saturation",
            "dataset.classify.augmentation_config.random_color.hue",
            "dataset.classify.augmentation_config.random_rotate.rotate_probability",
            "dataset.classify.augmentation_config.random_flip.vflip_probability",
            "dataset.classify.augmentation_config.random_flip.hflip_probability",
        ],
        "ranges": {
            # THE imbalance lever — best rec (57.6%) sampled 0.93; search heavy
            # defect oversampling. 556:1 PASS:defect KPI ratio demands it.
            "dataset.classify.fpratio_sampling": {"valid_min": 0.3, "valid_max": 0.98},
            # Winners clustered at 3e-5..5e-5; big LRs gave 99-100% FAR.
            "train.optim.lr": {"valid_min": 5e-6, "valid_max": 1e-4},
            # Strong regularization generalizes better from 210 rows (winners 0.14-0.94).
            "train.optim.weight_decay": {"valid_min": 0.05, "valid_max": 1.0},
            # Best recs at 0.18-0.32 — well below the 0.9 default region.
            "train.optim.momentum": {"valid_min": 0.1, "valid_max": 0.7},
            # Winners at 1-4; 8 is overcapacity for 210 rows.
            "model.classify.learnable_difference_modules": {"valid_min": 1, "valid_max": 4},
            # Moderate photometric jitter; extreme jitter (1.5+) correlated with 86-91% FAR.
            "dataset.classify.augmentation_config.random_color.brightness": {"valid_min": 0.1, "valid_max": 0.6},
            "dataset.classify.augmentation_config.random_color.contrast":   {"valid_min": 0.1, "valid_max": 0.6},
            "dataset.classify.augmentation_config.random_color.saturation": {"valid_min": 0.1, "valid_max": 0.6},
            "dataset.classify.augmentation_config.random_color.hue":        {"valid_min": 0.0, "valid_max": 0.3},
            "dataset.classify.augmentation_config.random_rotate.rotate_probability": {"valid_min": 0.1, "valid_max": 0.5},
            "dataset.classify.augmentation_config.random_flip.vflip_probability": {"valid_min": 0.3, "valid_max": 0.7},
            "dataset.classify.augmentation_config.random_flip.hflip_probability": {"valid_min": 0.3, "valid_max": 0.7},
        },
    },

    # Run C: hand-picked space + schema-unblocked params. cls_weight[1] is the
    # defect-class CE weight (schema-gated as automl_enabled=false on the list;
    # exposed via a scalar cls_weight[1] property added to the model skill schema
    # — verified end-to-end via generate_hyperparams_to_search dry-run).
    # batch_size needed no schema edit (only explicit false blocks a param).
    "bfbo_llm_40_unblocked": {
        "algorithm": "bfbo",
        "settings": _bayesian_settings(
            num_recommendations=40,
            enable_llm_range_narrowing=True, **LLM_BASE_SETTINGS),
        "metric": "far_pct", "direction": "minimize", "use_far_eval_fn": True,
        "hyperparameters": [
            "train.classify.cls_weight[1]",
            "dataset.classify.batch_size",
            "dataset.classify.fpratio_sampling",
            "train.optim.lr",
            "train.optim.weight_decay",
            "train.optim.momentum",
            "model.classify.learnable_difference_modules",
            "dataset.classify.augmentation_config.random_color.brightness",
            "dataset.classify.augmentation_config.random_color.contrast",
            "dataset.classify.augmentation_config.random_color.saturation",
            "dataset.classify.augmentation_config.random_color.hue",
            "dataset.classify.augmentation_config.random_rotate.rotate_probability",
            "dataset.classify.augmentation_config.random_flip.vflip_probability",
            "dataset.classify.augmentation_config.random_flip.hflip_probability",
        ],
        "ranges": {
            # Defect-class CE weight — the direct loss-level imbalance lever.
            "train.classify.cls_weight[1]": {"valid_min": 2.0, "valid_max": 60.0},
            # Schema max is inf — bound it. Small data, sampler interaction.
            "dataset.classify.batch_size": {"valid_min": 8, "valid_max": 32},
            "dataset.classify.fpratio_sampling": {"valid_min": 0.3, "valid_max": 0.98},
            "train.optim.lr": {"valid_min": 5e-6, "valid_max": 1e-4},
            "train.optim.weight_decay": {"valid_min": 0.05, "valid_max": 1.0},
            "train.optim.momentum": {"valid_min": 0.1, "valid_max": 0.7},
            "model.classify.learnable_difference_modules": {"valid_min": 1, "valid_max": 4},
            "dataset.classify.augmentation_config.random_color.brightness": {"valid_min": 0.1, "valid_max": 0.6},
            "dataset.classify.augmentation_config.random_color.contrast":   {"valid_min": 0.1, "valid_max": 0.6},
            "dataset.classify.augmentation_config.random_color.saturation": {"valid_min": 0.1, "valid_max": 0.6},
            "dataset.classify.augmentation_config.random_color.hue":        {"valid_min": 0.0, "valid_max": 0.3},
            "dataset.classify.augmentation_config.random_rotate.rotate_probability": {"valid_min": 0.1, "valid_max": 0.5},
            "dataset.classify.augmentation_config.random_flip.vflip_probability": {"valid_min": 0.3, "valid_max": 0.7},
            "dataset.classify.augmentation_config.random_flip.hflip_probability": {"valid_min": 0.3, "valid_max": 0.7},
        },
    },
}

# ── 40-rec × 100-epoch algorithm sweep (all on the unblocked 14-param space) ──
# bfbo_llm × unblocked is already covered by "bfbo_llm_40_unblocked" above.
_U = ALGO_CONFIGS["bfbo_llm_40_unblocked"]

def _algo40(algorithm, llm):
    if algorithm == "bohb":
        s = _mf_settings(num_recommendations=40, automl_max_epochs=100)
    else:
        s = _bayesian_settings(num_recommendations=40)
    if llm:
        s.update(enable_llm_range_narrowing=True, **LLM_BASE_SETTINGS)
    return {"algorithm": algorithm, "settings": s,
            "metric": "far_pct", "direction": "minimize", "use_far_eval_fn": True,
            "hyperparameters": _U["hyperparameters"], "ranges": _U["ranges"]}

ALGO_CONFIGS["bayesian_40_unblocked"]     = _algo40("bayesian", llm=False)
ALGO_CONFIGS["bayesian_llm_40_unblocked"] = _algo40("bayesian", llm=True)
ALGO_CONFIGS["bfbo_40_unblocked"]         = _algo40("bfbo",     llm=False)
ALGO_CONFIGS["bohb_40_unblocked"]         = _algo40("bohb",     llm=False)
ALGO_CONFIGS["bohb_llm_40_unblocked"]     = _algo40("bohb",     llm=True)

ALGO_CONFIGS["bfbo_llm_40_default_strict"] = {
    "algorithm": "bfbo",
    "settings": _bayesian_settings(
        num_recommendations=40,
        enable_llm_range_narrowing=True, **LLM_BASE_SETTINGS),
    "metric": "far_pct", "direction": "minimize",
    "use_val_far_eval_fn": True,      # strict: brain sees validation FAR only
    "hyperparameters": AUTOML_HPS,    # same default 11-param space as the leader
    "ranges": None,
}

# ── v3 relaunch (post direction-fix) ─────────────────────────────────────────
# Deduped: enable_llm_range_narrowing is consumed ONLY by the hybrid brain
# (factory.py) — for bayesian/bfbo/bohb it was a silent no-op, so *_llm arms
# duplicated their plain siblings. v3 runs one arm per genuinely distinct cell.
# Requires the patched tao_automl (factory passes metric; bfbo/bayesian honor
# 'loss'-name minimize via signed GP fit; bfbo drops failed trials).
_V3_BASE = {
    "algorithm": "bfbo",
    "settings": _bayesian_settings(num_recommendations=40),
    "metric": "far_pct", "direction": "minimize", "use_far_eval_fn": True,
}

ALGO_GROUPS = {
    "all_bayesian": ["bfbo", "bohb"],
    "all_llm":      ["bayesian_llm", "bfbo_llm", "bohb_llm"],
    "all":          ["bfbo", "bohb", "bayesian_llm", "bfbo_llm", "bohb_llm"],
    "all_far":      ["bayesian_far"],
    "all_40":       ["bfbo_llm_40_default", "bfbo_llm_40_custom", "bfbo_llm_40_unblocked"],
    "algo_sweep_40": ["bayesian_40_unblocked", "bayesian_llm_40_unblocked",
                       "bfbo_40_unblocked", "bohb_40_unblocked", "bohb_llm_40_unblocked"],
}

# v3 config registrations (after ALGO_CONFIGS/_algo40 mutations above)
_CUSTOM_SPACE = ALGO_CONFIGS["bfbo_llm_40_custom"]
ALGO_CONFIGS["bfbo_40_default"] = {**_V3_BASE,
    "hyperparameters": AUTOML_HPS, "ranges": None}
ALGO_CONFIGS["bfbo_40_custom"] = {**_V3_BASE,
    "hyperparameters": _CUSTOM_SPACE["hyperparameters"],
    "ranges": _CUSTOM_SPACE["ranges"]}
ALGO_CONFIGS["bfbo_40_default_strict"] = {**_V3_BASE,
    "hyperparameters": AUTOML_HPS, "ranges": None,
    "use_far_eval_fn": False, "use_val_far_eval_fn": True}
# Genuine LLM-guided arms. NOTE (2026-08-08): the local tao_automl package
# has been patched so enable_llm_range_narrowing now ALSO works for
# bayesian/bfbo/bohb directly (LLMAnalyzer wired into the brains with
# design-point renormalization + best-observed guardrail; see
# brain/*.bak_prenarrow backups). The *_llm configs above are therefore no
# longer no-ops for new runs. hybrid = Bayesian+LLM exploration with phase
# narrowing; llm = pure LLM-driven proposals.
_U3 = ALGO_CONFIGS["bfbo_llm_40_unblocked"]   # unblocked 14-param space source
ALGO_CONFIGS["hybrid_40_unblocked"] = {
    "algorithm": "hybrid",
    "settings": _bayesian_settings(
        num_recommendations=40,
        enable_llm_range_narrowing=True, **LLM_BASE_SETTINGS),
    "metric": "far_pct", "direction": "minimize", "use_far_eval_fn": True,
    "hyperparameters": _U3["hyperparameters"], "ranges": _U3["ranges"],
}
ALGO_CONFIGS["llm_40_unblocked"] = {
    "algorithm": "llm",
    "settings": _bayesian_settings(
        num_recommendations=40, **LLM_BASE_SETTINGS),
    "metric": "far_pct", "direction": "minimize", "use_far_eval_fn": True,
    "hyperparameters": _U3["hyperparameters"], "ranges": _U3["ranges"],
}

ALGO_GROUPS["relaunch_v3"] = [
    "bfbo_40_default", "bfbo_40_custom", "bfbo_40_unblocked",
    "bayesian_40_unblocked", "bohb_40_unblocked",
]
# LLM-guided search-space restriction (two-stage) for each base algorithm.
# Stage 1 = 12 recs full default space; LLM narrows bounds (minimize-aware,
# guardrailed: clamped to schema, best-observed config kept inside every range);
# Stage 2 = 28 recs, fresh brain, narrowed space.
for _algo in ["bfbo", "bayesian", "bohb"]:
    _s = (_mf_settings(num_recommendations=40, automl_max_epochs=100)
          if _algo == "bohb" else _bayesian_settings(num_recommendations=40))
    ALGO_CONFIGS[f"{_algo}_40_default_llmnarrow"] = {
        "algorithm": _algo, "settings": _s,
        "metric": "far_pct", "direction": "minimize", "use_far_eval_fn": True,
        "hyperparameters": AUTOML_HPS, "ranges": None,
        "llm_narrow": {"warmup_recs": 12},
    }

ALGO_GROUPS["llmnarrow_v3"] = ["bfbo_40_default_llmnarrow",
                                "bayesian_40_default_llmnarrow",
                                "bohb_40_default_llmnarrow"]



def make_sdk():
    from tao_sdk.platforms.slurm import SlurmSDK
    # SlurmSDK reads all credentials from env (SLURM_USER, SLURM_HOSTNAME,
    # SSH_KEY_PATH, SLURM_BASE_RESULTS_DIR, SLURM_PARTITION, SLURM_ACCOUNT, NGC_KEY …)
    return SlurmSDK()


def far_eval_fn(rec, train_job_id, sdk, results_dir):
    """Submit a SLURM inference+KPI job on the rec's checkpoint and return FAR."""
    import tempfile, time
    ckpt_path = rec.get("checkpoint_path") or rec.get("result_path", "")
    if not ckpt_path:
        log.warning("eval_fn: no checkpoint path for rec, skipping FAR eval")
        return None

    infer_results = f"{results_dir}/infer_{rec['id']}"
    infer_spec = {
        **BASE_SPEC,
        "task": "classify",
        "inference": {
            "checkpoint": ckpt_path,
            "batch_size": 16,
            "results_dir": infer_results,
        },
    }

    try:
        job_id = sdk.submit(
            action="inference",
            spec=infer_spec,
            job_name=f"vcn-infer-{rec['id'][:8]}",
        )
        log.info("FAR eval: submitted inference job %s for rec %s", job_id, rec['id'])

        # Poll until done
        for _ in range(120):
            time.sleep(30)
            status = sdk.status(job_id)
            if status in ("COMPLETE", "COMPLETED"):
                break
            if status in ("ERROR", "FAILED", "CANCELLED"):
                log.warning("FAR eval: inference job %s failed (%s)", job_id, status)
                return None

        # Pull inference.csv and run analyze_kpi locally
        infer_csv_remote = f"{infer_results}/inference.csv"
        local_tmp = f"/tmp/far_eval_{rec['id'][:8]}_inference.csv"
        sdk.download(infer_csv_remote, local_tmp)

        import subprocess, json as _json, pathlib
        analyze = str(SB / "skills/applications/tao-run-deft-aoi/scripts/analyze_kpi.py")
        out_dir = f"/tmp/far_eval_{rec['id'][:8]}"
        pathlib.Path(out_dir).mkdir(exist_ok=True)

        venv_py = str(WORKSPACE / ".venv/bin/python")
        result = subprocess.run(
            [venv_py, analyze, local_tmp, "--output-dir", out_dir],
            capture_output=True, text=True,
        )
        metric_json = pathlib.Path(out_dir) / "metric_result.json"
        if metric_json.exists():
            d = _json.loads(metric_json.read_text())
            far = d["value"]
            log.info("FAR eval rec %s: FAR=%.2f%%", rec['id'], far)
            return {"metric_value": far, "record_path": str(metric_json), "job_id": job_id}
        else:
            log.warning("FAR eval: analyze_kpi produced no output\n%s", result.stderr[:500])
            return None
    except Exception as e:
        log.warning("FAR eval failed: %s", e)
        return None


def score_bayesian_winner():
    """Run inference on the retained v1 bayesian winner and report FAR."""
    import time, pathlib, subprocess, json as _json
    log.info("Scoring bayesian v1 winner (job %s) ...", BAYESIAN_WINNER_JOB)
    sdk = make_sdk()

    # Find checkpoint via SSH (Lustre is POSIX — SDK list_path only handles S3)
    import subprocess as _sp
    ssh_host = os.environ.get("SLURM_HOSTNAME","").split(",")[0]
    ssh_user = os.environ.get("SLURM_USER","")
    key      = os.environ.get("SSH_KEY_PATH","")
    ssh_find = _sp.run(
        ["ssh", "-i", key, "-o", "StrictHostKeyChecking=no",
         f"{ssh_user}@{ssh_host}",
         f"find {BAYESIAN_V1_RESULTS} -name '*.pth' 2>/dev/null | head -3"],
        capture_output=True, text=True,
    )
    ckpt_candidates = [l.strip() for l in ssh_find.stdout.splitlines() if l.strip()]
    if not ckpt_candidates:
        sys.exit(f"No .pth found under {BAYESIAN_V1_RESULTS}. "
                 "Artifact may have been deleted by the v1 runner cleanup. "
                 "Re-run bayesian with --algorithm bayesian to get a fresh best checkpoint.")
    ckpt = ckpt_candidates[0]
    log.info("Using checkpoint: %s", ckpt)

    # scp checkpoint down locally and run inference via local Docker
    local_ckpt = "/tmp/bayesian_winner.pth"
    local_csv  = "/tmp/bayesian_winner_inference.csv"
    local_infer_dir = "/tmp/bayesian_winner_infer"
    pathlib.Path(local_infer_dir).mkdir(exist_ok=True)

    log.info("Downloading checkpoint via scp ...")
    _sp.run(
        ["scp", "-i", key, "-o", "StrictHostKeyChecking=no",
         f"{ssh_user}@{ssh_host}:{ckpt}", local_ckpt],
        check=True,
    )
    log.info("Checkpoint downloaded to %s", local_ckpt)

    # Write inference spec (all local paths)
    infer_spec_path = "/tmp/bayesian_winner_infer_spec.yaml"
    import yaml as _yaml
    local_spec = {
        **{k: v for k, v in BASE_SPEC.items() if k != "dataset"},
        "task": "classify",
        "dataset": {
            "classify": {
                **BASE_SPEC["dataset"]["classify"],
                "infer_dataset": {
                    "csv_path": str(WORKSPACE / "kpi" / "testing_set.csv"),
                    "images_dir": str(WORKSPACE / "kpi" / "images"),
                },
            }
        },
        "inference": {
            "checkpoint": "/tmp/bayesian_winner.pth",
            "batch_size": 16,
            "results_dir": "/tmp/bayesian_winner_infer",
        },
        "model": {
            **BASE_SPEC["model"],
            "backbone": {
                **BASE_SPEC["model"]["backbone"],
                "pretrained_backbone_path": "/data/pretrained_models/C-RADIOv2_B.safetensors",
            },
        },
    }
    pathlib.Path(infer_spec_path).write_text(_yaml.dump(local_spec, default_flow_style=False))

    log.info("Running local Docker inference ...")
    docker_cmd = [
        "docker", "run", "--rm", "--gpus", "device=0",
        "--shm-size=8g",
        "-v", f"{WORKSPACE}:/data/workspace",
        "-v", f"{local_infer_dir}:/tmp/bayesian_winner_infer",
        "-v", f"{local_ckpt}:/tmp/bayesian_winner.pth",
        "-v", f"{WORKSPACE / 'kpi' / 'images'}:/data/kpi/images",
        "-v", f"{WORKSPACE / 'kpi' / 'testing_set.csv'}:/data/kpi/testing_set.csv",
        "-v", f"{WORKSPACE / 'augmentation/backbone/c_radio_v2_b.safetensors'}:"
               "/data/pretrained_models/C-RADIOv2_B.safetensors",
        "-v", f"{infer_spec_path}:/tmp/infer_spec.yaml",
        IMAGE,
        "visual_changenet", "inference", "-e", "/tmp/infer_spec.yaml",
    ]
    _sp.run(docker_cmd, check=True)
    local_csv = f"{local_infer_dir}/inference.csv"
    log.info("Inference complete, CSV at %s", local_csv)

    # Run analyze_kpi locally
    analyze = str(SB / "skills/applications/tao-run-deft-aoi/scripts/analyze_kpi.py")
    out_dir = "/tmp/bayesian_winner_kpi"
    pathlib.Path(out_dir).mkdir(exist_ok=True)
    venv_py = str(WORKSPACE / ".venv/bin/python")
    subprocess.run([venv_py, analyze, local_csv, "--output-dir", out_dir], check=True)

    metric_json = pathlib.Path(out_dir) / "metric_result.json"
    d = _json.loads(metric_json.read_text())
    far = d["value"]
    threshold = d["threshold"]
    print(f"\n{'='*60}")
    print(f"Bayesian AutoML Winner — KPI Evaluation")
    print(f"  Checkpoint : {ckpt}")
    print(f"  FAR@100%R  : {far:.2f}%")
    print(f"  Threshold  : {threshold:.6f}")
    print(f"  (val_loss during HPO: 0.460)")
    print(f"{'='*60}\n")

    # Save to log dir
    pathlib.Path(args_log_dir).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(args_log_dir) / "bayesian_winner_far.json").write_text(
        _json.dumps({"far_pct": far, "threshold": threshold, "checkpoint": ckpt,
                     "val_loss_hpo": 0.460}, indent=2))
    return far


def launch_one(algo_name, cfg, log_dir):
    from tao_automl.runner import AutoMLRunner
    sdk = make_sdk()
    import datetime
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_dir = str(WORKSPACE / f"automl_{algo_name}_{run_ts}")
    runner = AutoMLRunner(sdk=sdk, skill_dir=str(SKILL_DIR), action="train")

    log.info("Launching AutoML [%s] — algo=%s recs=%s metric=%s",
             algo_name, cfg["algorithm"],
             cfg["settings"].get("num_recommendations", "?"),
             cfg["metric"])

    settings = {"algorithm": cfg["algorithm"], **cfg["settings"]}
    # CRITICAL: direction must live inside automl_settings. Without it, the
    # runner's implicit rule maximizes any metric whose name lacks 'loss' —
    # which would make the brain actively maximize far_pct.
    settings.setdefault("direction", cfg.get("direction", "minimize"))

    merged_overrides = dict(BASE_SPEC)

    def _clamp_intervals(rec):
        """Per-rec safety: BOHB rung overrides set train.num_epochs but never
        touch intervals (brain passes interval=None). TAO asserts
        checkpoint_interval <= num_epochs, so clamp both intervals to the
        rec's actual epoch budget. Full-fidelity recs keep interval 10."""
        n_ep = rec.specs.get("train.num_epochs")
        n_ep = int(n_ep) if n_ep else BASE_SPEC["train"]["num_epochs"]
        rec.specs["train.checkpoint_interval"] = min(10, n_ep)
        rec.specs["train.validation_interval"] = min(10, n_ep)

    # Attach FAR eval_fn when requested (bayesian_far config)
    eval_fn = None
    if cfg.get("use_val_far_eval_fn"):
        eval_fn = make_val_far_eval_fn(sdk)
        log.info("[%s] STRICT protocol: brain sees validation FAR only; "
                 "KPI reserved for one final scoring", algo_name)
    elif cfg.get("use_far_eval_fn"):
        lustre_far_root = f"{LUSTRE_AOI_ROOT}/far_eval_automl/{algo_name}"
        eval_fn = make_far_eval_fn(sdk, IMAGE, ACCOUNT, lustre_far_root)
        log.info("[%s] FAR eval_fn attached — each rec runs KPI inference on SLURM", algo_name)

    # Per-config HP space; None ranges → schema-default bounds.
    hp_list = cfg.get("hyperparameters", AUTOML_HPS)
    hp_ranges = cfg.get("ranges", CUSTOM_RANGES) if "ranges" in cfg else CUSTOM_RANGES
    log.info("[%s] search space: %d params, custom ranges: %s",
             algo_name, len(hp_list), "yes" if hp_ranges else "schema defaults")

    result = runner.run(
        image=IMAGE,
        spec_overrides=merged_overrides,
        automl_settings=settings,
        automl_hyperparameters=hp_list,
        custom_param_ranges=hp_ranges,
        workspace_path=workspace_dir,
        eval_fn=eval_fn,
        on_recommendation=_clamp_intervals,
        account=ACCOUNT or None,
        num_nodes=1,
        env_vars={"PYTHONPATH": f"{LUSTRE_AOI_ROOT}/patches/valfar"},
    )

    best = result.get("best", {})
    log.info("[%s] DONE — best metric=%s HPs=%s",
             algo_name, best.get("metric"), best.get("spec_overrides"))

    # Write summary
    summary = {
        "algorithm": algo_name,
        "best_metric_name": cfg["metric"],
        "best_metric_value": best.get("metric"),
        "best_spec_overrides": best.get("spec_overrides", {}),
        "best_job_id": best.get("job_id"),
        "workspace": workspace_dir,
    }
    out = Path(log_dir) / f"summary_{algo_name}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[{algo_name}] Best {cfg['metric']}: {best.get('metric')} | "
          f"Summary: {out}")
    return result


def llm_narrow_ranges(observations, param_names, base_ranges, algo_name):
    """LLM-guided search-space restriction (two-stage protocol, stage boundary).

    observations: list of {"params": {name: value}, "far_pct": float} from stage 1.
    base_ranges:  {name: {"valid_min": x, "valid_max": y}} — stage-1 sampling box
                  (schema defaults resolved to finite numbers by the caller).
    Returns narrowed {name: {"valid_min", "valid_max"}} with hard guardrails:
    clamped inside base_ranges, min<max, and the best-observed (lowest-FAR)
    config is always inside every narrowed range. Falls back to base_ranges
    on any LLM failure.
    """
    import json as _json
    best = min(observations, key=lambda o: o["far_pct"])
    try:
        from openai import OpenAI
        client = OpenAI(base_url=LLM_ENDPOINT, api_key=LLM_API_KEY)
        obs_lines = "\n".join(
            f"FAR={o['far_pct']:.2f}% :: " + ", ".join(f"{k}={v:.6g}" for k, v in o["params"].items())
            for o in sorted(observations, key=lambda o: o["far_pct"])
        )
        range_lines = "\n".join(
            f"{n}: [{base_ranges[n]['valid_min']:.6g}, {base_ranges[n]['valid_max']:.6g}]"
            for n in param_names
        )
        prompt = f"""You are guiding hyperparameter optimization for a PCB defect-detection model.
OBJECTIVE: MINIMIZE far_pct (false-alarm rate at 100% recall). Lower is strictly better.

Stage-1 observations (sorted best first):
{obs_lines}

Current sampling ranges:
{range_lines}

Propose NARROWED [min, max] ranges per parameter that likely contain the optimum,
based on which regions produced LOW far_pct. Narrow each range to roughly 30-60%
of its current width unless the evidence clearly says otherwise. Every range MUST
contain the corresponding value of the best observation (FAR={best['far_pct']:.2f}%).
Respond with ONLY a JSON object: {{"param_name": {{"valid_min": x, "valid_max": y}}, ...}}
covering every parameter listed above."""
        # NOTE: endpoint models with hidden reasoning (e.g. Gemini) spend
        # thinking tokens against max_tokens — budget generously or the JSON
        # gets truncated mid-string.
        resp = client.chat.completions.create(
            model=LLM_MODEL, temperature=0.2, max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content.strip()
        start, stop = text.find("{"), text.rfind("}")
        if start < 0 or stop <= start:
            raise ValueError(f"no JSON object in LLM response: {text[:200]!r}")
        proposed = _json.loads(text[start:stop + 1])
    except Exception as e:
        log.warning("[%s] LLM narrowing failed (%s) — continuing with stage-1 ranges",
                    algo_name, e)
        return dict(base_ranges)

    narrowed = {}
    int_params = {"model.classify.learnable_difference_modules",
                  "dataset.classify.batch_size"}
    for n in param_names:
        base = base_ranges[n]
        prop = proposed.get(n) or {}
        lo = float(prop.get("valid_min", base["valid_min"]))
        hi = float(prop.get("valid_max", base["valid_max"]))
        # guardrails: clamp inside base box; keep best-observed inside; sane order
        lo = max(lo, base["valid_min"]); hi = min(hi, base["valid_max"])
        bv = best["params"].get(n)
        if bv is not None:
            lo = min(lo, float(bv)); hi = max(hi, float(bv))
        if not (lo < hi):
            lo, hi = base["valid_min"], base["valid_max"]
        if n in int_params:
            lo, hi = int(round(lo)), int(round(hi))
            if lo >= hi:
                lo, hi = int(base["valid_min"]), int(base["valid_max"])
        narrowed[n] = {"valid_min": lo, "valid_max": hi}
        log.info("[%s] LLM narrowing %s: [%.6g, %.6g] -> [%.6g, %.6g]",
                 algo_name, n, base["valid_min"], base["valid_max"], lo, hi)
    return narrowed


def resolve_base_ranges(hp_list, hp_ranges):
    """Finite stage-1 sampling box per param: explicit ranges win, else the
    schema's minimum/maximum, with finite caps for the schema-unbounded
    params (lr, batch_size) matching what the brains effectively sample."""
    import json as _json
    import math
    import re as _re
    schema = _json.loads((SKILL_DIR / "schemas" / "train.schema.json").read_text())
    caps = {"train.optim.lr": (1e-6, 1e-3), "dataset.classify.batch_size": (8, 32)}
    out = {}
    for name in hp_list:
        if hp_ranges and name in hp_ranges:
            out[name] = {"valid_min": float(hp_ranges[name]["valid_min"]),
                         "valid_max": float(hp_ranges[name]["valid_max"])}
            continue
        node = schema
        for part in name.split("."):
            part = _re.sub(r"\[\d+\]$", "", part)
            node = node.get("properties", node).get(part)
            if node is None:
                break
        if node is None:
            raise KeyError(f"schema has no node for {name}")
        lo = float(node.get("minimum", 0.0))
        hi = float(node.get("maximum", float("inf")))
        default = node.get("default", 0.0)
        if name in caps:
            clo, chi = caps[name]
            lo = max(lo, clo)
            hi = min(hi, chi) if math.isfinite(hi) else chi
        elif not math.isfinite(hi):
            hi = max(1.0, abs(float(default or 0.0)) * 10.0)
        out[name] = {"valid_min": lo, "valid_max": hi}
    return out


def make_val_far_eval_fn(sdk):
    """STRICT-protocol eval_fn: report the selected checkpoint's VALIDATION FAR
    to the brain. The KPI set is never queried during the search — it gets
    scored exactly once, on the final winner, after the run completes.

    Zero GPU cost: the patched training already computes val_far every
    validation epoch and the checkpointer keeps the min-val_far checkpoint,
    so min(val_far over status.json) IS the selected checkpoint's val FAR
    (verified equivalent to external inference to ~2e-6 pp).
    """
    import re as _re, subprocess as _sp

    def _ssh(cmd):
        key  = os.environ.get("SSH_KEY_PATH", "")
        user = os.environ.get("SLURM_USER", "")
        host = os.environ.get("SLURM_HOSTNAME", "").split(",")[0]
        r = _sp.run(["ssh", "-i", key, "-o", "StrictHostKeyChecking=no",
                     f"{user}@{host}", cmd],
                    capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(f"SSH failed: {r.stderr[:300]}")
        return r.stdout

    def eval_fn(rec, train_job_id):
        rec_id = str(train_job_id)[:8]
        try:
            raw = sdk.get_job_results_dir(train_job_id)
            job_dir = raw.replace("lustre:///", "/").replace("lustre://", "/")
            status = _ssh(
                f"cat {job_dir}/results_dir/train/status.json 2>/dev/null || "
                f"cat {job_dir}/train/status.json 2>/dev/null"
            )
        except Exception as e:
            log.warning("[val_eval] rec=%s: cannot read status.json: %s", rec_id, e)
            return None
        vals = [float(v) for v in _re.findall(r'"val_far":\s*([0-9.]+)', status)]
        if not vals:
            log.warning("[val_eval] rec=%s: no val_far entries in status.json", rec_id)
            return None
        best = min(vals)
        log.info("[val_eval] rec=%s: val FAR@100%%R = %.2f%% "
                 "(min over %d validations; KPI untouched)", rec_id, best, len(vals))
        return best

    return eval_fn


def make_far_eval_fn(sdk, image, account, lustre_far_root):
    """
    Returns an eval_fn(rec, train_job_id) that:
      1. Finds the best checkpoint from the training job on Lustre
      2. Submits a SLURM inference job on the KPI test set (7.0.1-pyt, cached SQSH)
      3. Polls for completion
      4. Runs the FAR script on Lustre and reads metric_result.json
      Returns FAR value (lower = better) or None on failure.
    """
    import subprocess as _sp, time as _time, yaml as _yaml, pathlib as _pl

    FAR_IMAGE = "nvcr.io/nvidia/tao/tao-toolkit:7.0.1-pyt"
    FAR_SCRIPT_LOCAL = str(SB / "far_eval_inline.py")

    # Write FAR computation script once
    from far_eval import _FAR_INLINE_PY
    _pl.Path(FAR_SCRIPT_LOCAL).write_text(_FAR_INLINE_PY)

    def _ssh(cmd):
        key  = os.environ.get("SSH_KEY_PATH", "")
        user = os.environ.get("SLURM_USER", "")
        host = os.environ.get("SLURM_HOSTNAME", "").split(",")[0]
        r = _sp.run(["ssh", "-i", key, "-o", "StrictHostKeyChecking=no",
                     f"{user}@{host}", cmd],
                    capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(f"SSH failed: {r.stderr[:300]}")
        return r.stdout.strip()

    def _scp(local, remote):
        key  = os.environ.get("SSH_KEY_PATH", "")
        user = os.environ.get("SLURM_USER", "")
        host = os.environ.get("SLURM_HOSTNAME", "").split(",")[0]
        _sp.run(["scp", "-i", key, "-o", "StrictHostKeyChecking=no",
                 local, f"{user}@{host}:{remote}"], check=True)

    def eval_fn(rec, train_job_id):
        # Use first 8 chars of training job UUID as rec_id — always a clean hex string
        rec_id = str(train_job_id)[:8]
        log.info("[eval_fn] rec=%s train_job=%s — submitting KPI inference", rec_id, train_job_id)

        # Locate checkpoint and experiment.yaml via SDK (correct Lustre path)
        try:
            lustre_raw = sdk.get_job_results_dir(train_job_id)
            # SDK returns URI like "lustre:///path" — strip scheme for SSH filesystem access
            lustre_job = lustre_raw.replace("lustre:///", "/").replace("lustre://", "/")
            log.info("[eval_fn] rec=%s: results_dir=%s", rec_id, lustre_job)
        except Exception as e:
            log.warning("[eval_fn] rec=%s: cannot get results_dir: %s", rec_id, e)
            return None

        try:
            train_dir = _ssh(
                f"[ -d {lustre_job}/results_dir/train ] && echo {lustre_job}/results_dir/train "
                f"|| echo {lustre_job}/train"
            )
            exp_yaml_str = _ssh(f"cat {train_dir}/experiment.yaml 2>/dev/null")
        except Exception as e:
            log.warning("[eval_fn] rec=%s: cannot read train dir/yaml: %s", rec_id, e)
            return None

        if not exp_yaml_str:
            log.warning("[eval_fn] rec=%s: missing experiment.yaml", rec_id)
            return None

        # Template spec for the sweep driver (strips 7.1.0-only fields for the
        # 7.0.1 eval container; driver fills checkpoint + dataset per pass)
        import yaml as _yaml
        spec = _yaml.safe_load(exp_yaml_str)
        spec["task"] = "classify"
        spec.get("train", {}).pop("checkpointer", None)
        spec.get("train", {}).pop("precision", None)
        spec.get("train", {}).pop("sync_batchnorm", None)
        spec.get("train", {}).pop("use_distributed_sampler", None)

        # The training-side checkpointer (monitor=val_far, patched package) has
        # already selected the best-val-FAR epoch → single checkpoint to score.
        try:
            ckpt = _ssh(
                f"find {train_dir} -name 'model_best_*.pth' 2>/dev/null | sort | tail -1 "
                f"|| readlink -f {train_dir}/changenet_model_classify_latest.pth 2>/dev/null"
            )
        except Exception as e:
            log.warning("[eval_fn] rec=%s: checkpoint lookup failed: %s", rec_id, e)
            return None
        if not ckpt:
            log.warning("[eval_fn] rec=%s: no checkpoint found in %s", rec_id, train_dir)
            return None
        import re as _re
        m = _re.search(r"model_best_(\d+)", ckpt)
        best_epoch = int(m.group(1)) if m else None

        lustre_out = f"{lustre_far_root}/rec_{rec_id}"
        infer_out  = f"{lustre_out}/inference"
        far_out    = f"{lustre_out}/far"
        spec["dataset"]["classify"]["infer_dataset"] = {
            "csv_path": LUSTRE_KPI_CSV, "images_dir": LUSTRE_IMAGES_DIR,
        }
        spec["inference"] = {
            "checkpoint": ckpt, "batch_size": 16,
            "results_dir": infer_out, "num_gpus": 1, "gpu_ids": [0], "num_nodes": 1,
        }
        spec["results_dir"] = infer_out

        local_spec = f"/tmp/far_eval_rec_{rec_id}.yaml"
        lustre_spec   = f"{lustre_out}/infer_spec.yaml"
        lustre_script = f"{lustre_out}/far_compute.py"
        _pl.Path(local_spec).write_text(_yaml.dump(spec, default_flow_style=False))
        try:
            _ssh(f"mkdir -p {lustre_out} {infer_out} {far_out}")
            _scp(local_spec, lustre_spec)
            _scp(FAR_SCRIPT_LOCAL, lustre_script)
        except Exception as e:
            log.warning("[eval_fn] rec=%s: scp failed: %s", rec_id, e)
            return None

        far_json = f"{far_out}/metric_result.json"
        log.info("[eval_fn] rec=%s: scoring best-val_far checkpoint (epoch %s)",
                 rec_id, best_epoch)
        command = (
            f"visual_changenet inference -e {lustre_spec} && "
            f"python3 {lustre_script} {infer_out}/inference.csv {far_json}"
        )

        try:
            job = sdk.create_job(
                image=FAR_IMAGE, command=command, gpu_count=1,
                account=account or None, num_nodes=1,
            )
            job_id = job.id if hasattr(job, "id") else str(job)
            log.info("[eval_fn] rec=%s: FAR inference job=%s", rec_id, job_id)
        except Exception as e:
            log.warning("[eval_fn] rec=%s: submit failed: %s", rec_id, e)
            return None

        # Poll (sweep evals ~30-40 min + queue; allow up to 2h)
        terminal_ok  = {"COMPLETE", "COMPLETED", "SUCCESS"}
        terminal_err = {"ERROR", "FAILED", "CANCELLED", "FAILURE", "TIMEOUT"}
        for _ in range(240):
            _time.sleep(30)
            try:
                raw = sdk.get_job_status(job_id)
                if hasattr(raw, "status"):
                    status = str(raw.status).upper()
                elif hasattr(raw, "value"):
                    status = str(raw.value).upper()
                else:
                    status = str(raw).upper()
            except Exception:
                continue
            log.info("[eval_fn] rec=%s: FAR job %s → %s", rec_id, job_id, status)
            if status in terminal_ok:
                break
            if status in terminal_err:
                log.warning("[eval_fn] rec=%s: FAR job failed (%s)", rec_id, status)
                return None
        else:
            log.warning("[eval_fn] rec=%s: FAR job timed out", rec_id)
            return None

        # Read result
        try:
            import json as _json
            raw_result = _ssh(f"cat {far_json}")
            d = _json.loads(raw_result)
            far = d["value"]
            diag = d.get("diagnostics", {})
            log.info("[eval_fn] rec=%s: FAR@100%%R = %.2f%% (best_epoch=%s, "
                     "val_far=%.2f%%, curve=%s)",
                     rec_id, far, diag.get("best_epoch"),
                     diag.get("val_far_selected", float("nan")),
                     diag.get("val_far_curve"))
            return far
        except Exception as e:
            log.warning("[eval_fn] rec=%s: cannot read FAR result: %s", rec_id, e)
            return None

    return eval_fn


def launch_llm_narrowed(algo_name, cfg, log_dir, warmup_recs=12):
    """Two-stage LLM-guided search-space restriction for any base algorithm.

    Stage 1: <warmup_recs> recs over the full space (fresh brain).
    Boundary: LLM proposes narrowed per-param bounds from stage-1 observations
              (direction-aware prompt: MINIMIZE far_pct; guardrailed in code).
    Stage 2: remaining budget with a FRESH brain inside the narrowed box —
             restarting avoids warping the GP's normalized coordinate space
             (the same reason TAO's hybrid brain re-instantiates sub-brains
             per phase instead of mutating ranges mid-run).
    """
    import datetime
    import json as _json
    from tao_automl.runner import AutoMLRunner

    total = cfg["settings"].get("num_recommendations", 40)
    hp_list = cfg.get("hyperparameters", AUTOML_HPS)
    hp_ranges = cfg.get("ranges")
    base_box = resolve_base_ranges(hp_list, hp_ranges)

    def _clamp(rec):
        n_ep = rec.specs.get("train.num_epochs")
        n_ep = int(n_ep) if n_ep else BASE_SPEC["train"]["num_epochs"]
        rec.specs["train.checkpoint_interval"] = min(10, n_ep)
        rec.specs["train.validation_interval"] = min(10, n_ep)

    def _stage(stage_tag, n_recs, ranges):
        sdk = make_sdk()
        runner = AutoMLRunner(sdk=sdk, skill_dir=str(SKILL_DIR), action="train")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ws = str(WORKSPACE / f"automl_{algo_name}_{stage_tag}_{ts}")
        settings = {"algorithm": cfg["algorithm"], **cfg["settings"],
                    "num_recommendations": n_recs}
        settings.setdefault("direction", cfg.get("direction", "minimize"))
        eval_fn = make_far_eval_fn(
            sdk, IMAGE, ACCOUNT,
            f"{LUSTRE_AOI_ROOT}/far_eval_automl/{algo_name}_{stage_tag}")
        log.info("[%s] %s: %d recs, %s space", algo_name, stage_tag, n_recs,
                 "narrowed" if stage_tag == "s2" else "full")
        res = runner.run(
            image=IMAGE, spec_overrides=dict(BASE_SPEC),
            automl_settings=settings,
            automl_hyperparameters=hp_list, custom_param_ranges=ranges,
            workspace_path=ws, eval_fn=eval_fn, on_recommendation=_clamp,
            account=ACCOUNT or None, num_nodes=1,
            env_vars={"PYTHONPATH": f"{LUSTRE_AOI_ROOT}/patches/valfar"},
        )
        return res, ws

    r1, ws1 = _stage("s1", warmup_recs, hp_ranges)

    # Observations come from the controller state (the returned history does
    # not serialize per-rec specs). Each entry: id/specs/status/result.
    import glob as _glob
    obs = []
    hp_set = set(hp_list)
    for cf in _glob.glob(f"{ws1}/run_*/.automl/controller/*.json"):
        for rec in _json.load(open(cf)):
            if rec.get("status") == "success" and rec.get("result") is not None:
                params = {k: v for k, v in (rec.get("specs") or {}).items()
                          if k in hp_set}
                if params:
                    obs.append({"params": params,
                                "far_pct": float(rec["result"])})
    if len(obs) < 3:
        log.warning("[%s] only %d usable stage-1 observations — skipping "
                    "narrowing, stage 2 runs the full space", algo_name, len(obs))
        narrowed = base_box
    else:
        narrowed = llm_narrow_ranges(obs, hp_list, base_box, algo_name)

    r2, _ws2 = _stage("s2", max(1, total - warmup_recs), narrowed)

    bests = []
    for r, stage in [(r1, "stage1"), (r2, "stage2")]:
        b = (r or {}).get("best") or {}
        mv = b.get("metric_value", b.get("metric"))
        if mv is not None:
            bests.append((float(mv), stage, b))
    if bests:
        best_val, best_stage, best_rec = min(bests, key=lambda t: t[0])
        out = Path(log_dir) / f"summary_{algo_name}.json"
        out.write_text(_json.dumps({
            "algorithm": algo_name, "protocol": "two-stage-llm-narrowing",
            "warmup_recs": warmup_recs,
            "best_metric_value": best_val, "best_stage": best_stage,
            "best_rec_id": best_rec.get("rec_id"),
            "best_specs": best_rec.get("specs", {}),
            "narrowed_ranges": narrowed,
        }, indent=2, default=str))
        log.info("[%s] DONE two-stage: best FAR=%.2f%% (%s)",
                 algo_name, best_val, best_stage)
    return {"stage1": r1, "stage2": r2, "narrowed": narrowed}


# ── DEFT-iteration AutoML (run #2: "AutoML within DEFT") ─────────────────────
# Narrowed search box proven by bayesian_llmnarrow stage 2 (the 13.44% arm).
DEFT_ITER_RANGES = {
    "train.optim.lr":                     {"valid_min": 1e-06, "valid_max": 0.0005},
    "train.optim.weight_decay":           {"valid_min": 0.1,   "valid_max": 0.8},
    "train.optim.momentum":               {"valid_min": 0.4,   "valid_max": 0.95},
    "model.classify.train_margin_euclid": {"valid_min": 1,     "valid_max": 8},
    "model.classify.eval_margin":         {"valid_min": 0.5,   "valid_max": 8},
    "model.classify.learnable_difference_modules": {"valid_min": 2, "valid_max": 3},
    "dataset.classify.fpratio_sampling":  {"valid_min": 0.35,  "valid_max": 0.95},
    "dataset.classify.augmentation_config.random_color.brightness": {"valid_min": 0.8, "valid_max": 2},
    "dataset.classify.augmentation_config.random_color.contrast":   {"valid_min": 0.8, "valid_max": 2},
    "dataset.classify.augmentation_config.random_color.saturation": {"valid_min": 0.6, "valid_max": 1.8},
    "dataset.classify.augmentation_config.random_rotate.rotate_probability": {"valid_min": 0.1, "valid_max": 0.7},
}

# Warm-start iteration search space: ARCHITECTURE params (ldm) must be pinned
# to the chained checkpoint's architecture — searching them breaks state_dict
# loading (decoder width mismatch). 10 non-architectural params remain.
DEFT_ITER_PARAMS_WARMSTART = [p for p in AUTOML_HPS
                              if p != "model.classify.learnable_difference_modules"]
DEFT_ITER_RANGES_WARMSTART = {k: v for k, v in DEFT_ITER_RANGES.items()
                              if k != "model.classify.learnable_difference_modules"}


def launch_fixed_train(tag, pinned_values, spec_extra=None, num_epochs=10,
                       train_csv=None, images_dir=None, prev_ckpt=None):
    """Fixed-spec training on SLURM, implemented as a 1-rec AutoML with every
    searched param pinned (valid_min == valid_max == value): reuses spec
    upload, job submission, val_far checkpointing, and the KPI FAR eval_fn."""
    import datetime, copy
    from tao_automl.runner import AutoMLRunner

    sdk = make_sdk()
    runner = AutoMLRunner(sdk=sdk, skill_dir=str(SKILL_DIR), action="train")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ws = str(WORKSPACE / f"automl_{tag}_{ts}")

    spec = copy.deepcopy(BASE_SPEC)
    spec["train"]["num_epochs"] = num_epochs
    spec["train"]["validation_interval"] = min(10, num_epochs)
    spec["train"]["checkpoint_interval"] = min(10, num_epochs)
    if prev_ckpt:
        spec["train"]["pretrained_model_path"] = prev_ckpt
    if train_csv:
        spec["dataset"]["classify"]["train_dataset"] = {
            "csv_path": train_csv, "images_dir": images_dir or LUSTRE_IMAGES_DIR}
    for k, v in (spec_extra or {}).items():
        spec[k] = v

    # Epsilon-width ranges (not zero-width): the value generator's boundary
    # clamp rewrites values sitting exactly on valid_min/valid_max (x1.1/x0.9),
    # so a degenerate min==max pin comes out 10% off. A +/-1e-6 relative window
    # keeps the sample strictly interior => exact pin (ints unaffected).
    def _pin(val):
        if isinstance(val, int):
            return {"valid_min": val, "valid_max": val}
        eps = max(abs(val) * 1e-6, 1e-12)
        return {"valid_min": val - eps, "valid_max": val + eps}
    ranges = {name: _pin(val) for name, val in pinned_values.items()}
    settings = {"algorithm": "bayesian", "num_recommendations": 1,
                "train_sample_count": 260, "metric": "far_pct",
                "direction": "minimize"}
    eval_fn = make_far_eval_fn(
        sdk, IMAGE, ACCOUNT, f"{LUSTRE_AOI_ROOT}/far_eval_automl/{tag}")

    def _clamp(rec):
        rec.specs["train.checkpoint_interval"] = min(10, num_epochs)
        rec.specs["train.validation_interval"] = min(10, num_epochs)

    log.info("[%s] fixed-spec SLURM train: %d epochs, pinned %d params",
             tag, num_epochs, len(pinned_values))
    result = runner.run(
        image=IMAGE, spec_overrides=spec, automl_settings=settings,
        automl_hyperparameters=list(pinned_values), custom_param_ranges=ranges,
        workspace_path=ws, eval_fn=eval_fn, on_recommendation=_clamp,
        account=ACCOUNT or None, num_nodes=1,
        env_vars={"PYTHONPATH": f"{LUSTRE_AOI_ROOT}/patches/valfar"},
    )
    import glob as _glob, json as _json
    out = {"job_id": None, "far": None, "specs": {}, "workspace": ws}
    for cf in _glob.glob(f"{ws}/run_*/.automl/controller/*.json"):
        for r in _json.load(open(cf)):
            if r["status"] == "success" and r.get("result") is not None:
                out.update(job_id=r["job_id"], far=r["result"], specs=r["specs"])
    log.info("[%s] fixed train done: FAR=%s job=%s", tag, out["far"], out["job_id"])
    return out


# Original DEFT baseline defaults (workspace2/specs/baseline_spec.yaml)
DEFT2_BASELINE_PINS = {
    "train.optim.lr": 1e-5,
    "train.optim.weight_decay": 0.01,
    "train.optim.momentum": 0.9,
    "model.classify.train_margin_euclid": 2.0,
    "model.classify.eval_margin": 0.3,
    "model.classify.learnable_difference_modules": 4,
    "dataset.classify.fpratio_sampling": 0.2,
    "dataset.classify.augmentation_config.random_color.brightness": 0.3,
    "dataset.classify.augmentation_config.random_color.contrast": 0.3,
    "dataset.classify.augmentation_config.random_color.saturation": 0.3,
    "dataset.classify.augmentation_config.random_rotate.rotate_probability": 0.5,
}


def launch_deft_iter_automl(iter_label, train_csv_lustre, images_dir_lustre,
                            prev_ckpt_lustre, num_recs=6, num_epochs=10,
                            log_dir=None, search_params=None, search_ranges=None):
    """One DEFT iteration's retrain implemented as a small BFBO search.

    Each recommendation trains num_epochs from prev_ckpt (pretrained_model_path
    init — fresh epoch counter) on the iteration's combined CSV, with val_far
    checkpointing; eval_fn scores KPI FAR on SLURM. Returns the winner dict
    {job_id, far, specs} for the DEFT loop to download/commit.
    """
    import datetime, copy
    from tao_automl.runner import AutoMLRunner

    algo_name = f"deft2_{iter_label}"
    sdk = make_sdk()
    runner = AutoMLRunner(sdk=sdk, skill_dir=str(SKILL_DIR), action="train")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ws = str(WORKSPACE / f"automl_{algo_name}_{ts}")

    spec = copy.deepcopy(BASE_SPEC)
    spec["train"]["num_epochs"] = num_epochs
    spec["train"]["validation_interval"] = min(10, num_epochs)
    spec["train"]["checkpoint_interval"] = min(10, num_epochs)
    spec["train"]["pretrained_model_path"] = prev_ckpt_lustre
    spec["dataset"]["classify"]["train_dataset"] = {
        "csv_path": train_csv_lustre, "images_dir": images_dir_lustre}

    settings = {"algorithm": "bfbo", "num_recommendations": num_recs,
                "train_sample_count": 260, "metric": "far_pct",
                "direction": "minimize"}
    eval_fn = make_far_eval_fn(
        sdk, IMAGE, ACCOUNT, f"{LUSTRE_AOI_ROOT}/far_eval_automl/{algo_name}")

    def _clamp(rec):
        # use the REC's sampled epoch count (num_epochs may be searched),
        # not the fixed default — otherwise short recs violate
        # checkpoint_interval <= num_epochs and abort before epoch 1
        n_ep = int(rec.specs.get("train.num_epochs") or num_epochs)
        rec.specs["train.checkpoint_interval"] = min(10, n_ep)
        rec.specs["train.validation_interval"] = min(10, n_ep)

    log.info("[%s] DEFT-iter AutoML: %d recs x %d epochs from %s",
             algo_name, num_recs, num_epochs, prev_ckpt_lustre)
    result = runner.run(
        image=IMAGE, spec_overrides=spec, automl_settings=settings,
        automl_hyperparameters=search_params or AUTOML_HPS,
        custom_param_ranges=search_ranges or DEFT_ITER_RANGES,
        workspace_path=ws, eval_fn=eval_fn, on_recommendation=_clamp,
        account=ACCOUNT or None, num_nodes=1,
        env_vars={"PYTHONPATH": f"{LUSTRE_AOI_ROOT}/patches/valfar"},
    )
    best = (result or {}).get("best", {})
    out = {"job_id": None, "far": best.get("metric_value", best.get("metric")),
           "specs": best.get("specs", {}), "workspace": ws}
    # controller state carries the winner job id
    import glob as _glob, json as _json
    cands = []
    for cf in _glob.glob(f"{ws}/run_*/.automl/controller/*.json"):
        for r in _json.load(open(cf)):
            if r["status"] == "success" and r.get("result") is not None:
                cands.append(r)
    if cands:
        w = min(cands, key=lambda r: r["result"])
        out.update(job_id=w["job_id"], far=w["result"], specs=w["specs"])
    log.info("[%s] iteration winner: FAR=%.4f%% job=%s", algo_name,
             out["far"], out["job_id"])
    return out


def _run_and_log(algo_name, log_dir):
    """Module-level worker for multiprocessing.Pool (must be picklable)."""
    import logging as _logging
    log_file = Path(log_dir) / f"automl_{algo_name}.log"
    handler = _logging.FileHandler(log_file)
    handler.setFormatter(_logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _logging.getLogger().addHandler(handler)
    try:
        cfg = ALGO_CONFIGS[algo_name]
        if cfg.get("llm_narrow"):
            return launch_llm_narrowed(
                algo_name, cfg, log_dir,
                warmup_recs=cfg["llm_narrow"].get("warmup_recs", 12))
        return launch_one(algo_name, cfg, log_dir)
    except Exception as e:
        _logging.getLogger().exception("[%s] failed: %s", algo_name, e)
        return None
    finally:
        _logging.getLogger().removeHandler(handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoML v2 VCN SLURM runner")
    parser.add_argument("--algorithm", default="all",
                        choices=list(ALGO_CONFIGS) + list(ALGO_GROUPS) + ["far_only"])
    parser.add_argument("--log-dir", default=str(WORKSPACE / "automl_logs_v2"))
    args = parser.parse_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # Validate LLM config if needed
    llm_algos = {"llm_range", "hybrid", "hybrid_llm"}
    running = (ALGO_GROUPS.get(args.algorithm, [args.algorithm])
               if args.algorithm != "far_only" else [])
    if any(a in llm_algos for a in running):
        missing = [k for k,v in [("base_url/llm_endpoint", LLM_ENDPOINT),
                                  ("model/llm_model", LLM_MODEL),
                                  ("AUTOML_LLM_API_KEY", LLM_API_KEY)] if not v]
        if missing:
            sys.exit(f"ERROR: LLM config missing: {missing}. "
                     "Set base_url, model, AUTOML_LLM_API_KEY in ~/.tao/config.env")
        print(f"LLM endpoint: {LLM_ENDPOINT} | model: {LLM_MODEL} | key: SET")

    args_log_dir = args.log_dir  # make accessible inside score_bayesian_winner

    if args.algorithm == "far_only":
        score_bayesian_winner()
        sys.exit(0)

    algos_to_run = ALGO_GROUPS.get(args.algorithm, [args.algorithm])

    if len(algos_to_run) == 1:
        _run_and_log(algos_to_run[0], args.log_dir)
    else:
        import multiprocessing
        print(f"Launching {len(algos_to_run)} algorithms in parallel: {algos_to_run}")
        log_dir = args.log_dir
        with multiprocessing.Pool(len(algos_to_run)) as pool:
            pool.starmap(_run_and_log, [(a, log_dir) for a in algos_to_run])
        print("\nAll done. Summaries in:", args.log_dir)
