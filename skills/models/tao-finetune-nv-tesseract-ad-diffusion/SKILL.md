---
name: tao-finetune-nv-tesseract-ad-diffusion
description: >-
  NV-Tesseract AD Diffusion — diffusion-based anomaly detection and fine-tuning
  for multivariate time series. Use when the user asks to "fine-tune NV-Tesseract",
  "run AD diffusion inference", "detect anomalies with diffusion", "time series
  anomaly detection", "finetune ad-diffusion", "use perform_anomaly_analysis_with_diffusion",
  "automl ad-diffusion", "hyperparameter search ad-diffusion", "hyperparamter optimization" 
  or mentions "curriculum_medium.yaml", "final_model.pth",
  "nv-tesseract-ad-diffusion", "ad_diffusion", or "TSDiffuser_Generic".
license: Apache-2.0
compatibility: Requires Python 3.12+ and uv. CUDA GPU recommended; CPU-only supported.
metadata:
  author: NVIDIA Corporation
  version: "0.2.0"
allowed-tools: Read Bash
tags:
  - anomaly-detection
  - time-series
  - diffusion
  - finetune
  - inference
  - automl
  - nv-tesseract
---

# NV-Tesseract AD Diffusion

Diffusion-based anomaly detection and fine-tuning for multivariate time series. The model
reconstructs randomly masked segments and scores each timestep by MAE between reconstruction
and original signal; adaptive thresholding (SCS or MACS) converts scores to binary labels.

**Source code:** https://github.com/NVIDIA/NV-Tesseract
**Pretrained weights:** https://huggingface.co/nvidia/nv-tesseract-ad-diffusion

> **For the most up-to-date usage information**, refer to the README files in
> the NV-Tesseract repository:
>
> - **[`ad_diffusion/README.md`](https://github.com/NVIDIA/NV-Tesseract/blob/main/ad_diffusion/README.md)** — full SDK reference, model architecture, and API docs
> - **[`ad_diffusion/examples/datasets/README.md`](https://github.com/NVIDIA/NV-Tesseract/blob/main/ad_diffusion/examples/datasets/README.md)** — dataset format, synthetic data generation, and CSV conventions

## External dependencies

| Dependency | Purpose | Install |
|---|---|---|
| Python 3.12+ | Runtime | https://www.python.org/downloads/ |
| uv | Package + environment manager | `pip install uv` |
| CUDA toolkit (optional) | GPU acceleration | https://developer.nvidia.com/cuda-downloads |
| huggingface_hub | Weight download from HF | Bundled via `uv sync` |

## Credentials

`nvidia/nv-tesseract-ad-diffusion` is a public repo — no token required for downloading weights.
If you ever hit a `401`/`403` (gated access or private fork) or a `504` on first download, see the Known pitfalls section.

## Quick start

```bash
cd /path/to/NV-Tesseract/ad_diffusion  # git clone https://github.com/NVIDIA/NV-Tesseract
uv sync                              # install dependencies (one-time)

# Inference — synthetic data, auto-downloads weights from HF on first run
uv run python examples/quick_example.py

# Inference — your own CSV
uv run python examples/quick_example.py \
  --model-path final_model.pth \
  --config-path curriculum_medium.yaml \
  --dataset-path /path/to/data.csv

# Pre-download weights only (warm the cache before going offline)
uv run python examples/quick_example.py --download-weights

# Fine-tune on your own normal-behavior data
uv run python examples/finetune_example.py \
  --csv /path/to/normal_training_data.csv \
  --timestamp-col timestamp \
  --label-col is_anomaly \
  --epochs 20 \
  --output-dir artifacts/finetune_my_data
```

## Inference

Use `perform_anomaly_analysis_with_diffusion` in `sdk/anomaly_analysis.py`. It validates
input, auto-dispatches across all visible GPUs, applies adaptive thresholding (SCS or MACS),
and returns the original DataFrame with `Anomaly` (0/1) and `MAE` columns appended.

```python
import sys, pandas as pd
sys.path.append("/path/to/NV-Tesseract/ad_diffusion")  # git clone https://github.com/NVIDIA/NV-Tesseract
from sdk.anomaly_analysis import perform_anomaly_analysis_with_diffusion

df = pd.read_csv("your_data.csv")

# The API raises ValueError on non-numeric columns — drop timestamp, IDs, and labels first.
df = df.select_dtypes(include="number")

results = perform_anomaly_analysis_with_diffusion(
    df=df,
    threshold_strategy="scs",       # "scs" (fast) or "macs" (adaptive)
    model_path=None,                 # None → auto-download final_model.pth from HF
    config_path=None,                # None → auto-download curriculum_medium.yaml from HF
    nsample=15,                      # diffusion samples per window; ↑ accuracy, ↑ latency
    preprocess_model_dir=None,       # optional preprocessing model directory
)
# results columns: Anomaly (0/1), MAE (float), plus all original columns
print(results[["Anomaly", "MAE"]].describe())
```

### Inference CLI reference

| Argument | Default | Description |
|---|---|---|
| `--dataset-path` | synthetic | CSV with numeric feature columns |
| `--model-path` | auto-download | Path to `.pth` checkpoint |
| `--config-path` | auto-download | Path to `curriculum_medium.yaml` |
| `--download-weights` | — | Fetch weights from HF and exit |
| `--skip-download` | false | Require local weights; skip HF fetch |

## Fine-tuning

Fine-tune on your own data. The training CSV should contain mostly normal behavior. Validate the pretrained model on your domain before fine-tuning.

```bash
uv run python examples/finetune_example.py \
  --csv /path/to/normal_data.csv \
  --val-csv /path/to/val_data.csv \   # optional; otherwise --val-ratio splits --csv
  --pretrained-model final_model.pth \
  --epochs 20 --batch-size 16 --lr 1e-5 \
  --output-dir artifacts/finetune_my_data
```

### Fine-tuning arguments

| Argument | Default | Description |
|---|---|---|
| `--run-config` | — | JSON/YAML config file generated by AutoMLRunner (`{config_path}`). All fields below can be set here; explicit CLI flags override the file. |
| `--csv` | **required** | Training CSV (ideally containing normal behavior). Required if not supplied via `--run-config`. |
| `--val-csv` | — | Separate validation CSV |
| `--val-ratio` | `0.3` | Validation fraction when `--val-csv` not used (temporal split) |
| `--timestamp-col` | `timestamp` | Column to drop from features |
| `--label-col` | — | Label column to drop |
| `--drop-cols` | — | Comma-separated extra columns to drop |
| `--pretrained-model` | `final_model.pth` | Warm-start checkpoint (auto-downloaded if missing) |
| `--config` | `curriculum_medium.yaml` | Model config YAML |
| `--repo-id` | `nvidia/nv-tesseract-ad-diffusion` | HuggingFace repo for auto-download |
| `--no-download` | false | Fail if pretrained weights are not local |
| `--epochs` | `10` | Training epochs |
| `--batch-size` | `16` | Per-GPU batch size |
| `--lr` | `1e-5` | AdamW learning rate |
| `--weight-decay` | `1e-6` | AdamW weight decay |
| `--grad-clip` | `1.0` | Gradient norm clip |
| `--num-workers` | `0` | DataLoader workers |
| `--seed` | `42` | Random seed |
| `--output-dir` | `artifacts/finetune` | Output directory |
| `--window-length` | config (100) | Sliding window length in timesteps |
| `--window-stride` | `1` | Step between consecutive windows |
| `--split` | config (10) | Alternating mask segments per window |
| `--mask-ratio` | `0.7` | Fraction of each window masked during training |
| `--scale-factor` | config (1) | Scale multiplier after min-max normalization |
| `--num-gpus` | all available | Number of GPUs for DDP fine-tuning; set `1` to force single-GPU |

### Running inference with a fine-tuned checkpoint

```python
results = perform_anomaly_analysis_with_diffusion(
    df=df,
    threshold_strategy="scs",
    model_path="artifacts/finetune_my_data/best_finetuned_model.pth",
    config_path="artifacts/finetune_my_data/finetune_config.yaml",
    nsample=15,
)
```

## AutoML (HPO: hyperparameter optimization)

> For algorithm selection, budget configuration, preflight checks, monitoring,
> and result handoff, use the **`tao-skill-bank:tao-run-automl`** skill. This
> section documents what AD Diffusion tunes, the VirtualEnvSDK setup required
> for this containerless model, and the inference HPO trial script.

This skill supports AutoML for two use cases — both run locally via
`VirtualEnvSDK` with no container required:

| Use case | Action | Metric | Requires labels? |
|---|---|---|---|
| Fine-tuning HPO | `train` | `val_loss` (minimize) | No |
| Inference HPO | `inference_hpo` | `f1_score` (maximize) | Yes |

### What AutoML tunes

#### Fine-tuning

| Parameter | Spec key | Default | Search range | Why it matters |
|---|---|---|---|---|
| Learning rate | `train.lr` | `1e-5` | `[1e-7, 1e-3]` | Most impactful — diffusion backbone is sensitive to LR |
| Weight decay | `train.weight_decay` | `1e-6` | `[1e-9, 1e-3]` | Regularization; prevents overfitting on small datasets |
| Batch size | `train.batch_size` | `16` | `[4, 32]` | Larger batches stabilize gradients; capped at 32 (MPS OOM) |
| Gradient clip | `train.grad_clip` | `1.0` | `[0.1, 10.0]` | Prevents gradient explosion in diffusion fine-tuning |
| Mask ratio | `dataset.mask_ratio` | `0.7` | `[0.3, 0.9]` | Fraction of each window masked; shapes reconstruction difficulty |
| Window length | `dataset.window_length` | `100` | `[50, 140]` | Temporal context per training sample; must be divisible by `split` |

**Fixed (not searched):** `dataset.csv`, `dataset.val_csv`, `dataset.split`,
`dataset.window_stride`, `train.seed`, `train.output_dir`, `train.pretrained_model`,
`train.config`.

#### Inference HPO

| Parameter | Spec key | Default | Search range | Effect |
|---|---|---|---|---|
| Diffusion samples | `inference.nsample` | `15` | `[5, 30]` | More samples → better MAE estimate, slower per trial |

**`threshold_strategy`** (`scs` or `macs`) is fixed per AutoML run — run once
per strategy and compare F1 to pick the winner.

**Fixed:** `inference.model_path`, `inference.config_path` — set these in
`spec_overrides`, not the search space.

### AutoML artifacts

| Use case | Schema | Spec template |
|---|---|---|
| Fine-tuning | `schemas/train.schema.json` | `references/spec_template_train.yaml` |
| Inference HPO | `schemas/inference_hpo.schema.json` | `references/spec_template_inference_hpo.yaml` |

### Installing the AutoML runner

```bash
cd /path/to/NV-Tesseract/ad_diffusion
uv sync                              # project deps already installed

# AutoML packages (install into the same venv)
uv pip install "nvidia-tao-sdk>=7.1.0" "nvidia-tao-automl>=7.1.0"

# Verify
uv run python -c "
from tao_sdk.platforms.virtualenv import VirtualEnvSDK
from tao_automl.runner import AutoMLRunner
print('VirtualEnvSDK OK')
"
```

> **Note:** `VirtualEnvSDK` and `python_script` execution require
> `nvidia-tao-automl>=7.1.0` and `nvidia-tao-sdk>=7.1.0`.
> Install from PyPI: `pip install nvidia-tao-automl>=7.1.0 nvidia-tao-sdk>=7.1.0`.

### Fine-tuning AutoML

```python
from pathlib import Path
from tao_sdk.platforms.virtualenv import VirtualEnvSDK
from tao_automl.runner import AutoMLRunner

ad_diffusion_dir = Path("/path/to/NV-Tesseract/ad_diffusion")
skill_dir = Path("/path/to/tao-skill-bank/skills/models/tao-finetune-nv-tesseract-ad-diffusion")

sdk = VirtualEnvSDK(venv_path=str(ad_diffusion_dir / ".venv"))  # created by `uv sync`

runner = AutoMLRunner(sdk=sdk, skill_dir=str(skill_dir), action="train")

result = runner.run(
    automl_settings={
        "algorithm":                  "bayesian",
        "metric":                     "val_loss",
        "direction":                  "minimize",
        "automl_max_recommendations": 10,           # adjust to your compute budget
    },
    # spec_overrides MUST use dotted keys ("dataset.csv"), NOT nested dicts
    # ({"dataset": {"csv": ...}}). Nested dicts replace the entire sub-tree in
    # the spec template, which removes sibling keys like window_length from the
    # search space — AutoML silently excludes them as "deleted parameters".
    spec_overrides={
        "dataset.csv":        "/path/to/normal_data.csv",
        "train.epochs":       20,
        "train.output_dir":   "automl_workspace/finetune/trials",
    },
    # window_length must be divisible by split (default 10). The schema allows
    # any integer in [50, 140], so restrict to multiples of 10 explicitly —
    # otherwise the Bayesian sampler proposes values like 81 or 99 and every
    # trial fails with "window_length must be divisible by split".
    custom_param_ranges={
        "dataset.window_length": {
            "valid_options": [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
            "value_type":    "ordered_int",
        },
    },
    workspace_path="automl_workspace/finetune",
    execution={
        "type":          "python_script",
        "script":        str(ad_diffusion_dir / "examples/finetune_example.py"),
        "args":          ["--run-config", "{config_path}"],
        "cwd":           str(ad_diffusion_dir),
        "config_format": "yaml",
    },
)

best = result["best"]
print(f"Best val_loss : {best['metric_value']:.6f}")
print(f"Best params   : {best['specs']}")
```

The winning checkpoint is at `train.output_dir/rec_<N>/best_finetuned_model.pth`.
Pass it as `model_path` to `perform_anomaly_analysis_with_diffusion` for inference.

#### How the config flows (fine-tuning)

Each trial:
1. Samples hyperparameters from `schemas/train.schema.json`
2. Merges with `spec_template_train.yaml` defaults
3. Writes merged spec to `{config_path}` (YAML)
4. Calls `<venv>/bin/python finetune_example.py --run-config {config_path}`
5. Reads `val_loss` from `metrics.json` in the trial's `output_dir`

#### Window length constraint

`window_length` must be divisible by `split` (default `10`). The schema allows
any integer in `[50, 140]`, so the Bayesian sampler will propose non-multiples
like 81 or 99 — causing every trial to fail immediately with
`ValueError: window_length must be divisible by split`.

Always pass `custom_param_ranges` to restrict candidates to valid multiples:

```python
custom_param_ranges={
    "dataset.window_length": {
        "valid_options": [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
        "value_type":    "ordered_int",
    },
}
```

If your dataset is larger and you want a wider window, add more multiples of 10
to `valid_options` rather than raising the schema cap.

### Inference HPO with AutoML

Requires **labeled data** — a CSV with a ground-truth `label` column (0/1).
AutoML tunes `nsample` to maximize F1 on that eval set.

The trial script is not shipped with NV-Tesseract — generate it for the user's
working directory based on the template below, then point `execution.script` at it.

**Trial script template** (save anywhere, e.g. `inference_hpo_trial.py`):

```python
"""Single inference HPO trial — called by AutoMLRunner with --run-config {config_path}."""
import argparse, json, sys
from pathlib import Path
import yaml, pandas as pd
from sklearn.metrics import f1_score

AD_DIR = Path("/path/to/NV-Tesseract/ad_diffusion")   # adjust to actual clone path
sys.path.insert(0, str(AD_DIR))

from sdk.anomaly_analysis import perform_anomaly_analysis_with_diffusion

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-config", required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.run_config).read_text())
    ds  = cfg["dataset"]
    inf = cfg["inference"]
    out = Path(cfg.get("train", {}).get("output_dir", "artifacts/inference_hpo"))
    out.mkdir(parents=True, exist_ok=True)

    df      = pd.read_csv(ds["csv"]).head(ds.get("eval_rows", 2000))
    labels  = df[ds["label_col"]].values
    df_feat = df.drop(columns=[ds["label_col"]], errors="ignore").select_dtypes(include="number")

    results = perform_anomaly_analysis_with_diffusion(
        df=df_feat,
        nsample=inf["nsample"],
        threshold_strategy=inf["threshold_strategy"],
        model_path=inf.get("model_path") or None,
        config_path=inf.get("config_path") or None,
    )
    f1 = f1_score(labels, results["Anomaly"].values, zero_division=0)
    (out / "metrics.json").write_text(json.dumps({"f1_score": f1}))
    print(f"nsample={inf['nsample']}  strategy={inf['threshold_strategy']}  F1={f1:.4f}")

if __name__ == "__main__":
    main()
```

#### Run AutoML (once per strategy, compare F1)

```python
from pathlib import Path
from tao_sdk.platforms.virtualenv import VirtualEnvSDK
from tao_automl.runner import AutoMLRunner

ad_diffusion_dir = Path("/path/to/NV-Tesseract/ad_diffusion")
skill_dir = Path("/path/to/tao-skill-bank/skills/models/tao-finetune-nv-tesseract-ad-diffusion")

sdk    = VirtualEnvSDK(venv_path=str(ad_diffusion_dir / ".venv"))
runner = AutoMLRunner(sdk=sdk, skill_dir=str(skill_dir), action="inference_hpo")

results_by_strategy = {}
for strategy in ["scs", "macs"]:
    result = runner.run(
        automl_settings={
            "algorithm":                  "bayesian",
            "metric":                     "f1_score",
            "direction":                  "maximize",
            "automl_max_recommendations": 8,          # nsample range is narrow; 8 covers it well
        },
        # spec_overrides MUST use dotted keys — see fine-tuning section for why.
        spec_overrides={
            "dataset.csv":                  "/path/to/labeled_data.csv",
            "inference.threshold_strategy": strategy,
            "inference.model_path":         None,   # None → pretrained HF weights; or path to fine-tuned .pth
            "train.output_dir":             f"automl_workspace/inference_hpo_{strategy}/trials",
        },
        workspace_path=f"automl_workspace/inference_hpo_{strategy}",
        execution={
            "type":          "python_script",
            "script":        "/path/to/inference_hpo_trial.py",   # generated trial script
            "args":          ["--run-config", "{config_path}"],
            "cwd":           str(ad_diffusion_dir),
            "config_format": "yaml",
        },
    )
    best = result["best"]
    results_by_strategy[strategy] = best
    # best["specs"] uses the same dotted keys as spec_overrides
    print(f"{strategy:4s}  F1={best['metric_value']:.4f}  nsample={best['specs']['inference.nsample']}")

winner = max(results_by_strategy, key=lambda s: results_by_strategy[s]["metric_value"])
print(f"\nBest: strategy={winner}  nsample={results_by_strategy[winner]['specs']['inference.nsample']}")
```

**Timestamp / label columns:** drop them before passing to the inference function — the template above does this via `errors="ignore"`.

**Labeled data requirement:** skip inference HPO when you have no ground-truth
labels — without them there is no meaningful objective. Use `nsample=15` + `scs`
as reasonable defaults.


## Data requirements

| Property | Requirement |
|---|---|
| Rows | ≥ `window_length` (default **100**); ≥ `target_dim` (default **18**) for PCA |
| Columns | Must be numeric — the API raises `ValueError` on non-numeric columns; drop `timestamp`, IDs, and labels before calling |
| Values | No NaN / ±Inf — fill before passing to the API |
| Feature count > `target_dim` | PCA reduction to `target_dim`; needs ≥ `target_dim` rows |
| Feature count < `target_dim` | Zero-padded to `target_dim` |

```
timestamp,sensor_1,sensor_2,sensor_3
2024-01-01 00:00:00,0.42,1.10,-0.33
...
```

Pass only numeric feature columns to the inference API — it raises `ValueError` on
non-numeric columns rather than dropping them. Use `df.select_dtypes(include="number")`
or drop by name before calling. Fine-tuning handles this via `--timestamp-col`,
`--label-col`, and `--drop-cols` CLI args.

## Output structure

**Inference** (`examples/quick_example.py`):

```
examples/datasets/
└── anomaly_results.csv      # original columns + Anomaly (0/1) + MAE
```

**Fine-tuning** (`--output-dir artifacts/finetune_my_data`):

```
artifacts/finetune_my_data/
├── best_finetuned_model.pth     # checkpoint with lowest validation loss
├── final_finetuned_model.pth    # checkpoint from last epoch
├── metrics.json                 # scalar for AutoML: {"val_loss": <best>}
├── epoch_metrics.json           # per-epoch log: [{"epoch": N, "train_loss": …, "val_loss": …}]
└── finetune_config.yaml         # config used during training (for reproducibility)
```

## Model configuration (`curriculum_medium.yaml`)

| Field | Default | Description |
|---|---|---|
| `model.target_dim` | `18` | Internal feature dim; data is PCA'd/padded to this |
| `dataset.window_length` | `100` | Sliding window size in timesteps |
| `dataset.split` | `10` | Alternating mask segments per window |
| `dataset.scale_factor` | `1` | Scale multiplier after min-max normalization |
| `diffusion.num_steps` | `500` | Full diffusion steps (overridden by DPM-Solver) |
| `diffusion.channels` | `128` | Model hidden dimension |
| `diffusion.layers` | `6` | Transformer encoder layers |

## Hardware

| Tier | Setup | Notes |
|---|---|---|
| Minimum | 1× CPU | Functional; DPM-Solver reduces steps 500 → 20 |
| Recommended | 1× NVIDIA GPU (≥8 GB VRAM) | Strongly recommended for fine-tuning |
| Multi-GPU inference | 2–8× NVIDIA GPUs | auto-dispatched by `perform_anomaly_analysis_with_diffusion` |
| Multi-GPU fine-tuning | 2+× NVIDIA GPUs | Auto DDP via `--num-gpus` (defaults to all visible GPUs) |

## Known pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `HfHubHTTPError: 401` | Repo gated or token missing | `export HUGGINGFACE_HUB_TOKEN="hf_..."` or `huggingface-cli login` |
| `504` / timeout on first weight download | HF CDN throttles unauthenticated requests — public repos are still subject to this on first download | Set `export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"` before running; authenticated requests use a more reliable CDN path |
| `ValueError: No numeric columns` | All columns are strings/dates | Drop non-numeric columns before calling API |
| `ValueError: PCA needs at least target_dim rows` | Fewer rows than `target_dim` (18) | Provide a longer time series |
| `ValueError: Need at least N rows` (finetune) | Split shorter than `window_length` | Ensure each train/val split has ≥ 100 rows |
| `RuntimeError: CUDA out of memory` | Batch too large | Reduce `--batch-size` or `nsample` |
| All MAE scores identical | Constant-value columns | Drop zero-variance columns before calling API |
| `ModuleNotFoundError: sdk` | Wrong working directory | `cd ad_diffusion/` before `uv run`, or add it to `sys.path` |
| Slow inference on CPU | Many diffusion windows | Reduce `nsample` to 5–10 for smoke tests |
