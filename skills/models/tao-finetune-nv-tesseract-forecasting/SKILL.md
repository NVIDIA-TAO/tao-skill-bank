---
name: tao-finetune-nv-tesseract-forecasting
description: >-
  NV-Tesseract Forecasting — transformer-based multivariate time series forecasting
  with DARR (context-enhanced kNN retrieval), interpretability, and fine-tuning.
  Use when the user asks to "forecast with NV-Tesseract", "run forecasting inference",
  "use perform_forecasting", "DARR mode", "context-enhanced forecasting",
  "lag horizon attribution", "interpretability", or "fine-tune forecasting",
  or mentions "nv-tesseract-forecasting", "moment_head_512_6hr", or "run8_best_model_cr".
license: Apache-2.0
compatibility: Requires Python 3.10+ and uv. CUDA GPU recommended; Apple Silicon (MPS) and CPU supported.
metadata:
  # Upstream NV-Tesseract repo + uv workflow, not a TAO container: exempt from the
  # docker-native rule for skills/models/. See README "Docker-native first".
  requires_external: true
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
  - forecasting
  - time-series
  - darr
  - interpretability
  - finetune
  - inference
  - nv-tesseract
---

# NV-Tesseract Forecasting

Transformer-based multivariate time series forecasting using self-supervised pretraining on diverse temporal data.
Three inference modes: **standard** (direct forecast), **DARR** (context-enhanced kNN retrieval blending),
and **interpretability** (latent trajectory extraction, semantic flow, lag×horizon attribution,
trajectory stability, and diagnostic ratios — full explanation bundle with PDF report).
Fine-tuning adapts the forecasting head — and optionally the cross-channel layer — to your domain.

**Source code:** https://github.com/NVIDIA/NV-Tesseract  
**Pretrained weights:** https://huggingface.co/nvidia/nv-tesseract-forecasting

## External dependencies

| Dependency | Purpose | Install |
|---|---|---|
| Python 3.10+ | Runtime | https://www.python.org/downloads/ |
| uv | Package + environment manager | `pip install uv` |
| CUDA toolkit (optional) | GPU acceleration | https://developer.nvidia.com/cuda-downloads |
| matplotlib (optional) | Interpretability PDF report, heatmap PNG, flow + stability charts | `uv add matplotlib` |

## Credentials

`nvidia/nv-tesseract-forecasting` is a public repo — no token required for downloading weights.
If you hit a `401`/`403` (gated access or license not accepted) or a `504` on first download, see the Known pitfalls section.

## Quick start

```bash
git clone https://github.com/NVIDIA/NV-Tesseract
cd NV-Tesseract/forecasting
uv sync --group dev
uv pip install -e .          # editable install — required for clean sdk.* imports

# Standard inference (auto-downloads weights from HF on first run, no auth needed)
uv run python sdk/quick_example.py
```

## Inference

Import and call `perform_forecasting` from `sdk/forecasting.py`. It auto-downloads weights,
standardizes input, runs autoregressive rollout for long horizons, and returns a DataFrame
with `{target_column}_forecast` rows for the requested horizon.

```python
import sys, pandas as pd
sys.path.append("/path/to/NV-Tesseract/forecasting")  # git clone https://github.com/NVIDIA/NV-Tesseract
from sdk.forecasting import perform_forecasting

df = pd.read_csv("your_data.csv")   # must have timestamp + numeric target column

results = perform_forecasting(
    df=df,
    timestamp_column="timestamp",    # parseable datetime column
    target_column="target",          # primary target to forecast
    seq_len=512,                     # input context length (rows consumed)
    forecast_horizon=72,             # steps ahead to predict (max 512)
    model_horizon=72,                # native model horizon; change when using custom weights
    standardizer_pkl="standardizer.pkl",   # auto-downloaded from HF if missing
    ckpt="run8_best_model_cr.pt",          # auto-downloaded; see Checkpoints table
)
# Returns DataFrame: timestamp | {target_column}_forecast  (forecast_horizon rows)
print(results.head())
```


### Checkpoints

| File | Mode | Downloaded when |
|---|---|---|
| `run8_best_model_cr.pt` | Default (cross-channel on) | `use_cross_channel=True` (default) |
| `moment_head_512_6hr.pt` | Standard (no cross-channel) | `use_cross_channel=False` |
| `standardizer.pkl` | Both | Always |

Pass `use_cross_channel=False` to use the standard checkpoint:

```python
results = perform_forecasting(df=df, use_cross_channel=False, ...)
```

## DARR mode (context-enhanced forecasting)

Supply `context_df` to enable DARR: the SDK builds a kNN memory from historical windows and
blends direct predictions with retrieved neighbors (`alpha * direct + (1 - alpha) * kNN`).

```python
context_df = pd.read_csv("historical_data.csv")   # needs ≥ seq_len + model_horizon rows

results = perform_forecasting(
    df=df,
    context_df=context_df,      # enables DARR
    forecast_horizon=72,
    alpha=0.2,                  # 0.2 = 20% direct, 80% kNN (default: 0.01)
    k=64,                       # number of nearest neighbors
    temperature=0.05,           # kNN softmax temperature
)
```

Context and input datasets do not need identical columns — the SDK aligns to common features
and warns when columns differ. Both must share `timestamp_column` and `target_column`.

## Interpretability

Set `interpretability=True` to activate the Model-Agnostic Interpretability Framework. It produces localized, horizon-specific, time-aware explanations — including lag×horizon attribution, semantic flow, trajectory stability, diagnostic ratios, and (for multivariate inputs) channel-axis attribution and coupling analysis.

> For the full parameter reference, output bundle, and component descriptions, see
> **[`forecasting/README.md`](https://github.com/NVIDIA/NV-Tesseract/blob/main/forecasting/README.md)**.

## Fine-tuning

Fine-tune the forecasting head (encoder/embedder frozen by default) on your own time series.
`--ckpt-init auto` warm-starts from the published NV-Tesseract checkpoint; `--ckpt-init none`
trains a fresh head from the base backbone.

```bash
cd /path/to/NV-Tesseract/forecasting
# Without cross-channel (uses moment_head_512_6hr.pt)
uv run python examples/finetune_example.py \
  --csv /path/to/timeseries.csv \
  --timestamp-col timestamp \
  --target-cols target \
  --seq-len 512 --forecast-horizon 72 \
  --epochs 5 --batch-size 8 --lr 1e-4 \
  --output-dir artifacts/finetune_my_data

# With cross-channel layer (uses run8_best_model_cr.pt)
uv run python examples/finetune_example.py \
  --csv /path/to/timeseries.csv \
  --timestamp-col timestamp \
  --target-cols sensor_1,sensor_2,sensor_3 \
  --use-cross-channel --cross-channel-heads 8 \
  --epochs 5 \
  --output-dir artifacts/finetune_cross_channel
```

### Fine-tuning arguments

| Argument | Default | Description |
|---|---|---|
| `--csv` | **required*** | Single CSV split temporally into train/val |
| `--train-csv` | **required*** | Training CSV (mutually exclusive with `--csv`) |
| `--val-csv` | — | Validation CSV when `--train-csv` is used |
| `--timestamp-col` | `timestamp` | Datetime column to exclude from features |
| `--target-cols` | all numeric | Comma-separated columns to forecast |
| `--ckpt-init` | `auto` | `auto` = published NV-Tesseract weights; `none` = fresh head |
| `--seq-len` | `512` | Input context length |
| `--forecast-horizon` | `72` | Steps ahead to predict |
| `--val-ratio` | `0.1` | Validation fraction when `--csv` is used |
| `--epochs` | `5` | Training epochs |
| `--batch-size` | `8` | Batch size |
| `--lr` | `1e-4` | AdamW learning rate (OneCycleLR scheduler) |
| `--weight-decay` | `0.0` | AdamW weight decay |
| `--head-dropout` | `0.1` | Forecasting head dropout |
| `--max-norm` | `5.0` | Gradient norm clip |
| `--unfreeze-encoder` | false | Train the transformer encoder too |
| `--unfreeze-embedder` | false | Train the patch embedder too |
| `--use-cross-channel` | false | Add cross-channel attention layer |
| `--cross-channel-heads` | `8` | Attention heads in cross-channel layer |
| `--seed` | `13` | Random seed |
| `--output-dir` | `artifacts/finetune` | Output directory |

*One of `--csv` or `--train-csv` is required.

### Inference with fine-tuned checkpoint

```python
results = perform_forecasting(
    df=df,
    timestamp_column="timestamp",
    target_column="target",
    seq_len=512,
    forecast_horizon=72,
    model_horizon=72,
    standardizer_pkl="artifacts/finetune_my_data/standardizer.pkl",
    ckpt="artifacts/finetune_my_data/best_model.pt",
    use_cross_channel=False,   # set True if trained with --use-cross-channel
)
```

> **Fair comparison rule:** `perform_forecasting` defaults to `use_cross_channel=True`
> (loads `run8_best_model_cr.pt`). When comparing pretrained inference against a finetuned
> checkpoint, both must use the same base architecture. Either:
> - Finetune with `--use-cross-channel` (warm-starts from `run8_best_model_cr.pt`) and
>   run inference with `use_cross_channel=True` (default), or
> - Run inference with `use_cross_channel=False` and finetune without `--use-cross-channel`
>   (warm-starts from `moment_head_512_6hr.pt`, the finetune default).
>
> Mixing architectures — cross-channel pretrained vs standard finetuned — conflates model
> quality with architectural differences and makes the comparison uninterpretable.

## Data requirements

| Property | Requirement |
|---|---|
| Rows | ≥ `seq_len` (default **512**) for inference; validation split must also have ≥ `seq_len + forecast_horizon` rows |
| Columns | `timestamp` + one or more numeric columns; NULLs filled with zeros automatically |
| Timestamp | Parseable by pandas; no NULLs; uniform frequency inferred from mode of diffs |
| Target | Must be numeric; NULLs filled with zeros |
| `forecast_horizon` | Max **512** steps; beyond model's native 72 triggers autoregressive rollout |
| DARR context | ≥ `seq_len + model_horizon` rows; must share timestamp + target columns with input |

## Output structure

**Inference** (standard / DARR):
```
DataFrame: timestamp | {target_column}_forecast   (forecast_horizon rows)
```

**Fine-tuning** (`--output-dir artifacts/finetune_my_data`):
```
artifacts/finetune_my_data/
├── best_model.pt            # checkpoint with lowest validation loss
├── standardizer.pkl         # normalization statistics for this dataset
├── finetune_metadata.json   # model config, channels, best epoch, all args
└── metrics.json             # per-epoch train/val loss and MAE
```


## Hardware

| Tier | Setup | Notes |
|---|---|---|
| Minimum | 1× CPU | Functional; slow for long horizons |
| Recommended | 1× NVIDIA GPU (≥8 GB VRAM) | Strongly recommended for fine-tuning |
| Apple Silicon | MPS | Auto-detected; on par with CPU for this workload |
| Multi-GPU | Not supported | Single-device only |

## Known pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: backbone` | Editable install missing | Run `uv pip install -e .` from `forecasting/` |
| `HfHubHTTPError: 401` / `403` | Model license not accepted or gated fork | Accept license on HF repo page; or `huggingface-cli login` |
| `504` / timeout on first weight download | HF CDN throttles unauthenticated requests — public repos are still subject to this on first download | Set `export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"` before running; authenticated requests use a more reliable CDN path |
| `ValueError: DataFrame has X rows but seq_len requires Y` | Input too short | Provide ≥ `seq_len` (512) rows or reduce `--seq-len` |
| `ValueError: forecast_horizon must be <= 512` | Horizon too large | Split into multiple `perform_forecasting` calls |
| `ValueError: No common numeric columns` (DARR) | Context has no overlapping features | Ensure context shares ≥ 1 numeric column with input |
| `ValueError: Context DataFrame has X rows but requires Y` | Context too small | Context needs ≥ `seq_len + model_horizon` rows |
| `Interpretability PDF skipped: matplotlib not installed` | Missing optional dep | `uv add matplotlib` or use `interpretability_output="json"` |
| `ValueError: No training windows` (finetune) | Data too short for windows | Reduce `--seq-len` / `--forecast-horizon`, or increase dataset size |
| Stale environment errors mentioning `backbone` package | Old lock file | `uv cache clean && uv sync --group dev` |
