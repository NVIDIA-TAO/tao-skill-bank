# FAR-optimizing AutoML tooling for visual-changenet AOI

Tooling for running FAR@100%recall-minimizing AutoML on SLURM
(`automl_vcn_slurm_v2.py`) with exact-KPI evaluation (`far_eval*.py`).

## Required environment

Site-specific values are read from environment variables (keep them in a
gitignored `*.local.env` you source alongside your credentials env):

| Var | Meaning |
|---|---|
| `LUSTRE_AOI_ROOT` | Lustre root holding the AOI dataset mirror, eval outputs, and patches |
| `TAO_SKILL_BANK_PATH` | Absolute path of this repo checkout |
| `AOI_WORKSPACE` | Local workspace root (AutoML state, logs) |
| `SLURM_ACCOUNT` / `SLURM_PARTITION` | Cluster scheduling values |
| `SLURM_USER` / `SLURM_HOSTNAME` / `SSH_KEY_PATH` | SlurmSDK credentials |
| `base_url` / `model` / `AUTOML_LLM_API_KEY` | LLM endpoint for range narrowing |

## Prerequisite: val_far training overlay

The runner's checkpointer config (`monitor: val_far`, `mode: min`) selects the
best-FAR epoch **in-training**. The `val_far` metric is added to
visual_changenet by NVIDIA-TAO/tao-pytorch PR #92; until that ships in the TAO
container you must stage the patched module as a PYTHONPATH overlay:

```
${LUSTRE_AOI_ROOT}/patches/valfar/nvidia_tao_pytorch/cv/visual_changenet/classification/models/cn_pl_model.py
```

The runner injects `PYTHONPATH=${LUSTRE_AOI_ROOT}/patches/valfar` into every
training job. **Without the overlay (or a container containing PR #92), any
spec monitoring `val_far` fails at the first checkpoint attempt** — either
stage the overlay or strip the `checkpointer` block from `BASE_SPEC`.

## Companion fixes

The search direction and LLM-range-narrowing behavior of the brains require
NVIDIA-TAO/tao-automl PR #18 (metric-direction fix; without it,
minimize-direction metrics such as FAR are silently **maximized** by
Bayesian/BFBO, and `enable_llm_range_narrowing` is a no-op outside hybrid).

## Hard-won invariants encoded in the runner

- **Warm-start searches must pin architecture params** (use
  `DEFT_ITER_PARAMS_WARMSTART` / `DEFT_ITER_RANGES_WARMSTART`): searching
  `learnable_difference_modules` while initializing from a checkpoint fails at
  state_dict load (decoder width mismatch).
- **Fine-tune lr is regime-dependent**: HPs tuned for 100-epoch from-scratch
  training destroy a converged model when reused for short warm-start
  fine-tunes (measured cliff ~1e-4, dropping as the model improves). Search
  the fine-tune lr per iteration; do not transfer the from-scratch value.
- **`custom_param_ranges` keys are `valid_min`/`valid_max`/`valid_options`**;
  `min`/`max`/`values` are silently ignored by the brains.
- **Fixed-spec training via pinning** uses epsilon-width ranges
  (`launch_fixed_train`): degenerate min==max pins are rewritten ×1.1/×0.9 by
  the value generator's boundary clamp.

## Reproducing the guarded post-DEFT campaign

Build deterministic, duplicate-free 50% and 100% DEFT mixtures first:

```bash
python prepare_post_deft_data.py \
  --source /path/to/train_combined_final.csv \
  --output-dir /path/to/post_deft_data \
  --ratios 0.50,1.00 \
  --seed 20260814
```

Stage those CSVs and the DEFT incumbent checkpoint where SLURM workers can
read them. Then inspect the exact four-arm campaign without launching:

```bash
python aoi_post_deft_campaign.py \
  --campaign-dir /path/to/local/campaign \
  --mix50-csv /remote/data/train_unique_mix_050.csv \
  --mix100-csv /remote/data/train_unique_mix_100.csv \
  --images-dir /remote/aoi/workspace \
  --incumbent-checkpoint /remote/checkpoints/deft_winner.pth \
  --incumbent-far 16.096636665087637 \
  --dry-run
```

Remove `--dry-run` only after the normal TAO launch preflight and review. The
launcher writes `campaign_manifest.json` before submission, runs warm-full,
warm-head, scratch-mix50, and scratch-mix100 concurrently, and preserves each
branch summary independently for recovery.

The search objective is global validation FAR. `far_eval_calibrated.py`
calibrates the 100%-recall threshold on validation and applies it unchanged to
the KPI inference output. `max_regression=0` keeps the incumbent when a final
challenger metric is missing, non-finite, or worse. A bad fine-tune can still
occur; it cannot replace the incumbent through this guarded path.

The study used eight GPUs per trial with per-GPU batch size 2, giving the same
effective batch size of 16 as the prior one-GPU runs. The default campaign
contains 8 recommendations for each warm-start arm and 16 for each scratch
arm. Every path, incumbent FAR, seed, and bound is materialized in the written
manifest so a run can be audited or reproduced on another machine.
