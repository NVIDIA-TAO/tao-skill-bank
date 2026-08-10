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
