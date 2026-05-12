---
name: automl-deft-pipeline
description: Chain AutoML hyperparameter search and DEFT (Data-Efficient Fine-Tuning) into one pipeline — use AutoML to find the best training hyperparameters, then feed the resulting checkpoint into a DEFT loop as the SFT starting point. Trigger this skill whenever the user asks about combining AutoML with DEFT, running DEFT from an AutoML-tuned baseline, "AutoML then DEFT", "warm-start DEFT", "tune hyperparams before DEFT", or any workflow that involves finding good hyperparameters before kicking off iterative data augmentation. Also trigger when the user has an AutoML-trained checkpoint and asks how to use it for DEFT.
license: Apache-2.0
---

# AutoML + DEFT Pipeline

A workflow-bridge skill that wires two existing skills together:

- **`tao-automl`** — runs hyperparameter optimization on a model + dataset, produces a best checkpoint
- **`deft-cosmos-rl`** (or any DEFT application skill) — runs iterative data augmentation: gap analysis on a base model, generates synthetic training data, merges, retrains, repeats

This skill teaches the agent the *handoff* between the two: how to make the AutoML checkpoint serve as the SFT initialization for DEFT, plus the pitfalls and quality checks worth doing in between.

This skill does **not** re-implement AutoML or DEFT. The agent should invoke the existing skills via their normal entry points and only use this doc as the playbook for the connective tissue.

## When this skill applies

- User wants to do AutoML and then DEFT on the same model/dataset
- User has an AutoML-tuned checkpoint and asks "how do I use this for DEFT now"
- User wants a stronger SFT init than vanilla pre-trained weights before starting DEFT iterations
- User mentions "warm-starting DEFT", "tuned baseline for DEFT", or describes a two-phase pipeline matching the above

## When this skill does NOT apply

- User wants only AutoML (with no follow-on DEFT) → use `tao-automl` directly
- User wants only DEFT from a fixed pre-trained base model → use the appropriate DEFT application skill directly
- User is doing zero-shot eval, RAG, or non-training workflows

---

## The mental model

The pipeline is two phases that the agent runs in sequence:

```
Phase 1 (AutoML)                            Phase 2 (DEFT)
─────────────────                           ──────────────
train_dataset                               train_dataset
     │                                           │
     ▼                                           ▼
[ AutoML HPO sweep ]                        [ Gap analysis ]
     │  many recs                                │
     │  picks best by val_loss                   ▼
     ▼                                      [ Synth gen ]
best checkpoint URI ───────────────────►    [ Data merge ]
(s3://.../safetensors/epoch_N)                   │
                                                 ▼
                                            [ SFT retrain ]
                                                 │
                                                 ▼
                                            (loop N iterations)
```

The handoff is a **single URI**: the best safetensors directory produced by AutoML's winning run. DEFT's SFT stage starts from that URI instead of the vanilla pre-trained weights.

## Why bother chaining them

Without AutoML, DEFT starts from defaults. Default LR / weight decay / decay schedule are rarely optimal for the user's specific dataset, so the SFT baseline DEFT iterates from is weaker than it could be. AutoML produces a model that has already adapted to the dataset's geometry — DEFT then spends its iteration budget on the harder problem (filling distributional gaps), not on finding a workable optimizer config.

The trade-off is compute: AutoML is N training jobs, then DEFT is M more. Tell the user up front so they aren't surprised by the cost.

---

## Phase 1: run AutoML

Invoke the `tao-automl` skill (read its SKILL.md for the full interface). At minimum the agent needs to know:

| Input | Notes |
|---|---|
| `network_arch` | Same model the DEFT skill expects (e.g. `cosmos-rl`, `clip`, etc.) |
| `train_dataset_uri` | Same dataset the DEFT loop will use |
| `eval_dataset_uri` | Held-out set for AutoML's val_loss; ideally NOT the same set DEFT will use for gap analysis (see **Cleanliness** below) |
| `metric` | Default is `val_loss`. See **Metric pitfalls** for when to override |
| `algorithm` | Bayesian is a fine default. LLM-brain or autoresearch can be more sample-efficient if compute is tight |
| `automl_max_recommendations` | 5–20 is typical. More recs = better picks but linear in compute |
| `spec_overrides` | Pin the things you care about (epochs, FSDP shape, batch size). AutoML should only sweep the optimizer hyperparams unless you say otherwise |

After the sweep finishes, the agent reads `result["best"]` from the runner's return value:

```python
best = result["best"]
best_specs    = best["specs"]            # dict of winning hyperparams
best_metric   = best["metric_value"]     # the val_loss (or whatever metric)
best_train_id = best["job_id"]           # the training job that produced the winning model
```

The winning checkpoint lives at:

```
{output_root}/{best_train_id}/{output_subdir}/
```

Where `output_root` and `output_subdir` come from how the AutoMLRunner is configured (commonly `s3://<bucket>/results/<job_id>/train_output_dir/<timestamp>/safetensors/epoch_<N>`). The exact layout is network-specific — read it from the AutoMLRunner workspace state file or from the training job's container logs (look for a "saved safetensors to ..." line).

### Quality check before handing off

Before passing the checkpoint to DEFT, sanity-check that it's actually a usable model. **A low val_loss is not sufficient** — see Metric pitfalls below. The minimum check is to run a quick eval against a held-out set and look at the prediction distribution:

- Count predictions per class. If it predicted one class for ~all examples, the model has mode-collapsed and is useless for DEFT regardless of val_loss.
- Spot-check a few predictions vs ground truth.
- Compare accuracy to a zero-shot baseline of the same pre-trained model. If AutoML did not improve over zero-shot, something is wrong — surface that to the user before continuing.

If the best checkpoint is bad, evaluate the 2nd or 3rd best. AutoML's pick is a guess based on val_loss; the actual best deployable model among the runs may not be the val_loss winner.

---

## Phase 2: run DEFT with the AutoML checkpoint

Invoke the relevant DEFT application skill (e.g. `deft-cosmos-rl` for video-QA, or whatever DEFT variant fits the model). The DEFT skill's SKILL.md will document its own arguments — the agent should read it and follow it.

The bridge: instead of letting DEFT run its default `sft_training` stage from the vanilla pre-trained weights, point its SFT init at the AutoML checkpoint. Two patterns are common:

**Pattern A — DEFT skill exposes a `base_model` / `sft_init` arg**
Pass the AutoML checkpoint URI directly. DEFT will skip its own SFT and start the loop with that model.

**Pattern B — DEFT runs its own SFT stage**
Set the SFT stage's `base_model_path` (or equivalent) to the AutoML checkpoint URI. DEFT's SFT stage will resume from there. Use the **AutoML's winning hyperparameters** as the SFT stage's hyperparameters too — copy them out of `result["best"]["specs"]`. Otherwise DEFT will overwrite the AutoML signal with whatever defaults its SFT stage uses.

If neither pattern is supported by the DEFT skill cleanly, fall back to **Pattern C**: skip DEFT's SFT stage entirely (e.g. by not providing `train_dataset_uri` if the DEFT skill conditions SFT on that), and let the gap-analysis / synthetic-gen / retrain loop start from the AutoML checkpoint provided as the eval base.

After DEFT finishes its iterations, the agent reports:

- The accuracy at each phase: zero-shot → AutoML-best → after each DEFT iteration
- Which iteration gave the best accuracy (DEFT does not always monotonically improve)
- The total compute spent

---

## Pitfalls and quality checks

These are the lessons learned from running this pipeline. Bake them into the agent's behavior, don't just paste them once.

### Metric pitfalls

`val_loss` (cross-entropy) is the default AutoML metric and it has a well-known failure mode on **imbalanced classification tasks** (binary or k-way with skewed classes): the model can minimize CE by confidently predicting the majority class for everything, achieving very low val_loss while having near-zero recall on the minority class. The val_loss winner of an AutoML sweep can be a mode-collapsed model.

If the dataset is imbalanced (any class with <30% prevalence), prefer one of:

- Balanced accuracy or macro-F1 as the AutoML metric on a held-out set
- Run val_loss as usual but **eval all top-K configs** by accuracy/F1 before picking
- Add `pred_counts` sanity checks in the AutoML loop and discard any rec whose predictions collapse to one class

For regression and well-balanced classification, val_loss is fine.

### Run-to-run noise

AutoML can show **2–3× variance** in val_loss for the *same* hyperparameter config across runs (different seeds, dataloader shuffles, FSDP gradient accumulation order). If a single rec's val_loss looks too good to be true, it might be — replicate it before celebrating. A good practice: when the AutoML winner is suspiciously better than the runner-up, re-run the winner with a fresh seed and confirm the metric holds.

### Cleanliness (data leakage)

If AutoML's `eval_dataset_uri` is the same set DEFT will later use for gap analysis or final reporting, hyperparameter selection has *already touched* that data. AutoML's "winner" is biased upward on that set. To stay honest:

- Hold out a small split (e.g. 5–10%) from the training data as the AutoML validation set, distinct from any DEFT eval set
- Or accept that the final reported number on the shared eval set is optimistically biased, and flag this when reporting to the user

For a quick research compare, sharing eval sets is often acceptable — but the agent should **always** name the cleanliness boundary explicitly so the user understands what number they're getting.

### Compute budget

The pipeline is sequential and most of the cost is GPU training. Total wall clock is roughly:

- **AutoML**: `recs × train_time_per_rec` (training dominates; if optimizer-state upload to remote storage is slow on the user's backend, that adds per-rec overhead too)
- **DEFT**: `iterations × (gap_analysis + synth_gen + merge + sft_time)` — SFT is usually the largest term

Wall-clock per stage depends entirely on the user's compute (GPU count, model size, dataset size, network/storage speed), so don't quote specific minute numbers unless you've actually measured them on this user's setup. Instead, tell the user the **structure** of the cost (number of training jobs × per-job time) and ask them for a per-job time estimate if they have one. That gives an honest bound without making up numbers.

---

## Quick Start

This is what the agent might say to the user when starting fresh:

> I'll run the pipeline in two phases:
>
> **Phase 1 — AutoML:** I'll sweep `<N>` configs over `<hyperparam list>` against `<eval set>` using `<algorithm>` algorithm with `val_loss` as the metric. After it finishes I'll evaluate the top `<K>` configs by accuracy on a held-out set and pick the actual best — not just the val_loss winner — to guard against mode collapse.
>
> **Phase 2 — DEFT:** I'll start the DEFT loop with the Phase-1 checkpoint as the SFT init. The loop will run `<M>` iterations: gap analysis → synthetic data generation → merge → retrain. I'll report accuracy after each iteration so we can see which one peaks.
>
> Total cost is `<N>` AutoML training jobs + `<M>` DEFT SFT jobs (plus per-iteration synth-gen). If you can tell me roughly how long one training run takes on your setup I can give you a wall-clock estimate. OK to proceed?

After confirmation, the agent invokes `tao-automl`, evaluates the top configs, then invokes the DEFT skill with the chosen checkpoint, then summarizes the trajectory zero-shot → AutoML → DEFT[1] → DEFT[2] → … so the user can see where the gains came from.

---

## See also

- `tao-automl` skill — full AutoML interface, algorithm selection, hyperparameter ranges
- `deft-cosmos-rl` skill — full DEFT pipeline for Cosmos-RL video QA
- Other DEFT application skills as they appear under `applications/deft-*` — same handoff pattern applies
