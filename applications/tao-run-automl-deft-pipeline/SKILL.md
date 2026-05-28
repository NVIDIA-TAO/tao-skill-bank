---
name: tao-run-automl-deft-pipeline
description: >
  Run the canonical NVIDIA AOI three-phase training pipeline — Phase 1 AutoML baseline (HPO),
  Phase 2 DEFT loop (RCA → SDG → mining → plain-train retrain), Phase 3 AutoML refinement on
  the DEFT-augmented dataset. This is the default entry point for any "run the AOI workflow",
  "fine-tune my PCB AOI model end-to-end", "improve my AOI ChangeNet model", or "AOI workflow
  with AutoML" request — route here instead of tao-run-deft-aoi directly unless the user
  explicitly asks for the DEFT loop ONLY (e.g. "run JUST the DEFT loop", "skip AutoML, only
  DEFT"). Also handles the same three-phase pattern for non-AOI DEFT applications — AutoML
  baseline then DEFT loop warm-started from AutoML's winning HPs then post-DEFT AutoML
  refinement on the iteration-augmented dataset. Trigger phrases include "run the AOI
  workflow", "AOI end-to-end", "AutoML + DEFT", "AutoML then DEFT", "tune hyperparameters then
  DEFT", "DEFT with AutoML at both ends", "warm-start DEFT", "improve my AOI model".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit. Sub-skills (tao-run-automl, tao-run-deft-aoi) declare additional requirements.
metadata:
  author: NVIDIA Corporation
  version: "0.4"
allowed-tools: Read Bash Write Skill
---

# AutoML + DEFT Pipeline

A workflow-bridge skill that runs **three phases** in sequence by delegating to two existing skills — `tao-run-automl` for HPO and a DEFT application skill (default `tao-run-deft-aoi` for AOI; other `applications/deft-*` skills for non-AOI cases) for the iterative data-improvement loop.

This skill **does not** re-implement AutoML or DEFT. It owns only the connective tissue: HPO spec inputs, the spec-handoff between AutoML and DEFT, and the post-DEFT AutoML re-run on the augmented dataset.

## When this skill applies

- User asks to "run the AOI workflow" or "improve my AOI ChangeNet model" — **default to this skill**, not `tao-run-deft-aoi` directly. The bare DEFT loop is the inner stage of this pipeline.
- User wants AutoML and DEFT chained on the same model/dataset
- User says "AutoML at both ends", "tune HPs then DEFT", "warm-start DEFT", "AutoML before and after DEFT"
- User has an AutoML-tuned spec and asks how to feed it into DEFT

## When this skill does NOT apply

- User explicitly asks for the DEFT loop only ("run JUST the DEFT loop", "skip AutoML") → use `tao-run-deft-aoi` directly
- User wants only AutoML with no follow-on DEFT → use `tao-run-automl` directly
- User is doing zero-shot eval, RAG, or non-training workflows

---

## The mental model

```
Phase 1 (AutoML baseline)        Phase 2 (DEFT loop, plain train)        Phase 3 (AutoML refinement)
─────────────────────────        ────────────────────────────────        ───────────────────────────
specs/baseline_spec.yaml         (Phase 1 winner pre-seeds baseline      ${RESULTS_DIR}/iter${N}/dataset/
train/base/training_set.csv       — DEFT skips its baseline train)       train_combined_iter${N}.csv
        │                                       │                                       │
        ▼                                       ▼                                       ▼
[ AutoML HPO sweep ]               [ DEFT: baseline-inference → RCA       [ AutoML HPO sweep ]
   N recommendations                 → iter 1..N (plain retrain) ]        re-tunes HPs against the
   pick best by val_loss / FAR      RCA / route / SDG / mining             DEFT-augmented dataset
        │                                       │                                       │
        ▼                                       ▼                                       ▼
best HPs spec + ckpt ─────►      DEFT-augmented CSV ───────────►        final best checkpoint
                                 + iter winner checkpoint               (the deliverable; no
                                 (Phase 3 warm-starts from it)           further retrain)
```

The handoffs are:

- **Phase 1 → Phase 2**: a *spec file* AND the *winning checkpoint* (Phase 1 already trained a model at those HPs — retraining the same HPs in DEFT's baseline step is wasted compute). The bridge:
  1. Deep-merges Phase 1's winning HPs onto `<workspace>/specs/baseline_spec.yaml` → writes `specs/baseline_spec_automl.yaml` (DEFT reads this).
  2. Copies the Phase 1 winning checkpoint into `${RESULTS_DIR}/baseline/train/` with the filename DEFT expects.
  3. Pre-populates `${RESULTS_DIR}/deft_state.json` and `${RESULTS_DIR}/loop_log.jsonl` so DEFT sees baseline train as already completed and resumes at baseline inference → evaluate → RCA → iter 1.

  DEFT itself stays plain-train (`automl_policy: off` inside the DEFT loop is preserved).

- **Phase 2 → Phase 3**: a *training CSV* AND the *iter winner's checkpoint*. The CSV (`train_combined_iter${N_final}.csv`) is fed to AutoML as the training data; the checkpoint (`iterations.<best>.best_ckpt_path` from `deft_state.json`) is wired into each rec's `train.pretrained_model_path` so Phase 3 **fine-tunes from Phase 2's winner** rather than training from scratch. Without this warm-start Phase 3 routinely regresses vs the iter winner — small epoch budgets aren't enough to reconverge a from-scratch model on the augmented dataset, and AutoML ends up tuning a worse base. Phase 3's winning checkpoint is the pipeline's deliverable — no separate retrain step after Phase 3.

## Why three phases instead of two

- **Phase 1 alone** finds good HPs on the *original* training distribution, but the model still has the distributional gaps DEFT is designed to fill.
- **Phase 2 alone** (just DEFT) fills the gaps but uses whatever HPs `specs/baseline_spec.yaml` was hand-authored with — usually not optimal.
- **Phase 3 alone** would run AutoML against the augmented dataset, but without a tuned baseline the DEFT loop's iteration cost is higher (slower convergence, more iterations to hit the KPI).

Running all three: AutoML cheap-tunes once on the original data, DEFT does the heavy data work with reasonable HPs, then AutoML tunes again on the now-richer dataset. Phase 3 is the most important of the three for the final deployed FAR/recall.

## Cost up-front

The pipeline is sequential. Total wall-clock ≈ Phase 1 (N_automl × per-rec train) + Phase 2 (M iterations × per-iter cost) + Phase 3 (N_automl × per-rec train).

Note that **Phase 2 has no separate baseline train** — Phase 1's winning checkpoint is reused as DEFT's baseline, so the baseline cost lands inside Phase 1's N_automl trainings rather than as an extra retrain. Surface this to the user before kickoff. Typically Phase 2's iterations still dominate (each includes SDG + retrain), but Phase 1 and Phase 3 each add several hours on a single-GPU box. Use the per-job estimate from the user's setup (if they have one) rather than guessing minutes.

---

## Consolidated Pre-Flight — one gate, all three phases

**The pipeline has exactly one user gate.** Before any side-effecting action (docker pull, docker login, any job-launch call delegated to a downstream skill, file mutations under `${RESULTS_DIR}/`), the agent must produce a single consolidated Pre-Flight Summary that subsumes every downstream skill's preflight. Once the user approves, the run is autonomous through all three phases — no further interactive pauses.

The user explicitly does not want to be paged between phases. The DEFT loop's own inline `## Pre-Flight Summary` gate becomes a **zero-question display step** (every value pre-supplied from this consolidated gate) rather than a fresh interrogation. Same for `tao-run-automl`'s shared launch preflight in Phase 1 and Phase 3.

### How to build the consolidated summary

Before printing anything to the user, **open and read every downstream skill's preflight section in full**:

- `applications/tao-run-automl/SKILL.md` → `## Preflight` (Phases 1 and 3). Specifically: shared launch preflight (platform credentials, dataset visibility, model credentials, container image confirmation, compute shape), required inputs (`platform`, `image`, `network_arch`, `train_dataset_uri`, `eval_dataset_uri`, `metric`, `algorithm`, `automl_max_recommendations`), and the runner-freshness rule.
- The DEFT skill invoked in Phase 2 (AOI default: `applications/tao-run-deft-aoi/SKILL.md` → `## Pre-Flight` + `### Pre-Flight Summary`; for non-AOI runs, the corresponding `applications/deft-*` SKILL.md). Specifically: workspace/specs/CSV resolution, `.env` sourcing, NGC + HF token presence, `docker login nvcr.io`, container image resolution from `versions.yaml`, local image inspect, GPU memory rule of thumb (AOI ChangeNet: `batch_size ≤ 16` on 48 GB GPUs, `≤ 8` on 24 GB GPUs), pre-gen ingestion source verification + basename pairing, leakage check, and the loop's defaults (`max_iterations=3`, `top_k_per_target=5`, `min_similarity=0.9`).
- The `tao-launch-workflow` shared intake (referenced by `tao-run-automl`) — platform-specific credentials and compute-shape questions.

Then run **every read-only check** those preflight sections prescribe — image resolution, `docker image inspect`, file existence, basename pairing, row counts, value-count distributions, leakage diff, GPU memory query, host Python dependency check. The user should see the *outcome* of each check in the summary, not be asked to run it themselves.

#### Required: run every step of the DEFT skill's `## Pre-Flight`

Run **every check in `applications/tao-run-deft-aoi/SKILL.md` `## Pre-Flight`** (or, for non-AOI runs, the corresponding `applications/deft-*` SKILL.md `## Pre-Flight`) as part of the consolidated pre-flight, before printing the summary. If any step is skipped, the consolidated gate is invalid and the pipeline must not advance.

### Mandatory contents of the consolidated summary

The summary must include, in this order:

1. **Workspace, host, platform, network** — workspace root, GPU model + memory, docker version, platform choice (never default; if user hasn't said, ask in the consolidated gate, not later), `network_arch`.
2. **Credentials status** — `[ -n "$VAR" ]` SET/UNSET for each variable each downstream skill requires. Never print the value.
3. **Container images** — fully resolved URIs from `versions.yaml` (per the DEFT skill's `scripts/resolve_versions_key.py` pattern), with a PRESENT/MISSING column from `docker image inspect`. Missing images are not blockers — the post-approval autonomous run will `docker login nvcr.io` and pull them — but the user must see what will be pulled.
4. **Dataset table** — train/val/test/mining-pool/pre-gen counts; KPI label distribution; train↔val leakage check (must show `0 overlapping rows`).
5. **Phase 1 config** — algorithm, sweep size, metric, HPs to sweep, HPs pinned, results dir, spec source.
6. **Phase 2 config** — every field from the DEFT skill's `## DEFT Loop — Pre-Flight Summary` table (KPI target, max_iterations, training_epochs, top-K, mining cutoff, GPUs, resuming flag) **plus** the pre-seeded baseline source (`${RESULTS_DIR}/baseline/train/` populated from Phase 1's winning checkpoint). Mark the DEFT skill's inline gate as "auto-approved by consolidated gate above".
7. **Phase 3 config** — sweep size, metric, warm-start checkpoint policy, val set (must match Phase 1).
8. **Compute estimate** — Phase 1 train count × per-rec time + Phase 2 iteration count × per-iter time + Phase 3 train count × per-rec time. If per-job time is unknown, ask the user once in this same gate or offer a 1-epoch dry-run option.
9. **Confirmation line** — "Approve all three phases? After 'go' I will not pause again until DEFT's iter-level KPI gate (if reached) or pipeline completion."

### Suppressing downstream interactive gates

When invoking each downstream skill after the consolidated gate, pass through the values collected in the summary so the downstream skill has nothing to ask:

- `tao-run-automl` (Phases 1 + 3): supply `platform`, `image`, `network_arch`, dataset URIs, `metric`, `algorithm`, `automl_max_recommendations`, `spec_overrides`, and (Phase 3 only) the warm-start `pretrained_model_path`. The shared launch preflight then runs as a non-interactive validation pass.
- DEFT loop (Phase 2): write `deft_state.json` with the Phase 1 baseline pre-seed (per the Phase 1 → Phase 2 handoff below) **and** pre-populate the DEFT skill's config inputs (`max_iterations`, `top_k_per_target`, `min_similarity`, `training_epochs`, KPI threshold). The DEFT loop's inline summary still prints as an audit-trail display; it must not re-prompt.

The only places the pipeline is *allowed* to pause for user input after the consolidated gate are:

- Mid-run hard-stop gates the downstream skill cannot bypass on safety grounds (e.g. DEFT's KPI regression gate, an unrecoverable preflight failure surfaced after `docker pull`). These are exceptional, not routine. Call them out in the consolidated summary so the user knows when, if ever, they'll be paged.

### When the skill bank version doesn't yet support gate suppression

Older DEFT skill versions that hard-code "STOP — wait for explicit user approval" cannot be silenced by pre-supplied inputs alone. In that case, the agent must still produce the consolidated summary up front and tell the user: "the DEFT skill will re-print its preflight as a display before iter 1 — type 'go' both times, the second one is a known limitation of skill version X." Then file an issue / open a PR against the DEFT skill to make the gate honour pre-supplied approval.

---

## Phase 1 — AutoML baseline

Invoke `tao-skill-bank:tao-run-automl` with:

| Input | AOI default | Notes |
|---|---|---|
| `network_arch` | `visual-changenet` | Same model the DEFT loop expects |
| `train_dataset_uri` | `<workspace>/train/base/training_set.csv` | Same training set DEFT will start from |
| `eval_dataset_uri` | `<workspace>/train/base/validation_set.csv` | Held-out — must NOT be the KPI test set (`<workspace>/kpi/testing_set.csv`), since that set is reserved for DEFT's final reporting |
| `metric` | FAR @ 100% recall (preferred) or `val_loss` | See **Metric pitfalls** below — ChangeNet AOI is class-imbalanced, val_loss alone can mode-collapse |
| `algorithm` | `bayesian` | LLM-brain or `autoresearch` if compute is tight |
| `automl_max_recommendations` | 5–10 for AOI | More recs = better HPs but linear in compute |
| `spec_overrides` | Pin epochs / batch_size; sweep optimizer-related HPs only | Otherwise AutoML wanders into long-train regimes that blow Phase 2's budget |

After the sweep finishes, AutoML's `result["best"]["specs"]` is the winning hyperparameter dict.

### Handoff to Phase 2

Phase 1 hands over **two artifacts**: the winning *spec* and the winning *checkpoint*. Retraining the same HPs in DEFT's baseline step is wasted compute — instead, pre-seed DEFT's baseline state from Phase 1's outputs so DEFT starts at baseline inference → evaluate → RCA → iter 1.

**Step 1 — Write the merged spec.** Deep-merge `result["best"]["specs"]` onto `<workspace>/specs/baseline_spec.yaml` (preserve dataset paths, model architecture, lighting layout; overwrite only the HPs AutoML tuned) and write to `<workspace>/specs/baseline_spec_automl.yaml`. Copy this onto the path DEFT reads:

```bash
cp <workspace>/specs/baseline_spec_automl.yaml <workspace>/specs/baseline_spec.yaml
```

**Step 2 — Pre-seed DEFT's baseline.** Locate the winning AutoML rec's best checkpoint (the AutoMLRunner writes `result["best"]["best_checkpoint_path"]` — pass through `eval_fn` for FAR-@-100%-recall metric capture). Pick the DEFT run-id (timestamped subdir under `<workspace>/results/`) and create `${RESULTS_DIR}/baseline/train/`. Copy the AutoML checkpoint into that directory using the filename convention DEFT expects (`model_epoch_<EEE>_step_<SSS>.pth`).

**Step 3 — Initialise `deft_state.json` with baseline already done.** Use `tao-run-deft-aoi/scripts/init_deft_state.py` to write the initial state, then patch in the `iterations.baseline` entry:

```python
import json, pathlib, shutil

state_path = pathlib.Path(f"{RESULTS_DIR}/deft_state.json")
state = json.loads(state_path.read_text())
state["iterations"]["baseline"] = {
    "stage_completed": "train",                      # so DEFT's resume picks up at inference
    "best_ckpt_path": str(baseline_ckpt_path),       # absolute host path
    "train_metric": phase1_winning_metric,            # FAR @ 100% recall captured by Phase 1's eval_fn
    "source": "automl_phase1",                        # provenance flag — not a DEFT-generated checkpoint
}
state_path.write_text(json.dumps(state, indent=2))
```

Append a matching `baseline.train` entry to `loop_log.jsonl` via `scripts/log_stage.py` with `--status ok --summary "baseline train skipped — reused Phase 1 AutoML winning checkpoint"`.

**Step 4 — Invoke DEFT.** When the DEFT loop reads its state on startup it will see `iterations.baseline.stage_completed == "train"` and skip directly to baseline inference → evaluate → RCA → iter 1. `automl_policy: off` inside the loop is preserved.

> **DEFT honors this handoff.** `tao-run-deft-aoi` checks `iterations.baseline.stage_completed == "train"` on startup (Workflow step 2 / Pipeline baseline block in its `SKILL.md`) and resumes at baseline inference against the pre-seeded checkpoint — no retrain.

### Quality check before handing off

Run a quick eval of the winning checkpoint against the held-out set:

- Per-class prediction counts — if it collapsed to one class, the winning HPs are useless for Phase 2. Evaluate the 2nd or 3rd best instead.
- Compare to a zero-shot ChangeNet baseline. If AutoML did not improve over zero-shot, surface that to the user and pause before continuing.

---

## Phase 2 — DEFT loop (plain training, baseline pre-seeded from Phase 1)

Invoke `tao-skill-bank:tao-run-deft-aoi` (read its `SKILL.md` for the full interface). For non-AOI applications, invoke the matching DEFT skill; the handoff shape is the same.

**The DEFT loop's baseline-train sub-step is skipped.** Phase 1 already produced a checkpoint trained at the winning HPs, and Phase 1's handoff (see above) pre-populated `${RESULTS_DIR}/baseline/train/` and `${RESULTS_DIR}/deft_state.json` so DEFT resumes at baseline inference → evaluate → RCA → iter 1. The rest of the DEFT loop runs unchanged. **Do not modify its `automl_policy: off` invariant.**

The DEFT loop owns:

- The Pre-Flight Summary display step — this is **not** a fresh user gate. The pipeline-level Consolidated Pre-Flight (above) is the single gate. The DEFT skill's summary still prints as an audit-trail display showing the pre-seeded `baseline/train/` source; it must not re-prompt for approval since every input was already collected in the consolidated gate
- Baseline inference → evaluate → RCA on the pre-seeded checkpoint
- The full per-iteration RCA → routing → SDG → mining → assemble → train cycle
- KPI gating and stop conditions
- `${RESULTS_DIR}/` layout, `deft_state.json`, `loop_log.jsonl`, `DEFT_Loop_Report.html`

After the loop exits (KPI met or `max_iterations` reached), capture two values from `deft_state.json`:

- `iterations.<best>.best_ckpt_path` — the loop's best plain-train checkpoint
- The final iteration label `N_final` — used to locate the augmented training CSV

If the DEFT loop hard-stops on an unrecoverable gate, **skip Phase 3**. There is no validated augmented CSV to feed AutoML.

---

## Phase 3 — AutoML refinement on the DEFT-augmented dataset

Re-invoke `tao-skill-bank:tao-run-automl` with the augmented training CSV as the train dataset, the same held-out validation CSV as before, and **Phase 2's iter winner checkpoint as the warm-start**:

| Input | AOI value |
|---|---|
| `network_arch` | `visual-changenet` |
| `train_dataset_uri` | `${RESULTS_DIR}/iter${N_final}/dataset/train_combined_iter${N_final}.csv` |
| `eval_dataset_uri` | Same as Phase 1 (`<workspace>/train/base/validation_set.csv`) — keep the comparison apples-to-apples |
| `metric` | Same metric as Phase 1 |
| `algorithm` | Same as Phase 1 |
| `automl_max_recommendations` | 5–10 |
| Initial spec | Start from `<workspace>/specs/baseline_spec_automl.yaml` (Phase 1's winner) — gives the sweep a strong centroid to refine around |
| **Warm-start checkpoint** | **`iterations.<best>.best_ckpt_path` from `${RESULTS_DIR}/deft_state.json`** — set `spec_overrides["train"]["pretrained_model_path"]` to this path. Each Phase 3 rec then **fine-tunes from Phase 2's winner** instead of training from scratch. |

### Why the warm-start is mandatory

Phase 3 receives a small augmented dataset (often a few hundred rows) and a tight epoch budget per rec (typically the same `num_epochs` Phase 1 used). With **no warm-start**, every rec starts from random init and only has 10-20 epochs to reconverge — not enough to outperform the iter winner which already trained for ~baseline + N×iter epochs. Result: Phase 3's `val_loss` regresses by 0.03-0.05 vs iter1, and the `_pick_best` safety net silently rolls back to the iter winner, wasting Phase 3's entire compute.

With warm-start, each rec is doing **targeted HP refinement on a converged model** instead of "train from scratch with slightly different LR". Empirically, this is the difference between Phase 3 routinely regressing and Phase 3 routinely improving.

Tradeoff: warm-starting from `iterations.<best>.best_ckpt_path` means Phase 3 is exploring a narrower region around the iter winner's weights, so it won't discover radically different optima — but for HP *refinement* on a small augmented set, that's the right inductive bias. If you want broad exploration instead, run a separate `tao-run-automl` sweep with no warm-start; don't conflate the two.

### Concrete `spec_overrides` pattern

```python
import json
state = json.loads((RESULTS_DIR / "deft_state.json").read_text())
# _pick_best preferred: lowest far_pct among iterations
best_iter, best_entry = min(
    (k, v) for k, v in state["iterations"].items() if v.get("far_pct") is not None
    and k not in ("final_automl",)                  # don't warm-start from a prior Phase 3
), key=lambda kv: kv[1]["far_pct"])
warmstart_ckpt = best_entry["best_ckpt_path"]
spec_overrides["train"]["pretrained_model_path"] = warmstart_ckpt
```

Output goes to `${RESULTS_DIR}/final_automl/`. The winning checkpoint of this sweep is the pipeline's deliverable.

### Wiring Phase 3's output back into the DEFT report

`tao-run-deft-aoi`'s `scripts/prepare_inference_spec.py` selects the lowest-`far_pct` entry from `deft_state.json["iterations"]`. To make Phase 3's checkpoint visible to the handoff:

1. Append an entry to `${RESULTS_DIR}/deft_state.json` under `iterations.final_automl` with the same shape as iteration entries (`best_ckpt_path`, `threshold`, `far_pct`) — populate from Phase 3's eval output.
2. Re-run `python ${TAO_SKILL_BANK_PATH}/applications/tao-run-deft-aoi/scripts/prepare_inference_spec.py --results-dir ${RESULTS_DIR}`. The script's `_pick_best` will now see the Phase 3 entry and select it on `far_pct` (or fall back to the loop's best if Phase 3 regressed — see safety note below).

**Safety note.** Phase 3 is not guaranteed to beat the loop's best iteration — AutoML can over-fit a small augmented dataset. The `_pick_best` lowest-`far_pct` tie-break protects against this: if Phase 3's checkpoint is worse, the iteration winner is still selected. Surface both numbers to the user in the final summary so the regression is visible.

---

## Pitfalls and quality checks

These apply to both AutoML phases. Bake them into agent behavior — don't just paste once.

### Metric pitfalls — AOI is class-imbalanced

ChangeNet AOI datasets are typically PASS-dominant (90%+ PASS rate). `val_loss` (cross-entropy) on imbalanced data has a well-known failure mode: the model can minimize CE by confidently predicting PASS for everything, achieving very low val_loss while having zero recall on defects. The val_loss winner of an AutoML sweep can be a mode-collapsed model.

For AOI, prefer:

- **FAR @ 100%-recall** as the AutoML metric directly (matches the deployment KPI; never collapses)
- Or run val_loss with a **`pred_counts` sanity check**: discard any rec whose predictions collapse to one class
- Or eval all top-K configs by FAR @ 100%-recall on the held-out set before picking — val_loss is the sort key, FAR @ 100%-recall is the decision rule

For balanced datasets and regression tasks (non-AOI DEFT applications), val_loss is fine.

### Run-to-run noise

AutoML can show 2–3× variance in metric for the same HP config across runs (seeds, dataloader shuffles). If the AutoML winner is suspiciously better than the runner-up, re-run with a fresh seed and confirm the metric holds before committing the spec to Phase 2.

### Cleanliness (data leakage)

Both AutoML phases must use a validation set distinct from the KPI test set (`<workspace>/kpi/testing_set.csv`). The KPI test set is reserved for DEFT's final reporting — touching it during AutoML biases the final number upward. The standard split: `train/base/training_set.csv` for AutoML training, `train/base/validation_set.csv` for AutoML val, `kpi/testing_set.csv` left alone until DEFT's evaluate stage.

Phase 3's train_dataset is the DEFT-augmented CSV, which contains synthetic + mined real samples beyond the base training set. The validation set stays the same — that keeps Phase 1 and Phase 3 metric numbers comparable.

### Compute budget

Total cost is roughly:
- Phase 1: `N_automl × per-rec train` — the winning rec's checkpoint *is* DEFT's baseline; no separate baseline train below
- Phase 2: `M_iter × (RCA + SDG + mining + retrain)` — usually the largest term because SDG generates synthetic images
- Phase 3: `N_automl × per-rec train` on the (larger) augmented dataset, so per-rec time is somewhat higher than Phase 1. Phase 3's winner is the deliverable; no follow-up retrain.

Surface the structure to the user up front. Ask them for their per-job time and give a wall-clock range only after that — don't make up minute numbers.

---

## Quick Start (AOI worked example)

This is what the agent says to the user when starting fresh from "run the AOI workflow":

> I'll run the canonical AOI training pipeline in three phases:
>
> **Phase 1 — AutoML baseline.** I'll sweep `<N>` configs over `<HP list>` against `<workspace>/train/base/validation_set.csv` using `bayesian` with FAR @ 100%-recall as the metric (AOI is class-imbalanced, val_loss alone risks mode collapse). After it finishes I'll spot-check per-class prediction counts before declaring a winner. The winning spec is saved to `specs/baseline_spec_automl.yaml` and the winning **checkpoint** is staged into `${RESULTS_DIR}/baseline/train/`.
>
> **Phase 2 — DEFT loop.** Phase 1's checkpoint is reused as DEFT's baseline — no redundant retrain. DEFT resumes at baseline inference → evaluate → RCA → iter 1 and continues plain-train inside the loop (`automl_policy: off` preserved). The loop runs until the KPI target is met or `max_iterations` is reached. The DEFT skill's inline Pre-Flight Summary still prints (audit trail showing the pre-seeded baseline), but is **not** a second approval point — every input was collected in the consolidated gate above.
>
> **Phase 3 — AutoML refinement.** Final AutoML sweep on the DEFT-augmented CSV (`train_combined_iter${N_final}.csv`), **warm-starting each rec from Phase 2's iter winner checkpoint** so the sweep is HP refinement on a converged model rather than from-scratch retraining (without this, Phase 3 routinely regresses against iter1 on small datasets). The winning checkpoint of this sweep is the deliverable — there's no follow-up retrain. I'll register it under `state.iterations.final_automl` and re-run `prepare_inference_spec.py` so `best_model.json` and `best_model_inference_spec.yaml` point to it — unless Phase 3 regresses, in which case the loop's best iteration wins on the same metric.
>
> Total cost is `<N_automl>` AutoML training jobs × 2 sweeps + `<M_iter>` DEFT iterations (each with SDG + retrain). No extra baseline retrain at the front; no extra retrain at the end — Phase 1's winner is DEFT's baseline, Phase 3's winner is the deliverable. If you can tell me roughly how long one ChangeNet training run takes on your hardware I can give you a wall-clock estimate. OK to proceed?

After confirmation, invoke `tao-skill-bank:tao-run-automl` (Phase 1), write the merged spec, pre-seed `deft_state.json`, invoke `tao-skill-bank:tao-run-deft-aoi` with every input pre-supplied so its inline summary is a display step rather than a re-prompt, then `tao-skill-bank:tao-run-automl` again (Phase 3). No further user pauses unless a downstream skill hits an unrecoverable hard-stop gate (called out in the consolidated summary). Summarize the trajectory at the end: baseline AutoML best → DEFT iter 1 → ... → DEFT iter N_final → Phase 3 best, so the user sees where the gains came from.

## Non-AOI DEFT applications

Same three-phase pattern applies to other DEFT skills. Swap:

- `network_arch` to the relevant model
- The DEFT skill invoked in Phase 2
- The "best HP spec file" and "best HP checkpoint" path conventions to whatever the target DEFT skill expects
- The augmented-CSV path in Phase 3 to whatever the target DEFT skill produces

The handoff shape — Phase 1 emits a *spec + checkpoint* (the checkpoint pre-seeds the DEFT baseline), Phase 2 consumes both and emits an augmented dataset, Phase 3 emits the final checkpoint — is identical. The Phase 1 → Phase 2 baseline-skip mechanism is generic: any DEFT-style loop that exposes a resumable baseline state can be seeded the same way.

---

## See also

- `tao-skill-bank:tao-run-automl` — AutoML interface, algorithms, HP ranges
- `tao-skill-bank:tao-run-deft-aoi` — full DEFT AOI loop (Phase 2 default)
- `tao-skill-bank:tao-train-visual-changenet` — underlying ChangeNet train/eval/infer skill (used by both AutoML and DEFT)
- Other `applications/deft-*` skills — non-AOI Phase 2 targets
