---
name: tao-run-deft-aoi
description: >
  Run the full DEFT AOI improvement loop for NVIDIA TAO VisualChangeNet / ChangeNet PCB inspection models:
  baseline evaluate, RCA, ingestion of customer-supplied pre-generated AnomalyGen images, k-NN mining,
  retraining, and deployment gating until FAR / recall KPI targets are met. EA variant — does not run
  AnomalyGen inline; the customer pre-generates synthetic NG/OK pairs out-of-band and the loop ingests them.
  Use for prompts like "run the DEFT loop", "fine-tune until FAR < 0.1% at recall=100%", or "improve my AOI
  ChangeNet model with RCA and pre-generated synthetic defects"; do not use for standalone TAO training,
  one-off inference, generic anomaly generation, or RCA-only analysis.
license: Apache-2.0 AND CC-BY-4.0
compatibility: Requires docker + nvidia-container-toolkit. Sub-skills declare additional requirements.
metadata:
  author: NVIDIA Corporation
  version: '0.1-ea'
allowed-tools: Read Bash Write Task
tags:
- application
- workflow
- deft
- aoi
- loop
---

# Skill: tao-run-deft-aoi

## When to Use This Skill

Use this skill when the user wants an agent to run the full DEFT AOI improvement loop for an NVIDIA TAO VisualChangeNet / ChangeNet PCB inspection model: baseline evaluation, RCA, ingestion of pre-generated synthetic defects, data mining, retraining, and deployment gating until a KPI target is met. AnomalyGen is **not** run inline in this EA variant — the customer pre-generates NG/OK pairs out-of-band and places them under `<workspace>/augmentation/anomalygen/`.

- "Run the DEFT loop"
- "Fine-tune until FAR < 0.1% at recall=100%"
- "Improve my AOI ChangeNet model using RCA and synthetic defects"
- "Iterate training until false accept rate meets the target"

Do not use this skill for a single standalone TAO training run, one-off inference, generic anomaly generation, or RCA-only analysis. Use the relevant agent directly when the user asks for only that step.

## Base Model

The loop operates on **NVIDIA TAO Visual ChangeNet** classify with the **NVIDIA C-RADIOv2-B** backbone, fine-tuned end-to-end. The architecture is defined in `specs/baseline_spec.yaml` — that file is the source of truth. All pretrained weights come from HuggingFace (`HF_TOKEN` required); `NGC_API_KEY_*` only gate container pulls. ChangeNet backbone resolution + the staged-file/HF-URL fallback for `model.backbone.pretrained_backbone_path` are owned by `references/visual-changenet.md`. SigLIP for k-NN mining is owned by `references/tao-mine-aoi-images.md`. **No AnomalyGen-side checkpoints are required in this EA variant** — pre-generated synthetic pairs are ingested directly from `<workspace>/augmentation/anomalygen/{reconstructed_image,original_image}/`; see Pipeline step 3 below.

## Train AutoML Policy

DEFT AOI owns the iterative data-improvement loop, retraining cadence, and KPI
checkpoint selection. For this workflow only, bypass model-level AutoML even
when the underlying Visual ChangeNet model metadata has `automl_enabled: true`.
Invoke every Visual ChangeNet train stage, including baseline and iteration
retrain, with the run override `automl_policy: off` / plain training. This is a
workflow-level override only; do not change model metadata, and do not apply this
policy to other workflows.

## Launch Intake

After the user confirms they want to run this workflow, ask which supported
platform they intend to run on. Generate the platform choices with:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_tao_platforms.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} --format text
```

After platform selection, run:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_tao_platforms.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} \
  --platform <platform> --format text
```

Ask only for credentials relevant to that platform, plus model-specific
credentials required by the selected workflow.

## Agent Behavior

> **There is exactly one user gate: pre-flight confirmation.** Print the Pre-Flight Summary
> (see `## Pre-Flight Summary`), then STOP and wait for the user to type "go", "yes",
> "looks good", or similar explicit approval. Do not launch any side-effecting step
> (`docker run`, training, SDG, mutations under `${RESULTS_DIR}/`) before that approval —
> reading specs, listing files, `docker image inspect`, and populating the summary table
> are fine. **"Autonomous" describes behavior *after* this gate, not before it.** Do not
> skip the gate even if the user's original prompt sounded urgent ("just run it", "go
> ahead") — the summary itself is the artifact they need to see before approving.
>
> **After the gate, the skill is fully autonomous.** Run the entire loop without asking
> for confirmation. Do not pause between steps. Do not ask "want me to continue?" — just
> continue. Only stop if a step fails with an unrecoverable error or a hard-stop gate
> fires. Print a one-line status update at each step milestone so the user can follow
> progress.

## Workflow

Execute the loop in this order (full detail in `## Pipeline` and `## Stage Execution` below):

1. **Pre-Flight.** Run every check in `## Pre-Flight`. Resolve workspace, specs, CSVs, checkpoints, container images. Hard stop on any missing input.
2. **Baseline.** If `deft_state.json` already has `iterations.baseline.stage_completed == "train"` and a `best_ckpt_path` pointing at an existing file (the upstream `tao-run-automl-deft-pipeline` pre-seeds these from its Phase 1 AutoML winner — see its Phase 1 → Phase 2 handoff), **skip the train sub-step** and resume at `inference -> evaluate` against the pre-seeded checkpoint. Otherwise run `train -> inference -> evaluate` by invoking the `tao-skill-bank:tao-train-visual-changenet` skill. Either way, then `rca` by invoking `tao-skill-bank:tao-analyze-gaps-visual-changenet`. Read `references/visual-changenet.md` and `references/tao-analyze-gaps-visual-changenet.md` first for DEFT-loop-specific args (mounts, output dirs, `deft_state.json` updates).
3. **Iterate.** For each iteration up to `max_iterations`, execute Pipeline steps 1-7. Between every step, re-read `results/loop_log.jsonl` tail + `results/deft_state.json` from disk — disk is canonical.
4. **Stop** when the KPI target is met, `max_iterations` is reached, or a hard-stop gate fires (silent-drop, AMP allocation mismatch, train/val leakage). Never auto-retry hard stops.
5. **Render** `results/DEFT_Loop_Report.html` after each completed iteration (and once more at loop end) by spawning the `reporter` subagent (`agents/reporter.md`). Per-stage renders are not done — every stage already appends one line to `loop_log.jsonl`, which is enough for a tail-watching user; the HTML render carries an iteration's worth of state and one render per iteration keeps the per-loop token cost roughly linear in iteration count, not in stage count. Do not render inline.

All pipeline stages run inline in the parent context — the parent invokes the underlying `tao-skill-bank:*` skills directly via the Skill tool, layering DEFT-loop conventions on top via the matching `references/*.md` file. The **only** delegated work is HTML report rendering, handled by the `reporter` subagent in a fresh context so an end-of-loop render is never silently dropped when the parent's context is saturated. See `## Agents` below.

### Using Bundled Scripts

Run bundled scripts from `scripts/` via `run_script()` when the harness provides it (it is a Claude Code plugin runtime helper, not a function defined in this repo); otherwise fall back to direct `python` invocation. Resolve every path argument to an absolute host path before calling. For invocation examples, see `references/SCRIPT_USAGE.md`.

Never write `loop_log.jsonl` via `echo` or inline `jq` — the `seq` invariant requires reading the live tail through `next_seq()`.

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/log_stage.py` | Append a stage event to `results/loop_log.jsonl` (computes `seq` from disk; guarantees valid JSON). `--context-tokens` is an optional placeholder; real values come from `align_token_usage.py`. | `--log-path PATH --iter-label STR --stage {evaluate,rca,anomalygen,data_mining,train,loop_stop} --status {ok,error} --summary STR --duration-sec INT [--context-tokens INT]` |
| `scripts/align_token_usage.py` | Backfill per-stage LLM token usage into `results/loop_log.jsonl` by parsing the Claude Code transcript JSONL. Run after the loop (or any time). Adds a `tokens` field per entry and refreshes `context_tokens`. | `--log-path PATH [--cwd PATH \| --project-dir PATH \| --transcript PATH ...] [--dry-run]` |
| `scripts/analyze_kpi.py` | Compute FAR / threshold sweep on a ChangeNet inference CSV and pick the FAR @ 100%-recall operating point. | `csv_path` (positional) `[--output-dir PATH]` `[--label-column NAME=label]` `[--score-column NAME=siamese_score]` `[--pass-label NAME=PASS]` `[--bins INT=40]` |
| `scripts/validate_training_csv.py` | Validate an assembled ChangeNet training CSV before launching training. Checks required columns and that every `input_path` / `golden_path` exists on disk. Stdlib only — no pandas required. | `--csv PATH --workspace-root PATH` |
| `scripts/init_deft_state.py` | Write a fresh `${RESULTS_DIR}/deft_state.json` from CLI args. Guarantees unique top-level keys. Atomic write; refuses to overwrite without `--force`. Use only on fresh runs; never on resume. EA variant: no AnomalyGen container args — pre-gen ingestion only. | `--results-dir PATH --workspace PATH --kpi-target STR --max-iterations INT --num-gpus INT --num-epochs INT [--batch-size INT] [--top-k-per-target INT] [--knn-metric STR] [--min-similarity FLOAT] [--train-container STR] [--force]` |
| `scripts/changenet_data_pair_prepare.py` | Build the ChangeNet `(input, golden, label, object_name)` CSV from `_ng/` + `_ok/` image directories. NV_PCB_Siamese mode (`--images-dir`) emits the 14-column siamese CSV and copies images into the staged tree. | `--input-dir PATH --golden-dir PATH` `[--output PATH=dataset.csv]` `[--label STR]` `[--images-dir PATH]` `[--subdir NAME=sdg]` `[--light NAME=SolderLight]` `[--image-ext EXT=.jpg]` |
| `scripts/prestage_pregen.py` | **Pre-flight one-shot.** Stages every pre-gen NG/OK pair from `<workspace>/augmentation/anomalygen/` into `${RESULTS_DIR}/synth_pool/images/synth_{ng,ok}/` once, assembles `source_pool.{csv,parquet}` (real mining_pool + sdg, with `provenance` + absolute `filepath`), writes `manifest.json`. With `--embed-with-siglip`, also runs the data-services container once on the source pool so per-iter mining can skip step 2. | `--workspace PATH --results-dir PATH [--light NAME=SolderLight] [--image-ext EXT=.jpg] [--embed-with-siglip] [--ds-image URI] [--siglip-model ID=google/siglip-base-patch16-224] [--force]` |
| `scripts/prepare_inference_spec.py` | Write `best_model.json` + `best_model_inference_spec.yaml` from `deft_state.json` + the training spec. Run once at loop end. See `references/prepare-for-inference.md`. | `--results-dir PATH` |

## Agents

| Agent | Purpose | Invoke when |
|---|---|---|
| `agents/reporter.md` | Render `results/DEFT_Loop_Report.html` from disk state (`deft_state.json` + `loop_log.jsonl` + iter summaries + RCA artifacts) following `references/REPORT_RENDERING.md`. Atomic write; verifies all placeholders filled. | After each iteration completes (with `trigger="after-iteration"`) and once more at loop end (with `trigger="loop-end"`). Note: a per-stage trigger existed in earlier revisions and is no longer recommended — the spawn cost dominated for short stages. |

Spawn via the Task tool. Pass paths only, never values — the agent reads disk as the single source of truth:

```
Task(
  description="Render DEFT report",
  subagent_type="general-purpose",
  prompt=(
    f"Read {skill_root}/agents/reporter.md and follow its instructions exactly.\n"
    f"Inputs:\n"
    f"  results_dir = {RESULTS_DIR}\n"
    f"  skill_root  = {skill_root}\n"
    f"  trigger     = after-stage   # or 'loop-end' at the very end\n"
  ),
)
```

The agent prints one status line and exits. Never render `DEFT_Loop_Report.html` inline in the parent — the whole point of this agent is to keep rendering alive when the parent's context is saturated.

## Stage Reference Modules

Each pipeline stage maps to one underlying skill in the bank. The matching
`references/*.md` file layers DEFT-loop conventions (mounts, output dirs,
`deft_state.json` updates, `log_stage.py` summary string) on top of the
skill's generic instructions. **Read the reference file first, then invoke
the skill via the Skill tool.** If a reference file is missing, stop and
ask the user to reinstall the plugin.

| Stage(s) | Reference file | Underlying skill | Owns |
|---|---|---|---|
| `train`, `evaluate` | `references/visual-changenet.md` | `tao-skill-bank:tao-train-visual-changenet` | TAO training, inference, evaluation, checkpoint discovery, TAO spec edits, two-checkpoint compare, `${TAO_PYT_IMAGE}` (resolved from `tao_toolkit.pyt` in `versions.yaml`) invocation. |
| `anomalygen` | Pre-Flight step 10 + Pipeline step 3 (both inline — no skill, no reference doc) | _inline — no skill_ | Pre-Flight stages every pre-gen NG/OK pair into `${RESULTS_DIR}/synth_pool/` once per run via `scripts/prestage_pregen.py` (basename pairing validation, copy, ChangeNet-row emission, `source_pool.{csv,parquet}` assembly, optional source SigLIP embedding). Pipeline step 3 is then a per-iter no-op that just reads `synth_pool/manifest.json` for the cached paths. **No SDG container is launched.** |
| `rca` (VCN Classify) | `references/tao-analyze-gaps-visual-changenet.md` | `tao-skill-bank:tao-analyze-gaps-visual-changenet` | Threshold sweep, per-label weakness ranking, per-lighting expansion, `gaps.parquet` schema, and `deft_state.json` output for VCN Classify models. |
| `routing` | `references/tao-route-visual-changenet-samples.md` | `tao-skill-bank:tao-route-visual-changenet-samples` *(only when AnomalyGen runs on the fly)* | VCN weak-sample routing to mining vs AnomalyGen, `mining_gaps.parquet` + `anomalygen_gaps.parquet` outputs, dropped-label warnings. **Skipped when AnomalyGen is pre-generated** — there is no AG consumer to route to, so the loop instead promotes all `kpi_gaps.parquet` rows directly into `mining_gaps.parquet` inline (see Pipeline step 2). |
| `data_mining` (VCN path) | `references/tao-mine-aoi-images.md` | `tao-skill-bank:tao-mine-aoi-images` | Embed-then-mine workflow: target embedding, source-pool embedding, k-NN nearest-neighbour mining, `mined.parquet` output schema, encoder consistency requirement. |

### Invariants

**Path rule.** Use absolute host paths under `${RESULTS_DIR}/iter${ITER}/` for every stage's output, mount `<workspace>` into the container at the same path, pre-create dirs world-writable, and reject any config containing `output: /results/...` or any path outside `<workspace>`.

## Data Contract

Inputs (all paths under `<workspace>` unless absolute):

```text
<workspace>/
├── .env                                     # NGC_API_KEY (nvcr.io/* image pulls), HF_TOKEN (HuggingFace pre-flight pulls). No AnomalyGen credentials required — this EA variant ingests pre-generated pairs.
├── specs/baseline_spec.yaml                 # ChangeNet train/eval spec
├── train/base/
│   ├── training_set.csv                     # seed training rows; ChangeNet 14-column siamese schema
│   └── validation_set.csv                   # held-out rows; checked for leakage against every train CSV
├── kpi/
│   ├── images/                              # KPI test images (real data only — no generated images here)
│   └── testing_set.csv                      # labels live in the CSV
├── augmentation/
│   ├── mining_pool/
│   │   ├── mining_pool.csv                  # append-only production-line samples; paths relative to this dir
│   │   └── images/                          # source images referenced by mining_pool.csv (e.g. *_SolderLight.jpg)
│   └── anomalygen/                          # customer-supplied pre-generated synthetic pairs (this EA variant does not run AnomalyGen)
│       ├── reconstructed_image/             # NG images (will become ChangeNet input_path); flat dir of *.jpg or *.png
│       ├── original_image/                  # OK partner images, same stems as reconstructed_image/ (will become ChangeNet golden_path)
│       └── defect_spec.jsonl                # OPTIONAL — one entry per defect_type if defect-type accounting is wanted in deft_state.json
│                                            # Stems in reconstructed_image/ and original_image/ must match 1-to-1; extensions may differ.
└── results/run_<YYYYMMDD_HHMMSS>/           # created/resumed by this workflow (= ${RESULTS_DIR})
```

**ChangeNet CSV schema (VCN).** Mandatory columns: `input_path`, `golden_path`, `label`, `object_name` (siamese change-detector — a row without `golden_path` is unusable). Preserve `boardname`, scores, and provenance fields when present. TAO builds the full image path as `{images_dir}/{input_path}/{object_name}_{light}{image_ext}` — `input_path` is a directory, not a file.

## Output Layout

Relative to `<workspace>`:

```text
results/run_<YYYYMMDD_HHMMSS>/               # = ${RESULTS_DIR}
├── deft_state.json                          # current resume snapshot (schema: references/deft_state.json)
├── loop_log.jsonl                           # append-only stage log; single source of truth
├── DEFT_Loop_Report.html                    # re-rendered after every stage by agents/reporter.md
├── best_model.json                          # inference handoff metadata (see references/prepare-for-inference.md)
├── best_model_inference_spec.yaml           # ready-to-run TAO inference spec built from training config
├── iter${ITER}_summary.md                   # ≤300-word per-iteration summary
├── synth_pool/                              # built ONCE at Pre-Flight step 10 via scripts/prestage_pregen.py
│   ├── manifest.json                        # paths + counts for the loop to reference
│   ├── images/synth_{ng,ok}/                # ChangeNet-staged pre-gen pairs (single copy, shared across iters)
│   ├── sdg_rows.csv                         # 14-col + provenance + filepath; the SDG half of source_pool
│   ├── source_pool.{csv,parquet}            # real (mining_pool) + sdg unified pool with provenance
│   ├── source_embeddings.parquet            # written only when --embed-with-siglip was passed to prestage_pregen.py
│   └── source_embed.log                     # data-services log for the source embedding (if run)
├── baseline/
│   ├── train/                               # TAO train output: model_epoch_<EEE>_step_<SSS>.pth × N, status.json, experiment.yaml, train.log
│   ├── inference/{best_val,latest}/         # per-checkpoint inference.csv + KPI plots from scripts/analyze_kpi.py
│   └── rca_results/<TS>/                    # kpi_gaps.parquet, threshold.txt, weak_samples_breakdown.txt
└── iter${ITER}/
    ├── routing_results/<TS>/                # mining_gaps.parquet, anomalygen_gaps.parquet, routing_summary.txt
    ├── anomalygen/                          # per-iter bookkeeping (just records the synth_pool/manifest.json path)
    │   └── ingest_summary.json              # per-iter audit: which synth_pool manifest was reused, counts at iter start
    ├── mining_filter/
    │   ├── mining_pool.csv                  # top-K-per-target k-NN survivors from synth_pool/source_pool (synth + real subject to same filter)
    │   ├── knn_summary.csv                  # candidate_count, kept_count, rejected_count, similarity_threshold=0.9
    │   ├── target_embeddings.parquet        # embeddings of weak-target images (per-iter — targets change each iter)
    │   └── mining_summary.txt               # per-label breakdown emitted by mining container
    ├── dataset/
    │   ├── train_combined_iter${ITER}.csv
    │   └── train_combined_iter${ITER}_provenance.csv  # source ∈ {base_train, previous_iter_train, mining_pool}
    ├── train/                               # TAO train output for iter${ITER}
    ├── inference/{best_val,latest}/
    └── rca_results/<TS>/                    # next iteration's RCA reads inference/{best_val|latest}/inference.csv
```

A previous combined CSV's rows already include every prior contribution — assemble iter N+1 from `train_combined_iter${N}.csv` plus the new `mining_filter/mining_pool.csv`, not from `train/base/training_set.csv` again.

## Pre-Flight

Resolve everything possible before asking the user. In order:

1. Locate workspace root, specs, CSVs, checkpoints, augmentation assets. Derive a timestamped run directory: `RESULTS_DIR=<workspace>/results/run_$(date +%Y%m%d_%H%M%S)`. If resuming an existing run, set `RESULTS_DIR` to the existing run directory instead (detect by checking for `results/run_*/deft_state.json`). All references to `results/` throughout this skill mean `${RESULTS_DIR}/`.

   **Host Python deps.** `scripts/analyze_kpi.py` needs `pandas`, `numpy`, `matplotlib`. Verify with `python3 -c "import pandas, numpy, matplotlib"`. If missing, set up a venv (`python3 -m venv ~/.venvs/deft && ~/.venvs/deft/bin/pip install pandas numpy matplotlib`) and invoke via that interpreter — on Ubuntu 24.04+ / fresh Brev boxes a bare `pip3 install --user` hits PEP 668. Alternatively run analysis inside the TAO toolkit image. Do not silently skip — KPI plots are part of every loop's output.
2. Read the relevant `references/*.md` files for command syntax and output contracts. See `## Stage Reference Modules` for the stage→skill mapping.
3. Source `<workspace>/.env` if it exists (`set -a; source <workspace>/.env; set +a`). Then verify the credentials the workflow actually consumes:

   | Variable | Required for | Image prefix it gates |
   |---|---|---|
   | `NGC_API_KEY` | All nvcr.io image pulls — TAO toolkit (training, inference, deploy, data services) | `nvcr.io/nvstaging/tao/*` |
   | `HF_TOKEN` | Pre-Flight HuggingFace model downloads (ChangeNet backbone, SigLIP for mining) | huggingface.co |

   Both variables must be non-empty. If either is missing, show the user `.env.example` (next to this skill), ask them to copy it to `<workspace>/.env` and fill in values, and do not proceed until set.

   **Note (EA variant):** `NGC_API_KEY_METROPOLIS_DEV` and the AnomalyGen container are **not** required — this loop ingests pre-generated AnomalyGen output.
4. `docker login nvcr.io` once with `NGC_API_KEY` (username `$oauthtoken`, password = the key). nvcr.io stores one credential per host. Do not fall back to host-side TAO wrappers.
5. **Resolve container image refs from `versions.yaml`.** The rest of this skill — including the Pre-Flight Summary's `docker image inspect` line, every stage launch, and the `references/*.md` files — references two env vars (this EA variant has no AnomalyGen container, so `AG_IMAGE` is intentionally absent). They are **not** defined elsewhere; resolve them here using `scripts/resolve_versions_key.py` (the single owner of `versions.yaml` schema knowledge) and `export` them so all downstream commands see them:

   ```bash
   SB=${TAO_SKILL_BANK_PATH:-~/tao-skills-external}
   export TAO_PYT_IMAGE=$($SB/scripts/resolve_versions_key.py images.tao_toolkit.pyt)
   export TAO_DS_IMAGE=$($SB/scripts/resolve_versions_key.py  images.tao_toolkit.data_services)
   ```

   | Env var | `versions.yaml` key | Used by |
   |---|---|---|
   | `TAO_PYT_IMAGE` | `images.tao_toolkit.pyt` | `train`, `evaluate`, `rca` (TAO toolkit pyt container) |
   | `TAO_DS_IMAGE` | `images.tao_toolkit.data_services` | `data_mining` (TAO data services container) |

   The script exits non-zero (with a diagnostic on stderr) if a key is missing or empty. Hard stop here — without the export, bash silently substitutes `""`, the next step's `docker image inspect` reports `0` MISSING for every image, and the failure mode points at the wrong root cause.
6. Verify every image resolved in step 5 is present locally (`docker image inspect "$TAO_PYT_IMAGE" "$TAO_DS_IMAGE"`).
7. Apply the path rule: pre-create iter dirs under `${RESULTS_DIR}/iter${ITER}/` and mount `<workspace>` into containers at the same absolute path. Sub-skills enforce their own container-level invariants (entrypoints, env vars); the loop just supplies the workspace mount and the resolved image URI.
8. **Verify pre-generated AnomalyGen ingestion source.** Confirm `<workspace>/augmentation/anomalygen/reconstructed_image/` and `<workspace>/augmentation/anomalygen/original_image/` both exist and are non-empty. Validate basename pairing: every file under `reconstructed_image/` must have a same-stem partner under `original_image/`. Record the pair count and, if `augmentation/anomalygen/defect_spec.jsonl` is present, the per-defect-type breakdown — both surface in the Pre-Flight Summary. Hard stop on missing dirs, empty dirs, or unpaired files (Invariants §6). Also confirm GPU count. **Stage the ChangeNet C-RADIOv2-B backbone** per `references/visual-changenet.md` → *ChangeNet backbone resolution* — always pre-download to `<workspace>/augmentation/backbone/c_radio_v2_b.pth`, then rewrite `specs/baseline_spec.yaml::model.backbone.pretrained_backbone_path` to the canonical container path. Do not leave an `https://huggingface.co/...` URL in the spec — the TAO container does not auto-pull, it treats the URL as a literal filesystem path.
9. **GPU memory sanity check.** ChangeNet classify with C-RADIOv2-B (ViT-B) at the spec defaults (`batch_size: 64`, `image_width/height: 224`, `cls_weight: [1.0, 10.0]`, learnable difference modules) OOMs on a single 48GB-class GPU. Inspect `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits` and warn if the assembled spec's `dataset.classify.batch_size` is too large for the available memory: as a rule of thumb, **≤ 16 on 48GB GPUs, ≤ 8 on 24GB GPUs**. Surface the recommendation in the Pre-Flight Summary's `GPUs` row — let the user accept or override before launch rather than failing 30 seconds into training.
10. **Stage pre-gen AnomalyGen pairs once via `scripts/prestage_pregen.py`.** The pre-gen NG/OK directories do not change between iterations, only the k-NN target set does — so file staging, `source_pool.{csv,parquet}` assembly, and source-pool SigLIP embedding all hoist here instead of running in every Pipeline iteration. The script writes everything under `${RESULTS_DIR}/synth_pool/` and emits `manifest.json`; per-iter Pipeline step 3 reads that manifest and proceeds directly to k-NN.

    ```bash
    SKILL_ROOT=${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/skills/tao-run-deft-aoi
    python3 $SKILL_ROOT/scripts/prestage_pregen.py \
        --workspace "$WORKSPACE" \
        --results-dir "$RESULTS_DIR" \
        --embed-with-siglip --ds-image "$TAO_DS_IMAGE"
    ```

    The `--embed-with-siglip` flag is strongly recommended: it embeds the source pool (~1000-2000 rows) once per run, and the per-iter mining stage then reuses `source_embeddings.parquet` (cheap re-embedding only of the ~50 weak targets). Without it, each iter re-embeds the full source pool from scratch (~50s wasted per iter).

    Record the manifest path in `deft_state.json[config.pregen]` so the per-iter Pipeline step 3 can read it without re-discovery. **Do not re-stage on resume**: a non-empty `synth_pool/manifest.json` means staging is already done; verify pair counts match and continue.
11. Run train/validation leakage check before resuming any prior run.

Ask one consolidated question only for missing required inputs. Never ask about a parameter with a default.

**Defaults:**

- `max_iterations`: 3 (the loop's value emerges only across multiple iterations; 1 disables convergence detection entirely)
- `training_epochs`: `num_epochs` from `specs/baseline_spec.yaml`, else 20
- `top_k_per_target`: 5 (k-NN survivors per weak target; governs the emergent per-iter synth budget — see Augmentation Pool)
- `min_similarity` (optional mining cosine cutoff): 0.9 — read from `config.mining_filter.min_similarity` in `deft_state.json`; the literal `0.9` referenced in Pipeline step 4 below is just the fallback default.
- workspace root: user prompt, else `~/workspace`
- pretrained backbone: first `*.pth` or `*.ckpt` under `augmentation/backbone/`; if absent, fall through to `https://huggingface.co/nvidia/C-RADIOv2-B` (HF_TOKEN required)

### Pre-Flight Summary

Once all checks pass, print this summary and **STOP — wait for explicit user approval before launching anything**. This is the one user gate in the entire workflow (see `## Agent Behavior`); the loop is autonomous *after* this point, never before.

```
## DEFT Loop — Pre-Flight Summary

### Run config
| Field                          | Value                                                                          |
| ------------------------------ | ------------------------------------------------------------------------------ |
| KPI Target                     | FAR < X% at Recall=100%                                                        |
| Max Iterations                 | N                                                                              |
| Training Epochs                | N per iteration                                                                |
| Mining top-K per target        | N (default 5; emergent synth/real per-iter budget = topn × num_weak_targets)   |
| Mining cutoff                  | cosine ≥ <min_similarity> (default 0.9)                                        |
| GPUs                           | N                                                                              |
| Resuming                       | yes — iter N complete / no                                                     |

### Dataset
| Field                          | Value                                                                          |
| ------------------------------ | ------------------------------------------------------------------------------ |
| Training CSV                   | <path> (N rows)                                                                |
| Validation CSV                 | <path> (N rows)                                                                |
| KPI test CSV                   | <path> (N rows, X defect types)                                                |
| Images dir                     | <path>                                                                         |

### Augmentation
| Field                          | Value                                                                          |
| ------------------------------ | ------------------------------------------------------------------------------ |
| Pre-gen NG dir                 | <path> (N images)                                                              |
| Pre-gen OK dir                 | <path> (N images, all paired by stem)                                          |
| Defect spec (optional)         | <N types: type1, type2, ...> / not provided                                    |
| SigLIP model                   | <cached / download / local path>                                               |
| Backbone                       | <path> (FOUND / will auto-download from HF ~393 MB)                            |

### Docker Images
Fill the `Image` column with the actual URI resolved in Pre-Flight step 5
(i.e. the value of the env var), not the literal `${VAR}` placeholder.
Print one row per env var so the audit trail shows exactly which tag will run.

| Env var          | Image (resolved from `versions.yaml`)                                          | Status     |
| ---------------- | ------------------------------------------------------------------------------ | ---------- |
| `TAO_PYT_IMAGE`  | `<$TAO_PYT_IMAGE>` (key: `images.tao_toolkit.pyt`)                             | OK/MISSING |
| `TAO_DS_IMAGE`   | `<$TAO_DS_IMAGE>` (key: `images.tao_toolkit.data_services`)                    | OK/MISSING |
```

To populate the summary, run:
```bash
wc -l <training_csv> <validation_csv> <kpi_testing_csv>
python3 -c "import pandas as pd; df=pd.read_csv('<kpi_testing_csv>'); print(df['label'].value_counts().to_string())"
# Pre-gen pair count + basename-pairing check
PG=<workspace>/augmentation/anomalygen
ls "$PG/reconstructed_image/" | wc -l
ls "$PG/original_image/" | wc -l
# Same stems on both sides? (empty diff output = paired)
diff <(ls "$PG/reconstructed_image/" | sed 's/\.[^.]*$//' | sort) \
     <(ls "$PG/original_image/"      | sed 's/\.[^.]*$//' | sort) | head
# Defect spec (optional)
[ -f "$PG/defect_spec.jsonl" ] && python3 -c "import sys,json; [print(json.loads(l)['defect_type']) for l in open('$PG/defect_spec.jsonl')]" || echo "(no defect_spec.jsonl — defect-type breakdown unavailable)"
nvidia-smi --list-gpus | wc -l
# ${TAO_PYT_IMAGE}, ${TAO_DS_IMAGE} are exported by Pre-Flight step 5
# from versions.yaml via scripts/resolve_versions_key.py. Loop per-image so the
# output maps 1:1 to the Docker Images table rows above (you can't fill a
# per-row Status column from a single aggregate "grep -c sha256" count).
for var in TAO_PYT_IMAGE TAO_DS_IMAGE; do
  ref="${!var:?$var unset — re-run Pre-Flight step 5}"
  if docker image inspect "$ref" --format '{{.Id}}' >/dev/null 2>&1; then
    printf '%-14s OK       %s\n' "$var" "$ref"
  else
    printf '%-14s MISSING  %s\n' "$var" "$ref"
  fi
done
```

**Ask the user to confirm before proceeding.** Wait for explicit approval ("looks good", "go", "yes"). Do not start the loop until the user confirms.

## Augmentation Pool

Each iteration builds **one** source CSV that feeds mining:

```
mining_filter/source_pool.csv
  = augmentation/mining_pool/mining_pool.csv   (provenance=real, paths normalized to workspace-root)
  + mining_filter/sdg_rows.csv                 (provenance=sdg,  paths already workspace-root-relative)
```

Step 3 assembles `source_pool.csv`; step 4 embeds every row with SigLIP and writes the top-K-per-target survivors (deduped, `provenance` preserved) to `mining_filter/mining_pool.csv`. `train_combined_iter${N}.csv` = base training rows + surviving mining rows. **No SDG bypass — synthetic rows go through the same k-NN as real rows.**

**Per-iter mining bounds.** With `topn` (default 5) survivors per weak target and ~30–60 weak mining-routable targets per iter:

```
total mining winners per iter ≤ topn × num_weak_mining_targets   (deduped, upper bound)
synth share of winners       = fraction of top-K slots whose nearest neighbour was a synth row (k-NN, not a knob)
```

E.g. topn=5, 50 targets, 100 real + 1000 synth in the source pool → upper bound 250 total winners; synth share falls out of SigLIP proximity, not pool sizes. Customers worried about synth dominance should grow the real pool or lower `top_k_per_target` rather than capping pre-gen pool size.

The pre-gen contribution is **per-run, not per-iteration**: the loop re-reads `augmentation/anomalygen/` every iteration. The per-iter synth winners differ because the weak-target set shifts as the model evolves — so the loop naturally picks different synth pairs each iter without any explicit ingest cap. To get new synthetic coverage between runs, the customer regenerates offline and replaces the directory before launching the next run.

**Source pool growth.** `augmentation/mining_pool/mining_pool.csv` is append-only — the production line contributes new real-image samples daily (Day 1 → Day N). Each iteration mines against the current accumulated state of the pool; later iterations naturally benefit from a richer pool. Before running the mining step, verify the file exists and is non-empty; a missing or zero-row pool is a hard stop.

**Schema.** Base training rows arrive with production metadata populated. `augmentation/mining_pool/mining_pool.csv` and `mining_filter/sdg_rows.csv` carry the 4 mandatory columns. `source_pool.csv` and `mining_filter/mining_pool.csv` add a `provenance` column. Merging into `train_combined_iter${N}.csv` follows the Data Contract CSV schema: pad the 10 optional metadata columns with empty strings when absent.

**Quirk: `mining_pool.csv`'s `input_path` is file-style** (e.g. `images/R821@1_SolderLight.jpg` — includes the basename), but TAO's dataloader formula is `{images_dir}/{input_path}/{object_name}_{light}{ext}` which requires dir-style. Before mining or training reads these rows, strip the basename (`input_path = os.path.dirname(orig_input_path)`), then prepend `augmentation/mining_pool/` to make the path workspace-root-relative. `scripts/prestage_pregen.py` does this internally during Pre-Flight source_pool assembly — do not hand-roll the rewrite in iter code; route through the script so the logic stays in one place. Failure mode if you skip the strip: `{images_dir}/augmentation/mining_pool/images/X.jpg/X_SolderLight.jpg` → file-not-found ~30 s into training.

## Pipeline

All stages run inline in the parent context. For SKILL stages, read the matching `references/*.md` first, then invoke the underlying `tao-skill-bank:*` skill via the Skill tool. INLINE stages have no underlying skill — the parent does the work directly.

Baseline runs once before the loop: `train` → `inference` → `evaluate` (skill: `tao-skill-bank:tao-train-visual-changenet`), then `rca` (skill: `tao-skill-bank:tao-analyze-gaps-visual-changenet`). The `train` sub-step is **skipped** when `deft_state.json` arrives with `iterations.baseline.stage_completed == "train"` and a `best_ckpt_path` pointing at an existing file — the `tao-run-automl-deft-pipeline` parent skill pre-seeds these from its Phase 1 AutoML winner so DEFT doesn't retrain at the same HPs. In that case, baseline picks up at `inference` against the pre-seeded checkpoint, then `evaluate`, then `rca`. Then each iteration:

1. **[SKILL — `tao-skill-bank:tao-analyze-gaps-visual-changenet`] RCA** on the previous inference result. Output: `rca_results/`. Write `iterations.<iter>.rca_target_defects` and `rca_gaps_parquet` into `deft_state.json` before advancing. See `references/tao-analyze-gaps-visual-changenet.md`.

2. **Route weak samples.** Behaviour depends on whether AnomalyGen is run on the fly or pre-generated:

   - **AnomalyGen runs on the fly** (Cosmos container is configured — `state.config.anomalygen.sub_skill` is set): **[SKILL — `tao-skill-bank:tao-route-visual-changenet-samples`]** Split `rca_gaps_parquet` into `routing_mining_parquet` and `routing_anomalygen_parquet` in `deft_state.json`. Downstream mining and AnomalyGen stages read those paths from disk. See `references/tao-route-visual-changenet-samples.md`.

   - **AnomalyGen is pre-generated** (`state.config.anomalygen.mode == "pregen_ingest"` and `sub_skill == null`): **[INLINE]** Skip the routing skill — there is no AG consumer to route to. Copy `rca_gaps_parquet` verbatim to `routing_results/<TS>/mining_gaps.parquet` and set `routing_anomalygen_parquet` to null in `deft_state.json`. **All weak gaps become mining targets**, regardless of label. The mining step (already configured with `filter_by_label: false`) will let k-NN retrieve whichever source-pool rows are visually closest to each target — real PASS or pre-gen synth NG — without any label-based pre-filter.

     ```python
     # Pre-generated AnomalyGen — one shutil.copyfile, then state update.
     import shutil, json, pathlib
     rca_pq = state["iterations"][iter_label]["rca_gaps_parquet"]
     rt_dir = pathlib.Path(f"{RESULTS_DIR}/{iter_label}/routing_results/{ts}")
     rt_dir.mkdir(parents=True, exist_ok=True)
     mining_pq = rt_dir / "mining_gaps.parquet"
     shutil.copyfile(rca_pq, mining_pq)
     state["iterations"][iter_label]["routing_mining_parquet"] = str(mining_pq)
     state["iterations"][iter_label]["routing_anomalygen_parquet"] = None
     ```

     **Why the simplification matters.** When AnomalyGen is pre-generated, the previous behaviour ran the full routing-vcn skill, which filters `mining_gaps` by *real-pool labels only* (`augmentation/mining_pool/mining_pool.csv['label'].unique()`). For customers whose mining_pool is PASS-only (the common case — production lines collect a stream of nominal samples, not defective ones), this drops every weak NG target from mining. They then get routed to `anomalygen_gaps.parquet`, which has no consumer when AG is pre-generated — silently dropped. Net effect: the loop never gets k-NN neighbours for the very defect classes the model needs to learn. Measured on a real run: every iter dropped 38/88 (43%) of weak samples this way, identically each iter. Promoting all gaps to mining recovers them.

     Log via `scripts/log_stage.py --stage routing --status ok --summary "pre-gen single-bucket: <N> gaps -> mining; no AG fanout"`.

3. **[INLINE] Read the cached pre-gen manifest.** Staging + source-pool assembly were done **once** at Pre-Flight step 10 (`scripts/prestage_pregen.py`). Per iter, this step is now a thin reader: load `${RESULTS_DIR}/synth_pool/manifest.json`, verify the artefacts referenced by it still exist (`source_pool.csv`, `source_pool.parquet`, and `source_embeddings.parquet` if `--embed-with-siglip` was used at pre-flight), and record the manifest pointer into `state.iterations.<iter>.anomalygen_ingest` so the per-iter audit trail still names the source. Log via `scripts/log_stage.py --stage anomalygen --status ok --summary "reused pre-staged synth_pool: R real + S sdg rows"`.

   The previous design re-staged all 1000 pairs + reassembled `source_pool.csv` every iteration, even though neither the pre-gen NG/OK directory nor the real mining_pool changed between iterations. That cost ~70 GB of duplicate disk on a 10-iter run, plus ~50 s of redundant SigLIP source-pool embedding per iter. Only the k-NN target set (`routing_mining_parquet`) and the per-iter `mining_pool.csv` survivors actually need to be recomputed — and those still happen in step 4.

   **Sanity checks** the per-iter step should still run (cheap, < 1 s each):
   - `synth_pool/manifest.json` exists and parses; `counts.sdg_rows` > 0.
   - The NG/OK directory listing has not changed since pre-flight (compare against `manifest.counts.sdg_rows`). Mid-run mutation is still flagged as a hard stop here — *not* silently re-ingested.
   - `augmentation/mining_pool/mining_pool.csv` still exists and is non-empty (production line append-only growth is fine; deletion is not).

   **If a customer wants to refresh the pre-gen pool**, they must re-launch the loop with a new `RESULTS_DIR` (or pass `--force` to `prestage_pregen.py` and rerun pre-flight). The loop does not re-stage mid-run.

4. **[SKILL — `tao-skill-bank:tao-mine-aoi-images`] Mine the cached source pool against the iter's weak targets.** Input: `${RESULTS_DIR}/synth_pool/source_pool.parquet` (built once at pre-flight, real + sdg). Two cases:

   - **Pre-flight ran `--embed-with-siglip`** (recommended path): skip the source-pool embedding step entirely. Embed only the iter's `routing_mining_parquet` targets (~50 rows, < 5 s), then run k-NN against the cached `synth_pool/source_embeddings.parquet`. Cost: one embedding call per iter instead of two.
   - **Pre-flight did not embed**: behave as before — embed source pool from scratch each iter. This is a documented fallback, not the recommended path.

   In both cases keep the **top-K nearest neighbours per target** (`topn=state.config.mining_filter.top_k_per_target`, default 5; deduped). The `provenance` column rides verbatim through embedding so the post-join recovers it. Optionally enforce `cosine ≥ state.config.mining_filter.min_similarity` (default 0.9) as a second filter on top of top-K. Output: `mining_filter/{target_embeddings.parquet, mined.parquet, mining_summary.txt, mining_pool.csv, knn_summary.csv}`. **Synthetic rows go through the same k-NN as real rows — no SDG bypass.** See `references/tao-mine-aoi-images.md`.

   **Mid-iteration leakage check.** Right after mining finishes — before any further CSV assembly — diff `mining_filter/mining_pool.csv` against `train/base/validation_set.csv` on `(input_path, golden_path, label, object_name, boardname)` (use `scripts/validate_training_csv.py --csv <mining_pool.csv> --workspace-root <ws> --validation-csv <validation_set.csv>`). Hard-stop on any hit. Catching leakage here, with only the new rows in scope, is cheap and isolates the offending source. The post-assembly leakage check in step 6b stays as a defence-in-depth backstop.

5. **[INLINE] Assemble training CSV** with monotonic growth:
   - Iter 1: `train/base/training_set.csv` + `mining_filter/mining_pool.csv`.
   - Iter N/resume: previous `train_combined_iter${N-1}.csv` + current `mining_filter/mining_pool.csv`. Never re-add `base_train` when using a previous combined CSV.
   - Write a sibling `_provenance.csv` for every output row; `source ∈ {base_train, previous_iter_train, mining_pool}`.
   - **`images_dir` for the iteration training spec** must be set to the workspace root (e.g. `/data/workspace/`), not `kpi/images/`. SDG rows already carry workspace-root-relative paths. Base training rows carry paths relative to `kpi/images/` — prepend `kpi/images/` to their `input_path` and `golden_path` so all rows share the same coordinate space.
   - **Normalize `label` case — preserve `PASS` uppercase, lowercase+strip everything else.** See `references/visual-changenet.md` for the dataloader rule and the failure mode if you violate it.

6. **[INLINE] Pre-train CSV validation** — run **both** checks below; hard stop on either failure. Both must pass before launching the training container; an invalid CSV burns a full GPU run before the container surfaces the root cause.

   a. **Existence check.** Run `scripts/validate_training_csv.py --csv ${RESULTS_DIR}/iter${ITER}/dataset/train_combined_iter${ITER}.csv --workspace-root <workspace>`. It hard-stops if any `input_path` / `golden_path` refers to a file missing on disk or if a required column is missing.

   b. **Train/validation leakage check.** `scripts/validate_training_csv.py` accepts `--validation-csv`; pass `train/base/validation_set.csv` so the diff on `(input_path, golden_path, label, object_name, boardname)` runs as part of the single validation pass. Hard stop on any validation row appearing in training. (Step 4 already runs the mid-iteration variant on `mining_filter/mining_pool.csv`; this check is the defence-in-depth backstop against leakage introduced by base-CSV reassembly.)

7. **[SKILL — `tao-skill-bank:tao-train-visual-changenet`] Fine-tune + evaluate.** Invoke the skill for the `train` and `evaluate` tasks. For the train task, pass the workflow override `automl_policy: off` so Visual ChangeNet runs plain training instead of model-level AutoML. It owns TAO training, checkpoint discovery, inference, KPI analysis, and best-checkpoint selection. Write the selected checkpoint and KPI metrics into `deft_state.json`. Stop the loop if KPI met or `max_iterations` reached. See `references/visual-changenet.md`.

## State & Logging

Two artifacts persist loop state:

- `results/deft_state.json` — current resume snapshot. Schema: `references/deft_state.json`. **Initialize once on a fresh run via `scripts/init_deft_state.py`** — the script builds the dict with literal-once keys so duplicates are impossible. After initialization, update with Python/jq (never `echo`) after every step; never re-init on resume.
- `results/loop_log.jsonl` — append-only event stream, one JSON line per stage:

```json
{
  "seq":            <int, monotonically increasing from 1>,
  "ts":             "<ISO-8601 UTC; stage end time>",
  "iter":           "baseline|iter1|iter2|...",
  "stage":          "evaluate|rca|routing|anomalygen|data_mining|train|loop_stop",
  "status":         "ok|error",
  "summary":        "<one-line outcome, e.g. 'FAR=52.0% threshold=0.31'>",
  "duration_sec":   <int seconds from stage start to end>,
  "context_tokens": <0 at write time; backfilled at loop end by align_token_usage.py>,
  "tokens":         <object added at loop end: input, output, cache_read, cache_create, n_messages, models>
}
```

`context_tokens` is a placeholder written as 0 by `scripts/log_stage.py` (the bash caller cannot measure LLM context size in-flight). The loop-end sequence runs `scripts/align_token_usage.py` to read the Claude Code transcript at `~/.claude/projects/<slug>/<session-id>.jsonl`, attribute each assistant message to the stage whose timestamp window it falls in, and rewrite the file with real `context_tokens` plus a per-stage `tokens` object.

**Disk is the source of truth.** Before every stage, *unconditionally* re-read the last line of `loop_log.jsonl` and the full `deft_state.json`; overwrite any in-memory state. Compaction is invisible — there is nothing to detect. `seq` is always `last_seq + 1` from disk; `seq = 1` if the file does not exist.

Use `scripts/log_stage.py` to write entries (guarantees valid JSON and computes `seq` from disk). Pass `log_path` as `pathlib.Path`, not `str` — `append_stage()` calls `.exists()` on it directly. **Never emit JSON via `echo` or inline jq** — the `seq` invariant requires reading the live tail through `next_seq()`.

**On startup / resume:** Print the last 5 entries of `loop_log.jsonl` so the user can see recent progress, then proceed using the disk-loaded state.

## Stage Execution

Every stage runs in the parent's context. The disk contracts
(`deft_state.json` + `loop_log.jsonl` + `results/iter${ITER}/`) are the
canonical interface between stages — never assume in-memory state survives.

Three stage types:

- **SKILL** — read `references/<stage>.md` first, then invoke the matching `tao-skill-bank:*` skill via the Skill tool. Stage→skill mapping is the **Stage Reference Modules** table above.
- **INLINE** — parent does the work directly (pre-flight, CSV assembly, leakage check).
- **AGENT** — parent spawns a subagent. The only AGENT stage is `agents/reporter.md` for HTML rendering.

For `tao-skill-bank:tao-train-visual-changenet`, pass a separate task name (`train`, `inference`, or `evaluate`); the `stage` value in `loop_log.jsonl` is still only `train` or `evaluate`.

If the matching `references/*.md` file is missing, stop. Do not replace it with generic shell commands. Artifacts must stay under the stage-specific output directory defined by the reference file.

### Post-stage check

After every stage finishes, before advancing:

1. Re-read the last line of `loop_log.jsonl` and the full `deft_state.json` from disk. Trust disk over in-memory.
2. If `status=error` — halt, surface the disk evidence verbatim, **do not auto-retry**.
3. If `status=ok` — print one status line and advance. Render `DEFT_Loop_Report.html` only at iteration end (`trigger="after-iteration"`) and at loop end (`trigger="loop-end"`); never inline.

## Reports

- `results/iter${ITER}_summary.md` — ≤300 words; readable after context compaction.
- `results/iter${ITER}/report.html` — RCA targets, branch outputs, filter decision, metric delta.
- `results/DEFT_Loop_Report.html` — re-rendered **after every stage** and at loop end by the `reporter` subagent (`agents/reporter.md`). The agent owns the entire render: it reads the template, the rendering protocol (`references/REPORT_RENDERING.md`), and disk state, then writes atomically. The parent's only responsibility is to spawn the agent — never render inline.

## Runtime Behavior

Run without pausing. Between stages, follow `## Stage Execution`: re-read `loop_log.jsonl` tail + `deft_state.json` from disk, print a one-line status from the disk-loaded summary, then spawn the `reporter` subagent (`agents/reporter.md`, `trigger="after-stage"`) to re-render `DEFT_Loop_Report.html`. Append exactly one `loop_log.jsonl` entry per stage — never both before and after a skill invocation.

**Loop-end sequence** (run in order, each step depends on the previous):

1. Append the final `loop_stop` entry via `scripts/log_stage.py`.
2. Backfill real per-stage token usage into `loop_log.jsonl` from the Claude Code transcript:

   ```bash
   python ${TAO_SKILL_BANK_PATH}/skills/tao-run-deft-aoi/scripts/align_token_usage.py \
       --log-path ${RESULTS_DIR}/loop_log.jsonl \
       --project-dir ~/.claude/projects/$(pwd | sed 's|/|-|g')
   ```

   This rewrites every entry's `context_tokens` field with the real context size at stage end and adds a `tokens` object (`input`, `output`, `cache_read`, `cache_create`, `n_messages`, `models`). The next step's report includes the numbers.
3. Spawn `reporter` with `trigger="loop-end"` to re-render `DEFT_Loop_Report.html` against the now-aligned log.
4. Run `scripts/prepare_inference_spec.py` (see below).

**Stop conditions:**

- KPI met → run the loop-end sequence.
- `max_iterations` reached → run the loop-end sequence with the best-iteration report + final RCA on the best checkpoint.
- Unrecoverable gate failure → halt and report the exact missing artifact. Do not run a reduced loop. Do not fabricate CSVs. Skip prepare-for-inference (no valid checkpoint to hand off); steps 1–3 of the loop-end sequence still apply.

**Prepare-for-inference (final step).** Run `scripts/prepare_inference_spec.py` to emit the inference handoff:

```bash
python scripts/prepare_inference_spec.py --results-dir ${RESULTS_DIR}
```

This writes two artifacts under `${RESULTS_DIR}/`:

- `best_model.json` — handoff metadata (checkpoint, threshold, far_pct, backbone, images_dir, training_spec)
- `best_model_inference_spec.yaml` — runnable TAO inference spec built from the training config so model architecture, lighting layout, image size, and difference module match the checkpoint exactly

Downstream inference skills consume these — they should never read `deft_state.json` or the training spec directly. Full contract, consumer workflow, and silent-failure modes are documented in `references/prepare-for-inference.md`.

If a partial `${RESULTS_DIR}/` is missing iteration artifacts or fails the leakage check, restart from the last valid checkpoint instead of resuming. Starting a fresh run always creates a new timestamped `results/run_<YYYYMMDD_HHMMSS>/` — prior runs are preserved under their own directories.
