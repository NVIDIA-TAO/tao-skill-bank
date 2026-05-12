---
name: workflow-deft-aoi-loop
description: >
  Run the full DEFT AOI improvement loop for NVIDIA TAO VisualChangeNet / ChangeNet PCB inspection models:
  baseline evaluate, RCA, Cosmos AnomalyGen / AMP synthetic defects, k-NN mining, retraining, and deployment
  gating until FAR / recall KPI targets are met. Use for prompts like "run the DEFT loop", "fine-tune until
  FAR < 0.1% at recall=100%", or "improve my AOI ChangeNet model with RCA and synthetic defects"; do not use
  for standalone TAO training, one-off inference, generic anomaly generation, or RCA-only analysis.
license: Apache-2.0 AND CC-BY-4.0
compatibility: Requires docker + nvidia-container-toolkit. Sub-skills declare additional requirements.
metadata:
  author: NVIDIA Corporation
  version: '0.1'
allowed-tools: Read Bash Write Task
tags:
- application
- workflow
- deft
- aoi
- loop
---

# Skill: workflow-deft-aoi-loop

## When to Use This Skill

Use this skill when the user wants an agent to run the full DEFT AOI improvement loop for an NVIDIA TAO VisualChangeNet / ChangeNet PCB inspection model: baseline evaluation, RCA, synthetic defect generation, data mining, retraining, and deployment gating until a KPI target is met.

- "Run the DEFT loop"
- "Fine-tune until FAR < 0.1% at recall=100%"
- "Improve my AOI ChangeNet model using RCA and synthetic defects"
- "Iterate training until false accept rate meets the target"

Do not use this skill for a single standalone TAO training run, one-off inference, generic anomaly generation, or RCA-only analysis. Use the relevant agent directly when the user asks for only that step.

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

> **This is a fully autonomous skill.** After the pre-flight summary is confirmed, run the entire loop
> without asking for confirmation. Do not pause between steps. Do not ask "want me to
> continue?" — just continue. Only stop if a step fails with an unrecoverable error or a
> hard-stop gate fires. Print a one-line status update at each step milestone so the user
> can follow progress.

## Workflow

Execute the loop in this order (full detail in `## Pipeline` and `## Stage Execution` below):

1. **Pre-Flight.** Run every check in `## Pre-Flight`. Resolve workspace, specs, CSVs, checkpoints, container images. Hard stop on any missing input.
2. **Baseline.** Run `train -> inference -> evaluate` by invoking the `tao-skill-bank:visual-changenet` skill, then `rca` by invoking `tao-skill-bank:deft-aoi-rca-vcn`. Read `references/visual-changenet.md` and `references/deft-aoi-rca-vcn.md` first for DEFT-loop-specific args (mounts, output dirs, `deft_state.json` updates).
3. **Iterate.** For each iteration up to `max_iterations`, execute Pipeline steps 1-7. Between every step, re-read `results/loop_log.jsonl` tail + `results/deft_state.json` from disk — disk is canonical.
4. **Stop** when the KPI target is met, `max_iterations` is reached, or a hard-stop gate fires (silent-drop, AMP allocation mismatch, train/val leakage). Never auto-retry hard stops.
5. **Render** `results/DEFT_Loop_Report.html` after every stage and at loop end.

All stages run inline in the parent context. There are no subagents — the parent invokes the underlying `tao-skill-bank:*` skills directly via the Skill tool, layering DEFT-loop conventions on top via the matching `references/*.md` file.

### Using Bundled Scripts

Run bundled scripts from `scripts/` via `run_script()` or direct `python` when `run_script()` is unavailable. Resolve every path argument to an absolute host path before calling. For invocation examples, see `references/SCRIPT_USAGE.md`.

Never write `loop_log.jsonl` via `echo` or inline `jq` — the `seq` invariant requires reading the live tail through `next_seq()`.

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/log_stage.py` | Append a stage event to `results/loop_log.jsonl` (computes `seq` from disk; guarantees valid JSON). | `--log-path PATH --iter-label STR --stage {evaluate,rca,anomalygen,data_mining,train,loop_stop} --status {ok,error} --summary STR --duration-sec INT --context-tokens INT` |
| `scripts/analyze_kpi.py` | Compute FAR / threshold sweep on a ChangeNet inference CSV and pick the FAR @ 100%-recall operating point. | `csv_path` (positional) `[--output-dir PATH]` `[--label-column NAME=label]` `[--score-column NAME=siamese_score]` `[--pass-label NAME=PASS]` `[--bins INT=40]` |
| `scripts/changenet_data_pair_prepare.py` | Build the ChangeNet `(input, golden, label, object_name)` CSV from `_ng/` + `_ok/` image directories. NV_PCB_Siamese mode (`--images-dir`) emits the 14-column siamese CSV and copies images into the staged tree. | `--input-dir PATH --golden-dir PATH` `[--output PATH=dataset.csv]` `[--label STR]` `[--images-dir PATH]` `[--subdir NAME=sdg]` `[--light NAME=SolderLight]` `[--image-ext EXT=.jpg]` |
| `scripts/prepare_inference_spec.py` | Write `best_model.json` + `best_model_inference_spec.yaml` from `deft_state.json` + the training spec. Run once at loop end. See `references/prepare-for-inference.md`. | `--results-dir PATH` |

## Stage Reference Modules

Each pipeline stage maps to one underlying skill in the bank. The matching
`references/*.md` file layers DEFT-loop conventions (mounts, output dirs,
`deft_state.json` updates, `log_stage.py` summary string) on top of the
skill's generic instructions. **Read the reference file first, then invoke
the skill via the Skill tool.** If a reference file is missing, stop and
ask the user to reinstall the plugin.

| Stage(s) | Reference file | Underlying skill | Owns |
|---|---|---|---|
| `train`, `evaluate` | `references/visual-changenet.md` | `tao-skill-bank:visual-changenet` | TAO training, inference, evaluation, checkpoint discovery, TAO spec edits, two-checkpoint compare, `nvcr.io/nvidia/tao/tao-toolkit` invocation. |
| `anomalygen` | `references/cosmos-anomalygen.md` | `tao-skill-bank:cosmos-anomalygen` | AMP / AnomalyGen synthetic defect generation, `defect_spec.jsonl` routing, testcase prep, allocation recovery, and SDG output schema. |
| `rca` (VCN Classify) | `references/deft-aoi-rca-vcn.md` | `tao-skill-bank:deft-aoi-rca-vcn` | Threshold sweep, per-label weakness ranking, per-lighting expansion, `gaps.parquet` schema, and `deft_state.json` output for VCN Classify models. |
| `routing` | `references/deft-aoi-routing-vcn.md` | `tao-skill-bank:deft-aoi-routing-vcn` | VCN weak-sample routing to mining and/or AnomalyGen, `mining_gaps.parquet` + `anomalygen_gaps.parquet` outputs, dropped-label warnings. |
| `data_mining` (VCN path) | `references/deft-aoi-mining.md` | `tao-skill-bank:deft-aoi-mining` | Embed-then-mine workflow: target embedding, source-pool embedding, k-NN nearest-neighbour mining, `mined.parquet` output schema, encoder consistency requirement. |

### Invariants

**Path rule.** Use absolute host paths under `${RESULTS_DIR}/iter${ITER}/` for every stage's output, mount `<workspace>` into the container at the same path, pre-create dirs world-writable, and reject any config containing `output: /results/...` or any path outside `<workspace>`.

## Data Contract

Inputs (all paths under `<workspace>` unless absolute):

```text
<workspace>/
├── specs/baseline_spec.yaml                 # ChangeNet train/eval spec
├── train/base/
│   ├── training_set.csv                     # seed training rows
│   └── validation_set.csv                   # held-out rows; never appears in any training_set
├── kpi/
│   ├── images/                              # KPI test images (real data only — no generated images here)
│   └── testing_set.csv                      # labels live in the CSV
├── augmentation/mining_pool/
│   └── mining_pool.csv                      # mining pool — append-only; production line contributes new samples each day (Day 1 → Day N)
├── augmentation/anomalygen/checkpoints/<project>/
│   ├── defect_spec.jsonl
│   ├── checkpoints/
│   │   └── latest_checkpoint.txt
│   └── dataset/                             # reference data — one folder per defect_type
│       ├── semantic_segmentation_labels.json
│       └── <defect_type>/                   # folder name == defect_type from defect_spec.jsonl
│           ├── anomaly_image/               # defect reference images
│           ├── mask/                        # paired defect masks
│           ├── cad_mask/                    # CAD component masks (cad spatial_dependency)
│           └── clean_image/                 # clean reference images for AMP inpainting
└── results/run_<YYYYMMDD_HHMMSS>/            # created/resumed by this workflow (= ${RESULTS_DIR})
```

**ChangeNet CSV schema (VCN).** Mandatory columns: `input_path`, `golden_path`, `label`, `object_name` (siamese change-detector — a row without `golden_path` is unusable). Preserve `boardname`, scores, and provenance fields when present.

## Output Layout

Relative to `<workspace>`:

```text
results/
├── deft_state.json                          # current resume snapshot (schema: references/deft_state.json)
├── loop_log.jsonl                           # append-only stage log; single source of truth
├── DEFT_Loop_Report.html                    # re-rendered after every stage
├── best_model.json                          # inference handoff metadata (see references/prepare-for-inference.md)
├── best_model_inference_spec.yaml           # ready-to-run TAO inference spec built from training config
├── iter${ITER}_summary.md                   # ≤300-word per-iteration summary
├── baseline/{train,inference,evaluate,rca_results}/
└── iter${ITER}/
    ├── rca_results/
    ├── pool_anomalygen/                     # inputs/ — see references/cosmos-anomalygen.md → Pool Layout
    │   └── outputs/                         # AMP / AnomalyGen output — masks synthesized here
    │       ├── amp/                         # optional; present when AMP is invoked
    │       └── sdg/                         # SDG generation output
    │           ├── SDG_result.csv
    │           └── ...
    ├── mining_filter/
    │   ├── mining_pool.csv                  # combined SDG rows + real mined rows (similarity ≥ 0.9); used for training
    │   ├── knn_summary.csv                  # candidate_count, kept_count, rejected_count, similarity_threshold=0.9
    │   ├── source_embeddings.parquet        # embeddings of mining_pool candidates
    │   ├── target_embeddings.parquet        # embeddings of weak-target images
    │   └── mining_summary.txt               # per-label breakdown emitted by mining container
    ├── dataset/
    │   ├── train_combined_iter${ITER}.csv
    │   ├── train_combined_iter${ITER}_provenance.csv  # source ∈ {base_train, previous_iter_train, generated_kept}
    │   └── images/synthetic_iter${ITER}_{ng,ok}/  # staged synthetic images for ChangeNet dataloader
    └── {train,inference,export,evaluate}/
```

Never feed a previous combined CSV's rows back into training — `train_combined_iter${N-1}.csv` already contains all prior contributions.

## Pre-Flight

Resolve everything possible before asking the user. In order:

1. Locate workspace root, specs, CSVs, checkpoints, augmentation assets. Derive a timestamped run directory: `RESULTS_DIR=<workspace>/results/run_$(date +%Y%m%d_%H%M%S)`. If resuming an existing run, set `RESULTS_DIR` to the existing run directory instead (detect by checking for `results/run_*/deft_state.json`). All references to `results/` throughout this skill mean `${RESULTS_DIR}/`.
2. Read the relevant `references/*.md` files for command syntax and output contracts. See `## Stage Execution` for the stage routing table.
3. Source `<workspace>/.env` if it exists (`set -a; source <workspace>/.env; set +a`). Then verify `NGC_API_KEY` and `HF_TOKEN` are set. If either is missing, show the user `.env.example` (next to this skill) and ask them to copy it to `<workspace>/.env` and fill in the values — do not proceed until both are confirmed set.
4. `docker login nvcr.io`. Do not fall back to host-side TAO wrappers.
5. Verify every image in **Container Inventory** is present locally (`docker image inspect <ref>`).
6. Apply Path rule: pre-create iter dirs world-writable; verify a container can write to the exact output root.
7. Verify `augmentation/anomalygen/checkpoints/<project>/` (checkpoint + `latest_checkpoint.txt` + `defect_spec.jsonl`), backbone weights, GPU count. **Skip mask checks** (Mask rule: no masks on disk).

   **AnomalyGen texture** — Read `ag_config.yaml` from `augmentation/anomalygen/checkpoints/<project>/` and extract the top-level `texture` key (e.g., `PCB`). Compare it against the texture prefix encoded in each `defect_type` entry of `defect_spec.jsonl` (format: `<texture>+<anomaly>`). If they differ, note the mismatch and carry `ANOMALYGEN_TEXTURE=<value from ag_config.yaml>` as a resolved variable — Pipeline step 3 will auto-correct the staged copy. No hard stop here.

   **Cosmos base models** — Locate the directory that contains `Cosmos-Predict2-2B-Text2Image/` (the base diffusion model weights required by the AnomalyGen container at `/workspace/cosmos-anomalygen/checkpoints`). Search in order: (1) `$COSMOS_MODELS_DIR` env var if set, (2) `<workspace>/augmentation/cosmos_models/`, (3) sibling workspace directories. If found, carry the resolved path as `COSMOS_MODELS_DIR`. If not found, download them using `download_checkpoints.sh` inside the AnomalyGen container — target directory defaults to `<workspace>/augmentation/cosmos_models/` (~140 GB, idempotent):

   ```bash
   mkdir -p <workspace>/augmentation/cosmos_models
   docker run --rm \
       -e HF_TOKEN \
       -v <workspace>/augmentation/cosmos_models:/cosmos_models \
       -w /workspace/cosmos-anomalygen \
       nvcr.io/nv-metropolis-dev/metropolis-sdg/cosmos-anomalygen:1.0.3-006434bb.main \
       conda run -n cosmos-predict2 \
           bash scripts/download_checkpoints.sh --checkpoint-dir /cosmos_models
   ```

   After download, set `COSMOS_MODELS_DIR=<workspace>/augmentation/cosmos_models`. Hard stop only if `HF_TOKEN` is unset (cannot download) — in that case ask the user to add it to `<workspace>/.env` or point `COSMOS_MODELS_DIR` to an existing local copy.

   **SigLIP model** — Resolve the embedding model used for k-NN mining. Default is `google/siglip-base-patch16-224` (HuggingFace). Check in order: (1) `$SIGLIP_MODEL_PATH` env var — if it points to a local `.pth`/`.ckpt` or a HuggingFace cache dir, use it directly; (2) HuggingFace local cache at `~/.cache/huggingface/hub/models--google--siglip-base-patch16-224/`; (3) online download — only viable if network is available and `HF_TOKEN` is confirmed set. If none of the above apply (local path set but file missing, or HF_TOKEN unset with no cache), hard stop. Carry the resolved value as `SIGLIP_MODEL_PATH`; pass it to the mining stage so it does not re-resolve at runtime.

8. Run train/validation leakage check before resuming any prior run.

Ask one consolidated question only for missing required inputs. Never ask about a parameter with a default.

**Defaults:**

- `max_iterations`: 1
- `training_epochs`: `num_epochs` from `specs/baseline_spec.yaml`, else 20
- `num_SDG`: 20 (per-iteration AnomalyGen output budget; raise explicitly when more synthetic coverage is needed)
- workspace root: user prompt, else `~/workspace`
- baseline checkpoint: first `*.pth` or `*.ckpt` under `augmentation/backbone/`

### Pre-Flight Summary

Once all checks pass, print this summary and **wait for the user to confirm before starting**. This is the last thing the user sees before the loop runs autonomously.

```
╔══════════════════════════════════════════════════════════╗
║              DEFT Loop — Pre-Flight Summary              ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  KPI Target:        FAR < X% at Recall=100%              ║
║  Max Iterations:    N                                    ║
║  Training Epochs:   N per iteration                      ║
║  Num SDG:           N synthetic samples per iteration    ║
║  GPUs:              N                                    ║
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║  Dataset                                                 ║
║    Training CSV:    <path> (N rows)                      ║
║    Validation CSV:  <path> (N rows)                      ║
║    KPI test CSV:    <path> (N rows, X defect types)      ║
║    Images dir:      <path>                               ║
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║  Augmentation                                            ║
║    AnomalyGen ckpt: <path> (step N)                      ║
║    Defect spec:     <N types: type1, type2, ...>         ║
║    Texture (ag_config.yaml): <value> (matches / REMAPPED)║
║    Cosmos base models: <path> (FOUND / will download ~140GB)║
║    SigLIP model:    <cached / download / local path>     ║
║    Clean images:    N SolderLight images staged          ║
║    Backbone:        <path> (FOUND / MISSING)             ║
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║  Docker Images                                           ║
║    TAO toolkit:     ✅ / ❌                               ║
║    AnomalyGen:      ✅ / ❌                               ║
║    TAO DS (RCA/route/embed/mine): ✅ / ❌                 ║
║                                                          ║
║  Resuming:          <yes — iter N complete> / no         ║
╚══════════════════════════════════════════════════════════╝
```

To populate the summary, run:
```bash
wc -l <training_csv> <validation_csv> <kpi_testing_csv>
python3 -c "import pandas as pd; df=pd.read_csv('<kpi_testing_csv>'); print(df['label'].value_counts().to_string())"
cat <workspace>/augmentation/anomalygen/checkpoints/<project>/latest_checkpoint.txt
cat <workspace>/augmentation/anomalygen/checkpoints/<project>/defect_spec.jsonl | python3 -c "import sys,json; [print(json.loads(l)['defect_type']) for l in sys.stdin]"
nvidia-smi --list-gpus | wc -l
docker image inspect nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt nvcr.io/nv-metropolis-dev/metropolis-sdg/cosmos-anomalygen:1.0.3-006434bb.main nvcr.io/nvidian/iva/tao-toolkit-ds:aoi --format '{{.Id}}' 2>&1 | grep -c "sha256"
```

**Ask the user to confirm before proceeding.** Wait for explicit approval ("looks good", "go", "yes"). Do not start the loop until the user confirms.

### Container Inventory

Every container the loop touches, pinned to the version this orchestrator has been validated against. Sub-skills may reference `versions.yaml` keys; this table is the **application-level** truth — drift here is a hard failure.

| Stage | Image | Tag | Reference file |
|---|---|---|---|
| Train / inference / evaluate | `nvcr.io/nvidia/tao/tao-toolkit` | `6.26.3-pyt` | `references/visual-changenet.md` |
| AnomalyGen / AMP / SDG | `nvcr.io/nv-metropolis-dev/metropolis-sdg/cosmos-anomalygen` | `1.0.3-006434bb.main` | `references/cosmos-anomalygen.md` |
| VCN gap analysis / routing / embedding / mining | `nvcr.io/nvidian/iva/tao-toolkit-ds` | `aoi` | `references/deft-aoi-rca-vcn.md`, `references/deft-aoi-routing-vcn.md`, `references/deft-aoi-mining.md` |

## Augmentation Pool

Each iteration builds one **mining pool** from two complementary sources:

| Source | Selection | Contribution |
|---|---|---|
| AnomalyGen synthetic generation (step 3) | All generated images — no filtering | Defect-type diversity |
| Real images from `augmentation/mining_pool/` (step 4) | k-NN cosine similarity ≥ 0.9 to weak-target embeddings | Real-distribution anchor |

Both sources are appended into a single `mining_filter/mining_pool.csv` before fine-tuning. `train_combined_iter${N}.csv` = base training rows + mining pool rows.

**Source pool growth.** `augmentation/mining_pool/mining_pool.csv` is append-only — the production line contributes new real-image samples daily (Day 1 → Day N). Each iteration mines against the current accumulated state of the pool; later iterations naturally benefit from a richer pool. Before running the mining step, verify the file exists and is non-empty; a missing or zero-row pool is a hard stop (no real-image contribution to the mining pool for this iteration).

## Pipeline

All stages run inline in the parent context. For SKILL stages, read the matching `references/*.md` first, then invoke the underlying `tao-skill-bank:*` skill via the Skill tool. INLINE stages have no underlying skill — the parent does the work directly.

Baseline runs once before the loop: `train` → `inference` → `evaluate` (skill: `tao-skill-bank:visual-changenet`), then `rca` (skill: `tao-skill-bank:deft-aoi-rca-vcn`). Then each iteration:

1. **[SKILL — `tao-skill-bank:deft-aoi-rca-vcn`] RCA** on the previous inference result. Output: `rca_results/`. Write `iterations.<iter>.rca_target_defects` and `rca_gaps_parquet` into `deft_state.json` before advancing. See `references/deft-aoi-rca-vcn.md`.

2. **[SKILL — `tao-skill-bank:deft-aoi-routing-vcn`] Route weak samples.** Split `rca_gaps_parquet` into `routing_mining_parquet` and `routing_anomalygen_parquet` in `deft_state.json`. Downstream mining and AnomalyGen stages read those paths from disk. See `references/deft-aoi-routing-vcn.md`.

3. **[INLINE] Stage the AnomalyGen pool.** Produce the canonical layout per `references/cosmos-anomalygen.md` → **Pool Layout** at `pool_anomalygen/inputs/`. **The pool's only source is `augmentation/anomalygen/checkpoints/<project>/`** — its `dataset/` (per-defect `mask/`, `cad_mask/`, `clean_image/`, `semantic_segmentation_labels.json`), `defect_spec.jsonl`, and `ag_config.yaml`. **No real-sample injection from routing parquets, no fallback to `train/base/`, no KPI images, no prior-iteration SDG outputs, no other sources.** The reference owns the mechanical mapping table, mask format, and the texture-remap one-liner; do not duplicate or override here.

   Hard stops: `text` `spatial_dependency` with empty `roi_prompt_defect_location` in `defect_spec.jsonl`; any staged `defect_type` that fails the layout invariants in `references/cosmos-anomalygen.md`; required subdir under `<project>/dataset/<T>+<A>/` is empty (do not substitute).

4. **[SKILL — `tao-skill-bank:cosmos-anomalygen`] Run AMP / AnomalyGen** once. Invoke with `mode=inference_only`, `num_SDG=<per-iter budget>`, `num_gpus=1`, and absolute paths under `pool_anomalygen/{inputs,outputs}/`. Discover `<project>` and `step` from `latest_checkpoint.txt`. Pass `cosmos_models_dir=<COSMOS_MODELS_DIR>` (resolved in Pre-Flight step 7) so the Cosmos base weights are mounted into the container. See `references/cosmos-anomalygen.md` for the full param list, mount contract, and input hygiene.

   **SDG training contribution (INLINE).** Convert passed AnomalyGen outputs into ChangeNet paired training rows. Stage generated NG/OK image pairs under `results/iter${N}/dataset/images/synthetic_iter${N}_{ng,ok}/`, run `scripts/changenet_data_pair_prepare.py` with explicit `--input-dir`, `--golden-dir`, `--output`, `--images-dir`, and `--subdir synthetic_iter${N}`, then append the resulting rows to `mining_filter/mining_pool.csv`. SDG rows are included without k-NN filtering; only real-image mining applies the cosine threshold.

5. **[SKILL — `tao-skill-bank:deft-aoi-mining`] Mining pool — real-image contribution.** Mine real images from `augmentation/mining_pool/mining_pool.csv` against the current iteration's weak samples (`routing_mining_parquet` from `deft_state.json`) using SigLIP k-NN embeddings. **Retain only entries with cosine similarity ≥ 0.9** — lower-similarity candidates are rejected. Append the retained rows into `mining_filter/mining_pool.csv` (same file as the SDG contribution above). Output: updated `mining_filter/mining_pool.csv` and `mining_filter/knn_summary.csv` (`candidate_count`, `kept_count`, `rejected_count`, `similarity_threshold=0.9`). See `references/deft-aoi-mining.md`.

6. **[INLINE] Assemble training CSV** with monotonic growth:
   - Iter 1: `train/base/training_set.csv` + `mining_filter/mining_pool.csv`.
   - Iter N/resume: previous `train_combined_iter${N-1}.csv` + current `mining_filter/mining_pool.csv`. Never re-add `base_train` when using a previous combined CSV.
   - Write a sibling `_provenance.csv` for every output row; `source ∈ {base_train, previous_iter_train, mining_pool}`.
   - **`images_dir` for the iteration training spec** must be set to the workspace root (e.g. `/data/workspace/`), not `kpi/images/`. SDG rows already carry workspace-root-relative paths. Base training rows carry paths relative to `kpi/images/` — prepend `kpi/images/` to their `input_path` and `golden_path` so all rows share the same coordinate space.

7. **[INLINE] Train/validation leakage check.** Diff `train_combined_iter${ITER}.csv` (step 6 output) against `train/base/validation_set.csv` on `(input_path, golden_path, label, object_name, boardname)` where present. Hard stop on any validation row appearing in training.

8. **[SKILL — `tao-skill-bank:visual-changenet`] Fine-tune + evaluate.** Invoke the skill for the `train` and `evaluate` tasks. It owns TAO training, checkpoint discovery, inference, KPI analysis, and best-checkpoint selection. Write the selected checkpoint and KPI metrics into `deft_state.json`. Stop the loop if KPI met or `max_iterations` reached. See `references/visual-changenet.md`.

## State & Logging

Two artifacts persist loop state:

- `results/deft_state.json` — current resume snapshot. Schema: `references/deft_state.json`. Write with Python/jq (never `echo`) after every step.
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
  "context_tokens": <approximate current context size at write time, integer>
}
```

**Disk is the source of truth.** Before every stage, *unconditionally* re-read the last line of `loop_log.jsonl` and the full `deft_state.json`; overwrite any in-memory state. Compaction is invisible — there is nothing to detect. `seq` is always `last_seq + 1` from disk; `seq = 1` if the file does not exist.

Use `scripts/log_stage.py` to write entries (guarantees valid JSON and computes `seq` from disk). Pass `log_path` as `pathlib.Path`, not `str` — `append_stage()` calls `.exists()` on it directly. **Never emit JSON via `echo` or inline jq** — the `seq` invariant requires reading the live tail through `next_seq()`.

**On startup / resume:** Print the last 5 entries of `loop_log.jsonl` so the user can see recent progress, then proceed using the disk-loaded state.

## Stage Execution

Every stage runs in the parent's context. The disk contracts
(`deft_state.json` + `loop_log.jsonl` + `results/iter${ITER}/`) are the
canonical interface between stages — never assume in-memory state survives.

Two stage types:

- **SKILL stages** — read `references/<stage>.md` first, then invoke the
  matching `tao-skill-bank:*` skill via the Skill tool with DEFT-loop args.
  When the skill returns, update `deft_state.json` per the reference file
  and append a `loop_log.jsonl` entry via `scripts/log_stage.py`.
- **INLINE stages** — parent does the work directly (pre-flight, pool
  staging, CSV assembly, leakage check, report render). Reasons a stage
  stays INLINE: decision-dense, may need user interaction (hard stops in
  pool staging), trivial output, or meta-logic over disk state.

If the matching `references/*.md` file is missing or cannot be read, stop.
Do not replace it with generic shell commands.

### Stage Routing

Every stage uses a real `stage` value from `scripts/log_stage.py` / `references/deft_state.json::_completed_step_values`.

| Stage key | Task | Type | Skill | Required stage output |
|---|---|---|---|---|
| pre-flight / pool / CSV / resume / report | same as parent step | INLINE | — | Parent-owned artifacts only. |
| `anomalygen` | `anomalygen` | SKILL | `tao-skill-bank:cosmos-anomalygen` | `pool_anomalygen/outputs/`, generation summary, one `loop_log.jsonl` entry. |
| `train` | `train` | SKILL | `tao-skill-bank:visual-changenet` | Train artifacts and checkpoint files, one `loop_log.jsonl` entry. |
| `evaluate` | `inference` + `evaluate` | SKILL | `tao-skill-bank:visual-changenet` | Inference CSVs, KPI analysis, FAR/threshold metrics in `deft_state.json`, one `loop_log.jsonl` entry. |
| `rca` (VCN Classify) | `rca` | SKILL | `tao-skill-bank:deft-aoi-rca-vcn` | `rca_results/` with `gaps.parquet`, RCA target defects + `rca_gaps_parquet` path in `deft_state.json`, one `loop_log.jsonl` entry. |
| `routing` | `routing` | SKILL | `tao-skill-bank:deft-aoi-routing-vcn` | `routing_results/` with `mining_gaps.parquet` + `anomalygen_gaps.parquet`, routing paths in `deft_state.json`, one `loop_log.jsonl` entry. |
| `data_mining` (VCN path) | `data_mining` | SKILL | `tao-skill-bank:deft-aoi-mining` | `mining_results/` with `mined.parquet`, mining count in `deft_state.json`, one `loop_log.jsonl` entry. |

For `tao-skill-bank:visual-changenet`, pass a separate task name: `train`, `inference`, or `evaluate`. `stage` is still only `train` or `evaluate`; `inference` is a task, not a `loop_log.jsonl` / `deft_state.json` stage.

Artifacts must stay under the stage-specific output directory defined by the matching reference file. Do not invent a generic `results/iter${ITER}/<stage>/` layout.

### Post-stage check

After every stage finishes, before advancing to the next:

1. Re-read the last line of `loop_log.jsonl` and the full `deft_state.json` from disk. Trust the disk over any in-memory belief.
2. If `status=error` — halt, surface the disk evidence verbatim to the user, **do not auto-retry**. Hard stops (silent-drop gate, AMP allocation mismatch, train/val leakage) must reach the user.
3. If `status=ok` — re-render `DEFT_Loop_Report.html` and advance to the next stage per Pipeline order.

## Reports

- `results/iter${ITER}_summary.md` — ≤300 words; readable after context compaction.
- `results/iter${ITER}/report.html` — RCA targets, branch outputs, filter decision, metric delta.
- `results/DEFT_Loop_Report.html` — re-rendered **after every stage** and at loop end. Template, placeholder map, in-progress stub values, doc-comment stripping, base64 image embedding, and verification counts all live in `references/REPORT_RENDERING.md` next to the template.

## Runtime Behavior

Run without pausing. Between stages, follow `## Stage Execution`: re-read `loop_log.jsonl` tail + `deft_state.json` from disk, print a one-line status from the disk-loaded summary, re-render `DEFT_Loop_Report.html`. Append exactly one `loop_log.jsonl` entry per stage — never both before and after a skill invocation.

**Stop conditions:**

- KPI met → stop, write final report, run prepare-for-inference.
- `max_iterations` reached → stop with best-iteration report + final RCA on the best checkpoint, run prepare-for-inference.
- Unrecoverable gate failure → halt and report the exact missing artifact. Do not run a reduced loop. Do not fabricate CSVs. Skip prepare-for-inference (no valid checkpoint to hand off).

**Prepare-for-inference (final step).** Run `scripts/prepare_inference_spec.py` to emit the inference handoff:

```bash
python scripts/prepare_inference_spec.py --results-dir ${RESULTS_DIR}
```

This writes two artifacts under `${RESULTS_DIR}/`:

- `best_model.json` — handoff metadata (checkpoint, threshold, far_pct, backbone, images_dir, training_spec)
- `best_model_inference_spec.yaml` — runnable TAO inference spec built from the training config so model architecture, lighting layout, image size, and difference module match the checkpoint exactly

Downstream inference skills consume these — they should never read `deft_state.json` or the training spec directly. Full contract, consumer workflow, and silent-failure modes are documented in `references/prepare-for-inference.md`.

If a partial `${RESULTS_DIR}/` is missing iteration artifacts or fails the leakage check, restart from the last valid checkpoint instead of resuming. Starting a fresh run always creates a new timestamped `results/run_<YYYYMMDD_HHMMSS>/` — prior runs are preserved under their own directories.
