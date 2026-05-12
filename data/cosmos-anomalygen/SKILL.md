---
name: cosmos-anomalygen
description: >-
  Full Cosmos AnomalyGen pipeline — fine-tune on a new anomaly dataset, generate
  synthetic anomaly images (SDG), evaluate quality (nn_score), and search per-sample
  (guidance, crop_ratio) parameters. Three modes: full (Phase 0→7: finetune then
  generate), finetune_only (Phase 0→1: train only), inference_only (Phase 0, 2→7:
  generate from an existing checkpoint). Always invoke this skill when the user asks
  to fine-tune, train, generate anomaly images, run SDG, run AnomalyGen, evaluate
  SDG output quality, run per-sample search/refinement, or run any part of the
  AnomalyGen pipeline — even if they only mention one phase.
license: Apache-2.0
compatibility: Requires Cosmos AnomalyGen assets, checkpoints, Python helpers, and GPU runtime access.
metadata:
  author: NVIDIA Corporation
  version: '0.1'
allowed-tools: Read Bash Write
tags:
- data
- anomalygen
- sdg
---

# Cosmos AnomalyGen

Multi-phase pipeline (0–7). The `mode` flag selects which phases run.

| Phase | What runs | Mode(s) |
|---|---|---|
| 0 | Verify / download pretrained checkpoints | all |
| 1 | Fine-tune on `dataset_dir` | `full`, `finetune_only` |
| 2 | Prepare inference JSONL (AMP routing) | `full`, `inference_only` |
| 3 | SDG — generate synthetic anomaly images → `original/` (up to `num_SDG`; AMP failures may reduce count) | `full`, `inference_only` |
| 4 | Eval `original/` — emit `per_sample.csv` | `full`, `inference_only` |
| 5 | Per-sample `(guidance, crop_ratio)` search → `searched/` | `full`, `inference_only` |
| 6 | Aggregate eval on `original/` and `searched/` | `full`, `inference_only` |
| 7 | Threshold filter → `filtered/` (only if `nn_threshold` set) | `full`, `inference_only` |

Run every phase through to completion without mid-run pauses. Collect all
required parameters up front. Run every command from the repo root.

When inside a product container (`ANOMALYGEN_PRODUCT_MODE=1`), invoke
`anomalygen-guard` before any GPU work. If it reports `BLOCKED`, fix the
listed issues before continuing.

## Reference files — read before executing phases

Read **`references/finetune.md`** before Phase 0 or Phase 1 (env check,
checkpoint download details, dataset validation, config generation, training
commands, best-checkpoint selection).

Read **`references/inference.md`** before any of Phases 2–7 (AMP routing,
JSONL validation, SDG flags, eval interpretation, search loop, filtering).

**On-demand deep references** — read when troubleshooting or needing full detail for a specific phase:

| File | Read when |
|---|---|
| `references/setup.md` | Checkpoint download fails; first-time setup; HF_TOKEN / disk issues |
| `references/prep-testcase.md` | AMP fails; need full param table, helper script descriptions, allocation invariant |
| `references/sdg-inference.md` | NCCL hang; checkpoint validation error; multi-GPU VRAM question; full step list |
| `references/eval.md` | Unexpected scores; FID column order confusion; eval output format reference |
| `references/sdg-refine.md` | draws.json alignment; re-AMP heuristics; search output layout |

For `mode=full`: read **both** `finetune.md` and `inference.md` before starting.

---

## Required parameters

`num_SDG` is always allocated proportionally to training mask counts scanned
from `dataset_dir` — no distribution mode to pick.

| Parameter | Description |
|---|---|
| `mode` | `full` (Phase 0→7), `inference_only` (skip Phase 1), or `finetune_only` (Phase 0→1 only). |
| `name` | Experiment label. |
| `dataset_dir` | Training/reference dataset root. Drives mask-count allocation, AMP submask templates, and holds `semantic_segmentation_labels.json` for `cad` defects. |
| `defect_spec` | JSONL tagging each defect `spatial_dependency` as `free`/`text`/`cad`. `text` entries need `roi_prompt_defect_location`. Template: `assets/defect_spec_template.jsonl`. |
| `num_SDG` | Total output samples per bucket. *(Ignored when `mode=finetune_only`.)* |
| `num_search_run` | Per-sample search budget (`0` → skip search, only `original/` produced). *(Ignored when `mode=finetune_only`.)* |

## Conditionally required

| Parameter | Required when | Description |
|---|---|---|
| `checkpoint_dir` / `step` | `mode=inference_only` | Pre-existing fine-tuned model. In `mode=full` these are auto-derived after Phase 1; passing them is an error. In `mode=finetune_only` silently ignored — Phase 1 always trains from scratch (no resume-from-checkpoint support). Both must be present together — supplying only one is an error. |

## Optional parameters

| Parameter | Default | Description |
|---|---|---|
| `clean_dir` | `dataset_dir` | Clean images. Set only when they live outside the training dataset. Forwarded as `--clean-dir` to prep-testcase and `--clean-image-path` to finetune. |
| `validation_jsonl` | auto-generated | Pre-built validation JSONL for Phase 1. When supplied, preflight verifies every `defect_spec` type appears and paths exist. |
| `nn_threshold` | unset | `nn_score` (DINOv2 correspondence to real defects — key KPI) cutoff for Phase 7. If set, filters the final bucket (`searched/` when search ran, else `original/`) into `filtered/`, keeping only `nn_score >= nn_threshold`. Omit to skip Phase 7. |
| `max_iter` | `75000` | Phase 1 only. Total fine-tune iterations. |
| `save_iter` | `5000` | Phase 1 only. Checkpoint save interval. |
| `validation_iter` | `5000` | Phase 1 only. Validation (`nn_score`) logging interval. |
| `num_gpus` | `1` | Forwarded to Phase 1 (finetune) and Phase 3 (SDG). Eval and search rounds stay single-GPU. |
| `model_size` | `2b` | `2b` or `14b`. Used by finetune and SDG. On-disk checkpoint path encodes in upper-case (`2b`→`2B`, `14b`→`14B`). |
| `lr` | `0.02` | Phase 1 only. Learning rate. |
| `batch_size` | `2` | Phase 1 only. Per-GPU batch size. |
| `image_size` | `512` | Phase 1 only. Training resolution (square). |
| `guidance_range` | `1.5 10.0` | Phase 5 search draw range for guidance. |
| `crop_ratio_range` | `1.5 10.0` | Phase 5 search draw range for crop_ratio. |

---

## Mode validation (fail fast before any phase)

- `mode` unset → halt: *"`mode` is required (`full` | `inference_only` | `finetune_only`)."*
- `mode=inference_only` missing either `checkpoint_dir` or `step` → halt: *"inference_only requires both `checkpoint_dir` and `step`."*
- `mode=full` with `checkpoint_dir` or `step` supplied → halt: *"full mode runs finetune; use `mode=inference_only` to reuse an existing checkpoint."*

## Shared variables

Set once before Phase 0:

```bash
MODE=<full|inference_only|finetune_only>
NAME=<exp>
DATASET_DIR=<dataset_dir>
CLEAN_DIR=${clean_dir:-${DATASET_DIR}}
CKPT=<checkpoint_dir>      # required iff MODE=inference_only; auto-derived after Phase 1 when MODE=full
STEP=<iter>                # required iff MODE=inference_only; auto-derived after Phase 1 when MODE=full
NUM_SDG=<N>
DEFECT_DESC=<defect_spec.jsonl>
DEFECTS=(T+A T+B)          # TEXTURE+TYPE names derived from DEFECT_DESC
NUM_SEARCH_RUN=<N>
MODEL_SIZE=<2b|14b>
NUM_GPUS=${num_gpus:-1}
MAX_ITER=${max_iter:-75000}
SAVE_ITER=${save_iter:-5000}
VALIDATION_ITER=${validation_iter:-5000}
LR=${lr:-0.02}
BATCH_SIZE=${batch_size:-2}
IMAGE_SIZE=${image_size:-512}
VALIDATION_JSONL=${validation_jsonl:-}  # optional; set by Phase 1 Step 2 if not user-supplied

BASE=results/${NAME}
JSONL=ag_inference/${NAME}/testcase.jsonl
ORIGINAL=${BASE}/original
SEARCHED=${BASE}/searched
ROUNDS=${BASE}/rounds
```

## Guard preflight (product mode only)

```bash
if [[ "${ANOMALYGEN_PRODUCT_MODE:-}" == "1" ]]; then
    python3 ../anomalygen-guard/scripts/preflight.py \
        --mode ${MODE} \
        --name ${NAME} \
        --dataset-dir ${DATASET_DIR} \
        --defect-spec ${DEFECT_DESC} \
        --num-search-run ${NUM_SEARCH_RUN} \
        --model-size ${MODEL_SIZE} \
        ${CLEAN_DIR:+--clean-dir ${CLEAN_DIR}} \
        ${NUM_SDG:+--num-sdg ${NUM_SDG}} \
        ${CKPT:+--checkpoint-dir ${CKPT}} \
        ${STEP:+--step ${STEP}} \
        ${VALIDATION_JSONL:+--validation-jsonl ${VALIDATION_JSONL}}
fi
```

`--validation-jsonl` is forwarded only when the user supplied one; preflight
then verifies every `defect_spec` type appears in the file and that
`image_filename` / `mask_filename` paths exist. Auto-generated validation
JSONLs are caught upstream by `allocate_samples.py`, which refuses to
allocate 0 entries to any defect.

For `MODE=finetune_only`, omit `--num-sdg` if the user did not supply one.

---

## Phase 0 — checkpoints

Read `references/finetune.md §Phase 0` for HF_TOKEN requirements and what
gets downloaded (~140 GB). Verify first; download only what is missing.

```bash
scripts/check.sh \
    || scripts/download_checkpoints.sh
```

---

## Phase 1 — fine-tune (skip when `MODE=inference_only`)

Read `references/finetune.md §Phase 1` for dataset structure, config
template details, and best-checkpoint selection.

```bash
# Step 1: Validate dataset structure — derive anomaly types
python3 scripts/validate_dataset.py ${DATASET_DIR}

# Step 2: Generate validation JSONL (skip if user provided VALIDATION_JSONL)
# num_SDG = total training mask count from Step 1 output
scripts/prep_testcase.sh \
    --name validation_${NAME} \
    --num-sdg <total_mask_count> \
    --dataset-dir ${DATASET_DIR} \
    --clean-dir ${CLEAN_DIR} \
    --defect-spec ${DEFECT_DESC} \
    --amp-output-dir ag_inference/validation_${NAME}/amp \
    --output-jsonl ag_inference/validation_${NAME}/testcase.jsonl
VALIDATION_JSONL=ag_inference/validation_${NAME}/testcase.jsonl

# Step 3: Generate training config — show to user and confirm before writing
python3 scripts/generate_config.py \
    --name ${NAME} --dataset-dir ${DATASET_DIR} \
    --defect-spec ${DEFECT_DESC} --validation-jsonl ${VALIDATION_JSONL} \
    --output ag_configs/${NAME}.yaml \
    --model-size ${MODEL_SIZE} --max-iter ${MAX_ITER} \
    --save-iter ${SAVE_ITER} --validation-iter ${VALIDATION_ITER} \
    --num-gpus ${NUM_GPUS} --lr ${LR} --batch-size ${BATCH_SIZE} \
    --image-size ${IMAGE_SIZE}

# Step 4: Launch training (run in background)
scripts/launch_training.sh \
    --ag-config ag_configs/${NAME}.yaml \
    --num-gpus ${NUM_GPUS} \
    --model-size ${MODEL_SIZE}
```

After training, derive `CKPT` and `STEP` (uppercase model_size in path):

```bash
MODEL_SIZE_UPPER="${MODEL_SIZE^^}"
CKPT="./results/anomaly_gen/${NAME}/${NAME}_training_FP32_lr${LR}_bs=${BATCH_SIZE}_${MODEL_SIZE_UPPER}_${IMAGE_SIZE}x${IMAGE_SIZE}"
# STEP = highest nn_score step from validation logs (see references/finetune.md)
```

If `MODE=finetune_only`: stop here.

---

## Phase 2 — prep-testcase (skip when `MODE=finetune_only`)

Read `references/inference.md §Phase 2` for AMP routing detail and n_seeds
sizing. Do NOT pass `--seeds` — it is auto-computed and is not a recognized
flag.

```bash
scripts/prep_testcase.sh \
    --name ${NAME} --num-sdg ${NUM_SDG} \
    --dataset-dir ${DATASET_DIR} \
    --clean-dir ${CLEAN_DIR} \
    --defect-spec ${DEFECT_DESC} \
    --amp-output-dir ag_inference/${NAME}/amp \
    --output-jsonl ${JSONL}
```

---

## Phase 3 — SDG → `original/`

Read `references/inference.md §Phase 3` for JSONL validation against the
checkpoint, multi-GPU caveats, and output verification before eval.

```bash
scripts/validate_checkpoint.py ${CKPT} --step ${STEP}
scripts/validate_jsonl.py ${CKPT} ${JSONL}

scripts/run_sdg.sh \
    --checkpoint_dir ${CKPT} --step ${STEP} \
    --input_jsonl ${JSONL} --output_dir ${ORIGINAL} \
    --model_size ${MODEL_SIZE} --num_gpus ${NUM_GPUS}

scripts/verify_output.sh ${JSONL} ${ORIGINAL}
```

---

## Phase 4 — eval `original/`

Read `references/inference.md §Eval` for score interpretation and feature
count explanation.

```bash
scripts/run_eval.sh \
    --real-path ${DATASET_DIR} --generated-path ${ORIGINAL} \
    --anomaly-types ${DEFECTS[@]}
# per_sample.csv emitted to ${ORIGINAL}/per_sample.csv by default
```

---

## Phase 5 — per-sample search → `searched/`

Read `references/inference.md §Phase 5` for draw strategy, ranges, and
re-AMP guidance. For `r` in `1..NUM_SEARCH_RUN`:

1. Read prior round's `per_sample.csv` (or `${ORIGINAL}/per_sample.csv` for `r=1`).
2. Write `${ROUNDS}/round_${r}/draws.json` with Claude-chosen `(guidance, crop_ratio)` per sample.
3. Run round:

```bash
scripts/run_round.sh \
    --base-jsonl ${JSONL} \
    --draws ${ROUNDS}/round_${r}/draws.json \
    --output-dir ${ROUNDS}/round_${r} \
    --real-path ${DATASET_DIR} --anomaly-types ${DEFECTS[@]} \
    --checkpoint-dir ${CKPT} --step ${STEP} \
    [--model-size ${MODEL_SIZE}]
```

After all rounds, assemble best-seen:

```bash
scripts/assemble_searched.py \
    --original-dir ${ORIGINAL} --original-csv ${ORIGINAL}/per_sample.csv \
    --rounds-dir ${ROUNDS} --searched-dir ${SEARCHED}
```

---

## Phase 6 — aggregate eval

```bash
scripts/run_eval.sh \
    --real-path ${DATASET_DIR} --generated-path ${ORIGINAL} \
    --anomaly-types ${DEFECTS[@]} | tee ${BASE}/eval_original.log

scripts/run_eval.sh \
    --real-path ${DATASET_DIR} --generated-path ${SEARCHED} \
    --anomaly-types ${DEFECTS[@]} | tee ${BASE}/eval_searched.log
```

---

## Phase 7 — threshold filter (only if `nn_threshold` is set)

```bash
SRC=$([[ ${NUM_SEARCH_RUN} -gt 0 ]] && echo ${SEARCHED} || echo ${ORIGINAL})
FILTERED=${BASE}/filtered

python3 scripts/filter_by_nn.py \
    --searched-dir ${SRC} \
    --per-sample-csv ${SRC}/per_sample.csv \
    --threshold ${NN_THRESHOLD} \
    --filtered-dir ${FILTERED}

scripts/run_eval.sh \
    --real-path ${DATASET_DIR} --generated-path ${FILTERED} \
    --anomaly-types ${DEFECTS[@]} | tee ${BASE}/eval_filtered.log
```

---

## Output layout

```
results/<name>/
├── original/                   # up to num_SDG samples, baseline
│   ├── reconstructed_image/
│   ├── original_mask/
│   ├── overlay_image/
│   ├── original_image/
│   ├── SDG_result.csv
│   └── per_sample.csv
├── searched/                   # same count as original/, best-seen per input
│   └── (same layout as original/)
├── filtered/                   # only if nn_threshold set; count ≤ num_SDG
│   └── (same layout as searched/)
├── rounds/
│   ├── round_001/
│   ├── round_002/
│   └── search_summary.csv
├── eval_original.log
├── eval_searched.log
└── eval_filtered.log           # only if nn_threshold set
```

## Verification

1. `${ORIGINAL}/reconstructed_image/` has up to `num_SDG` images.
2. `${SEARCHED}/reconstructed_image/` matches `${ORIGINAL}/` count.
3. `${ROUNDS}/search_summary.csv` has one row per sample.
4. Both `eval_*.log` contain per-type `nn_score` (key KPI), `mnn_score`, and `fid`.
5. If `nn_threshold` set: `${FILTERED}/reconstructed_image/` count ≤ `num_SDG`; `filter_by_nn.py` stdout reports `kept N/num_SDG`.

## Error handling

- `dataset_dir` missing per-type mask dir → allocation scans zero and errors.
- AMP output short of `num_SDG` → `build_jsonl.py` logs and cycles with replacement.
- SDG failure mid-round in Phase 5 → halts; re-run resumes from the next round (rounds are append-only).
- `mode=inference_only` with a `step` not on a `save_iter` boundary → `torch.load` FileNotFoundError; `ls ${CKPT}/checkpoints/model/iter_*.pt` to find valid steps.
- See `references/finetune.md` and `references/inference.md` for phase-specific error handling.
