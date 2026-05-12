---
name: deft-cosmos-rl
description: 'DEFT pipeline for Cosmos-RL video QA: gap analysis, Qwen captioning, Cosmos Predict 2.5 video generation, data
  merge, and fine-tuning. Use when running iterative data augmentation for video QA models.'
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit. Sub-skills declare additional requirements.
metadata:
  author: NVIDIA Corporation
  version: '0.1'
allowed-tools: Read Bash Write
tags:
- video
- qa
- cosmos
- deft
- sda
- iterative
---

# DEFT Cosmos-RL Workflow

## 1. Pipeline Overview

This workflow runs iterative DEFT for the Cosmos-RL video QA model. Each iteration identifies what the model gets wrong, generates new synthetic training videos for those failure cases, merges them with existing data, and retrains.

The pipeline uses three concrete skills:
- **cosmos-rl** -- training and evaluation (see `cosmos-rl.md`)
- **qwen-caption** -- caption generation for gap-analysis videos (see `qwen-caption.md`)
- **cosmos-predict-2-5** -- synthetic video generation from captions (see `cosmos-predict-2-5.md`)

## Launch Intake

After the user confirms they want this DEFT workflow, ask which supported
platform they intend to run on. Generate the platform choices with
`${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_tao_platforms.py --format text`;
do not scan platform docs or folders.

Also ask whether long-running monitoring should stay enabled and how many
minutes between status updates. Defaults: enabled, 5 minutes.

After platform selection, run
`${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_tao_platforms.py --platform <platform> --format text`
and ask only for credentials relevant to that platform, plus model-specific
credentials such as HuggingFace tokens when required.

## 2. Init Phase

Before the DEFT loop begins, the workflow runs optional initialization stages:

| Stage | Skill | Type | Condition |
|-------|-------|------|-----------|
| `zero_shot_eval` | cosmos-rl (eval) | GPU | Always runs -- establishes baseline accuracy |
| `sft_training` | cosmos-rl (train) | GPU | Only if `train_dataset_uri` is provided |
| `sft_eval` | cosmos-rl (eval) | GPU | Only after `sft_training` completes |

Init stage results are stored at job-specific paths (`s3://{bucket}/results/{job_id}`), not under the `storage_root`. The planner uses `{step_results:N}` references so that iteration stages (e.g., gap_analysis) can find init stage outputs at runtime — the orchestrator resolves these from the job store.

## 3. The 8 Iteration Stages

Each DEFT iteration runs these 8 stages in order.

### Stage 1: gap_analysis (CPU)

Compares previous eval predictions to ground truth and extracts failure cases.

- **Script:** `model/cosmos-rl/scripts/analyze_gaps.py`
- **Image:** `nvcr.io/nvidian/iva/smart_data_augmentation`
- **Input:**
  - `{prev_storage_root}` -- previous iteration's eval results. For iteration 1, this resolves to `{step_results:N}` (the last init step's job results_dir, resolved at runtime by the orchestrator from the job store). For iteration N>1, this is `iter_{N-1}`.
  - `{kpi_dataset_uri}/{kpi_ann_filename}` -- ground truth annotation file (defaults to `annotations.json`)
  - `{kpi_dataset_uri}` -- media directory for gap videos
- **Output:** `{storage_root}/gaps/gaps.parquet` -- rows with video paths, questions, wrong answers, and ground truth
- **Logic:** String comparison on `results.json` entries where `response != gt`

### Stage 2: caption_generation (CPU)

Generates descriptive captions for the gap-analysis videos using Qwen-2.5-VL.

- **Skill:** `qwen-caption` (see `qwen-caption.md` for model params and prompt format)
- **Input:** Gap videos from `{storage_root}/gaps/`
- **Output:** `{storage_root}/captions/captions.jsonl` -- one JSON line per video with caption text

### Stage 3: split_data (CPU)

Splits captions into N chunks for parallel video generation.

- **Script:** `workflow/deft-cosmos-rl/scripts/split_data.py`
- **Image:** `nvcr.io/nvidian/iva/smart_data_augmentation`
- **Input:**
  - `{storage_root}/captions/captions.jsonl`
  - `{storage_root}/gaps/gaps.parquet`
  - HuggingFace token file (`hf_tokens`)
- **Output:** `{storage_root}/splits/` -- 8 split files (split_0.jsonl through split_7.jsonl)
- **Config:** `num-splits: 8` (fixed -- matches parallel video generation)

### Stage 4: video_generation (GPU, 8-way parallel)

Generates synthetic videos from captions using Cosmos Predict 2.5.

- **Skill:** `cosmos-predict-2-5` (see `cosmos-predict-2-5.md` for generation params)
- **Parallelism:** The agent launches **8 concurrent `job-dispatch` calls**, one per split file. Each job processes one split independently.
- **Input:** `{storage_root}/splits/split_{i}.jsonl` for i in 0..7
- **Output:** `{storage_root}/generated/` -- generated videos + `all_generated.parquet` (merged by the last job or a post-step)
- **GPU requirement:** Each split job requires GPU resources; 8 jobs run simultaneously on the platform

### Stage 5: merge_outputs (CPU)

Converts generated video metadata into LLaVA-format training annotations.

- **Script:** `workflow/deft-cosmos-rl/scripts/merge_annotations.py` (mode: `create-llava`)
- **Image:** `nvcr.io/nvidian/iva/smart_data_augmentation`
- **Input:** `{storage_root}/generated/all_generated.parquet`
- **Output:** `{storage_root}/augmented/training_data.json` -- LLaVA JSON with video paths, questions, and answers

### Stage 6: data_merge (CPU)

Merges current iteration's augmented data with previous training annotations.

- **Script:** `workflow/deft-cosmos-rl/scripts/merge_annotations.py` (mode: `merge`)
- **Image:** `nvcr.io/nvidian/iva/smart_data_augmentation`
- **Input:**
  - `{storage_root}/augmented/training_data.json` -- this iteration's augmented data
  - `{prev_annotations_path}` -- previous iteration's merged annotations (or initial train dataset)
- **Output:** `{storage_root}/training_data/annotations.json` -- combined LLaVA JSON for training

### Stage 7: training (GPU)

Fine-tunes Cosmos-RL on the merged training dataset.

- **Skill:** `cosmos-rl` (train action; see `models/cosmos-rl/SKILL.md` for spec params like `dp_shard_size`, `dp_replicate_size`).
- **Input:**
  - `{storage_root}/training_data/annotations.json`
  - Checkpoint: resolved via continual_model rules (see Section 5)
- **Output:** `{storage_root}/model/` -- trained checkpoint
- **Multi-node:** set `dp_replicate_size = num_nodes` in the spec and submit with `num_nodes>1` on a multi-node-capable platform (`platform/lepton`, `platform/slurm`, `platform/kubernetes`). Cosmos-RL drives FSDP from those spec keys; see `models/cosmos-rl/SKILL.md` "Parallelism" section. Brev and local Docker are single-host only.

### Stage 8: evaluation (GPU)

Evaluates the trained model on the held-out KPI dataset.

- **Skill:** `cosmos-rl` (eval action)
- **Input:**
  - `{storage_root}/model/` -- checkpoint from training
  - `{kpi_dataset_uri}` -- KPI dataset
- **Output:** `{storage_root}/eval/` -- results.json with per-sample predictions and accuracy metrics

## 4. Storage Root Pattern

The agent computes the storage root for each iteration and passes it to `generate_plan(..., workflow="deft-cosmos-rl")`:

```
storage_root = s3://{bucket}/results/deft-cosmos-rl/{plan_id}/iter_{iteration}
```

Init stages (zero-shot eval, SFT) store results at **job-specific paths** (`s3://{bucket}/results/{job_id}`), not under storage_root. The planner uses `{step_results:N}` so iteration stages can reference these outputs — the orchestrator resolves them at runtime from the job store (SQLite, keyed by plan_id + step_id).

Iteration stages use subdirectories under `storage_root`:

```
s3://{bucket}/results/{job_id}/            # Zero-shot eval results (init)
s3://{bucket}/results/{job_id}/            # SFT results (init, conditional)
s3://{bucket}/results/deft-cosmos-rl/{plan_id}/
  iter_1/
    gaps/                                  # gap_analysis output
    captions/                              # caption_generation output
    splits/                                # split_data output (8 split files)
    generated/                             # video_generation output
    augmented/                             # merge_outputs output (LLaVA JSON)
    training_data/                         # data_merge output
    model/                                 # training checkpoint
    eval/                                  # evaluation results
  iter_2/
    ...
```

## 5. Iteration Control

### continual_model (default: true)

Controls checkpoint chaining between iterations.

| Scenario | Checkpoint Used |
|----------|----------------|
| Iter 1, continual_model=true, SFT ran | SFT checkpoint (resolved via parent_job_id chain in job store) |
| Iter 1, continual_model=true, no SFT | `init_checkpoint` (Cosmos-RL PTM) |
| Iter N>1, continual_model=true | Previous iteration's checkpoint (resolved via parent_job_id chain) |
| Any iter, continual_model=false | Always `init_checkpoint` |

### continual_dataset (default: true)

Controls training data accumulation between iterations.

| Scenario | Training Annotations Used |
|----------|--------------------------|
| Iter 1, continual_dataset=true | `train_dataset_uri` (if provided) merged with iter_1 augmented data |
| Iter N>1, continual_dataset=true | Previous iteration's merged annotations from `iter_{N-1}/training_data/annotations.json` |
| Any iter, continual_dataset=false | Only `train_dataset_uri` (if any) + current iteration's augmented data |

The `data_merge` stage resolves `{prev_annotations_path}` based on these rules.

## 6. Agent Decision Points

After each iteration's evaluation completes, the agent must:

1. **Compare against baseline:** Report accuracy vs zero-shot eval and vs previous iteration
2. **Check for plateau:** Less than 1% improvement for two consecutive iterations -- suggest stopping
3. **Check for regression:** Accuracy decreased -- investigate augmentation quality, learning rate, or distribution mismatch
4. **Report metrics:** Current accuracy, delta from previous iteration, delta from baseline, number of synthetic samples added
5. **Recommend next action:** Continue to next iteration, adjust parameters (e.g., reduce learning rate, increase num_splits), or stop

The agent should present these findings and wait for user confirmation before proceeding to the next iteration.

## 7. Parallel Video Generation

The `video_generation` stage runs 8 jobs in parallel. The agent must:

1. Call `job-dispatch` 8 times, once per split file (`split_0.jsonl` through `split_7.jsonl`)
2. Track all 8 job IDs independently
3. Wait for all 8 to complete before proceeding to `merge_outputs`
4. If a subset of splits fail, retry only the failed splits (do not re-run successful ones)
5. All 8 jobs write to `{storage_root}/generated/` -- the merge step combines their outputs

See `cosmos-predict-2-5.md` for per-job GPU requirements and generation parameters.

## 8. KPI Dataset Validation

Before generating the plan, the agent should validate the KPI dataset and auto-discover the annotation file:

1. List the dataset contents with `aws s3 ls <kpi_dataset_uri>` (or storage CLI equivalent)
2. Look for a `.json` file in the root — this is the annotation file
3. If the annotation file is **not** `annotations.json` (the default), pass `kpi_ann_filename` as an override in `extra_args`:
   ```python
   extra_args={'kpi_ann_filename': 'actual_annotation_filename.json', ...}
   ```
4. If no `.json` file is found at the root, ask the user for the annotation filename

This avoids an extra turn asking the user for the annotation filename — the agent discovers it automatically from the bucket listing.

## 9. Error Patterns

### gap_analysis
- **No results found:** Previous eval did not produce `results.json`. The `results-dir` arg should resolve to the prior eval step's job results_dir via `{step_results:N}`. If the referenced step hasn't been run, the orchestrator will raise a clear error. Re-run the eval step first.
- **Empty parquet:** Model got everything right (no gaps). Report to user -- DEFT may not be needed.
- **Path mismatch:** `{kpi_dataset_uri}/{kpi_ann_filename}` does not point to a valid annotation file. Verify with `aws s3 ls` or your storage CLI.

### caption_generation
- **Qwen endpoint timeout:** Retry with backoff. Check Qwen service health on the platform.
- **Empty captions:** Input videos may be corrupt or unreadable. Check gap_analysis video paths.

### split_data
- **HF token error:** `hf_tokens` file path is wrong or token is expired. User must provide a valid token file.
- **Zero splits produced:** captions.jsonl is empty. Check caption_generation output.

### video_generation
- **GPU OOM on a split:** Reduce batch size in cosmos-predict-2-5 spec params. Retry the failed split only.
- **Split N failed, others succeeded:** Retry only split N. Do not re-run all 8 jobs.
- **All splits timeout:** Platform may be under load. Check resource availability, retry with backoff.

### merge_outputs
- **Missing generated parquet:** Not all video_generation splits completed. Check for failed splits.

### data_merge
- **prev_annotations_path not found:** Previous iteration's training_data/annotations.json is missing. Check continual_dataset setting and prior iteration success.
- **Schema mismatch:** Augmented data JSON does not match LLaVA format. Check merge_outputs output.

### training
- **NaN loss:** Learning rate too high. Reduce by 10x and retry.
- **Checkpoint not found (continual_model=true):** Previous iteration's training failed. Fall back to init_checkpoint or re-run prior iteration.
- **GPU eviction:** Platform evicted the job. Retry, or use a dedicated node group.

### evaluation
- **Low accuracy despite good training loss:** Overfitting on augmented data. Reduce epochs or increase data diversity.
- **Results.json missing after job completes:** Check job logs for internal container failure (Execution status: FAIL).
