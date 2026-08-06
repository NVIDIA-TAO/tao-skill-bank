# DEFT OD — Staging Stage Overlay (bundled glue)

This stage has no leaf skill. It turns the miner's flat list of filepaths into a trainable ODVG source, validates it, and extends the exclude set for the next iteration. All three scripts are bundled and run on the host through `scripts/deft_python.sh` — no container.

## Why this is glue and not a skill

The reference pipeline performed these steps inside internal container images (`smart_data_augmentation`, `data_utils`, `mining:v2`) whose scripts are not published. Each is pure file manipulation — pandas, `shutil`, `json` — so the loop owns them directly rather than depending on images it cannot pull or audit.

## Step 1 — Stage mined images into an ODVG source

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/stage_mined_odvg.py \
  --mined-parquet "${RESULTS_DIR}/iter${N}/mining/final_unique_files.parquet" \
  --annotations-base-dir "<source pool ODVG dir>" \
  --output-images-dir "${RESULTS_DIR}/iter${N}/tmm/images" \
  --output-annotations-dir "${RESULTS_DIR}/iter${N}/tmm/annotations" \
  --report-json "${RESULTS_DIR}/iter${N}/tmm/staging_report.json"
```

For each mined filepath it copies the image, looks up that image's ODVG record by **basename** under `--annotations-base-dir`, renumbers `image_id` sequentially from 0, remaps `instances[].label` through the labelmap, and appends the record to `tmm_odvg.jsonl`. It then writes `labelmap.json` — reusing the first `*labelmap.json` found under the annotations tree, or synthesizing one from the observed categories when none exists.

Two deliberate differences from the reference implementation:

- **`tmm_odvg.jsonl` is truncated, not appended.** The reference opened it in append mode, so re-running an iteration silently doubled every annotation. Re-running this script is idempotent.
- **Both `filepath` and `source_filepath` are accepted** as the mined-image column, because the miner generations disagree on the name.

`--min-success-rate` (default `0.0`) hard-fails when the fraction of mined images that resolved to an annotation falls below the given ratio. The script always fails when *zero* annotations were staged, since training would then have no new data.

## Step 2 — Validate the staged source

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/validate_odvg_images.py \
  --image-dir "${RESULTS_DIR}/iter${N}/tmm/images" \
  --odvg "${RESULTS_DIR}/iter${N}/tmm/annotations/tmm_odvg.jsonl" \
  --key-field file_name
```

Hard-fails when an ODVG record references an image missing on disk, when the file has no usable records, or when duplicate records are present. Catching this here is the point: without it, training fails partway through an epoch with a GPU already allocated. Images with no ODVG record are reported as orphans and are harmless; pass `--prune` to delete them.

This is a hard-stop gate. Do not train on a source that fails validation.

## Step 3 — Extend the exclude set

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/merge_exclude_parquet.py \
  --parquet-a "${RESULTS_DIR}/iter${N}/mining/final_unique_files.parquet" \
  --parquet-b "${RESULTS_DIR}/iter$((N-1))/mined_cumulative.parquet" \
  --output "${RESULTS_DIR}/iter${N}/mined_cumulative.parquet"
```

**Omit `--parquet-b` entirely on iteration 1** — there is no previous cumulative. The script also tolerates a `--parquet-b` path that does not exist, logging a note and falling back to `--parquet-a` alone.

The output feeds the *next* iteration's miner as `exclude_path`, so the loop never re-mines an image it already added.

## Mining budget

`desired_unique_count` is computed once and held constant across iterations, matching the reference pipeline:

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/compute_mining_budget.py \
  --weak-parquet "${RESULTS_DIR}/iter1/gaps/weak_images.parquet" \
  --multiplier "<multiplier>" \
  --max-count "<source pool row count>" \
  --report-json "${RESULTS_DIR}/iter${N}/mining/budget.json"
```

Point `--weak-parquet` at **iteration 1's** weak-images parquet on every iteration. Iteration 1 has the largest gap set (the model is weakest against the zero-shot baseline), so the budget stays constant as later iterations improve, instead of decaying toward zero. Only the number is written to stdout, so the caller can capture it directly.

## Commit

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/commit_stage.py \
  --results-dir "${RESULTS_DIR}" --iter-label "iter${N}" --stage stage \
  --odvg "${RESULTS_DIR}/iter${N}/tmm/annotations/tmm_odvg.jsonl" \
  --label-map "${RESULTS_DIR}/iter${N}/tmm/annotations/labelmap.json" \
  --staged-images-dir "${RESULTS_DIR}/iter${N}/tmm/images" \
  --exclude-parquet "${RESULTS_DIR}/iter${N}/mined_cumulative.parquet" \
  --summary "staged <N> images with annotations"
```
