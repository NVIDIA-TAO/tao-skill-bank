# Dataset Setup and Layout

Read during pre-flight and when the audit selects
`baseline/dataset_setup`.

## Contents

- [Input contract](#input-contract)
- [Approved extraction and rebuild](#approved-extraction-and-rebuild)
- [Materialize run splits and pool](#materialize-run-splits-and-pool)
- [Run output layout](#run-output-layout)

## Input contract

The IAA TAO-FT export contains:

| File | Required | Expected content |
|---|---|---|
| `images_raw.tar` | yes | source images under `images_raw/<source_split>/...` |
| `meta.tar.gz` | yes | pair/list metadata, README/vocabulary files, and `rebuild.py` |
| `SHA256SUMS` | no | checksums for the two archives |

There is no download branch. Preserve the archives in place; extraction goes
to the approved `DATASET_ROOT`. Do not copy multi-gigabyte archives into the
dataset tree.

## Approved extraction and rebuild

Only after the pre-flight gate:

```bash
set -e
set -o pipefail
mkdir -p "$DATASET_ROOT" "$RESULTS_DIR/dataset_setup"

# Execute only when a manifest was approved. Persist pass/fail, not digest
# values or the manifest contents.
if [ -n "${CHECKSUMS_FILE:-}" ]; then
  if (cd "$(dirname "$CHECKSUMS_FILE")" && \
      sha256sum --check --status "$(basename "$CHECKSUMS_FILE")"); then
    printf 'CHECKSUM_VERIFY: PASS\n' \
      >"$RESULTS_DIR/dataset_setup/checksum_verify.log"
  else
    printf 'CHECKSUM_VERIFY: FAIL\n' \
      >"$RESULTS_DIR/dataset_setup/checksum_verify.log"
    exit 1
  fi
fi

tar -xzf "$METADATA_ARCHIVE" -C "$DATASET_ROOT"
tar -xf "$IMAGES_ARCHIVE" -C "$DATASET_ROOT"

"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$DATASET_ROOT/rebuild.py" --workers 16 \
  2>&1 | tee "$RESULTS_DIR/dataset_setup/rebuild_verify.log"
```

Omit the checksum command when no manifest was approved. A checksum mismatch,
tar failure, missing `rebuild.py`, or nonzero rebuild is a hard stop. The
rebuild log must contain `VERIFY: PASS`; `VERIFY: FAIL` or a missing pass line
is not valid evidence. Do not continue using a partly rebuilt dataset.

The rebuild creates the canonical TAO-facing structure:

```text
DATASET_ROOT/
├── images_raw/
├── images/
├── captions/
├── train_pairs.json
├── val_pairs.json
└── rebuild.py
```

## Materialize run splits and pool

Run the deterministic adapter; it calls the bundled
`materialize_iaa_eval_split`, `materialize_iaa_pool_split`, and
`convert_clip_image_list_to_parquet` with the immutable config:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/run_iaa_stage.py" dataset-materialize \
    --results-dir "$RESULTS_DIR" \
    --deft-config "$RESULTS_DIR/config/deft_config.yaml"
```

Required outputs are exact, not glob-selected:

```text
iaa_splits/eval_list.txt
iaa_splits/eval_pairs.json
iaa_splits/val_list.txt
iaa_splits/aug_pool_list.txt
iaa_splits/aug_pool_pairs.json
embeddings/source/source_pool.parquet
```

The commit validator checks all split files, JSON structure, non-empty parquet
rows/schema, and dataset links/files. When a checksum manifest was approved,
also pass its exact verification log. Commit only after the adapter succeeds:

```bash
CHECKSUM_ARGS=()
if [ -n "${CHECKSUMS_FILE:-}" ]; then
  CHECKSUM_ARGS+=(--checksum-verify-log \
    "$RESULTS_DIR/dataset_setup/checksum_verify.log")
fi

"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/commit_stage.py" \
    --results-dir "$RESULTS_DIR" --iter-label baseline \
    --stage dataset_setup \
    --iaa-splits-dir "$RESULTS_DIR/iaa_splits" \
    --source-pool-parquet "$RESULTS_DIR/embeddings/source/source_pool.parquet" \
    --verify-log "$RESULTS_DIR/dataset_setup/rebuild_verify.log" \
    --dataset-materialize-status \
      "$RESULTS_DIR/dataset_setup/dataset-materialize.host.status.json" \
    "${CHECKSUM_ARGS[@]}" \
    --summary "IAA dataset rebuilt and run splits materialized"
```

Then run the audit. The only legal next stage is `baseline/pool_embed`.

## Run output layout

These names are canonical IAA conventions and must not be renamed:

```text
RESULTS_DIR/
├── config/                         immutable copied specs
├── deft_state.json
├── loop_log.jsonl
├── dataset_setup/                  rebuild_verify.log; optional checksum_verify.log
├── iaa_splits/                     eval, validation, and mining-pool files
├── embeddings/source/              source_pool.parquet, embeddings.parquet
├── caption_selection_history.json
├── mining_selection_history.json
├── zs/                             baseline eval
└── iter_N/
    ├── gaps/
    ├── embeddings/
    ├── mining/
    ├── visualization/
    ├── specs/
    ├── train/
    ├── pretrained/
    ├── evaluate/
    └── iteration_summary.json
```

The caption mining pool comes from training pairs. The eval split comes from
validation pairs. History selection still rechecks basename disjointness
before every training iteration; source assumptions never replace that gate.
