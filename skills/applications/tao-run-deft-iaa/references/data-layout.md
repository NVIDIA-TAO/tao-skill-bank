# Dataset Setup and Layout

For SLURM, Kubernetes, or Brev, dispatch rebuild and materialization as the
allowlisted `dataset_rebuild` and `dataset_materialize` zero-GPU actions in
`platform-execution.md`; shell examples below are local-frame examples.

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
| `meta.tar.gz` | yes | pair/list metadata and README/vocabulary files |
| `SHA256SUMS` | no | checksums for the two archives |

There is no download branch. Preserve the archives in place; extraction goes
to the approved `DATASET_ROOT`. Do not copy multi-gigabyte archives into the
dataset tree. If an export also contains a legacy `rebuild.py`, leave it
unused; the workflow always executes the hash-bound copy bundled with this
skill.

## Approved extraction and rebuild

On remote platforms use the exact `dataset_rebuild` adapter prepare/execute/
finalize pattern in `platform-execution.md`, with `dataset_setup` as the stage
directory, `rebuild_verify.log` as the fresh output, and label `baseline`.
The adapter extracts into a deterministic run-scoped staging directory,
verifies it, and atomically promotes it. A verified final dataset is reused;
an incomplete final or staging directory is retained and reported instead of
being merged or deleted.

On a quota-managed remote filesystem, check both byte and per-user file/inode
quota before approving a new rebuild. A global `df -i` pass is insufficient:
for Lustre, inspect `quota -s` and `lfs quota -u "$(id -u)" <mount>` without
printing credentials. If the remaining file quota cannot hold every expected
caption and image-link entry, block rebuild. When the same archive-provenance
dataset is already verified on both the controller and SLURM, initialize the
new run with that existing controller dataset root and give
`airflow_slurm_action.py` its exact `--backend-dataset-root`. The signed mount
mapping reuses it read-only. Never infer a sibling dataset, silently hardlink
one, or delete an older run to make quota.

After a finalized `dataset_rebuild` failure whose synchronized log proves
`Disk quota exceeded`, inspect the deterministic run-owned staging tree. If
its diagnostic value is exhausted and deletion was approved, remove only that
non-promoted tree with `cleanup_failed_dataset_rebuild.py --confirm`. The
helper requires matching local and remote failure evidence, refuses an
existing final dataset, derives the exact staging path from immutable state,
and writes a non-recoverable cleanup receipt. Never issue a broad manual
recursive delete.

The following direct command is only for local Docker/virtualenv execution
after the pre-flight gate:

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
  "$SKILL_ROOT/scripts/iaa_deft/rebuild.py" \
    --metadata-root "$DATASET_ROOT" --out "$DATASET_ROOT" --workers 16 \
  2>&1 | tee "$RESULTS_DIR/dataset_setup/rebuild_verify.log"
```

Omit the checksum command when no manifest was approved. A checksum mismatch,
tar failure or nonzero rebuild is a hard stop. The
rebuild log must contain `VERIFY: PASS`; `VERIFY: FAIL` or a missing pass line
is not valid evidence. Do not continue using a partly rebuilt dataset.

The rebuild creates the canonical TAO-facing structure:

```text
DATASET_ROOT/
├── images_raw/
├── images/
├── captions/
├── train_pairs.json
└── val_pairs.json
```

## Materialize run splits and pool

Run the deterministic adapter through the selected platform; it calls the bundled
`materialize_iaa_eval_split`, `materialize_iaa_pool_split`, and
`convert_clip_image_list_to_parquet` with the immutable config:

```bash
STAGE_DIR="$RESULTS_DIR/dataset_setup"
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" prepare \
    --results-dir "$RESULTS_DIR" --image ds \
    --stage-dir "$STAGE_DIR" --name dataset_materialize \
    --fresh-output "$STAGE_DIR/dataset-materialize.host.status.json" -- \
    python3 /iaa-runtime/run_iaa_compute.py dataset_materialize \
      --results-dir /results --label baseline
```

Execute and finalize this request through `platform-execution.md`.

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
    --dataset-rebuild-status \
      "$RESULTS_DIR/dataset_setup/dataset_rebuild.status.json" \
    --dataset-materialize-status \
      "$RESULTS_DIR/dataset_setup/dataset_materialize.status.json" \
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
