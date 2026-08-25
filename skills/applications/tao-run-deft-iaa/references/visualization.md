# Visualization

On every platform, dispatch `visualize_prepare` and `visualize_finish` as
allowlisted zero-GPU actions in the selected compute frame.

Visualization is one ordered stage with two independently configured outputs:
contact sheets (`visualize`) and an embedding t-SNE
(`visualize_embeddings`). It is the only skippable stage.

Every `run_deft_action.py prepare` call only writes an action request. Execute
and finalize each one through `platform-execution.md` before starting the next
embedding or the finish adapter.

## Contents

- [Prepare host artifacts](#prepare-host-artifacts)
- [Embed image sets](#embed-image-sets)
- [Finish and commit](#finish-and-commit)

If both approved flags are false, commit the stage with `--skip` and no
artifacts. If either is true, a failure is not silently downgraded to a skip.

## Prepare platform artifacts

```bash
ITER_DIR="$RESULTS_DIR/iter_$N"

"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" prepare \
    --results-dir "$RESULTS_DIR" --image ds \
    --stage-dir "$ITER_DIR/visualization" --name visualize_prepare \
    --fresh-output "$ITER_DIR/visualization/visualize-prepare.host.status.json" -- \
    python3 /iaa-runtime/run_iaa_compute.py visualize_prepare \
      --results-dir /results --label "iter$N"
```

When contact sheets are enabled, this must populate
`iter_N/visualization/samples/`. When embedding visualization is enabled, it
creates exact input parquets for weak and mined images, plus a previous-data
pool only when prior continual training data exists.

## Embed image sets

Skip this section when `visualize_embeddings=false`. Otherwise launch weak and
mined embedding commands, each with its exact output marked fresh:

```bash
HF_ARGS=()
if [ "${REQUIRES_HF_TOKEN:-false}" = true ]; then
  HF_ARGS=(--pass-hf-token)
fi
WEAK_DIR="$ITER_DIR/embeddings/viz_weak"
WEAK_OUT="$WEAK_DIR/embeddings.parquet"
MINED_EMBED_DIR="$ITER_DIR/embeddings/augmented"
MINED_EMBED_OUT="$MINED_EMBED_DIR/mined_embeddings.parquet"

"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" prepare \
    --results-dir "$RESULTS_DIR" --image ds \
    --stage-dir "$WEAK_DIR" --name viz_weak_embed \
    "${HF_ARGS[@]}" \
    --fresh-output "$WEAK_OUT" -- \
    embedding image_embeddings -e /specs/image_embed_spec.yaml \
    input_parquet=/results/iter_$N/embeddings/viz_weak/input.parquet \
    output_parquet=/results/iter_$N/embeddings/viz_weak/embeddings.parquet

"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" prepare \
    --results-dir "$RESULTS_DIR" --image ds \
    --stage-dir "$MINED_EMBED_DIR" --name viz_mined_embed \
    "${HF_ARGS[@]}" \
    --fresh-output "$MINED_EMBED_OUT" -- \
    embedding image_embeddings -e /specs/image_embed_spec.yaml \
    input_parquet=/results/iter_$N/mining/mined_unique_images.parquet \
    output_parquet=/results/iter_$N/embeddings/augmented/mined_embeddings.parquet
```

If `iter_N/embeddings/previous/prev_pool.parquet` exists and is non-empty,
also run:

```bash
PREV_DIR="$ITER_DIR/embeddings/previous"
PREV_OUT="$PREV_DIR/embeddings.parquet"
HF_ARGS=()
if [ "${REQUIRES_HF_TOKEN:-false}" = true ]; then
  HF_ARGS=(--pass-hf-token)
fi

"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" prepare \
    --results-dir "$RESULTS_DIR" --image ds \
    --stage-dir "$PREV_DIR" --name viz_previous_embed \
    "${HF_ARGS[@]}" \
    --fresh-output "$PREV_OUT" -- \
    embedding image_embeddings -e /specs/image_embed_spec.yaml \
    input_parquet=/results/iter_$N/embeddings/previous/prev_pool.parquet \
    output_parquet=/results/iter_$N/embeddings/previous/embeddings.parquet
```

Do not synthesize an empty previous-data input; absence at iteration 1 without
a seed training set is the normal branch.

## Finish and commit

When `VISUALIZE_EMBEDDINGS=true`, run t-SNE only after all required embedding
actions finalize successfully. Do not run this adapter for contact-sheets-only mode:

```bash
if [ "$VISUALIZE_EMBEDDINGS" = true ]; then
  "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
    "$SKILL_ROOT/scripts/run_deft_action.py" prepare \
      --results-dir "$RESULTS_DIR" --image ds \
      --stage-dir "$ITER_DIR/visualization" --name visualize_finish \
      --fresh-output "$ITER_DIR/visualization/visualize-finish.host.status.json" -- \
      python3 /iaa-runtime/run_iaa_compute.py visualize_finish \
        --results-dir /results --label "iter$N"
fi
```

Execute and finalize each adapter through `platform-execution.md`; the selected
platform frame applies the required thread caps for scikit-learn t-SNE.

Build commit arguments only for enabled/configured outputs. Repeat
`--visualize-command-status` for weak, mined, and—when run—previous embedding
statuses:

```bash
VIS_ARGS=()
VIS_ARGS+=(--visualize-prepare-status \
  "$ITER_DIR/visualization/visualize_prepare.status.json")
if [ "$VISUALIZE" = true ]; then
  VIS_ARGS+=(--samples-dir "$ITER_DIR/visualization/samples")
fi
if [ "$VISUALIZE_EMBEDDINGS" = true ]; then
  VIS_ARGS+=(--tsne-plot "$ITER_DIR/visualization/tsne_plot.png")
  VIS_ARGS+=(--visualize-finish-status \
    "$ITER_DIR/visualization/visualize_finish.status.json")
  VIS_ARGS+=(--visualize-command-status "$WEAK_DIR/viz_weak_embed.status.json")
  VIS_ARGS+=(--visualize-command-status "$MINED_EMBED_DIR/viz_mined_embed.status.json")
  if [ -n "${PREV_OUT:-}" ]; then
    VIS_ARGS+=(--visualize-command-status "$PREV_DIR/viz_previous_embed.status.json")
  fi
fi

"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/commit_stage.py" \
    --results-dir "$RESULTS_DIR" --iter-label "iter$N" --stage visualize \
    "${VIS_ARGS[@]}" --summary "iter$N approved visualizations completed"
```

When both flags are false, use instead:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/commit_stage.py" \
    --results-dir "$RESULTS_DIR" --iter-label "iter$N" \
    --stage visualize --skip \
    --summary "visualization disabled in approved config"
```

Empty contact sheets, missing t-SNE, nonzero embedding status, or stale image
embeddings are stage failures. Apply the normal one-correction limit; do not
change the approved visualization flags to make the audit pass.

For Airflow over SLURM, the bridge synchronizes the adapter host status and
every status-bound visualization side output before another exact-tree action
may stage. This includes the contact-sheet directory, weak/mined input
parquets, an optional previous-data parquet, the final t-SNE image, and the
adapter host log referenced by each nested host status. If a completed action
from an older bridge has retained that log remotely, recover it without
rerunning compute with the bridge's `recover-visualization-host-log`
operation. The same operation can recover a missing
`visualize_prepare` log from its immutable platform action log only when the
original output-sync receipt is valid and a later successful
`visualize_finish` exact-tree action proves the historical deletion shape.
The recovery writes explicit digest evidence and never regenerates adapter
outputs. If a
run produced by an older bridge has a successful `visualize_prepare` action,
both controller and backend side outputs are absent, and the immediately
following weak embedding failed only because its canonical input was missing,
classify that one evidence shape with
`airflow_slurm_action.py classify-visualize-output-loss`. The classifier
archives the successful status, records local/backend absence plus downstream
failure digests, and converts it to the standard artifact-error form so the
normal, single attempt-2 retry can run. It refuses existing side outputs,
unrelated failures, a second recovery, or any nonterminal job; never edit or
delete the status by hand.
