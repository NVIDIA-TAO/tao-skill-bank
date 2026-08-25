# CLIP Train and Evaluate

On a remote platform, eval/train configuration, checkpoint publication, metric
parsing, and iteration summary are allowlisted zero-GPU actions. Only commit
and audit metadata remain controller-local.

Read with `metric-contract.md` when the audit selects `train` or `evaluate`.
The IAA workflow uses plain TAO CLIP train/evaluate commands. There is no
AutoML branch and no hand-authored per-stage YAML.

Every `run_deft_action.py prepare` call only writes the immutable action
request. Execute and finalize it through `platform-execution.md` before parsing,
publishing, or committing its outputs.

## Contents

- [Evaluate](#evaluate)
- [Train](#train)
- [Failure handling](#failure-handling)

## Evaluate

The baseline uses the public SigLIP2 base weights and the `zs/` directory.
Iteration N uses its freshly published best checkpoint and `iter_N/`.

1. Generate the exact eval config through the selected platform's signed
   zero-GPU adapter action:

   ```bash
   PHASE_DIR="$RESULTS_DIR/zs"
   [ "$LABEL" = baseline ] || PHASE_DIR="$RESULTS_DIR/iter_${LABEL#iter}"
   "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
     "$SKILL_ROOT/scripts/run_deft_action.py" prepare \
       --results-dir "$RESULTS_DIR" --image ds \
       --stage-dir "$PHASE_DIR/specs" --name eval_config \
       --fresh-output "$PHASE_DIR/specs/eval_config.yaml" -- \
       python3 /iaa-runtime/run_iaa_compute.py eval_config \
         --results-dir /results --label "$LABEL"
   ```

   Execute and finalize this request through `platform-execution.md`. Its
   `eval_config.status.json`, not a controller-host status, is commit evidence.

2. Set `PHASE_DIR="$RESULTS_DIR/zs"` and `CONTAINER_PHASE=zs` for `baseline`;
   otherwise set `PHASE_DIR="$RESULTS_DIR/iter_$N"` and
   `CONTAINER_PHASE="iter_$N"`. Launch evaluation with every canonical metric
   output marked fresh. The detailed metrics CSV is a required input to the
   next gap-analysis stage, so omitting it from the platform output contract is
   a hard workflow error:

   ```bash
   EVAL_DIR="$PHASE_DIR/evaluate"
   DETAIL_METRICS="$EVAL_DIR/nvidia_pas_metrics.csv"
   METRICS="$EVAL_DIR/nvidia_pas_metrics_aggregate.csv"
   WEIGHTED_METRICS="$EVAL_DIR/nvidia_pas_metrics_weighted_aggregate.csv"
   TAO_STATUS="$EVAL_DIR/status.json"
   HF_ARGS=()
   if [ "${REQUIRES_HF_TOKEN:-false}" = true ]; then
     HF_ARGS=(--pass-hf-token)
   fi

   "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
     "$SKILL_ROOT/scripts/run_deft_action.py" prepare \
       --results-dir "$RESULTS_DIR" --image pyt \
       --stage-dir "$EVAL_DIR" --name evaluate \
       "${HF_ARGS[@]}" \
       --fresh-output "$DETAIL_METRICS" \
       --fresh-output "$METRICS" \
       --fresh-output "$WEIGHTED_METRICS" \
       --fresh-output "$TAO_STATUS" -- \
       clip evaluate -e "/results/$CONTAINER_PHASE/specs/eval_config.yaml"
   ```

   The native backend exit must be zero, all three metric CSVs must be
   non-empty, and the TAO status must contain `Evaluate finished successfully`.
   A stale CSV next to a failed status is not evidence.
3. Parse the approved metric contract exactly as shown in
   `metric-contract.md`. For an iteration, also bind the canonical summary:

   ```bash
     "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
       "$SKILL_ROOT/scripts/run_deft_action.py" prepare \
         --results-dir "$RESULTS_DIR" --image ds \
         --stage-dir "$PHASE_DIR" --name iteration_summary \
         --fresh-output "$PHASE_DIR/iteration_summary.json" -- \
         python3 /iaa-runtime/run_iaa_compute.py iteration_summary \
           --results-dir /results --label "iter$N"
   ```

   Do not run `iteration-summary` for baseline.
4. Commit evaluation with exact paths:

   ```bash
   SUMMARY_ARGS=()
   if [ "$LABEL" != baseline ]; then
     SUMMARY_ARGS=(
       --iteration-summary "$PHASE_DIR/iteration_summary.json"
       --iteration-summary-status \
         "$PHASE_DIR/iteration_summary.status.json"
     )
   fi

   "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
     "$SKILL_ROOT/scripts/commit_stage.py" \
       --results-dir "$RESULTS_DIR" --iter-label "$LABEL" --stage evaluate \
       --metrics-aggregate-csv "$METRICS" \
       --eval-status-json "$TAO_STATUS" \
       --metric-result "$EVAL_DIR/metric_result.json" \
       --metric-parse-status "$EVAL_DIR/metric_parse.status.json" \
       --eval-command-status "$EVAL_DIR/evaluate.status.json" \
       --eval-config "$PHASE_DIR/specs/eval_config.yaml" \
       --eval-config-status "$PHASE_DIR/specs/eval_config.status.json" \
       "${SUMMARY_ARGS[@]}" \
       --summary "$LABEL IAA evaluation completed"
   ```

   The commit also binds the detailed and weighted CSVs at their canonical
   paths. They need no additional user arguments because their filenames are
   fixed by TAO and by the action contract.

The commit reopens the CSV and re-derives the metric. It rejects a result from
another label/path even when its numeric value is plausible.

## Train

1. Generate the canonical train config through the selected platform's signed
   zero-GPU adapter action:

   ```bash
   ITER_DIR="$RESULTS_DIR/iter_$N"
   "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
     "$SKILL_ROOT/scripts/run_deft_action.py" prepare \
       --results-dir "$RESULTS_DIR" --image ds \
       --stage-dir "$ITER_DIR/specs" --name train_config \
       --fresh-output "$ITER_DIR/specs/train_config.yaml" -- \
       python3 /iaa-runtime/run_iaa_compute.py train_config \
         --results-dir /results --label "iter$N"
   ```

   In the default continual-dataset/continual-model mode, each iteration uses
   the accumulated mined datasets and the prior iteration's normalized model
   state. When non-continual model mode is explicitly approved, training starts
   from the configured base model each iteration. Never hand-edit the generated
   YAML.
2. Recheck occupancy for the approved GPU IDs. If the shape must change, stop
   for a revised pre-flight approval and begin a new immutable run; do not
   patch the current config.
3. Launch plain training. Mark TAO's own training status fresh; the canonical
   best checkpoint does not exist until the following publisher step:

   ```bash
   TRAIN_DIR="$ITER_DIR/train"
   BEST="$TRAIN_DIR/best/clip_best_val_t2i_mAP.pth"
   TRAIN_TAO_STATUS="$TRAIN_DIR/status.json"
   HF_ARGS=()
   if [ "${REQUIRES_HF_TOKEN:-false}" = true ]; then
     HF_ARGS=(--pass-hf-token)
   fi

   "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
     "$SKILL_ROOT/scripts/run_deft_action.py" prepare \
       --results-dir "$RESULTS_DIR" --image pyt \
       --stage-dir "$TRAIN_DIR" --name train \
       "${HF_ARGS[@]}" \
       --fresh-output "$TRAIN_TAO_STATUS" -- \
       clip train -e "/results/iter_$N/specs/train_config.yaml"
   ```

4. Only after a zero native backend exit and successful action finalization,
   select the best validation
   checkpoint and create the normalized warm-start state:

   ```bash
   "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
     "$SKILL_ROOT/scripts/run_deft_action.py" prepare \
       --results-dir "$RESULTS_DIR" --image pyt \
       --stage-dir "$TRAIN_DIR" --name publish_checkpoint \
       --fresh-output "$TRAIN_DIR/publish-checkpoint.host.status.json" -- \
       python3 /iaa-runtime/run_iaa_compute.py publish_checkpoint \
         --results-dir /results --label "iter$N"
   ```

   This must produce both:

   ```text
   iter_N/train/best/clip_best_val_t2i_mAP.pth
   iter_N/pretrained/model_state.pth
   ```

   The first is the raw evaluation checkpoint; the second is the normalized
   model-only warm-start form. Selection uses `val/t2i_mAP` when metric
   evidence exists; otherwise checkpoint metadata records
   `selection_strategy=newest_fallback`. Freshness is anchored to the first
   attempt in the bounded train-attempt lineage, and is validated before the
   canonical link/copy is created, so a retry can safely reuse a checkpoint
   produced by its earlier attempt.
5. Commit:

   ```bash
   "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
     "$SKILL_ROOT/scripts/commit_stage.py" \
       --results-dir "$RESULTS_DIR" --iter-label "iter$N" --stage train \
       --best-ckpt "$BEST" \
       --pretrained-state "$ITER_DIR/pretrained/model_state.pth" \
       --train-config "$ITER_DIR/specs/train_config.yaml" \
       --train-config-status "$ITER_DIR/specs/train_config.status.json" \
       --train-command-status "$TRAIN_DIR/train.status.json" \
       --train-tao-status-json "$TRAIN_TAO_STATUS" \
       --publish-checkpoint-status \
         "$TRAIN_DIR/publish_checkpoint.status.json" \
       --summary "iter$N TAO CLIP training and checkpoint publication completed"
   ```

The validator requires TAO's `Train finished successfully.` marker, an exact
approved `clip train` argv digest, and a selected raw checkpoint newer than
the train attempt lineage. The canonical relative symlink is allowed only when it
resolves directly to a regular checkpoint inside this iteration's `train/`;
its metadata-backed hardlink/copy fallbacks are also accepted. Symlink chains,
stale targets, and cross-iteration targets are rejected.

## Failure handling

- Nonzero native backend exit: inspect the captured action log's last meaningful error block,
  do not publish or evaluate outputs, and apply at most one documented retry.
- CUDA OOM caused by changed occupancy: retry the same approved shape once
  after those GPUs are free. Config reshaping requires a new run.
- Hydra reports an unknown key: regenerate the config; never add `workflow` or
  `automl_policy` to TAO YAML.
- A PyTorch 2.6+ NumPy dtype allowlist error: the platform action already
  mounts the bundled `sitecustomize.py`. If the error persists, hard-stop
  rather than weakening checkpoint loading.
- Eval success marker absent: treat all CSVs from that launch as partial and
  do not parse or commit them.
