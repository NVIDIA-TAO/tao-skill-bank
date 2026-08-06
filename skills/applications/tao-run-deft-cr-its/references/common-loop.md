# DEFT CR ITS Common Stages

Read this reference for every workflow mode. It covers state logging, baseline evaluation, gap analysis, annotation assembly, Cosmos Reason training/evaluation, and the final accuracy report. Mode-specific data construction remains in `mining-loop.md` and `genai-loop.md`.

Resolve `DEFT_SKILL_ROOT` to the installed `tao-run-deft-cr-its` directory. Use `run_script()` for bundled helpers when available; otherwise invoke the shown `python3 "$DEFT_SKILL_ROOT/scripts/<name>.py"` commands.

## Underlying Skills

| Stage | Registered skill | Owns |
| --- | --- | --- |
| Cosmos Reason train/evaluate | `tao-finetune-cosmos-reason` | Action command, image, credentials, model runtime, and submitted job |
| VLM BCQ gap analysis | `tao-analyze-gaps-vlm-bcq` | Gap spec generation, data-services action, image, and outputs |
| Execution | Selected `tao-run-on-*` skill | Resources, submission, logs, status, and terminal-state detection |

Read each underlying skill before invoking its stage. Keep its platform job id and monitor status and logs until the job reaches a terminal state. The presence of a partially written output file does not prove that a job completed.

For local or remote single-node Docker, run Cosmos Reason through `tao-run-on-docker`. Require `--ipc=host --ulimit memlock=-1 --ulimit stack=67108864` in the submitted container command. Kubernetes and Slurm need equivalent shared-memory and memlock resources.

## Stage State

Append an event after every completed, skipped, or failed stage:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/log_stage.py" \
  --log-path "$RUN_DIR/loop_log.jsonl" \
  --iter-label "iter_${ITER}" \
  --stage "$STAGE" \
  --status "$STATUS" \
  --summary "$SUMMARY" \
  --duration-sec "$DURATION_SEC" \
  --artifact "$ARTIFACT"
```

Use `baseline` as the iteration label for initialization and baseline events. Add `--artifact` repeatedly when a stage has multiple outputs. Before resuming, refresh state and use the returned stage:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/resume_position.py" --run-dir "$RUN_DIR"
```

Do not infer completion from directory names or rerun iteration 1 by default. `loop_log.jsonl` is the event source; each append atomically refreshes `deft_state.json`.

## Baseline Evaluation

Generate the baseline evaluation TOML:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/setup_cosmos_reason_stage.py" \
  baseline-evaluate \
  --workspace "$WORKSPACE" \
  --workflow-yaml "$WORKSPACE/specs/workflow.yaml" \
  --run-dir "$RUN_DIR"
```

Run `tao-finetune-cosmos-reason` evaluate with `$RUN_DIR/baseline/evaluate/specs/evaluate.toml`. After the submitted job exits successfully, locate its one completed result and compute authoritative binary metrics:

```bash
BASELINE_RESULTS_JSON="$(python3 "$DEFT_SKILL_ROOT/scripts/setup_cosmos_reason_stage.py" \
  find-results-json \
  --evaluate-dir "$RUN_DIR/baseline/evaluate")"

python3 "$DEFT_SKILL_ROOT/scripts/compute_bcq_accuracy_metrics.py" \
  --results-json "$BASELINE_RESULTS_JSON" \
  --output-json "$RUN_DIR/baseline/evaluate/bcq_accuracy_metrics.json"
```

The metrics parser accepts `response`/`gt` and `answer`/`ground_truth`. Unparseable predictions count as incorrect and are reported; an unparseable ground truth is a hard error. Log both `results.json` and `bcq_accuracy_metrics.json`. Iteration 1 uses `BASELINE_RESULTS_JSON` as `PREVIOUS_RESULTS_JSON`.

## Gap Analysis

For later iterations, set `PREVIOUS_RESULTS_JSON` to the prior iteration's completed evaluation result. Rewrite Cosmos Reason annotation ids to absolute KPI media paths before gap analysis:

```bash
PREDICTIONS_JSON="$RUN_DIR/iter_${ITER}/gaps/predictions.json"

python3 "$DEFT_SKILL_ROOT/scripts/prepare_gap_analysis_predictions.py" \
  --results-json "$PREVIOUS_RESULTS_JSON" \
  --annotations-json "$KPI_ANNOTATIONS_JSON" \
  --media-dir "$KPI_MEDIA_DIR" \
  --output-json "$PREDICTIONS_JSON"
```

Invoke `tao-analyze-gaps-vlm-bcq` with:

```text
predictions_json = $PREDICTIONS_JSON
videos_dir      = omitted because prediction video ids are absolute
results_dir     = $RUN_DIR/iter_<N>/gaps
output_spec     = $RUN_DIR/iter_<N>/gaps/vlm_bcq_spec.yaml
```

After the submitted job exits successfully, inspect its output:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/inspect_gap_analysis.py" \
  --gaps-jsonl "$RUN_DIR/iter_${ITER}/gaps/kpi_gaps.jsonl" \
  --status-json "$RUN_DIR/iter_${ITER}/gaps/gap_status.json"
```

The prediction helper preserves every evaluation row and changes only `video_id`, so multiple questions for one video remain separate weak samples. When `gap_status.json` reports `has_gaps: false`, log `loop_stop` with reason `no_weak_samples` and do not run either data-construction branch.

## Assemble Training Annotations

After every enabled data-construction branch has completed or been explicitly skipped, call `assemble_train_annotations.py` with its nonempty current outputs:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/assemble_train_annotations.py" \
  --current-annotations "$CURRENT_ANNOTATIONS" \
  --output-json "$RUN_DIR/iter_${ITER}/train/train_annotations.json"
```

Pass `--current-annotations` twice in `both` mode when mining and GenAI both produced annotations. Follow the mode table in `SKILL.md` to decide whether to add `--previous-annotations`. For a GenAI-only iteration 1 seed whose LLaVA `video` values may be relative, also pass `--previous-media-dir` from `train_dataset.media_dir`.

The assembler deduplicates by LLaVA `id`, preserves earlier accumulated records when ids repeat, rewrites every video path as absolute, and fails if the current branches collectively contain zero rows.

## Train Cosmos Reason

Generate the iteration train TOML:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/setup_cosmos_reason_stage.py" \
  iteration-train \
  --workspace "$WORKSPACE" \
  --workflow-yaml "$WORKSPACE/specs/workflow.yaml" \
  --run-dir "$RUN_DIR" \
  --iteration "$ITER" \
  --train-annotations "$RUN_DIR/iter_${ITER}/train/train_annotations.json" \
  --checkpoint-path "$STARTING_CHECKPOINT"
```

Run `tao-finetune-cosmos-reason` train with `$RUN_DIR/iter_<N>/train/specs/train.toml`. The stage is complete only after the submitted job exits successfully and checkpoint discovery returns a path:

```bash
TRAINED_CHECKPOINT="$(python3 "$DEFT_SKILL_ROOT/scripts/setup_cosmos_reason_stage.py" \
  latest-checkpoint \
  --train-dir "$RUN_DIR/iter_${ITER}/train")"
```

Iteration 1 uses `cosmos_reason.baseline_model_path` as `STARTING_CHECKPOINT`. Later iterations use the previous checkpoint only when `cosmos_reason.continual_model` is true; otherwise they use the baseline again.

## Evaluate The Trained Checkpoint

```bash
python3 "$DEFT_SKILL_ROOT/scripts/setup_cosmos_reason_stage.py" \
  iteration-evaluate \
  --workspace "$WORKSPACE" \
  --workflow-yaml "$WORKSPACE/specs/workflow.yaml" \
  --run-dir "$RUN_DIR" \
  --iteration "$ITER" \
  --checkpoint-path "$TRAINED_CHECKPOINT"
```

Run `tao-finetune-cosmos-reason` evaluate with `$RUN_DIR/iter_<N>/evaluate/specs/evaluate.toml`. After the submitted job exits successfully:

```bash
ITERATION_RESULTS_JSON="$(python3 "$DEFT_SKILL_ROOT/scripts/setup_cosmos_reason_stage.py" \
  find-results-json \
  --evaluate-dir "$RUN_DIR/iter_${ITER}/evaluate")"

python3 "$DEFT_SKILL_ROOT/scripts/compute_bcq_accuracy_metrics.py" \
  --results-json "$ITERATION_RESULTS_JSON" \
  --output-json "$RUN_DIR/iter_${ITER}/evaluate/bcq_accuracy_metrics.json"
```

Log both result files. Use `ITERATION_RESULTS_JSON` as the next iteration's `PREVIOUS_RESULTS_JSON`.

## Final Accuracy Report

Generate the report after reaching `max_iterations`, finding no weak samples, or stopping after a failure when baseline metrics exist:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/summarize_bcq_accuracy_metrics.py" \
  --run-dir "$RUN_DIR"
```

This writes `$RUN_DIR/bcq_accuracy_report.md` and `$RUN_DIR/bcq_accuracy_summary.json`. Include the Markdown accuracy table in the final response and log both files as `loop_stop` artifacts together with the stop reason.
