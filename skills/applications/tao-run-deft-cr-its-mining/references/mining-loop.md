# DEFT CR ITS Mining Loop Reference

Use this reference when running the iteration loop after the DEFT workspace and `workflow.yaml` have been validated.

Resolve `DEFT_SKILL_ROOT` to the absolute directory containing the installed `tao-run-deft-cr-its-mining/SKILL.md`. The plugin runtime or agent resolves it; never ask the user for this path. Use `run_script()` for bundled helpers when available, otherwise use the direct `python3 "$DEFT_SKILL_ROOT/scripts/<name>.py"` commands below. The user's working directory does not need to be the plugin or repository root.

## Stage Skills

Read each underlying skill before invoking its stage. The workflow layers iteration paths and completion rules on top; the underlying skill owns action execution.

| Stage | Registered skill | Owns |
| --- | --- | --- |
| Cosmos Reason train/evaluate | `tao-finetune-cosmos-reason` | Action command, image, credentials, model runtime, and submitted job. |
| Cosmos Embed inference | `tao-finetune-cosmos-embed` | Inference command, image, credentials, mounts, and submitted job. |
| VLM BCQ gap analysis | `tao-analyze-gaps-vlm-bcq` | Gap spec generation, data-services action, image, and outputs. |
| Nearest-neighbor mining | `tao-mine-nearest-neighbors` | Default template, spec validation, data-services action, image, and outputs. |
| Job execution | Selected `tao-run-on-*` platform skill | Resource allocation, submission, status, logs, and terminal-state detection. |

## Initialization

If the user does not provide custom Cosmos Reason or Cosmos Embed templates, copy them from `$DEFT_SKILL_ROOT/assets/` into `<deft_workspace>/specs/`. Use `tao-mine-nearest-neighbors` to copy its bundled default template when the user does not provide a mining template. Point `workflow.yaml` at the workspace copies.

Initialize a run once:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/init_deft_cr_mining_state.py" \
  --workspace "$WORKSPACE" \
  --workflow-yaml "$WORKSPACE/specs/workflow.yaml"
```

Use the printed `run_dir` as `RUN_DIR` for every later command. If `run.name` is `null`, never let another setup script derive a new run directory; pass `--run-dir "$RUN_DIR"`.
An orchestrator that resolves the run directory before initialization may also pass that absolute path to this command with `--run-dir "$RUN_DIR"`.

`--force` rewrites `deft_state.json` and the workflow snapshot but does not remove the existing loop log or stage artifacts. Use it only to repair the snapshot for the same run. Start a clean run with a new `run.name` instead.

Append one stage event after each completed, skipped, or failed stage:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/log_stage.py" \
  --log-path "$RUN_DIR/loop_log.jsonl" \
  --iter-label "iter_${ITER}" \
  --stage mine_nearest_neighbors \
  --status ok \
  --summary "mined nearest train neighbors" \
  --duration-sec "$DURATION_SEC" \
  --artifact "$RUN_DIR/iter_${ITER}/mining/mined_neighbors.parquet"
```

Each append rebuilds `deft_state.json` atomically from valid log events. A truncated or otherwise malformed log line is ignored without blocking later appends. Before resuming, refresh state and ask the helper for the next unfinished stage:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/resume_position.py" \
  --run-dir "$RUN_DIR"
```

Use the reported stage; do not infer resume position from directory names or rerun iteration 1 by default.

## Cosmos Reason Platform Runtime

Run every Cosmos Reason train/evaluate action through `tao-finetune-cosmos-reason` and the selected platform skill. For local or remote single-node Docker, select `tao-run-on-docker`; do not construct a competing unmanaged Docker launch. Require its submitted container command to include `--ipc=host --ulimit memlock=-1 --ulimit stack=67108864`. Kubernetes or Slurm must provide equivalent shared-memory and memlock resources. Keep the platform job id and use the platform's status and log operations until the job reaches a terminal state.

## Baseline Evaluation

Generate the baseline evaluate TOML:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/setup_cosmos_reason_stage.py" \
  baseline-evaluate \
  --workspace "$WORKSPACE" \
  --workflow-yaml "$WORKSPACE/specs/workflow.yaml" \
  --run-dir "$RUN_DIR"
```

Use `tao-finetune-cosmos-reason` evaluate with `$RUN_DIR/baseline/evaluate/specs/evaluate.toml`. After the job exits successfully, locate its one result and compute the authoritative binary metrics:

```bash
BASELINE_RESULTS_JSON="$(python3 "$DEFT_SKILL_ROOT/scripts/setup_cosmos_reason_stage.py" \
  find-results-json \
  --evaluate-dir "$RUN_DIR/baseline/evaluate")"

python3 "$DEFT_SKILL_ROOT/scripts/compute_bcq_accuracy_metrics.py" \
  --results-json "$BASELINE_RESULTS_JSON" \
  --output-json "$RUN_DIR/baseline/evaluate/bcq_accuracy_metrics.json"
```

The metrics script reads `response`/`gt` and also accepts `answer`/`ground_truth`. It extracts `yes` and `no` defensively from capitalization, punctuation, and short free-form answers. An unparseable prediction is reported and counted as incorrect; an unparseable ground truth is a hard error. Report the printed accuracy, balanced accuracy, false-positive count, false-negative count, and unparseable count in the baseline stage update. Log both `results.json` and `bcq_accuracy_metrics.json` as `baseline_evaluate` artifacts. The stage is not complete until the metrics file exists.

## Mining Embeddings

Prepare fixed KPI/train Cosmos Embed inputs once:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/setup_for_cosmos_embed.py" \
  --workspace "$WORKSPACE" \
  --workflow-yaml "$WORKSPACE/specs/workflow.yaml" \
  --run-dir "$RUN_DIR"
```

Run `tao-finetune-cosmos-embed` inference for every generated spec under:

```text
$RUN_DIR/cosmos_embed_output/kpi/specs/
$RUN_DIR/cosmos_embed_output/train/specs/
```

Only run specs that exist. KPI specs follow `mining.embeddings_modality`; train specs always cover text and video. If `mining.embedding_parquets.kpi` or `.train` supplied a complete combined dataset Parquet, there are no specs for that dataset and its mining-ready output is already staged under `$RUN_DIR/embedding_parquets/<dataset>/embeddings.parquet`. Staging remaps text identifiers to the current run's lookup by reading the source question files, so those files must remain readable; video identifiers and embedding vectors are unchanged.

Read `inference.num_gpus` from each generated spec and request exactly that many GPUs from the selected platform. The bundled template defaults to 8 GPUs. The setup script prints the count for every generated modality; stop before launch if the platform cannot satisfy it.

Keep the exact container image selected by `tao-finetune-cosmos-embed` as `COSMOS_EMBED_IMAGE`. After each Cosmos Embed container exits, restore host write access if needed with that same image:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/restore_docker_mount_permissions.py" \
  --path "$RUN_DIR/cosmos_embed_output/kpi" \
  --docker-image "$COSMOS_EMBED_IMAGE"
```

Convert completed outputs only for datasets that generated inference specs:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/cosmos_embed_outputs_to_parquet.py" \
  --output-dir "$RUN_DIR/cosmos_embed_output/kpi" \
  --parquet-dir "$RUN_DIR/embedding_parquets/kpi" \
  --embedding-modality "$EMBEDDING_MODALITY"

python3 "$DEFT_SKILL_ROOT/scripts/cosmos_embed_outputs_to_parquet.py" \
  --output-dir "$RUN_DIR/cosmos_embed_output/train" \
  --parquet-dir "$RUN_DIR/embedding_parquets/train" \
  --embedding-modality both
```

These commands write `$RUN_DIR/embedding_parquets/kpi/embeddings.parquet` and `$RUN_DIR/embedding_parquets/train/embeddings.parquet`. Skip the corresponding command when setup already staged a supplied combined Parquet at that path. The KPI file contains the selected modality or modalities. The train file always contains both, and each row records its `modality` alongside `filepath` and `embedding`.

For text outputs, conversion matches each Cosmos Embed metadata `text` value to the corresponding lookup question. Repeated questions consume their lookup rows in occurrence order. `npy_row` selects only the embedding vector; it is not treated as a lookup-row index. Conversion fails if Cosmos Embed returns missing, extra, or unmatched text occurrences.

## Iteration Loop

Run iterations `1..run.max_iterations`. Before each stage, run `resume_position.py`; `loop_log.jsonl` is the event source and `deft_state.json` is its refreshed snapshot.

1. **Gap analysis**: for iteration 1, source predictions come from the baseline `results.json`; for later iterations, they come from the previous iteration's `results.json`. Cosmos Reason may write a LLaVA annotation id in `video_id`, which is not necessarily the annotation's media path and is ambiguous for downstream video joins when one video has multiple questions. Rewrite each result through the fixed KPI annotations before generating the gap-analysis spec:

```bash
PREDICTIONS_JSON="$RUN_DIR/iter_${ITER}/gaps/predictions.json"

python3 "$DEFT_SKILL_ROOT/scripts/prepare_gap_analysis_predictions.py" \
  --results-json "$PREVIOUS_RESULTS_JSON" \
  --annotations-json "$KPI_ANNOTATIONS_JSON" \
  --media-dir "$KPI_MEDIA_DIR" \
  --output-json "$PREDICTIONS_JSON"
```

Invoke `tao-analyze-gaps-vlm-bcq` with `predictions_json=$PREDICTIONS_JSON`, no `videos_dir` because the prepared ids are absolute paths, `results_dir=$RUN_DIR/iter_${ITER}/gaps`, and `output_spec=$RUN_DIR/iter_${ITER}/gaps/vlm_bcq_spec.yaml`. Let that skill use its bundled helper to generate the spec and run the action. After its submitted job exits successfully, inspect the output:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/inspect_gap_analysis.py" \
  --gaps-jsonl "$RUN_DIR/iter_${ITER}/gaps/kpi_gaps.jsonl" \
  --status-json "$RUN_DIR/iter_${ITER}/gaps/gap_status.json"
```

`KPI_ANNOTATIONS_JSON` is `kpi_dataset.annotations_path` from `workflow.yaml`. The preparation script preserves every prediction row and changes only `video_id`; multiple annotation ids may therefore resolve to the same video path while retaining their separate questions. The container must exit successfully before inspection. A missing or empty `kpi_gaps.jsonl` after a successful container run means there are no weak samples. In that case, log `loop_stop` with reason `no_weak_samples`; otherwise continue with mining.

2. **Build mining targets**:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/setup_iteration_mining.py" \
  --workspace "$WORKSPACE" \
  --workflow-yaml "$WORKSPACE/specs/workflow.yaml" \
  --run-dir "$RUN_DIR" \
  --iteration "$ITER" \
  --gaps-jsonl "$RUN_DIR/iter_${ITER}/gaps/kpi_gaps.jsonl"
```

The command writes one `$RUN_DIR/iter_<N>/mining/target.parquet`. Gap-analysis `video_id` remains the external resolved media path; internally it joins to lookup `video_path`. Text target rows are failed `(video_path, question)` embeddings, and video target rows are unique failed-video embeddings. With `embeddings_modality: both`, both row types are appended to the same target. When `mining.mine_unique_only` is true or omitted, the single generated spec points at `$RUN_DIR/iter_<N>/mining/filtered_source.parquet`, which excludes source filepaths already recorded in `$RUN_DIR/mining/mined_paths_log.parquet`.

3. **Mine nearest neighbors**: use `tao-mine-nearest-neighbors` once with `$RUN_DIR/iter_<N>/mining/nearest_neighbors.yaml`. Every target row is queried independently against the combined text/video train source. The stage is complete when `$RUN_DIR/iter_<N>/mining/mined_neighbors.parquet` and its mining summary exist.

4. **Record mined paths**: when `mining.mine_unique_only` is true or omitted, update the cumulative mined-path log after the nearest-neighbor run completes.

```bash
python3 "$DEFT_SKILL_ROOT/scripts/record_mined_paths.py" \
  --mined-neighbors-parquet "$RUN_DIR/iter_${ITER}/mining/mined_neighbors.parquet" \
  --mined-log-parquet "$RUN_DIR/mining/mined_paths_log.parquet"
```

Future iterations filter their train/source embedding pool with the cumulative log. This does not remove records from the already assembled training annotation JSON.

When `mining.mine_unique_only` is false, do not run the command; append a `record_mined_paths` event with `status=skipped` so resume can advance unambiguously.

5. **Convert mined rows to LLaVA**:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/build_llava_from_mining.py" \
  --mined-neighbors-parquet "$RUN_DIR/iter_${ITER}/mining/mined_neighbors.parquet" \
  --train-embeddings-parquet "$RUN_DIR/embedding_parquets/train/embeddings.parquet" \
  --train-lookup-parquet "$RUN_DIR/cosmos_embed_output/train/lookup.parquet" \
  --output-llava-json "$RUN_DIR/iter_${ITER}/mining/mined_train_annotations.json"
```

Nearest-neighbor output contains only selected source filepaths. The conversion script joins those paths back to the train embeddings parquet to recover each source row's modality. A text source path selects its corresponding train question; a video source path selects every train annotation row whose `video_path` matches. Output records preserve the source `annotation_id` as their LLaVA `id` and are deduplicated by that stable id.

6. **Assemble training annotations**: iteration 1 uses mined annotations only. Do not seed the first iteration with `train_dataset.annotations_path`; that file is the mining source pool, not the first iteration's previous-training file. For later iterations, add `--previous-annotations "$RUN_DIR/iter_<N-1>/train/train_annotations.json"`.

```bash
python3 "$DEFT_SKILL_ROOT/scripts/assemble_train_annotations.py" \
  --mined-annotations "$RUN_DIR/iter_${ITER}/mining/mined_train_annotations.json" \
  --output-json "$RUN_DIR/iter_${ITER}/train/train_annotations.json"
```

For iteration 2 and later, include the previous iteration's merged annotations with `--previous-annotations`. The script dedupes by LLaVA `id`.

7. **Train Cosmos Reason**:

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

Use `tao-finetune-cosmos-reason` train with `$RUN_DIR/iter_<N>/train/specs/train.toml`. The stage is complete when the job exits successfully and the latest checkpoint command returns a path:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/setup_cosmos_reason_stage.py" \
  latest-checkpoint \
  --train-dir "$RUN_DIR/iter_${ITER}/train"
```

Iteration 1 starts from `cosmos_reason.baseline_model_path`. Later iterations use the previous iteration's checkpoint only when `cosmos_reason.continual_model: true`; otherwise they start from the baseline checkpoint again.

8. **Evaluate trained checkpoint**:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/setup_cosmos_reason_stage.py" \
  iteration-evaluate \
  --workspace "$WORKSPACE" \
  --workflow-yaml "$WORKSPACE/specs/workflow.yaml" \
  --run-dir "$RUN_DIR" \
  --iteration "$ITER" \
  --checkpoint-path "$TRAINED_CHECKPOINT"
```

Use `tao-finetune-cosmos-reason` evaluate with `$RUN_DIR/iter_<N>/evaluate/specs/evaluate.toml`. After the job exits successfully, locate the result and compute that iteration's metrics:

```bash
ITERATION_RESULTS_JSON="$(python3 "$DEFT_SKILL_ROOT/scripts/setup_cosmos_reason_stage.py" \
  find-results-json \
  --evaluate-dir "$RUN_DIR/iter_${ITER}/evaluate")"

python3 "$DEFT_SKILL_ROOT/scripts/compute_bcq_accuracy_metrics.py" \
  --results-json "$ITERATION_RESULTS_JSON" \
  --output-json "$RUN_DIR/iter_${ITER}/evaluate/bcq_accuracy_metrics.json"
```

Report the printed accuracy, balanced accuracy, false-positive count, false-negative count, and unparseable count in the iteration update. Log both files as `evaluate` artifacts. The evaluate stage is not complete until `bcq_accuracy_metrics.json` exists. Use `ITERATION_RESULTS_JSON` as the next iteration's `PREVIOUS_RESULTS_JSON`.

## Final Accuracy Report

After the loop stops because it reached `max_iterations` or found no weak samples, generate the baseline/iteration report. Also generate it after a failed run when baseline metrics are available, so completed evaluations are not lost from the final account.

```bash
python3 "$DEFT_SKILL_ROOT/scripts/summarize_bcq_accuracy_metrics.py" \
  --run-dir "$RUN_DIR"
```

This writes `$RUN_DIR/bcq_accuracy_report.md` and `$RUN_DIR/bcq_accuracy_summary.json`. Read `bcq_accuracy_report.md` and include its full accuracy table in the agent's final response; providing only the artifact path is not sufficient. Log both report files as `loop_stop` artifacts together with the stop reason.

## Completion Criteria

| Stage | Complete when |
| --- | --- |
| `validate_workflow` | `verify_workflow_yaml.py` exits successfully. |
| `init_state` | `$RUN_DIR/workflow.yaml` and `$RUN_DIR/deft_state.json` exist. |
| `baseline_evaluate` | The evaluate job exits successfully, exactly one baseline `results.json` is found, and `baseline/evaluate/bcq_accuracy_metrics.json` exists. |
| `setup_embeddings` | Each dataset has a lookup parquet and either all required Cosmos Embed specs or a staged combined embedding Parquet. |
| `cosmos_embed` | Every generated Cosmos Embed inference spec has completed successfully through the underlying skill. |
| `convert_embeddings` | `embedding_parquets/{kpi,train}/embeddings.parquet` exist with the required modalities. |
| `gap_analysis` | `$RUN_DIR/iter_<N>/gaps/predictions.json` exists, the container exits successfully, and `gap_status.json` records the weak-sample count. |
| `build_mining_target` | Requested target parquets and nearest-neighbor specs exist. |
| `mine_nearest_neighbors` | One mined-neighbor parquet and mining summary exist. |
| `record_mined_paths` | The cumulative log exists when enabled, or a skipped event is logged when disabled. |
| `build_llava_from_mining` | `$RUN_DIR/iter_<N>/mining/mined_train_annotations.json` exists. |
| `assemble_train_annotations` | `$RUN_DIR/iter_<N>/train/train_annotations.json` exists and is valid JSON. |
| `train` | Cosmos Reason training job exits successfully and `latest-checkpoint` returns a checkpoint path. |
| `evaluate` | Cosmos Reason evaluation job exits successfully, exactly one iteration `results.json` is found, and its `bcq_accuracy_metrics.json` exists. |
| `loop_stop` | Stop reason is logged; the run-level Markdown and JSON accuracy reports cover the baseline and every completed iteration. |
