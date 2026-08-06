# DEFT CR ITS Mining Branch

Read this reference only when `data_generation.mode` is `mining` or `both`. Resolve `DEFT_SKILL_ROOT` to the installed `tao-run-deft-cr-its` directory; it is not a user input.

This reference owns fixed Cosmos Embed initialization and the per-iteration mining branch. Gap analysis, annotation assembly, Cosmos Reason training/evaluation, state, and reporting remain in `SKILL.md`.

## Underlying Skills

| Stage | Registered skill | Owns |
| --- | --- | --- |
| Cosmos Embed | `tao-finetune-cosmos-embed` | Inference command, image, credentials, mounts, and submitted job |
| Nearest neighbors | `tao-mine-nearest-neighbors` | Default template, spec validation, data-services action, image, and outputs |
| Execution | Selected `tao-run-on-*` skill | Resources, submission, logs, status, and terminal-state detection |

## Fixed Embeddings

Prepare KPI and train/source inputs once before the iteration loop:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/setup_for_cosmos_embed.py" \
  --workspace "$WORKSPACE" \
  --workflow-yaml "$WORKSPACE/specs/workflow.yaml" \
  --run-dir "$RUN_DIR"
```

The KPI dataset uses `mining.embeddings_modality`; the train/source dataset always uses text and video. Setup writes lookup Parquets and any missing inference specs under:

```text
$RUN_DIR/cosmos_embed_output/kpi/
$RUN_DIR/cosmos_embed_output/train/
```

If `mining.embedding_parquets.kpi` or `.train` supplies a complete combined Parquet, setup stages it under `$RUN_DIR/embedding_parquets/<dataset>/embeddings.parquet` and writes no inference specs for that dataset. Supplied text embedding rows still require their source question files so setup can remap them to the current lookup by question content.

Run `tao-finetune-cosmos-embed` once for every generated spec, requesting exactly its `inference.num_gpus`. After each container exits, restore host write access when necessary:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/restore_docker_mount_permissions.py" \
  --path "$RUN_DIR/cosmos_embed_output/kpi" \
  --docker-image "$COSMOS_EMBED_IMAGE"
```

Convert datasets that ran inference:

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

The resulting KPI Parquet contains selected target modalities. The train/source Parquet always contains both modalities, with `filepath`, `embedding`, and `modality` columns.

## Build Iteration Target

After common gap analysis writes `$ITER_DIR/gaps/kpi_gaps.jsonl`:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/setup_iteration_mining.py" \
  --workspace "$WORKSPACE" \
  --workflow-yaml "$WORKSPACE/specs/workflow.yaml" \
  --run-dir "$RUN_DIR" \
  --iteration "$ITER" \
  --gaps-jsonl "$ITER_DIR/gaps/kpi_gaps.jsonl"
```

This writes one `$ITER_DIR/mining/target.parquet` and one `$ITER_DIR/mining/nearest_neighbors.yaml`. Text targets preserve each failed question. Video targets deduplicate failed videos. With `both` embedding modalities, both target row types are appended to the same target and queried independently.

When `mine_unique_only` is true, the generated mining spec uses `filtered_source.parquet`, excluding train/source filepaths already recorded in `$RUN_DIR/mining/mined_paths_log.parquet`.

## Mine And Record

Run `tao-mine-nearest-neighbors` once with:

```text
$ITER_DIR/mining/nearest_neighbors.yaml
```

Require the submitted job to exit successfully and produce `$ITER_DIR/mining/mined_neighbors.parquet` plus its summary.

When `mine_unique_only` is true, update the cumulative source-path log:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/record_mined_paths.py" \
  --mined-neighbors-parquet "$ITER_DIR/mining/mined_neighbors.parquet" \
  --mined-log-parquet "$RUN_DIR/mining/mined_paths_log.parquet"
```

This only removes candidates from future mining source pools. It never removes annotations already accumulated for training.

## Convert To LLaVA

```bash
python3 "$DEFT_SKILL_ROOT/scripts/build_llava_from_mining.py" \
  --mined-neighbors-parquet "$ITER_DIR/mining/mined_neighbors.parquet" \
  --train-embeddings-parquet "$RUN_DIR/embedding_parquets/train/embeddings.parquet" \
  --train-lookup-parquet "$RUN_DIR/cosmos_embed_output/train/lookup.parquet" \
  --output-llava-json "$ITER_DIR/mining/mined_train_annotations.json"
```

Text source paths select their matching train question. Video source paths select every train annotation row for that video. Output records preserve source `annotation_id`, use absolute `video_path`, and deduplicate by annotation id.

Return to `SKILL.md` after this file exists. In `both` mode, do not assemble or train until the GenAI branch has also completed or been explicitly skipped.

## Completion Criteria

| Stage | Complete when |
| --- | --- |
| Embedding setup | Each dataset has `lookup.parquet` and either complete specs or a staged combined Parquet |
| Cosmos Embed | Every generated spec exits successfully |
| Conversion | KPI and train/source combined embedding Parquets contain their required modalities |
| Target setup | `target.parquet` and `nearest_neighbors.yaml` exist |
| Mining | `mined_neighbors.parquet` and its summary exist |
| Path recording | Cumulative log exists when enabled, otherwise the stage is skipped |
| LLaVA conversion | `mined_train_annotations.json` exists and contains valid LLaVA records |
