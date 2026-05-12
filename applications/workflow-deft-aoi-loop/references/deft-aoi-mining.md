# DEFT AOI Mining — DEFT Loop Reference

Read this when the parent runs the `data_mining` stage (embed-then-mine workflow).
The underlying skill `tao-skill-bank:deft-aoi-mining` (`data/deft-aoi-mining/SKILL.md`)
owns the full docker invocation (three `nvcr.io/nvidian/iva/tao-toolkit-ds:aoi`
calls), encoder consistency requirement, output schema, and common pitfalls.
This file only covers the DEFT-loop-specific overlay: required inputs, three-step
order, output layout, and `deft_state.json` / `loop_log.jsonl` updates.

## DEFT-Loop Inputs

- `target_parquet` — absolute path from `deft_state.json` (`routing_mining_parquet` field set by the routing stage); required columns: `filepath` (and `label` if `filter_by_label=true`)
- `source_pool_parquet` — parquet of candidate images to mine against with a `filepath` column; convert from CSV up front if needed (preserve `filepath` and `label`)
- `model` — embedding model: `CLIP`, `SigLIP`, or a TAO `.pth`/`.ckpt` checkpoint; default `SigLIP`
- `model_path` — resolved by the parent during Pre-Flight as `SIGLIP_MODEL_PATH`; do not re-resolve at runtime. Default `google/siglip-base-patch16-224` (HuggingFace ID) applies only if Pre-Flight did not set a value. If a local path is set, mount it into the container; if a HuggingFace cache dir is set, mount `~/.cache/huggingface` read-only so the container can load from cache without a network call.
- `topn` — nearest neighbours per target (default `5`)
- `knn_metric` — `cosine` (default, recommended for CLIP/SigLIP), `euclidean`, or `manhattan`
- `filter_by_label` — `true` or `false` (default `false`); requires `label` in both embedding parquets

If `routing_mining_parquet` is absent from `deft_state.json` or the file does not exist on disk, stop and return failure without running any docker steps.

## Three-Step Execution Order

1. **Embed targets** (`embedding image_embeddings … input_parquet=<target_parquet>`) → `target_embeddings.parquet`
2. **Embed source pool** (`embedding image_embeddings … input_parquet=<source_pool_parquet>`) → `source_embeddings.parquet`; use the **identical** `model` and `model_path` as Step 1
3. **Mine nearest neighbours** (`tmm nearest_neighbors …`) → `mined.parquet` + `mining_summary.txt`

All three steps use `nvcr.io/nvidian/iva/tao-toolkit-ds:aoi`. Mount the workspace root at an identical path inside the container (`-v $WORKSPACE:$WORKSPACE`) so absolute paths in parquet args resolve the same on both sides.

## Output Directory

`results/<baseline|iter${N}>/mining_results/<timestamp>/`

Required files:
- `mined.parquet` — unique mined source filepaths (columns: `filepath`)
- `mining_summary.txt` — query count, neighbour count, duplicates removed, kept/dropped pairs
- `target_embeddings.parquet` — Step 1 output (reusable across future mining runs against the same targets)
- `source_embeddings.parquet` — Step 2 output (reusable against the same source pool)

## Output to deft_state.json

```python
state["baseline" | f"iter{N}"]["mining_mined_parquet"] = "<abs_path>/mined.parquet"
state["baseline" | f"iter{N}"]["mining_mined_count"]   = <int>   # rows in mined.parquet
```

## Log Stage

```bash
python3 <skill_root>/scripts/log_stage.py \
    --log-path results/loop_log.jsonl \
    --iter-label <baseline|iter${N}> \
    --stage data_mining --status ok \
    --summary "Mining (VCN): mined=N_mined source images for N_targets targets"
```
