# Coverage-stratified candidate selector

`config.mining.candidate_selector` accepts `nearest_neighbor` (the unchanged
default) or `coverage_stratified_hardness_v1`. The launch phrase **“use
coverage_stratified_hardness_v1”** maps exactly to
`init_deft_state.py --candidate-selector coverage_stratified_hardness_v1`.
Proxy failures then have role `quota_statistics_only`; Proxy embeddings never
rank or choose Mining parents.

## Inventory and eligibility

Build `${RESULTS_DIR}/coverage_inventory.parquet` once from the sealed Mining
annotations and the existing normalized SigLIP whole-image embedding cache.
The parquet schema metadata records both input SHA-256 values and its payload
hash. A later round reuses the cache only when those hashes match; otherwise it
atomically rebuilds it. Exact duplicates, Proxy/Benchmark leakage, split and
lineage exclusions remain upstream gates. `source_group_id` is the selection
atom for an original board/image and its derivatives. Selection de-duplicates
that group and emits its canonical parent record, never a crop or derivative.
Ordered reference pairs remain one atomic parent.

Detection inventory cells are:

```text
task × source_dataset × canonical_phenotype × log_bbox_area_quartile
     × local_contrast_quartile × gt_count_bin
```

where `gt_count_bin` is `0`, `1`, `2-3`, or `4+`. Classification cells use
`task × source_dataset × label`. Quartiles are deterministic ranks over the
inventory. Every cell retains its normalized embedding and visual-cluster ID.

## Quotas, coverage, and hardness

For each Proxy capability cell, compute `error_mass = FN + 0.5 × FP` and
shrink it toward the task mean. Cell quota weights are proportional to
`sqrt(error_mass + epsilon)`. Query-vector cosine is deliberately absent from
this computation and from parent ranking.

Within a task, capped water-filling gives each source at most 35% of selected
rows and every source with enough eligible supply at least 10%. When the
constraints cannot fill the request, reduce the usable budget and record the
reason; do not backfill from a dominant source. Within every cell,
k-center/farthest-first cycles through visual clusters before revisiting one,
with one row per parent and the launch-recorded per-cluster cap.

The launch-recorded `hardness_schedule` contains exactly five rounds:

| Round | Coverage positive | Hard positive | Coverage negative | FP-hard negative |
| --- | ---: | ---: | ---: | ---: |
| 1 | 45% | 15% | 25% | 15% |
| 2 | 40% | 20% | 20% | 20% |
| 3 | 35% | 25% | 15% | 25% |
| 4 | 30% | 30% | 15% | 25% |
| 5 | 30% | 30% | 15% | 25% |

Hard positives come from higher-error/lower-recall cells. FP-hard negatives
come only from empty/no-change visual clusters represented among Proxy false
positives. Both retain the same parent, source, and cluster caps.

## Calibration and gates

Single-image Defect Detection empty rows target 41.8% of that task and
reference Defect Detection no-change rows target 42.4%; both are clipped to
38–45%. Few-box rows remain a 15–25% stratum of non-empty positives. Across
the whole materialized iteration, empty plus no-change may not exceed 25%.
These rows are unique and have `rep=1`; no extra global-512 calibration quota
is added.

Each round writes `coverage_selector_manifest.json`, validated by
`references/coverage_selector_manifest.schema.json`, with at least
`available_unique`, `target_share`, `realized_share`, `source_share`,
`cell_share`, `unique_parent_ratio`, `rep`, `effective_exposure`, and
`shortage_reason`. `effective_exposure = rep × epochs` must be at most 8.
`task_mining_router.py` exits nonzero and `commit_stage.py` refuses the
`data_mining` commit when any target-minus-realized share gap exceeds five
percentage points or another selector gate fails.

## CPU selector step

After source/target embedding artifacts exist, use the complete Proxy
`gap_candidates.parquet` for quota statistics:

```bash
"$PYTHON" "$SKILL_ROOT/scripts/task_mining_router.py" \
  --target-embeddings "$MINING_DIR/target_embeddings.parquet" \
  --source-embeddings "$MINING_DIR/source_embeddings.parquet" \
  --source-annotations "$MINING_ANNOTATIONS" \
  --media-root "$MEDIA_ROOT" \
  --candidate-selector coverage_stratified_hardness_v1 \
  --proxy-errors "$RCCA_DIR/gap_candidates.parquet" \
  --round-index "$ITERATION" --epochs "$EPOCHS" \
  --iteration-budget "$MAX_TRAINING_ROWS" \
  --inventory-cache "$RESULTS_DIR/coverage_inventory.parquet" \
  --selector-manifest "$MINING_DIR/coverage_selector_manifest.json" \
  --mode "$MINING_ROUTER_MODE" --top-k-per-target "$TOPN" \
  --output "$MINING_DIR/mined_candidates.parquet" \
  --summary "$MINING_DIR/router_summary.json"
```

Pass the resulting manifest to the mining commit with
`--coverage-selector-manifest`. The nearest-neighbor path neither requires nor
creates this manifest.
