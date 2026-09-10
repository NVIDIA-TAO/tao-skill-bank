# Cosmos3 AOI real-pair Mining

Read `skills/data/tao-mine-aoi-images/SKILL.md` before launch. Embed unique
Proxy atomic samples and the recorded Mining atomic source pool with the same encoder.
Dispatch each GPU invocation through the selected platform's four verbs and
track it with its own job-record.

## Inputs and isolation

- query targets: selected Proxy RCCA gaps only, with each reference pair
  represented by either one canvas asset or two full-resolution component
  embeddings according to the frozen similarity mode;
- source pool: canonical `annotations/mining.jsonl` and its media, materialized
  by `build_mining_source_pool.py` at `atomic_sample_id` granularity (canvas
  mode additionally requires `--pair-assets-dir ...`);
- top-K, cosine floor, router mode, candidate selector, and five-round
  hardness schedule: frozen DEFT state;
- output root: `${RESULTS_DIR}/iterN/mining`.

Benchmark records or errors must never enter query or source inputs. Proxy
records are query targets, not trainable source samples.

Before enabling bbox-driven Defect Detection mining, invoke
`$PYTHON "$SKILL_ROOT/scripts/audit_bbox_retrieval.py"` to run the proxy-only
`audit_bbox_retrieval_v1` gate. Its `prepare` phase reproduces the authoritative
Hungarian IoU `> 0.5` false-negative assignment, freezes the explicit NVPAW
label-to-phenotype map, and materializes EXIF-aware 1.5x/3.0 square RGB-224
crops with clipped visible-region mean padding; its `compute` phase compares
those arms and their multiscale maximum against the existing whole-image
SigLIP cache with parent-de-duplicated rankings, paired query bootstrap, all
required slices, lineage checks, and blinded review grids. Seal the report
gates before `prepare`, never pass Benchmark data to the tool (there is no
Benchmark CLI argument), and consume only the emitted full-image parent rows;
the audit crops are embedding intermediates and must never become training
samples.

With `candidate_selector=coverage_stratified_hardness_v1`, Proxy records are
quota statistics rather than similarity queries. Follow
`references/coverage-stratified-selector.md`: reuse the source embedding
artifact to build one hash-bound parent inventory, pass the complete Proxy
`gap_candidates.parquet`, and require the selector manifest before commit.
The default `nearest_neighbor` path below is unchanged.

`config.mining.pair_similarity` selects `canvas` (the backward-compatible
default) or `two_vector`. Canvas uses one 1024x512 golden/test composite and
therefore halves each board's effective input resolution; two-vector uses the
same full-resolution single-image encoder separately for golden and test,
deduplicates shared golden inputs, and reuses test embeddings when their
single-image cache key matches. The router combines the two cosine scores with
`config.mining.pair_similarity_combine = mean | min` (default `mean`), writes
`sim_golden`, `sim_test`, and `sim_pair` on every candidate, and reports
same-golden-path-or-board-prefix hits for each reference query. Both modes keep
`atomic_sample_id`, history, leakage exclusion, budgets, and canonical
golden-then-test materialization pair-atomic; two-vector never synthesizes a
golden/test combination absent from Mining annotations. Pass the same
`--pair-similarity` value to `build_mining_source_pool.py`,
`route_selected_gaps.py`, and `task_mining_router.py` so the embedding inputs
and router interpretation remain identical; pass the recorded combine value
only to the router. When migrating an existing canvas cache, use
`merge_source_embedding_cache.py --allow-cached-superset` so matching
full-resolution test rows are retained while obsolete canvas rows are ignored
and counted.

## Task-aware routing

The target embedding parquet carries atomic sample IDs, ordered original image
paths, and selected task types emitted by `route_selected_gaps.py`. The Mining
annotation provides the available task types for every real source atomic
sample. Pass the same source pair-asset root to the router so it resolves
embedding paths back to exact ordered pairs.

Routing (`_source_catalog`) and emission (`_source_index`) use
`lookup_embedding_filepath`: single-image paths or deterministic pair-asset
keys, without opening images, checking their contents, or creating canvases.
Ordered atomic IDs and unambiguous hash-name aliases still match cached vectors
from another root. Rendering remains in `route_selected_gaps.materialize_embedding_inputs`
and `build_mining_source_pool._embedding_inputs`. For
`render_iteration_mining_runner.py --request request.json --output plan.json
--runner-output iteration_mining_runner.py`, set independent request paths
`source_pair_assets_dir` (the source pool's recorded cache) and
`query_pair_assets_dir` (the run's query output cache). Optional
`mining_commands` maps stage names to argv lists, run in this order:
`source_inputs`, `source_embeddings`, `query_inputs`, `query_embeddings`,
`routing`, `history`, `emission`, then the existing selector/assembler handoff;
omit stages already complete or unused. Use `build_mining_source_pool.py` for
`source_inputs`, `route_selected_gaps.py` for `query_inputs`,
`task_mining_router.py` for `routing`, and `emit_mined_sharegpt.py` for `emission`.
The renderer owns `--pair-assets-dir`: source inputs/routing/emission get the
source root, query inputs get the query root, and a missing root or caller
override is rejected. Other stage arguments remain explicit; emission must
write an intermediate, never cumulative Train. Each generated stage prints
flushed JSON `stage_start`/`stage_end` events with timestamps, elapsed seconds
on completion, and the exit code; failures stop later stages. Generation only
writes the plan/runner: execute it solely through an approved platform job.

```bash
"$PYTHON" "$SKILL_ROOT/scripts/task_mining_router.py" \
  --target-embeddings "$MINING_DIR/target_embeddings.parquet" \
  --source-embeddings "$MINING_DIR/source_embeddings.parquet" \
  --source-annotations "$MINING_ANNOTATIONS" \
  --media-root "$MEDIA_ROOT" \
  --pair-assets-dir "$RESULTS_DIR/source_pair_assets" \
  --pair-similarity "$PAIR_SIMILARITY" \
  --pair-similarity-combine "$PAIR_SIMILARITY_COMBINE" \
  --candidate-selector "$CANDIDATE_SELECTOR" \
  --mode "$MINING_ROUTER_MODE" \
  --top-k-per-target "$TOPN" \
  --defect-detection-top-k-per-target "$DD_TOPN" \
  --min-similarity "$MIN_SIMILARITY" \
  --output "$MINING_DIR/mined_candidates.parquet" \
  --summary "$MINING_DIR/router_summary.json"
```

For the coverage selector, also pass `--proxy-errors
"$RCCA_DIR/gap_candidates.parquet" --round-index "$ITERATION" --epochs
"$EPOCHS" --iteration-budget "$MAX_TRAINING_ROWS" --inventory-cache
"$RESULTS_DIR/coverage_inventory.parquet" --selector-manifest
"$MINING_DIR/coverage_selector_manifest.json"`. Proxy similarity never ranks
parents on this path.

`image_only` applies global cosine top-K, `task_strict` requires an exact task
match, and `task_then_fallback` fills strict shortfalls from the global pool.
The optional Defect Detection override is applied independently per exact task
route; all other tasks keep `--top-k-per-target`.
All modes use the same deterministic router and record rank, cosine, task
types, query IDs, and route tier. A zero-row result is a hard stop.

## History-aware selection

Remove atomic sample IDs selected by previous iterations while preserving the
pre-history candidate parquet:

```bash
"$PYTHON" "$BANK_ROOT/skills/data/tao-mine-aoi-images/scripts/filter_mined_history.py" \
  --candidate-parquet "$MINING_DIR/mined_candidates.parquet" \
  --output-parquet "$MINING_DIR/mined_filtered.parquet" \
  --history-file "$RESULTS_DIR/mining_history.json" \
  --summary "$MINING_DIR/mining_history_summary.json" \
  --iteration "$ITERATION" \
  --topn "$TOPN" \
  --pool-size "$UNIQUE_MINING_TARGETS" \
  --max-cumulative-fraction "$MINING_POOL_FRACTION_CAP"
```

When the candidate parquet contains `source_group_id`, the history tool uses
that parent/derivative group automatically; otherwise it uses
`atomic_sample_id`, then the legacy `filepath`. It records
`selected_identities`, so a derivative cannot re-enter a later round. The
filtered parquet must contain at least one new real atomic sample. The history ledger hard-caps
cumulative unique atomic-sample selection against the sealed pool size and
fraction. An all-duplicate or budget-exhausted result is a hard stop; never
silently exceed the recorded Mining budget.

## Handoff

Commit the final filtered parquet, the pre-history candidate parquet, router
summary, history ledger and summary, both embedding parquets, and the exact
positive row count. `emit_mined_sharegpt.py --pair-assets-dir ...` then
recovers canonical JSONL messages and both ordered original images for each
selected reference pair; it never matches or recombines individual sides.

```bash
"$PYTHON" "$SKILL_ROOT/scripts/commit_stage.py" \
  --results-dir "$RESULTS_DIR" --iter-label "iter$ITERATION" \
  --stage data_mining \
  --mining-parquet "$MINING_DIR/mined_filtered.parquet" \
  --mining-candidates "$MINING_DIR/mined_candidates.parquet" \
  --mining-summary "$MINING_DIR/router_summary.json" \
  --mining-history "$RESULTS_DIR/mining_history.json" \
  --mining-history-summary "$MINING_DIR/mining_history_summary.json" \
  --mining-target-embeddings "$MINING_DIR/target_embeddings.parquet" \
  --mining-source-embeddings "$MINING_DIR/source_embeddings.parquet" \
  --mining-count <positive-int> \
  --duration-sec <measured-positive-seconds> \
  --summary "task-aware history-filtered Mining selected novel real records"
```

For `coverage_stratified_hardness_v1`, the same commit must additionally pass
`--coverage-selector-manifest
"$MINING_DIR/coverage_selector_manifest.json"`. The commit re-runs the
training-eligibility gate and refuses a manifest with a share gap above five
percentage points.
