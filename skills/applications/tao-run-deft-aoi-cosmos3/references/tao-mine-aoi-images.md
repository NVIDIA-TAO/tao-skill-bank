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
- top-K, cosine floor, and router mode: frozen DEFT state;
- output root: `${RESULTS_DIR}/iterN/mining`.

Benchmark records or errors must never enter query or source inputs. Proxy
records are query targets, not trainable source samples.

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

```bash
"$PYTHON" "$SKILL_ROOT/scripts/task_mining_router.py" \
  --target-embeddings "$MINING_DIR/target_embeddings.parquet" \
  --source-embeddings "$MINING_DIR/source_embeddings.parquet" \
  --source-annotations "$MINING_ANNOTATIONS" \
  --media-root "$MEDIA_ROOT" \
  --pair-assets-dir "$RESULTS_DIR/source_pair_assets" \
  --pair-similarity "$PAIR_SIMILARITY" \
  --pair-similarity-combine "$PAIR_SIMILARITY_COMBINE" \
  --mode "$MINING_ROUTER_MODE" \
  --top-k-per-target "$TOPN" \
  --defect-detection-top-k-per-target "$DD_TOPN" \
  --min-similarity "$MIN_SIMILARITY" \
  --output "$MINING_DIR/mined_candidates.parquet" \
  --summary "$MINING_DIR/router_summary.json"
```

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

When the candidate parquet contains `atomic_sample_id`, the history tool uses
that field automatically and records `selected_identities`; legacy
single-image parquets continue to use `filepath`. The filtered parquet must
contain at least one new real atomic sample. The history ledger hard-caps
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
