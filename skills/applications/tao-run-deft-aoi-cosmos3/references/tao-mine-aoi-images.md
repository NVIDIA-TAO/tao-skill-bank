# Cosmos3 AOI real-pair Mining

Read `skills/data/tao-mine-aoi-images/SKILL.md` before launch. Embed unique
Proxy target images and the recorded Mining source pool with the same encoder.
Dispatch each GPU invocation through the selected platform's four verbs and
track it with its own job-record.

## Inputs and isolation

- query targets: selected Proxy RCCA gaps only;
- source pool: canonical `annotations/mining.jsonl` and its media;
- top-K, cosine floor, and router mode: frozen DEFT state;
- output root: `${RESULTS_DIR}/iterN/mining`.

Benchmark records or errors must never enter query or source inputs. Proxy
records are query targets, not trainable source samples.

## Task-aware routing

The target embedding parquet carries the physical target IDs and selected task
types emitted by `route_selected_gaps.py`. The Mining annotation provides the
available task types for every real source target.

```bash
"$PYTHON" "$SKILL_ROOT/scripts/task_mining_router.py" \
  --target-embeddings "$MINING_DIR/target_embeddings.parquet" \
  --source-embeddings "$MINING_DIR/source_embeddings.parquet" \
  --source-annotations "$MINING_ANNOTATIONS" \
  --media-root "$MEDIA_ROOT" \
  --mode "$MINING_ROUTER_MODE" \
  --top-k-per-target "$TOPN" \
  --min-similarity "$MIN_SIMILARITY" \
  --output "$MINING_DIR/mined_candidates.parquet" \
  --summary "$MINING_DIR/router_summary.json"
```

`image_only` applies global cosine top-K, `task_strict` requires an exact task
match, and `task_then_fallback` fills strict shortfalls from the global pool.
All modes use the same deterministic router and record rank, cosine, task
types, query IDs, and route tier. A zero-row result is a hard stop.

## History-aware selection

Remove filepaths selected by previous iterations while preserving the
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

The filtered parquet must contain at least one new real target. The history
ledger hard-caps cumulative unique target-image selection against the sealed
pool size and fraction. An all-duplicate or budget-exhausted result is a hard
stop; never silently exceed the recorded Mining budget.

## Handoff

Commit the final filtered parquet, the pre-history candidate parquet, router
summary, history ledger and summary, both embedding parquets, and the exact
positive row count. `emit_mined_sharegpt.py` then recovers the canonical JSONL
messages and ordered media for the selected real records.

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
