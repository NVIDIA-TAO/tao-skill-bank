# Cosmos3 AOI Real-Pair Mining

Read `skills/data/tao-mine-aoi-images/SKILL.md` before launch. Reuse its first
two steps unchanged: embed unique Proxy targets, then embed the Mining source
pool with the same encoder. Bare mode also reuses its native nearest-neighbor
step. Rich mode instead runs the host-side `task_mining_router.py` so task
eligibility and fallback provenance survive into training. Dispatch each GPU
embedding invocation through the selected platform's four verbs and give it
its own job-record.

## Inputs

- targets: Proxy false accepts/rejects only;
- source pool: recorded Mining annotations/media;
- model: the mining skill's configured SigLIP embedding model;
- top-K and metric: recorded DEFT config;
- router mode: recorded `config.mining.router_mode` (`image_only`,
  `task_strict`, or `task_then_fallback`);
- output root: `${RESULTS_DIR}/iterN/mining`.

Never use Benchmark errors as targets. The candidate/source side contains only
the recorded Mining pool; Proxy errors are query targets, not source samples.
The DEFT default top-K is 5. Preserve a user-supplied value; increase it only
when the history summary shows that the current neighborhood contains too few
novel candidates.

## Container user

The mining skill's setup notes tell you to drop `--user` because it raises a
`getpwuid()` `KeyError` during the `transformers` import. Do not drop it here.
That error is only the mapped uid having no entry inside the image, and this
workflow already carries the fix — pass the account databases along with the
mapping:

```bash
--user $(id -u):$(id -g) -e USER="$(id -un)" -e HOME=/tmp \
-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro
```

Verified 2026-07-30 against the pinned data-services image: `pwd.getpwuid`
resolves and `transformers` imports cleanly. This matters because mining writes
into `${RESULTS_DIR}/iterN/mining`, a tree the operator owns; run as root and
the parquet files come back root-owned and cannot be cleaned up afterwards.

If a future image genuinely rejects the mapping, repair ownership through a
container rather than assuming sudo:

```bash
docker run --pull=never --rm -v "$WORKSPACE:/ws" busybox:latest \
  chown -R "$(id -u):$(id -g)" /ws/<relative/path/to/mining>
```

## Rich task-aware router

The target embedding parquet carries the `task_types` retained by
`route_selected_gaps.py`. Mining annotations provide the supported task types
for every source target. Run one deterministic policy over the same embedding
artifacts for every ablation arm:

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

`image_only` selects one global cosine top-K per physical target and fans out
the source's available tasks. `task_strict` reuses that target embedding but
allocates a separate top-K for each selected task and emits only the matching
task. `task_then_fallback` takes those per-task strict neighbors first and fills
each target-task top-K shortfall from the global pool;
those additions are visibly marked `route_tier=fallback`. A zero-row output is
a hard stop. Do not run `filter_mined_by_cosine.py` afterward: this router
already applies and records the same floor.

## Bare cosine floor

For `bare_okng`, the native nearest-neighbor output is not sufficient proof of
the configured floor. Preserve raw outputs, then write cosine-qualified rows
to the distinct pre-history candidate parquet:

```bash
"$PYTHON" "$SKILL_ROOT/scripts/filter_mined_by_cosine.py" \
  --mined-parquet "$MINING_DIR/mined_raw.parquet" \
  --source-embeddings "$MINING_DIR/source_embeddings.parquet" \
  --target-embeddings "$MINING_DIR/target_embeddings.parquet" \
  --min-similarity "$MIN_SIMILARITY" \
  --output "$MINING_DIR/mined_candidates.parquet" \
  --summary "$MINING_DIR/cosine_filter_summary.json"
```

The output must differ from the raw parquet. A missing embedding, dimension
mismatch, zero-norm vector, non-finite value, missing path, or zero kept rows
is a hard stop.

## History-aware selection

After the rich router or bare cosine floor, drop filepaths selected by prior
iterations:

```bash
"$PYTHON" "$BANK_ROOT/skills/data/tao-mine-aoi-images/scripts/filter_mined_history.py" \
  --candidate-parquet "$MINING_DIR/mined_candidates.parquet" \
  --output-parquet "$MINING_DIR/mined_filtered.parquet" \
  --history-file "$RESULTS_DIR/mining_history.json" \
  --summary "$MINING_DIR/mining_history_summary.json" \
  --iteration "$ITERATION" \
  --topn "$TOPN"
```

`mined_filtered.parquet` is now the final novel-only handoff. Preserve
`mined_candidates.parquet`, `mining_history_summary.json`, and the run-level
ledger. Cosmos3 requires at least one mined row, so an all-duplicate result is a
hard stop with the summary's recommendation to increase `topn` or expand the
Mining pool; do not replay an earlier sample into the monotonic Train lineage.

## Handoff

Commit `data_mining` with the final filtered parquet, pre-history candidate
parquet, router summary (rich) or cosine summary (bare), history ledger plus
per-iteration summary, both embedding
parquets, and exact positive row count. Set `MINING_SELECTION_SUMMARY` to
`router_summary.json` for rich mode or `cosine_filter_summary.json` for bare
mode. The next stage uses `emit_mined_sharegpt.py` to recover the compatible
Mining prompts, golden images, and labels.

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/commit_stage.py \
  --results-dir "$RESULTS_DIR" --iter-label "iter$ITERATION" \
  --stage data_mining \
  --mining-parquet "$MINING_DIR/mined_filtered.parquet" \
  --mining-candidates "$MINING_DIR/mined_candidates.parquet" \
  --mining-summary "$MINING_DIR/$MINING_SELECTION_SUMMARY" \
  --mining-history "$RESULTS_DIR/mining_history.json" \
  --mining-history-summary "$MINING_DIR/mining_history_summary.json" \
  --mining-target-embeddings "$MINING_DIR/target_embeddings.parquet" \
  --mining-source-embeddings "$MINING_DIR/source_embeddings.parquet" \
  --mining-count <positive-int> \
  --summary "history-aware mining selected novel real pairs"
```
