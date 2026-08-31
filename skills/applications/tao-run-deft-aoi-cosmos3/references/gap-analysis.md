# Proxy RCCA and frozen Benchmark KPI

Both roles run `cfw_jsonl_runtime.py` inside the Cosmos Framework image against
canonical NVPAW JSONL. It emits normalized rows atomically; use
`cfw_predictions.py` as the strict normalization/coverage gate when merging or
accepting external Framework shards. Prediction coverage is exact: missing,
duplicate, or unknown IDs fail.

## Proxy RCCA

Proxy is the sole error source for routing and Mining. Build candidates using
the evaluator path and SHA-256 frozen in `deft_state.json`:

```bash
"$PYTHON" "$SKILL_ROOT/scripts/analyze_gaps.py" \
  --evaluator "$EVALUATOR" \
  --source "$PROXY_JSONL" \
  --predictions "$RESULTS_DIR/$LABEL/evaluate_proxy/predictions.jsonl" \
  --output-dir "$RESULTS_DIR/$LABEL/proxy_rcca" \
  --gap-analysis-profile deficit_weighted_round_robin
```

`analyze_gaps.py` dynamically loads only the recorded evaluator's parsers and
matching helpers. It does not calculate the Benchmark KPI. It writes:

- `gaps_summary.json`;
- `gap_candidates.parquet`;
- `selected_gaps.parquet`.

Candidate rows retain task type, evaluator family, reference cohort, dataset,
physical target ID/path, parse status, and raw prediction. Multiple task rows
for one physical target share a target ID so the target image is embedded only
once. Write `RCCA_Report.md` from these artifacts using
`RCCA_REPORT_TEMPLATE.md`, then commit all four files.

Route selected rows with `route_selected_gaps.py`. `task_mining_router.py`
then applies the immutable `config.mining.router_mode`:

- `image_only`: global cosine top-K;
- `task_strict`: exact task candidates only;
- `task_then_fallback`: strict candidates first, then global fill.

Every routed row records its tier, task types, matched query IDs, rank, and
cosine. Benchmark annotations or predictions are forbidden at this boundary.

## Frozen Benchmark gate

The app has no F1 implementation. Invoke the recorded workspace evaluator
only through the adapter:

```bash
"$PYTHON" "$SKILL_ROOT/scripts/exact_f1_adapter.py" \
  --evaluator "$EVALUATOR" \
  --source "$BENCHMARK_JSONL" \
  --predictions "$RESULTS_DIR/$LABEL/evaluate_benchmark/predictions.jsonl" \
  --raw-output "$RESULTS_DIR/$LABEL/benchmark_metrics/raw_f1.json" \
  --metric-output "$RESULTS_DIR/$LABEL/benchmark_metrics/metric_result.json" \
  --component-threshold "$KPI_THRESHOLD"
```

The raw report path/SHA-256, evaluator path/SHA-256, exact five-component
vector, coverage, threshold, minimum attainment, and tie breakers are committed
together. All five components must meet the frozen threshold and both missing
and unknown prediction counts must be zero. Benchmark output can stop the loop
but can never seed RCCA, routing, or Mining.

## Selection replay

The packaged selection profiles live under `assets/gap-analysis/`. Use
`run_gap_analysis.py` or `replay_gap_analysis.py` against a frozen candidate
parquet to compare profile/seed behavior without another model evaluation.
The output records the input hash, resolved configuration, seed, quotas,
realized budget, selected-ID hash, and group composition.
