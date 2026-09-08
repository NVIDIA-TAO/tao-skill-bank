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
"$PYTHON" "$SKILL_ROOT/scripts/exact_f1_adapter.py" \
  --evaluator "$EVALUATOR" \
  --source "$PROXY_JSONL" \
  --predictions "$RESULTS_DIR/$LABEL/evaluate_proxy/predictions.jsonl" \
  --raw-output "$RESULTS_DIR/$LABEL/proxy_rcca/raw_f1.json" \
  --metric-output "$RESULTS_DIR/$LABEL/proxy_rcca/metric_result.json" \
  --component-threshold "$KPI_THRESHOLD"
```

This exact Proxy KPI is committed with the RCCA artifacts, compared with prior
Proxy results using the frozen KPI and tie breakers, and never used as the
Benchmark stopping result.

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
ordered atomic sample ID/paths, parse status, and raw prediction. Multiple task
rows for one physical single image or exact reference pair share an atomic ID
so that unit is embedded only once. Reference rows with a common test image but
different golden images remain distinct. Write `RCCA_Report.md` from these artifacts using
`RCCA_REPORT_TEMPLATE.md`, then commit all four files.

Route selected rows with `route_selected_gaps.py`, passing `--media-root` and
`--pair-assets-dir`; reference queries become one deterministic pair asset for
the encoder while retaining both original paths in the query parquet. The Defect Detection
ablation may select `all_proxy_severity`, which anchors on every Proxy DD row
and preserves deterministic FN/partial-overlap, FP, correct ordering; the
historical `hard_only` policy remains available. `task_mining_router.py`
then applies the immutable `config.mining.router_mode`:

- `image_only`: global cosine top-K;
- `task_strict`: exact task candidates only;
- `task_then_fallback`: strict candidates first, then global fill.

Every routed row records its tier, task types, matched query IDs, rank, and
cosine. Benchmark annotations or predictions are forbidden at this boundary.
The default `top_k_per_target` may be overridden per task; the DD ablation
records its Defect Detection override separately so maintenance tasks retain
their launch value.
When the launch contract enables detection calibration,
`select_detection_calibration.py` derives the single-image and reference
empty-GT rates from the frozen Proxy, deterministically rounds each cohort's
requested calibration total into empty and few-box quotas, and prepends the
matching real Mining examples. Reference candidates carry one atomic pair ID,
one deterministic two-image asset, both original paths, and explicit
`calibration_reference_no_change_ground_truth` evidence for empty/no-change
negatives. These rows carry the explicit `calibration` tier, pass through the
same atomic history and cumulative-pool budget, and never weaken the
`task_strict` policy applied to gap-routed neighbors. A cohort shortage fails
closed instead of borrowing quota from the other cohort.

When the frozen Proxy has no Component Count cohort, a launch may separately
prepend a bounded `count_replay` tier selected directly from real Mining rows.
That tier is restricted to `Component Count`, passes through the identical
history and cumulative-pool budget, is included in training, and does not
relax strict routing for any gap-derived neighbor.

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

`benchmark_cadence=final_and_best` scores the frozen Benchmark for the
checksum-validated reusable zero-shot baseline, a checkpoint whose exact Proxy
KPI ranking is best so far, and the final checkpoint. `every` scores every
checkpoint. Proxy evaluation and RCCA still run every iteration under both
policies.

## Selection replay

The packaged selection profiles live under `assets/gap-analysis/`. Use
`run_gap_analysis.py` or `replay_gap_analysis.py` against a frozen candidate
parquet to compare profile/seed behavior without another model evaluation.
The output records the input hash, resolved configuration, seed, quotas,
realized budget, selected-ID hash, and group composition.
