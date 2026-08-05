# Cosmos3 DEFT Pipeline and State

## Transition graph

The frozen Benchmark gate is evaluated first, before any Proxy work. Proxy
evaluate and RCCA run only when the gate is unmet, because their sole purpose
is to seed the next iteration's mining targets. When the gate passes there is
no next iteration, so the run stops without spending a Proxy evaluation.

```text
baseline/evaluate_benchmark
  -> benchmark_metrics
       -> loop_stop                         (KPI met)
       -> evaluate_proxy                    (KPI unmet)
  evaluate_proxy
  -> proxy_rcca
  -> iter1/routing

iterN/routing
  -> anomalygen                          (or documented skip)
  -> data_mining
  -> assemble_data
  -> validate_data
  -> train
  -> evaluate_benchmark
  -> benchmark_metrics
       -> loop_stop                         (KPI met or N=max_iterations)
       -> evaluate_proxy                    (continuing)
  evaluate_proxy
  -> proxy_rcca
  -> iterN+1/routing
```

A completed iteration therefore ends at `benchmark_metrics` when the gate
stopped the run, and at `proxy_rcca` when it continued. The final iteration of
any run has Benchmark artifacts but no Proxy artifacts.

An error event may transition only to `loop_stop`. Every successful stage has
exactly one state update and exactly one ordered log entry. The bundled
`commit_stage.py` owns both writes.

## Baseline

1. Zero-shot evaluate the unmodified base model on frozen Benchmark with
   exact-output prompting and normalize the last standalone `OK`/`NG` token.
   This establishes the zero-shot KPI every later iteration is compared against.
2. Analyze Benchmark with `analyze_gaps.py --evaluation-role benchmark`; write
   `metrics_summary.json` and `metric_result.json`. Commit the configured
   metric only at `benchmark_metrics`.
3. Stop here when the gate passes: the base model already meets the KPI and no
   training is warranted.
4. Only when the gate is unmet, zero-shot evaluate the same base model on Proxy
   with identical prompting and generation settings.
5. Analyze Proxy results with `--evaluation-role proxy`. Preserve false accepts
   and false rejects as the only RCCA source.

Baseline may stop immediately when the Benchmark gate passes. Baseline does not
count against `max_iterations`.

## Iteration

1. Build `mining_targets.json` only from the preceding Proxy false-accept /
   false-reject artifacts. Never read Benchmark per-sample errors here.
2. Run `paidf-anomalygen` in `inference_only` mode against the recorded
   AnomalyGen project, then `emit_sdg_sharegpt.py` to turn each generated pair
   into a bare `NG` record. When the driving Proxy RCCA recorded zero false
   accepts, commit `--skip` instead of launching the generator; the audit
   re-proves that against `false_accepts_json` on disk. See
   `references/paidf-anomalygen.md`.
3. Invoke `tao-mine-aoi-images` on the recorded Mining pool. Persist raw mined
   paths and source/target embeddings under the current iteration.
4. Run `filter_mined_by_cosine.py`; a zero-row result is a hard stop.
5. Run `emit_mined_sharegpt.py` to align every mined path to exactly one Mining
   source record. It inherits the source prompt, golden reference, and exact
   label.
6. After RCA and Mining selection, run `assemble_training_json.py` without a
   seed for `iter1`, passing the mined records and — when AnomalyGen ran — the
   synthetic records as separate `--new-json` inputs; together they become
   `train_iter_1.json`. Later iterations use `train_iter_<N-1>.json` as the
   seed and write `train_iter_<N>.json`. Dedupe by the two-image pair and
   exclude Proxy and Benchmark targets.
7. Run `validate_sharegpt.py --require-files` and
   `validate_split_contract.py` against the assembled Train file, passing
   `--synthetic` when AnomalyGen produced records this iteration and
   `--previous-train train_iter_<N-1>.json` for N>1. The latter makes historical
   records eligible while proving that the current Train retained all of them.
8. Retrain, then Benchmark-evaluate/gate. Stop when the gate passes or
   `N = max_iterations`. Only when the loop continues, Proxy-evaluate and run
   RCCA to seed the next iteration's routing.

Training data is monotonic after its creation: `train_iter_1.json` contains
only newly mined and newly generated, validated records; `train_iter_<N>.json`
for `N > 1` starts from the preceding Train artifact and adds only newly mined
and newly generated, validated records.

## State fields

`deft_state.json` contains:

- immutable run identity, results directory, metric contract, and maximum
  iterations;
- platform, model, image, spec, annotation, media-root, compute, and mining
  configuration;
- the frozen Benchmark SHA-256;
- one `iterations.<label>` object whose `stage_completed` matches the last
  successful log event for that label;
- absolute artifact paths under `${RESULTS_DIR}/<label>`.

`loop_log.jsonl` contains a strict, gap-free `seq`, UTC timestamp, iteration,
stage, `ok|error`, non-empty summary, duration, and context-token placeholder.

## Stage commit examples

All paths are absolute.

```bash
"$PYTHON" "$SKILL_ROOT/scripts/commit_stage.py" \
  --results-dir "$RESULTS_DIR" --iter-label iter1 --stage train \
  --best-ckpt "$RESULTS_DIR/iter1/train/safetensors/epoch_10" \
  --training-spec "$WORKSPACE/specs/train_spec.toml" \
  --summary "first mined-data Cosmos3 LoRA SFT completed"
```

```bash
"$PYTHON" "$SKILL_ROOT/scripts/commit_stage.py" \
  --results-dir "$RESULTS_DIR" --iter-label baseline \
  --stage benchmark_metrics \
  --benchmark-metrics-summary \
    "$RESULTS_DIR/baseline/benchmark_metrics/metrics_summary.json" \
  --metric-result \
    "$RESULTS_DIR/baseline/benchmark_metrics/metric_result.json" \
  --summary "frozen Benchmark KPI analyzed"
```

Run the audit before constructing each next command. Its
`read_before_action` output is the resume oracle.

## Stop and completion

For an ordinary stop, commit `loop_stop` after a completed
`benchmark_metrics`. The audit accepts a successful stop only when:

- at least one completed Benchmark result passes the metric contract; or
- a completed `iterN` has `N >= max_iterations`.

For a hard stop, commit the failed stage with `--status error`, then commit
`loop_stop`. A failed terminal run is not KPI completion.

Render the report after each completed iteration and once at loop end. The
final completion claim requires `audit_deft_run.py --require-complete`.
