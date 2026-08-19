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

Every successful or failed stage has exactly one state update and one ordered
event. The bundled `commit_stage.py` owns that write.

## Profile initialization examples

After launch approval, staged specs, annotation validation, platform preflight,
and image resolution, initialize a legacy run explicitly:

```bash
"$PYTHON" "$SKILL_ROOT/scripts/init_deft_state.py" \
  --results-dir "$RESULTS_DIR" --workspace "$WORKSPACE" \
  --platform "$PLATFORM" --max-iterations "$MAX_ITERATIONS" \
  --gpu-model "$GPU_MODEL" \
  --annotation-profile bare_okng --kpi-profile bare_okng_v1 \
  --gap-analysis-profile legacy_bare_okng
```

For rich NVPaw prompts, keep prompt, KPI, gap selection, and mining routing as
separate explicit axes:

```bash
"$PYTHON" "$SKILL_ROOT/scripts/init_deft_state.py" \
  --results-dir "$RESULTS_DIR" --workspace "$WORKSPACE" \
  --platform "$PLATFORM" --max-iterations "$MAX_ITERATIONS" \
  --gpu-model "$GPU_MODEL" \
  --annotation-profile nvpaw_multitask_v1 \
  --prompt-variant official_v1 \
  --kpi-profile task_balanced_v1 \
  --gap-analysis-profile deficit_weighted_round_robin \
  --gap-analysis-seed 17 \
  --mining-router-mode task_strict \
  --anomalygen-policy disabled
```

The resolved gap YAML is copied into results and its hash is frozen in state.
Use `--gap-analysis-config` instead of `--gap-analysis-profile` for one custom
ablation arm; the flags are mutually exclusive. Change only
`--mining-router-mode image_only|task_strict|task_then_fallback` when comparing
router policies against the same selected gaps and embeddings.
Change only `--anomalygen-policy auto|disabled` for a separate synthetic-data
ablation; `disabled` commits an unconditional stage skip and leaves mining
enabled.

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

1. Build `mining_targets.json` only from preceding Proxy gaps. Bare mode uses
   false-accept/reject artifacts; rich mode selects from the frozen candidate
   parquet with the state-recorded gap profile, then collapses records by
   `target_id`. Never read Benchmark per-sample errors here.
2. If `config.anomalygen.policy` is `disabled`, commit `anomalygen --skip`
   without gap evidence and continue to mining. Otherwise run
   `paidf-anomalygen` in `inference_only` mode against the recorded AnomalyGen
   project. Commit Phase 2's defect-to-count `allocation.json` as
   the canonical AMP-allocation evidence, then run `emit_sdg_sharegpt.py` to turn each generated pair
   into a bare `NG` record. Rich mode permits only known-label defect
   classification output and rejects detection without geometry. When the driving Proxy RCCA recorded zero false
   accepts, commit `--skip` instead of launching the generator after checking
   the recorded `false_accepts_json` on disk. See
   `references/paidf-anomalygen.md`.
3. Invoke the embedding stages of `tao-mine-aoi-images` on the recorded Mining
   pool and the unique Proxy targets. Bare mode keeps its native image-only k-NN
   and `filter_mined_by_cosine.py` path. Rich mode runs
   `task_mining_router.py` over both embedding artifacts with the frozen router
   mode, top-K, and cosine floor, writing `mined_candidates.parquet` plus
   `router_summary.json`. A zero-row result is a hard stop.
4. Run the mapped skill's
   `filter_mined_history.py` to remove filepaths selected by every prior
   iteration and write the final `mined_filtered.parquet`, per-iteration
   summary, and run-level ledger. A zero-row novel result is also a hard stop;
   surface the recommendation to increase top-K above the default 5 or expand the
   Mining pool.
5. Run `emit_mined_sharegpt.py` to align every mined path to its Mining source.
   Bare mode requires exactly one match. Rich strict rows fan out only to
   `routed_task_types`; image-only and explicitly marked fallback rows keep the
   source target's available QA records while preserving image roles.
6. After RCA and Mining selection, run `assemble_training_json.py` without a
   seed for `iter1`, passing the mined records and — when AnomalyGen ran — the
   synthetic records as separate `--new-json` inputs; together they become
   `train_iter_1.json`. Later iterations use `train_iter_<N-1>.json` as the
   seed and write `train_iter_<N>.json`. Bare mode deduplicates by media pair;
   rich mode deduplicates by record fingerprint. Exclude Proxy and Benchmark
   targets in both modes.
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

## State

`deft_state.json` is the only persistent loop record. Initialize it once with
`init_deft_state.py`, then mutate it only through `commit_stage.py`; never
hand-edit or reinitialize it. It contains:

- immutable run identity, results directory, metric contract and hash, execution
  policy, selected Python, and maximum iterations;
- platform, model, image, spec, annotation, media-root, compute, and mining
  configuration;
- frozen annotation hashes, annotation/prompt/KPI profiles, resolved
  gap-analysis config/hash/seed, mining-router mode, AnomalyGen policy, and
  Benchmark SHA-256;
- one `iterations.<label>` object whose `stage_completed` matches the latest
  successful event for that label;
- absolute artifact paths under `${RESULTS_DIR}/<label>`;
- terminal `final_artifacts` only after validated finalization;
- an `events` array with a strict, monotonically increasing `seq`, UTC
  timestamp, iteration, stage, `ok|error`, non-empty summary, measured
  duration, and context-token placeholder.

Before every stage, re-read `deft_state.json`. Use the latest event plus
`iterations.<label>.status` and `stage_completed` to resume. A failed stage may
be retried after its cause is fixed; the retry becomes a new state event.

## Stage commit examples

All paths are absolute.

```bash
"$PYTHON" "$SKILL_ROOT/scripts/commit_stage.py" \
  --results-dir "$RESULTS_DIR" --iter-label iter1 --stage train \
  --best-ckpt "$RESULTS_DIR/iter1/train/safetensors/epoch_10" \
  --training-spec "$WORKSPACE/specs/train_spec.toml" \
  --duration-sec "$STAGE_DURATION_SEC" \
  --summary "first mined-data Cosmos3 LoRA SFT completed"
```

For rich `benchmark_metrics`, also commit `--task-metrics`,
`--sample-metrics`, and `--prediction-coverage`. Rich `proxy_rcca` additionally
requires gap candidates, selected gaps, and gap-analysis summary. Rich routing
commits both JSON and parquet targets plus `routing_summary.json`.

```bash
"$PYTHON" "$SKILL_ROOT/scripts/commit_stage.py" \
  --results-dir "$RESULTS_DIR" --iter-label baseline \
  --stage benchmark_metrics \
  --benchmark-metrics-summary \
    "$RESULTS_DIR/baseline/benchmark_metrics/metrics_summary.json" \
  --metric-result \
    "$RESULTS_DIR/baseline/benchmark_metrics/metric_result.json" \
  --duration-sec "$STAGE_DURATION_SEC" \
  --summary "frozen Benchmark KPI analyzed"
```

Re-read `deft_state.json` before constructing each next command and continue
from its latest event and `stage_completed` value.

## Stop and completion

For an ordinary stop, run `scripts/deft_context.py --stage finalize`, then
`scripts/finalize_run.py` after a completed `benchmark_metrics`. Pass exactly
one stop reason:

- `--stop-reason metric_met` when the final Benchmark result passes; or
- `--stop-reason max_iterations` when a non-passing completed `iterN` has
  `N >= max_iterations`.

For a hard stop, commit the failed stage with `--status error` and halt. The
state becomes `failed`; a failed terminal run is not KPI completion.

`finalize_run.py` renders the report before committing `loop_stop`; the commit
validates and records that report and refreshes it again. After optional token
alignment, run `render_report.py --require-terminal` once so the final artifact contains the
aligned evidence. The final completion claim requires
a fresh state read showing `status == "complete"`, a complete baseline, and a
complete final iteration. A hard-stop claim instead requires
`status == "failed"`.
