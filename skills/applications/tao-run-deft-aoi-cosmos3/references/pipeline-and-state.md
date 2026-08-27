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
6. Before committing `proxy_rcca`, write `proxy_rcca/RCCA_Report.md` from the
   three machine artifacts using `references/RCCA_REPORT_TEMPLATE.md` and pass
   it with `--rcca-report`; `references/rcca-artifact-manifest.json` is the
   machine-readable source for artifact requirements and state fields.

Baseline may stop immediately when the Benchmark gate passes. Baseline does not
count against `max_iterations`.

## Iteration

1. Build `mining_targets.json` only from the preceding Proxy false-accept /
   false-reject artifacts. Never read Benchmark per-sample errors here.
2. Run `tao-generate-anomalies` in `inference_only` mode against the recorded
   AnomalyGen project. Commit Phase 2's defect-to-count `allocation.json` as
   the canonical AMP-allocation evidence, then run `emit_sdg_sharegpt.py` to turn each generated pair
   into a bare `NG` record. When the driving Proxy RCCA recorded zero false
   accepts, commit `--skip` instead of launching the generator after checking
   the recorded `false_accepts_json` on disk. See
   `references/tao-generate-anomalies.md`.
3. Invoke `tao-mine-aoi-images` on the recorded Mining pool. Persist raw mined
   paths and source/target embeddings under the current iteration.
4. Run `filter_mined_by_cosine.py` into `mined_candidates.parquet`; a zero-row
   result is a hard stop. Then run the mapped skill's
   `filter_mined_history.py` to remove filepaths selected by every prior
   iteration and write the final `mined_filtered.parquet`, per-iteration
   summary, and run-level ledger. A zero-row novel result is also a hard stop;
   surface the recommendation to increase top-K above the default 5 or expand the
   Mining pool.
5. Run `emit_mined_sharegpt.py` to align every mined path to exactly one Mining
   source record. It inherits the source prompt and exact label.
6. After RCA and Mining selection, run `assemble_training_json.py` without a
   seed for `iter1`, passing the mined records and — when AnomalyGen ran — the
   synthetic records as separate `--new-json` inputs; together they become
   `train_iter_1.json`. Later iterations use `train_iter_<N-1>.json` as the
   seed and write `train_iter_<N>.json`. Dedupe by the image path and
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

## Train backend boundary

Train alone uses the resolver-selected `images.tao_toolkit.cosmos_framework`
image and `python -m cosmos_framework.scripts.train --sft-toml=<config>`.
Proxy and Benchmark evaluate remain on the existing cosmos-rl vLLM action.
The Framework stage consumes the prepared HF Qwen3-VL snapshot directly,
trains native VLM LoRA through BF16 single-node FSDP2, writes DCP on the
profile's cadence, and runs the image-owned exact-key DCP exporter before the
next evaluate stage. Evaluation receives only the verified merged HF
safetensors directory.

The pinned image does not yet natively accept CR3's two-image ShareGPT rows.
The Train submission therefore mounts the skill-owned interim adapter as one
read-only experiment module; no generated annotation is rewritten. Remove
that mount only after source inspection proves a native equivalent.

Assembled annotations retain absolute host image paths, matching the evaluate
stages' identity-mount convention. `submit_cfw_train.py` therefore always adds
a read-only same-path mount for the host media root and a read-only same-path
mount for each containing directory referenced outside that root.
Relative annotation paths continue to resolve through the configured
compute-frame media mount. Operators do not need to discover or add these
identity mounts by hand.

Every Train submission records all of the following before launch:

- job-record `image`: immutable `repository@sha256:...`, not a mutable tag;
- `train_submission.json`: `train_backend=cosmos-framework`, versions key,
  resolved URI, immutable image digest, job id, config, and exact argv;
- `deft_state.json` `config.training`: `backend`, `image`, and `image_digest`;
- Train terminal transition message/DEFT stage summary: backend plus the same
  immutable digest.

Treat disagreement among those four evidence surfaces as a hard stop. Continue
to poll Docker/Kubernetes/SLURM/Brev rather than reading runtime state from the
submission manifest or DEFT state.

## State

`deft_state.json` is the only persistent loop record. Initialize it once with
`init_deft_state.py`, then mutate it only through `commit_stage.py`; never
hand-edit or reinitialize it. It contains:

- immutable run identity, results directory, metric contract, execution
  policy, selected Python, and maximum iterations;
- platform, model, image, spec, annotation, media-root, compute, and mining
  configuration;
- separate `cosmos_framework_train` and `cosmos_rl_evaluate` container
  identities, plus the Framework Train backend and immutable image digest;
- the frozen Benchmark SHA-256;
- one `iterations.<label>` object whose `stage_completed` matches the latest
  successful event for that label;
- absolute artifact paths under `${RESULTS_DIR}/<label>`;
- terminal `final_artifacts` only after validated finalization;
- an `events` array with a strict, monotonically increasing `seq`, UTC
  timestamp, iteration, stage, `ok|error|skipped`, non-empty summary, explicit
  `skip_reason` on skips, measured positive duration for executed stages,
  non-negative duration for skips, and context-token placeholder.

Before every stage, re-read `deft_state.json`. Use the latest event plus
`iterations.<label>.status` and `stage_completed` to resume. A failed stage may
be retried after its cause is fixed; the retry becomes a new state event.

## Stage commit examples

All paths are absolute.

```bash
"$PYTHON" "$SKILL_ROOT/scripts/commit_stage.py" \
  --results-dir "$RESULTS_DIR" --iter-label iter1 --stage train \
  --best-ckpt "$RESULTS_DIR/iter1/train/eval_model" \
  --export-verification "$RESULTS_DIR/iter1/train/export_action.json" \
  --training-spec "$WORKSPACE/specs/train_spec.toml" \
  --duration-sec "$STAGE_DURATION_SEC" \
  --summary "cosmos-framework LoRA SFT completed; image=$COSMOS_FRAMEWORK_IMAGE_DIGEST"
```

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

```bash
"$PYTHON" "$SKILL_ROOT/scripts/commit_stage.py" \
  --results-dir "$RESULTS_DIR" --iter-label "$LABEL" \
  --stage proxy_rcca \
  --proxy-gaps-summary "$RESULTS_DIR/$LABEL/proxy_rcca/gaps_summary.json" \
  --false-accepts "$RESULTS_DIR/$LABEL/proxy_rcca/false_accepts.json" \
  --false-rejects "$RESULTS_DIR/$LABEL/proxy_rcca/false_rejects.json" \
  --rcca-report "$RESULTS_DIR/$LABEL/proxy_rcca/RCCA_Report.md" \
  --duration-sec "$STAGE_DURATION_SEC" \
  --summary "Proxy RCCA analyzed and next-iteration targets identified"
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
