# Cosmos3 DEFT Scripts and Agents

Run host Python through `scripts/deft_python.sh`. Resolve all paths to absolute
paths before invoking a script.

## Available scripts

| Script | Purpose |
|---|---|
| `init_deft_state.py` | Initialize version-3 Cosmos3 state once; freeze Benchmark hash and bare mode. |
| `audit_deft_run.py` | Read-only state/log/artifact audit and resume oracle, including AMP-allocation sum proof. |
| `commit_stage.py` | Atomically commit one stage and roll back on failed audit; AnomalyGen records both generated CSV and allocation JSON. |
| `log_stage.py` | Ordered JSONL writer used by `commit_stage.py`; do not call for normal stage commits. |
| `metric_contract.py` | Validate/compare the Benchmark KPI contract. |
| `record_metric_result.py` | Bind `benchmark_metrics/metric_result.json` to an iteration. |
| `validate_sharegpt.py` | Enforce two-image, exact bare OK/NG ShareGPT records. |
| `validate_split_contract.py` | Enforce split isolation, monotonic generated-Train lineage, and Benchmark hash. |
| `check_annotations.py` | Per-role field-contract check over all three workspace annotation files. `ROLE_CONTRACT` is the authoritative field list. |
| `patch_eval_image_cap.py` | Raises the pinned image's 1-image-per-prompt evaluation cap to what bare_okng needs, and returns the read-only mount. Retires itself once the image is fixed. |
| `analyze_gaps.py` | Proxy RCCA artifacts or Benchmark aggregate metric evidence. |
| `emit_sdg_sharegpt.py` | AnomalyGen SDG output as bare `NG` ShareGPT records. |
| `filter_mined_by_cosine.py` | Recompute max cosine to Proxy targets and enforce the floor. |
| `emit_mined_sharegpt.py` | Align filtered paths to Mining prompts, golden images, and labels. |
| `assemble_training_json.py` | Monotonic bare training-data merge with dedupe/leakage checks. |
| `align_token_usage.py` | Backfill stage token accounting after a run when a transcript is available. |
| `render_report.py` | Deterministically render the self-contained NVIDIA-styled HTML report from canonical state/log and recorded artifacts, including escaped annotation prompt examples; validate required sections/placeholders and replace atomically. |

`init_deft_state.py` requires `--gpu-model` with the exact model string from
the selected platform's Preflight. `commit_stage.py` requires a positive
`--duration-sec` on every success, error, skip, and `loop_stop` commit. Use the
backend's measured elapsed wall time for submitted jobs and a directly measured
host duration for inline stages; round a measured sub-second stage up to `1`.
An omitted, zero, or negative duration is rejected rather than recorded as an
unknown value.

Train, Proxy evaluate, and Benchmark evaluate reuse the current
`tao-finetune-cosmos-reason` action commands. Mining reuses
`tao-mine-aoi-images`. The application owns only the DEFT-specific state,
isolation, OK/NG analysis, filtering, and assembly scripts.

## Automatic report hook

`init_deft_state.py` invokes `render_report.py` after writing canonical state.
`commit_stage.py` invokes it again after every valid state/log commit,
including error and `loop_stop` commits. The hook is outside the state
transaction: it reports `report hook failed` without rolling back a valid
stage result.

Normally do not render separately. After optional loop-end token alignment, or
to recover from a reported presentation error, run:

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/render_report.py \
  --results-dir "${RESULTS_DIR}" --require-terminal
```

The legacy `agents/reporter.md` is a compatibility wrapper around this exact
command and contains no HTML-generation logic.

## Path invariants

- Every recorded stage artifact is absolute.
- Every iteration artifact is under `${RESULTS_DIR}/baseline` or
  `${RESULTS_DIR}/iterN`.
- Train checkpoints are under `${RESULTS_DIR}/<label>/train`.
- Proxy artifacts never contain Benchmark paths.
- Mining, routing, and assembly never consume Benchmark per-sample errors.
- User data is never mounted over the cosmos-rl `/workspace` package root.
