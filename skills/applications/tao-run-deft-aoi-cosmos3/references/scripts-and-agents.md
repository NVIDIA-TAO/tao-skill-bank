# Cosmos3 DEFT Scripts and Agents

Set `PYTHON=$(bash scripts/deft_python.sh)`, then run host Python scripts with
`"$PYTHON"`. Resolve all paths to absolute paths before invoking a script.

## Available scripts

| Script | Purpose |
|---|---|
| `init_deft_state.py` | Initialize state once; freeze annotation/prompt/KPI/gap/mining-router profiles, AnomalyGen policy, contract and hashes, media root, Benchmark, and execution policy. |
| `deft_context.py` | Re-read state and print the deterministic next stage plus network/Python policy; reject a requested-stage mismatch. |
| `deft_exec.py` | Enforce state-backed offline/no-install/no-pull policy for external commands. |
| `commit_stage.py` | Validate one stage's inputs and atomically update state. `anomalygen.policy=disabled` authorizes an unconditional skip; `auto` requires zero-gap evidence. Terminal commits require reason/report evidence. |
| `metric_contract.py` | Validate/compare the Benchmark KPI contract. |
| `record_metric_result.py` | Bind `benchmark_metrics/metric_result.json` to an iteration. |
| `materialize_nvpaw_annotations.py` | Convert rich JSONL authoring records into a typed manifest and deterministic Cosmos JSON array. |
| `validate_sharegpt.py` | Validate explicit bare or rich ShareGPT contracts and image roles. |
| `validate_split_contract.py` | Enforce split isolation, monotonic generated-Train lineage, and Benchmark hash. |
| `check_annotations.py` | Per-role field-contract check over all three workspace annotation files. `ROLE_CONTRACT` is the authoritative field list. |
| `patch_eval_image_cap.py` | Source-classifies the selected image's evaluation cap, raises a recognized undersized literal, and returns the read-only mount only when required. |
| `multitask_metrics.py` | Parse and score rich classification/detection outputs with coverage evidence and balanced KPI artifacts. |
| `analyze_gaps.py` | Profile-dispatched Proxy RCCA/selection artifacts or Benchmark metric evidence. |
| `run_gap_analysis.py` / `replay_gap_analysis.py` | Run or compare deterministic gap-selection profiles against one frozen candidate table. |
| `route_selected_gaps.py` | Collapse selected rich record gaps into one mining query per physical target. |
| `task_mining_router.py` | Apply image-only, task-strict, or strict-then-fallback cosine top-K to the same target/source embeddings and preserve route provenance. |
| `emit_sdg_sharegpt.py` | Resolve documented or PAIDF 1.0.1 SDG paths and emit bare `NG` or capability-checked rich defect-classification records. |
| `filter_mined_by_cosine.py` | Recompute max cosine to Proxy targets and enforce the floor. |
| `emit_mined_sharegpt.py` | Align filtered paths to Mining prompts, golden images, and labels; honor routed task types for rich strict rows. |
| `assemble_training_json.py` | Monotonic profile-aware training merge with record/media dedupe and leakage checks. |
| `align_token_usage.py` | Backfill stage token accounting after a run when a transcript is available. |
| `render_report.py` | Deterministically render the self-contained NVIDIA-styled HTML report from state and recorded artifacts, including escaped annotation prompt examples; validate required sections/placeholders and replace atomically. |
| `finalize_run.py` | Render final evidence, validate the explicit stop reason, commit `loop_stop`, and record the report path. |

`init_deft_state.py` requires `--gpu-model` with the exact model string from
the selected platform's Preflight. `commit_stage.py` requires a positive
`--duration-sec` on every executed success, error, and `loop_stop` commit. Use
the backend's measured elapsed wall time for submitted jobs and a directly
measured host duration for inline stages; round a measured sub-second executed
stage up to `1`. A documented `--skip` records `status=skipped`, copies the
summary into `skip_reason`, and accepts a non-negative duration including `0`.
An omitted duration or any negative duration is rejected.

Before every stage, call `deft_context.py --state <deft_state.json> --stage
<stage>`. Wrap local external/container commands with `deft_exec.py --state
<deft_state.json> -- <command>`; the selected remote platform must enforce the
same policy when it constructs a job.

Train, Proxy evaluate, and Benchmark evaluate reuse the current
`tao-finetune-cosmos-reason` action commands. Mining reuses the embedding and
history stages from `tao-mine-aoi-images`; rich mode replaces its provenance-
losing final k-NN output with `task_mining_router.py`. The application owns only
the DEFT-specific state, isolation, profile-aware analysis, routing, filtering,
and assembly scripts.

## Automatic report hook

`init_deft_state.py` invokes `render_report.py` after writing canonical state.
`commit_stage.py` invokes it again after every valid state commit,
including error and `loop_stop` commits. The hook is outside the state
transaction: it reports `report hook failed` without rolling back a valid
stage result.

Normally do not render separately. After optional loop-end token alignment, or
to recover from a reported presentation error, run:

```bash
PYTHON=$(bash <skill_root>/scripts/deft_python.sh)
"$PYTHON" <skill_root>/scripts/render_report.py \
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
