# Correct-row anchors (and the planned coverage blend)

## Why

Cumulative continue-SFT on mined residual rows collapses abilities the loop
does not mine for (single-image yes/no fell from 55 to 42 in the proxy-steered
Ablation-1 arm while boxes improved). Anchors keep a fixed share of the
training corpus on rows the model already answers correctly, chosen across
tasks by the KPI set's task shares, so rehearsal is spread over every ability
instead of the one the RCCA is chasing.

## Assets (operator-owned, built once per scored checkpoint)

1. Score the pool with `score_pool_residual.py` (`analysis-tools.md`) and
   export ids: one JSON line per pool row with `id, task_type, dataset,
   row_score, error_type, is_residual` (from `pool_residual.parquet`).
2. Build the candidate file (CPU, streams the canonical Mining JSONL):

   ```bash
   python3 "$SKILL_ROOT/scripts/build_anchor_candidates.py" \
     --mining "$WORKSPACE/annotations/mining.jsonl" \
     --scored-ids "$RESIDUAL_DIR/pool_scored_ids.jsonl" \
     --output-dir "$WORKSPACE/annotations/anchors_<checkpoint>_<date>" \
     --per-cell 3000 --seed 17
   ```

   Only the six supported tasks, only rows with `row_score == 1.0` and
   `is_residual == false`, at most `--per-cell` rows per (task, dataset) chosen
   by a stable seed/id hash. The manifest seals both inputs and the output.

## Launch-recorded options (default off)

`init_deft_state.py --anchor-share 0.10 --anchor-source <anchor_candidates.jsonl>
[--anchor-task-shares <eval.jsonl>] [--anchor-source-cap 0.35] [--anchor-seed 17]`
records `config.mining.anchor` (share, unit `rows`, source + sha256, task-share
source = the KPI set by default). The controller passes the same values on the
selector command; `render_iteration_mining_runner.py` moves every `--anchor-*`
option to the assembler, which is the only writer of the cumulative Train.

## What the assembler does each iteration

- `N0` = previous rows that are not anchors + unique current mined rows.
- Total anchors wanted `T = round(N0 · share / (1 − share))`; new anchors
  `= T − anchors already present` (anchor rows carry the top-level marker
  `deft_anchor: true`; the runtime reads only id/task_type/messages, so the
  marker is inert, and previous anchors are retained like every previous row).
- Task quotas = largest-remainder split of the new anchors by the task-share
  source's row shares; inside a task, datasets are filled round-robin under
  `--anchor-source-cap` (relaxed to `ceil(quota / n_datasets)` when fewer
  datasets exist), ordered by seed/id hash.
- Skipped, never fatal: candidates whose id or fingerprint is already in the
  corpus (`already_in_corpus`) or whose atomic identity is an evaluation target
  (`evaluation_target`). Counts are in the manifest.
- Materialization order under the row cap: current mined rows first, then
  anchors, then all previous rows; anchors are trimmed before current rows.
- Not combinable with the repetition blend (fail closed).

Outputs: `assemble_summary.json` gains `materialized_anchor_records` and an
`anchor` block (requested share, realized share in rows, per task × dataset
counts, skips, shortages); `anchor_manifest.json` beside `train.jsonl`
repeats it with the training-JSONL binding. Provenance rows carry
`source_kind = anchor_correct` and `purpose_tags = ["anchor"]`.

## Exposure accounting

The share is in rows (`unit: rows`); supervised-token share is not measured
here. Report both the requested and realized row share per iteration, and
compare the retention cohorts (single-image BCQ, reference BCQ, Component
Detection, reference clean FP) against the parent run.

## Coverage blend (planned, not implemented)

A 5% cross-dataset slice with a per-dataset floor, `plain` (no correctness
filter) or `residual` (`is_residual == true`) mode, will use the same
candidate/exclusion machinery; datasets with no residual rows fall back to
anchors. Its options (`--coverage-blend-*`) will also be assembler-only.
