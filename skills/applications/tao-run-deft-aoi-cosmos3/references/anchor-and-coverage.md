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
- Under `--max-rows` / `--row-multiple` the share is taken on the
  *materialized* row count: `round(M · share)` anchor slots are reserved first
  (minus anchors already retained from previous iterations), all previous rows
  are kept, and current mined rows fill the rest in task-balanced order, so the
  cap or global-batch rounding displaces mined rows, never the anchors
  (`anchor.cap_reservation` reports the displaced count). Output order stays
  current mined → anchors → previous rows.
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
`assemble_summary.json.exposure_rows` is the per-purpose ledger (current
mined / previous mined / previous anchor / previous coverage / new anchor /
new coverage / total and the realized shares).

## Coverage blend (Phase 2 step 2b) — default off

A small fixed share of the corpus drawn so every eligible (task, dataset) cell
of the mining pool is represented regardless of what the gap-driven miner
picked. Two launch-recorded modes give a controlled comparison:

- `plain` — rows sampled uniformly from each pool cell (no correctness filter);
- `residual` — rows the scored checkpoint got wrong (`is_residual`); a cell
  whose residual rows run out falls back to its correct rows, tagged
  `purpose_tags = ["coverage", "anchor"]` and counted in
  `materialized_fallback_correct_records` (they do not carry `deft_anchor`,
  so anchor accounting stays separate).

Candidates: `build_coverage_candidates.py --mining <Mining JSONL> --scored-ids
<pool ids JSONL> --mode plain|residual --output-dir <dir> [--per-cell 3000]
[--fallback-per-cell 300] [--seed 17]` writes
`coverage_candidates_<mode>.jsonl` (+ manifest); each row carries the inert
key `deft_pool_status` (`correct` / `residual` / `unscored`), stripped again at
materialization.

Launch: `init_deft_state.py --coverage-blend-share 0.05 --coverage-blend-mode
plain|residual --coverage-blend-source <candidates.jsonl>
[--coverage-blend-min-rows-per-dataset 8] [--coverage-blend-seed 17]` records
`config.mining.coverage_blend`; the same `--coverage-blend-*` values go on
every iteration's selector command and the runner moves them to the
assembler (assembler-only; the selector never sees them).

Assembler rules (mirroring anchors):
- Slice totals are solved jointly: with base rows `N0` (neither anchor nor
  coverage), corpus `= N0 / (1 − share_anchor − share_coverage)`; each slice
  wants `round(corpus · share)` and adds only what previous iterations do not
  already carry (`deft_coverage: true` marker).
- Allocation is a deterministic round-robin that always feeds the cell with
  the fewest cumulative coverage rows (prior + new), so cells below the floor
  `K` fill first and the remainder spreads evenly; exhausted cells are skipped
  and reported (`shortage`, `cells_below_floor_after_selection`).
- Floor fail-closed: `cells × K` must fit the eventual budget under the row
  cap, `round(aligned max_rows · share)`; the first iteration's slice may be
  smaller than `cells × K` and is topped up as the corpus grows.
- Under `--max-rows` / `--row-multiple` the coverage slots
  `round(M · share) − prior coverage` are reserved like anchor slots; current
  mined rows absorb the trim. Output order: current mined → coverage →
  anchors → previous rows. Not combinable with the repetition blend.

Outputs: `materialized_coverage_records`, a `coverage_blend` block (mode,
requested/realized share, per-cell new and cumulative counts, fallback counts,
selected pool-status mix so plain vs residual are comparable, floor check,
cap reservation) and `coverage_blend_manifest.json` beside `train.jsonl`.
Provenance rows carry `source_kind = coverage_blend`, `purpose_tags` and
`pool_status`.

## Coverage blend (planned, not implemented)

A 5% cross-dataset slice with a per-dataset floor, `plain` (no correctness
filter) or `residual` (`is_residual == true`) mode, will use the same
candidate/exclusion machinery; datasets with no residual rows fall back to
anchors. Its options (`--coverage-blend-*`) will also be assembler-only.
