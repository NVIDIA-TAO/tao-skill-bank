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
  (`evaluation_target`). Counts are in the manifest. When the empty-answer
  guard is enabled (`empty-answer-guard.md`), candidates with an empty ground
  truth are skipped as well (`empty_ground_truth`, reported as
  `anchor_empty_rows_excluded` with `prefer_non_empty_rows = true`), because
  anchors are never trimmed and an empty-answer anchor could only leave an
  untrimmable excess behind.
- Anchors are not in the mining history, so a later iteration's miner may
  select a retained anchor again as a plain row. The assembler de-duplicates
  current mined rows by id and by marker-free content against the retained
  corpus (`duplicates_skipped`); the retained marked copy stays, and ids in
  `train.jsonl` remain unique for the canonical validator.
- Under `--max-rows` / `--row-multiple` the share is taken on the
  *materialized* row count: `round(M · share)` anchor slots are reserved first
  (minus anchors already retained from previous iterations), all previous rows
  are kept, and current mined rows fill the rest in task-balanced order, so the
  cap or global-batch rounding displaces mined rows, never the anchors
  (`anchor.cap_reservation` reports the displaced count). Output order stays
  current mined → anchors → previous rows. Calibration rows the materializer
  marked `deft_calibration` are never displaced either
  (`cap_reservation.calibration_rows_protected`), so the verified calibration
  contract survives assembly (2026-09-14: 382 of 3,000 reference pairs were
  trimmed). When the calibration rows alone exceed the aligned-down capacity
  (a calibration-dominated iteration with almost no trimmable mined rows), the
  corpus is rounded **up** to the next `--row-multiple` and the gap is filled
  with extra anchors (`alignment_fill_anchors`, provenance tag
  `alignment_fill`, `alignment_policy = round_up_fill_with_anchors`); the
  realized anchor share then exceeds the requested share for that iteration
  and the next iteration's top-up accounts for it. Fail closed only when the
  rounded-up size exceeds `--max-rows` or the anchor supply runs out.
- `--anchor-share-exclude-rows <cumulative rows>` (default 0) takes a
  launch-recorded acquisition slice out of the share base, both for the
  uncapped target and for the cap reservation; `realized_share_rows` is then
  relative to that base and `realized_share_of_all_rows` keeps the plain ratio.
- Not combinable with the repetition blend (fail closed).

Outputs: `assemble_summary.json` gains `materialized_anchor_records` and an
`anchor` block (requested share, realized share in rows, per task × dataset
counts, skips, shortages); `anchor_manifest.json` beside `train.jsonl`
repeats it with the training-JSONL binding. Provenance rows carry
`source_kind = anchor_correct` and `purpose_tags = ["anchor"]`.

## Share ceiling under the empty-answer guard (Feature B3)

Under `--max-rows` / `--row-multiple` two paths could add anchors beyond the
configured share: the leftover fill (when the current candidates do not fill
the aligned size, spare anchors from the uncapped target took the slots) and
the round-up alignment fill above. With the empty-answer guard on, the guard
trims empty rows and the candidates run out routinely, so the leftover fill
became a steady drain: run r4 (2026-09-15) drew 230 / 478 / 386 anchors in
iterations 2–4 and reached 1,094 of 5,376 rows = 20.4% against a 0.10 share.

`--anchor-overfill allow|forbid` (assembler; `init_deft_state.py` records it
under `config.mining.empty_answer_guard.anchor_overfill`, default `forbid`
when any empty-answer cap is given, `allow` otherwise, so runs without the
guard are byte-identical to before):

- `forbid`: the leftover fill may not draw anchors beyond `round(M × share)`
  (`extra_coverage` and the mined back-fill stay). When the retained rows, the
  share-bound anchors, the coverage rows and this iteration's candidates do not
  fill the aligned size `M`, `M` shrinks to the largest `--row-multiple`
  multiple they do fill (`cap_reservation.aligned_rows_shrunk_for_share`; the
  guard's `aligned_rows_shrunk` keeps counting the first-pass-to-final
  difference). The round-up alignment fill is not available either (a
  calibration-dominated iteration that does not fit fails closed with the
  existing message plus "anchor over-fill forbidden"). The iteration fails
  closed **only** when the growth would be zero rows, i.e. nothing of this
  iteration fits above the previous corpus:
  `training materialization would add zero rows of this iteration under the
  empty-answer guard (anchors may not over-fill the global batch):
  current_rows_after_guard=…, anchor_slots=…, coverage_slots=…,
  row_multiple=…, previous_rows=…, aligned_rows=…`.
- `allow`: the parent behaviour; `cap_reservation.leftover_fill_anchors`
  counts the anchors drawn beyond the share.

Every iteration the summary reports `anchor.cumulative_share_rows` (anchor
rows, retained and new, over all materialized rows; also when anchors are off
this iteration) with `cumulative_anchor_rows` and `share_tolerance = 0.02`, and
`operator_attention` lists `anchor_share_exceeded` when that share is above
`requested_share + 0.02` (the assembler prints the same line to stderr). Under
`forbid` the share stays at the configured value up to global-batch rounding;
`tests/test_cosmos3_guard_aware_calibration.py` drives three iterations whose
mined candidates run out every time and checks the share never leaves the
tolerance, while `allow` exceeds it in the first iteration.

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

## Zero-new-candidate policy — default `fail_closed`

The materializer (`defect_detection_ablation.py`) verifies
`all_five_maintenance_tasks_present` over the rows it selects for the current
iteration; a missing maintenance task makes the quota manifest unverified, the
CLI exits 2 and `bind_cumulative_manifest` rejects the manifest. In iteration 2
of run 4c-B (2026-09-15) two tasks (Component Classification, Ref_based Defect
Classification) had zero new candidates after history filtering while the
others had 4 / 2 / 1,024 / 529, so the whole iteration failed although the
launch-recorded operator policy was "skip individually exhausted tasks, record
the shortage, continue".

`init_deft_state.py --zero-new-candidate-policy fail_closed|skip_exhausted`
records `config.mining.zero_new_candidate_policy` (plus a
`zero_new_candidate_policy_rule` sentence). The runner request field
`zero_new_candidate_policy` makes `render_iteration_mining_runner.py` append
`--zero-new-candidate-policy <value>` to the materializer command (renderer-owned
when the field is set). With `skip_exhausted`:

- a maintenance task absent from the current selection is acceptable only
  when the routed candidate set of this iteration has **zero eligible rows for
  that task** after history / identity exclusion (counted from the routed
  candidates input: `maintenance_tasks.routed_candidates` before exclusion,
  `maintenance_tasks.eligible_after_exclusion` after previous-row, evaluation
  and duplicate exclusion); a task that had eligible rows but ended up absent
  (for example every row was a near-duplicate) still fails
  (`zero_new_candidate_block_reason = maintenance_task_absent_with_eligible_candidates`);
- the manifest records `exhausted_tasks` (per task: routed, eligible, selected,
  materialized counts), `skipped_tasks` (the exhausted tasks accepted by the
  policy) and the verification key `maintenance_tasks_present_or_exhausted`;
  the raw `all_five_maintenance_tasks_present` fact is kept but listed in
  `verification_policy_exclusions`, so `verified` and the bound v2 manifest
  (`bind_cumulative_manifest`, which copies `exhausted_tasks` / `skipped_tasks`
  under `current_selection`) treat an exhausted task as the only acceptable
  unmet presence;
- it still fails closed when every maintenance task is exhausted
  (`all_maintenance_tasks_exhausted`) or the iteration would add zero new rows
  (a ValueError before the repetition blend, under both policies).

The empty-answer guard cannot make a task exhausted silently: it only removes
empty rows and the standard back-fill draws from a task's own remaining
candidates, so a task with no spare candidates keeps its rows.
