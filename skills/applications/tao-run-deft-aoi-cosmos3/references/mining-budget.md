# Mining budget (Phase 5-S) — Defect Detection fraction, mined per-task pool caps, fill order — defaults off

## Why

Phase 5-S scales the mining budget from 0.87% of the Mining pool (8,448 rows over
five iterations) to about 10% (three iterations of 32,256 new rows = 96,768). The
materializer (`defect_detection_ablation.py`) used to fill each iteration's target
with a hard-coded 0.5 lower bound for single-image Defect Detection and then fill
the maintenance tasks by availability, round-robin. At 32k rows per iteration that
breaks twice:

1. the single-image Defect Detection pool holds 21,849 rows, so a 50% reserve
   (16,128 rows per iteration) is infeasible;
2. with no per-task ceiling the fill drifts to Ref_based Defect Classification
   (827,493 rows, 85% of the pool) while the thin single-image tasks (Defect
   Classification 35,648; Component Detection 12,424; Component Classification
   10,851) are where DEFT is weakest.

Decision (Sean Lin, 2026-09-17): the task mix is controlled by per-task caps
expressed as a fraction of each task's own pool, single-image tasks are filled
first, and the Defect Detection fraction is a launch-recorded option. All three
are launch-recorded; their defaults reproduce the parent snapshot byte-for-byte
(`tests/test_cosmos3_mined_task_pool_caps.py`, golden test).

## The three options

| init flag (`init_deft_state.py`) | state path (`config.mining.*`) | runner request field | materializer flag |
|---|---|---|---|
| `--defect-detection-fraction F` (default 0.5, (0, 1]) | `defect_detection_fraction` (+ `_rule`) | `defect_detection_fraction` | `--defect-detection-fraction F` |
| `--mined-task-pool-cap "TASK=FRACTION"` (repeatable; six task types; (0, 1]) | `mined_task_pool_caps` = `{task: fraction}` (+ `_rule`; `{}` = uncapped) | `mined_task_pool_caps` (object) | `--mined-task-pool-cap "TASK=FRACTION"` per task |
| `--mined-task-fill-order "T1,T2,..."` (maintenance tasks only, no repeats) | `mined_task_fill_order` = `[T1, T2, ...]` (+ `_rule`; `[]` = today's round-robin) | `mined_task_fill_order` (list) | `--mined-task-fill-order "T1,T2,..."` |

`render_iteration_mining_runner.py` appends the flags to the materializer command
whenever the request field is set (an empty cap object / empty order adds no
flag) and records the three values in the plan; when a field is set the caller
must not pass that flag itself (renderer-owned, like `zero_new_candidate_policy`).
Build the request from the state: copy the three `config.mining` values into the
request fields of the same names.

## Cap semantics

- **Base.** `pool_rows` = the task's row count in the canonical Mining pool, the
  `--source-annotations` file the materializer already reads.
  `cap_rows = floor(pool_rows * fraction)`.
- **Cumulative over iterations.** `used_before` = the task's *mined* rows already
  in the cumulative Train JSONL (`--previous-jsonl`, the history the materializer
  already receives). The remainder for this iteration is
  `max(0, cap_rows - used_before)`; the materializer never selects more mined
  rows of the task than the remainder.
- **Mined rows only.** Calibration rows (`deft_calibration`), anchors
  (`deft_anchor`) and coverage-blend rows (`deft_coverage`) are not mined rows:
  they neither count in `used_before` nor in `selected_now`. Under the fixed
  calibration slot the Defect Detection cap bounds the task-strict (mined) rows
  only; under the proxy-rate policy (no fixed slot) it bounds every selected
  Defect Detection row (conservative, never above the cap).
- **Slots flow on.** A task at its remainder frees its slots to the next task in
  the fill order and then to the round-robin tasks, so the iteration's target row
  count is preserved whenever candidates exist. A cap that leaves Defect
  Detection below its fraction lower bound makes the target shrink to the largest
  feasible batch-aligned size (or fails closed without `--minimum-rows`), exactly
  as any other shortage does.
- **Capped, not exhausted.** `mined_task_pool_usage[task].capped_this_iteration`
  is true when the remainder is used up (before or by this selection) and the
  task is listed in `capped_tasks`. `exhausted_tasks`, `skipped_tasks` and the
  zero-new-candidate policies are unchanged: a task whose cap is fully consumed
  is *absent* from later selections while it still has eligible candidates, and
  the unchanged policies treat it like any other absent task (`fail_closed`:
  `policy_fail_closed`; `skip_exhausted`:
  `maintenance_task_absent_with_eligible_candidates`). Size the caps so that the
  last planned iteration still has a remainder for every task you cap, or treat a
  populated `capped_tasks` as the planned end of the run. The empty-answer guard
  (B3/B3.1/B4) and the anchors are untouched and still run after materialization.

## Fill order

Within one iteration the materializer fills:

1. Defect Detection to its fraction lower bound (`ceil((target - acquisition_rows) * F)`;
   it may exceed the bound only when the maintenance tasks cannot fill the batch);
2. one mined row of every maintenance task that has an eligible, uncapped
   candidate (so a later task is never starved by the priority tasks and the
   presence policies stay satisfiable — the materializer's later trim of the
   selection to the accepted target removes rows from the tail);
3. the tasks of `mined_task_fill_order`, in that order, each up to
   `min(eligible novel candidates, cap remainder)`;
4. the remaining maintenance tasks round-robin in the materializer's fixed task
   order (today's behaviour), each still bounded by its cap remainder.

An empty fill order skips steps 2 and 3, which is today's round-robin. The
reserved reference calibration pairs are selected before and outside this fill
and are unaffected.

## Records (quota manifest, `defect_detection_quota_manifest_v1` / v2)

| field | meaning |
|---|---|
| `defect_detection_fraction` | the lower bound used (also `configuration.defect_detection_minimum_fraction`, `row_counts.defect_detection_minimum_target`) |
| `mined_task_pool_caps` | the input caps `{task: fraction}` |
| `mined_task_pool_cap_policy` | `floor_of_task_mining_pool_rows_times_fraction_cumulative_over_iterations_mined_rows_only` |
| `mined_task_pool_usage` | per task (all six): `pool_rows`, `cap_fraction`, `cap_rows`, `used_before`, `selected_now`, `remaining_after`, `capped_this_iteration` (`null` cap fields for uncapped tasks) |
| `mined_task_fill_order` | the input order (`[]` = round-robin) |
| `mined_task_fill_realized` | mined rows selected this iteration per task (all six; calibration rows excluded) |
| `capped_tasks` | tasks with `capped_this_iteration` true |
| `verification.mined_task_pool_caps_respected` | no task selected more mined rows than its remainder (part of `verified`) |

`bind_cumulative_manifest` copies `mined_task_pool_usage`,
`mined_task_fill_realized` and `capped_tasks` into `current_selection` of the v2
manifest like the other current-selection facts.

## Worked example — Phase 5-S

Pool sizes (2026-09-17): Defect Detection 21,849; Defect Classification 35,648;
Component Detection 12,424; Component Classification 10,851; Ref_based Defect
Classification 827,493; Ref_based Defect Detection: read `pool_rows` from the
manifest. Growth 32,256 rows per iteration, three iterations (96,768 rows), the
pinned fixed calibration slot (1,024 single-image + 500 reference pairs per
iteration) included in every target.

```bash
python3 "$SKILL_ROOT/scripts/init_deft_state.py" ... \
  --defect-detection-fraction 0.15 \
  --mined-task-pool-cap "Defect Classification=0.6" \
  --mined-task-pool-cap "Defect Detection=0.6" \
  --mined-task-pool-cap "Component Detection=0.6" \
  --mined-task-pool-cap "Component Classification=0.6" \
  --mined-task-pool-cap "Ref_based Defect Detection=0.4" \
  --mined-task-fill-order "Defect Classification,Component Detection,Component Classification"
```

| task | pool_rows | cap_fraction | cap_rows (whole run) | if spread over 3 iterations |
|---|---:|---:|---:|---:|
| Defect Detection | 21,849 | 0.6 | 13,109 | 4,369 |
| Defect Classification | 35,648 | 0.6 | 21,388 | 7,129 |
| Component Detection | 12,424 | 0.6 | 7,454 | 2,484 |
| Component Classification | 10,851 | 0.6 | 6,510 | 2,170 |
| Ref_based Defect Detection | pool_rows | 0.4 | floor(0.4 * pool_rows) | cap_rows / 3 |
| Ref_based Defect Classification | 827,493 | uncapped | — | the remainder |

Iteration 1 (no `used_before`): Defect Detection fills to
`ceil(32,256 * 0.15) = 4,839` rows (1,024 calibration + 3,815 mined). The other
27,417 slots hold the 500 reference calibration pairs and 26,917 mined
maintenance rows: one row per task, then Defect Classification up to
`min(routed candidates, 21,388)`, Component Detection up to 7,454, Component
Classification up to 6,510, then Ref_based Defect Classification / Ref_based
Defect Detection round-robin (the latter up to its 0.4 remainder). Iteration 2
subtracts the iteration-1 mined rows per task (`used_before`) from every cap
before selecting; iteration 3 again. Over the run the single-image caps bound the
mined Defect Detection rows at 13,109 (0.15 is feasible: 3 x 3,815 = 11,445) and
the three single-image maintenance tasks at 35,352 rows together; everything
above that goes to the uncapped Ref_based Defect Classification. Read
`mined_task_pool_usage` after every iteration; when a task shows
`remaining_after = 0` it will be absent next iteration and the unchanged
presence policy blocks (see "Capped, not exhausted"), so the third iteration is
the last one the caps above can carry when the routing offers more candidates
than the remainders.

A default launch (`--defect-detection-fraction 0.5`, no caps, no fill order)
records `defect_detection_fraction = 0.5`, `mined_task_pool_caps = {}`,
`mined_task_fill_order = []`, and the materializer's selection is byte-identical
to the parent snapshot (golden test on the guard-aware fixture and on the Phase
5-S synthetic pool).
