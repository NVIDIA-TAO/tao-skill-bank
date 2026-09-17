# Mining budget (Phase 5-S) — Defect Detection fraction, mined per-task pool caps, fill order, cross-task visual de-duplication switch — defaults off

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
(`tests/test_cosmos3_mined_task_pool_caps.py`, golden test). Feature P5-S.1 adds
a fourth launch-recorded switch, the cross-task visual de-duplication (see the
section of that name): the first Phase 5-S run showed the materializer's
cross-task image exclusion starving the very tasks the caps and fill order aim at.

## The launch-recorded options

| init flag (`init_deft_state.py`) | state path (`config.mining.*`) | runner request field | materializer flag |
|---|---|---|---|
| `--defect-detection-fraction F` (default 0.5, (0, 1]) | `defect_detection_fraction` (+ `_rule`) | `defect_detection_fraction` | `--defect-detection-fraction F` |
| `--mined-task-pool-cap "TASK=FRACTION"` (repeatable; six task types; (0, 1]) | `mined_task_pool_caps` = `{task: fraction}` (+ `_rule`; `{}` = uncapped) | `mined_task_pool_caps` (object) | `--mined-task-pool-cap "TASK=FRACTION"` per task |
| `--mined-task-fill-order "T1,T2,..."` (maintenance tasks only, no repeats) | `mined_task_fill_order` = `[T1, T2, ...]` (+ `_rule`; `[]` = today's round-robin) | `mined_task_fill_order` (list) | `--mined-task-fill-order "T1,T2,..."` |
| `--cross-task-visual-dedup on\|off` (default `on`; Feature P5-S.1) | `cross_task_visual_dedup` = `"on"` / `"off"` (+ `_rule`) | `cross_task_visual_dedup` (string) | `--cross-task-visual-dedup on\|off` |

`render_iteration_mining_runner.py` appends the flags to the materializer command
whenever the request field is set (an empty cap object / empty order adds no
flag) and records the four values in the plan; when a field is set the caller
must not pass that flag itself (renderer-owned, like `zero_new_candidate_policy`).
Build the request from the state: copy the four `config.mining` values into the
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
  task is listed in `capped_tasks`. A task whose cap was consumed *before* this
  selection is absent while it still has eligible candidates; it is recorded in
  `capped_absent_tasks` (per task: routed, eligible, cap_rows, used_before,
  selected, materialized), never in `exhausted_tasks` / `skipped_tasks`. Under
  `--zero-new-candidate-policy skip_exhausted` such a task is accepted like an
  exhausted one: the binding verification key is
  `maintenance_tasks_present_or_exhausted_or_capped` (the older
  `maintenance_tasks_present_or_exhausted` stays a raw fact and is listed in
  `verification_policy_exclusions`), and the CLI line reports `capped_tasks=[...]`.
  `fail_closed` still fails on any absent task (`policy_fail_closed`), and every
  maintenance task absent still fails under both policies
  (`all_maintenance_tasks_exhausted_or_capped`). The empty-answer guard
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
| `cross_task_visual_dedup` | `on` / `off` (Feature P5-S.1; see "Cross-task visual de-duplication") |
| `maintenance_rows_unlocked_by_cross_task` | when `off`: `{task: n}` maintenance rows per task that today's cross-task exclusion would have dropped; `null` when `on` |
| `uniqueness.visual_identity_scope` | `all_tasks` (`on`) or `within_task` (`off`): the scope of `verification.unique_target_images` / `unique_image_content` / `near_duplicate_free` |

`bind_cumulative_manifest` copies `mined_task_pool_usage`,
`mined_task_fill_realized`, `capped_tasks`, `cross_task_visual_dedup` and
`maintenance_rows_unlocked_by_cross_task` into `current_selection` of the v2
manifest like the other current-selection facts.

## Cross-task visual de-duplication (Feature P5-S.1)

**Why.** Run `v12_p5s_pool10_r8` iteration 1: the router routed 858 candidate
images to Defect Classification (842 of them also to Defect Detection), yet the
materializer reported `maintenance_marginal_quota.available["Defect Classification"] = 19`
and materialized 19 Defect Classification rows (r7 iteration 1: 275 routed, 10
available). The materializer de-duplicated the maintenance rows against every
image already selected for Defect Detection (and for the other maintenance
tasks). In the NVPAW pool the single-image Defect Classification (MCQ) rows and
the Defect Detection rows are asked on the SAME board images, so Defect Detection
consumed the images first and Defect Classification starved; the same happens
between Component Classification and Component Detection. For Phase 5-S the
single-image MCQ tasks are the target, so this exclusion defeated the per-task
caps and the fill order.

**Semantics.**

- `on` (default): today's rule, byte-identical to the parent snapshot. A
  maintenance-task row is dropped when its image path / content (or, with the
  near-duplicate filter, its perceptual hash) was already selected for any task,
  Defect Detection or another maintenance task.
- `off`: visual de-duplication still applies WITHIN each task type (no two rows
  of one task on the same image content), and the exact-record, previous-record
  and benchmark / proxy leakage exclusions are unchanged. A maintenance row is no
  longer dropped because its image was selected for a DIFFERENT task: every
  task's exclusion set holds only its own already-selected rows (Defect
  Detection selections do not exclude Defect Classification / Component
  Classification / Component Detection / reference rows; reference-pair tasks
  keep their pair identity). The guard-aware calibration re-selection (Feature
  B3) applies the same same-task rule, so a Defect Classification row on a
  few-box board no longer evicts that few-box calibration row. The novel-image
  accounting (`novel_image_limit`) counts selected rows as today.
- Under `off` the verification keys `unique_target_images`,
  `unique_image_content` and `near_duplicate_free` are evaluated within each task
  type (`uniqueness.visual_identity_scope = within_task`); `uniqueness.unique_images`
  and `uniqueness.unique_image_content` stay the counts over all emitted rows, so
  `row_counts.total - uniqueness.unique_images` is the number of rows that share
  an image with a row of another task.

**Records.** `cross_task_visual_dedup` (`on` / `off`) and, when `off`,
`maintenance_rows_unlocked_by_cross_task` = `{task: n}`: the rows per maintenance
task that survived only because the cross-task exclusion was off (the same
candidates are also run through today's shared exclusion and the difference is
recorded; `null` when `on`). Both are copied into `current_selection` of the v2
manifest.

**Rule.** Phase 5-S runs launch with `--cross-task-visual-dedup off`: the
single-image MCQ tasks are the target and the caps / fill order steer the mix,
so the cross-task exclusion only starves them. Runs that must keep every image
unique across all tasks (the pre-Phase-5-S Defect Detection ablations) keep the
default.

Synthetic check (`tests/test_cosmos3_cross_task_visual_dedup.py`): a pool where
every board carries one Defect Detection and one Defect Classification row (12
boards plus one solo Defect Classification row). `on` leaves
`available["Defect Classification"] = 1` and the batch shrinks to 24 rows; `off`
gives 13 (= routed), `maintenance_rows_unlocked_by_cross_task["Defect Classification"] = 12`,
the 30-row target is reached and the manifest verifies. Without shared images
`off` selects exactly the rows `on` selects and unlocks nothing.

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
  --mined-task-fill-order "Defect Classification,Component Detection,Component Classification" \
  --cross-task-visual-dedup off
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
`remaining_after = 0` it will be absent next iteration, which `skip_exhausted`
accepts and records in `capped_absent_tasks` while `fail_closed` blocks (see
"Capped, not exhausted"), so launch Phase 5-S with
`--zero-new-candidate-policy skip_exhausted`.

Read `maintenance_rows_unlocked_by_cross_task` next to `mined_task_pool_usage`
after every iteration: it is the Defect Classification / Component
Classification / Component Detection availability the `off` switch restored
(r8 iteration 1 would have shown about 840 Defect Classification rows).

A default launch (`--defect-detection-fraction 0.5`, no caps, no fill order,
`--cross-task-visual-dedup on`) records `defect_detection_fraction = 0.5`,
`mined_task_pool_caps = {}`, `mined_task_fill_order = []`,
`cross_task_visual_dedup = "on"`, and the materializer's selection is
byte-identical to the parent snapshot (golden test on the guard-aware fixture and
on the Phase 5-S synthetic pool).
