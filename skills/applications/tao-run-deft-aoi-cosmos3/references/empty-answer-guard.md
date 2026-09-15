# Empty-answer guard (Phase 4 step 4c-B) — report always, caps default off

## Why

Measured 2026-09-14 on the anchors run: the final cumulative corpus is 43%
empty-ground-truth rows (the full training pool is 22%), because the detection
calibration quota (512 empty single images + no-change reference pairs per
iteration) and the reference mining amplify a property NVPAW itself does not
have; the corpus holds 10 single-image defect MCQ rows. On the new benchmark
the model answers `[]` on 58% of the single-image MCQ (Full-train 10K, same
prompt templates: 5%). Deleting the "return []" clause from the questions in
an evaluation-only probe removed the empty answers but left accuracy near
guessing (defect MCQ 12% → 23%), so the missing capability is defect-type
classification (`calibration-profile.md`, "Classification calibration"); the
empty-answer share is the symptom this guard caps so it cannot grow again.

## What the assembler reports every iteration

`assemble_summary.json` always carries:

- `answer_profile`: task × format (`BCQ` / `MCQ` / `DET`; `COUNT` for the
  training-only Component Count rows) × image count → `rows`, `empty_rows`,
  `empty_share`, plus `by_task` and a `classification` block (BCQ + MCQ rows
  together and per classification task). *Empty* = the parsed assistant
  answer is `[]`, `{}` or blank after stripping code fences; a BCQ "No" is not
  empty. Formats are read from the native messages (the yes/no sentence
  "Answer with the complete option text" or a leading Yes/No marks BCQ,
  every other classification prompt is MCQ) — nothing is rewritten.
- `answer_profile_new_rows`: the same profile over the rows added this
  iteration (mined, calibration, new anchors / coverage rows).
- `empty_answer_guard`: caps, `before` / `after` shares of the aligned corpus,
  `exceeded_before` / `exceeded_after`, `rows_trimmed_total` /
  `rows_trimmed_by_source` / `rows_trimmed_by_task`, `aligned_rows_before` /
  `aligned_rows_after` / `aligned_rows_shrunk`, `backfilled_rows`, `passes`
  and `status`.

The materializer's quota manifest (`defect_detection_ablation.py`) records
the empty count of the rows it adds as `new_rows_empty` /
`new_rows_empty_by_task` (moved under `current_selection` in the bound v2
manifest); its selection is unchanged.

## Caps (assembler-only options, launch-recorded by `init_deft_state.py`)

| option | scope |
| --- | --- |
| `--max-empty-answer-share 0.30` | whole cumulative corpus |
| `--max-empty-answer-share-task "Defect Detection=0.45"` (repeatable) | that task's rows |
| `--max-classification-empty-share 0.10` | BCQ + MCQ rows together **and** each classification task |
| `--empty-answer-guard-mode enforce\|report` | default `enforce` when any cap is given; without caps the guard only reports |

`init_deft_state.py --max-empty-answer-share / --max-empty-answer-share-task /
--max-classification-empty-share [--empty-answer-guard-mode]` records
`config.mining.empty_answer_guard`; put the same options on every iteration's
selector command and `render_iteration_mining_runner.py` moves them to the
assembler (whitelist; the selector never sees them). Shares are compared
strictly: a share equal to its cap is within the cap.

## Recommended defaults (anchored to the full-pool shares)

Full pool `annotations/mining.jsonl` (972,799 rows): 22.2% empty overall;
Defect Detection 12.8%, Ref_based Defect Detection 26.9%, Ref_based Defect
Classification MCQ 23.5%, Component Classification 9.0%, Component Detection
7.8%, Defect Classification 0%. The anchors run reached 43.1% overall, Defect
Detection 44.5%, Ref_based Defect Detection 56.5%, Ref_based DC MCQ 66.7%.
Recommended caps: overall `<= 0.30`, `Defect Detection <= 0.45`,
`Ref_based Defect Detection <= 0.50`, classification `<= 0.10` (about the pool
share × 1.5 for the detection tasks, the pool's classification share for the
BCQ/MCQ rows). Tighten them only together with a corpus-size decision, because
the caps trim the detection calibration negatives first.

## Order of trimming (enforce mode)

The guard runs **before** the `--max-rows` / `--row-multiple` alignment, on
the candidate set (previous rows + this iteration's mined, calibration, anchor
and coverage rows). Each pass materializes the aligned corpus with the
standard cap step, measures its shares, plans which of its trimmable empty
rows to drop, removes those rows from the candidate set and re-materializes.
The standard selection then back-fills the freed slots from the remaining
(non-empty) mined candidates, so the aligned corpus size stays unchanged
whenever enough candidates remain (`aligned_rows_before == aligned_rows_after`;
`backfilled_rows` = rows that entered the corpus because trimmed rows freed
their slots); it shrinks to the next lower multiple only when candidates run
out (`aligned_rows_shrunk`, reported). Passes are bounded (`passes`, at most
25) and every pass removes at least one row.

Within one pass rows are removed one group at a time, a row only when it
lowers a cap that is still exceeded (the overall cap: any empty row; a task
cap: that task's rows; the classification cap: BCQ/MCQ rows of the union or of
the exceeded task):

1. detection calibration negatives — `deft_calibration` rows with empty
   ground truth whose `deft_calibration_kind` is `detection` (written by the
   materializer), absent or unknown;
2. mined empty rows — current mined rows without a calibration marker.

Never trimmed: previous-iteration rows, anchors (`anchor_correct`), coverage
rows (`coverage_blend`), classification calibration rows
(`deft_calibration_kind = classification`). Rows are removed from the tail of
the current slice so the freshest corrective rows stay in front. The anchor
reservation is re-solved on the trimmed candidate set, so the realized anchor
share follows the (possibly smaller) corpus and spare anchors may back-fill a
leftover slot exactly as in the standard step.

### Status

| `status` | meaning | enforce mode | report mode |
| --- | --- | --- | --- |
| `within_caps` | no cap exceeded (nothing trimmed) | exit 0 | exit 0 |
| `trimmed_to_caps` | caps met after trimming this iteration's empty rows | exit 0 | n/a (no trimming) |
| `exceeded_untrimmable` | a cap is still exceeded but every empty row counting toward it is never-trimmed (previous rows, anchors, coverage, classification calibration) | exit 0, corpus written, residual recorded, stderr note | exit 0, residual recorded |
| `exceeded` | a cap is still exceeded and trimmable empty rows remain (pass budget hit, or report mode with trimmable rows) | exit 2: `assemble_summary.json` written for diagnosis, `train.jsonl` not written | exit 0, corpus unchanged |

`untrimmable_excess` lists every still-exceeded cap with `cap`, `share_after`,
`rows_over_cap` (empty rows that would have to go, without replacement, to
reach the cap), `rows_by_source` (empty rows counting toward the cap by
provenance: `previous_iteration`, `anchor_correct`, `coverage_blend`,
`classification_calibration`, or the trimmable groups) and
`trimmable_rows_left`. 4c-B r3 iteration 2 (2026-09-15) is the motivating
case: Component Classification had no new candidates, its only new rows were
anchors with an empty ground truth, and the 0.10 classification cap could not
be met by trimming; the run now continues with the residual on record.

To stop that excess from arising, anchor selection prefers rows with a
non-empty answer whenever the guard is enabled: `select_anchors` skips
empty-ground-truth candidates (`skipped.empty_ground_truth`, reported as
`anchor.anchor_empty_rows_excluded`, `anchor.prefer_non_empty_rows`), both for
the standard anchor slice and for the alignment fill. Without a cap the anchor
selection is unchanged. Enforcement cannot be combined with the repetition
blend (report mode can).

## Guard-aware calibration and anchor share (Feature B3)

### Worked example: run v12_p4b_emptyguard_r4, iteration 4 (snapshot 6fb5dcbc)

Iteration 4 died in the assembler with `training materialization cannot retain
all previous iteration records and include current Mining data under the
configured cap`. The chain: (1) the single-image tasks were exhausted, so the
1,536 new rows were almost all detection calibration rows (Defect Detection
1,024 incl. 512 empty boards, Ref_based Defect Detection 509 at the KPI empty
rate, Ref_based Defect Classification 35); (2) the cumulative corpus already sat
at 0.29 empty share against the 0.30 cap, so the guard trimmed most of this
iteration's empty rows; (3) the surviving current rows plus the available
anchors were fewer than 768, the 768 row-multiple rounding dropped the
materialized size back to the prior corpus, the current capacity became 0 and
the assembler failed closed. Separately, the guard back-fill of iterations 2–3
drew extra anchors through the `leftover -> extra_anchors` path whenever the
mined candidates ran out: 230 / 478 / 386 anchors per iteration, 1,094 of 5,376
rows = 20.4% against the configured 0.10 share, a confound for the guard arm.

Two mechanisms, one launch-recorded option group (defaults flip only when the
guard is enabled, so runs without the guard reproduce byte-for-byte; the golden
fixture `tests/fixtures/guard_aware_calibration_golden.json` proves it):

1. **Guard-aware detection calibration selection** (materializer
   `defect_detection_ablation.py`, `--calibration-guard-aware on`): before the
   feasibility loop the materializer computes, under every applicable cap
   (overall, per-task for Defect Detection and Ref_based Defect Detection), the
   *empty-row headroom* `floor(cap × rows − empty_rows)` over the cumulative
   corpus (`--previous-jsonl`) plus this iteration's non-calibration rows, with
   the calibration slot's row count already added to `rows`. It selects at most
   that many empty calibration rows (per-task caps first, the shared overall
   headroom split by largest remainder) and fills the rest of the slot with
   few-box rows from the same source, same ordering, same per-source rules; the
   slot's row count (512 + 512 single-image, 500 reference pairs) is unchanged,
   only the empty / few-box split moves. Rule details: `calibration-profile.md`,
   "Empty / few-box substitution under the empty-answer guard".
2. **No anchor over-fill** (assembler `--anchor-overfill forbid`, the default
   under the guard): the leftover fill may not draw anchors beyond their share
   (the `extra_anchors` step is dropped; `extra_coverage` and the mined back-fill
   stay). When the candidates are short, the aligned size shrinks to the largest
   row-multiple the retained rows, the share-bound anchors, the coverage rows
   and the current candidates fill; the iteration fails closed only when the
   growth would be zero rows, with a message naming the shortage
   (`current_rows_after_guard`, `anchor_slots`, `coverage_slots`,
   `row_multiple`, `previous_rows`, `aligned_rows`). Rule details:
   `anchor-and-coverage.md`, "Share ceiling under the empty-answer guard".

Synthetic reproduction of the r4 numbers
(`tests/test_cosmos3_guard_aware_calibration.py`): a 5,376-row corpus at
0.294 empty share (Defect Detection 880 / 2,000 empty, Ref_based Defect
Detection 700 / 1,500), an iteration of 1,024 single-image calibration rows,
500 reference pairs, 9 mined pairs (5 empty) and 3 Ref_based Defect
Classification rows, caps 0.30 / 0.45 / 0.50. Headroom: overall
`floor(0.30 × 6,912 − 1,585) = 488`, Defect Detection
`floor(0.45 × 3,024 − 880) = 480`, Ref_based Defect Detection
`floor(0.50 × 2,009 − 705) = 299`; the shared 488 splits 312 / 176, so 200
single-image and 95 reference empties become few-box rows. The assembler then
keeps every one of the 1,536 rows (`growth_rows = 1536`,
`empty_answer_guard.status = within_caps`, overall share 2,073 / 6,912). With
guard-aware selection off the same iteration must trim 421 calibration
negatives, the aligned size falls by one global batch and the 1,103 protected
calibration rows left no longer fit its 768 slots: before Feature B4 the
assembler failed closed as r4 did; now 347 more Defect Detection negatives
yield to the growth slot and the step grows by 768 rows (see "Calibration
yields to the growth slot").

### Worked example: run v12_p4b_emptyguard_r5, iteration 1 (snapshot 3c4e042b) — Feature B3.1

With guard-aware selection on, the headroom lowered the empty targets to 426
single-image / 151 no-change (KPI 512 / 278) and the materializer asked for 86
few-box rows and 127 changed pairs in their place. The calibration FEED had no
reserve: the run-local runtime had called
`select_detection_calibration.select_calibration` with the hard-coded
`cohort_bucket_quotas` 512 / 512 and 278 / 222, so every few-box (512) and
changed (222) candidate was already in the slot. Result: `fewbox_shortfall`
86 / 127, slots filled 938 / 1,024 and 373 / 500, "Defect Detection
materialization quota is not verified" (exit 2), no assembly, run aborted under
criterion (d), while 86 unused EMPTY candidates sat in the feed.

Two mechanisms, both only with `--calibration-guard-aware on` (guard-off
selection stays byte-identical; the same golden fixture proves it):

1. **Contract-driven feed reserve.** `init_deft_state.py` writes
   `config.mining.calibration_quota_contract.feed_bucket_quotas`: with
   guard-aware on `non_reference_based = {empty: max_empty, few: max_empty +
   max_few_box}` and `reference_based = {empty: KPI no-change, few: total}`
   (512 / 1,024 and 278 / 500 for the pinned recipe), so the whole slot can be
   filled with non-empty rows when the headroom demands it; with guard-aware
   off the block equals the contract quotas (512 / 512 and 278 / 222). The
   runtime passes it as `select_calibration(feed_bucket_quotas=...)`; the
   contract quotas stay the fail-closed floor, the reserve is best-effort and
   reported per cohort as `feed_reserve_rows`. Rule and controller
   instruction: `calibration-profile.md`, "Feed reserve".
2. **Best-effort substitution with recorded overflow.** When the few-box /
   changed reserve runs out, the materializer fills the remaining slot rows
   with the next EMPTY calibration candidates of the same ordering (never more
   than the KPI-rate selection carried) instead of leaving the slot short. The
   slot is full, the total-slot verification passes, and the manifest records
   `calibration_headroom_overflow_rows` (per task + `total`),
   `fewbox_shortfall` (few-box / changed rows asked for and not found), an
   honest `calibration_fewbox_substituted` (rows actually replaced) and
   `guard_aware_calibration.status = substituted_with_overflow`. The
   assembler's guard then trims the overflow empties as calibration negatives.
   A substitution shortfall never fails the materializer by itself; a slot the
   feed cannot fill at all still fails closed.

Synthetic reproduction (`FeedReserveAndOverflowTests` in
`tests/test_cosmos3_guard_aware_calibration.py`): a 6,144-row corpus (8 global
batches) at 0.280 empty share, the r4-shaped iteration with today's feed (512
few-box, 222 changed) or the reserve feed (1,024 / 500), caps 0.30 / 0.45 /
0.50, KPI reference empty rate 0.556. Headroom: overall
`floor(0.30 × 7,680 − 1,727) = 577`, Defect Detection 480, Ref_based Defect
Detection 170; the shared 577 splits 426 / 151.

| feed | empty selected | substituted | overflow | `status` | materializer | assembler, 768-row batch |
|---|---|---|---|---|---|---|
| today's (512 / 222) | 512 / 278 | 0 / 0 | 86 / 127 | `substituted_with_overflow` | verified, 1,536 rows | guard trims 335 calibration negatives (every trimmed row also shrinks the denominator; no back-fill candidates), then the 1,189 protected calibration rows left do not fit the one 768-row batch that remains: before Feature B4 this failed closed (`cannot retain the calibration rows under the configured cap`); now 433 more Defect Detection negatives yield, `growth_rows = 768`, overall share 0.253, `calibration_yielded` |
| reserve (1,024 / 500) | 426 / 151 | 86 / 127 | 0 / 0 | `substituted` | verified, 1,536 rows | `within_caps`, `growth_rows = 1536`, overall share 0.300 |

The overflow turns the r5 materializer abort into a recorded, inspectable
outcome; only the reserve feed lets the iteration through. Put
`feed_bucket_quotas` from the state on the runtime's `select_calibration` call
(the r6 prompt says so) and read `guard_aware_calibration.status` in every
quota manifest.

### Calibration yields to the growth slot (Feature B4)

#### Worked example: run v12_p4b_emptyguard_r6, iteration 2 (snapshot bab8537b)

Iteration 2 had the three single-image tasks exhausted (new candidates:
Component Detection 6, Defect Detection 1,536, Ref_based Defect Detection 807,
others 0). The materializer emitted 1,536 verified rows: 1,524 detection
calibration rows (B3.1 substitution 177 / 96, overflow 0) and about 221 mined
rows. In the assembler, after the duplicate rule (1,387 current rows, 1,166
protected calibration rows) and the guard's empty trimming, prior 2,304 +
current + anchors fell below 3,840, so the 768-row alignment rounded the corpus
down to 3,072: current slot 768, `anchor_slots` 77 (0.10 share), `current_limit`
691. `materialize_capped` then hit `len(protected) > current_limit` (1,077 >
691). Its only remedy was the round-up anchor fill, which `--anchor-overfill
forbid` disables, so it raised `training materialization cannot retain the
calibration rows under the configured cap` and the run aborted; r4 had passed
this point only through the anchor over-fill (the 20.4% anchors confound). The
contradiction: a fixed detection calibration quota (1,024 single-image + 500
reference pairs) plus the empty-share caps plus exhausted single-image mining
plus a 10% anchor ceiling do not fit one growth step. Something must yield, and
the rule (DECISIONS 2026-09-15) is that the calibration quota is a fraction of
the growth slot, not the other way round.

#### Rule (guard on, i.e. `--anchor-overfill forbid`; guard-off behaviour is byte-identical)

When the protected calibration rows alone exceed the current slot, the pass
does not raise. Inside the same capped pass that reserves the anchor and
coverage slots:

1. **Mined rows first.** Every trimmable mined current row that fits is kept;
   when the mined rows alone exceed the slot, the existing task-balanced mined
   trim applies and every detection calibration row is dropped.
2. **Calibration fills the rest**, dropping rows in this order: empty-ground-
   truth calibration rows first (detection negatives / no-change pairs), then
   non-empty rows (few-box boards / changed pairs). Inside each bucket the
   kept rows are split over Defect Detection and Ref_based Defect Detection
   proportionally to their counts (largest remainder, the materializer's
   `allocate_empty_headroom`), so neither task loses everything while the
   other keeps all; each task keeps a prefix of its rows in materializer order
   (the tail goes, like the guard's trim plan). Classification calibration
   rows (4c-A) stay protected as before.
3. **Anchors stay exactly at the share**: no round-up, no over-fill; the
   aligned size is unchanged by the yield.
4. **Zero growth still fails closed.** When the aligned size minus the previous
   rows is below one `--row-multiple` (the iteration would add fewer than one
   global batch), the existing zero-growth message with its numbers is raised
   (`current_rows_after_guard`, `anchor_slots`, `coverage_slots`,
   `row_multiple`, `previous_rows`, `aligned_rows`).

The guard loop is unchanged: it measures the materialized (post-yield) corpus,
so `empty_answer_guard.after` describes the final rows, and every trim pass
strictly shrinks the candidate set, so the loop converges. Because the yield
drops empties first, the rows it removes are the rows the guard would have
trimmed; when the guard still trims after a yield (the slot could only be
filled with more empties than the caps allow), the loop ends in the
zero-growth failure or in `exceeded`, which is the correct fail-closed reading
of a step that cannot be filled within the caps.

Synthetic reproduction (`CalibrationYieldTests` in
`tests/test_cosmos3_guard_aware_calibration.py`; the fixture keeps 240 retained
anchors so the same inputs exercise both paths): prior 2,304 rows (450 empty),
1,387 current rows = 221 mined non-empty rows + 1,166 detection calibration
rows (500 empty: 350 boards / 150 no-change; 666 non-empty: 450 few-box / 216
changed), 200 anchor candidates, caps 0.30 / 0.45 / 0.50.

| `--anchor-overfill` | aligned | anchors new | current slot | mined kept | calibration kept | dropped (`by_kind`, `by_task`) | `growth_rows` |
|---|---|---|---|---|---|---|---|
| `forbid` (guard on) | 3,072 | 67 (share) | 701 | 221 | 480 = 324 few-box + 156 changed | 686 = empty 500 + non-empty 186; DD 476 / Ref DD 210 | 768 |
| `allow` (guard off, HEAD) | 3,840 (round-up) | 149 (67 + 76 spare + 6 `alignment_fill`) | 1,387 | 221 | 1,166 | 0 | 1,536 |

With the guard the corpus stays within the caps in one pass (overall share 450
/ 3,072); `operator_attention` lists `calibration_yielded`.

#### Fields added (assembler summary)

`calibration_rows_dropped_for_cap = {"by_task": {task: n}, "by_kind":
{"empty": n, "non_empty": n}, "total": n}` (this iteration's detection
calibration rows that yielded; zeros when nothing yielded),
`calibration_rows_kept = {task: n}` (calibration rows of this iteration in the
corpus, per task), `calibration_yielded` (bool) and the `operator_attention`
entry `calibration_yielded` when `total > 0` (informational; printed to stderr
like `anchor_share_exceeded`, with the dropped counts).
`anchor.cap_reservation.alignment_policy` reads
`round_down_calibration_yields_to_growth_slot` for a yielding pass;
`calibration_rows_protected` keeps counting the calibration rows offered.
`growth_rows` and `aligned_rows_shrunk` keep their meaning.

### Launch record and pass-through

`init_deft_state.py --max-empty-answer-share ... [--calibration-guard-aware
on|off] [--anchor-overfill allow|forbid]` records
`config.mining.empty_answer_guard.calibration_guard_aware` (default `true`
when any cap is given, else `false`) and `.anchor_overfill` (default `forbid`
when any cap is given, else `allow`); `--calibration-guard-aware on` without a
cap is rejected. Put the recorded values on every iteration's selector command
next to the caps: `render_iteration_mining_runner.py` moves `--anchor-overfill`
to the assembler, passes `--calibration-guard-aware` to **both** the
materializer and the assembler and, when it is `on`, mirrors
`--max-empty-answer-share` / `--max-empty-answer-share-task` to the
materializer as read-only copies (plan invariant
`calibration_guard_caps_mirrored_to_materializer`; the guard mode and the
classification cap stay assembler-only). The assembler cross-checks the
current quota manifest's `calibration_guard_aware` against the launch value
(its default is on when a cap is given) and fails closed before writing
anything when they disagree, so a selector command that forgot the flag cannot
silently produce a non-guard-aware corpus in a guard-aware run.

### Fields added

Materializer quota manifest (copied under `current_selection` in the bound v2
manifest): `calibration_guard_aware` (bool), `calibration_empty_headroom`
(`overall`, `task:Defect Detection`, `task:Ref_based Defect Detection`; `null`
for caps not configured or when off), `calibration_empty_selected` (per task +
`total`), `calibration_fewbox_substituted` (per task + `total`),
`guard_aware_calibration` (caps, KPI empty targets, the ledger of previous /
current non-calibration rows, `fewbox_shortfall`); `single_image_calibration`
gains `max_few_box_effective` and `empty_substituted_by_few_box`;
`reference_calibration` gains `kpi_target_no_change` and
`no_change_substituted_by_changed` (`target_no_change` is the guard-aware
target the content gate verifies).

Feature B3.1 additions. Materializer quota manifest (also copied under
`current_selection`): `calibration_headroom_overflow_rows` (per task +
`total`; empties kept beyond the headroom because the reserve ran out);
`guard_aware_calibration.status` (`no_substitution_needed` | `substituted` |
`substituted_with_overflow`; `null` when off),
`guard_aware_calibration.headroom_empty_targets` (the per-task empty targets
after the headroom split; `null` when off),
`guard_aware_calibration.substitution` (policy name);
`single_image_calibration.empty_beyond_headroom` and
`reference_calibration.no_change_beyond_headroom` (the per-slot overflow;
`target_no_change` counts the kept no-change pairs, headroom target plus
overflow). Selector summary (`detection_calibration_v3`, per cohort):
`feed_bucket_quotas` and `feed_reserve_rows` (`empty` / `few` / `total` rows
selected beyond the contract). State:
`config.mining.calibration_quota_contract.feed_bucket_quotas`,
`.feed_bucket_quotas_rule` and `.owner`.

Assembler summary: `growth_rows` (output − previous rows),
`anchor.cumulative_share_rows` (anchor rows over all rows, every iteration) with
`anchor.cumulative_anchor_rows` and `anchor.share_tolerance` (0.02),
`operator_attention` (a list; `anchor_share_exceeded` when the cumulative share
is above `requested_share + 0.02`, also printed to stderr),
`empty_answer_guard.calibration_guard_aware` and `empty_answer_guard.anchor_overfill`,
`anchor.cap_reservation.anchor_overfill` / `leftover_fill_anchors` /
`aligned_rows_shrunk_for_share`.

## Summary schema (`empty_answer_guard`)

```json
{
  "enabled": true, "mode": "enforce",
  "caps": {"overall": 0.30, "per_task": {"Defect Detection": 0.45}, "classification": 0.10},
  "empty_definition": "...", "trim_order": ["detection_calibration_negative", "mined_empty"],
  "never_trimmed": ["previous_iteration", "anchor_correct", "coverage_blend", "classification_calibration"],
  "before": {"rows": 8448, "empty_rows": 3637, "overall_share": 0.4305,
             "per_task": {"Defect Detection": 0.445}, "classification_share": 0.08,
             "classification_per_task": {"Component Classification": 0.137}},
  "after": {"...": "same keys"},
  "exceeded_before": ["overall", "task:Ref_based Defect Detection"], "exceeded_after": [],
  "policy": "trim_empty_candidates_before_alignment_backfill_from_remaining_candidates",
  "rows_trimmed_total": 1200,
  "rows_trimmed_by_source": {"detection_calibration_negative": 900, "mined_empty": 300},
  "rows_trimmed_by_task": {"Defect Detection": 700, "Ref_based Defect Detection": 500},
  "aligned_rows_before": 2304, "aligned_rows_after": 2304, "aligned_rows_shrunk": 0,
  "backfilled_rows": 1200, "passes": 2,
  "untrimmable_excess": {"classification_task:Component Classification": {
      "cap": 0.10, "share_after": 0.18, "rows_over_cap": 4,
      "rows_by_source": {"anchor_correct": 6, "previous_iteration": 3}, "trimmable_rows_left": 0}},
  "status": "exceeded_untrimmable"
}
```
