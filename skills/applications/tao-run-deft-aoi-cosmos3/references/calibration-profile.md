# Profile-matched calibration (`kpi_profile_count_bins`) — default off

## Why

Calibration rows are the loop's only lever on *what a detection answer looks
like* when the mined rows are all hard positives. The fixed buckets (512 empty
+ 512 rows with <= 2 boxes per iteration for single-image Defect Detection;
reference pairs split into no-change / <= 2-box changed at the KPI empty rate)
teach a box-count distribution the evaluation never asks for. Measured on the
frozen NVPAW benchmark vs. the training sets of finished runs (tool A profiles,
2026-09-10):

| task | benchmark empty | benchmark rows with >= 4 boxes | training set (v3b / v9b-redo) empty | training rows with >= 4 boxes |
|---|---:|---:|---:|---:|
| Defect Detection | 24.8% | 19% | 68.5% / 44.8% | 1% / 0.3% |
| Ref_based Defect Detection | 54.2% | 19% | 48.9% / 42.3% | 14% / 0.9% |

The two-image failure the image review found ("we find one defect and stop")
is the same mismatch: changed pairs in training carry one box, the benchmark's
carry 3.4 on average.

## What the policy does

`init_deft_state.py --calibration-task-total "Defect Detection=1024"
--calibration-task-total "Ref_based Defect Detection=500" [--calibration-min-fill 0.9]`
records `config.mining.calibration_quota_contract` with
`policy = kpi_profile_count_bins`:

- `task_profiles`: for every detection task, the KPI set's rows per
  ground-truth box-count bin `0 / 1 / 2-3 / 4-9 / 10+` and their shares
  (`select_detection_calibration.derive_task_count_profiles`);
- `task_bin_quotas`: each task total split across the bins by largest
  remainder (`profile_bin_quotas`);
- `min_fill_fraction`: fail-closed threshold per task (default 0.9).

Per iteration the mining runtime calls
`select_detection_calibration.select_calibration(records, media_root=...,
excluded_identities=..., pair_assets_dir=..., task_bin_quotas=contract["task_bin_quotas"],
min_fill_fraction=contract["min_fill_fraction"])` (or the CLI with
`--profile-task-total TASK=ROWS`); no `max_boxes`, cohort or legacy quota
arguments may be combined with it. Rows are read from
`config.annotations.calibration` (the Mining file unless a calibration pool was
recorded, see `preflight.md`). Selection walks the pool once, fills each
`(task, bin)` bucket in file order, keeps reference pairs content-unique and
skips previously mined / evaluation identities exactly as the cohort policy.

Rows keep the existing evidence labels (`calibration_empty_ground_truth`,
`calibration_few_box_ground_truth`, `calibration_reference_no_change_ground_truth`)
so downstream quota accounting is unchanged, and add `calibration_count_bin`
and `calibration_policy`. The summary (`detection_calibration_v4`) reports per
task and bin the requested / selected rows, the fill fraction and shortages; a
task below `min_fill_fraction` fails closed with the short bins listed.

Tasks not named in `--calibration-task-total` get no calibration rows
(Component Detection stays mining-only unless a total is given; its benchmark
profile is 0% empty and 5.3 boxes per row, so give it a total only together
with a corpus-size decision, because the row cap and the Defect Detection
floor act on the whole iteration).

## Selection order

Profile mode collects every eligible detection row first and fills each
`(task, bin)` bucket round-robin across datasets in seed/id-hash order
(`profile_seed`, default 17), so a pool sorted by dataset cannot fill a bucket
from its first dataset alone; the summary reports `datasets_per_bin`.
Reference pairs stay content-unique and any reference shortfall fails closed.

## Capability gate and acquisition mode (Phase 3)

`capability_gate.py --kpi-annotations <KPI set> --raw-report <raw_f1.json of
the gating checkpoint> --acquisition-rows N --refinement-rows TASK=ROWS
--output capability_gate.json` compares each detection task's F1 with the
KPI set's trivial always-empty F1 `2E / (2E + B)`. A task at or below it is in
**acquisition** mode: the skill is absent, so error-driven mining has nothing
to refine and the task gets a volume budget with broad dataset coverage.
Launch-record it with `init_deft_state.py --acquisition-task TASK=ROWS
--acquisition-gate capability_gate.json` (requires the profile policy): the
task's profile-binned calibration total becomes ROWS and
`calibration_quota_contract.acquisition` records tasks, rows and the gate
file. Pass the materializer `--acquisition-rows <rows>` so those rows are
excluded from the Defect Detection floor base; otherwise a large acquisition
slice forces an unsatisfiable DD quota. Re-run the gate every iteration; when
the task crosses the trivial baseline drop the override (refinement mode).

Two assembler rules keep the acquisition slice intact under the row cap and
the anchor share (2026-09-14: the task-balanced trim that makes room for the
anchor slots dropped 382 of 3,000 reference pairs): emitted calibration rows
carry the inert marker `deft_calibration` and are never displaced by the trim
(`cap_reservation.calibration_rows_protected`), and the assembler's
`--anchor-share-exclude-rows <cumulative rows>` takes the slice's increment
over the parent (e.g. `2500 × iteration` for 3,000 vs 500 pairs) out of the
share base, so the anchor volume matches the parent instead of growing with
the acquisition rows (`realized_share_rows` is relative to that base;
`realized_share_of_all_rows` keeps the plain ratio).

The quota manifest checks the reference empty rate over calibration **and**
mined reference rows together; the materializer therefore seeds the mined
slice's running total with the reserved calibration counts so both slices
complete one combined `floor(total × rate + 0.5)` target (separately rounded
slices missed it by one row at 3,000 + 80 rows, 2026-09-14).

## Supply check before launching

Pool box-count bins (full pool, 2026-09-10): Defect Detection `4-9` 1,358 rows,
`10+` 190; Ref_based Defect Detection `4-9` 13,558, `10+` 639. With the panel
shares above a 1,024-row DD quota asks for about 184 `4-9` and 23 `10+` rows
per iteration, so five iterations fit; report the shortage table anyway.

## Empty / few-box substitution under the empty-answer guard (Feature B3)

Applies to the fixed calibration slot of `defect_detection_ablation.py`
(`--single-image-calibration-max-empty/-few`, `--reference-calibration-total`)
when `--calibration-guard-aware on` is given together with the guard caps
(`--max-empty-answer-share`, `--max-empty-answer-share-task`; the runner mirrors
them from the assembler options). Without the flag the selection is
byte-identical to before (golden fixture in
`tests/test_cosmos3_guard_aware_calibration.py`). The flag without a cap, or
without the fixed slot, fails closed.

Rule, after the strict Defect Detection rows and the mined maintenance rows of
the iteration are selected and before the feasibility loop:

1. **Headroom per cap.** For the overall cap and for the Defect Detection /
   Ref_based Defect Detection task caps:
   `headroom = max(0, floor(cap × rows − empty_rows))`, where `rows` counts the
   cumulative corpus (`--previous-jsonl`), this iteration's non-calibration
   rows **and** the calibration slot's rows (their count is fixed), and
   `empty_rows` counts the empty ground truths of the corpus and the
   non-calibration rows only. A share equal to its cap is within the cap, as in
   the assembler.
2. **Empty targets.** Start from the KPI targets (the empties the slot would
   select today: `max_empty` single images, `floor(total × KPI empty rate + 0.5)`
   no-change pairs), bound each by its task headroom, then split the overall
   headroom over the two by largest remainder (ties by task name) when their
   sum exceeds it (`allocate_empty_headroom`). Negative headroom means zero
   empties.
3. **Substitution, same source.** Reference pairs: the reserved calibration is
   re-selected with the effective empty rate `target / total`, so the kept
   no-change pairs are a prefix of the KPI-rate selection and changed pairs fill
   the remaining slots (content-unique, deduplicated against every row already
   kept). Single images: the first `target` empties are kept in their original
   order and the balanced few-box selection is re-run with the target
   `max_few + substituted` over the few-box calibration candidates (evidence
   `calibration_few_box_ground_truth`, deduplicated against the kept rows). The
   mined maintenance and strict rows are not re-selected. **Best-effort
   (Feature B3.1):** when the few-box / changed supply is short, the remaining
   slot rows are the next empties of the same ordering, never more than the
   KPI-rate selection carried, so the slot stays full: `fewbox_shortfall`
   records the non-empty rows asked for and not found,
   `calibration_headroom_overflow_rows` the empties kept beyond the headroom,
   `calibration_fewbox_substituted` only the rows actually replaced, and
   `guard_aware_calibration.status` becomes `substituted_with_overflow`
   (`substituted` without overflow, `no_substitution_needed` when the headroom
   covered the KPI targets). The assembler's guard trims the overflow. A
   substitution shortfall alone never fails the materializer; a slot the feed
   cannot fill at all (fewer candidates than slot rows) fails closed as before.
4. **Verification.** `single_image_calibration_caps_respected` compares the
   few-box count with `max_few_box_effective = max_few_box + substituted`; the
   reference targets `target_no_change` (calibration; the kept no-change pairs,
   headroom target plus overflow) and the combined calibration + mined target
   behind `reference_empty_rate_matched` are lowered by
   `no_change_substituted_by_changed` (the mined slice keeps tracking the KPI
   rate, so the guard never gets more empties to trim from it);
   `kpi_target_no_change` keeps the original number.

Manifest: `calibration_guard_aware`, `calibration_empty_headroom`,
`calibration_empty_selected`, `calibration_fewbox_substituted`,
`calibration_headroom_overflow_rows` and the `guard_aware_calibration` block
(caps, `kpi_empty_targets`, `headroom_empty_targets`, `ledger`,
`fewbox_shortfall`, `status`), all copied under `current_selection` in the
bound v2 manifest; `single_image_calibration.empty_beyond_headroom` and
`reference_calibration.no_change_beyond_headroom` give the per-slot overflow.
Worked examples with the r4 iteration-4 and r5 iteration-1 numbers:
`empty-answer-guard.md`, "Guard-aware calibration and anchor share".

## Feed reserve (Feature B3.1) — read `feed_bucket_quotas` from the state

The substitution can only draw few-box / changed rows the calibration FEED
holds. Run v12_p4b_emptyguard_r5 iteration 1 (snapshot 3c4e042b) fed the
materializer with the hard-coded contract numbers
(`cohort_bucket_quotas = {non_reference_based: {empty: 512, few: 512},
reference_based: {empty: 278, few: 222}}`), the headroom asked for 86 / 127
more non-empty rows than that, and the run aborted with the slot short while
86 unused empty candidates sat in the feed.

`init_deft_state.py` therefore records
`config.mining.calibration_quota_contract.feed_bucket_quotas` next to the
contract quotas (fixed-slot policy `fixed_single_image_proxy_rate_reference`
only):

| `empty_answer_guard.calibration_guard_aware` | `non_reference_based` | `reference_based` |
|---|---|---|
| `true` | `{"empty": max_empty, "few": max_empty + max_few_box}` → 512 / 1,024 | `{"empty": KPI no-change, "few": total}` → 278 / 500 |
| `false` | `{"empty": max_empty, "few": max_few_box}` → 512 / 512 (today) | `{"empty": KPI no-change, "few": total − no-change}` → 278 / 222 (today) |

Controller rule: the per-iteration runtime call MUST read this block from the
state and pass it through, never hard-code the numbers (the r6 prompt says the
same):

```python
contract = state["config"]["mining"]["calibration_quota_contract"]
select_detection_calibration.select_calibration(
    records, media_root=..., excluded_identities=..., pair_assets_dir=...,
    cohort_bucket_quotas={
        "non_reference_based": {"empty": contract["single_image_max_empty"], "few": contract["single_image_max_few_box"]},
        "reference_based": {"empty": contract["reference_empty"], "few": contract["reference_few_box"]},
    },
    feed_bucket_quotas=contract["feed_bucket_quotas"],
    cohort_rates=contract["cohorts"],
)
```

The selector fills the feed quotas, fails closed only against the contract
quotas (the reserve is best-effort: a pool with fewer few-box rows than the
reserve is fine) and reports per cohort `feed_bucket_quotas` and
`feed_reserve_rows` (`empty` / `few` / `total` rows selected beyond the
contract) in its `detection_calibration_v3` summary. `feed_bucket_quotas`
requires the fixed-slot `cohort_bucket_quotas` contract and may not undercut
it. Worked example and outcome table: `empty-answer-guard.md`, "Worked
example: run v12_p4b_emptyguard_r5".

## Calibration quota under the guard is an upper bound (Feature B4)

**Rule: under the empty-answer guard the calibration quota is an upper bound;
the growth slot is the binding constraint.** The materializer still verifies
the fixed slot (1,024 single-image + 500 reference pairs for the pinned recipe)
and the assembler still protects those rows from the ordinary cap trim, but
when the detection calibration rows alone exceed the current slot of one
growth step (previous rows + one `--row-multiple`, minus the share-bound
anchors and coverage rows) the assembler keeps every mined row that fits and
lets calibration rows yield: empty-ground-truth rows first (negatives,
no-change pairs), then non-empty rows (few-box, changed pairs), split
proportionally over Defect Detection and Ref_based Defect Detection with each
task keeping a prefix in materializer order. Anchors do not round the corpus up
(that fill is guard-off only). Run v12_p4b_emptyguard_r6 iteration 2 (1,077
calibration rows against 691 slots) is the worked example, and
`calibration_rows_dropped_for_cap` / `calibration_rows_kept` /
`calibration_yielded` in the assembly summary record what yielded:
`empty-answer-guard.md`, "Calibration yields to the growth slot". Read
`calibration_yielded` in every guard-on iteration's summary; a yield means the
calibration slot is larger than the growth step can carry once the single-image
mining is exhausted, so a smaller `--single-image-calibration-max-*` /
`--reference-calibration-total` or a larger `--row-multiple` is the lever, not
more anchors.

## Classification calibration (Phase 4 step 4c-A) — default off

### Why

The anchors run's final corpus is 43% empty-ground-truth rows (full training
pool: 22%) and holds 10 single-image defect MCQ rows. On the new benchmark the
model answers `[]` on 58% of the single-image MCQ (Full-train 10K: 5%). A probe
that deleted the "return []" clause from the questions removed the empty
answers but accuracy stayed near guessing (defect MCQ 12% → 23%): the missing
capability is defect-type classification, and the empty-answer share is only
the symptom (capped separately by `empty-answer-guard.md`). This quota adds
real pool rows that *choose a class*.

### Selection rule

```bash
python3 "$SKILL_ROOT/scripts/select_classification_calibration.py" \
  --pool "$WORKSPACE/annotations/mining.jsonl" --kpi "$WORKSPACE/annotations/proxy_kpi.jsonl" \
  --task-total "Defect Classification=256" [--task-total "Component Classification=128"] \
  [--exclude-identities-file train.jsonl] [--exclude-identities-file anchor_candidates.jsonl] \
  [--media-root "$WORKSPACE"] [--seed 17] [--min-fill-fraction 0.9] [--allow-shortfall] \
  --output classification_calibration.jsonl --manifest classification_calibration_manifest.json
```

Eligible rows per task: a single-image classification task (`Defect
Classification`, `Component Classification`; the two-image reference task is
rejected), exactly one image in the user message, the lettered MCQ form (the
`current possible classes` options block; the yes/no "Answer with the complete
option text" form is not eligible), and a non-empty parsed option set (`F`,
`[B,D]`, `["B","D"]`; `[]` is rejected, unknown letters and unparsable answers
are counted under `rejected`). Excluded: atomic identities / ids found in the
exclusion files (`.jsonl` = rows such as the cumulative `train.jsonl` or the
anchor candidates, any other file = one identity per line; the identity is the
assembler's `record_identity`, so pass the same `--media-root`), marker-free
content duplicates and same-image duplicates within the pool.

### Class shares

The target distribution per task comes from the KPI set's ground-truth option
labels (the semantic option text before the colon, never the letter, because
letters differ between prompts; a multi-label row counts once per label).
Quotas are a largest-remainder split over the classes the pool can supply:
KPI classes with no eligible pool row are dropped and their share redistributed
(`classes_missing_in_pool`); a task without usable KPI labels falls back to a
uniform split over the pool's classes (`class_share_source =
uniform_pool_fallback`). Inside a class the rows are ordered by seed/id hash
and filled round-robin across datasets. A shortage inside a present class is
not backfilled from another class; a task below `--min-fill-fraction` makes the
CLI exit 2 unless `--allow-shortfall`, and the manifest records
`shortfall_tasks`, `fill_ok` and `accepted` either way.

### Markers, assembly, launch record

Output rows are the pool rows unchanged plus the inert markers
`deft_calibration: true` and `deft_calibration_kind: "classification"` (the
materializer's detection calibration rows carry `deft_calibration_kind:
"detection"`; the empty-answer guard trims only the detection kind).
`assemble_training_json.py --classification-calibration-jsonl <output>` adds
them as current rows (provenance `source_kind = classification_calibration`),
de-duplicates them by id / marker-free content against the corpus, never trims
them under the row cap or the anchor reservation (like detection calibration
rows), and reports `materialized_classification_calibration_records` plus a
`classification_calibration` block (input rows, duplicates, prior rows,
new / total). `render_iteration_mining_runner.py` renders the selector as the
stage between the mined-row selector and the assembler when the request carries
`classification_calibration_command`; the renderer owns
`--exclude-identities-file` (previous Train, the `--anchor-source` when anchors
are configured, the current `mined.jsonl`), `--output` and `--manifest`, and
passes the output to the assembler.

`init_deft_state.py --calibration-task-total "Defect Classification=256"`
records `config.mining.classification_calibration = {task: rows}` and a
`classification_calibration_contract` (eligible tasks, class-share source =
the KPI set, pool = the calibration annotations, `min_fill_fraction` from
`--calibration-min-fill`, `--classification-calibration-seed`, markers). Detection
tasks named in the same flag still feed the box-count profile above.

### Manifest (`classification_calibration_v1`)

Per task: `requested`, `eligible`, `selected`, `fill_fraction`, `shortfall`,
`class_share_source`, `classes_missing_in_pool`, `datasets`, and `per_class`
`{kpi_rows, target_share, target, eligible, selected}`; overall
`excluded_by_identity`, `deduplicated_content`, `deduplicated_identity`,
`rejected` by reason, `seed`, `min_fill_fraction`, `inputs` (pool / KPI /
exclusion files with SHA-256) and `output` (rows, SHA-256).
