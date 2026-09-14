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

`status`: `within_caps` (nothing to trim), `trimmed_to_caps`, or `exceeded`.
In enforce mode `exceeded` after trimming makes `assemble_training_json.py`
write `assemble_summary.json` (for diagnosis) but not `train.jsonl`, and exit
2; in report mode the corpus is written unchanged and the status is recorded.
Enforcement cannot be combined with the repetition blend (report mode can).

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
  "status": "trimmed_to_caps"
}
```
