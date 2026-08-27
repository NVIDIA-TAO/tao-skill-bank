# RCCA Report — `<run>` / `<iteration>`

Template for the Proxy RCCA report that `commit_stage.py --rcca-report`
validates. Every `##` heading below is **required**: the validator checks for
each by name and refuses the commit if one is missing, because a report that
silently drops a section reads as complete while hiding the analysis the next
iteration's routing depends on.

Keep the headings verbatim. Everything under them is yours.

---

## Verdict

One paragraph: did this iteration's Proxy evaluation meet the KPI, and if not,
what is the dominant failure mode. State the metric and the target, not an
impression.

> Example: `recall_ng=0.94` against a target of `1.00`. The gap is dominated by
> false rejects on `PCB+bridge`, not by a uniform drop across defects.

## False-Accept Breakdown

NG records the model called OK. Group by defect type and by any structural
attribute that separates them (board, lighting, component class). A count with
no grouping is not a breakdown.

## False-Reject Breakdown

OK records the model called NG. Same grouping. These usually matter more for the
AOI KPI, since the contract is recall-first — say so explicitly when the two
directions disagree about where the model is weak.

## Top-K Worst Samples

The individual records that most influenced the verdict, with their paths under
`${RESULTS_DIR}`, so a reader can look at the images rather than trust the
summary. Include the model's response verbatim; a response that is neither `OK`
nor `NG` is a finding in itself, not a miscount.

## Per-Defect Analysis

One subsection per defect type present in the annotation set. For each: how many
records, how many wrong, and whether the errors cluster. A defect with too few
records to conclude anything should say that rather than report a rate.

## Recommended Actions

What the next iteration should do, tied to the evidence above: which defects to
route to mining, which to AnomalyGen, and which to leave alone. Say when the
evidence does not support acting — an empty recommendation with a reason is a
valid outcome, and `anomalygen --skip` requires exactly that proof.
