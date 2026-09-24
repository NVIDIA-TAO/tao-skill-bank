# Gap matching and retrieval routing

Read this when preparing KPI gap jobs or interpreting retrieval requests.

## Dual gap passes

Run `tao-analyze-gaps-od-map` action `object_detection` twice over the same
KPI ground truth and predictions:

| Pass | Confidence | Consumed rows | Route |
|---|---:|---|---|
| Loose | 0.3 | false positives | near miss → real; background-like → clean |
| Strict | 0.8 | false negatives | real and optional synthesis |

Both passes use same-class one-to-one matching at IoU 0.5. An unmatched loose
prediction below IoU 0.05 is background-like. From 0.05 inclusive to 0.5
exclusive it is a near miss. A strict FN may represent a missed object or an
object covered only by a lower-confidence prediction.

The inference threshold remains 0.001 so gap analysis, rather than inference,
owns the loose and strict operating points.

## SigLIP retrieval

Build the candidate cache once:

- each real annotation produces a 1.5× contextual crop;
- each verified-clean image produces the whole image and a 2×2 grid;
- both roles use the same frozen SigLIP encoder.

For each iteration, crop strict FNs and near-miss FPs as real queries, and
background-like loose FPs as clean queries. Embed queries with the identical
encoder. Invoke `tao-mine-od-images` using the emitted role-specific specs.

Retrieval is global within the real or clean role. Provenance metadata does not
partition the index. Empty query roles emit no action. Admission recomputes
maximum cosine similarity from the frozen embeddings, applies the frozen
minimum, deduplicates parent images, and enforces cumulative caps. The mining
request overfetches crop candidates by the frozen factor, while admission
measures capacity in unique parent images after cumulative exclusions. Parent
budgets are targets rather than hard minimums: admission takes up to each
target and writes `admission_preview.json` with desired parents, mined crops,
unique and novel parents, shortfalls, branch counts, and selected parents by
source dataset. A completely empty first admission still fails.

Strict-FN real budgets use the selected factor within the frozen `1..6` range.
Near-miss budgets are computed independently for each frozen
`benchmark/texture/defect_type` pocket as `min(2 * queries, 20)` by default;
this cap is not a per-dataset allocation quota. Clean retrieval remains global
and is capped only by its factor and the cumulative clean-to-real bound.

The initial `-1.0` similarity threshold is an explicit calibration policy,
not evidence that all candidates are equally useful. Review retrieval outputs
before freezing a stricter value for a later run.
