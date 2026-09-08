# AnomalyGenNext OD generation contract

Read this when adapting generation to a new platform or AnomalyGenNext release.
This action performs inference only and requires an existing task checkpoint
with its matching canonical recipe.

## Input modes

Exactly one input mode is allowed:

- Prepared mode: `--inputs-dir` points to a completed
  `tao-prepare-anomalygennext-inputs` result containing
  `anomalygen_next_generation_plan.json`.
- Native mode: pass `--input-data-path`, `--checkpoint`, and `--recipe`;
  optional dataset and anomaly-type selectors may narrow existing rows.

Every testcase image and mask must exist. Each `anomaly_type` must be present
in the recipe, and the requested count must equal the testcase row count.
Prepared-input hashes are verified when the integrity manifest is present.

## Native stages

From the release source baked into the image, the wrapper invokes generation,
optional evaluation and quality refinement when a real-image root is present,
and pseudo-labeling. Generation uses `torchrun` with the requested GPU count;
AnomalyGenNext owns distributed row partitioning and rank-zero metadata merge.

## Completion accounting

For every dataset:

```text
generated + guardrail_blocked == requested
```

Every generated image must have exactly one pseudo-label image record and at
least one valid in-bounds annotation. The native COCO retains the exact
fine-grained anomaly categories. The binary companion maps every annotation to
category id 1 named `defect`.

The leaf never admits generated images into a detector training pool. That
decision belongs to the calling application.
