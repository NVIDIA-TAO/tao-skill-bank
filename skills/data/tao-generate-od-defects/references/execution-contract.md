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

## Image-safety reporting

The native guardrail path includes three separate components: text screening,
image content-safety screening, and face-blur post-processing. The observed
image-safety state applies only to image content-safety enforcement; it does not
imply that text screening or face blurring was active. Report component states
independently when the runtime exposes them.

A workflow that records the observed image-safety state must use one of these
values:

| Value | Meaning |
|---|---|
| `enforcing` | The runtime positively confirmed that image-safety enforcement was active. |
| `not_enforcing` | The runtime positively confirmed that image-safety enforcement was inactive. |
| `unknown` | The wrapper could not determine whether enforcement was active. |

These values describe an observed result, not a requested mode. Treat
`not_enforcing` and `unknown` as unscreened; neither is equivalent to
`enforcing`. A zero `guardrail_blocked` count does not prove that enforcement
ran. This run-level observation is distinct from per-image guardrail results.

## Quality metrics

When native evaluation is enabled, interpret FID and `nn_score` according to
the [AnomalyGenNext evaluation reference](../../tao-generate-anomalies/references/eval.md).
They are diagnostic signals with sample-count and visual-plausibility
limitations, not standalone acceptance evidence.
