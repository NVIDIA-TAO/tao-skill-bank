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

The wrapper exposes the native Boolean guardrail choice. Guardrails default on
without changing the native invocation; disabling them appends
`--no-guardrail` explicitly.
`--no-guardrail` disables the text guardrail, image content-safety path, and
face-blur postprocessor together. The validation summary records the selected
mode.
The pinned framework's default video preset has no active image safety model:
its SigLIP-based `VideoContentSafetyFilter` is disabled because of excessive
false positives. Enabling an enforcing image classifier requires a separately
validated container preset rather than another argument to this wrapper.

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

With `--no-guardrail`, `generated == requested` and `guardrail_blocked == 0`.

Every generated image must have exactly one pseudo-label image record and at
least one valid in-bounds annotation. The native COCO retains the exact
fine-grained anomaly categories. The binary companion maps every annotation to
category id 1 named `defect`.

The leaf never admits generated images into a detector training pool. That
decision belongs to the calling application.
