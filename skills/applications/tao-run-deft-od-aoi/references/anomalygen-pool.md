# AnomalyGen 1.2 synthesis contract

Read this only when synthesis is enabled.

## Reference pool

`synthesis.pool_dataset_root` points to the normalized pool consumed by
`tao-generate-anomalies`:

```text
POOL/
  TEXTURE/
    clean_image/*
    mask/DEFECT/*
```

An enabled synthesis policy must start with at least one supported image under
some `TEXTURE/clean_image` directory; initialization rejects a globally empty
reference pool. Individual texture pools may still be absent or empty because
their routed FNs are handled independently during iteration preparation.

The frozen `defect_spec.jsonl` must define every selected
`TEXTURE+DEFECT`. Text-routed definitions require
`roi_prompt_defect_location`. Every KPI FN provides `dataset_id` so the
application can decide whether it is routed for synthesis. FNs whose dataset
ID is not in `synthesis.routes` remain in the normal real-data path and are
counted in the synthesis request report. A missing or empty `dataset_id` is
malformed metadata, not an opt-out mechanism. Routed FNs must also provide
explicit `texture_id`, `defect_class`, and `fn_mask_source`. A box is not a
mask, and this application never invents masks or prompts.

## Existing task weights

Each key in `synthesis.routes` matches a KPI `dataset_id` and provides:

```yaml
routes:
  line_a:
    checkpoint: /models/line_a/adapter.pt
    recipe: /models/line_a/canonical_recipe.yaml
```

Both files must exist before iteration preparation. The recipe must declare the
types requested from that route.

## Missing task weights

A route may instead carry a one-time fine-tuning plan:

```yaml
routes:
  line_a:
    finetune:
      dataset_root: /data/line_a/anomalygen
      validation_testcase: /data/line_a/validation.jsonl
      base_checkpoint: /models/Cosmos3-Nano
      vae_path: /models/Wan2.2_VAE.pth
      checkpoint_root: /models/anomalygen-checkpoints
      result_handoff: /results/line_a/training_handoff.json
      recipe_template: /data/line_a/recipe.yaml   # optional
      defect_spec: /data/line_a/defect_spec.jsonl # optional override
```

`checkpoint_root` must contain the required Qwen tokenizer model assets under
`hf/` and `facebook/dinov2-large/` with its Transformers config and weights.
The training action mounts this entire directory over the image's
`/workspace/cdf-anomalygen/checkpoints`; mounting only DINOv2 is insufficient.
Reuse that same complete root for `tao-generate-anomalies prepare --source od-gaps --prepare-stage amp`
and `tao-generate-anomalies inference_only`. Both actions require the canonical
mount rather than independent SAM2 or Hugging Face cache inputs, and each
validates its action-specific assets before GPU work.

Run `resolve_deft_od_aoi_synthesis.py` before the candidate cache. A missing
handoff produces `finetune_requests.json`. Execute each request through
`tao-generate-anomalies finetune_only` once, then resolve again into a new directory.

Accept a completed handoff only when dataset identity matches, anomaly types
are nonempty, and recipe/checkpoint hashes verify. Freeze the emitted
`resolved_synthesis_policy.yaml` for the run. Never start or resume
AnomalyGen 1.2 training inside an iteration.

## Iteration handoff

`prepare_deft_od_aoi_synthesis.py` converts exact strict FN/annotation matches
to a filtering YAML for `tao-generate-anomalies`. The resulting
generation plan is passed to `tao-generate-anomalies`. Only its validated
logical `binary_coco` output enters admission, under the cumulative synthetic
fraction cap. Its exact relative path comes from the authoritative action
contract; the native fine-grained COCO remains separately available as
`native_coco`. Resolve that relative path against the generation action's
results directory before admission, then pass the concrete file through
`--synthetic-coco` and the results directory through `--synthetic-images`.
Admission removes undersized, extreme-aspect, and full-frame boxes, then
allocates available capacity proportionally across source `dataset_id` values
with deterministic selection.

If there are no routed FNs, `synthesis_request.json` is a typed `SKIPPED`
contract whose evidence lists the observed dataset IDs, configured route keys,
total skipped count, per-dataset skipped counts, and a human-readable summary.
If every routed FN fails mask eligibility, the preparation leaf emits a typed
`SKIPPED` `input_contract.json` with per-FN reasons. Either contract completes
the optional synthesis stage without AMP, generation, or synthetic re-admission.

Clean reference images are a synthesis-only input, distinct from the DEFT clean
retrieval role. A missing or empty reference directory marks only its routed FNs
with `no_clean_reference_images`; other eligible types and real-data mining
continue. When all routed FNs have that reason, the input contract is typed
`SKIPPED` with preparation-level reason `no_eligible_false_negatives`, carries
the specific per-FN reason and warning evidence, and emits no embedding or AMP
work.

## Runtime-generated FN selection

The default generates a bounded plan deterministically after real admission:

```yaml
synthesis:
  cumulative_fraction_of_total_defects: 0.25
  fn_selection:
    mode: generated_per_type_plan
  candidate_topn: 3
  max_neighbors_per_fn: 1
```

Pass the admitted real COCO with `--real-coco`. The planner solves
`synthetic / (real + synthetic) = fraction`, subtracts previously admitted
synthetic images, and distributes the remaining whole-FN budget across anomaly
types in proportion to eligible FN counts with deterministic remainder ties.
The planner derives the expected finalized generation yield for one selected
FN as:

```text
retained neighbors per FN × mask branches per neighbor × images per branch
```

The accepted configuration uses one retained neighbor, two mask branches, and
one generated image per branch, so the value is `1 × 2 × 1 = 2`. This derived
value is recorded as `images_per_fn` in the frozen plan evidence. The
synthesis-only `candidate_topn` defaults to three because it supplies fallback
neighbors before AMP finalization; it is independent of both max-similarity
candidate overfetch and round-robin refill. It does not increase the expected
yield once `max_neighbors_per_fn` retains only one neighbor. It is the AMP
compute knob: each eligible candidate produces two AMP requests, so one FN
costs up to `2 * candidate_topn` placement attempts to retain
`2 * max_neighbors_per_fn` finalized images. Increasing
`max_neighbors_per_fn` automatically increases the bounded-plan yield by two
images per added neighbor.

Set `synthesis.fn_selection.mode: all_eligible` explicitly only when every
routed FN should proceed without pre-generation budgeting. That compatibility
mode does not require `--real-coco`, but the cumulative admission cap still
applies after generation.

It writes `synthetic_plan.json`, binds its SHA-256 into the request, and records
the complete budget calculation. If no budget remains, preparation emits a
typed `SKIPPED/no_synthetic_budget` result.

AnomalyGen 1.2 synthesis uses the single `tao-generate-anomalies` skill:
`prepare --source od-gaps` for inputs/AMP/finalize, `finetune_only` for one-time
bootstrap, and `inference_only --pseudo-label` for each finalized group. Resolve
`images.metropolis_sdg.anomalygen`; mount the complete selected-stage cache at
`/workspace/cdf-anomalygen/checkpoints`. Keep the prepared manifest and its
hash-bound artifacts. Emit COCO with application `scripts/emit_sdg_coco.py`
using `--runs-manifest` and `--prepared-manifest`; admission uses the converter's
`pseudo_labels/coco_annotations_od_defect.json`. Pass caller `--exclude-stems`
at preparation, generation and conversion, and set `synthesis.exclude_stems`
for admission. The preparation CLI option overrides the frozen YAML exclusion path; omission uses the application path. The original source/donor provenance
remains in the prepared bundle. Existing NN improvement, box-quality, fraction,
whole-FN planning and split/KPI policies are unchanged. Fully blocked output is
COMPLETE with zero synthetic contribution; preserve blocked IDs without retry
or top-up. Guardrail capability limitations remain explicit.

Application COCO admission requires each generated image's `anomalygen_prepared`
binding to the hash-bound prepared manifest and dataset group. Conversion and
admission recheck explicit `source_mask`, `fn_filepath`, `clean_filepath`, and
`aligned_mask` provenance against current caller exclusions, including after
mask alignment changes the filename. Missing or altered ancestry is a hard
stop. Standalone smoke COCO conversion without a prepared bundle demonstrates
conversion only and is not admissible as application training data.

Historical AnomalyGen 1.1 OD checkpoint execution and the full OD AOI runtime
remain unverified in this migration; CPU converter and contract tests establish
only their tested interfaces.
