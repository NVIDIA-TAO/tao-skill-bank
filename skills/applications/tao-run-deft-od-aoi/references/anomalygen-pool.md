# AnomalyGenNext synthesis contract

Read this only when synthesis is enabled.

## Reference pool

`synthesis.pool_dataset_root` points to the normalized pool consumed by
`tao-prepare-anomalygennext-inputs`:

```text
POOL/
  TEXTURE/
    clean_image/*
    mask/DEFECT/*
```

The frozen `defect_spec.jsonl` must define every selected
`TEXTURE+DEFECT`. Text-routed definitions require
`roi_prompt_defect_location`. KPI FNs provide explicit
`dataset_id`, `texture_id`, `defect_class`, and `fn_mask_source`.
A box is not a mask, and this application never invents masks or prompts.

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
      nn_backbone: /models/dinov2-large
      result_handoff: /results/line_a/training_handoff.json
      recipe_template: /data/line_a/recipe.yaml   # optional
      defect_spec: /data/line_a/defect_spec.jsonl # optional override
```

Run `resolve_deft_od_aoi_synthesis.py` before the candidate cache. A missing
handoff produces `finetune_requests.json`. Execute each request through
`tao-finetune-anomalygennext` once, then resolve again into a new directory.

Accept a completed handoff only when dataset identity matches, anomaly types
are nonempty, and recipe/checkpoint hashes verify. Freeze the emitted
`resolved_synthesis_policy.yaml` for the run. Never start or resume
AnomalyGenNext training inside an iteration.

## Iteration handoff

`prepare_deft_od_aoi_synthesis.py` converts exact strict FN/annotation matches
to a filtering YAML for `tao-prepare-anomalygennext-inputs`. The resulting
generation plan is passed to `tao-generate-od-defects`. Only its validated
binary COCO enters admission, under the cumulative synthetic fraction cap.
