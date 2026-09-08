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

## Iteration handoff

`prepare_deft_od_aoi_synthesis.py` converts exact strict FN/annotation matches
to a filtering YAML for `tao-prepare-anomalygennext-inputs`. The resulting
generation plan is passed to `tao-generate-od-defects`. Only its validated
binary COCO enters admission, under the cumulative synthetic fraction cap.
