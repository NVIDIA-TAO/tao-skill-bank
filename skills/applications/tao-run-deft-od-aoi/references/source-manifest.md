# Normalized source handoff

Use this reference when adapting one or more customer datasets into the four
roles accepted by DEFT OD AOI. Dataset discovery and normalization are upstream
of this application; the application does not contain dataset-name-specific
parsers.

## Handoff shape

Each role supplies an image directory and COCO file:

```yaml
sources:
  kpi:   {images: /data/kpi/images,   coco: /data/kpi.json}
  test:  {images: /data/test/images,  coco: /data/test.json}
  real:  {images: /data/real/images,  coco: /data/real.json}
  clean: {images: /data/clean/images, coco: /data/clean.json}
```

Preserve a stable absolute `source_path` on each image when the normalized
view does not physically own the source file. The application hashes the COCO
contracts and resolves every referenced image during initialization.

## Normalization rules

- Map all foreground categories to one category named `defect`.
- Keep boxless images only in explicitly verified clean or held-out roles.
- Keep clean images as COCO image rows with zero annotations.
- Never place one resolved image identity in more than one role.
- Preserve provenance metadata needed for audit.
- For synthesis, preserve exact `dataset_id`, `texture_id`, `defect_class`,
  and pixel-mask paths on eligible KPI records.
- Do not infer AnomalyGenNext types from filenames at this boundary.

Validate the normalized handoff with `init_deft_od_aoi.py`; its output policy
is the immutable downstream contract. If a customer layout needs parsing,
implement or run that conversion before this boundary and keep its own
provenance report beside the normalized COCO files.
