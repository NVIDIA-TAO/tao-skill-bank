# DEFT OD AOI data contract

Read this before freezing the application policy or admitting new training
images.

## Normalized roles

The application consumes four normalized COCO roles:

| Role | Boxes | May enter training | Purpose |
|---|---:|---:|---|
| KPI | labeled or empty | never | gap queries and checkpoint selection |
| Test | labeled or empty | never | report-only measurement |
| Real | at least one per image | after retrieval and admission | positive mining |
| Clean | exactly zero per image | after retrieval and admission | negative mining |

Every COCO document declares exactly one foreground category named `defect`.
Background is implicit. Clean images remain explicit `images` rows with zero
annotations; files absent from the COCO document do not train the detector.

Each image resolves through `source_path` when present, otherwise
`images_dir/file_name`. Resolved identities must be disjoint across KPI,
test, real, and clean roles. The initializer verifies paths, IDs, boxes,
category names, role-specific annotation counts, and cross-role overlap before
freezing hashes in the policy.

## Cumulative admission

`admit_deft_od_aoi_coco.py` recomputes similarity from the frozen embeddings,
deduplicates selected crops to source images, rejects previously admitted
sources, and publishes a new binary COCO. Pass `--previous-coco` from
iteration 2 onward. Clean negatives are capped by
`routing.clean_cumulative_cap_per_real`. Existing images and boxes are retained
unchanged.
