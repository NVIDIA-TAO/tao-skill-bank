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
| Clean | exactly zero per image when nonempty | after retrieval and admission | negative mining |

Every COCO document declares exactly one foreground category named `defect`.
Background is implicit. Clean images remain explicit `images` rows with zero
annotations; files absent from the COCO document do not train the detector.
Every bbox must have a nonnegative origin, positive dimensions, and remain
fully within its image dimensions. Boundary overflow is an input-contract
violation; source ground truth is rejected rather than clipped or rewritten.

Each image resolves through `source_path` when present, otherwise
`images_dir/file_name`. Resolved identities must be disjoint across KPI,
test, real, and clean roles. The initializer verifies paths, IDs, boxes,
category names, role-specific annotation counts, and cross-role overlap before
freezing hashes in the policy.

The real role must start with at least one boxed image. The clean role may start
with zero images; initialization emits a warning and a typed `UNAVAILABLE`
clean retrieval capability. The clean retrieval role is independent from the
AnomalyGenNext synthesis reference pool.

## Optional synthesis metadata

Every KPI sample must retain its real nonempty `dataset_id`. Each routed KPI
annotation/image pair must also resolve `texture_id`, `defect_class`, and a real
pixel `fn_mask_source`. Metadata may appear directly on the record or under
`deft_od_aoi`. A box never substitutes for the mask.

`synthesis.routes` is an allowlist: a valid ID absent from it skips synthesis
but still participates in the separate real-data DEFT path. A missing ID is
malformed metadata, not a synthesis opt-out. For routed FNs,
`texture_id+defect_class` must match a type declared by that route's recipe and
defect specification.

AnomalyGen clean-reference images come from the synthesis pool and are not the
DEFT `clean` retrieval role. Missing references skip only affected synthesis
FNs; they do not disable real-defect or clean-negative retrieval.

## Cumulative admission

`admit_deft_od_aoi_coco.py` recomputes similarity from the frozen embeddings,
deduplicates selected crops to source images, rejects previously admitted
sources, and publishes a new binary COCO. Pass `--previous-coco` from
iteration 2 onward. Clean negatives are capped cumulatively by
`routing.clean_cumulative_cap_per_real`. Real retrieval may become exhausted
while clean retrieval continues until the existing cumulative real count's
clean allowance is full. When synthesis is enabled, pass both
the generated binary COCO and its image root. Synthetic admission is capped by
`synthesis.cumulative_fraction_of_real_defects`. Existing records remain
unchanged.
