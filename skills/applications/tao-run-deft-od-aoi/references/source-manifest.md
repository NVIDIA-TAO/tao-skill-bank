# Normalized source handoff

Use this reference to convert one or more customer COCO datasets into the four
roles accepted by DEFT OD AOI. The application provides a generic manifest
preparer; it does not contain dataset-name-specific parsers.

## Source manifest

Create `dataset_sources.json` with one or more COCO inputs per role:

```json
{
  "schema_version": 1,
  "inputs": {
    "benchmark": [{"coco": "/data/benchmark.json",
                   "images_dir": "/data/benchmark/images"}],
    "test": [{"coco": "/data/test.json", "images_dir": "/data/test/images"}],
    "mining": [{"coco": ["/data/train.json", "/data/mine.json"],
                "images_dir": "/data/images"}],
    "clean": [{"coco": "/data/clean.json", "images_dir": "/data/clean/images"}]
  }
}
```

Paths may be absolute or relative to the manifest. `coco` accepts one path or
a list. Benchmark and test inputs may contain boxed and boxless images. The
preparer maps user-facing `benchmark` to the downstream internal `kpi` role.
Every mining image must have a box, while every explicit clean COCO must have
zero annotations.
The preparer maps all input categories to the one `defect` category and rejects
cross-role image overlap.

## Canonical metadata boundary

The generic preparer preserves metadata; it does not invent dataset-specific
identity. Before referencing a benchmark COCO in `dataset_sources.json`, its
upstream dataset ingestion must populate these canonical fields on every boxed
KPI image:

- `dataset_id`: synthesis route/dataset identity;
- `texture_id`: namespaced product or texture identity;
- `defect_class`: source defect label;
- `fn_mask_source`: a real same-size segmentation mask when `dataset_id` is
  synthesis-applicable.

Dataset-specific ingestion owns any path, category, or source-schema mapping
needed to produce these values. This application accepts only the canonical
contract and does not define legacy aliases or infer identities from filenames.

Validate without writing output, then materialize into a new directory:

```bash
scripts/prepare_deft_od_aoi_sources.py \
  --manifest /data/dataset_sources.json --check-only

scripts/prepare_deft_od_aoi_sources.py \
  --manifest /data/dataset_sources.json \
  --output-dir /new/results/normalized --link-mode symlink
```

Use `--link-mode copy` when the normalized directory must own portable image
copies. `--check-only` needs no container; materialization runs in the pinned
TAO Data Services image. It retains canonical per-source files under
`merge_inputs/`, invokes the existing `annotations merge` action once per role,
and post-validates each merged COCO. Existing output directories are never
overwritten. The output includes the frozen input manifest, merge evidence,
normalized COCO files and image views, a provenance report, and `sources.json`
ready for initialization or the SLURM acceptance driver.

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
- Preserve canonical `dataset_id`, `texture_id`, and `defect_class` values plus
  pixel-mask paths on synthesis-applicable benchmark records.
- Do not infer AnomalyGenNext types from filenames at this boundary.

Validate the resulting handoff with `init_deft_od_aoi.py`; its output policy is
the immutable downstream contract. Raw-image discovery, filename-based routing,
and dataset-specific metadata inference remain explicit adapters before this
generic COCO boundary.
