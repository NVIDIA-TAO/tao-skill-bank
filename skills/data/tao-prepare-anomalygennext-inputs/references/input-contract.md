# Normalized AnomalyGenNext preparation contract

The preparation leaf accepts normalized identity instead of legacy path rules.
The producer must freeze `dataset_id`, `texture_id`, `defect_class`,
`anomaly_type`, and `fn_mask_source` on every gap row before launch.

`bbox` is `[x1, y1, x2, y2]` in source-image pixels. `fn_mask_source` must be a
same-size pixel mask; a detector box is not a replacement. `split` is an opaque
selection bucket such as `kpi` or `test`.

The YAML `datasets` mapping assigns each `dataset_id` an existing checkpoint
and recipe. This action verifies that both files exist and that the recipe's
`anomaly_types` contains the normalized `TEXTURE+TYPE`. A later synthesis
integration may resolve those two paths from a completed fine-tuning handoff,
but they must be concrete before invoking this leaf.

Selection supports:

- `all_eligible`: every eligible FN in the listed datasets.
- `per_dataset`: the first deterministic `per_dataset` rows from the named
  `split` for each listed dataset.

The same frozen encoder identity is written to both embedding specs. The next
action must use those specs unchanged so clean and FN vectors remain comparable.

`run_amp` joins the unique source-image embedding back to every box-level FN,
ranks clean images within the normalized texture pool, and creates two AMP
requests for every eligible pair. It rejects missing, non-finite, zero-norm, or
width-mismatched embeddings. AMP itself remains owned by the container-native
`anomalygen.scripts.auto_mask_placement.roi_place` entry point.

`finalize_inputs` retains the configured number of successful neighbors per FN.
Both aligned mask branches must match the clean image and cover neither zero nor
all pixels. It copies accepted masks under the prepared root and hashes every
generation contract artifact, without changing the source training pool.
