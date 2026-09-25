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

`pool_dataset_root` is one user-level typed folder input, not an incidental path
hidden only inside YAML. The platform stages the source once, records its
source-to-compute binding, and passes the resolved compute path to preparation.
Preparation freezes that path in `filtering_config.yaml` and in every pool
filepath carried by the clean embedding parquet. It also records the pool under
`input_contract.json.downstream_inputs` as a required read-only folder for the
clean-image embedding and AMP actions. Each action has a separate container
mount namespace, so staging may be reused but the source must be remounted at
the same frozen compute path for each consumer.

The frozen contract shape is:

```json
{
  "downstream_inputs": {
    "clean_embeddings": {
      "pool_dataset_root": {
        "type": "folder",
        "compute_path": "/inputs/anomalygen_pool",
        "read_only": true
      }
    },
    "run_amp": {
      "pool_dataset_root": {
        "type": "folder",
        "compute_path": "/inputs/anomalygen_pool",
        "read_only": true
      }
    }
  }
}
```

`compute_path` is the required mount destination, not a second source input.
The platform retains the original user-supplied source binding and uses it to
construct each later container mount.

The emitted embedding YAML files remain native Data Services configs; do not
add mount-only fields to them. The platform action request carries the typed
folder mount separately. The clean embedding consumes the frozen pool binding.
The FN embedding does not consume the pool, but its container must separately
mount every source-image root referenced by `fn_embedding_inputs.parquet`.

Selection supports:

- `all_eligible`: every eligible FN in the listed datasets.
- `per_dataset`: the first deterministic `per_dataset` rows from the named
  `split` for each listed dataset.

The same frozen encoder identity is written to both embedding specs. The next
action must use those specs unchanged so clean and FN vectors remain comparable.

`run_amp` joins the unique source-image embedding back to every box-level FN,
ranks clean images within the normalized texture pool, and creates two AMP
requests for every eligible pair. It rejects missing, non-finite, zero-norm, or
width-mismatched embeddings. It always loads the frozen config from the
prepared root; no caller-supplied config is compared or accepted. The typed pool
argument is generated from the retained binding and must resolve to the pool
path frozen during preparation. AMP itself remains owned by the
container-native `anomalygen.scripts.auto_mask_placement.roi_place` entry point.

`finalize_inputs` retains the configured number of successful neighbors per FN.
Both aligned mask branches must match the clean image and cover neither zero nor
all pixels. It copies accepted masks under the prepared root and hashes every
generation contract artifact, without changing the source training pool.
