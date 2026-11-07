# Independent light-mode final evaluation

Run this host command only after both native editors have frozen and exited:

```sh
python3 skills/applications/tao-run-deft-icl-aoi-cosmos3/scripts/wsp03_light_final.py final-config.json
```

The driver checks the two pinned FREEZE artifacts and the native Docker state
of every listed editor container before opening final sources. Keep exited
containers until this check; an unknown/removed container is not evidence of
stopping. It does not invoke an editor or expose final results to one.

Use the existing frozen development recipe and final-input configuration.
The explicit normalized R4 selection avoids depending on an editor-specific
FREEZE field layout. Both FREEZE documents must have `status: FROZEN`. Preserve
the original R3 export, including its `patch/manifest.json` and replacement
safetensors files. A missing/invalid/nonzero-patch failure stops; the original
base is never substituted as an R3 result.

Required configuration structure (replace the placeholders):

```json
{
  "output": "/private/new-final-output",
  "freezes": {
    "R3": {"path": "/private/R3/FREEZE.json", "sha256": "R3_FREEZE_SHA"},
    "R4": {"path": "/private/R4/FREEZE.json", "sha256": "R4_FREEZE_SHA"}
  },
  "editor_containers": {"R3": ["R3_CONTAINER_ID"], "R4": ["R4_CONTAINER_ID"]},
  "r3_export": "/private/R3/frozen-r3",
  "r4_selection": {
    "freeze_sha256": "R4_FREEZE_SHA",
    "profile": "P01",
    "patch_manifest_sha256": "FROZEN_PATCH_MANIFEST_FILE_SHA"
  },
  "model": "/private/original-model",
  "model_files": {"config.json": "ORIGINAL_CONFIG_SHA"},
  "recipe": {"path": "/private/development-recipe.json", "sha256": "RECIPE_SHA"},
  "reference_state": {"path": "/private/frozen-reference-state.json", "sha256": "REFERENCE_STATE_SHA"},
  "evaluator_directory": "/private/frozen-evaluator",
  "image": "registry/model-image@sha256:IMAGE_DIGEST",
  "data_image": "registry/data-image@sha256:DATA_IMAGE_DIGEST",
  "data_python": "python3",
  "gpu": 0,
  "pass_timeout_seconds": 7200,
  "final_inputs": {
    "public_package": "/private/public-package",
    "prepare": {
      "mining_sha256": "MINING_MANIFEST_SHA",
      "benchmark_queries_sha256": "LABEL_FREE_QUERY_MANIFEST_SHA",
      "dev_embeddings_sha256": "DEVELOPMENT_EMBEDDING_SHA",
      "mining_count": 3642, "benchmark_count": 271,
      "dev_count": 2917, "holdout_count": 723,
      "holdout_ids_sha256": "FROZEN_ORDERED_HOLDOUT_ID_DIGEST"
    },
    "holdout_identities": {"path": "/private/holdout/queries.jsonl", "sha256": "HOLDOUT_QUERY_SHA"},
    "siglip_path": "/private/frozen-siglip",
    "historical_c": {"path": "/private/C/requests.json", "sha256": "FROZEN_C_REQUEST_SHA"},
    "holdout_gold": {"path": "/private/holdout/gold.jsonl", "sha256": "HOLDOUT_GOLD_SHA"},
    "benchmark_gold": {"path": "/private/benchmark/gold.jsonl", "sha256": "BENCHMARK_GOLD_SHA"}
  },
  "historical_predictions": {
    "B": {"path": "/private/B/final.json", "sha256": "B_RESULT_SHA"},
    "C": {"path": "/private/C/final.json", "sha256": "C_RESULT_SHA"}
  }
}
```

`model_files` must contain the complete already-frozen original-model file
inventory, not only the abbreviated example. Its values are file SHA-256
strings. This is the original HF model; the worker replays the frozen absolute
replacement patch exactly once before CUDA. Every arm uses this same state.
Do not point `model` at an already edited HF export and apply the patch again.

Normalize the already frozen development reference state, without recomputing
attributions or using final labels:

```json
{
  "r3_freeze_sha256": "R3_FREEZE_SHA",
  "helpfulness": {"MEASURED_DEVELOPMENT_REFERENCE_ID": 0.125},
  "atlas": ["FROZEN_DEVELOPMENT_REFERENCE_ID"]
}
```

`helpfulness` contains measured per-reference means, so the existing selector
performs the frozen class-mean/global-mean imputation for unmeasured references.
Do not pass all imputed development values as if they were measured. Keep the
normalization and source-artifact hashes with the owner's final configuration.

Only the two required source manifests and referenced images enter preparation;
no gold enters its container. Development embeddings are reused. Missing final
embeddings are computed once after freeze. Holdout retrieves from development
2917; Benchmark retrieves from original Mining3642 with existing identity,
image-byte, joint and component exclusions. The frozen C request file and PNGs
are opened only at this stage and checked against P01 semantics/pixels before
exact reuse for R4-C Benchmark. A mismatch stops rather than silently falling
back to a new C request.

Each physical arm/split has an exclusive `attempt.json` before submission,
native job records, original requests, worker attempts and outputs. There is no
automatic retry. R4 P00 aliases R3; R4 P01 aliases R4-C, with zero extra model
passes. A failure retains partial output and stops. Do not create a different
output directory to repeat already attempted final predictions.

Only after all final model passes complete does the scorer open final gold and
historical B/C predictions. The frozen external evaluator and existing paired
joint bootstrap produce `statistics.json` (2000 draws, seed42, 95% percentile
intervals, fixed groups). `results.json`, `terminal.json`, native job records,
worker token/latency/peak-memory fields, and 10-second `gpu-samples.jsonl` retain
best-effort costs. Failed in-flight usage and kernel-active GPU seconds may be
unknown. No heavy Service, Ledger, UID sandbox or provider relay is instantiated.
