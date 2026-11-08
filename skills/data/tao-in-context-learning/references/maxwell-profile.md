# Maxwell grid and bounded tools: explicit external adapter

Use `maxwell_profile.py` through the application's `icl_toolset.py` in a
platform job. The user separately supplies the authorized internal checkout
and frozen assets. No Maxwell source, dataset, cache or report is vendored in
this skill. Model resolution remains model-owned. The adapter accepts a
configured local checkpoint, while recording the native declared model path.

Profiles:

| Profile | Native arm | Initial context | Further tools |
|---|---|---|---|
| maxwell-grid32-cap128 | E_atlas_only | Fixed32 atlas + query + detailed task | None |
| maxwell-grid32-tools-guidance-cap128 | A_full | Same atlas + query + task/tools/guidance | Native search and class examples |

Pin source revision1711959a40f89cbf4072ea343de318a204903dfd, with the exact
`analysis.tht_ablation.Mode` and `analysis.tht_ablation_runtime.run_case` source
hashes enforced by the adapter. Do not route through the newer shared
conversation engine: it adds a tool-result reminder absent from this ablation.
Do not use historical module-level `run`, `prepare`, `verify` or `submit`, which
include grading, historical paths or scheduler behavior.

The original native backend uses greedy4096/seed0. These profiles explicitly
intercept only the output cap to128, retaining native rendering, decoding,
seed0 and tool semantics. Receipts record both native4096 and effective128.
This is a disclosed adaptation, not an exact4096 replay. The one-shot suite's
seed42/HF profile is a separate runtime. HistoricalR97 formatting-repair loops
are also separate; these profiles never repair a final answer.

Tools are optional, one call per model turn, at most10 tool rounds then one
final-only response. Stop at the first non-tool response, save it, then parse;
INVALID is terminal. A previous incomplete trace blocks automatic retry.
Terminal valid/invalid traces can be reused without a new generation.

Receipt v2 binds every input inventory, freshly prepared native settings, the
stored settings artifact and rendered atlas. Each terminal case seals its query,
trace and actual request JSON files, and revalidates every image artifact before
reuse. The final seal binds the exact set of case receipts. Any existing case
directory without a valid terminal seal requires explicit reconciliation before
continuing; do not automatically regenerate it. Pre-v2 outputs are not silently
upgraded or accepted as v2 evidence.

The initial atlas has four distinct Mining joints per class and remains fixed.
Native tools exclude atlas members/query joint, rank the next two classes by
max eligible C-RADIO spatial_mean_rotation_mean cosine, then return four
within-class nearest unrotated summary references. Search pages0–3 and class
pages0–9 remain native. These are not SigLIP features. Both profiles require
the frozen caches because native preparation constructs the same bank.

Native atlas construction has no per-query exclusion. This adapter adds an
explicit preflight overlap block; it never silently replaces a tile. Report
this stronger gate. Keep exact original train manifest bytes for native cache
binding; stage label-free val/test manifests instead of mounting the original
labeled evaluation directory. Query projection keys are exactly filename,
object_name, scan_id, image_sha256. No target label or score is accepted.

## Configuration

All inventories are JSON objects mapping root-relative filenames to SHA256.
Use a code/prompt-only external source view with every Python/text/JSON file
bound; exclude historical results/reports and datasets. Dataset inventory binds
the staged Mining manifest, label-free query manifest and permitted images.
Model inventories must cover the complete configured snapshot, including actual
weights and processor/configuration files. Cache inventories cover the complete
snapshot and must contain metadata.json and features.npz. This versioned THT
profile pins the original atlas/selection pair and independently verifies all32
selected image bytes, labels and joint identities against Mining. Register other
atlas pairs separately; the generic atlas selector remains available.

The no-tools profile above is the current E_atlas_only ablation. It is not the
historical Atlas v1 static protocol: that uses original JPEG queries, its original
system/user prompt and original unmodified atlas, with native greedy128. Keep
those experiment identities distinct when comparing reported scores.

```json
{
  "schema": "tao-icl-maxwell-native/1",
  "profile": "maxwell-grid32-tools-guidance-cap128",
  "native": {
    "checkout": "/native",
    "revision": "1711959a40f89cbf4072ea343de318a204903dfd",
    "source_manifest": "/input/native-source-hashes.json",
    "prompt_config": "config/current/tht-inspection.json"
  },
  "assets": {
    "dataset_view": "/dataset",
    "dataset_manifest": "/input/dataset-hashes.json",
    "atlas": {"path": "/input/reference_atlas.png", "sha256": "<hash>"},
    "selection": {"path": "/input/reference_selection.json", "sha256": "<hash>"},
    "queries": {"path": "/input/queries.jsonl", "sha256": "<hash>"},
    "training_cache": {"path": "/cache/train", "manifest": "/input/train-cache-hashes.json"},
    "query_cache": {"path": "/cache/query", "manifest": "/input/query-cache-hashes.json"}
  },
  "query_split": "val",
  "model": {"path": "/model", "inventory": "/input/model-hashes.json"},
  "protocol": {
    "examples_per_class": 4, "seed": 0, "reasoning": "off", "label_mode": "names",
    "execution_policy": "tht_ablation_no_format_repair_v1",
    "format_retries": 0, "max_new_tokens": 128
  },
  "output": "/results/profile"
}
```

Native preparation has a hardcoded settings split `val`; receipts retain that
native declaration beside the actual adapter split. They also distinguish
native declared model from actual configured checkpoint. Source and rendered
atlas hashes are separate. Native settings/traces contain per-query identities
and remain private. Standard output is an aggregate JSON receipt only.

Preparation performs no inference; `run` does. Use the platform's approved job
and narrow mounts, never mount gold or historical predictions. A completed
adapter run is raw classification evidence; a separate trusted scorer joins
gold after freeze. Synthetic cap/path/identity tests establish adapter behavior,
not measured model capacity, complete native preparation or score parity.
Require development-only preparation/tool/capacity smoke before a formal run.
