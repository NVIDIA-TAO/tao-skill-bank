# Simple harness: native grid and bounded tools

Use `simple_harness.py` through the application's `simple_harness.py` in a
platform job. The user separately supplies the authorized internal checkout
and frozen assets. No Maxwell source, dataset, cache or report is vendored in
this skill. Model resolution remains model-owned. The adapter accepts a
configured local checkpoint, while recording the native declared model path.

Profiles:

| Profile | Native arm | Initial context | Further tools |
|---|---|---|---|
| simple-harness-grid32-tools-guidance-native4096 | A_full | Fixed32 atlas + query + task/tools/guidance | Native search and class examples |
| simple-harness-grid32-cap128 | E_atlas_only | Fixed32 atlas + query + detailed task | None |
| simple-harness-grid32-tools-guidance-cap128 | A_full | Same atlas + query + task/tools/guidance | Native search and class examples |
| simple-harness-grid32-explore-cap128-v1 | A_full + TAO guidance | Same context plus progressive evidence guidance | Model-selected new search pages / class examples |

The exploration profile appends [versioned guidance](simple-harness-explore-v1.txt)
to the native user prompt. It changes neither tool schemas nor tool selection
in the runner: Cosmos may answer directly, request another page, or stop.
It retains greedy128/seed0 and the ten-tool-round bound, so compare it with
the cap128 profile to avoid simultaneously changing the generation budget.
The full effective prompt and guidance SHA256 are bound in preparation and
revalidated at run time. Receipts disclose `prompt_deviation` and
`generation_deviation`. A modified guidance file invalidates prior preparation.

This is a TAO extension, not an exact Maxwell reproduction. Search pagination
changes the candidate-class pair; it does not rewrite the embedding query.
Search pages are 0–3; class-example pages are 0–9 for one or two class names.
Duplicate successful requests and invalid tool retries are not new evidence.

After completion, audit without gold or new inference:

```bash
python scripts/simple_harness_audit.py --results-dir /results/profile \
  --output /results/exploration-audit.json
```

Report successful distinct search pages per query, queries that actually
performed additional searches, new reference IDs from those searches,
duplicate calls, tool errors, INVALID and calls/costs. New-reference counts
are relative to earlier tool results for that query, not to the initial atlas.
Use a separate trusted Proxy scorer for accuracy. More calls do not establish
better classification; do not infer hidden reasoning from these statistics.

Pin source revision1711959a40f89cbf4072ea343de318a204903dfd, with code/prompt hashes in `simple-harness-source-pins.json`, including the
mode, native backend, conversation helpers, tool schemas, retrieval, rendering
and prompts enforced by the adapter. These are mandatory core pins, not a
complete source-slice recipe: supply the complete code/prompt-only view and
bind all transitive imports with its source inventory. Do not route through the newer shared
conversation engine: it adds a tool-result reminder absent from this ablation.
Do not use historical module-level `run`, `prepare`, `verify` or `submit`, which
include grading, historical paths or scheduler behavior.

The default native4096 profile calls the original greedy4096/seed0 backend
without a generation override. The explicitly named cap128 profiles intercept
only the output cap to128. Both retain native rendering, decoding and tools.
Receipts record the native and effective limits. Cap128 is a disclosed adaptation. The one-shot suite's
seed42/HF profile is a separate runtime. HistoricalR97 formatting-repair loops
are also separate; these profiles never repair a final answer.

Tools are optional, one call per model turn, at most10 tool rounds then one
final-only response. Stop at the first non-tool response, save it, then parse;
INVALID is terminal. A previous incomplete trace blocks automatic retry.
Terminal valid/invalid traces can be reused without a new generation.

Receipts bind every input inventory, freshly prepared native settings, the
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
pages0–9 remain native. These are not SigLIP features. All profiles require
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
atlas pairs separately; use the separate SigLIP toolset for the existing adaptive one-shot workflow.

The optional no-tools profile above is the current E_atlas_only ablation. It is not the
historical Atlas v1 static protocol: that uses original JPEG queries, its original
system/user prompt and original unmodified atlas, with native greedy128. Keep
those experiment identities distinct when comparing reported scores.

```json
{
  "schema": "tao-icl-simple-harness-native/1",
  "profile": "simple-harness-grid32-tools-guidance-cap128",
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
