# Toolset contract

Common policy fields: `toolset`, `version`, `rationale`, `max_calls`, and
toolset-owned `parameters`. A toolset declares required artifacts, allowed
role views, operation costs, capacity limits measured in smoke tests, and its
output type. It must emit actual prompts/calls, ordered reference identities,
input/output hashes, job handles, wall time and model/agent usage. Unknown
measurements are null with a reason. Future toolsets may emit a call plan or
execute a bounded interaction instead of static messages.

`siglip-retrieval-v1` version 1 uses `reference_count`, `ordering`
(`nearest_first`, `farthest_first`, `class_grouped`) and `instruction`.
Its `max_calls` is 1. Reference count must be positive and no larger than the
capacity validated by the application. The agent selects these values from
Proxy evidence, one trial at a time. The toolset makes no initial grid.

Encoder: `google/siglip-base-patch16-224`, immutable revision supplied in the
task profile. Both source and target use the exact local snapshot, processor
and embedding configuration. The first implementation uses the existing
TAO Data Services `embedding image_embeddings` and `tmm nearest_neighbors`
entrypoints. The latter outputs unique source paths, not query/neighbor pairs;
invoke it with a single target per query so provenance is unambiguous. Source
exclusions happen before that invocation. Reorder its returned candidates by
cosine similarity from the same embeddings (stable source-ID tie break).

Input queries have `id`, `image`, `image_sha256`. Trusted exclusion identities
add `source_id`, `object_name`, `scan_id` without gold. References add one
allowed `labels` entry and those identity fields. Use a validated label-free identity projection; never mount private source
rows into classifiers or proposal agents. `SCN_` is removed from the scan identifier before equality.

Static output: an OpenAI-style `messages` array, ordered reference IDs,
policy hash and image hashes. Query and reference PNGs have the declared task
preprocessing. No atlas or label-derived visual hint is added implicitly.
The THT adapter applies its RGB padded256/NEAREST query profile to both source
and target embedding views before the pinned native SigLIP processor, and
records both original and embedding-view hashes.

Cache native retrieval by embedding hashes, allowed IDs, query identity, metric,
capacity and runtime digest. Prompt/rationale/presentation order are not retrieval
inputs. Mine the validated maximum once, then select the nearest requested count
and apply the proposed order. Record shared cache costs once.

## Simple harness (explicit route)

`simple-harness-grid32-tools-guidance-native4096` delegates to the pinned
external native Cosmos loop. It emits a multi-turn trace rather than one
preassembled single-call message. The fixed initial grid, C-RADIO feature
caches, two optional tools and guidance are declared together. Cosmos owns
tool selection; the runner only executes and records requests. See
[simple-harness.md](simple-harness.md) for isolation, cap128 adaptation,
source/model pinning and terminal-answer/resume semantics. This route does
not change `siglip-retrieval-v1` or use its policy validator.
