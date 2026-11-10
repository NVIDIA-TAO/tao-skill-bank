---
name: tao-in-context-learning
description: Select, evaluate and freeze in-context learning policies from Proxy evidence, or run an explicitly requested Simple harness with a fixed class grid and Cosmos-driven example/search tool calls. Use for image reference retrieval, ICL search and bounded multi-turn visual evidence gathering.
license: Apache-2.0
metadata:
  version: "0.2.0"
  author: NVIDIA Corporation
allowed-tools: Read Bash Write
---

# In-context learning

The application owns dataset roles, budgets, scoring and final freeze. This
skill owns policy proposals and toolset contracts. Read the current Proxy
evidence, identify an unresolved error pattern, propose **one** candidate with
a rationale, run its complete Proxy evaluation, then reconsider. Do not plan
a grid or preselect all candidates. Never use Benchmark feedback to choose a
policy. An exhausted budget with no completed candidate is incomplete.

For an explicit **Simple harness** request, use the separately versioned
[Simple harness contract](references/simple-harness.md) and
`scripts/simple_harness.py`. The application selects this route without
starting DEFT or the adaptive policy search. A fixed 8-class × 4-example
grid initializes the context; Cosmos chooses optional `get_class_examples`
and `search_similar_examples` calls, sees their returned images, and decides
again. No Astra agent chooses the tools on Cosmos's behalf. The runner uses
the pinned, externally supplied Maxwell source; it does not vendor that code.
Keep this path distinct from the one-call SigLIP toolset below.

Read [references/toolsets.md](references/toolsets.md) for the toolset interface.
The initial `siglip-retrieval-v1` toolset composes
`tao-generate-image-embeddings` and `tao-mine-nearest-neighbors`. Use their
packaged validators, immutable image and platform four verbs. There is no
encoder sweep. Use the task's pinned local SigLIP snapshot and keep its
processor, revision and embedding-spec hashes with both embedding parquets.

The trusted pool builder excludes the query's source identity, original image
bytes and normalized physical joint **before** mining. Query manifests contain
no labels. The classifier receives only that query and selected, allowed
Mining references. A classifier never reads private provenance to perform
its own exclusions. Missing identity evidence blocks retrieval.

For this toolset use `scripts/retrieval.py`: validate policy; produce label-free
query and allowed-source parquets; call the two existing skills; consume the
per-query mined output; deterministically order references; build the actual
multimodal messages. Store IDs, image order, original/view hashes, prompts,
policies, specs and job handles. A future toolset can perform multiple calls
or interactive tool use by implementing the same provenance/budget/result
contract; do not restrict the application to a fixed prompt grid.

The THT suite uses at most 12 **complete** Proxy trials per arm, and equal
frozen wall/token budgets for base and trained ICL. Failed attempts consume
resources. Choose highest MCC; candidates within 0.01 of the maximum prefer
fewer images, fewer calls, shorter prompt, then trial ID. Record the selected
policy as an immutable artifact before Benchmark. An incomplete trial cannot
win selection.
