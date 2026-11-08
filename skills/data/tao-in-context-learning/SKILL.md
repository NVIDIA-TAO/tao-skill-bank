---
name: tao-in-context-learning
description: Prepare, compare and freeze image-reference ICL policies, including one-shot similarity, nearest-N, examples from different classes, fixed class-balanced grids and a bounded Maxwell-compatible tool harness. Use when users request ICL, one/few-shot examples, grid atlases, similarity-only controls or reference/tool policy changes; compose existing TAO embedding and platform skills without changing frozen experiments.
license: Apache-2.0
metadata:
  version: "0.2.0"
  author: NVIDIA Corporation
allowed-tools: Read Bash Write
---

# In-context learning

The application owns dataset roles, model/prompt/runtime, budgets, scoring and
freeze. This skill owns reference selection, presentation and bounded toolset
contracts. Keep these three different questions explicit:

- **One-shot similarity**: a retriever selects one labeled image; a VLM predicts
  from that reference and the query. This is reference-based ICL, not zero-shot.
- **Similarity only (1-NN, no VLM)**: return the nearest reference's label. This
  is a retrieval classifier, not a Cosmos model result. Include this control when
  the question is whether the VLM adds value beyond retrieval.
- **Grid / tool-assisted ICL**: multiple class examples and, when selected, tools
  for the model to request more evidence. A fixed grid alone is not a tool harness.

Report weight status independently: Zero-shot has no task fine-tuning or image
examples; ICL has examples but no weight update; Fine-tuned can be combined with
ICL. Do not label every pretrained model invocation zero-shot.

## Select the requested execution mode

Read [toolsets.md](references/toolsets.md) and, for Maxwell, the exact
[native-profile contract](references/maxwell-profile.md).

| Toolset | Purpose | Entry point |
|---|---|---|
| `siglip-retrieval-v1` | Reproduce frozen legacy TMM-backed policies | `scripts/retrieval.py`; existing application suite |
| `similarity-reference-v2` | Nearest-N or top candidate classes with a fixed examples-per-class quota | `scripts/reference_policy.py --config CONFIG` |
| `fixed-atlas-selection-v1` | Query-independent, deterministic class-balanced atlas membership | Same preparer, `mode: fixed_atlas` |
| `maxwell-grid32-cap128` | ICL (1 grid: 8 classes × 4 images), no tools | `scripts/maxwell_profile.py prepare|run --config CONFIG` |
| `maxwell-grid32-tools-guidance-cap128` | Same initial grid plus native search/class-example tools and guidance | Same native adapter |

Do not silently replace a frozen policy with a new mode. The legacy six-arm
suite remains reproducible. New modes use separate run/configuration/freeze
artifacts. Other encoders, toolsets, class counts and datasets may be registered
explicitly; the generic selector does not hardcode THT labels or a model path.
The optional Maxwell adapter is explicitly THT-specific external integration.

## Evidence and selection

When the user supplies fixed comparison arms, register those arms and run the
bounded comparison as requested; do not invent adaptive selection. Otherwise,
read Proxy evidence and propose one candidate at a time with a rationale. The
application determines the trial limit and tie-breaks. Never use Benchmark
feedback to select or alter a reference policy. An incomplete trial cannot win.

Freeze the exact encoder revision/processor, source manifest, query identity
projection, embeddings, reference algorithm/order, prompt, model inventory,
runtime, output parser and call budget before final evaluation. Record a JSON
receipt for every operation. Failed attempts consume resources and remain in
the report. Keep per-query provenance private and return aggregates to the
coordinator when the experiment requires an evaluation firewall.

The default query projection contains no target labels. Exclude the query's
source identity, original image bytes and normalized physical joint before
similarity ranking. Reference labels are allowed only from the frozen Mining
pool. Missing identity or an unsatisfied class quota blocks preparation; do not
silently use evaluation labels, substitute a class or relax an exclusion.

The fixed atlas is deterministic across queries. Audit overlap before use;
never quietly replace individual tiles and still call it the original fixed
grid. Its selection manifest and rendered pixels are separate frozen artifacts.

## Capacity, execution and costs

Measure the actual maximum image/pixel context on development inputs before
Benchmark. One grid is one model image but contains many examples; report both
counts. More references are not assumed better. Three images and nine images
are different capacities; old five-image evidence does not prove nine.

Use the application's selected platform four verbs and durable job records.
Mount the classifier's actual query/references and model only, with its approved
code. Gold, historical predictions and broader datasets stay outside that
runtime. Native tool execution may hold its approved Mining bank; tool results
expose only allowed examples, labels and counts, not target gold.

Preserve completed answers, including INVALID. Resume infrastructure failures
only at the documented boundary; an ambiguous in-flight generation blocks
automatic replay. Record actual input/output tokens, tool/model calls, wall
time and peak memory where available. Unknown GPU kernel time/currency is null,
not zero. Keep pure retrieval costs distinct from VLM costs.

The new native adapter is a cap128 adaptation of the screenshot's **no-format-
repair** executor. Historical R97's formatting repair is a different protocol.
Do not claim full Maxwell score parity from synthetic adapter tests, static
reference tests or the nearest-one result. Report validation scope explicitly.
