# Simple harness with user data

`simple-harness-siglip-v1` is a packaged, inference-only Cosmos tool loop.
It needs this skill bank, user-owned data and local model snapshots. It does
not import `aoi_harness` or require Maxwell's source checkout or THT assets.
New code follows the skill bank's Apache-2.0 license; separately acquired
models, containers and data retain their own terms.

## What runs

```text
TAO image-embedding skill → SigLIP reference/query vectors
                                    ↓
query + fixed labeled grid → Cosmos chooses an action
                              ├─ search_similar_examples(page)
                              ├─ get_class_examples(classes, page)
                              └─ final class label
                                       ↑
                  requested references and their images
```

Cosmos makes each per-image tool decision. The runner validates arguments,
returns evidence, and enforces the budget. It neither chooses a tool on
Cosmos's behalf nor forces a search. No Astra controller enters this loop.
DEFT, training and generation are skipped.

| Component | Packaged behavior |
|---|---|
| Encoder | Existing `tao-generate-image-embeddings`, local SigLIP snapshot |
| Initial grid | Configured class order, four examples/class by default; fixed for the run |
| Search | Rank eligible classes by their best cosine match; two classes/page, up to four references/class |
| Class examples | Cosmos names one or two classes; nearest references within each; page advances within class |
| Model loop | One optional call/turn, at most ten tool rounds plus one final-only response |
| Final output | Strict `{"labels":["one configured class"]}` after decoding complete native thinking/answer wrappers; invalid/truncated answer is terminal |
| Tool error | Returned to Cosmos within the budget; not automatically a wrong final prediction |
| Repeated search | New pages expose other classes, using the same query embedding; no new encoder forward per call |

This follows the grid/tools/guidance structure of the Simple harness. It is
**a separate SigLIP protocol**, with generic classes and TAO-owned prompts and
runtime. It does not promise exact native rendering, trace parity or scores.
The old C-RADIO profiles remain available through the [historical adapter](simple-harness.md).
C-RADIOv2 is not supported by the existing TAO image-embedding skill; it is
not silently emulated with a different encoder. The one-call SigLIP ICL path
is also separate from this multi-turn runtime.

## Inputs

Copy [packaged-simple-harness.example.json](packaged-simple-harness.example.json)
outside the repository. All paths must resolve identically in preparation,
embedding and inference containers. `model.path` is a local Cosmos3-Nano VLM
snapshot prepared by the model skill (`model_type=qwen3_vl`); `siglip_model`
is a pinned local HF SigLIP snapshot, including its processor configuration.
The runtime uses local files only and never downloads weights.

Reference JSONL (one object per image):

```json
{"id":"reference-001","image":"reference-001.png","label":"Scratch","group_id":"board-4-joint-3"}
```

Query JSONL, **without labels, gold or extra fields**:

```json
{"id":"query-001","image":"query-001.png","group_id":"board-8-joint-9"}
```

Image paths are relative to `image_root` or absolute paths inside it. IDs and
paths must be unique within each inventory. Supply physical `group_id` for
every image to exclude other views of the same component/joint. If unavailable,
omit it everywhere: the receipt explicitly reports that physical-group
isolation is **not established**. The runner cannot infer physical identity
from filenames. It always excludes matching IDs and original image bytes.
For a formal benchmark, establish and review group identities beforehand.

The fixed grid selects references deterministically by ID within class, after
excluding overlaps with any query. It requires the configured number of
distinct eligible images/groups per class. Tools additionally exclude grid
members, their byte duplicates and supplied groups. Keep enough other
references to make tool queries useful. Missing groups, missing classes or
too few safe grid images should be addressed in dataset preparation, never
by borrowing query labels. Empty search pages are legitimate tool results.

## Quick start

Run commands below from the bank root, in the approved runtime with identical
mount paths. Preparation uses Python, pandas/pyarrow, NumPy, Pillow and PyYAML;
inference additionally requires PyTorch, Transformers with Qwen3-VL support,
and CUDA. Use the existing model/platform runtime and its environment check;
do not install Maxwell's package. No new container or Python dependency is
introduced to the bank's dependency manifests.

1. Resolve the model with `scripts/resolve_tao_model.py`, prepare its local
   VLM checkpoint through the model skill, and prepare the pinned SigLIP snapshot.
   Select the Docker platform, reserve the GPU, and resolve images from
   `versions.yaml`; the embedding image is `images.tao_toolkit.data_services`.
   The model runtime must support `Qwen3VLForConditionalGeneration`.
2. Stage input parquets and two specs. This performs **no inference**:

   ```bash
   python skills/data/tao-in-context-learning/scripts/simple_harness_runtime.py stage \
     --config /work/input/simple-harness.json
   ```

   `staged.json` binds the inventories, images, source and encoder snapshot.
   Both parquet inputs contain only `filepath`, even for labeled references.
3. Use [tao-generate-image-embeddings](../../tao-generate-image-embeddings/SKILL.md)
   to validate each staged YAML and dispatch two embedding jobs through the
   platform `submit`/`status`/`logs`/`cancel` contract. Its native command is
   `embedding image_embeddings -e /work/staging/simple-harness/references.yaml`
   (then `queries.yaml`). Use the exact staged specs and same encoder snapshot;
   keep job records and image digests. Do not override native CLI parameters
   after staging. A parquet alone cannot prove which encoder generated it.
   Mount both input parquets **and their image roots** at the staged paths.
4. Run the application CPU preflight, which consumes the embeddings, renders
   the grid, hashes the prepared local model, and freezes the complete inputs:

   ```bash
   python skills/applications/tao-run-deft-icl-aoi-cosmos3/scripts/simple_harness.py preflight \
     --config /work/input/simple-harness.json --output /work/preflight.json
   ```

   Preflight does not prove GPU capacity or model tool competence. Run a
   separately approved development smoke first, in a different output directory.
   Inspect actual Cosmos calls and later requests containing returned images.
   Synthetic scripted tests prove the runner, not model agency or accuracy.
5. With the user's approval bound to that preflight (`preflight_sha256`,
   `approved_by`, `approved_at`), open the platform job record **before launch**.
   Submit this worker with that job ID:

   ```bash
   python skills/applications/tao-run-deft-icl-aoi-cosmos3/scripts/simple_harness.py run \
     --preflight /work/preflight.json --approval /work/input/approval.json \
     --job-id "$JOB_ID"
   ```

   Mount code, models and inputs read-only, output writable. Do not mount
   benchmark gold, prior predictions/reports, host home or credentials. Keep
   query filenames/IDs opaque when they encode answers; the model needs pixels,
   task/class text and allowed reference labels only.
6. Inspect `status --results-dir /work/results/simple-harness` for artifact
   completion; poll the platform for live execution state. Join predictions
   to gold only in an independent evaluator after freeze. Never tune from test
   scores or assign this profile the historical 0.727/0.731 result.

Changing config, code, images, embeddings or model files invalidates the
frozen run. Use fresh staging/output directories for a different experiment.
Completed valid and INVALID cases are reused without calls. Interrupted cases
preserve their last request and stop for reconciliation; they are not retried
automatically. The runner locks the output directory against duplicate workers.

## Evidence and limits

`prepared.json` and its seal bind the reference/query inventories, grid,
encoder/model/source hashes, normalized vectors and exclusion coverage.
Each `cases/<hashed-id>/trace.json` records actual messages, image paths and
hashes, tool arguments/results/errors, raw output, parse outcome, tokens,
generation time and measured peak GPU allocation. `predictions.json` contains
no gold; `complete.json` summarizes completion and seals the outputs.

These file hashes detect accidental drift, not malicious tampering or an
external embedding job using the wrong CLI. Mount isolation and job provenance
remain platform responsibilities. A valid JSON answer is not evidence of a
correct class. Repeated calls are not evidence of useful exploration; inspect
distinct successful search pages and new reference IDs. Context growth can
exhaust memory before the round limit; a context/OOM failure preserves the
case and requires a new reviewed run with a smaller image budget, not silent
truncation or extra trials.

## Suggested user prompt

```text
Run the packaged Cosmos Simple harness on my dataset, using simple-harness-siglip-v1.
Use my classes and labeled reference pool. Generate SigLIP embeddings with the
existing TAO embedding skill; let Cosmos choose example/search tools and inspect
their returned images over multiple turns. Skip DEFT, training and AnomalyGen.
Start with a development smoke, keep the tool traces and costs, and freeze the
configuration before any separately authorized test evaluation.
```
