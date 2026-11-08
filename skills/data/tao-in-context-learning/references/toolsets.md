# Reference and toolset contracts

Every policy records toolset/version, rationale, max_calls, parameters, input
hashes, selected identities/order, actual rendered prompt and a JSON receipt.
Model weights and labeled query answers are not policy inputs. The application
owns role visibility, preprocessing, runtime, capacity and final freeze.

## Legacy siglip-retrieval-v1

Version1 parameters remain exactly `reference_count`, `ordering`
(`nearest_first`, `farthest_first`, `class_grouped`) and `instruction`;
max_calls1. `retrieval.py` and original suite behavior are unchanged. It uses
existing `tao-generate-image-embeddings` and `tao-mine-nearest-neighbors`, pinned
SigLIP snapshot/processor/specs. TMM outputs unique source paths, so invoke one
query per mining operation; reorder returned candidates by exact cosine and
reference ID. Mine the measured capacity once and share the retrieval cache
across prompt/order variants. Do not use its truncated candidate list as a full
class-discovery pool.

Original exclusion order is source identity, original bytes, normalized
physical joint, then reference source-ID/byte deduplication in manifest order.
The legacy selector does not enforce diversity between selected reference
joints. Keep this distinction when changing reference count or class quotas.

## similarity-reference-v2

Version2 parameters are all explicit:

```json
{
  "toolset": "similarity-reference-v2",
  "version": 2,
  "rationale": "Compare two likely classes using one example each",
  "max_calls": 1,
  "parameters": {
    "strategy": "different_classes",
    "reference_count": 2,
    "class_count": 2,
    "per_class": 1,
    "distinct_joints": false,
    "ordering": "class_ranked",
    "instruction": "Use the labeled reference images as examples. Follow the original query instructions and output format."
  }
}
```

`nearest` selects reference_count from the full eligible pool, with class_count
and per_class null. `different_classes` ranks classes by maximum eligible
cosine, configured class-order ties, then takes per_class references from the
first class_count classes. Within a class, descending cosine/reference-ID ties.
Reference count equals class_count×per_class. `nearest_first` presents all
selected references globally sorted; `class_ranked` groups candidate classes.
`distinct_joints` adds within-group joint diversity; it never relaxes the
query exclusion. Insufficient quota blocks instead of replacing a class.

The output class vocabulary remains the original configured task vocabulary;
two candidate classes do not implicitly constrain the answer to those two.
Class-coverage analysis belongs to trusted scoring after selection, never to
retrieval. Use complete hash-bound embedding caches; no query gold.

The generic CLI consumes JSON arrays of identity/reference records with
embeddings and an explicit class vocabulary:

```json
{
  "schema": "tao-reference-preparation/1",
  "mode": "similarity",
  "queries": {"path": "/input/query-identities.json", "sha256": "<hash>"},
  "references": {"path": "/input/mining-references.json", "sha256": "<hash>"},
  "classes": ["<class-a>", "<class-b>"],
  "reference_capacity": 2,
  "policy": "<replace with the complete policy object above>",
  "output": "/results/selected-references.private.json"
}
```

The application renders selected image/label pairs before the query using the
frozen task profile. These are static one/few-shot inputs, not model-driven
tool calls. A separate1NN control copies the first label without invoking a VLM.

## fixed-atlas-selection-v1

Use the same CLI with `mode: fixed_atlas`, classes/per_class/salt and the same
bound query/reference assets; omit policy/reference_capacity. Selection sorts
SHA256(salt+referenceID), keeps distinct physical joints per class, and emits
a fixed membership manifest. Default salt is `atlas-v1:`. Native Maxwell
pixel parity additionally requires its exact filename identities, selection
manifest, renderer/font and frozen atlas; the generic selector alone does not
establish that parity. Fixed-grid overlap audit blocks, without per-query tile
substitution. For the eight-class four-example grid, show `ICL (1 grid: 8
classes × 4 images)`, distinct from `ICL (1 grid + tools + guidance)`.

## Future toolsets

Declare input artifacts/roles, atomic operations, measured capacity, call/token
budget, stopping/resume rules and output schema. A bounded interaction can
emit multiple requests instead of static messages. Keep the application-level
ICL section high-level; detailed strategies belong here or a toolset reference.
Do not expand the frozen legacy suite schema as a side effect of adding a new
executor. Register and test the new entrypoint explicitly.
