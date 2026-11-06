# Native generator addendum

Use `scripts/generator_addendum.py` for an explicitly authorized, separate
`generator-addendum-v1` experiment (arm G). The historical six-arm `suite.py`
continues to skip SDG. This route adds native AnomalyGen 1.2 to D's single-image
JSON classification contract; it does not change the paired OK/NG validator or
insert a partially populated arm into the six-arm store/report.

All run inputs, prepared source data, model weights, recipes, evidence, approval
and results stay outside the code checkout. This application consumes an
externally prepared package; it does not prepare or publish private datasets.
Read the consolidated `tao-generate-anomalies` skill before native execution.
Use the caller's approved Docker execution host and cached digest-pinned images.

## Start and resume

Copy [generator-task.example.json](generator-task.example.json) outside the
checkout. Replace every placeholder, zero count and zero storage estimate with
verified inputs. The example carries the complete frozen `generator.policy`
object, exported by `generator_policy.POLICY`; do not tune it during the run.

```bash
python scripts/generator_addendum.py preflight --task TASK.json --output PREFLIGHT.json
python scripts/generator_addendum.py run --preflight PREFLIGHT.json --approval APPROVAL.json
python scripts/generator_addendum.py status --results-dir RUN
python scripts/generator_addendum.py export --results-dir RUN
# Only after G's frozen final evaluation is complete:
python scripts/generator_addendum.py export --results-dir RUN --comparison-run ORIGINAL_SIX_ARM_RUN
```

Approval is external JSON with `preflight_sha256`, `approved_by` and
`approved_at`, binding the exact effective preflight. Preserve existing explicit
authorization; never manufacture an approval record. After that approval the
harness owns the fixed dependencies, generation, training, selection and final
evaluation. No supervisor chooses the next allocation, clean image, round or
checkpoint. A technical blocker saves evidence and stops.

Restart the identical command and inputs. `--resume-origin operator` is the
honest default for a manual restart; a supervising automatic recovery may use
`--resume-origin autonomous-recovery`. Both are recorded, neither expands the
budget or changes policy. A file lock prevents concurrent drivers. Completed
answers, including INVALID and length-limited answers, remain immutable. Only
infrastructure failures get a second attempt, in a fresh native job directory.
The generator's two-hour deadline starts at its first attempt, and survives
restarts; the overall run and other stage caps are 96 and 48 hours respectively.
A deterministic or quality/lineage error stops without a free retry.

## Task inputs and preflight boundary

Every file binding is `{ "path": "/absolute/file", "sha256": "...", "bytes": N }`.
The F selection uses `{path,sha256}` and must be the completed full-Proxy
seven-epoch selection. The classifier starts from the original base; F supplies
only the epoch count.

| Input | Required meaning |
|---|---|
| `public_package`, `private_package` | Separate frozen original data roles, original public/private checksum manifests; no resplit |
| `model_path`, `framework_image`, `direct_profile` | Shared converted base, original digest-pinned classifier runtime and exact original DirectMode prompt |
| `siglip_path`, `siglip_revision`, `data_services_image` | Original native retrieval model/files/revision and runtime |
| `epoch_selection` | Frozen F artifact; seven epochs, Proxy only |
| `m3_evidence` | Passed scorer, training, continuation, export, isolation, retrieval, loss-mask, paired and new-round gates; matching training settings |
| `generator_evidence` | Passed `native_byo_quality`, `amended_source_selection`, `source_data_review`; each contains nonempty `artifacts` file bindings |
| `environment_manifest` | External measured environment record including nonempty `host` |
| `storage_required_bytes` | Positive measured additional retention requirement; include failed attempts, five DCPs/exports and native generation/search artifacts |
| `retrieval_cache` | Original Mining/Proxy vector, input, spec, launch and terminal bindings; `approved_preflight` and `provenance` bindings |
| `generator` | Prepared package, exact adapter and training recipe, native base/checkpoint roots, immutable image, source counts and exact policy |

The cache provenance receipt contains `original_preflight_semantic_sha256`,
`inventory`, `readonly_original_stage_query.db` and
`completed_original_producer_stages`. Preflight verifies actual original
artifacts and performs its own fixed read-only query for only the two
Mining/Proxy producer stages. It requires COMPLETE/exit-zero receipts, exact
output hashes, original approved model/revision/runtime, full identity coverage,
finite nonzero vectors and original source/preprocessing parity. A descriptive
new attestation cannot substitute for these original records.

Preflight reads Proxy gold to freeze every class support. It never opens
Benchmark gold, scores or answers and starts no model. It binds all public
assets, private checksum manifest, classifier weights, source inventory,
checkpoint/recipe, native weight trees, symlink targets, code commit, budgets
and policy. Every current class support and real Mining count must be positive.
Resume verifies the sealed inputs and completed artifacts before new work.
A/C/E Codex runtimes, credentials and token-cap probes are not prerequisites.

`generator.runtime_mounts` is an optional list of `{source,target}` read-only
native runtime assets. The worker automatically binds the prepared package,
adapter, recipe, base/checkpoint paths and helper code at host-identical paths;
checkpoint root also appears at `/workspace/cdf-anomalygen/checkpoints`, and the
base source at `/base-source`. Declare any additional aliases required by the
staged cache, such as its VAE alias. Declare the optional HF CLI environment via
`generator.environment.IMAGINAIRE_HF_CLI_ENVIRONMENT` and a matching runtime
mount. Other environment settings are rejected.

A virtualenv may contain an absolute interpreter symlink that exists only in
the pinned native image. Supply `generator.container_symlink_targets` keyed by
that exact container path, with `{image,artifact}`. `artifact` binds a captured
copy of the original image file. Preserve extraction/image/no-masking evidence
in the acceptance artifacts. Preflight accepts only explicitly bound targets;
arbitrary dangling symlinks fail. Relative symlink chains and target bytes are
sealed too. Completed native manifests keep their original absolute paths.

## External prepared package

The package contains `dataset/`, ordered `class_mapping.json`, `inventory.json`
and `audit/{admitted_sources.jsonl,protected_joints.json,exclude_stems.txt}`.
The native dataset carries its defect specification and semantic CAD palette.
The map has exactly one native `texture+defect` key for each non-Good class;
order is preserved explicitly in the effective preflight.

Each admitted source row has `kind` (`clean` or `donor_reference`),
`source_stem`, `matched_identity` (original Mining ID, role, one class, normalized
physical joint, image path/hash), `classifier_training_admission:false`, and
exactly two assets. Clean assets are `clean_image` and `cad_mask`; donor assets
are `donor_image` and `donor_mask`. Each asset binds original archive member,
prepared relative path, original/prepared SHA-256, byte count and the declared
byte-copy/name-normalization transformation. `pixel_proof` proves decoded RGB
identity with frozen Mining. Inventory covers every file except itself.

Protected joints/stems, unregistered files, wrong classes, changed source or
mask bytes and mismatched CAD dimensions stop admission. Original donor
references are never appended as extra classifier training rows. Only D's
selected real Mining IDs and verified generated results enter training.

## Frozen amended allocation and source choice

For each defect class compute incoming full-Proxy error rate (including
INVALID), `rarity=1/sqrt(full real Mining class count)` and persistent decay.
From round two, halve decay when the previous allocation was positive and the
error rate did not decrease; floor .125, no upward reset. Compare rates using
integer cross-products. Use binary64 weights and class-map order for ties.

Allocate 25 weighted slots using Hamilton largest remainders. Allocate up to
three distinct zero-error exploration classes by least cumulative successful
final logical generation before training deduplication, then class-map order.
Redistribute any remaining exploration slots with a separate Hamilton call.
If all weights are zero use rarity alone. Keep zero counts in the seven-key
vector. Failed attempts, blocked cases and refinement alternatives do not enter
this historical logical counter. Each round requests exactly 28; five rounds
request at most 140 logical cases.

Cycle sorted erroneous Proxy IDs per class; if none, cycle all same-class Proxy
IDs. Select the highest-cosine unused approved clean using the original native
Mining vectors and the existing `retrieval.cosine` scalar formula. Freeze its
source hash; do not round or use an epsilon when ordering. Canonical stems must
be unique and break exact ties. One used-source set spans all classes within a
round and resets next round. Generic no-anchor fallback delegates clean choice
to native pairing from the unused allowed set; frozen formal supports make
that branch unreachable in this experiment. An insufficient pool stops.

`generation-requests/round-NNN.json` seals allocation, incoming Proxy/history
hashes, cache/policy/source bindings and work items before native launch. Work
items include global slot, round/class ordinal, class, anchor/fallback, full
precision similarity and authoritative clean/CAD paths and hashes. Reporting
includes anchored/allocated cases and distinct-used/eligible anchors separately.

## Native composition and immutable admission

The worker first calls native `roi_allocate` with the complete explicit count
vector and verifies all seven realized allocation values. It then makes narrow
per-class clean/CAD views with the approved donor
masks, calls native `roi_pair --per_clean_pairing on` with one positive class,
verifies exact coverage, then merges native pair records in frozen slot order.
One native `roi_place` call owns all mask placement. Native CAD/THT placement,
largest-component donor transform and one placement per pair remain unchanged.
The seed is `42+round-1`; merged pair ordering fixes the native placement stride.

Preserve the native sorted testcase. A separate final testcase is reordered to
work-slot order and carries explicit `index=slot`, fixing diffusion seed
`42+round-1+1024*slot`. Seal the original, filtered, final and permutation
artifacts. Bind original donor image/mask, actual retained transformed mask,
CAD/palette, placed mask, native pair row and selected source. `TMPDIR` is inside
the durable attempt directory before native placement starts.

The shared 1.2 runner generates the entire batch once and runs three NN quality
rounds. Native per-sample BO uses minimum two distinct observations and 1024
candidates, with deterministic uniform fallback, draw seeds 1/2/3, two decimal
parameter precision, guidance 1–10 and the frozen trained crop band 1.5–2.5.
Keep initial diffusion controls (35 steps, guidance3, crop2, one output, no
Poisson blending) and diffusion seeds. Keep best finite NN, original/earliest
on ties; verify selected image bytes against the winning native bucket. No
filtering or top-up. Valid all-blocked contributes zero new synthetic rows;
missing, incomplete or malformed output stops.

`load_generation` and `validate_lineage` remain the shared native completion
boundary. Application admission adds exact target mapping, source ancestry,
per-class counts, refinement identity/control checks and selected-bucket
verification. All cumulative records, including rows later removed as
duplicates, are revalidated on reassembly/restart. Deduplicate decoded RGB
256x256 NEAREST padded training pixels; real rows precede earliest synthetic
round/index. Conflicting labels on identical training pixels stop.

The actual admitted population sets native seven-epoch step counts and the
cumulative scheduler horizon. Each round resumes only its own immediately
previous DCP model/optimizer/scheduler/trainer; reset the dataloader for a new
population, preserve its cursor for recovery within a round. Full Proxy chooses
the earliest maximum-MCC terminal round. Freeze model/policy before exactly one
final Benchmark candidate. Retain raw completed answers without repair.

## Acceptance and export

The root-owned runtime acceptance can invoke the same worker with
`--request REQUEST.json --output NEW_HOST_IDENTICAL_DIRECTORY --prepare-only`.
A subsequent fresh full smoke exercises normal quality and admission. These
are setup evidence; neither smoke outputs nor fixture answers enter formal
training or formal results. CPU tests do not claim model or visual quality.

The addendum exports `generator-addendum-report/1`: G only, round allocations,
source/anchor coverage, native initial/refinement/selected evidence, real and
synthetic composition/exposure, original raw invocation costs and failed
attempt ledgers. Selected copies are not counted as extra renders. Missing
measurements and prices stay unknown. Training telemetry measures actual total
attempt exposure; per-source exposure is separately reported as admitted
population times epochs. Operator resumes and failure stops remain visible.

`report_data.json` is the first completed snapshot. Other snapshots live under
`exports/`; `latest-report.json` points to the newest immutable export. Re-export
never changes a sealed completed snapshot. Optional old-D comparison opens the
original store read-only only after G's freeze and complete final evaluation;
it verifies old frozen prediction artifacts and uses the original external
scorer with paired physical-joint bootstrap (seed42, 2000, 95%). It never reruns D.

Reports must disclose unknown generator-adapter training overlap with
Benchmark, original retained Mining/Benchmark joint overlap, and intended
synthetic labels without independent human class verification. This is an
engineering addendum, not a leakage-free causal estimate. Existing six-arm,
supplementary and final-delivery artifacts remain separate.
