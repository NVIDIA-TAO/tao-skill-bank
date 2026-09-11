# Customer Adapter Contracts

The controller invokes customer logic only through command templates. Adapter
parameters are ordinary YAML values under `actions.<stage>.parameters` and can
be referenced as `{parameter_name}`. Existing local parameter files are hashed
into `data.lock.json`; secrets must be supplied through the execution platform,
not the workflow configuration.

## Source Data

`data.source_payload_contract` is a small JSON document binding source locators
to versioned, immutable payload roots. It has this shape:

```json
{
  "schema_version": "1.0",
  "immutability": "immutable",
  "datasets": [
    {
      "dataset_id": "wfm-aoi",
      "version": "dss-snapshot-12345",
      "root_uri": "file:///datasets/wfm-aoi-v12345"
    }
  ]
}
```

Use a DSS snapshot, object version, archive-set digest, or an equivalent stable
version identifier. The workflow hashes this contract on approval and resume.
Registered embedding stores are committed artifacts whose `shards` entries must
each include `relative_path`, `bytes`, and `sha256`; the controller verifies the
actual bytes and binds the resulting inventory digest into `data.lock.json`.
Create them with Data Services `register-store --source-payload-contract ...`.
Registration checks every locator against a contracted root and binds the exact
contract file plus a deterministic locator-audit digest into the store artifact.
The legacy `data.source_parts` path recomputes the same locator audit during
approval and resume; use a registered store for large corpora.
This gives image/archive provenance without materializing or hashing every
individual image in a large pool.

## Task Scoring

Custom score adapters must declare `actions.score.implementation_files`; custom
evaluators must declare `actions.evaluate.implementation_files`. These lists are
the complete executable dependency closures approved for the run, not merely the
entrypoint. The packaged GRIT adapter closure is discovered and sealed
automatically.

Large CUDA GRIT cohorts require `execution.capabilities.gpu_faiss: true`.
Local execution probes `StandardGpuResources` and `index_cpu_to_gpu` before the
score command is launched. External runners receive the capability in the stage
request and must reject an image that cannot provide it. The default TAO pip
dependency is CPU-only; build the scoring image with the supported GPU FAISS
package instead of allowing an implicit CPU fallback.

Runner `cancel` is idempotent and must return a normalized terminal state
(`COMPLETE`, `ERROR`, or `CANCELED`) only after the backend workload is terminal.
Unknown or nonterminal acknowledgements leave the workflow in `canceling`; a
later `cancel` or `run` reconciles it. Every external runner verb is bounded by
`execution.call_timeout_seconds`.

The scoring command receives the current DINO checkpoint, the fixed adaptive
target manifest, customer head configuration, and an output directory. It must
write `task_scores.parquet`, `score_commit.json`, and `_SUCCESS`.

Required Parquet columns:

| Column | Contract |
| --- | --- |
| `sample_id` | Identity from the target manifest |
| `task` | One configured `multi_task.tasks` value |
| `weakness_score` | Finite scalar; larger always means weaker |
| `embedding` | Fixed C-RADIO vector used only for source relevance |

Each `(sample_id, task)` pair must be unique. The adapter may use a frozen head
or refit a head on a fixed training split, but that policy and its inputs must
remain unchanged within a run.

The output identity set must exactly equal the target manifest identity set.
For GRIT this means every row whose `role` is `query`; for multi-task scoring it
means every declared `(sample_id, task)` pair. Partial score coverage is an
error, not an implicit sampling policy. Every output embedding must also match
the corresponding target-manifest C-RADIO vector exactly; scoring changes only
the weakness value, never the mining geometry.

`score_commit.json` must bind `input_sha256` for the target manifest,
`checkpoint_sha256` for the checkpoint used to produce weakness, and
`output_sha256` for `task_scores.parquet`. It must also record a nonempty
`implementation_sha256` covering the adapter and head-scoring implementation.
List imported implementation dependencies under
`actions.score.implementation_files`; the built-in GRIT action locks its CLI,
formula, and extraction pipeline automatically.
The controller also supplies `TAO_REFINEMENT_REQUEST_SHA256` and
`TAO_REFINEMENT_ENTRYPOINT_SHA256`, plus the approved implementation digest.
The commit must echo them as `request_sha256`, `entrypoint_sha256`, and
`implementation_sha256`; the trusted entrypoint must calculate its own digests
rather than copying environment values.
`_SUCCESS` contains the SHA-256 of `score_commit.json`. The controller verifies
this chain before target selection and again when adopting a cached score stage;
a score from a prior checkpoint is never accepted by path alone.

The controller converts `targets_per_round` and `task_weights` into a fixed
deterministic task allocation. Data Services ranks weakness independently in
each task, alternates among active tasks, and deduplicates `sample_id` globally.
Task preference therefore does not inflate the total per-round target budget.
For `multi_task.policy: balanced`, Data Services computes the allocation over
all configured tasks even when scoring has no actionable rows for one of them.
The unfilled quota is preserved. Materialization joins every selected neighbor
to its query task, keeps the cumulative manifest unique, and publishes a
separate task-balanced training view.

## Training

The training command receives the current checkpoint, cumulative training
manifest, pass count, output directory, and the resolved allocation through
`{training_nodes}`, `{training_gpus_per_node}`, `{training_world_size}`, and
`{training_optimizer_updates}` placeholders. It must publish the configured
checkpoint, `training_contract.json`, `training_commit.json`, the runtime
`experiment.yaml`, and `_SUCCESS`. The contract binds checkpoint and runtime
spec paths, byte sizes, and SHA-256 values. `training_commit.json` binds the
final contract, checkpoint, and runtime-spec digests; `_SUCCESS` contains the
commit record's SHA-256. The controller recomputes the full chain before
evaluation and whenever a completed run is reopened.

With `training.node_scaling`, the controller derives the allocation from the
cumulative manifest row count, the base spec's `dataset.batch_size`, and an
explicit optimizer-update floor. It writes `training_allocation.json` before
the leaf runs and includes the resolved resources in the stage request. Custom
adapters should apply the supplied node and GPU values rather than recomputing
them.

The resolved allocation also carries `sqrt_global_batch` learning-rate scaling
and its reference world size. Numeric base learning rates are scaled by the
square root of the effective/reference global-batch ratio. Interpolated rates
must explicitly depend on dataset batch size, node count, and GPU count. The
effective rule, multiplier, global batch, and allocation are recorded per
round.

Manifest-backed training uses a shard-aware distributed sampler: shard order
and rows within a shard are shuffled each epoch, but archive rows stay
contiguous enough for each worker's bounded handle cache. Random access is
limited to uncompressed tar or zip shards.

The packaged `nvidia_tao_pytorch.ssl.dinov3.data_refinement.train_cli` adapter
records the workflow's `checkpoint_policy`, same-round resume checkpoint,
sealed input-spec digest, effective overrides, and launch implementation
identities. `base_checkpoint_each_round` and `previous_round_checkpoint` must
therefore remain distinguishable in every round contract.

For multi-node training, platform runners launch exactly one adapter process per
node and provide `WORLD_SIZE` as **node count**, `NODE_RANK`, `MASTER_ADDR`,
`MASTER_PORT`, `NUM_GPU_PER_NODE`, and a unique
`TAO_REFINEMENT_LAUNCH_ID` for every gang attempt. The ID must change across
requeues and retries; a bare scheduler job ID is insufficient. The values must
exactly match the stage allocation, and standard process-level
`WORLD_SIZE=nodes*GPUs` is invalid. Rank zero seals the input spec and contract,
all launchers use private temporary copies, and only rank zero finalizes.

DINOv3 uses static `torchrun` membership. The runner must submit the nodes as a
gang, cancel all peers when any launcher fails, and restart every launcher with
a fresh rendezvous and attempt-scoped launch ID. Partial pod or rank retry is
invalid. Its terminal status reports the gang `attempt_id`. Every node requires
the same immutable base spec, manifest, parent/resume checkpoints, and shared
output path with atomic rename and hard-link support; the adapter verifies their
identities before spawning workers. Platform-provided `CUDA_VISIBLE_DEVICES`
must contain exactly the requested allocation and is preserved. The runner
should provide job-local temporary storage through `TAO_REFINEMENT_TMPDIR` or
`TMPDIR`.

Runtime placement, code transport, cache placement, and cleanup belong to the
platform runner, not this workflow. A runner may use an installed package,
container image, content-addressed archive, shared filesystem, or another
platform-native mechanism as long as it preserves the immutable request and
committed-output contracts. Deployments may impose stricter storage policies
without changing the controller or action interface.

GRIT feature arrays are disk-backed and require `actions.score.settings.work_dir`
or `TAO_LOCAL_SCRATCH`. The image reader projects locator columns only and loads
fixed mining embeddings only for query rows. View neighborhoods are computed
once at the largest configured `k`; cohorts above 50,000 rows require exact
FAISS so the reference index remains resident instead of being recopied for
every query block.

## Evaluation

When evaluation is enabled, round 0 evaluates the immutable base checkpoint.
Every later metrics payload must have the exact same unique `(task, name)`,
direction, and sample count contract, with finite numeric values. The evaluator
commit binds the checkpoint, metrics, benchmark identity, request digest, and
locked implementation. Declare imported evaluator dependencies through
`actions.evaluate.implementation_files`. The report presents raw values and
direction-normalized gain versus round 0. Evaluation remains optional; customer
heads and evaluators use this same leaf contract.

Registered source stores have two validation modes. `full_sha256` hashes every
shard at approval and resume and works without a prior local seal.
`sealed_inventory` requires a payload-binding artifact created by Data Services
after one complete shard hash pass; subsequent checks compare the committed
digest inventory and POSIX device, inode, size, mtime, and ctime identities.

GRIT actions declare node-local scratch through
`resources.local_scratch.path_environment`. The runner maps that environment
to platform scratch; the leaf computes required feature bytes from the approved
manifest and model width and fails before inference when capacity is insufficient.

Parent and cumulative training identities are always sent to retrieval as
exclusions so already-trained rows cannot consume the finite neighbor budget.
`parent_history_policy: exclude` also makes materialization fail closed if an
adapter nevertheless replays a row. A zero-row novel delta stops the loop only
after retrieval had the opportunity to search unseen neighbors.
Use `full_sha256` when a platform can replace files without changing those
identities or when the storage contract does not guarantee POSIX semantics.

`execution.backend: local` supports one controller process. With the default
`actions.train.execution_mode: runner`, multi-node fixed resources or node scaling
require `execution.backend: external` and these declared capabilities:
`gang_scheduling`, `gang_retry`, `attempt_scoped_launch_id`, and
`shared_filesystem`. An adapter that durably submits and adopts its own training
allocation instead uses `actions.train.execution_mode: adapter_managed`; its
`wrapper_resources` must remain one node, while the resolved allocation is
recorded separately in `execution_contract.adapter_managed_resources`.

The same execution choice applies to score, data, search, and evaluation
actions. This keeps the loop independent of the compute platform: a command may
be launched directly by a local, Kubernetes, Slurm, or custom runner, or it may
be a durable adapter that owns a platform-native child job. Adapter-managed
actions must make submission idempotent, adopt an existing matching child job,
and return only after committed outputs or a terminal failure are visible.

Each training output directory is bound to one preparation request. Repeating a
completed identical request is an idempotent success; changing its base spec,
manifest, parent checkpoint, passes, resources, or checkpoint policy requires a
new output directory. The adapter fails rather than replacing existing lineage.

## Evaluation

Evaluation is optional. A configured command receives the current checkpoint,
sealed benchmark manifest, customer parameters, and output directory. It must
write `metrics.json` matching `metrics.schema.json`,
`evaluation_commit.json`, and `_SUCCESS`. The commit binds the exact checkpoint
fingerprint, evaluation scope, and metrics file identity; `_SUCCESS` contains
the commit file's SHA-256. Additional per-sample outputs should also be listed
in the commit. A scheduler job ID alone is not a success seal.

Evaluation output is recorded in the report only. It cannot change selection,
mining radius, persistence, or stopping. The controller validates that the
adaptive target cohort and benchmark-unit manifest are disjoint.

With `scope: diagnostic_replay`, overlap is allowed for operational feedback,
but the output is not held-out evidence and must be labeled accordingly.

## Continuation

Never rewrite a prior run's state or lock files to adopt work produced under a
different workflow release. A continuation uses a new `run_dir`, sets
`workflow.start_round`, keeps `model.base_checkpoint` as the immutable training
initializer, sets `model.initial_scoring_checkpoint` to the accepted candidate,
and supplies `data.previous_training_manifest` plus a `continuation` contract.
The contract names the prior run, adopted round, sealed training contract,
commit, success marker, and prior data/release locks.

Run `adopt-training` before `run`. Adoption validates manifest bytes and rows,
base checkpoint, pass count, native topology, optimizer steps, final EMA teacher
name, runtime spec, checkpoint, and every seal. It records old and new lock
identities additively, marks only the adopted training stage complete, clears
stale jobs, and leaves evaluation and the round incomplete. The next `run`
performs exactly that evaluation before starting a new mining round.
The adoption record also seals the previous `state.json` and carries forward
`persistent_targets`, so recovery cannot reset noisy-sample suppression.

## Large-Scale Search

Use `actions.search.backend: audited_ann` for a persistent large-pool index.
Configure a committed dense-store manifest, index manifest, recall-audit
manifest, audited probe count, audited candidate depth, and separate candidate
and rerank command prefixes. The controller appends the concrete Data Services
subcommand and arguments. Candidate retrieval commits `ann_candidates.npz` in
`search_candidates`; reranking commits the normal search outputs below. These
are independent resumable jobs and may use different Python/CUDA runtimes.

The recall audit must belong to the configured index and approve the exact
probe count and candidate depth. cuVS output is never a final mining decision:
the rerank runtime reads immutable float32 C-RADIO vectors and applies exact
cosine relevance, parent/round exclusion, adaptive radius, and global duplicate
rejection. Relevance-radius expansion and duplicate rejection retain the same
meaning as exact search.

An optional `actions.search.backend: custom` plus `command` remains available
for customer indexes. It receives queries, the registered embedding-store
manifest, cumulative exclusion manifest, mining thresholds, and output
directory. Both indexed implementations publish:

- `neighbors.parquet`
- `search_summary.json`
- `artifact.json`
- `_SUCCESS`

The committed artifact records the exact locked search input: every
`source_shard` for direct Parquets, the `source_store_manifest` identity and
artifact ID for a registered exact/custom store, or the `dense_vector_store`
identity and artifact ID for audited ANN. Registered searches also record the
`query_embedding_contract`. The controller compares these inputs with
`data.lock.json` after every search, so a same-path store replacement fails the
stage instead of entering the cumulative training manifest.

`search_summary.json.search_proof` is either `exact_all_declared_shards` or an
audited ANN identifier beginning with `ann_audited_`. An empty exact result can
prove `radius_exhausted`; an empty ANN result produces
`search_budget_exhausted` unless the eligible source count is zero. Exact search
remains the terminal proof path when corpus exhaustion must be distinguished
from insufficient ANN depth.
