# DEFT IAA SDG on SLURM

Load this reference only for the `tao-run-deft-iaa` SDG action. Normal TAO
training, evaluation, inference, and ordinary multi-node jobs continue to use
the standard SLURM contracts in `SKILL.md`.

## Topology

`generation_nodes=N` means **up to N independent image-worker jobs**, not one
N-node allocation or a gang-start requirement:

- Each image-worker job requests exactly one node and eight GPUs.
- Each worker launches eight deterministic image-edit services: one service per
  scheduler-visible GPU, TP=1, capacity=1, on base port plus GPU ordinal.
- A distinct coordinator job requests one node and two GPUs. It owns the
  one-GPU VLM service, one-GPU LLM service, component execution, shared SDG
  runtime, validation, and final evidence.
- Requested image-edit capacity is `8 * generation_nodes`. The coordinator is
  independently schedulable and waits a bounded time for the first complete
  eight-service worker descriptor, then uses every complete descriptor that
  arrives during a short settle window. This lets useful work proceed when
  backfill starts only a subset of the approved maximum. Active capacity is
  always recorded explicitly; it is never inferred or silently reported as
  the requested maximum. Malformed, unreachable, duplicated, misordered, or
  wrongly owned services still block the action.

The image workers are separate scheduler jobs so each node has an explicit
native ID, exact job name, failure boundary, and cleanup target. Do not replace
this topology with `#SBATCH --nodes=N` or one aggregated capacity-eight
endpoint per node. Pending workers that were not selected are exact-owned and
canceled with the rest of the group when the coordinator completes or fails.

## Signed request

The immutable action request binds the workflow, action/run/iteration IDs,
attempt, shared paths, images, model IDs and revisions, limits, expected
outputs, and its canonical SHA-256. It additionally requires:

- `started_at`: timezone-aware ISO timestamp from action preparation.
- `started_ns`: positive preparation timestamp in nanoseconds.
- `generation_nodes`: integer in `[1, 64]`.
- `resources`: exactly one two-GPU coordinator and one-node/eight-GPU image
  workers with capacity eight per worker, separate signed CPU allocations for
  each role, and the approved wall-time limit. The default image-worker
  allocation is 64 CPUs because it launches eight exact, exclusive 8-CPU
  endpoint steps; the two-GPU coordinator default is 60 CPUs.
- All three model roles use TP=1. The image-edit base port reserves eight
  consecutive ports per worker node.
- `config_sha256` and `runtime_sha256`: the immutable SDG config and active IAA
  Python runtime lineage. Every worker verifies both before GPU or component
  work.
- `component_sources`: immutable original image references from
  `sdg_config.yaml`, separate from the prepared runtime SQSH paths.

All data, runtime, image, cache, request, result, and job-evidence paths are
absolute shared SLURM paths. The controller path used to stage a request is not
substituted for its compute-visible path.

### Deterministic preparation

`prepare-request` is a local preparation operation, not a fifth execution
verb. Run it after `history_select` is committed and before launch approval:

```bash
python3 "$SLURM_SKILL/scripts/slurm_sdg_action.py" prepare-request \
  --deft-state "$RESULTS_DIR/deft_state.json" \
  --sdg-config "$RESULTS_DIR/config/sdg_config.yaml" \
  --iteration "$ITERATION" \
  --runtime-root "$REMOTE_RUNTIME_ROOT" \
  --cache-dir "$REMOTE_CACHE_ROOT" \
  --augmentation-image "$AUGMENTATION_SQSH" \
  --auto-labeling-image "$AUTO_LABELING_SQSH" \
  --image-edit-image "$IMAGE_EDIT_SQSH" \
  --text-serving-image "$TEXT_SERVING_SQSH" \
  --image-worker-cpus-per-task 64 \
  --coordinator-cpus-per-task 60 \
  --output "$RESULTS_DIR/iter_$ITERATION/datagen/.tao-runtime/controller/sdg.action.json"
```

When Airflow orchestrates the application while SLURM remains the compute
platform, controller and Lustre run roots may differ. In that composed case
only, the packaged IAA bridge adds `--backend-results-dir` and
`--backend-dataset-root`: local state/config still authorize preparation,
while the signed request binds the exact remote paths consumed by SLURM. Its
status verb adds `--local-results-dir`; terminal `COMPLETE` then atomically
copies the declared SDG outputs plus endpoint and platform-status evidence to
the controller after remote size and SHA-256 verification. Omitting these
mapping flags preserves native SLURM behavior. They never provide a compute
fallback.

Add `--account`, `--partition`, `--image-worker-cpus-per-task`,
`--coordinator-cpus-per-task`, or `--time-minutes` only for approved
site/resource overrides. The packaged defaults are 64 CPUs for an eight-GPU
image worker and 60 CPUs for the two-GPU coordinator. Each distinct allocation
must pass `sbatch --test-only` before submission. The helper derives generation topology,
models, revisions, ports, limits, source images, paths, outputs, run/action
identity, credential variable names, and scheduler resources from immutable
state/config. It reads no credential value. Identical execution returns
`unchanged`; a different existing output, non-SLURM state, altered config or
runtime lineage, unsafe path, out-of-order iteration, committed SDG, failed
stage, or stopped run is a hard error.

`--cache-dir` is an already-existing shared Hugging Face cache directory, not a
path beneath `results_dir`. Resolve and verify it on the SLURM login frame
before approval; do not construct it by joining a prior run path to the current
workspace/results path. Immediately before staging or `sbatch`, `submit`
revalidates that cache and the dataset as readable non-symlink directories,
the immutable config file and digest, and all four non-symlink SQSH files with
SquashFS magic. Missing or changed shared input fails before GPU allocation.

For the single scheduler retry, use the same command with a distinct output
path and all three authoritative attempt-1 inputs:

```bash
  --retry-from-request "$ATTEMPT1_REQUEST" \
  --retry-from-job-record "$ATTEMPT1_JOB_RECORD" \
  --retry-login "$SLURM_LOGIN" \
  --output "$RESULTS_DIR/iter_$ITERATION/datagen/.tao-runtime/controller/sdg.attempt-2.action.json"
```

The producer accepts only the exact attempt-1 request and a redacted terminal
`ERROR` TAO job record classified `ERR_INFRA`. This also covers a batch-shell
failure before coordinator Python could write its terminal JSON. It loads the
immutable job group, proves every native ID/name through SLURM accounting, and
requires one terminal failed group whose coordinator matches the job-record
backend. It verifies current committed `history_select` state, preserves the
attempt-1 artifacts byte-for-byte, and derives a distinct, deterministic
attempt-2 action with matching resume semantics. Attempt 2 must use a distinct
new pending TAO job record opened with `--retry-of "$ATTEMPT1_JOB_ID"`; the
request itself binds hashes of the prior job record, job group, and captured
native states. There is no attempt 3.

## Public four verbs

Use `scripts/slurm_sdg_action.py` through the same job-record and launch-review
boundary as other platform consumers:

1. `submit` validates the signed request and job record, stages immutable
   runtime files by digest, creates the protected run-scoped endpoint-auth
   sidecar, submits `N` exact image-worker names, then submits the distinct
   independently schedulable coordinator. It returns the coordinator
   as the primary backend reference plus every owned native worker ID/name.
   Before each `sbatch`, it durably creates one immutable submit intent binding
   request, action, attempt, job ID/name, and rendered-script digest. If the
   submit response is lost, exactly one same-name job owned through that intent
   is reconciled and recorded through a bounded accounting-propagation window;
   zero before submit proceeds to submit, while multiple jobs, mismatched or
   missing pre-existing intent, or ambiguous ownership fail closed. Discovery
   of an existing job never mints retroactive intent. A partial submit
   preserves its jobs, sidecar, and intents for the same job-record recovery;
   it never deletes authentication needed by already-created workers or
   submits duplicate exact names. The complete operation is serialized by one
   request/job-scoped, owner-token-checked lock on shared storage. A concurrent
   invocation fails before staging or scheduler submission; the lock is always
   released by the owning invocation on ordinary success or failure.
2. `status` reloads the strict job group, proves the coordinator backend
   reference and every image-worker native ID still bind to their recorded
   exact names, maps scheduler state, and accepts completion only with matching
   terminal evidence.
3. `logs` verifies coordinator ownership before reading the exact shared log
   paths and redacts recognized credential forms.
4. `cancel` requires confirmation, verifies every member of the job group, and
   cancels only those exact native IDs. A partial or ambiguous ownership result
   blocks cancellation rather than widening its scope.

`image-worker`, `coordinator`, and `component` are template/runtime entrypoints,
not additional user-facing launch verbs.

### Exact recovery for a historical duplicate-submit race

`recover-duplicate-submit` is an exceptional evidence-repair operation, not a
fifth launch verb or another workload attempt. Use it only when a pre-lock
attempt-1 invocation demonstrably created exactly two same-name coordinators.
It requires explicit `--confirm`, the immutable attempt-1 request, its active
TAO job record, exact job ID, and login. It fails closed unless all of the
following hold:

```bash
python3 "$SLURM_SKILL/scripts/slurm_sdg_action.py" recover-duplicate-submit \
  --request "$ATTEMPT1_REQUEST" \
  --login "$SLURM_LOGIN" \
  --job-id "$ATTEMPT1_JOB_ID" \
  --job-record "$ATTEMPT1_JOB_RECORD" \
  --confirm
```

- the job-record backend and current job-group coordinator are the two
  distinct exact-name coordinator IDs;
- each expected image-worker name resolves to exactly the ID in the immutable
  group, and every worker/coordinator is ownership-verified;
- all request/job/name/script submit intents are intact;
- no canonical SDG output exists; and
- every affected native job reaches a terminal scheduler state within the
  bounded cancellation deadline.

The operation cancels only a still-active, exactly owned member. It quarantines
any untrusted coordinator terminal JSON, records its size and digest, records
all native states and the overwritten job-group digest in a digest-bound
recovery artifact, removes launch authentication, and accepts no workload
output. Repeating it against the same still-active record is idempotent. Then
terminalize attempt 1 as `ERR_INFRA` with the packaged job-record helper and
prepare the one permitted attempt 2 with `--retry-from-request`,
`--retry-from-job-record`, and `--retry-login`.

Attempt-2 preparation validates the recovery artifact and quarantined file and
binds both coordinator IDs plus every worker state into retry lineage. Any
other job count, changed intent/group/artifact, missing ownership, output,
nonterminal job, or attempt 2 is rejected. Never edit the job group or bind the
overwritten coordinator retroactively. A controller-authored terminal record
is accepted on this fully proven duplicate-recovery path. An agent-authored
record is accepted only for the narrower cleanup-only failure: the producer
must independently validate the exact coordinator terminal JSON, the cleanup
record and every worker ID/name, terminal scheduler accounting, and all four
canonical output hashes. Those evidence hashes are embedded in attempt-2 retry
lineage. Every other ordinary retry requires terminalization by the backend
hook.

One attempt-2 adapter repair is separately allowlisted when a newly created
endpoint pool is rejected before preprocessing or generation. Prepare it with
`--repair-from-request`, `--repair-from-job-record`, and `--repair-login`.
The producer requires exact terminal native ownership, cleanup, execute-log,
zero-generation progress, and absent-output evidence. It keeps `attempt=2`,
retains the original retry lineage, records the immediate parent job through
`retry_of`, and passes `--explicit-unstarted-pool-rebind` to the shared
runtime. No other program error and no already repaired request can enter this
path. If the repair follows an approved runtime refresh, the request binds the
exact final old-to-new runtime-lineage record and validation evidence; all
other data, model, scheduler, and resource semantics must remain unchanged.

One scheduler reschedule is separately allowlisted when an attempt-2
pool-rebind repair is canceled before execution for capacity/backfill reasons.
Prepare it with `--reschedule-from-request`,
`--reschedule-from-job-record`, `--reschedule-login`, and a strictly smaller
`--time-minutes`. The producer verifies the immediate parent record, exact job
group and ownership, `CANCELLED` plus zero elapsed seconds for every native
allocation, unchanged repair-bound progress, and absent outputs/descriptors.
It keeps attempt 2, retains retry and repair lineage, binds the reschedule
evidence, and creates a distinct action ID. No executed allocation, semantic
change besides the shorter walltime, or second reschedule is accepted.

One image-service launch repair is allowlisted after that reschedule when a
terminal worker log and its exact endpoint log prove vLLM-Omni failed before
generation with `EADDRINUSE` on the inherited diffusion `MASTER_PORT`. Prepare
it with `--launch-repair-from-request`,
`--launch-repair-from-job-record`, and `--launch-repair-login`. The producer
binds terminal ownership/accounting, terminal and cleanup evidence, failure-log
digests, any partial readiness descriptors, unchanged zero-generation progress,
and absent canonical outputs. The repaired worker preflights and assigns one
deterministic internal port per GPU through the process environment while
keeping API ports, GPU count, models, walltime, and all prior lineage unchanged.
No other endpoint error or second launch repair is accepted.

### Exact recovery for a cleanup-only terminal failure

`recover-cleanup-failure` is an exceptional evidence repair, not a workload
retry. Use it only while the exact job record is still nonterminal and after
the workload produced all canonical outputs but the coordinator failed solely
because worker cancellation could not be confirmed. It requires explicit
confirmation:

```bash
python3 "$SLURM_SKILL/scripts/slurm_sdg_action.py" recover-cleanup-failure \
  --request "$REQUEST" \
  --login "$SLURM_LOGIN" \
  --job-id "$JOB_ID" \
  --job-record "$JOB_RECORD" \
  --confirm
```

The operation independently validates the request, active record, exact job
group and ownership, cleanup-only coordinator terminal, failed cleanup entry,
all terminal native states, and current hashes of all four outputs. It cancels
only an active, reverified worker under the shared cleanup deadline. It copies
the error terminal to immutable recovery evidence, writes a digest-bound
recovery record, and publishes a canonical success terminal bound to that
record. Public `status` maps a natively failed coordinator to `COMPLETE` only
while this recovery evidence revalidates. Repetition is idempotent. Missing,
changed, ambiguous, nonterminal, or unrelated evidence fails closed; never use
this operation to repair a workload or output failure.

## Readiness and endpoint pool

Every image-worker binds services to its coordinator-reachable node hostname,
performs bounded model readiness checks, and atomically publishes one signed
descriptor only after all eight services pass. The coordinator requires at
least one complete descriptor within the startup deadline, gathers other
complete descriptors during the bounded settle window, verifies native
ID/name ownership through SLURM, and probes every selected remote service
before committing `<stage_dir>/endpoint_pool.json`.
Malformed readiness payloads remain inside the same bounded deadline and are
reported as classified readiness failures rather than escaping as parser
exceptions.

Before any endpoint work, workers verify the signed config digest, staged
active-runtime Python-tree digest, and all four prepared SQSH files including
SquashFS magic. `endpoint_manifest.json` records auxiliary model readiness and
augmentation/auto-labeling provenance: immutable original image reference,
runtime SQSH path, and conversion verification. This is the evidence consumed
by the application SDG commit gate.

The pool is strict and contains exactly `8 * M` ordered, distinct capacity-one
entries for the selected active workers, where `1 <= M <= N`.
Each entry binds its endpoint ID and URL, GPU identity, and owner
native ID/name. The manifest also binds the image-edit model/revision, request
digest, active and requested capacity, platform, UTC creation time, and the authentication
environment-variable name. It never contains an authentication value. A
failed reachability or ownership check leaves no committed pool.

The shared SDG dispatcher acquires one endpoint slot per augmentation request.
For each component call, the SLURM adapter reloads the pool and proves the
selected endpoint ID/URL pair belongs to it. Endpoint failure is handled by the
shared bounded retry/quarantine contract; the adapter does not add another
retry loop or independent progress journal.

The pinned auto-labeling component stores its managed interpreter beneath the
container root home. Its component step therefore uses Pyxis
`--container-remap-root` together with `--no-container-mount-home`; the latter
prevents the default host-home mount from hiding the image-bundled runtime.
Pyxis maps only the submitting job user to container root and explicitly does
not grant elevated host permissions. Apply these flags only to auto-labeling,
keep the input mount read-only, and keep output limited to the run-owned stage
mount. Augmentation and model-serving steps remain unmapped, and no host home
is exposed to the auto-labeling component.

## Authentication, resume, and cleanup

- Endpoint authentication uses one random launch-scoped value delivered to
  every managed image-edit, VLM, and LLM service and client through process
  environment only. Values never enter
  requests, argv, job-group records, endpoint manifests, logs, or reports.
- The shared-storage auth sidecar and parent directory must be owned by the
  submitting user, non-symlinks, and mode 0600/0700. Normal completion,
  cancellation, and partial-submit recovery securely remove the sidecar.
- The shared SDG runtime exclusively owns stage progress, bounded component
  retry, endpoint quarantine, normalization, and committed-stage resume. The
  SLURM adapter owns scheduler/service evidence only.
- Request, auth, ownership, descriptor, log, cleanup, and terminal evidence is
  scoped to the exact minted job record. A successful terminal is idempotent
  only when its request/action/job binding and all expected outputs still
  match; an infrastructure retry cannot consume or replace another launch's
  evidence.
- Never resubmit an unchanged failed attempt-1 request under another job ID.
  Use only the authoritative attempt-2 preparation above; preserve all prior
  request, job-record, terminal, log, sidecar, ownership, and progress evidence.
- Coordinator terminal evidence preserves signed `started_at` and `started_ns`,
  the coordinator native ID, request and resume digests, exact attempt, output
  list, and worker start/finish timestamps.
- On coordinator exit, stop auxiliary endpoint steps, cancel only verified
  image-worker jobs, record the result for every worker, sanitize endpoint
  logs, and remove endpoint authentication. Issue cancellation to every
  verified worker before polling any worker. Retry transient exact-name
  ownership queries for at most 60 seconds, then use one shared bounded
  `squeue`/`sacct` deadline for the whole group. Transient scheduler-query
  failures remain pending inside that deadline. A timed-out or nonzero
  `scancel` is ambiguous: query state, reverify the exact name, and retry only
  while that worker remains active. `COMPLETING` remains nonterminal. Each
  exact allocation must reach a terminal state; `scancel` acceptance alone is
  not completion. A failed or unverified owned-worker cancellation prevents
  successful terminal evidence. Never stop a job or service whose exact
  ownership cannot be proven.
