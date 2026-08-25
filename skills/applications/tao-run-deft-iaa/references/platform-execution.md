# Platform Execution Contract

Use this reference for every TAO-backed IAA stage and deterministic IAA adapter.
For a remote workflow, dataset rebuild/materialization, gap/mining/history,
visualization preparation/finalization, train/eval configuration, checkpoint
publication, metric parsing, iteration summary, and report rendering are
zero-GPU platform actions. The controller only prepares, stages, submits,
monitors, synchronizes, finalizes, commits, and audits metadata.

## One producer, five compute consumers

Remote adapter names are exact: `dataset_rebuild`, `dataset_materialize`,
`gap_analysis`, `mining_postprocess`, `history_select`, `visualize_prepare`,
`visualize_finish`, `eval_config`, `train_config`, `publish_checkpoint`,
`iteration_summary`, `metric_parse`, and `report`. Prepare them with the same
producer and the exact command returned by `expected_container_command`; use
the DS image except `publish_checkpoint`, which uses PyTorch. Their bundle has
`gpus=0` and `gpu_ids=[]`. The selected platform must use a CPU allocation and
must not widen that shape to a GPU request.

`run_deft_action.py prepare` is the only producer. It validates the requested
image, argv, stage scope, token forwarding, output set, and immutable run
hashes; deletes the exact stale outputs; then writes:

```text
<stage>/<name>.action.json
```

`sdg_normalize_repair` is a narrow zero-GPU adapter available only when the
typed coarse-filesystem recovery journal is in `prepared` state. It declares
exactly the three normalized manifest/list/pairs files and binds every
accepted image, verification record, label, vocabulary, config, and evaluation
exclusion input by digest. It runs no endpoint or model code. Any changed
input, extra output, absent backup, or nonzero-GPU rendering is invalid. After
terminal success, run the typed `verify`; on action failure, run `restore`.

The request contains a `tao-artifacts` schema-valid `spec_bundle`, the selected
platform, logical mounts, non-secret environment settings, approved credential
variable names, expected fresh outputs, the exact job-record state directory,
and an attempt number. Preparation atomically snapshots the exact controller
Python files and compatibility patches beneath the action's `.tao-runtime`
directory; `controller_snapshot` and `patches_snapshot` bind every file by
size and SHA-256. The controller manifest covers a minimal skill-bank layout;
stage its root and mount its
`skills/applications/tao-run-deft-iaa/scripts` subdirectory at `/iaa-runtime`.
Mount and stage those request-owned paths, never a plugin
cache path. This keeps prepared and in-flight actions valid across plugin
refreshes. Never edit the request or either snapshot.
The selected platform skill is the consumer:

| state platform | required platform skill | native lifecycle |
|---|---|---|
| `docker` | `tao-run-on-docker` | `docker` |
| `slurm` | `tao-run-on-slurm` | `ssh` + `sbatch/squeue/sacct/scancel` |
| `kubernetes` | `tao-run-on-kubernetes` | `kubectl` Job |
| `brev` | `tao-run-on-brev` | `brev exec` + remote Docker |
| `virtualenv` | `tao-run-on-virtualenv` | vendored `virtualenv_runner.py` |

Read that platform's complete `SKILL.md`, run its preflight, and use its exact
`submit`/`status`/`logs`/`cancel` contract. When the run records
`orchestrator=airflow`, read `airflow-execution.md` and wrap those exact verbs
in the application-local signed envelope. The request, job record, GPU shape,
backend handle, and final evidence remain native to the selected compute
platform. Do not translate one platform into another or fall back to Docker.
Legacy initialized `platform=airflow` runs use their original compatibility
consumer and are not migrated in place.

## Prepare

Each stage reference supplies the exact arguments:

```bash
ACTION_JSON=$(
  "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
    "$SKILL_ROOT/scripts/run_deft_action.py" prepare \
      --results-dir "$RESULTS_DIR" --image <pyt|ds> \
      --stage-dir "$STAGE_DIR" --name <stable-name> \
      "${HF_ARGS[@]}" \
      --fresh-output "$EXPECTED_OUTPUT" \
      -- <exact TAO argv>
)
ACTION_REQUEST=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["request"])' \
  <<<"$ACTION_JSON")
```

Repeat `--fresh-output` in the documented order when an action has several.
Preparation is part of the approved launch and is a mutation: do not call it
before the launch review.

For example, `gap_analysis` that feeds iteration 1 is prepared exactly as
`label=iter1`, including when it analyzes the baseline evaluation. Gap actions
bind the iteration they produce, not the evaluation label committed afterward:

```bash
STAGE_DIR="$RESULTS_DIR/iter_1/gaps"
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" prepare \
    --results-dir "$RESULTS_DIR" --image ds \
    --stage-dir "$STAGE_DIR" --name gap_analysis \
    --fresh-output "$STAGE_DIR/kpi_gaps.parquet" -- \
    python3 /iaa-runtime/run_iaa_compute.py gap_analysis \
      --results-dir /results --label iter1
```

Use each reference's exact stage, label, and fresh-output set. On every
platform, including local Docker and virtualenv, invoke the bundled mutator
only through its signed platform action; never invoke `run_iaa_stage.py`
directly from the controller.

`prepare` is crash-safe. Repeating it before finalization returns the existing
immutable request and does not delete outputs or mint a new attempt. Inspect
the embedded `reconciliation` result, or run the equivalent explicit check:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" reconcile \
    --request "$ACTION_REQUEST"
```

`NO_JOB_RECORD` means it is safe to open the record. `JOB_OPENED_UNBOUND`
means bind that exact record before submitting. `BOUND` means a native handle
is already recorded: never submit a second native job, and poll/finalize the
bound job instead. `BOUND_BACKEND_RECONCILIATION_REQUIRED` is the crash window
after the immutable pre-submit binding but before a native handle became
durable. Query the selected backend for the deterministic `JOB_ID`/object name
and repair the existing record only when that native object is proven to
exist. A narrower recovery is available for any original deterministic SLURM
action whose controller canceled the record before submit:
exact `PENDING -> CANCELED` transitions, agent terminal ownership, no backend
handle, no status/log/output evidence, and no native job returned by bounded
exact-name `squeue` and `sacct` queries. After reviewing those facts, archive
the stale binding and restore the safe open boundary with:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" recover-bound-presubmit \
    --request "$ACTION_REQUEST" --job-record "$JOB_RECORD" \
    --login "$SLURM_LOGIN" --confirm
```

The helper writes job-scoped digest-bound recovery evidence and is idempotent
before a replacement binding is created. At most three proven pre-submit
records are recoverable for one action; exhaustion is a hard stop rather than
an unbounded correction loop. Reconcile again and require `NO_JOB_RECORD`,
then perform the normal attest, open, bind, and submit sequence exactly once.
It refuses launched records, changed evidence, or any native-name match. If
absence cannot be proven, stop for operator
reconciliation; never blindly submit a replacement. Stop on any malformed,
duplicate, or mismatched record rather than guessing. One exact `PENDING ->
CANCELED` record with no backend handle and no binding is a typed pre-submit
abandonment receipt and does not poison a corrected open. The producer still
binds its action, path, timestamp, and canceled shape; launched, bound, stale,
nonterminal, or malformed records fail closed.

## Stage inputs and preserve logical paths

Read `mounts` from the request. The same source may have two targets by design:

- workflow results at `/results` and at their approved absolute path;
- the dataset parent at `/data` and at its approved absolute path;
- immutable config at `/specs`;
- the request-owned compatibility snapshot at `/patches`;
- persistent model cache at `/cache`.
- for adapter actions only, the request-owned controller snapshot at
  `/iaa-runtime`.

The producer emits only inputs used by that action. Text-only `pool_embed`,
`target_embed`, and `knn` requests intentionally omit the dataset parent;
evaluation, training, and image-embedding requests include both dataset
aliases because they dereference image paths. Consumers must stage every
declared input and must not add undeclared workflow-wide inputs.

When a run deliberately reuses an existing dataset, the producer also emits
exact dataset-root aliases beneath `/data/<dataset-name>` and at the canonical
host path. A remote consumer may bind those two request-owned targets to one
verified backend dataset root through the staging receipt. This is distinct
from remapping the writable dataset parent and cannot be inferred from a
similarly named directory.

TAO model actions carry a digest-bound `cache_subset` manifest for their
SigLIP2 cache. Adapter actions intentionally omit it and consumers must skip
cache-subset staging for those requests. Remote TAO consumers materialize it with
`scripts/stage_action_cache.py` and mirror only that verified directory to the
compute-side `/cache` source. Never mirror the shared control cache, which may
contain unrelated image-edit, VLM, LLM, or Xet data. The helper is idempotent
and rejects changed/missing files, digest mismatch, traversal, non-regular
targets, or unrelated destination files. The request digest in the staging
receipt transitively binds this manifest.

Embedded parquet rows and generated TAO specs can contain the approved absolute
dataset path, while TAO commands use `/results`, `/data`, and `/specs`. Every
container platform must therefore render **all** request mounts, including the
absolute-path aliases. Do not remove an alias because it looks host-specific.

Local Docker and virtualenv are tier A and use local paths. Docker through an
approved remote `DOCKER_HOST`, SLURM, Kubernetes, and Brev use the request's
remote staging contract: use `tao-data-io` to stage every declared input, then
render the remote source at each request target. The workflow results tree is
mutable shared application state, so synchronize it to the compute side before
each action and synchronize it back after terminal success.

Freshness on a remote platform is an absence contract, not a clock comparison:
mirror the prepared local results tree with delete semantics so every path in
`staging_absent_paths` (all `fresh_outputs` plus the native action log) is
absent on the compute side before submit. Never use an incremental copy that
can retain stale evidence. The finalizer then accepts the fetched artifact
without comparing clocks, because remote and launcher clocks need not agree.
Local Docker and virtualenv instead require local mtime after preparation.
Remote Docker never uses launcher-local mtimes. The producer also gives every
bundle action a digest suffix bound to the run, stage, attempt, and preparation
time; do not shorten or rederive it when opening the job-record.

After the platform-native absence check succeeds and **before** opening the
job-record, persist that result. Repeat every request output in order and use a
non-secret, canonical durable results scope. It must be the exact absolute
compute-side action-stage path (for example `/lustre/.../embeddings/source`) or
a credential-free persistence URI (for example
`s3://bucket/runs/<run>/embeddings/source`); relative labels, credentials,
queries, fragments, and `..` traversal are rejected:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" attest-staged \
    --request "$ACTION_REQUEST" \
    --backend-scope "$BACKEND_SCOPE" \
    --absent-path "$EXPECTED_OUTPUT" \
    --absent-path "$ACTION_LOG" \
    --mount-map "$LOCAL_RESULTS" "$REMOTE_RESULTS"
```

Repeat `--mount-map LOCAL_SOURCE BACKEND_SOURCE` once for every distinct
request mount source, in request order, on a remote backend whose filesystem
paths differ from the controller. The signed receipt is the only authority
for rendered remote sources; the target and read/write mode remain the
immutable request values. Omit `--mount-map` only when controller and backend
paths are literally identical.

The finalizer rejects a remote action without this digest-bound staging receipt
or when the job-record was opened before the receipt. This prevents a retained
remote artifact from being accepted merely because clocks differ.

## Submit and job record

Validate the bundle before submit. Open the job-record before native launch:

```bash
JOB_ID=$("$BANK/scripts/tao_job_record.py" open \
  --platform "$PLATFORM" \
  --image "$RECORD_IMAGE" \
  --network-arch "$NETWORK_ARCH" \
  --action "$ACTION" \
  --storage-tier "$STORAGE_TIER" \
  --results-dir "$JOB_RESULTS_SCOPE" \
  --upload-exclude .tao-runtime/ \
  --upload-exclude '*.launch.lock')
```

Pass every `spec_bundle.upload_excludes` item, in order, as one
`--upload-exclude`; do not copy a static list from this example. For local
Docker and virtualenv, `JOB_RESULTS_SCOPE` is the exact local
`request.stage_dir`. For remote Docker, SLURM, Kubernetes, and Brev it is the
exact `backend_scope` already recorded by `attest-staged`. The producer rejects
a different path or exclusion list even if the rest of the record matches.

Take `RECORD_IMAGE`, `NETWORK_ARCH`, and `ACTION` from `record_image` and
`spec_bundle`; never rederive them. Immediately bind the still-PENDING record,
before any native submit:

```bash
RECONCILIATION=$("$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" reconcile \
    --request "$ACTION_REQUEST")
JOB_RECORD=$(python3 -c \
  'import json,sys; print(json.load(sys.stdin)["job_record"])' \
  <<<"$RECONCILIATION")
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" bind-job \
    --request "$ACTION_REQUEST" --job-record "$JOB_RECORD"
```

Binding is serialized by a request-scoped lock. It validates the request-owned
record path, platform, image, action, upload exclusions, submission time, exact
results scope, and (for remote platforms) the staging receipt, then persists
an immutable identity digest. It fails once a backend handle or any post-open
transition exists; a distinct concurrent record cannot replace the first
binding. Only after it succeeds may the platform skill submit with `JOB_ID`,
every rendered mount, the bundle command (`command` followed by `args`),
`compute_shape.gpus`, the non-secret `environment` values, and only the
variable names in `forward_env`. Mark `RUNNING` with the native handle.
For SLURM, render the request's `mounts` in their recorded order as explicit
`source:target:ro|rw` Pyxis mappings. The packaged submit gate compares every
receipt-bound backend source, target, and mode with the immutable request
before scheduler submit;
never reuse a prior stage's rendered mount list. It also derives adapter versus
model execution from `spec_bundle.network_arch`, requires the exact request
environment, and binds `#SBATCH --gres=gpu:N` to both `compute_shape.gpus` and
`gpu_ids`; an environment marker alone never classifies an action.

For Docker, SLURM, Kubernetes, and Brev the executable argv is the bundle argv
inside the pinned image. Virtualenv mirrors the two image families with two
immutable profiles. `image_kind=pyt` selects the approved `pyt` environment
for `clip`; `image_kind=ds` selects the approved `ds` environment for
`embedding` or `tmm`. A single environment is compatible only when it passes
both contracts. Config preparation checks Python 3.12/platform ABI, pinned
distribution versions, exact console-script metadata ownership/target,
profile imports, PyTorch CUDA build, and `pip check`. Action preparation
rechecks the selected action entrypoint and runs the real per-GPU CUDA tensor
probe so a stale or replaced environment cannot reach submit.

Use `request.virtualenv` (not a shell default or the control `.venv`) as the
runner `--venv`, use the attempt-specific `request.platform_runtime_dir` as
the runner `--job-dir`, and submit the packaged shim:

```text
script = request.virtualenv_shim
args   = --request <action.json> -- <bundle command> <bundle args...>
```

Pass each token as a separate runner `--arg` (use `--arg=TOKEN` for tokens
beginning with `-`). Pass `--gpus request.spec_bundle.compute_shape.gpus` and
the comma-joined immutable `request.gpu_ids` as `--gpu-ids`; never infer IDs,
substitute local defaults, or omit the selection. The runner binds those host
IDs through `CUDA_VISIBLE_DEVICES`, while generated TAO configs use the
corresponding CUDA-local ordinals `0..N-1`. The shim refuses a runner GPU frame
that differs from the prepared request. The shim verifies the request digest, full immutable
workflow state and paths, exact argv, exact equality between `VIRTUAL_ENV` and
the action's selected profile, and the prepared console-script content digest.
It opens the entrypoint without symlink traversal, verifies its executable file
type and profile-bound shebang, and executes the pinned file descriptor so
replacement races cannot redirect execution. It translates compute-frame paths
and YAML values to local paths, sets local cache locations, constructs a
minimal child environment (approved runtime tuning, validated certificate
paths, and explicitly forwarded `HF_TOKEN` only), and executes only the
allowlisted TAO CLI at the exact `<selected-profile>/bin/<command>` path.

## Status, logs, synchronization, and finalize

Poll the backend, not the record, and map native state to the fixed vocabulary.
On terminal state, mark the job-record. Preserve the ordered
`PENDING -> RUNNING -> terminal` transition lineage; finalization rejects an
earlier terminal state, a missing transition, or timestamps that move
backwards. Capture the selected platform's complete action log at the exact
request `log_path`; a missing, empty, symlinked, or credential-bearing log is
failure. Redact native logs before finalization when the repository credential
linter reports literal secret material.

For a remote `COMPLETE`, synchronize the workflow results tree back before
finalization and verify all declared outputs arrived. Do not tear down tier-C
storage or an ephemeral backend object before output synchronization and log
capture.

For an Airflow-orchestrated SLURM action, `slurm_action.py status` performs
this synchronization from the receipt-bound remote results scope before it
reports `COMPLETE`. The Airflow DAG therefore cannot validate outputs ahead of
the transfer. A missing, unsafe, stale, or empty declared output keeps the
backend action non-successful.
For `ERROR` or `CANCELED`, it synchronizes the complete sanitized native log
and any safe nonempty declared output as diagnostic-only evidence before
reporting the terminal state; the bridge then records the matching job-record
transition and finalizes the failed platform status. Diagnostic outputs never
satisfy success. A retry is not eligible until that evidence is complete.

Then bind native and artifact evidence:

```bash
JOB_RECORD="${TAO_STATE_DIR:-$HOME/.tao}/jobs/$JOB_ID.json"
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" finalize \
    --request "$ACTION_REQUEST" \
    --job-record "$JOB_RECORD" \
    --native-exit-code "$NATIVE_EXIT_CODE"
```

Finalization verifies the pre-submit job binding, exact job-record path and
results scope, platform/image/action identity, backend handle, terminal state,
native exit code, immutable request digest, log, and all fresh outputs. It
writes `<stage>/<name>.status.json`; only that status may be passed to
`commit_stage.py`. Commit and audit additionally require the evidence platform
to equal the platform initialized in workflow state; legacy schema-v1
container evidence is accepted only for Docker runs.

On `ERROR` or `CANCELED`, still capture logs and call `finalize` (omit a native
exit code only when the backend cannot report one). It records terminal error
evidence and returns nonzero. A retry is permitted only from that complete,
schema-valid status plus its immutable request, binding, terminal job-record,
backend handle, and log. The producer archives attempt-1 status as
`<name>.attempt-1.status.json`, retains its request/binding/log, and creates
attempt-specific request, receipt, binding, log, and platform-runtime paths for
attempt 2. Repeating preparation returns the same attempt-2 request. The second
attempt receives a new job-record and can never re-submit under the old job id;
a third attempt is rejected.

SLURM creates no stdout/stderr when a pending job is canceled before allocation.
For only that case, run packaged `run_deft_action.py
capture-preallocation-cancel --request ... --job-record ...` on the SLURM login
frame before `finalize`. It queries `sacct` itself and writes a digest-bound,
credential-free receipt at the immutable action log path only when the exact
bound job is `CANCELED` with zero elapsed time, no start/node allocation, exit
`0:0`, and no declared output. It is idempotent for the same receipt and rejects
ERROR/COMPLETE jobs, runtime, output, a native log, or changed accounting.

`dispatch-repair` is one deliberately narrower recovery path, not a third
workload attempt. It is available only for a virtualenv action whose two normal
attempts both have complete terminal ERROR lineage, runner exit-code-2 evidence,
captured/native log equality, no declared output, and the two exact allowlisted
shim failures that predate TAO CLI execution. Invoke it with the same arguments
as `prepare`, changing only the verb. It atomically archives attempt-2 status
and emits distinct `.dispatch-repair` request, log, binding, staging, runtime,
action-id, and job-record evidence while retaining logical `attempt: 2` and
marking `dispatch_repair: 1`. Repeating it before launch returns that same
request; after it is finalized, another repair is forbidden. Do not use this
path for an unknown error, a started TAO command, a model/data/CUDA failure, an
active native job, or any action that produced a declared workload output.

`launcher-repair` is the corresponding one-shot recovery only for SLURM IAA
`clip train`. Both normal attempts must have immutable terminal lineage:
attempt 1 must be the exact Lightning `devices=2` versus SLURM
`--ntasks-per-node=1` rejection before distributed initialization; attempt 2
must be explicitly canceled after reaching only rank 0 `MEMBER: 1/2`, with no
traceback, Hydra execution error, rank 1, first batch, output, or checkpoint.
Invoke the same producer arguments with `launcher-repair` in place of
`prepare`. It archives the active attempt-2 status and emits distinct
`.launcher-repair` request, log, binding, staging, runtime, action-id, and
job-record evidence while retaining logical `attempt: 2` and marking
`launcher_repair: 1`. Repeating it before submit returns the same request; a
terminal repair forbids another. Submit the emitted request through the normal
SLURM launch gate and wrapper contract. Never use this path for an arbitrary
exhausted retry or runtime, data, model, or unknown failure.

`unbound-replay` is a separate one-shot recovery for a controller interruption
or ordering defect where one allowlisted deterministic SLURM adapter reached a
terminal COMPLETE native job but its request-owned job binding was never
created. The producer requires attempt 1, exactly one owned terminal job, no
binding or platform status, a captured log, and non-empty regular declared
outputs. It moves those outputs to a run-owned digest-bound quarantine, records
the immutable request/job/log lineage, and mints a distinct attempt-2 action.
Only `gap_analysis`, `mining_postprocess`, `history_select`, `eval_config`,
`train_config`, `iteration_summary`, and `metric_parse` are eligible. The
normal SLURM submit gate must validate the new request and binding before
submit. Never bind attempt 1 retroactively, reuse its outputs, replay a GPU or
mutating adapter, or mint attempt 3.
