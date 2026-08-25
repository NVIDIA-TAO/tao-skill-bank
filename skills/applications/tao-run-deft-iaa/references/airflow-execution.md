# IAA Airflow Orchestration

Airflow is an optional, application-scoped orchestrator for
`tao-run-deft-iaa`. It is not a sixth TAO compute platform. New composed runs
record both dimensions explicitly:

```json
{
  "platform": "slurm",
  "orchestrator": "airflow"
}
```

`platform` remains the authoritative compute backend. It owns staging, GPU
allocation, workload identity, native status and logs, cancellation, SDG
topology, and the TAO job record. Airflow schedules and observes that backend
consumer through one signed envelope; it never translates the request into
Airflow GPU tasks and never creates a second DEFT state store.

Legacy initialized runs with `platform=airflow` remain resumable through the
v1 direct-Airflow compatibility path. Do not initialize a new run in that
form.

For Airflow over SLURM, model jobs use the cluster-validated single-node NCCL
transport profile `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`, and
`NCCL_NET=Socket`. These fixed non-secret values pass through the platform
allowlist into Pyxis. This avoids treating two GPUs in one node as an
InfiniBand path when the cluster's external NCCL plugin cannot establish the
local queue pair; no credential or user-supplied environment name is added.

## Support matrix

All workloads, including SDG, TAO actions, CPU adapters, training, evaluation,
and result production, remain on the selected backend.

| compute `platform` | ordinary action consumer | composite SDG consumer | material constraint |
|---|---|---|---|
| `docker` | `docker_action.py` | `local_sdg_action.py` | Airflow worker has Docker, NVIDIA runtime, approved GPUs, and the shared path |
| `slurm` | `tao-run-on-slurm/scripts/slurm_action.py` | `tao-run-on-slurm/scripts/slurm_sdg_action.py` | Airflow worker has noninteractive SSH and the same approved SLURM/storage view |
| `kubernetes` | `tao-run-on-kubernetes/scripts/kubernetes_action.py` | `tao-run-on-kubernetes/scripts/kubernetes_sdg_action.py` | Airflow worker has the approved context, namespace, PVC, ServiceAccount, and Secrets |
| `brev` | `tao-run-on-brev/scripts/brev_action.py` | `tao-run-on-brev/scripts/brev_sdg_action.py` | Airflow worker has Brev authentication and the immutable resolved instance inventory |
| `virtualenv` | `tao-run-on-virtualenv/references/virtualenv_runner.py` | `local_sdg_action.py` | TAO/CPU actions use the approved venv profiles; SDG remains container-backed on that host |

Virtualenv support is intentionally hybrid, not container-free. Airflow uses
the selected `pyt` or `ds` virtualenv for TAO and typed CPU actions, while the
locally built, pinned SDG component and serving images continue to run through
Docker on the same compute host.

## Entry and preflight

Run the selected compute platform's complete preflight first. Then validate
the Airflow layer:

- `AIRFLOW_BASE_URL` is a credential-free HTTPS origin; HTTP is allowed only
  through loopback port forwarding.
- Authentication is present as `AIRFLOW_API_TOKEN`, or both
  `AIRFLOW_USERNAME` and `AIRFLOW_PASSWORD`. Values remain in the environment.
- `TAO_IAA_AIRFLOW_DAG_ID`, when set, names the installed IAA DAG; otherwise
  the ID is `tao_deft_iaa_action_v1`.
- The unpaused DAG advertises the exact `tao-deft-iaa-action-v1` tag.
- `TAO_IAA_AIRFLOW_SHARED_ROOT` is a normalized non-root path visible with the
  same contents to the controller and Airflow worker.
- The worker can execute the selected backend's packaged consumer and native
  CLI. Validate credentials and access from the worker frame, not merely the
  controller.
- The shared frame has enough disk and preserves regular-file, symlink, and
  atomic-rename semantics required by the request, receipt, log, and output
  contracts.

The Airflow coordinator needs only one CPU pool slot. Backend resources remain
in the backend request and must not be copied into Airflow pool sizes.

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/airflow_action.py" preflight \
    --pool "$AIRFLOW_IAA_COORDINATOR_POOL:1"
```

Missing backend access, an incompatible DAG, inaccessible shared evidence, or
missing environment variables blocks launch. Do not fall back to another
backend or to the Airflow worker's local Docker daemon.

### Packaged local Airflow service

For approved local-machine validation, use the bundled service helper. Its
`plan` action is read-only; install, deploy, start, and stop are launch-review
operations. Deploy stages both the DAG runtime and the orchestration runtime by
digest.

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/manage_local_airflow.py" plan \
    --root "$AIRFLOW_LOCAL_ROOT" --shared-root "$AIRFLOW_SHARED_ROOT" --port "$AIRFLOW_PORT"

"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/manage_local_airflow.py" install --root "$AIRFLOW_LOCAL_ROOT"
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/manage_local_airflow.py" deploy --root "$AIRFLOW_LOCAL_ROOT"
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/manage_local_airflow.py" start \
    --root "$AIRFLOW_LOCAL_ROOT" --shared-root "$AIRFLOW_SHARED_ROOT" --port "$AIRFLOW_PORT"
```

The local service is loopback-only and for validation, not production. Stop
only its receipt-bound process group with `stop --confirm`.

## Immutable orchestration envelope

Prepare and bind the normal backend-native request first. The selected
platform's request still says `platform=<compute backend>` and its TAO job
record uses that same platform. For an ordinary action, finish remote staging,
the absence attestation, job-record open, and `bind-job` before composing
Airflow. For SDG, use the selected platform's canonical composite request;
Docker and virtualenv use `airflow_sdg_action.py prepare-request` only to
produce the local composite request, followed by `local_sdg_action.py`.

For an ordinary SLURM action selected by the audit, use the packaged bridge
instead of manually assembling those boundaries:

```bash
export TAO_STATE_DIR="$TAO_IAA_AIRFLOW_SHARED_ROOT/.tao"
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/airflow_slurm_action.py" \
    --results-dir "$RESULTS_DIR" --name "$ACTION" --label "$LABEL" \
    --login "$SLURM_LOGIN" --remote-workspace "$REMOTE_WORKSPACE" \
    --shared-root "$TAO_IAA_AIRFLOW_SHARED_ROOT" \
    --pyt-sqsh "$PYT_SQSH" --ds-sqsh "$DS_SQSH" \
    --account "$SLURM_ACCOUNT" \
    --cpu-partition "$SLURM_CPU_PARTITION" \
    --gpu-partition "$SLURM_GPU_PARTITION" \
    ${CPU_TIME_MINUTES:+--cpu-time-minutes "$CPU_TIME_MINUTES"} \
    ${GPU_TIME_MINUTES:+--gpu-time-minutes "$GPU_TIME_MINUTES"} \
    ${BACKEND_DATASET_ROOT:+--backend-dataset-root "$BACKEND_DATASET_ROOT"}
```

Export the shared `TAO_STATE_DIR` before either standalone action preparation
or bridge invocation. The bridge checks this binding before transferring the
results tree. If an unsubmitted request was prepared against another state
directory, first use `recover-bound-presubmit` when it has a stale binding,
then run `rebind-airflow-state --request <action.json> --confirm` with the
approved shared `TAO_STATE_DIR`; this archives the prior request and staging
evidence and never retries workload compute.

The helper advances only that named audit-selected action. It snapshots the
controller, stages exact request inputs, records the controller-to-Lustre
mount mapping in the absence receipt, opens and binds the SLURM job, renders
the immutable GPU shape, stages the digest-bound consumer, submits the Airflow
envelope, and polls both layers. SLURM terminal success synchronizes every
declared output and the sanitized complete native log into shared Airflow
storage before `COMPLETE` can reach the DAG. For `dataset_materialize`, it also
validates the adapter's exact six-file output inventory, copies those nested
split/source artifacts by remote size and SHA-256, and writes
`dataset_materialize.output-sync.json` before finalization. The normal action
finalizer still owns the resulting platform status. Repeat the same invocation
after a controller interruption; immutable receipts reconcile the existing
work.

For `train`, the bridge inventories only publishable `.pth`, `.ckpt`, or
`.safetensors` checkpoints plus TensorBoard event files. It rejects symlinks,
empty files, traversal, and a successful job with no checkpoint; copies each
artifact by exact relative path, size, and SHA-256; and writes
`train.output-sync.json` before marking the job complete. This prevents a
later exact-tree adapter stage from deleting backend-only checkpoints.
`publish_checkpoint` applies the same nested-output rule to its host log,
normalized pretrained state, best-checkpoint metadata, and best checkpoint,
recording `publish_checkpoint.output-sync.json`. A successful legacy publisher
whose nested outputs remain on SLURM can be reconciled without recompute via
`recover-publish-checkpoint-sync`; the recovery accepts only the exact
terminal COMPLETE request/job/status lineage.

Once a run-owned remote results tree has a valid prior staging receipt, later
actions use copy-on-write incremental staging. The helper hard-links the
existing tree into an action-specific candidate, runs checksum-based `rsync`
with exact deletion into that candidate, requires the protocol's
receiver-acknowledged zero exit, writes the new receipt, and atomically promotes it. An
absent or invalid prior receipt falls back to the full tar stream. This keeps
unchanged checkpoints on Lustre without weakening exact-tree semantics or
mutating the currently committed remote tree.

One compatibility recovery exists for a run completed by an older bridge. Use
the same command with positional operation `recover-train-output-loss` only
after a successful attempt-2 train is followed by publisher attempt 1 failing
with the canonical “No checkpoints found under …/train” classifier. The
recovery proves that both controller and SLURM checkpoint inventories are
empty, binds the successful train and failed publisher requests, job records,
statuses, and logs by digest, archives the prior statuses, and replays the
exact train command once under `train.output-replay.*`. Any existing
checkpoint, sync receipt, publisher retry, changed evidence, different
failure, or second replay is rejected. This is not an ordinary third attempt.

The optional CPU/GPU time flags are approved scheduler bounds in `[10, 240]`
minutes. Defaults remain 120 minutes for zero-GPU adapters and 240 minutes for
model actions. Use a shorter value only when the launch review's workload
estimate supports it; the rendered SBATCH records the exact bound and SLURM
`TIMEOUT` remains terminal for that attempt.
If the Airflow consumer rejects the request before `sbatch`, the bridge proves
the deterministic job name absent through both remote `squeue` and `sacct`,
agent-cancels the never-submitted record, archives its stale binding with the
packaged bound-pre-submit recovery, and restores `NO_JOB_RECORD`. Repeating the
same action then opens and binds exactly one corrected record. Consumer plans
and Airflow envelopes are named by that job ID, so the corrected record never
overwrites the rejected attempt's orchestration evidence. Any exact native
match blocks that recovery and preserves the binding for reconciliation.
If Airflow itself terminates after native submit because its status SSH call
times out, the bridge directly rechecks the exact bound SLURM handle. A still
pending/running/unknown native job remains bound and is never marked terminal.
After the native job succeeds, rerun the same bridge command with positional
operation `recover-monitoring`. It accepts only an Airflow `ERROR` whose
orchestration log contains the bounded status timeout and no cancel, polls the
exact existing SLURM job, synchronizes its declared output and log, archives
any false controller terminal record, writes digest-bound recovery evidence,
and finalizes without resubmitting compute. Three consecutive `UNKNOWN`
results or any native error/cancel is a hard stop.
Use `--backend-dataset-root` only when the initialized run deliberately reuses
an existing verified controller dataset and the same dataset already exists on
Lustre at another absolute path. The bridge hashes the six pair/list metadata
files on both sides, rejects symlinked roots or canonical subtrees, and binds
the exact backend root into the staging receipt. It does not copy or rebuild
that million-file dataset and does not weaken the action's own full-count and
sample-link verification.

For the audit-selected `sdg` stage on SLURM, use the packaged composite bridge:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/airflow_slurm_sdg_action.py" \
    --results-dir "$RESULTS_DIR" --iteration "$ITERATION" \
    --login "$SLURM_LOGIN" --remote-workspace "$REMOTE_WORKSPACE" \
    --shared-root "$TAO_IAA_AIRFLOW_SHARED_ROOT" \
    --backend-dataset-root "$BACKEND_DATASET_ROOT" \
    --cache-dir "$SLURM_HF_CACHE" \
    --augmentation-sqsh "$AUGMENTATION_SQSH" \
    --auto-labeling-sqsh "$AUTO_LABELING_SQSH" \
    --image-edit-sqsh "$IMAGE_EDIT_SQSH" \
    --text-serving-sqsh "$TEXT_SERVING_SQSH" \
    --account "$SLURM_ACCOUNT" --partition "$SLURM_GPU_PARTITION"
```

The controller validates the committed `history_select` boundary, verifies the
reused dataset metadata on both filesystems, maps only signed backend paths,
and stages the exact run snapshot before opening the native SLURM job record.
Airflow then invokes `slurm_sdg_action.py` through its four direct verbs. The
native consumer submits `generation_nodes` independent one-node/eight-GPU
image-worker jobs plus one two-GPU coordinator job; it never widens GPU
selection or runs SDG on the Airflow host. Terminal status atomically mirrors
the four normalized SDG outputs, both endpoint manifests, and the platform
status into the controller results tree by remote size and SHA-256 before the
DAG may return `COMPLETE`. A failed or timed-out launch retains owned jobs and
evidence for the native bounded recovery; it does not silently switch backend
or submit a second composite action.

After the native group is terminal, cleanup is proven, and its TAO job record
is terminal `ERROR/ERR_INFRA`, the single approved scheduler retry uses the
same command plus both authoritative inputs:

```bash
  --retry-from-request "$ATTEMPT1_REQUEST" \
  --retry-from-job-record "$ATTEMPT1_JOB_RECORD"
```

The bridge asks the native producer to validate attempt-1 lineage, writes a
distinct attempt-2 request, opens a new job record with `retry_of`, and retains
the existing Lustre result tree rather than replacing its failure evidence.
Missing, active, program-error, or mismatched attempt-1 evidence is rejected;
there is no attempt 3.

If attempt 2 reaches all endpoints but fails before preprocessing because the
retained attempt-1 progress rejects the new attempt-owned image endpoint pool,
use the single allowlisted pool-rebind repair:

```bash
  --repair-from-request "$ATTEMPT2_REQUEST" \
  --repair-from-job-record "$ATTEMPT2_JOB_RECORD"
```

This is not a third controller retry. The signed request remains `attempt=2`,
retains the original retry lineage, and adds digest-bound proof of the exact
terminal job group, cleanup, execute log, and zero-generation progress. It is
accepted only for `image-edit endpoint pool changed outside explicit unfinished
resume`, with `preprocessed=false`, empty augmentation and labeling journals,
no normalized outputs, and every owned job terminal. Partial generation,
changed evidence, another repair, or any other error is a hard stop.
When the fix required an approved runtime rebind, the repair also binds the
exact final old-to-new runtime-lineage record and its validated PASS evidence;
no other request field may differ.

If that repair is canceled before any allocation executes because its requested
walltime cannot enter scheduler backfill, one shorter reschedule is allowed:

```bash
  --time-minutes 60 \
  --reschedule-from-request "$ATTEMPT2_REPAIR_REQUEST" \
  --reschedule-from-job-record "$ATTEMPT2_REPAIR_JOB_RECORD"
```

This remains attempt 2 and preserves both retry and pool-rebind repair lineage.
The producer requires the immediate parent job record to be terminal
`CANCELED`, or `ERROR/ERR_INFRA`, every exactly owned native allocation to be
`CANCELLED` with zero elapsed seconds, the repair-bound progress digest to be
unchanged, and all canonical outputs plus parent endpoint descriptors to be
absent. The new walltime must be strictly shorter; all other data, model, GPU,
scheduler, runtime, and output semantics must match. A rescheduled request
cannot be rescheduled again.

If the rescheduled group reaches the image workers but a same-node vLLM-Omni
service fails before generation because multiple replicas inherited one
`MASTER_PORT`, use the single launch repair:

```bash
  --time-minutes 60 \
  --launch-repair-from-request "$RESCHEDULE_REQUEST" \
  --launch-repair-from-job-record "$RESCHEDULE_JOB_RECORD"
```

The producer accepts only a terminal owned group with at least one failed image
worker, an unchanged zero-generation progress digest, absent canonical outputs,
and exact worker/endpoint logs proving `EADDRINUSE` on the inherited port. The
new consumer assigns eight deterministic, preflighted internal ports through
per-process `MASTER_PORT` environment values; the values are not placed in the
request, job record, argv, or report. The request remains attempt 2, retains all
retry/repair/reschedule lineage, and may receive this repair only once.

For a terminal successful action produced by an older bridge that synchronized
the adapter status but not its six bound nested outputs, insert the positional
operation `recover-materialize-sync` immediately after
`airflow_slurm_action.py` in the same command. This recovery accepts only the
exact successful baseline `dataset_materialize` request, platform status,
binding, and terminal SLURM job record. It fetches only missing controller
outputs, rejects any differing existing file, and writes the same digest-bound
sync receipt; it never resubmits compute or mutates the remote dataset.

Stage the exact packaged consumer into the Airflow shared root. The envelope
binds its SHA-256. The consumer plan contains four direct Python
argv arrays—`submit`, `status`, `logs`, and `cancel`—plus bounded polling,
deadline, expected-output, retention, and environment-name policy. It contains
no shell, credential value, wildcard, or reconstructed GPU shape.

`airflow_orchestrator.py prepare` validates the plan and emits the strict
schema in `airflow-orchestration-request.schema.json`:

```bash
ORCHESTRATION="$ACTION_STAGE/airflow-orchestration.json"
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/airflow_orchestrator.py" prepare \
    --compute-platform "$PLATFORM" \
    --compute-kind "$COMPUTE_KIND" \
    --compute-request "$COMPUTE_REQUEST" \
    --job-record "$JOB_RECORD" \
    ${JOB_BINDING:+--job-binding "$JOB_BINDING"} \
    --consumer-plan "$CONSUMER_PLAN" \
    --output "$ORCHESTRATION"
```

The plan is accepted only when every verb invokes Python directly and the
script basename exactly matches the matrix above. `cancel` must retain
`--confirm`. Credential flags, credential values, unexpected environment
names, malformed paths, request/job identity drift, consumer-byte drift, and
`--gpus all` are rejected before DAG submission. `DAG conf` contains only the contract,
orchestration ID, envelope path, and envelope digest.

## Canonical DAG behavior

The packaged DAG remains deliberately small:

```text
validate envelope and immutable backend evidence
  -> reconcile or submit the exact backend consumer once
  -> persist backend handle in an atomic delegation receipt
  -> poll backend-native status within the approved deadline
  -> capture sanitized bounded backend logs
  -> validate declared outputs
```

The Airflow task does not execute TAO or SDG logic itself for composed runs.
It invokes the backend's consumer, and a retry reuses the digest-bound receipt
instead of resubmitting. Three consecutive `UNKNOWN` statuses by default, or
the approved smaller/larger bound, fail closed. Deadline expiry records
`UNKNOWN` and retains owned work for diagnosis; it does not silently destroy
containers, pods, instances, or scheduler jobs.

Completion is conjunctive:

1. the Airflow DAG run is `success`;
2. the backend consumer reports `COMPLETE`;
3. the receipt still binds the request, job identity, platform, and backend
   handle;
4. every declared shared output is non-empty, regular, non-symlink, and fresh;
5. the normal backend finalizer, stage commit, and IAA audit succeed.

An Airflow success alone is never TAO completion evidence.

## Four verbs

Submit the envelope, not the backend request, to Airflow:

```bash
AIRFLOW_RESULT=$(
  "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
    "$SKILL_ROOT/scripts/airflow_orchestrator.py" submit \
      --envelope "$ORCHESTRATION"
)
BACKEND_REF=$(python3 -c \
  'import json,sys; print(json.load(sys.stdin)["backend_ref"])' \
  <<<"$AIRFLOW_RESULT")

"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/airflow_orchestrator.py" status \
    --envelope "$ORCHESTRATION" --backend-ref "$BACKEND_REF"
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/airflow_orchestrator.py" logs \
    --envelope "$ORCHESTRATION" --backend-ref "$BACKEND_REF" --tail 200
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/airflow_orchestrator.py" cancel \
    --envelope "$ORCHESTRATION" --backend-ref "$BACKEND_REF" --confirm
```

`status` reports Airflow and compute states separately. It returns `COMPLETE`
only when both are complete. `logs` reads only the receipt-bound sanitized
orchestration log. `cancel` invokes the exact backend consumer's confirmed
cancel verb first and deletes the DAG run only after compute cancellation or
already-terminal completion is proven. A missing handle, foreign ownership,
or unconfirmed compute cancellation retains the DAG run and returns an
actionable error.

## Backend-specific invariants

- Docker: preserve the request's explicit GPU IDs and exact owned container
  labels. Stop, but do not remove, the owned container on cancellation.
- SLURM: ordinary actions use `slurm_action.py`; SDG reuses
  `slurm_sdg_action.py` and its `generation_nodes=N` independent one-node,
  eight-GPU workers plus coordinator. Never turn N into one N-node service.
  Airflow controllers use `airflow_slurm_action.py` for each ordinary action;
  it records remote mount sources in the staging receipt and synchronizes
  results before terminal success.
- Kubernetes: ordinary actions use `kubernetes_action.py`; SDG reuses
  `kubernetes_sdg_action.py`. Preserve namespace/PVC/ServiceAccount/Secret and
  exact request ownership; never `apply` over a same-name foreign Job.
- Brev: reuse the immutable instance inventory and the canonical ordinary/SDG
  consumers. Single-host and multi-host topology remain Brev-owned. Never run
  local Docker as a fallback when Brev is unavailable.
- Virtualenv: use `virtualenv_runner.py` for TAO/CPU actions with PID-safe
  process-group ownership. `local_sdg_action.py` owns the detached composite
  controller and container-backed endpoints on the same host.

## Recovery

- Lost Airflow submit response: repeat the exact submit. A deterministic DAG
  run is accepted only when its complete conf equals the envelope binding.
- Airflow task restart after compute submit: reuse the atomic receipt and
  backend handle. Never invoke submit twice.
- Backend `ERROR` or `CANCELED`: capture bounded logs, retain diagnostic
  resources according to the approved policy, finalize the failed attempt,
  and use only the application's remaining corrected attempt.
- When attempt 1 ends as `ERROR/ERR_INFRA` because no image worker became
  ready, attempt 2 may move to another launch-reviewed SLURM partition. The
  signed retry lineage records the old and new partitions and rejects every
  other workflow, model, dataset, resource, time-limit, or command change.
  Live or merely canceled groups are ineligible: the coordinator must be
  terminal `FAILED` and every exact-owned worker must be terminal.
- Repeated `UNKNOWN`: stop launching. Reconcile ownership in the selected
  backend; do not infer success from files or Airflow state.
- Controller interruption: query both Airflow and the backend through the
  envelope. The TAO job record remains backend-native.
- Legacy `platform=airflow`: resume only with the original direct-Airflow v1
  request and evidence. Do not migrate its immutable state in place.

The launch review must show the compute platform and Airflow orchestrator as
separate rows, the exact consumer and request digest, shared evidence scope,
backend resource shape, polling/deadline/unknown bounds, credential names
(presence only), and retention/cancellation policy.
