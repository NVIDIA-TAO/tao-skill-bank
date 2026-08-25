---
name: tao-run-on-slurm
description: Remote SLURM GPU cluster execution over SSH with sbatch/srun, Pyxis/Enroot containers, and Lustre-backed
  results. Use when running TAO training/eval/inference jobs on an on-prem or DGX SLURM cluster. Trigger phrases include
  "run on SLURM", "submit sbatch", "DGX SLURM cluster", "Pyxis/Enroot container", "Lustre dataset".
license: Apache-2.0
compatibility: Requires SSH access to a SLURM login node (passwordless via key auth) and SLURM_USER + SLURM_HOSTNAME env vars.
  No nvidia-tao-sdk install is required; jobs are driven directly over ssh + sbatch/squeue/sacct/scancel.
metadata:
  author: NVIDIA Corporation
  version: "0.1.1"
allowed-tools: Read Bash
tags:
- platform
- slurm
---

# SLURM

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

Remote GPU compute platform for clusters managed by SLURM. Jobs are submitted
from the launch host to a login node over SSH, staged on a shared
filesystem, submitted with `sbatch`, and executed with `srun` container support.

## When to use

Use SLURM when the user has access to a managed GPU cluster, shared Lustre
storage, and scheduler-owned GPU allocation. Do not use SLURM for local files
that exist only on the agent machine; data and outputs must be reachable from
the cluster.

## Preflight + SSH

Confirm `SLURM_USER` and `SLURM_HOSTNAME` are exported and passwordless SSH to a
login host works (`ssh -o BatchMode=yes`).
The launch host needs `ssh`, not local `sbatch`, `srun`, Enroot, or a Lustre
mount. Preflight those scheduler, Pyxis, Enroot, and shared-storage dependencies
on the selected remote login/compute frame. Model-specific inspectors may be
streamed from the installed skill over SSH stdin; do not stage an ad-hoc source
patch or treat the launch host as the SLURM frame.
For private `nvcr.io` images, install `~/.config/enroot/.credentials` on the
cluster once per (cluster, user): Pyxis/Enroot does not read `NGC_KEY` from the
job env, and without persistent credentials, auth-gated pulls fail with "Could
not process JSON input" at job startup. Install it via the `printf | ssh`
heredoc so the `NGC_KEY` value never lands in shell history, intermediate files,
or chat output; never `cat`/`echo` the value.

If a preflight check fails, the agent prompts the user to authorize the
install/fix via Bash. Pip-installable Python requirements are the exception:
install them automatically, then rerun preflight.

See `references/slurm-ssh-credentials.md` for the full preflight script, the
enroot-credentials heredoc, prerequisite key setup (keypair, `ssh-copy-id`,
`known_hosts`, container key mounts, 2FA handling), and the SSH failure
remediation prompt.

## Execution — the four verbs

`tao-run-on-slurm` is a platform **consumer**: it runs a spec-bundle over
`ssh + sbatch/squeue/sacct/scancel`, mutating only the job-record. Storage is
**tier A** (Lustre) — the dataset is staged to a shared path *before* submit and
read through Pyxis; never fetch S3 inside the allocation (the scheduler-idle
timeout kills GPU-idle jobs and bills the wasted time). `$BANK` =
`${TAO_SKILL_BANK_PATH}`; `$LOGIN` = a resolved `SLURM_HOSTNAME`.

Ordinary typed IAA actions and IAA Airflow envelopes use packaged
`scripts/slurm_action.py` for all four verbs. It delegates submit to the
hardened gate and proves exact native ownership for every other verb; do not
substitute the freehand examples below.

### submit

For a DEFT IAA SDG action, `generation_nodes=N` selects the maximum service
pool, not a multi-node batch allocation. Submit **N independent one-node/eight-GPU image
worker jobs**; each owns eight TP=1, one-GPU image-edit services. Submit one
separate one-node/two-GPU coordinator for the VLM, LLM, shared SDG runtime, and
component steps. Use only the packaged `slurm_sdg_action.py` four-verb consumer
for this composite action. Before the launch review, use its deterministic
`prepare-request` operation to derive and sign JSON from initialized DEFT state;
never hand-author the action request. It records exact native job ownership,
requires at least one complete eight-service worker, and records the selected
active capacity separately from the approved `8 * N` maximum.
Read `references/slurm-sdg-action.md` for the signed request, endpoint-pool,
serialized-submit, bounded duplicate-submit recovery, resume, authentication,
status, and cleanup contracts before approval or launch. Do not route this
action through the ordinary multi-node template.

1. **Reuse what's already staged — never redo (tier A):**
   - *Image:* `@@IMAGE@@` is a Lustre `.sqsh` — **reuse an existing one if present**
     (`ssh $LOGIN ls <sqsh>`); only if missing, convert once with `enroot import`
     (cached by name — see `references/slurm-container-execution.md`).
   - *Dataset:* **confirm it is already on Lustre** (`ssh $LOGIN test -e …`) and
     reference those paths; `tao-data-io` stages *only* a small auxiliary input
     that is not there yet — never re-stage existing data, and never the training
     set inside the allocation.
   Then author the spec at `<job_dir>/specs/spec.yaml` on Lustre with those paths.
   For large trees, digest-bound subsets, and IAA controller/patch snapshots,
   follow the atomic `slurm_stage_tree.py` contracts in
   `references/slurm-container-execution.md`. Stage only request-declared
   inputs, never a plugin-cache path or the full mining-pool tree.
2. **Credentials → sidecar (never inline):** if the run needs session creds
   (e.g. `HF_TOKEN`), write them to a mode-600 sidecar on Lustre and let the
   template shred it on exit; NGC image pulls use the one-time
   `~/.config/enroot/.credentials` (see `references/slurm-ssh-credentials.md`),
   not the job env. For a platform-neutral action request, write exactly the
   variable names listed in `request.forward_env`; never copy a whole launching
   environment or credentials file into the sidecar:
   ```bash
   set -a; source /path/to/.env; set +a   # omit if already exported
   printf 'export HF_TOKEN=%s\n' "$HF_TOKEN" | ssh $LOGIN "umask 077; cat > <job_dir>/job_$JOB_ID.env"
   ```
3. **Open the record — mints the id, binds `results_dir` on Lustre, before launch:**
   ```bash
   JOB_ID=$("$BANK/scripts/tao_job_record.py" open --platform slurm --image "$IMAGE" \
     --network-arch "$ARCH" --action "$ACTION" --storage-tier A --results-root "$SLURM_BASE_RESULTS_DIR")
   ```
4. **Render** `templates/slurm/singlenode.sbatch.tmpl` — substitute every
   `@@<NAME>@@` (`JOB_NAME=$JOB_ID`, `NUM_GPUS`, `CPUS_PER_TASK`, `TIME`, `LOG_DIR`,
   `IMAGE`, `CONTAINER_MOUNTS=<RUNTIME_SUPPLIED_MOUNTS>`, `COMMAND=<bundle command reading the shared-storage
   spec>`, `SBATCH_EXTRA=` account/partition lines, `ENV_FILE=` the sidecar path or
   empty, `EXTRA_ENV=` any cluster NCCL knobs, and `REQUEUE_DIRECTIVE=` exactly
   `#SBATCH --requeue` or `#SBATCH --no-requeue` from the approved setting) →
   `<job_dir>/sbatch/job_$JOB_ID.sbatch`. Create `LOG_DIR` before submit; the
   template intentionally writes flat `%x-%j.out` and `%x-%j.err` files because
   SLURM does not create intermediate directories in output paths.
   For an IAA action request, require `len(request.gpu_ids)` to equal
   `spec_bundle.compute_shape.gpus`, reject duplicates or negative IDs, and
   render that count as `NUM_GPUS`. Those IDs bind the user's approved GPU
   quantity; they are not login-host device ordinals. Do not export them as
   `CUDA_VISIBLE_DEVICES`: SLURM selects physical devices and Pyxis exposes only
   the allocation inside the container. This preserves the explicit selection
   without widening the request to every cluster GPU.
   For only an IAA `clip train` bundle, prefix the exact bundle argv with the
   request-mounted `/patches/run_clip_train_slurm.sh`. Keep the single parent
   `srun` task: TAO owns the per-GPU launcher, while the wrapper removes only
   `SLURM_NTASKS`, `SLURM_NTASKS_PER_NODE`, `SLURM_PROCID`, `SLURM_LOCALID`,
   `SLURM_NODEID`, and the inherited single-parent distributed frame
   (`WORLD_SIZE`, `RANK`, `LOCAL_RANK`, `NODE_RANK`, `MASTER_ADDR`,
   `MASTER_PORT`, `NUM_GPU_PER_NODE`) inside that container process so TAO and Lightning do not
   mistake the scheduler's one parent task for externally launched DDP ranks.
   It retains job/account variables and GPU visibility. Do not fake topology
   counts, apply the wrapper to evaluate/embedding, or change the request argv.
   Both single- and multi-node templates bind only the fixed non-secret
   `NCCL_DEBUG`, `LOGLEVEL`, `NCCL_P2P_DISABLE`, `NCCL_IB_DISABLE`,
   `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, and `NCCL_NET` names through Pyxis
   `--container-env`; a host-shell export alone does not prove the setting is
   visible inside the container. Never add a credential, whole environment,
   or user-supplied arbitrary name to this allowlist.
   If both normal IAA training attempts have already terminated in the two
   exact pre-workload topology failures recognized by the application
   producer, use its one-shot `launcher-repair` verb. Submit only the emitted
   repair request through this same path. Never hand-edit or reuse an earlier
   request, and never use this exception for a runtime/data/model failure, an
   unknown log, a started batch, an output/checkpoint, or a second repair.
   Submit the rendered file only through packaged
   `scripts/slurm_submit_action.py`. It secret-lints and syntax-checks the local
   input. For every typed IAA action, pass its immutable `--request` and
   request-owned `--job-binding`; the gate validates their digests, action/job
   ownership, staged-absence receipt, fresh PENDING record, and exact ordered
   Pyxis mount, environment, model/adapter, and scheduler GPU contracts before any
   scheduler submit. It then proves the exact job name is absent, copies to a target-scoped
   temporary file, verifies non-emptiness plus byte-identical SHA-256 before
   and after atomic promotion, runs remote `bash -n` and `sbatch --test-only`,
   rechecks exact-name absence, then submits and returns the parsed native
   handle. A lost or malformed submit reply triggers exact-name reconciliation
   every two seconds for at most 60 seconds. Treat any rejection as blocking;
   in particular, an explicit-account error means request `SLURM_ACCOUNT`, add
   only its `#SBATCH --account` directive, and rerun lint, syntax, and test-only.
   `--test-only` validates the scheduler contract without submitting a job.
   A signed action with `spec_bundle.compute_shape.gpus=0` is an IAA
   compute-frame adapter, not permission to run it on the controller. Render
   `templates/slurm/cpu.sbatch.tmpl`, use an approved CPU partition, omit every
   GPU/GRES directive, and preserve the same staging receipt, job binding,
   exact mounts, four verbs, synchronization, and finalization. Reject
   `--gres=gpu:0` and controller-local execution.
5. **Submit + record RUNNING:**
   ```bash
   SUBMIT_JSON=$(python3 "$BANK/skills/platform/tao-run-on-slurm/scripts/slurm_submit_action.py" \
     --login "$LOGIN" --job-id "$JOB_ID" --rendered-script <local-rendered.sbatch> \
     --remote-script <job_dir>/sbatch/job_$JOB_ID.sbatch \
     --request "$ACTION_REQUEST" --job-binding "$JOB_BINDING")
   SLURM_ID=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["backend_ref"])' \
     <<<"$SUBMIT_JSON")
   "$BANK/scripts/tao_job_record.py" mark "$JOB_ID" --state RUNNING --backend-ref "$SLURM_ID"
   ```

A submit that skipped the gate or the open has no id — so it cannot launch.

### status

```bash
# sacct ANNOTATES states ("CANCELLED by 12345") and truncates them to the
# default column width, so a cancelled job reads back as "CANCELLED+" and
# matches nothing in the table below — reporting UNKNOWN instead of CANCELED.
# Widen the column, take the first word, drop the truncation marker.
st=$(ssh $LOGIN "sacct -j $SLURM_ID -X -n -o State%30" | awk '{print $1}' | tr -d '+')
# (use squeue while the job is still PENDING; sacct lags briefly after submit)
```

| SLURM state | vocab |
|---|---|
| `PENDING` | `PENDING` |
| `RUNNING` / `COMPLETING` | `RUNNING` |
| `COMPLETED` | `COMPLETE` (confirm `status.json` in `results_dir`) |
| `FAILED` / `TIMEOUT` / `OUT_OF_MEMORY` | `ERROR` (infra-vs-program classify → retry, M6) |
| `NODE_FAIL` / `BOOT_FAIL` | `ERROR`, `err_class=ERR_INFRA` (`--requeue` re-queues these) |
| `CANCELLED` / `PREEMPTED` / `REVOKED` | `CANCELED` |
| (not found) | `UNKNOWN` |

Native sub-state rides in the transition `message`. Poll at the chosen interval;
long queue waits are normal — do not stop on elapsed time.

### logs

```bash
ssh $LOGIN "tail -n ${N:-200} <log_dir>/$JOB_ID-$SLURM_ID.out"
```

### cancel

```bash
ssh $LOGIN "scancel $SLURM_ID"
"$BANK/scripts/tao_job_record.py" mark "$JOB_ID" --state CANCELED --source agent
```

Treat an already-terminated SLURM job as a successful cancel.
For a bound IAA job canceled while still pending, SLURM may create no native
log. Use the application producer's packaged `capture-preallocation-cancel`
verb on the login frame; it accepts only exact zero-runtime/no-node `sacct`
evidence and writes the typed receipt required by finalization. Never create a
placeholder log by hand.

### Multi-node (nodes > 1)

Same four verbs, with three additions at submit:

1. **Render `templates/slurm/multinode.sbatch.tmpl`** instead of the single-node
   one — it's a strict superset (adds `--nodes` / `--wait-all-nodes` + the
   rendezvous block). `WORLD_SIZE` is the **node count** (TAO's misnomer); never
   change it to a global-rank count.
2. **NCCL probe first** — before the real job, run a cheap 2-node all-reduce
   (`scripts/nccl_allreduce_probe.py` under the container's torchrun) with a
   ~120s timeout. Before invoking torchrun, preserve the TAO rendezvous values
   as `TAO_NODE_COUNT=$WORLD_SIZE`,
   `TAO_GPUS_PER_NODE=$NUM_GPU_PER_NODE`, and
   `TAO_NODE_RANK=$SLURM_PROCID`; torchrun overwrites its standard
   `WORLD_SIZE` with the global process count. `NCCL_PROBE_OK` → proceed.
   **Timed out** (the collective hung)
   → set the cluster's NCCL knob in `EXTRA_ENV` and re-probe — on CS-OCI-ORD that
   is `export NCCL_P2P_DISABLE=1` (the intra-node P2P hang), often with
   `NCCL_SOCKET_IFNAME=eth0` / `NCCL_IB_DISABLE=1`. When an external NCCL
   network plugin still selects IB despite that setting, bind
   `NCCL_NET=Socket` and prove its in-container value before the bounded
   re-probe. **Cache the working env per
   cluster** so later jobs skip the probe. Gate on **`gpus_per_node > 1` too** —
   the P2P hang triggers on a single node with 2+ GPUs.
3. Tier-A Lustre, sidecar creds, record, and lint are unchanged.

### Cosmos backend guardrails

Read [`references/cosmos-slurm-guardrails.md`](references/cosmos-slurm-guardrails.md)
before rendering a Cosmos command. It defines image staging, planner
materialization, Framework and Cosmos-RL launch contracts, worker/runtime
requirements, and exit/status handling.

## Storage

Use shared-filesystem URIs, not local or `file://` paths; `tao-core` rejects
local/file paths for remote backends.

- `lustre:///absolute/path` for user-provided datasets on Lustre.
- `slurm://` paths may appear in microservices metadata and are converted to
  Lustre paths before the container starts.

Accept either dataset roots (model skills map them to required files) or direct
spec-key paths. After SSH succeeds and before generating scripts, `test -e` each
required dataset path from the login host; if it fails, stop and ask for
corrected paths or staged data rather than producing scripts that fail in the
first training job. See `references/slurm-ssh-credentials.md` for root vs.
direct-spec modes, backend details, and the results-dir default.

## Container execution

Stage compact specs and metadata on Lustre, acquire and validate cached `.sqsh`
images on a CPU partition before allocation, render the run-owned batch file,
and execute it through Pyxis/Enroot with explicit mounts. Never fall back to a
registry pull inside a GPU allocation after conversion failure. Read
`references/slurm-container-execution.md` for accepted image forms, conversion
cache and node-local temporary-directory rules, NCCL environment forwarding,
multi-node rendezvous, and failure recovery.

Direct Enroot conversion must use a job-unique node-local directory and a
stable working directory. Preserve these exact submission settings:
`ENROOT_TEMP_PATH=/tmp/enroot-tao-\${SLURM_JOB_ID}`,
`SLURM_ENROOT_TEMP_PATH=\${ENROOT_TEMP_PATH}`, and `--chdir=/tmp`.

## Monitoring and cancellation

Poll the stored native ID through `squeue`/`sacct`, bind completion to terminal
evidence in shared results, and continue requested monitoring across normal
queue waits. Read logs and cancel only after exact ownership resolution. The
full state map, terminal-evidence rules, log paths, detach behavior, and
infrastructure retry classification are in
`references/slurm-container-execution.md`.

## Required inputs

Ask for these in the SLURM intake; see `references/slurm-ssh-credentials.md`
for the full credential list, microservices schema keys, and defaults.

- **SLURM_USER** (required): SSH username for the login node.
- **SLURM_HOSTNAME** (required): Comma-separated login hostnames for failover.
- **SLURM_PARTITION** (optional): Partition list for GPU submission. The
  packaged default is `polar,polar3,polar4,grizzly`, treated as 4-hour queues.
  Ask only when the user wants a different partition or the scheduler rejects
  the default.
- **SSH_KEY_PATH** (preferred, expected before launch): private key for
  non-interactive public-key auth. Ask for this first in remediation; prefer it
  over the `SSH_AUTH_SOCK` agent-socket fallback.
- **SLURM_BASE_RESULTS_DIR** (optional): base shared-filesystem path; default
  a shared-storage root supplied and verified at runtime.
- **SLURM_ACCOUNT** (usually required by site policy): account for `#SBATCH --account`.

Do not ask for `SLURM_ACCOUNT` or `SLURM_BASE_RESULTS_DIR` in the initial
intake unless the user says their site requires an account, wants a custom
results root, or the workflow cannot proceed without overriding defaults.

## Resource defaults

Use `references/skill_info.yaml` as the packaged source for node, GPU, CPU,
wall-time, partition, mount, SQSH, and conversion defaults. Do not substitute a
12-hour default: verify any override against the selected partition before
approval. The SDG composite topology is the explicit exception documented in
`references/slurm-sdg-action.md`.

## Multi-node and retries

Ordinary distributed training uses the multi-node template, bounded NCCL probe,
and TAO rendezvous contract described above. Stage large inputs on Lustre before
allocation. On failure, classify infrastructure versus program errors, honor
workload-specific requeue policy, create a new retry record when agent-level
resubmission is justified, and reconcile ambiguous submission by exact name.
The complete decision table is in `references/slurm-container-execution.md`.

## References

- `references/slurm-ssh-credentials.md` — preflight script, SSH/key setup,
  enroot credentials, full credential list, backend details, storage rules,
  SSH remediation prompt.
- `references/slurm-container-execution.md` — container execution steps,
  monitoring, status mapping, cancellation, multi-node detail,
  Lustre-not-S3, retries, failure modes.
- `references/slurm-preflight-storage.md` — extended preflight/storage notes.
- `references/slurm-sdg-action.md` — DEFT IAA SDG fan-out topology, signed
  request, serialized four-verb execution, duplicate-submit recovery,
  readiness, endpoint pool, resume, authentication, and exact-owned cleanup.
- `references/cosmos-slurm-guardrails.md` — Cosmos Framework and Cosmos-RL
  launch and status guardrails.
- `references/detailed-guide.md` — navigation map for the split references.
