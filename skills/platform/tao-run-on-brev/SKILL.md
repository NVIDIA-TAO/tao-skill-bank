---
name: tao-run-on-brev
description: Run a TAO training/evaluation/inference container on an NVIDIA Brev GPU instance. Instance provisioning (create/search/stop/delete/login) is delegated to the official brev-cli agent skill or the Brev MCP server; this skill covers only the TAO-specific part — running the container over `brev exec` via the four-verb docker contract. Trigger phrases include "run on Brev", "Brev GPU instance", "TAO on Brev", "submit job to Brev".
license: Apache-2.0
compatibility: Requires the brev CLI (https://github.com/brevdev/brev-cli) and an active brev login. Instance provisioning is handled by the official brev-cli agent skill or the Brev MCP server.
metadata:
  author: NVIDIA Corporation
  version: "0.3.0"
allowed-tools: Read Bash
tags:
- gpu
- compute
- instance-based
- brev
---

# Brev — TAO execution glue

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

NVIDIA Brev provides on-demand GPU instances (pre-loaded with NVIDIA drivers,
CUDA, Docker, and the NVIDIA Container Toolkit). Brev is **instance-based**: you
provision an instance, run commands on it over `brev exec`, and delete it when
done.

This skill is deliberately thin. **Provisioning and managing instances — create,
search by GPU/price, start/stop, delete, login — is owned by NVIDIA Brev's own
agent skill, not duplicated here.** This skill covers only the TAO-specific part:
running a TAO container on a reached instance through the **four-verb docker
contract**, deferring the container-how to `tao-run-on-docker` over `brev exec`.

## Provisioning: use the official Brev skill or MCP

NVIDIA Brev publishes an agent skill that manages instances in natural language
("create an A100 instance", "search for GPUs under $3/hr", "stop all my
instances"). Install it once — it self-registers into your agent's skills dir and
is discovered at runtime:

```bash
curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/scripts/install-agent-skill.sh | bash
# installs to ~/.claude/skills/brev-cli/ , ~/.codex/skills/brev-cli/ , ~/.agents/skills/brev-cli/
```

Or connect the **Brev MCP server** (`https://docs.nvidia.com/brev/_mcp/server`).
Either one owns login/auth quirks, placement IDs, GPU search, and teardown flags.
It does **not** cover container execution on the instance — that is this skill.

**Preflight for this skill:** the `brev` CLI is on `PATH` and has an active
authenticated session. An existing interactive/cached login is valid; headless
automation may instead set `BREV_API_TOKEN` and run
`brev login --token "$BREV_API_TOKEN"` before any other call. Do not require or
request a token when `brev ls --json` already succeeds. You must also be able to
reach a target instance — poll with a **two-word** command until it succeeds
before issuing real work (a fresh instance reports `RUNNING` before sshd is up):

```bash
BREV_TRANSPORT="${TAO_SKILL_BANK_PATH:?}/skills/platform/tao-run-on-brev/scripts/brev_transport.py"
for i in $(seq 1 60); do
  python3 "$BREV_TRANSPORT" ready --instance <instance> && break
  sleep 5
done
python3 "$BREV_TRANSPORT" ready --instance <instance> || {
  echo "instance not exec-ready"
  exit 1
}
```

The helper sends a **two-word command as one argument** and accepts a complete
`TAO_BREV_READY` output line. Current Brev CLI releases may append an instance
name to successful output, so byte-for-byte stdout equality is not a readiness
test. A single-token probe
(`brev exec <instance> -- true`) passes even when every real command is broken,
because `brev exec [instance...] <command>` treats only the LAST positional as
the command — so a lone `true` lands in the right slot by accident while
`docker run ...` does not. See *`brev exec` argument form* below.

Allow **≥ 600 s** for the first `brev exec` on a new instance (SSH bring-up +
first container pull); a 60–120 s wrapper timeout truncates startup and looks
like a spurious `exec failed`.

## Storage

No shared NFS/Lustre — storage tier **B/C** via `tao-data-io`: stage inputs from
S3 to the instance's local disk (or fetch in-container) and **upload results to S3
before deleting the instance**. Instance-local `~/` persists across stop/start but
**not** across delete/create, so the results upload must precede teardown.

## Execution — the four verbs (a compound over Docker)

Brev is a **compound consumer**: `submit` reaches an instance, then **defers the
container-how to the four docker verbs** (`tao-run-on-docker`) run over
`brev exec`. Use the packaged `brev_action.py`; do not reconstruct Docker or
`brev exec` quoting from prose. It validates the immutable action digest,
requires every staged mount, preserves the request's exact `gpu_ids`, runs as
the remote non-root UID/GID, forwards only approved credential names, labels
the container with the job id, and refuses to replace an existing container.
The only zero-GPU exception is a signed, allowlisted `tao-run-deft-iaa` adapter
request. For that exception it omits `--gpus`, inventories the complete staged
controller skills-root and `/patches` trees against their signed per-file
manifests and aggregate digests before submit, and separately checks the IAA
runtime digest. The `/iaa-runtime` mount must be the fixed
`applications/tao-run-deft-iaa/scripts` subdirectory of that skills-root. This
covers the controller entry point, application references, core artifact
schemas, and all code imported through `PYTHONPATH`. It adds the action,
request, and runtime digests as container labels. Zero-GPU TAO actions and
arbitrary Python requests fail closed. The adapter environment must equal the
producer's five-entry non-secret environment contract; the request digest is
an integrity check, not authorization for added or changed variables.
It is not a symmetric peer — teardown must additionally delete an ephemeral
instance to stop billing. `$BANK` = `${TAO_SKILL_BANK_PATH}`.

For a workflow action, first follow the producer's remote staging contract:
mirror every declared input to instance-local absolute paths, delete every
`staging_absent_paths` item, persist the producer's staging attestation, open
the job-record against that exact remote results scope, and bind it before
native submit. Build one `--mount COMPUTE_TARGET=/remote/source` argument for
every request mount target. Do not omit absolute-path aliases.
For an IAA adapter, stage the declared controller skills-root and patches
snapshots to instance-local directories. Map the derived application scripts
subdirectory to `/iaa-runtime` and the patches root to `/patches`;
`brev_action.py` inventories the complete roots and rejects any mutated,
extra, missing, symlinked, or non-regular file before starting Docker. Result
mirroring, fresh-output deletion, the staging receipt, and job binding remain
owned by the producer's remote staging/finalization contract.

When an IAA request contains `cache_subset`, do not mirror the shared cache
root. Materialize it with the packaged IAA `stage_action_cache.py` helper,
then transfer that directory to the instance's `/cache` source with delete
semantics. The helper verifies the manifest and source digests, rejects
traversal and unrelated output, and is idempotent. The staging receipt's
request digest transitively binds the cache manifest. This excludes unrelated
model and Xet caches from TAO-only remote actions.

- **submit** — reach an instance (provision/reuse via the official Brev skill or
  MCP; reuse an existing instance by its `instance_id`; wait for readiness, above).
  Open the record to mint `$JOB_ID` **before** launch, bind it through the
  producer, then run the Docker `submit` verb inside the instance and mark
  RUNNING:

  ```bash
  BREV_ACTION="$BANK/skills/platform/tao-run-on-brev/scripts/brev_action.py"
  JOB_ID=$("$BANK/scripts/tao_job_record.py" open \
    --platform brev --image "$IMG" \
    --network-arch "$ARCH" --action "$ACTION" \
    --storage-tier "$TIER" --results-dir "$BACKEND_SCOPE" \
    "${UPLOAD_EXCLUDE_ARGS[@]}")
  # Bind this exact record with the producing workflow before submit.
  SUBMIT_JSON=$(python3 "$BREV_ACTION" submit --json --request "$ACTION_REQUEST" \
    --instance <instance> --job-id "$JOB_ID" \
    "${REMOTE_MOUNT_ARGS[@]}")  # one --mount target=source per request mount
  BACKEND_REF=$(python3 -c \
    'import json,sys; print(json.load(sys.stdin)["backend_ref"])' <<<"$SUBMIT_JSON")
  "$BANK/scripts/tao_job_record.py" mark "$JOB_ID" --state RUNNING \
    --backend-ref "$BACKEND_REF"              # instance is part of the ref: the
                                             # container is unreachable without it
  ```
  Optional IAA Airflow plans add `--reconcile`; that mode adopts only an
  existing container whose job and signed-request labels both match after a
  lost task response. Direct repeated submit retains its fail-closed behavior.
- **status / logs** — run `python3 "$BREV_ACTION" status --instance <instance>
  --job-id "$JOB_ID"` and `python3 "$BREV_ACTION" logs --instance <instance>
  --job-id "$JOB_ID"`. Status returns the fixed vocabulary. Capture complete
  logs at the producer's immutable action log path. Recover `<instance>` from
  the record's `backend-ref`.
- **cancel / teardown** — remove the container, then for an ephemeral instance
  delete it (stops billing), then mark the record. Never leave an ephemeral
  instance running:

  ```bash
  python3 "$BREV_ACTION" cancel --instance <instance> \
    --job-id "$JOB_ID" --confirm
  brev delete <instance>                      # ephemeral instances only
  "$BANK/scripts/tao_job_record.py" mark "$JOB_ID" --state CANCELED --source agent
  ```

### IAA generation topology

IAA synthetic-data generation is a specialized compound Brev job. Use the
packaged `brev_sdg_action.py`; do not translate this topology into independent
generic Docker submissions. For `generation_nodes=1`, the canonical and fully
supported layout is one existing, exec-ready Brev host with exactly eight
visible GPUs of at least 80000 MiB each. All work stays on that host:

- GPUs 0-3 each serve one independent image-edit endpoint;
- GPU 4 serves the verification VLM;
- GPU 5 serves the query-generation LLM;
- GPUs 6-7 are reserved for TAO mining, training, and evaluation.

The image-edit services are reached at `127.0.0.1`; no control-host relay,
tunnel, cross-instance route, or locally running endpoint is part of this
topology. The signed hardware inventory, request, endpoint plans, readiness
evidence, and manager ownership must all agree with these exact GPU IDs. Any
`--gpus all`, missing slot, wrong model, insufficient VRAM, or foreign ownership
fails closed.

The existing multi-host layout remains available only for
`generation_nodes>1`: one distinct coordinator using exactly two explicitly
selected GPUs (VLM GPU 0 and LLM GPU 1; any additional coordinator GPUs remain
unused during SDG) plus exactly `N` distinct eight-GPU workers. The coordinator must directly
reach all `8*N` worker endpoints. The adapter never proxies, tunnels, or relays
image payloads through the machine running Codex; routing or firewall failure
is actionable and never silently shrinks the pool.

Do not hand-assemble the composite request. After `history_select` is committed,
have the Brev provisioning layer write a resolved inventory with exactly these
fields and a canonical `inventory_sha256` over the object excluding that digest:

```json
{
  "schema_version": "1", "platform": "brev", "status": "resolved",
  "topology": "single_host",
  "coordinator": {
    "instance": "<instance>", "gpu_count": 8,
    "gpu_memory_mib": [81920, 81920, 81920, 81920, 81920, 81920, 81920, 81920]
  },
  "workers": [
    {"id": "worker-0", "instance": "<instance>", "address": "127.0.0.1"}
  ],
  "inventory_sha256": "<lowercase-sha256>"
}
```

Then run the adapter's read-only preparation command before launch review:

```bash
BREV_SDG="$BANK/skills/platform/tao-run-on-brev/scripts/brev_sdg_action.py"
python3 "$BREV_SDG" prepare-request \
  --state "$RESULTS_DIR/deft_state.json" --iteration "$ITERATION" \
  --inventory "$BREV_INVENTORY" --remote-root "$REMOTE_ROOT" \
  --remote-cache "$REMOTE_CACHE" \
  --remote-controller-python "$REMOTE_ROOT/.venv/bin/python" \
  --output "$SDG_REQUEST"
```

`--remote-root` is the coordinator/worker mirror of the initialized local
workspace; `--remote-cache` is the approved remote model cache. Provision the
documented SDG controller dependencies in the run-scoped virtualenv named by
`--remote-controller-python` on every selected instance. The adapter binds that
absolute interpreter path into the signed request and probes that it is a real
virtualenv with `yaml`, `pandas`, and `pyarrow` before any endpoint starts.
It uses the same interpreter for endpoint management, SDG prepare, and SDG
execute; bare system `python3` is never a fallback. Preparation is
not an execution verb: it reads no credentials, reaches no instance, and starts
no service. It derives and binds the ordered workers, exact GPU roles and ports,
models, limits, canonical local/remote paths, expected outputs, state/config/
inventory digests, action identity, and request digest. The provisioning layer
must populate GPU memory from read-only inventory evidence, not an assumed SKU
label. Repeating preparation with
unchanged inputs reuses the byte-identical request. It rejects a wrong-platform,
duplicate, malformed, or topology-inconsistent inventory; a worker-count mismatch; an
already committed SDG stage; changed state/config; or a different existing
output.

On each `submit`, generate an ephemeral endpoint key and forward it only through
the process environment. Workers receive the same value under
`IMAGE_EDIT_API_KEY` and `VLLM_API_KEY`; the coordinator receives
the same ephemeral value under both `IMAGE_EDIT_API_KEY` and `VLLM_API_KEY`.
Only the environment variable names may appear in the
request, pool manifest, argv, state, logs, or report. A resume generates a new
key and may use the shared endpoint manager's `--recreate-owned` operation only
after exact run ownership has been reconciled. It must never replace or stop a
user-managed endpoint.

Each worker must first report every signed, owned, running, unit-capacity endpoint
(four in the canonical single-host layout; eight per multi-host worker).
The coordinator then directly validates `/v1/models` and a minimal image-edit
request against every slot. Commit the strict endpoint-pool manifest only after
the exact signed capacity passes; the runtime consumes that committed manifest through
`--image-edit-endpoint-pool`. A partial or stale pool is not runnable.

The public four verbs operate on the composite job:

- **submit** stages the signed request and shared helpers, reconciles the named
  instances in parallel, starts the exact-owned worker pools, proves direct
  coordinator readiness, starts the two coordinator services, and launches the
  SDG runtime. Pass `--resume` only for the same bound job and request.
- **status** aggregates the coordinator and every worker; it cannot report the
  composite job ready while any required worker is missing or unhealthy.
- **logs** returns the coordinator's canonical action log. The sanitized
  submit/status evidence carries each worker manager result and its evidence
  paths.
- **cancel** stops the coordinator and then only the exactly owned worker
  services. Confirmation remains required.

The lifecycle is bounded. Resume gets at most two controller attempts and does
not repeat committed stages. COMPLETE and CANCELED jobs stop exactly owned
services; ERROR retains them when useful for diagnosis, with cleanup made
explicit in the result. Instance deletion remains a separate, approved Brev
provisioning action.

### Large result-tree transfer and cache evidence

Do not use file-by-file `rsync` for a high-member-count tree. Always use the
packaged receipt-bound archive helper for immutable request/controller/patches
snapshots: their mode-`0500` directories can make recursive copy apply a
non-writable destination mode before children exist. Do not chmod or manually
reshape a signed snapshot. When a measured single-stream bottleneck requires
chunking, use the helper's deterministic `split`/`transfer-chunks`/`join` path
with at most four manifest-owned streams; it disables SSH ControlMaster so the
streams cannot silently share one connection. Never hand-build a tar, `xargs`,
or reassembly pipeline. Read
`references/archive-transfer.md` before approval or execution. It defines the
limits, atomic promotion, interruption recovery, symlink policy, cleanup, and
model-cache evidence contracts.

### `brev exec` argument form

The remote command is **one argument**. The CLI signature is
`brev exec [instance...] <command>`: every positional except the last is an
instance name, and `--` only ends flag parsing — it does not group the words
after it. So `brev exec <inst> -- docker inspect "$JOB_ID"` is read as
instances `<inst> docker inspect` plus command `"$JOB_ID"`, and fails with
`could not look up instance "docker"` / `ssh: illegal option -- -` (exit 255) —
an error that reads like an instance or SSH fault but is a syntax fault. Quote
every remote command as a single string, exactly as `brev exec --help` shows.

NGC auth once per instance — **never put `NGC_KEY` on argv** (it lands in the
remote process table). Brev CLI consumes its own stdin as piped instance names;
it does not forward that stream to the remote command. Use the packaged helper,
which sends the password over the Brev-managed SSH alias directly to
`--password-stdin`:

```bash
IMG=nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt  # versions-key: images.tao_toolkit.pyt

# NGC auth (one-time per instance) — value never appears on argv.
set -a; source /path/to/user-approved.env; set +a
printf %s "$NGC_KEY" | python3 \
  "${TAO_SKILL_BANK_PATH:?}/skills/platform/tao-run-on-brev/scripts/brev_transport.py" \
  registry-login --instance <instance> --registry nvcr.io \
  --username '$oauthtoken'

# Verify auth without reading ~/.docker/config.json. Failure before a successful
# login = not authenticated; failure after = the key's org lacks entitlement.
brev exec <instance> "docker manifest inspect $IMG >/dev/null && echo AUTH_OK || echo AUTH_FAIL"

# Pull BEFORE the GPU run. `docker run` would pull implicitly, but the instance
# bills from boot, so a multi-GB first-time TAO pull is billed GPU-idle time.
# Pulling as its own step also separates a pull failure (auth/entitlement) from
# a training failure in the logs.
brev exec <instance> "docker image inspect $IMG >/dev/null 2>&1 || docker pull $IMG"

# Submit only through brev_action.py after staging, job open, and producer bind.
# GPU actions render `--gpus '"device=<approved ids>"'`; the typed CPU adapter
# omits `--gpus`. Neither path ever widens GPU selection to all GPUs.
python3 "$BREV_ACTION" submit --request "$ACTION_REQUEST" \
  --instance <instance> --job-id "$JOB_ID" "${REMOTE_MOUNT_ARGS[@]}"
```

## Multi-GPU and multi-node

Distributed TAO training is not supported on Brev. Multi-GPU training **on a
single instance** is supported (up to 8× H100 / A100 / L40S); `torchrun
--nproc-per-node=N` or PyTorch DDP work within the instance. The IAA generation
pool above is the narrow cross-instance exception: its workers are independent
model-serving nodes coordinated through HTTP, not a distributed training job.
