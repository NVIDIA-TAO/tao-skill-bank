# Running a DEFT stage on any platform

Every stage runs the same way. A stage reference tells you *which* stage and
*what* to pass; it does not tell you how to launch, because that is the
platform skill's job.

```bash
BANK="${TAO_SKILL_BANK_PATH:?}"
AOI="$BANK/skills/applications/tao-run-deft-aoi/scripts"

# 1. Emit the stage as a spec-bundle.
python3 "$AOI/stage_bundle.py" anomalygen.amp \
  --results-dir "$RUN_DIR" \
  --param dataset_dir="$DS" \
  --param defect_spec="$DS/defect_spec.jsonl" \
  --param cosmos_models="$COSMOS" \
  --arg '${ANOMALYGEN_SCRIPTS}/prep_testcase.sh --name iter1 --num-sdg 20' \
  > "$RUN_DIR/amp.bundle.json"

# 2. Launch it on the chosen platform, and record it.
JOB=$(python3 "$AOI/deft_exec.py" --state "$STATE" --submit \
        --bundle "$RUN_DIR/amp.bundle.json" \
        --platform "$PLATFORM" $PLATFORM_CTX)

# 3. Poll to a terminal state.
python3 "$AOI/deft_exec.py" --state "$STATE" --await-job "$JOB" $PLATFORM_CTX

# 4. On failure, ask the platform — do not hand-write an ssh/kubectl probe.
python3 "$AOI/deft_exec.py" --state "$STATE" --logs "$JOB" --tail 100 $PLATFORM_CTX

# 4. Preserve the launch evidence per container stage. A grader (or you, next
#    week) cannot distinguish "launched on kubernetes" from "claimed to" from
#    transcript snippets; the job record carries platform and backend_ref.
STAGE_EVIDENCE="$RUN_DIR/launch/$JOB"; mkdir -p "$STAGE_EVIDENCE"
"$BANK/scripts/tao_job_record.py" show "$JOB" > "$STAGE_EVIDENCE/job_record.json"
cp "$RUN_DIR"/*.bundle.json "$STAGE_EVIDENCE/" 2>/dev/null || true
cp "$RUN_DIR"/configs/*.yaml "$STAGE_EVIDENCE/" 2>/dev/null || true
python3 "$AOI/deft_exec.py" --state "$STATE" --logs "$JOB" --tail 500 \
  $PLATFORM_CTX > "$STAGE_EVIDENCE/tail.log" 2>&1 || true
```

`--list` prints the stage table with each stage's required `--param` names:

```bash
python3 "$AOI/stage_bundle.py" --list
```

## Why the stage references no longer show `docker run`

They used to read:

```bash
docker run --gpus all --rm --ipc=host --shm-size=16g \
  --user $(id -u):$(id -g) -v /etc/passwd:/etc/passwd:ro \
  -v $WS:$WS -w /workspace/paidf-anomalygen $AG_IMAGE bash -lc "…"
```

That string is not a description of the stage — it is a description of *docker*.
It hardcodes a runtime, a GPU flag spelled docker's way, mount syntax spelled
docker's way, and flags that mean nothing on a scheduler. A stage written that
way cannot move platforms, and every new DEFT workflow forks the same lines
again.

A stage is only ever six facts: which image, what to run, what it reads, what it
writes, how much compute, and config-vs-args. Those are the spec-bundle fields,
and the platform skills turn a bundle into `docker run`, `srun
--container-image=`, or `kubectl apply` without the stage knowing which.

**Nothing is lost on docker.** The flags above are still emitted — by the docker
renderer, which is where they belong:

| docker recipe | where it lives now |
|---|---|
| `--gpus all` | `compute_shape.gpus`; each platform spells the request itself |
| `-v $WS:$WS` | `declared_inputs` (read-only) + `results_dir` (writable) |
| `-w <dir>` | `workdir` in the bundle; `-w` / `--container-workdir` / `workingDir` |
| `--shm-size` | docker renderer, always — docker's `/dev/shm` default is 64 MB |
| `--user`, `--group-add` | docker renderer, refusing UID 0 on a writable bind |
| `HOME`/`USER`/cache vars | docker renderer; torch calls `getpass.getuser()` at import |
| `--ipc=host` | not needed once `--shm-size` is set |
| `--rm` | **never** — it deletes the exit code `status()` reads |

On SLURM the same bundle needs none of the shm or identity flags: enroot is
rootless and exposes the host tmpfs (89 GB measured on CS-OCI-ORD). On
Kubernetes the pod template provides a `dshm` volume. That divergence is exactly
why these belong to the renderer and not to the stage.

## Finding your inputs

A stage command must never name a host path directly. Every declared input is
exported as `TAO_INPUT_<SPEC_KEY>` (upper-cased, non-alphanumerics collapsed),
and the output path as `TAO_RESULTS_ROOT`:

```sh
: "${TAO_INPUT_DATASET_DIR:?input not exported by the platform}"
n=$(ls "$TAO_INPUT_DATASET_DIR" | wc -l)
[ "$n" -gt 0 ] || { echo "read 0 inputs" >&2; exit 3; }
mkdir -p "$TAO_RESULTS_ROOT"   # a fresh volume may not have the per-job dir yet
```

**Fail closed like that.** A wrong or missing input path does not raise: the
directory is simply absent, so a stage that does not check reads nothing, writes
empty output, and exits 0 — and every downstream stage treats the empty result
as real. An end-to-end run in this repo went green exactly that way.

## Platform context

`$PLATFORM_CTX` carries what the platform needs and the bundle must not:

| platform | typical ctx |
|---|---|
| `docker` | *(none)*; add `--ctx shm_size=16g` if you hit `Bus error` |
| `slurm` | `--ctx login=… --ctx sqsh_dir=… --ctx job_dir=… --ctx account=… --ctx partition=…` |
| `kubernetes` | `--ctx namespace=… --ctx pvc_claim=… --ctx mount_path=… --ctx job_dir=… --ctx cred_secret=… --ctx uid=$(id -u) --ctx gid=$(id -g)` |

Take SLURM's `account` and `partition` from `sinfo` / `sacctmgr` at preflight —
the packaged values describe one cluster. See
`skills/platform/tao-run-on-slurm/references/slurm-preflight-storage.md`.

## What has not changed

Stage semantics, the four AnomalyGen invariants, spec authoring, and every
`commit_stage.py` call are unchanged. Only the launch mechanism moved. Each
stage reference still owns its parameters, its output layout, and its
`commit_stage.py` arguments.

On Kubernetes pass `--ctx uid`/`--ctx gid` — the renderer emits the
`securityContext` only when `uid` is present; without it stages run as the
image's default user and die writing the uid-owned results tree. Workloads
default their working directory to the stage's results mount on docker (`-w`)
and kubernetes (`workingDir`), since image WORKDIRs are commonly root-owned and
a relative-path write there fails with PermissionError; a bundle-declared
`workdir` still wins.
