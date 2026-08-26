# Running a Cosmos3 stage on any platform

Container stages run through the platform contract. Host-side stages do not —
see *Which stages are containers* below.

```bash
BANK="${TAO_SKILL_BANK_PATH:?}"
C3="$BANK/skills/applications/tao-run-deft-aoi-cosmos3/scripts"

# 1. Emit the stage. Command, mode, config_format and image are RESOLVED from
#    tao-finetune-cosmos-reason -- not restated here.
python3 "$C3/stage_bundle.py" train \
  --results-dir "$RUN_DIR" \
  --spec-file "$RUN_DIR/train_spec.yaml" \
  --param workspace="$WORKSPACE_DIR" \
  --param annotations="$RUN_DIR/annotations/train_iter1.json" \
  > "$RUN_DIR/train.bundle.json"

# 2. Launch and record it.
JOB=$(python3 "$C3/deft_exec.py" --state "$STATE" --submit \
        --bundle "$RUN_DIR/train.bundle.json" \
        --platform "$PLATFORM" $PLATFORM_CTX)

# 3. Poll to a terminal state, then read logs if it failed.
python3 "$C3/deft_exec.py" --state "$STATE" --await-job "$JOB" $PLATFORM_CTX
python3 "$C3/deft_exec.py" --state "$STATE" --logs "$JOB" --tail 100 $PLATFORM_CTX
```

`--list` prints the stage table, each stage's required `--param` names, and the
host-side stages with the script that owns each:

```bash
python3 "$C3/stage_bundle.py" --list
```

## Which stages are containers

Only four families. The rest of the loop is bundled Python that runs on the
host and needs no platform at all:

| Stage | Runs |
|---|---|
| `train` | container — `tao-finetune-cosmos-reason` train |
| `evaluate_proxy`, `evaluate_benchmark` | container — the same `evaluate` action, different annotations |
| `anomalygen.amp`, `anomalygen.sdg` | container — `paidf-anomalygen` |
| `data_mining.*` | container — TAO data-services |
| `proxy_rcca`, `benchmark_metrics` | host — `scripts/analyze_gaps.py` |
| `routing` | host — bundled routing over the Proxy gaps |
| `assemble_data`, `validate_data` | host — `assemble_training_json.py`, `validate_sharegpt.py` |
| `loop_stop` | host — `scripts/finalize_run.py` |

`stage_bundle.py` refuses a host-side stage by name and says which script owns
it, rather than reporting an unknown stage.

## Why the commands are not written down here

They are read from `tao-finetune-cosmos-reason/references/skill_info.yaml` at
build time. That skill owns them, and a second copy drifts:

- this reference used to state the train command as
  `cosmos-rl --config {config_path} /opt/cosmos_rl/tao_sft_example.py`
- the model skill actually computes the hook path from `cosmos_rl.__file__`,
  landing at `…/tools/custom_hooks/tao_sft_example.py`, and guards it with
  `test -f` first

Those are different files. A run following the old text passed cosmos-rl a
script that does not exist. The same failure mode hit DEFT AOI, whose table
carried `visual_changenet classify train` — a subcommand that does not exist —
because it copied the command instead of resolving it.

The image resolves the same way, from the model skill's `backend_contracts`
entry for the selected backend (`cosmos-rl` by default; `--backend` overrides).

## The workspace must appear at `/tao-workspace`

Every path in a cosmos-rl spec is under that root, and the packaged template
warns: **NEVER mount over `/workspace`**, where cosmos-rl itself is installed.
The `workspace` input therefore declares `target: /tao-workspace`, and each
platform binds it there — `-v src:/tao-workspace:ro` on docker,
`--container-mounts=src:/tao-workspace:ro` on SLURM, and a second `volumeMount`
of the same claim with a `subPath` on Kubernetes.

Mounting it anywhere else does **not** fail: the spec's paths simply do not
resolve, and the stage dies inside the workload with a missing-file error that
names the spec path rather than the mount.

## Host identity

Do not paste `--user`/`-e USER`/`/etc/passwd` flags into a stage. The docker
renderer emits them, refuses to LAUNCH as UID 0 for a writable bind, and
redirects `HOME` and the framework caches onto the results mount. SLURM needs
none of it (enroot is rootless); Kubernetes uses a `securityContext`.

This is the same reason the flags left the AOI stage references: a pasted
runtime pins the loop to one platform, and the renderer already supplies what
that platform needs.

## Platform context

| platform | typical ctx |
|---|---|
| `docker` | *(none)*; `--ctx shm_size=16g` if you hit `Bus error` |
| `slurm` | `--ctx login=… --ctx sqsh_dir=… --ctx job_dir=… --ctx account=… --ctx partition=…` |
| `kubernetes` | `--ctx namespace=… --ctx pvc_claim=… --ctx mount_path=… --ctx job_dir=…` |

On Kubernetes `job_dir` must be under `mount_path`: a config-mode stage writes
its TOML spec there and the pod reads it through the single bound volume, so a
`job_dir` outside it is refused at render time.
