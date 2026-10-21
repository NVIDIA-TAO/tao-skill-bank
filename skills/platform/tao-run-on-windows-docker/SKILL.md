---
name: tao-run-on-windows-docker
description: >-
  Windows-container execution platform that runs TAO GPU containers on a
  Windows Server host through the docker CLI, using process isolation and a
  GPU device class instead of the NVIDIA Container Toolkit. Single node and
  single GPU only with stock torch wheels. Use when the workload must run on
  Windows rather than Linux. Verified on a real Windows-container daemon,
  including the GPU device class, four verbs, and vLLM serving.
license: Apache-2.0
compatibility: Requires a Windows Server host on the same Windows Server release as the image base, an NVIDIA display driver installed on the host, and a Docker daemon in Windows-container mode reachable over the local named pipe. Process isolation must be available — Hyper-V isolation has no GPU path. GPU is attached with a Windows device class; --gpus, --shm-size and --user do not exist for Windows containers. No nvidia-tao-sdk — jobs run with the plain docker CLI.
metadata:
  author: NVIDIA Corporation
  version: "0.1.1"
allowed-tools: Read Bash
tags:
- windows
- windows-containers
- docker
- gpu
- compute
- container
---

# Windows Docker (Windows containers)

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

## Status

Executed against a real Windows-container daemon (moby 29.7.1, `OSType: windows`,
default isolation `hyperv`) on an RTX PRO 6000 host:

- `--isolation=process` + the device class gives the container a working GPU —
  `GPU_OK`, cuDNN 92000, bf16 matmul. Dropping **only** `--device` on the same
  image yields `NO GPU VISIBLE`, so the device class is load-bearing, not decorative.
- The image's own build check passes (`TAO_BUILD_CHECK_OK`), including all five
  `cosmos-framework-*` entrypoints and both Windows source patches.
- The four verbs behave as the status table below says: `running 0` → RUNNING,
  `exited 0` → COMPLETE, and `docker logs` still works **after** exit, which is
  the entire reason `--rm` is omitted on submit.
- vLLM serves a bind-mounted exported checkpoint and answers
  `/v1/chat/completions` over a published port (~53 GB resident, released on
  `docker rm -f`).

Not exercised here: multi-node, `DOCKER_HOST` over TCP/SSH to a remote Windows
daemon, and image **building** (the image was already present locally).

The individual W-numbered traps in `references/` are inherited from prior
findings and are not each independently re-triggered here; treat them as
documented failure modes rather than as reproduced ones.

The sibling **`tao-run-on-windows-venv`** covers the same five actions with no
daemon, no image and no device class. If a Windows container is not a hard
requirement, it is the lower-overhead path.

Runs a TAO GPU container on a **Windows** host, as a **Windows container** — a
process tree on the host Windows kernel, not a Linux VM. Every example below is
PowerShell.

This skill owns **execution mechanics only**: how a Windows container gets a GPU,
what mounts survive, how the four verbs map onto the docker CLI, and the
build/runtime traps that apply to *any* Windows GPU image. It carries no
workload-specific configuration. The image tag, entrypoints, training overrides,
and model-level traps belong to the selected model skill. For Cosmos3, read
`skills/models/tao-finetune-cosmos-reason/references/cosmos3-windows-contract.toml`
for the image URI (`[image].uri`) and run flags
(`[runtime].required_run_flags`). Read the tag from there; do not hardcode it.

## What is different from Linux Docker

Do not translate a Linux run line by search-and-replace. These flags are not
"optional" on a Windows container — passing one is an error, not a no-op.

| Linux | Windows containers |
|---|---|
| `--gpus all` | **does not exist.** GPU arrives via `--device class/<GUID>` |
| NVIDIA Container Toolkit | not used; the driver libraries are projected by the device class |
| `--shm-size` | **does not exist.** No `/dev/shm` — dataloader workers must be configured accordingly by the application skill |
| `--user`, uid/gid, rootless | **do not exist.** No user mapping of any kind |
| default isolation | **`hyperv`, which has NO GPU path.** `--isolation=process` is mandatory at run time |
| bind-mount the output dir | **breaks after the run completes** (see [Results extraction](#results-extraction)) |
| cgroups / MIG partitioning | none: no per-user GPU quota, no isolation, no scheduler |
| multi-GPU / multi-node | impossible with stock torch wheels (see [Limits](#limits)) |

## Quick Start

```powershell
$env:DOCKER_HOST = 'npipe:////./pipe/docker_engine'   # the local Windows daemon
$Image = 'nvcr.io/<org>/<image>:<tag>'                  # from [image].uri in the application contract
$Class = 'class/5B45201D-F2F2-4F3B-85BB-30FF1F953599'

docker version --format '{{.Server.Os}}'              # must print: windows

# GPU smoke — process isolation + device class, nothing else.
docker run --rm --isolation=process --device $Class $Image `
  powershell -Command "nvidia-smi"
```

`nvidia-smi` passing is **not** proof the GPU is usable — see W11 in
`references/windows-container-runtime.md`. Probe with a real allocation.

## Preflight

Four checks. Each one fails a different way and the first masks the rest.

```powershell
# 1. Docker CLI present, pointed at the LOCAL Windows daemon.
$env:DOCKER_HOST = 'npipe:////./pipe/docker_engine'
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  throw "MISSING: docker CLI not on PATH."
}

# 2. Daemon is in WINDOWS-container mode. 'linux' here means every later step
#    fails with 'no matching manifest for linux/amd64'.
$serverOs = docker version --format '{{.Server.Os}}'
if ($serverOs -ne 'windows') {
  throw "MISSING: daemon is in $serverOs-container mode. Switch to Windows containers: & `"$Env:ProgramFiles\Docker\Docker\DockerCli.exe`" -SwitchWindowsEngine"
}

# 3. Host NVIDIA driver present (the device class projects the HOST driver
#    libraries into the container; there is no in-container driver).
nvidia-smi.exe --query-gpu=name,driver_version,memory.total --format=csv,noheader
if ($LASTEXITCODE -ne 0) { throw "MISSING: no host NVIDIA driver / nvidia-smi." }

# 4. Process isolation actually works with the device class attached.
docker run --rm --isolation=process --device class/5B45201D-F2F2-4F3B-85BB-30FF1F953599 `
  $Image powershell -Command "nvidia-smi"
if ($LASTEXITCODE -ne 0) { throw "GPU device class not usable — see references/windows-troubleshooting.md" }
```

**Host/image release must match.** Process isolation requires the container base
image and the host to be the same Windows Server release. Hyper-V isolation is
what normally papers over a mismatch — and Hyper-V has no GPU path, so on this
platform there is no fallback. A mismatch reports *"The container operating
system does not match the host operating system."*

## Credentials & configuration

- **NGC** — for `nvcr.io` pulls. Feed the key on **stdin**; `--password` puts the
  secret in argv, where it is visible in the process table and in PowerShell
  history:
  ```powershell
  $env:NGC_KEY | docker login nvcr.io --username '$oauthtoken' --password-stdin
  ```
- **DOCKER_HOST** (optional) — `npipe:////./pipe/docker_engine` is the local
  Windows daemon and is the default. Set it explicitly when a stale
  `tcp://`/Linux-engine value is inherited from the shell profile.
- **Workload env** (`HF_TOKEN`, TAO job/results env, …) — defined by the
  application skill, passed with `-e NAME=value` or `--env-file`. Prefer
  `--env-file` so no secret reaches argv.

Do not ask for Brev, SLURM or kubeconfig credentials for a Windows-container run.
There is no S3 sidecar on this platform, so do not ask for S3 credentials either:
inputs are host paths (see below). Before launch, verify the host input path
exists **on the daemon host** and that the image tag resolves.

## Storage — tier A only

**Tier A** (a host directory bind-mounted into the container) is the only
supported tier here. There is no init-container fetch step and no in-container
object-store client on the Windows image, so data must already be on a local or
SMB-mounted host path.

Mount **inputs** read-write or read-only as you like. Do **not** mount the
**output** directory — see the next section.

## Results extraction

Keep the container's output directory **container-internal** and copy it out
after the run:

```powershell
docker cp "${Name}:C:/results" $ResultsDir
```

Distributed-checkpoint writers write `<name>.tmp` and then rename it. On a
Windows bind mount that rename fails with `WinError 3`, and it fails **after
training has already completed** — the work is done and then thrown away. This
is W2 in `references/windows-container-runtime.md`.

`--results-dir` recorded at `open` is therefore the **host** directory that
`docker cp` writes into, not a mount visible to the container.

## Execution — the four verbs

`tao-run-on-windows-docker` is a platform **consumer**: it runs a spec bundle via
the docker CLI, mutating only the job record. There is no `tao_sdk` import and no
in-container agent. `${BANK}` = `${TAO_SKILL_BANK_PATH}`.

### submit

1. **Preflight gates first** — daemon in Windows-container mode, host driver
   present, device class usable. All three are hard-fails; a Windows container
   without a GPU starts happily and fails minutes later inside the workload.
2. **Memory gate.** A `--memory` limit below roughly **2x the model size** kills
   the process with `0xC0000005` *after* the run prints its completion line,
   because the checkpoint is staged through host memory on the way out. Either
   size the limit generously or omit `--memory` entirely (W3).
3. **Serialize yourself.** There is no cgroup or MIG equivalent and no
   scheduler: two concurrent GPU containers on one Windows host simply fight over
   VRAM. Check for a running TAO container before launching another (W10).
4. **Open the record — mints the id, binds `results_dir`, before launch:**
   ```powershell
   $JobId = python "${BANK}/scripts/tao_job_record.py" open --platform windows-docker --image $Image --network-arch $Arch --action $Action --storage-tier A --results-dir $ResultsDir
   ```
5. **Launch detached, named after the job id.** The container name is the backend
   handle for all three remaining verbs:
   ```powershell
   $Name = "tao-$JobId"
   docker run -d --name $Name --isolation=process --device $Class `
     -v "${HostData}:C:/data" --env-file $EnvFile `
     $Image <the application skill's action command>
   ```
   Container-side paths use **forward slashes** (`C:/data`). Under a bash-family
   shell a backslash is eaten and the argument arrives relative to `WORKDIR`
   (W8). Do not change `WORKDIR` (W9).
6. **Record RUNNING:**
   ```powershell
   python "${BANK}/scripts/tao_job_record.py" mark $JobId --state RUNNING --backend-ref $Name
   ```

A submit that skipped the gates or the `open` has no id — so it cannot launch.

### status

```powershell
docker inspect -f '{{.State.Status}} {{.State.ExitCode}} {{.State.OOMKilled}}' $Name
```

| docker signal | vocab |
|---|---|
| `created`, image still pulling | `PENDING` (a first Windows base-layer pull is measured in GB and minutes) |
| `running` | `RUNNING` |
| `exited`, exit code `0` | `COMPLETE` |
| `exited`, exit code `3221225477` (`0xC0000005`) | `ERROR` / `ERR_INFRA` — access violation, no traceback. Almost always the memory limit (W3) or a GPU collective (see [Limits](#limits)) |
| `exited`, exit code `3221225786` (`0xC000013A`) | `CANCELED` — Ctrl-C / console shutdown |
| `exited`, any other non-zero | `ERROR` — classify from `logs` |
| `No such object` | `UNKNOWN` (container was removed; the job record is the source of truth) |

Windows exit codes are unsigned 32-bit NTSTATUS values, so a crash shows up as a
large positive number, never as `137`-style POSIX signal arithmetic.

### logs

```powershell
docker logs --tail 200 $Name
```

Add `-f` to follow. If the log shows mojibake or the workload dies with
`UnicodeEncodeError`, set `PYTHONIOENCODING=utf-8` in the env file — the Windows
console default is not UTF-8.

### cancel

**Copy results out before removing the container** — the output directory is
container-internal, so `docker rm` destroys any partial results:

```powershell
docker cp "${Name}:C:/results" $ResultsDir   # optional, but irreversible if skipped
docker stop -t 60 $Name
docker rm $Name
python "${BANK}/scripts/tao_job_record.py" mark $JobId --state CANCELED --source agent
```

Windows containers have no signal model: `docker stop` asks the init process to
shut down and then terminates the process tree when the timeout expires. There is
no `SIGTERM` handler to catch, so a graceful in-process checkpoint-on-exit cannot
be relied on.

## Limits

**Multi-node is not supported** — and neither is multi-GPU. With **stock torch
wheels** this is structural, not "untested", for two independent reasons:

1. **gloo cannot run any collective on a CUDA tensor on Windows.** Every such
   call dies with `0xC0000005` and no Python traceback.
2. **PyTorch gates all NCCL code behind the `USE_C10D_NCCL` build flag, which is
   off in the official Windows wheels.** `init_process_group("nccl")` therefore
   fails no matter what NCCL binary is present, which leaves gloo as the only
   selectable backend — and see (1).

Single-GPU works only because the distributed wrapper is skipped at
`world_size == 1` and the remaining collectives stage through host memory. Do not
"just try two GPUs" — `max_gpus = 1`, `max_nodes = 1`.

*Do not restate this as "NCCL has no Windows build."* A maintained community
port exists (`github.com/SystemPanic/nccl-windows`), and vLLM's Windows builds
load its `nccl.dll` directly via `VLLM_NCCL_SO_PATH`, bypassing
`torch.distributed` altogether. That does not lift the limit here: torch would
have to be rebuilt from source with `USE_C10D_NCCL=1`, inside a Windows
container image. **UNTESTED — do not plan a multi-GPU run on it.**

Other standing limits: no GPU quota or isolation between containers on a shared
host (W10); no `/dev/shm`; no user mapping; Hyper-V isolation has no GPU path at
all.

## Common error patterns

Full error → cause → fix table: `references/windows-troubleshooting.md`.

The four that account for most first-run failures:

- **`no matching manifest for windows/amd64`** or **`for linux/amd64`** — the
  daemon is in the wrong container mode. Preflight check 2.
- **`The container operating system does not match the host operating system`** —
  host release differs from the image base, and process isolation cannot bridge
  it. Match the host release; there is no Hyper-V fallback because Hyper-V has no
  GPU.
- **`WinError 126` loading `c10.dll`** — Server Core has no Visual C++ runtime.
  A build-time fix, not a run-time one: `references/windows-image-build.md` (W4).
- **`RuntimeError: NVENC library file is not found`** — a decoder that probes for
  the NVENC *encoder* at init. The device class projects `nvcuvid.dll` (NVDEC)
  but not `nvEncodeAPI64.dll`. Select a decoder that does not need the encoder
  (the application skill names the setting) and **never bake a driver DLL into
  the image** — driver libraries must match the host driver (W1).

## What this skill does NOT support

- **Multi-GPU or multi-node.** See [Limits](#limits). Structural with stock torch
  wheels, not a gap someone forgot to fill.
- **Linux containers.** A Linux-mode daemon cannot run this image and vice versa;
  one daemon serves one mode at a time.
- **Object-store staging (tiers B/C).** Tier A host paths only.
- **GPU quotas, fractional GPUs, MIG.** No Windows equivalent exists.
- **Building the image.** `references/windows-image-build.md` documents the
  build-time traps, but the image is produced by the application's release
  pipeline, not by this skill.

## References

- `references/windows-container-runtime.md` — isolation, device class, mounts,
  memory, ports, GPU probing, the output hazard (W1, W2, W3, W10, W11, W12, W13).
- `references/windows-image-build.md` — build-time traps that apply to any
  Windows GPU image (W4–W9).
- `references/windows-troubleshooting.md` — error → cause → fix.
- `references/skill_info.yaml` — machine-readable platform metadata.
- Windows container version compatibility:
  <https://learn.microsoft.com/virtualization/windowscontainers/deploy-containers/version-compatibility>
- Windows container isolation modes:
  <https://learn.microsoft.com/virtualization/windowscontainers/manage-containers/hyperv-container>
