# Windows container runtime mechanics

How a Windows container actually gets a GPU, what survives a mount, and the
run-time traps that apply to any Windows GPU image. Build-time traps are in
`windows-image-build.md`. Everything here is platform mechanics — no workload
configuration.

Trap identifiers (W1–W13) match the prose in SKILL.md and in the application
skill, so a symptom can be traced across both.

> **UNVERIFIED — none of this was executed.** Like the rest of
> `tao-run-on-windows-docker`, this file is transcribed from previously
> validated findings; the Windows container daemon was not available on the
> validation host, so no container here was ever started. See the VALIDATION
> STATUS section at the top of SKILL.md.

---

## 1. Isolation mode — the flag everything else depends on

A Windows container runs in one of two isolation modes:

| mode | what it is | GPU |
|---|---|---|
| `hyperv` | a lightweight utility VM per container | **none, ever** |
| `process` | a process tree on the host kernel | yes, via device class |

`--isolation=process` is **mandatory at run time**. On many hosts (notably
Windows client SKUs, and any daemon configured with `"exec-opts":
["isolation=hyperv"]`) the *default is hyperv*, and a hyperv container has no GPU
path whatsoever — `nvidia-smi` is simply absent, and the failure looks like a
broken image rather than a wrong flag.

**Build under hyperv, run under process.** Hyper-V isolation is the safer,
version-tolerant mode for a build, and the build does not need a GPU.

**Consequence — the host release must match the image base.** Process isolation
requires the container base image and the host to be the same Windows Server
release; hyperv isolation is what normally tolerates a mismatch, and it has no
GPU. So on this platform there is no fallback: an image built on Windows Server
LTSC 2025 needs an LTSC 2025 host. Mismatch reports:

```
The container operating system does not match the host operating system.
```

Reference: <https://learn.microsoft.com/virtualization/windowscontainers/deploy-containers/version-compatibility>

## 2. The GPU device class

There is no NVIDIA Container Toolkit on Windows and no `--gpus` flag. The GPU is
attached by **device-class projection**: the daemon projects the *host's* driver
libraries into the container's device namespace.

```
--device class/5B45201D-F2F2-4F3B-85BB-30FF1F953599
```

That GUID is `GUID_DEVINTERFACE_DISPLAY_ADAPTER`. It is the only argument form
that works; `--device nvidia.com/gpu=all`, `--gpus all` and a raw device path all
fail.

Three consequences that bite:

1. **Driver libraries come from the host, so they must match the host driver.**
   Never bake a driver DLL into the image (W1). A baked `nvEncodeAPI64.dll` or
   `nvcuda.dll` will be the wrong version the first time the host driver is
   updated, and the failure mode is a crash, not a version warning.
2. **The projection is partial.** `nvcuvid.dll` (NVDEC, decode) is projected;
   `nvEncodeAPI64.dll` (NVENC, encode) is **not**. A video library that probes
   for the *encoder* during initialization — even one that only ever decodes —
   dies at import with:
   ```
   RuntimeError: NVENC library file is not found
   ```
   The fix is to select a decoder that does not need NVENC. The setting name
   belongs to the application skill; the missing DLL belongs here (W1).
3. **CUDA user-mode libraries live under the projected path,** not in a
   toolkit-installed `CUDA_PATH`. Code that resolves CUDA by `CUDA_HOME` needs a
   shim (see §7).

## 3. W11 — query health is not allocation health

`torch.cuda.is_available()`, `torch.cuda.get_device_name()` and `nvidia-smi` can
all pass on a container where a **four-byte allocation segfaults**. Enumeration
goes through a different code path than allocation, so a broken projection is
invisible to every "is the GPU there?" check.

Probe with a real allocation **and** a compute op:

```python
import torch
assert torch.cuda.is_available()
x = torch.zeros(1, device="cuda")                    # allocation
a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
b = (a @ a).float().sum().item()                     # compute, forced to sync
print("GPU OK", torch.cuda.get_device_name(0), b)
```

Bake that in as `C:/gpu_check.py` at build time and run it as the first step of
any GPU container (`docker run ... $Image python C:/gpu_check.py`). Do not paste
it inline into a `RUN` or into `docker run` as `python -c` — see W7 in
`windows-image-build.md`.

## 4. W2 — the output-directory hazard

**Do not bind-mount the directory a distributed checkpoint writer writes to.**

Distributed checkpoint (DCP) writes `.metadata.tmp` and then renames it to
`.metadata`. On a Windows bind mount that rename fails:

```
[WinError 3] The system cannot find the path specified: '...\.metadata.tmp' -> '...\.metadata'
```

The damage is maximal because it happens **after** training completes: the run
prints its completion line, the GPU work is finished, and then the checkpoint is
discarded.

The pattern that works:

```powershell
# inputs: bind-mounted. output: container-internal.
docker run -d --name $Name --isolation=process --device $Class `
  -v "${HostData}:C:/data" $Image <action command writing to C:/results>

# after the run reaches a terminal state:
docker cp "${Name}:C:/results" $ResultsDir
docker rm $Name
```

`docker cp` works on a **stopped** container, so the copy can happen after exit —
but it cannot happen after `docker rm`. Never launch with `--rm`.

## 5. W3 — the memory limit kills the job at the finish line

A `--memory` limit below roughly **2x the model size** produces:

```
Done with training.
<exit code 3221225477>
```

`0xC0000005` is an access violation with no Python traceback. The cause is that
the checkpoint writer stages the whole model through **host** memory on the way
out, so peak host RSS is highest *after* the GPU work is done. A limit that
comfortably held the entire training run kills it in the last second.

Either size `--memory` at 2x model size or more, or omit it entirely. The Windows
job object does not report an OOM kill the way a cgroup does: `docker inspect`
shows `OOMKilled: false` and the raw NTSTATUS exit code.

## 6. W10 — no quota, no isolation, no scheduler

There is no cgroup equivalent, no MIG, no fractional GPU, no per-user GPU quota,
and no scheduler. Two containers started on the same Windows host both see the
whole GPU and both allocate against it until one dies. On a shared box:

```powershell
# before launching, refuse if another TAO container already holds the GPU
$busy = docker ps --filter "name=tao-" --format '{{.Names}}'
if ($busy) { throw "GPU already in use by: $busy — serialize, there is no quota on this platform." }
```

Serialization is the agent's responsibility. `--cpus` and `--memory` do apply
(they map to Windows job-object limits) but neither constrains the GPU.

## 7. W12 — vLLM serving needs a CUDA_HOME shim

If the workload serves through vLLM, it resolves CUDA by `CUDA_HOME` and fails on
a projected-driver container. Create a minimal shim directory in the image and
point `CUDA_HOME` at its **root**:

```
C:/cudashim/bin/cudart64_13.dll
```

```powershell
docker run ... -e CUDA_HOME=C:/cudashim $Image <serve command>
```

Two traps inside this one:

- `VLLM_ATTENTION_BACKEND` and `VLLM_USE_FLASHINFER_SAMPLER` do **not** avoid the
  requirement. The FlashInfer attention backend is imported during **model
  loading**, before any backend selection takes effect.
- The error message names `CUDA_LIB_PATH`. Setting `CUDA_LIB_PATH` does nothing —
  set `CUDA_HOME`, and use forward slashes.

This applies only to a vLLM serving path. Evaluation and inference that generate
through HF Transformers do not touch it.

## 8. W13 — port binding fails with WinError 10013

```
OSError: [WinError 10013] An attempt was made to access a socket in a way
forbidden by its access permissions
```

Two causes, and the message does not distinguish them:

1. Something else already holds the port.
2. The port falls inside a **Hyper-V reserved range**. Windows reserves large,
   effectively random TCP ranges for Hyper-V/WSL/containers at boot, and a
   reserved port refuses `bind()` even though nothing is listening on it.

```powershell
netsh interface ipv4 show excludedportrange protocol=tcp
```

Pick a port outside every listed range. The exclusions change on reboot, so a
port that worked yesterday can be reserved today — treat this as a startup check,
not a one-time configuration.

## 9. Mounts, paths and the shell

- **Container-side paths use forward slashes**: `C:/data`, `C:/gpu_check.py`.
  Under a bash-family shell a backslash is consumed as an escape and the argument
  arrives as a *relative* path resolved against `WORKDIR` — silently pointing at
  the wrong place instead of erroring (W8).
- **Host-side paths** in PowerShell may use either separator.
- **Do not change `WORKDIR`.** A documented action command that passes a relative
  recipe/spec path breaks if `WORKDIR` moves — and it breaks *only* that action
  while the others keep working, which makes the break look like a workload bug
  (W9).
- **SMB/UNC sources**: bind-mount a drive letter that is mapped **for the account
  the daemon runs as** (usually `LocalSystem`), not for the interactive user. A
  path that resolves in your shell can be invisible to the daemon.
- **No `/dev/shm`, no `--shm-size`, no `--ipc`.** Anything relying on POSIX shared
  memory between worker processes must be disabled by the application skill.

## 10. Process, signal and exit-code model

- **No signals.** `docker stop -t N` asks the container's init process to shut
  down and terminates the process tree when the timeout expires. There is no
  `SIGTERM` to trap, so checkpoint-on-cancel cannot be relied on.
- **`docker kill` supports only `KILL`** on Windows containers.
- **Exit codes are unsigned NTSTATUS values**, not POSIX signal arithmetic:

  | code | hex | meaning |
  |---|---|---|
  | `3221225477` | `0xC0000005` | access violation — memory limit (§5), or a GPU collective (§11) |
  | `3221225786` | `0xC000013A` | Ctrl-C / console shutdown |
  | `3221225781` | `0xC0000135` | a DLL was not found (see `windows-image-build.md`, W4) |

  There is no `137`-style "OOMKilled" code to key on.
- **Console encoding.** The Windows console default is not UTF-8. Set
  `PYTHONIOENCODING=utf-8` in the env file or non-ASCII log output either garbles
  or raises `UnicodeEncodeError` mid-run.

## 11. Why multi-GPU and multi-node are structurally impossible with stock torch

Not "untested" and not "planned". Two independent blocks:

- **PyTorch's Windows wheels have NCCL compiled out.** All NCCL code in torch is
  gated behind the `USE_C10D_NCCL` build flag, and the official Windows wheels
  are built with it **off**, so `init_process_group("nccl")` fails regardless of
  what NCCL binary is present on the host or in the image. gloo is the only
  backend that can be selected.
- **gloo cannot run any collective on a CUDA tensor on Windows.** Every such call
  dies with `0xC0000005` and no Python traceback.

Single-GPU works only because the distributed wrapper is skipped at
`world_size == 1`, and the few remaining collectives stage through host memory.
Raising `world_size` above 1 — on one host or across hosts — crashes at the first
collective. There is no configuration that avoids this, so do not attempt a
two-GPU run to "see what happens".

**Do not write "NCCL has no Windows build" — it is false and it invites a
correction that loses the real reason.** A community port of NCCL for Windows
exists and is actively maintained (`github.com/SystemPanic/nccl-windows`); vLLM's
Windows builds require building it and then load `nccl.dll` directly through
`VLLM_NCCL_SO_PATH`, which works precisely because vLLM bypasses torch's process
group. Nothing in that path helps a torch-distributed workload: torch itself
would have to be rebuilt from source with `USE_C10D_NCCL=1`, inside a Windows
container image, and gloo's CUDA-tensor limitation would still apply to any
fallback. **UNTESTED — the accurate claim is "impossible with stock torch
wheels", and no one should plan a multi-GPU run on the port.**
