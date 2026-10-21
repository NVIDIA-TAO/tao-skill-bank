# Windows containers — error, cause, fix

Look the message up here before changing the workload configuration. Most
first-run failures on this platform are host or flag problems wearing a
workload's error message.

`W*` identifiers cross-reference `windows-container-runtime.md` (run time) and
`windows-image-build.md` (build time). Anything about the *workload's* own
settings belongs to the application skill, not here.

> **UNVERIFIED — none of these messages was reproduced by us.** They are
> transcribed from previously validated findings; the Windows container daemon
> was not available on the validation host, so no command in this skill has been
> executed. See the VALIDATION STATUS section at the top of SKILL.md.

---

## Daemon and image

| Message | Cause | Fix |
|---|---|---|
| `no matching manifest for linux/amd64 in the manifest list entries` | The daemon is in **Linux**-container mode and is trying to pull a Windows image. | Switch the engine: `& "$Env:ProgramFiles\Docker\Docker\DockerCli.exe" -SwitchWindowsEngine`. Verify with `docker version --format '{{.Server.Os}}'` → `windows`. |
| `no matching manifest for windows/amd64` | The reverse: a Windows-mode daemon pulling a Linux-only image. | Confirm you are pulling the Windows image tag from the application contract (`[image].uri`). |
| `The container operating system does not match the host operating system` | Process isolation requires the image base and host to be the same Windows Server release. | Run on a host of the matching release. There is **no** Hyper-V fallback here — Hyper-V isolation has no GPU path. |
| `error during connect ... open //./pipe/docker_engine: The system cannot find the file specified` | Daemon not running, or `DOCKER_HOST` points somewhere else. | `$env:DOCKER_HOST = 'npipe:////./pipe/docker_engine'` (the local Windows daemon), then start the Docker service. |
| `unauthorized: authentication required` on `nvcr.io` | Not logged in, or the credential helper holds a stale token. | `$env:NGC_KEY \| docker login nvcr.io --username '$oauthtoken' --password-stdin`. Never pass the key with `--password` — argv is visible in the process table. |
| Pull sits for many minutes with no progress | Normal: Windows base layers are multi-GB. | Pre-pull the image before the first `submit`; report the job as `PENDING` while `docker inspect` shows `created`. |

## GPU attachment

| Message | Cause | Fix |
|---|---|---|
| `nvidia-smi` not found inside the container, GPU absent | Container ran under **hyperv** isolation (often the daemon default). Hyper-V has **no GPU path at all**. | Add `--isolation=process`. W1/§1. |
| `unknown flag: --gpus` / `--gpus is not supported on Windows` | `--gpus` does not exist for Windows containers. | Use `--device class/5B45201D-F2F2-4F3B-85BB-30FF1F953599`. |
| `invalid device request` / device not found | Wrong device argument form (`nvidia.com/gpu=all`, a raw device path). | Only the `class/<GUID>` form works. |
| `nvidia-smi` prints the GPU but a tiny CUDA allocation segfaults | Enumeration and allocation take different paths. Query health is not allocation health. | Probe with a real allocation plus a compute op (`C:/gpu_check.py`). W11. |
| `RuntimeError: NVENC library file is not found` | A video library probes for the NVENC **encoder** at init even though it only decodes. The device class projects `nvcuvid.dll` (NVDEC) but **not** `nvEncodeAPI64.dll`. | Select a decoder that does not require NVENC — the application skill names the setting. Do **not** bake the encoder DLL into the image; driver libraries must match the host driver. W1. |
| `nvidia-smi` fails on the **host** | No host driver, or the display adapter is not exposed. | Install the NVIDIA display driver on the host. There is no in-container driver on this platform. |

## Crashes and exit codes

| Signal | Cause | Fix |
|---|---|---|
| Exit `3221225477` (`0xC0000005`) right after a completion message | Access violation while the checkpoint is staged through **host** memory. `--memory` below ~2x model size. `docker inspect` reports `OOMKilled: false`. | Raise `--memory` to 2x model size or omit it. W3. |
| Exit `3221225477` at the first distributed call | gloo running a collective on a CUDA tensor. gloo is the only selectable backend because torch's Windows wheels are built with `USE_C10D_NCCL` **off** — not because NCCL has no Windows build (a community port exists; see §11). | Nothing to configure — with stock torch wheels multi-GPU and multi-node are structurally impossible. Run `world_size = 1`. See `windows-container-runtime.md` §11. |
| Exit `3221225781` (`0xC0000135`) | `STATUS_DLL_NOT_FOUND` — a dependent DLL is missing at process start. | Usually the VC++ runtime (W4) or an unregistered DLL directory. See `windows-image-build.md`. |
| Exit `3221225786` (`0xC000013A`) | Ctrl-C / console shutdown. | Expected on cancel; record `CANCELED`, not `ERROR`. |
| Process dies with no traceback at all | Native crash — Python never got control. | Do not look for a Python bug first; check the memory limit and the collective path above. |

## DLLs and imports

| Message | Cause | Fix |
|---|---|---|
| `[WinError 126] The specified module could not be found. Error loading "...\c10.dll" or one of its dependencies` | The named DLL exists; its **dependency**, the Visual C++ runtime, does not. Server Core ships without it. | `COPY vc_redist.x64.exe` into the build context and run it `/quiet`. Copy, never download — the CA store is empty until certifi is wired. W4. |
| `ImportError: DLL load failed while importing <ext>` from a package that installed cleanly | Since Python 3.8, Windows extension modules do **not** resolve dependent DLLs from `PATH`. | Register the directories with `os.add_dll_directory()` at interpreter startup, via a one-line `.pth` in `site-packages`. See `windows-image-build.md`. |
| A media/codec extension imports but cannot open files | Missing or wrong FFmpeg build. "Essentials" builds contain **no** DLLs. | Install **shared** FFmpeg builds and register their `bin` directories in the same `.pth`. |

## Filesystem and mounts

| Message | Cause | Fix |
|---|---|---|
| `[WinError 3] The system cannot find the path specified: '...\.metadata.tmp' -> '...\.metadata'`, **after** the run completed | Distributed checkpoint writes a temp file then renames it; that rename fails on a Windows bind mount. | Keep the output directory **container-internal** and `docker cp` it out after the run. Never launch with `--rm`. W2. |
| `pip` dies mid-install with an `OSError` naming a file that exists | `MAX_PATH` 260-character limit. | Set `LongPathsEnabled=1` **before** any pip step and rebuild from that layer. W5. |
| A container path resolves to the wrong place, no error | A backslash in a `docker` argument was eaten by a bash-family shell; the argument arrived relative to `WORKDIR`. | Forward slashes for all container-side paths (`C:/data`). W8. |
| Training breaks while evaluate and inference still work after an image change | `WORKDIR` moved; the training command passes a **relative** recipe path. | Restore the contract's `WORKDIR` (`[runtime].workdir`). W9. |
| A mounted host path is empty inside the container | The drive/UNC mapping exists for the interactive user, not for the account the daemon runs as. | Bind-mount a local path, or map the share for the daemon's service account. |

## Build

| Symptom | Cause | Fix |
|---|---|---|
| Build reports success, image is broken | `$ErrorActionPreference` does not apply to **native executables**, so a failed `RUN` does not fail a Windows build. | Check `$LASTEXITCODE` after every native command and `throw`. W6. |
| `SyntaxError` about an unterminated string from an inline `python -c` in a `RUN` — or worse, it "succeeds" and does the wrong thing | PowerShell mangles quotes and semicolons before the interpreter sees them. | Put the code in a `.py`/`.pth` file and `COPY` it. W7. |
| Build fails only on a GPU-less builder | The build was attempted under process isolation on a mismatched host. | Build with `--isolation=hyperv` (no GPU needed); the **run** uses process isolation. |

## Networking

| Message | Cause | Fix |
|---|---|---|
| `OSError: [WinError 10013] An attempt was made to access a socket in a way forbidden by its access permissions` | The port is taken, **or** it falls inside a Hyper-V reserved range — reserved ports refuse `bind()` even with nothing listening. | `netsh interface ipv4 show excludedportrange protocol=tcp`, then pick a port outside every listed range. Exclusions change on reboot, so re-check at startup. W13. |
| vLLM serving fails during model load naming `CUDA_LIB_PATH` | The FlashInfer attention backend is imported during **model loading**; `VLLM_ATTENTION_BACKEND` and `VLLM_USE_FLASHINFER_SAMPLER` are too late to help. | Set `CUDA_HOME=C:/cudashim` (forward slashes) with `C:/cudashim/bin/cudart64_13.dll` present. Set `CUDA_HOME`, **not** the `CUDA_LIB_PATH` the error names. W12. |
| Published port unreachable from another machine | Windows container NAT plus the host firewall. | Publish with `-p`, then open the port in Windows Firewall on the host. |

## Logs and output

| Symptom | Cause | Fix |
|---|---|---|
| Garbled non-ASCII in `docker logs`, or `UnicodeEncodeError` mid-run | The Windows console default encoding is not UTF-8. | Set `PYTHONIOENCODING=utf-8` in the env file. |
| `docker logs` empty for a container that is clearly working | Output buffered, or written to a file rather than stdout. | Add `-u` / `PYTHONUNBUFFERED=1`; check for a log file inside the container with `docker exec`. |
| Results missing after a successful run | The container was removed before `docker cp`, or the output was bind-mounted (W2). | Copy first, remove second. `docker cp` works on a stopped container but not on a removed one. |

## When nothing above matches

Collect, in this order, before touching the workload configuration:

```powershell
docker version --format '{{.Server.Os}} {{.Server.Version}}'
[System.Environment]::OSVersion.Version                     # host build
nvidia-smi.exe --query-gpu=name,driver_version --format=csv,noheader
docker inspect -f '{{.State.Status}} {{.State.ExitCode}} {{.HostConfig.Isolation}} {{.HostConfig.Memory}}' $Name
docker logs --tail 200 $Name
```

An exit code in the `3221225xxx` range with an empty Python traceback is a
**platform** failure. A Python traceback naming the workload's own modules is an
**application** failure — take it to the application skill.
