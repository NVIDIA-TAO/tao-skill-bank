# Building a Windows GPU container image

Build-time traps W4–W9. They apply to **any** Windows GPU image, not to one
workload — the same six mistakes break a PyTorch image, a TensorRT image, or a
plain diagnostic image.

This skill does not build the shipped image; that comes from the application's
release pipeline. Read this when a build fails, when you are reproducing the
image locally, or when you are adding a layer on top of it.

**Build under `hyperv` isolation, run under `process`.** The build needs no GPU
and Hyper-V isolation is version-tolerant, which removes an entire class of
"container OS does not match host OS" failures from the build. The *run* must be
`--isolation=process` because that is the only mode with a GPU path.

> **UNVERIFIED — no build in this file was executed.** Transcribed from
> previously validated findings; the Windows container daemon was not available
> on the validation host. See the VALIDATION STATUS section at the top of
> SKILL.md.

---

## W6 — a failed `RUN` does not fail a Windows build

This is the first trap because it hides all the others.

In a Windows `RUN` (which executes under PowerShell), `$ErrorActionPreference`
governs **cmdlets only**. It has no effect on **native executables**. A native
command that exits non-zero prints its error, PowerShell continues to the next
line, the `RUN` layer exits 0, and the build reports success. You get a green
build and an image with a half-installed dependency.

Check `$LASTEXITCODE` after **every** native command:

```dockerfile
SHELL ["powershell", "-NoProfile", "-Command", "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue';"]

RUN pip install --no-cache-dir -r C:/requirements.txt; `
    if ($LASTEXITCODE -ne 0) { throw "pip install failed with $LASTEXITCODE" }
```

Setting `$ErrorActionPreference='Stop'` in `SHELL` is still worth doing — it
catches the cmdlet half — but it is not a substitute for the `$LASTEXITCODE`
check. Every `pip`, `curl`, `msiexec`, `7z`, `git` and `python` invocation needs
one.

## W5 — enable long paths before the first `pip`

The Windows default `MAX_PATH` is 260 characters. A deep dependency tree blows
through it and `pip` dies partway through an install with a truncated,
misleading error — often an `OSError` naming a file that plainly exists.

Set the registry key **before any pip step**, in its own early layer:

```dockerfile
RUN New-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' `
      -Name 'LongPathsEnabled' -Value 1 -PropertyType DWORD -Force | Out-Null
```

Retrofitting it after a failed install does not repair the partially-installed
site-packages tree; rebuild from that layer.

## W4 — Server Core has no Visual C++ runtime

Server Core ships without the Visual C++ redistributable, so the first native
extension module fails to load:

```
OSError: [WinError 126] The specified module could not be found.
  Error loading "...\torch\lib\c10.dll" or one of its dependencies.
```

The named DLL exists. Its *dependency* — the VC++ runtime — does not.

**COPY the installer into the build context; do not download it.** The base
image's certificate store is effectively empty until `certifi` (or an explicit
root-certificate import) is wired up, so an `Invoke-WebRequest` to the Microsoft
CDN fails TLS validation inside the build:

```dockerfile
COPY vc_redist.x64.exe C:/tmp/vc_redist.x64.exe
RUN Start-Process -FilePath C:/tmp/vc_redist.x64.exe `
      -ArgumentList '/quiet','/norestart' -Wait -PassThru | `
      ForEach-Object { if ($_.ExitCode -ne 0) { throw "vc_redist failed: $($_.ExitCode)" } }; `
    Remove-Item C:/tmp/vc_redist.x64.exe
```

`WinError 126` after this is installed means a *different* missing dependency —
usually a DLL search-path problem, covered in the next section.

## Dependent-DLL resolution — the `.pth` that must exist

Since Python 3.8, Windows extension modules **do not** resolve dependent DLLs
from `PATH`. `os.add_dll_directory()` is the only mechanism, and it must run at
**interpreter startup**, before any import that triggers a load.

A one-line `.pth` file dropped into `site-packages` does exactly that (Python
executes lines in a `.pth` that begin with `import`):

```
import os; [os.add_dll_directory(p) for p in (r"C:\ffmpeg-8.1\bin", r"C:\ffmpeg-7.1\bin") if os.path.isdir(p)]
```

Write it as a file and `COPY` it in (see W7). Symptoms of a missing one:
`WinError 126` or a bare `ImportError: DLL load failed while importing <ext>`
from a package that installed cleanly.

## W7 — no inline `python -c` in a `RUN`

PowerShell mangles quotes and semicolons before the interpreter sees them. An
inline `python -c "import x; print(x.__file__)"` inside a `RUN` becomes something
that either fails with a `SyntaxError` about an unterminated string, or — worse —
parses into a *different* program that succeeds and does the wrong thing.

Put the code in a `.py` (or `.pth`) file and `COPY` it:

```dockerfile
COPY gpu_check.py C:/gpu_check.py
COPY dll_dirs.pth C:/Python312/Lib/site-packages/dll_dirs.pth
RUN python C:/gpu_check.py; if ($LASTEXITCODE -ne 0) { throw "gpu probe failed" }
```

The same applies to `docker run ... python -c "..."` from a bash-family shell at
run time.

## W8 — forward slashes for container paths

Under a bash-family shell a backslash in a `docker` argument is consumed as an
escape, and the surviving text arrives as a **relative** path resolved against
`WORKDIR`. It does not error — it silently points somewhere else.

```
docker run ... $Image python C:/gpu_check.py     # correct
docker run ... $Image python C:\gpu_check.py     # arrives as "Cgpu_check.py"
```

Use forward slashes for every container-side path, in `COPY` destinations, in
`WORKDIR`, in `ENV`, and in run arguments. Host-side paths in PowerShell may use
either separator.

## W9 — do not change `WORKDIR`

The documented action commands pass **relative** spec/recipe paths. Moving
`WORKDIR` breaks exactly those actions and leaves every other action working, so
the failure presents as "training is broken" rather than "the image changed".
Keep the `WORKDIR` the application contract specifies (`[runtime].workdir`).

## Never bake driver libraries

The GPU device class projects the **host's** driver libraries into the container.
Baking a driver DLL (`nvcuda.dll`, `nvEncodeAPI64.dll`, `nvml.dll`, …) into the
image creates a version skew the first time the host driver is updated, and the
resulting failure is a crash rather than a version error.

If a library is missing at run time (e.g. NVENC — see
`windows-container-runtime.md` §2), the answer is to stop requiring it, not to
ship it.

## Build checklist

1. `SHELL` set to PowerShell with `$ErrorActionPreference='Stop'` **and** an
   explicit `$LASTEXITCODE` check after every native command (W6).
2. `LongPathsEnabled=1` in an early layer, before any `pip` (W5).
3. `vc_redist.x64.exe` **copied** into the context and run `/quiet` (W4).
4. A `.pth` in `site-packages` registering DLL directories via
   `os.add_dll_directory()`.
5. No inline `python -c`; code lives in `COPY`'d files (W7).
6. Forward slashes everywhere on the container side (W8).
7. `WORKDIR` unchanged from the contract (W9).
8. No driver DLLs baked in.
9. Build with `--isolation=hyperv`; document that the run needs
   `--isolation=process` plus the device class.
10. A GPU probe (`C:/gpu_check.py`) baked in that **allocates and computes**, not
    one that only calls `is_available()` (W11).
