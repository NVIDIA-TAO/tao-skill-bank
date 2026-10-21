# Native Windows host setup — full five-action parity, no WSL and no container

This is the container recipe inverted. Every step in the verified
`Dockerfile` / `Dockerfile.tao` pair is a `pip install` or a file drop, so the
same stack installs directly on a Windows host with no image, no daemon and no
device class. Pins below are the ones that were actually resolved in the working
host venv (`C:\cosmos-win\venv-tao`) — reproduce them verbatim.

When you are done, `preflight_cosmos3_windows_host.py` must report zero failures — see
section 12 for the exact invocation, which needs more flags than you would guess.

## Validated target layout

This is what is actually on disk on the reference box. Adjust the drive; keep
the shape, and keep the FFmpeg directory names exactly as the BtbN zips extract
them (see section 6 — the short names in older revisions of this doc never
existed):

```
C:\cosmos-win\
  python\                                       python-build-standalone 3.12.13; `home` of BOTH venvs
  venv\                                         TRAINING-ONLY venv, 109 packages
                                                  no cosmos_rl, no cosmos-framework-*.exe
  venv-tao\                                     THE FULL FIVE-ACTION VENV, 202 packages
                                                  all five cosmos-framework-*.exe entrypoints
                                                  <- this is the one the 2026-08-13 runs used
  ffmpeg-n8.1-latest-win64-gpl-shared-8.1\bin\  FFmpeg 8.1 SHARED (unmodified BtbN zip name)
  ffmpeg-n7.1-latest-win64-gpl-shared-7.1\bin\  FFmpeg 7.1 SHARED (unmodified BtbN zip name)
  cosmos-framework\                             editable (`pip install -e`) source, branch windows-port-pr164
  out\                                          IMAGINAIRE_OUTPUT_ROOT
```

There is **no `src\` level** — `cosmos-framework` sits directly under
`C:\cosmos-win\`. And there is no `cosmos-rl` working tree in the layout at all:
in the validated environment `cosmos-rl` was installed **non-editable**, so its
code lives in `C:\cosmos-win\venv-tao\Lib\site-packages\cosmos_rl\` (section 11).

---

## 0. Host prerequisites
Every row below is satisfied on the reference box, which is a Windows 11
Enterprise host with one RTX PRO 6000 (97887 MiB) running CPython 3.12.13 and
torch 2.13.0+cu130 (cuDNN 92000).

| Requirement | Why | Check |
|---|---|---|
| Windows 11 / Server 2022 or 2025, x64 | every wheel is `win_amd64` | `[System.Environment]::OSVersion` |
| NVIDIA driver at CUDA 13.0 or newer, WDDM | the torch cu130 wheels | `nvidia-smi` (CUDA version, top right) |
| Python **3.12** x64 | every wheel is cp312 | `python -VV` |
| Visual C++ x64 redistributable | torch's DLLs link against it | see step 1 |
| FFmpeg **shared** builds, 8.1 **and** 7.1 | torchcodec and PyNvVideoCodec | step 6 |
| `LongPathsEnabled = 1` | pip dies mid-install at 262 chars | step 1 |
| ~120 GB free on the venv drive | torch alone is 1.9 GB; models dominate | — |

No CUDA toolkit is required. The torch Windows wheel bundles the CUDA runtime
internally — that is why it is 1.9 GB — and the driver supplies the rest.

---

## 1. Two registry/runtime prerequisites, before any pip
**Long paths.** Set this FIRST. A fresh Server Core (and many locked-down
Windows 11 images) default to `LongPathsEnabled = 0`, so `MAX_PATH` is 260 and
pip dies part-way through an install with
`OSError: [Errno 2] No such file or directory: 'C:\\...\\..._instance_part1.cpp'`.
CPython 3.6+ is already long-path aware through its manifest, so the registry
key is the only missing piece. Each new process picks it up.

```powershell
New-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' `
  -Name 'LongPathsEnabled' -Value 1 -PropertyType DWORD -Force
```

**Visual C++ runtime.** Server Core ships none, and `import torch` then dies
with `OSError: [WinError 126] The specified module could not be found. Error
loading "...\torch\lib\c10.dll" or one of its dependencies.` Install
`vc_redist.x64.exe /quiet /norestart`. Copy the installer in rather than
downloading it on a fresh Server Core: the CA store is empty until `certifi` is
wired up in step 4, so an in-place `Invoke-WebRequest` is unreliable.

---

## 2. Python 3.12
Validated: the reference box runs CPython **3.12.13** out of
`C:\cosmos-win\python`, and that directory is the recorded `home` of both venvs —
so the standalone-tarball layout is the one in production here. Transcribed: the
download/extract commands themselves were not re-run.

If the python.org MSI is blocked by corporate policy it fails with
`exit 1625 ERROR_INSTALL_PACKAGE_REJECTED`. `python-build-standalone` is a plain
tarball extraction — no installer, no admin, nothing for policy to reject:

```powershell
curl.exe -sL -o py312.tar.gz `
  https://github.com/astral-sh/python-build-standalone/releases/download/20260807/cpython-3.12.13%2B20260807-x86_64-pc-windows-msvc-install_only.tar.gz
tar.exe -xzf py312.tar.gz -C C:\cosmos-win     # -> C:\cosmos-win\python\python.exe
```

---

## 3. The venvs — there are TWO, and why `uv sync` cannot be used
The validated host does **not** have "the one and only TAO venv". It has two,
both with `home = C:\cosmos-win\python` and both on CPython 3.12.13:

| Venv | Packages | What is in it | What it can run |
|---|---|---|---|
| `C:\cosmos-win\venv` | 109 | cosmos-framework + the torch stack. **No `cosmos_rl`. No `cosmos-framework-*.exe`.** | training only |
| `C:\cosmos-win\venv-tao` | 202 | the training stack **plus** `nvidia-tao-core 6.25.11`, `tao-cosmos-actions`, `pynvvideocodec 2.2.0`, Flask, `opencv-python-headless`, `orjson`, `nvidia-cuda-runtime-cu12 12.9.79` | all five actions; **all five** `cosmos-framework-*.exe` entrypoints are present |

**Use `venv-tao`.** The 2026-08-13 LoRA SFT run and `evaluate` run both executed
out of `venv-tao`. `C:\cosmos-win\venv` is the older training-only environment;
it is kept because it is far smaller, but `export`, `evaluate`, `inference` and
`inference-microservice` cannot start from it — their console scripts simply do
not exist there, and `import cosmos_rl` fails.

**On the single-venv shape.** Sections 4–11 read as one linear build into one
venv. That is what the **container** does: `Dockerfile` / `Dockerfile.tao` layer
the framework stack and the TAO stack into a single site-packages. That shape has
**never been validated natively on this box.** What has been validated natively
is the two-venv split above. If you build a single venv on Windows you are
following the container recipe, not the verified host layout — it may well work,
but you are the first to try it, and any divergence in sections 9–11 is yours to
debug. The conservative native path is: build `venv` through section 8, then
build `venv-tao` by repeating sections 3–8 into a second directory and continuing
through sections 9–11 there.

Wherever `$PY` appears below, it is the interpreter of whichever venv you are
building. For the validated five-action environment that is
`C:\cosmos-win\venv-tao\Scripts\python.exe`.

```powershell
# training-only venv (validated as "training only")
C:\cosmos-win\python\python.exe -m venv C:\cosmos-win\venv

# the full five-action venv (validated end-to-end for train + evaluate)
C:\cosmos-win\python\python.exe -m venv C:\cosmos-win\venv-tao
$PY = "C:\cosmos-win\venv-tao\Scripts\python.exe"
& $PY -m pip install --upgrade pip certifi
```

`uv sync` is unusable: `pyproject.toml` declares
`required-environments = ["sys_platform == 'linux' ...]`, so Windows is not a
resolvable environment at all. The dependency set is hand-carried below.

**TLS.** On a fresh Server Core the CA store is empty and every
`requests`/`huggingface_hub` call fails on TLS until certifi's bundle is pointed
at explicitly:

```powershell
$cert = & $PY -c "import certifi; print(certifi.where())"
[System.Environment]::SetEnvironmentVariable('SSL_CERT_FILE', $cert, 'Machine')
```

---

## 4. torch — the load-bearing pin
`venv-tao` carries exactly these pins, and the real train + evaluate runs went
through them: torch 2.13.0+cu130, cuDNN reported as 92000.

```powershell
& $PY -m pip install "torch==2.13.0+cu130" --index-url https://download.pytorch.org/whl/cu130
& $PY -m pip install "torchcodec==0.14.0+cu130" "torchvision==0.28.0+cu130" `
    --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
```

**2.13.0+cu130 is required, not preferred.** It is the only line shipping cuDNN
**9.20**, and `cosmos_framework/model/attention/cudnn/__init__.py` sets
`CUDNN_MIN_BACKEND_VERSION = 92000`. On torch 2.11 (cuDNN 9.19 — one hundred
short) the cuDNN attention backend silently disables itself and falls back to
NATTEN, which has **no Windows build**. That backend is what makes generator
inference work here.

---

## 5. `requirements-cosmos3-win.txt` — the cosmos-framework dependency set (104 entries)
**There is no `requirements-cosmos3-win.txt` to download — the block below IS the file.**
Nothing fetches it for you and no upstream repo publishes it. Save the block
verbatim next to this document, as
`skills/models/tao-finetune-cosmos-reason/references/requirements-cosmos3-win.txt`, and
point pip at that path. If your copy of the bank already carries a file at that
path, use it and diff it against this block; if it does not, author it from this
block before running the pip line.

Validated: `venv-tao` — 202 packages once sections 5, 9, 10 and 11 are all in —
runs train and evaluate on this box, and its torch line is exactly the section-4
pin. Also validated the hard way: the `pytest` entry (see below), whose absence
killed a real run on 2026-08-13. Transcribed: the remaining pins were captured
from the earlier working host venv and were not re-audited package by package on
that date, and installing them as one pip invocation was not re-executed.

```powershell
& $PY -m pip install -r <bank>\skills\models\tao-finetune-cosmos-reason\references\requirements-cosmos3-win.txt `
    --extra-index-url https://download.pytorch.org/whl/cu130
```

```
accelerate==1.14.0
aiohappyeyeballs==2.7.1
aiohttp==3.14.3
aiosignal==1.4.0
annotated-types==0.8.0
antlr4-python3-runtime==4.9.3
anyio==4.14.2
attrs==26.1.0
av==18.0.0
boto3==1.43.68
botocore==1.43.68
cattrs==26.1.0
certifi==2026.7.22
charset-normalizer==3.4.9
click==8.4.2
colorama==0.4.6
contourpy==1.3.3
cycler==0.12.1
datasets==5.0.1
diffusers==0.39.0
dill==0.4.1
docstring_parser==0.18.0
einops==0.8.2
einx==0.4.3
filelock==3.29.0
fonttools==4.63.0
frozendict==2.4.7
frozenlist==1.8.0
fsspec==2026.4.0
ftfy==6.3.1
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==0.36.2
hydra-core==1.3.5
idna==3.18
ImageIO==2.37.4
imageio-ffmpeg==0.6.0
importlib_metadata==9.0.0
iopath==0.1.10
Jinja2==3.1.6
jmespath==1.1.0
kiwisolver==1.5.0
loguru==0.7.3
MarkupSafe==3.0.3
matplotlib==3.11.1
mpmath==1.3.0
msgpack==1.2.1
multidict==6.7.1
multiprocess==0.70.19
networkx==3.6.1
numpy==2.5.2
nvidia-cudnn-frontend==1.27.0
nvidia-ml-py==13.610.43
obstore==0.11.0
omegaconf==2.3.1
packaging==26.3
pandas==3.0.5
peft==0.20.0
pillow==12.3.0
platformdirs==4.11.2
portalocker==4.1.0
propcache==0.5.2
protobuf==7.35.1
psutil==7.2.2
pyarrow==25.0.1
pydantic==2.13.4
pydantic_core==2.46.4
pyparsing==3.3.2
pytest==9.1.1
python-dateutil==2.9.0.post0
pytz==2026.3.post1
PyYAML==6.0.3
qwen-vl-utils==0.0.14
regex==2026.7.19
requests==2.34.2
s3transfer==0.19.2
safetensors==0.8.0
scipy==1.18.0
sentry-sdk==2.67.1
setuptools==78.1.0
six==1.17.0
sympy==1.14.0
termcolor==3.3.0
tokenizers==0.22.2
torch==2.13.0+cu130
torchcodec==0.14.0+cu130
torchvision==0.28.0+cu130
tqdm==4.70.0
transformers==4.57.6
typeguard==4.6.0
typing-inspection==0.4.3
typing_extensions==4.15.0
tyro==1.0.15
tzdata==2026.3
urllib3==2.7.0
uv==0.12.3
wandb==0.28.1
wcwidth==0.8.2
websockets==17.0.1
win32_setctime==1.2.0
xxhash==3.8.1
yarl==1.24.5
zipp==4.1.0
```

**`pytest==9.1.1` in that list is a RUNTIME dependency of TRAINING, not a
leftover test pin. Do not prune it.**

`cosmos_framework/configs/base/reasoner/config.py:53` recursively imports every
module in the experiment package, and
`configs/base/reasoner/experiment/tao_video_sft_test.py:10` does `import pytest`
at module scope. The recursion helper (`utils/config_helper.py:199`) only skips
modules whose name starts with `_` — it does **not** skip `*_test.py`. So every
training run imports the test module during config resolution, before a single
tensor is allocated.

This is not theoretical: on 2026-08-13 the first training attempt on this box
died at config load with `ModuleNotFoundError: No module named 'pytest'`, and
installing pytest is what made the run that later reached "Done with training."
possible. The container image carries `pytest==9.1.1` for the same reason, which
is why earlier revisions of this recipe never noticed the omission — the prose
warned about it while the pinned list left it out.

If you are repairing an existing venv rather than building one:

```powershell
& $PY -m pip install pytest==9.1.1
```

### Packages that do not exist on Windows

`transformer-engine`, `natten`, `flash-attn`, `apex`, `megatron-core`,
`nvidia-dali`, `decord`, `multi-storage-client` (no wheel *and* no sdist — its
dependency `xattr` needs POSIX extended attributes). `triton` proper has no
Windows wheel, but **`triton-windows` does** and installs as the real `triton`
module.

---

## 6. FFmpeg — both lines, shared builds, 8.1 first
The commonly-downloaded "essentials" build contains only `.exe` files and no
DLLs; it cannot work. Use the BtbN **shared** zips.

| Build | Extracted directory on the reference box | Provides | Consumer |
|---|---|---|---|
| FFmpeg 8.1 shared | `C:\cosmos-win\ffmpeg-n8.1-latest-win64-gpl-shared-8.1\bin` | `avformat-62`, `avcodec-62` | PyNvVideoCodec 2.2.0 |
| FFmpeg 7.1 shared | `C:\cosmos-win\ffmpeg-n7.1-latest-win64-gpl-shared-7.1\bin` | `avformat-61`, `avcodec-61`, `avutil-59`, `swresample-5`, `swscale-8` | torchcodec 0.14.0 |

**Keep the BtbN directory names exactly as they extract.** Those long names are
what the validated host has, and they are what the `.pth` in section 7 and the
preflight invocation in section 12 hard-code. Earlier revisions of this document
claimed `C:\cosmos-win\ffmpeg8\bin` and `C:\cosmos-win\ffmpeg\bin`; **neither
directory has ever existed on this machine**, and `preflight_cosmos3_windows_host.py`
carries a third, equally fictitious pair of defaults (see section 12). If you do
rename the directories, you must update all three places yourself.

Extract both. Keep both. torchcodec is happy against 8.1, so 8.1 goes first in
every ordering below; 7.1 stays as the fallback that the verified host and image
both carry.

---

## 7. The DLL discovery `.pth` — one line, three directories
Since Python 3.8, Windows extension modules do **not** resolve dependent DLLs
from `PATH`. Putting FFmpeg on `PATH` is not enough: `os.add_dll_directory()`
has to have run first, and it has to run in *every* process. A `.pth` in
site-packages executes at interpreter startup, which is the only mechanism that
covers subprocesses, DataLoader workers and console entrypoints alike.

The file on the validated host is
`C:\cosmos-win\venv-tao\Lib\site-packages\_win_tao_dlldirs.pth` — note the
**plural** `dlldirs`. Its entire body is the following single line, reproduced
verbatim from the working environment. It **must stay one line**: a `.pth`
executes only lines beginning with `import`.

```python
import os, sys; globals().setdefault('_win_tao_handles', [os.add_dll_directory(x) for x in [r'C:\cosmos-win\venv-tao\Lib\site-packages\nvidia\cuda_runtime\bin', r'C:\cosmos-win\ffmpeg-n8.1-latest-win64-gpl-shared-8.1\bin', r'C:\cosmos-win\ffmpeg-n7.1-latest-win64-gpl-shared-7.1\bin'] if sys.platform == 'win32' and os.path.isdir(x)])
```

**Keeping the handles alive is load-bearing — do not "simplify" this into a bare
list comprehension.** `os.add_dll_directory()` does not perform a fire-and-forget
registration: it returns a handle object (a context manager) whose finalizer
calls `RemoveDllDirectory`, undoing the registration. Bind that handle to nothing
and the object is immediately garbage-collectable, so the search path entry can
vanish at an arbitrary later moment — typically long after startup, in a
DataLoader worker or on the first `import` that actually needs `avformat-62.dll`.
The failure then looks like a random `ImportError: DLL load failed` on a machine
where the directory demonstrably exists. `globals().setdefault('_win_tao_handles',
[...])` parks the handles in the module globals of the `.pth` execution context
for the life of the interpreter, and `setdefault` makes a second execution a
no-op instead of a double registration. Earlier revisions of this document showed
the discard form (`[os.add_dll_directory(_d) for _d in ...]` with the result
thrown away); that is a latent bug, not a style choice.

Three more details:

- **Per-venv.** Each venv has its own site-packages, so each venv that must load
  these DLLs needs its own copy of the file, with its own
  `...\Lib\site-packages\nvidia\cuda_runtime\bin` path baked in. The path above
  is `venv-tao`'s; the training-only `venv` needs `C:\cosmos-win\venv\Lib\...`.
- **`PATH` ordering.** Put
  `C:\cosmos-win\ffmpeg-n8.1-latest-win64-gpl-shared-8.1\bin` ahead of
  `C:\cosmos-win\ffmpeg-n7.1-latest-win64-gpl-shared-7.1\bin` on `PATH`, so that
  `ffmpeg.exe` itself resolves in the same order the DLL directories are
  registered.
- **Write the file with a real editor.** Do **not** generate it from an inline
  `python -c` inside PowerShell: the quotes and semicolons are mangled before
  Python ever sees them.

---

## 8. cosmos-framework
Validated: on the reference box `cosmos_framework` is an **editable** install
whose source tree is `C:\cosmos-win\cosmos-framework` — directly under
`C:\cosmos-win\`, with **no `src\` level**. Transcribed: the clone/checkout/patch
sequence, and the contents of the port table below.

```powershell
git clone https://github.com/NVIDIA/cosmos-framework C:\cosmos-win\cosmos-framework
# check out PR #164 (TAO/PEFT), then apply the Windows port described below
& $PY -m pip install -e C:\cosmos-win\cosmos-framework --no-deps
```

The Windows port touches 13 files and adds one new module
(`cosmos_framework/utils/win_compat.py`).

Use the vendored Framework patch `references/cosmos-framework-windows-port-pr164.patch`;
the exact source and rebase rule are in `cosmos3-windows-source-patches.md`.
Do not reconstruct or publish a second patch from this host setup document.

| Area | Fix |
|---|---|
| `os.sched_setaffinity` / `sched_getaffinity` | `hasattr` guard — POSIX-only; the `AttributeError` escapes an `NVMLError`-only handler and aborts every entrypoint |
| `signal.SIGALRM` / `signal.alarm` | `hasattr` guard — no Windows equivalent; hang watchdog disabled |
| NCCL backend | fall back to **gloo** on win32; a process group is mandatory (unconditional `dist.barrier()` at trainer init) |
| DDP wrapper | **skip at `world_size == 1`** |
| gloo CUDA collectives | stage tensors through host memory |
| `import fcntl` x3 | `msvcrt`-based `flock` shim |
| `os.symlink` for the job dir | symlink to directory junction to pointer file; Developer Mode is usually off |
| `multistorageclient` x4 | optional imports; the `s3`/`msc` backends simply do not register |
| unguarded `import triton` | guard, or install `triton-windows` |

**Rebasing onto PR #164.** #164 independently added `ImportError` guards for
`multi-storage-client`, so that patch becomes unnecessary. Its affinity change
does **not** replace the guard: it adds `sched_getaffinity` and widens the
handler to `OSError`, which still misses `AttributeError` — keep the `hasattr`
guard. #164 also selects sdpa automatically, but only on the Edge recipe. Leave
the Nano recipe on its validated native `cosmos` attention default; the older
instruction to force `attn_implementation=sdpa` caused the W14 silent-learning
failure documented in `cosmos3-windows-gaps.md`.

---

## 9. Extra runtime dependencies
Validated: `venv-tao` contains all of these — `nvidia-cuda-runtime-cu12 12.9.79`,
`pynvvideocodec 2.2.0`, Flask, `opencv-python-headless`, `orjson`. Transcribed:
the `dumpbin` diagnosis below.

```powershell
& $PY -m pip install nvidia-cuda-runtime-cu12 PyNvVideoCodec flask opencv-python-headless orjson
```

`nvidia-cuda-runtime-cu12` is **not** redundant with torch's bundled CUDA 13
runtime. `PyNvVideoCodec_130.pyd` — the CUDA **13.0** build — links
`cudart64_12.dll`, i.e. the CUDA **12** runtime. Confirmed with
`dumpbin /DEPENDENTS`. It is an upstream packaging inconsistency that goes
unnoticed on Linux, where a system-wide `libcudart` is usually already present.
Without it the import dies with
`ImportError: DLL load failed while importing _PyNvVideoCodec: The specified module could not be found.`

`flask`, `opencv-python-headless` and `orjson` are plain missing-dependency
installs, discovered by iterating `--help` on each entrypoint.

---

## 10. `requirements-cosmos3-tao-win.txt` — the curated tao-core dependency set (16 entries)
**As with section 5, there is no `requirements-cosmos3-tao-win.txt` to download — the
block at the end of this section IS the file.** Save it as
`skills/models/tao-finetune-cosmos-reason/references/requirements-cosmos3-tao-win.txt`
(next to this document) and install from that path; if your copy of the bank
already carries that file, diff it against the block before trusting either.

`nvidia-tao-core` declares **uWSGI** as a hard requirement
(`requirements.txt:49`), and uWSGI cannot be built on Windows at all. Its build
script calls a POSIX-only API on the first line it executes:

```
File ".../uwsgiconfig.py", line 8, in <module>
  uwsgi_os = os.uname()[0]
AttributeError: module 'os' has no attribute 'uname'. Did you mean: 'name'?
ERROR: Failed to build 'uWSGI' when getting requirements to build wheel
```

That failure is at the *metadata* stage, so no wheel or build flag can help.

**uWSGI is not a blocker.** It serves only the TAO microservices HTTP app;
nothing on the train / export / evaluate / inference path imports it. Install
tao-core with `--no-deps` and supply its dependencies from this file instead.

Of tao-core's 54 requirements, only 28 are present in the venv where eval and
inference were verified working; 12 of those already arrive with
`requirements-cosmos3-win.txt`, leaving the 16 below. The 26 omitted requirements are
genuinely unused by the five entrypoints, several cannot install on Windows
anyway (`pycocotools` needs a compiler, `tritonclient[all]`), and
**`torchao==0.11.0` is deliberately excluded** because it can drag torch off the
2.13.0+cu130 pin that the cuDNN 92000 attention gate depends on.

Where tao-core's pin is older than an already-installed package (Flask 3.1.1 vs
3.1.3, Werkzeug 3.1.3 vs 3.1.8, grpcio 1.67.1 vs 1.83.0) the pin is relaxed
rather than forcing a downgrade of a working package.

```
cryptography>43.0.1
docker
Flask
grpcio
jsonschema
kubernetes==32.0.1
leptonai>=0.26.0
ngcsdk==3.151.0
orjson
pycryptodome==3.23.0
pymongo==4.13.0
requests_toolbelt==1.0.0
ruamel.yaml==0.18.12
toml
validators==0.35.0
Werkzeug
```

---

## 11. The TAO integration layer
Validated: `venv-tao` carries `nvidia-tao-core 6.25.11`, `tao-cosmos-actions` and
**all five** `cosmos-framework-*.exe` entrypoints, and `cosmos_rl` is installed
**non-editable** — it lives in `C:\cosmos-win\venv-tao\Lib\site-packages\cosmos_rl\`,
not in a working tree. Transcribed: the install order, the commit table and the
`pwd` guard.

Four of the five actions live in `cosmos-rl` (the private TAO integration repo),
not in `cosmos-framework`:

| Console script | Target |
|---|---|
| `cosmos-framework-train` | `cosmos_framework.scripts.train:main` |
| `cosmos-framework-export` | `cosmos_rl.framework.export:main` |
| `cosmos-framework-evaluate` | `cosmos_rl.evaluation.evaluate:main` |
| `cosmos-framework-inference` | `cosmos_rl.inference.inference:main` |
| `cosmos-framework-inference-microservice` | `cosmos_rl.inference.infer_microservices:main` |

Install in dependency order:

```powershell
& $PY -m pip install -r <bank>\skills\models\tao-finetune-cosmos-reason\references\requirements-cosmos3-tao-win.txt
git clone <cosmos-rl remote> C:\cosmos-win\cosmos-rl   # any path; it is only a build directory
cd C:\cosmos-win\cosmos-rl
& $PY -m pip install --no-deps .\tao-core
& $PY -m pip install .\nvidia-tao-daft
& $PY -m pip install .          # NOT -e — see below
```

**`cosmos-rl` is installed non-editable in the validated environment.** The last
line is a plain `pip install .`; the resulting package sits in
`C:\cosmos-win\venv-tao\Lib\site-packages\cosmos_rl\`, and the clone is disposable
build input. Two consequences: the clone directory is **not** part of the runtime
layout (which is why it does not appear in the layout block at the top), and
editing the clone after installation changes nothing at runtime. If you need to
iterate on the four Windows commits below, either reinstall after each change or
switch to `pip install -e .` — but the `-e` form is *not* what the validated
environment uses; `cosmos_framework` (section 8) is the only editable install
observed on the reference box.

**Keep the `.git` directories.** `pyproject.toml` declares
`dynamic = ["version"]` via setuptools-scm, so the install needs git metadata to
derive a version. Without it the install dies with
`setuptools-scm was unable to detect version`. Git must be on `PATH`; MinGit is
the portable Git-for-Windows distribution and is a plain extract, so there is no
installer for policy to reject.

### The `cosmos-rl` Windows branch — exactly four commits

| Commit | Change | File |
|---|---|---|
| `ed752be` | load NVDEC via `nvcuvid.dll` on win32 | `cosmos_rl/utils/pynv_video_reader.py` |
| `5d191f6` | bump the tao-core submodule to the pwd-guard commit | submodule pointer |
| `c00a158` | allow `torchcodec` as an evaluation video decoder | `cosmos_rl/evaluation/base.py` |
| `62da6f8` | pin torchcodec explicitly instead of deferring to the environment | `cosmos_rl/evaluation/base.py` |

`ed752be` is worth understanding, because its failure mode was three layers from
its cause: `ctypes.CDLL("libnvcuvid.so.1")` hardcodes the Linux ELF SONAME, so
the load raised on Windows and NVDEC backend registration failed. qwen-vl-utils
then silently demoted itself to its torchvision reader — and torchvision >= 0.23
removed `torchvision.io.read_video`. The reported error was
`module 'torchvision.io' has no attribute 'read_video'`.

### The one tao-core source guard — apply it locally

`nvidia_tao_core/microservices/handlers/execution_handlers/slurm_handler.py:24`
imports the POSIX-only `pwd` module unconditionally, so importing anything that
reaches this handler fails on Windows:

```python
# before (line 24)
import pwd

# after
try:
    import pwd
except ImportError:  # Windows has no pwd module
    pwd = None
```

**The published commit carrying this guard (`544b23f`) is NOT reachable from
public tao-core.** Do not expect `git fetch` to produce it — apply the guard to
your checkout before `pip install --no-deps ./tao-core`.
`preflight_cosmos3_windows_host.py` verifies the installed copy.

---

## 12. Verify
Point it at **`venv-tao`**, and pass both FFmpeg directories explicitly:

```powershell
& "C:\cosmos-win\venv-tao\Scripts\python.exe" `
  <bank>\skills\models\tao-finetune-cosmos-reason\scripts\preflight_cosmos3_windows_host.py `
  --venv C:\cosmos-win\venv-tao `
  --ffmpeg-dir C:\cosmos-win\ffmpeg-n8.1-latest-win64-gpl-shared-8.1\bin `
  --ffmpeg-dir C:\cosmos-win\ffmpeg-n7.1-latest-win64-gpl-shared-7.1\bin
```

`--venv C:\cosmos-win\venv` will fail the entrypoint check by design: the
training-only venv has none of the five console scripts (section 3). Only
`venv-tao` can pass the whole gate.

**Pass `--ffmpeg-dir` twice, or the ffmpeg check fails on a healthy machine.**
`preflight_cosmos3_windows_host.py:188` falls back to `C:\ffmpeg8\bin` and
`C:\ffmpeg\bin` when no hint is given — a third pair of paths that has never
existed here, distinct from both the real BtbN names and the short names older
revisions of this document claimed. Run on this box with no hints,
`check_ffmpeg(report, [])` returns
`{'status': 'FAIL', 'detail': 'found avformat majors none'}` — on the same
machine, in the same hour, that had just completed a real LoRA training run and a
real evaluation run using those very DLLs. A hintless ffmpeg FAIL is a defect in
the preflight defaults, not evidence about your environment; a FAIL *with* both
hints supplied is real.

Zero failures is the gate. A green preflight proves the interpreter, the VC++
runtime, the torch/cuDNN pins, a real bf16 allocation, both FFmpeg lines, the
DLL registration, all five entrypoints and the tao-core `pwd` guard.

---

## 13. Runtime environment
`windows_job_runner.py submit --tao-windows-env` injects this block for you, and
that is exactly how the 2026-08-13 train and evaluate runs were launched
(`{"status": "COMPLETE", "return_code": 0}` for both). To run an action by hand,
export it first:

```powershell
$env:MASTER_ADDR="127.0.0.1"; $env:MASTER_PORT="29500"
$env:RANK="0"; $env:WORLD_SIZE="1"; $env:LOCAL_RANK="0"
$env:USE_LIBUV="0"; $env:PYTHONIOENCODING="utf-8"
$env:TOKENIZERS_PARALLELISM="false"; $env:WANDB_MODE="disabled"
$env:IMAGINAIRE_OUTPUT_ROOT="C:\cosmos-win\out"
$env:HF_HOME="C:\cosmos-win\hf"
$env:TAO_API_JOB_ID="my-job"; $env:TAO_API_RESULTS_DIR="C:\cosmos-win\out\my-job"
```

Use plain `python`, **not `torchrun`**: the shipped launchers are bash and
hardcode `--nproc_per_node=8`.

`TAO_API_JOB_ID` is mandatory for every action and appears in no `--help`
(`ValueError: TAO_API_JOB_ID environment variable is required for TAO status
logging`). But for **training**, setting it alone is fatal at iteration 0 with
`TAO_API_RESULTS_DIR is required when TAO_API_JOB_ID is set`. The requirements
are opposite, so if one environment block serves everything, set **both**.

`MASTER_PORT` can collide with a Hyper-V reserved range, which surfaces as
`WinError 10013` on bind. Check with
`netsh interface ipv4 show excludedportrange protocol=tcp` and pick a port
outside every listed range.

---

## 14. Training overrides — the eleven that are mandatory
Validated in aggregate: with this stack, a LoRA SFT run on 2026-08-13 injected
LoRA into `['q_proj', 'k_proj', 'v_proj', 'o_proj']` (144 modules wrapped,
15,335,424 trainable LoRA params against 8,767,123,696 frozen, 0.175%), reached
"Done with training." and saved a checkpoint. Transcribed: the individual
justification for each override below, which comes from earlier debugging rather
than from an A/B on that run.

They go after a bare `--` separator:

```
optimizer=adamw
trainer.distributed_parallelism=ddp
model.config.parallelism.data_parallel_shard_degree=1
model.config.parallelism.data_parallel_replicate_degree=1
model.config.policy.lora_enabled=true
dataloader_train.num_workers=0
dataloader_train.persistent_workers=false
dataloader_train.prefetch_factor=null
dataloader_val.num_workers=0
dataloader_val.persistent_workers=false
dataloader_val.prefetch_factor=null
```

- **Do not override `attn_implementation` in the normal route.** The recipe's
  `cosmos` attention is a PyTorch/cuDNN-backed adapter and is the validated
  Windows training path. SDPA/eager is debug-only: apply the W14 mask-gate patch
  first, then set both the top-level and nested policy keys because `VLMModel`
  reads the latter.
- **`optimizer=adamw`**, not the default `fusedadamw` — the latter imports
  `transformer_engine`.
- **`num_workers=0`, `prefetch_factor=null`.** Windows uses `spawn`, not `fork`.

**Nine of these cannot be folded into a recipe TOML.** `SFTExperimentConfig`
rejects `model.config.*` and `dataloader_*` with `Extra inputs are not
permitted`, and TOML has no `null` literal. They must be passed on the command
line every time.

---

## 15. Optional: a third venv, for vLLM serving
Built and served on the reference box on 2026-08-13: vllm 0.26.0 loaded the
exported checkpoint (`Model loading took 16.79 GiB`, `GPU KV cache size: 237,712
tokens`, ~53 GB resident) and answered `/v1/chat/completions` over HTTP. What was
run is vLLM's own `vllm.entrypoints.openai.api_server`; the
`cosmos-framework-inference-microservice` entrypoint was **not** run.

This is a third venv, separate from both `venv` and `venv-tao` in section 3, and
it is needed only if you want to serve.

Only the serving path wants vLLM. Evaluate and inference generate through HF
Transformers (`cosmos_rl/framework/runtime.py` `_load()` branches into
`Cosmos3OmniModel` or `AutoModelForImageTextToText` and calls
`self.model.generate(...)` under `torch.inference_mode()`); SFT validation is an
in-process forward pass. There is no `LLM(...)`, no `SamplingParams`, no
`EngineArgs` anywhere on the TAO path.

vLLM pins `torch==2.11.0+cu130` with `==`, and cosmos needs 2.13.0+cu130 for the
cuDNN 9.20 gate. The two genuinely cannot share a venv, so serving gets its own:

```powershell
C:\cosmos-win\python\python.exe -m venv C:\cosmos-win\venv-vllm
C:\cosmos-win\venv-vllm\Scripts\python.exe -m pip install `
  vllm-0.26.0+cu132-cp312-cp312-win_amd64.whl `
  --extra-index-url https://download.pytorch.org/whl/cu130
```

Upstream publishes no Windows wheels; this is the SystemPanic build
(`github.com/SystemPanic/vllm-windows`), cp312 / cu132. **Keep the wheel's
original filename** — pip parses the version and ABI tags out of it (PEP 427),
and a renamed `vllm.whl` fails with
`ERROR: Invalid wheel filename (wrong number of parts)`.

That venv needs its own copy of the DLL-directory `.pth` from section 7 (a venv
has its own site-packages, so the `venv-tao` copy does not apply), with the
`cuda_runtime\bin` path rewritten to
`C:\cosmos-win\venv-vllm\Lib\site-packages\nvidia\cuda_runtime\bin` and the same
two FFmpeg directories left as they are.

Serving needs a `CUDA_HOME` shim:

```powershell
mkdir C:\cudashim\bin
copy C:\cosmos-win\venv-vllm\Lib\site-packages\torch\lib\cudart64_13.dll C:\cudashim\bin\
$env:CUDA_HOME = "C:/cudashim"          # forward slashes
$env:VLLM_USE_FLASHINFER_SAMPLER = "0"
```

`VLLM_ATTENTION_BACKEND` and `VLLM_USE_FLASHINFER_SAMPLER` do **not** avoid
this: the flashinfer attention backend is imported during **model loading**, and
it raises `ValueError: CUDA_LIB_PATH is not set`. Despite the message, the
variable to set is `CUDA_HOME`, not `CUDA_LIB_PATH`.
No CUDA toolkit is needed. flashinfer wants only
`<root>\bin\cudart64_<major>.dll`, where major comes from `torch.version.cuda`,
and that DLL already ships at `Lib\site-packages\torch\lib\`. The shim supplies
the directory shape, nothing else.

---

## 16. Native versus container — what changes
Validated natively on 2026-08-13: training wrote its checkpoint straight to the
job's results directory (`dataloader/ model/ optim/ scheduler/ trainer/`, 33 GB
of `model/`, 124 MB of `optim/`, `SaveDone succeeded=True`) with no rename
failure and no `docker cp`, and `evaluate` completed with
`video_decoder = "torchcodec"`. Transcribed: the container-side failure modes
they are contrasted against.

The native path is strictly simpler; these container-only problems disappear:

- No `--isolation=process` requirement, and no
  `--device class/5B45201D-F2F2-4F3B-85BB-30FF1F953599`.
- No NVENC probe failure. The device class projects `nvcuvid.dll` (NVDEC) but
  not `nvEncodeAPI64.dll`, so the default decoder cannot initialize in a
  container. On the host both are present. Setting
  `video_decoder = "torchcodec"` is still recommended for portability.
- No DCP-on-bind-mount failure. In a container, write-temp-then-rename gives
  `WinError 3` on `.metadata.tmp` to `.metadata` **after** training completes,
  forcing a container-internal output directory and a `docker cp`. On the host,
  write straight to the job's `results_dir`.
- No `--memory` sizing trap (below roughly twice the model size, DCP staging
  through host memory faults with `0xC0000005` after "Done with training.").
- No "a failed `RUN` does not fail a Windows docker build" hazard.

What does **not** change: multi-GPU is impossible either way **with the stock
torch Windows wheels**, because PyTorch guards all of its NCCL code behind
`USE_C10D_NCCL`, which those wheels are not built with — so
`init_process_group("nccl")` fails no matter what else is installed. (A community
NCCL port for Windows does exist and is actively maintained, and vLLM sidesteps
torch's process group by loading `nccl.dll` directly via `VLLM_NCCL_SO_PATH`;
neither has been tested here, and neither changes the stock-wheel conclusion.
**Multi-node is not supported** on Windows in any configuration.) Every one of
the model-level traps (`answer_type`, `num_frames`, `--media <dir>`,
`checkpoint.load_path`, export `--base-model-path-or-uri`) is
operating-system independent.
