# Cosmos3 on native Windows ARM64

This is an experimental native-host route for Cosmos3 LoRA. It does not change
the CPython 3.12 x64 reference environment in
`cosmos3-windows-x64-host-setup.md`, and it does not validate WSL, Docker,
multi-GPU, or multi-node execution.

## Required route contract

- Use an ARM64 Windows interpreter: `platform.machine()` must report `ARM64`.
  AMD64 Python under emulation does not validate native wheels.
- Use one RTX Spark-class GPU with CUDA exposed through a native ARM64 Torch
  wheel. The generic architecture and real-allocation gate is
  `skills/platform/tao-run-on-windows-venv/references/windows-arm64.md`.
- Keep Cosmos Framework at PR #164 (`git fetch origin pull/164/head`) plus the
  vendored `references/cosmos-framework-windows-port-pr164.patch`,
  Cosmos-RL on its `windows-port` checkout, and the TAO Core `pwd` import guard.
  The complete rebase contract is `cosmos3-windows-source-patches.md`.

Install native Torch before the rest of the Python stack:

```powershell
$PY = "<arm64-venv>\Scripts\python.exe"
& $PY -m pip install --pre --upgrade --only-binary=:all: `
  --extra-index-url 'https://pypi.nvidia.com/nvtorch_oot_nightly' `
  torch torchvision torchaudio
& $PY <bank>\skills\platform\tao-run-on-windows-venv\scripts\preflight_windows_arm64.py `
  --allocation-gib 72
```

The 72 GiB allocation is a model-planning gate, not a Windows reservation. It
allocates, writes, synchronizes, and frees a CUDA buffer, so serialize use of a
shared GPU. On the RTX Spark N1X validation host, Torch
`2.15.0.dev20260902+cu134` reported 101.50 GiB total and this gate passed. The
WDDM `nvidia-smi` framebuffer counter reported 8 GiB; it is not the
unified-memory capacity limit.

## Public weights and media packages

Download the public `nvidia/Cosmos3-Nano` release, then convert it to a local
VLM directory such as `C:\models\Cosmos3-Nano-VLM`. Set
`VLM_SAFETENSORS_PATH` to the converted directory, never the raw release or the
Hugging Face ID. The converter resolves the public
`Qwen/Qwen3-VL-8B-Instruct` processor dependency. Budget local storage for the
raw release (~32.6 GiB), Qwen cache (~16.3 GiB), converted VLM (~16.3 GiB), and
a DCP checkpoint (~32.8 GiB).

While converting, put the ARM64 venv's `Scripts` directory on `PATH` so the
nested `uv` call resolves. The Framework conversion helper must retain the
Windows patch that skips generation-only audio/video tokenizers; the VLM export
does not use them and the audio helper can stall on native Windows.

No public win_arm64 TorchCodec wheel exists. Build TorchCodec from source on
the ARM64 host against the WoA Torch nightly installed above (TorchCodec's
`CMakeLists` build with CUDA enabled; FFmpeg 9 ARM64 shared libraries on the
build path), then install the resulting wheel with `--no-deps`. The validated
build was `torchcodec 0.17.0a0` with CUDA 13.4/NVDEC video, nvJPEG CUDA JPEG,
CPU JPEG, and PNG support.

The wheel is dynamically linked. The host still needs matching ARM64
Torch/CUDA, FFmpeg 9 ARM64, and GStreamer SDK JPEG/PNG DLLs. Add their `bin`
directories, plus Torch `lib` and `<cuda>\bin\arm64`, with a `.pth` that retains
the handles returned by `os.add_dll_directory()`. Python 3.13 does not reliably
resolve these extension dependencies from `PATH` alone. An import failure for
`libtorchcodec_image.dll` commonly means one of its dependencies is missing.

The packaged source contains an ARM64 correction for an AVIF-disabled stub.
WebP, AVIF, GIF, and HEIC are intentionally omitted. Cosmos's ordinary static
image route remains Pillow-based; JPEG and PNG are the native TorchCodec image
formats covered by this wheel.

## Framework, Cosmos-RL, and TAO Core install

This page pins only what differs on ARM64: Torch, TorchCodec, and (below)
vLLM. The framework itself, its dependency set, Cosmos-RL, and the TAO
integration layer install exactly as in `cosmos3-windows-x64-host-setup.md`,
sections 5 through 11, run with the ARM64 interpreter **after** native Torch is
in place. Known ARM64 deviations, by x64 section:

- **Section 5, `requirements-cosmos3-win.txt`** installs unchanged. Its
  `platform_machine` markers skip the x64-only pins on ARM64: the cu130
  torch/torchvision/torchcodec lines (Torch comes from the WoA nightly above,
  TorchCodec from the ARM64 source build above), `pyarrow` and `obstore` (no win_arm64
  wheels, trap W16), and install `duckdb` in pyarrow's place. The mining
  scripts read and write the same parquet files through `parquet_io.py` with
  either backend. Drop the x64 `--extra-index-url .../cu130` from the install
  line; it has nothing to offer an ARM64 interpreter.
- **Sections 6 and 7, FFmpeg and the DLL `.pth`**: the x64 table lists two x64
  BtbN shared builds. On ARM64 the TorchCodec wheel needs the FFmpeg 9 ARM64
  build named under "Public weights and media packages", and the `.pth` lists
  its `bin`, Torch `lib`, and `<cuda>\bin\arm64`. Confirm which FFmpeg major
  PyNvVideoCodec loads on ARM64 before adding a second build.
- **Section 8, cosmos-framework**: identical — clone, check out PR #164, apply
  the vendored Windows port patch
  (`cosmos3-windows-source-patches.md`), then `pip install -e ... --no-deps`.
  Keep the Nano recipe on the default `cosmos` attention (W14).
- **Section 9, extra runtime dependencies**: `opencv-python-headless` had no
  cp313 win_arm64 wheel at the validated point ("Remaining boundaries"); direct
  CLI inference does not need it, only the optional inference microservice.
  Re-verify the rest of that list against the ARM64 interpreter.
- **Section 10, `requirements-cosmos3-tao-win.txt`** installs unchanged; its
  markers pin the abi3 `cryptography==46.0.3` and skip `grpcio` on ARM64 (W16).
- **Section 11, TAO integration layer**: identical, including the tao-core
  `pwd` import guard.

```powershell
$PY = "<arm64-venv>\Scripts\python.exe"          # native Torch already installed (above)
& $PY -m pip install -r <bank>\skills\models\tao-finetune-cosmos-reason\references\requirements-cosmos3-win.txt
git clone https://github.com/NVIDIA/cosmos-framework <src>\cosmos-framework
# check out PR #164 and apply the Windows port patch (cosmos3-windows-source-patches.md)
& $PY -m pip install -e <src>\cosmos-framework --no-deps
& $PY -m pip install -r <bank>\skills\models\tao-finetune-cosmos-reason\references\requirements-cosmos3-tao-win.txt
# then x64 sections 9 and 11 (Cosmos-RL build, TAO integration layer)
```

The 2026-09-02 validation venv (contract `[validation.arm64]`) is the reference
for anything not listed here. The 104 x64 pins were not re-audited one by one
on ARM64: if another pin fails to resolve, record it as a W16 line and marker
it out — do not build from source on a demo host.

## Decode and validation scope

Do not set `COSMOS_ALLOW_CPU_VIDEO_DECODE` for the GPU route. The WTS reader
requests `device="cuda"` and rejects CPU fallback. Set
`FORCE_QWENVL_VIDEO_READER=torchcodec` for native direct inference.

The optional reproducible WTS fixture, its layout, command, measured decode
comparison, and lifecycle evidence live in `cosmos3-wts-validation.md`. It is
not a requirement for a customer dataset or a claim of general model quality.

The completed native ARM64 coverage is a 500-step, two-epoch LoRA run, DCP
checkpoint, merged HF export, direct Cosmos-RL inference, and one-item
Framework evaluation through TorchCodec. The WTS reference records the exact
fixture-specific evidence. It does not establish model quality for WTS or any
other dataset.

## vLLM serving

vLLM is not part of TAO training, export, direct inference, or evaluation. It
is an optional serving consumer of the exported HF artifact. Native ARM64
serving is validated with a wheel built from the public Windows fork; follow
`cosmos3-vllm-serving-windows.md` for the source build, install,
launch, request examples, dynamic-LoRA boundary, and measured limitations.

Use the portable `0.26.0+cu134` wheel by default. The Spark-only
`0.26.0+cu134.sm121a` wheel also passed, but its measured single-sequence decode
throughput was equivalent and does not justify making a machine-specific wheel
the normal route. Keep the public NVIDIA WoA Torch nightly index in the setup:
the vLLM wheel supplies vLLM, not PyTorch. Install `openai-harmony==0.0.8`
(built from source on ARM64 if no win_arm64 wheel is published); this
satisfies upstream vLLM's Responses dependency without a Windows-specific API
patch. Cosmos3 itself uses the standard, non-Harmony Responses context.

## Remaining boundaries

For the validated direct actions, retain these source changes:

- `TAO_DISABLE_CLOUD_STATUS_CALLBACK=1` prevents local evaluate/inference from
  importing optional cloud orchestration clients merely to emit status.
- Select Matplotlib's `Agg` backend before importing `pyplot`; the ARM64 Python
  install has no Tcl/Tk GUI runtime and evaluation only saves plots.
- Keep `CosmosInferenceMicroservice` as a lazy Cosmos-RL import. Direct CLI
  inference does not need OpenCV; the optional server does.

At the validated point, CPython 3.13 ARM64 PyPI wheels were unavailable for
`pyarrow`, `obstore`, `grpcio`, and `opencv-python-headless`. The ARM64 vLLM
wheel omits or isolates features requiring those unavailable packages, so
those optional integrations remain unsupported even though the OpenAI HTTP
server works. The packages do not block the validated direct video route. The
ABI3 `cryptography==46.0.3` wheel is a working ARM64 substitute where newer
versions attempt an OpenSSL source build.
