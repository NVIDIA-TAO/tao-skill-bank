# Windows-specific gaps, W1-W18

Everything here fails **because the host is Windows**. Each entry was hit on a real
box; the contract encodes the ones that are machine-checkable.

## Not owned by this file

Six traps that produce a wrong answer at exit 0 are **OS-independent** — they
reproduce identically on Linux, and filing them here would put the fix in the
wrong place:

| Trap | What it is | Owner |
|---|---|---|
| T1 | `answer_type = "letter"` clamps `max_tokens` to 10 and reports `Accuracy: 0.0000` at exit 0 | `skills/models/tao-finetune-cosmos-reason/references/cosmos-reason-gaps.md` |
| T2 | `num_frames` never reaches the decoder; bound with `fps` and `total_pixels` | same |
| T3 | `--media <dir>` does not recurse; pass a single file | same |
| T4a | `TAO_API_JOB_ID` is mandatory for every action and appears in no `--help` | same |
| T5 | `checkpoint.load_path` is ignored when `latest_checkpoint.txt` exists | same |
| T6 | export without `--base-model-path-or-uri` still contains `lora` tensors and serves as the baseline | same |

Only **T4b** below belongs to Windows.

Two more accepted-but-ignored settings are **ours**, because it is this bank's
Windows recipe that makes them inert. Both are filed where they are used, not
here:

| Setting | What happens | Filed in |
|---|---|---|
| `trainer.max_iter=<n>` | accepted, unlogged, ignored — we passed 6 and the run executed all 20 iterations, because `wts_vlm_windows.toml` sets `cycle_lengths = [20]` | [`cosmos3-windows-overrides.md`](cosmos3-windows-overrides.md), contract `[overrides.inert]` |
| `WEIGHTED_SCORE: 0.000000` | printed by a **successful** evaluate; not a failure and not a metric | below |

---

## W1 — the default video decoder cannot start in a container

`cosmos-framework-evaluate` registers PyNvVideoCodec, which probes for the NVENC
**encoder** library at initialisation even though it only ever decodes. The
Windows GPU device class projects `nvcuvid.dll` (NVDEC) into `System32` but **not**
`nvEncodeAPI64.dll` — verified by listing both host and container:

```
RuntimeError: NVENC library file is not found. Please ensure NV driver is installed
```

So evaluate cannot run in **any** Windows container as shipped. `video_decoder`
looks like a knob but stock builds raise on every value except the default, so
there is no config-only escape; selecting `torchcodec` requires the port patch.
`cosmos-framework-inference` is unaffected — it uses qwen-vl-utils' default
backend.

**Do not bake `nvEncodeAPI64.dll` into the image.** Driver libraries must match
the host driver, so a baked-in copy breaks on every other machine.
Contract: `evaluate.video_decoder`, `image.bake_driver_dlls = false`.

Measured accuracy is identical either way (0.4118 host and container), so this
costs nothing but the flag.

## W2 — DCP checkpointing on a bind mount destroys the run after it succeeds

torch DCP and safetensors both write a temp file and atomically rename it. On a
Windows bind mount that rename fails with `WinError 3` on
`.metadata.tmp -> .metadata` — **after** training has completed and printed its
final loss. Ordinary files (`config.yaml`, `status.json`, `launch_info.yaml`)
land fine, which makes the output directory look almost right.

Keep the output directory container-internal, create the container **without
`--rm`**, `docker cp` the result out, then remove the container.
Contract: `runtime.output_dir_must_be_container_internal`.

Reads through a bind mount are fine: a 17.5 GB model bind-mounted read-only
`from_pretrained`s and mmaps without trouble, and measured *faster* than the same
weights baked in with `docker commit` (11.06 s vs 17.25 s to load 16.79 GiB). The
hazard is on the write path only.

## W3 — `--memory` faults instead of erroring

A cap below roughly 2x model size gives `0xC0000005` after "Done with training."
prints: DCP stages the whole model through host memory on the way out. Do not set
`--memory` on a training container.
Contract: `runtime.memory_limit_min_multiple_of_model_size`.

## W4 — Server Core has no VC++ runtime

`WinError 126` loading `c10.dll`. `COPY` `vc_redist.x64.exe` into the build
context and run it `/quiet`. Copy it, never download it: the base image's CA
store is empty until certifi is wired up, so the download fails with a TLS error
that looks like a network problem.

## W5 — `LongPathsEnabled=0` kills pip mid-install

pip dies at 262 characters partway through a large dependency set, leaving a
half-installed environment. Set the registry key **before any pip step**, not as
a fix afterwards.

## W6 — a failed `RUN` does not fail a Windows docker build

`$ErrorActionPreference` does not apply to native executables, so a `RUN` step
whose command exits non-zero still produces a layer and the build reports
success. Check `$LASTEXITCODE` after **every** native command. This is how a
broken image ships looking green.

## W7 — inline `python -c` inside `RUN` gets mangled

PowerShell rewrites the quotes and semicolons. Put the code in a `.py` or `.pth`
file and `COPY` it in.

## W8 — backslashes are eaten under a bash-family shell

`C:\gpu_check.py` arrives as `C:gpu_check.py`, i.e. relative to `WORKDIR`, and
the failure reads as a missing file. Use forward slashes for container paths
(`C:/gpu_check.py`). The same rule applies to `-e CUDA_HOME=C:/cudashim` (see
W12), where the mangled value silently becomes bare `C:`.

## W9 — changing `WORKDIR` breaks training only

The documented training command passes a **relative** recipe path, so a stage-2
`WORKDIR` on top of the image gives
`FileNotFoundError: 'examples/toml/sft_config/wts_vlm.toml'`. Evaluate and
inference keep working because they resolve absolute paths from config, so only a
training regression test catches it. Preserve `WORKDIR C:/cosmos-framework`.
Contract: `runtime.workdir`.

## W10 — there is no cgroup or MIG equivalent

No per-user GPU quota, no isolation, no scheduler on a shared Windows box. Two
users starting jobs on the same GPU simply collide. Serialize GPU stages
yourself; there is nothing to configure.
Contract: `runtime.gpu_quota_supported = false`.

## W11 — query health is not allocation health

`torch.cuda.is_available()`, `get_device_name()` and `nvidia-smi` can all pass
while a 4-byte allocation segfaults. A preflight built on the query APIs reports
a healthy GPU on a box that cannot train. Probe with a real allocation plus a
bf16 matmul.
Contract: `runtime.gpu_probe_must_allocate = true`.

## W12 — vLLM serving needs a `CUDA_HOME` shim, and the error names the wrong variable

Serving an exported checkpoint dies during **model loading**:

```
File "vllm\v1\attention\backends\flashinfer.py", line 12, in <module>
ValueError: CUDA_LIB_PATH is not set. CUDA_LIB_PATH need to be set with the
absolute path to CUDA root folder on Windows
```

Three non-obvious facts:

1. `VLLM_ATTENTION_BACKEND=FLASH_ATTN` does not avoid it. The import is reached
   from `model_loader/base_loader.py` -> `load_model`, i.e. during model loading,
   not attention-backend selection.
2. `VLLM_USE_FLASHINFER_SAMPLER=0` does not avoid it either. That flag governs the
   *sampler*; the log confirms it is honoured *after* the failing import would
   have run.
3. Set **`CUDA_HOME`**, not the `CUDA_LIB_PATH` the error names. flashinfer uses
   `CUDA_HOME` / `CUDA_ROOT` / `CUDA_PATH` verbatim but treats `CUDA_LIB_PATH` as
   `<root>\lib\x64` and walks up two levels, so `CUDA_LIB_PATH=C:/cudashim`
   becomes `C:\` and it looks for `C:\bin\cudart64_13.dll`.

No CUDA toolkit is required. flashinfer only wants
`<root>/bin/cudart64_<major>.dll`, where major comes from `torch.version.cuda`,
and that DLL already ships in the image under `torch/lib/`. Only the directory
shape is missing, so the fix is a two-line shim plus `-e CUDA_HOME=C:/cudashim`
(forward slashes, per W8).

Healthy startup logs `Model loading took 16.79 GiB and 11.06 seconds` and
`Application startup complete.`

## W13 — `WinError 10013` on bind

Either a port collision or a Hyper-V reserved range. Check the reserved ranges
before assuming the port is in use:

```
netsh interface ipv4 show excludedportrange protocol=tcp
```

Related: Windows containers cannot publish to a specific host IP —
`-p 127.0.0.1:8000:8000` fails with "Windows does not support host IP addresses in
NAT settings". A published port binds every interface, so treat any published
service as reachable from the network.

## W14 — training completes but learns nothing under `sdpa`/`eager`

The trainer's `training_step`/`validation_step` unconditionally
`data.pop("attention_mask", None)` (`model/generator/vlm_model.py`). The
custom varlen kernels (`cosmos`, flash-attn) never read that mask, so on
Linux the pop is harmless. The HF-native `sdpa`/`eager` paths — used by the
former Windows `attn_implementation=sdpa` override and now retained only for
debugging with the W14 gate — DO need it: with `attention_mask=None`
transformers 4.57.6 feeds the M-RoPE temporal position row into
`create_causal_mask`, whose packed-sequence detection reads the constant
temporal index across each image span as fake sequence boundaries. The
answer token is walled off from the image and prompt, and the model
converges to the label prior at exit 0. Evaluation is immune (mask
supplied on that path), which is what makes this trap invisible to every
post-training check. Fixed upstream in transformers v5.3.0 (HF PR #44158);
no 4.x backport exists.

Fix at the trainer: gate the pop on the attention implementation — keep the
mask for `sdpa`/`eager`, keep popping it for `cosmos`/flash (the `cosmos`
adapter raises on a non-None mask). One `if` at both pop sites. Validated
2026-08-27: in-training val loss on the all-NG head 0.669 → 0.0226 at epoch 1.
Contract: `[validation.train] learns`.

Upstream status (checked 2026-09-16): cosmos-framework fixed this in its
Release 2026-08-14 (GitHub NVIDIA/cosmos-framework #198, commit `18d90ea`).
`training_step`/`validation_step` now forward `attention_mask` instead of
popping it, and no longer precompute `position_ids` (the model derives mRoPE
positions itself, which also removes the fabricated boundaries). The Windows
pin in this skill is built on the PR #164 base, which predates that release,
so the mask-gate patch is still required on the pin and nowhere else. When the
Windows port is rebased onto that release or later, drop
`w14-attention-mask-gate.patch` and this trap becomes history. The cosmos-rl
SFT path (`hf_models`, HF-wrapped Qwen3-VL) still passes precomputed 3-row
mRoPE `position_ids` without an `attention_mask` as of its 2026-09-16 main;
that is a separate code base and a separate fix.

## W15 — an orphaned checkpoint writer poisons every later training run on the host

cosmos-framework saves DCP checkpoints asynchronously through a background
process spawned at trainer init, which binds its own gloo store on
**`MASTER_PORT + 2`** (`checkpoint/dcp.py`, 29502 with the default 29500).
If any previous run's writer is still alive — an aborted run's orphan, or a
run hung at teardown — the new run's writer dies at init with
`DistNetworkError ... failed to bind ... (system error: 10048)` buried in
the log. Training then proceeds **normally for the entire run**, prints
"completed successfully", logs "Submitted checkpoint to background process",
and blocks forever. The checkpoint directory contains only `dataloader/`;
the weights die with the process.

The hang itself creates the next orphan, so one aborted run poisons every
subsequent run on the host until the port is cleared (observed 2026-08-27:
a 10:06 abort orphaned its writer; the 12:05 and 12:12 runs both trained
fully — 30 GPU-minutes each — and hung with empty checkpoint dirs).

Before every training launch: `netstat -ano | findstr :29502` must come back
empty; kill any squatter **by exact PID**. Never run two framework trainings
concurrently on one host — they share the fixed port. The tell after the
fact: "completed successfully" in the log but the process never exits, GPU
memory still held, `epoch_N/` holding only `dataloader/`.

## W16 — pyarrow, obstore, and grpcio ship no Windows ARM64 wheels

`pip install -r requirements-cosmos3-win.txt` on an ARM64 interpreter fails at
`pyarrow==25.0.1`: PyPI has never published a win_arm64 pyarrow wheel and
conda-forge has no win-arm64 build (checked 2026-09-06 against both indexes;
Arrow issue #47195 is open and PR #48539 is nightly-first and explicitly
non-release-blocking). `obstore` and `grpcio` are in the same position, and
`cryptography` newer than 46.0.3 attempts an OpenSSL source build on ARM64.

pyarrow's only consumer here was the mining stage, purely as parquet I/O.
`parquet_io.py` (vendored into both the DEFT and mining skill `scripts`
directories; a test keeps the copies byte-identical) uses pyarrow where it
imports and `duckdb` otherwise — duckdb ships a cp313 win_arm64 wheel. The two
backends write byte-identical schemas, verified on every artifact of a real
run including `list<float32>` embeddings and the empty table `pool_exhausted`
produces, and the full mining stage was exercised end-to-end with pyarrow
hidden. Both requirement files carry `platform_machine` markers for the four
packages, so the x64 files install unchanged on ARM64. Object-store checkpoint
backends (`obstore`) stay unsupported there; `grpcio`'s optional cloud status
clients are already disabled by `TAO_DISABLE_CLOUD_STATUS_CALLBACK=1`.
The same wheel gap makes cosmos-rl's evaluator unimportable (`text_metrics.py`
imports `evaluate` -> `datasets` -> pyarrow at module scope); apply
`cosmos-rl-evaluate-import-guard.patch` (see `cosmos3-windows-source-patches.md`).

## W17 — the Windows AOI recipe's `[dataloader_train]` block trains one sample per step

`aoi_vlm_windows.toml` (and `wts_vlm_windows.toml`, which it was cloned from)
ships `max_samples_per_batch = 1` / `max_sequence_length = 81920` under
`[dataloader_train]`. The structured-TOML loader remaps those two fields onto
the VLM CosmosDataLoader's PoolPackingBatcher (`batcher.max_batch_size`,
`batcher.max_tokens`), so with the recipe unchanged every optimizer step
consumes exactly one record (`input_ids torch.Size([1, 128])` in the
`log_tensor_shape` callback) while the DEFT driver sizes the run as
`trainer.steps_per_epoch = ceil(records/8)`. Each "epoch" therefore sees an
eighth of the assembled set, the cosine schedule anneals over an eighth of the
intended samples, and the run exits 0 looking normal.

Observed on the RTX Spark N1X, 2026-09-08, 2-epoch learning check on 1,600
balanced wafer records (800 NG first), `cosmos` attention, 7-module LoRA
r16/a32/d0.05, lr 1e-4 annealed over 410 steps:

| `max_samples_per_batch` | batch shape | val loss epoch 1 (all-NG head) | val loss epoch 2 | wall s/step |
|---|---|---|---|---|
| 1 (recipe as shipped) | `[1, 128]` | 0.4030 | 0.0705 | 0.72 |
| 8 (validated wafer config) | `[8, 128]` | 0.1689 (9/10 samples ≤ 0.065, median 0.009; one label-noise record at 1.52) | 0.1691 (9/10 ≤ 0.036, median 0.007; same record at 1.58) | 4.13 (≈3 s of it backward) |

The trainer *is* learning in the first row (train loss falls, 6/10 val samples
at p(NG) > 0.99), so this is not W14 — W14 reads as a flat ~0.67 plateau on a
50/50 set. It is simply eight times less data per epoch than the validated
configuration. The one high-loss validation record in both rows is the same
file: it sits in the export's `defect/` folder (label NG) but its filename
class token is `good`, and the model confidently answers OK — the wafer
export's folder/filename disagreement is documented in the rig README; do not
read a single such sample as a trainer regression. Put `max_samples_per_batch = 8` (and a sane
`max_sequence_length`, 8192 matches the experiment's `data_setting.max_tokens`)
in the run spec; the DEFT skill's `prepare_roboflow_classification_aoi.py`
workspace builder does so.
Verify from the first `[Tensor Shape] Iteration 0` line of the training log,
not from the TOML.

## W18 — `cosmos-rl evaluate` refuses to start without PyNvVideoCodec on a venv that never needed video

`cosmos_rl/evaluation/base.py::load_model` registers a video decoder before it
loads the VLM. The default `video_decoder = "pynvvideocodec"` calls
`register_pynv_video_reader(strict=True)`, which imports `PyNvVideoCodec` and
raises `GPU video decoding requires PyNvVideoCodec and qwen-vl-utils` when the
module is absent — for an images-only AOI spec too. The x64 native venv had the
package; the ARM64 TAO venv provisioned for training did not, and the run died
at the baseline benchmark leg with exit 1 (RTX Spark N1X, 2026-09-08).

Resolution on Windows ARM64: install the `PyNvVideoCodec==2.2.2` win_arm64 wheel
(`pip install --no-deps PyNvVideoCodec==2.2.2`; no dependencies) into the TAO venv. Verify with the evaluator's own call,
`register_pynv_video_reader(cache_size=0, video_override_map=None, strict=True)`.
Config-only alternative when no wheel is available: `[vision] video_decoder =
"torchcodec"` in the evaluate spec (W1's container fix; qwen-vl-utils then pins
torchcodec). Details and package provenance: the DEFT skill's
`windows-arm64-native.md`, item 13.

---

## T4b — the status file lands drive-relative

This is the Windows half of the `TAO_API_*` trap. With `TAO_API_JOB_ID` set and
no `TAO_API_RESULTS_DIR`, the compiled-in POSIX default `/results/<job>/status.json`
resolves **drive-relative** to `C:\results\<job>\status.json` instead of into
`results_dir` — so an orchestrator polling the job directory never sees the file
and the job reads as hung. Setting both variables is the only configuration that
satisfies every entrypoint *and* puts the status file where it belongs.

The same drive-relative rule explains `IMAGINAIRE_OUTPUT_ROOT`
(`/tmp/imaginaire4-output` -> `<drive>:\tmp\imaginaire4-output`). A leading slash
is never an error on Windows, only a different place.

---

## Observed artifact — `WEIGHTED_SCORE: 0.000000` on a run that PASSED

Not a W-numbered trap: it is a reporting artifact, and nothing establishes that
it is caused by Windows. It is recorded here because this is where a Windows
reader meets it. Confirmed observed 2026-08-13 on the native evaluate run that
succeeded. The same stdout carried, in this order:

```
Overall accuracy: 0.4118   Total samples: 17   Correct samples: 7
WEIGHTED_SCORE: 0.000000
Evaluation completed successfully in 148.6 seconds
```

exit code 0, runner status `COMPLETE`. The zero is an **artifact of the reporting
line**, not a score, not a failure, and not something to debug. It is dangerous
for exactly one reason: it looks like the `Accuracy: 0.0000` symptom of the
`answer_type = "letter"` trap (T1), and someone chasing it will "fix" a working
configuration.

Tell the two apart by the accuracy line, which is the metric that matters:

| | `answer_type = "letter"` (broken) | this (fine) |
|---|---|---|
| `Overall accuracy` | `0.0000` | a real number, e.g. `0.4118` |
| `WEIGHTED_SCORE` | `0.000000` | `0.000000` |

**Report `overall.accuracy`.** Never gate, alert or summarize on
`WEIGHTED_SCORE`. Contract: `[evaluate].weighted_score_line_always_zero`,
`[evaluate].report_metric`, and the run itself in `[validation.evaluate]`.

---

## Environment gaps, not code

| Gap | Workaround |
|---|---|
| Corporate policy blocks the python.org MSI (`exit 1625`) | `python-build-standalone` tarball: extract, no installer, no admin |
| FFmpeg "essentials" builds ship **no DLLs** | use a **shared** build; install **both 8.1 and 7.1**, 8.1 first — PyNvVideoCodec needs `avformat-62` (8.x) and torchcodec is happy on 8.1 |
| Since Python 3.8, Windows extension modules do not resolve dependent DLLs from `PATH` | `os.add_dll_directory()` must run at interpreter startup — a one-line `.pth` in site-packages |
| `PyNvVideoCodec_130.pyd` (a CUDA 13 build) links `cudart64_12.dll` | `pip install nvidia-cuda-runtime-cu12` — not redundant, found with `dumpbin /DEPENDENTS` |
| `uv sync` refuses to resolve | `pyproject.toml` declares `required-environments = ["sys_platform == 'linux' ...]`; hand-carry the dependency set |
| `nvidia-tao-core` hard-requires uWSGI, which cannot build on Windows | uWSGI's `uwsgiconfig.py` calls `os.uname()` on its first executed line and fails at the **metadata** stage, where no wheel or build flag helps. `pip install --no-deps ./tao-core` plus a curated requirement list. uWSGI serves only the microservices HTTP app. |
| torch must be exactly `2.13.0+cu130` | it is the only line shipping cuDNN 9.20, and `CUDNN_MIN_BACKEND_VERSION = 92000`. On cuDNN 9.19 the cuDNN attention backend disables itself and falls back to NATTEN, which has no Windows build. |

Install ordering and the exact wheel commands belong to
`skills/platform/tao-run-on-windows-venv`; the image build steps belong to
`skills/platform/tao-run-on-windows-docker`.

## One failure mode worth naming twice

The Linux SONAME load (`ctypes.CDLL("libnvcuvid.so.1")`) did not surface as a
decoder error. It made NVDEC registration fail, which silently demoted
qwen-vl-utils to its `torchvision` backend — and torchvision >= 0.23 removed
`torchvision.io.read_video`. The reported error was
`module 'torchvision.io' has no attribute 'read_video'`, three layers away from
the cause. On this platform, assume the reported error names the last thing to
break, not the first.

## Latent, lower severity

| Site | Problem |
|---|---|
| `data/reasoner/local_sft_dataset.py` | `open(path, "r")` with no `encoding=` reads cp1252 on Windows. A pure-ASCII slice hides it; a full annotation set will not. |
| `videophy2_sft_nano.py` | dataloader defaults `num_workers=2, persistent_workers=True` — the reason overrides 8-13 exist. |
| DCP async checkpointing | the background process binds `MASTER_PORT + 2` on the hostname rather than `MASTER_ADDR`; the exception is swallowed, so async checkpointing is silently dead. |
| `policy_config.py` | `trainable_map` is declared and never referenced — a dead knob that misleads anyone hunting for a partial-fine-tune mechanism. |
