# Cosmos3 vLLM serving on native Windows

Use this optional route after a Cosmos Framework export when an
OpenAI-compatible HTTP service is required. TAO train, export, evaluate, and
direct inference do not call vLLM; they continue to use Cosmos Framework,
Cosmos-RL, and Hugging Face Transformers.

## Artifact boundary

For ordinary serving, pass the verified merged HF directory produced by the
Framework exact-key exporter. It must contain `config.json`, tokenizer and
processor files, `model.safetensors.index.json`, all referenced shards, and
zero keys containing `lora`. A DCP checkpoint is not loadable by vLLM.

For dynamic LoRA, start vLLM with the public/base HF model and a standard PEFT
adapter directory containing `adapter_config.json` and
`adapter_model.safetensors`. The merged Framework export is not an adapter. An
adapter-only export is a separate packaging operation and must preserve the
same rank, alpha, target modules, and base-model revision as training.

Standard vLLM accepts text, image, and video input for Cosmos3 Reason and emits
text/tool tokens. It does not emit generated images or videos for a Cosmos
generator. `vllm-omni` and the Cosmos inference-microservice action are outside
this validated route.

## Windows x64

Use the third venv from `cosmos3-windows-x64-host-setup.md`; it is separate
because the validated SystemPanic wheel pins Torch 2.11 while the TAO action
environment requires Torch 2.13 and cuDNN 9.20.

The command below is the exact validated x64 bring-up profile. It intentionally
uses `--enforce-eager` to minimize compiler and CUDA-graph variables during
compatibility validation; it is not a production-performance recommendation.

```powershell
$PY = "C:\cosmos-win\venv-vllm\Scripts\python.exe"   # the third venv; contract [validation.vllm_serving].venv
& $PY -m pip install `
  C:\path\to\vllm-0.26.0+cu132-cp312-cp312-win_amd64.whl `
  --extra-index-url https://download.pytorch.org/whl/cu130

New-Item -ItemType Directory -Force C:\cudashim\bin | Out-Null
Copy-Item C:\cosmos-win\venv-vllm\Lib\site-packages\torch\lib\cudart64_13.dll `
  C:\cudashim\bin\
$env:CUDA_HOME = "C:/cudashim"
$env:VLLM_USE_FLASHINFER_SAMPLER = "0"
& $PY -m vllm.entrypoints.openai.api_server `
  --model C:\models\Cosmos3-Nano-VLM `
  --served-model-name cosmos3 --host 127.0.0.1 --port 8421 `
  --dtype bfloat16 --max-model-len 2048 --enforce-eager
```

The CUDA shim supplies only the directory shape expected by FlashInfer; it is
not a CUDA Toolkit installation. Copy the DLL-directory `.pth` described in
the x64 host reference into this venv as well. For a production candidate,
omit `--enforce-eager`, then validate compiled startup, the required request
modalities and endpoints, clean shutdown, and representative throughput before
adopting that profile. Compiled x64 execution has not yet been validated or
benchmarked here.

Validated on x64 to date: model load, `/v1/models`, text `/v1/chat/completions`
(2026-08), and — with the eager profile above plus `--max-model-len 16384`,
`--limit-mm-per-prompt '{"image": 8}'`, `--enable-prefix-caching`,
`--max-num-seqs 1` — **image input at one to four images per request**, 512
requests with zero errors (2026-09-10).
Three things that run taught, all of which apply to any x64 image serving:

- The OpenAI request carries no per-image pixel budget. If a workload specifies
  one, pin it server-side with `--mm-processor-kwargs
  '{"min_pixels": N, "max_pixels": N}'`; the checkpoint's
  `preprocessor_config.json` leaves both `null`. 1,048,576 px is 1,024 visual
  tokens per image — check `usage.prompt_tokens`.
- vLLM adopts the checkpoint's `generation_config.json` (temperature 0.7,
  top-k 20, top-p 0.8) as request defaults; a greedy run must send
  `temperature: 0` explicitly.
- Beside other servers, size `--gpu-memory-utilization` to *free* memory, not
  total: vLLM budgets `utilization × total` and refuses to start when that
  exceeds what is free. 0.30 (≈29 GB) held the Nano at 16k context with 4-image
  prefill while two other servers occupied 61 GB.

Do not describe x64 video, Responses, dynamic-LoRA, or compiled execution as
validated from the ARM64 results.

## Windows ARM64

Use native CPython 3.13 ARM64 and install NVIDIA's matching WoA Torch nightly
before vLLM. There is no public win_arm64 vLLM wheel yet: build vLLM from the
public Windows fork (`vortex-captain/vllm-windows`, branch `v029_win_arm64`)
following that repository's build notes, then install the wheel you built:

Building vLLM itself is not Cosmos3-specific. If no published wheel matches the
required Python, Torch/CUDA, vLLM revision, or GPU architecture, follow
`skills/platform/tao-run-on-windows-venv/references/vllm-windows-arm64-build.md`,
then return here for the Cosmos3 artifact, media, serving, and validation
contract. Do not copy the generic compiler procedure into this model reference.

```powershell
$PY = "<arm64-vllm-venv>\Scripts\python.exe"
& $PY -m pip install --pre --upgrade --only-binary=:all: `
  --extra-index-url "https://pypi.nvidia.com/nvtorch_oot_nightly" `
  torch torchvision torchaudio

git clone -b v029_win_arm64 https://github.com/vortex-captain/vllm-windows <src>\vllm-windows
# build per the fork's Windows instructions (CUDA 13.x toolkit, MSVC, ARM64 host), then:
& $PY -m pip install <src>\vllm-windows\dist\vllm-*.whl
& $PY -m pip install "openai-harmony==0.0.8"      # build from source if no win_arm64 wheel is published
& $PY -m pip install --only-binary=:all: "winloop==0.6.3"
```

Authenticate to the private GitLab index with an approved PAT, deploy token,
CI job token, or credential helper. Never put a token in this reference, a
command history committed to Git, or pip configuration checked into a repo.

The portable wheel is
`vllm-0.26.0+cu134-cp313-cp313-win_arm64.whl`, SHA-256
`4bcfed901989462c50abc7c7a177b0bee2013ce597b1087e4fa7b0dd5c1f5554`.
It was installed and validated with Torch `2.15.0.dev20260902+cu134` from the
public NVIDIA WoA nightly index. `winloop` is explicit because this published
wheel predates the Windows ARM64 `winloop` wheel; the next vLLM wheel build can
restore it as an ordinary dependency. The optional exact-Spark package is
`vllm==0.26.0+cu134.sm121a`, SHA-256
`4f853e4eeff79efe013d3a9e57972d161e90a32ff1f799edea0fc3e56a50c959`;
use it only for controlled architecture comparisons. Measured 2026-09-12 to
09-16 on the N1X: the portable `+cu134` wheel already carries the CUTLASS FP8
and NVFP4 kernels for SM 12.1 (`CutlassFP8ScaledMMLinearKernel`,
`CutlassNvFp4LinearKernel` are selected for ModelOpt and online-FP8
checkpoints), so quantized serving does not need the `sm121a` build. The
earlier assumption in this bank that only `sm121a` could run them was wrong.

The private index also carries the unmodified upstream OpenAI Harmony v0.0.8
wheel `openai_harmony-0.0.8-cp38-abi3-win_arm64.whl`, SHA-256
`6a1b3a74cca34aeb29a634ae04069766d7435446f05aed33feb85a8c434bfca3`.
It was built from the official v0.0.8 tag with native ARM64 Rust, MSVC, and
LLVM/Clang 22.1.8, then passed all 42 upstream tests on CPython 3.13 ARM64.
Because this wheel now satisfies vLLM's ordinary dependency, future vLLM
builds should keep upstream Responses imports unchanged rather than carrying
the broader optional-Harmony compatibility patch used during initial bring-up.

Install PyNvVideoCodec 2.2.2 for the preferred in-memory NVDEC path. For the
alternate TorchCodec CPU/CUDA backend, build and configure TorchCodec for
ARM64 exactly as described in `cosmos3-windows-arm64.md`.

### ARM64 execution profiles

For normal, production-oriented ARM64 serving, omit `--enforce-eager` as in
the command below. This allows the validated TorchInductor/CUDA-graph path.
It can improve steady-state throughput for some workloads, but it adds
compilation and graph-capture startup work and is not guaranteed to outperform
eager execution for a single sequence. Benchmark the representative request
length, modality, batching, and concurrency before choosing a production
profile.

Start a merged-model server with compiled execution enabled:

```powershell
$env:VLLM_USE_FLASHINFER_SAMPLER = "0"
& $PY -m vllm.entrypoints.cli.main serve `
  C:\models\Cosmos3-Nano-VLM `
  --host 127.0.0.1 --port 8421 --served-model-name cosmos3 `
  --dtype bfloat16 --max-model-len 2048 --max-num-seqs 1 `
  --kv-cache-memory-bytes 4G --disable-custom-all-reduce `
  --attention-backend FLASH_ATTN `
  --media-io-kwargs '{"video":{"num_frames":4,"video_backend":"pynvvideocodec","hw_decoders":2}}'
```

Add `--enforce-eager` only for initial bring-up, deterministic lifecycle or
signal-handling validation, diagnosing a compiler/CUDA-graph failure, or
temporarily working around one. Eager mode is a validated fallback, not a
Cosmos3 or Windows-on-ARM64 requirement.

To use TorchCodec instead, replace the media kwargs with:

```json
{"video":{"num_frames":4,"backend":"torchcodec","torchcodec_device":"cuda"}}
```

For dynamic LoRA, serve the base model and append:

```powershell
--enable-lora --max-loras 1 --max-lora-rank 16 `
--lora-modules cosmos-lora=C:\models\adapters\cosmos-lora
```

Then address the adapter as model `cosmos-lora`; requests to the base model remain
available while the adapter is active.

## Request checks

Use an ordinary Chat Completions request for both architectures:

```powershell
$body = @{
  model = "cosmos3"
  messages = @(@{ role = "user"; content = "Reply with exactly: ready" })
  max_tokens = 16
  temperature = 0
} | ConvertTo-Json -Depth 8
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8421/v1/chat/completions `
  -ContentType application/json -Body $body
```

On ARM64, the standard Responses endpoint was also validated:

```powershell
$body = @{ model = "cosmos3"; input = "Reply with exactly: ready"; max_output_tokens = 16 } |
  ConvertTo-Json -Depth 8
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8421/v1/responses `
  -ContentType application/json -Body $body
```

Cosmos3 uses vLLM's standard Responses context, not the Harmony parser. Harmony
is selected for model configurations that declare that protocol (notably the
GPT-OSS family), but the upstream vLLM package imports and declares Harmony even
when a served model does not use it. Before the ARM64 Harmony wheel existed,
Vortex's source contained `/v1/responses`, but a normal dependency install could
not resolve and those eager imports could fail. The wheel fixes that packaging
gap without changing the endpoint.

For image and video requests, use OpenAI multimodal content parts with a data
URL or reachable URL. Keep a labeled clip and assert semantic content in the
reply; a 200 response alone does not prove the decoder was used correctly.

## Validated ARM64 scope and limits

The portable wheel passed Cosmos3 text, image, video, Chat Completions,
standard Responses, eager execution, TorchInductor/CUDA graphs, and dynamic
rank-16 LoRA. PyNvVideoCodec direct-bytes decode and TorchCodec CPU/CUDA decode
all passed. The Harmony ARM64 wheel separately passed its upstream unit suite
and vLLM Responses-module import; a Harmony-model serving run was not part of
the Cosmos3 validation.

Measured median single-sequence decode was 13.86 tokens/s for portable eager,
13.72 for portable compiled, and 13.83 for exact SM121a eager. At 128 output
tokens, dynamic LoRA measured 11.79 tokens/s versus 12.79 for the base request
in the same server, about 8 percent overhead. These are same-host functional
comparisons, not Linux parity or multi-request throughput claims. The eager
and compiled single-sequence measurements do not establish a meaningful
performance advantage for either profile; use a representative concurrent
serving benchmark for that decision.

The wheel is private and experimental. It does not establish multi-GPU or
multi-node support, `vllm-omni`, media generation, every optional vLLM plugin,
or models requiring unavailable Windows ARM64 Triton kernels. Revalidate after
any Torch nightly, CUDA Toolkit, compiler, vLLM commit, or wheel change.

## RTX Spark N1X: measured serving facts and traps (2026-09-12)

Measured on the N1X with the validated ARM64 profile (`vllm 0.26.0+cu134`, torch 2.15 nightly, FLASH_ATTN, CUDA
graphs on, `--max-num-seqs 1`, `--kv-cache-memory-bytes 8G`, `--limit-mm-per-prompt {"image": 24}`,
`--enable-prefix-caching`).

- **Decode speed and the GPU power state.** Nano BF16 decodes at ~6.0 tok/s text-only and ~5.6 tok/s on a
  4.1k-token 9-image prompt (256-token streaming runs, temperature 0, `ignore_eos`), identical at 8k and 16k
  context and with a second idle server resident. That is ~40% of the LPDDR5x bound (17.5 GB weights per token at
  ~273 GB/s gives ~15 tok/s; 13.9 tok/s has been measured on this class of box). One-second `nvidia-smi` samples show
  the GPU 90-96% busy but in **P8 at 430-900 MHz and 4-6 W** throughout decode; the same GPU runs 1.5-1.6 GHz at
  ~32 W under training and hit 2.1 GHz during torch.compile. Decode alone does not make the driver ramp clocks. A
  clock lock or high-performance power plan is the first thing to test before any quantization or Super decision.
- **Online FP8 works on SM 12.1 with this wheel.** `--quantization fp8` on the BF16 checkpoint selects
  `CutlassFP8ScaledMMLinearKernel for Fp8PerTensorOnlineLinearMethod`, loads (weights 78.9 s incl. quantization,
  compile 334 s cold) and decodes at 8.82 tok/s text / 8.05 tok/s multi-image (1.48x / 1.43x over BF16 measured
  at the same time), 19/20 agreement with BF16 on 20 wafer verdicts, correct spec6 episode in 76 s vs 115 s.
  Bytes per token halve but the gain is capped by the same P8 clock.
- **Co-resident servers need `--gpu-memory-utilization` lowered.** With one Nano engine resident, a second
  server fails vLLM's startup check: `Free memory on device (77.31/101.5 GiB) on startup is less than desired GPU
  memory utilization (0.92, 93.38 GiB)`. Because KV is fixed by `--kv-cache-memory-bytes`, the flag only gates
  that check; `--gpu-memory-utilization 0.60` on the second server is enough.
- **Fresh-process `torch.cuda.mem_get_info` is blind on this host.** It reports ~101.3 GiB free with nothing
  resident, with one engine resident, and with two. Use vLLM's own `Initial free memory` line (99.4 GiB empty,
  ~77-79 GiB with one Nano engine) and process memory (`Get-Process` WorkingSet64 / PagedMemorySize64: BF16
  engine WS ~20.8 GiB, commit ~28.8 GiB; online-FP8 engine WS ~20.2 GiB, commit ~21.9 GiB).
- **The EngineCore is a `multiprocessing.spawn` child under the base interpreter path**
  (`...\Python313-arm64\python.exe`), not the venv launcher. `Stop-Process` on the launcher or venv-path filters
  leave it alive holding ~20 GiB, the port may stay open on a surviving API server, and a readiness probe right
  after a relaunch can be answered by the old process. Stop a server by walking its process tree from the launcher
  pid (`taskkill /T` works only while the launcher is alive); never sweep every vLLM process on the box, that
  kills the reference server too.
- **ModelOpt reasoner checkpoints live under `nim/nvidia`, not `nvstaging/nim`:** `nim/nvidia/cosmos3-nano-reasoner`
  and `cosmos3-super-reasoner`, tags `bf16-final`, `modelopt-fp8-final_format_fix` (Nano 9.87 GiB, Super
  33.08 GiB), `modelopt-nvfp4-full-quantize-final_format_fix` (Nano 7.04 GiB, Super 20.37 GiB). HF format,
  `hf_quant_config.json` (producer modelopt 0.45, quant_algo FP8, kv_cache_quant_algo FP8, excludes `lm_head` and
  `model.visual*`), architecture Qwen3VLForConditionalGeneration. Without the NGC CLI the REST API works from a
  venv with `httpx`: mint a token at `authn.nvidia.com/token?service=ngc&scope=group/ngc:nim` with
  `Authorization: ApiKey`, then list `/v2/org/nim/team/nvidia/models/<name>/versions`; the `/files` and download
  endpoints additionally require the headers `nv-ngc-org: nim` and `nv-ngc-team: nvidia` or they answer 400 "Org
  context missing". Public Hugging Face has only the omni checkpoints (`nvidia/Cosmos3-Nano`, 32.6 GB, architecture
  `Cosmos3ForConditionalGeneration`, which this wheel registers as a Qwen3-VL subclass with a weight mapper) and
  community FP8/NVFP4 repos that quantize the generation transformer, not the reasoner.
- Speculative decoding is available in this wheel (ngram, suffix, EAGLE/EAGLE3, Medusa, MTP, draft-model
  proposers) but every server so far ran with `speculative_config=None`; it is the cheapest batch-1 lever on
  JSON-shaped output and is untested here.

- **ModelOpt checkpoints on the N1X (2026-09-12 evening).** Both `nim/nvidia/cosmos3-nano-reasoner` ModelOpt
  tags are auto-detected by this wheel (no `--quantization` flag needed) and select `CutlassFP8ScaledMMLinearKernel
  for ModelOptFp8LinearMethod` / `CutlassNvFp4LinearKernel for NVFP4 GEMM`. **They fail engine init unless
  `--kv-cache-dtype bfloat16` is passed:** the checkpoints carry `kv_cache_quant_algo: FP8`, vLLM then picks
  `kv_cache_dtype=fp8_e4m3`, and FLASH_ATTN here is FA2 ("FP8 KV cache requires FA3 on SM90 or FA4"). With the
  pin: FP8 weights load in 51.6 s and decode **13.61 tok/s** text / 11.47 tok/s on a 4.1k-token 9-image prompt
  (2.28x / 2.04x over BF16 measured at the same time), spec6 episode 62 s, 18/20 WAFER20 agreement with BF16;
  NVFP4 loads in 36.8 s and decodes 12.86 / 11.75 tok/s (no gain over FP8: the floor is now ~75 ms/token of
  clock-limited time, not bandwidth), 13/20 WAFER20 agreement (7 NG->OK shifts). Engine process: FP8 WS 21.1 /
  commit 22.2 GiB, NVFP4 WS 18.2 / commit 19.4 GiB. Predicted Super FP8 at this power state: ~7 tok/s text.

- **Super FP8 on the N1X (2026-09-14).** `nim/nvidia/cosmos3-super-reasoner:modelopt-fp8-final_format_fix` (33.08 GiB,
  64 layers x 5120) serves with the same profile plus `--kv-cache-dtype bfloat16`: weights load in 143 s, compile
  422 s, engine WS 44.7 GiB / commit 46.0 GiB, KV 32,768 tokens at 8 GB. Decode **2.45 tok/s** text / 2.28 tok/s
  multi-image (single client, Nano BF16 resident) — not the 7 tok/s a bytes-ratio estimate gives, because decode here
  is bound by the P8 clock floor, so time per token scales with compute (~3.7x Nano's 75 ms). One no-think spec6
  episode 262 s; a 40-image no-think read 135 min, **0.800 (32/40)**, audit 10/10; think episode (1024 cap) 549 s.
  Correct and servable, too slow for interactive use until the clock behaviour is fixed. While Super was busy,
  Windows trimmed the idle Nano engine's working set from 20 GiB to 0.45 GiB (commit unchanged); expect a slow first
  request on an idle co-resident server.

- **ROOT CAUSE of the slow decode (found 2026-09-15): a "Reliability" clock limiter after a GPU fault.** Windows
  System log shows nvlddmkm event 153 ("Error occurred on GPUID: f0100") at 2026-09-10 12:41:52 and 12:49:27 (the
  DEFT trainer died at the second one with `iter1_train failed rc=1`). Since then `clocks_event_reasons.active` =
  0x400 (Reliability) at idle and under load, and the SM clock stays at 656-1020 MHz / 6-8 W under ANY load: a bf16
  4096^3 matmul loop sustains 1.2 TFLOP/s; `nvidia-smi -lgc 3003` is accepted (idle rises to 2619 MHz) but under load
  the clock drops straight back to ~660 MHz and decode is unchanged (5.85 vs 6.28 tok/s). Before the fault the same
  GPU trained at P0, 1.5-2.3 GHz, 32-39 W (and 74-94 C). So every tok/s number above from 2026-09-10 onward is a
  degraded-GPU number; accuracy/agreement numbers are unaffected. Check `nvidia-smi --query-gpu=clocks_event_reasons.active
  --format=csv` before trusting any speed measurement on this box, but do not read the 0x400 bit alone as the fault:
  after the 2026-09-15 reboot the healthy GPU still reports 0x400 at idle and in 765 of 953 one-second samples under
  load while running 1.9-2.5 GHz and 42-44 TFLOP/s. The fault signature is 0x400 **plus** SM clock stuck at
  ~650 MHz under load (spinner ~1 TFLOP/s). If you see that, reboot first (no reboot had happened since 2026-09-02)
  and keep the event IDs and clock samples for follow-up.

- **Healthy-GPU numbers after the 2026-09-15 reboot (the fault-state numbers above are superseded).** Same profile:
  Nano BF16 **14.6 tok/s** text / 14.0 multi-image (97% of the 273 GB/s ceiling), spec6 episode 37.5 s, compile 38 s;
  Nano FP8 ModelOpt **24.5 / 22.7 tok/s**, episode 25.3 s; Super FP8 ModelOpt **6.4 / 6.2 tok/s**, episode 88 s,
  weights 82 s, compile 32 s. TTFT 0.23-0.44 s. GPU during decode ~1.9 GHz, 26-29 W, P0/P4. Agreement pre/post reboot
  19/20 (answers unchanged; only speed). Rule: always record `clocks_event_reasons.active` and a 20-second matmul
  TFLOP/s figure (`gpu_spinner.py`: ~44 TFLOP/s healthy, ~1 TFLOP/s in the fault state) next to any tok/s number.
- **Healthy-GPU 40-image spec6 no-think reads (2026-09-15).** Nano FP8: 0.450 5-way (14 min 26 s, median episode
  21.9 s, audit 10/10). Super FP8: **0.775** 5-way (31/40), binary 0.850, recall bridge/excess/missing/no_defect
  0.7/0.8/0.9/0.7, median episode 74.4 s, audit 10/10, 202/202 first-try JSON; the degraded-GPU read was 0.800, so
  the two Super reads sit inside the README's single-read spread. Trap: the machine hard-reset mid-read (Kernel-Power
  41, BugcheckCode 0, no minidump, no WHEA, no driver event; last server line 16:39:48, boot 16:53:17) at episode 27.
  The harness resumes (skips completed episodes) so the read finished with a plain re-launch of the same `--out`; a
  1 s `nvidia-smi` sampler during the 16-minute remainder showed SM mean 1.9 GHz (max 2.5), power mean 30 W (max
  41.6) and **GPU temperature up to 89 C** with no further reset. Fourth GPU-side incident under sustained load on this
  box in a week (nvlddmkm 153 on 09-08, 09-10 x2, event 41 on 09-15): run every multi-hour job with the sampler
  going and a resumable output dir, and keep the event history and temperature samples.
- **Thermal ceiling of this N1X unit under sustained Super FP8 decode (2026-09-15/16).** GPU temperature climbs about
  2 C per minute from idle; at 91 C the driver raises the SW thermal-slowdown flag (0x20) and clocks fall to 760-880 MHz;
  after several minutes at 93-94 C the platform hard-resets (Kernel-Power 41, code 0, no dump, no driver event) - seen
  twice (16:40 and 23:51). Separately, the persistent Reliability limiter (clock pinned at 663 MHz, 0x400 only, spinner
  ~6 TFLOP/s) can return WITHOUT any nvlddmkm event: on 09-16 00:40 it engaged at 84 C after 12 minutes of Super load.
  Working rule: run long Super reads through a temperature-paced, resumable wrapper (`paced_read.py`: stop the client at
  86 C, resume at 78 C), check the spinner after every read, and treat a pinned 663 MHz clock as end of measurement
  (only a reboot has cleared it).
