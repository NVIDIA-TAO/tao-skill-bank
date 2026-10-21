# Native Windows ARM64 AOI execution

This is the no-container route for an RTX Spark Windows-on-ARM64 host. It uses
the `windows-venv` platform runner, one GPU, and one process tree per action.

## Scope and limits

- Native mining, train, evaluate, export, direct inference, and the later
  local RTVI parity check are in scope.
- AnomalyGen SDG is **not** ported to native Windows. Start with mining-only
  iterations. A later run may use approved synthetic records that were staged
  from another platform, but must not represent that as a local SDG execution.
- `pyarrow` has no Windows ARM64 wheel (trap W16). Mining, filter, history, and
  emit read and write the same `.parquet` artifacts through `parquet_io.py`,
  which uses pyarrow where it imports and `duckdb` otherwise; nothing about the
  artifact names or the DEFT contract changes. Model annotations stay bare
  ShareGPT JSON arrays.

## Host layout

Keep the existing RTVI environment untouched. Recommended local NTFS paths:

```text
C:\rtvi-cosmos3\venv                         existing RTVI-VLM environment
C:\rtvi-cosmos3\venv-tao-cosmos3             isolated TAO environment
C:\rtvi-cosmos3\SRC\cosmos-framework         Framework with PR #164 + Windows port
C:\rtvi-cosmos3\SRC\cosmos-rl                windows-port checkout
C:\rtvi-cosmos3\SRC\tao-core                 pwd-import guard applied
C:\WORK\aoi-cosmos3-pcb                       AOI workspace and annotation splits
C:\models\Cosmos3-Nano-VLM                    prepared VLM safetensors directory
```

The three source checkouts are not supplied by this skill bank. Obtain them
through an already-authenticated approved Git configuration. Never place a
personal access token in a command line, source file, pip configuration, or
job record.

Retain at least 40 GiB free on the output drive after a prepared VLM is present.
That is the native checkpoint floor; the complete raw-release + conversion +
DCP lifecycle needs about 100 GiB plus dataset and source space.

## TorchCodec DLL bootstrap

After installing the verified ARM64 TorchCodec wheel, install the per-venv DLL
bootstrap before any import, training, or evaluation command. It retains the
Windows DLL-directory handles needed for Torch, CUDA, cuDNN, and the installed
ARM64 GStreamer JPEG/PNG DLLs; `PATH` alone is insufficient on Python 3.13.

```powershell
$PY = 'C:\rtvi-cosmos3\venv-tao-cosmos3\Scripts\python.exe'
$BANK = 'C:\tao-skill-bank'

& $PY "$BANK\skills\applications\tao-run-deft-aoi-cosmos3\scripts\configure_windows_arm64_dll_paths.py" `
  --venv 'C:\rtvi-cosmos3\venv-tao-cosmos3'

& $PY -c "import torch, torchcodec; print(torch.__version__); print(torchcodec.__version__)"
```

The bootstrap refuses to replace a different existing `.pth` file unless given
`--force`; use `--dry-run` to inspect its detected directories. Do not copy DLLs
into TorchCodec or modify the existing RTVI venv.

## Public PCB dataset preparation

The selected Roboflow `poc-ongfq` v2 project is an image-classification export
with `good` and `defect` folders. Download and expand it through an approved
local Roboflow credential mechanism, then preserve the original export outside
the workspace. Its expected layout is `train`, `valid`, and `test`, each with
`good` and `defect` subdirectories.

Run the deterministic converter once against an empty AOI workspace:

```powershell
$PY = 'C:\rtvi-cosmos3\venv-tao-cosmos3\Scripts\python.exe'
$BANK = 'C:\tao-skill-bank'

& $PY "$BANK\skills\applications\tao-run-deft-aoi-cosmos3\scripts\prepare_roboflow_classification_aoi.py" `
  --dataset-root 'C:\WORK\aoi-cosmos3-pcb\incoming\poc-ongfq-v2' `
  --workspace 'C:\WORK\aoi-cosmos3-pcb'

& $PY "$BANK\skills\applications\tao-run-deft-aoi-cosmos3\scripts\check_annotations.py" `
  --workspace 'C:\WORK\aoi-cosmos3-pcb' --require-files

& $PY "$BANK\skills\applications\tao-run-deft-aoi-cosmos3\scripts\validate_split_contract.py" `
  --workspace 'C:\WORK\aoi-cosmos3-pcb' `
  --manifest 'C:\WORK\aoi-cosmos3-pcb\provenance\roboflow_poc_v2_manifest.json'
```

The source labels become `OK` and `NG`; `train` becomes Mining, `valid` becomes
Proxy, and `test` becomes the frozen Benchmark. Since classification images do
not supply a physical paired golden reference, this converter chooses another
good image deterministically from the same source split. Its provenance
manifest marks the result `production_eligible: false`: it is valid only for a
pipeline demonstration. Use actual matched inspection/golden pairs before any
production KPI claim.

## Read-only preflight

The preflight performs the real allocation probe but does not install packages,
download assets, change sources, or submit a job.

```powershell
$PY = 'C:\rtvi-cosmos3\venv-tao-cosmos3\Scripts\python.exe'
$BANK = 'C:\tao-skill-bank'

& $PY "$BANK\skills\applications\tao-run-deft-aoi-cosmos3\scripts\preflight_windows_arm64_aoi.py" `
  --venv 'C:\rtvi-cosmos3\venv-tao-cosmos3' `
  --workspace 'C:\WORK\aoi-cosmos3-pcb' `
  --model-path 'C:\models\Cosmos3-Nano-VLM' `
  --framework-root 'C:\rtvi-cosmos3\SRC\cosmos-framework' `
  --cosmos-rl-root 'C:\rtvi-cosmos3\SRC\cosmos-rl' `
  --tao-core-root 'C:\rtvi-cosmos3\SRC\tao-core' `
  --output 'C:\WORK\aoi-arm64-preflight.json'
```

Do not submit a GPU action unless `ready_for_gpu_stage` is true. The report
must show native ARM64 Python, CUDA Torch, a passed 72 GiB allocation probe,
free ports 29500 and 29502, a staged workspace, adequate disk, and all Windows
source patch markers.

## Source and trainer rules

Copy `scripts/tao_aoi_pair_sft.py` to exactly
`cosmos_framework/configs/base/reasoner/experiment/tao_aoi_pair_sft.py` in the
patched Framework checkout. The `tao_` filename prefix is functional because
the Framework config loader's reload order otherwise leaves stale class objects.

Use `aoi_vlm_windows.toml` with the eleven Windows overrides from the Cosmos3
contract and AOI LoRA dropout `0.05`. Keep native `cosmos` attention; never add
`sdpa` or `eager` in a normal run because that can exit successfully while only
learning the label prior. Run `python -m cosmos_framework.scripts.train`, not
`torchrun`, with the single-process gloo environment and serialized ports 29500
and 29502.

Install NVIDIA's CUDA 13.4 Torch nightly and the private ARM64 TorchCodec wheel
first, then the model skill's `requirements-cosmos3-win.txt` unchanged: its
`platform_machine` markers skip the x64-only pins (cu130 torch family,
`pyarrow`, `obstore`) on ARM64 and install `duckdb` for the mining handoff.
The full sequence is the "Framework, Cosmos-RL, and TAO Core install" section
of `cosmos3-windows-arm64.md` in the model skill.

## Native mining handoff

The mining scripts take the same `.parquet` paths on ARM64 as on x64; there is
no format flag to pass. `parquet_io.py` (vendored into the DEFT and mining
skill `scripts` directories) selects pyarrow where it imports and `duckdb`
otherwise, and the two write byte-identical files -- verified on every
artifact of a real run, including `list<float32>` embeddings and the empty
table `pool_exhausted` produces. Install `duckdb` in the ARM64 venv (the
requirements markers do this) and run `native_mine_aoi.py`,
`filter_mined_by_cosine.py`, `filter_mined_history.py`, and
`emit_mined_sharegpt.py` exactly as documented for x64.

## Evaluation and RTVI evidence

Use `analyze_gaps.py` as the authoritative evaluator for native bare-label AOI;
the Cosmos-RL console summary is not the KPI. After an export with zero LoRA
tensor keys, load it through the existing native RTVI-VLM launcher and send
ordered JPEG pair requests. Compare exact labels with the frozen Benchmark and
record model version, request pair order, API responses, and the confusion-cell
slice before claiming ARM64 DeepStream parity.

## Traps confirmed on this host (2026-09-05 run)

Each item below broke a real run on Windows 11 ARM64 / RTX Spark N1X, CPython 3.13 ARM64.
They are ordered by how expensive they are to misdiagnose.

### 1. `task.type = "binary"` silently reports accuracy 0.0000 for OK/NG

`cosmos_rl/evaluation/evaluator.py:35-37` defines `POSITIVE_LABELS = {"yes","a"}` and
`NEGATIVE_LABELS = {"no","b"}`. AOI's `OK`/`NG` are in neither set. When `task.type` is
`"binary"`, `evaluator.py:451-457` runs `_compute_binary_classification_metrics` and
**overwrites** `overall.accuracy` with its result — which is 0.0 because the confusion matrix
matched nothing (TP=FP=TN=FN=0). The run exits 0 and prints `Accuracy: 0.0000` while
`per_task` holds the correct value.

Use **`task.type = "bcq"`**. `extract_answer` treats `bcq` and `binary` identically
(`metrics/task_accuracy.py:36`), and `bcq` is in `ACCURACY_TASKS`, so accuracy is still
computed — it simply skips the mislabelled confusion matrix.

The AOI confusion matrix comes from `analyze_gaps.py`, which is written for these labels
(NG positive, `TP = NG->NG`, `FN = NG->OK` false accept). Never take it from the evaluator.

Note the two code paths disagree: the shard-aggregation path (`evaluator.py:513-516`) also
requires every `gt` to be in `BINARY_LABELS` before treating a run as binary, so it would not
have zeroed the result. Only the single-shard path does.

### 2. `evaluate` is unreachable — pyarrow has no ARM64 wheel

`cosmos_rl/evaluation/metrics/text_metrics.py` imported `evaluate` at module scope, making the
whole evaluator unimportable. The chain is `evaluate -> datasets (all versions) -> pyarrow>=21`,
and `pip install --only-binary=:all: --dry-run pyarrow` reports **`from versions: none`**.

The guarded import now disables BLEU/ROUGE/BERTScore and logs a warning; `compute()` already
returned `{"WEIGHTED_SCORE": 0.0}` in that state, which is the documented normal output. OK/NG
accuracy is unaffected because it comes from `task_accuracy.py`, which does not import
`evaluate`.

Apply any cosmos-rl patch to **both** `C:\rtvi-cosmos3\SRC\cosmos-rl\...` and
`C:\rtvi-cosmos3\venv-tao-cosmos3\Lib\site-packages\cosmos_rl\...`. Unlike `cosmos_framework`,
`cosmos_rl` is **not** editable-installed, so the runtime uses the site-packages copy.

### 3. `TAO_DISABLE_CLOUD_STATUS_CALLBACK=1` is mandatory

`nvidia_tao_core/loggers/logging.py:25 _status_callback` imports the Lepton/NGC/Kubernetes cloud
handler tree on every status write. Set this variable on every native job. Chasing the import
instead leads to `ngcbase`, which is **not on PyPI at all**.

The runner's `--tao-windows-env` does not set it; pass it with `-e`.

### 4. wandb cannot be downgraded; `wandb.util.generate_id` is gone

Recent wandb moved the helper to `wandb.sdk.lib.runid.generate_id`, so
`cosmos_framework/utils/wandb_util.py` raised `AttributeError` at train launch. Pinning an older
wandb is **not** an option: every release that still exports `wandb.util.generate_id` requires
`setproctitle`, which has no ARM64 wheel. A version-tolerant shim in `wandb_util.py` is the fix.

This failure is masked: the secondary exception raised inside the error handler
(`ModuleNotFoundError: ruamel`) is what gets reported. Read the **first** traceback in the log.

### 5. Windows PowerShell 5.1 strips quotes from native-exe arguments

`{"video":{...}}` arrives as `{video:{...}}` and `json.loads` rejects it, so
`Start-Cosmos3Arm64.ps1 --media-io-kwargs` kills the vLLM launcher before it binds a port.
PowerShell 7 forwards quotes verbatim; `pwsh` is not installed on this host. Guard with
`if ($PSVersionTable.PSVersion.Major -lt 6) { $v = $v -replace '"', '\"' }`.

### 6. `recall_ng >= 1.0` is satisfiable by a degenerate model

A constant-`NG` predictor never emits `OK`, so it cannot produce a false accept and scores
`recall_ng = 1.0` by construction. On this dataset the zero-shot base model did exactly that:
910/910 `NG`, accuracy 0.2308 (the NG base rate), 700 false rejects, **0 true negatives** — and
it *passed* the default gate at baseline, which would have stopped the loop before any
iteration ran.

Gate on `accuracy` or `f1_ng` for AOI. Declare the metric **before** the evaluation it governs
and never swap it after seeing results.

### 7. Mining can return single-class data

When every Proxy error is a false reject, routing yields only `OK` targets and mining returns
only `OK` records. Training on one class collapses the model to a constant `OK` predictor — the
mirror of trap 6. AnomalyGen normally supplies the `NG` side but is unsupported here.

Balance the assembled Train set with `NG` records drawn from **`mining_pool.json` only**, never
Proxy or Benchmark, and record the deviation in `assemble_summary.json`. Re-run
`validate_split_contract.py` afterwards.

Related: `anomalygen --skip` is legal exactly when the driving Proxy RCCA recorded zero false
accepts, which is the same condition that produces this single-class case.

### 8. No pyarrow anywhere in the mining chain

`pip install --only-binary=:all: --dry-run pyarrow` reports `from versions: none` on
CPython 3.13 ARM64. The mining chain no longer needs it: `native_mine_aoi.py`,
`filter_mined_by_cosine.py`, `filter_mined_history.py`, `emit_mined_sharegpt.py`, and the
`--mining-parquet` row count in `commit_stage.py` all go through `parquet_io.py`, which uses
pyarrow where it imports and `duckdb` otherwise, and writes the same `.parquet` artifacts
either way (trap W16 in the model skill). Install `duckdb` in the ARM64 venv --
`requirements-cosmos3-win.txt` does so through its `platform_machine` marker. `routing`
still emits both `mining_targets.json` and the parquet, as SKILL.md asks.

`init_deft_state.py` likewise required two container images unconditionally; it now exempts
containerless platforms (`windows-venv`, `local-venv`, `slurm-venv`) via `CONTAINERLESS_PLATFORMS`.

### 9. Checkpoints must live inside the iteration result tree

`commit_stage.py --best-ckpt` rejects a path outside `<results>/<iter>/train/`, and SKILL.md
lists "a checkpoint outside the iteration result tree" as a hard stop. The Windows job runner
writes under its own job directory, so move the DCP checkpoint into
`<results>/<iter>/train/checkpoints/` before committing the `train` stage.

### 10. Export must merge, and must be verified

Pass `--base-model-path-or-uri` to `cosmos_rl.framework.export` for any LoRA run. Without it the
export still succeeds, still contains lora tensors, and vLLM silently serves the **baseline**.
Post-condition, checked every time:

```
python -c "import json;m=json.load(open('<export_dir>/model.safetensors.index.json'))['weight_map'];print(sum(1 for k in m if 'lora' in k.lower()))"
```

Must print `0`.

## Learning check on the RTX Spark N1X (2026-09-08)

Second run on this host, wafer single-image workspace (`C:\WORK\wafer\workspace-full`,
rebuilt offline from the verified `poc-ongfq` v2 export by the rig's
`build_workspace_from_export.py`; 7,346 / 909 / 910). Driver:
`windows_native_deft_loop.py` training path via the rig's `learning_check.py`, recipe-default
`cosmos` attention, 7-module LoRA, `TAO_DISABLE_CLOUD_STATUS_CALLBACK=1`, `duckdb` parquet
backend. Torch 2.15.0.dev20260904+cu134, cuDNN 9.25, transformers 4.57.6, peft 0.20.0.

### 11. The Sept-5 spec comment "`cosmos` routes to NATTEN on Windows" is wrong on this host

`C:\WORK\output\cfg\roboflow_native.toml` (Sept-5 run) carries the comment that `cosmos`
attention routes to NATTEN, which has no Windows build, and switches to `sdpa` — which on
transformers 4.57.x trains to the label prior (W14). Probed directly on the N1X:

```
arch_tag 121  cudnn 92500  CUDNN_MIN_BACKEND_VERSION 91500
backend preference list: ['cudnn', 'flash2', 'natten']
natten_supported: False
```

`cosmos_framework.model.attention.backends.get_backend_list` puts cuDNN first for the
Blackwell RTX arch tag, the installed cuDNN clears the floor, and NATTEN is neither
installed nor selected. `cosmos` therefore runs on torch's cuDNN SDPA here, exactly as on
x64. Keep the recipe default; never set `attn_implementation=sdpa|eager` on this host
without the W14 mask-gate patch.

### 12. Recipe batch size (W17)

The first learning check inherited `[dataloader_train] max_samples_per_batch = 1` from
`aoi_vlm_windows.toml` and trained one record per step (`input_ids [1, 128]`): val loss on
the all-NG head 0.4030 at epoch 1, 0.0705 at epoch 2 — learning, but an eighth of the
intended data per epoch. With `max_samples_per_batch = 8` (`input_ids [8, 128]`, the
validated x64 "batch 8"): 0.1689 at epoch 1 — nine of the ten all-NG validation records
between 0.002 and 0.065 (median 0.009) and one at 1.52 — and 0.1691 at epoch 2 (nine
records between 0.003 and 0.036, the same outlier at 1.58). Over the nine correctly
labelled records the mean is 0.018 at epoch 1 and 0.012 at epoch 2, i.e. the ~0.02 the
W14 validation established; the check passes. The 1.52 record is the same file in both runs: `defect/` folder (label NG), filename class token
`good`, model answers OK with confidence. That folder/filename disagreement affects
427 / 7,346 Mining, 49 / 909 Proxy and 61 / 910 Benchmark records of this export (rig
README, "Data note"); it is label noise, not a trainer signal. Full entry: W17 in the
model skill's `cosmos3-windows-gaps.md`.

### 13. `cosmos-rl evaluate` needs PyNvVideoCodec at `load_model` even for still images (W18)

The first loop launch died at the baseline benchmark evaluate:

```
RuntimeError: GPU video decoding requires PyNvVideoCodec and qwen-vl-utils
  base.py:106 load_model -> register_pynv_video_reader(strict=True)
```

`cosmos_rl/evaluation/base.py` registers a video decoder before it loads the model; the
default `video_decoder = "pynvvideocodec"` imports `PyNvVideoCodec` with `strict=True`, and
the provisioned ARM64 TAO venv did not carry it (the rtvi venv does — its installer pulls
`PyNvVideoCodec==2.2.2` from PyPI). Fix used here: install the PyNvVideoCodec 2.2.x
win_arm64 wheel into the TAO venv.

- Install: `python -m pip install --no-deps PyNvVideoCodec==2.2.2` (win_arm64 wheel on
  PyPI; metadata declares no dependencies). A fix for the interpreter-exit crash seen with
  this build is pending in the next PyNvVideoCodec release; pin by version until then.
- Verify exactly what the evaluator does:
  `from cosmos_rl.utils.pynv_video_reader import register_pynv_video_reader;
  register_pynv_video_reader(cache_size=0, video_override_map=None, strict=True)` →
  `{'backend': 'pynvvideocodec', 'version': '2.2.2', ...}`; `nvcuvid.dll` and
  `nvEncodeAPI64.dll` are in `System32` on this host, so the W1 NVENC probe passes natively.
- Config-only fallback if no wheel is available: `[vision] video_decoder = "torchcodec"`
  in the evaluate spec (torchcodec is installed here; route doc trap W1). Not used for the
  N1X run so the evaluator stays on the same decoder path as x64.

### 14. The loader kills the run on the last batch of epoch 1 unless the train set is a multiple of the batch size

Not a Windows trap (the guard is in `cosmos_framework/data/generator/dataflow/loader.py`
and fires on any host), recorded here because it cost 27 GPU-minutes on this box. With
`max_samples_per_batch = 8` and an assembled iter1 set of 3,444 records, training ran
430 of 431 steps of epoch 1 and then raised

```
ValueError: Map-style resume cannot safely stamp a multi-sample batch whose _dp_stream_pos
values are non-contiguous or span epochs (reordering batcher + batch_size>1). Use
max_batch_size=1 with pool packing, a sequential (order-preserving) batcher, or an
iterable (non-resumable) source.
```

The 431st batch holds the last 4 records of epoch 0 and the first 4 of epoch 1. The
route doc's "keep assembled train sets at a batch-size multiple" is this guard; the
reference driver now enforces it after `assemble_training_json.py` by dropping the
`len % 8` tail records (the lowest-similarity newly mined OK records, since assembly
appends new records in mining order) and recording them under `batch_multiple_trim` in
`assemble_summary.json`. The 1,600-record learning checks never hit it because 1,600 is a
multiple of 8. Recovery: the failed attempt is kept as
`results\wafer-win-arm-1-attempt1-epoch-span\` and the run was restarted from scratch
under the same name (baseline evals and mining are ~17 min; state-file surgery to redo
only assembly was rejected as a provenance hazard).

### 15. Native rtvi-vlm deploy fails its own preflight after the DeepStream workspace is rebuilt

Two absolute paths into the DeepStream team's rel-39 build workspace are baked into the
serving stack, and both went stale when that workspace was rebuilt with a new CMake preset
(2026-09-08; output moved from `deepstream\sdk\cmake\build\release\bin` to
`deepstream\sdk\build\arm64-release\bin`). The rtvi venv itself is untouched.

1. `Start-RtviVlmArm64.ps1` asserts `<Rel39Root>\deepstream\sdk\cmake\build\release\bin`
   and `<Rel39Root>\3rdparty\vcpkg\installed\arm64-windows\bin` exist (default root
   `C:\Users\lab\Downloads\rel-39`) and prepends them to `PATH`:
   `rel-39 DeepStream build runtime directory was not found`.
2. The venv's `sitecustomize.py` (written by `Install-PydsArm64Artifacts.ps1`) registers
   the same build-bin path plus the vcpkg bin, glib tools and GStreamer bin with
   `os.add_dll_directory`, **skipping any directory that does not exist**. Python 3.13 does
   not search `PATH` for a `.pyd`'s dependencies, so with the build bin gone
   `import pyds` fails — `DLL load failed while importing pyds: The specified module could
   not be found` — and the launcher's runtime preflight throws. Fixing the `PATH` side
   alone (a `-Rel39Root` shim) passes assert #1 and still dies here; the missing DLLs are
   the freshly built DeepStream helpers (`nvdsgst_helper.dll` and friends) that only the
   build bin and `C:\opt\...\lib` carry, and `C:\opt` is not a registered DLL directory.

Fix used (venv untouched): recreate the documented path as a junction to the new output,
`mklink /J C:\Users\lab\Downloads\rel-39\deepstream\sdk\cmake\build\release\bin
C:\Users\lab\Downloads\rel-39\deepstream\sdk\build\arm64-release\bin`; `import pyds` then
succeeds bare and the launcher starts. Verify with the venv interpreter,
`python -c "import pyds"`, before every deploy. The rig's `deploy_champion.py` also gained
an `RTVI_REL39_ROOT` passthrough (`-Rel39Root`) for hosts where the workspace lives
elsewhere; it covers the launcher's asserts, not the venv's `sitecustomize` paths.

### 16. The in-process rtvi-vlm rewrites the model's input: 608×320 frame resize and a timestamp prompt

First parity run of the iter2 champion through the native server: **7/12** on the
stratified benchmark sample and NG on all six shoot-kit wafers including the three good
ones, while `analyze_gaps.py` on the same merged weights scored 0.9385. Same weights,
different verdicts, so the serving path was changing the model's input. Two causes, both in
`src/models/vllm_compatible` / `src/vlm_pipeline` of the rtvi-microservices checkout:

1. **Decode-time resize.** `get_input_config()` returns `608×320` for
   `cosmos-reason2/3`, and every frame — a still JPEG included — is resized to it before
   vLLM's processor (`VLM_INPUT_WIDTH/HEIGHT` override it). A 224×224 wafer becomes a
   stretched 608×320 image: the served request reported `input_tokens=277` (190 visual
   tokens for the 19×10 grid) against 49 visual tokens for the 224×224 image the trainer and
   evaluator saw (`input_ids` length 123–128 in the training log). The A/B through
   `/v1/chat/completions` with the image inline gave the same NG verdicts, so the DeepStream
   decode itself is not the cause. Fix: `VLM_INPUT_WIDTH=0 VLM_INPUT_HEIGHT=0` — the fast
   image path only resizes when both are non-zero, and native-resolution frames then reach
   vLLM's own processor exactly as the evaluator's PIL image does.
2. **Timestamp prompt.** `Start-RtviVlmArm64.ps1` hard-codes
   `RTVI_ADD_TIMESTAMP_TO_VLM_PROMPT=true`, so file requests get the prefix
   `These are images sampled from the same video at times 0.00 . ` in front of the AOI
   question the model was fine-tuned on. The env cannot be overridden through the launcher,
   but any `RTVI_TIMESTAMP_PROMPT_{PREFIX,SUFFIX}_FILE_SOURCE` template replaces the legacy
   wording; the pieces are joined with newlines, so a single-space suffix template
   (`RTVI_TIMESTAMP_PROMPT_SUFFIX_FILE_SOURCE=' '`) reduces the injection to one trailing
   whitespace token.

The x64 12/12 parity proof ran rtvi-vlm in openai-compat mode against an external vLLM,
which hands the image to vLLM's processor untouched — this trap is specific to the native
in-process (`cosmos-reason3` model type) lane. Verify with the response's `input_tokens`
before trusting any served verdict.

**Result after the fix (2026-09-09, `results\wafer-win-arm-1\parity\parity_result.json`).**
Served `input_tokens` 136 versus 135 for the evaluator's own path (replicated on CPU:
`process_vision_info(image_patch_size=16)` + HF processor → grid `[1,16,16]`, 64 visual
tokens); the one extra token is the timestamp suffix. Parity on the stratified sample rose
from 7/12 to **11/12**; on the six shoot-kit wafers (all benchmark images, matched by
sha256) the three NG heroes match the iter2 evaluator and two of the three good wafers
are served NG where the evaluator said OK — 15/18 overall, every mismatch served-stricter.
Passing the trainer's pixel bounds per request (`mm_processor_kwargs`
`{"min_pixels": 50176, "max_pixels": 401408}`, 121 tokens) does not change those three,
so the residual is the decoder (in-process vLLM bf16 versus HF transformers) and/or the
single suffix token; attributing between them needs an HF run on the three images with
the suffix appended, which requires the GPU and is pending. Note for the demo script: on
this lane only `wafer-good-2.jpg` is served OK among the shoot-kit "safe" wafers. Caveat
on method: an A/B through `/v1/chat/completions` with an inline `image_url` reported
`prompt_tokens=65`, i.e. the image was dropped, so that endpoint is not a valid control
here. Served latency on the file path is ~5.3 s per image (VLM ~0.45 s, the rest asset
upload and per-request pipeline setup); `RTVI_FAST_IMAGE_ASSET_CHUNK_DECODE=true` is the
untested speed knob.

### 17. `localhost` costs 2 s per HTTP request against the native server — use 127.0.0.1

`Start-RtviVlmArm64.ps1` is launched with `-ApiHost 127.0.0.1`, and on this host
`localhost` resolves to `::1` first: a Python `urllib` request to
`http://localhost:8000/v1/health/ready` takes 2.06 s every time (an explicit `[::1]`
connect hangs ~2 s before failing, then IPv4 succeeds), against 0.02 s for
`http://127.0.0.1:8000`. curl's happy-eyeballs hides it (0.2 s), which is why a manual
probe looks fine. The classify tool makes four requests per verdict and the parity check
two per image, so this alone was the "5 s per image" and "7–8 s per classify" seen on the
N1X while the server itself reported ~0.3 s per image (decode ~120–150 ms, VLM ~170 ms,
identical after a long idle — there is no wake-up penalty). One unexplained 50 s first
call was seen once and did not reproduce after the fix. Rig tools now default the native
lane to `127.0.0.1`; classify is 1.0 s end to end.

### 18. Pick demo wafers through the served path, not from evaluator verdicts — the zero-shot base flips hardest

With the parity-correct env (item 16) the served **base** model still disagrees with the HF
evaluator far more than the champion does: all six shoot-kit wafers that the evaluator's
baseline called OK are served NG, and of the evaluator's 24 baseline false accepts only 4 are
served OK. Framing is identical (served 136 input tokens = HF 135 + the suffix token; the chat
template's assistant prefix is the same with and without `enable_thinking`), so the residual
is the decoder — in-process vLLM bf16 versus HF transformers — acting on a model whose
zero-shot decisions sit near 50/50 (baseline accuracy 0.4956). The fine-tuned champion is
confident enough that the same difference moves 3 of 18 verdicts.

Consequence for a "before/after" demo: the x64 shoot-kit heroes do not work on this lane
(the served base already says NG), so hero and good wafers must be screened through the
served base *and* the served champion. Screen done 2026-09-09
(`results\wafer-win-arm-1\parity\served_base_candidates.json`,
`served_champion_candidates.json`): 2 heroes (truth NG, served base OK, champion NG) and 9
good wafers (served OK under both) out of 24 + 16 screened.

Also measured here: the **first request after a (re)deploy takes 25–50 s** because the
launcher runs with `SKIP_PIPELINE_WARMUP=true`, so the decode pipeline is built on the first
real request (subsequent requests 1–3 s). `deploy_champion.py --check` now fires one warm-up
classify when it first sees READY; without it, the first on-camera verdict is slow.

### 19. The idle server also cold-starts: ~55 s on the first request after sitting idle

Item 17 said the 50 s first call did not reproduce. It did, on 2026-09-10: after the native
server had served nothing for ~50 minutes, the first classify took 54.8 s and every later one
2.7 s, i.e. the same deferred-pipeline build as after a deploy (item 18), not a network effect.
The `--check` warm-up only covers the first request after READY. Before any timed or on-camera
verdict, fire one throwaway classify yourself; the demo tools now do this on the operator's cue.

### 20. Stopping and resuming the loop: ledger resume works, but kill the notifier first

The reference driver resumes by reading `deft_state.json` and skipping every `(iter, stage)`
whose event is `ok` (`done_stages()`); relaunching with the same run name re-logs the skipped
stages' result lines and continues at the first missing stage. Verified 2026-09-10 on
`wafer-win-arm-2`: killed at `iter1 routing/data_mining`, restarted 49 min later, it redid
`evaluate_proxy`/`routing`/`data_mining` (~10 min) and went into `train`. Three rules:

1. **Training has no mid-run resume** (`save_freq_in_epoch` = epochs, checkpoint at the end), so
   a stop during `train` loses that iteration's progress; stop between stages.
2. **Kill the WhatsApp notifier before the driver.** It tails the driver log and posts on
   `failed rc=|Traceback`; killing the driver first can put a false "failed" line in the chat.
   After a restart the fresh notifier re-posts the start and the re-logged result lines once.
3. Every venv Python shows up **twice** in process lists (Windows venv launcher + real
   interpreter, same command line, same second): match on the command line and expect pairs;
   two `run_loop.py` entries are one run, not a duplicate launch.

### 21. The zero-shot baseline is bit-reproducible across runs

`wafer-win-arm-2` (2026-09-10, fresh run on the same rebuilt workspace) scored the identical
baseline to `wafer-win-arm-1`: accuracy 0.4956, recall_ng 0.886. The HF evaluator on the base
model is deterministic on this host, so a different baseline on a new run means the workspace
or the evaluator environment changed, not the model.

### Loop result on the N1X — `wafer-win-arm-1` (2026-09-08/09), gate met at iter2

Same driver, recipe and knobs as the x64 reference (`DEFT_OK_TOPN=15`, `DEFT_BALANCE=none`,
7-module LoRA, `cosmos` attention, 10 epochs/iteration, batch 8), on the workspace rebuilt
from the same Roboflow export. Frozen benchmark = 910 images (700 OK / 210 NG), scored by
`analyze_gaps.py`, zero unknown predictions at every stage.

| Stage | Accuracy | NG recall | NG precision | Train set | x64 reference (wafer-win-8) |
|---|---|---|---|---|---|
| baseline (zero-shot) | 0.4956 | 0.886 | 0.300 | — | 0.4890 / 0.881 |
| iter1 | 0.8736 | 0.505 | 0.906 | 3,440 (9.6% NG; 3,444 mined, 4 OK trimmed) | 0.8659 / 0.495 / 3,470 |
| iter2 — **gate met** | **0.9385** | **0.824** | 0.901 | 4,104 (23.2% NG; +622 NG / +45 OK, 3 OK trimmed) | **0.9220** / 0.814 / 4,095 |

Confusion at iter2: 173 NG→NG, 37 NG→OK, 19 OK→NG, 681 OK→OK. Champion:
`results\wafer-win-arm-1\iter2\export\merged` (17 GB, 0 LoRA keys); the iter1 merged
model was deleted by the driver's keep-best rule. Stage wall times (s, from
`deft_state.json`): baseline benchmark 357, proxy 323; iter1 mining 322, train 16,362
(4.5 h, 4,300 steps), benchmark 358, proxy 335; iter2 mining 57 (pool embeddings reused),
train 16,683 (4.6 h, 5,130 steps), benchmark 366. End to end 9 h 55 min against ~65 min on
the x64 workstation; the difference is the training step (item 12 / measured table below).
In-training validation on the all-NG head: iter1 0.016 → 0.053 → 0.153 → 3e-5 → ~0 from
epoch 5; iter2 0.379 → 0.017 → 0.013 → … ~0. GPU during training: 92% utilisation,
32.5 W, SM clock mean 1.50 GHz (max 3.0 GHz).

### Measured on the N1X (learning check, 1,600 records, 2 epochs, 200 steps/epoch)

| | 1 sample/step | 8 samples/step |
|---|---|---|
| seconds/step, `iter_speed` every-N line ("seconds per iteration") | 0.29 | 4.1 |
| seconds/step, wall clock first step → "Done with training" (400 steps, incl. 2 validations) | 0.72 | 4.13 |
| split of one 8-sample step (log timestamps, steps 300–303) | — | ≈1 s data fetch + forward, ≈3 s backward (full activation checkpointing), ≈0.3 s optimizer + callbacks |
| wall per sample | 0.72 s | 0.52 s |
| wall incl. model load + checkpoint save | 6.8 min | 29.2 min |
| trainer allocator peak (`peak_mem_gb` in the `tokens_per_sec` callback line) | 20.2 GB | 20.3 GB |
| DCP checkpoint (`epoch_2/`: model, optim, scheduler, trainer, dataloader) | 17 GB | 17 GB |
| async checkpoint save | 50 s, `succeeded=True` | 63 s, `succeeded=True` |
| peak system committed memory above idle (external sampler, 15–20 s) | 65.7 GB (sampler covered model load + save staging) | 18.2 GB (sampler started after launch) |
| `nvidia-smi` memory.used | pinned at ~8.0 GiB of the 8,128 MiB WDDM counter | same |

`nvidia-smi`'s 8 GiB counter is the WDDM dedicated segment, not the budget on this
unified-memory host (`torch.cuda.mem_get_info` reports 101.5 GiB); read the trainer's
`peak_mem_gb` (20.3 GB for this LoRA config) or the committed-memory delta instead.

**The step is GPU-bound, and the backward pass is three quarters of it.** Do not read the
per-step `iter_speed` callback's `Time: 0.55s` as compute time — it is the interval since
the callback's previous hit, not a bracket around the step. The trainer's own log
timestamps for consecutive steps give the split: previous step end → `train/loss_avg`
≈1 s (data fetch + forward), `train/loss_avg` → `clip_grad_norm` ≈3 s (backward with
`activation_checkpointing.mode = "full"`, i.e. a forward recompute plus the backward), →
next step end ≈0.3 s (optimizer + callbacks). The 1-sample run splits the same way at
0.72 s. Projection at the driver's `steps_per_epoch = ceil(records/8)` × 10 epochs: an
x64-sized iter1 (3,470 records, 4,340 steps) is ~5 h and iter2 (4,095 records) ~6 h on
this host, versus ~25 min per iteration on the x64 workstation. The one speed-only knob
in the validated recipe is activation checkpointing (kept on for the x64 validation; the
20.3 GB allocator peak leaves ~80 GB of headroom here); turning it off should remove the
forward recompute from the backward, roughly a quarter of the step. `num_workers` stays
0 (contract override #6: Windows spawns worker processes) and is not where the time goes. The checkpoint background process logs
`client socket has failed to connect to [kubernetes.docker.internal]:29502 (10049)` at
init because it binds `MASTER_PORT+2` on the hostname (latent item in the gaps file), then
reports ready and saves normally; both 29500 and 29502 are released at exit (W15 pre-flight
still applies before every launch). Disk: DCP 17 GB + merged export 17 GB per iteration,
so a two-iteration loop peaks near 51 GB above the base model with the driver's
delete-after-export hygiene.
