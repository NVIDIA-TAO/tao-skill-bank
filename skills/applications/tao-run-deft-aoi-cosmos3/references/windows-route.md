<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DEFT AOI on native Windows

The full mining-only DEFT loop runs natively on one Windows 11 host and meets
its gate: on the public `poc-ongfq` wafer set (7,346-image pool, frozen
910-image benchmark), **baseline 0.4890 → iter1 0.8659 → iter2 0.9220,
`accuracy >= 0.90` gate met, NG recall 0.814** — mining, LoRA training,
export, evaluation, and deployment all on the same machine (RTX PRO 6000,
96 GB). The run is deterministic: a replay reproduces every stage number
bit-exactly.

Read `skills/models/tao-finetune-cosmos-reason/references/cosmos3-windows.md`
first: it owns the Cosmos3 Windows pins, source patches, and support
boundaries. This file owns what the DEFT loop does differently on Windows.
The full investigation record (run ledgers, elimination ladder, root-cause
forensics) lives in `windows-route-evidence.md` — read it for *why*, not
*how*.

## The validated training configuration (2026-08-27)

- **Leave `attn_implementation` at the recipe default (`cosmos`).** Do not
  override to `sdpa`/`eager`: on the pinned transformers 4.57.x they train to
  the label prior at exit 0 (trap W14 — the trainer pops the attention mask
  and HF fabricates packed-sequence walls at every image token). The `cosmos`
  implementation is an AttentionInterface adapter over torch's cuDNN SDPA and
  runs natively on Windows. If sdpa/eager is ever required, apply the model
  skill's `w14-attention-mask-gate.patch` first. The model skill's contract
  carries the 11-string override set — the two attention overrides that used
  to be in it are deliberately gone.
- **LoRA**: r16 / alpha 32 / dropout 0.05, 7-module targets
  (`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` — pass with
  embedded quotes; bare commas are a hydra sweep).
- **Schedule**: lr 1e-4 cosine-annealed over the actual run:
  `scheduler.cycle_lengths=[steps_per_epoch*epochs+10]` and explicit
  `trainer.num_epochs` (the recipe default cycle silently pins lr at ~99% of
  peak; stepping past the last cycle crashes the scheduler). Batch 8 —
  concretely `[dataloader_train] max_samples_per_batch = 8` in the run's
  spec (the TOML field is remapped to the PoolPackingBatcher's
  `max_batch_size`; the batch arrives as `input_ids [8, 128]`). The
  checked-in `aoi_vlm_windows.toml` still ships the WTS smoke value `1`,
  which trains one sample per step and turns `steps_per_epoch=ceil(records/8)`
  into an eighth of a pass per "epoch" (trap W17 in the model skill's
  `cosmos3-windows-gaps.md`; measured on the N1X 2026-09-08). 10
  epochs/iteration, `trainer.steps_per_epoch=ceil(records/8)`.
- **Mining**: per-class with `--filter-label`, symmetric top-15, **no balance
  floor** when the benchmark class prior is skewed. The rule that transfers:
  accuracy peaks when the cumulative training mix matches the benchmark
  prior (23% NG here); NG-heavy mixes trade accuracy for NG recall. Pick the
  operating point deliberately — a recall-forward line wants NG 15 / OK 3
  with a 40% NG floor instead (evidence file, "Loop closure").
- **Pre-flight, every training launch**: ports 29500 and 29502 must be free
  (a leftover async-checkpoint writer on `MASTER_PORT+2` lets the run train
  fully then hang forever at the final save with an empty checkpoint — trap
  W15; check `netstat -ano | findstr :29502`, kill squatters by exact PID,
  never run two trainings concurrently); >= 40 GB free disk (33 GB DCP peak);
  assembled train sets at a batch-size multiple — the loader's map-style
  resume guard raises `Map-style resume cannot safely stamp a multi-sample
  batch ... span epochs` on the last batch of epoch 1 otherwise, after a full
  epoch of GPU time (N1X, 3,444 records, step 430/431); the reference driver
  trims the `len % 8` lowest-similarity new OK records after assembly and
  records them in `assemble_summary.json`; drop the recipe's
  `[dataloader_train]` block if swapping batchers (its fields are
  PoolPackingBatcher-specific).
- **Disk hygiene per iteration**: delete each DCP after its verified export;
  delete a previous merged model only when the newer iteration scores at
  least as well.

## Platform selection on x86: prefer the container for GPU stages

The Linux loop is hybrid by design — host orchestration, containerized GPU
stages — and the same split is the recommended shape on Windows x86:
orchestration and mining run on the host (mining has no Windows container;
`native_mine_aoi.py` is the mining stage regardless), while train / evaluate /
export run in the validated `cosmos-framework-windows` image through
`tao-run-on-windows-docker`. The pinned image (tag `20260826.1` onward)
already carries the AOI experiment (`tao_aoi_pair_sft.py`, one- or two-image
records); when the base is rebuilt, re-derive with a single `COPY` of the
experiment file, or read-only bind-mount it onto the same path. **The AOI
evaluate spec needs `video_decoder = "torchcodec"` under `[vision]`
in-container even though AOI is images-only** (trap W1: the evaluator probes
NVENC at startup and the device class does not project it). Container and
native are numerically interchangeable for these stages (validated:
bit-identical validation losses, identical eval scores); choose on
operational grounds. The all-native route is required on **ARM64** (no
Windows container exists) and is fine for x86 dev iteration.

## The two substitutions

Everything else in the loop runs unmodified — the orchestration scripts are
pure Python, and `cosmos-rl-evaluate` is validated natively.

### 1. Trainer: cosmos-framework LoRA instead of the `cosmos-rl` launcher

The Cosmos-RL SFT launcher is not in the Windows port. Training runs through
`cosmos_framework.scripts.train` with an AOI experiment config that accepts
one- or two-image records:

- Install `scripts/tao_aoi_pair_sft.py` into the model source checkout at
  `cosmos_framework/configs/base/reasoner/experiment/`. The filename is
  load-bearing: the config loader re-imports that package **alphabetically
  with `reload=True`**, and any module sorting before `dataflow_roles`
  captures pre-reload class objects and dies at pickle time with
  "it's not the same object". Keep the `tao_` prefix.
- The experiment (`aoi_pair_conversation`) reuses the shared `VLMProcessor`
  pipeline; every image in a bare ShareGPT record attaches to the first user
  turn in record order (pairs or single captures).
- Recipe: `references/aoi_vlm_windows.toml`, plus the model skill's Windows
  override set on the command line (the recipe cannot absorb the
  `model.config.*` / `dataloader_*` overrides) and
  `model.config.policy.lora_dropout=0.05`. `trainer.max_iter` is
  **non-binding** on this recipe family — control run length through epochs.
- Env contract mirrors WTS: `AOI_TRAIN_ANNOTATION`, `AOI_TRAIN_MEDIA`,
  `AOI_VAL_ANNOTATION`, `AOI_VAL_MEDIA`, `AOI_SYSTEM_PROMPT`
  (`Return exactly OK or NG. Do not include reasoning or extra text.`),
  `AOI_MAX_PIXELS=401408`, `AOI_MIN_PIXELS=50176`. Use Proxy for in-training
  validation; the frozen Benchmark is the gate and nothing else.
- In-training validation is the honest learning signal (the val slice is the
  all-NG head of the assembled JSON): healthy runs read ~0.02 at epoch 1 and
  fall to <=1e-3 by epoch 4; a plateau near 0.5+ means the model is not
  learning — check the attention configuration before anything else.

After training, run `cosmos-framework-export --base-model-path-or-uri <base>`
and verify **zero** `lora` keys in `model.safetensors.index.json`. Point the
iterN evaluate spec's `model_name` at the exported directory.

### 2. Mining: native SigLIP instead of the data-services container

`skills/data/tao-mine-aoi-images/scripts/native_mine_aoi.py` reproduces the
container's three-step contract (embed targets, embed pool, top-K cosine)
with `google/siglip-base-patch16-224` via transformers, emitting the same
parquet schemas. Parity evidence: the skill's own `filter_mined_by_cosine.py`
reproduced the miner's scores exactly. `filter_mined_history.py` runs
unchanged — its Windows `EBADF` (fsync on a read-only handle) is fixed in
this bank. All mining parquet I/O goes through `parquet_io.py`: pyarrow where
it imports, duckdb otherwise (Windows ARM64 has no pyarrow wheel — trap W16);
the two backends write byte-identical files.

### ARM64 correction: JSON mining tables and executable preflight

The preceding Parquet wording applies only where `pyarrow` is available.
Native CPython 3.13 ARM64 has no compatible `pyarrow` wheel, so
`native_mine_aoi.py` now automatically selects the dependency-free atomic JSON
table handoff: `target_embeddings.json`, `source_embeddings.json`, and
`mined_raw.json`. `filter_mined_by_cosine.py`, `filter_mined_history.py`, and
`emit_mined_sharegpt.py` accept the same JSON table format and preserve the
raw/filtered/history audit semantics. Do not mix `.parquet` and `.json` tables
within one mining iteration. The legacy CLI argument names remain unchanged;
their value may be a native `.json` table on ARM64.

Run `"$PYTHON" scripts/preflight_windows_arm64_aoi.py` before any native GPU stage. It
checks the target venv, a real 72 GiB CUDA allocation, Framework ports
29500/29502, the three patched source trees, the prepared VLM path, and the
checkpoint/output disk floor. See `windows-arm64-native.md` for the complete
native setup and no-container commands.

## Reference driver for the native route

`"$PYTHON" scripts/windows_native_deft_loop.py` is the driver that produced every
number on this page (wafer-win-7/8/9 and the recorded demo). The agent-driven
state machine in SKILL.md remains canonical; this file is the executable
transcript of one conforming drive: baseline gate, per-class mining with
`--filter-label`, cumulative balance policy (`DEFT_BALANCE=band|none`),
monotonic assembly, training with the recipe-default attention and an
annealed schedule, export with the zero-LoRA-keys check, the frozen gate,
keep-best model hygiene, W15 port pre-flight, and `pool_exhausted` handling.
It imports the skill's own scripts and `parquet_io` by path.

Host layout is environment-driven (defaults are the x64 reference
workstation): `DEFT_PYTHON` (the TAO venv interpreter), `DEFT_BANK`,
`DEFT_FRAMEWORK_CWD` (the patched cosmos-framework checkout — training runs
with that cwd, trap W9), `DEFT_MODEL_BASE` (the prepared VLM directory);
run knobs `DEFT_WAFER_WS`, `DEFT_WAFER_RUN`, `DEFT_WAFER_TRAIN_TOML`,
`DEFT_NG_TOPN`, `DEFT_OK_TOPN`, `DEFT_BALANCE`, `DEFT_LORA_TARGETS`.
Evaluate and export are invoked as `python -m cosmos_rl.evaluation.evaluate`
/ `python -m cosmos_rl.framework.export`, so the venv needs no console-script
wrappers (the ARM64 venv has none). `TAO_DISABLE_CLOUD_STATUS_CALLBACK=1` is
set by the driver. On the RTX Spark N1X the only differences from x64 are the
four host-layout variables above and `duckdb` in place of pyarrow.

## Evaluate semantics on this route

The Windows `cosmos-rl-evaluate` generates through HF Transformers, not the
pinned Linux image's vLLM engine. Consequences:

- The Linux image's `limit_mm_per_prompt` two-image cap does not exist here;
  `patch_eval_image_cap.py --probe` reporting the cap absent is expected.
- **The console summary lies in bare-label mode.** With the shipped
  `task.type = ""`, the evaluator prints `Total samples: 0 … Accuracy: n/a`
  on a fully successful run. All generations are in `results.json`, and
  `analyze_gaps.py` is the loop's authoritative scorer. Never grade a run
  from the evaluator console.
- Windows-vs-Linux numbers are same-items, different-decoder comparisons;
  label the evaluator when reporting across OSes.

## Deployment (DeepStream rtvi-vlm) — validated

The champion's merged export serves natively: vLLM inside the
`cosmos-framework-windows` container image (model bind-mounted **read-only** — no
`docker cp` needed for reads), fronted by the DeepStream rtvi-vlm still-image
API, Windows build (single JPEG or ordered multi-image requests,
`system_prompt` forwarded). A 12-image parity check stratified across all
four confusion cells scored **12/12** against the benchmark evaluator — the
pipeline reproduces the model's mistakes, not just its wins. Deployment
traps: (1) launching the serve container through an MSYS/Git-Bash shell
mangles `2>nul` in the `cmd /c` chain, silently emptying the `C:\cudashim`
CUDA_HOME shim — flashinfer then dies at import with `CUDA_LIB_PATH is not
set`; drop the redirects or launch from PowerShell. (2) the rtvi image bakes
`VLLM_MODEL=cosmos3-nano-reasoner` — set `VLLM_MODEL=` (explicitly empty) so
the chunker discovers the served model from `/v1/models`, or every job 404s
while `/ready` stays green. (3) the DeepStream still-decode branch is
JPEG-only (`jpegparse ! nvimagedec`).

### ARM64 deployment boundary

The preceding 12/12 proof is an x64 Windows container result. It is not an
ARM64 result. On this host, RTVI-VLM itself runs through the native Windows
launcher, but the champion-model handoff is still a required follow-up after
export: load the merged model in its isolated native serving environment, send
ordered JPEG pairs to the local API, and compare the exact `OK`/`NG` responses
with `analyze_gaps.py` on the frozen Benchmark. Do not claim ARM64 DeepStream
parity from the x64 container evidence.

## Stage-by-stage Windows status

| Loop stage | On Windows |
|---|---|
| checkpoint preparation | `prepare_cosmos3_vlm_checkpoint.py` (POSIX-guarded in this bank) |
| evaluate (proxy/benchmark) | native `cosmos-rl-evaluate`, validated |
| RCCA / routing / assemble / validate / report | pure Python, unchanged |
| mining | native SigLIP script above |
| **AnomalyGen SDG** | **not ported** — run remotely, pre-stage output, or run mining-only |
| train | cosmos-framework, native, converges (see validated configuration) |
| export | `cosmos-framework-export`, validated (0 lora keys) |
| deploy | x64 container proof exists; native ARM64 RTVI model-parity check is pending after champion export |

## Reference result (wafer-win-8, 2026-08-27)

| Stage | Accuracy | NG recall | Train set |
|---|---|---|---|
| baseline (zero-shot) | 0.4890 | 0.881 | — |
| iter1 | 0.8659 | 0.495 | 3,470 (9.6% NG) |
| iter2 — **gate met** | **0.9220** | 0.814 | 4,095 (22.6% NG) |

Early stop at iteration 2 of 5; total wall ~65 minutes on one GPU. The same
error-driven mining that overshot toward OK at iter1 self-corrected the mix
to the benchmark prior at iter2 — that correction, not any manual knob, is
what met the gate. Full ledgers, the failed configurations, the trainer
cross-checks, and the W14 root-cause forensics: `windows-route-evidence.md`.
