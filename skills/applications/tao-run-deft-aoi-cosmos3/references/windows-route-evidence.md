<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DEFT AOI on Windows — investigation record (2026-08-25 → 2026-08-28)

This is the evidence chain behind `windows-route.md`, kept verbatim as the
investigation unfolded. **Read the runbook for how to run; read this to know
why the runbook says what it says.**

> SUPERSESSION NOTE: sections written before 2026-08-27 attribute the
> training failure to "trainer optimization dynamics" and estimate the
> substitution cost at 9–12 accuracy points, and the early findings recommend
> lr 1e-5 and strict class balancing. All of that described a stack whose
> attention mask was silently corrupted (trap W14, transformers 4.57.6 +
> sdpa) and is superseded by "Root cause, named" below. It is preserved
> unchanged because the elimination ladder itself is what localized the bug.

## Validation evidence (run `deft-win-1`, 2026-08-25)

Workspace: `deft_cr3_nvpcb_0809` PCB subset — Proxy 28, frozen Benchmark 110,
Mining pool 54; all bare OK/NG two-image records; `check_annotations.py
--require-files` clean; gate `accuracy >= 0.99` (deliberately unmet).

| Frozen benchmark (110) | baseline | iter1 |
|---|---|---|
| accuracy | 0.800 | 0.755 |
| recall_ng | 1.000 | 0.100 |
| precision_ng | 0.577 | 1.000 |
| false accepts / rejects | 0 / 22 | 27 / 0 |
| unknowns / unparseable | 0 / 0 | 0 / 0 |

A second run (`deft-win-2`, same day) isolated the causes: with the mined set
balanced to 16 OK / 16 NG (target-similar pool NG supplements) and roughly
half the per-record training exposure, iter1 improved to accuracy 0.773 /
recall_ng 0.167 / 25 false accepts — better than run 1, still below baseline,
still flipped to over-acceptance. Both regressions are the loop working, not
the port failing: **lr 1e-4 LoRA on ~18-32 records destabilises the zero-shot
decision boundary regardless of label balance**, and the frozen gate caught it
in both runs. Knobs with remaining headroom: lower lr (1e-5), one epoch, a
larger mined pool, and — structurally — AnomalyGen synthetic NG, which is what
the SDG stage exists to supply and is not yet ported to Windows.

## Validation evidence (run `wafer-win-1`, 2026-08-26, public single-image dataset)

The single-image route is validated end-to-end on the CC BY 4.0 Roboflow
`poc-ongfq` wafer classification subset (mining pool 800 = 640 OK / 160 NG;
proxy 91 and frozen benchmark 91, both ~balanced), reproducing the semicon
mining-only recipe: gate accuracy >= 0.90, <= 5 iterations, 10 epochs/iter,
LoRA r16/a32/dropout 0.05, lr 1e-4, per-class mining (NG top-15 / OK top-3
via the miner's `--filter-label`), cosine + history filtering, NG >= 40%
batch floor, monotonic assemble.

| Frozen benchmark (91) | baseline | iter1 | iter2 | iter3 |
|---|---|---|---|---|
| accuracy | 0.6813 | 0.4945 | 0.5055 | **0.6923** |
| recall_ng | 0.911 | 1.000 | 0.000 | 0.889 |

- iter1/iter2 collapsed to the two constant answers (all-NG, then all-OK):
  lr 1e-4 over 10 epochs on ~150–190 records memorizes the newest batch's
  majority direction. The frozen gate rejected both.
- iter3, trained on the near-balanced monotonic set (141 NG / 119 OK),
  recovered discrimination and finished **above baseline** — the first
  positive mining-only delta on this port.
- iter4 mining produced 148 candidates and the history ledger dropped all of
  them, so the run closed with the **`pool_exhausted` stop reason**
  (`commit_stage.py` / `finalize_run.py` accept it with ledger evidence:
  `candidate_row_count > 0`, `selected_count == 0`). The 800-record subset
  pool is the binding constraint — the full 7,346-record `poc-ongfq` pool is
  the first knob for a higher final number, then lr 1e-5 / fewer epochs.
- Recipe deviations on this route, all deliberate: batch 8 is approximated as
  `steps_per_epoch = ceil(records/8)` under the packing batcher; activation
  checkpointing stays on (validated Windows setting; speed-only); a mined
  batch with zero NG is kept rather than emptied by the NG floor —
  rebalancing an over-rejecting model requires OK examples.

### Full-pool runs (`wafer-win-2` / `wafer-win-3`, 2026-08-26)

Same recipe on the full 7,346-record pool (1,677 NG / 5,669 OK; frozen
benchmark 910, majority-class floor **0.769**), with the balance policy
tightened per operator guidance to a cumulative band: keep every mined NG,
cap new OK so the combined set stays >= 40% NG (lowest-similarity dropped
first; assembled records never modified). Zero-shot baseline reproduced
bit-exactly across both runs: **0.4890** (recall_ng 0.881).

| Frozen benchmark (910) | lr | iter1 | iter2 | iter3 | terminal |
|---|---|---|---|---|---|
| `wafer-win-2` | 1e-4 | 0.3242 (rn 0.971) | 0.7692 = all-OK | 0.2308 = all-NG | stopped by operator |
| `wafer-win-3` | 1e-5 | **0.6791** (rn 0.538) | 0.3571 (rn 0.967) | 0.5066 (rn 0.833) | `pool_exhausted` at iter4 |
| `wafer-win-4`* | 1e-5 | 0.7791 (rn 0.195) | **0.7363 (rn 0.443)** | — | `pool_exhausted` at iter3 |
| `wafer-win-5`† | 1e-4 annealed | 0.7692 = all-OK | 0.4571 (rn 0.962) | — | stopped by operator |
| `wafer-win-6`† | 1e-4 annealed, 7-module LoRA | 0.7692 = all-OK | — | — | stopped by operator |

\* win-4 ran error-profile-weighted assembly: top-k 15 for BOTH classes, no
balance cap — the mined set mirrors the failure distribution instead of a
class-share target.

† win-5/6 are the external-reproduction attempts (a colleague reported >0.90
in one cosmos-rl iteration from the same 0.4890 baseline, train size ~1,700):
top-k 15/3, no cap — iter1 mines 1,446 records (334 NG / 1,112 OK), matching
that train set. win-5 added true cosine annealing over the actual step count
(the recipe's huge default cycle had silently pinned lr at ~99% of peak in
every earlier run); win-6 additionally matched cosmos-rl's 7-module LoRA
(`lora_target_modules` default here is attention-only q,k,v,o — cosmos-rl's
spec adds gate/up/down). Both still collapsed to all-OK at iter1 —
**data, schedule, and adapter capacity are all eliminated; the residual
variable is the trainer itself** (real-batch gradient averaging vs
token-packing normalization over single-token OK/NG supervision).

### Trainer calibration verdict (cosmos-rl on identical data, 2026-08-27)

The deciding experiment ran the pinned Linux cosmos-rl image on the exact
`train_iter_1.json` (1,446 records) with the reference run's own resolved
iter1 config — first under Docker Desktop/WSL2 on the Windows box, then to
completion as SLURM job 33140267 (cs-oci-ord, polar4, 1 GPU, ~1h45m at
3.4 s/step), with the adapter scored by the same native Windows evaluator
used for every other number:

| Same data, same knobs | trainer | accuracy | recall_ng |
|---|---|---|---|
| wafer-win-5 / win-6 | cosmos-framework (Windows) | 0.7692 | 0.000 (all-OK) |
| SLURM calibration | **cosmos-rl (Linux)** | **0.8615** | **0.576** |

Supporting evidence: full-pool supervised training (all 7,346 records) on the
Windows trainer scored 0.7681 / recall_ng 0.257 — the ~0.78 ceiling is the
trainer's, not the data's. The training curves show the mechanism: the
packing trainer collapses its loss to ~0.02 within one epoch (single-token
OK/NG supervision receives outsized effective steps under packed
normalization), while cosmos-rl's per-batch averaging keeps gradient signal
alive into epoch 4 at the same nominal lr.

Root cause, established by a forensic chain on identical data (unpacked
`SimpleBatcher(8)` isolation run + in-process replay of the exporter on a
kept DCP checkpoint):

- **Export/merge are value-perfect** — teacher-forced p(ground-truth) on
  training records is bit-equal before and after
  `merge_and_strip_lora_adapters_`; `dcp.load` mutates the module tensors as
  intended. All earlier suspicion of the export path is withdrawn.
- **Batching is exonerated** — unpacked training scores identically to
  packed at eval (0.7692 / recall_ng 0.000).
- **The framework training step underfits the task.** With honest (unpacked)
  batching, training loss plateaus at ~0.21 and the converged adapter
  assigns only p≈0.27 to NG on its *own training records* (teacher-forced),
  i.e. it never leaves the OK-prior — while cosmos-rl reaches ~1e-5 loss on
  the identical data, LoRA, lr, and step count. The packed runs' apparent
  convergence (loss ~0.02 in one epoch) was the packing shortcut — neighbor
  answers visible in-context — not learning. The honest signal all along was
  the in-training validation curve (never below ~0.5; note the val slice is
  the first 16 records of the assembled JSON, which are all-NG by assembly
  order). A component-isolation ladder (identical data throughout) localizes
  the defect further: raw model + LoRA + manual AdamW fits a
  trainer-identical fixed batch in 15 steps; the framework's
  OptimizersContainer, LambdaCosine scheduler (lr verified 1e-4 at the param
  groups), fused-AdamW + bf16 adapters, collator pixel ordering, activation
  checkpointing, and the real config-composed `VLMModel.training_step` on a
  fixed batch ALL pass — but the same composed stack on **streamed** batches
  plateaus at 0.2–0.39 for 4 epochs with no trainer loop involved, matching
  the full trainer's 10-epoch plateau, while cosmos-rl fits the identical
  stream to ~1e-5 by epoch 4. **Learning does not accumulate across
  batches** despite healthy gradient norms (0.3–0.9) and correct lr;
  per-batch memorization works. File upstream with this ladder: the repro
  isolates a cross-batch optimization-dynamics defect in the composed
  cosmos-framework VLM SFT stack, reproducible without the trainer in ~30
  lines (probe scripts preserved in the wafer workspace).
  (Support notes: the recipe TOML's `[dataloader_train]` block maps to
  PoolPackingBatcher-specific fields and must be dropped when swapping
  batchers; keep assembled train sets at a batch-size multiple — the
  loader's resume guard rejects epoch-spanning batches.)

**Consequence for this route: the cosmos-framework trainer substitution costs
roughly 9–12 accuracy points on bare-label AOI data.** A Windows port of
cosmos-rl SFT is now evidence-backed (NCCL→Gloo is the hard seam; the
launcher and 4 of 5 actions already run natively). A reference >0.90 result
in one iteration remains unmatched even by the calibration (residuals:
~1,700-vs-1,446 record sets, native-reasoner vs converted-PTM base, vLLM vs
HF eval decode).

Running the Linux cosmos images on the Windows box via Docker Desktop (WSL2)
requires four workarounds, all measured here: `LD_PRELOAD` the WSL-injected
`libcuda.so.1` (torch CUDA init otherwise fails), `NCCL_CUMEM_ENABLE=0`
(cuMem VMM unsupported over GPU-PV), patch cosmos-rl's hard-injected
`expandable_segments:True` to False (same VMM failure in the allocator), and
stage models off 9p bind mounts. Expect ~GPU-native step time regardless of
platform for this trainer (~3.4 s/step) — but per-epoch DCP checkpoints will
balloon the WSL2 vhdx (~35 GB each); save at the final epoch only.

Findings, in causal order:

1. **lr 1e-4 saturates LoRA to a constant answer** on this dataset at every
   set size and balance — win-2 pinballed between the all-OK score (0.7692)
   and the all-NG score (0.2308). At **lr 1e-5 the model discriminates**;
   the best mining-only result is win-3 iter1: **0.6791, +19 points over
   baseline** (835 records, 40% NG).
2. **Cumulative class share steers the prior** even at 1e-5: 40% NG -> 0.679;
   59% NG -> 0.357 (over-reject); back to 49% -> 0.507. The discriminating
   iterations are exactly the balanced ones — balance the set before turning
   any other knob.
3. The oscillation never fully damps at 10 epochs/iteration because every
   iteration retrains from base on the cumulative set; remaining headroom for
   this dataset is fewer epochs (1–3) or train-val early stopping, a
   symmetric band (cap NG additions too), and a deeper OK top-k for
   corrective batches. The 0.90 gate was not met mining-only.
4. The `pool_exhausted` stop ran end-to-end unattended in win-3 and win-4.
5. **Error-profile weighting trades the two error types against each other**
   (win-4): the 3,470-record iter1 set (9.6% NG — mirroring a baseline whose
   errors are 93% false alarms) produced the highest raw accuracy of any run,
   0.7791, barely above the all-OK floor with recall_ng collapsed to 0.195 —
   the aggregate-up/escapes-shipped trap. iter2's NG correction (cumulative
   4,174 records at 24.9% NG, the pool's natural ratio) was the first
   *damped* step: 0.7363 with recall_ng 0.443 and both classes genuinely
   traded. The mining-only ceiling on this pool sits around accuracy
   0.68–0.78 depending on where recall_ng is priced; **no configuration met
   the 0.90 gate**. A reported >0.90-in-one-iteration result (~1,700 train
   records) is not reproduced by data composition alone — calibrate against
   that run's baseline, eval split, trainer, and lr before treating 0.90 as a
   mining-only target.

Operational note: per-iteration disk hygiene deletes each ~33 GB DCP after
its verified export, and deletes a previous merged model only when the newer
iteration scores at least as well — the best-so-far weights always survive
(win-3 predates that guard; regenerate its iter1 by retraining from the
committed `iter1/assemble/train_iter_1.json`, ~7 minutes).

### Root cause, named (2026-08-27): the attention implementation

The final attribution matrix — identical data (1,440-record mined set),
identical unpacked recipe, LoRA r16/a32, lr 1e-4 annealed, in-training
validation on the all-NG head of the train JSON:

| Platform | attn_implementation | val @ epoch 1 | terminal |
|---|---|---|---|
| Windows | sdpa | 0.669 | plateau through 10 epochs |
| Windows | eager | 0.615 | plateau |
| Linux (pinned framework image) | sdpa | 0.540 | plateau trajectory |
| Linux (pinned framework image) | cosmos (default) | 0.116 | **0.000045 at epoch 4** |
| Windows | sdpa + mask gate (W14 fix) | 0.019 | **4e-6 at epoch 10; benchmark 0.8780** |
| Windows | cosmos (recipe default) | 0.0179 | converging; 2-epoch trajectory matches sdpa+gate |

**`sdpa` and `eager` do not train this model on either platform** — but not
because of the kernels themselves. The defect, run to ground on 2026-08-27:

The trainer's `training_step`/`validation_step` unconditionally
`data.pop("attention_mask", None)` (`model/generator/vlm_model.py`). The
`cosmos` and flash-attn paths never read that mask (the `cosmos` adapter
in fact raises on a non-None mask), so on Linux the pop is invisible. The
HF-native `sdpa`/`eager` paths need it: under transformers 4.57.6 with
`attention_mask=None`, `Qwen3VLTextModel` feeds the M-RoPE temporal
position row into `create_causal_mask`
(`modeling_qwen3_vl.py`), and the packed-sequence detection in
`masking_utils.py` reads the constant temporal index across each image
span as fabricated sequence boundaries. The answer token is masked off
from the image and the prompt; the model converges to the label prior at
exit 0. Evaluation supplies the mask on its own path and is immune, which
kept every post-training check green. Upstream this is fixed in
transformers v5.3.0 (HF PR #44158); no 4.x backport exists, and the
pinned venv is 4.57.6.

**The fix (trap W14)** is a one-condition gate at both pop sites: keep the
mask when `attn_implementation` is `sdpa`/`eager`, keep popping it
otherwise (the `cosmos` adapter requires None). Validation ladder:
mask-kept streaming probe 0.0005; real trainer val@epoch-1 0.669 → 0.019;
full 10-epoch pipeline on the identical 1,440-record mined set scored
**0.8780 accuracy / 0.605 NG recall / 0.819 NG precision** on the
910-record benchmark — above the cosmos-rl Linux calibration (0.8615 /
0.576) on the same data.

**`cosmos` attention runs natively on Windows.** It is not a compiled
Linux kernel: `hf_model.py` registers it via `AttentionInterface` as an
adapter over torch's built-in cuDNN SDPA op. Verified forward+backward on
this box (sm_120, max|diff| 0.0039 vs sdpa), and it trains: val@epoch-1
0.0179 with zero overrides. The Windows-mandatory
`attn_implementation=sdpa` override was never necessary. Caveats:
`deterministic=True` disables the cudnn SDPA backend; head_dim 128 is the
sm120 cudnn ceiling (this model sits exactly at it); the adapter raises on
a non-None attention mask, which the W14 gate preserves.

Consequences:

- **Windows checkpoints trained before the fix are invalid** — their
  numbers reflect the broken mask path, not DEFT or the recipe.
- **Preferred Windows training config: leave `attn_implementation` at the
  recipe default (`cosmos`).** No override, no source patch on the
  attention path, full config parity with Linux. Keep the W14 gate as
  defense-in-depth for anyone who selects `sdpa`/`eager` explicitly.
- Remote Linux training is no longer required. The SLURM route remains
  documented above as a calibration/cross-check tool (framework val
  4.5e-5; cosmos-rl benchmark 0.8615 on the same mined set).
- A cosmos-rl SFT Windows port is no longer assumed blocked; it would need
  its own mask-path audit, not a kernel port.

### Loop closure (2026-08-27): accuracy >= 0.90 gate met natively

First post-fix full DEFT runs on the wafer benchmark (910 records,
700 OK / 210 NG, baseline 0.4890), cosmos attention, 7-module LoRA,
lr 1e-4 annealed, 10 epochs/iter:

- `wafer-win-7` (NG 15 / OK 3, cumulative 40% NG floor): iter1
  0.7835/rn 0.800, iter2 0.7473/rn 0.914, iter3 0.7890/rn 0.867 —
  the recall-forward operating point.
- `wafer-win-8` (symmetric top-15, no floor): iter1 3,470 recs at 9.6% NG
  -> 0.8659/rn 0.495 (NG-diluted); iter2's error-driven mining re-landed
  the mix at 22.6% NG (4,095 recs) -> **0.9220 / rn 0.814, gate met,
  early stop at iteration 2**.

**Deployment through DeepStream rtvi-vlm — validated (2026-08-27).** The
gate champion's merged export serves natively: vLLM inside the
cosmos-framework-windows container image (model bind-mounted read-only — no
`docker cp` needed for reads), fronted by the DeepStream rtvi-vlm still-image
API (Windows build). A 12-image parity check stratified across all
four confusion cells scored 12/12 against the benchmark evaluator's
per-sample verdicts — the pipeline reproduces the model's mistakes, not
just its wins. Two deployment traps: (1) launching the serve container
through an MSYS/Git-Bash shell mangles `2>nul` in the `cmd /c` chain, which
silently empties the `C:\cudashim` CUDA_HOME shim and surfaces three layers
away as a flashinfer `CUDA_LIB_PATH` import crash — drop the redirects or
launch from PowerShell; (2) the rtvi image bakes `VLLM_MODEL=cosmos3-nano-reasoner`
as a default — set `VLLM_MODEL=` (explicitly empty) so the chunker
discovers the served model name from `/v1/models`, or 404s follow.

Two findings transfer beyond this dataset. (1) Accuracy peaks when the
cumulative train mix matches the benchmark class prior (23% NG here);
over- or under-shooting either direction costs points, and NG recall
moves opposite to the OK fraction. (2) Routing policy is the size lever:
this route forwards every Proxy gap as a mining query (458 OK targets ->
3,136 mined OK at iter1), while target-side label balancing (as in the
DeepStream demo pipeline) caps the query list first and yields ~2x
smaller sets. The upstream skill constrains target provenance (Proxy
FA/FR only) but is silent on selection policy — pin one explicitly when
comparing runs across implementations.
