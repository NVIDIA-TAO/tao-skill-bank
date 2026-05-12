---
name: tao-hf-finetune
description: >
  End-to-end fine-tune, evaluation, inference, and HF Hub push for any HuggingFace
  CV or VLM model on local NVIDIA GPUs using an NGC PyTorch container.
  Research-first: fetch the model card, author's finetune example, and HF task
  docs live before generating any code. References under references/ act as a
  fallback safety net when live research is silent or a known-issue is detected.
  Six-step workflow: inspect & qualify, hardware & NGC image, research, generate
  & smoke, train + eval + infer, push & emit rerun skill. Produces a trained
  checkpoint, baseline + post-train eval JSON, 5 inference samples, a HuggingFace
  Hub push with a model card, and a self-contained rerun skill. Optional
  deliverables (PROGRESS.md log, PDF/HTML report, fake-data unit tests) opt in
  via config flags. Supported tasks: image-classification, object-detection,
  semantic-segmentation, instance-segmentation, depth-estimation,
  image-text-to-text (VLM SFT / LoRA), LLM SFT / DPO / GRPO.
  Rejects models whose AutoConfig fails to load.
license: Apache-2.0
version: "0.1.0"
author: NVIDIA CORPORATION
tags:
  - finetuning
  - huggingface
  - nvidia-tao
  - computer-vision
  - training
tools:
  - Read
  - Bash
  - Write
  - WebFetch
compatibility: Requires docker + nvidia-container-toolkit, NVIDIA GPU (driver ≥ 545, ≥ 24 GB VRAM for ≤3B models), ~40 GB free disk, HF_TOKEN, and WANDB_API_KEY/WANDB_PROJECT.
metadata:
  author: NVIDIA Corporation
  version: '0.1'
allowed-tools: Read Bash Write WebFetch
---
<!--
Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->


# tao-hf-finetune

Local NVIDIA GPU fine-tuning for HuggingFace models, grounded in live-fetched
documentation, with curated references as a fallback safety net. One NGC
container, a small set of focused scripts, one push to HF Hub. Behavior is
governed by the rules in this file — follow them, do not improvise around them.

**Order of authority (highest first):**

1. **User input** — explicit `model_id`, `dataset_id`, `training_method`, `config.yaml` overrides.
2. **Live research** — model card, HF repo example, author finetune script, HF task docs, paper. Always fetched. See Step 3 + `references/research-priorities.md`.
3. **Curated references** (`references/*.md`) — fallback when live research is silent or ambiguous.
4. **Your training-data memory** — last resort. Treat as suspect; cross-check against (2) or (3).

If (2) and (3) conflict on an API call, (2) wins (newer). If they conflict on a
method detail (collator, LoRA targets, augmentation), (2) wins for the *specific*
model; (3) for the generic shape. Note the discrepancy in a comment at the source
line.

---

## Inputs

**Required:**
- `model_id` — HuggingFace model ID, e.g. `google/vit-base-patch16-224`
- `HF_TOKEN` — read access to the model; write access if `push_to_hub` is on
- `WANDB_API_KEY`, `WANDB_PROJECT` — monitoring

**Dataset — exactly one:**
- `dataset_id` — HuggingFace dataset ID *(source: `hf`)*
- `local_dataset_path` — local folder or file *(source: `local`)*. Optionally
  `local_dataset_format` ∈ {auto, imagefolder, coco, voc, jsonl, arrow,
  parquet, csv}. Default: auto-detect.
- *(omit)* — agent recommends popular datasets *(source: `recommend`)*

**Optional (have defaults):**
- `task_type` — auto-detected from config + model card
- `n_train=10000`, `n_eval=1000`, `n_epochs=3`, `lora_r=16`
- `output_dir=./output/<model_short_name>`
- `hf_model_repo` — push target. If unset and HF_TOKEN has write access,
  auto-derived as `<whoami>/<model_short_name>-finetuned`.
- `push_to_hub=True` — set explicitly to `False` to skip
- `skip_baseline=False` — skip zero-shot baseline eval

**Optional deliverables (off by default):**
```yaml
emit_progress_log: false   # output_dir/PROGRESS.md (per-step ✅/⚠️/❌ journal)
emit_report:       false   # reports/report.{pdf,html} with curves & samples
emit_unit_tests:   false   # tests/ with fake-data heterogeneous-batch tests
```

All values live in `output_dir/config.yaml`. Never hardcode in Python.

---

## References — fallback safety net

Consulted **only** when live research is silent, ambiguous, or unavailable. Live
docs always win for the specific model and current API.

### Always-on (consulted in the workflow)

| File | Step | Role |
|---|---|---|
| `compat-workarounds.md` | 1 | Known-issue registry; auto-applied via `detect` rules |
| `model-discovery.md` | 1 | `model_type` → AutoModel/processor mapping (when card silent) |
| `dataset-recommendations.md` | 1 | Vetted datasets for `source = recommend` |
| `dataset-sources.md` | 1 | Local format detectors + COCO/VOC/imagefolder/jsonl loaders |
| `dataset-patterns.md` | 4 | Universal `prepare_data.py` skeleton |
| `hardware-container.md` | 2 | NGC selection (offline fallback), GPU/disk audit, multi-GPU |
| `research-priorities.md` | 3 | 6-priority live-fetch ladder + extract/record + conflict rules |
| `cv-scripts.md` | 4 | CV scaffold (file names, CLI, config schema). **Don't copy `[FETCH LIVE]` blocks** |
| `vlm-scripts.md` | 4 | VLM/LLM scaffold (TRL/PEFT). **Don't copy `[FETCH LIVE]` blocks** |
| `docker-runs.md` | 4, 5 | Canonical `docker run` invocations for every command |
| `hub-push.md` | 6 | HF Hub push Python block + model card template |
| `pipeline-skill-template.md` | 6 | `run-<short>/SKILL.md` rerun template |
| `deliverables.md` | 4, 6 | Final directory layout + README results section |

### Opt-in (only when their flag is set)

| File | Flag | Adds |
|---|---|---|
| `progress-tracking.md` | `emit_progress_log: true` | PROGRESS.md template |
| `testing.md` | `emit_unit_tests: true` | Fake-data heterogeneous-batch tests |
| `reporting.md` | `emit_report: true` | `report.py` (PDF + HTML, reads `trainer_state.json`) |

**Rule:** before falling back to a reference, log the live source you tried and
why it was insufficient (in `config.yaml` `notes:`, and PROGRESS.md if enabled).

**`[FETCH LIVE]` markers in `cv-scripts.md` / `vlm-scripts.md`** are a
research checklist, not code to inline. If a `[FETCH LIVE]` block has no Step 3
finding, refetch the listed URL.

---

## Core rules

### Your knowledge of HF libraries is outdated

You do not know current APIs for `transformers`, `trl`, `datasets`, `peft`, or
`accelerate`. Your internal knowledge WILL produce wrong imports, wrong trainer
arguments, wrong collator constructors, and hallucinated config fields. Before
writing any ML code, fetch the live sources listed in
`references/research-priorities.md`. Never generate training code from memory
alone.

### Mistakes you WILL make without research

- **HALLUCINATED IMPORTS** — modules renamed or removed. Read one current
  example script first.
- **WRONG TRAINER ARGUMENTS** — args that don't exist in the installed
  `transformers`/`trl`. Fetch the docs for `TrainingArguments` / `SFTConfig`.
- **WRONG DATASET FORMAT** — assuming columns. Stream 20 rows, print columns
  *before* writing the collator.
- **BATCH FAILURES** — launching multiple runs before verifying one. Smoke-test
  (`--max_steps 1`) on real data before the full run.
- **SILENT DATASET SUBSTITUTION** — requested dataset fails, you quietly switch.
  Stop. Tell the user. Ask.
- **SCOPE-CHANGING FIXES** — on OOM you switch SFT→LoRA, shrink `max_length`,
  disable monitoring. Don't. Fix with the minimal change that preserves the
  request.
- **LOST MODELS** — local disk can be cleared. `push_to_hub=True` always unless
  user explicitly says `False`.
- **HIDDEN LOSS** — `tqdm` bars hide loss. In `TrainingArguments`:
  `disable_tqdm=True`, `logging_strategy="steps"`, `logging_first_step=True`,
  `logging_steps=10`.
- **NO AUGMENTATION (CV)** — `AutoImageProcessor` only resizes+normalizes.
  Without `RandomResizedCrop` + `RandomHorizontalFlip` you can drop ~30-40 points
  on small datasets. Always fetch training transforms from the HF task doc or
  author's script — not memory.

### Never without user approval

- Change `model_id`, `dataset_id`, or `training_method`.
- Change task type mid-run (e.g. full → LoRA, classification → detection).
- Skip the smoke test or preflight check.
- Disable monitoring to "fix" an error.

### Error recovery — minimal change, same approach

- **OOM**: halve `per_device_train_batch_size`, double
  `gradient_accumulation_steps` (effective batch unchanged), enable
  `gradient_checkpointing=True`. Still OOM → ask user for bigger GPU.
- **NaN loss**: reduce LR 10×, set `max_grad_norm=1.0`.
- **Flat loss**: inspect label masking and LR. Usually a collator bug.
- **Same error 3× in a row**: stop, summarize, ask. Do not loop.
- **Import/API error**: refetch the relevant doc page — the API moved.

### Dataset format by task

Verify columns BEFORE writing the collator:

- `image-classification` — `image` + `label` (or `labels`)
- `object-detection` — `image` + `objects` with `bbox` + `category` (or `label`)
- `semantic-segmentation` — `image` + `segmentation` (or `label`, or `mask`)
- `depth-estimation` — `image` + `depth_map`
- `image-text-to-text` (VLM SFT) — `image` + `messages` (conversation), or
  `image` + `text` / `question` + `answer`

Mismatch + rename fixes it → do it in `prepare_data.py`. Restructuring needed →
stop and ask.

### Hardware sizing (bf16)

| Model size | GPU |
|---|---|
| ≤3B | 24 GB (A10, L4, T4-medium) |
| 7-13B | 80 GB (A100-80, H100) |
| 30B+ | multi-GPU (2-4× 80 GB) or LoRA on 1× 80 GB |
| 70B+ | 8× 80 GB or LoRA |

Rule of thumb: bf16 weights ≈ 2 B/param; optimizer states add ≈ 3-4× weights for
full finetune, ~0 for LoRA. If full won't fit and user didn't ask for LoRA, ask
before switching.

---

## Workflow — 6 steps

Single pass, sequential. Each step has a clear gate before the next begins.

### Step 1 — Inspect & qualify

**Goal:** decide whether to proceed at all. Probe model, probe dataset, apply
accept/reject, register applicable compat fixes, write the initial `config.yaml`.

**1a. Probe model:**

```bash
python3 -m venv /tmp/hf-inspect-venv && source /tmp/hf-inspect-venv/bin/activate
pip install -q transformers huggingface_hub datasets Pillow
python - <<'PY'
import os, sys
from transformers import AutoConfig
from huggingface_hub import model_info
mid = os.environ["MODEL_ID"]; tok = os.environ["HF_TOKEN"]
try:
    cfg = AutoConfig.from_pretrained(mid, token=tok, trust_remote_code=True)
except Exception as e:
    print(f"REJECT: AutoConfig failed — {e}"); sys.exit(1)
info = model_info(mid, token=tok)
print("model_type:", cfg.model_type)
print("architectures:", getattr(cfg, "architectures", []))
print("tags:", info.tags)
print("hidden_size:", getattr(cfg, "hidden_size", None))
print("num_kv_heads:", getattr(cfg, "num_key_value_heads", None))
print("num_attn_heads:", getattr(cfg, "num_attention_heads", None))
PY
```

Detect `task` from `architectures` + `tags` + model-card body. If the card
doesn't show `from transformers import AutoModelFor...`, fall back to
`references/model-discovery.md` and log the fallback under `notes:`.

**1b. Probe dataset:**

For `source = recommend`, present 3–5 picks from
`references/dataset-recommendations.md` to the user, then re-run with the chosen
`dataset_id` / `local_dataset_path`.

```python
# HF source — loadability + schema probe (catches gated / script-based / missing)
from datasets import load_dataset, load_dataset_builder
import os
DID = os.environ["DATASET_ID"]; TOK = os.environ["HF_TOKEN"]
try:
    load_dataset_builder(DID, token=TOK)
    ds = load_dataset(DID, split="train[:20]", token=TOK)
except Exception as e:
    print(f"REJECT dataset: {type(e).__name__}: {e}"); raise
rows = list(ds)
print("columns:", list(rows[0].keys()))
for col, val in rows[0].items():
    print(f"  {col}: {type(val).__name__}")
```

For `source = local`, see `references/dataset-sources.md` for format detection
and loaders.

Verify columns match the task schema (Core rules → Dataset format). Mismatch +
rename fixes it → write the rename into `prepare_data.py`. Otherwise stop.

**1c. Apply accept/reject:**

REJECT if:
- `AutoConfig` raised
- task can't be determined
- task is not CV / VLM / SFT-LLM (out of scope)
- no recipe source exists at all (no card example, no HF repo script, no author
  finetune, no task doc, no paper)
- dataset is gated / script-based / missing (loadability probe failed)

Stop and report the specific reason. Do not proceed.

**1d. Walk compat-workarounds:**

For every entry in `references/compat-workarounds.md`, evaluate its `detect`
expression against `cfg` and the detected `task`. Hardware-dependent rules
(those needing `hw`) are deferred to Step 2.

Record matches in `config.yaml` under `applicable_workarounds:` (id + fix type +
one-line reason). Each becomes a Dockerfile block, requirements pin, config
override, or runtime env in Step 4.

**1e. Write `config.yaml` skeleton:**

```yaml
model_id: <…>
task: <…>
dataset_id: <…>             # or local_dataset_path
research_sources: []         # filled in Step 3
applicable_workarounds: [<…>]
notes: []                    # log any reference fallback
push_to_hub: true            # default
```

deactivate; rm -rf /tmp/hf-inspect-venv

**Gate:** `config.yaml` exists with model, dataset, task, applicable_workarounds.
Do not proceed if any field is missing.

---

### Step 2 — Hardware audit & NGC image

**Goal:** verify Docker + GPU + disk, pick the NGC PyTorch image live, finalize
hardware-dependent compat rules.

**2a. Audit (hard gate):** run the preflight script. It hard-fails on missing
driver, docker daemon, `nvidia-container-toolkit` registration, `--gpus all`
smoke (against a CUDA tag derived from the driver's max supported CUDA), or
missing `HF_TOKEN`. Free-disk on `/` is a **soft warn** at 100 GB (override via
`MIN_DISK_GB`); the script continues so the user can decide. On hard-fail it
prints a distro-aware install hint parsed from `/etc/os-release` and exits
non-zero. **Do not proceed to Step 4 on a hard-fail** — Step 4's `docker build`
pulls a 20+ GB NGC base image, and a missing `nvidia-container-toolkit` only
surfaces at `prepare_data.py` time as the cryptic
`could not select device driver "" with capabilities: [[gpu]]`.

```bash
bash scripts/preflight.sh    # see scripts/preflight.sh — all 6 checks live there
```

Record `gpu_count`, `gpu_name`, `driver_major`, `vram_gb_per_gpu` in
`config.yaml`.

**2b. Pick NGC image (live):**

```
WebFetch https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
```

Find the **PyTorch NGC container** section. Pick the highest-versioned image
where:
- `Min driver ≤ detected driver_major`
- PyTorch is a **stable release** (not `aN` / `rcN`)

If WebFetch fails: fallback rules in `references/hardware-container.md`. Default
fallback: `nvcr.io/nvidia/pytorch:24.09-py3` (driver ≥ 545; SDPA+GQA bug — if
the model has `num_key_value_heads < num_attention_heads`, set
`attn_implementation: "eager"` in config).

Record `ngc_image` in `config.yaml`.

**2c. Re-evaluate hardware-dependent compat rules:**

Re-run the `compat-workarounds.md` walk for entries whose `detect` expression
needs `hw`. Update `applicable_workarounds:` in place.

**2d. Model-fit check:** estimate `param_bytes ≈ 2×param_count` (bf16). If
> 60% of `vram_gb_per_gpu × 1e9`, recommend LoRA in the user-facing summary.

**Gate:** `config.yaml` has `ngc_image`, `gpu_count`, `gpu_name`, `driver_major`,
`vram_gb_per_gpu`. Hardware-dependent compat fixes are recorded.

---

### Step 3 — Research the recipe

**Goal:** fetch the live recipe. The agent's training-data knowledge of
`transformers`/`trl`/`peft` is treated as suspect — Step 3 is non-negotiable.

Walk `references/research-priorities.md` in priority order (Priority 1 → 6).
Stop once you have, for the detected task:

- `AutoModel` / processor class
- Train + eval transforms
- Collator
- `compute_metrics`
- Hyperparameter hints (LR, batch size, epochs, scheduler)

Record findings in `meta/recipe.md` and append source URLs to
`config.yaml: research_sources:`. If a slot has no live finding, fall back to
the matching scaffold reference (`cv-scripts.md` / `vlm-scripts.md`) and log
"fallback to scaffold — no live source for <slot>" under `notes:`.

**Conflict resolution rules** are in `references/research-priorities.md`.

**Gate:** every required slot above is filled, with a source URL or an explicit
scaffold-fallback note.

---

### Step 4 — Generate project & smoke-test

**Goal:** write all scripts, build the image, prepare data, run a 1-step smoke
on real data. One `docker build`, two `docker run`s.

**4a. Generate project files** in `output_dir/`:

| File | From | Notes |
|---|---|---|
| `config.yaml` | Steps 1-3 + user input | already started |
| `Dockerfile` | template below + compat injections | layer order: deps → compat → code |
| `requirements.txt` | task baseline + compat pins | don't pin without cause |
| `prepare_data.py` | scaffold + Step 3 | save Arrow to `data/{train,eval}` |
| `train.py` | scaffold + Step 3 recipe | reads `config.yaml`, supports `--smoke --max_steps N` |
| `run_eval.py` | scaffold + Step 3 | **MUST** be `run_eval.py` (collides with HF `evaluate` lib if named `evaluate.py`) |
| `infer.py` | scaffold + Step 3 | writes `reports/inference_samples/<i>_input.jpg`, `_pred.jpg`, `_meta.json` |
| `merge_lora.py` | scaffold | only for VLM with LoRA |
| `.gitignore` | `data/`, `checkpoints/`, `logs/`, `wandb/`, `reports/inference_samples/`, `.env`, `__pycache__/`, `*.pyc` | |

Authority order while writing: live research from Step 3 → scaffold reference
(`cv-scripts.md` / `vlm-scripts.md`) for **structure only**, never their
`[FETCH LIVE]` blocks. Apply each `applicable_workarounds` entry: Dockerfile
blocks, requirements pins, config overrides, runtime env vars.

Every generated `.py` file (`prepare_data.py`, `train.py`, `run_eval.py`,
`infer.py`, `merge_lora.py`, and any `tests/*.py`) must start with the NVIDIA
Apache-2.0 copyright header as a `#`-prefixed comment block — same text as the
HTML copyright comment used in the rerun skill, just commented for Python:

```python
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

If you generate an emitter script, make it fail unless every emitted `.py`
begins with that header.

If `emit_unit_tests: true`, also generate `tests/` per `references/testing.md`.

**Dockerfile template:**

```dockerfile
ARG NGC_IMAGE=nvcr.io/nvidia/pytorch:24.09-py3
FROM ${NGC_IMAGE}

ENTRYPOINT ["/bin/bash", "-c"]
WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# {{COMPAT_DOCKERFILE_BLOCKS}}     ← injected from applicable_workarounds
# {{COMPAT_ENV_VARS}}                ← injected from applicable_workarounds

COPY *.py ./
COPY config.yaml ./
```

**4b. Build, prepare, smoke** (commands: `references/docker-runs.md` §1-3):

```bash
docker build -t run-<short>:latest .
# → docker-runs.md §2: prepare_data
# → docker-runs.md §3: smoke (--smoke --max_steps 1)
```

Smoke pass criteria (in `logs/smoke.log`):
- No exception
- Loss is finite (not `0.0`, not `NaN`)
- `grad_norm > 0` at step 1

If `emit_unit_tests: true`, also run `pytest tests/` inside the container.
Failure → STOP. Do not proceed.

**4c. Preflight summary** — print and verify every field is filled before
launching full training:

```
─ PREFLIGHT ────────────────────────────────────────
reference implementation:  <URL from Step 3>
dataset columns verified:  <col1, col2, …>
push_to_hub:               <repo_id>
monitoring:                wandb <project>/<run_name>
ngc_image:                 <image tag>
hardware:                  <gpu_count>× <gpu_name>
smoke test:                PASSED (loss=X.XX, grad_norm=Y.YY)
────────────────────────────────────────────────────
```

**Gate:** project files written, image built, smoke PASSED, preflight has no
blank fields.

---

### Step 5 — Train, evaluate, infer

**Goal:** baseline eval, full training, post-train eval, optional LoRA merge,
5 inference samples. All commands: `references/docker-runs.md` §4-8.

| Sub-step | docker-runs.md | Skip if |
|---|---|---|
| 5a. Baseline eval (zero-shot) | §4 | `skip_baseline: true` |
| 5b. Full training (detached) | §5 | — |
| 5c. LoRA merge | §6 | not VLM-with-LoRA |
| 5d. Post-train eval | §7 | — |
| 5e. Inference (5 samples) | §8 | — |

Multi-GPU: prepend `torchrun --nproc_per_node=$gpu_count` to `python train.py`.

While training streams, watch `docker logs -f hft_train` for:
- Loss drops within 10-20 steps → working
- Flat loss → collator / label-masking bug; stop
- NaN loss → LR too high; stop, reduce LR, retry
- OOM → halve batch, double grad_accum, enable gradient checkpointing

If `emit_report: true`, run `report.py` after Step 5e per `references/reporting.md`.

**Gate:** all of:
- `checkpoints/final/` (or `checkpoints/merged/` for LoRA) exists
- `reports/eval_results.json` has a numeric primary metric
- `reports/baseline_results.json` exists (unless skipped)
- `reports/inference_samples/` has 5 samples
- wandb URL shows descending loss

---

### Step 6 — Push & emit rerun skill

**Goal:** publish the run and ensure it can be reproduced without re-research.

**6a. Push to HF Hub** — use the script in `references/hub-push.md`. Pushes:
- model weights (merged or final)
- model card (`README.md`) generated from `config.yaml` + eval JSONs
- `results/{eval,baseline}_results.json`, `config.yaml`, `Dockerfile`,
  `requirements.txt`, `inference_samples/*.jpg`
- `report.{pdf,html}` if `emit_report: true`

Skip iff `push_to_hub: false` is explicit in `config.yaml`.

**6b. Emit rerun skill** at `<output_dir>/skills/run-<short>/SKILL.md` per the
template in `references/pipeline-skill-template.md`. Every `<placeholder>` must
be substituted with a real value. Literal placeholders in the output are a bug.
Include full YAML (`license`, `compatibility`, `metadata`, `allowed-tools`) and
the NVIDIA copyright notice in an HTML comment (`<!--` … `-->`) immediately after
the closing `---`, as in that template. If you generate an emitter script, make it fail unless the emitted `SKILL.md` contains those fields and the HTML copyright comment.

**Gate (Done criteria):** all of:
- Step 5 gate met
- HF Hub repo exists at the resolved URL with weights + card + `results/`
  (unless `push_to_hub: false`)
- `<output_dir>/skills/run-<short>/SKILL.md` exists, no `<placeholder>` left,
  with metadata + copyright HTML comment per `pipeline-skill-template.md`

**Final message to user** — terse, with direct URLs:
- wandb URL
- HF Hub URL
- primary metric: baseline → fine-tuned (Δ)
- path to `reports/inference_samples/`
- path to `<output_dir>/skills/run-<short>/SKILL.md`

---

## Error playbook

When you hit an error, consult this table before redesigning anything. Apply the
minimal fix that keeps the user's original request intact.

The compat-workarounds registry at `references/compat-workarounds.md` is the
durable form of this table — entries there are auto-detected at Step 1d, before
the error has a chance to fire. **When the same row in this table fires twice
across runs, lift it into `compat-workarounds.md` with a `detect` rule.** Tell
the user when you do.

| Symptom | Fix |
|---|---|
| `DataLoader worker ... Bus error` | Add `--shm-size=16g` to `docker run`. |
| Container starts then hangs | NGC ENTRYPOINT. Use `--rm` for one-shots; `ENTRYPOINT ["/bin/bash","-c"]` in Dockerfile. |
| `ImportError: cannot import name 'main' from 'evaluate'` | Script named `evaluate.py`. Rename to `run_eval.py` — HF `evaluate` lib shadows it. |
| `pip cache purge` fails in build | NGC disables pip cache. Remove the line. |
| `TypeError: ... enable_gqa` at step 0 | PyTorch 2.5.0 SDPA+GQA bug (NGC 24.09). Set `attn_implementation: "eager"`. |
| `TypeError: Missing **kwargs in ... @check_model_inputs` (Idefics3 / Llava / Mllama) | `transformers>=4.51` regression. Pin `transformers==4.49.0 tokenizers==0.21.0`. |
| `trl>=1.0` breaking API on import | Pin `trl>=0.18.0,<1.0.0`. |
| `ValueError: ... CVE-2025-32434` torch.load | NGC 25.01 PyTorch 2.6.0a + `transformers>=4.51` refuses `.bin` checkpoints. If model ships only `pytorch_model.bin`, pin `transformers==4.49.0 tokenizers==0.21.0`. Safetensors models unaffected. |
| `ImportError: numpy.core.multiarray failed to import` | numpy 2.x ABI break. Pin `numpy<2`. |
| Albumentations `y_max <= y_min for bbox` | Degenerate bboxes. Add `filter_invalid_bboxes=True`, `min_area=1` to `A.BboxParams`. |
| Detection: `'list' object has no attribute 'logits'` in `compute_metrics` | Trainer with `eval_do_concat_batches=False`. Drop in-trainer metric, use `metric_for_best_model=eval_loss`, run mAP via `run_eval.py` post-training. |
| PEFT + `gradient_checkpointing`: `element 0 ... does not require grad` | After `get_peft_model(...)`, call `model.enable_input_require_grads()`. |
| Idefics3/SmolVLM: vision tower SDPA error | Set `_attn_implementation="eager"` on every model load. Store in `config.yaml: attn_implementation:`. |
| Model barely learns, loss ≈ random | Don't set `torch_dtype=torch.bfloat16`. Load fp32, set `bf16=True` in `TrainingArguments`. |
| Labels saved as `LABEL_0/1` not class names | Pass `id2label=` from `ClassLabel.names` to `from_pretrained`. |
| Arrow drops `PIL.Image` after `load_from_disk` | `ds.cast_column("image", datasets.Image())`. |
| LoRA reports 5-10% trainable (expected 0.1-1%) | Target regex too broad. VLMs: `target_modules=".*language_model.*"`. |
| UCX segfault on container exit | Harmless NCCL cleanup. Check `checkpoints/final/` exists. |
| Step 0 hangs for minutes | Streaming dataset. Run `prepare_data.py` first. |
| CV: ~57% accuracy where SOTA is 94%+ | Missing augmentation. Add `RandomResizedCrop` + `RandomHorizontalFlip`. |
| OOM at step 0 | Halve `per_device_train_batch_size`, double `gradient_accumulation_steps`, enable `gradient_checkpointing`. |

---

## Communication style

- Terse. No filler, no restating the request. One-word answers when appropriate.
- Always include direct Hub and wandb URLs when referencing artifacts.
- On error: state what went wrong, why, what you changed. No menus.
- Never present "Option A/B/C" for a request that has a clear answer. Act.
