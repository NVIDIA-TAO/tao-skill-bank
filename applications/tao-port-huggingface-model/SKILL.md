---
name: tao-port-huggingface-model
description: >
  Integrate a HuggingFace Computer Vision model into the NVIDIA TAO Toolkit
  ecosystem (tao-core config, tao-pytorch trainer, tao-deploy TensorRT
  pipeline). Use when the user asks to "integrate a HuggingFace model into
  TAO", "add an HF model to TAO Toolkit", "wire a HuggingFace ViT/DETR/
  SegFormer into tao-pytorch", "build a TAO trainer + deploy pipeline for an
  HF CV model", or pastes a HuggingFace model URL/ID and wants it turned
  into a TAO model. Covers the full 7-phase loop: prerequisites check,
  HuggingFace inspection and validation, codebase exploration, tao-core
  configuration and native trainer implementation, ONNX export plus TensorRT
  deploy integration, packaging and L0 testing, container-based end-to-end
  validation, and (conditional) accuracy/latency tuning. Supports
  classification, object detection, semantic / instance / panoptic
  segmentation, zero-shot detection, and depth estimation.
license: Apache-2.0
compatibility: Requires Python 3.10+, NVIDIA driver, CUDA 13.0+, docker + nvidia-container-toolkit, an NGC API key (`docker login nvcr.io`), an HF_TOKEN, and access to the TAO Toolkit container images on `nvcr.io` for `tao-pytorch`, `tao-deploy`, and (optionally) `tao-dataservices` — Phase 0 asks the user for the exact image references and prepares them locally as `tao-pytorch-base:latest`, `tao-deploy-base:latest`, `tao-dataservices-base:latest`. Local clones of `tao-core`, `tao-pytorch`, `tao-deploy`, and `tao-dataservices` are required (the skill drives modifications across all four). All work is local-only — the skill never pushes to remote git, container registries, or HF Hub.
metadata:
  author: NVIDIA Corporation
  version: '0.1'
allowed-tools: Read Bash Write Edit Grep Glob WebFetch
tags:
- tao
- huggingface
- integration
- computer-vision
- deploy
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


# TAO-HF Integration Skill

Integrate a HuggingFace (HF) Computer Vision model into the NVIDIA TAO Toolkit ecosystem. Work the phases iteratively — not purely linearly. Real implementation follows a **build → test → debug → fix → retest** loop at every step. When something fails, diagnose and fix before moving on. When everything passes, move to the next step.

This SKILL.md is the workflow coordinator. Each phase has a dedicated reference file under `references/phase-N-*.md` containing the full step-by-step content, code blocks, docker invocations, and gates. Read the matching reference at the start of each phase — the summaries below are not sufficient on their own.

---

## Local-Only Rule

All work is strictly local. Do NOT:

- `git commit`, `git push`, or create branches on any remote (GitLab, GitHub, HuggingFace).
- Create merge requests, pull requests, or issues.
- Upload, publish, or push Docker images to any registry.
- Push to any remote container registry or artifact store.

You may only read/clone from remotes. All file edits, Docker builds, and test runs stay on the local machine.

---

## Submodule Override Strategy

The user clones the four TAO repos (`tao-core`, `tao-pytorch`, `tao-deploy`, `tao-dataservices`) independently into one working directory:

```
working-directory/
├── tao-core/             ← independently cloned — modifications go HERE
├── tao-pytorch/
│   └── tao-core/        ← submodule at original commit (stale — DO NOT use)
├── tao-deploy/
│   └── tao-core/        ← submodule at original commit (stale — DO NOT use)
└── tao-dataservices/
    ├── tao-core/        ← submodule at original commit
    └── tao-pytorch/     ← submodule at original commit
```

The nested `tao-core/` submodules inside each repo point to the **original unmodified commit**. Modifications only exist in the top-level `tao-core/`. **Always install from the top-level `tao-core/`, never from `<repo>/tao-core/`.**

In CI, Jenkinsfiles run `pip install tao-core/` (the submodule). The local override:

1. **Mount the entire working directory** into the container: `-v $(pwd):/workspace`.
2. **pip install order:** Always `pip install /workspace/tao-core` FIRST, before installing tao-pytorch or tao-deploy. This ensures modified config schemas are used instead of the stale submodule.
3. **PYTHONPATH:** Top-level tao-core first, e.g. `-e PYTHONPATH=/workspace/tao-core:/workspace/tao-pytorch`.

Using the nested submodule silently ignores all modifications — model configs, backbone mappings, etc. would not be present.

---

## Execution platform

This skill executes every test, smoke run, and end-to-end validation inside a
locally prepared TAO Toolkit container (`tao-pytorch-base:latest`,
`tao-deploy-base:latest`, optionally `tao-dataservices-base:latest` — all
prepared in Phase 0). The platform skills own the *how* of running those
containers; this skill only specifies *what* to run inside them.

| Concern | Authoritative skill |
|---|---|
| GPU host runtime — NVIDIA driver 580, CUDA Toolkit 13.0, NVIDIA Container Toolkit 1.19.0 | [`tao-skill-bank:tao-setup-nvidia-gpu-host`](../../platform/tao-setup-nvidia-gpu-host/SKILL.md) |
| `docker run` flags, NGC auth, `--gpus`, mounts, env passthrough, `--ipc=host`/`--shm-size`, container inspection, common error modes | [`tao-skill-bank:tao-run-on-docker`](../../platform/tao-run-on-docker/SKILL.md) |
| Local Docker daemon preflight + per-job invocation | [`tao-skill-bank:tao-run-on-local-docker`](../../platform/tao-run-on-local-docker/SKILL.md) |

**Default platform:** `local-docker`. This workflow requires bind-mounting
your local clones of `tao-core`, `tao-pytorch`, `tao-deploy`, and
`tao-dataservices` into the container at `/workspace`, then installing the
modified source via `pip install /workspace/tao-core` and `setup.py develop`.
That layout only makes sense against a Docker daemon you control. The Local
Only Rule above is the corollary: no remote registry pushes, no remote job
submissions.

**GPU runtime preflight:** Phase 0 delegates the driver / CUDA / NCT checks
to the `tao-setup-nvidia-gpu-host` skill rather than duplicating them here. NGC
`docker login`, image pulls, and the published-image preparation step remain
in Phase 0 — those are the only TAO-Toolkit-specific bits.

**Docker run conventions:** every `docker run` invocation in Phases 3 / 4 /
6 follows the canonical flag set from `platform/tao-run-on-docker/SKILL.md` (`--gpus
all`, `-v` bind mounts, `-e VAR` passthrough, `--shm-size=16G` for
DataLoader-heavy pytest, `--rm` for one-shots). The phase reference files
only specify the *workflow-specific* additions (`-w /workspace/<repo>`,
`PYTHONPATH=/workspace/tao-core:/workspace/<repo>`, the inner
`pip install /workspace/tao-core && python setup.py develop && pytest ...`
shell). If anything about the generic conventions changes, change it in the
docker platform skill — do not fork them inside this skill.

---

## Phase Map

| Phase | Goal | Reference |
|---|---|---|
| 0 | Verify prerequisites + ask user for TAO Toolkit images + prepare local image tags | [phase-0-prereqs.md](references/phase-0-prereqs.md) |
| 1 | Gather inputs, launch the containerized HF-inspection environment, validate HF model + dataset | [phase-1-inspection.md](references/phase-1-inspection.md), [hf-inspection.md](references/hf-inspection.md) |
| 2 | Find the closest existing TAO reference model | [phase-2-codebase.md](references/phase-2-codebase.md), [task-type-guide.md](references/task-type-guide.md) |
| 3 | tao-core config + tao-pytorch trainer / native eval / inference | [phase-3-implementation.md](references/phase-3-implementation.md), [tao-patterns.md](references/tao-patterns.md), [repo-structure.md](references/repo-structure.md) |
| 4 | ONNX export + tao-deploy TRT engine, inference, evaluation | [phase-4-deploy.md](references/phase-4-deploy.md) |
| 5 | Packaging (`setup.py` console_scripts) + L0 tests | [phase-5-packaging.md](references/phase-5-packaging.md) |
| 6 | Container-based testing + end-to-end pipeline validation | [phase-6-container-tests.md](references/phase-6-container-tests.md), [docker-patterns.md](references/docker-patterns.md) |
| 7 | (conditional) Accuracy / latency / size tuning | [phase-7-optimization.md](references/phase-7-optimization.md) |

Cross-cutting:

- [workflow-consistency.md](references/workflow-consistency.md) — end-to-end CLI flow, config field paths, and cross-phase data dependencies.

**IMPORTANT — Continuous Execution Through Phase 6:** Do NOT stop after finishing implementation (Phases 3–5) and wait for the user to run tests. After completing Phase 5 (Packaging & L0 Testing), immediately proceed to Phase 6. The implementation is not considered complete until tests pass inside the TAO Toolkit containers and the end-to-end pipeline is validated. Phase 6 is mandatory, not optional.

---

## Development Loop

At every implementation step:

```
1. Write code
2. Test immediately (import check, unit test, or dry-run)
3. If it fails → read traceback → diagnose root cause → fix → go to 2
4. If it passes → move to next step
```

Do NOT accumulate untested code across multiple steps. Test early, test often. Writing all files first and only testing at the end makes debugging much harder because multiple bugs compound.

---

## Debugging Playbook

When something fails, consult this before trying random fixes:

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | Missing `__init__.py` or wrong PYTHONPATH | Add `__init__.py` to every package dir; check PYTHONPATH in docker command |
| `KeyError` in `BACKBONE_REGISTRY` | Backbone not registered or not imported | Add import to `backbone_v2/__init__.py`; verify `@BACKBONE_REGISTRY.register()` |
| Shape mismatch in forward pass | `head.in_channels` doesn't match backbone output dim | Check `model_params_mapping.py`; print backbone output shape |
| NaN loss after first epoch | LR too high, or wrong data normalization | Reduce LR by 10×; verify `augmentation.mean/std` matches model expectations |
| ONNX export fails | Unsupported op or dynamic control flow | Identify failing op; try `opset_version=17`; rewrite the op if needed |
| TRT engine build fails | ONNX graph has unsupported TRT ops | Run `trtexec --onnx=model.onnx` to identify failing layer; may need plugin |
| TRT accuracy << PyTorch | Preprocessing mismatch or precision loss | Compare `augmentation.mean/std` across specs; try FP32 engine first |
| OOM during training | Batch size too large or activation memory | Reduce `dataset.batch_size`; enable activation checkpointing; use FP16 |
| DDP hangs | Unused parameters in forward | `strategy='ddp_find_unused_parameters_true'` |
| Checkpoint load fails (missing keys) | State dict key mismatch | `strict=False` in `load_state_dict()`; check key mapping |
| `results_dir` files not created | Path doesn't exist or wrong permissions | `os.makedirs(results_dir, exist_ok=True)` |
| Config changes not taking effect | Stale submodule copy of tao-core | Verify `-v $(pwd):/workspace`; `pip install /workspace/tao-core` runs first |

---

## Environment Isolation Strategy

All Python work runs **inside Docker containers** — no host venvs, no
`pip install`s into host Python. The same `tao-pytorch-base:latest` image
that Phases 3/4/6 use is also used for Phase 1's HF inspection, so the host
needs only Docker (provided by `tao-setup-nvidia-gpu-host`) and never needs
`python3-pip` / `python3-venv` / a particular Python version.

- **Context A — HF model inspection (Phase 1):** launch a long-lived
  `tao-pytorch-base:latest` container named `tao-hf-inspect`, bind-mount a
  host scratch dir at `/workspace`, and run each probe step via `docker
  exec`. A `python:3.12-slim` fallback is documented for environments where
  Phase 0 hasn't been run yet. Full commands in `phase-1-inspection.md`.
- **Context B — Incremental smoke tests (Phase 3/4):** run inside the
  prepared TAO Toolkit container (`docker run ... tao-pytorch-base:latest`)
  with the local source bind-mounted and installed via `pip install
  /workspace/tao-core && python setup.py develop`.
- **Context C — Temporary files:** scratch lives under the host bind-mount
  (e.g. `./.phase1`) so files end up host-user-owned (`--user $(id -u):$(id -g)`).
  Remove the scratch dir after the phase that created it, or keep it
  between runs to skip model redownloads.

Rules:

1. `pip install` — NEVER into the host/system Python. Always inside a
   container.
2. Host-level system packages (`docker`, `git`, kernel headers, NVIDIA
   Container Toolkit) are owned by the `tao-setup-nvidia-gpu-host` skill, which
   handles the distro-specific package manager (`apt-get` on Debian/Ubuntu
   and derivatives, `dnf` / `yum` on Fedora/RHEL/Rocky/Alma, `zypper` on
   openSUSE/SLES, manual instructions for other distros). This skill never
   issues `apt`/`dnf`/`zypper` commands directly — it only invokes
   `tao-setup-nvidia-gpu-host --check-only` and surfaces the error.
3. **Container UID convention — depends on the workload:**
   - Phase 1 inspection (Context A) — runs `python -c "..."` against
     pre-installed wheels in `tao-pytorch-base:latest`. **Pass
     `--user $(id -u):$(id -g)`**; HF cache + the `tao_hf_test.onnx`
     scratch file end up host-user-owned. The fallback path on
     `python:3.12-slim` does pip-install-at-startup, so it also sets
     `HOME=/workspace` + `PIP_USER=1` to route the install into a
     bind-mounted user-site instead of the root-owned system
     `site-packages`.
   - Phase 3 / 4 / 6 (Context B) — every smoke test, L0 test, and the
     end-to-end pipeline run `pip install /workspace/tao-core && python
     setup.py develop` against the container's **system** site-packages
     (root-owned). These invocations therefore run **as root** (no
     `--user`) and accept the trade-off that `*.egg-info/`, `build/`,
     `.pytest_cache/`, `dist/`, and `__pycache__/` left in
     `/workspace/tao-*` end up `root:root`. `sudo rm -rf` them or leave
     them between iterations — none of them is a source artifact.
4. Remove the long-lived inspection container (`docker rm -f
   tao-hf-inspect`) at the end of Phase 1.

---

## Phase 0 — Prerequisites Check

**Goal:** verify Python 3.10+ and `git`; delegate the NVIDIA driver / CUDA /
Docker / NVIDIA Container Toolkit host check to the `tao-setup-nvidia-gpu-host` skill
(see the Execution platform section above); verify NGC `docker login` for
`nvcr.io`. Then **ask the user** for the TAO Toolkit container image
references (tao-pytorch, tao-deploy, optionally tao-dataservices), pull them,
and prepare them as local image tags `tao-pytorch-base:latest`,
`tao-deploy-base:latest`, and `tao-dataservices-base:latest` for use by
Phases 3–6.

The TAO Toolkit images come with the released TAO Python packages already installed; the preparation step removes those pre-installed packages so the user's local clones (mounted at `/workspace/...` in later phases) can be installed and picked up at run time. After preparation, `pip install /workspace/tao-core && python setup.py develop` cleanly registers the local source and its `console_scripts` inside the container.

**Hard stop** if any check fails — resolve before proceeding.

Full commands, the user-prompt wording, and the per-image preparation `Dockerfile` snippets: see [phase-0-prereqs.md](references/phase-0-prereqs.md).

**Gate:** all prerequisite checks pass; the user has supplied the required image references; `tao-pytorch-base:latest` and `tao-deploy-base:latest` exist locally; `tao-dataservices-base:latest` exists if dataservices work is anticipated.

---

## Phase 1 — Information Gathering & Validation

**Goal:** decide whether to proceed at all. Gather credentials, locate (or clone) the four TAO repos, create a consistent local working branch across all of them, launch the long-lived `tao-hf-inspect` container (Context A in the Environment Isolation Strategy above), validate that the HF model is a CV model with a supported `pipeline_tag`, extract config + state-dict schema, sanity-check ONNX export, and clean up.

Full step-by-step (1.1–1.7) including the AutoConfig probe, dataset loadability probe, ONNX sanity export, and container cleanup: see [phase-1-inspection.md](references/phase-1-inspection.md). Generic HF-inspection patterns: [hf-inspection.md](references/hf-inspection.md).

**Reject if:**

- `pipeline_tag` is NLP / audio / LLM (out of CV scope).
- `AutoConfig` raises.
- ONNX export fundamentally cannot work (and has no rewrite path).

**Gate:**

- [ ] All 4 TAO repos located or cloned; consistent working branch created across all of them.
- [ ] `pipeline_tag` confirmed CV.
- [ ] `model_type`, `image_size`, `hidden_size`, `num_labels` extracted.
- [ ] State-dict keys documented; HF→TAO remapping plan drafted.
- [ ] ONNX export sanity check passed (or failure mode understood).
- [ ] User confirmed `model_short_name` and task type.

Present findings to the user and get confirmation before proceeding.

---

## Phase 2 — Codebase Exploration

**Goal:** find the closest existing TAO reference model for the detected `pipeline_tag`, read its full implementation across `tao-core`, `tao-pytorch`, and `tao-deploy`, and decide whether the backbone already exists in `backbone_v2/` or needs implementation.

The HF `pipeline_tag` → TAO reference model mapping (classification → `classification_pyt`, detection → `dino`/`rtdetr`, segmentation → `segformer`, instance → `mask2former`, panoptic → `oneformer`, zero-shot → `grounding_dino`, depth → `mono_depth`) drives **everything downstream**: config structure, model architecture, loss, ONNX export shape, TRT builder, deploy inferencer/loader, evaluation metrics, and dataset format.

Full reference list (12 files per reference model), backbone coverage check (`backbone_v2/` already provides `vit`, `swin`, `resnet`, `convnext`, `dino_v2`, `fan`, `fastervit`, `gcvit`, `hiera`, `mit`, `edgenext`, `efficientvit`, `radio`, `siglip2`, `open_clip`, etc.), and `tao-dataservices` coverage check: see [phase-2-codebase.md](references/phase-2-codebase.md). Per-task architectural details: [task-type-guide.md](references/task-type-guide.md).

If a new backbone is needed, decide the implementation strategy (timm wrap > re-implement from scratch > HF black-box wrap) before Phase 3 — it changes weight loading, ONNX export, and the deploy pipeline. **Never dual-inherit from `transformers.PreTrainedModel` and `BackboneBase`** — metaclass conflict.

**Gate:**

- [ ] Reference TAO model identified; all 12 reference locations read.
- [ ] Task-type implications understood (architecture, loss, ONNX outputs, deploy classes, metrics, dataset).
- [ ] Backbone coverage decided (reuse existing / wrap timm / new implementation).
- [ ] Dataservices coverage checked (existing converters vs. new needed).

---

## Phase 3 — TAO Core Configuration & Native Implementation

**Goal:** write the tao-core config schema and the tao-pytorch trainer + native inference + native evaluation, with intermediate smoke tests. Use `<model_name>` as the `snake_case` short-name from Phase 1 and `<ModelName>` as the `PascalCase` form.

Steps (each builds on the previous, smoke-test in between):

1. **Step 1 — `tao-core` config:** `nvidia_tao_core/config/<model_name>/{__init__.py, default_config.py, model_params_mapping.py}`. `ExperimentConfig(CommonExperimentConfig)` MUST contain `model`, `dataset`, `train`, `evaluate`, `inference`, `export`, `gen_trt_engine`, `quantize`. After writing, smoke-test the import inside `tao-pytorch-base:latest`.
2. **Step 2 — `tao-pytorch` trainer:** create `cv/<model_name>/{__init__.py, model/, dataloader/, scripts/, entrypoint/, experiment_specs/, utils/}` with `build_model()`, `<ModelName>PlModel(TAOLightningModule)`, `train.py`, `entrypoint/<model_name>.py`, and `experiment_specs/experiment_spec.yaml`. (If a new backbone is needed, add `cv/backbone_v2/<backbone_name>.py` and register it in `backbone_v2/__init__.py`.) Smoke-test config import → model build → PLModel instantiation inside the container.
3. **Step 3 — Multi-GPU/multi-node:** handled by the entrypoint's `launch()` (sets `TAO_VISIBLE_DEVICES`, wraps with `torchrun`); the train script reads it via `initialize_train_experiment()`. Use `strategy='ddp_find_unused_parameters_true'`, `sync_batchnorm=True`, `use_distributed_sampler=False`.
4. **Step 4 — Native inference (`scripts/inference.py`):** mirrors training (Hydra runner, monitor_status, `initialize_inference_experiment`). Output: `result.csv`.
5. **Step 5 — Native evaluation (`scripts/evaluate.py`):** same pattern, `dm.setup(stage="test")`, `trainer.test()`. Output: `results.json`.
6. **Step 6 — MLOps for training:** `self.log(...)` in `training_step`/`on_train_epoch_end`, `TensorBoardLogger`, `TAOStatusLogger`, `LearningRateMonitor`.
7. **Step 7 — MLOps for eval/infer:** `@monitor_status` writes `status.json`; eval writes `results.json`, infer writes `result.csv`.

Full code (build_model body, BackboneBase abstract methods, the 6 abstract methods to override, HF state-dict converter pattern, PLModel skeleton, train.py with precision mapping, entrypoint, the canonical `experiment_spec.yaml` with all sections, the in-container smoke-test command): [phase-3-implementation.md](references/phase-3-implementation.md). Canonical code snippets: [tao-patterns.md](references/tao-patterns.md). File layout: [repo-structure.md](references/repo-structure.md). Per-task variations: [task-type-guide.md](references/task-type-guide.md).

**Critical consistency rules** (also enforced in the cross-phase checklist below):

- `augmentation.mean`/`std` in training spec MUST be identical to the deploy specs.
- `model.head.in_channels` MUST match `model_params_mapping.py` for the chosen backbone.
- `<model_name>_model_latest.pth` MUST match `self.checkpoint_filename` in the PLModel.
- `export.onnx_file` MUST match what `gen_trt_engine.onnx_file` references.
- All `???` fields are `MISSING` (required) — user supplies via YAML or CLI override.

**Step 1 gate:** `ExperimentConfig` imports cleanly inside the container.
**Step 2 gate:** `build_model(cfg)` runs and the PLModel instantiates inside the container.
**Phase 3 overall:** all 7 steps complete, smoke tests pass, no missing `__init__.py`.

---

## Phase 4 — Export, Deployment & TensorRT Integration

**Goal:** ship ONNX export from tao-pytorch, then a TRT engine builder + TRT inference + TRT evaluation in tao-deploy that reuse the tao-core `ExperimentConfig`.

Steps:

1. **Step 8 — ONNX exporter (`tao-pytorch/cv/<model_name>/scripts/export.py`):** load PLModel, extract raw `nn.Module` (`sf_model.model`), build dummy input from `export.input_*`, call `ONNXExporter().export_model(...)` with input/output names per task type (classification: `["input"]`/`["output"]`; detection: `["pred_logits", "pred_boxes"]`; segmentation: `["output"]`; instance seg: `["pred_logits", "pred_masks"]`). `batch_size=-1` ⇒ dynamic batch (`dynamic_axes={0: "batch"}`).
2. **Step 9 — TensorRT engine builder (`tao-deploy/cv/<model_name>/scripts/gen_trt_engine.py`):** uses tao-deploy's own `hydra_runner` and `monitor_status` (separate imports from tao-pytorch). Subclass `nvidia_tao_deploy.engine.builder.EngineBuilder`, or reuse `ClassificationEngineBuilder` for classification. Also write `specs/{gen_trt_engine,inference,evaluate}.yaml` — same `ExperimentConfig` schema as training; `augmentation.mean`/`std` MUST match training.
3. **Step 10 — TRT inference (`tao-deploy/cv/<model_name>/scripts/inference.py`):** load classes, build `ClassificationInferencer` (or task-appropriate), feed via `ClassificationLoader` (NumPy-only, no PyTorch), write `result.csv`.
4. **Step 11 — TRT evaluation (`tao-deploy/cv/<model_name>/scripts/evaluate.py`):** same pattern with `is_inference=False`; compute task-appropriate metrics via sklearn / pycocotools / etc.; write `results.json`.

Full code (`run_export`, `<ModelName>EngineBuilder` template, the three deploy spec YAMLs, ClassificationLoader wiring, sklearn metrics code), plus the Phase 3+4 verification gate (3 in-container checks: imports, model build + forward, ONNX export round-trip): [phase-4-deploy.md](references/phase-4-deploy.md).

**Module pitfalls:**

- tao-pytorch and tao-deploy have **separate** `hydra_runner` and `monitor_status` implementations. Use the deploy versions in deploy scripts.
- The `ExperimentConfig` is imported from `nvidia_tao_core` in both repos — same schema, same field paths.

**Phase 3+4 gate:** all three in-container checks (`tao-pytorch` imports + model + ONNX export, `tao-deploy` imports) pass.

---

## Phase 5 — Packaging & L0 Testing

**Goal:** register the model as a console_script in both repos and add unit tests.

1. **Step 12 — `tao-pytorch/setup.py`:** add `'<model_name>=nvidia_tao_pytorch.cv.<model_name>.entrypoint.<model_name>:main'` to `console_scripts`.
2. **Step 13 — `tao-deploy/setup.py`:** create `cv/<model_name>/entrypoint/<model_name>.py` (using `nvidia_tao_deploy.cv.common.entrypoint.entrypoint_hydra`); add `'<model_name>=nvidia_tao_deploy.cv.<model_name>.entrypoint.<model_name>:main'` to `console_scripts`.
3. **Step 14 — Deploy L0 tests:** `tao-deploy/tests/<model_name>/test_<model_name>.py` covering `gen_trt_engine`, inference, evaluate (subprocess + `--buildOnly` `trtexec`).
4. **Step 15 — Trainer L0 tests:** `tao-pytorch/tests/cv_unit_test/<model_name>/{conftest.py, test_model.py, test_trainer.py, test_dataloader.py, test_config.py, test_export.py}`. Pattern: `Trainer(..., fast_dev_run=True)` + `@pytest.mark.cv_unit @pytest.mark.<model_name>`.

Full code: [phase-5-packaging.md](references/phase-5-packaging.md).

**Gate:** entrypoints registered; pytest files exist and follow the marker convention. **Do NOT stop here — proceed directly to Phase 6.**

---

## Cross-Phase Data Flow & Consistency Verification

Before Docker testing, verify the entire chain:

```
train → export → gen_trt_engine → inference / evaluate

train produces:           <results_dir>/train/<model_name>_model_latest.pth
export.checkpoint reads:  ${results_dir}/train/<model_name>_model_latest.pth
export produces:          <results_dir>/export/<model_name>.onnx
gen_trt_engine reads:     ${export.results_dir}/<model_name>.onnx
gen_trt_engine produces:  <results_dir>/trt/<model_name>.engine
inference reads:          inference.trt_engine = <engine_path>
evaluate reads:           evaluate.trt_engine = <engine_path>
```

Consistency checklist (verify before proceeding):

- [ ] `self.checkpoint_filename` in PLModel produces the `*_latest.pth` name that `evaluate.checkpoint` and `export.checkpoint` reference.
- [ ] `augmentation.mean`/`std` are identical in: training spec, `inference.yaml`, `evaluate.yaml`, and `preprocess_mode` in the engine builder.
- [ ] `input_names=['input']` and `output_names=['output']` in ONNX export (TRT engine builder expects these names; detection/instance-seg use task-specific names).
- [ ] `export.input_width`/`input_height` match `dataset.img_size` from training.
- [ ] `model.head.in_channels` in config matches `model_params_mapping.py` for each backbone variant.
- [ ] `classes.txt` at `dataset.root_dir` is readable by both tao-pytorch dataset and tao-deploy dataloader.
- [ ] All `__init__.py` files exist in every package directory (required for wheel build and imports).
- [ ] `scripts/__init__.py` exists (required for `get_subtasks()` to discover scripts via `pkgutil`).

Full config field paths and cross-phase dependencies: [workflow-consistency.md](references/workflow-consistency.md).

---

## Phase 6 — Container Testing & End-to-End Validation

**Mandatory — start immediately after Phase 5.** All TAO models ship as Docker images. Code that only works outside a container is incomplete.

TAO testing runs **directly inside the TAO Toolkit container** — no Docker image build is involved in the test loop. Phase 0 already prepared the local image tags from the user-supplied references; this phase mounts the local source into those containers, installs it via `setup.py develop`, and invokes `pytest` / `pylint` / `pydocstyle` / `flake8` directly. End-to-end flow: mount → install local source → test.

> **Note:** Use vanilla `pytest` + lint commands rather than any `ci/run_functional_tests.py` / `ci/run_static_tests.py` wrappers. Those wrappers only exist in NVIDIA's internal mirrors of the TAO repos; the public github mirrors at `github.com/NVIDIA-TAO/` do not ship a `ci/` directory. Targeting `pytest` and the lint binaries directly keeps the workflow public-repo-friendly.

Steps:

- **Step 16** — Verify `tao-pytorch-base:latest` and `tao-deploy-base:latest` exist locally (prepared in Phase 0 from the user-supplied TAO Toolkit images).
- **Step 17** — `pytest --cov=nvidia_tao_core` inside `tao-pytorch-base` against `tao-core`.
- **Step 18** — `pytest tests/cv_unit_test/<model_name>/ -m cv_unit -v` for `tao-pytorch` (with `--shm-size=16G`); the full suite is `pytest tests/ -v -m "not slow"`.
- **Step 19** — Same pattern for `tao-deploy` (no `--shm-size` needed): `pytest tests/<model_name>/ -v`; full suite is `pytest tests/ -v`.
- **Step 20** — Static / lint tests on the new code: `python -m pylint --errors-only nvidia_tao_pytorch/cv/<model_name>/ nvidia_tao_deploy/cv/<model_name>/ nvidia_tao_core/config/<model_name>/`. Optionally also run `pydocstyle` and `flake8` over the same paths.
- **Step 21** — Build wheels: `python setup.py bdist_wheel` in tao-pytorch and `make build` (or `python setup.py bdist_wheel` if `make` is missing) in tao-deploy, both inside their respective TAO Toolkit containers.
- **Step 22** — End-to-end pipeline (a) train dry-run + export in **one** tao-pytorch container session (b) gen_trt_engine + inference + evaluate in **one** tao-deploy container session. (Same container session is critical — `--rm` discards installed packages between sessions.)
- **Step 23** — Cross-check native PyTorch vs TRT engine predictions on the same images. FP32 ≈ exact, FP16 ≈ small delta. Significant divergence = ONNX export or TRT build issue.
- **Step 24** — Interactive debugging shells: `docker run -it --rm --gpus all ... bash`.
- **Step 25** — (Optional) build release Docker images via `release/docker/Dockerfile{,.release}`. Distribution-only; not needed for validation.

Full commands (every `docker run` invocation, the exact env-var set for each container, the train/export/gen_trt_engine/inference/evaluate one-liner with all CLI overrides, fix-and-retest loop): [phase-6-container-tests.md](references/phase-6-container-tests.md). Build scripts, runner patterns, requirements files, and CI conventions: [docker-patterns.md](references/docker-patterns.md).

**Phase 6 gate (Done criteria):**

- [ ] tao-core / tao-pytorch / tao-deploy unit tests pass inside their respective TAO Toolkit containers.
- [ ] Static tests pass (or only legacy lint warnings).
- [ ] Wheels build successfully.
- [ ] End-to-end: `<model_name>_model_latest.pth` → `model.onnx` → `model.engine` → non-empty `result.csv` and `results.json`.
- [ ] Native vs TRT predictions agree within tolerance.

---

## Phase 7 — Optimization & Tuning (conditional)

Enter only if Phase 6 passes but accuracy / latency / model size needs improvement. **Ask the user for target metrics first.**

Diagnostic categories:

1. **Accuracy too low** — verify `augmentation.mean/std`, `load_state_dict()` missing/unexpected keys, LR/schedule vs HF reference, longer training, EMA, freeze backbone for small datasets, knowledge distillation.
2. **TRT vs native gap** — try FP32 first (isolates precision vs preprocessing), compare output tensors numerically, per-layer FP16 fallback for sensitive layers.
3. **Training too slow** — `torch.profiler`, more `workers` + `pin_memory`, gradient checkpointing.
4. **Inference too slow** — `trtexec --fp16 --verbose`, fixed batch size, larger workspace.

Optimization techniques covered:

- **Step 27** — Hyperparameter tuning (LR, optimizer, schedule, augmentation, EMA, backbone freezing).
- **Step 28** — INT8 quantization (PTQ via torchao / modelopt, TRT INT8 with calibration data).
- **Step 29** — Channel pruning (amount-based, L1 importance) + retrain.
- **Step 30** — Knowledge distillation (FD, logits, summary, spatial).
- **Step 31** — Resolution tuning (TAO interpolates positional embeddings automatically for ViT).

Full config blocks, YAML overrides, decision tree, and rationale per technique: [phase-7-optimization.md](references/phase-7-optimization.md).

---

## Argument

`$ARGUMENTS`

If provided, interpret `$ARGUMENTS` as the HuggingFace model ID or URL to use as the starting point for Phase 1. If credentials or model short-name are not included, ask the user for them before proceeding.
