# Cosmos3 on native Windows

This is the model-specific Windows route for Cosmos3. It is intentionally part
of `tao-finetune-cosmos-reason`, not an application skill: the dependencies,
patches, model conversion, LoRA overrides, action behavior, and support
boundaries all belong to Cosmos3. Use the Windows platform skills only for
reusable host execution mechanics.

## Select a route

| Route | Model-specific reference | Platform mechanics |
|---|---|---|
| Windows container, x64 | This file and `cosmos3-windows-contract.toml` | `skills/platform/tao-run-on-windows-docker` |
| Native Windows, x64 | `cosmos3-windows-x64-host-setup.md` | `skills/platform/tao-run-on-windows-venv` |
| Native Windows on ARM64 | `cosmos3-windows-arm64.md` | `skills/platform/tao-run-on-windows-venv` and `references/windows-arm64.md` |

All routes are one GPU and one node. Do not use WSL for this route. A model
request must select one route and preserve its interpreter, Torch ABI, and
source revisions for the whole job; x64 and ARM64 dependency sets are not
interchangeable.

## Source of truth and required reading

Read these in order before launching a Windows Cosmos3 action:

1. `cosmos3-windows-contract.toml` is the machine-readable source of truth for
   the container image, action entrypoints, required environment, the eleven
   training overrides, evaluate/export settings, and model limits. Read it with
   `tomllib`; do not duplicate its values in a launch script.
2. `cosmos3-windows-overrides.md` explains why the overrides must remain
   verbatim. In particular, native `cosmos` attention is the default; the two
   SDPA keys are retained only as a paired, W14-gated debug option, and
   `trainer.max_iter` is inert when the recipe has `cycle_lengths`.
3. `cosmos3-windows-gaps.md` records Windows-only failure modes. The
   cross-platform evaluate, inference, checkpoint, and LoRA-export traps remain
   in `cosmos-reason-gaps.md`.
4. `cosmos3-windows-source-patches.md` is the three-repository patch and PR #164
   rebase contract. Do not rebase from memory.
5. Select the matching host reference below. For OpenAI-compatible serving,
   also read `cosmos3-vllm-serving-windows.md`. `cosmos3-wts-validation.md` is
   an optional validation fixture, not a data requirement for Cosmos3 users.

## x64 reference route

`cosmos3-windows-x64-host-setup.md` is the recorded CPython 3.12 / CUDA 13.0
reference environment. It owns its pinned requirements and the
`preflight_cosmos3_windows_host.py` model-environment check. It must not be
applied to an ARM64 interpreter.

The recorded x64 native results establish LoRA training, evaluation, export,
direct inference, and vLLM Chat Completions serving with text and with one to
four images per request at 16k context. Do not silently inherit ARM64
video, Responses API, dynamic-LoRA, or compiled-engine validation for x64. The
Cosmos inference-microservice action itself remains unverified.

## Experimental native ARM64 route

`cosmos3-windows-arm64.md` uses CPython 3.13 ARM64, NVIDIA's ARM nightly Torch
wheels, an ARM64 TorchCodec source build, the Framework PR #164 plus the
vendored Windows port patch, and a single RTX Spark-class GPU. The model
reference owns the package and decoder contract; the platform ARM reference
owns the architecture and real-allocation probe.

The completed ARM64 lifecycle includes a 500-step, two-epoch WTS LoRA run,
merged HF export, direct CLI inference, one-item Framework evaluation, and
native vLLM serving. vLLM coverage includes text, image, video, Chat
Completions, standard Responses, eager and compiled execution, and a
standard PEFT adapter loaded dynamically. These results prove the recorded
native paths and artifacts; they do not establish Linux throughput parity,
multi-request scale, general model quality, or the Cosmos
inference-microservice action.

## Required action rules

- The five action entrypoints span two packages: train is in Cosmos Framework;
  export, evaluate, inference, and the microservice are in Cosmos-RL.
- `TAO_API_JOB_ID` and `TAO_API_RESULTS_DIR` must be set together for training.
- Export a LoRA run with `--base-model-path-or-uri`, then verify that the HF
  safetensors index has zero residual `lora` keys.
- Evaluate with `answer_type = "freeform"`; bound video work with `fps` and
  `total_pixels`, not `num_frames`; pass one media file to direct inference.
- Report `overall.accuracy`, never the known-always-zero `WEIGHTED_SCORE` line.
- Confirm the LoRA injection line before treating an 8.8B run as LoRA. A missing
  line means the run may be performing an unfittable dense update.

## Scope

Cosmos3 remains dataset agnostic. Validate every user dataset through the
normal dataset contracts in the parent skill; do not infer layout, task, media
format, sequence budget, or quality from its name. WTS is one optional fixture
for a reproducible native Windows ARM64 smoke, not a preferred or required
training dataset.

`cosmos3-windows-integration.md` records the current DeepStream delegation.
Describe that work as available for adoption, not integrated, until the
DeepStream follow-up lands.
