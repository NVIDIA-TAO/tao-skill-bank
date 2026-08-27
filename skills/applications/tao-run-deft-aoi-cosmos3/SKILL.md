---
name: tao-run-deft-aoi-cosmos3
description: >
  Run the disk-backed DEFT AOI improvement loop for NVIDIA Cosmos Reason 3
  models with exact OK/NG labels: evaluate a frozen Benchmark, diagnose Proxy
  errors, mine labeled real images, train with Cosmos Framework, and repeat.
license: Apache-2.0 AND CC-BY-4.0
compatibility: Requires the companion skills named in eval.config, host Python with pyarrow and yaml, and the selected platform native CLI.
metadata:
  author: NVIDIA Corporation
  version: "0.2.0"
allowed-tools: Read Task Bash Write
tags:
- application
- workflow
- deft
- aoi
- cosmos
---

# Skill: tao-run-deft-aoi-cosmos3

This is the canonical Cosmos3 DEFT AOI application. It is a disk-backed,
gate-first loop. Training data comes only from labeled real images selected
from the Mining split. The base model is a complete local HF-format VLM
snapshot under the workspace and is consumed directly. There is no model
preparation stage and no explicit checkpoint export stage.

## Required reading and model-first routing

Before planning an action:

1. Read this file and the reference for the next durable stage.
2. Resolve the supplied model ID before choosing a workflow:

   ```bash
   python "$TAO_SKILL_BANK_PATH/scripts/resolve_tao_model.py" \
     --model nvidia/Cosmos3-Nano --action evaluate \
     --backend cosmos-framework --workload deft-aoi
   ```

3. Resolve the shared Cosmos frontend with the explicit supported backend:

   ```bash
   python "$TAO_SKILL_BANK_PATH/skills/models/tao-finetune-cosmos-reason/scripts/cosmos_workflow.py" \
     --backend cosmos-framework --action evaluate --workload deft-aoi
   ```

   Repeat with `--action train` or `--action inference` for those actions.
   Use the returned `cosmos-framework` implementation because it owns native
   VLM training, evaluation, inference, DCP handling, and the image pin used by
   this application.

4. Read `skills/models/tao-finetune-cosmos-reason/SKILL.md`, its
   `references/skill_info.yaml`, and the selected Framework contract before
   resolving the image or building a nested TOML spec.

Supported canonical IDs are `nvidia/Cosmos3-Nano`,
`nvidia/Cosmos3-Edge`, and `nvidia/Cosmos3-Super`. Aliases `nano`, `edge`, and
`super` normalize to those IDs. Nano is the application default; never switch
an explicitly selected variant.

## Fixed validation profile

For the fresh validation run on this machine, use these explicit values:

- platform: Docker;
- compute: one NVIDIA H200 GPU, one node;
- iterations: 5 maximum;
- training: 10 epochs per iteration;
- mining: `topk=15`, cosine distance, `filter_by_label`, with `OK` and `NG`
  queried separately;
- evaluation: frozen Benchmark at baseline and after every trained iteration;
- target: `accuracy >= 0.99` and `unknown_predictions <= 0`;
- assistant label: exactly `OK` or `NG`.

For other requests, preserve explicit user values. If the application is
available on several installed supported platforms and none was selected, ask
once among those peers; never invent a default. The profile above already
selects Docker and therefore does not require a platform question.

## Preflight and approval

Run `references/preflight.md` in order. In particular:

- validate Docker, the H200, disk space, host UID/GID, and the skill Python;
- resolve `images.tao_toolkit.cosmos_framework` and
  `images.tao_toolkit.data_services` from `versions.yaml`;
- locate a complete local HF VLM snapshot beneath `${WORKSPACE}/models` with
  `config.json` and safetensor weights;
- validate Proxy, frozen Benchmark, and Mining JSON arrays with
  `scripts/validate_sharegpt.py` and `scripts/validate_split_contract.py`;
- require exactly one image per record and a unique safe `id` on evaluation
  records;
- construct nested Train and per-role Evaluate TOML files;
- show effective values, their sources, planned commands, immutable image
  digest, mounts, results directory, and checkpoint handoff.

Do not create specs/state/results, pull or log into a registry, download an
asset, or submit a container before the user confirms the launch review. A
missing small Python helper may be installed under the bank-wide exception;
report it and rerun preflight.

## Direct model and checkpoint contract

`--base-model-path` is an absolute workspace path to a complete local
HF-format VLM snapshot. Use it unchanged for baseline evaluation, iteration-1
training, and standalone inference. Do not copy, translate, or reserialize its
weights.

Framework Train writes a native DCP checkpoint. Record the selected DCP
directory and the rendered SFT TOML in the `train` stage. For iteration N:

- evaluate the DCP directly by setting Evaluate `model.model_name` to the DCP
  path, `model.config_file` to that iteration's SFT TOML,
  `model.export_dir` to a writable action-model directory, and
  `model.vit_checkpoint_path` to the original HF base directory;
- infer from that DCP with the equivalent `--config_file`, `--export_dir`, and
  `--vit_checkpoint_path` arguments;
- warm-start iteration N+1 by rendering its SFT TOML with
  `checkpoint.load_path` equal to iteration N's DCP path. Keep
  `checkpoint.keys_to_skip_loading=[]` so LoRA keys are restored.

The public evaluation and inference commands perform their required local DCP
materialization internally. Do not add an app-level conversion or export
stage.

## Native Framework actions

All three GPU model actions use the same immutable Framework image and its
public console scripts:

- Train: `cosmos-framework-train --sft-toml=/tao/config/train.toml`
- Evaluate: `cosmos-framework-evaluate --config /tao/config/evaluate.toml`
- Inference: `cosmos-framework-inference --model_path ... --type image
  --media ... --prompt ... --results_dir ...`

Use the bundled render/submit helpers:

- `scripts/render_cfw_sft.py` and `scripts/submit_cfw_train.py`;
- `scripts/render_cfw_evaluate.py` and `scripts/submit_cfw_evaluate.py`;
- `scripts/submit_cfw_inference.py`.

The evaluate renderer fixes a one-GPU BF16 profile with
`vision.video_decoder="torchcodec-cuda-on-demand"`, one frame, batch size 1,
and four output tokens. Do not patch installed image source. Training uses
native VLM LoRA on language projections and one image per record.

Every actual spec is a nested dictionary serialized to TOML. Never send flat
dotted keys across a container boundary. Do not mount user data over
`/workspace`; mount data and results at their absolute compute-frame paths.
Run writable containers as the invoking UID:GID with `USER`, `LOGNAME`,
`HOME=/tmp`, and read-only `/etc/passwd` and `/etc/group` mounts.

## Bare-label and split contract

- Each ShareGPT record has exactly one image.
- The first human/user turn is the inspection prompt.
- The final assistant/gpt value is exactly `OK` or `NG`.
- `NG` is positive; `NG -> OK` is a false accept and `OK -> NG` a false
  reject.
- Output parsing may select the last standalone OK/NG token, but training
  labels are never normalized.
- Proxy, Benchmark, and Mining targets are disjoint.
- Benchmark is frozen at initialization and its SHA-256 must never change.
- Only Proxy results drive RCCA, routing, and mining. Benchmark results only
  drive the stop gate.
- `train_iter_1.json` contains current real Mining selections. Later Train
  JSON files preserve the preceding Train JSON and add only newly selected
  real Mining records, deduplicated by target path.

## Durable workflow

After approval, initialize state exactly once with
`scripts/init_deft_state.py`. Record version 6, the absolute local base path,
single Framework image/digest, platform, compute, split paths and hash, metric
contract, five-iteration budget, 10 epochs, and mining top-K 15. Never edit
`deft_state.json` manually.

Before every stage and after context compaction, run:

```bash
scripts/deft_context.py --state "$RESULTS_DIR/deft_state.json" --stage STAGE
```

Baseline:

1. `evaluate_benchmark`
2. `benchmark_metrics`; finalize immediately if the target passes.
3. `evaluate_proxy`
4. `proxy_rcca`

Each iteration while the gate is unmet:

1. `routing`: derive target rows only from Proxy false accepts/rejects.
2. `data_mining`: call `tao-mine-aoi-images`; run `filter_by_label` separately
   for OK and NG with top-K 15, apply the cosine floor, then apply the durable
   filepath history so earlier selections cannot re-enter.
3. `assemble_data`: map selections back to Mining prompts and exact labels
   with `scripts/emit_mined_sharegpt.py`, then call
   `scripts/assemble_training_json.py --mined-json ...`; add
   `--previous-json` from iteration 2 onward.
4. `validate_data`: validate files, exact labels, lineage, deduplication,
   monotonic retention, split isolation, and frozen Benchmark hash.
5. `train`: iteration 1 uses the local HF base; later iterations also load the
   preceding DCP.
6. `evaluate_benchmark` directly from the new DCP.
7. `benchmark_metrics`; finalize on pass or after iteration 5.
8. `evaluate_proxy` and `proxy_rcca` only when another iteration is needed.

Write the Proxy RCCA Markdown artifact from
`references/RCCA_REPORT_TEMPLATE.md` before committing `proxy_rcca`. Commit
every success or error through `scripts/commit_stage.py` with measured positive
duration. Finish only through `scripts/finalize_run.py`, then verify fresh
state says `status == "complete"` and render the terminal report.

## Platform and job-record contract

Every GPU action is submitted with the selected platform's native four verbs:
`submit`, `status`, `logs`, and `cancel`. Before native launch,
`tao-launch-workflow` opens a job-record and binds its `results_dir`; its id is
the only handle. Poll Docker for live state and map it to
`PENDING RUNNING COMPLETE ERROR CANCELED UNKNOWN`. Records are audit evidence,
not a live scheduler. Never launch an unrecorded container.

## References by stage

| Concern | Reference |
|---|---|
| Ordered preflight | `references/preflight.md` |
| State transitions and artifact fields | `references/pipeline-and-state.md` |
| Framework action specs and DCP handoff | `references/cosmos-reason.md` |
| Annotation and mining-only assembly | `references/aoi-annotation.md` |
| Mining invocation | `references/tao-mine-aoi-images.md` |
| Data layout and split isolation | `references/data-layout.md` |
| State/report helpers | `references/scripts-and-agents.md` |
| Report evidence | `references/REPORT_RENDERING.md` |

## Hard stops

Commit an error and stop for invalid state; missing local HF model files;
unsupported model/backend/action; non-array JSON; non-exact labels; anything
other than one image; split leakage; changed Benchmark hash; missing or
ambiguous Mining alignment; reused mined filepath; missing DCP metadata;
missing rendered SFT TOML; unpinned image; path not visible in the compute
frame; insufficient H200 memory; a native backend error; or final evidence
that does not satisfy the metric contract. Do not silently fall back to a
different model, backend, platform, checkpoint, dataset, or label format.
