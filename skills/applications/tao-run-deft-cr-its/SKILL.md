---
name: tao-run-deft-cr-its
description: >-
  Run iterative DEFT improvement for ITS Cosmos-Reason binary video questions using nearest-neighbor
  mining, PAIDF Cosmos Predict generation, or both. Use for traffic-camera collision classification
  workflows that evaluate, analyze BCQ gaps, construct training data, retrain Cosmos-RL, and repeat.
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit. Workflows declare additional requirements.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash Write Task
tags:
- application
- workflow
- deft
- its
- cosmos-reason
- cosmos-rl
- mining
- genai
- paidf
---

# TAO Run DEFT CR ITS

## Purpose

Improve the non-reasoning binary-classification path of Cosmos Reason by repeatedly evaluating a KPI dataset, identifying false-positive and false-negative questions, constructing new training annotations through mining, PAIDF generation, or both, training Cosmos-RL, and evaluating the new checkpoint.

## Bundled Resources

Resolve `DEFT_SKILL_ROOT` to the installed directory containing this `SKILL.md`; it is not a user input. Use `run_script("scripts/<name>.py", ...)` when the runtime provides it, otherwise use `python3 "$DEFT_SKILL_ROOT/scripts/<name>.py"`. Never require a source-repository checkout or change the user's working directory to a repository root.

Invoke `tao-finetune-cosmos-reason`, `tao-finetune-cosmos-embed`, `tao-analyze-gaps-vlm-bcq`, `tao-mine-nearest-neighbors`, `paidf-cosmos-predict`, and the selected platform skill by registered skill name. Those skills own their commands, images, credentials, and bundled defaults. This workflow owns only its helpers, workflow configuration, branch selection, and artifact contracts.

## Workspace

The user provides an absolute DEFT workspace path. Do not use `/workspace`, which conflicts with Cosmos Reason runtime conventions.

```text
<deft_workspace>/
├── data/       # KPI and optional training/source datasets
├── hf_cache/   # persistent Hugging Face cache
├── model/      # baseline Cosmos Reason and optional local Cosmos Embed models
├── specs/      # workflow.yaml and user-editable templates
└── results/    # all run outputs
```

Each run writes to `<deft_workspace>/results/<run.name>`. When `run.name` is null, initialization creates `run_<YYYYMMDD_HHMMSS>` and records that resolved path in `deft_state.json`. The run contains the workflow snapshot, append-only loop log, state snapshot, accuracy report, baseline evaluation, optional fixed mining embeddings, and one `iter_<N>/` directory per improvement iteration.

## Workflow Configuration

Create `<deft_workspace>/specs/workflow.yaml` with absolute filesystem paths:

```yaml
run:
  name: null
  max_iterations: 1

data_generation:
  mode: both  # mining, genai, or both

kpi_dataset:
  annotations_path: /abs/deft_workspace/data/kpi/annotations.json
  media_dir: /abs/deft_workspace/data/kpi

train_dataset:
  annotations_path: /abs/deft_workspace/data/train/annotations.json
  media_dir: /abs/deft_workspace/data/train

cosmos_reason:
  baseline_model_path: /abs/deft_workspace/model/baseline
  base_evaluate_toml: /abs/deft_workspace/specs/cr_base_evaluate.toml
  base_train_toml: /abs/deft_workspace/specs/cr_base_train.toml
  continual_model: false

mining:
  embeddings_spec_template: /abs/deft_workspace/specs/cosmos_embed_inference.yaml
  embeddings_modality: both
  cosmos_embed_checkpoint_path: null
  embedding_parquets:
    kpi: null
    train: null
  mining_spec_template: /abs/deft_workspace/specs/nearest_neighbors.yaml
  mine_unique_only: true

genai:
  vlm_captioning_endpoint: https://captioning.example.com/v1
  paidf_num_gpus: 4
  generation_settings: null
  caption_prompt_file: null
```

`kpi_dataset`, `cosmos_reason`, and `data_generation` are always required. `train_dataset` and `mining` are required when the mode is `mining` or `both`. In those modes, `train_dataset` is the mining source pool and its raw annotations are never inserted directly into training. In `genai` mode, `train_dataset` is optional; when present, it seeds iteration 1 training.

`genai` is required when the mode is `genai` or `both`. The user must provide `genai.vlm_captioning_endpoint`; this workflow never starts a captioning server. `paidf_num_gpus` is also required. A null `generation_settings` uses the `paidf-cosmos-predict` default. A null `caption_prompt_file` uses `assets/qwen_its_caption_prompt.txt`.

All configured filesystem paths must be inside the DEFT workspace. The only exception is the workflow-owned default caption prompt resolved from the installed skill.

## Preflight And Initialization

Validate before launching any workload:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/verify_workflow_yaml.py" \
  --workspace "$WORKSPACE" \
  --workflow-yaml "$WORKSPACE/specs/workflow.yaml"
```

When GenAI is enabled, immediately use the `paidf-cosmos-predict` endpoint verifier against the configured base URL. Stop if its OpenAI-compatible models endpoint is unavailable; do not run the baseline evaluation and do not attempt to launch vLLM.

When the user does not supply custom templates, copy `assets/cr_base_evaluate.toml` and `assets/cr_base_train.toml` into `<deft_workspace>/specs/`. When mining is enabled, also copy `assets/default_cosmos_embed_inference.yaml` and the nearest-neighbor skill's default template. Point `workflow.yaml` at the workspace copies before validation.

Initialize exactly once:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/init_deft_cr_state.py" \
  --workspace "$WORKSPACE" \
  --workflow-yaml "$WORKSPACE/specs/workflow.yaml"
```

Use its printed `run_dir` for every later command. Follow [references/common-loop.md](references/common-loop.md) for exact common-stage commands, logging, resume behavior, and terminal-state checks.

## Baseline Evaluation

Evaluate `cosmos_reason.baseline_model_path` on the KPI dataset once. Generate the baseline TOML, run it through `tao-finetune-cosmos-reason`, require the job to exit successfully, locate exactly one completed `results.json`, and write `baseline/evaluate/bcq_accuracy_metrics.json`. Use [references/common-loop.md](references/common-loop.md) for the commands.

## Mining Initialization

Run mining initialization only for `mining` or `both`. The KPI and source datasets are fixed, so Cosmos Embed setup, inference, lookup generation, and consolidated embedding Parquets run once before the loop. Use [references/mining-loop.md](references/mining-loop.md) for the exact commands and completion checks. Skip every mining initialization stage in GenAI-only mode and log it as skipped only when a static stage ledger requires an event.

## Iteration Loop

For iterations `1..run.max_iterations`:

1. Use the previous completed evaluation's `results.json`; iteration 1 uses the baseline result.
2. Rewrite Cosmos Reason annotation ids to absolute KPI video paths with `prepare_gap_analysis_predictions.py`.
3. Run `tao-analyze-gaps-vlm-bcq`, then write `gap_status.json` with `inspect_gap_analysis.py`.
4. Stop the loop when no weak samples remain.
5. Run the enabled data-construction branches:
   - `mining`: follow [references/mining-loop.md](references/mining-loop.md).
   - `genai`: follow [references/genai-loop.md](references/genai-loop.md).
   - `both`: run mining and GenAI from the same gap file.
6. Assemble current branch outputs and accumulated annotations.
7. Train Cosmos Reason, discover the completed checkpoint, evaluate it, and compute iteration metrics.

### Annotation Assembly

Use `assemble_train_annotations.py --current-annotations` once for each enabled branch output. The command requires at least one new annotation across the enabled branches and rewrites all output `video` fields as absolute paths.

| Mode | Iteration 1 | Later iterations |
| --- | --- | --- |
| `mining` | Current mined annotations | Previous assembled + current mined |
| `genai` | Optional initial training annotations + current generated | Previous assembled + current generated |
| `both` | Current mined + current generated | Previous assembled + current mined + current generated |

For GenAI-only iteration 1, pass `train_dataset.annotations_path` as `--previous-annotations` and `train_dataset.media_dir` as `--previous-media-dir` when that optional dataset exists. Never pass the initial dataset in `both` mode. In later iterations, pass the prior iteration's `train/train_annotations.json` as `--previous-annotations`; it already contains absolute media paths.

If one branch in `both` mode produces no annotations, continue with the other branch. If neither branch produces new annotations, stop without retraining. Failed PAIDF rows remain in `failed_videos.jsonl` and are never converted to LLaVA.

## Training And Evaluation

Generate each train TOML with `setup_cosmos_reason_stage.py iteration-train`. Mining-only uses the source dataset media root. Modes containing GenAI use the DEFT workspace as the training media root because assembled annotations may span source data and multiple PAIDF iteration directories. All annotation `video` paths are absolute.

Iteration 1 trains from the baseline checkpoint. Later iterations use the previous checkpoint only when `cosmos_reason.continual_model` is true; otherwise each starts from the baseline. Annotation accumulation is independent of checkpoint accumulation and always follows the table above.

After training exits successfully, discover the latest safetensors checkpoint, generate the iteration evaluate TOML, run evaluation, require exactly one completed `results.json`, and compute `bcq_accuracy_metrics.json`. At loop termination run `summarize_bcq_accuracy_metrics.py` and include its baseline/iteration accuracy table in the final response.

## Completion Criteria

| Stage | Complete when |
| --- | --- |
| Validate | `verify_workflow_yaml.py` exits successfully; GenAI endpoint verification also succeeds when enabled. |
| Initialize | `workflow.yaml`, `deft_state.json`, and the resolved run directory exist. |
| Baseline | Cosmos Reason evaluation exits successfully and baseline results plus metrics exist. |
| Mining initialization | Required lookup and combined embedding Parquets exist, or the stage is disabled by mode. |
| Gap analysis | Prepared predictions, gap outputs, and `gap_status.json` exist. |
| Mining branch | Mined LLaVA annotations exist, or mining is disabled. |
| GenAI branch | PAIDF generated/failed handoffs and generated LLaVA annotations exist, or GenAI is disabled. |
| Assembly | `train/train_annotations.json` contains at least one new current annotation and valid absolute video paths. |
| Train | Training exits successfully and checkpoint discovery succeeds. |
| Evaluate | Evaluation exits successfully and iteration results plus metrics exist. |
| Stop | Accuracy report and summary compare the baseline with every completed iteration. |

## Troubleshooting

**Captioning endpoint unavailable**: Report the configured base URL and endpoint-verifier error. Ask the user to provide or repair an externally managed endpoint. Do not start vLLM.

**PAIDF exits nonzero**: Follow `paidf-cosmos-predict` troubleshooting. Report the Docker error, generated and failed counts, retained log path, and ask the user how to proceed before retrying or accepting partial results.

**No new annotations**: Report mining and successful PAIDF counts separately. Stop without training when both enabled branches produced zero current rows; do not train only on accumulated history.

**Initial GenAI seed has relative videos**: Pass its `train_dataset.media_dir` to the assembler. The assembled output must contain absolute paths before Cosmos Reason training.

**Prediction id does not match KPI annotations**: Confirm evaluation used the configured KPI annotations. Every result id must match a LLaVA annotation id or its resolved video path.

**Docker-created output is not writable**: Identify the producing container and use `restore_docker_mount_permissions.py` only after informing the user which path and ownership change are required.
