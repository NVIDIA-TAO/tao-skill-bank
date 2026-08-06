---
name: tao-run-deft-cr-its
description: >-
  Run iterative ITS Cosmos-Reason DEFT improvement using mining, PAIDF GenAI generation, or both.
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

Improve the non-reasoning binary-classification path of Cosmos Reason by evaluating a KPI dataset, identifying false-positive and false-negative questions, constructing training annotations through nearest-neighbor mining, PAIDF generation, or both, retraining Cosmos Reason, and evaluating each new checkpoint.

## Bundled Resources

Resolve `DEFT_SKILL_ROOT` to the absolute directory containing this installed `SKILL.md`. The agent or plugin runtime resolves this path; it is not a user input. Run bundled helpers with `run_script("scripts/<name>.py", ...)` when the runtime provides it. Otherwise invoke them directly with `python3 "$DEFT_SKILL_ROOT/scripts/<name>.py"`. Never require a source-repository checkout or change the user's working directory to a repository root.

Invoke `tao-finetune-cosmos-reason`, `tao-finetune-cosmos-embed`, `tao-analyze-gaps-vlm-bcq`, `tao-mine-nearest-neighbors`, `paidf-cosmos-predict`, and the selected platform skill by registered skill name. Those skills own their action commands, credentials, and bundled assets. This workflow overrides only the Cosmos Reason runtime image: resolve `images.tao_toolkit.deft_cosmos_reason` from `versions.yaml` and pass it as the planner `image_tag` and submitted action image for every Cosmos Reason train and evaluate launch.

## Workspace

The user provides an absolute DEFT workspace path. Do not use `/workspace`, which conflicts with Cosmos Reason runtime conventions.

```text
<deft_workspace>/
├── data/       # KPI and optional training/source datasets
├── hf_cache/   # persistent Hugging Face cache
├── model/      # baseline Cosmos Reason and optional local Cosmos Embed models
├── specs/      # workflow.yaml and user-editable templates
└── results/    # all generated workflow artifacts
```

Each run writes under `<deft_workspace>/results/<run_name>`. When `run.name` is null, initialization creates `run_<YYYYMMDD_HHMMSS>` and records the resolved path in `deft_state.json`. The run contains the workflow snapshot, append-only loop log, baseline evaluation, optional fixed mining embeddings, accuracy report, and one `iter_<N>/` directory per improvement iteration. Modes containing GenAI also write `paidf/` and `genai/` inside each iteration directory.

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

`run`, `data_generation`, `kpi_dataset`, and `cosmos_reason` are always required. `data_generation.mode` must be `mining`, `genai`, or `both`.

`train_dataset` and `mining` are required when the mode includes mining. In those modes, the training dataset is the mining source pool; its raw annotations are never inserted directly into training. In GenAI-only mode, `train_dataset` is optional and, when present, seeds iteration 1 training.

`genai` is required when the mode includes GenAI. The user must provide `genai.vlm_captioning_endpoint`; this workflow never starts a captioning server. `genai.paidf_num_gpus` is also required. A null `generation_settings` uses the `paidf-cosmos-predict` default. A null `caption_prompt_file` uses `assets/qwen_its_caption_prompt.txt`.

All user-configured filesystem paths must be absolute and inside the DEFT workspace. The workflow-owned default caption prompt is resolved from the installed skill and is the only exception.

## Preflight And Initialization

When custom templates are not supplied, copy `assets/cr_base_evaluate.toml` and `assets/cr_base_train.toml` into the workspace `specs/` directory. When mining is enabled, also copy `assets/default_cosmos_embed_inference.yaml` and the nearest-neighbor skill's default template. Point `workflow.yaml` at those workspace copies.

Validate before launching any workload:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/verify_workflow_yaml.py" \
  --workspace "$WORKSPACE" \
  --workflow-yaml "$WORKSPACE/specs/workflow.yaml"
```

When GenAI is enabled, immediately invoke the `paidf-cosmos-predict` endpoint verifier against `genai.vlm_captioning_endpoint`. Stop if its OpenAI-compatible `/models` endpoint is unavailable; do not run baseline evaluation and do not start vLLM.

Initialize exactly once:

```bash
python3 "$DEFT_SKILL_ROOT/scripts/initialize_workflow.py" \
  --workspace "$WORKSPACE" \
  --workflow-yaml "$WORKSPACE/specs/workflow.yaml"
```

Use its printed `run_dir` for every later command. Before resuming, run `resume_position.py --run-dir "$RUN_DIR"`; `loop_log.jsonl` is the event source and `deft_state.json` is its refreshed snapshot.

## Baseline Evaluation

Evaluate `cosmos_reason.baseline_model_path` on the KPI dataset once. Use `prepare_cosmos_reason_evaluate.py`, run the resulting TOML through `tao-finetune-cosmos-reason` with the DEFT image override, require terminal job success, locate exactly one completed `results.json`, and write `baseline/evaluate/bcq_accuracy_metrics.json`. Follow [references/mining-loop.md](references/mining-loop.md) for the exact common-stage commands.

## Mining Initialization

Run mining initialization only for `mining` or `both`. KPI and source datasets are fixed, so Cosmos Embed setup, inference, lookup generation, and consolidated embedding Parquets run once before the loop. Skip these stages entirely in GenAI-only mode. Follow [references/mining-loop.md](references/mining-loop.md).

## Iteration Loop

For iterations `1..run.max_iterations`:

1. Use the previous completed evaluation's `results.json`; iteration 1 uses the baseline result.
2. Rewrite Cosmos Reason annotation ids to absolute KPI video paths with `prepare_gap_analysis_predictions.py`.
3. Run `tao-analyze-gaps-vlm-bcq` and count valid rows in `gaps/kpi_gaps.jsonl` after its job reaches terminal success.
4. Stop when no weak samples remain.
5. Run each enabled data-construction branch from the same gaps file:
   - `mining`: follow [references/mining-loop.md](references/mining-loop.md).
   - `genai`: follow [references/genai-loop.md](references/genai-loop.md).
   - `both`: run both branches.
6. Run `prepare_cosmos_reason_train.py` to construct current annotations, merge accumulated annotations, and generate `train/specs/train.toml`.
7. Train Cosmos Reason, evaluate the completed checkpoint, compute metrics, and remove resumable checkpoint directories while preserving safetensors exports.

### Annotation Assembly

`prepare_cosmos_reason_train.py` deduplicates by LLaVA `id` and rewrites every output `video` field as an absolute path.

| Mode | Iteration 1 | Later iterations |
| --- | --- | --- |
| `mining` | Current mined annotations | Previous assembled + current mined |
| `genai` | Optional initial training annotations + current generated | Previous assembled + current generated |
| `both` | Current mined + current generated | Previous assembled + current mined + current generated |

Never insert the initial training dataset into the assembled annotations in `both` mode. If one branch in `both` mode produces no current annotations, continue with the other. If neither branch produces new annotations, stop without training. Failed PAIDF rows remain in `failed_videos.jsonl` and are never converted to LLaVA.

## Training And Evaluation

Mining-only training uses `train_dataset.media_dir`. Modes containing GenAI use the DEFT workspace as the configured training media root because assembled records have absolute paths spanning source data and PAIDF iteration directories.

Iteration 1 trains from the baseline checkpoint. Later iterations use the previous checkpoint only when `cosmos_reason.continual_model` is true; otherwise each iteration starts from the baseline. Annotation accumulation is independent of checkpoint accumulation.

After training reaches terminal success, `prepare_cosmos_reason_evaluate.py` discovers the latest completed safetensors checkpoint and writes the iteration evaluation TOML. Require evaluation success and exactly one completed `results.json`, compute `bcq_accuracy_metrics.json`, then run `cleanup_cosmos_reason_training.py`. At loop termination, run `summarize_bcq_accuracy_metrics.py` and include its baseline/iteration accuracy table in the final response.

## Completion Criteria

| Stage | Complete when |
| --- | --- |
| `validate_workflow` | The validator exits successfully; GenAI endpoint verification also succeeds when enabled. |
| `initialize_workflow` | `$RUN_DIR/workflow.yaml` and `$RUN_DIR/deft_state.json` exist. |
| `baseline_evaluate` | Evaluation succeeds, exactly one `results.json` is found, and baseline metrics exist. |
| `prepare_cosmos_embed_inference` | Required lookups and specs or staged embedding Parquets exist, or mining is disabled. |
| `cosmos_embed` | Every required inference job succeeds, or mining is disabled. |
| `convert_embeddings` | Mining-ready KPI/train Parquets exist, or mining is disabled. |
| `gap_analysis` | The job succeeds and `kpi_gaps.jsonl` has been counted. |
| `prepare_nearest_neighbor_mining` | Target/source inputs and one mining spec exist, or mining is disabled. |
| `mine_nearest_neighbors` | The mined-neighbor Parquet exists, or mining is disabled. |
| `prepare_paidf_input` | `paidf/media.jsonl` exists, or GenAI is disabled. |
| `paidf` | PAIDF reaches terminal success and generated/failed handoffs account for every input row, or GenAI is disabled. |
| `llava_conversion` | Generated LLaVA annotations exist for successful PAIDF rows, or GenAI is disabled. |
| `prepare_cosmos_reason_train` | At least one current annotation and the accumulated annotations plus train TOML exist. |
| `train` | Cosmos Reason training reaches terminal success. |
| `evaluate` | Checkpoint discovery and evaluation succeed and iteration metrics exist. |
| `cleanup_cosmos_reason_training` | Raw checkpoint directories are removed, safetensors remain, and the cleanup report exists. |
| `loop_stop` | The accuracy report compares baseline with every completed iteration. |

## Troubleshooting

**Captioning endpoint unavailable**: Report the configured base URL and endpoint-verifier error. Ask the user whether to provide or repair an externally managed endpoint. Do not start vLLM.

**PAIDF exits nonzero**: Follow `paidf-cosmos-predict` troubleshooting. Report the Docker error, generated and failed counts, retained log path, and ask the user how to proceed before retrying or accepting partial results.

**No new annotations**: Report mining and successful PAIDF counts separately. Stop without training when all enabled branches produced zero current rows; do not train only on accumulated history.

**Prediction id does not match KPI annotations**: Confirm evaluation used the configured KPI annotations. Every result id must match a LLaVA annotation id or its resolved video path.

**Docker-created output is not writable**: Identify the producing container and use `restore_docker_mount_permissions.py` only after informing the user which path and ownership change are required.

**Cosmos Reason reports `No space left on device` for `/dev/shm/nccl-*`**: For local Docker, require `--ipc=host --ulimit memlock=-1 --ulimit stack=67108864`. Other platforms must provide equivalent shared-memory and memlock settings.
