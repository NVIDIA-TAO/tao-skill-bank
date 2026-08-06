---
name: tao-run-automl
description: Run skill-owned AutoML / hyperparameter optimization (HPO) for NVIDIA TAO networks and actions. Handles Bayesian, BFBO, Hyperband, BOHB, ASHA, PBT, DEHB, Hyperband-ES, LLM, hybrid, and autoresearch algorithms; GEPA prompt optimization; multi-objective scoring; WandB tracking; platform jobs; and durable result handoff. Use when the user mentions TAO AutoML, hyperparameter
  optimization, HPO, automl, automl_settings, bayesian search, hyperband, ASHA, LLM-guided search,
  autoresearch, or wants to tune selected action parameters such as train/evaluate/inference/distill/prune/quantize for any TAO network. Platform-agnostic across Brev, SLURM, Kubernetes, and Docker.
license: Apache-2.0
compatibility: Requires Python 3.10+, PyYAML, jsonschema, numpy, scipy, and scikit-learn; docker + nvidia-container-toolkit for local GPU jobs. GEPA and WandB helpers are optional.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash Write
tags:
- automl
- hpo
- workflow
- training
- optimization
- llm
---

# TAO AutoML

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

Run automated hyperparameter optimization for a TAO model by combining:

1. The selected model skill under `skills/models/<model_skill>/`.
2. The selected platform skill under `skills/platform/<platform>/`.
3. The skill-owned `automl_step.py` engine for every algorithm and action, plus
   `gepa_step.py` when optimizing batched prompts.

Do not launch until model metadata, platform preflight, data visibility,
credentials, image choice, and compute shape are all proven.

## Reference Map

- `references/skill_info.yaml`: this workflow's structured metadata.
- `references/automl-sdk-free-bayesian.md`: complete step engine, durable state,
  algorithms, four-verb handoff, retry, observation, and finalize.
- `scripts/live_training_matrix.py`: opt-in GPU regression harness that drives
  every algorithm through real `classification_pyt` training, canonical job
  and metric records, checkpoint promotion/resume, and final evaluation.
- Split detailed references: `automl-preflight-concepts.md` for prerequisites
  and support checks; `automl-intent-algorithms.md` for search policy;
  `automl-compression-literature.md` for distill/prune/quantize algorithm
  sufficiency and future compression-search roadmap;
  `automl-runner-configuration.md` for engine CLI and WandB details;
  `automl-advanced-monitoring.md` for hooks, resume, and pitfalls; and
  `automl-examples.md` for conversation examples. `detailed-guide.md` is only
  the map.
- `skills/models/<network>/SKILL.md`: model-specific dataset requirements, metrics,
  HPO notes, checkpoint handoff, and known failures.
- `skills/models/<network>/references/skill_info.yaml`: action contract,
  container image, inputs, outputs, upload exclusions, and `mode`.
- `skills/platform/<platform>/SKILL.md`: selected platform preflight, credentials,
  resource shape, monitoring, and cancellation.
- `skills/core/tao-launch-workflow/SKILL.md`: shared intake pattern for platform,
  credentials, dataset visibility, image confirmation, and user confirmation.

## Preflight

1. Run the shared launch intake. If the user has not chosen a platform, ask;
   Brev, SLURM, Kubernetes, and Docker are equal peers.
2. Run the selected platform skill's preflight before generating experiment
   state or job files.
3. Verify the step engine's ordinary helpers:

```bash
python -c "import yaml, jsonschema, numpy, scipy, sklearn; print('AutoML helpers OK')"
```

4. If using GEPA or offline/online WandB, verify only the requested optional
helper:

```bash
python -c "import gepa; print('GEPA OK')"       # prompt optimization only
python -c "import wandb; print('WandB OK')"    # experiment tracking only
```

For release qualification on a Docker GPU host, run the all-policy live gate:

```bash
python skills/applications/tao-run-automl/scripts/live_training_matrix.py \
  --workspace /path/outside/the/repository/automl-live --clean
```

The harness is destructive only to the explicit workspace when `--clean` is
passed. It requires Pillow in addition to the ordinary helpers, and it never
downloads or installs an optimizer wheel.

## Model Support Gate

Before every run:

1. Read the model `SKILL.md` and `references/skill_info.yaml`.
2. Confirm `automl_enabled: true` for the model or that the model skill
   explicitly routes the selected action to AutoML.
3. Confirm `<skill_dir>/schemas/<action>.schema.json` exists and parses. This
   is the AutoML search-space gate.
4. For non-TAO-Core models such as Cosmos-RL and CLIP, also require
   `references/spec_template_<action>.yaml`; otherwise the engine has no
   complete action defaults.
5. If any gate fails, do not improvise a search space. Report the missing
   packaged artifact.

## Inputs

Collect these before experiment construction:

| Input | Requirement |
|---|---|
| `model_skill` | Resolved model skill directory under `skills/models/`. Accept user aliases such as `network_arch` only after resolving them to the packaged skill directory. |
| `network_arch` | Read from the resolved model skill metadata. |
| `action` | Action to optimize, usually `train`, `evaluate`, `inference`, `distill`, `prune`, or `quantize` when that action has a packaged schema/template. |
| `platform` | One of the supported TAO platform skills. |
| `train_dataset` / `eval_dataset` / action inputs | Use model-specific spec keys and dataset layout. Non-train actions often also require parent checkpoints, teacher checkpoints, calibration data, or pruned artifacts. |
| `results_root` | Local, Lustre, or S3 path appropriate for the platform. |
| `gpu_count`, `num_nodes` | Respect model and platform limits. |
| `container_image` | Resolve through model metadata and `versions.yaml`; show it to the user. |
| `automl_algorithm` | Default `bayesian` unless user asks for another algorithm or the model skill recommends one. |
| `metric`, `direction` | Prefer the model skill's validation/task metric. |
| `automl_budget` | Recommendation count, max epochs/rungs, concurrency, or population size as required by the algorithm. |

Never ask for secret values. Verify required env vars with
`[ -n "$VAR_NAME" ] && echo SET || echo UNSET`.

## Pre-Launch Review Gate

Before launching any recommendation jobs, show a concrete launch review and get
user confirmation. This gate applies to every AutoML run for every
AutoML-supported model/network; it is not Cosmos-specific and must not be
scoped to a single model skill. This applies even when platform and image
preflight already passed. The review must include:

- model/network, platform, image, GPU/node shape, and result/workspace root
- dataset mode and concrete spec keys, including train/eval sample counts when
  they can be read cheaply
- algorithm, budget, max concurrent jobs, metric, and direction
- searchable parameters and ranges, including default values when the user did
  not provide an explicit search space
- exact generated recommendation configs for the initial launch batch, produced
  in a review-only step before any recommendation job is submitted
- estimated runtime per recommendation and total expected wall time, with the
  assumptions used
- the automatic baseline eval job id, metric value, and result path from the
  post-preflight eval job, or an explicit blocker if the model has no runnable
  evaluate action or validation data
- the post-AutoML final evaluation plan for the selected best checkpoint/model,
  including metric, dataset, and record path

If the estimate is longer than the user's stated limit or materially longer
than a normal interactive run, ask whether to reduce recommendations, epochs,
dataset size, validation frequency, or search space before launch. Do not hide
multi-day estimates in logs.

## Automatic Baseline Eval Job

After platform, image, credential, data, and model preflight pass, run the
model's evaluate action once on the selected validation/eval data before
submitting any AutoML recommendation jobs. This is required AutoML setup, not an
optional "pretrained eval" question for the user. Use the same base model or
checkpoint that the AutoML training run starts from, the model skill's evaluate
spec/template, and the selected platform's normal job submission path. If the
model skill recommends a smaller shape for evaluation than training, use that
shape and call it out in the launch review.

Share the eval metric number with the user in the launch review before asking
for confirmation to launch recommendations. If the model has no packaged
evaluate action, the eval dataset is missing, or the eval job fails, stop and
report the blocker instead of silently falling back to a training-loss-only
AutoML run. Continue without this baseline only when the user explicitly accepts
that the run will optimize a proxy metric and will not have an impact baseline.

The AutoML workflow owns final evaluation of the selected best
checkpoint/model. Finalize `best_rec.json`, submit the model's evaluate action
through the selected platform skill, and persist its canonical metric record in
the experiment workspace. Use the baseline's metric, dataset, and direction and
record a concrete final-evaluation status and reason.

## Dependency And Data Preflight

If the selected workflow needs object storage or a platform CLI and the tool is
missing, report the missing dependency and offer the exact install command
before continuing. After user approval, rerun
`scripts/check_tao_launch_preflight.py` with `--install-missing-tools` so it
installs the smallest needed package and immediately retries path verification.
For S3 paths, verify both credentials and path readability from the launch
platform before creating experiment artifacts. Do not wait for the first training
container to discover a missing AWS CLI, S3 client, or unreadable URI.

For models that read large media archives or directories during every training
trial, stage or extract the dataset once to storage visible from the execution
platform, then point all recommendation specs at that staged path. Record the
source URI, staged path, byte/file-count evidence when available, and timestamp
in `<workspace>/evaluations/data_staging.json`. If staging is not possible,
include the repeated S3 I/O risk in the pre-launch review and ask before
spending a long AutoML budget on it.

When the model skill defines sample-count-sensitive constraints, enforce them
before launch. Reject or cap every batch-size recommendation that would create
zero training steps for the selected dataset and GPU shard count. Use
`scripts/check_tao_launch_preflight.py --effective-batch-limit
train_annotation=<batch_size>,<shard_count>` for each generated recommendation
before submitting it. If a recommendation later fails because the data is too
small for the effective batch size, classify it as an invalid configuration,
replace or adjust it only when remaining budget exists, and report the
correction in the final summary.
When train sample count is known from an annotation file or cheap manifest
read, check every recommendation against it before submit and record any cap or
adjustment in recommendation metadata.

## Algorithm Policy

| Algorithm | Good fit | Required knobs |
|---|---|---|
| `bayesian` | Default for small/medium budgets and few parameters. | `num_recommendations`, metric, direction |
| `bfbo` | Batch-friendly expensive trials with active-point diversity. | recommendation count and concurrency |
| `hyperband`, `asha` | Many configs with cheap early rungs; ASHA supports parallelism. | `max_epochs`, `reduction_factor`, optional `max_concurrent` |
| `bohb`, `dehb` | Mixed Bayesian/evolutionary search with multi-fidelity budgets. | same rung budget fields as Hyperband |
| `hyperband_es` | Learning curves reliably predict weak jobs before their rung ends. | Hyperband knobs plus observation/cancel thresholds |
| `pbt` | Long training where schedules should mutate during training. | population and generation budget |
| `llm`, `hybrid`, `autoresearch` | User explicitly wants LLM-guided search and has an endpoint configured. | LLM endpoint config plus budget |

For `evaluate` or `inference`, default to Bayesian/BFBO-style search over the
selected action's prompt, decoding, preprocessing, or runtime config knobs.
Use a task metric from the action outputs/logs and set `direction` explicitly
when the metric name is ambiguous. Do not use training-loss assumptions for
actions that do not update weights.

For `distill`, use the same train-like policy when the distill action performs
epoch-based optimization and writes checkpoints. For single-shot `prune` and
`quantize`, default to `bayesian` or `bfbo` unless the action schema/model skill
declares an epoch-like or calibration-budget field that makes
`hyperband`/`asha`/`bohb`/`dehb` meaningful. Use `eval_fn` when the selected
metric must be computed by a follow-up evaluate/inference action after the
compression action completes.

Prefer the model skill's recommendation over generic defaults. Avoid ASHA or
Hyperband when the model skill says startup, validation, or checkpoint cost
dominates short trials.

## Spec And Search Space

Build specs as nested dictionaries. If a model skill lists paths in dotted
notation for readability, walk the path and assign the nested leaf; do not store
flat dotted strings as spec keys.

Use the packaged selected-action schema for:

- `automl_default_parameters`
- `automl_disabled_parameters`
- valid min/max ranges
- enums, option weights, conditions, dependencies, and popular parameters

User-provided search spaces must stay inside schema constraints. For integer
knobs with discrete choices, include the schema's required integer option shape
instead of a loose list if the model skill calls that out.

Data source overrides are mandatory unless the model skill says the launcher can
derive them. Preserve exact user-provided spec keys when the dataset uses direct
annotation/media paths.

## Metric Policy

Training loss is cheap but can be misleading. Prefer the model skill's task
metric. Use one of these:

- Log metric: `metric=<name>`, `direction=maximize|minimize`.
- `metric_extractor(logs, metric_name)`: parse the model's logs when the
  default resolver is ambiguous.
- `eval_fn(rec, train_job_id)`: run the model's evaluate action after each
  recommendation when the user wants a downstream task metric.

Do not map `kpi` to a metric unless the model skill explicitly defines that
mapping.

For every AutoML run with a runnable evaluate action and validation/eval data,
run the automatic baseline eval job after preflight and before recommendations.
The final report must compare that baseline metric, each recommendation's
metric, and the selected best metric so users can see the impact of tuning. For
model skills that require an `eval_fn` to compute the real task metric, use
that evaluator instead of optimizing a convenient training loss unless the user
explicitly accepts the proxy metric.

## Execution Routing

For every action and algorithm, follow
`references/automl-sdk-free-bayesian.md`: call `automl_step.py`, submit each
returned nested spec through the selected platform skill, bind its job-record
id, then report the canonical metric record. Never silently downgrade the
user's algorithm. Use `observe` for intermediate learning curves and honor a
`should_cancel` response through the platform's cancel verb.

For GEPA, use `gepa_step.py`; its batch callback submits the selected TAO action
through the same platform contract.

Only resume an existing workspace when the user explicitly asks to resume,
continue, recover, or inspect an existing experiment. Treat a plain "run
AutoML" request as a fresh run.

## Monitoring

Combine `automl_step.py status` with the selected platform skill's `status` and
`logs` verbs.
For active jobs, report:

- recommendation id / trial id
- platform job id
- status
- current metric
- best metric so far
- selected hyperparameters for the current/best recommendation
- elapsed time and updated ETA when enough timing data exists

On failure, classify whether it is infrastructure, data visibility, image,
credential, spec/schema, or model-code failure. Fix only the minimal cause and
do not silently spend additional budget on repeated invalid recommendations.
If a blocker is fixed during run setup, continue from the original task after
showing the updated preflight/launch review instead of leaving the user to
restate the request.

For LLM-based algorithms, inspect recommendation metadata before calling the run valid.
Verify that LLM calls succeeded, proposals were generated, prior metrics were
used to choose later parameter changes, and logs show keep/discard or
equivalent algorithm decisions. If the brain falls back to random sampling,
classify the LLM workflow as failed or blocked instead of treating it as a
valid LLM-guided run.

## Result Handoff

At completion:

1. Identify the best recommendation by the selected metric and direction.
2. Return the best child job id and its result path.
3. Resolve the model checkpoint or action artifact using the model skill's
   declared output metadata. Do not guess filenames such as `latest`.
4. Report the exact search space, algorithm, budget, metric, and platform.
5. Report the automatic baseline eval job id/result path/metric, all
   recommendation metrics, final evaluation status/result path/metric, failed
   recommendations and root causes, elapsed time, and final runtime notes.
6. If this feeds a workflow such as AutoML + DEFT, pass the winning spec
   overrides and checkpoint through the workflow's declared handoff fields.

## Common Pitfalls

- Do not expect `~/tao-core` at runtime. Schemas and templates must be packaged
  inside the model skill.
- Do not infer dataset URIs from previous runs.
- Do not precompute platform-managed output paths; use the job-record's bound
  `results_dir`.
- For SLURM, stage large datasets on Lustre rather than burning GPU allocation
  time on large S3 downloads.
- For gated HuggingFace models, verify `HF_TOKEN` is set without reading it.
- If all recommendations fail, stop and summarize the shared root cause instead
  of launching more trials.
