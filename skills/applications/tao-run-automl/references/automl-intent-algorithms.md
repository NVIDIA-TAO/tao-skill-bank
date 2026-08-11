# AutoML Intent And Algorithms

## Default behavior

A normal “run AutoML” request is a fresh experiment. Resolve the model skill,
selected action, platform, complete base spec, data, metric/direction, and image
from packaged metadata. Default to Bayesian search with a small recommendation
budget unless the model skill or user chooses another algorithm. Ask only for
choices that materially change cost or outcome.

Do not launch from a historical workspace, silently resume, switch algorithms,
or invent an action schema. Resume only on explicit continue/recover/resume
intent.

## Required automatic evaluation

After platform, image, credential, model, and data preflight, run the model's
evaluate action once on the chosen validation data as the automatic baseline eval job.
The launch review must give its job id, result path, and metric value.
If the evaluate action or validation data is unavailable, stop unless the user
explicitly accepts a proxy-only run.

After optimization, evaluate the frozen best checkpoint/artifact on the same
dataset and objective. Compare baseline, recommendation, and final metrics.

## Inputs and defaults

| Input | Default/rule |
|---|---|
| `model_skill`, `network_arch` | Resolve aliases to one packaged model skill |
| `action` | `train`, unless the request names another schema-backed action |
| `algorithm` | `bayesian` |
| `metric`, `direction` | Model task metric and explicit direction |
| `max_recommendations` | Small interactive budget, commonly 5–10 |
| `max_concurrent` | 1 for Bayesian; platform-safe parallelism for batch/multi-fidelity methods |
| `skill_dir` | Absolute packaged model-skill directory |
| `workspace` | New experiment directory unless resuming explicitly |
| `image` | Selected action's resolved model image |

Run dataset conversion or staging once before optimization and reuse its
artifact across all trials. Every recommendation must remain valid for known
sample-count, batch-size, GPU-shard, and model constraints.

## Algorithm choice

| Algorithm | Choose when | Main knobs |
|---|---|---|
| `bayesian` | Sequential, expensive trials and modest dimensions | recommendation count |
| `bfbo` | Several expensive trials should launch together | recommendation count, concurrency |
| `hyperband` | Short rungs are representative and checkpointing is cheap | max epochs, reduction |
| `bohb` | Hyperband fidelity plus learned density proposals | Hyperband knobs, KDE warm-up |
| `asha` | Parallel workers should refill asynchronously | max trials, concurrency, top configs |
| `pbt` | Long runs benefit from exploit/perturb schedules | population, generations, interval |
| `dehb` | Mixed/discrete spaces benefit from differential evolution | Hyperband and mutation/crossover knobs |
| `hyperband_es` | Intermediate curves can safely stop weak jobs | Hyperband and curve thresholds |
| `llm` | Domain reasoning should propose individual configs | endpoint, model, recommendation count |
| `hybrid` | A language model should plan classical search phases | endpoint, model, experiment budget |
| `autoresearch` | Reflective search should use failures and evaluation feedback | endpoint, model, research program |

Do not use multi-fidelity methods for one-shot actions unless their schema
exposes a meaningful resource axis. Prefer Bayesian/BFBO/LLM-family methods for
evaluate, inference, prune, or quantize configuration search. Distill can use
training-style budgets when it is epoch based.

## LLM requirements

For `llm`, `hybrid`, or `autoresearch`, resolve endpoint and model from explicit
input or `AUTOML_LLM_ENDPOINT`/`AUTOML_LLM_MODEL`. Providers read credentials
from `AUTOML_LLM_API_KEY` or `NVIDIA_API_KEY`; never print or persist them. A
run is genuinely language-guided only if recommendation metadata records
`fallback: false`. Treat fallback as a degraded/blocked LLM validation even
though it preserves a deterministic optimizer path.

Evolvable text must be explicitly named and string-valued in the action schema.
Without that opt-in, categorical text remains bounded to declared options.

## Compression and non-training actions

- `distill`: optimize task metric, student/teacher-compatible knobs, and epoch
  budget; preserve both input checkpoints.
- `prune`: jointly optimize sparsity and retained task quality; evaluate each
  artifact when the action output does not contain the task metric.
- `quantize`: optimize calibration/runtime knobs with quality and latency/size
  objectives; use multi-fidelity only for a representative calibration budget.
- `evaluate`/`inference`: optimize prompts, decoding, preprocessing, and runtime
  config using declared action artifacts, not checkpoint assumptions.

GEPA is preferred for decomposable prompt objectives with per-example feedback.
Use validation-only candidate selection and touch the test split only once with
the frozen winner.

## Launch review

Before submitting trials, show model/action, platform, image, GPU/node shape,
workspace, visible data paths/counts, objective definitions, algorithm and full
budget, exact initial configs, search bounds/defaults, baseline record, runtime
estimate, and final evaluation plan. Ask for confirmation when the run is long
or side effecting.

## Naming and isolation

Use one experiment id/state file per run and one job record per recommendation
attempt. Retry ids link to their failed infrastructure attempt. Promotions and
PBT children preserve config id plus parent/resume lineage. Never share a state
file between concurrent experiments.
