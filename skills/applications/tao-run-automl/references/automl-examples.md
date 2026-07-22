# TAO AutoML Example Conversations

Example user requests and the correct high-level agent response shape.

Every example uses the skill-owned step engine and a native platform skill.

Load this file only when the compact `SKILL.md` points here for the current task. If this reference conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the compact/current source wins.

## Contents

- User: "Optimize `<network_arch>` training on my dataset"
- User: "I want the real task metric, not the default proxy"
- User: "Use the LLM to figure out the best hyperparameters"
- User: "Run a fully autonomous search — let the AI figure everything out"
- User: "Resume the experiment from yesterday"
- User: "Switch to ASHA with 30 trials and track in WandB"
- User: "I don't know what parameters to tune for `<network_arch>`"

## Example Conversations

### User: "Optimize `<network_arch>` training on my dataset"

```
Agent: I'll read the model skill for `<network_arch>` first, then initialize a fresh SDK-free Bayesian experiment using its train schema, required datasets, metric, and recommended search space. I need:
1. The dataset URI or URIs required by that model skill
2. How many trials? (default: 10)

User: s3://bucket/data/train, 10 recs

Agent: Launch review before I spend GPU budget:
- model skill, network_arch, platform, image, GPU/node shape, workspace, dataset paths/counts, metric/direction, algorithm, budget, search bounds, estimated runtime, and automatic baseline eval job id/result path/metric
- exact generated recommendation configs for the initial launch batch, created in a review-only step before any recommendation job is submitted

User: confirmed

[alternates automl_step.py with the selected platform's submit/status/logs verbs]
```

### User: "I want the real task metric, not the default proxy"

```
Agent: This metric needs a separate evaluation job per recommendation, so it
adds one model-skill evaluation action per trial. I’ll include that cost in the
budget and declare the task metric name and direction explicitly.
[shows launch review with training and evaluation cost, metric/direction, and
the canonical metric-record handoff]
User: confirmed
[submits each train job, submits its evaluation job with the train job as
parent, then reports the evaluation metric record to automl_step.py]
```

### User: "Use the LLM to figure out the best hyperparameters"

```
Agent: I'll use the LLM algorithm — it reasons about your network architecture and learns from each experiment.
I need three things for the LLM brain:
1. OpenAI-compatible endpoint URL
2. Model name served by that endpoint
3. API key available in `AUTOML_LLM_API_KEY` or `NVIDIA_API_KEY`

User: endpoint "https://inference-api.nvidia.com", model "gcp/google/gemini-3.1-pro-preview"

Agent: I verified the endpoint and key environment variable are available, the
selected train image is confirmed, and platform/data preflight passed. Launch
review: 10 LLM-guided recs, metric/direction, search bounds, workspace,
GPU/node shape, and estimated runtime.
User: confirmed
[initializes automl_step.py with `--algorithm llm`, `--llm-endpoint`, and
`--llm-model`; the engine reads the key only from the environment, then the
agent alternates recommend/bind-job/report with the platform skill]
```

### User: "Run a fully autonomous search — let the AI figure everything out"

```
Agent: I'll use the autoresearch algorithm. It will:
1. Ground the research program in the model skill and bounded schema
2. Reflect on successful metrics, failures, and sanitized feedback
3. Propose bounded modifications with recorded reasoning
4. Stop at the explicit experiment cap

I'll cap it at 30 experiments. I need:
- Dataset URI
- LLM endpoint (default: https://inference-api.nvidia.com)
- LLM model (default: gcp/google/gemini-3.1-pro-preview)
- LLM API key in `AUTOML_LLM_API_KEY` or `NVIDIA_API_KEY`

User: dataset s3://bucket/data, endpoint https://inference-api.nvidia.com, model gcp/google/gemini-3.1-pro-preview

[shows launch review with 30-experiment cap, LLM config source, data/image/platform checks, exact generated configs for the initial launch batch, and estimated runtime]
User: confirmed

[initializes automl_step.py with `--algorithm autoresearch`,
`--max-experiments 30`, the endpoint/model flags, and an optional reviewed
research-program file; alternates recommendations and platform jobs]
```

### User: "Resume the experiment from yesterday"

```
Agent: Resuming. I'll read `automl_experiment.json`, reconcile the bound child
job through the selected platform's status/logs verbs, report its terminal
metric if available, then show a resume review before submitting new work.
[shows resume review with completed/failed/in-flight recs, remaining budget, metric state, image/platform/data checks, and ETA]
User: confirmed
[continues with automl_step.py status/recommend/bind-job/report]
```

### User: "Switch to ASHA with 30 trials and track in WandB"

```
Agent: Switching to ASHA — parallel + early-stopping. I’ll first check the model skill for any warning that short trials are dominated by startup, validation, or checkpoint cost, then initialize a fresh experiment with WandB tracking.
[shows launch review with ASHA rungs, max concurrent jobs, WandB project, search bounds, and estimated runtime]
User: confirmed
[initializes automl_step.py with `--algorithm asha --max-trials 30`, the rung
and concurrency flags, and `--wandb --wandb-project tao-hpo`; launches and
reports trials through the selected platform skill]
```

### User: "I don't know what parameters to tune for `<network_arch>`"

```
Agent: I can generate a configuration from your goal description. What are you trying to achieve?

User: Maximize my task metric on a small custom dataset with 1 GPU.

Agent: Based on your constraints, I recommend:
- Algorithm: bayesian (sample-efficient for small budgets)
- Parameters: from the model skill's AutoML / HPO Notes and the generated schema
- Budget: 12 recs
- Ranges: from the model skill and user constraints
[reads the schema search space, shows launch review with generated config and estimated runtime]
User: confirmed
[runs the skill-owned Bayesian step loop]
```
