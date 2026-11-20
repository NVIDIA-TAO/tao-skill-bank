## Description: <br>
Run the full DEFT AOI improvement loop for NVIDIA TAO VisualChangeNet / ChangeNet PCB inspection models: baseline evaluate, RCA, Cosmos AnomalyGen / AMP synthetic defects, k-NN mining, retraining, and deployment gating against a customer-defined primary metric and optional constraints. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers and ML engineers building NVIDIA TAO VisualChangeNet / ChangeNet automated optical inspection (AOI) models for PCB defects use this skill to have an agent iteratively evaluate, analyze root causes, generate synthetic defects, mine data, retrain, and gate deployment until a customer-defined KPI target is met, including air-gapped runs with pre-staged assets. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [air-gap.md](references/air-gap.md) <br>
- [data-layout.md](references/data-layout.md) <br>
- [metric-contract.md](references/metric-contract.md) <br>
- [network-bootstrap.md](references/network-bootstrap.md) <br>
- [pipeline-and-state.md](references/pipeline-and-state.md) <br>
- [preflight.md](references/preflight.md) <br>
- [prepare-for-inference.md](references/prepare-for-inference.md) <br>
- [scripts-and-agents.md](references/scripts-and-agents.md) <br>
- [visual-changenet.md](references/visual-changenet.md) <br>
- [tao-analyze-gaps-visual-changenet.md](references/tao-analyze-gaps-visual-changenet.md) <br>
- [tao-generate-anomalies.md](references/tao-generate-anomalies.md) <br>
- [tao-mine-aoi-images.md](references/tao-mine-aoi-images.md) <br>
- [tao-route-visual-changenet-samples.md](references/tao-route-visual-changenet-samples.md) <br>
- [REPORT_RENDERING.md](references/REPORT_RENDERING.md) <br>
- [TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Files, Analysis] <br>
**Output Format:** [Markdown status updates with inline shell commands; JSON state file (deft_state.json), CSV, YAML specs, model checkpoints, and an HTML loop report] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Verbose training/inference/Docker output is redirected to files; state is updated only through bundled commit scripts] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
3 evaluation tasks (3 positive): tao-run-deft-aoi-basic, tao-run-deft-aoi-customer-metric, and tao-run-deft-aoi-resume-state; 1 attempt per task, each in its own isolated sandbox pod (environment `k8s-sandbox`), compared against a no-skill baseline. Evaluator version `1.6.10`, evaluation date 2026-11-20. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from `security` (100%). <br>
- Correctness: Is the answer correct? Scored from `accuracy` (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from `skill_execution` (100%). <br>
- Effectiveness: Did the skill help complete the task? Scored from `goal_accuracy` (50%) + `behavior_check` (50%). <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Scored from `skill_efficiency` (50%) + `token_efficiency` (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Security: unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Skill Execution: whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool Productivity: tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Accuracy: final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Goal Accuracy: whether the user's goal was achieved. <br>
- `behavior_check`: Behavior Check: whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Token Efficiency: actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
> ✅ **Overall verdict: PASS — Recommended for publication**

### Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 79.2% — baseline ran, but no comparable score was available; uplift unavailable | 78.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 66.7% → 100.0% (+33.3 points) |
| Correctness | 33.3% → 93.3% (+60.0 points) | 60.0% → 93.3% (+33.3 points) |
| Discoverability | 56.7% — baseline ran, but no comparable score was available; uplift unavailable | 56.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 26.1% → 55.0% (+28.9 points) | 45.6% → 48.9% (+3.3 points) |
| Efficiency | 90.9% — baseline ran, but no comparable score was available; uplift unavailable | 95.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.9% → 94.3% (-3.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 89.4% → 91.1% (+1.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 105 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 5 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 3 task(s) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
