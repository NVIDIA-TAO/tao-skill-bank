## Description: <br>
Run the full DEFT smart-data-augmentation loop for NVIDIA TAO Grounding DINO object detection: zero-shot baseline inference, KPI analysis, per-class gap analysis, SigLIP embedding of weak images, unique-neighbor mining against a source pool, ODVG dataset staging, and retraining — repeated for a fixed number of iterations. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and ML engineers use this skill to have an agent run an iterative gap-analysis, mining, and retraining loop that grows the training set and improves mAP for a TAO Grounding DINO detector, and to prepare the pseudo-labeled, embedded source pool that the loop mines from. <br>

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
- [references/data-layout.md](references/data-layout.md) <br>
- [references/prep-source-pool.md](references/prep-source-pool.md) <br>
- [references/preflight.md](references/preflight.md) <br>
- [references/pipeline-and-state.md](references/pipeline-and-state.md) <br>
- [references/scripts-and-agents.md](references/scripts-and-agents.md) <br>
- [references/grounding-dino.md](references/grounding-dino.md) <br>
- [references/stage-mined-data.md](references/stage-mined-data.md) <br>
- [references/tao-analyze-detection-kpi.md](references/tao-analyze-detection-kpi.md) <br>
- [references/tao-analyze-gaps-od-map.md](references/tao-analyze-gaps-od-map.md) <br>
- [references/tao-generate-image-embeddings.md](references/tao-generate-image-embeddings.md) <br>
- [references/tao-mine-od-images.md](references/tao-mine-od-images.md) <br>
- [TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Files, Analysis] <br>
**Output Format:** [Markdown status updates with inline bash code blocks; on-disk JSON state, JSONL loop log, model checkpoints, and a Markdown loop report] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Verbose train, inference, and Docker output is redirected to files; the loop report is re-rendered on every committed stage] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (4 positive), 1 attempt per task, each attempt in its own isolated sandbox pod in a `k8s-sandbox` environment; each task was also attempted without the skill as a baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from `security` (100%). <br>
- Correctness: Is the answer correct? Scored from `accuracy` (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from `skill_execution` (100%). <br>
- Effectiveness: Did the skill help complete the task? Scored from `goal_accuracy` (50%) + `behavior_check` (50%). <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Scored from `skill_efficiency` (50%) + `token_efficiency` (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
> ✅ **Overall verdict: PASS — Recommended for publication**

### Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 84.6% — baseline ran, but no comparable score was available; uplift unavailable | 64.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 75.0% → 100.0% (+25.0 points) |
| Correctness | 15.0% → 80.0% (+65.0 points) | 25.0% → 75.0% (+50.0 points) |
| Discoverability | 75.0% — baseline ran, but no comparable score was available; uplift unavailable | 23.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 14.4% → 78.1% (+63.7 points) | 14.4% → 38.8% (+24.4 points) |
| Efficiency | 89.7% — baseline ran, but no comparable score was available; uplift unavailable | 86.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.7% → 96.1% (-2.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 93.3% → 93.1% (-0.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
