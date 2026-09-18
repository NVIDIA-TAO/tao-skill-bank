## Description: <br>
Full PAIDF AnomalyGen pipeline — fine-tune on a new anomaly dataset, generate synthetic anomaly images (SDG), evaluate quality (nn_score), and search per-sample (guidance, crop_ratio) parameters. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers who need to fine-tune NVIDIA Cosmos-based models on anomaly datasets and generate synthetic defect images for industrial inspection use cases. <br>

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
- [TAO Skill Bank Repository](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>
- [Agent Skills Open Standard](https://agentskills.io) <br>
- [references/finetune.md](references/finetune.md) <br>
- [references/inference.md](references/inference.md) <br>
- [references/setup.md](references/setup.md) <br>
- [references/datasets.md](references/datasets.md) <br>
- [references/docker.md](references/docker.md) <br>
- [references/eval.md](references/eval.md) <br>
- [references/output-layout.md](references/output-layout.md) <br>
- [references/error-handling.md](references/error-handling.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
1 evaluation task (1 positive) across 3 attempts per task in isolated sandbox pods. Dataset digest: sha256:f16074998b03223a85c3a4a806ad96050ff9fba4c9d9125dd3ab7b9e54d2c850. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was selected and the workflow executed. <br>
- Effectiveness: Checks goal completion (50%) and expected workflow behavior adherence (50%). <br>
- Efficiency: Checks tool-call productivity (50%) and token efficiency (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Verifies no unsafe operations, secret leakage, or unauthorized access. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `skill_execution`: Verifies the expected skill was selected and decoys were avoided. <br>
- `goal_accuracy`: Verifies whether the user's goal was achieved. <br>
- `behavior_check`: Verifies whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Verifies tool-call productivity. <br>
- `token_efficiency`: Verifies actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 94.1% | 79.7% |
| Security | 100.0% → 100.0% (±0.0 pts) | 100.0% → 100.0% (±0.0 pts) |
| Correctness | 0.0% → 100.0% (+100.0 pts) | 20.0% → 100.0% (+80.0 pts) |
| Discoverability | 100.0% | 0.0% |
| Effectiveness | 16.7% → 100.0% (+83.3 pts) | 32.5% → 100.0% (+67.5 pts) |
| Efficiency | 70.6% | 99.7% → 98.4% (-1.3 pts) |

## Skill Version(s): <br>
1.0.1 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
