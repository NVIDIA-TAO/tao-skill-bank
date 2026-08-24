## Description: <br>
NV-Tesseract AD Diffusion — diffusion-based anomaly detection and fine-tuning for multivariate time series. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to fine-tune and run inference with NV-Tesseract AD Diffusion for diffusion-based anomaly detection on multivariate time series data. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Optional] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [NV-Tesseract Source Repository](https://github.com/NVIDIA/NV-Tesseract) <br>
- [AD Diffusion README (SDK reference, architecture, API docs)](https://github.com/NVIDIA/NV-Tesseract/blob/main/ad_diffusion/README.md) <br>
- [Dataset Format and CSV Conventions](https://github.com/NVIDIA/NV-Tesseract/blob/main/ad_diffusion/examples/datasets/README.md) <br>
- [Pretrained Weights (Hugging Face)](https://huggingface.co/nvidia/nv-tesseract-ad-diffusion) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Code, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash and Python code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
1 evaluation task (1 positive) — skill identification and workflow outline for fine-tuning and inference with NV-Tesseract AD Diffusion. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks whether the answer is correct against the reference answer. <br>
- Discoverability: Checks whether the right skill was found and executed when needed. <br>
- Effectiveness: Checks whether the skill helped complete the user’s goal and expected workflow. <br>
- Efficiency: Checks for routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Verifies absence of unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies whether the expected skill was found and executed. <br>
- `skill_efficiency`: Verifies routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Verifies whether the user’s goal was achieved. <br>
- `behavior_check`: Verifies whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 52% → 95% (+43 points) | 43% → 47% (+4 points) |
| Security | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Correctness | 40% → 100% (+60 points) | 60% → 80% (+20 points) |
| Discoverability | 50% → 100% (+50 points) | 0% → 0% (±0 points) |
| Effectiveness | 37% → 100% (+63 points) | 53% → 53% (±0 points) |
| Efficiency | 35% → 75% (+40 points) | 0% → 0% (±0 points) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
