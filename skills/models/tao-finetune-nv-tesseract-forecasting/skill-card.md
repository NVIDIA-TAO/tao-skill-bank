## Description: <br>
NV-Tesseract Forecasting — transformer-based multivariate time series forecasting with DARR (context-enhanced kNN retrieval), interpretability, and fine-tuning. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers who need to perform multivariate time series forecasting, fine-tuning, and interpretability analysis using the NV-Tesseract model. <br>

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
- [NV-Tesseract GitHub Repository](https://github.com/NVIDIA/NV-Tesseract) <br>
- [Forecasting SDK README](https://github.com/NVIDIA/NV-Tesseract/blob/main/forecasting/README.md) <br>
- [Pretrained Weights (Hugging Face)](https://huggingface.co/nvidia/nv-tesseract-forecasting) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands, Files] <br>
**Output Format:** [Markdown with inline Python and bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
1 evaluation task (1 positive) run on the trusted local host. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use (unsafe operations, secret leakage, unauthorized access). <br>
- Correctness: Whether the answer produced is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user’s goal and expected workflow (goal_accuracy 50% + behavior_check 50%). <br>
- Efficiency: Whether the skill avoided wasted tool or skill usage (routing quality, workspace-aware reads, productive tool use). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user’s goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 52% → 97% (+45 points) | 54% → 51% (-3 points) |
| Security | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Correctness | 40% → 100% (+60 points) | 100% → 100% (±0 points) |
| Discoverability | 50% → 100% (+50 points) | 0% → 0% (±0 points) |
| Effectiveness | 27% → 100% (+73 points) | 68% → 53% (-15 points) |
| Efficiency | 41% → 83% (+42 points) | 0% → 0% (±0 points) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
