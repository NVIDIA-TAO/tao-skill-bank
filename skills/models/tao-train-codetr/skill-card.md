## Description: <br>
Trains, evaluates, and runs inference with NVIDIA TAO Co-DETR (CoDINO) object detection — a DETR-family detector using collaborative hybrid assignment for accuracy-first closed-set detection. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>

## Use Case: <br>
Developers and engineers training, evaluating, or running inference with Co-DETR for object detection, including using it as a high-accuracy teacher to label unlabeled data. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

Risk: Co-DETR may not ship in every TAO PyTorch build, producing a confusing mid-run failure. <br>
Mitigation: The skill requires probing `codetr --help` against the resolved image during Preflight and hard-stopping when absent, rather than substituting another detector. <br>

Risk: When inference output is used as pseudo-labels, those are model predictions rather than ground truth, so a downstream model inherits the teacher's errors. <br>
Mitigation: `inference.conf_threshold` is documented as the primary quality control and output is inspectable as plain KITTI text before any downstream use. <br>

## Skill Output: <br>
**Output Type(s):** [Shell commands, Files] <br>
**Output Format:** [Markdown with inline bash code blocks, KITTI label files] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`claude-code`) <br>
- Codex (`codex`) <br>

## Evaluation Tasks: <br>
Evaluated against 1 evaluation task in the NVSkills-Eval external profile (astra-sandbox environment). <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
