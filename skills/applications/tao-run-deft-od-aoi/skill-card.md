## Description: <br>
Runs a binary RT-DETR DEFT object-detection loop for industrial inspection with frozen KPI/test roles, SigLIP hard-example retrieval, cumulative COCO admission, KPI-selected adaptive training, and optional AnomalyGenNext synthesis. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>

## Use Case: <br>
Developers and engineers iteratively improving a binary AOI defect detector from normalized real, clean, KPI, and test datasets. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Unverified clean images, KPI leakage, or test-driven selection can inflate apparent quality. <br>
Mitigation: Validate disjoint normalized roles, admit only explicitly verified clean records, select on KPI only, and keep test report-only. <br>

Risk: Synthetic defects may not preserve requested identity or localization. <br>
Mitigation: Require exact FN metadata and pixel masks, verify route identity, and admit only outputs that pass the generation contract. <br>

## Skill Output: <br>
**Output Type(s):** [Shell commands, JSON state, YAML specifications, COCO dataset, Model checkpoints, Analysis] <br>
**Output Format:** [Markdown, JSON, YAML, Parquet, COCO JSON] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`claude-code`) <br>
- Codex (`codex`) <br>

## Evaluation Tasks: <br>
Evaluated against the packaged planning and workflow-disambiguation tasks and runtime smoke tests. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: security, correctness, discoverability, effectiveness, and efficiency. <br>

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility. Users should review selected data, generated outputs, thresholds, and deployment requirements with their internal teams. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
