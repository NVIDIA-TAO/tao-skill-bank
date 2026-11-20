## Description: <br>
Token-efficient execution kit for repeated TAO agentic workflows that compiles a skill into stage cards once (with a strong model), then executes them in fresh, small headless sessions coordinated by a bash driver, with measured 74-91% token reduction and 4-5x lower peak context versus one long conversation. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers running NVIDIA TAO workflows repeatedly use this skill to compile application skills into stage-card packs and execute them through a bash driver and harness adapters (Claude Code hooks or Pi extensions), reducing token cost and enabling execution on models with small context windows. <br>

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
- [CONTRACTS.md — progress log, STAGE_DONE, commands.log, env contract](references/CONTRACTS.md) <br>
- [MEASUREMENT.md — long-conversation vs card measurement protocol, thresholds, provenance](references/MEASUREMENT.md) <br>
- [authoring_prompt.md — card compilation prompt (round 0)](references/authoring_prompt.md) <br>
- [authoring_prompt_v2.md — card authoring rules from measured small-model failures](references/authoring_prompt_v2.md) <br>
- [CARD_AUTHORING.md — annotated real authoring session](references/CARD_AUTHORING.md) <br>
- [Claude Code adapter README](adapters/claude-code/README.md) <br>
- [Pi adapter README](adapters/pi/README.md) <br>
- [Pi coding agent](https://pi.dev) <br>
- [TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Files, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown stage cards, bash driver scripts, on-disk state files (progress log, STAGE_DONE, commands.log), and per-session token accounting reports] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Driver halts on committed errors with no auto-retry and exits 0 only after verified completion; starting a pack driver requires the tao-launch-workflow confirmation gate.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
2 evaluation tasks (2 positive: routing and run-pack cases), 1 attempt per task, each attempt in its own isolated sandbox pod (`k8s-sandbox` environment); evaluator version 1.6.10, evaluated 2026-11-20. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the `security` signal (100%). <br>
- Correctness: Is the answer correct? Scored from the `accuracy` signal (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from the `skill_execution` signal (100%). <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` (50%) and `behavior_check` (50%). <br>
- Efficiency: Did it avoid wasted tool calls and token usage? `skill_efficiency` (50%) plus `token_efficiency` (50%). <br>

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
| Overall | 82.6% — baseline ran, but no comparable score was available; uplift unavailable | 52.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 10.0% → 90.0% (+80.0 points) | 0.0% → 50.0% (+50.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 0.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 10.0% → 60.0% (+50.0 points) | 10.0% → 20.0% (+10.0 points) |
| Efficiency | 63.0% — baseline ran, but no comparable score was available; uplift unavailable | 93.7% → 93.6% (-0.1 points) |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.4% → 93.5% (-4.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 24 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 2 task(s) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
