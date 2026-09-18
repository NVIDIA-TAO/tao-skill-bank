# Skill Benchmark: tao-run-deft-pas

> ❌ **Overall verdict: FAIL — Publication blocked**

The skill should be reviewed before publication. Address the blocking findings below, then rerun Skill Evaluator.

## Evaluation Metadata

- Skill: `tao-run-deft-pas`
- Evaluation date: 2026-09-18
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 14 evaluation tasks (14 positive)
- Dataset digest: `sha256:40d7b333cefc143c30dd17c4ffbefe571c121a4003d1bf967029bf231d1321d9` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 3
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 78.0% — baseline ran, but no comparable score was available; uplift unavailable | 63.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 24.4% → 81.1% (+56.7 points) | 33.9% → 70.9% (+37.0 points) |
| Discoverability | 45.6% — baseline ran, but no comparable score was available; uplift unavailable | 3.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 22.8% → 65.4% (+42.6 points) | 25.1% → 45.6% (+20.5 points) |
| Efficiency | 97.9% — baseline ran, but no comparable score was available; uplift unavailable | 99.5% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 977,870 | 991,740 | N/A | N/A | skill 18/18; base 32/32 |
| claude-code | tao-run-deft-pas-attached-loop-liveness | 32,255 | 31,824 | +431 | +1.35% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-pas-basic | 70,284 | 89,362 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-run-deft-pas-bounded-nested-discovery | 71,724 | 101,369 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-run-deft-pas-clean-intake | 135,105 | 93,186 | +41,919 | +44.98% | skill 3/3; base 3/3 |
| claude-code | tao-run-deft-pas-discovery-provenance | 123,476 | 100,117 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-run-deft-pas-explicit-root-ambiguity | 72,682 | 30,787 | +41,895 | +136.08% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-pas-history-recovery-plan | 70,425 | 32,069 | +38,356 | +119.60% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-pas-metric-contract | 71,679 | 34,272 | +37,407 | +109.15% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-pas-plain-language-routing | 29,319 | 86,999 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-run-deft-pas-plain-language-routing-kpi-paraphrase | 29,350 | 87,090 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-run-deft-pas-plain-language-routing-no-model-name | 29,312 | 86,976 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-run-deft-pas-plain-language-routing-stage-paraphrase | 29,347 | 87,081 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-run-deft-pas-resume-plan | 69,226 | 96,158 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-run-deft-pas-virtualenv-contract | 143,686 | 34,450 | N/A | N/A | skill 3/3; base 1/1 |
| codex | All cases | 328,379 | 450,310 | N/A | N/A | skill 22/22; base 33/33 |
| codex | tao-run-deft-pas-attached-loop-liveness | 14,095 | 13,534 | +561 | +4.15% | skill 1/1; base 1/1 |
| codex | tao-run-deft-pas-basic | 13,797 | 40,589 | N/A | N/A | skill 1/1; base 3/3 |
| codex | tao-run-deft-pas-bounded-nested-discovery | 43,071 | 41,727 | +1,344 | +3.22% | skill 3/3; base 3/3 |
| codex | tao-run-deft-pas-clean-intake | 41,820 | 40,774 | +1,046 | +2.57% | skill 3/3; base 3/3 |
| codex | tao-run-deft-pas-discovery-provenance | 43,113 | 41,805 | +1,308 | +3.13% | skill 3/3; base 3/3 |
| codex | tao-run-deft-pas-explicit-root-ambiguity | 14,023 | 27,319 | N/A | N/A | skill 1/1; base 2/2 |
| codex | tao-run-deft-pas-history-recovery-plan | 14,227 | 13,762 | +465 | +3.38% | skill 1/1; base 1/1 |
| codex | tao-run-deft-pas-metric-contract | 15,800 | 14,604 | +1,196 | +8.19% | skill 1/1; base 1/1 |
| codex | tao-run-deft-pas-plain-language-routing | 13,466 | 39,666 | N/A | N/A | skill 1/1; base 3/3 |
| codex | tao-run-deft-pas-plain-language-routing-kpi-paraphrase | 13,524 | 39,867 | N/A | N/A | skill 1/1; base 3/3 |
| codex | tao-run-deft-pas-plain-language-routing-no-model-name | 13,477 | 39,779 | N/A | N/A | skill 1/1; base 3/3 |
| codex | tao-run-deft-pas-plain-language-routing-stage-paraphrase | 13,477 | 39,918 | N/A | N/A | skill 1/1; base 3/3 |
| codex | tao-run-deft-pas-resume-plan | 42,766 | 14,150 | N/A | N/A | skill 3/3; base 1/1 |
| codex | tao-run-deft-pas-virtualenv-contract | 31,723 | 42,816 | N/A | N/A | skill 1/1; base 3/3 |
| ALL AGENTS | Dataset aggregate | 1,306,249 | 1,442,050 | N/A | N/A | skill 40/40; base 65/65 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **FAILED** | 11 validator(s); 128 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 4 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 14 task(s) |

## Blocking Findings

- **MEDIUM** BANDIT/B614:pytorch_load: Use of unsafe PyTorch load (CWE-502) (`skills/applications/tao-run-deft-pas/scripts/pas_deft/utils.py:652`)
- **MEDIUM** BANDIT/B614:pytorch_load: Use of unsafe PyTorch load (CWE-502) (`skills/applications/tao-run-deft-pas/scripts/pas_deft/utils.py:658`)
- **MEDIUM** BANDIT/B108:hardcoded_tmp_directory: Probable insecure usage of temp file/directory. (CWE-377) (`skills/applications/tao-run-deft-pas/scripts/run_deft_action.py:333`)

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across scripts/init_deft_state.py and scripts/prepare_deft_config.py and scripts/run_deft_container.py:
  "_workspace_child()" in scripts/init_deft_state.py (lines 123-131)
  vs "_workspace_child()" in scripts/prepare_deft_config.py (lines 74-82)
  vs "_workspace_child()" in scripts/run_deft_container.py (lines 122-130) (`scripts/init_deft_state.py:123`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across scripts/deft_action_contract.py and scripts/run_deft_container.py:
  "launch_label()" in scripts/deft_action_contract.py (lines 263-275)
  vs "_launch_label()" in scripts/run_deft_container.py (lines 407-419) (`scripts/deft_action_contract.py:263`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across scripts/audit_deft_run.py and scripts/commit_stage.py:
  "_results_root_for_scope()" in scripts/audit_deft_run.py (lines 607-613)
  vs "_results_root_for_scope()" in scripts/commit_stage.py (lines 347-353) (`scripts/audit_deft_run.py:607`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across scripts/deft_action_contract.py and scripts/run_deft_container.py:
  "fresh_output_path()" in scripts/deft_action_contract.py (lines 125-141)
  vs "_fresh_output_path()" in scripts/run_deft_container.py (lines 82-103) (`scripts/deft_action_contract.py:125`)
- **MEDIUM** BANDIT/B614:pytorch_load: Use of unsafe PyTorch load (CWE-502) (`skills/applications/tao-run-deft-pas/scripts/pas_deft/utils.py:652`)
- 127 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
