# Evaluation Report

## Maintenance Validation — 2026-10-06, Skill 0.2.0

This update changes workflow guidance, action input contracts, generated schemas/templates, and the no-execution evaluation set. The NVSkills-Eval results below are historical results for version 0.1.0; they do not certify this revision. The managed validation/signing pipeline must generate current publication evidence.

Local validation:

- 32 Sparse4D CPU tests pass: six schema/template pairs, declared action inputs, evaluation and artifact routing, regeneration against public source dataclasses, co-training fragment merges for ResNet-50 and ResNet-101 in both Core and PyTorch, annotation-free conversion configuration, and the depth-repair CLI with real temporary HDF5/PKL fixtures.
- 10 existing command-hygiene, image-resolution, and repository-URL tests pass (42 tests total).
- Independent no-execution plan evaluation covers backbone selection, released-LTT adaptation with RN101, an unspecified backbone, evaluation without training data, incompatible single-view/taxonomy inputs, exact resume, and a one-GPU wiring smoke test. This is a local workflow review, not a fresh NVSkills performance score.
- No GPU job or full training/accuracy comparison was run for this skill revision. Configuration/CPU checks do not establish container execution, convergence, accuracy equivalence, or support in an older image.

Schema sources: TAO Core `cfa016e1c5314d50d2dbaa272eb0b01403997c63` and TAO Data Services `8a9503287094cf54f32f54d395e088d31d5fad7e`. The additional PyTorch dataclass merges use `2db3de4217b3f25c1d60ae71e6acd320016fdc9e`. Schemas and templates were emitted by `scripts/generate_dataclass_schemas.py` through its `generate_for_model` entry point, using the selected skill's action metadata in a temporary generation directory.

Run the focused tests with compatible source checkouts on `PYTHONPATH` to enable all source-parity checks:

```bash
python -m pytest scripts/tests/test_sparse4d_skill.py \
  scripts/tests/test_skill_command_hygiene.py \
  scripts/tests/test_resolve_tao_image.py \
  scripts/tests/test_repository_urls.py -q -ra
```

Without the source packages, the source-parity tests report skips. The skill bank's validator is authoritative for frontmatter: the generic skill-creator quick validator rejects `compatibility` and `tags`, which this repository requires for signing; retain those fields.

## Historical NVSkills-Eval — Skill 0.1.0

Evaluation of the `tao-train-sparse4d` skill before publication through NVSkills-Eval.

This benchmark summarizes 3-Tier Evaluation from NVSkills-Eval results for the skill. The goal is to document whether the skill is safe, discoverable, effective, and useful for agents before it is published for broader workflow use.

## Evaluation Summary

- Skill: `tao-train-sparse4d`
- Evaluation date: 2026-06-22
- NVSkills-Eval profile: `external`
- Environment: `astra-sandbox`
- Dataset: 1 evaluation tasks
- Attempts per task: 1
- Pass threshold: 50%
- Overall verdict: PASS

## Agents Used

- `claude-code`
- `codex`

## Metrics Used

Reported benchmark dimensions:

- Security: checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access.
- Correctness: checks whether the agent follows the expected workflow and produces the correct final output.
- Discoverability: checks whether the agent loads the skill when relevant and avoids using it when irrelevant.
- Effectiveness: checks whether the agent performs measurably better with the skill than without it.
- Efficiency: checks whether the agent uses fewer tokens and avoids redundant work.

Underlying evaluation signals used in this run:

- `security` (Security): checks for unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): verifies that the agent loaded the expected skill and workflow.
- `skill_efficiency` (Efficiency): checks routing quality, decoy avoidance, and redundant tool usage.
- `accuracy` (Accuracy): grades final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): checks whether the overall user task completed successfully.
- `behavior_check` (Behavior Check): verifies expected behavior steps, including safety expectations.
- `token_efficiency` (Token Efficiency): compares token usage with and without the skill.

## Test Tasks

The benchmark dataset contained 1 evaluation tasks:

- Positive tasks: 1 tasks where the skill was expected to activate.
- Negative tasks: 0 tasks where no skill was expected.
- Unlabeled tasks: 0 tasks where positive/negative intent could not be inferred.

Task composition is derived from the evaluation dataset when possible. Entries with `expected_skill` set are treated as positive skill-activation cases, while entries with `expected_skill: null` are treated as negative activation cases.

## Results

| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | 1 | 100% (+0%) | 100% (+0%) |
| Correctness | 1 | 100% (+100%) | 50% (+50%) |
| Discoverability | 1 | 92% (+92%) | 0% (+0%) |
| Effectiveness | 1 | 100% (+90%) | 88% (+74%) |
| Efficiency | 1 | 79% (+52%) | 28% (-0%) |

Score values show skill-assisted performance. Values in parentheses show uplift versus the no-skill baseline when baseline data is available.

## Tier 1: Static Validation Summary

Tier 1 validation passed with observations. NVSkills-Eval ran 1 checks and found 5 total findings.

Top findings:

- MEDIUM SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/models/tao-train-sparse4d`)
- MEDIUM SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/models/tao-train-sparse4d/SKILL.md`)
- MEDIUM SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/models/tao-train-sparse4d/SKILL.md`)
- LOW SCHEMA/unexpected_file: Unexpected 'schemas' in skill root (`skills/models/tao-train-sparse4d/schemas`)
- LOW SCHEMA/author_format: Author must be of the form 'Name <email@host>' (`skills/models/tao-train-sparse4d/SKILL.md`)

## Tier 2: Deduplication Summary

This tier was not run or did not produce findings in this report.

## Publication Recommendation

The skill is suitable to proceed toward NVSkills-Eval publication based on this benchmark. Skill owners should keep this file with the skill and refresh it when the evaluation dataset, skill behavior, or target agents materially change.
