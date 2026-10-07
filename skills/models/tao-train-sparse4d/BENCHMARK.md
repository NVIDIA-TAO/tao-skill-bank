# Evaluation Report

## Maintenance Validation — 2026-10-06, Skill 0.2.0

This update changes workflow guidance, action input contracts, generated schemas/templates, and the evaluation set. The release target is TAO 7.3.0. The NVSkills-Eval results below are historical results for version 0.1.0; they do not certify this revision. The managed validation/signing pipeline must generate current publication evidence.

Local validation:

- 36 Sparse4D CPU tests pass: six schema/template pairs, declared action inputs, evaluation and artifact routing, regeneration against public source dataclasses, packaged-template and co-training fragment merges for ResNet-50 and ResNet-101 in both Core and PyTorch, annotation-free conversion configuration, and the depth-repair CLI with real temporary HDF5/PKL fixtures.
- 10 existing command-hygiene, image-resolution, and repository-URL tests pass (46 tests total).
- Independent no-execution plan evaluation covers backbone selection, released-LTT adaptation with RN101, an unspecified backbone, evaluation without training data, incompatible taxonomy inputs, exact resume, and a one-GPU wiring smoke test. This is a local workflow review, not a fresh NVSkills performance score.
- The same 46 CPU tests and the skill-bank validator also pass after applying the two scoped skill commits (`d2123f3`, `081224b`) to skill-bank `release/7.3.0` at `bea0d96f798380cabf35011850054c3ba238132b`. The release candidate retains that branch's image pins; it does not import the default branch's image selection.

### Local TAO 7.3 runtime validation

Bounded tests ran on one RTX PRO 6000 Blackwell GPU, with existing datasets and checkpoint bundles mounted read-only and each action writing to its own tracked output directory. Images came from the 7.3 release branch:

- PyTorch: `nvcr.io/nvstaging/tao/tao-toolkit-pyt:7.3.0-rc-78-multiarch`, manifest digest `sha256:72e6057cf4351f940d13e12c29320959961a64a271c40e3aa6f93b4cf96ce2d5`.
- Data Services: `nvcr.io/nvstaging/tao/tao-toolkit-ds:7.3.0-rc-75-multiarch`, manifest digest `sha256:12c73e28ff1037e2df80c636fbce051c75525a7dcc42aaaa23fabad3ea4f2d62`.

Passed execution and artifact checks:

- ResNet-50: four FP32 training steps, two BF16 training steps with gradient scrubbing disabled, baseline evaluation, and inference from the new FP32 checkpoint.
- ResNet-101: baseline evaluation, eight steps of 2D-to-3D geometric distillation with the released frozen LTT adapter, and checkpoint continuation from step 8 to step 10. Saved model and optimizer tensors are finite and checkpoint step counters match the requested budgets.
- Data Services teacher-cache preparation and lazy-index generation. A separate eight-step ResNet-101 run consumed these newly generated artifacts with lazy loading enabled. Supplemental observation of the installed training methods recorded seven real-scene pseudo-2D batches and one labeled 3D batch, with finite, nonzero losses for each active route. Standalone evaluation of that fresh checkpoint passed.
- Standalone evaluation and inference use `dataset.test_dataset.ann_file` while the train/validation paths are deliberately unavailable. Both backbone tests preserve their paired seven-class taxonomy, preprocessing, anchors, and temporal dimensions; the two baseline metric configurations differ and their scores are not compared.
- ResNet-50 ONNX export from the new checkpoint, followed by ONNX structural validation. The actual graph image input is `[batch_size, num_cams, 3, 540, 960]`; setting `export.input_height: 544` did not change that signature in the tested exporter. The fresh mixed-training checkpoint also passes finite model/optimizer checks.

**Open release blocker:** annotation-free conversion fails in Data Services `7.3.0-rc-75-multiarch` because its installed `AICityConfig` lacks `load_annotations` (and `fps`). The converter and approved SDU 2.0.2 dependency update are merged and backported to Data Services `release/7.3.0` at `4eceb550b10cad122f6c4e4929c5f51e5e2e968f`, but this image predates that update and contains SDU 1.0.0. Build and pin a newer 7.3 Data Services image, then repeat raw annotation-free conversion and its downstream handoff. Existing compatible PKLs allowed the model tests to proceed; this is not a passing raw-data-to-model end-to-end test.

These are small wiring tests using four labeled training frames, four real-scene frames, and two held-out evaluation frames. They establish local execution, not convergence, production accuracy, multi-GPU behavior, quantization support, TensorRT execution, or numerical parity. Continuation with an extended step budget is not a claim of equivalence to an uninterrupted training schedule. Fresh managed NVSkills evaluation/signing remains required.

The observed issues are reflected in the skill: select release-matched images, inspect JPEG/HDF5 storage across mixed inputs, bind action output paths, use tracked Docker jobs with the submitting identity, and verify actual supervision-route coverage within a bounded smoke budget.

Schema sources: TAO Core `cfa016e1c5314d50d2dbaa272eb0b01403997c63` and TAO Data Services `8a9503287094cf54f32f54d395e088d31d5fad7e`. The additional PyTorch dataclass merges use `2db3de4217b3f25c1d60ae71e6acd320016fdc9e`. Schemas and templates were emitted by `scripts/generate_dataclass_schemas.py` through its `generate_for_model` entry point, using the selected skill's action metadata in a temporary generation directory. The generator limits packaged Sparse4D fields to calibrated multi-camera workflows, preserving the retained dataclass defaults and metadata.

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
