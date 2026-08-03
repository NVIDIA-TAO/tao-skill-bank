# TAO Skill Bank Agent Performance Review

## Contents

- Scope
- Review Criteria
- Summary
- Changes Made
  - 1. Validate all skill metadata, including deploy metadata
  - 2. Repair invalid deploy YAML
  - 3. Make action serialization explicit
  - 4. Tighten data IO contracts
  - 5. Normalize image references where appropriate
  - 6. Remove stale platform routing language
  - 7. Refresh authoring guidance and templates
  - 8. Improve non-workflow trigger descriptions
  - 9. Remove harness-specific wording from non-workflow skills
  - 10. Remove runtime placeholder from DINO
  - 11. Clean stale removed-platform references outside workflow skills
  - 12. Run the skill-creator re-review across the full bank
  - 13. Refine remaining in-scope SKILL.md bodies
  - 14. Add reference navigation across the bank
- Remaining Recommendations
  - Explicitly excluded or reverted skills
  - Near-limit skills to watch
  - Add navigation to long human-authored references
  - Add UI metadata when plugin surfaces support it
- Validation


Date: 2026-05-29

Branch: `rarunachalam/improve-skill-agent-performance`

Base: `origin/release/7.0.1`

## Scope

Reviewed 84 non-template `SKILL.md` files, 5 skill templates, and all packaged `skill_info.yaml`
metadata under `skills/models/`, `skills/data`, `skills/platform`, `skills/applications`, and helper
`skills/core/`.

Latest scope update: `skills/applications/tao-run-deft-aoi/SKILL.md` and
`skills/data/tao-mine-aoi-images/SKILL.md` are intentionally left untouched. The AOI
mining skill was reverted back to the `release/7.0.1` content. Per follow-up
request, `skills/applications/tao-analyze-changenet-rca/SKILL.md` was also reverted
back to `release/7.0.1`. Remaining workflow, model, data, and platform skills
were refined against the skill-creator guidance.

## Review Criteria

The review used these skill authoring practices as the performance baseline:

- Progressive disclosure: frontmatter triggers stay precise; long details move
  to references that the agent can load only when needed.
- Structured contracts: `skill_info.yaml` is parseable and declares the exact
  command, serialization mode, inputs, outputs, upload exclusions, and image.
- Grounded launch behavior: agents should read structured metadata and platform
  preflight instructions instead of inventing commands, output paths, or
  platform defaults.
- Validation integrity: CI should catch metadata that would make the SDK or an
  agent infer behavior.
- Trigger clarity: descriptions should include literal "Use when..." phrases
  that match user language.
- Harness portability: skill bodies should not assume harness-specific files,
  hooks, tools, or session artifact names.
- Progressive-disclosure hygiene: `SKILL.md` should stay under the 500-line
  target where practical, and long references should be navigable.

## Summary

The syntax-level validator was already passing, but the deeper audit found
several issues that directly increase hallucination risk:

- `deploy/skill_info.yaml` files were not validated at all.
- Four deploy metadata files were invalid YAML.
- Most command actions lacked explicit `mode`, so agents had to infer
  `config`, `args`, or `passthrough` behavior.
- Several data actions omitted `upload_excludes`, leaving IO wrapping less
  deterministic.
- One deploy image was pinned to an old absolute URI despite a matching
  `versions.yaml` key.
- One platform skill still named the removed Lepton SDK as a valid choice.
- Several non-workflow descriptions had weaker trigger phrasing.

This pass fixes the structural issues in model/data/platform metadata and
tooling, refreshes the authoring templates, and now applies the skill-creator
cleanup to all in-scope skills.

After the follow-up skill-creator-guided re-review, all 81 in-scope production
skills have complete trigger coverage, no runtime placeholder markers, no
body-level trigger sections, no tool-harness-specific wording, and no
`SKILL.md` over 500
lines. Remaining findings are confined to the explicitly excluded/reverted
skills.

## Changes Made

### 1. Validate all skill metadata, including deploy metadata

Before:

- `scripts/validate-skills.sh` only walked `references/skill_info.yaml`.
- `models/*/deploy/skill_info.yaml` could be malformed and still pass CI.

Fixed:

- The validator now scans every `skill_info.yaml`, including
  `deploy/skill_info.yaml`.
- It validates top-level and action-level `container_image` keys against
  `versions.yaml`.
- It requires command actions to declare a valid `mode`.
- It requires model/data command actions to declare `inputs`, `outputs`, and
  `upload_excludes`.

Impact:

- Agents and SDK launch helpers no longer have to infer missing action
  serialization behavior.
- Broken deploy metadata fails validation instead of becoming runtime guesswork.

### 2. Repair invalid deploy YAML

The audit found four deploy metadata files that failed YAML parsing because
plain scalar note strings contained unquoted colon patterns:

- `skills/models/tao-train-fast-foundation-stereo/deploy/skill_info.yaml`
- `skills/models/tao-train-mask2former/deploy/skill_info.yaml`
- `skills/models/tao-train-ocrnet/deploy/skill_info.yaml`
- `skills/models/tao-train-oneformer/deploy/skill_info.yaml`

Fixed:

- Quoted the affected note strings and preserved their content.

Impact:

- Deploy metadata can now be parsed reliably by validation, tooling, and agents.

### 3. Make action serialization explicit

Before:

- Existing command actions overwhelmingly depended on implicit config-mode
  behavior.
- Missing `mode` is a high-risk ambiguity because the same launch layer supports
  `config`, `args`, and `passthrough`.

Fixed:

- Added explicit `mode: config` to config-file model/deploy actions.
- Added explicit `mode: passthrough` to GPU-host setup actions.
- Kept existing `mode: args` data actions intact.
- Updated `skills/platform/tao-run-platform/SKILL.md` to treat missing mode as invalid
  metadata instead of defaulting to config.

Impact:

- Agents can select the correct `build_entrypoint(...)` shape from metadata.
- No launch path needs to guess whether to write a spec file, expand args, or
  run a passthrough command.

### 4. Tighten data IO contracts

Before:

- Args-mode data actions had structured `inputs` and `outputs`, but no
  `upload_excludes`.

Fixed:

- Added `upload_excludes: [inputs/]` to:
  - `skills/data/tao-analyze-gaps-vlm-bcq/references/skill_info.yaml`
  - `skills/data/tao-generate-image-grounding/references/skill_info.yaml`
  - `skills/data/tao-generate-referring-expressions/references/skill_info.yaml`
  - `skills/data/tao-generate-video-reasoning-annotations/references/skill_info.yaml`

Impact:

- S3-backed runs have clearer upload behavior and less chance of re-uploading
  staged input trees.

### 5. Normalize image references where appropriate

Before:

- CenterPose deploy metadata pinned
  `nvcr.io/nvidia/tao/tao-toolkit:6.26.3-deploy`, while `versions.yaml` already
  defines `tao_toolkit.deploy`.

Fixed:

- Replaced the CenterPose deploy top-level and action-level image references
  with `tao_toolkit.deploy`.
- Updated `scripts/migrate-to-version-keys.py` to inspect every
  `skill_info.yaml`, including deploy metadata and action-level overrides.

Remaining:

- `skills/models/tao-train-bevfusion/references/skill_info.yaml` still uses
  `nvcr.io/nvidia/tao/tao-toolkit:5.5.0-pyt`. The migration tool reports it as
  intentionally kept because there is no matching `versions.yaml` key. Decide
  whether BevFusion must stay pinned or should get a manifest key.

### 6. Remove stale platform routing language

Before:

- `skills/platform/tao-run-platform/SKILL.md` still listed `LeptonSDK` as a selectable
  backend even though this release branch has removed Lepton support.

Fixed:

- Removed `LeptonSDK` from the platform selection sentence.

Impact:

- Agents are less likely to propose an unavailable backend.

### 7. Refresh authoring guidance and templates

Fixed:

- `docs/authoring.md` now documents:
  - `deploy/skill_info.yaml` as a first-class metadata file.
  - required action `mode`.
  - required `upload_excludes` for model/data actions.
  - the stricter validation expectations.
- Model and data skill skeletons now include `mode` and `upload_excludes`.

Impact:

- New skills are less likely to reintroduce ambiguous metadata.

### 8. Improve non-workflow trigger descriptions

Fixed several non-application descriptions to use the recommended literal
"Use when..." trigger form:

- `skills/data/tao-analyze-gaps-vlm-bcq/SKILL.md`
- `skills/data/paidf-anomalygen/SKILL.md`
- `skills/data/tao-route-visual-changenet-samples/SKILL.md`
- `skills/platform/tao-run-on-docker/SKILL.md`
- `skills/platform/tao-setup-nvidia-gpu-host/SKILL.md`
- `skills/core/tao-list-capabilities/SKILL.md`

Impact:

- Skill auto-invocation has more concrete phrases to match. After this pass,
  every in-scope skill description contains literal `Use when` trigger wording.
  The only remaining trigger misses are the excluded DEFT AOI and AOI mining
  skills.

### 9. Remove harness-specific wording from non-workflow skills

Before:

- Several data skills described `TAO_SKILL_BANK_PATH` as coming from a
  harness-specific session hook.
- Data output trees named a harness-specific session capture file as if every
  runtime produced the same artifact.
- `skills/platform/tao-run-on-brev/SKILL.md` hard-checked a harness-local skill path
  before using the Brev CLI.
- `skills/platform/tao-run-platform/SKILL.md` still described "mode inference" even
  though action `mode` is now required metadata.

Fixed:

- Reworded data setup/output instructions to refer to the installed skill bank
  and optional runtime-dependent session capture.
- Removed remaining harness/delegation wording from the PAIDF AnomalyGen and VCN
  gap-analysis data skills.
- Replaced the Brev harness-local agent-skill check with a portable `brev --help`
  availability check.
- Reworded body-level "Use when" guidance in non-workflow model/platform skills
  so trigger guidance lives in frontmatter descriptions.
- Reframed `build_entrypoint` guidance around declared `actions.<action>.mode`
  rather than inference from missing metadata.

Impact:

- Non-workflow skills are less tied to one harness and give Codex a clearer
  contract to follow.

### 10. Remove runtime placeholder from DINO

Before:

- `skills/models/tao-train-dino/SKILL.md` contained a runtime placeholder about extending
  launcher data-source mapping support.

Fixed:

- Replaced the placeholder with the current supported behavior: provide DINO dataset
  paths explicitly until the launcher advertises first-class support for that
  mapping shape.

Impact:

- Agents no longer see an internal implementation placeholder as a user-flow step.

### 11. Clean stale removed-platform references outside workflow skills

Before:

- `Jenkinsfile.release` still referenced `tao_sdk_lepton` and a stale
  `cosmos_predict_2_5` key that does not match `versions.yaml`.
- `skills/models/tao-train-visual-changenet/eval.slow-manual.config` described a
  Lepton/DGX Cloud manual eval even though this release branch removed Lepton
  support.

Fixed:

- Removed the Lepton SDK patch from `Jenkinsfile.release`.
- Updated the release pipeline variable/key from `COSMOS_PREDICT_2_5_TAG` /
  `cosmos_predict_2_5` to `COSMOS_PREDICT_TAG` / `cosmos_predict`.
- Deleted the stale Visual ChangeNet Lepton slow-manual eval config.

Impact:

- Release/eval surfaces no longer steer agents toward a removed platform.

### 12. Run the skill-creator re-review across the full bank

The follow-up audit checked the skill bank against the explicit skill-creator
rubric:

- `SKILL.md` inventory: 89 total, 84 production/non-template, 5 templates.
- Required `name` and `description` frontmatter: 89/89 present and parseable.
- In-scope literal description `Use when` trigger coverage: 81/81.
- In-scope body-level `When to Use` / `Use when` trigger sections: 0.
- In-scope tool-harness/provider-specific wording: 0.
- In-scope `SKILL.md` files over 500 lines: 0.
- Runtime placeholder markers: 0.
- `skill_info.yaml` parse errors: 0.
- Model/data/platform command action `mode` coverage: 225/225.

Explicit exclusions:

- `skills/applications/tao-run-deft-aoi/SKILL.md`
- `skills/data/tao-mine-aoi-images/SKILL.md`
- `skills/applications/tao-analyze-changenet-rca/SKILL.md` (reverted by request)

Local schema exception:

- The upstream skill-creator guide recommends only `name` and `description` in
  frontmatter. This repo intentionally carries additional packaging metadata
  (`license`, `compatibility`, `metadata`, `allowed-tools`, `tags`) on every
  skill and template. Those fields were not removed because existing TAO
  packaging/validation expects them.

### 13. Refine remaining in-scope SKILL.md bodies

Fixed workflow/example and long model/platform skill-body issues that were
previously report-only:

- Added literal `Use when` frontmatter triggers to the four HuggingFace rerun
  example skills and `skills/applications/tao-run-automl-deft-pipeline/SKILL.md`.
- Removed hard-coded web-fetch tool, harness prompt, delegation, and
  harness-cache wording from in-scope workflow skills.
- Renamed body-level trigger guidance so trigger routing lives in frontmatter.
- Rewrote `skills/applications/tao-run-automl/SKILL.md` from a 1182-line reference dump
  into a compact workflow coordinator with explicit preflight, model support
  gates, algorithm policy, nested-spec guidance, metrics, monitoring, and
  result handoff.
- Split the large preservation dumps for substantial rewrites into focused
  one-level references, with `detailed-guide.md` / `detailed-workflow.md` kept
  as small maps only. This preserves the original operational detail without
  forcing agents to load a giant archive file.
- Added compact contents blocks to long human-authored Markdown references
  outside the explicitly excluded/reverted areas. Reference files also state
  that current `SKILL.md`, `skill_info.yaml`, schemas, and model/platform
  skills win on conflicts.
- Compressed duplicated reference-level detail in
  `skills/applications/tao-finetune-huggingface-model/SKILL.md` to 496 lines while
  preserving gates and reference routing.
- Reduced long in-scope model/platform skills under the target:
  `skills/models/tao-train-dino/SKILL.md` 488 lines,
  `skills/models/tao-finetune-cosmos-reason/SKILL.md` 498 lines, and
  `skills/platform/tao-run-on-slurm/SKILL.md` 497 lines.

### 14. Add reference navigation across the bank

The full Markdown follow-up treated long reference files as agent-facing skill
surfaces. Fixed:

- Converted the five large preservation references into map files and split
  their detailed logic into 16 focused one-level references.
- Added navigation blocks to 46 pre-existing long Markdown files that lacked
  early contents/reference-map sections.
- Added conflict-precedence wording to long `references/*.md` files so agents
  prefer current skill bodies, structured metadata, schemas, and platform/model
  skills over stale examples.
- Kept the explicitly excluded/reverted areas unchanged while still reporting
  their residual findings below.

## Remaining Recommendations

### Explicitly excluded or reverted skills

These files still contain guideline findings, but are intentionally unchanged:

- `skills/applications/tao-run-deft-aoi/SKILL.md`: missing literal `Use when`, has
  body trigger guidance, harness-specific wording, and is over 500 lines.
- `skills/data/tao-mine-aoi-images/SKILL.md`: missing literal `Use when` and contains
  harness-specific wording.
- `skills/applications/tao-analyze-changenet-rca/SKILL.md`: reverted by request; it
  has harness-specific wording and is over 500 lines in the base version.

### Near-limit skills to watch

No in-scope production `SKILL.md` is over 500 lines after this pass. Near-limit
skills that should be split before adding more detail:

- `skills/models/tao-finetune-cosmos-reason/SKILL.md`: 498 lines.
- `skills/platform/tao-run-on-slurm/SKILL.md`: 497 lines.
- `skills/applications/tao-finetune-huggingface-model/SKILL.md`: 496 lines.
- `skills/models/tao-train-depth-anything-v2/SKILL.md`: 492 lines.
- `skills/models/tao-train-visual-changenet/SKILL.md`: 490 lines.
- `skills/data/paidf-anomalygen/SKILL.md`: 490 lines.
- `skills/models/tao-train-dino/SKILL.md`: 488 lines.
- `skills/models/tao-train-fast-foundation-stereo/SKILL.md`: 476 lines.
- `skills/platform/tao-run-platform/SKILL.md`: 473 lines.
- `skills/applications/tao-port-huggingface-model/SKILL.md`: 470 lines.

Recommended split pattern:

- Keep quick start, critical overrides, action map, and known hard failures in
  `SKILL.md`.
- Move long recipes, variant matrices, and historical debugging notes to
  one-level `references/*.md` files.
- Add a short reference index with exact loading conditions.

### Maintain navigation on new references

The follow-up pass fixed current long human-authored Markdown outside the
explicitly excluded/reverted areas. Future edits should keep the same pattern:
long references need an early contents block, and generated YAML/spec files
should not receive Markdown TOCs.

### Add UI metadata when plugin surfaces support it

No skill currently has `agents/openai.yaml`. The skill-creator guidance
recommends it for UI-facing display names, short descriptions, and default
prompts. This is not required for current validation, but it would improve
human browsing and future connector surfaces.

Recommended approach:

- Generate `agents/openai.yaml` only after agreeing on naming style and whether
  all 84 production skills should get it at once.
- Keep values derived from `SKILL.md` frontmatter to avoid a second stale
  trigger surface.

## Validation

Commands run:

```bash
./scripts/validate-skills.sh
python3 scripts/migrate-to-version-keys.py
python3 -m py_compile scripts/*.py
git diff --check
custom skill-creator / full-Markdown audit
```

Results:

- `validate-skills` passed.
- The skill-creator audit found 81/81 in-scope production skills with literal
  `Use when`, 0 in-scope body trigger sections, 0 in-scope harness-specific
  wording hits, 0 in-scope `SKILL.md` files over 500 lines, and 0 runtime
  placeholder markers.
- The refactored-Markdown audit found 5/5 large detailed references converted
  to map files, 16/16 split references directly linked from their parent
  `SKILL.md`, no refactored Markdown file over 500 lines, and no
  harness-specific wording or runtime placeholder markers in the refactored
  Markdown set.
- The full Markdown navigation audit found 62/62 long non-SKILL Markdown files
  with an early `Contents` or `Reference Map` section outside the explicitly
  excluded/reverted areas, and no hard-coded harness/tool names or runtime
  placeholder markers in non-excluded Markdown.
- The remaining findings are the explicitly excluded/reverted
  `skills/applications/tao-run-deft-aoi/SKILL.md` and
  `skills/data/tao-mine-aoi-images/SKILL.md`, plus reverted
  `skills/applications/tao-analyze-changenet-rca/SKILL.md`.
- Model/data/platform command actions now have `mode` coverage of 225/225.
- Model/data command actions now have `upload_excludes` coverage of 221/221.
- The migration dry-run reported zero applicable migrations and one intentionally
  kept absolute image path for BevFusion.
- `py_compile` and `git diff --check` passed.
