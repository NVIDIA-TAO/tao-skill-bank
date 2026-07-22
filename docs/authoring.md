# Authoring a new skill

## Contents

- 0. Decide the layer
- 1. Minimum viable skill
- 2. Frontmatter
  - Required fields
  - Optional fields (validator warns when missing)
  - Body must be agent-runnable
- 3. Body structure (DAFT-style)
- 4. When to add `references/`
  - Version references
  - Skills that require a Python wheel
  - `references/skill_info.yaml` schema
- 5. Optional: `example/` reference output
- 5b. Required: `evals/evals.json` (Tier-3 signing)
- 6. Templates
- 7. Add to `marketplace.json`
- 8. Validate
- 9. Test locally
- Checklist
- Common pitfalls
- Agent identity (cross-cutting)


The minimum viable skill is a single `SKILL.md`. Everything else — `references/skill_info.yaml`, `scripts/`, `example/`, templates, defaults — is optional and only added when it earns its keep.

## 0. Decide the layer

| Layer | Use when the skill is… | Examples |
|---|---|---|
| `skills/models/` | A trainable network with `train` / `evaluate` / `inference` / `export` actions | `tao-finetune-cosmos-reason`, `tao-train-visual-changenet`, `tao-finetune-clip` |
| `skills/data/` | A data transformation — preparation, analysis, embedding, filtering | `omniverse-sdg`, `mining`, `changenet-data-prepare` |
| `skills/platform/` | A compute backend or runtime convention (where/how jobs run) | `tao-run-on-docker`, `tao-run-on-brev`, `tao-run-on-slurm`, `tao-run-on-kubernetes`, `tao-data-io` |
| `skills/applications/` | A workflow composing multiple skills (orchestrator) | `tao-run-deft-aoi`, `tao-train-single-step` |

If you're unsure: produces a trained model artifact → model. Transforms data → data. Infrastructure → platform. Orchestrates → application.

## 1. Minimum viable skill

```
tao-train-my-network/
└── SKILL.md
```

That's it. The body must contain enough info for an agent to run the skill — typically a Quick Start section that shows the literal `docker run` (or `python script.py`) line. No `references/`, no `scripts/`, no JSON files required.

## 2. Frontmatter

YAML between `---` markers at the top of `SKILL.md`.

All fields below are **required**. The CI validator (`scripts/validate-skills.sh`) and the signing pipeline both enforce them — see [`skill-requirements.md`](skill-requirements.md) for the full gate list.

### Required fields

```yaml
---
name: tao-<verb>-<object>     # kebab-case; product prefix `tao-`, verb after the prefix.
                              # Recommended verbs include: train, deploy, finetune, tune, generate,
                              # run, analyze, setup, call, list, prepare, convert, mine, port,
                              # route, validate. Drop the word "skill" from the name (it is
                              # redundant). ≤ 5 tokens, ≤ 64 chars.
                              # MUST equal the directory name (kebab-case) — the validator
                              # and the signer both enforce this.
description: >-               # 1-3 sentences. Include literal trigger phrases.
  What the skill does and when to use it. Use when the user asks to
  "convert annotations to DAFT", "package annotations in DAFT format",
  or mentions "Data Factory exports". List every trigger phrase the user
  is likely to say.
                              # No literal `<` / XML-like tokens (the signer rejects
                              # them as "Description contains XML tags"). Use "below"
                              # or "under" instead of `<`.
license: Apache-2.0           # Apache-2.0 unless your skill has a different license
compatibility: Requires docker + nvidia-container-toolkit + NGC API key.
                              # Runtime requirements only. ≤ 500 characters.
metadata:
  author: NVIDIA Corporation  # MUST be exactly this string
  version: "0.1.0"            # Strict semver "x.y.z". "0.1", "0.4", "0.1-ea" all fail.
tags:                         # Non-empty list. Short keywords for browsing / tooling.
  - <domain>
  - <task>
---
```

#### Naming conventions

The frontmatter `name:` is the skill's trigger surface — the agent matches the user's
words against it. Follow these rules:

- **Shape:** `tao-<verb>-<object-or-outcome>`. The product prefix `tao-` always
  leads; the verb comes immediately after the prefix.
- **Recommended verbs (not exhaustive):** `train`, `finetune`, `tune`, `deploy`,
  `run`, `generate`, `analyze`, `setup`, `call`, `list`, `prepare`, `inspect`,
  `audit`, `migrate`, `summarize`, `search`, `query`, `ask`, `ingest`,
  `convert`, `mine`, `port`, `route`, `validate`, `launch`. Pick a verb a user
  would actually say. If none fits, the skill probably does too many things —
  split it.
- **Outcome over implementation.** Pick the user's word — "rca" or "frag" alone
  reads like internal jargon; pair it with `analyze-` / `generate-` or expand it.
- **Drop the word "skill"** from the name (it is redundant per Marketing).
- **Length:** ≤ 5 tokens, ≤ 64 characters. Multi-token network names like
  `mask-grounding-dino` count as one logical noun.
- **Lowercase, kebab-case, no underscores, no filler words** (`a`, `an`, `the`).
- **Per-product subscope** (e.g. `tao-daft-` for TAO DAFT, `tao-deft-` for the
  DEFT loop) is fine — `tao-` still leads, then the subscope, then the verb,
  then the object: `tao-convert-dataset-format`, `tao-mine-aoi-images`.
- **No personal namespacing.** Differentiate variants by scope, not by author
  (e.g. `tao-deploy-edge` vs `tao-deploy-cloud`, not `<author>/deploy`).
- **`name:` must equal the directory name** (kebab-case). The validator and
  signer both enforce this.

Examples that pass:
`tao-train-visual-changenet`, `tao-deploy-dino`, `tao-run-automl`,
`tao-run-deft-aoi`, `tao-finetune-huggingface-model`, `tao-run-on-kubernetes`,
`tao-setup-nvidia-gpu-host`, `tao-launch-workflow`, `tao-list-capabilities`.

Examples that fail:
`dino` (no verb, no prefix), `visual-changenet-deploy` (verb at end), `mine-skill`
(contains "skill"), `train_changenet` (underscore), `the-tao-trainer` (filler word).

The validator fails CI when any of the required fields above is missing or malformed (strict semver, non-empty tags, ≤500-char compatibility, no `<` in description, `name == directory name`, etc.).

**Description style guide** — DAFT-influenced:

- Keep the first sentence factual (what the skill does).
- Follow with a *"Use when the user asks to '...', '...', '...'"* clause listing 2-5 literal trigger phrases. This drives auto-invocation; abstract descriptions don't trigger reliably.
- Mention domain terms users actually say ("convert annotations", "fine-tune cosmos-rl", "k-NN mining"). Synonyms help.

### Field notes (required fields above)

**`compatibility:`** — runtime requirements only. Tools, packages, env vars, services the skill needs. **≤ 500 characters** (signer cap).

> **Important:** the skill bank is **agent-harness-agnostic**. Do NOT prefix `compatibility:` with "Designed for <runtime>" or any specific harness — the same skill must work in any Agent Skills compatible agent. Describe runtime requirements only.

| Skill type | Recommended `compatibility:` value |
|---|---|
| Containerized model/data | `Requires docker + nvidia-container-toolkit + NGC API key.` |
| `skills/platform/tao-run-on-docker` | `Requires docker + nvidia-container-toolkit.` |
| `skills/platform/tao-run-on-brev` | `Requires the brev CLI (https://github.com/brevdev/brev-cli) and an active brev login.` |
| `skills/applications/tao-run-automl` | `Requires Python 3.10+, PyYAML, jsonschema, numpy, scipy, and scikit-learn; GEPA and WandB helpers are optional.` |
| Local Python script (no container) | `Requires Python 3.8+ and Pillow.` (or whatever) |
| Agent-prompt-driven | `Standalone — no external runtime requirements.` |

**`metadata.author`** — must be exactly `NVIDIA Corporation`. The validator fails CI on any other value (including personal names or all-caps variants).

**`metadata.version`** — skill version (NOT tool/model version). **Strict semver `"x.y.z"`** — `"0.1"`, `"0.4"`, `"0.1-ea"` all fail at signing. Start at `"0.1.0"`; bump when the SKILL.md materially changes (new actions, schema changes, etc.).

**`tags`** — non-empty list of short keywords for documentation, browsing, and our own catalog tooling. Tags are NOT used for skill auto-invocation — that's driven by `description` (and trigger phrases within it). Tags exist for human browsing and tooling. Lives in `SKILL.md` frontmatter only — `references/skill_info.yaml` does NOT carry tags (single source of truth). Example:

```yaml
tags:
  - pcb
  - aoi
  - defect
  - classification
```

### Optional fields (validator warns when missing)

**`allowed-tools`** — declares frequently used tools for compatible runtimes. Whitespace-separated list. Common values: `Read Bash`, `Read Bash Write`. Use sparingly — only for tools the skill genuinely needs frequently. Keep `Write` separated from `Skill` / `Task` (co-presence trips the signer's privilege-escalation scanner — see [`skill-requirements.md`](skill-requirements.md) § 2.8).

```yaml
allowed-tools: Read Bash
```

### Body must be agent-runnable

Body must contain at least one of:
- A `## Quick Start` (or `## Quick start`) section
- A `docker run` code block
- A `references/skill_info.yaml` file on disk
- A `scripts/` or `hooks/` directory on disk

The validator enforces this.

## 3. Body structure (DAFT-style)

The validator accepts any of the runnable-info markers above, but **for non-trivial skills**, follow this structure:

```
# Skill Name
[2-line summary]

## External dependencies
[table: dependency / purpose / install command]

## Quick start
[multiple example commands: simple / advanced / dry-run]

## CLI Reference
[arg / required / default / description table]

## Output structure
[directory tree showing what gets produced]

## Inputs / Outputs / Credentials
[what the user provides, what the skill emits, env vars needed]

## Instructions   ← especially important for application/workflow skills
[Step 1 — gather inputs. Step 2 — run. Step 3 — handle edge cases. Step 4 — validate.]

## Known pitfalls
[symptom / cause / fix table]
```

Skip sections that don't apply (e.g., a 5-line orchestrator skill doesn't need a CLI Reference table). Keep `SKILL.md` under ~500 lines; move long reference material to `references/`.

## 4. When to add `references/`

Add a `references/` directory when one or more applies:

| Add this file | When |
|---|---|
| `references/skill_info.yaml` | Multi-action skill (train/evaluate/inference) AND/OR the agent needs structured action metadata (image, command, mode, inputs/outputs) to construct the container command. |
| `deploy/skill_info.yaml` | Deploy-only metadata paired with a `deploy/SKILL.md`. It follows the same schema and validator rules as `references/skill_info.yaml`. |
| `references/spec_template_<action>.yaml` | Action takes a config file (YAML/TOML) and you want users to start from a known-good default. |
| `references/scripts/<file>.py` | The skill ships a reference implementation of a script that runs inside the container. |

Pure agent-only skills (e.g., HF model wrappers driven by a single `docker run`) don't need any of these.

### Version references

Skills reference container images and SDK wheel versions through a single canonical file: `versions.yaml` at the bank's repo root. This is the **only** place to bump TAO container tags, IVA images, or SDK wheel versions when an RC ships.

`references/skill_info.yaml` carries a stamped literal for `container_image`,
annotated with the versions.yaml key it is stamped from — skills stay standalone
(no runtime versions.yaml lookup) while release bumps stay one-file edits:

```yaml
container_image: nvcr.io/nvidia/tao/tao-toolkit:7.0.1-pyt  # versions-key: images.tao_toolkit.pyt
```

`scripts/stamp_versions.py` rewrites every annotated line from `versions.yaml`
(CI runs `--check` to reject drift). Two legacy forms also parse:

```yaml
# Legacy: dotted key — resolved against versions.yaml at runtime (pre-standalone skills)
container_image: tao_toolkit.pyt

# Also valid: absolute registry URI (for experimental / third-party / one-off images)
container_image: nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt
```

Use a key when the image is shared across more than one skill or expected to be bumped on a release cadence. Use an absolute URI for experiments or external images not worth promoting to the manifest.

### Version pinning rules

Every image pin in a skill must be one of three things — the versions
check in CI (`scripts/stamp_versions.py --check`) reports anything else as a
stray:

1. **Release-managed pin** — stamped literal + marker on the same line. Use for
   any `nvcr.io/nvidia/tao/*` image:

   ```bash
   container_image: nvcr.io/nvidia/tao/tao-toolkit:7.0.1-pyt  # versions-key: images.tao_toolkit.pyt
   ```

2. **Deliberate one-off** — third-party registry, experimental image, or
   anything not bumped on the TAO release cadence. Annotate it so the stray
   scan skips it, with a reason:

   ```bash
   docker run --rm alpine:3.20 chown -R "$(id -u)" /w  # unpinned: generic helper image, not release-managed
   ```

3. **Variable indirection for multi-line commands** — a `# versions-key:`
   comment cannot sit on a `docker run` continuation line (the `#` would
   swallow the trailing `\`). Define a stamped variable once, then reference
   it:

   ```bash
   TAO_DEPLOY_IMAGE=nvcr.io/nvidia/tao/tao-toolkit:7.0.1-deploy  # versions-key: images.tao_toolkit.deploy

   docker run --gpus all --rm \
     "$TAO_DEPLOY_IMAGE" \
     dino gen_trt_engine -e /specs/spec.yaml
   ```

On a release bump, edit `versions.yaml` and run `scripts/stamp_versions.py` —
never hand-edit a stamped value (CI rejects the drift either way).

The validator enforces:

- A **key reference** must resolve in `versions.yaml`'s `images` tree (else error).
- An **absolute URI** is accepted as-is (no further check).

To add a new image, edit `versions.yaml`:

```yaml
images:
  tao_toolkit:
    pyt:        nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt
    cosmos_rl:  nvcr.io/nvidia/tao/tao-toolkit:6.26.3-cosmos-rl
    # ← add new entries here
```

To bump an RC, change one line — that's the entire diff.

### Skills that require Python helpers

The bank uses ordinary, narrowly scoped Python helpers where an algorithm or
file format warrants one. Declare those packages in `compatibility`, test for
their imports in Preflight, and install them into the active environment when
the workflow permits. Do not add NVIDIA SDK or optimizer wheels: execution and
AutoML state machines must remain in the skill bank.

### `references/skill_info.yaml` schema

```yaml
name: tao-train-my-network                      # follow the kebab + verb-object convention
type: model | data | application | platform     # optional, useful for tooling
required_credentials: [HF_TOKEN, NGC_KEY]

# Models only: exact HF repo IDs for which this skill owns the user-facing
# train/fine-tune workflow. Required when applicable; do not list a dependency
# backbone that this model merely downloads.
huggingface_model_ids: [namespace/repository]

# Models and data skills (containerized) — stamped literal + versions-key marker
container_image: nvcr.io/nvidia/tao/tao-toolkit:7.0.1-pyt  # versions-key: images.tao_toolkit.pyt
actions:
  train:
    command: visual_changenet train -e {config_path}
    config_format: yaml
    mode: config
    inputs:
      dataset.train_csv: { type: file }
    outputs:
      results_dir: { type: folder }
    upload_excludes:
      - inputs/

# Action mode controls how launch tooling serializes the spec:
# - config: write a YAML/TOML/JSON spec file and substitute {config_path}
# - args: build command arguments from actions.<action>.args
# - passthrough: run the command as declared, without generated config or args

# Models only — parallelism wiring (which spec keys carry the GPU/node counts)
gpu_spec_key: train.num_gpus
node_spec_key: train.num_nodes

# Application/workflow skills
stages:
  - { skill: omniverse-sdg, action: generate, condition: always }

# Platform skills
features: [tracking, multi-node, lustre]

tags: [classification, my-domain]
```

## 5. Optional: `example/` reference output

Add an `example/` directory when the skill produces a non-trivial structured output and users benefit from seeing a sample. Keep examples small (KB-scale), strip sensitive content, use synthetic input.

When to add:

- ✅ Skills with multi-file output trees users must conform to.
- ✅ Skills with format-sensitive outputs (specific JSON schemas).
- ❌ Single-file outputs whose schema is documented inline.
- ❌ Very large outputs (don't bloat the repo).

## 5b. Required: `evals/evals.json` (Tier-3 signing)

> **Don't confuse the two eval files.**
>
> | File | Required? | What it drives |
> |---|---|---|
> | `evals/evals.json` (this section) | **Required for signing** | Tier-3 AGENT_EVAL — no-execution routing/plan check. |
> | `eval.config` (at the skill root) | **Optional, opt-in only** | TAO skill-execution-eval — live `docker run`, real datasets, real metrics. Onboard this only if you want live-execution coverage. Examples: [`skills/models/tao-train-visual-changenet/eval.config`](../skills/models/tao-train-visual-changenet/eval.config), [`skills/applications/tao-run-deft-aoi/eval.config`](../skills/applications/tao-run-deft-aoi/eval.config). |
>
> Adding `eval.config` does not waive the `evals/evals.json` requirement.

The signing pipeline runs a Tier-3 agent-eval (AGENT_EVAL) against every skill. Presence of `evals/evals.json` at the skill root is what triggers that stage — skills shipped without it cannot be signed.

```
skills/<layer>/<skill-name>/
├── SKILL.md
└── evals/
    └── evals.json
```

The file is a top-level JSON array. Each entry needs `id`, `question`, `expected_skill` (must equal the skill's `name:`), `ground_truth`, and a non-empty `expected_behavior` list. `expected_script` is optional (`null` if unused).

Phrase the `question` as a **no-execution routing/plan check** — the eval sandbox rejects the `web_search` tool (HTTP 400) and crashes if the skill drives live research.

```json
[
  {
    "id": "tao-mine-aoi-images-basic",
    "question": "A user request: \"Runs the DEFT embed-then-mine workflow for VCN AOI iterations — embeds the gap-analysis target parquet, embeds a source pool, and mines nearest-neighbour source images for downstream augmentation.\" Identify which TAO skill applies and, reading only that skill's documentation, outline the steps it prescribes. Do NOT run any commands, scripts, web searches, or other tools — describe the plan only.",
    "expected_skill": "tao-mine-aoi-images",
    "expected_script": null,
    "ground_truth": "Identify tao-mine-aoi-images as the applicable skill and summarize its documented workflow from SKILL.md without executing anything.",
    "expected_behavior": [
      "Identifies tao-mine-aoi-images as the relevant skill",
      "Outlines the documented workflow steps from SKILL.md",
      "Does not run commands, scripts, or web searches"
    ]
  }
]
```

In-tree reference: [`skills/data/tao-mine-aoi-images/evals/evals.json`](../skills/data/tao-mine-aoi-images/evals/evals.json).

See [`skill-requirements.md`](skill-requirements.md) for the full signing-pipeline rule set (frontmatter strictness, size cap, security scanner, etc.) — those checks gate release independent of CI.

## 6. Templates

Copy a starting point from `templates/skill-skeleton/`:

```bash
cp -r templates/skill-skeleton/minimal skills/models/<your-skill>      # bare SKILL.md only
cp -r templates/skill-skeleton/model   skills/models/<your-skill>      # full DAFT-style scaffolding
cp -r templates/skill-skeleton/data    skills/data/<your-skill>
cp -r templates/skill-skeleton/platform skills/platform/<your-skill>
cp -r templates/skill-skeleton/workflow skills/applications/<your-skill>
```

Rename the directory to your skill's kebab-case name. Fill in the placeholders.

## 7. Add to `marketplace.json`

List your skill under `tao-skills` (the marketplace's main plugin) so it ships with the standard install.

```json
{
  "name": "tao-skill-bank",
  "plugins": [
    {
      "name": "tao-skills",
      "skills": [
        "./skills/models/my-new-network",
        ...
      ]
    }
  ]
}
```

Users install with `/plugin install tao-skills@tao-skill-bank`. The plugin name (`tao-skills`) is what they type; the marketplace name (`tao-skill-bank`) is the source.

Do not also add the skill under `skills/core/`. That directory is only for Codex helper/router skills that generate capability answers or launch intake from the packaged manifests. Mirroring model, data, platform, or application skills under both places gives agents duplicate trigger surfaces and increases the chance of stale or hallucinated routing.

## 8. Validate

```bash
./scripts/validate-skills.sh
```

Errors (fail CI — at parity with the signing pipeline; see [`skill-requirements.md`](skill-requirements.md)):

- `marketplace.json` skill paths must resolve.
- `skills/core/` must not contain symlink mirrors of canonical skills.
- `SKILL.md` frontmatter:
  - `name` present and equal to the directory name (kebab-case).
  - `description` present and contains no `<` / XML-like tokens.
  - `license` present.
  - `compatibility` present and ≤ 500 characters.
  - `metadata.author` present and equal to `NVIDIA Corporation`.
  - `metadata.version` present and strict semver `"x.y.z"` (`"0.1"` fails).
  - `tags` present and non-empty.
- `SKILL.md` body must have runnable info (Quick Start, docker run, scripts/, hooks/, or `references/skill_info.yaml`).
- `evals/evals.json` must exist at the skill root, parse as a non-empty JSON array, and each entry must have `id`, `question`, `expected_skill`, `ground_truth`, and a non-empty `expected_behavior` list.
- No removed SDK or AutoML package symbol leaks anywhere under `skills/`.
- Hook paths in frontmatter must resolve.
- Any `skill_info.yaml` or `model_info.yaml` parses, including `deploy/skill_info.yaml`.
- Container image keys resolve through `versions.yaml` (top-level and action-level).
- Model/data action contracts declare `command`, `mode`, `inputs`, `outputs`, and `upload_excludes`.

Warnings (printed but don't fail CI):

- Missing `allowed-tools`.

CI runs the same script — fix errors before opening a PR; address warnings opportunistically.

## 9. Test locally

Use the applicable local plugin/runtime command for the target environment and
point it at `/path/to/tao-skills-external`.

Start a session, ask the agent to exercise the skill. Verify the agent reads it, constructs a valid invocation, and produces the expected output.

## Checklist

- [ ] Skill directory is kebab-case at the right layer.
- [ ] Frontmatter has `name`, `description` with trigger phrases, `license: Apache-2.0`.
- [ ] Optional: `compatibility`, `metadata.author`, `metadata.version`, `allowed-tools` populated.
- [ ] Body has Quick Start (or scripts/, hooks/, references/skill_info.yaml) — agent-runnable.
- [ ] If the skill is non-trivial: External Dependencies, CLI Reference, Output Structure, Known Pitfalls sections present.
- [ ] If using `skill_info.yaml`: `container_image` set, each model/data action has `command`, `mode`, `inputs`, `outputs`, and `upload_excludes`.
- [ ] If a model skill owns training/fine-tuning for a Hugging Face repo ID: add every exact public ID to `huggingface_model_ids`; do not rely only on prose, tags, or human aliases.
- [ ] Every release-managed image/wheel pin carries a `# versions-key:` marker (or a stamped variable for multi-line commands); one-off images are annotated `# unpinned: <reason>`.
- [ ] SKILL.md carries the standalone breadcrumb under its title (run `tao-setup` first when the session was not plugin-initialized) — copy it from any existing skill or the skeleton.
- [ ] `scripts/stamp_versions.py --check` reports no errors and no new stray-pin warnings from your skill.
- [ ] No removed SDK or AutoML package symbols anywhere under `skills/`.
- [ ] Added to the marketplace manifest under the right plugin(s), when the packaging surface requires it.
- [ ] No mirrored copy or symlink added under `skills/core/`.
- [ ] `scripts/validate-skills.sh` passes (no errors; warnings are informational).
- [ ] Tested locally with the applicable plugin/runtime harness.

## Common pitfalls

**Naming the skill file wrong.** It must be `SKILL.md` (uppercase, exact). Files like `dino.md` or `<skill_name>.md` are NOT discovered as skills by plugin runtimes — they're treated as supporting docs.

**Mentioning the agent harness in `compatibility`.** The skill bank is harness-agnostic. Don't write "Designed for <runtime>." Restrict the `compatibility` field to runtime requirements.

**Abstract description.** "Visual Changenet model" is bad. "Fine-tune Visual ChangeNet for PCB defect detection. Use when the user asks to 'train ChangeNet', 'PCB defect detection', or mentions 'siamese classification'." is good.

**Duplicating docker boilerplate.** If your skill explains `--gpus`, NGC login, or nvidia-container-toolkit, delete it and link to `tao-skill-bank:tao-run-on-docker`.

**Mirroring skills under `skills/core/`.** Keep one canonical skill location under `skills/models/`, `skills/data/`, `skills/platform/`, or `skills/applications/`. The `skills/core/` directory is a Codex helper surface, not a flat copy of the bank.

**Over-long SKILL.md.** Keep it under ~500 lines. Move long reference material to `references/` and link.

**Assuming an NVIDIA Python wheel is available.** Keep execution and AutoML
orchestration in skills. Use only ordinary helper packages declared by the
skill.

**Stale `skill_info.yaml`.** When you change the docker command in `SKILL.md`, update the YAML too, including deploy metadata under `deploy/skill_info.yaml`. The agent reads the YAML to construct the command; if they drift, it builds a stale invocation.

## Agent identity (cross-cutting)

The agent's identity — who it is, the discovery flow, what it must never do — lives in **`AGENTS.md`** at the repo root. This is the cross-runtime instruction-loading file per the [agents.md](https://agents.md/) spec. Compatible runtimes should load the same file from the project root or plugin hook so one file drives all runtimes.

**Edit `AGENTS.md`, not the hooks or plugin manifests.** When you add a new runtime (e.g., once Codex's plugin-bundled `SessionStart` hook is wired up — see [openai/codex#16430](https://github.com/openai/codex/issues/16430)), make it `cat AGENTS.md` from `${<RUNTIME>_PLUGIN_ROOT}/AGENTS.md`. Do not duplicate the prompt inline in a hook or in a plugin manifest's `description` / `longDescription` / `defaultPrompt` field — duplicating means future drift across runtimes.

This is distinct from individual `SKILL.md` files, which describe one skill. `AGENTS.md` is the cross-cutting "what is this agent" prompt.
