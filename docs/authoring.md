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
  - Skills that require the SDK
  - `references/skill_info.yaml` schema
- 5. Optional: `example/` reference output
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
| `skills/platform/` | A compute backend or runtime convention (where/how jobs run) | `tao-run-on-docker`, `tao-run-on-brev`, `tao-run-on-slurm`, `tao-run-platform` |
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

### Required fields

```yaml
---
name: tao-<verb>-<object>     # kebab-case; product prefix `tao-`, verb after the prefix.
                              # Approved verbs: train, deploy, finetune, tune, generate, run,
                              # analyze, deploy, setup, call, list, prepare. Drop the word
                              # "skill" from the name (it is redundant). ≤ 4 tokens, ≤ 64 chars.
description: >-               # 1-3 sentences. Include literal trigger phrases.
  What the skill does and when to use it. Use when the user asks to
  "convert annotations to DAFT", "package annotations in DAFT format",
  or mentions "Data Factory exports". List every trigger phrase the user
  is likely to say.
license: Apache-2.0           # required — Apache-2.0 unless your skill has a different license
---
```

#### Naming conventions

The frontmatter `name:` is the skill's trigger surface — the agent matches the user's
words against it. Follow these rules:

- **Shape:** `tao-<verb>-<object-or-outcome>`. The product prefix `tao-` always
  leads; the verb comes immediately after the prefix.
- **Approved verbs:** `train`, `finetune`, `tune`, `deploy`, `run`, `generate`,
  `analyze`, `setup`, `call`, `list`, `prepare`, `inspect`, `audit`, `migrate`,
  `summarize`, `search`, `query`, `ask`, `ingest`. If you can't find a verb that
  fits, the skill probably does too many things — split it.
- **Outcome over implementation.** Pick the user's word — "rca" or "frag" alone
  reads like internal jargon; pair it with `analyze-` / `generate-` or expand it.
- **Drop the word "skill"** from the name (it is redundant per Marketing).
- **Length:** ≤ 4 tokens, ≤ 64 characters. Multi-word network names like
  `mask-grounding-dino` count as one logical noun and are allowed.
- **Lowercase, kebab-case, no underscores, no filler words** (`a`, `an`, `the`).
- **Per-product subscope** (e.g. `tao-daft-` for TAO DAFT, `tao-deft-` for the
  DEFT loop) is fine — `tao-` still leads, then the subscope, then the verb,
  then the object: `tao-convert-dataset-format`, `tao-mine-aoi-images`.
- **No personal namespacing.** Differentiate variants by scope, not by author
  (e.g. `tao-deploy-edge` vs `tao-deploy-cloud`, not `<author>/deploy`).

Examples that pass:
`tao-train-visual-changenet`, `tao-deploy-dino`, `tao-run-automl`,
`tao-run-deft-aoi`, `tao-finetune-huggingface-model`, `tao-run-platform`,
`tao-setup-nvidia-gpu-host`, `tao-launch-workflow`, `tao-list-capabilities`.

Examples that fail:
`dino` (no verb, no prefix), `visual-changenet-deploy` (verb at end), `mine-skill`
(contains "skill"), `train_changenet` (underscore), `the-tao-trainer` (filler word).

The validator fails CI when `license` is missing. `name` and `description` follow the standard Agent Skills spec.

**Description style guide** — DAFT-influenced:

- Keep the first sentence factual (what the skill does).
- Follow with a *"Use when the user asks to '...', '...', '...'"* clause listing 2-5 literal trigger phrases. This drives auto-invocation; abstract descriptions don't trigger reliably.
- Mention domain terms users actually say ("convert annotations", "fine-tune cosmos-rl", "k-NN mining"). Synonyms help.

### Optional fields (validator warns when missing)

```yaml
compatibility: Requires docker + nvidia-container-toolkit + NGC API key.
metadata:
  author: NVIDIA Corporation
  version: "1.0"
allowed-tools: Read Bash
```

**`compatibility:`** — runtime requirements only. Tools, packages, env vars, services the skill needs.

> **Important:** the skill bank is **agent-harness-agnostic**. Do NOT prefix `compatibility:` with "Designed for <runtime>" or any specific harness — the same skill must work in any Agent Skills compatible agent. Describe runtime requirements only.

| Skill type | Recommended `compatibility:` value |
|---|---|
| Containerized model/data | `Requires docker + nvidia-container-toolkit + NGC API key.` |
| `skills/platform/tao-run-on-docker` | `Requires docker + nvidia-container-toolkit.` |
| `skills/platform/tao-run-on-brev` | `Requires the brev CLI (https://github.com/brevdev/brev-cli) and an active brev login.` |
| `skills/platform/tao-run-platform` | `Requires Python 3.10+ and the nvidia-tao-sdk package (pip install nvidia-tao-sdk).` |
| Local Python script (no container) | `Requires Python 3.8+ and Pillow.` (or whatever) |
| Agent-prompt-driven | `Standalone — no external runtime requirements.` or omit the field. |

**`metadata.author`** — must be exactly `NVIDIA Corporation`. The validator fails CI on any other value (including personal names or all-caps variants).

**`metadata.version`** — skill version (NOT tool/model version). Start at `"0.1"` for new skills; bump when the SKILL.md materially changes (new actions, schema changes, etc.).

**`allowed-tools`** — declares frequently used tools for compatible runtimes. Whitespace-separated list. Common values: `Read Bash`, `Read Bash Write`. Use sparingly — only for tools the skill genuinely needs frequently.

**`tags`** — list of short keywords for documentation, browsing, and our own catalog tooling. Examples:

```yaml
tags:
  - pcb
  - aoi
  - defect
  - classification
```

Tags are NOT used for skill auto-invocation — that's driven by `description` (and trigger phrases within it). Tags exist for human browsing and tooling. Lives in `SKILL.md` frontmatter only — `references/skill_info.yaml` does NOT carry tags (single source of truth).

### Body must be agent-runnable

Body must contain at least one of:
- A `## Quick Start` (or `## Quick start`) section
- A `docker run` code block
- A `references/skill_info.yaml` file on disk
- A `scripts/` or `hooks/` directory on disk
- An SDK invocation example (`sdk.create_job`, `BrevSDK`, etc.) — for skills like `skills/platform/tao-run-platform`

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
| `references/skill_info.yaml` | Multi-action skill (train/evaluate/inference) AND/OR you want SDK-orchestrated execution. The TAO SDK reads this for structured action metadata. |
| `deploy/skill_info.yaml` | Deploy-only metadata paired with a `deploy/SKILL.md`. It follows the same schema and validator rules as `references/skill_info.yaml`. |
| `references/spec_template_<action>.yaml` | Action takes a config file (YAML/TOML) and you want users to start from a known-good default. |
| `references/scripts/<file>.py` | The skill ships a reference implementation of a script that runs inside the container. |

Pure agent-only skills (e.g., HF model wrappers driven by a single `docker run`) don't need any of these.

### Version references

Skills reference container images and SDK wheel versions through a single canonical file: `versions.yaml` at the bank's repo root. This is the **only** place to bump TAO container tags, IVA images, or SDK wheel versions when an RC ships.

`references/skill_info.yaml` accepts two forms for `container_image`:

```yaml
# Preferred: dotted key — resolved against versions.yaml at runtime
container_image: tao_toolkit.pyt

# Also valid: absolute registry URI (for experimental / third-party / one-off images)
container_image: nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt
```

Use a key when the image is shared across more than one skill or expected to be bumped on a release cadence. Use an absolute URI for experiments or external images not worth promoting to the manifest.

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

### Skills that require the SDK

Most skills run with just docker (no Python SDK). A few skills are SDK-orchestrated by design (e.g., `skills/platform/tao-run-platform` (`tao-run-platform`), `skills/applications/tao-run-automl` (`tao-run-automl`)). These need a **preflight** block at the top of `SKILL.md`:

````markdown
## Preflight

This skill needs the TAO SDK. `nvidia-tao-sdk` is on public PyPI and pinned in `versions.yaml`; Preflight blocks resolve the pin via `scripts/resolve_versions_key.py` (swap `wheels.tao_sdk_brev` for the extra you need — `_docker`, `_slurm`, `_kubernetes`, `_all`):

```bash
PIN=$("${TAO_SKILL_BANK_PATH:?}/scripts/resolve_versions_key.py" wheels.tao_sdk_brev)
python -c "import tao_sdk" 2>/dev/null || {
  echo "MISSING: nvidia-tao-sdk not installed. Run:"
  echo "  pip install \"$PIN\""
  exit 1
}
```

If missing, the agent prompts the user to authorize the install via Bash, then re-runs the preflight before continuing. Never auto-install silently.
````

The preflight is documentation-only; the validator does not enforce it. But every SDK-dependent skill is expected to start with this block so users get a clean install instruction instead of a `ModuleNotFoundError`.

### `references/skill_info.yaml` schema

```yaml
name: tao-train-my-network                      # follow the kebab + verb-object convention
type: model | data | application | platform     # optional, useful for tooling
required_credentials: [HF_TOKEN, NGC_KEY]

# Models and data skills (containerized) — prefer the dotted key form (resolved against versions.yaml)
container_image: tao_toolkit.pyt
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

# Models only — parallelism wiring for SDK orchestration
gpu_spec_key: train.num_gpus
node_spec_key: train.num_nodes

# Application/workflow skills
stages:
  - { skill: omniverse-sdg, action: generate, condition: always }

# Platform skills
sdk_module: tao_sdk.platforms.brev.sdk
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

Errors (fail CI):

- `marketplace.json` skill paths must resolve.
- `skills/core/` must not contain symlink mirrors of canonical skills.
- `SKILL.md` frontmatter must have `name`, `description`, and `license`.
- `SKILL.md` body must have runnable info (Quick Start, docker run, scripts/, hooks/, or `references/skill_info.yaml`).
- No `tao_sdk` symbol leaks into model/data/application skills (skills/platform/* exempt; tao-run-automl exempted as SDK-native workflow).
- Hook paths in frontmatter must resolve.
- Any `skill_info.yaml` or `model_info.yaml` parses, including `deploy/skill_info.yaml`.
- Container image keys resolve through `versions.yaml` (top-level and action-level).
- Model/data action contracts declare `command`, `mode`, `inputs`, `outputs`, and `upload_excludes`.

Warnings (printed but don't fail CI):

- Missing `compatibility`.
- Missing `metadata.author` or `metadata.version`.
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
- [ ] No SDK symbols (`tao_sdk`, `sdk.create_job`, etc.) in model/data/application skills (allowed in `skills/platform/*`).
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

**Assuming the SDK is available.** Write the skill to be runnable with just docker. SDK usage should be in an "Optional: via TAO SDK" section, not the primary path.

**Stale `skill_info.yaml`.** When you change the docker command in `SKILL.md`, update the YAML too, including deploy metadata under `deploy/skill_info.yaml`. The SDK reads the YAML; if they drift, agent and SDK diverge.

## Agent identity (cross-cutting)

The agent's identity — who it is, the discovery flow, what it must never do — lives in **`AGENTS.md`** at the repo root. This is the cross-runtime instruction-loading file per the [agents.md](https://agents.md/) spec. Compatible runtimes should load the same file from the project root or plugin hook so one file drives all runtimes.

**Edit `AGENTS.md`, not the hooks or plugin manifests.** When you add a new runtime (e.g., once Codex's plugin-bundled `SessionStart` hook is wired up — see [openai/codex#16430](https://github.com/openai/codex/issues/16430)), make it `cat AGENTS.md` from `${<RUNTIME>_PLUGIN_ROOT}/AGENTS.md`. Do not duplicate the prompt inline in a hook or in a plugin manifest's `description` / `longDescription` / `defaultPrompt` field — duplicating means future drift across runtimes.

This is distinct from individual `SKILL.md` files, which describe one skill. `AGENTS.md` is the cross-cutting "what is this agent" prompt.
