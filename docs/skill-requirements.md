# Skill requirements: naming + signing

> **Audience:** new skill authors. This doc captures the **must-follow** rules
> that gate a skill's path from PR → merge → signed release. Anything that
> isn't in here is style; anything in here will block.
>
> **TL;DR — two failure surfaces:**
>
> | Surface | Triggered by | What runs the check | Fix in |
> |---|---|---|---|
> | **CI validator** | basic frontmatter presence, body has runnable info, SDK leaks, marketplace paths, `skill_info.yaml` shape | [`scripts/validate-skills.sh`](../scripts/validate-skills.sh) (runs as the `validate-skills` job in [`.gitlab-ci.yml`](../.gitlab-ci.yml) on every MR and on `main`) | this PR |
> | **Signing pipeline** | strict semver, `name == dir`, top-level `tags`, `compatibility`, size, `evals/evals.json` (Tier-3), security scanner, structure/dedup rules | Harbor / signing job after merge (also surfaces in `.skill-eval.yml` Tier-3 stage) | this PR (block before merge) |
>
> The signing pipeline is stricter than the CI validator. A skill can pass
> CI and still get rejected at signing. Read both sections.
>
> **Important — the CI validator does NOT enforce naming shape.** It only
> checks that `name:` and `description:` are present. The `tao-<verb>-<object>`
> shape, length caps, approved verb list, and "drop the word `skill`" are
> **conventions**, not gates. The single hard rule on the naming side is
> `name == directory name (kebab-case)` — and that's the **signer**, not CI.
> Treat the conventions as required anyway: bad names trigger unpredictably
> at runtime, which is worse than a fast CI fail.

## Contents

- 1. Naming
  - 1.1 Hard rules (gated)
  - 1.2 Conventions (not gated — follow anyway)
- 2. Signing requirements (block release)
  - 2.1 Frontmatter strictness
  - 2.2 Size — "Large skill"
  - 2.3 Evals / Tier-3 (AGENT_EVAL)
  - 2.4 Structure / atomicity
  - 2.5 Code integrity (dedup)
  - 2.6 SDK leak
  - 2.7 Security scanner — PII / paths
  - 2.8 Security scanner — credentials / destructive ops / tool permissions
  - Meta-rules
- 3. Reference: `evals.json` example

---

## 1. Naming

The frontmatter `name:` is the skill's **trigger surface** — the agent matches
a user's words against it. Most of the naming guidance below is convention,
not a gate — but bad names trigger unpredictably or collide across teams, so
treat them as required.

### 1.1 Hard rules (gated)

Only two naming checks actually block:

| Rule | Where it fires | What blocks |
|---|---|---|
| `name:` and `description:` must be present in frontmatter | CI validator (`scripts/validate-skills.sh`) | MR pipeline |
| `name:` value must equal the directory name (kebab-case) | Signing pipeline | release |

Everything below in § 1.2 is convention.

### 1.2 Conventions (not gated — follow anyway)

#### Shape

`tao-<verb>-<object-or-outcome>`

- `tao-` prefix **always leads**. The verb comes immediately after the prefix.
- Lowercase, kebab-case, **no underscores**, no filler words (`a`, `an`, `the`).
- **Drop the word "skill"** from the name (it's redundant per Marketing).
- **No personal namespacing in `name:`.** Folder paths may use
  `skills/<author>/...`, but `name:` differentiates by **scope**, not by author
  (e.g. `tao-deploy-edge` vs `tao-deploy-cloud`, not `<author>/deploy`).
- `name:` **must equal the directory name** (the signer also enforces this — see § 2.1).

#### Approved verbs

Recommended (not exhaustive): `train`, `finetune`, `tune`, `deploy`, `run`,
`generate`, `analyze`, `setup`, `call`, `list`, `prepare`, `inspect`,
`audit`, `migrate`, `summarize`, `search`, `query`, `ask`, `ingest`,
`create`, `bootstrap`, `install`, `bump`, `fix`, `review`, `profile`,
`manage`, `scaffold`, `convert`, `mine`, `port`, `route`, `validate`,
`launch`.

Pick a verb a user would actually say. If you can't find one that fits, the
skill probably does too many things — split it. New verbs get added here as
they're used; the list isn't a gate.

#### Length

- **≤ 5 tokens.** (Multi-token network names like `mask-grounding-dino` count
  as one logical noun — they're allowed; the cap is on the post-collapse
  count, e.g. `tao-train-mask-grounding-dino` = 3 logical tokens.)
- **≤ 64 characters.**

#### Outcome over implementation

Pick the user's word, not the internal microservice or wire-protocol name.

| Don't | Do | Why |
|---|---|---|
| `tao-frag` | `tao-generate-frag-report` | "Frag" alone reads like jargon — pair it with a verb. |
| `tao-rt-vlm` | `tao-call-rtvi-vlm` *or* `tao-generate-captions` | The bare microservice name doesn't say what the skill produces. |
| `tao-va-mcp-query` | `tao-query-analytics` | Wire protocol is implementation; analytics is the outcome. |

#### Sibling disambiguation

When two skills touch the same domain, each name has to make its **unique**
job obvious. The verb does the work.

| Backend | Bad | Good |
|---|---|---|
| Embedding-based fusion search | `tao-video-search` | `tao-search-archive` |
| Elasticsearch over metrics | `tao-video-analytics` | `tao-query-analytics` |
| Live VLM inference on pixels | `tao-video-understanding` | `tao-ask-video` |

Test: *if a teammate saw only the name in a list, would they know which one to pick?*

#### Acronyms

Prefer full words unless the acronym is canonical user vocabulary. Keep:
`tao-`, `vlm`, `aoi`, `daft`, `deft`. Expand or rename to the outcome:
`rt-vlm`, `va-mcp`.

Test: *does the acronym appear, unexplained, in a Slack message between two
engineers on the team? If yes, keep. If not, expand.*

#### Examples

Pass: `tao-train-visual-changenet`, `tao-deploy-dino`, `tao-run-automl`,
`tao-finetune-huggingface-model`, `tao-mine-aoi-images`,
`tao-setup-nvidia-gpu-host`.

Fail: `dino` (no verb, no prefix), `visual-changenet-deploy` (verb at end),
`mine-skill` (contains "skill"), `train_changenet` (underscore),
`the-tao-trainer` (filler word), `<author>/deploy` (personal namespace).

---

## 2. Signing requirements (block release)

A skill that passes `scripts/validate-skills.sh` can still fail the signing
pipeline. The checks below run there.

> **What the CI validator does NOT catch — signer-only:**
>
> - `SKILL.md` size ≤ 5000 tokens (§ 2.2)
> - No nested `SKILL.md` (§ 2.4)
> - Internal/structural meta wording (§ 2.4)
> - Content dedup across `SKILL.md` and references (§ 2.5)
> - Security-scanner findings — PII paths, leaked usernames, credential
>   shapes, destructive flags, tool-permission combos (§ 2.7 – § 2.8). The
>   scanner is an LLM and is non-deterministic, which is why it isn't in CI.
>
> Self-check these before opening a release-track PR.

### 2.1 Frontmatter strictness

| Field | Rule |
|---|---|
| `metadata.version` | **Strict quoted semver `"x.y.z"`**. `"0.1"`, `"0.4"`, `"0.1-ea"` all fail. Start at `"0.1.0"`. |
| `metadata.author` | Must be present. |
| `name` | Must equal the directory name (kebab-case). |
| `license` | Must be present (`Apache-2.0` unless your skill has a different license). |
| `tags` | Top-level `tags:` must be present and **non-empty**. |
| `compatibility` | Must be present and **≤ 500 characters**. |
| `description` | **No angle-bracket / XML-like tokens.** A literal `<` (e.g. `FAR < 0.1%`, `<workspace>`) trips "Description contains XML tags." Write "below" / "under" or describe in words. |

The body should also have a top-level heading. `## Instructions`, `## Examples`,
and `## Purpose` are advisory nudges (non-blocking).

> **Note:** [`authoring.md`](authoring.md) tells authors to start `metadata.version`
> at `"0.1"` and treats `tags` / `compatibility` as optional. The signer
> disagrees — see § 4.

### 2.2 Size — "Large skill"

`SKILL.md` must be **≤ 5000 tokens** (≈ chars ÷ 4). Target **≤ ~18,000 chars
(~4,500 tokens)** for margin.

Relocate deep detail (param tables, troubleshooting, spec templates, long
examples) into `references/*.md` and leave a summary + pointer in `SKILL.md`.

### 2.3 Evals / Tier-3 (AGENT_EVAL)

> **Two different "eval" files — do not confuse them:**
>
> | File | Required? | What it drives | What it does |
> |---|---|---|---|
> | `evals/evals.json` (this section) | **Required** for skill signing | Tier-3 **AGENT_EVAL** (routing/plan check) | No-execution: agent reads `SKILL.md` and outlines the documented plan. Cheap. |
> | `eval.config` (at skill root) | **Optional** — opt-in only | TAO **skill-execution-eval** (live run) | Pulls real datasets, runs real `docker run`, measures real metrics. Only onboard if you want live-execution coverage. See [`skill-execution-eval-container.yml`](https://gitlab-master.nvidia.com/nvidia-tao-toolkit/skill-eval) and existing examples in [`skills/models/tao-train-visual-changenet/eval.config`](../skills/models/tao-train-visual-changenet/eval.config) / [`skills/applications/tao-run-deft-aoi/eval.config`](../skills/applications/tao-run-deft-aoi/eval.config). |
>
> Shipping `eval.config` does **not** waive the `evals/evals.json` signing
> requirement. Every skill needs `evals/evals.json`; `eval.config` is a
> separate, optional live-execution layer on top.

**`evals/evals.json` must exist** at the skill root — its presence is what
triggers the Tier-3 stage. The file must be a top-level JSON array.

Each entry needs:

| Field | Required | Notes |
|---|---|---|
| `id` | yes | Stable, kebab-case identifier. |
| `question` | yes | Phrase as a **no-execution routing/plan check** — "identify the skill and outline its documented steps; do NOT run commands, scripts, or web searches." |
| `expected_skill` | yes | Must equal the skill's `name:`. |
| `ground_truth` | yes | One-sentence summary of the expected reasoning. |
| `expected_behavior` | yes | Non-empty list of expected agent actions. |
| `expected_script` | optional | Path to a reference script if applicable; `null` otherwise. |

**Do not truncate or ellipsize entries.** The eval sandbox's agent rejects
the `web_search` tool (HTTP 400) and crashes if the skill drives live
research — phrase every question to forbid execution.

A full example is in [§ 3](#3-reference-evalsjson-example).

### 2.4 Structure / atomicity

- **No nested skills.** A skill directory may not contain another `SKILL.md`.
  Fold any child into `references/<name>.md`.
- **No links to nested-skill paths.** `deploy/SKILL.md` is both a dead link
  and forbidden — point to `references/tao-deploy-<model>.md`.
- **All relative reference links must resolve** to a file that exists.
- **No internal/structural/meta wording** anywhere (`SKILL.md` and references):
  `sub-skill`, `parent skill`, `child skill`, `mother skill`, `folded`,
  `relocated`, `refactor`, `restructure`, `moved from/to`, `originally in
  SKILL.md`. (Allowed: `parent_model` / "Parent Model Inference SDK" terms,
  and plain English like "restructuring".)

### 2.5 Code integrity (dedup)

**No duplicated content** between `SKILL.md` and a reference, or across
references. Move a block to exactly one place; leave a pointer, not a copy.

### 2.6 SDK and optimizer-package leak

No SDK or removed optimizer-package symbols (`tao_sdk`, `tao_automl`,
`AutoMLRunner`, `script_runner`, `sdk.create_job`, `build_entrypoint`, …)
anywhere under `skills/`—SKILL.md or a reference. Platform execution and AutoML
state machines are skill-owned. `scripts/validate-skills.sh` enforces this.

### 2.7 Security scanner — PII / paths (HIGH, blocking)

The detector matches the **path shape**, not the token. Renaming the value
does not help.

| Forbidden shape | Examples that fail |
|---|---|
| `/home/<anything>/` | `/home/ubuntu/`, `/home/user/`, `/home/shadeform/` |
| `/Users/<anything>/` | `/Users/jdoe/`, `/Users/ci/` |
| `.../users/<anything>/` | `lustre/users/$USER/` |

Use an accepted form instead: `~/...`, `$HOME/...`, `/path/to/...`, or a
`<your-dir>`-style placeholder with a one-line gloss ("where `<your-dir>` is
your per-user directory"). Tilde and `$HOME` are confirmed-accepted.

**No leaked usernames anywhere** — `SKILL.md`, references, JSON state files,
scripts, comments, and path slugs. Don't ship runtime artifacts (e.g.
`deft_state.json`) with real paths; scrub to placeholders.

### 2.8 Security scanner — credentials, destructive ops, tool permissions (HIGH, blocking)

- **Sensitive credentials (NGC key, `HF_TOKEN`):** add explicit secure-handling
  guidance — don't inline secrets in generated YAML, prefer `--env-file` /
  secrets over `-e`, rotate after use. (This finding is non-deterministic —
  the same content may pass one run and flag HIGH the next; rerun before
  declaring it a content bug.)
- **Destructive / auto-confirm flags (`--yes`, unattended installs):** keep the
  flag but add a warning disclosing what it auto-confirms and recommending a
  `--check-only` / dry-run first. **Do not bake `--yes` / auto-confirm into
  declared `skill_info.yaml` commands or static examples** — keep declared
  commands interactive and instruct the agent to append the assume-yes flag
  at runtime (skill runs have no TTY).
- **PII assignment shape:** the detector matches the `secret="<…>"` assignment
  shape. Describe credentials / kwargs in **prose**, never as
  `x="<placeholder>"`.
- **Tool permissions (`allowed-tools`):** keep minimal. Don't place `Write`
  adjacent to `Skill` ("Write Skill" reads as "write/modify skills" →
  "Rogue Agent" HIGH). `Write` + `Skill` / `Task` co-presence trips
  privilege-escalation flags. Only request tools the skill truly needs.

### Meta-rules

1. **Structure beats tokens.** For any PII / path / permission flag, change
   the **shape** or use an accepted placeholder — don't just rename the value.
2. **`gate_severity` blocks on two things only:** a HIGH-severity finding **or**
   a failed Tier-3 result. MEDIUM / LOW (phone-number false positives, missing
   `## Examples`, author format, long description) are reported but don't
   block. Fixing the Tier-3 blocker can surface a previously-hidden HIGH on
   the next run.
3. **The security scanner is an LLM and is non-deterministic** — a clean run
   isn't proof; a flaky HIGH can appear on rerun with identical content.
4. **Tier-3 (Harbor) is capacity-limited** — don't trigger many batch
   pipelines at once or you get "Harbor produced no scored trials" (infra
   failure, not content). Stagger reruns.
5. **The Sign stage pushes signatures back to the branch** and needs
   branch-protection / blossom-ci bypass — that's a CI-owner / repo-config
   item, not a content fix.

---

## 3. Reference: `evals.json` example

Save at `skills/<layer>/<skill-name>/evals/evals.json`. Top-level JSON array;
one or more entries.

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

Reference copy in-tree:
[`skills/data/tao-mine-aoi-images/evals/evals.json`](../skills/data/tao-mine-aoi-images/evals/evals.json).

**Question phrasing rule:** every `question` must explicitly forbid
execution ("Do NOT run any commands, scripts, web searches, or other tools —
describe the plan only"). The eval sandbox crashes if the skill drives live
research.
