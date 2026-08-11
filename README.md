# NVIDIA [TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skills-bank)

Portable agent skills for training, evaluating, and running inference on NVIDIA TAO models. Works with Claude Code, Codex, Gemini CLI, or any coding agent that speaks the [Agent Skills open standard](https://agentskills.io). Most local Docker model/data actions need only Docker plus NVIDIA Container Toolkit; application workflows may declare small host-Python requirements for deterministic state and analysis adapters. Advanced features—job tracking, multi-node, S3 I/O—are built in, not bolted on: platform skills implement a four-verb execution contract over their native CLI with no `nvidia-tao-sdk`. See [Execution: no SDK required](#execution-no-sdk-required).

## Install

The skill bank works with both Claude Code and Codex. Pick the runtime you use.

### Claude Code

In a Claude Code session, add the marketplace and install the plugin:

```
/plugin marketplace add git@github.com:NVIDIA-TAO/tao-skills-bank.git
/plugin install tao-skills@tao-skill-bank
```

That's it — no `git clone`, no `pip install`. The TAO Skill Bank plugin bundles all skills (every model, data, platform, and application). The plugin's [`SessionStart`](hooks/session_start.sh) hook loads the [`AGENTS.md`](AGENTS.md) identity at the start of every session.

### Codex

Codex setup has **two independent pieces** — the plugin (which surfaces the skills to Codex) and `AGENTS.md` (which loads the agent identity). You need both for parity with Claude Code.

#### One command (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/NVIDIA-TAO/tao-skills-bank/main/scripts/install-codex-agents.sh | bash
```

…or, if you've already cloned or extracted the repo from a zip, run
`scripts/install-codex-agents.sh` from that directory. The script registers the
marketplace, installs the TAO Skill Bank plugin, and copies `AGENTS.md` to
`~/.codex/AGENTS.md` so the TAO identity loads in every Codex session. It's
idempotent and backs up any existing `~/.codex/AGENTS.md` before overwriting.
Override the source with `TAO_SKILL_BANK_MARKETPLACE=…` and
`TAO_SKILL_BANK_REF=…` to use a fork, pinned ref, or local absolute path:

```bash
cd /absolute/path/to/tao-skills-external
TAO_SKILL_BANK_MARKETPLACE=/absolute/path/to/tao-skills-external \
  scripts/install-codex-agents.sh
```

#### Manual steps

If you'd rather drive each step yourself:

**1. Install the plugin.** Either use the VS Code Codex extension's plugin UI (select **TAO Skill Bank**), or from the CLI:

```bash
codex plugin marketplace add git@github.com:NVIDIA-TAO/tao-skills-bank.git
codex plugin add tao-skill-bank@tao-local-plugins
```

This installs the bundle to `~/.codex/plugins/cache/tao-local-plugins/tao-skill-bank/<version>/` (the `tao-local-plugins` segment comes from the `name` field in `.agents/plugins/marketplace.json`).

For a local zip or clone, use the absolute path instead of the Git URL:

```bash
codex plugin marketplace add /absolute/path/to/tao-skills-external
codex plugin add tao-skill-bank@tao-local-plugins
```

**2. Load the agent identity (`AGENTS.md`).** The plugin install does **not** auto-load [`AGENTS.md`](AGENTS.md) — Codex's `AGENTS.md` discovery walks down from the project root, not into the plugin cache (see [openai/codex#16430](https://github.com/openai/codex/issues/16430) for why plugin-bundled `SessionStart` hooks don't fix this yet). Pick one:

- **Per-project**: `git clone` this repo and launch `codex` from inside the clone. Codex auto-loads `AGENTS.md` from the project root per the [agents.md](https://agents.md/) cross-runtime spec.
- **Globally** (one-time copy): `cp ~/.codex/plugins/cache/tao-local-plugins/tao-skill-bank/<version>/AGENTS.md ~/.codex/AGENTS.md`. The identity then loads in every Codex session, anywhere.

Once Codex starts honoring plugin-bundled hooks, the identity will install automatically alongside the plugin — until then, this manual step is needed.

### Credentials

The skill bank reads credentials from the **session environment** — export what you need in your shell **before launching**, and the session inherits them:

```bash
export NGC_KEY=...            # nvcr.io image pulls
export HF_TOKEN=...           # gated HuggingFace models
```

The vars each skill looks for (export only the ones your workflow needs):

| Var | Used for |
|---|---|
| `NGC_KEY` | `nvcr.io` image pulls — required by almost everything |
| `HF_TOKEN` | gated HuggingFace models / `push_to_hub` |
| `BREV_API_TOKEN` | `tao-run-on-brev` (optional — `brev login` also works) |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_BUCKET_NAME`, `AWS_ENDPOINT_URL`, `AWS_DEFAULT_REGION` | S3 / object-storage I/O via `tao-data-io` (legacy `ACCESS_KEY`/`SECRET_KEY` still accepted) |
| `WANDB_API_KEY`, `WANDB_PROJECT` | WandB experiment logging (AutoML / HF fine-tune) |

Some runtimes (notably Codex) do not reliably inherit shell exports; there, put bare `KEY=value` lines — no `export`, which `docker --env-file` rejects — in your own credential env file and load it with `set -a; source /path/to/.env; set +a`. Unprompted, the agent sources only the fixed user-owned locations (`~/.tao/secrets.env`, `~/.config/tao/.env`); a repo-relative `./.env` or `<workspace>/.env` is loaded only when you point the agent at it. On session start the hook reports which vars it detects in the environment (names only). The agent never reads credential values — it only checks presence, and never writes a credential to a file you did not ask for; registry logins (`docker login nvcr.io`), the one-time enroot credentials install on a SLURM cluster (`~/.config/enroot/.credentials`, which persists your NGC key in plaintext on shared storage), and SLURM job sidecars happen only after you approve them.

When a workflow needs Hugging Face access, get a token from [Hugging Face settings](https://huggingface.co/settings/tokens) and accept the model or dataset license before launch.

If a readiness check reports a missing CLI, container image, backbone, or credential, the TAO skills can often install or stage the missing piece after you approve the action. Ask the agent to continue the original workflow after a blocker is resolved; it should rerun preflight and proceed from the same task.

> **Persisting secrets is your own responsibility.** If you'd rather not re-export each session, keep them in an env file and either source it yourself or point the agent at it; a shell rc or a secrets manager works just as well. The agent never creates that file for you.

### Do I need to install anything else?

Usually there is no up-front setup beyond the chosen platform prerequisites.
Model and data skills run with just `docker run`; platform skills add job
tracking, S3 I/O, and multi-node over their native CLI with no
`nvidia-tao-sdk` — see [Execution: no SDK required](#execution-no-sdk-required).
Application workflows such as IAA DEFT may provision documented host-Python
dependencies into an isolated workspace environment, but only after the
workflow's approval gate. AutoML search (`tao-run-automl`) remains the one
special SDK exception: its Preflight lazily installs the
`nvidia-tao-automl` wheel, whose pin lives in [`versions.yaml`](versions.yaml)
(`wheels.tao_automl_*`).

### Updating

**Claude Code:**

```
/plugin marketplace update tao-skill-bank
/reload-plugins
```

If skills look stale (cached contents):

```bash
rm -rf ~/.claude/plugins/cache/tao-skill-bank
```

then re-run `/plugin install`.

**Codex:**

```bash
codex plugin marketplace upgrade tao-skill-bank
```

If you copied `AGENTS.md` to `~/.codex/AGENTS.md`, re-copy from the upgraded plugin cache to pick up identity changes.

## Getting started (5 minutes)

The quickest way to verify your setup: run a Visual ChangeNet inference on a sample image.

### Prerequisites

```shell
set -a; source /path/to/.env; set +a   # omit if already exported
docker --version
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
echo "$NGC_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

If any check fails, see `skills/platform/tao-run-on-docker/SKILL.md` for install/troubleshooting.

### Smoke test

In a Claude Code session with the plugin installed, ask:

> *"Run Visual ChangeNet inference on this sample image: /tmp/sample.png. Write results to /tmp/vcn-out/."*

The agent will read `skills/models/tao-train-visual-changenet/SKILL.md` (skill name `tao-train-visual-changenet`, plus its `references/skill_info.yaml` if present), construct a `docker run --gpus all ...` invocation, and execute via Bash. **No Python needed.** No SDK install. Just docker + the plugin. For classify mode, expect per-image PASS/NO_PASS-style predictions and result files under `/tmp/vcn-out/`. For segment mode, expect binary change-mask outputs under the requested results directory.

For more complex workflows, see `skills/applications/tao-run-deft-aoi/SKILL.md`
(`tao-run-deft-aoi`, shorthand `tao-deft-aoi`) for AOI iterative fine-tuning,
`skills/applications/tao-run-deft-iaa/SKILL.md` (`tao-run-deft-iaa`, shorthand
`tao-deft-iaa`) for the self-contained local-Docker IAA loop, and
`skills/applications/tao-run-automl/SKILL.md` (`tao-run-automl`) for
hyperparameter optimization. AutoML launch reviews should show the number of
recommendations, metric, search space, expected runtime, and resolved train
image before long-running jobs start.

## What's in the bank

| Layer | Purpose | Examples |
|---|---|---|
| `skills/models/` | Network-centric skills: containers, commands, data formats, checkpoints | `tao-finetune-cosmos-reason`, `tao-train-visual-changenet`, `tao-finetune-clip`, `tao-train-dino`, `tao-train-segformer`, … |
| `skills/data/` | Data preparation, analysis, and enhancement | `tao-mine-aoi-images`, `tao-analyze-gaps-visual-changenet`, `tao-route-visual-changenet-samples`, `tao-analyze-gaps-vlm-bcq`, `tao-convert-dataset-format`, `tao-validate-dataset-format`, `tao-generate-image-grounding`, `tao-generate-referring-expressions`, `tao-generate-video-reasoning-annotations` |
| `skills/platform/` | Where and how jobs run | `tao-run-on-docker` (local daemon or `DOCKER_HOST=ssh://`), `tao-run-on-brev` (instance-based GPU), `tao-run-on-slurm` (remote SLURM cluster), `tao-run-on-kubernetes` (k8s), `tao-run-on-virtualenv` (docker-free local venv), `tao-data-io` (S3/data staging), `tao-setup-nvidia-gpu-host` (host runtime) |
| `skills/applications/` | End-to-end workflows composing the layers above | `tao-run-deft-aoi`, `tao-run-deft-iaa`, `tao-run-automl-deft-pipeline`, `tao-analyze-changenet-rca`, `tao-train-single-step`, `tao-run-automl`, `tao-finetune-huggingface-model`, `tao-port-huggingface-model`, `tao-run-inference-service` |

Each skill is a directory with `SKILL.md` (agent-readable instructions). Optional `references/skill_info.yaml` provides structured metadata (container image, per-action command/mode/inputs/outputs) the agent uses to construct the container command; optional `scripts/` bundles supporting code.

The `skills/core/` directory is not a second copy of the skill bank. It is the Codex plugin surface for small helper/router skills, such as capability discovery and launch intake. Canonical model, data, platform, and application skills live once in the layer directories above; do not add symlinks or copies under `skills/core/`.

## Execution: no SDK required

Job tracking, S3 I/O, multi-node training, and failure-classified retries are
all built into the bank — there is **no `nvidia-tao-sdk`**. Every platform skill
implements the same **four-verb consumer contract** (`submit`/`status`/`logs`/
`cancel`) over its native CLI (`docker`, `kubectl`, `ssh`+`sbatch`, `brev`, or
the vendored virtualenv runner):

- **Job tracking** — `scripts/tao_job_record.py` mints a job id and binds
  `results_dir` before launch (record-then-launch), then records state
  transitions in a fixed vocabulary.
- **S3 / data staging** — the `tao-data-io` skill stages inputs (storage tiers
  A/B/C) and uploads results.
- **Multi-node** — the SLURM/K8s multi-node templates in `templates/` plus the
  NCCL probe (`scripts/nccl_allreduce_probe.py`).
- **Retries** — infra-vs-program failure classification lives in
  `tao-launch-workflow` (prose the agent applies, not a brittle regex table).

These need only the platform CLI plus the bank's helper scripts — no wheel to
install. The **one** exception is AutoML hyperparameter search
(`tao-run-automl`), which uses the `nvidia-tao-automl` wheel (and its transitive
`nvidia-tao-sdk` dependency) to pick each next config; its Preflight installs
the right extra on first use. The pins live in [`versions.yaml`](versions.yaml)
(`wheels.tao_automl_*`).

## Contributing a new skill

> **Read first:** [`docs/skill-requirements.md`](docs/skill-requirements.md) —
> the **must-follow** rules for naming, frontmatter, size, `evals/evals.json`,
> and the security-scanner gotchas that block at signing. CI errors and
> signing-pipeline blockers are called out separately. Treat this as the
> authoritative gate list; [`docs/authoring.md`](docs/authoring.md) is the
> longer walkthrough.

See [`docs/authoring.md`](docs/authoring.md) for the full authoring guide. The minimum viable skill is just `SKILL.md` — `references/skill_info.yaml` and friends are optional and only added when they earn their keep.

In brief:

1. Pick the layer (`skills/models/`, `skills/data/`, `skills/platform/`, `skills/applications/`).
2. Copy a template from [`templates/skill-skeleton/`](templates/skill-skeleton/) — `minimal/` for the bare path, `model/`, `data/`, `platform/`, or `workflow/` for richer scaffolding.
3. Fill in frontmatter and SKILL.md body. Body must contain a `## Quick Start` section, a `docker run` block, or a link to `references/skill_info.yaml`.
4. Add `evals/evals.json` (required for Tier-3 signing — see [`docs/skill-requirements.md`](docs/skill-requirements.md) § 2.3). `eval.config` is optional and only needed if you want live-execution coverage.
5. Add the skill path to [`.claude-plugin/marketplace.json`](.claude-plugin/marketplace.json) under the relevant plugin(s).
6. Do not add a mirror entry under `skills/core/`; Codex helper skills route to the canonical layer directories.
7. Validate with `scripts/validate-skills.sh` before submitting a PR.

## Repository structure

```
tao-skills-external/
├── .claude-plugin/
│   ├── marketplace.json              # marketplace catalog (plugin definitions)
│   └── plugin.json                   # plugin manifest (fallback when loaded directly)
├── hooks/
│   ├── hooks.json                    # SessionStart hook registration
│   └── session_start.sh              # emits agent guidance; reports credential vars present in the env
├── .codex-plugin/
│   └── plugin.json                   # Codex plugin manifest
├── .agents/
│   └── plugins/marketplace.json      # Codex marketplace entry
├── versions.yaml                     # single source of truth: container images + AutoML wheel versions
├── README.md
├── docs/
│   ├── skill-requirements.md         # must-follow rules: naming + signing gates (read first)
│   ├── authoring.md                  # guide for adding new skills
│   └── maintenance.md                # RC bump procedure for versions.yaml
├── templates/skill-skeleton/         # copy-paste starting points (minimal + per-layer)
├── scripts/
│   ├── validate-skills.sh            # CI validator
│   ├── verify-standalone.sh          # end-to-end smoke (docker-only path)
│   ├── install-codex-agents.sh       # one-shot Codex install: marketplace + plugin + AGENTS.md
│   └── migrate-to-version-keys.py    # one-shot: literal nvcr.io paths → versions.yaml keys
└── skills/
    ├── applications/                 # 13 end-to-end workflow skills
    ├── data/                         # 10 data preparation/analysis skills
    ├── models/                       # 53 network-centric skills
    ├── platform/                     # 7 compute backend / runtime skills
    └── core/                         # 2 Codex helper/router skills; no mirrored skill symlinks
```

## CI

The repo runs three CI suites in parallel:

- **NV-ACES skill evaluation** (`.skill-eval.yml`) — Tier 1/2 quality scoring, security scan.
- **Skill execution eval** (`.gitlab-ci.yml`) — runs each skill's `eval.config` on a real GPU runner.
- **`validate-skills`** (`scripts/validate-skills.sh`) — marketplace path resolution, no `skills/core/` mirrors, frontmatter, body has runnable info, no SDK leaks, hook references resolve.

PRs must pass all three before merge.

## Design rules

- **Docker-native first.** Every model/data skill should be runnable with just `docker run` + the contents of `SKILL.md`. Platform skills add tracking/staging/multi-node via the four-verb contract — no SDK.
- **Generic docker conventions live once** in `skills/platform/tao-run-on-docker`. Other skills defer to it for `--gpus`, NGC auth, mount patterns, data-root relocation, etc.
- **No SDK in the bank.** `tao_sdk`-specific imports and `sdk.create_job`/`build_entrypoint` calls are allowed only under `skills/applications/tao-run-automl` (it keeps the `nvidia-tao-automl` wheel + its transitive SDK). `scripts/validate-skills.sh` enforces this.
- **Minimum-viable skill is `SKILL.md` only.** Add `references/skill_info.yaml` only when multi-action structured metadata earns its keep.
- **One canonical location per skill.** Model, data, platform, and application skills live only in their layer directories; `skills/core/` is for Codex helper/router skills, not mirrored copies.
- **Prefer portability over cleverness.** A skill that works across three coding agents is more valuable than a skill that works perfectly in one.
