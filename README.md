# NVIDIA TAO Skill Bank

Portable agent skills for training, evaluating, and running inference on NVIDIA TAO models. Works with Claude Code, Codex, Gemini CLI, or any coding agent that speaks the [Agent Skills open standard](https://agentskills.io). **Zero Python required** for local docker workflows — install the plugin, install docker + nvidia-container-toolkit, and an agent can run every skill by constructing `docker run` commands directly. For advanced features (job tracking, multi-node, Lepton access, S3 I/O wrapping), an optional Python layer — the [TAO Execution SDK](#optional-python-layer) — sits on top.

## Install

The skill bank works with both Claude Code and Codex. Pick the runtime you use.

### Claude Code

In a Claude Code session, add the marketplace and install a plugin. Two choices:

```
/plugin marketplace add git@github.com:NVIDIA-TAO/tao-skills-bank.git
/plugin install tao-skills@tao-skill-bank             # everything (recommended)
# or
/plugin install deft-aoi-loop-plugin@tao-skill-bank   # just the DEFT AOI loop
```

That's it — no `git clone`, no `pip install`. The `tao-skills` plugin bundles all 56 skills (every model, data, platform, and application). If you want only the focused DEFT loop bundle, install `deft-aoi-loop-plugin` instead. The plugin's [`SessionStart`](hooks/session_start.sh) hook loads the [`AGENTS.md`](AGENTS.md) identity at the start of every session.

### Codex

Codex setup has **two independent pieces** — the plugin (which surfaces the skills to Codex) and `AGENTS.md` (which loads the agent identity). You need both for parity with Claude Code.

#### 1. Install the plugin

**Option A — VS Code Codex extension (recommended for VS Code users).** Open the extension's plugin UI, add the marketplace URL, and install `tao-skill-bank` — all from the UI. Most discoverable, one click.

**Option B — CLI + TUI.** Add the marketplace from the shell, then install the plugin from inside the Codex TUI (no CLI `install` subcommand exists yet — [openai/codex#17431](https://github.com/openai/codex/issues/17431)):

```bash
codex plugin marketplace add git@github.com:NVIDIA-TAO/tao-skills-bank.git
codex                # opens TUI
/plugins             # then: select tao-skill-bank → Install plugin
```

Either path installs the bundle to `~/.codex/plugins/cache/<marketplace>/tao-skill-bank/<version>/` (the `<marketplace>` segment comes from the `name` field in `.agents/plugins/marketplace.json`).

#### 2. Load the agent identity (`AGENTS.md`)

The plugin install does **not** auto-load [`AGENTS.md`](AGENTS.md) — Codex's `AGENTS.md` discovery walks down from the project root, not into the plugin cache (see [openai/codex#16430](https://github.com/openai/codex/issues/16430) for why plugin-bundled `SessionStart` hooks don't fix this yet). Pick one:

- **Per-project (preferred)**: `git clone` this repo and launch `codex` from inside the clone. Codex auto-loads `AGENTS.md` from the project root per the [agents.md](https://agents.md/) cross-runtime spec.
- **Globally** (one-time copy): `cp ~/.codex/plugins/cache/<marketplace>/tao-skill-bank/<version>/AGENTS.md ~/.codex/AGENTS.md`. The identity then loads in every Codex session, anywhere.

Once Codex starts honoring plugin-bundled hooks, the identity will install automatically alongside the plugin — until then, this manual step is needed.

### Credentials

On first session start, the plugin looks for `~/.config/tao/.env` and auto-loads it. To set up:

```bash
mkdir -p ~/.config/tao
cp "${CLAUDE_PLUGIN_ROOT}/.env.example" ~/.config/tao/.env  # template ships in the plugin
# Edit ~/.config/tao/.env and fill in NGC_KEY, LEPTON_*, S3 keys, etc.
```

The `.env.example` is also at the [repo root](.env.example) for direct reference. The agent never reads credential values — it only checks presence.

### When does the SDK get installed?

The TAO SDK is **opt-in** and installed lazily. Most skills (any model or data skill) run with just `docker run` and need no Python. Only `platform/lepton`, `platform/tao-sdk`, the managed-platform skills (slurm/kubernetes/docker), and `applications/tao-automl` require the SDK; their Preflight blocks tell the agent to run a `pip install "nvidia-tao-sdk[<platform>] @ git+https://..."` direct-URL the first time the skill is invoked. (The SDK isn't on public PyPI yet — see each skill's Preflight for the exact command.)

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
docker --version
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
echo "$NGC_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

If any check fails, see `platform/docker/SKILL.md` for install/troubleshooting.

### Smoke test

In a Claude Code session with the plugin installed, ask:

> *"Run visual-changenet inference on this sample image: /tmp/sample.png. Write results to /tmp/vcn-out/."*

The agent will read `models/visual-changenet/SKILL.md` (and `references/skill_info.yaml` if present), construct a `docker run --gpus all ...` invocation, and execute via Bash. **No Python needed.** No SDK install. Just docker + the plugin.

For more complex workflows (iterative fine-tuning with synthetic data augmentation), see `applications/workflow-deft-aoi-loop/SKILL.md`.

## What's in the bank

| Layer | Purpose | Examples |
|---|---|---|
| `models/` | Network-centric skills: containers, commands, data formats, checkpoints | `cosmos-rl`, `visual-changenet`, `clip`, `vila`, `dino`, `segformer`, … |
| `data/` | Data preparation, analysis, and enhancement | `knn-mining`, `siglip-embed`, `qwen-caption`, `nim-embedding`, `vcn-*`, `deft-aoi-*` |
| `platform/` | Where and how jobs run | `docker` (conventions), `brev` (instance-based GPU), `lepton` (DGX Cloud API), `slurm` (remote SLURM cluster), `local-docker` (local Docker daemon), `tao-sdk` (optional Python) |
| `applications/` | End-to-end workflows composing the layers above | `workflow-deft-aoi-loop`, `deft-cosmos-rl`, `deft-vcn-aoi`, `rca-changenet`, `normal-train`, `tao-automl` |

Each skill is a directory with `SKILL.md` (agent-readable instructions). Optional `references/skill_info.yaml` provides structured metadata for SDK-orchestrated execution; optional `scripts/` bundles supporting code.

## Optional Python layer

For users who want job handles, S3 I/O wrapping via `script_runner`, state persistence, multi-node distributed training, Lepton access, or failure analysis, the [TAO Execution SDK](https://gitlab-master.nvidia.com/nvidia-tao-toolkit/tao-sdk) provides a single wheel with optional extras:

```shell
# The SDK is not on public PyPI yet — install via pip direct-URL:
REPO='git+https://gitlab-master.nvidia.com/nvidia-tao-toolkit/tao-sdk.git'
pip install "nvidia-tao-sdk @ $REPO"                  # core
pip install "nvidia-tao-sdk[lepton] @ $REPO"          # + Lepton (required — no docker-run equivalent)
pip install "nvidia-tao-sdk[brev] @ $REPO"            # + Brev (wraps brev CLI with Job handles)
pip install "nvidia-tao-sdk[slurm] @ $REPO"           # + SLURM
pip install "nvidia-tao-sdk[kubernetes] @ $REPO"      # + Kubernetes
pip install "nvidia-tao-sdk[docker] @ $REPO"          # + local Docker
pip install "nvidia-tao-sdk[all] @ $REPO"             # all platforms
```

You don't have to pre-install — the relevant skills (`platform/lepton`, `platform/tao-sdk`, `applications/tao-automl`) run a Preflight that prompts the agent to install the right extra on first use. If you're running locally on your own GPU or on Brev via `brev exec`, you don't need the SDK at all.

## Contributing a new skill

See [docs/authoring.md](docs/authoring.md) for the full guide. The minimum viable skill is just `SKILL.md` — `references/skill_info.yaml` and friends are optional and only added when they earn their keep.

In brief:

1. Pick the layer (`models/`, `data/`, `platform/`, `applications/`).
2. Copy a template from [`templates/skill-skeleton/`](templates/skill-skeleton/) — `minimal/` for the bare path, `model/`, `data/`, `platform/`, or `workflow/` for richer scaffolding.
3. Fill in frontmatter and SKILL.md body. Body must contain a `## Quick Start` section, a `docker run` block, an SDK call, or a link to `references/skill_info.yaml`.
4. Add the skill path to [`.claude-plugin/marketplace.json`](.claude-plugin/marketplace.json) under the relevant plugin(s).
5. Validate with `scripts/validate-skills.sh` before submitting a PR.

## Repository structure

```
tao-skills-bank/
├── .claude-plugin/
│   ├── marketplace.json              # marketplace catalog (lists tao-skills + deft-aoi-loop-plugin)
│   └── plugin.json                   # plugin manifest (fallback when loaded directly)
├── hooks/
│   ├── hooks.json                    # SessionStart hook registration
│   └── session_start.sh              # emits agent guidance; sources ~/.config/tao/.env
├── .env.example                      # credential template (copy to ~/.config/tao/.env)
├── versions.yaml                     # single source of truth: container images + SDK wheel versions
├── README.md
├── docs/
│   ├── authoring.md                  # guide for adding new skills
│   └── maintenance.md                # RC bump procedure for versions.yaml
├── templates/skill-skeleton/         # copy-paste starting points (minimal + per-layer)
├── scripts/
│   ├── validate-skills.sh            # CI validator
│   ├── verify-standalone.sh          # end-to-end smoke (docker-only path)
│   └── migrate-to-version-keys.py    # one-shot: literal nvcr.io paths → versions.yaml keys
├── applications/
├── data/
├── models/
└── platform/
```

## CI

The repo runs three CI suites in parallel:

- **NV-ACES skill evaluation** (`.skill-eval.yml`) — Tier 1/2 quality scoring, security scan.
- **Skill execution eval** (`.gitlab-ci.yml`) — runs each skill's `eval.config` on a real GPU runner.
- **`validate-skills`** (`scripts/validate-skills.sh`) — marketplace path resolution, frontmatter, body has runnable info, no SDK leaks, hook references resolve.

PRs must pass all three before merge.

## Design rules

- **Docker-native first.** Every model/data skill should be runnable with just `docker run` + the contents of `SKILL.md`. SDK invocation is an optional enhancement, documented in `platform/tao-sdk`.
- **Generic docker conventions live once** in `platform/docker`. Other skills defer to it for `--gpus`, NGC auth, mount patterns, data-root relocation, etc.
- **No SDK leaks in model/data/application skills.** `tao_sdk`-specific imports, `sdk.create_job` calls, and credential-file references belong only in `platform/tao-sdk` and (for platform-specific reasons) `platform/lepton`.
- **Minimum-viable skill is `SKILL.md` only.** Add `references/skill_info.yaml` only when SDK orchestration or multi-action structured metadata earn their keep.
- **Prefer portability over cleverness.** A skill that works across three coding agents is more valuable than a skill that works perfectly in one.
