---
name: tao-setup
description: One-time session setup and orchestration map for the TAO skill bank. Run this first when the TAO skills were installed individually (e.g. from a public skills catalog) so the session gets the cross-skill discovery flow, credential checks, and host preflight that the bundled plugin hook would otherwise inject automatically. Trigger phrases include "set up TAO skills", "TAO session setup", "prepare TAO environment", "TAO getting started".
license: Apache-2.0
compatibility: Requires bash and Python 3.10+. Docker plus the NVIDIA container toolkit are needed by most downstream TAO skills but are only checked (not installed) here.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- setup
- orchestration
- discovery
---

# TAO Setup

One-time session bootstrap for the TAO skill bank. TAO skills are standalone —
each model, data, and platform skill carries its own pinned container image and
instructions — but multi-skill workflows chain them (data prep, train,
evaluate, deploy). This skill provides the session-level pieces that make that
chaining work when skills are installed individually: the discovery flow, the
credential conventions, and the host preflight.

When the full skill bank is installed as a plugin from this repository, a
SessionStart hook injects this guidance automatically and you do not need to
run this skill. When skills were installed one-by-one from a skills catalog,
run this skill first.

## Quick Start

```bash
# 1. Host preflight — most TAO skills dispatch docker containers on a GPU host.
docker info > /dev/null && echo "OK: docker" || echo "MISSING: docker"
nvidia-smi > /dev/null && echo "OK: GPU" || echo "MISSING: NVIDIA GPU/driver"

# 2. Credential presence check — names only, never read values.
for v in NGC_KEY HF_TOKEN WANDB_API_KEY ACCESS_KEY SECRET_KEY S3_BUCKET_NAME S3_ENDPOINT_URL BREV_API_TOKEN; do
  [ -n "${!v:-}" ] && echo "SET:   $v" || echo "unset: $v"
done

# 3. NGC registry login (needed for nvcr.io image pulls). Key goes over
#    stdin — never as an argv flag, where it lands in the process table.
[ -n "${NGC_KEY:-}" ] && printf '%s' "$NGC_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

If Docker or the NVIDIA host runtime is missing, use the
`tao-setup-nvidia-gpu-host` skill — it checks and (with approval) installs
NVIDIA driver 580 or newer, CUDA Toolkit 13.0 or newer, and NVIDIA Container
Toolkit 1.19.0 or newer, and can install Docker itself on Debian/RHEL/SUSE-family
hosts. These are TAO-wide minimums. If the selected model's
`references/skill_info.yaml` declares `runtime_requirements.gpu_host`, pass
those model-specific minimums to the host setup skill instead.

## Credentials

Credentials come from the session environment — export them in your shell
before launching the agent. This skill (and the bank) never reads credential
values and never creates or loads a credentials file; verify presence only
with `[ -n "$VAR" ]`.

- `NGC_KEY` — nvcr.io image pulls (most skills)
- `HF_TOKEN` — gated HuggingFace weights (several model skills)
- `WANDB_API_KEY` — experiment tracking (optional)
- `ACCESS_KEY` / `SECRET_KEY` / `S3_BUCKET_NAME` / `S3_ENDPOINT_URL` — S3 I/O
- `BREV_API_TOKEN` — Brev platform dispatch

## Discovery flow (how TAO skills chain)

1. **Read the task skill.** Model skills (`tao-train-*`, `tao-finetune-*`)
   own network specifics; data skills (`tao-generate-*`, `tao-analyze-*`,
   `tao-mine-*`, …) own transforms; application skills (`tao-run-automl`,
   `tao-run-deft-aoi`, …) compose model + data + platform into workflows.

2. **Read the skill's `references/skill_info.yaml`** (when present) for the
   structured contract: `container_image` (a pinned URI), per-action
   `command`, `mode`, `config_format`, `inputs`, `outputs`, and optional
   `runtime_requirements.gpu_host`. Model runtime requirements override the
   TAO-wide platform defaults for that workflow.

3. **Pick an execution platform and read its skill** for mounts, env vars,
   and resource conventions: `tao-run-on-docker` conventions apply to any
   local `docker run`; `tao-run-on-slurm`, `tao-run-on-kubernetes`,
   `tao-run-on-brev`, and `tao-run-on-local-docker` cover managed dispatch.
   The platforms are equal-class peers — if the user has not chosen, ask;
   never default silently.

4. **Construct the spec as nested dicts** (`{"train": {"num_epochs": 12}}`,
   never flat dotted keys), confirm with the user, then dispatch via the
   chosen platform's pattern and monitor per that platform's skill.

## Conventions all TAO skills follow

- **Confirm before side effects.** `docker run`, job submission, pushes, and
  file mutations outside the working directory need user confirmation first.
  Installing a missing Python package prerequisite is the one exception:
  install it by default and report what was installed.
- **Never ask for credentials in chat** and never read credential values or
  files; name the missing variable and let the user export it.
- **Container images are pinned per skill.** Each skill carries the exact
  image URI it was validated against; do not swap tags silently. Offer
  overrides only when the skill documents an override path.
- **Runtime requirements are layered.** Platform skills own the default host
  requirements and the check/install mechanism. A model may override only the
  minimum versions it has validated by declaring `runtime_requirements.gpu_host`
  in `references/skill_info.yaml`; pass those values to the shared host setup
  check rather than changing the defaults for unrelated models.

## Optional: Codex agent identity

For Codex sessions, `scripts/install-codex-agents.sh` registers the TAO skill
marketplace, installs the plugin, and copies the TAO agent identity to
`~/.codex/AGENTS.md` so it loads in every session:

```bash
bash scripts/install-codex-agents.sh
```

