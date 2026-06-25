#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SessionStart hook for the TAO skill bank.
#
# Stdout is loaded into the agent's context as additionalContext at session
# start. Keep it tight — every line lands in context for every session.
#
# Responsibilities:
#   1. Emit TAO orchestration guidance (the agent's identity + discovery flow).
#   2. Report which credential env vars are present in the session (names only).
#      This hook does NOT read or load any credentials file — users export
#      credentials in their own shell; the session inherits that environment.
#   3. Surface clear setup hints if docker is missing.
#
# This hook does NOT install Python packages. The TAO SDK is opt-in and
# installed lazily by the skills that need it (skills/platform/tao-run-platform,
# skills/applications/tao-run-automl) via their Preflight blocks.

set -u

# Idempotency guard: both `tao-skills` and `deft-aoi-loop-plugin` share the
# same source dir, so hook auto-discovery fires this script once per enabled
# plugin. Emit the guidance only on the first invocation per session.
if [[ -n "${TAO_SESSION_INIT_DONE:-}" ]]; then
  exit 0
fi
if [[ -n "${CLAUDE_ENV_FILE:-}" ]]; then
  echo "export TAO_SESSION_INIT_DONE=1" >> "$CLAUDE_ENV_FILE"
fi

# ─── 1. Agent guidance ────────────────────────────────────────────────────
# Single source of truth: AGENTS.md at the plugin root (cross-runtime spec —
# https://agents.md/). Edit there to update Claude + Codex + any future
# runtime in one place. Do not duplicate the prompt inline here or in other
# hooks.
if [[ -n "${CLAUDE_PLUGIN_ROOT:-}" && -f "${CLAUDE_PLUGIN_ROOT}/AGENTS.md" ]]; then
  cat "${CLAUDE_PLUGIN_ROOT}/AGENTS.md"
  echo
fi

# ─── 1b. Make versions.yaml + skill bank discoverable to the SDK ──────────
# The SDK's tao_sdk.versions module checks $TAO_SKILL_BANK_PATH for
# versions.yaml. Plugin-installed users (pip install nvidia-tao-sdk + plugin
# install tao-skill-bank) need this to resolve container_image keys like
# `tao_toolkit.pyt`.
if [[ -n "${CLAUDE_PLUGIN_ROOT:-}" && -n "${CLAUDE_ENV_FILE:-}" ]]; then
  echo "export TAO_SKILL_BANK_PATH=\"${CLAUDE_PLUGIN_ROOT}\"" >> "$CLAUDE_ENV_FILE"
fi

# ─── 2. Credentials ───────────────────────────────────────────────────────
# This hook does NOT read or load any credentials file. Export credentials in
# your own shell (or shell rc / secrets manager) before launching; the session
# inherits that environment. Whether and how to persist secrets to disk is the
# user's own responsibility — the skill bank neither writes nor reads a
# credentials file. The agent only checks presence (names), never values.
echo "## Credentials"
echo
# Known credential vars across the platform/model skills. Names only.
_tao_cred_vars="NGC_KEY BREV_API_TOKEN \
ACCESS_KEY SECRET_KEY S3_BUCKET_NAME S3_ENDPOINT_URL HF_TOKEN WANDB_API_KEY"
_tao_present=""
for _v in $_tao_cred_vars; do
  [[ -n "${!_v:-}" ]] && _tao_present="${_tao_present} ${_v}"
done
if [[ -n "${_tao_present// /}" ]]; then
  echo "Detected in this session's environment (names only):"
  for _v in $_tao_present; do echo "- $_v"; done
else
  echo "No TAO credential vars detected in this session's environment."
fi
echo
echo "Credentials are read from the environment — export what you need in your"
echo "shell **before launching**, e.g.:"
echo "\`\`\`bash"
echo "export NGC_KEY=...            # nvcr.io image pulls"
echo "export HF_TOKEN=...           # gated HuggingFace models"
echo "# platform-specific: BREV_API_TOKEN, ACCESS_KEY/SECRET_KEY/S3_*"
echo "\`\`\`"
echo "See the Credentials section of the skill bank README for the full var list."
echo "The skill bank does not create or load a credentials file; persisting"
echo "secrets to disk is your own responsibility."
echo

# ─── 3. Docker preflight ──────────────────────────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
  echo "## ⚠ Docker missing"
  echo
  echo "Most TAO skills need Docker plus the pinned TAO GPU host runtime:"
  echo "- NVIDIA driver branch 580"
  echo "- CUDA Toolkit 13.0"
  echo "- NVIDIA Container Toolkit 1.19.0"
  echo
  echo "Use the \`tao-setup-nvidia-gpu-host\` skill to check / install the NVIDIA pieces;"
  echo "its \`--backend docker --install --yes\` path also installs Docker on"
  echo "Debian/RHEL/SUSE-family hosts and adds you to the \`docker\` group."
  echo "Manual install reference: https://docs.docker.com/engine/install/"
  echo
fi
