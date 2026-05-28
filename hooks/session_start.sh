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
#   2. Persist user credentials from ~/.config/tao/.env into the session via
#      $CLAUDE_ENV_FILE. The agent never reads values; only checks presence.
#   3. Surface clear setup hints if docker is missing.
#
# This hook does NOT install Python packages. The TAO SDK is opt-in and
# installed lazily by the skills that need it (platform/tao-run-on-lepton, platform/tao-run-platform,
# applications/tao-run-automl) via their Preflight blocks.

set -u

# Idempotency guard: emit the guidance only on the first invocation per session.
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
TAO_ENV_FILE="${HOME}/.config/tao/.env"
if [[ -f "$TAO_ENV_FILE" ]]; then
  # Persist to the session env file so subsequent Bash tool calls inherit them.
  if [[ -n "${CLAUDE_ENV_FILE:-}" ]]; then
    cat "$TAO_ENV_FILE" >> "$CLAUDE_ENV_FILE"
  fi
  echo "## Credentials"
  echo
  echo "Loaded from \`~/.config/tao/.env\`. The following vars are now in the session:"
  # List only NAMES — never values.
  awk -F= '/^[[:space:]]*(export[[:space:]]+)?[A-Z_][A-Z0-9_]*=/ {
    sub(/^[[:space:]]*export[[:space:]]+/, "")
    split($0, a, "=")
    print "- " a[1]
  }' "$TAO_ENV_FILE" | sort -u
  echo
else
  echo "## Credentials"
  echo
  echo "No \`~/.config/tao/.env\` found. To set up:"
  echo "\`\`\`bash"
  echo "mkdir -p ~/.config/tao"
  echo "cp \"\${CLAUDE_PLUGIN_ROOT}/.env.example\" ~/.config/tao/.env"
  echo "# Edit ~/.config/tao/.env and fill in values."
  echo "\`\`\`"
  echo "Future sessions will auto-load it."
  echo
fi

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
