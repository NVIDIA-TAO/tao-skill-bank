#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One-shot Codex installer for the TAO skill bank.
#
# What it does:
#   1. Registers the tao-skills-external marketplace with the Codex CLI.
#   2. Installs the `tao-skill-bank` plugin (skills surface).
#   3. Copies AGENTS.md to ~/.codex/AGENTS.md so the TAO identity loads in
#      every Codex session, not only when codex is launched from a clone.
#      (Codex's AGENTS.md discovery walks the project tree from the git root —
#      plugin-bundled SessionStart hooks do not yet install identity globally;
#      see https://github.com/openai/codex/issues/16430.)
#
# Override the source via env var if you need a fork or a pinned ref:
#   TAO_SKILL_BANK_MARKETPLACE=ssh://git@host/path/repo.git \
#   TAO_SKILL_BANK_REF=release/7.0.0 \
#       scripts/install-codex-agents.sh

set -euo pipefail

MARKETPLACE_SOURCE="${TAO_SKILL_BANK_MARKETPLACE:-git@github.com:NVIDIA-TAO/tao-skills-bank.git}"
MARKETPLACE_REF="${TAO_SKILL_BANK_REF:-}"
MARKETPLACE_NAME="tao-local-plugins"   # `name` in .agents/plugins/marketplace.json
PLUGIN_NAME="tao-skill-bank"

log() { printf '[install-codex-agents] %s\n' "$*"; }
die() { printf '[install-codex-agents] ERROR: %s\n' "$*" >&2; exit 1; }

# 0. Preflight
command -v codex >/dev/null 2>&1 \
  || die "'codex' CLI not found. Install it first: https://developers.openai.com/codex"

# 1. Marketplace
if codex plugin marketplace list 2>/dev/null | grep -qw "$MARKETPLACE_NAME"; then
  log "Marketplace '$MARKETPLACE_NAME' already registered — refreshing."
  codex plugin marketplace upgrade "$MARKETPLACE_NAME"
else
  log "Adding marketplace from $MARKETPLACE_SOURCE"
  if [[ -n "$MARKETPLACE_REF" ]]; then
    codex plugin marketplace add "$MARKETPLACE_SOURCE" --ref "$MARKETPLACE_REF"
  else
    codex plugin marketplace add "$MARKETPLACE_SOURCE"
  fi
fi

# 2. Plugin
if codex plugin list 2>/dev/null | grep -qw "$PLUGIN_NAME"; then
  log "Plugin '$PLUGIN_NAME' already installed."
else
  log "Installing plugin ${PLUGIN_NAME}@${MARKETPLACE_NAME}"
  codex plugin add "${PLUGIN_NAME}@${MARKETPLACE_NAME}"
fi

# 3. Global AGENTS.md identity
# Prefer the freshly-installed plugin cache so the identity matches the
# installed plugin version. Fall back to the repo root when running from a
# clone before the plugin cache exists.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_AGENTS="${SCRIPT_DIR}/../AGENTS.md"
CACHE_ROOT="${HOME}/.codex/plugins/cache/${MARKETPLACE_NAME}/${PLUGIN_NAME}"

SRC_AGENTS=""
if [[ -d "$CACHE_ROOT" ]]; then
  LATEST_VERSION="$(ls -1 "$CACHE_ROOT" 2>/dev/null | sort -V | tail -n1 || true)"
  if [[ -n "$LATEST_VERSION" && -f "${CACHE_ROOT}/${LATEST_VERSION}/AGENTS.md" ]]; then
    SRC_AGENTS="${CACHE_ROOT}/${LATEST_VERSION}/AGENTS.md"
  fi
fi
if [[ -z "$SRC_AGENTS" && -f "$REPO_AGENTS" ]]; then
  SRC_AGENTS="$(cd "$(dirname "$REPO_AGENTS")" && pwd)/AGENTS.md"
fi
if [[ -z "$SRC_AGENTS" ]]; then
  log "WARN: could not locate AGENTS.md; skipping global identity install."
  log "      Verify with: codex plugin list"
  exit 0
fi

DEST_AGENTS="${HOME}/.codex/AGENTS.md"
mkdir -p "${HOME}/.codex"
if [[ -f "$DEST_AGENTS" ]] && ! cmp -s "$SRC_AGENTS" "$DEST_AGENTS"; then
  BACKUP="${DEST_AGENTS}.bak.$(date +%Y%m%d%H%M%S)"
  log "Backing up existing $DEST_AGENTS -> $BACKUP"
  cp "$DEST_AGENTS" "$BACKUP"
fi
cp "$SRC_AGENTS" "$DEST_AGENTS"
log "Installed TAO agent identity -> $DEST_AGENTS"

log "Done. Launch 'codex' from any directory to use the TAO skill bank."
