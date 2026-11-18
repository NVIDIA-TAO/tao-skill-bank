#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One-shot Codex installer for the TAO skill bank.
#
# What it does:
#   1. Registers the tao-skill-bank marketplace with the Codex CLI.
#   2. Installs the `nvidia-tao-toolkit` plugin (skills surface). Releases through 7.3.0
#      ship it as `tao-skill-bank`; the script installs whichever name the
#      chosen release offers and removes the other one if it is installed.
#      Releases after 7.3.0 need Codex CLI 0.142.0 or newer.
#   3. Copies AGENTS.md to ~/.codex/AGENTS.md so the TAO identity loads in
#      every Codex session, not only when codex is launched from a clone.
#      (Codex's AGENTS.md discovery walks the project tree from the git root —
#      plugin-bundled SessionStart hooks do not yet install identity globally;
#      see https://github.com/openai/codex/issues/16430.)
#
# By default it installs the latest release build (DEFAULT_MARKETPLACE_REF
# below) rather than whatever `main` holds. Override the source or the ref via
# env var if you need a fork, a different release, or unreleased work:
#   TAO_SKILL_BANK_MARKETPLACE=ssh://git@host/path/repo.git \
#   TAO_SKILL_BANK_REF=main \
#       scripts/install-codex-agents.sh
# `TAO_SKILL_BANK_REF=` (empty) uses the source's default branch.

set -euo pipefail

# Latest release tag — keep in sync with the pins in README.md on every release.
DEFAULT_MARKETPLACE_REF="7.3.0"

MARKETPLACE_SOURCE="${TAO_SKILL_BANK_MARKETPLACE:-https://github.com/NVIDIA-TAO/tao-skill-bank.git}"
MARKETPLACE_REF="${TAO_SKILL_BANK_REF-$DEFAULT_MARKETPLACE_REF}"

# Plugin and marketplace names, newest first. This script is fetched from `main`
# but installs a release tag, so it cannot assume which names that tag ships:
#   after 7.3.0      nvidia-tao-toolkit  in marketplace tao-skill-bank
#   through 7.3.0    tao-skill-bank      in marketplace tao-local-plugins
# (`name` in .codex-plugin/plugin.json and .agents/plugins/marketplace.json.)
PLUGIN_NAMES=("nvidia-tao-toolkit" "tao-skill-bank")
MARKETPLACE_NAMES=("tao-skill-bank" "tao-local-plugins")

# Releases after 7.3.0 use the repository root as the plugin source. Codex CLIs
# older than this register the marketplace and then report that the plugin was
# not found in it.
MIN_CODEX_VERSION="0.142.0"

# A local path marketplace (clone or unzipped release) has no ref to check out.
if [[ -d "$MARKETPLACE_SOURCE" ]]; then
  MARKETPLACE_REF=""
fi

log() { printf '[install-codex-agents] %s\n' "$*"; }
die() { printf '[install-codex-agents] ERROR: %s\n' "$*" >&2; exit 1; }

# 0. Preflight
command -v codex >/dev/null 2>&1 \
  || die "'codex' CLI not found. Install it first: https://developers.openai.com/codex"

codex_older_than() {
  # True when the installed CLI reports a version lower than $1. A version that
  # cannot be read counts as new enough: the caller only uses this for a hint.
  local have
  have="$(codex --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1 || true)"
  [[ -n "$have" && "$have" != "$1" ]] \
    && [[ "$(printf '%s\n%s\n' "$have" "$1" | sort -V | head -n1)" == "$have" ]]
}

# 1. Marketplace
# `codex plugin marketplace add` refuses to overwrite a registration that points
# at a different source or ref, and `upgrade` only re-fetches the ref already
# recorded — so an existing registration is dropped and re-added. That is what
# moves a previously unpinned (or older-release) install onto this ref.
#
# Both generations of names are cleared first. Leaving an older plugin enabled
# next to the new one lists every skill twice, which makes Codex cut each skill
# description to a few words.
marketplace_registered() {
  # Exact match on the name column: marketplace roots are paths that can
  # contain these names too.
  codex plugin marketplace list 2>/dev/null \
    | awk -v name="$1" '$1 == name { found = 1 } END { exit !found }'
}
for i in "${!PLUGIN_NAMES[@]}"; do
  codex plugin remove "${PLUGIN_NAMES[$i]}@${MARKETPLACE_NAMES[$i]}" >/dev/null 2>&1 || true
done
for name in "${MARKETPLACE_NAMES[@]}"; do
  if marketplace_registered "$name"; then
    log "Marketplace '$name' already registered — re-pointing it."
    codex plugin marketplace remove "$name"
  fi
done

log "Adding marketplace from ${MARKETPLACE_SOURCE}${MARKETPLACE_REF:+ (ref: ${MARKETPLACE_REF})}"
if [[ -n "$MARKETPLACE_REF" ]]; then
  codex plugin marketplace add "$MARKETPLACE_SOURCE" --ref "$MARKETPLACE_REF"
else
  codex plugin marketplace add "$MARKETPLACE_SOURCE"
fi

# 2. Plugin
# `codex plugin add` is idempotent and re-resolves the version from the
# marketplace snapshot. Try the current name first and fall back to the name
# older releases ship, so one script serves every release tag.
PLUGIN_NAME=""
MARKETPLACE_NAME=""
ADD_ERRORS=""
for i in "${!PLUGIN_NAMES[@]}"; do
  candidate="${PLUGIN_NAMES[$i]}@${MARKETPLACE_NAMES[$i]}"
  if err="$(codex plugin add "$candidate" 2>&1 >/dev/null)"; then
    PLUGIN_NAME="${PLUGIN_NAMES[$i]}"
    MARKETPLACE_NAME="${MARKETPLACE_NAMES[$i]}"
    log "Installed plugin ${candidate}"
    break
  fi
  ADD_ERRORS="${ADD_ERRORS}  ${candidate}: ${err}"$'\n'
done
if [[ -z "$PLUGIN_NAME" ]]; then
  printf '%s' "$ADD_ERRORS" >&2
  if codex_older_than "$MIN_CODEX_VERSION"; then
    printf '[install-codex-agents] This Codex CLI is older than %s, which releases after 7.3.0 need. Upgrade Codex and run this script again.\n' \
      "$MIN_CODEX_VERSION" >&2
  fi
  die "could not install the TAO plugin from ${MARKETPLACE_SOURCE}${MARKETPLACE_REF:+ (ref: ${MARKETPLACE_REF})}"
fi

# 3. Global AGENTS.md identity
# Prefer the marketplace snapshot: it is checked out at exactly the ref that was
# just installed, so the identity matches the release the plugin came from. The
# plugin cache keeps every version ever installed, so picking its highest
# version number would hand back a newer identity than a pinned older release
# ships. Fall back to the cache, then to the repo root when running from a clone.
CODEX_ROOT="${CODEX_HOME:-${HOME}/.codex}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_AGENTS="${SCRIPT_DIR}/../AGENTS.md"
CACHE_ROOT="${CODEX_ROOT}/plugins/cache/${MARKETPLACE_NAME}/${PLUGIN_NAME}"

SRC_AGENTS=""
MARKETPLACE_ROOT="$(codex plugin marketplace list 2>/dev/null \
  | awk -v name="$MARKETPLACE_NAME" '$1 == name { sub(/^[^ \t]+[ \t]+/, ""); print; exit }' || true)"
if [[ -n "$MARKETPLACE_ROOT" && -f "${MARKETPLACE_ROOT}/AGENTS.md" ]]; then
  SRC_AGENTS="${MARKETPLACE_ROOT}/AGENTS.md"
fi
if [[ -z "$SRC_AGENTS" && -d "$CACHE_ROOT" ]]; then
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

DEST_AGENTS="${CODEX_ROOT}/AGENTS.md"
mkdir -p "$CODEX_ROOT"
if [[ -f "$DEST_AGENTS" ]] && ! cmp -s "$SRC_AGENTS" "$DEST_AGENTS"; then
  BACKUP="${DEST_AGENTS}.bak.$(date +%Y%m%d%H%M%S)"
  log "Backing up existing $DEST_AGENTS -> $BACKUP"
  cp "$DEST_AGENTS" "$BACKUP"
fi
cp "$SRC_AGENTS" "$DEST_AGENTS"
log "Installed TAO agent identity -> $DEST_AGENTS"

log "Done. Launch 'codex' from any directory to use the TAO skill bank."
