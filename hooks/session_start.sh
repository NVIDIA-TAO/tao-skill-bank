#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SessionStart hook for the TAO skill bank.
#
# Stdout is loaded into the agent's context as additionalContext at session
# start. Keep it tight — every line lands in context for every session.
#
# Usage: session_start.sh [identity|context|all]   (default: all)
#
# Claude Code caps each hook command's output at 10,000 characters and replaces
# anything longer with a short preview, so hooks/hooks.json registers this
# script twice and each run prints one part:
#   identity  AGENTS.md up to the split heading (identity + discovery flow)
#   context   the rest of AGENTS.md, then credentials, card packs, docker check
# `all` prints both parts in order, for anything that runs the script directly.
# scripts/tests/test_session_start_hook.py keeps each part under the cap.
#
# Responsibilities:
#   1. Emit TAO orchestration guidance (the agent's identity + discovery flow).
#   2. Report which credential env vars are present in the session (names only).
#      Credentials arrive from the user's own shell or from a user-approved env
#      file the agent sources; this hook never reads or prints values.
#   3. Surface clear setup hints if docker is missing.
#
# This hook does NOT install Python packages. The bank runs SDK-free over native
# platform CLIs; the one wheel-dependent skill, skills/applications/tao-run-automl,
# installs nvidia-tao-automl lazily via its Preflight block.

set -u

TAO_PART="${1:-all}"
case "$TAO_PART" in
  identity|context|all) ;;
  *) echo "usage: session_start.sh [identity|context|all]" >&2; exit 2 ;;
esac

# Plugin root. Claude Code sets CLAUDE_PLUGIN_ROOT; other harnesses that run this
# hook may set their own variable or none, so fall back to this script's parent.
TAO_ROOT="${CLAUDE_PLUGIN_ROOT:-${CURSOR_PLUGIN_ROOT:-${PLUGIN_ROOT:-}}}"
if [[ -z "$TAO_ROOT" || ! -f "$TAO_ROOT/AGENTS.md" ]]; then
  TAO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

# Line number of the level-2 heading where AGENTS.md is split between the two
# parts: the first "## " heading at or past the midpoint, outside fenced code
# (a "## " line inside a fence is a comment, not a heading). 0 = do not split.
tao_split_line() {
  awk -v total="$(wc -c < "$1")" '
    { if (/^```/) fence = !fence
      if (!found && !fence && /^## / && off >= total / 2) { print NR; found = 1 }
      off += length($0) + 1 }
    END { if (!found) print 0 }' "$1"
}

# ─── 1. Agent guidance ────────────────────────────────────────────────────
# Single source of truth: AGENTS.md at the plugin root (cross-runtime spec —
# https://agents.md/). Edit there to update Claude + Codex + any future
# runtime in one place. Do not duplicate the prompt inline here or in other
# hooks.
if [[ -f "${TAO_ROOT}/AGENTS.md" ]]; then
  _tao_split="$(tao_split_line "${TAO_ROOT}/AGENTS.md")"
  if [[ "$TAO_PART" == "all" || "$_tao_split" -le 1 ]]; then
    # No split point: the identity part carries the whole file.
    [[ "$TAO_PART" != "context" ]] && { cat "${TAO_ROOT}/AGENTS.md"; echo; }
  elif [[ "$TAO_PART" == "identity" ]]; then
    head -n "$((_tao_split - 1))" "${TAO_ROOT}/AGENTS.md"
  else
    echo "<!-- TAO agent guidance, part 2 of 2 (part 1: identity and discovery flow) -->"
    tail -n "+${_tao_split}" "${TAO_ROOT}/AGENTS.md"
    echo
  fi
fi
[[ "$TAO_PART" == "identity" ]] && exit 0

# ─── 1b. Make versions.yaml + skill bank discoverable to the helper scripts ──
# The bank's scripts (resolve_tao_image.py, tao_job_record.py) and templates/
# read $TAO_SKILL_BANK_PATH to find versions.yaml. Claude Code persists it for
# the session's shell commands through CLAUDE_ENV_FILE; harnesses without that
# file get the path spelled out instead, and the helper scripts fall back to
# the bank they ship in.
if [[ -n "${CLAUDE_ENV_FILE:-}" ]]; then
  _tao_export="export TAO_SKILL_BANK_PATH=\"${TAO_ROOT}\""
  grep -qxF "$_tao_export" "$CLAUDE_ENV_FILE" 2>/dev/null \
    || echo "$_tao_export" >> "$CLAUDE_ENV_FILE"
else
  echo "TAO skill bank root: \`${TAO_ROOT}\`. Helper scripts and skills read it from"
  echo "\`TAO_SKILL_BANK_PATH\`; export that variable before running them from another directory."
  echo
fi

# ─── 2. Credentials ───────────────────────────────────────────────────────
# Presence only: names, never values. The credential policy itself lives in
# AGENTS.md, which this hook already emits above.
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
echo "shell **before launching**, or source a user-approved env file, e.g.:"
echo "\`\`\`bash"
echo "export NGC_KEY=...            # nvcr.io image pulls"
echo "export HF_TOKEN=...           # gated HuggingFace models"
echo "# platform-specific: BREV_API_TOKEN, ACCESS_KEY/SECRET_KEY/S3_*"
echo "\`\`\`"
echo "Have an env file (~/.tao/secrets.env, ~/.config/tao/.env, or one you"
echo "point the agent at)? Load it with \`set -a; source /path/to/.env; set +a\`."
echo "See the Credentials section of the skill bank README for the full var list."
echo

# ─── 2b. Token-efficient execution — advertise shipped card packs ─────────
# Card packs run application workflows as fresh per-stage sessions (74-91%
# cheaper than in-conversation execution). This block injects the routing
# rule so every session sees it before executing a covered workflow.
if [[ -d "${TAO_ROOT}/skills/applications" ]]; then
  _tao_packs=""
  for _p in "${TAO_ROOT}"/skills/applications/*/cards; do
    [[ -d "$_p" ]] && _tao_packs="${_tao_packs} $(basename "$(dirname "$_p")")"
  done
  if [[ -n "${_tao_packs// /}" ]]; then
    echo "## Token-efficient execution"
    echo
    echo "Card packs are available for:${_tao_packs}. Before executing one of these"
    echo "workflows stage-by-stage in this conversation, prefer the pack's"
    echo "\`cards/driver.sh\` (fresh per-stage sessions; read the pack README for"
    echo "required env). The \`tao-launch-workflow\` gate still applies in full —"
    echo "the driver launch is the side-effecting action it reviews. Framework:"
    echo "\`skills/core/tao-token-efficient-execution\`."
    echo
  fi
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
