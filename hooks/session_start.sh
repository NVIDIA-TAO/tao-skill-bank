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
#      Credentials arrive from the user's own shell or from a user-approved env
#      file the agent sources; this hook never reads or prints values.
#   3. Surface clear setup hints if docker is missing.
#
# This hook does NOT install Python packages. The bank runs SDK-free over native
# platform CLIs; the one wheel-dependent skill, skills/applications/tao-run-automl,
# installs nvidia-tao-automl lazily via its Preflight block.

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

# ─── 1b. Make versions.yaml + skill bank discoverable to the helper scripts ──
# The bank's scripts (resolve_tao_image.py, tao_job_record.py) and templates/
# read $TAO_SKILL_BANK_PATH to find versions.yaml. Plugin-installed users need
# this set to resolve container_image keys like `tao_toolkit.pyt`.
if [[ -n "${CLAUDE_PLUGIN_ROOT:-}" && -n "${CLAUDE_ENV_FILE:-}" ]]; then
  echo "export TAO_SKILL_BANK_PATH=\"${CLAUDE_PLUGIN_ROOT}\"" >> "$CLAUDE_ENV_FILE"
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
