#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# End-to-end verification: skill bank works standalone (no tao_sdk import required).
#
# Requires: NVIDIA driver 580 + CUDA 13.0 + docker + nvidia-container-toolkit 1.19.0 + NGC login + 24GB+ VRAM GPU.
# Does NOT require: pip install tao-sdk.
#
# Pass = every step returns 0. Fail = any step returns non-zero.

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== 1. Confirm Python has no tao_sdk (validates 'no SDK required')"
if python3 -c "import tao_sdk" 2>/dev/null; then
  echo "FAIL: tao_sdk is importable. Re-run in a venv that lacks it:"
  echo "      python -m venv /tmp/standalone && source /tmp/standalone/bin/activate"
  exit 1
else
  echo "  OK: tao_sdk not importable — standalone env confirmed"
fi

echo
echo "=== 2. NVIDIA GPU runtime + Docker + NGC login"
bash skills/platform/tao-setup-nvidia-gpu-host/scripts/setup-nvidia-gpu-host.sh --backend docker --check-only
docker --version
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi >/dev/null && echo "  OK: GPU + toolkit"
grep -q 'nvcr.io' ~/.docker/config.json 2>/dev/null && echo "  OK: NGC login present" || {
  echo "  FAIL: not logged into nvcr.io. Run: echo \$NGC_KEY | docker login nvcr.io -u '\$oauthtoken' --password-stdin"
  exit 1
}

echo
echo "=== 3. Read tao-train-visual-changenet metadata from the skill bank"
INFO_FILE=""
for f in skills/models/tao-train-visual-changenet/references/skill_info.yaml \
         skills/models/tao-train-visual-changenet/references/model_info.yaml; do
  [ -f "$f" ] && INFO_FILE="$f" && break
done

if [ -z "$INFO_FILE" ]; then
  echo "  tao-train-visual-changenet has no references/skill_info.yaml or references/model_info.yaml"
  echo "  Skipping metadata read — agent must construct from SKILL.md prose alone."
else
  IMAGE=$(sed -n 's/^[[:space:]]*container_image:[[:space:]]*//p' "$INFO_FILE" | head -n 1)
  echo "  container_image: ${IMAGE:-<not declared>}"
fi

echo
echo "=== 4. Run validate-skills.sh"
./scripts/validate-skills.sh

echo
echo "✓ Standalone verification passed."
echo
echo "Next: in a fresh Claude Code session with only the plugin installed (no tao-sdk),"
echo "ask the agent to run a real Visual ChangeNet inference (skill name"
echo "tao-train-visual-changenet). It should construct docker run from SKILL.md"
echo "(and references/ if present), invoke via Bash, and produce output — no Python"
echo "imports required."
