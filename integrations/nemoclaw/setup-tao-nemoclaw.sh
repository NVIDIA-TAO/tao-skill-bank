#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Set up TAO capability on a NemoClaw sandbox via a host-side MCP server.
#
# The sandbox agent gets typed tools (list/read/write workspace, run/monitor/
# stop TAO containers) over the OpenShell host bridge. Docker, the GPU, and NGC
# credentials stay on the host — the agent never touches them.
#
# Usage:
#   ./setup-tao-nemoclaw.sh <sandbox-name> [workspace-root]
#
# Prerequisites (not handled here):
#   - NemoClaw installed, and <sandbox-name> already onboarded and Ready.
#   - Host logged into NGC:  docker login nvcr.io   (user: $oauthtoken)
#   - uv on PATH (https://astral.sh/uv).
#
# Run this on the NemoClaw HOST, in a login shell (nemoclaw on PATH).
set -euo pipefail

SB="${1:?usage: setup-tao-nemoclaw.sh <sandbox-name> [workspace-root]}"
WORKSPACE="${2:-$HOME/tao-workspace}"
PORT=9901
SERVER="$(cd "$(dirname "$0")" && pwd)/server.py"
# ── Skill bank source (three modes, priority order) ──────────────────────────
#   1. SKILL_LOCAL=<path> — copy a local working tree (e.g. an SQA checkout with
#      un-pushed changes) instead of cloning. Wins over the repo modes if set.
#   2. SKILL_REPO + SKILL_REF (alias TAO_RELEASE) — clone/checkout repo @ ref.
#
# The bank's dotted image keys resolve to the installed checkout's versions.yaml,
# so the source determines which TAO images every skill runs.
#
# Default (published): public GitHub distribution on `main`.
# Internal SQA / pre-release — release/7.x branches live on the internal GitLab
# repo (GitHub has only main + tags), so set BOTH:
#   export SKILL_REPO="ssh://git@gitlab-master.nvidia.com:12051/nvidia-tao-toolkit/tao-skills-external.git"
#   export SKILL_REF="release/7.1.0"          # or TAO_RELEASE=release/7.1.0
# …or skip the network entirely and use a local checkout directly:
#   export SKILL_LOCAL="$HOME/tao-skills-external"
SKILL_LOCAL="${SKILL_LOCAL:-}"
SKILL_REPO="${SKILL_REPO:-https://github.com/NVIDIA-TAO/tao-skills-bank}"
SKILL_REF="${SKILL_REF:-${TAO_RELEASE:-main}}"

log() { printf '\033[1;32m[tao-nemoclaw]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[tao-nemoclaw] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

command -v nemoclaw >/dev/null || die "nemoclaw not on PATH (use a login shell)"
command -v docker   >/dev/null || die "docker not on PATH"
command -v uv       >/dev/null || die "uv not on PATH"
[ -f "$SERVER" ] || die "server.py not found next to this script"

# ── 0. Resolve the sandbox container and its docker-bridge gateway ────────────
# Name-scope the filter: a bare 'openshell' matches every sandbox's container,
# and 'openshell-<sb>' can still match 'openshell-<sb>-local' — so match the
# UUID-suffixed form exactly.
CID=$(docker ps --format '{{.ID}} {{.Names}}' \
      | awk -v p="openshell-${SB}-" '$2 ~ "^"p {print $1; exit}')
[ -n "$CID" ] || die "no running container for sandbox '$SB' (nemoclaw list?)"
# The sandbox reaches the host at this gateway IP (== host.openshell.internal).
# Binding the server here (not 0.0.0.0) keeps it off the LAN.
GW=$(docker inspect "$CID" -f '{{range .NetworkSettings.Networks}}{{.Gateway}}{{end}}')
[ -n "$GW" ] || die "could not resolve bridge gateway for '$SB'"
log "sandbox=$SB container=$CID bridge-gateway=$GW workspace=$WORKSPACE"

# ── 1. Materialize the skill bank + resolve the tao_exec shell image ──────────
# Do this before the server starts so TAO_SHELL_IMAGE (the pyt image tao_exec
# runs its CPU shell in) can be read from the bank's versions.yaml. Clone the
# bank INTO the workspace (not /tmp) so tao_run containers also see it at
# /data/tao-skills-external — every skill's scripts, references, and
# versions.yaml are then runnable in-container without the agent copying files
# through its context. The same tree is copied into the sandbox in step 3.
mkdir -p "$WORKSPACE"
BANK="$WORKSPACE/tao-skills-external"
if [ -n "$SKILL_LOCAL" ]; then
  # Copy a local working tree into the workspace (drop its .git — the sandbox only
  # needs the files). Lets SQA test un-pushed release/7.x skills without a push.
  [ -d "$SKILL_LOCAL" ] || die "SKILL_LOCAL is not a directory: $SKILL_LOCAL"
  src="$(cd "$SKILL_LOCAL" && pwd -P)"
  if [ "$src" != "$(cd "$WORKSPACE" && pwd -P)/tao-skills-external" ]; then
    rm -rf "$BANK"; cp -a "$src" "$BANK"; rm -rf "$BANK/.git"
  fi
  log "skill bank: local tree $src ($(git -C "$src" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'no-git'))"
elif [ -d "$BANK/.git" ]; then
  git -C "$BANK" fetch -q --depth 1 "$SKILL_REPO" "$SKILL_REF" \
    && git -C "$BANK" checkout -q -B "$SKILL_REF" FETCH_HEAD
  log "skill bank: $SKILL_REPO @ $SKILL_REF"
else
  git clone --depth 1 -b "$SKILL_REF" "$SKILL_REPO" "$BANK"
  log "skill bank: $SKILL_REPO @ $SKILL_REF"
fi

# tao_exec runs its CPU shell in the bank's pinned pyt image, so the shell's
# Python / huggingface_hub / ngcsdk match the skills. Resolve it from
# versions.yaml unless the caller pinned TAO_SHELL_IMAGE; without this the
# agent's first tao_exec fails with "no shell image available".
if [ -z "${TAO_SHELL_IMAGE:-}" ]; then
  TAO_SHELL_IMAGE=$(awk '/^[[:space:]]*pyt:[[:space:]]*nvcr\.io\// {print $2; exit}' "$BANK/versions.yaml" 2>/dev/null || true)
fi
if [ -n "${TAO_SHELL_IMAGE:-}" ]; then
  export TAO_SHELL_IMAGE; log "tao_exec shell image: $TAO_SHELL_IMAGE"
else
  log "WARN: no pyt image in $BANK/versions.yaml — export TAO_SHELL_IMAGE so tao_exec can start its shell"
fi

# ── 2. Ensure the MCP server is running on the host ───────────────────────────
if pgrep -f "tao-mcp.*server.py|$(basename "$SERVER")" >/dev/null; then
  log "MCP server already running"
else
  log "starting MCP server (tokenless, bound to $GW:$PORT — bridge only)"
  # Tokenless is safe because the bind is the docker bridge IP, not the LAN.
  # The server inherits TAO_SHELL_IMAGE (resolved above) for tao_exec.
  ( unset TAO_MCP_TOKEN
    setsid nohup uv run --with mcp --with uvicorn python "$SERVER" \
      --workspace-root "$WORKSPACE" --host "$GW" --port "$PORT" \
      > "$WORKSPACE/tao-mcp-server.log" 2>&1 & )
  sleep 8
  pgrep -f "$(basename "$SERVER")" >/dev/null \
    || die "server failed to start — see $WORKSPACE/tao-mcp-server.log"
fi

# ── 3. Install the TAO skills into the sandbox ────────────────────────────────
docker cp "$BANK" "$CID":/sandbox/    # -> /sandbox/tao-skills-external (agent reads skills)
# OpenClaw discovers skills one level below its skills dir; the bank nests them.
nemoclaw "$SB" exec -- bash -c 'cd /sandbox/tao-skills-external && find skills -name SKILL.md | while read -r f; do d=$(dirname "$f"); ln -sfn "/sandbox/tao-skills-external/$d" "/sandbox/.openclaw/skills/$(basename "$d")"; done'
log "skills installed (workspace: $BANK ; sandbox: /sandbox/tao-skills-external)"

# ── 4. Register MCP server + skill-bank path + enable orchestration tools ─────
# Edits openclaw.json directly (openclaw config set refuses to run in-sandbox).
# The "coding" tool profile enables exec + fs + subagents so the agent can run
# multi-step workflows (e.g. spawn the DEFT report subagent). All three operate
# INSIDE the sandbox (exec.host defaults to "auto" = sandbox when available,
# always true here) — host execution stays exclusively behind the MCP server.
# chmod 660 is REQUIRED: json.dump would leave the file 600, which trips
# OpenShell's GATEWAY_UNSAFE_CONFIG_PATH check on the next gateway restart.
nemoclaw "$SB" exec --stdin -- python3 <<PY
import json, os
p = "/sandbox/.openclaw/openclaw.json"; d = json.load(open(p))
d.setdefault("env", {})["TAO_SKILL_BANK_PATH"] = "/sandbox/tao-skills-external"
d.setdefault("mcp", {}).setdefault("servers", {})["tao"] = {
    "type": "http", "url": "http://host.openshell.internal:${PORT}/mcp"}
d.setdefault("tools", {})["profile"] = "coding"   # exec + fs + subagents (sandbox-scoped)
json.dump(d, open(p, "w"), indent=2)
os.chmod(p, 0o660)
print("openclaw.json configured (mcp + coding tools profile)")
PY

# ── 4b. Give the agent runtime awareness (AGENTS.md in its workspace) ─────────
# Without this the agent falls into the skill bank's default flow and asks which
# platform to use; the note tells it to use the tao MCP tools on the host.
# Appended (idempotent via the heading grep) so an existing AGENTS.md is kept.
AGENTS_SRC="$(cd "$(dirname "$0")" && pwd)/AGENTS.md"
if [ -f "$AGENTS_SRC" ]; then
  docker cp "$AGENTS_SRC" "$CID":/tmp/tao-AGENTS.md
  # Pipe via stdin — OpenShell exec rejects newlines in argv, so no `bash -c '<multiline>'`.
  nemoclaw "$SB" exec --stdin -- bash <<'EOS'
dst=/sandbox/.openclaw/workspace/AGENTS.md
mkdir -p /sandbox/.openclaw/workspace
grep -q "TAO on NemoClaw" "$dst" 2>/dev/null || { printf "\n" >> "$dst"; cat /tmp/tao-AGENTS.md >> "$dst"; }
chmod 660 "$dst" 2>/dev/null || true
EOS
  log "runtime-awareness AGENTS.md installed"
fi

# ── 5. Allow the sandbox to reach the host bridge port ────────────────────────
# access:full alone is denied by OpenShell's SSRF guard for private gateway IPs;
# allowed_ips must explicitly permit the docker-bridge range. rules cover the
# MCP streamable-HTTP verbs (GET stream, POST call, DELETE session end).
POLICY="${TMPDIR:-/tmp}/tao-mcp-policy.$$.yaml"   # nemoclaw requires a .yaml/.yml extension
cat > "$POLICY" <<EOF
preset:
  name: tao-mcp
  description: "TAO MCP server on host via OpenShell bridge"
network_policies:
  tao_mcp:
    name: tao_mcp
    endpoints:
      - host: host.openshell.internal
        port: ${PORT}
        protocol: rest
        enforcement: enforce
        allowed_ips: [10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16]
        rules:
          - allow: { method: GET,    path: "/**" }
          - allow: { method: POST,   path: "/**" }
          - allow: { method: DELETE, path: "/**" }
    binaries:
      - { path: /usr/local/bin/openclaw }
      - { path: /usr/local/bin/node }
      - { path: /usr/bin/node }
      - { path: /usr/bin/curl }
      - { path: /usr/bin/python3 }
EOF
# --yes is REQUIRED non-interactively: without it policy-add hangs with no output.
nemoclaw "$SB" policy-add --from-file "$POLICY" --yes
rm -f "$POLICY"

# ── 6. Reload the sandbox gateway so OpenClaw picks up env + MCP tools ─────────
if ! nemoclaw "$SB" gateway restart 2>&1 | tee /tmp/tao-gw.log | grep -q "restarted"; then
  if grep -q "GATEWAY_UNSAFE_CONFIG_PATH" /tmp/tao-gw.log; then
    log "config-path guard tripped — running doctor --fix and retrying"
    nemoclaw "$SB" doctor --fix >/dev/null 2>&1 || true
    nemoclaw "$SB" gateway restart
  fi
fi

# ── 7. Verify the bridge reaches the server ───────────────────────────────────
CODE=$(nemoclaw "$SB" exec -- curl -sS --max-time 8 -o /dev/null \
       -w '%{http_code}' "http://host.openshell.internal:${PORT}/mcp" 2>/dev/null || echo 000)
case "$CODE" in
  400|406|200) log "✓ bridge OK (HTTP $CODE — server answered)";;
  403) die "bridge blocked by policy (HTTP 403) — check policy-list";;
  *)   die "server unreachable (HTTP $CODE) — check the server bind matches gateway $GW";;
esac

log "Done. In the agent (nemoclaw $SB connect -> openclaw tui), ask:"
log "  'What MCP tools do you have?'  -> expect tao_ls/read/write/pull/run/list/status/logs/stop/rm/cleanup_results"
log "Put datasets under $WORKSPACE/<name>/ ; the agent sees them via tao_ls."
