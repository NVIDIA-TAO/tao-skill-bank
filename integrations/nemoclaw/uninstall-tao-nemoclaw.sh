#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Remove TAO capability from a NemoClaw sandbox — the inverse of
# setup-tao-nemoclaw.sh.
#
# Undoes, in reverse order: the tao-mcp network policy, the openclaw.json MCP
# registration and TAO_SKILL_BANK_PATH, the appended AGENTS.md block, the
# sandbox-side skill tree and its symlinks, and the host MCP server process.
#
# Workspace DATA IS NEVER TOUCHED. Datasets, results, and checkpoints under the
# workspace root survive. --purge-bank additionally removes the cloned skill
# bank and the server log, which setup created.
#
# Usage:
#   ./uninstall-tao-nemoclaw.sh <sandbox-name> [workspace-root] [--purge-bank] [--yes]
#
# Works when the sandbox is already destroyed: sandbox-side steps are skipped
# and the orphaned host server is still stopped.
#
# Run this on the NemoClaw HOST, in a login shell (nemoclaw on PATH).
set -euo pipefail

SB=""; WORKSPACE=""; PURGE_BANK=0; ASSUME_YES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --purge-bank) PURGE_BANK=1 ;;
    --yes|-y)     ASSUME_YES=1 ;;
    -h|--help)    sed -n '5,22p' "$0"; exit 0 ;;
    -*)           printf 'unknown flag: %s\n' "$1" >&2; exit 2 ;;
    *)            if [ -z "$SB" ]; then SB="$1"; elif [ -z "$WORKSPACE" ]; then WORKSPACE="$1"; fi ;;
  esac
  shift
done
[ -n "$SB" ] || { echo "usage: uninstall-tao-nemoclaw.sh <sandbox-name> [workspace-root] [--purge-bank] [--yes]" >&2; exit 2; }
WORKSPACE="${WORKSPACE:-$HOME/tao-workspace}"
# Normalize: the server's own --workspace-root is an absolute, symlink-resolved
# path, and the match below is textual, so a trailing slash or a relative path
# would silently fail to find it.
[ -d "$WORKSPACE" ] && WORKSPACE="$(cd "$WORKSPACE" && pwd -P)"
PORT=9901
SERVER="$(cd "$(dirname "$0")" && pwd)/server.py"

log()  { printf '\033[1;32m[tao-nemoclaw]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[tao-nemoclaw] WARN:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[tao-nemoclaw] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

command -v nemoclaw >/dev/null || die "nemoclaw not on PATH (use a login shell)"
command -v docker   >/dev/null || die "docker not on PATH"

# ── 0. Resolve the sandbox container, if it still exists ──────────────────────
# Same UUID-suffixed match as setup. A missing container is not an error: the
# sandbox may already be destroyed, leaving only the host server to clean up.
CID=$(docker ps --format '{{.ID}} {{.Names}}' \
      | awk -v p="openshell-${SB}-" '$2 ~ "^"p {print $1; exit}' || true)
if [ -n "$CID" ]; then
  log "sandbox=$SB container=$CID workspace=$WORKSPACE"
else
  warn "no running container for sandbox '$SB' — skipping sandbox-side cleanup"
fi

# ── 0b. Identify the server process precisely ─────────────────────────────────
# Setup matches on `pgrep -f "$(basename "$SERVER")"`, i.e. the bare string
# "server.py", which collides with any unrelated process of that name. Removal
# must be exact, but NOT keyed to this script's own path: setup logs "already
# running" and reuses a server started from any other checkout, so the live
# process often runs different files than the ones next to this script. What
# identifies the instance is (port, workspace root) — match on those, and say
# so when the code is served from somewhere else.
PIDS=""; SEEN_SRC=""
for pid in $(pgrep -f 'integrations/nemoclaw/server\.py' 2>/dev/null || true); do
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
  case "$cmd" in *"--port $PORT"*) ;; *) continue ;; esac
  case "$cmd" in
    *"--workspace-root $WORKSPACE"*) PIDS="$PIDS $pid" ;;
    *) continue ;;
  esac
  # The `uv run` parent and its python child both match; report the checkout
  # once. Pick the server.py argument itself rather than a positional guess —
  # `uv run --with … python /path/server.py` has it in a different slot.
  case "$cmd" in
    *"$SERVER"*) continue ;;
  esac
  src=""
  for w in $cmd; do
    case "$w" in */integrations/nemoclaw/server.py) src="$w" ;; esac
  done
  [ -n "$src" ] || continue
  case " $SEEN_SRC " in *" $src "*) continue ;; esac
  SEEN_SRC="$SEEN_SRC $src"
  warn "this workspace is served from a different checkout: $src"
  warn "  (setup reuses any running server.py — the files next to this script are not the ones running)"
done
PIDS="${PIDS# }"

# ── 1. Confirm ────────────────────────────────────────────────────────────────
echo
echo "  About to remove TAO from sandbox '$SB':"
echo "    - tao-mcp network policy preset"
echo "    - openclaw.json: mcp.servers.tao + env.TAO_SKILL_BANK_PATH"
echo "    - /sandbox/tao-skills-external and its skill symlinks"
echo "    - the appended AGENTS.md block (a .bak is kept)"
[ -n "$PIDS" ] && echo "    - host MCP server pid(s):$PIDS" || echo "    - host MCP server: none matching this workspace"
[ "$PURGE_BANK" = 1 ] && echo "    - $WORKSPACE/tao-skills-external and tao-mcp-server.log  (--purge-bank)"
echo "  Datasets and results under $WORKSPACE are NOT touched."
echo
if [ "$ASSUME_YES" != 1 ]; then
  printf "  Proceed? [y/N] "; read -r reply
  case "$reply" in [yY]|[yY][eE][sS]) ;; *) die "aborted" ;; esac
fi

# ── 2. Remove the network policy preset ───────────────────────────────────────
if [ -n "$CID" ]; then
  if nemoclaw "$SB" policy remove tao-mcp --yes >/dev/null 2>&1; then
    log "policy preset tao-mcp removed"
  else
    warn "policy preset tao-mcp not removed (already absent?) — check: nemoclaw $SB policy list"
  fi
fi

# ── 3. Clean openclaw.json ────────────────────────────────────────────────────
# Drops only what setup added. tools.profile is deliberately LEFT AS IS: setup
# overwrote it without recording the prior value, so restoring it would be a
# guess. chmod 660 for the same reason setup needs it — 600 trips OpenShell's
# GATEWAY_UNSAFE_CONFIG_PATH check on the next gateway restart.
if [ -n "$CID" ]; then
  nemoclaw "$SB" exec --stdin -- python3 <<'PY'
import json, os
p = "/sandbox/.openclaw/openclaw.json"
d = json.load(open(p))
changed = []
if d.get("mcp", {}).get("servers", {}).pop("tao", None) is not None:
    changed.append("mcp.servers.tao")
    if not d["mcp"].get("servers"):
        d["mcp"].pop("servers", None)
    if not d["mcp"]:
        d.pop("mcp", None)
if d.get("env", {}).pop("TAO_SKILL_BANK_PATH", None) is not None:
    changed.append("env.TAO_SKILL_BANK_PATH")
    if not d["env"]:
        d.pop("env", None)
if changed:
    json.dump(d, open(p, "w"), indent=2)
    os.chmod(p, 0o660)
    print("openclaw.json: removed " + ", ".join(changed))
else:
    print("openclaw.json: nothing to remove")
if d.get("tools", {}).get("profile") == "coding":
    print('note: tools.profile is still "coding" (set by setup; prior value unknown) — '
          'clear it by hand if this sandbox should not have exec/fs/subagents')
PY
fi

# ── 4. Strip the appended AGENTS.md block ─────────────────────────────────────
# Setup appends the whole file after a blank line, guarded by a heading grep, so
# removing from the heading to EOF is the exact inverse. A .bak is kept because
# the user may have written their own content after it.
if [ -n "$CID" ]; then
  nemoclaw "$SB" exec --stdin -- python3 <<'PY'
import os
p = "/sandbox/.openclaw/workspace/AGENTS.md"
try:
    text = open(p).read()
except FileNotFoundError:
    print("AGENTS.md: absent"); raise SystemExit(0)
i = text.find("# TAO on NemoClaw")
if i < 0:
    print("AGENTS.md: no TAO block found"); raise SystemExit(0)
open(p + ".bak", "w").write(text)
rest = text[:i].rstrip()
open(p, "w").write(rest + "\n" if rest else "")
os.chmod(p, 0o660)
print("AGENTS.md: TAO block removed (backup at AGENTS.md.bak)")
PY
fi

# ── 5. Remove the sandbox skill tree and its symlinks ─────────────────────────
# Two different privilege levels, because setup used two:
#
#   - the symlinks were made by `nemoclaw exec` (sandbox user), so remove them
#     the same way. Only links resolving into /sandbox/tao-skills-external go,
#     so skills installed from another source survive.
#   - the tree itself arrived by `docker cp`, which preserves the HOST uid — it
#     lands root-owned, and the sandbox user cannot delete it ("Permission
#     denied" on .gitignore, templates/, etc). It has to come out the way it
#     went in: as root, through the docker socket.
#
# This is the one legitimate use of `docker exec` here. It is a host-side admin
# operation on files, not a test of agent-visible behavior — for anything the
# agent must be able to see or reach, always use `nemoclaw exec`, because
# docker exec escapes OpenShell's policy attribution and reports false results.
if [ -n "$CID" ]; then
  nemoclaw "$SB" exec --stdin -- bash <<'EOS'
set -eu
d=/sandbox/.openclaw/skills
n=0
if [ -d "$d" ]; then
  for l in "$d"/*; do
    [ -L "$l" ] || continue
    case "$(readlink "$l")" in
      /sandbox/tao-skills-external/*) rm -f "$l"; n=$((n+1)) ;;
    esac
  done
fi
echo "sandbox: removed $n skill symlink(s)"
EOS
  if docker exec -u 0 "$CID" rm -rf /sandbox/tao-skills-external /tmp/tao-AGENTS.md 2>/dev/null; then
    log "sandbox: removed /sandbox/tao-skills-external (as root, matching docker cp)"
  else
    warn "could not remove /sandbox/tao-skills-external — remove it by hand:"
    warn "    docker exec -u 0 $CID rm -rf /sandbox/tao-skills-external /tmp/tao-AGENTS.md"
  fi
fi

# ── 6. Reload the gateway so OpenClaw drops the MCP tools ─────────────────────
if [ -n "$CID" ]; then
  nemoclaw "$SB" gateway restart >/dev/null 2>&1 \
    && log "gateway restarted" \
    || warn "gateway restart failed — run: nemoclaw $SB gateway restart"
fi

# ── 7. Stop the host MCP server ───────────────────────────────────────────────
# setsid put the server in its own session, so signal the process group to take
# the `uv run` parent and the python child together. TERM, then KILL.
if [ -n "$PIDS" ]; then
  for pid in $PIDS; do
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    [ -n "$pgid" ] || continue
    kill -TERM "-$pgid" 2>/dev/null || true
  done
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    still=""
    for pid in $PIDS; do [ -d "/proc/$pid" ] && still="$still $pid"; done
    [ -z "$still" ] && break
    sleep 1
  done
  for pid in $PIDS; do
    [ -d "/proc/$pid" ] || continue
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    [ -n "$pgid" ] && kill -KILL "-$pgid" 2>/dev/null || true
  done
  log "MCP server stopped (pid(s):$PIDS)"
else
  log "no MCP server to stop for workspace $WORKSPACE"
fi

# ── 8. Optionally remove what setup created inside the workspace ──────────────
if [ "$PURGE_BANK" = 1 ]; then
  rm -rf "$WORKSPACE/tao-skills-external"
  rm -f  "$WORKSPACE/tao-mcp-server.log"
  log "purged $WORKSPACE/tao-skills-external and tao-mcp-server.log"
else
  [ -d "$WORKSPACE/tao-skills-external" ] \
    && log "kept $WORKSPACE/tao-skills-external (pass --purge-bank to remove)"
fi

# ── 9. Host firewall: print, never mutate ─────────────────────────────────────
# Symmetric with setup, which only ever prints the add rule. Removing a host
# firewall rule is a change the operator should make knowingly — and the rule
# may still be needed by another sandbox on the same bridge.
if command -v ufw >/dev/null 2>&1 && sudo -n ufw status 2>/dev/null | grep -q "$PORT/tcp"; then
  log "a ufw rule for port $PORT remains. If no other sandbox needs it:"
  log "    sudo ufw status numbered | grep $PORT   # find the number, then:"
  log "    sudo ufw --force delete <number>"
fi

# ── 10. Verify ────────────────────────────────────────────────────────────────
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  warn "something is still listening on port $PORT — another sandbox's server?"
  warn "    ss -ltnp | grep :$PORT"
else
  log "✓ port $PORT clear"
fi
log "Done. TAO removed from '$SB'."
