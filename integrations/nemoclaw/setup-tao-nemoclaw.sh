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

# --restart-server forces a server restart even when a healthy one is already
# serving this workspace. Setup restarts automatically when the running server
# is unusable (wrong workspace, no credentials), so this is only needed after
# editing server.py.
RESTART_SERVER=0
_pos=()
for _a in "$@"; do
  case "$_a" in
    --restart-server) RESTART_SERVER=1 ;;
    *) _pos+=("$_a") ;;
  esac
done
set -- ${_pos[@]+"${_pos[@]}"}

SB="${1:?usage: setup-tao-nemoclaw.sh <sandbox-name> [workspace-root] [--restart-server]}"
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
#   export SKILL_REPO="ssh://git@gitlab-master.nvidia.com:12051/nvidia-tao-toolkit/tao-skill-bank.git"
#   export SKILL_REF="release/7.1.0"          # or TAO_RELEASE=release/7.1.0
# …or skip the network entirely and use a local checkout directly:
#   export SKILL_LOCAL="$HOME/tao-skill-bank"
SKILL_LOCAL="${SKILL_LOCAL:-}"
SKILL_REPO="${SKILL_REPO:-https://github.com/NVIDIA-TAO/tao-skill-bank}"
SKILL_REF="${SKILL_REF:-${TAO_RELEASE:-main}}"

log() { printf '\033[1;32m[tao-nemoclaw]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[tao-nemoclaw] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# Who owns the bridge port. The only trustworthy way to find "the server":
# process names collide, ports do not.
_port_owner() {
  _p=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
  [ -n "${_p:-}" ] || return 1
  printf '%s\n' "$_p"
}

# Value of a flag in a running process's argv, e.g. _server_arg 123 --workspace-root
_server_arg() {
  tr '\0' '\n' < "/proc/$1/cmdline" 2>/dev/null \
    | awk -v f="$2" 'p{print;exit} $0==f{p=1}'
}

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

# ── 0b. Host firewall preflight ───────────────────────────────────────────────
# The sandbox reaches this server over the docker bridge, so its packets land on
# the host's INPUT chain. A default-deny host firewall — common on corporate /
# IT-managed Linux, while ufw ships *inactive* on stock Ubuntu and DGX Spark,
# which is why this never reproduces there — silently DROPs them. The symptom
# surfaces 150 lines later in step 7 as "HTTP 000", which reads as a bad bind
# even though the bind is correct. Warn here, with the exact rule.
#
# Deliberately does NOT run sudo: mutating a host firewall is a change the
# operator should make knowingly.
FW_HINT="sudo ufw allow from ${GW%.*}.0/16 to $GW port $PORT proto tcp"
if command -v ufw >/dev/null 2>&1 && sudo -n ufw status 2>/dev/null | grep -qi '^Status: active'; then
  if ! sudo -n ufw status 2>/dev/null | grep -q "$GW $PORT/tcp"; then
    log "WARN: ufw is active and no rule permits the sandbox bridge to reach $GW:$PORT."
    log "      The sandbox cannot reach the MCP server until you run:"
    log "        $FW_HINT"
  fi
fi

# ── 0c. Credential preflight ──────────────────────────────────────────────────
# The server forwards secrets into containers BY NAME, reading values from its
# own environment at launch — so a server started without them can never fetch
# a gated HuggingFace repo or an nvcr.io image, and nothing later can fix it
# without a restart. These live in ~/.bashrc for most operators, below the
# non-interactive guard, so `ssh host './setup-tao-nemoclaw.sh …'` silently
# starts an unauthenticated server. Empty counts as missing: that is exactly how
# server.py treats it, and `export NGC_KEY="$NGC_API_KEY"` in a non-interactive
# shell produces an empty NGC_KEY that *looks* set.
: "${NGC_KEY:=${NGC_API_KEY:-}}"
export NGC_KEY
_missing=""
[ -n "${HF_TOKEN:-}" ] || _missing="$_missing HF_TOKEN"
[ -n "${NGC_KEY:-}" ]  || _missing="$_missing NGC_KEY/NGC_API_KEY"
if [ -n "$_missing" ]; then
  die "missing credentials:$_missing
  Values are never printed or stored by this script; it only checks presence.
  If they are exported from ~/.bashrc, re-run through a login shell:
      ssh <host> 'bash -ic \"$0 $SB $WORKSPACE\"'"
fi

# ── 0d. Workspace-root sanity ─────────────────────────────────────────────────
# The bridge mounts WORKSPACE at /workspace, and every skill reference treats
# <workspace> as the root that directly contains train/, kpi/, results/ and
# augmentation/. Point this one level too high and the agent spends the run
# resolving paths that do not exist — `results/deft_state.json` missing 31 times
# in a single run, because the real tree was at /workspace/<dataset>/results/.
if [ ! -d "$WORKSPACE/results" ] && [ ! -d "$WORKSPACE/train" ]; then
  _cand=$(find "$WORKSPACE" -maxdepth 2 -type d -name results 2>/dev/null | head -1)
  if [ -n "$_cand" ]; then
    warn_root="$(dirname "$_cand")"
    log "WARN: $WORKSPACE has no train/ or results/, but $warn_root does."
    log "      Skills treat <workspace> as the directory holding train/, kpi/,"
    log "      results/, augmentation/. Consider:  $0 $SB $warn_root"
  fi
fi

# ── 1. Materialize the skill bank + resolve the tao_exec shell image ──────────
# Do this before the server starts so TAO_SHELL_IMAGE (the pyt image tao_exec
# runs its CPU shell in) can be read from the bank's versions.yaml. Clone the
# bank INTO the workspace (not /tmp) so tao_run containers also see it at
# /data/tao-skill-bank — every skill's scripts, references, and
# versions.yaml are then runnable in-container without the agent copying files
# through its context. The same tree is copied into the sandbox in step 3.
mkdir -p "$WORKSPACE"
BANK="$WORKSPACE/tao-skill-bank"
if [ -n "$SKILL_LOCAL" ]; then
  # Copy a local working tree into the workspace (drop its .git — the sandbox only
  # needs the files). Lets SQA test un-pushed release/7.x skills without a push.
  [ -d "$SKILL_LOCAL" ] || die "SKILL_LOCAL is not a directory: $SKILL_LOCAL"
  src="$(cd "$SKILL_LOCAL" && pwd -P)"
  if [ "$src" != "$(cd "$WORKSPACE" && pwd -P)/tao-skill-bank" ]; then
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
_start_server() {
  log "starting MCP server (tokenless, bound to $GW:$PORT — bridge only)"
  # Tokenless is safe because the bind is the docker bridge IP, not the LAN.
  # The server inherits TAO_SHELL_IMAGE (resolved above) for tao_exec.
  ( unset TAO_MCP_TOKEN
    # Pin mcp<2: the server imports mcp.server.fastmcp, which mcp 2.x removed.
    # Unpinned, `uv run --with mcp` resolves 2.x and the server dies on import.
    setsid nohup uv run --with 'mcp<2' --with uvicorn python "$SERVER" \
      --workspace-root "$WORKSPACE" --host "$GW" --port "$PORT" \
      > "$WORKSPACE/tao-mcp-server.log" 2>&1 & )
  sleep 8
  _port_owner >/dev/null \
    || die "server failed to start — see $WORKSPACE/tao-mcp-server.log"
}

_stop_server() {
  _pid=$(_port_owner) || return 0
  _pgid=$(ps -o pgid= -p "$_pid" | tr -d ' ')
  log "stopping MCP server pid=$_pid"
  kill -TERM "-$_pgid" 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do [ -d "/proc/$_pid" ] || break; sleep 1; done
  [ -d "/proc/$_pid" ] && kill -KILL "-$_pgid" 2>/dev/null || true
  return 0
}

# Identify the running server by PORT OWNER, never by `pgrep -f server.py`.
# That basename matches any unrelated process carrying the string — an orphaned
# shell once matched it and made this script report "already running" while
# starting nothing, so setup failed 150 lines later at the bridge check.
#
# Reuse is only safe when the running server serves THIS workspace and has the
# credentials it needs; its environment is frozen at launch, so a server started
# without HF_TOKEN can never fetch a gated model no matter what is exported now.
_cur_pid=$(_port_owner || true)
if [ -n "${_cur_pid:-}" ]; then
  _cur_ws=$(_server_arg "$_cur_pid" --workspace-root)
  _cur_hf=$(tr '\0' '\n' < "/proc/$_cur_pid/environ" 2>/dev/null | cut -d= -f1 | grep -cx 'HF_TOKEN' || true)
  if [ "${RESTART_SERVER:-0}" = 1 ]; then
    log "restarting MCP server (--restart-server)"; _stop_server; _start_server
  elif [ "$_cur_ws" != "$(cd "$WORKSPACE" && pwd -P)" ]; then
    log "running server serves a different workspace (${_cur_ws:-unknown}) — restarting"
    _stop_server; _start_server
  elif [ "${_cur_hf:-0}" = 0 ]; then
    log "running server has no HF_TOKEN — restarting so gated fetches work"
    _stop_server; _start_server
  else
    log "MCP server already running (pid=$_cur_pid, workspace=$_cur_ws)"
  fi
else
  _start_server
fi

# Report which secrets the running server can actually forward. Until this line
# existed there was no way to tell an authenticated server from an unauthenticated
# one without reading /proc — the failure surfaced hours later as an HTTP 401
# inside a model download, which reads like a HuggingFace problem rather than a
# setup one. Names only; values are never printed.
_srv_pid=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2 || true)
if [ -n "${_srv_pid:-}" ] && [ -r "/proc/$_srv_pid/environ" ]; then
  _fwd=$(tr '\0' '\n' < "/proc/$_srv_pid/environ" | cut -d= -f1 \
         | grep -xE 'HF_TOKEN|HUGGING_FACE_HUB_TOKEN|NGC_API_KEY|NGC_CLI_API_KEY|NGC_KEY' \
         | sort | tr '\n' ' ')
  # Group by secret, not by variable name. There are two secrets here; listing
  # NGC_KEY and NGC_API_KEY side by side reads as two credentials when it is one
  # value under the name the skills use and the name the NGC CLI sets.
  _hf=$(printf '%s' "$_fwd"  | tr ' ' '\n' | grep -xE 'HF_TOKEN|HUGGING_FACE_HUB_TOKEN' | paste -sd, -)
  _ngc=$(printf '%s' "$_fwd" | tr ' ' '\n' | grep -xE 'NGC_KEY|NGC_API_KEY|NGC_CLI_API_KEY' | paste -sd, -)
  log "credentials forwarded into containers:"
  log "  HuggingFace   ${_hf:-NONE}"
  log "  NGC           ${_ngc:-NONE}"
  case "${_ngc:-}" in
    *,*) log "                (one key, two names: skills read NGC_KEY, the NGC CLI sets NGC_API_KEY)" ;;
  esac
  if [ -z "${_hf:-}" ]; then
    log "WARN: HF_TOKEN is not in the server environment — gated HuggingFace"
    log "      repos (e.g. Cosmos-Predict2 base weights) will fail with 401."
    log "      Stop the server and re-run this script from a login shell."
  fi
fi

# ── 3. Install the TAO skills into the sandbox ────────────────────────────────
docker cp "$BANK" "$CID":/sandbox/    # -> /sandbox/tao-skill-bank (agent reads skills)
# OpenClaw discovers skills one level below its skills dir; the bank nests them.
nemoclaw "$SB" exec -- bash -c 'cd /sandbox/tao-skill-bank && find skills -name SKILL.md | while read -r f; do d=$(dirname "$f"); ln -sfn "/sandbox/tao-skill-bank/$d" "/sandbox/.openclaw/skills/$(basename "$d")"; done'
log "skills installed (workspace: $BANK ; sandbox: /sandbox/tao-skill-bank)"

# ── 4. Register MCP server + skill-bank path + enable orchestration tools ─────
# Edits openclaw.json directly (openclaw config set refuses to run in-sandbox).
# The "coding" tool profile enables exec + fs + subagents so the agent can run
# multi-step workflows (e.g. spawn the DEFT report subagent). All three operate
# INSIDE the sandbox (exec.host defaults to "auto" = sandbox when available,
# always true here) — host execution stays exclusively behind the MCP server.
# chmod 660 is REQUIRED: json.dump would leave the file 600, which trips
# OpenShell's GATEWAY_UNSAFE_CONFIG_PATH check on the next gateway restart.
#
# Model capabilities (4a) are corrected in the same pass. NemoClaw cannot probe
# a custom endpoint, so for provider `anthropicCompatible` it writes
# conservative guesses — reasoning=false, maxTokens=4096, contextWindow=131072
# — with no warning. A DEFT-class loop is dozens of stages of heavy tool use and
# dies on those: 4096 truncates a turn mid-loop, and 131072 overflows context
# after ~110 tool calls, surfacing only as "Agent couldn't generate a response".
# Values below verified against an Anthropic-compatible Opus 5 endpoint on 2026-08-10:
#   - max_tokens ceiling is exactly 128000 (the 400 names it), and 128000 is
#     accepted on a NON-streaming request, so it needs no streaming fallback;
#   - a single request with 432,015 input tokens returned HTTP 200;
#   - thinking blocks come back signed, so reasoning is real.
# Override per-host with MODEL_CONTEXT_WINDOW / MODEL_MAX_TOKENS /
# MODEL_REASONING if you point this at a different endpoint or model.
MODEL_CONTEXT_WINDOW="${MODEL_CONTEXT_WINDOW:-1000000}"
MODEL_MAX_TOKENS="${MODEL_MAX_TOKENS:-128000}"
MODEL_REASONING="${MODEL_REASONING:-true}"
# Level used when reasoning is enabled: off|minimal|low|medium|high|xhigh.
MODEL_THINKING="${MODEL_THINKING:-medium}"
# Turn cap for an orchestration turn and for each heartbeat turn. The stock 600s
# truncates stages that legitimately run longer.
AGENT_TIMEOUT_SECONDS="${AGENT_TIMEOUT_SECONDS:-3600}"
# Heartbeat cadence, and so the worst-case recovery time after an interrupted
# turn. OpenClaw defaults to 30m, which for an orchestration means a run can sit
# idle for up to half an hour before anything wakes it.
HEARTBEAT_EVERY="${HEARTBEAT_EVERY:-2m}"
case "$MODEL_REASONING" in true) MODEL_REASONING_PY=True ;; *) MODEL_REASONING_PY=False ;; esac
nemoclaw "$SB" exec --stdin -- python3 <<PY
import json, os
p = "/sandbox/.openclaw/openclaw.json"; d = json.load(open(p))
d.setdefault("env", {})["TAO_SKILL_BANK_PATH"] = "/sandbox/tao-skill-bank"
d.setdefault("mcp", {}).setdefault("servers", {})["tao"] = {
    "type": "http", "url": "http://host.openshell.internal:${PORT}/mcp"}
d.setdefault("tools", {})["profile"] = "coding"   # exec + fs + subagents (sandbox-scoped)

# Expose the tao_* MCP tools directly instead of behind Tool Search's generic
# tool_call wrapper. With the wrapper on, a whole class of calls fails
# validation with 'args: must be object' when the model serialises arguments as
# a Python-repr dict rather than JSON — observed 49 times in a single DEFT run,
# each one a silent retry the operator experiences as the agent stalling. The
# tao surface is 12 tools; it does not need progressive disclosure.
d["tools"].setdefault("toolSearch", {})["enabled"] = False

# 4a. Correct the guessed capabilities. Scoped to Claude Opus entries so a
# sandbox onboarded against a smaller model is left alone; widen deliberately
# rather than by accident. Declaring a window LARGER than the endpoint accepts
# is worse than one too small — the agent stops compacting and starts getting
# hard 400s instead — so only raise these against a verified endpoint.
patched = []
for prov in d.get("models", {}).get("providers", {}).values():
    for m in prov.get("models", []) or []:
        if "opus" not in "{} {}".format(m.get("id", ""), m.get("name", "")).lower():
            continue
        m["contextWindow"] = ${MODEL_CONTEXT_WINDOW}
        m["maxTokens"] = ${MODEL_MAX_TOKENS}
        m["reasoning"] = "${MODEL_REASONING}" == "true"
        patched.append(m.get("id", "?"))

# 4b. Turn off memory search. It embeds through an OpenAI-style provider, and
# NemoClaw's routed inference has no embeddings route — from the sandbox,
# https://inference.local/v1/models answers 200 but /v1/embeddings returns
# {"error":"no compatible inference route available"}. So on any sandbox whose
# brain is an Anthropic-compatible endpoint, memory_search cannot work: it fails
# with 'No API key found for provider "openai"' the first time the agent reaches
# for it, mid-task. Wiring it up would mean putting a live embeddings credential
# inside the sandbox, which is exactly the property this integration exists to
# avoid. Off is honest; the fix belongs upstream in the router.
d.setdefault("agents", {}).setdefault("defaults", {}) \
 .setdefault("memorySearch", {})["enabled"] = False

# 4c. Make the sandbox able to carry a long orchestration unattended.
#
# An orchestration workflow (DEFT AOI, AutoML, any multi-stage loop) runs for
# hours across a dozen stages. Two defaults stop it dead, and both fail silently:
#
#   * agents.defaults.timeoutSeconds defaults to 600. A single stage can exceed
#     that on its own — AnomalyGen SDG measured 520s — so an orchestration turn is
#     cut off mid-stage. Nothing reports it: the turn just ends.
#   * heartbeats fire on an interval to keep an always-on agent working, but each
#     heartbeat turn is bounded by the same timeout. Observed result:
#     {"reason":"interval","status":"ok-empty","durationMs":619326,"silent":true}
#     — the agent was woken, worked for exactly the cap, was truncated, and the
#     event was recorded as OK. The operator sees nothing and concludes the agent
#     is idle, then prompts it by hand every few minutes.
#
# skipWhenBusy stops a heartbeat landing on an agent that is mid-stage, which
# otherwise aborts the in-flight turn.
_orch = d.setdefault("agents", {}).setdefault("defaults", {})
_orch["timeoutSeconds"] = ${AGENT_TIMEOUT_SECONDS}
_hb = _orch.setdefault("heartbeat", {})
# every is the recovery interval, not just a liveness ping. A turn that dies —
# a truncated SSE stream from the provider is the common case — leaves the agent
# idle, and OpenClaw will not replay it: replay safety is decided by whether
# execution had started, and a tool cannot declare itself safe once it has. So
# the next heartbeat is what resumes the work, and at the stock 30m default that
# is up to half an hour of visible idle. skipWhenBusy means a short cadence costs
# nothing while the agent is actually working: those wakes are deferred.
_hb["every"] = "${HEARTBEAT_EVERY}"
_hb["skipWhenBusy"] = True
_hb["timeoutSeconds"] = ${AGENT_TIMEOUT_SECONDS}
_hb["includeSystemPromptSection"] = True

# Turn thinking on by default. models…models[].reasoning above only declares
# that the model CAN think; thinkingDefault decides whether it does, and it
# ships as "off" — so a sandbox can show reasoning=true in its model config and
# still run every turn without it. Levels: off|minimal|low|medium|high|xhigh.
if ${MODEL_REASONING_PY}:
    d["agents"]["defaults"]["thinkingDefault"] = "${MODEL_THINKING}"

json.dump(d, open(p, "w"), indent=2)
os.chmod(p, 0o660)
print("openclaw.json configured (mcp + coding tools profile; memorySearch off)")
if patched:
    print("model capabilities corrected for {}: contextWindow=${MODEL_CONTEXT_WINDOW}, "
          "maxTokens=${MODEL_MAX_TOKENS}, reasoning=${MODEL_REASONING}".format(", ".join(patched)))
else:
    print("WARN: no Claude Opus model entry found — capabilities left at NemoClaw's "
          "guessed defaults (likely maxTokens=4096, contextWindow=131072). "
          "A long agentic loop will truncate or overflow; set them by hand.")
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
  000) # No response at all (dropped, not refused). Step 2 already confirmed the
       # server process is up on $GW, so the usual cause is the host firewall
       # dropping bridge traffic on INPUT — not a bad bind.
       log "server unreachable (HTTP 000 — no response, i.e. dropped, not refused)."
       log "Most likely the host firewall is dropping sandbox->host traffic. Fix:"
       log "    $FW_HINT"
       log "Confirm the server itself is healthy (expect HTTP 400 or 406):"
       log "    curl -sS --noproxy '*' -o /dev/null -w '%{http_code}\\n' http://$GW:$PORT/mcp"
       die "bridge unreachable — see the hint above";;
  *)   die "server unreachable (HTTP $CODE) — check the server bind matches gateway $GW";;
esac

# ── 8. Dashboard ──────────────────────────────────────────────────────────────
# The authenticated URL carries a bearer token in the query string, so print it
# only to a terminal. Piped or redirected output is a log file, and a token that
# lands in one outlives the session it was minted for.
if DASH_URL=$(nemoclaw "$SB" dashboard-url --quiet 2>/dev/null) && [ -n "$DASH_URL" ]; then
  if [ -t 1 ]; then
    log "Dashboard: $DASH_URL"
  else
    log "Dashboard: nemoclaw $SB dashboard-url --quiet"
    log "           (URL carries an auth token; not written to non-terminal output)"
  fi
  DASH_PORT=$(printf '%s' "$DASH_URL" | sed -n 's|.*://[^:/]*:\([0-9]\{1,\}\).*|\1|p')
  if [ -n "${SSH_CONNECTION:-}" ] && [ -n "$DASH_PORT" ]; then
    log "           remote? forward it first:"
    log "             ssh -N -L ${DASH_PORT}:127.0.0.1:${DASH_PORT} $(id -un)@$(hostname -f 2>/dev/null || hostname)"
  fi
fi

log "Done. In the agent (nemoclaw $SB connect -> openclaw tui), ask:"
log "  'What MCP tools do you have?'  -> expect 12: tao_ls/read/write/exec/pull/run/list/status/logs/stop/rm/cleanup_results"
log "Put datasets under $WORKSPACE/<name>/ ; the agent sees them via tao_ls."
