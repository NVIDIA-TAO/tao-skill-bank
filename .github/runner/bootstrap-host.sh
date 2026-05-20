#!/usr/bin/env bash
# bootstrap-host.sh — one-shot setup for a fresh DGX-style host to run skill-eval in CI.
#
# Manual prerequisites on the host (one-time, can't be scripted):
#   1. ~/skill-eval.env  — provider tokens (S3, GitLab, GitHub, NGC, Artifactory, ...)
#   2. claude login      — writes ~/.claude.json
#   3. codex login --device-auth  — writes ~/.codex/auth.json
#
# Everything else is automated here. Idempotent; safe to re-run.
#
# Usage:
#   bash bootstrap-host.sh                 # install + pull image; print smoke command
#   bash bootstrap-host.sh --smoke         # also run the bundled hello-world smoke test
#
# Env overrides:
#   SKILL_EVAL_IMAGE   default: nvcr.io/nvstaging/tao/skill-eval:latest

set -euo pipefail

IMAGE="${SKILL_EVAL_IMAGE:-nvcr.io/nvstaging/tao/skill-eval:latest}"
RUN_SMOKE=0
[ "${1:-}" = "--smoke" ] && RUN_SMOKE=1

log() { printf '[bootstrap] %s\n' "$*"; }

# ---------------- 0. Sanity ----------------
[ -f "$HOME/skill-eval.env" ] || { echo "[bootstrap] FATAL: missing ~/skill-eval.env" >&2; exit 2; }
[ -f "$HOME/.claude.json" ] || [ -f "$HOME/.claude/.credentials.json" ] \
  || log "WARN: no claude auth found (claude backend won't work until 'claude login' runs)"
[ -f "$HOME/.codex/auth.json" ] \
  || log "WARN: no codex auth found (codex backend won't work until 'codex login --device-auth' runs)"

# ---------------- 1. apt: docker + nvidia-container-toolkit + fuse-overlayfs ----------------
log "step 1/6 — install docker.io, nvidia-container-toolkit, fuse-overlayfs"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends docker.io ca-certificates curl gnupg fuse-overlayfs >/dev/null
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y nvidia-container-toolkit >/dev/null
fi

# ---------------- 2. Register nvidia runtime ----------------
log "step 2/6 — register nvidia runtime in /etc/docker/daemon.json"
sudo nvidia-ctk runtime configure --runtime=docker >/dev/null

# ---------------- 3. Start dockerd ----------------
# This pod runs k8s under containerd, so we can't mount kernel overlay (the host
# already has overlayfs in use upstream). Try fuse-overlayfs first — it dedupes
# layers so disk usage stays close to overlay2 (vfs makes full copies of every
# layer and blew up disk usage in prior runs, triggering pod eviction). If
# fuse-overlayfs fails for any reason, fall back to vfs.
log "step 3/6 — start dockerd (storage=fuse-overlayfs preferred, vfs fallback)"
sudo usermod -aG docker "$USER" 2>/dev/null || true
start_dockerd() {
  local driver="$1"
  sudo pkill -TERM dockerd 2>/dev/null || true
  sleep 1
  sudo rm -rf /var/lib/docker 2>/dev/null || true
  sudo nohup dockerd --storage-driver="$driver" > /tmp/dockerd.log 2>&1 &
  for i in $(seq 1 30); do [ -S /var/run/docker.sock ] && break; sleep 1; done
  sg docker -c 'docker info' >/dev/null 2>&1
}
if ! sg docker -c 'docker info' >/dev/null 2>&1; then
  if start_dockerd fuse-overlayfs; then
    log "  dockerd up with storage=fuse-overlayfs"
  else
    log "  fuse-overlayfs failed (see /tmp/dockerd.log); retrying with vfs"
    start_dockerd vfs || { log "FATAL: dockerd refused to start"; exit 1; }
    log "  dockerd up with storage=vfs (slower, but works)"
  fi
fi
sg docker -c 'docker info' >/dev/null
sg docker -c 'docker info 2>&1 | grep "Storage Driver:"' || true

# ---------------- 4. Pull skill-eval image ----------------
log "step 4/6 — pull $IMAGE"
set -a; . "$HOME/skill-eval.env"; set +a
if [ -n "${NGC_KEY:-}" ]; then
  echo "$NGC_KEY" | sg docker -c "docker login nvcr.io -u '\$oauthtoken' --password-stdin" >/dev/null \
    && log "  nvcr.io login via NGC_KEY: OK"
fi
sg docker -c "docker pull $IMAGE" | tail -3
sg docker -c "docker tag $IMAGE skill-eval:latest"

# ---------------- 5. Install host launcher to ~/.local/bin/skill-eval ----------------
log "step 5/6 — install host launcher to ~/.local/bin/skill-eval"
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/skill-eval" <<'BOOTSTRAP_LAUNCHER_EOF'
#!/usr/bin/env bash
# Host launcher: docker-run the skill-eval image with auth + env mounts.
set -uo pipefail

IMAGE="${SKILL_EVAL_IMAGE:-skill-eval:latest}"

usage() {
  cat <<EOF
Usage: skill-eval [host-options] <command> [args]

Host options:
  --env-file FILE     Env vars to pass to the container (default: ./skill-eval.env if present)
  --output-dir DIR    Mount as /workspace/skill-test-outputs (default: ./skill-test-outputs)
  --skill-src DIR     Mount as /workspace/skill-source
  --image IMAGE       Docker image (default: skill-eval:latest, env: SKILL_EVAL_IMAGE)
  --gpus SPEC         Forwarded as docker --gpus
  -v|--volume SPEC    Extra docker volume mount (repeatable)
  -e VAR[=VAL]        Extra env (repeatable)
  -h|--help

Container commands:
  smoke [--backend claude|codex|both]
  eval  <skill-dir-inside-container> [--backend ...] [--overlay NAME]
  bash
  versions
EOF
}

env_file=""
output_dir="$(pwd)/skill-test-outputs"
skill_src=""
gpus=""
extra_volumes=()
extra_envs=()

while [ $# -gt 0 ]; do
  case "$1" in
    --env-file) env_file="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --skill-src) skill_src="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    -v|--volume) extra_volumes+=("$2"); shift 2 ;;
    -e) extra_envs+=("$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) break ;;
  esac
done

if [ -z "$env_file" ] && [ -f "$(pwd)/skill-eval.env" ]; then
  env_file="$(pwd)/skill-eval.env"
fi

mkdir -p "$output_dir"

# Shared workspace dir at SAME path host+container — makes docker-in-docker bind mounts work.
ws_dir="${WORKSPACE_DIR:-$HOME/skill-eval-workspace}"
mkdir -p "$ws_dir"

docker_args=(--rm -i --user "$(id -u):$(id -g)")
docker_args+=(-v "$output_dir:/workspace/skill-test-outputs")
docker_args+=(-v "$ws_dir:$ws_dir")
docker_args+=(-e "WORKSPACE_DIR=$ws_dir")

if [ -n "$skill_src" ]; then
  [ -d "$skill_src" ] || { echo "skill-src not found: $skill_src" >&2; exit 1; }
  docker_args+=(-v "$(cd "$skill_src" && pwd):/workspace/skill-source")
fi

mounted_any_auth=0
if [ -f "$HOME/.claude.json" ]; then
  docker_args+=(-v "$HOME/.claude.json:/skilleval-home/.claude.json"); mounted_any_auth=1
fi
if [ -f "$HOME/.claude/.credentials.json" ]; then
  docker_args+=(-v "$HOME/.claude/.credentials.json:/skilleval-home/.claude/.credentials.json"); mounted_any_auth=1
fi
if [ -f "$HOME/.codex/auth.json" ]; then
  docker_args+=(-v "$HOME/.codex/auth.json:/skilleval-home/.codex/auth.json"); mounted_any_auth=1
fi
if [ -f "$HOME/.codex/cloud-requirements-cache.json" ]; then
  docker_args+=(-v "$HOME/.codex/cloud-requirements-cache.json:/skilleval-home/.codex/cloud-requirements-cache.json")
fi

if [ -S "/var/run/docker.sock" ]; then
  docker_args+=(-v "/var/run/docker.sock:/var/run/docker.sock")
  if getent group docker >/dev/null 2>&1; then
    docker_args+=(--group-add "$(getent group docker | cut -d: -f3)")
  fi
fi
if [ "$mounted_any_auth" = "0" ]; then
  echo "WARN: no claude/codex auth file found on host." >&2
fi

if [ -n "$env_file" ]; then
  [ -f "$env_file" ] || { echo "env-file not found: $env_file" >&2; exit 1; }
  docker_args+=(--env-file "$env_file")
fi
for v in "${extra_volumes[@]:-}"; do [ -n "$v" ] && docker_args+=(-v "$v"); done
for e in "${extra_envs[@]:-}"; do [ -n "$e" ] && docker_args+=(-e "$e"); done
[ -n "$gpus" ] && docker_args+=(--gpus "$gpus")
[ -t 1 ] && docker_args+=(-t)

exec docker run "${docker_args[@]}" "$IMAGE" "$@"
BOOTSTRAP_LAUNCHER_EOF
chmod +x "$HOME/.local/bin/skill-eval"

# ---------------- 6. PATH ----------------
log "step 6/6 — ensure ~/.local/bin on PATH"
if ! grep -q 'HOME/.local/bin' "$HOME/.bashrc" 2>/dev/null; then
  echo 'export PATH=$HOME/.local/bin:$PATH' >> "$HOME/.bashrc"
  log "  appended PATH update to ~/.bashrc"
fi

log ""
log "bootstrap COMPLETE."
log "Smoke command (use sg or new shell so docker group takes effect):"
log "  sg docker -c '~/.local/bin/skill-eval --env-file ~/skill-eval.env smoke'"

# ---------------- Optional: run smoke test now ----------------
if [ "$RUN_SMOKE" = "1" ]; then
  log ""
  log "running smoke test..."
  cd "$HOME"
  sg docker -c '~/.local/bin/skill-eval --env-file ~/skill-eval.env smoke' || {
    rc=$?; log "smoke test exited with rc=$rc"; exit $rc
  }
fi
