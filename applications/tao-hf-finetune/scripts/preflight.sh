#!/usr/bin/env bash
# Step 2a hard-gate preflight for tao-hf-finetune.
#
# Verifies the host can run an NVIDIA GPU container before Step 4 attempts a
# 20+ GB NGC base-image pull. Every check exits 1 on failure with an
# actionable, distro-aware hint.
#
# Checks (in order):
#   1. nvidia-smi reports a GPU and exits 0
#   2. docker is installed and the daemon is reachable
#   3. nvidia-container-toolkit is registered as a docker runtime
#   4. `docker --gpus all` works against a CUDA base image whose tag is
#      derived from the driver's max supported CUDA (parsed from nvidia-smi)
#   5. free disk on / (root volume) — soft warn (default 100 GB, override
#      via MIN_DISK_GB env). Not a hard stop: NGC base + HF cache + checkpoints
#      + dataset can exceed 80 GB for ≤3B fine-tunes, but actual need depends
#      on model + dataset, so we warn and continue.
#   6. HF_TOKEN is set
#
# Usage:
#   bash scripts/preflight.sh
#
# Notes:
#   - This script does NOT auto-install dependencies. nvidia-container-toolkit
#     needs sudo + the NVIDIA libnvidia-container repo + a daemon restart;
#     silently sudo-ing those is the wrong default. We detect the distro and
#     print the single matching install command for the user to run.

set -u

err() { echo "REJECT: $*" >&2; }

# ─── 1. nvidia-smi ──────────────────────────────────────────────────────────
nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap \
           --format=csv,noheader,nounits \
  || { err "nvidia-smi failed; check NVIDIA driver."; exit 1; }

# ─── 2. docker daemon ───────────────────────────────────────────────────────
docker --version >/dev/null \
  || { err "docker not installed."; exit 1; }
docker info >/tmp/docker-info.log 2>&1 \
  || { err "docker daemon not reachable; see /tmp/docker-info.log."; exit 1; }

# ─── 3. nvidia-container-toolkit precheck (must precede --gpus all) ─────────
if ! grep -qi 'Runtimes:.*nvidia' /tmp/docker-info.log; then
  err "nvidia-container-toolkit not installed/registered as a docker runtime."

  # Detect distro for a precise install hint.
  distro_id=""; distro_like=""; distro_pretty=""
  if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    distro_id="${ID:-}"
    distro_like="${ID_LIKE:-}"
    distro_pretty="${PRETTY_NAME:-$ID}"
  fi

  case " $distro_id $distro_like " in
    *' ubuntu '*|*' debian '*)
      install_cmd="sudo apt-get install -y nvidia-container-toolkit" ;;
    *' fedora '*|*' rhel '*|*' centos '*|*' rocky '*|*' almalinux '*)
      install_cmd="sudo dnf install -y nvidia-container-toolkit" ;;
    *' amzn '*)
      install_cmd="sudo yum install -y nvidia-container-toolkit" ;;
    *' opensuse '*|*' opensuse-leap '*|*' opensuse-tumbleweed '*|*' sles '*|*' suse '*)
      install_cmd="sudo zypper install -y nvidia-container-toolkit" ;;
    *)
      install_cmd=""
      ;;
  esac

  echo "  Detected distro: ${distro_pretty:-unknown}" >&2
  if [ -n "$install_cmd" ]; then
    echo "  Install (after adding the NVIDIA libnvidia-container repo):" >&2
    echo "    $install_cmd" >&2
  else
    echo "  No matching install hint for this distro." >&2
  fi
  echo "  Repo setup + full install guide:" >&2
  echo "    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html" >&2
  echo "  After install:" >&2
  echo "    sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker" >&2
  exit 1
fi

# ─── 4. GPU smoke through a CUDA tag derived from the driver's max CUDA ─────
cuda_max=$(nvidia-smi | awk -F'CUDA Version: ' '/CUDA Version/ {print $2}' | awk '{print $1}')
[ -n "$cuda_max" ] \
  || { err "could not parse 'CUDA Version' from nvidia-smi output."; exit 1; }
smoke_image="nvidia/cuda:${cuda_max}.0-base-ubuntu22.04"
docker run --rm --gpus all "$smoke_image" nvidia-smi -L \
  || { err "'docker --gpus all' failed with $smoke_image (driver max CUDA $cuda_max);"; \
       echo "  if Docker Hub does not publish that exact tag, retry with the next-lower minor (e.g. ${cuda_max%.*}.0.0)" >&2; \
       echo "  or restart the docker daemon." >&2; \
       exit 1; }

# ─── 5. disk space (soft warn, not a hard stop) ────────────────────────────
# Check root volume — practical "whole machine" on common single-disk hosts,
# WSL, and cloud VMs. Override threshold via MIN_DISK_GB (default 100 GB).
min_disk_gb="${MIN_DISK_GB:-100}"
disk_free_gb=$(df -BG / | awk 'NR==2 {print $4}' | tr -d G)
if [ "${disk_free_gb:-0}" -lt "$min_disk_gb" ]; then
  echo "WARN: only ${disk_free_gb}G free on /; recommend ≥ ${min_disk_gb}G for NGC base (~20G) + HF cache + checkpoints + dataset." >&2
  echo "  Continuing — set MIN_DISK_GB to silence or run 'docker system prune' to reclaim." >&2
fi

# ─── 6. HF_TOKEN ────────────────────────────────────────────────────────────
[ -n "${HF_TOKEN:-}" ] \
  || { err "HF_TOKEN missing"; exit 1; }

echo "preflight OK: GPU ($cuda_max) + docker + nvidia-container-toolkit + ${disk_free_gb}G free on / + HF_TOKEN."
