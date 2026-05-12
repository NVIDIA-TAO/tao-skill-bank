#!/usr/bin/env bash
set -euo pipefail

DRIVER_BRANCH="${NVIDIA_DRIVER_BRANCH:-580}"
DRIVER_PACKAGE="${NVIDIA_DRIVER_PACKAGE:-nvidia-open-${DRIVER_BRANCH}}"
CUDA_PACKAGE="${NVIDIA_CUDA_PACKAGE:-cuda-toolkit-13-0}"
CUDA_PATH="${NVIDIA_CUDA_PATH:-/usr/local/cuda-13.0}"
CONTAINER_TOOLKIT_VERSION="${NVIDIA_CONTAINER_TOOLKIT_VERSION:-1.19.0-1}"
BACKEND="docker"
INSTALL=0
YES=0
CONFIGURE_DOCKER=1

usage() {
  cat <<'USAGE'
Usage: setup-nvidia-gpu-host.sh [--backend docker|kubernetes] [--check-only|--install] [--yes] [--skip-docker-config]

Checks for the TAO GPU host runtime:
  - NVIDIA driver branch 580
  - CUDA Toolkit 13.0
  - NVIDIA Container Toolkit 1.19.0-1

By default this script only checks. Pass --install to configure NVIDIA apt
repositories and install any missing runtime packages.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      BACKEND="${2:-}"
      shift 2
      ;;
    --check-only)
      INSTALL=0
      shift
      ;;
    --install)
      INSTALL=1
      shift
      ;;
    -y|--yes)
      YES=1
      shift
      ;;
    --skip-docker-config)
      CONFIGURE_DOCKER=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$BACKEND" in
  docker|local-docker|kubernetes|k8s) ;;
  *)
    echo "Unsupported backend: $BACKEND" >&2
    exit 2
    ;;
esac

SUDO=()
if [[ "${EUID}" -ne 0 ]]; then
  SUDO=(sudo)
fi

have() {
  command -v "$1" >/dev/null 2>&1
}

sudo_available() {
  [[ "${EUID}" -eq 0 ]] || sudo -n true >/dev/null 2>&1
}

driver_ok() {
  have nvidia-smi || return 1
  local version
  version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]')"
  [[ "$version" == "${DRIVER_BRANCH}".* ]]
}

cuda_ok() {
  [[ -x "${CUDA_PATH}/bin/nvcc" ]] || return 1
  "${CUDA_PATH}/bin/nvcc" --version 2>/dev/null | grep -q 'release 13\.0'
}

container_toolkit_ok() {
  local version
  version="$(dpkg-query -W -f='${Version}' nvidia-container-toolkit 2>/dev/null || true)"
  [[ "$version" == "$CONTAINER_TOOLKIT_VERSION" ]]
}

docker_runtime_ok() {
  have docker || return 1
  if docker info >/dev/null 2>&1; then
    docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'
    return $?
  fi
  if sudo_available; then
    sudo docker info >/dev/null 2>&1 || return 1
    sudo docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'
    return $?
  fi
  return 1
}

kubernetes_gpu_ok() {
  have kubectl || return 2
  local gpu
  gpu="$(kubectl get nodes -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' 2>/dev/null | grep -v '^$' | head -n 1 || true)"
  [[ -n "$gpu" && "$gpu" != "0" ]]
}

print_status() {
  if driver_ok; then
    echo "OK: NVIDIA driver branch ${DRIVER_BRANCH}"
  else
    echo "MISSING: NVIDIA driver branch ${DRIVER_BRANCH}"
  fi

  if cuda_ok; then
    echo "OK: CUDA Toolkit 13.0 at ${CUDA_PATH}"
  else
    echo "MISSING: CUDA Toolkit 13.0 at ${CUDA_PATH}"
  fi

  if container_toolkit_ok; then
    echo "OK: NVIDIA Container Toolkit ${CONTAINER_TOOLKIT_VERSION}"
  else
    echo "MISSING: NVIDIA Container Toolkit ${CONTAINER_TOOLKIT_VERSION}"
  fi

  if [[ "$BACKEND" == "docker" || "$BACKEND" == "local-docker" ]]; then
    if docker_runtime_ok; then
      echo "OK: Docker NVIDIA runtime configured"
    else
      echo "MISSING: Docker NVIDIA runtime not configured or Docker unreachable"
    fi
  fi

  if [[ "$BACKEND" == "kubernetes" || "$BACKEND" == "k8s" ]]; then
    if kubernetes_gpu_ok; then
      echo "OK: Kubernetes reports nvidia.com/gpu allocatable"
    else
      local rc=$?
      if [[ "$rc" -eq 2 ]]; then
        echo "WARN: kubectl not found; cannot check cluster GPU capacity"
      else
        echo "WARN: Kubernetes does not report nvidia.com/gpu allocatable"
      fi
    fi
  fi
}

runtime_ok() {
  driver_ok && cuda_ok && container_toolkit_ok || return 1
  if [[ "$BACKEND" == "docker" || "$BACKEND" == "local-docker" ]]; then
    docker_runtime_ok
    return $?
  fi
  return 0
}

detect_cuda_repo() {
  if [[ ! -r /etc/os-release ]]; then
    echo "Cannot read /etc/os-release" >&2
    exit 1
  fi

  # shellcheck disable=SC1091
  . /etc/os-release
  if [[ "${ID:-}" != "ubuntu" ]]; then
    echo "Unsupported OS: ${PRETTY_NAME:-unknown}. This installer supports Ubuntu apt hosts." >&2
    exit 1
  fi

  case "${VERSION_ID:-}" in
    22.04) CUDA_DISTRO="ubuntu2204" ;;
    24.04) CUDA_DISTRO="ubuntu2404" ;;
    *)
      echo "Unsupported Ubuntu version: ${VERSION_ID:-unknown}. Expected 22.04 or 24.04." >&2
      exit 1
      ;;
  esac

  case "$(dpkg --print-architecture)" in
    amd64) CUDA_ARCH="x86_64" ;;
    arm64) CUDA_ARCH="sbsa" ;;
    *)
      echo "Unsupported Debian architecture: $(dpkg --print-architecture)" >&2
      exit 1
      ;;
  esac
}

confirm_install() {
  if [[ "$YES" -eq 1 ]]; then
    return 0
  fi

  cat <<EOF
This will install or repair:
  - ${DRIVER_PACKAGE} (driver branch ${DRIVER_BRANCH})
  - ${CUDA_PACKAGE}
  - nvidia-container-toolkit=${CONTAINER_TOOLKIT_VERSION}

It will add NVIDIA apt repositories if missing and may restart Docker.
EOF
  read -r -p "Continue? [y/N] " answer
  case "$answer" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 1 ;;
  esac
}

install_prereqs() {
  export DEBIAN_FRONTEND=noninteractive
  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" apt-get install -y --no-install-recommends ca-certificates curl gnupg
}

install_cuda_repo() {
  detect_cuda_repo

  if dpkg-query -W cuda-keyring >/dev/null 2>&1; then
    return 0
  fi

  local deb
  deb="$(mktemp)"
  curl -fsSL \
    "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_DISTRO}/${CUDA_ARCH}/cuda-keyring_1.1-1_all.deb" \
    --output "$deb"
  "${SUDO[@]}" dpkg -i "$deb"
  rm -f "$deb"
}

install_container_repo() {
  local key_tmp keyring_tmp list_tmp
  key_tmp="$(mktemp)"
  keyring_tmp="$(mktemp)"
  list_tmp="$(mktemp)"

  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey --output "$key_tmp"
  gpg --dearmor --yes --output "$keyring_tmp" "$key_tmp"
  "${SUDO[@]}" install -m 0644 "$keyring_tmp" /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list --output "$list_tmp"
  sed -i 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' "$list_tmp"
  "${SUDO[@]}" install -m 0644 "$list_tmp" /etc/apt/sources.list.d/nvidia-container-toolkit.list

  rm -f "$key_tmp" "$keyring_tmp" "$list_tmp"
}

install_runtime_packages() {
  export DEBIAN_FRONTEND=noninteractive
  local kernel_headers
  kernel_headers="linux-headers-$(uname -r)"

  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" apt-get install -y --allow-downgrades \
    "$kernel_headers" \
    "nvidia-driver-pinning-${DRIVER_BRANCH}" \
    "$DRIVER_PACKAGE" \
    "$CUDA_PACKAGE" \
    "nvidia-container-toolkit=${CONTAINER_TOOLKIT_VERSION}" \
    "nvidia-container-toolkit-base=${CONTAINER_TOOLKIT_VERSION}" \
    "libnvidia-container-tools=${CONTAINER_TOOLKIT_VERSION}" \
    "libnvidia-container1=${CONTAINER_TOOLKIT_VERSION}"
}

configure_docker_runtime() {
  [[ "$CONFIGURE_DOCKER" -eq 1 ]] || return 0
  [[ "$BACKEND" == "docker" || "$BACKEND" == "local-docker" ]] || return 0

  if ! have docker; then
    echo "WARN: Docker is not installed; skipping Docker NVIDIA runtime configuration."
    return 0
  fi

  "${SUDO[@]}" nvidia-ctk runtime configure --runtime=docker
  if have systemctl; then
    "${SUDO[@]}" systemctl restart docker || {
      echo "WARN: could not restart Docker; restart it manually before running GPU containers."
    }
  else
    echo "WARN: systemctl not found; restart Docker manually before running GPU containers."
  fi
}

if [[ "$INSTALL" -eq 0 ]]; then
  print_status
  if runtime_ok; then
    exit 0
  fi
  echo
  echo "Run with --install --yes after user approval to install the pinned runtime."
  exit 1
fi

if ! sudo_available; then
  echo "MISSING: passwordless sudo/root is required for runtime installation." >&2
  exit 1
fi

confirm_install
install_prereqs
install_cuda_repo
install_container_repo
install_runtime_packages
configure_docker_runtime

if have modprobe; then
  "${SUDO[@]}" modprobe nvidia || true
fi

print_status
runtime_ok
