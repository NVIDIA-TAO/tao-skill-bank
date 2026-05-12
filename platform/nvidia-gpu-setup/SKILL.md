---
name: nvidia-gpu-setup
description: >-
  Host setup for TAO GPU backends. Checks and, after user approval, installs
  NVIDIA driver branch 580, CUDA Toolkit 13.0, and NVIDIA Container Toolkit
  1.19.0 for Docker/local-Docker and Kubernetes GPU worker hosts.
license: Apache-2.0
compatibility: Requires Ubuntu 22.04/24.04 with apt, sudo/root, internet access to NVIDIA package repositories, and Docker if configuring the Docker backend.
metadata:
  author: NVIDIA Corporation
  version: '0.1'
allowed-tools: Read Bash
tags:
- setup
- nvidia
- cuda
- docker
- kubernetes
---

# NVIDIA GPU Host Setup

Use this setup skill before TAO workflows run on the `docker`, `local-docker`,
or `kubernetes` backend. It standardizes the host GPU runtime on:

- NVIDIA driver branch `580`
- CUDA Toolkit package `cuda-toolkit-13-0`
- NVIDIA Container Toolkit `1.19.0-1`

The check is safe and read-only by default. Installation must be explicitly
authorized by the user and rerun with `--install`.

## Quick Start

From the skill bank root:

```bash
# Check the local Docker backend host.
bash platform/nvidia-gpu-setup/scripts/setup-nvidia-gpu-host.sh --backend docker --check-only

# Install or repair after user approval.
bash platform/nvidia-gpu-setup/scripts/setup-nvidia-gpu-host.sh --backend docker --install --yes

# Check a Kubernetes GPU worker host.
bash platform/nvidia-gpu-setup/scripts/setup-nvidia-gpu-host.sh --backend kubernetes --check-only
```

In an installed plugin copy that exposes `skills/`, use:

```bash
bash skills/nvidia-gpu-setup/scripts/setup-nvidia-gpu-host.sh --backend docker --check-only
```

## Workflow Contract

Docker and Kubernetes workflows must run the check before submitting GPU work:

```bash
SETUP_SCRIPT="${TAO_SKILL_BANK_ROOT:-$PWD}/skills/nvidia-gpu-setup/scripts/setup-nvidia-gpu-host.sh"
[ -x "$SETUP_SCRIPT" ] || SETUP_SCRIPT="${TAO_SKILL_BANK_ROOT:-$PWD}/platform/nvidia-gpu-setup/scripts/setup-nvidia-gpu-host.sh"

bash "$SETUP_SCRIPT" --backend docker --check-only || {
  echo "MISSING: TAO GPU host runtime is not ready."
  echo "After user approval, run:"
  echo "  bash \"$SETUP_SCRIPT\" --backend docker --install --yes"
  exit 1
}
```

Never install silently. If the check fails, explain what is missing, ask the
user to authorize the fix, then run the install command and rerun the check.

## What The Installer Does

For Ubuntu 22.04/24.04 apt hosts, the script:

1. Adds NVIDIA's CUDA apt keyring/repository if `cuda-keyring` is absent.
2. Adds NVIDIA Container Toolkit's apt repository/keyring.
3. Installs current-kernel headers, `nvidia-driver-pinning-580`,
   `nvidia-open-580`, `cuda-toolkit-13-0`, and the four Container Toolkit
   packages pinned to `1.19.0-1`.
4. For Docker backends, runs `nvidia-ctk runtime configure --runtime=docker`
   and restarts Docker when `systemctl` is available.
5. Attempts `modprobe nvidia` so verification can pass before reboot.

## Verification

After installation, verify:

```bash
nvidia-smi
/usr/local/cuda-13.0/bin/nvcc --version
docker info --format '{{json .Runtimes}}' | grep nvidia
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

Expected `nvidia-smi` output includes driver `580.x` and CUDA Version `13.0`.
Expected `nvcc` output includes `release 13.0`.

## Kubernetes Notes

For self-managed Kubernetes clusters, run the host installer on every GPU
worker node or bake the same package set into the node image before installing
the NVIDIA GPU Operator or device plugin.

The workflow check also warns if `kubectl` is available but the cluster reports
no `nvidia.com/gpu` allocatable capacity. In that case, install/configure the
NVIDIA GPU Operator after the worker host runtime is ready:

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
helm install --wait gpu-operator -n gpu-operator --create-namespace nvidia/gpu-operator
```

Managed Kubernetes providers may own driver installation through node images or
GPU Operator policy. Do not overwrite a provider-managed GPU node without user
approval and a rollback plan.

## Failure Modes

**Unsupported OS**: The installer supports Ubuntu 22.04/24.04 apt hosts. On
other distributions, use the same version targets with that distribution's
NVIDIA package-manager instructions.

**Docker runtime still missing**: Restart Docker, then rerun
`nvidia-ctk runtime configure --runtime=docker`.

**Driver installed but `nvidia-smi` fails**: Load the module with
`sudo modprobe nvidia` or reboot. Secure Boot may require MOK enrollment on
systems where it is enabled.

**Kubernetes still has no GPU capacity**: Confirm the driver works on each GPU
node with `nvidia-smi`, then check the GPU Operator/device plugin pods and node
labels.
