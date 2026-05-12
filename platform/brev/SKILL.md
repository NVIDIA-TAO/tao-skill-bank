---
name: brev
description: Brev managed GPU instances with Docker support. Use when running training or inference on Brev GPU instances
  or managing Brev deployments.
license: Apache-2.0
compatibility: Requires the brev CLI (https://github.com/brevdev/brev-cli) and an active brev login.
metadata:
  author: NVIDIA Corporation
  version: '0.1'
allowed-tools: Read Bash
tags:
- gpu
- compute
- instance-based
- brev
---

# Brev

NVIDIA Brev provides on-demand GPU instances across multiple cloud providers. Instances come pre-loaded with NVIDIA drivers, CUDA, Docker, and NVIDIA Container Toolkit.

Brev is instance-based (not job-based like Lepton). You create an instance, run commands on it via `brev exec`, and delete it when done. The TAO SDK's BrevHandler wraps this into the standard job interface.

## Preflight

This skill needs the `brev` CLI, its companion agent skill (`brev-cli`), and an active login. Check before proceeding:

```bash
# 1. brev CLI installed
command -v brev >/dev/null 2>&1 || {
  echo "MISSING: brev CLI not installed. Install:"
  echo "  https://docs.nvidia.com/brev/"
  exit 1
}

# 2. brev-cli agent skill installed — provides the brev CLI's command reference to the agent
[ -d "$HOME/.claude/skills/brev-cli" ] || [ -d ".claude/skills/brev-cli" ] || {
  echo "MISSING: brev-cli agent skill not installed. Run:"
  echo "  brev agent-skill install"
  exit 1
}

# 3. brev login active
brev ls >/dev/null 2>&1 || {
  echo "MISSING: not logged in to brev. Run:"
  echo "  brev login                                    # interactive (opens browser)"
  echo "  # or set BREV_API_TOKEN in ~/.config/tao/.env (then 'brev login --token \$BREV_API_TOKEN')"
  exit 1
}
```

If any step fails, the agent prompts the user to authorize the fix via Bash, then re-runs the preflight before continuing. The TAO SDK is **not** required for Brev — `brev exec docker run …` is sufficient. Reach for the SDK only if you want Job handles, S3 I/O wrapping via `script_runner`, or state persistence; the SDK is not on public PyPI yet, install with: `pip install "nvidia-tao-sdk[brev] @ git+https://gitlab-master.nvidia.com/nvidia-tao-toolkit/tao-sdk.git"`.

## Authentication

Two options:

1. **Automated (recommended)**: Get an API token from the Brev console settings page. Set `BREV_API_TOKEN` as an environment variable (e.g., in `~/.config/tao/.env`). The handler auto-authenticates via `brev login --token` on first use — same UX as Lepton.

2. **Manual**: Run `brev login` (opens browser). Tokens expire hourly — the handler refreshes automatically.

S3 credentials (ACCESS_KEY, SECRET_KEY) are needed separately for data transfer.

## Launch Preflight

Before generating scripts or submitting jobs:

1. Verify `BREV_API_TOKEN` is set.
2. Verify the `brev` CLI is installed and can list instances, for example
   `brev ls --json`. If needed, authenticate with `brev login --token`.
3. For `s3://` datasets/results, verify `ACCESS_KEY` and `SECRET_KEY` are set
   and the exact paths are readable with `aws s3 ls`.
4. Do not accept local `/path` inputs for Brev unless the user has proven those
   paths exist on the target Brev instance or are mounted into it.
5. Verify model-specific credentials such as `HF_TOKEN` before launch.

## Instance Lifecycle

The agent controls instance lifecycle:

- **Reuse**: Pass `instance_id` in `backend_details` to run multiple jobs on the same instance. Efficient for multi-step workflows.
- **Ephemeral**: Omit `instance_id` — the handler creates a new instance per job. Clean but slower (instance boot ~2-5 min).

## Multi-GPU and multi-node

**Multi-node is not supported on Brev.** Brev is instance-based — one job runs on one instance, with no cross-instance coordination.

Multi-GPU **on a single instance** is supported (instances available with up to 8× H100 / A100 / L40S). `gpu_count` maps to the GPU count on the instance; `torchrun --nproc-per-node=N` or PyTorch DDP work within the instance.

## GPU Types

Available via `brev search`:
- L40S, A100 80GB, H100 (availability varies by provider)
- Use `--gpu-name` to filter, `--min-vram` for memory requirements

## Storage

No shared NFS/Lustre. All data flows through S3 via the script_runner's fsspec integration. Instance-local disk at `/home/ubuntu/` persists across stop/start but not across delete/create.

## Docker on Brev

VM Mode instances have Docker pre-installed. For TAO container images:

```bash
# NGC auth (one-time per instance)
brev exec <instance> -- docker login nvcr.io -u '$oauthtoken' -p <NGC_KEY>

# Run a TAO training job
brev exec <instance> -- docker run --gpus all --rm \
  -v /home/ubuntu/data:/data \
  nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt \
  visual_changenet train -e /data/spec.yaml
```

## Mixed-Platform Workflows

Brev can be mixed with Lepton in the same workflow. Per-stage platform assignment:

```json
{"skill": "vcn-gap-analysis", "action": "analyze", "platform": "brev"},
{"skill": "visual-changenet", "action": "train", "platform": "lepton"}
```

CPU stages (gap analysis, data merge) run cheaply on Brev. GPU stages (training) run on Lepton H100s.

## Error Patterns

**brev CLI not found**: Install from https://docs.nvidia.com/brev/.

**Token expired**: Handler auto-refreshes via `brev ls`. If persistent, run `brev login` manually.

**Instance stuck in provisioning**: Some GPU types have limited availability. Try a different `--gpu-name` or provider.

**Docker pull fails on nvcr.io**: NGC_KEY not set or expired. Run `docker login nvcr.io` on the instance.
