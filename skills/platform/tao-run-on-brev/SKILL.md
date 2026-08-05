---
name: tao-run-on-brev
description: Run a TAO training/evaluation/inference container on an NVIDIA Brev GPU instance. Instance provisioning (create/search/stop/delete/login) is delegated to the official brev-cli agent skill or the Brev MCP server; this skill covers only the TAO-specific part — running the container over `brev exec` via the four-verb docker contract. Trigger phrases include "run on Brev", "Brev GPU instance", "TAO on Brev", "submit job to Brev".
license: Apache-2.0
compatibility: Requires the brev CLI (https://github.com/brevdev/brev-cli) and an active brev login. Instance provisioning is handled by the official brev-cli agent skill or the Brev MCP server.
metadata:
  author: NVIDIA Corporation
  version: "0.2.0"
allowed-tools: Read Bash
tags:
- gpu
- compute
- instance-based
- brev
---

# Brev — TAO execution glue

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

NVIDIA Brev provides on-demand GPU instances (pre-loaded with NVIDIA drivers,
CUDA, Docker, and the NVIDIA Container Toolkit). Brev is **instance-based**: you
provision an instance, run commands on it over `brev exec`, and delete it when
done.

This skill is deliberately thin. **Provisioning and managing instances — create,
search by GPU/price, start/stop, delete, login — is owned by NVIDIA Brev's own
agent skill, not duplicated here.** This skill covers only the TAO-specific part:
running a TAO container on a reached instance through the **four-verb docker
contract**, deferring the container-how to `tao-run-on-docker` over `brev exec`.

## Provisioning: use the official Brev skill or MCP

NVIDIA Brev publishes an agent skill that manages instances in natural language
("create an A100 instance", "search for GPUs under $3/hr", "stop all my
instances"). Install it once — it self-registers into your agent's skills dir and
is discovered at runtime:

```bash
curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/scripts/install-agent-skill.sh | bash
# installs to ~/.claude/skills/brev-cli/ , ~/.codex/skills/brev-cli/ , ~/.agents/skills/brev-cli/
```

Or connect the **Brev MCP server** (`https://docs.nvidia.com/brev/_mcp/server`).
Either one owns login/auth quirks, placement IDs, GPU search, and teardown flags.
It does **not** cover container execution on the instance — that is this skill.

**Preflight for this skill:** the `brev` CLI is on `PATH` and logged in (headless:
`brev login --token "$BREV_API_TOKEN"` before any other call), and you can reach a
target instance — poll `brev exec <instance> -- true` until it succeeds before <!-- lint-ok: brev-exec-form -->
issuing real work (a fresh instance reports `RUNNING` before sshd is up):

```bash
for i in $(seq 1 60); do brev exec <instance> -- true >/dev/null 2>&1 && break; sleep 5; done  # lint-ok: brev-exec-form
brev exec <instance> -- true >/dev/null 2>&1 || { echo "instance not exec-ready"; exit 1; }  # lint-ok: brev-exec-form
```

Allow **≥ 600 s** for the first `brev exec` on a new instance (SSH bring-up +
first container pull); a 60–120 s wrapper timeout truncates startup and looks
like a spurious `exec failed`.

## Storage

No shared NFS/Lustre — storage tier **B/C** via `tao-data-io`: stage inputs from
S3 to the instance's local disk (or fetch in-container) and **upload results to S3
before deleting the instance**. Instance-local `~/` persists across stop/start but
**not** across delete/create, so the results upload must precede teardown.

## Execution — the four verbs (a compound over Docker)

Brev is a **compound consumer**: `submit` reaches an instance, then **defers the
container-how to the four docker verbs** (`tao-run-on-docker`) run over
`brev exec`. It is not a symmetric peer — teardown must additionally delete the
instance to stop billing. `$BANK` = `${TAO_SKILL_BANK_PATH}`.

- **submit** — reach an instance (provision/reuse via the official Brev skill or
  MCP; reuse an existing instance by its `instance_id`; wait for readiness, above).
  Then run the docker `submit` verb *inside* it: `open` the record (`--platform
  brev`, `--backend-ref "<instance>/<container>"`), `brev exec <instance> --
  docker run -d --name "$JOB_ID" ...`, mark RUNNING.
- **status / logs** — `brev exec <instance> -- docker inspect/logs "$JOB_ID"`, <!-- lint-ok: brev-exec-form -->
  mapped to the vocab exactly as the docker verbs do.
- **cancel / teardown** — `brev exec <instance> -- docker rm -f "$JOB_ID"`, then <!-- lint-ok: brev-exec-form -->
  for an ephemeral instance **`brev delete <instance>`** (stops billing), then mark
  the record. Never leave an ephemeral instance running.

NGC auth once per instance — **never put `NGC_KEY` on argv** (it lands in the
remote process table); pipe it to `--password-stdin`:

```bash
# NGC auth (one-time per instance) — value never on argv
brev exec <instance> -- bash -lc 'printf %s "$NGC_KEY" | docker login nvcr.io -u "$oauthtoken" --password-stdin'  # lint-ok: brev-exec-form

# Run a TAO job (the docker `submit` verb, over brev exec)
IMG=nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt  # versions-key: images.tao_toolkit.pyt
brev exec <instance> "docker run -d --name $JOB_ID --gpus all -v ~/data:/data -e NGC_KEY $IMG visual_changenet train -e /data/spec.yaml"
```

## Multi-GPU and multi-node

**Multi-node is not supported on Brev** — instance-based, no cross-instance
coordination. Multi-GPU **on a single instance** is supported (up to 8× H100 /
A100 / L40S); `torchrun --nproc-per-node=N` or PyTorch DDP work within the
instance.
