# TAO on NemoClaw (host MCP server)

Give a [NemoClaw](https://github.com/NVIDIA/NemoClaw) sandbox agent the ability
to run NVIDIA TAO training/inference on the host GPU — **without** granting the
sandbox Docker, GPU, or NGC credentials.

A small MCP server runs on the host and exposes typed tools over the OpenShell
host bridge. The agent calls tools (list/read/write the workspace, run/monitor/
stop TAO containers); the host executes the containers. Everything the sandbox
cannot safely do — Docker, the GPU, holding secrets — stays host-side.

```
agent (sandbox)  --HTTP over OpenShell bridge-->  tao-mcp server (host)  -->  docker run TAO container (host GPU)
```

## Why a host MCP server

The OpenShell sandbox is deny-by-default on egress and has no Docker. Direct
approaches fail: raw SSH/TCP can't leave the sandbox, and the managed
`nemoclaw mcp add` path needs an attested public-DNS endpoint. A host MCP
server registered directly in `openclaw.json` (the pattern the
[VSS blueprint](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization)
uses) reaches the local bridge cleanly and moves execution to where it works.

## Quick start

Prerequisites: a NemoClaw sandbox already onboarded with the **OpenClaw** agent
(validated with **Claude Opus 5** as the agent brain — see Tested configuration),
and the host logged into
NGC (`docker login nvcr.io`). All agent-side setup is written into the
sandbox's `openclaw.json`, so an OpenClaw-based NemoClaw sandbox is required.

Then, on the host:

```bash
./setup-tao-nemoclaw.sh <sandbox-name>          # workspace defaults to ~/tao-workspace
```

The script starts the server (bound to the docker-bridge IP, off the LAN),
installs the TAO skills into the sandbox, registers the MCP server, opens the
bridge policy, and verifies reachability. Then in the agent
(`nemoclaw <sandbox> connect` → `openclaw tui`): *"What MCP tools do you have?"*

Put datasets under `<workspace>/<name>/`; the agent discovers them with `tao_ls`.

## Tools

| Tool | Purpose |
|------|---------|
| `tao_ls` / `tao_read` / `tao_write` | Inspect and author files in the host workspace |
| `tao_exec` | Run a shell command in a **CPU-only** container over the whole workspace — the agent's shell for everything that is not GPU compute: inspecting data, unpacking archives, authoring specs, staging models (`huggingface_hub` / `ngcsdk` / `curl`). Has outbound network and `HF_TOKEN`; runs as the server's host UID:GID, so nothing it writes is root-owned. Uses `$TAO_SHELL_IMAGE`, which must already be pulled |
| `tao_pull` | Pull an `nvcr.io/*` image into the host cache before launch |
| `tao_run` | Launch a cached container image on the host GPU without pulling (workspace-confined per-job results, host UID:GID ownership, `shm_size` for DataLoaders) |
| `tao_list` | List and recover jobs launched for this TAO workspace |
| `tao_status` / `tao_logs` | Monitor a job |
| `tao_stop` / `tao_rm` | Stop/remove a job's container layer (TAO containers only; bind-mounted outputs remain) |
| `tao_cleanup_results` | Remove a terminal job and its verified, isolated result tree without `sudo` |

## Output ownership and cleanup

`tao_run` runs the container as the UID:GID of the host user running the MCP
server, so checkpoints and results stay removable by that user without `sudo`.
Because overriding an image's root user makes `/root` unwritable, the bridge
prepares `HOME` and framework caches inside the job's own result path at
`.tao-runtime/home`. It refuses to launch writable jobs when the bridge itself
runs as root — start it as the submitting user.

Each `tao_run` mounts a unique `<results_subdir>/.tao-jobs/<token>/` at
`/results` and returns that exact path, so cleaning up one failed experiment
cannot delete another run's checkpoints.

`tao_stop` and `tao_rm` manage only the container — they do **not** delete
results, checkpoints or caches, so do not report container removal as output
cleanup. To dispose of a run: `tao_stop`, inspect any logs you need, then
`tao_cleanup_results`. Never `tao_rm` first; removing the container also removes
the metadata that authorizes cleanup. Results from older bridge versions may be
root-owned and need a one-time ownership repair.

## Security

**This server is the security boundary — keep it.** `tao_run` refuses any image
outside `nvcr.io/*` (the NVIDIA NGC registry) and confines all mounts to a fixed
`--workspace-root` the agent cannot escape. This admits TAO images plus
QA/staging and data-generation images from other NGC orgs, but still refuses
arbitrary registries (Docker Hub, private, etc.). The agent gets NGC-image
execution, not arbitrary host control.

> **Do not** substitute a generic public Docker MCP server (e.g.
> `ckreiling/mcp-server-docker`). Those expose unconstrained `run_container`
> with arbitrary images and host mounts — equivalent to giving the sandboxed
> agent root on the host (it can read the host filesystem and the OpenShell
> credential store). Use the constrained server here unless you fully accept
> that exposure.

Two properties that keep it safe: the server binds the **docker-bridge IP**
(reachable only by sandbox containers, not the LAN), and the sandbox reaches it
only through the **bridge egress policy** the setup applies.

## Scope

Runs TAO workflows on the host GPU. The agent reads the skill, authors the
spec, stages models (Docker's default outbound networking lets HuggingFace / NGC
/ S3 pulls work in-container), launches `tao_run`, and monitors — orchestrating
multi-step workflows itself over the tools. Verified on hardware (DINO, Visual
ChangeNet, DEFT AOI).

AutoML's managed search loop may still need the Claude Code / Codex plugin
runtime; other TAO workflows run through this surface.

## Files

| File | What |
|------|------|
| `server.py` | The MCP server (stdlib + `mcp` + `uvicorn`) |
| `setup-tao-nemoclaw.sh` | One-command setup for a sandbox |
| `uninstall-tao-nemoclaw.sh` | Reverses setup: policy, `openclaw.json`, skill tree, `AGENTS.md` block, host server. Never touches workspace data; `--purge-bank` also removes the cloned bank |
| `AGENTS.md` | Runtime operating guide appended to the sandbox's workspace `AGENTS.md` |

## Notes / gotchas

- Adding or changing tools requires `nemoclaw <sandbox> gateway restart` before the
  agent re-fetches the tool list.
- Docker must support `volume-subpath` mounts; setup fails closed if the host
  engine does not.
- **Experimental** — NemoClaw is alpha; see Tested configuration below.

## Tested configuration

Last full validation: **2026-08-10**, Ubuntu 22.04.5 x86_64, 36 vCPU / 31 GiB,
NVIDIA RTX A6000 (49 GB), corporate IT-managed host with `ufw` active.

### Resulting configuration

| Setting | Value | Where |
|---|---|---|
| MCP endpoint | `http://host.openshell.internal:9901/mcp` | `openclaw.json` → `mcp.servers.tao` |
| Server bind | docker-bridge gateway `172.19.0.1:9901` (never `0.0.0.0`) | `server.py` args |
| Gateway port | 8080 on the same bridge IP | OpenShell |
| Network policy | `tao-mcp` preset, GET/POST/DELETE to `host.openshell.internal:9901` | `nemoclaw <sb> policy list` |
| Tools profile | `coding` (exec + fs + subagents, sandbox-scoped) | `openclaw.json` → `tools.profile` |
| Skill bank | `<workspace>/tao-skill-bank` (host) and `/sandbox/tao-skill-bank` (sandbox) | `docker cp` |
| `contextWindow` | `1000000` | `openclaw.json` → `models.providers.*.models[]` |
| `maxTokens` | `128000` | same |
| `reasoning` | `true` | same |
| `memorySearch.enabled` | `false` | `agents.defaults.memorySearch` |
| `timeoutSeconds` | `3600` (agent and heartbeat) | `agents.defaults` |
| `heartbeat.every` | `2m`, `skipWhenBusy: true` | `agents.defaults.heartbeat` |
| ufw | `allow from 172.19.0.0/16 to 172.19.0.1 port 8080,9901 proto tcp` | host |

Capability values were measured against the endpoint, not assumed: the
`max_tokens` ceiling is reported verbatim by a deliberate overflow
(`max_tokens: 9999999 > 128000 … for anthropic.claude-opus-5`), 128000 is
accepted on a **non-streaming** request, a single request carrying 432,015
input tokens returned HTTP 200, and `thinking` blocks come back signed.

### Version matrix

| Component | Verified | Notes |
|---|---|---|
| NemoClaw | **v0.0.97** (`lkg`) | v0.0.97–v0.0.101 were previously blocked by an `npm audit` gate against a live advisory feed (upstream issue #8177) |
| OpenShell CLI | **0.0.85** | pinned exactly by `nemoclaw-blueprint/blueprint.yaml` (`min == max`), so it is not a floor — 0.0.72 in earlier docs is not reproducible. **This pin is NemoClaw's, not ours, and it moves:** 0.0.72 (07-03) → 0.0.85 (07-17) → 0.0.99 (08-07) → 0.0.101 (08-10) → 0.0.106 (08-20). Upgrading NemoClaw upgrades OpenShell with it, so never assume a version observed here still holds — see the note below |
| OpenClaw agent | **2026.7.1** (`2d2ddc4`) | the `nemoclaw onboard --agent openclaw` path; Hermes and Deep Agents are untested here |
| Agent brain | **Claude Opus 5** (`aws/anthropic/bedrock-claude-opus-5`) | supersedes Opus 4.8; see capability note below |
| Node | **≥ 22.19.0** | declared by `tools/mcp-tool-discovery-runtime/package.json`; the quickstart's "Node 20+" is wrong. The installer bootstraps 22.x itself |
| Python `mcp` | **< 2** (resolves 1.29.0) | `mcp` 2.x removed `mcp.server.fastmcp`, which `server.py` imports. Pinned in this script |
| TAO toolkit image | **7.1.0-pyt** | from the bank's `versions.yaml`; also the default `TAO_SHELL_IMAGE` for `tao_exec` |

This table records what was *verified*, not what is *enforced*. Nothing here
pins anything on our side — the versions an operator actually gets come from
whichever NemoClaw release they installed.

That distinction has already cost us one P0 (NVBug 6682592). OpenShell v0.0.88
renamed sandbox containers from `openshell-<name>-<id>` to
`openshell-<workspace>--<name>-<id>`, and NemoClaw's exact pin jumped 0.0.85 →
0.0.99 on 2026-08-07, straight over it. Both setup and uninstall were matching
that name and broke — on newer hosts only, so it looked like a flaky recurrence
rather than a version boundary. They now resolve the container by its
`openshell.ai/sandbox-name` label instead, which is stable across the rename.

The lesson generalises: **do not re-derive OpenShell's internal conventions**
(container names, network names, paths). Ask for them, or key off labels.

### Custom inference endpoints

NemoClaw cannot probe a custom (`anthropicCompatible`) endpoint, so it writes
conservative **guessed** model capabilities into `openclaw.json` with no
warning — typically `contextWindow 131072`, `maxTokens 4096`,
`reasoning false`. Those defaults truncate a turn mid-loop and overflow context
after roughly 110 tool calls, surfacing only as
`⚠️ Agent couldn't generate a response`; the real cause appears only in the
gateway log. This script corrects them for Claude Opus entries — override with
`MODEL_CONTEXT_WINDOW`, `MODEL_MAX_TOKENS`, `MODEL_REASONING` for a different
model or endpoint. Verify before raising them: declaring a window **larger**
than the endpoint accepts is worse than one too small, because the agent stops
compacting and starts taking hard 400s.

`memory_search` is disabled by this script. It embeds through an OpenAI-style
provider and NemoClaw's routed inference has no embeddings route
(`/v1/embeddings` → `no compatible inference route available`), so on an
Anthropic-compatible sandbox it fails with
`No API key found for provider "openai"` the first time the agent uses it.
Enabling it would require a live embeddings credential inside the sandbox,
which this integration exists to avoid.

### Long-running workflows

An orchestration workflow runs for hours across many stages, and two stock
defaults quietly stop it.

`agents.defaults.timeoutSeconds` is **600**. A single stage can exceed that on its
own — AnomalyGen SDG measured 520s — so an orchestration turn is cut off
mid-stage, and nothing reports it. Setup raises it to 3600
(`AGENT_TIMEOUT_SECONDS`).

The heartbeat is the **recovery path**, not just a liveness ping. When a turn dies
— a truncated SSE stream from the provider is the common case — OpenClaw will not
replay it: replay safety is decided by whether execution had already started, and
a tool cannot declare itself safe once it has. The agent is simply left idle, and
the next heartbeat is what resumes the work. OpenClaw's default cadence is
**30m**, so a dead turn can look like a hung run for half an hour. Setup sets 2m
(`HEARTBEAT_EVERY`), and `skipWhenBusy` means a short cadence costs nothing while
the agent is working — those wakes are deferred.

> A heartbeat turn is bounded by the same timeout, so the two settings must move
> together. Observed on a stock sandbox:
> `{"reason":"interval","status":"ok-empty","durationMs":619326,"silent":true}` —
> 619s against a 600s cap. The agent was woken, worked for exactly the limit, was
> truncated, and the event was recorded as **OK**. From outside that is
> indistinguishable from an idle agent.

Check it with `nemoclaw <sandbox> exec -- openclaw system heartbeat last`: a
`durationMs` close to your `timeoutSeconds` means truncation, not idleness.

### On corporate hosts

`ufw` ships **inactive** on stock Ubuntu and DGX Spark, so a default-deny host
firewall never reproduces there. On an IT-managed host it will silently drop
sandbox→host bridge traffic and surface much later as `HTTP 000`. Two rules are
needed, both scoped to the NemoClaw bridge (`172.19.0.0/16`, not the default
`172.17.0.0/16`): the gateway port (8080) and this server's port (9901). Setup
preflights and prints the exact rule; it deliberately never runs `sudo`.

Internal NVIDIA inference endpoints resolve to RFC1918 addresses, which
NemoClaw's SSRF guard rejects by default. Onboarding such an endpoint needs
`NEMOCLAW_TRUSTED_PRIVATE_INFERENCE_HOSTS=<host>`.

### Testing note

Use `nemoclaw <sandbox> exec` — never `docker exec` — to test anything the
agent must be able to see or reach. OpenShell attributes egress by process
tree, so `docker exec` falls outside it and the L7 proxy denies everything,
producing false failures. `docker exec` is correct only for host-side admin on
files (for example removing the root-owned tree that `docker cp` installs).

## Helper commands

```bash
# credentials — must be set in the shell that LAUNCHES the server
export HF_TOKEN=hf_...
export NGC_API_KEY=...                      # setup derives NGC_KEY from this

# onboard a sandbox
NEMOCLAW_AGENT=openclaw NEMOCLAW_PROVIDER=anthropicCompatible \
NEMOCLAW_ENDPOINT_URL=https://<endpoint>/v1/messages \
NEMOCLAW_MODEL=aws/anthropic/bedrock-claude-opus-5 \
NEMOCLAW_INFERENCE_API=anthropic-messages \
NEMOCLAW_TRUSTED_PRIVATE_INFERENCE_HOSTS=<endpoint-host> \
NEMOCLAW_PROVIDER_KEY="$KEY" \
  nemoclaw onboard --fresh --non-interactive --yes \
    --name <sb> --agent openclaw --sandbox-gpu

# install TAO — bash -ic, or ~/.bashrc is not sourced and the server gets no credentials
ssh <host> 'bash -ic "cd ~/tao-skill-bank/integrations/nemoclaw && ./setup-tao-nemoclaw.sh <sb> <workspace>"'
./setup-tao-nemoclaw.sh <sb> <workspace> --restart-server    # force restart after editing server.py

# verify
nemoclaw <sb> status
nemoclaw <sb> dashboard-url --quiet

# reset the agent between runs — close the TUI first, then confirm the session id changed
nemoclaw <sb> sessions reset agent:main:main --agent main --reason new
nemoclaw <sb> exec -- rm -f /sandbox/.openclaw/workspace/MEMORY.md
nemoclaw <sb> sessions list

# remove TAO
./uninstall-tao-nemoclaw.sh <sb> <workspace> --yes
```

Provider id is the internal `anthropicCompatible`, not the
`compatible-anthropic-endpoint` label `nemoclaw list` prints. Drop
`NEMOCLAW_TRUSTED_PRIVATE_INFERENCE_HOSTS` only for a public endpoint.
