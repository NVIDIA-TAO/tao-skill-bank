<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Tutorial: run the DEFT AOI workflow on NemoClaw

End to end on a single GPU workstation — install NemoClaw, onboard a sandbox,
install TAO, run the DEFT AOI improvement loop, tear it down.

Verified 2026-08-10 on Ubuntu 22.04.5 x86_64, 36 vCPU / 31 GiB, RTX A6000 (49 GB),
NemoClaw v0.0.97, OpenShell 0.0.85, OpenClaw 2026.7.1, Claude Opus 5.

---

## 0. Host prerequisites

```bash
docker --version                 # + nvidia-container-toolkit
nvidia-smi                       # GPU visible
node --version                   # >= 22.19.0 (the installer bootstraps this)
uv --version                     # https://astral.sh/uv
df -h /                          # >= 200 GB free; AnomalyGen base checkpoints alone are ~140 GB
docker login nvcr.io             # username: $oauthtoken, password: your NGC key
```

Export the two credentials the workflow needs. Put them in `~/.bashrc` if you
want them permanently — but see the login-shell warning in step 4.

```bash
export NGC_API_KEY=...           # nvcr.io image pulls
export HF_TOKEN=...              # gated HuggingFace weights
```

The HuggingFace account behind `HF_TOKEN` must have accepted the licence for
`nvidia/Cosmos-Predict2-2B-Text2Image`, which is `gated: auto`. Verify before you
start — a 401 here surfaces two hours into the run:

```bash
curl -sS -o /dev/null -w '%{http_code}\n' -r 0-1000 \
  -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image/resolve/main/model.pt
# 302 = authorized.  401 = accept the licence on the model page first.
```

## 1. Install NemoClaw

Follow <https://docs.nvidia.com/nemoclaw/latest/get-started/quickstart.html>, or
from a checkout:

```bash
git clone https://github.com/NVIDIA/NemoClaw && cd NemoClaw && ./installer.sh
nemoclaw --version               # expect v0.0.97 or newer
```

`nemoclaw` installs to `~/.local/bin`. If it is not on your PATH in a
non-interactive shell, prefix commands with
`export PATH=$HOME/.local/bin:$PATH`.

## 2. Onboard a sandbox

```bash
NEMOCLAW_AGENT=openclaw \
NEMOCLAW_PROVIDER=anthropicCompatible \
NEMOCLAW_ENDPOINT_URL=https://your-endpoint/v1/messages \
NEMOCLAW_MODEL=aws/anthropic/bedrock-claude-opus-5 \
NEMOCLAW_INFERENCE_API=anthropic-messages \
NEMOCLAW_PROVIDER_KEY="$YOUR_INFERENCE_KEY" \
  nemoclaw onboard --fresh --non-interactive --yes \
    --name deft --agent openclaw --sandbox-gpu
```

Three things that are easy to get wrong:

- `NEMOCLAW_PROVIDER` takes the **internal id** `anthropicCompatible`. The
  `compatible-anthropic-endpoint` label printed by `nemoclaw list` is rejected.
- `NEMOCLAW_ENDPOINT_URL` is required non-interactively and undocumented in the
  quickstart. The URL is normalised, so the full `/v1/messages` path is fine.
- If your endpoint resolves to a private (RFC1918) address, NemoClaw's SSRF
  guard fails it with `no HTTP response` even though `curl` gets 200. Add:

  ```bash
  export NEMOCLAW_TRUSTED_PRIVATE_INFERENCE_HOSTS=your-endpoint-host
  ```

Confirm:

```bash
nemoclaw list                    # deft, sandbox GPU
nemoclaw deft status             # Phase: Ready, Sandbox GPU: enabled (CUDA verified)
```

## 3. Lay out the workspace

The workspace root is what every skill means by `<workspace>`. It must contain
`train/`, `kpi/`, `results/` and `augmentation/` **directly** — pointing one
level too high is the single most common cause of a run spent resolving paths
that do not exist.

```
~/deft-workspace/
├── train/base/{training_set.csv,validation_set.csv}
├── kpi/{testing_set.csv,images/}
├── augmentation/
│   ├── backbone/c_radio_v2_b.safetensors
│   ├── encoders/siglip-base-patch16-224/
│   └── mining_pool/{mining_pool.csv,images/}
└── results/                     # empty
```

Labels must be canonical: `PASS` verbatim, every other label lowercase and
stripped. Mixed case fails `validate_training_csv.py` with exit 2, and splits
your taxonomy (`Missing` vs `missing`) across RCA and routing:

```bash
cd ~/deft-workspace/kpi && cp testing_set.csv testing_set.csv.orig && python3 -c "
import csv
rows=list(csv.DictReader(open('testing_set.csv')))
for r in rows:
    if r['label'] != 'PASS': r['label'] = r['label'].lower().strip()
w=csv.DictWriter(open('testing_set.csv','w',newline=''), fieldnames=rows[0].keys())
w.writeheader(); w.writerows(rows)"
```

Keep archives and anything you cannot lose **outside** the workspace root — the
agent has full read-write access to everything inside it.

## 4. Install TAO into the sandbox

Run this from a **login shell**. The server forwards secrets into containers by
name from its own environment, frozen at launch, and `~/.bashrc` is not sourced
by `ssh host 'command'` — that produces a server that silently cannot fetch
anything gated.

```bash
git clone https://github.com/NVIDIA-TAO/tao-skills-bank && cd tao-skills-bank/integrations/nemoclaw
./setup-tao-nemoclaw.sh deft ~/deft-workspace
```

Remotely:

```bash
ssh <host> 'bash -ic "cd ~/tao-skills-bank/integrations/nemoclaw && ./setup-tao-nemoclaw.sh deft ~/deft-workspace"'
```

Expect, near the end:

```
[tao-nemoclaw] credentials forwarded into containers: HF_TOKEN NGC_API_KEY NGC_KEY
[tao-nemoclaw] model capabilities corrected for aws/anthropic/bedrock-claude-opus-5
[tao-nemoclaw] ✓ bridge OK (HTTP 400 — server answered)
```

If credentials read `NONE`, stop and re-run from a login shell — nothing later
fixes it without restarting the server:

```bash
./restart-tao-mcp.sh ~/deft-workspace
```

On a corporate host with `ufw` active, setup prints the two rules it needs.
Both are scoped to the NemoClaw bridge, not the default docker bridge:

```bash
sudo ufw allow from 172.19.0.0/16 to 172.19.0.1 port 8080 proto tcp
sudo ufw allow from 172.19.0.0/16 to 172.19.0.1 port 9901 proto tcp
```

## 5. Verify the agent can see TAO

```bash
nemoclaw deft connect          # then: openclaw tui
```

Ask: **"What MCP tools do you have?"** — expect 12: `tao_ls`, `tao_read`,
`tao_write`, `tao_exec`, `tao_pull`, `tao_run`, `tao_list`, `tao_status`,
`tao_logs`, `tao_stop`, `tao_rm`, `tao_cleanup_results`.

## 6. Run the loop

Give every parameter explicitly and require the summary before execution:

> Run the DEFT AOI loop on this workspace. `max_iterations=1`, `num_epochs=2`,
> `batch_size=16`, 1 GPU. KPI target: FAR < 10% at recall=100%. Run the full
> loop including the AnomalyGen synthetic-defect arm. Show me the Pre-Flight
> Summary and wait for my approval before running anything.

Check the summary before approving: each parameter should be marked `user`,
`spec` or `default`, and your explicit values must appear as `user`. Then the
loop runs unattended through baseline train → inference → evaluate → RCA →
routing → AnomalyGen → mining → merge → iter1 train → evaluate → finalize.

Expect hours. The AnomalyGen base checkpoints dominate: ~140 GB of HuggingFace
downloads before any compute.

## 7. Monitor

```bash
nemoclaw deft logs --follow
tail -f ~/deft-workspace/results/run_*/loop_log.jsonl
tail -f ~/deft-workspace/tao-mcp-server.log
watch -n30 nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

Use `nemoclaw deft exec` — never `docker exec` — to check anything the agent
must be able to see. OpenShell attributes egress by process tree, so
`docker exec` falls outside it and reports false failures.

For the web dashboard instead of the TUI:

```bash
nemoclaw deft dashboard-url --quiet         # prints an authenticated URL
ssh -N -L 18790:127.0.0.1:18790 <host>      # from your laptop
```

## 8. Results

```bash
ls ~/deft-workspace/results/run_*/
#   deft_state.json         status == complete when the loop finished
#   DEFT_Loop_Report.html   rendered after every committed stage
#   loop_log.jsonl          one line per stage
```

Completion is what `deft_state.json` says, not what the agent says: `status`,
`iterations.baseline.status` and the final iteration's `status` must all read
`complete`.

Reference run for this dataset (211 train rows, 21,148 KPI rows, mining only,
AnomalyGen skipped): FAR at recall=100% improved **99.28% → 76.46%** over
baseline plus one iteration.

## 9. Teardown

```bash
./uninstall-tao-nemoclaw.sh deft ~/deft-workspace          # add --purge-bank to remove the cloned bank
```

Removes the policy preset, the MCP registration, the sandbox skill tree, the
`AGENTS.md` block and the host server. Workspace data is never touched.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `GatedRepoError: 401` on a HuggingFace model | server started without `HF_TOKEN` | `./restart-tao-mcp.sh <workspace>` from a login shell |
| `credentials forwarded into containers: NONE` | non-interactive shell, `~/.bashrc` not sourced | re-run setup via `bash -ic` |
| `MCP server already running` but nothing works | setup matches any process named `server.py` | `ss -ltnp \| grep 9901` to find the real owner; kill it by process group |
| `bridge unreachable (HTTP 000)` | host firewall dropping bridge traffic | add the two `ufw` rules from step 4 |
| Agent repeatedly misses `results/<name>.json` | workspace root one level too high | re-run setup pointing at the directory that holds `train/` and `kpi/` |
| `⚠️ Agent couldn't generate a response` | context overflow on guessed model capabilities | check `contextWindow`/`maxTokens` in `openclaw.json`; setup corrects these for Opus |
| `args: must be object`, repeatedly | Tool Search wrapper | setup disables it (`tools.toolSearch.enabled = false`); confirm in `openclaw.json` |
| `ValueError: subtask requires -e/--experiment_spec_file` | spec flag omitted | the spec path is an explicit `-e` argument in `tao_run`'s command list |
| A stage's outputs are not under its stage directory | `tao_run` isolates each job under `<results_subdir>/.tao-jobs/<token>/` | use the path `tao_run` returned; it is recorded in `deft_state.json` |
| `memory_search` unavailable | no embeddings route on an Anthropic-compatible endpoint | expected; setup disables memory search |
