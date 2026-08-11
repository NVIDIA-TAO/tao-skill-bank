# TAO on NemoClaw — Agent Operating Guide

You run NVIDIA TAO workflows on the host GPU from inside a NemoClaw (OpenShell)
sandbox. TAO executes on the **host** through the `tao` MCP tools; you never run
Docker or hold credentials yourself.

**Do not ask which platform to use, and do not use the Brev, SLURM, Kubernetes or
local-docker dispatch skills.** Under NemoClaw the execution platform *is* the host
`tao` MCP server.

Pick between the two execution tools by what the work needs — the tool
descriptions cover the rest:

- `tao_exec` — CPU shell over the whole workspace at `/workspace`. Everything that
  is not GPU compute: inspect data, move and unpack files, author specs, run a
  skill's helper scripts, and stage models or datasets (host network,
  `HF_TOKEN`/`NGC_KEY` forwarded). Never burn a GPU job on a file copy, a
  download, or a helper script.
- `tao_run` — the GPU job. It never pulls, so `tao_pull` first or it fails fast.

Never move a file through your own context (no base64, no chunking) — read and
write it in the shell.

## Orchestration: run to completion

A workflow runs to completion in this session. Do not finish a stage and wait to
be asked for the next one; the user should never have to say "continue".

- After each stage: commit it, print **one line** — stage, key metric, next stage
  — and start the next stage immediately.
- Resume from **disk, not from the conversation**. Re-read the workflow's state
  file and continue from the first uncommitted stage. This is what makes a resumed
  or restarted session correct.
- On a bare "continue", "status" or "progress": re-read state, say what stage is
  running and what is next, and carry on. Do not ask what to do.
- Stop only for the skill's own confirmation gate, a hard failure you cannot fix,
  or a genuinely missing input. Say which of the three it is, and exactly what you
  need to proceed.

## Heartbeats: you will be woken; keep going

This sandbox is always-on and the harness wakes you on an interval. A heartbeat is
not a greeting — it is "carry on". On every heartbeat, if a workflow is in flight:

1. Re-read the workflow's state file from disk.
2. Resume the first uncommitted stage. Do not re-run a committed one.
3. Emit **one line**: stage, key metric, next stage.

Never treat a heartbeat as a reason to summarise, ask a question, or wait. If the
workflow is finished, say so in one line and stop; if nothing is in flight, stay
silent.

Long stages outlive a single turn. If you are woken while work you launched is
still running on the host, check it with `tao_status`/`tao_logs`, report one line,
and let it continue — do not relaunch it. A second writer against the same
experiment corrupts the run.

## Batch independent checks

One tool call is one turn. Five `tao_exec` calls to answer five questions cost
five turns; one shell line answers all five at once:

```
tao_exec: ls -d /workspace/results/run_*/ | tail -1; wc -l /workspace/train/base/*.csv
```

Split them only when a later command depends on an earlier result.

## Long operations: background them, then poll

A `tao_exec` that blocks for minutes hits the MCP request timeout and returns an
error while the work keeps running on the host — leaving you no handle on it. For
a large download or any multi-minute step:

```
tao_exec: nohup <cmd> > /workspace/results/<stage>.log 2>&1 & echo started
tao_exec: tail -5 /workspace/results/<stage>.log     # poll at <=30s intervals
```

## Preflight — before every `tao_run`

There is no SDK here to stage inputs or resolve images for you: `tao_run` starts a
plain container that reads only **local** files under `/data` and `/results`.
Verify the skill's declared contract from its own `references/skill_info.yaml` —
one generic routine, no per-model rules. Do not launch until every check passes.

1. **Image.** Resolve `container_image` (absolute `nvcr.io/...`, or a
   `# versions-key:` comment looked up in `versions.yaml`) and `tao_pull` it.
2. **Registry auth.** If `tao_pull` returns an auth error, stop and report — do
   not retry blindly.
3. **Declared inputs.** For each entry under `actions.<action>.inputs`, confirm
   the path exists: datasets under `/data`, checkpoints under `/results`,
   backbones staged locally. A remote URI (`hf://`, `https://`, `ngc://`) is **not**
   resolved inside the container — stage it with `tao_exec` first, then point the
   spec at the local path.
4. **Resources.** Training needs `shm_size ≥ 8g`.

Report each check as `PRESENT` or `MISSING: <exact fix>`, resolve, re-check, then
launch. If a job goes missing after launch, `tao_list` recovers its `job_id`.

## Paths

Data and results live on the **host** — never look under `/sandbox` for datasets.

- `tao_exec` sees the whole workspace at `/workspace`. This is the `<workspace>`
  every skill reference means: `train/`, `kpi/`, `results/` sit directly under it.
- `tao_run` sees `<data_subdir>` at `/data` and its own isolated results tree at
  `/results`, and **nothing else**. Write spec paths as `/data/...` and
  `/results/...`, never `/workspace/...` and never a host path — a `/workspace`
  path staged by `tao_exec` does not exist inside a GPU job.
- `tao_run` returns that job's exact `results_subdir`; use the returned path.

Workflow state is not at `<workspace>/results/<name>.json`. Timestamped workflows
write under a per-run directory — DEFT AOI's is `results/run_<TS>/deft_state.json`.
Find it, don't guess it:

```
tao_exec: ls -d /workspace/results/run_*/ | tail -1
```

TAO subtasks need the spec flag explicitly, or they fail with
`requires the following argument: -e/--experiment_spec_file`:

```
command: ["visual_changenet", "train", "-e", "/results/<stage>_train.yaml"]
```

The skill bank is in the workspace at `tao-skills-external/`, so helper scripts
and `versions.yaml` are reachable from both tools.

## Cleanup

`tao_stop` and `tao_rm` act on the container only — they do **not** delete
results, checkpoints or caches. Do not report container removal as output cleanup.

To dispose of a failed or throwaway run: `tao_stop` if it is still running,
inspect any logs you need, then `tao_cleanup_results`. Never `tao_rm` first —
removing the container destroys the metadata that authorizes cleanup. Keep a
successful run's result path until its deliverables are handed off.

If `tao_run` reports an ambiguous Docker response, reconcile the job name in the
error with `tao_status`/`tao_stop` before resubmitting — never launch a second
writer against the same experiment.

## Read the skill first

Read the relevant `tao-*` skill's `SKILL.md` and its `references/skill_info.yaml`
before acting — they are the contract: the image, the action command, the spec
schema, and any mandatory steps. There is no plugin harness here, so a skipped
mandatory step simply does not happen.

Follow the skill's intake rules: use its defaults, and ask only for genuinely
required inputs plus its own confirmation gate. **Never ask about a parameter that
has a default.** Run workflows inline in this session — no cron, no task
scheduling.
