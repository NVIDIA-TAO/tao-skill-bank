---
name: tao-run-deft-iaa
description: >
  Use for requests such as "Improve my SigLIP2 image retrieval model on my
  attribute-labelled dataset until it stops getting better." Keep improving
  an NVIDIA TAO CLIP / SigLIP2 image-retrieval model on attribute-labelled
  data until improvement stops, its retrieval KPI reaches a target, or the
  iteration budget is exhausted. Route iterative attribute-labelled
  image-retrieval improvement here even when the customer does not know the
  DEFT or Image Attribute Augmentation (IAA) names. The self-contained loop performs dataset
  local model deployment or compatible endpoint validation, verified image
  generation, auto-labeling, dataset preparation, zero-shot evaluation,
  attribute gap analysis, caption-space k-NN mining, history-aware selection,
  retraining, and re-evaluation. Treat
  `tao-deft-iaa` as shorthand for this canonical `tao-run-deft-iaa` workflow.
  Do not use for standalone CLIP training, one-off evaluation or embedding,
  generic k-NN mining, or AOI/ChangeNet DEFT workflows.
license: Apache-2.0 AND CC-BY-4.0
compatibility: Requires one supported TAO compute platform (Docker, SLURM, Kubernetes, Brev, or virtualenv), accessible NVIDIA GPUs, the two IAA dataset export archives, and Python 3.9+ for control; virtualenv execution additionally requires the documented CPython 3.12 pyt and ds profiles. Airflow is an optional IAA-only orchestrator over any supported compute platform.
metadata:
  author: NVIDIA Corporation
  version: "0.6.0"
allowed-tools: Read Bash Write
tags:
- application
- workflow
- deft
- iaa
- clip
- retrieval
- loop
---

# IAA DEFT Workflow

Run IAA as one resumable, disk-backed workflow. All logic and adapters ship
here; no separate source checkout or remote generation service is required.

Docker, SLURM, Kubernetes, Brev, and virtualenv consume platform-neutral action
bundles through four verbs and job-records. Airflow optionally orchestrates any
of those five backends; it is not a sixth compute platform. The immutable
state keeps `config.platform` as the backend and adds
`config.orchestrator=airflow` only when requested. Read
`references/airflow-execution.md`.
Virtualenv execution uses separate immutable `pyt` and `ds` runtime profiles;
the workspace control `.venv` is not an execution runtime.

Every workload runs in the selected platform's compute frame; the control host
only prepares, monitors, synchronizes, and audits. Distributed generation uses
at most `generation_nodes=N` eight-GPU workers and a coordinator. Image-edit
endpoints are single-request slots, one per GPU.
Evidence records active versus approved `8*N` capacity; zero ready workers
fail. Docker and virtualenv use this contract on one machine. See
`references/platform-execution.md`.

## Entry Contract

`tao-deft-iaa` and `tao-run-deft-iaa` select this same workflow. If the user
did not choose a platform, ask once among Docker, SLURM, Kubernetes, Brev, and
virtualenv; never default to Docker. If the user requests Airflow, retain the
selected backend and record Airflow separately as the orchestrator. On resume,
the immutable platform and optional orchestrator in `deft_state.json` are
already selected and must not be changed. Legacy runs whose platform is
`airflow` remain resumable through the compatibility contract, but new runs
must not use that overloaded representation.

Use two intake phases:

1. Perform only bounded, lightweight path discovery. Two explicit archive file
   paths win and suppress all archive searching. An explicit archive directory
   is treated as one approved search root. Otherwise, resolve the conventional
   `~/workspace` and use only these deduplicated search roots: `~/iaa`, the
   workspace root, and its `iaa/`, `input/`, and `inputs/` children. Check the
   workspace root itself only; beneath each other search root, inspect the root
   and directories at most two levels below it. This AOI-style bounded-subtree
   lookup supports nested export/drop directories without turning into a home
   or repository scan. Never follow symlinks or add the current checkout,
   source repositories, development trees, or workspace dataset-output
   trees as implicit search roots.

   An archive candidate contains direct regular-file children
   `images_raw.tar` and `meta.tar.gz`; the workflow needs one such pair, not
   multiple dataset directories. Follow the classification and provenance
   format in `references/preflight.md` so the user sees every search root, its
   reason and depth bound, the directory in which a pair was found, and whether
   it is archive-only or mixed with extracted data.

   Enumerate run-state files only at
   `<workspace>/results/run_*/deft_state.json`. Read the minimal identity fields
   and present a resume candidate only when `workflow` is exactly
   `tao-run-deft-iaa`; never offer AOI or unidentified DEFT state as IAA.
   Do not validate large archives to EOF or inspect platforms, images, GPUs, or
   credentials yet.
2. If `max_iterations` or a time budget is absent, ask one consolidated
   question for that required value and any genuinely ambiguous path/run
   choice. State the documented defaults and that a complete read-only
   preflight plus approval summary follows. Do not ask whether the user wants
   optional KPI, authentication, or parameter overrides; apply their defaults
   unless the prompt already supplies an override.

After required intake is resolved, discover and validate:

- workspace and either a new `${RESULTS_DIR}` or one existing run to resume;
- `images_raw.tar` and `meta.tar.gz`; `SHA256SUMS` is optional;
- `max_iterations`, or a user-supplied time budget from which an iteration
  limit can be estimated;
- metric name, query type, operator, and optional target;
- whether the deployment requires authenticated Hugging Face model access;
- selected TAO execution platform and its platform-specific prerequisites;
- optional IAA-only Airflow orchestration and its API, DAG, shared-storage,
  credential-presence, and backend-consumer prerequisites;
- managed platform-local endpoints with explicit GPU allocation for image edit,
  VLM, and LLM; for distributed platforms also record `generation_nodes`;
- only when the user explicitly requests endpoint reuse, three user-supplied,
  already-running compatible local endpoint URLs;
- any explicit epoch, GPU, mining, continual-learning, or visualization
  overrides.

For a new run, `RESULTS_DIR` must be a child of `WORKSPACE`. `DATASET_ROOT`
must be below a workspace data directory (for example
`$WORKSPACE/data/iaa_v31_tao_ft`), not directly below the workspace; neither
path may contain the other. All approved paths are absolute and non-symlink.

Never replace an explicit value with a heuristic. Defaults apply only to
unspecified values and must be identified as defaults in the pre-flight
summary. `max_iterations` has no default. An absent metric target means an
ungated run that stops after `max_iterations`. Hugging Face token forwarding
defaults to disabled because the bundled model is public.

Managed deployment is the default customer path. Never discover listening
ports and switch to external mode, infer reuse from existing containers, or
offer endpoint reuse merely because a compatible service is present. External
mode requires the user's explicit reuse request, all three URLs, and the
materializer's `--reuse-external-endpoints` evidence flag.

If required information remains missing after full discovery, ask one
consolidated follow-up. A normal invocation should need no knowledge of stage
modules, container mounts, state files, or bundled-runtime function
signatures.

## Safety Gate

Perform only read-only discovery before approval: resolve paths, inspect file
metadata and archives, check process-environment variable presence, inspect
local images, inspect GPUs, and audit an existing run. Credentials come from
the launching process environment or a user-approved env file. Source only a
path the repository contract permits, in the same shell as the consuming
command; never print, grep, copy, or otherwise inspect its contents or echo a
credential value. If a required variable is absent, tell the user which name
to export in the shell that launches the agent; never ask for its value in
chat. Do not inspect credential-file metadata when no credential is needed. If
the user explicitly asks for a file-permission check, `stat` only that named
file and warn about group/other readability.

Show the summary defined in `references/preflight.md`, including every
parameter and source, planned file creation/extraction, image pulls, estimated
runtime, and resume status. Wait for explicit approval before registry login or
pulls, platform submit, package installation, archive extraction, config/state
creation, or any write under the workspace.

If an approved parameter later changes, show the changed summary rows and get
approval again before continuing. No confirmation is needed between unchanged,
already-approved stages.

## Execution Contract

1. After approval, follow `references/preflight.md` to prepare the runtime,
   materialize the immutable run config, and initialize state once. Never
   reinitialize an existing run.
2. Before every stage, after context compaction, and before any completion
   claim, run:

   ```bash
   "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
     "$SKILL_ROOT/scripts/audit_deft_run.py" --results-dir "$RESULTS_DIR"
   ```

   Continue only when the audit reports `IN_PROGRESS` and its `next_action`
   matches the intended stage. For `INVALID`, stop launching work and report
   the listed inconsistency. For nonterminal `FAILED` whose `next_action` is
   `loop_stop`, follow the pipeline reference and commit `hard_stop`; for
   terminal `FAILED`, report the failure and launch no more work. For
   `COMPLETE`, do not rerun a stage.
3. Read only the current stage reference named by `read_before_action`. Use
   `run_iaa_stage.py` directly only in a local compute frame. For remote runs,
   dispatch deterministic adapters through `run_deft_action.py` as zero-GPU
   platform actions. For every TAO action, use
   `run_deft_action.py prepare`, reconcile any interrupted launch, bind the
   exact request-owned job-record before native submit, dispatch the emitted
   bundle through the selected platform's four verbs, synchronize remote
   outputs, capture native logs, then use `run_deft_action.py finalize`. Follow
   `references/platform-execution.md`; never assemble an untracked launch. If
   Airflow orchestration is selected, wrap those exact four verbs with the
   signed application-local envelope in `references/airflow-execution.md`;
   Airflow must not reinterpret backend GPUs, staging, ownership, or topology.
   For `sdg`, use
   `manage_sdg_endpoints.py` for prebuilt-image checks and endpoint lifecycle,
   then `run_sdg_stage.py`, as documented in `references/local-sdg.md`. These
   helpers reconstruct paths and images from immutable state; do not depend on
   a previous `cd` or hidden environment mutation.
4. A command succeeds only when its exit status is zero and its documented
   output checks pass. Capture verbose output at the action-owned log path;
   inspect the final error block or at most the last 40 lines.
5. Commit each successful or terminally failed stage exactly once with
   `commit_stage.py`. It validates artifact structure, freshness, iteration
   scope, command evidence, leakage, and transition order before applying a
   recoverable, journaled update to `deft_state.json` and `loop_log.jsonl`.
   Never call
   `log_stage.py` or `record_metric_result.py` directly, and never hand-edit
   canonical state, log, metric, or history files.
6. Render the HTML report with `render_deft_report.py`. Reporting is a
   deterministic read of audited state; it does not require another agent.

All recorded artifact paths are absolute host paths under `${RESULTS_DIR}`.
Baseline artifacts live under `zs/`, iteration N under `iter_N/`, and run-wide
splits, source embeddings, and history files at the run root.

## Workflow

The normal path is linear, with only two meaningful decisions:

```text
pre-flight approval
  -> dataset_setup -> pool_embed -> baseline evaluate
  -> KPI met? yes: loop_stop
              no: baseline gap_analysis
  -> for N = 1..max_iterations:
       data_mining -> history_select -> sdg -> visualize -> train -> evaluate
       -> KPI met? yes: loop_stop
                   no and N < max: gap_analysis -> next iteration
                   no and N = max: loop_stop
  -> final audit -> HTML report
```

Stage ownership and exact commands are in
`references/scripts-and-agents.md`. Read the detailed reference only when its
stage is next:

| Stage | Reference | Required result |
|---|---|---|
| dataset setup | `references/data-layout.md` | verified rebuilt dataset, five split files, non-empty source-pool parquet |
| pool embedding and mining | `references/mining.md` | fresh command evidence plus non-empty, schema-checked parquet outputs |
| evaluate and train | `references/clip-train-eval.md`, `references/metric-contract.md` | successful TAO status, bound metric evidence; for train, a fresh best and normalized checkpoint |
| gap analysis | `references/gap-analysis.md` | non-empty iteration-scoped gaps parquet |
| history selection | `references/mining.md` | budgeted mined set, cumulative/history entry, zero eval leakage |
| platform-local generation | `references/local-sdg.md` | accepted provenance-bound crops, validated open QA, normalized image-text pairs, endpoint evidence |
| visualization | `references/visualization.md` | enabled artifacts and command evidence, or a config-authorized skip |

`visualize` is the only optional stage. Commit it with `--skip` only when both
visualization settings are false in the approved config. Do not turn off a
failing visualization mid-run without revising and reapproving the config.

## Loop and Recovery Rules

- The loop is bounded by `max_iterations`; never create an iteration outside
  that range. Once the KPI passes, the only legal next transition is
  `loop_stop`. Do not mine or train again.
- Do not use open-ended polling. Retain the job id, poll the selected platform's
  native status no more often than every 30 seconds while
  continuing to update the user.
- Never repeat an unchanged failed command speculatively. Classify the failure,
  inspect its final log block, make one evidence-based correction permitted by
  the stage reference, then retry once. Stable command status records persist
  attempt 1/2, and the wrappers refuse a third attempt after context loss. The
  audit must pass before advancing.
- A transient registry/network interruption or GPU contention may be retried
  once after evidence shows the condition changed. A second equivalent failure
  is terminal for the run.
- The platform producer's `dispatch-repair` verb is not a third workload
  retry. Use it only when both normal virtualenv attempts are finalized and
  the helper proves that each stopped inside an allowlisted shim check before
  the TAO CLI ran. It preserves both attempts, refuses any workload output,
  runtime/data/model failure, unknown log, active job, or second repair, and
  emits one distinct request/job lineage. Never use it to bypass an ordinary
  exhausted retry budget.
- The producer's `launcher-repair` verb is the equally narrow SLURM training
  counterpart. Use it only after both normal `clip train` attempts are
  terminal and the helper proves, in order, the exact one-task Lightning
  device mismatch and the canceled rank-0 `MEMBER: 1/2` initialization hang.
  It is forbidden after rank 1, a first batch, any output/checkpoint,
  runtime/data/model or unknown failure, an active job, or a prior repair.
  Dispatch its single distinct request normally through the SLURM skill; do
  not create another workload attempt.
- The producer's `unbound-replay` verb is the one bounded recovery for an
  allowlisted deterministic SLURM adapter whose terminal attempt-1 job lacks
  its pre-submit binding. It requires exactly one owned COMPLETE job, no
  platform status or binding, quarantines and hashes every untrusted output,
  preserves request/job/log evidence, and emits a distinct attempt-2 action.
  Never retroactively create a binding, accept the quarantined output, use the
  verb for GPU or mutating adapters, or run it more than once.
- Never retry checksum or rebuild verification failure, zero-row mining,
  schema/cardinality failure, missing eval-split evidence, eval leakage,
  history conflict that fails documented resume, metric-contract mismatch,
  stale/cross-iteration output, or an unsupported GPU architecture. Commit an
  error when state permits and stop with the exact recovery action.
- If a command wrote outputs but crashed before stage commit, run the audit.
  Reuse outputs only when their command status is successful, fresh, correctly
  scoped, and all stage validators pass. History selection has one supported
  recovery: rerun the deterministic adapter with `--resume`; never delete or
  edit a history entry by hand.
- The `sdg` stage has two generation attempts per source by default and permits
  only an approved bound from `1..5`. A failed verification rejects that
  attempt; exhaustion rejects the source. At least one accepted source is
  required. Its operation journal resumes completed preprocessing, accepted
  samples, splitting, and labeling without repeating them.
- Manual modification or truncation of `deft_state.json` or `loop_log.jsonl`
  invalidates the run. Preserve it for diagnosis and start a new results
  directory.

## Metric and Stop Semantics

The approved metric contract is immutable for the run. Evaluation must parse
the exact iteration's `nvidia_pas_metrics_aggregate.csv`; the result records
its source path and is re-derived during commit and audit. Checkpoint ranking
and best-run reporting follow the approved operator (`>=`/`>` chooses the
higher value, `<=`/`<` the lower), not a hard-coded metric convention.

Successful completion is exactly one of:

- `loop_stop(reason=kpi_met)` after a bound evaluation satisfies its target;
- `loop_stop(reason=max_iterations)` after the final allowed evaluation when
  the target is unmet or absent.

`hard_stop` is a failed terminal outcome, not successful completion.

## Completion

At a normal loop boundary:

1. commit `loop_stop` with the audit-selected reason;
2. render `${RESULTS_DIR}/DEFT_Loop_Report.html` with trigger `loop-end`;
3. require this command to exit zero:

   ```bash
   "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
     "$SKILL_ROOT/scripts/audit_deft_run.py" \
       --results-dir "$RESULTS_DIR" --require-complete
   ```

Report the stop reason, baseline and best metric, best iteration/checkpoint,
completed iteration count, report path, and any warnings. A checkpoint, CSV,
HTML file, or assistant statement alone is not completion evidence.

## Progressive References

| Need | Read |
|---|---|
| read-only checks, approval summary, initialization | `references/preflight.md` |
| stage commands and script interfaces | `references/scripts-and-agents.md` |
| platform staging, four verbs, job records, finalization | `references/platform-execution.md` |
| optional IAA-only Airflow orchestration over all five compute backends | `references/airflow-execution.md` |
| state transitions and resume behavior | `references/pipeline-and-state.md` |
| dataset/archive contract | `references/data-layout.md` |
| KPI parsing and evidence | `references/metric-contract.md` |
| gap generation | `references/gap-analysis.md` |
| embeddings, k-NN, selection | `references/mining.md` |
| platform-local endpoints, augmentation, labeling, normalization | `references/local-sdg.md` |
| contact sheets and t-SNE | `references/visualization.md` |
| train/evaluate/checkpoints | `references/clip-train-eval.md` |
