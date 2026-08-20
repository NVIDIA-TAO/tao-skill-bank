# Local Image Attribute Augmentation

Read this reference only when `iterN/sdg` is the audited next action. The stage
extends the existing iteration transaction; it does not introduce another
workflow engine or state file.

## Execution frame

This reference governs the control-host generation frame, not the selected TAO
action platform. The prebuilt augmentation and auto-labeling images, and any
skill-managed model endpoints, run through the control host's Docker daemon.
TAO embedding, mining, train, and evaluate actions continue through the
selected Docker, SLURM, Kubernetes, Brev, or virtualenv consumer and its four
verbs. For a remote TAO platform, the platform staging receipt must include the
committed generated dataset before a downstream action is submitted.

Do not silently point the Docker CLI at a remote daemon: bind paths, loopback
endpoint URLs, output synchronization, and ownership checks in this helper are
defined in the control-host frame. If control-host Docker is unavailable, stop
during intake. An external endpoint replaces only skill-managed model serving;
it does not remove the prebuilt component-image requirement.

## Stage map

| Functional operation | Local operation | Endpoint | Input | Validated output | Retry or recovery | State result |
|---|---|---|---|---|---|---|
| remaining-gap plan | `prepare` | none | committed mined pairs, evaluation exclusion list | bounded `sdg_plan.json`, staged person folders | one corrected host-adapter retry; zero selections is terminal | uncommitted `sdg` work |
| endpoint deployment | `plan`, then `start` or `validate-external` | all roles | immutable `sdg_config.yaml` | ownership manifest and successful model probes | bounded readiness deadline; preserve owned containers on failure | endpoint evidence |
| pane preparation | `execute` preprocessing | none | selected per-person views | pane image plus layout metadata for every source | resume if journal and outputs agree | operation journal |
| augmentation smoke | first `execute` sample | image edit, VLM, LLM | one pane and residual attributes | image, prompt, metadata, verification pass | approved per-source attempt bound | smoke evidence |
| full augmentation | remaining `execute` samples | image edit, VLM, LLM | selected panes | accepted and rejected attempts kept separately | finite attempts; exhausted source is rejected | operation journal |
| crop split | `execute` split | none | accepted panes and layout metadata | per-view generated crops | resume only with complete split evidence | operation journal |
| labeling smoke | first accepted crop | VLM, LLM | one generated crop | `task/open_qa.json` with three captions per difficulty | one corrected component retry | smoke evidence |
| full labeling | remaining accepted crops | VLM, LLM | generated crops | validated open-QA files | one corrected component retry | operation journal |
| mining normalization | `execute` normalization | none | accepted crops, verified metadata, open QA, vocabulary | image/caption dirs, list, pairs, provenance manifest | complete output is reused; partial output stops for recovery | committed `sdg` |

Only a verification-passed image with bound metadata may reach splitting,
labeling, or normalization. Rejected attempts remain under `datagen/augmentation`
for diagnosis and never enter the training list. The normalizer also rejects any
source basename present in `iaa_splits/eval_list.txt`.

## Immutable defaults

The run copy at `${RESULTS_DIR}/config/sdg_config.yaml` records these default
roles and immutable revisions:

| Role | Model | Backend | API |
|---|---|---|---|
| image edit | `Qwen/Qwen-Image-Edit-2511` | `vllm-omni` | `/v1/models`, `/v1/images/edits` through the augmentation smoke test |
| verification VLM | `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` | `vllm` | `/v1/models`, `/v1/chat/completions` |
| query LLM | `Qwen/Qwen2.5-14B-Instruct` | `vllm` | `/v1/models`, `/v1/chat/completions` |

Compatible OpenAI-style models may be substituted only before approval. Record
an immutable model revision, correct backend, served model ID, expected API,
VRAM estimate, and endpoint smoke result. Do not switch the image-edit role to
a text-only server. A compatible shared VLM/LLM can be used by setting both
roles explicitly to that model and endpoint before approval.

The augmentation and auto-labeling workflow containers are prebuilt release
artifacts recorded under `images` in the run config. Customers never build
these images and do not need either implementation repository. Inspect or pull
only the approved immutable references. Serving images are version-tagged;
mutable `latest` tags are rejected throughout the materialized configuration.

## Endpoint intake and preflight

Choose exactly one mode:

- `managed`: provide non-empty explicit GPU ID lists for all three roles. IDs
  may be shared only when aggregate free VRAM satisfies the summed allocation.
- `external`: provide three already-running local HTTP(S) base URLs. GPU IDs
  are empty because the skill does not own those processes.

Before approval, inspect only Docker/runtime availability, GPU inventory,
ports, disk capacity, local image presence, and environment-variable presence.
Do not create the cache directory, pull an image, download a model, or start a
container. The planning command is read-only:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/manage_sdg_endpoints.py" plan \
  --config "$RESULTS_DIR/config/sdg_config.yaml" \
  --run-id "$(basename "$RESULTS_DIR")" \
  --output "$RESULTS_DIR/endpoints/plan.json"
```

For a strictly read-only pre-approval plan, omit `--output`; writing the plan
belongs after approval. The helper checks Docker's NVIDIA runtime, compute
capability, aggregate free VRAM, at least 150 GiB model-cache disk space,
deterministic container-name conflicts, and port collisions. It never emits a
credential value or widens an explicit device list.

## Launch approval rows

Add these rows to the consolidated preflight summary:

```text
Generation
  endpoint mode: <managed | external> (source=<user | default>)
  image edit: model=<id>@<revision>; endpoint=<managed port | URL>;
              gpu_ids=<explicit list | user-managed>; VRAM estimate=<MiB>
  VLM:        model=<id>@<revision>; endpoint=<managed port | URL>;
              gpu_ids=<explicit list | user-managed>; VRAM estimate=<MiB>
  LLM:        model=<id>@<revision>; endpoint=<managed port | URL>;
              gpu_ids=<explicit list | user-managed>; VRAM estimate=<MiB>
  generation budget: <N sources/iteration>; verification attempts=<1..5>;
                     caption policy=<all | easy | medium | hard>
  component images: <two pinned prebuilt images and local/pull status>
  serving images: <two pinned images and local/pull status>
  lifecycle: reuse matching run-owned containers; never replace or stop
             user-managed endpoints; retain run-owned containers on failure;
             stop only by an explicit owned-only cleanup action
```

Approval covers the listed image pulls, model downloads, endpoint startup,
dataset mutation, and GPU work. Any GPU, model, URL, port, image, budget, or
cleanup-policy change requires a changed-row approval.

## Endpoint lifecycle

After approval, managed mode starts or resumes only deterministic run-owned
containers:

```bash
mkdir -p "$RESULTS_DIR/endpoints"
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/manage_sdg_endpoints.py" start \
  --config "$RESULTS_DIR/config/sdg_config.yaml" \
  --run-id "$(basename "$RESULTS_DIR")" \
  --output "$RESULTS_DIR/endpoints/manifest.json"
```

External mode substitutes `validate-external` and writes the same manifest
path. Model discovery plus minimal inference is bounded by
`startup_timeout_s`, `retry_interval_s`, and `request_timeout_s`. The first
augmentation is the image-edit inference smoke test and must pass verification
before the batch proceeds.

A same-name container is reusable only when all workflow, run, and role labels
match. A stopped matching container may be restarted. Any other same-name
container or occupied managed port is a hard conflict. User-managed endpoints
are validated but never started, restarted, stopped, or replaced.

On failure, keep run-owned containers and report the sanitized status plus the
recorded `docker logs --tail 200 <name>` command. Cleanup is explicit:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/manage_sdg_endpoints.py" stop \
  --config "$RESULTS_DIR/config/sdg_config.yaml" \
  --run-id "$(basename "$RESULTS_DIR")"
```

This stops matching owned containers without removing them or their cache.

## Generation execution

Prepare from the committed history selection:

```bash
DATAGEN="$RESULTS_DIR/iter_${N}/datagen"
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/run_sdg_stage.py" prepare \
  --config "$RESULTS_DIR/config/sdg_config.yaml" --output-root "$DATAGEN" \
  --mined-pairs "$RESULTS_DIR/iter_${N}/mining/mined_pairs.json" \
  --gaps-parquet "$RESULTS_DIR/iter_${N}/gaps/kpi_gaps.parquet" \
  --attribute-vocab "$DATASET_ROOT/attribute_vocab.json" \
  --dataset-root "$DATASET_ROOT" \
  --eval-list "$RESULTS_DIR/iaa_splits/eval_list.txt"
```

Then run the bounded sequence. It performs the first-item smoke gates, full
batch, split, labeling, and normalization. Each accepted item and completed
operation is journaled atomically; rerunning after interruption skips validated
work.

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/run_sdg_stage.py" execute \
  --config "$RESULTS_DIR/config/sdg_config.yaml" --output-root "$DATAGEN" \
  --eval-list "$RESULTS_DIR/iaa_splits/eval_list.txt" \
  --attribute-vocab "$DATASET_ROOT/attribute_vocab.json"
```

Inspect the smoke image, `augmentation_smoke.json`, verification metadata, and
`auto_label_smoke_open_qa.json` before accepting the stage. A component exit 0
is insufficient without these artifacts and the final normalization checks.

Commit the stage once:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/commit_stage.py" \
  --results-dir "$RESULTS_DIR" --iter-label "iter${N}" --stage sdg \
  --endpoint-manifest "$RESULTS_DIR/endpoints/manifest.json" \
  --sdg-execution-manifest "$DATAGEN/sdg_execution_manifest.json" \
  --sdg-manifest "$DATAGEN/dataset/sdg_manifest.json" \
  --sdg-pairs "$DATAGEN/dataset/sdg_pairs.json" \
  --sdg-image-list "$DATAGEN/dataset/sdg_image_list.txt" \
  --sdg-status "$DATAGEN/status/sdg-normalize.host.status.json" \
  --summary "verified local generation normalized for iteration ${N}"
```

The subsequent `train-config` adapter adds this generated dataset alongside
the mined set. Evaluation directories are never mounted as generation inputs.

## Failure classes

- Retry one unchanged readiness probe until its fixed deadline. Timeout,
  malformed response, or wrong served model blocks the stage with the exact
  endpoint and recovery action.
- Retry one component command only after identifying a transient registry,
  network, or process failure. Do not change models or GPU IDs speculatively.
- Verification failure is data rejection, not infrastructure retry. Generate
  at most the approved attempts for that source and retain every attempt.
- Port ownership conflict, insufficient GPU/VRAM/disk, unsupported compute
  capability, missing model access, partial normalization, malformed open QA,
  provenance mismatch, or evaluation leakage is non-retryable until the named
  condition is corrected and any changed approval row is reapproved.
- A committed `sdg` stage is never rerun. An uncommitted stage resumes only
  from its validated progress/status evidence.
