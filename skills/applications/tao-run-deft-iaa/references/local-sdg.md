# Platform-local Image Attribute Augmentation

Read this reference only when `iterN/sdg` is the audited next action. The stage
extends the existing iteration transaction; it does not introduce another
workflow engine or state file.

## Execution frame

All inference, component execution, and dataset mutation run in the selected
platform's compute frame. The control host may provision, submit, monitor,
synchronize completed outputs, and audit evidence only. It must not become a
fallback data plane for a remote SLURM, Kubernetes, or Brev run. Optional
Airflow orchestration does not change that compute boundary.

Managed distributed generation uses one platform-local coordinator and up to
`generation_nodes=N` independent single-node image workers. Each image worker
has exactly eight GPUs and starts eight independent, single-GPU image-edit
microservices. These are capacity-one endpoints: each GPU accepts one image
request at a time. The coordinator validates an ordered runtime pool selected from
the approved `8*N` maximum, runs the VLM and LLM on separately approved GPUs,
and dispatches each attempt to a free node/GPU slot. At least one complete
worker must be ready; requested and active capacities remain explicit in
evidence. Docker and virtualenv use the same
dispatcher on one machine and derive capacity from their explicit image-edit
GPU IDs. An external-endpoint request replaces only managed model serving; it
does not remove the prebuilt component-runtime requirement.

## Stage map

| Functional operation | Skill operation | Endpoint | Input | Validated output | Retry or recovery | State result |
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

- `managed` (default): provide explicit GPU allocations for all three roles.
  For SLURM, Kubernetes, and Brev, also provide `generation_nodes`; each node is
  an independent eight-GPU worker, while VLM/LLM use the coordinator allocation.
  For Docker and virtualenv, `generation_nodes` is 1 and the number of explicit
  image-edit GPU IDs is the generation capacity.
- `external`: use only after the user explicitly requests reuse and supplies
  three already-running local HTTP(S) base URLs. Materialize it with
  `--sdg-endpoint-mode external --reuse-external-endpoints`; GPU IDs are empty
  because the skill does not own those processes.

Never discover a listening port or compatible service and switch away from
managed mode. Do not ask about endpoint reuse unless the user brings it into
scope. Existing services are not implicit authorization to inspect or use
them.

Before approval, inspect only the selected platform runtime, GPU inventory or
available shapes/partitions, ports or service reachability, disk capacity,
image presence, and environment-variable presence.
Do not create the cache directory, pull an image, download a model, or start a
container. After approval materializes the proposed run configuration but
before `init_deft_state.py`, the following read-only plan is mandatory. This
ordering gives the deterministic helper the exact approved configuration while
keeping a failed proposal uninitialized and safe to replace:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/manage_sdg_endpoints.py" plan \
  --config "$RESULTS_DIR/config/sdg_config.yaml" \
  --run-id "$(basename "$RESULTS_DIR")" \
  --output "$RESULTS_DIR/endpoints/plan.json"
```

Omit `--output` for the pre-initialization gate; writing the plan belongs after
approval and is optional evidence. The helper checks Docker's NVIDIA runtime, compute
capability, aggregate free VRAM, and at least 150 GiB usable model-cache
capacity (free bytes plus files already present under the exact configured
model revisions),
deterministic container-name conflicts, and port collisions. It never emits a
credential value or widens an explicit device list.

For an operator-managed remote platform, execute its equivalent endpoint plan
in the platform compute frame. An Airflow-orchestrated Docker or virtualenv
backend executes this exact local plan on its compute worker; Airflow does not
allocate or reinterpret the GPUs. A port collision
must be resolved by proposing a new explicit port set before state
initialization; never edit an initialized run or treat a foreign listener as
implicit endpoint reuse.

On resume, the manager may record `disposition=reuse_no_acquisition` and skip
only the prospective 150-GiB model-acquisition capacity gate. This is legal
only when a read-only reconciliation proves every required service is running,
healthy, exactly run-owned, and matches the immutable image, model revision,
served model, GPU, and port allocation; every exact model-revision cache
receipt must also be present, and current `/v1/models` plus the role's minimal
readiness inference must pass. The image-edit inference gate remains the first
bounded augmentation smoke immediately after endpoint reconciliation. The
reuse path performs no pull, download, container create/restart, or cache
write. Missing, stale, unhealthy, mismatched, or incomplete evidence falls
back to the full acquisition-capacity check before any mutation. Normal
workspace/output free-space checks are unchanged.

## Launch approval rows

Add these rows to the consolidated preflight summary:

```text
Generation
  endpoint mode: <managed | external> (source=<user | default>)
  generation nodes: <up to N independent 8-GPU workers; distributed only>
  generation capacity: <8*N maximum distributed | explicit local image-edit GPU count>
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

After approval, Docker and virtualenv managed mode start or resume only
deterministic run-owned services. Set `DATAGEN` to the current iteration's
`$RESULTS_DIR/iter_${N}/datagen`; the iteration-local pool is immutable input
to dispatch and commit:

```bash
mkdir -p "$DATAGEN"
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/manage_sdg_endpoints.py" start \
  --config "$RESULTS_DIR/config/sdg_config.yaml" \
  --run-id "$(basename "$RESULTS_DIR")" \
  --platform "$PLATFORM" \
  --image-edit-pool "$DATAGEN/endpoint_pool.json" \
  --output "$DATAGEN/endpoint_manifest.json"
```

The manager starts one image-edit service per explicit image-edit GPU, with a
deterministic container name, base port plus ordinal, TP=1, and capacity 1.
VLM and LLM remain singleton services. SLURM, Kubernetes, and Brev must use
their platform composite action instead; do not run this lifecycle on the
control host for those platforms. With Airflow orchestration, wrap that same
platform composite consumer as documented in `airflow-execution.md`.
Docker/virtualenv use `airflow_sdg_action.py prepare-request` plus
`local_sdg_action.py`; Airflow itself does not run mapped GPU tasks. Never
hand-build DAG `conf`.

External mode substitutes `validate-external` and writes the same manifest
path. Model discovery plus minimal inference is bounded by
`startup_timeout_s`, `retry_interval_s`, and `request_timeout_s`. The first
augmentation is the image-edit inference smoke test and must pass verification
before the batch proceeds.

Image editing has a separate bounded
`generation.image_edit_request_timeout_s` (600 seconds by default) because a
validated 50-step diffusion request can legitimately take several minutes.
This does not widen readiness or text-model smoke-test deadlines.

A same-name service is reusable only when all workflow, run, role, and GPU
labels match. A stopped matching service may be restarted. Any other same-name
container or occupied managed port is a hard conflict. User-managed endpoints
are validated but never started, restarted, stopped, or replaced.

A successful managed manifest is reusable only while every matching owned
service remains running. If one later crashes during an uncommitted SDG stage,
`start` permits at most two owned-container restarts and repeats all readiness
probes, recording the bounded `restart_count` in the same manifest. A third
later crash exhausts the endpoint restart budget; it never widens a per-source
generation bound. Managed image-edit containers receive a fixed 16 GiB shared
memory allocation because the diffusion server is not safe with
Docker's 64 MiB default. A stopped, exactly run-owned image-edit container
created without that allocation is captured as recovery evidence, removed,
and recreated; running or non-owned containers are never replaced.

If an older installed helper left the exact run-owned image-edit container in
Docker's never-started `created` state with the known multi-GPU parser error,
use `repair-created` before a runtime rebind. It captures sanitized state/log
evidence and removes only that unusable container; it does not start a model.
All other owned states and every non-owned container are preserved.

On failure, keep run-owned containers and report the sanitized status plus the
recorded `docker logs --tail 200 <name>` command. Cleanup is explicit:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/manage_sdg_endpoints.py" stop \
  --config "$RESULTS_DIR/config/sdg_config.yaml" \
  --run-id "$(basename "$RESULTS_DIR")" \
  --roles image_edit,vlm,llm \
  --output "$DATAGEN/endpoint_manifest.json"
```

This stops matching owned containers without removing them or their cache and
persists sanitized intentional-shutdown evidence in the endpoint manifest. A
later iteration may restart that exact cleanly stopped owned set without
consuming the bounded unexpected-crash restart budget. A missing, foreign,
mixed, nonzero-exit, OOM-killed, or errored role is never classified as an
intentional shutdown. Runs created by an older helper are accepted only when a
successful prior readiness manifest and the same complete clean owned set are
both present.

One older helper could overwrite that successful manifest with an exact
restart-budget error after a planned clean stop. Use the following recovery
only for that signature; it preserves the error manifest, validates immutable
model/image/GPU identity, requires a committed successful SDG execution
receipt, records recovery lineage, and does not start a container:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/manage_sdg_endpoints.py" recover-overwritten-stop \
  --config "$RESULTS_DIR/config/sdg_config.yaml" \
  --run-id "$(basename "$RESULTS_DIR")" \
  --output "$RESULTS_DIR/endpoints/manifest.json" \
  --execution-receipt "$RESULTS_DIR/deft_state.json"
```

Do not use this action for another error, absent or uncommitted execution
evidence, changed configuration, missing/mixed ownership, or a nonzero, OOM, or
errored container exit. It never performs a generic retry-budget reset.

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
  --eval-list "$RESULTS_DIR/iaa_splits/eval_list.txt" \
  --eval-pairs "$RESULTS_DIR/iaa_splits/eval_pairs.json"
```

Then run the bounded sequence. It performs the first-item smoke gates, full
batch, split, labeling, and normalization. Each accepted item and completed
operation is journaled atomically; rerunning after interruption skips validated
work.

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/run_sdg_stage.py" execute \
  --config "$RESULTS_DIR/config/sdg_config.yaml" --output-root "$DATAGEN" \
  --execution-platform "$PLATFORM" \
  --image-edit-endpoint-pool "$DATAGEN/endpoint_pool.json" \
  --eval-list "$RESULTS_DIR/iaa_splits/eval_list.txt" \
  --attribute-vocab "$DATASET_ROOT/attribute_vocab.json"
```

Inspect the smoke image, `augmentation_smoke.json`, verification metadata, and
`auto_label_smoke_open_qa.json` before accepting the stage. A component exit 0
is insufficient without these artifacts and the final normalization checks.
For remote platforms, synchronization must copy and digest-verify the complete
normalized `dataset/` tree—not only its manifest, pairs, and image list. The
commit and audit require the exact listed image and caption files plus the
attribute vocabulary, preventing a metadata-only SDG success from reaching
training.

Commit the stage once:

If a successful legacy SLURM normalization is blocked only because its three
metadata files have a same-second, whole-second Lustre timestamp, use
`repair_sdg_normalize_freshness.py prepare`, dispatch the signed
`sdg_normalize_repair` adapter through the selected SLURM four verbs, then run
`verify` before commit. This recomputes metadata only and proves that the
result is byte-identical; it never repeats generation, verification, splitting,
or labeling. If the adapter fails, use `restore` to put back the exact journaled
files. The producer rejects this path without the exact prepared repair journal.

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/commit_stage.py" \
  --results-dir "$RESULTS_DIR" --iter-label "iter${N}" --stage sdg \
  --endpoint-pool "$DATAGEN/endpoint_pool.json" \
  --endpoint-manifest "$DATAGEN/endpoint_manifest.json" \
  --sdg-execution-manifest "$DATAGEN/sdg_execution_manifest.json" \
  --sdg-manifest "$DATAGEN/dataset/sdg_manifest.json" \
  --sdg-pairs "$DATAGEN/dataset/sdg_pairs.json" \
  --sdg-image-list "$DATAGEN/dataset/sdg_image_list.txt" \
  --sdg-status "$DATAGEN/status/sdg-normalize.${PLATFORM}.status.json" \
  --summary "verified platform-local generation normalized for iteration ${N}"
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
