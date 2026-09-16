# DINOv3 SSL DEFT container lifecycle

Data Services owns the controller and includes TAO PyTorch/Core. No host TAO
installation, nested Docker, Dockerfile change, or startup package replacement
is required by the single-container design.

## Release readiness

The currently declared general-purpose DS image is not a DEFT release.
Do not launch this application against it. Publication is blocked until TAO
Infra provides a build carrying the reviewed DS, PyTorch, and Core changes,
the packaged suite and GPU smoke tests pass, and a dedicated
`images.tao_toolkit.deft_dinov3_data_services` digest is pinned in
`versions.yaml` and `skill_info.yaml`. Keep the application PR draft until then.
A separate PyTorch deployment would also need a tested
`images.tao_toolkit.deft_dinov3_pyt` pin.

Source-overlay QA demonstrates source compatibility only. It is not evidence
that the published image contains these modules. The ANN path additionally
requires an approved CUDA-compatible cuVS dependency; the exact-search profile
does not. Never silently change an approved search backend.

## Docker: preparation and execution

Read the Docker platform skill for its host preflight, secure registry login,
job-record opening, UID/GID mapping, cache variables and GPU selection.
Resolve the approved application image on the host with:

```bash
python scripts/resolve_tao_image.py --application tao-run-dinov3-ssl-deft --action run
```

Resolve symbolic keys here, then pass explicit digest-pinned image references
to DS. It does not read Skill Bank or `versions.yaml`.

The example below assumes the platform has prepared these reviewed bindings:
`DEFT_IMAGE` is the tested image digest, `DEFT_ROOT` contains config, data,
checkpoints and outputs, and `DEFT_GPU` is the allocated host GPU identifier.
All paths in the YAML use the container's `/deft` namespace.
`DEFT_IDENTITY_ARGS` and `DEFT_CACHE_ARGS` are the Docker skill's non-root
identity and writable cache arguments. Do not inline credentials.

```bash
DEFT_MOUNTS=(-v "$DEFT_ROOT:/deft")
DEFT_RUNTIME=(--gpus "device=$DEFT_GPU" --ipc=host -e CUDA_VISIBLE_DEVICES=0)
DEFT_COMMAND=(python -m nvidia_tao_ds.mining.dinov3.workflow)

docker run --rm "${DEFT_RUNTIME[@]}" "${DEFT_IDENTITY_ARGS[@]}" "${DEFT_CACHE_ARGS[@]}" "${DEFT_MOUNTS[@]}" "$DEFT_IMAGE" "${DEFT_COMMAND[@]}" preflight --gpu
docker run --rm "${DEFT_IDENTITY_ARGS[@]}" "${DEFT_CACHE_ARGS[@]}" "${DEFT_MOUNTS[@]}" "$DEFT_IMAGE" "${DEFT_COMMAND[@]}" init --recipe grit-score --output /deft/run.yaml
# Fill reviewed paths/resources and set execution.backend: local.
docker run --rm "${DEFT_IDENTITY_ARGS[@]}" "${DEFT_CACHE_ARGS[@]}" "${DEFT_MOUNTS[@]}" "$DEFT_IMAGE" "${DEFT_COMMAND[@]}" validate /deft/run.yaml
docker run --rm "${DEFT_IDENTITY_ARGS[@]}" "${DEFT_CACHE_ARGS[@]}" "${DEFT_MOUNTS[@]}" "$DEFT_IMAGE" "${DEFT_COMMAND[@]}" plan /deft/run.yaml
```

After approval, the Docker platform opens the job-record and binds
`output.run_dir` to its results directory. Its returned `JOB_ID` names the
detached controller container. Preserve the platform's record-then-launch order:

```bash
docker run -d --name "$JOB_ID" --label "tao-job=$JOB_ID" "${DEFT_RUNTIME[@]}" "${DEFT_IDENTITY_ARGS[@]}" "${DEFT_CACHE_ARGS[@]}" "${DEFT_MOUNTS[@]}" "$DEFT_IMAGE" "${DEFT_COMMAND[@]}" run /deft/run.yaml
docker exec "$JOB_ID" "${DEFT_COMMAND[@]}" status /deft/results
# Obtain CLIENT_JOB_ID from status, then:
docker exec "$JOB_ID" "${DEFT_COMMAND[@]}" logs /deft/results "$CLIENT_JOB_ID"
```

The status/logs commands are explicit `docker exec` operations against the
running controller, not separate submissions to the platform. Use the platform's
Docker inspect/log verbs as well: a persisted workflow state is not proof that
the container is alive. Cancel through `workflow cancel /deft/results` while
the container is alive; SIGTERM to its main process also records cancellation.

After exit, `docker exec` is unavailable. Inspect the retained container with
the platform verbs and inspect durable state using the same image and a
read-only mount (with separately writable temporary caches):

```bash
docker run --rm "${DEFT_IDENTITY_ARGS[@]}" -e USER=tao -e LOGNAME=tao -e HOME=/tmp -v "$DEFT_ROOT:/deft:ro" "$DEFT_IMAGE" "${DEFT_COMMAND[@]}" status /deft/results
docker run --rm "${DEFT_IDENTITY_ARGS[@]}" -e USER=tao -e LOGNAME=tao -e HOME=/tmp -v "$DEFT_ROOT:/deft:ro" "$DEFT_IMAGE" "${DEFT_COMMAND[@]}" logs /deft/results "$CLIENT_JOB_ID"
```

For a failed (not canceled) workflow, submit a new container/job-record with the
same image, mounts and resources and `resume /deft/run.yaml`. Do not restart the
old container concurrently. The application data stays in the durable mount;
removing an inspected terminal container does not remove those results.

## Allocation, storage and recovery

Use `execution.backend: local`, omit `container_images`, and pin
`CUDA_VISIBLE_DEVICES` to the assigned allocation. GPU-consuming actions must
request exactly that visible GPU count; the controller does not discover or
claim unallocated devices. The recipe's one-GPU `torch_exact` scoring avoids
a GPU FAISS dependency. Explicit `faiss_exact` must pass a GPU FAISS probe.

The run directory must support reliable POSIX `flock`, atomic rename and
`fsync` (including directories). Do not use CIFS with `nobrl`, or NFS mounts
without working cross-client locks. Inputs are immutable and mounted read-only
where practical; outputs and scratch are separate writable locations.

Cancellation is terminal; `resume` adopts failed/interrupted work, not a canceled
run. When a dead local session leader has live descendants, the runner reports
UNKNOWN and does not signal unverified process identities. Inspect the recorded
PID/start time, descendants and allocation on the host, stop only verified
survivors, then confirm termination before retrying. Never remove a lock file to
force recovery. Config or native action-code changes require a new run and the
explicit `adopt-training` handoff where appropriate.

## External execution boundary

The DS runtime retains its four-verb external adapter interface, but neither DS
nor Skill Bank ships a Slurm/Kubernetes/separate-image runner executable for
this application. That deployment is unsupported until a named, tested runner
is supplied and reviewed. Do not present a placeholder executable as a runnable
recipe. The interface and attestation requirements remain in
[adapter-contracts.md](adapter-contracts.md) for integrators.
