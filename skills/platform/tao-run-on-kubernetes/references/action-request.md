# Producer Action Requests

Read this reference before consuming a platform-neutral action request that
contains a `mounts` array. The simple one-root spec-bundle template cannot
preserve duplicate-source aliases or per-target access modes.

The renderer accepts action-request schema version `"1"` only and fails closed
on a missing or unsupported `request.schema_version`; update the renderer and
this contract together before consuming a future version.

## Stage every declared source

Use `tao-data-io` when the producer source is not already visible from the
Kubernetes compute frame. Mirror each distinct `request.mounts[].source` into
a job-owned relative path on one bound PVC. Use delete semantics when the
producer requires remote freshness. Record exactly one row per distinct source:

```json
{
  "schema_version": "1",
  "sources": [
    {"source": "/launcher/run/results", "sub_path": "jobs/action-123/results"},
    {"source": "/launcher/run/config", "sub_path": "jobs/action-123/results/config"},
    {"source": "/launcher/cache", "sub_path": "jobs/action-123/cache"}
  ]
}
```

The staged subpaths must already exist. Verify read access and required write
access from a pod before opening the job-record. For an application freshness
contract, also verify every declared absent path and persist its staging
receipt before opening the record. Never add an undeclared source or use an
incremental copy that can retain stale outputs.

### Signed IAA CPU adapters

An IAA typed adapter is the only action request allowed to render with
`compute_shape.gpus=0`. The renderer verifies the signed request digest, fixed
adapter allowlist and argv, `gpu_ids=[]`, empty credential forwarding, and
the exact non-secret Kubernetes adapter environment. Stage the complete
controller root and patches snapshot. Mount only the exact derived
`<controller root>/applications/tao-run-deft-iaa/scripts` PVC subpath at
`/iaa-runtime`, and mount patches at `/patches`; both mounts are read-only.
Each staging row must bind the corresponding producer snapshot digest:

```json
{"source":"/run/.tao-runtime/input-snapshots/skills","sub_path":"jobs/action-123/controller","sha256":"<request.controller_snapshot.sha256>"}
{"source":"/run/.tao-runtime/input-snapshots/patches","sub_path":"jobs/action-123/patches","sha256":"<request.patches_snapshot.sha256>"}
```

The renderer verifies every local manifest path, size, file digest, and
aggregate digest, including `run_iaa_compute.py`, and requires the staging
receipt to attest both aggregate digests. A missing/different file or digest,
writable controller/patches mount, unsigned or
unknown zero-GPU action, or adapter requesting a GPU fails closed. A valid CPU
adapter renders `resources: {}` and therefore does not request the NVIDIA
device plugin. Its Job annotation records the job-record ID and runtime digest;
ordinary status/log/cancel and output evidence remain unchanged.

## Record, credentials, and backend name

Open the job-record only after the staging/freshness gate. Job-record ids allow
characters and lengths that Kubernetes object names do not. Derive, never
hand-edit, the collision-resistant backend name; the manifest retains the
original record id in an annotation:

```bash
K8S_JOB_NAME=$(python3 \
  "$BANK/skills/platform/tao-run-on-kubernetes/scripts/render_action_job.py" \
  name --job-id "$JOB_ID")
```

When `request.forward_env` is non-empty, create `CRED_SECRET` after the record
is open and feed exactly those approved variables through an env-file on
stdin. For example, for a request that forwards these three names:

```bash
CRED_SECRET="${K8S_JOB_NAME}-creds"
set -a; source /path/to/.env; set +a   # omit if already exported
printf 'AWS_ACCESS_KEY_ID=%s\nAWS_SECRET_ACCESS_KEY=%s\nHF_TOKEN=%s\n' \
  "$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY" "$HF_TOKEN" \
  | kubectl create secret generic "$CRED_SECRET" -n "$NAMESPACE" \
      --from-env-file=/dev/stdin
```

The manifest projects each approved `forward_env` name from the same-named key
with an individual `env.valueFrom.secretKeyRef`; unrelated keys in the Secret
are not imported. Omit both the credential Secret and `--credential-secret`
when `forward_env` is empty (the renderer rejects that mismatch). Likewise,
pass an image-pull Secret only when the namespace needs one; do not create a
dependency on a placeholder Secret.

## Render and gate

The renderer validates the immutable argv/image/GPU shape, exact source map,
unique targets, safe relative PVC subpaths, writable coverage for fresh
outputs, duplicate-source aliases, read-only flags, and conditional Secret
references. It emits native `command`/`args` arrays, not a shell string:

```bash
RENDER_ARGS=()
if [ -n "${CRED_SECRET:-}" ]; then
  RENDER_ARGS+=(--credential-secret "$CRED_SECRET")
fi
if [ -n "${IMAGE_PULL_SECRET:-}" ]; then
  RENDER_ARGS+=(--image-pull-secret "$IMAGE_PULL_SECRET")
fi

python3 "$BANK/skills/platform/tao-run-on-kubernetes/scripts/render_action_job.py" \
  render --request "$ACTION_REQUEST" --staging-map "$STAGING_MAP" \
  --job-id "$JOB_ID" --namespace "$NAMESPACE" --pvc-claim "$PVC_CLAIM" \
  "${RENDER_ARGS[@]}" >"$MANIFEST"

"$BANK/scripts/redact_secrets.py" lint "$MANIFEST"
kubectl apply --dry-run=server -f "$MANIFEST"
```

The renderer uses `templates/k8s/action-job.yaml.tmpl`. Do not use the legacy
`single-pod-job.yaml.tmpl` for an action request with multiple mounts.

Apply and bind the real backend name to the record:

```bash
K8S_OBJECT=$(kubectl apply -f "$MANIFEST" -o name)
[ "$K8S_OBJECT" = "job.batch/$K8S_JOB_NAME" ] || {
  echo "unexpected Kubernetes object: $K8S_OBJECT" >&2
  exit 1
}
"$BANK/scripts/tao_job_record.py" mark "$JOB_ID" --state RUNNING \
  --backend-ref "$NAMESPACE/$K8S_JOB_NAME"
```

On reattach, recover namespace and backend name from that `backend_ref`; never
assume the Kubernetes object name equals the record id. Status, logs, and
cancel use the backend name. Delete `CRED_SECRET` after terminal log/output
collection, when it was created.
