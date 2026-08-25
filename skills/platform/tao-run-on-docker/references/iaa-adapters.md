# Signed IAA CPU adapters on Docker

Read this reference only when consuming a platform-neutral
`tao-run-deft-iaa` action request whose signed `name` is one of the typed CPU
adapters.

Use `scripts/render_iaa_adapter.py` after opening the job record. The renderer
accepts only the fixed adapter allowlist, verifies the request digest and
active runtime digest, requires read-only `/iaa-runtime` and `/patches` binds,
and emits no
`--gpus` or `NVIDIA_VISIBLE_DEVICES` selector:

```bash
python3 "$BANK/skills/platform/tao-run-on-docker/scripts/render_iaa_adapter.py" \
  --request "$ACTION_REQUEST" --job-id "$JOB_ID" >"$JOB_DIR/docker-submit.json"
```

The JSON contains the exact native Docker argv and backend name. Lint it, then
use the packaged executor; never reconstruct the array with a shell, `eval`,
`xargs`, or command substitution:

```bash
python3 "$BANK/scripts/redact_secrets.py" lint "$JOB_DIR/docker-submit.json"
LAUNCH_JSON=$(python3 \
  "$BANK/skills/platform/tao-run-on-docker/scripts/execute_rendered_argv.py" \
  --submit "$JOB_DIR/docker-submit.json")
CID=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["backend_ref"])' \
  <<<"$LAUNCH_JSON")
```

The executor requires detached Docker argv, exact name binding, a full
container ID, and a matching post-launch inspection. Bind that ID to the
already-open job record. The renderer
adds exact job, action, request-digest, and runtime-digest labels so native
inspection remains bound to the signed request.

The complete `controller_snapshot` manifest must match its local minimal
skills root. `/iaa-runtime` must resolve to exactly
`<controller root>/applications/tao-run-deft-iaa/scripts`; the broader root is
validated but not mounted. `patches_snapshot` must separately match the exact
tree mounted at `/patches`. Every path, size, file digest, and aggregate digest
is checked, binding the entrypoint, application references, core artifact
schemas, `iaa_deft`, and compatibility patches. The `iaa_deft` tree must also
equal `request.runtime_sha256`. Workflow results retain their
producer-declared writable mounts and remain the native completion evidence.
A missing signature, unknown adapter, nonzero GPU shape, credential forwarding,
writable or changed controller/patches snapshot, duplicate mount target, or inconsistent image
fails before Docker is invoked.
