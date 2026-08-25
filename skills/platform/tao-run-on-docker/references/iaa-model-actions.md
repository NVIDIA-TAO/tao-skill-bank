# Signed IAA GPU model actions on Docker

Read this reference when consuming a signed `tao-run-deft-iaa` action request
for embedding, k-NN, visualization embedding, training, or evaluation.

After opening and binding the request-owned job record, render the native argv:

```bash
python3 "$BANK/skills/platform/tao-run-on-docker/scripts/render_iaa_model_action.py" \
  --request "$ACTION_REQUEST" --job-id "$JOB_ID" \
  >"$STAGE_DIR/docker-submit.json"
```

The renderer verifies the request digest, action allowlist, exact image and
command family, explicit GPU IDs, full controller and patch snapshots, mounts,
and credential-forwarding policy. It preserves the exact device IDs, adds the
required non-root Docker identity and cache environment, and emits an argv
array. Lint the JSON and execute it only with
`scripts/execute_rendered_argv.py` as shown in `iaa-adapters.md`; do not
reconstruct it with shell interpolation. Bind the returned container ID, then
use Docker `status`, `logs`, and `cancel` normally.

Never replace the emitted device selector with a GPU count or `--gpus all`.
