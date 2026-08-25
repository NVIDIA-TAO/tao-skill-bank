# IAA SDG composite action on Kubernetes

Read this reference only for the `tao-run-deft-iaa` SDG stage. This action is
not TAO multi-node training: there is no rank rendezvous, `torchrun`, or shared
training process group.

## Native topology

- `generation_nodes=N` renders one Indexed Job with `N` independently
  scheduled pods. Every pod contains eight image-edit containers; each
  container requests exactly one `nvidia.com/gpu`, serves TP=1, and owns one
  capacity-one endpoint at `base_port + ordinal`.
- A headless Service gives every worker a stable
  `<job>-<index>.<service>.<namespace>.svc` identity.
- A separate coordinator Job uses two restartable native sidecars, one GPU for
  VLM and one GPU for LLM. Its main controller container runs the shared
  `run_sdg_stage.py` workflow and creates bounded component Jobs from the
  approved augmentation and auto-labeling images.
- Kubernetes v1.29 or newer is required for Indexed Jobs and restartable native
  sidecars.

The coordinator does not publish a partial pool. It discovers exactly `N`
owned worker pods, probes `/v1/models` and minimal inference on every one of
the `N*8` image-edit endpoints and both auxiliary endpoints, then atomically
writes:

```text
iter_N/datagen/endpoint_pool.json
iter_N/datagen/endpoint_manifest.json
```

The pool uses the shared capacity-one schema and is consumed by the shared SDG
dispatcher. The execution manifest binds its path, SHA-256, signed request
digest, and required capacity.

## Read-only gates

Before approval and submit, verify:

- Kubernetes server v1.29+ and the selected namespace are reachable.
- The named ServiceAccount exists and can create/get/delete Jobs and read Pods.
- The PVC is `Bound`; multi-worker runs require `ReadWriteMany`.
- All request paths resolve beneath the PVC mount. Agent-local paths are not
  sufficient evidence.
- At least `N` schedulable nodes expose eight GPUs each, and total allocatable
  capacity is at least `N*8 + 2` GPUs.
- All five images are digest pinned and the immutable SDG model revisions and
  component images match the request.

Do not hand-assemble a Kubernetes request. `prepare-request` reads the canonical
initialized `results_dir/deft_state.json` and its hash-bound immutable
`config/sdg_config.yaml`. It requires `platform=kubernetes`, the requested
iteration to equal `current_iteration`, and `history_select` to be the last
committed stage. It derives the exact generation-node count, model identities
and revisions, component provenance, limits, run paths, expected outputs,
credential-variable allowlist, and runtime/state/config digests. Namespace,
PVC identity and mount, ServiceAccount, digest-pinned deployable images, and
the canonical staged runtime path remain explicit platform inputs.

The application controller runtime must be staged at
`iter_N/datagen/.tao-runtime/controller` on the PVC before submit. It contains
`run_sdg_stage.py`, `iaa_deft/`, and this adapter as
`kubernetes_sdg_action.py`; its approved runtime digest is inherited from the
initialized state. Preparation performs no Kubernetes API call and reads no
credential. Repeating it with unchanged inputs reuses the byte-identical
request. It refuses a changed state/config, mismatched component image,
different existing output, stopped run, wrong platform, or uncommitted/already
committed SDG iteration.

## Credentials

Submit generates one run-scoped endpoint key in memory and creates an exact-
owned Kubernetes Secret using `kubectl apply -f -` with JSON on stdin. The same
secret is projected under server and client environment names. Secret values
must never appear in argv, request JSON, rendered workload objects, files,
reports, status, or logs. `HF_TOKEN`, when approved, is copied from the process
environment into the same stdin-only Secret.

Never create the Secret with `--from-literal`, print the Secret object, or
persist the stdin payload. Delete it after successful endpoint/component work;
retain failed resources for diagnosis until explicit cancel.

## Public verbs

```bash
python3 scripts/kubernetes_sdg_action.py prepare-request \
  --deft-state "$RESULTS_DIR/deft_state.json" \
  --sdg-config "$RESULTS_DIR/config/sdg_config.yaml" --iteration "$ITER" \
  --namespace "$NAMESPACE" --pvc-claim "$PVC_CLAIM" \
  --pvc-mount "$PVC_MOUNT" --service-account "$SERVICE_ACCOUNT" \
  --runtime-root "$RESULTS_DIR/iter_${ITER}/datagen/.tao-runtime/controller" \
  --augmentation-image "$AUGMENTATION_IMAGE" \
  --auto-labeling-image "$AUTO_LABELING_IMAGE" \
  --image-edit-image "$IMAGE_EDIT_IMAGE" \
  --text-serving-image "$TEXT_SERVING_IMAGE" \
  --controller-image "$CONTROLLER_IMAGE" \
  --output "$RESULTS_DIR/iter_${ITER}/datagen/kubernetes_request.json"

python3 scripts/kubernetes_sdg_action.py submit \
  --request "$RESULTS_DIR/iter_${ITER}/datagen/kubernetes_request.json"

python3 scripts/kubernetes_sdg_action.py status \
  --request "$RESULTS_DIR/iter_${ITER}/datagen/kubernetes_request.json"

python3 scripts/kubernetes_sdg_action.py logs \
  --request "$RESULTS_DIR/iter_${ITER}/datagen/kubernetes_request.json" --tail 200

python3 scripts/kubernetes_sdg_action.py cancel \
  --request "$RESULTS_DIR/iter_${ITER}/datagen/kubernetes_request.json" --confirm
```

Submit is resumable only when every discovered object has the exact workflow,
action-kind, run, role, and signed-request labels. Status reports `COMPLETE`
only when the coordinator succeeds and all six canonical SDG artifacts are
present. Cancel refuses mixed ownership, deletes exact-owned Jobs, Services,
and Secret with foreground propagation, and reports only Kubernetes native
UIDs—not environment or Secret data.
