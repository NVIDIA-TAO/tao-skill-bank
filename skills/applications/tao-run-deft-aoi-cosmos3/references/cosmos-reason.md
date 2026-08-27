# Cosmos3 Train and Bare OK/NG Evaluate

## Prepare the Cosmos Reason 3 checkpoint

The model this loop fine-tunes is the published **Cosmos Reason 3** reasoner
(`nvidia/Cosmos3-Nano` by default). The published directory is native Omni
format — recognizable by `model_type="cosmos3_omni"` and
`Cosmos3ForConditionalGeneration` — while Framework Train and the unchanged
cosmos-rl evaluate job both load a Qwen3-VL VLM safetensors PTM.

So a one-time conversion turns the selected Cosmos Reason 3 reasoner into a
Qwen3-VL PTM. The reasoner stays the model's identity and lineage throughout;
the Qwen3-VL PTM is only the on-disk format Cosmos-RL can consume.

After the user approves the launch review, prepare Nano with the helper owned by
`tao-finetune-cosmos-reason`:

```bash
set -a; source /path/to/.env; set +a   # omit unless this env file was approved
PREPARED_MODEL_PARENT=$(dirname "$PREPARED_MODEL_HOST_PATH")
mkdir -p "$PREPARED_MODEL_PARENT"
probe="$PREPARED_MODEL_PARENT/.tao-write-probe.$$"
(umask 077 && : >"$probe" && rm -f "$probe") || {
  echo "FATAL: $PREPARED_MODEL_PARENT is not writable by uid $(id -u)" >&2
  exit 2
}

# Pre-Flight resolves this packaged Nano mapping to an immutable Hub revision.
COSMOS3_VLM_ARCHITECTURE_MODEL="${COSMOS3_VLM_ARCHITECTURE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
: "${COSMOS3_VLM_ARCHITECTURE_REVISION:?set the immutable revision resolved by the model planner}"
COSMOS3_CONVERSION_CACHE_DIR="${COSMOS3_CONVERSION_CACHE_DIR:-$PREPARED_MODEL_PARENT/cache}"
COSMOS_FRAMEWORK_IMAGE=$(
  "$PYTHON" "$TAO_SKILL_BANK_PATH/scripts/resolve_versions_key.py" \
    --skill-bank "$TAO_SKILL_BANK_PATH" images.tao_toolkit.cosmos_framework
)
COSMOS_FRAMEWORK_IMAGE_DIGEST=$(docker image inspect --format '{{.Id}}' "$COSMOS_FRAMEWORK_IMAGE")
: "${COSMOS_FRAMEWORK_IMAGE_DIGEST:?could not resolve the selected runtime image ID}"

"$PYTHON" \
  "$TAO_SKILL_BANK_PATH/skills/models/tao-finetune-cosmos-reason/scripts/prepare_cosmos3_vlm_checkpoint.py" \
  --base-model-path-or-uri nvidia/Cosmos3-Nano \
  --output-path "$PREPARED_MODEL_HOST_PATH" \
  --cache-dir "$COSMOS3_CONVERSION_CACHE"```

Nano conversion is fully resolved by the model skill. Do not ask the user to
choose a Qwen donor, a Cosmos Framework repository/commit, or an image. The
helper reads the immutable Nano source revision, Qwen3-VL donor revision,
Framework converter commit, and conversion-image digest from
`references/cosmos3-conversion-defaults.json`. It downloads model snapshots,
checks out the converter, and installs the locked converter environment only
when its cache does not already contain them.

The helper runs the conversion container as the invoking uid/gid, so no repair
container or `chown` step is needed. It forwards whichever of `HF_TOKEN` or
`HUGGING_FACE_HUB_TOKEN` is already set, by name and without printing values.
Use `--secrets-env` only for an env file the user approved.

Do not convert again when the helper reports `status="reused_verified"`; it has
already validated the complete `qwen3_vl` directory and matching provenance
before any clone, pull, or source-checkpoint access. Mount or stage that
directory for the selected platform, then use its compute-frame path for:
- Framework Train `[model.backbone].model_name` and `safetensors_path`;
- baseline and iteration evaluate `model.model_name`.

Keep the canonical Cosmos Reason 3 ID (`nvidia/Cosmos3-Nano` by default) as the
source-model lineage; it names the model, not the Cosmos-RL PTM path. The
automatic mapping is Nano-only. For Edge or Super, require a model-skill-owned,
variant-specific conversion policy; hard-stop instead of applying Nano's
Qwen3-VL donor.
## Run containers as the invoking user

Every container that writes into the bound results directory — Train, both
evaluate stages, AnomalyGen — must drop to the host user. Containers default to
root, so their outputs land root-owned inside a directory the operator owns,
and the run tree then cannot be deleted, re-rendered into, or cleaned up
without `sudo`. On Docker:

```bash
CR3_IDENTITY_ARGS=(
  --user "$(id -u):$(id -g)"
  -e USER="$(id -un)" -e LOGNAME="$(id -un)" -e HOME=/tmp
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro
)
```

Insert `"${CR3_IDENTITY_ARGS[@]}"` into every Docker Train, Proxy evaluate,
and Benchmark evaluate launch, including resumed actions, and use the
stage-local writable working directory:

```bash
-w "$RESULTS_DIR/<label>/<stage>/cwd"
```

All four parts are one unit; dropping any of them fails.

`HOME=/tmp` matters: a mapped uid has no home inside the image, and libraries
that write caches to `$HOME` fail or scatter files otherwise.

`-w` matters for the same reason. The image's `WORKDIR` is `/workspace`, owned
by root, and TAO's status decorator falls back to creating `./results` in the
current directory when `TAO_API_JOB_ID` is unset. As root that silently
succeeded; as a mapped uid the first job dies with
`PermissionError: [Errno 13] Permission denied: './results'`. Point `-w` at a
writable directory you create inside the stage's bound results tree.

That leaves a `cwd/results/status.json` stub per stage. It is inert, but treat
`cwd/` as run scaffolding rather than an artifact — never record it in state.
The cleaner alternative, read from the container's `decorators.py`, is to set
`TAO_API_JOB_ID` together with `TAO_API_RESULTS_DIR` pointing into the bound
results tree, which takes the branch that skips `./results` entirely; that path
has not been run end-to-end here, so verify it before relying on it. Other platforms
express the same intent differently — a Kubernetes `securityContext`
`runAsUser`/`runAsGroup`, or SLURM's already-unprivileged execution — so apply
the selected platform's equivalent rather than copying these flags blindly.

## Train producer

The Train action is native Cosmos Framework:

- image key: `images.tao_toolkit.cosmos_framework`, resolved only through
  `scripts/resolve_versions_key.py`;
- command: `python -m cosmos_framework.scripts.train
  --sft-toml=<config>`;
- mode/format: `config` / TOML;
- checkpoint: Framework DCP under the job-record-bound stage result tree;
- export: the image-owned `cosmos_framework.scripts.export_vlm_dcp` invoked
  and verified by `scripts/export_cfw_checkpoint.py`.

Use `nvidia/Cosmos3-Nano` by default. Accept `nvidia/Cosmos3-Edge` or
`nvidia/Cosmos3-Super` only when the user explicitly selects that variant.
Keep the canonical selected ID as lineage across baseline evaluation, Train,
and post-Train evaluation, and use its matching prepared PTM for each job. Treat
checkpoint loading/conversion and compute shape as variant-specific preflight
evidence. Give hardware recommendations consistent with the prompt and
selected variant; report insufficient resources instead of silently
substituting a different variant.

The pinned image's `TaoVlReasonDaftDataset` is video-only and its
`VideoConversationDataset` requires one video field. Until that gap lands
upstream, use the checked-in `scripts/cfw_cr3_aoi_adapter.py`; the submission
helper mounts only that file read-only into the baked experiment package. The
adapter validates the unchanged JSON-array `emit_sdg_sharegpt` shape, exact
two-image order, existing paths, and a final bare `OK`/`NG` label.

Render one of the reviewed profiles after Proxy RCA and Mining selection:

```bash
"$PYTHON" "$SKILL_ROOT/scripts/render_cfw_sft.py" \
  --profile full \
  --model-path "$PREPARED_MODEL_COMPUTE_PATH" \
  --annotation-path "$TRAIN_ANNOTATION_COMPUTE_PATH" \
  --media-root "$MEDIA_COMPUTE_ROOT" \
  --run-name "${RUN_ID}_${LABEL}_train" \
  --output "$WORKSPACE/specs/train_spec.toml"
```

Both profiles use BF16, Framework FSDP2, one node, seed 42, full activation
checkpointing, synchronous DCP cadence, Qwen3-VL's linear-equivalent PatchEmbed
compatibility, and native VLM LoRA over
`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`. The smoke profile is
five steps/save-at-five; full is 500 steps/save-every-100. Change those values
only as an explicit run decision. Do not replace native LoRA with a dense
freeze subset.

For Docker, keep the host-identity block above byte-for-byte and keep the
platform's job-record/four-verb pattern unchanged. The only Train-specific
submit step is delegated to the checked-in command composer:

```bash
COSMOS_FRAMEWORK_IMAGE=$(
  "$PYTHON" "$TAO_SKILL_BANK_PATH/scripts/resolve_versions_key.py" \
    --skill-bank "$TAO_SKILL_BANK_PATH" images.tao_toolkit.cosmos_framework
)
COSMOS_FRAMEWORK_IMAGE_DIGEST=$(docker image inspect --format '{{index .RepoDigests 0}}' "$COSMOS_FRAMEWORK_IMAGE")
JOB_ID=$("$TAO_SKILL_BANK_PATH/scripts/tao_job_record.py" open \
  --platform docker --image "$COSMOS_FRAMEWORK_IMAGE_DIGEST" \
  --network-arch cosmos3 --action train --storage-tier A \
  --results-dir "$TRAIN_RESULTS_DIR")
"$PYTHON" "$SKILL_ROOT/scripts/submit_cfw_train.py" \
  --skill-bank "$TAO_SKILL_BANK_PATH" --job-id "$JOB_ID" \
  --config "$WORKSPACE/specs/train_spec.toml" \
  --model-host "$PREPARED_MODEL_HOST_PATH" \
  --annotation-host "$TRAIN_ANNOTATION_HOST_PATH" \
  --media-host "$MEDIA_HOST_ROOT" --results-dir "$TRAIN_RESULTS_DIR" \
  --execute
CID=$(docker inspect --format '{{.Id}}' "$JOB_ID")
"$TAO_SKILL_BANK_PATH/scripts/tao_job_record.py" mark "$JOB_ID" \
  --state RUNNING --backend-ref "$CID"
```

Use the platform's unchanged `status`, `logs`, and `cancel` verbs after submit;
never infer runtime state from `train_submission.json`. That manifest records
`train_backend=cosmos-framework`, the resolver key, and immutable image digest.
It also records the automatically derived read-only identity mounts: the media
root at its host path, plus same-path mounts for directories containing
absolute annotation images outside that root. Do not add those mounts manually.
Pass the same URI and digest to `init_deft_state.py --train-container` and
`--train-image-digest`, and repeat both in the Train stage summary.

After Train reaches `COMPLETE`, identify the concrete final DCP plus its saved
resolved `config.yaml`, then export through the same resolved image:

```bash
set -o pipefail
"$PYTHON" "$SKILL_ROOT/scripts/export_cfw_checkpoint.py" \
  --skill-bank "$TAO_SKILL_BANK_PATH" \
  --checkpoint-path "$FRAMEWORK_DCP" --config-file "$FRAMEWORK_CONFIG_YAML" \
  --base-model-path "$PREPARED_MODEL_HOST_PATH" \
  --output-dir "$TRAIN_RESULTS_DIR/eval_model" \
  --manifest "$TRAIN_RESULTS_DIR/export_action.json" --execute
```

Never pipe this command through `grep`. If output is logged through `tee`, keep
`set -o pipefail` so the exporter exit code remains authoritative. The wrapper
refuses an incomplete export and verifies the DCP metadata hash, native LoRA
merge evidence, the exact indexed shard count, and tokenizer/processor runtime
files. Commit the concrete `eval_model` directory as `--best-ckpt` and pass
`export_action.json` as `--export-verification`; `commit_stage.py` revalidates
both before writing state. The unchanged cosmos-rl vLLM evaluate spec sets
`model.enable_lora=false` and `model.model_name` to that directory. Do not
record a launcher directory, partial export, or unverified DCP.

## Proxy and Benchmark evaluate producers

Read the current model skill's `evaluate` action contract. Use it for both
evaluation stages:

- command: `cosmos-rl-evaluate --config {config_path}`;
- mode/format: `config` / TOML;
- inputs: `dataset.annotation_path`, `dataset.media_dir`, and
  `model.model_name`;
- output: `results_dir`.

Start from the model skill's current packaged Evaluate template and materialize
it once for Proxy and once for Benchmark. Apply these AOI overrides to both:

- `task.type=""`;
- `dataset.system_prompt`: return exactly `OK` or `NG`;
- `evaluation.answer_type="freeform"` and
  `evaluation.soft_accuracy.enabled=false`;
- `generation.max_tokens=4` and `generation.temperature=0`;
- `metrics.names=[]`;
- save individual results; disable evaluator confusion-matrix and metric
  summary output because `analyze_gaps.py` owns the discrete OK/NG metrics.

Both stages must use the same checkpoint and generation settings; only the
annotation, bound output directory, and save-folder label differ. Framework's
native exporter merges LoRA into a complete HF checkpoint, so keep
`model.enable_lora=false`, point `model.model_name` at the verified
`eval_model` directory, and do not set an adapter `base_model_path`.

### Classify and, when needed, lift the evaluation image cap

Some `cosmos-rl` images build vLLM with
`limit_mm_per_prompt={"video": 1, "image": 1}`. This skill's single-image AOI
records already fit that default cap, so this stage is normally a no-op
(`already_sufficient`); it stays in the pipeline as a guardrail for any record
shape that carries more than one image per prompt, where evaluation would
otherwise fail until the cap is raised. Other images, including
framework-evaluator builds, contain neither that literal nor a vLLM engine
construction and need no patch. Image tags are not a stable way to tell these
apart.

Run this once per run, before the first evaluate job:

```bash
"$PYTHON" "$SKILL_ROOT/scripts/patch_eval_image_cap.py" \
  --image "$COSMOS_RL_IMAGE" \
  --output-dir "$RESULTS_DIR/patches/cosmos_rl_eval" \
  --summary "$RESULTS_DIR/patches/cosmos_rl_eval/summary.json"
```

It reads `base.py` out of the selected image and reports one source-driven
classification, independent of whether the tag is semver, a custom build name,
or a future tag:

- `patch_required`: the recognized cap is below `--images` (default 1); the
  script rewrites only that literal and prints `MOUNT_ARG=<host>:<container>:ro`.
  Add it as a read-only mount to every `cosmos-rl-evaluate` job.
- `already_sufficient`: the recognized cap already meets `--images`; no file
  or mount.
- `cap_absent`: the source contains neither the cap nor vLLM engine evidence;
  no file or mount.

If the source still references `limit_mm_per_prompt` or vLLM without the
recognized literal, the script fails with `classification=unknown`; verify the
new evaluator shape instead of guessing. Nothing is vendored into the skill,
and Train jobs never need this mount.

### Make the GPU video decoder registration optional

`cosmos_rl/evaluation/base.py`'s `load_model()` unconditionally registers a
PyNvVideoCodec-backed GPU video decoder before loading the model, even though
this skill's bare OK/NG records are single-image and never carry video.
`PyNvVideoCodec` dlopens both the NVIDIA encode and decode shared libraries at
import time; on a GPU with no hardware video encoder (for example H200, which
has no NVENC engine), `libnvidia-encode.so.1` does not exist and the whole
evaluate job hard-fails with `RuntimeError: Failed to load NVENC library:
libnvidia-encode.so.1: cannot open shared object file`, regardless of the fact
that no video will ever be decoded. There is no config key or environment
override for this in the source — the only accepted `vision.video_decoder`
value is `"pynvvideocodec"`.

Run this once per run, alongside the image-cap patch, before the first
evaluate job:

```bash
"$PYTHON" "$SKILL_ROOT/scripts/patch_eval_video_decoder.py" \
  --image "$COSMOS_RL_IMAGE" \
  --output-dir "$RESULTS_DIR/patches/cosmos_rl_video_decoder" \
  --summary "$RESULTS_DIR/patches/cosmos_rl_video_decoder/summary.json"
```

- `patch_required`: the recognized unconditional-registration block is
  present; the script rewrites it to register the GPU decoder only when
  `TAO_SKIP_PYNV_VIDEO_DECODER` is unset, and to skip it (logging why) when
  the run sets `TAO_SKIP_PYNV_VIDEO_DECODER=1`. Add the printed
  `MOUNT_ARG=<host>:<container>:ro` as a read-only mount, and pass
  `-e TAO_SKIP_PYNV_VIDEO_DECODER=1` to every `cosmos-rl-evaluate` job this
  single-image skill runs — a video-carrying workload on hardware that does
  have NVENC/NVDEC would leave that variable unset and keep registering the
  decoder exactly as before.
- `already_sufficient`: the source is already conditioned on
  `TAO_SKIP_PYNV_VIDEO_DECODER`; no file or mount.
- `pattern_absent`: the source contains no GPU video decoder registration
  block; no file or mount.

If the source still references `register_pynv_video_reader` or
`video_decoder` without the recognized block, the script fails with
`classification=unknown`; verify the new evaluator shape instead of guessing.
Nothing is vendored into the skill, and Train jobs never need this mount —
Train's dataset never touches this evaluator module.

The evaluator writes `results.json` with per-sample `video_id`, `response`,
`question`, and `gt`. Run `analyze_gaps.py` afterward:

- `--evaluation-role proxy` writes per-sample RCCA artifacts and never gates;
- `--evaluation-role benchmark` writes aggregate metrics and the stop-gate
  `metric_result.json`.

The analysis layer normalizes the last standalone `OK`/`NG` token. Source
training labels remain exact.

## Output commits

Baseline begins with zero-shot frozen Benchmark against the base model and
submits no Train job; zero-shot Proxy follows only when that gate is unmet.
After Proxy RCA and Mining selection create `train_iter_N.json`, submit Train,
Benchmark evaluate, and — only when the loop continues — Proxy evaluate, as
separate platform jobs with separate job-records. Commit Train with
`--best-ckpt`, `--export-verification`, and `--training-spec`; commit each
evaluation with its `results.json`. The `benchmark_metrics` commit records the
KPI; a stopping iteration completes there, and a continuing one completes at
`proxy_rcca`.
