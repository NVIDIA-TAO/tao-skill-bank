# AnomalyGen 1.2 — DEFT generation contract

Use the single `tao-generate-anomalies` skill. Resolve the runtime with
`python scripts/resolve_versions_key.py images.metropolis_sdg.anomalygen` and record
its immutable image digest. Its native repository is `/workspace/cdf-anomalygen`.
Read the data skill's `references/container-runtime.md` and `checkpoints.md`.

## Inputs and bootstrap

Default public adapter: `nvidia/Cosmos-AnomalyGen-PCB-Nano`. Download its actual
`model/iter_000015000.pt` and `model/exp_texture_ft_pcb.yaml` together into
`augmentation/anomalygen/checkpoints/<project>/`. Bind the adapter **file** and
its original recipe. The HF `model/` directory is not a training run directory.
For BYO use `--anomalygen-checkpoint <run-dir-or-iter_N.pt>` together with
`--anomalygen-recipe <exact-training-recipe>`. Native run directories resolve
through their authoritative best pointer and saved recipe. Never fabricate a
latest pointer or a flat-checkpoint view. Predict2 1.0.1 weights require retraining;
unstamped 1.1 texture adapters are native inference-only legacy inputs whose
actual compatibility must be verified. Public Metal-Nano and Glass-Nano are
other dataset examples, with their own original recipes and class sets.

Network-enabled bootstrap remains post-approval. Use `hf download
nvidia/Cosmos-AnomalyGen-PCB-Nano --include 'model/*' --local-dir <checkpoint-root>`.
Use native `scripts/skill_utility/download_required_weights.py --help` and the
selected-stage mapping for Nano DCP base, Wan VAE, Qwen tokenizer and both
guardrail repositories. AMP may need route-specific assets; quality evaluation
needs its backbones. Do not download Edge/Gemma just for global preflight.
The native `preflight_env_ckpt.sh` output and a CUDA 13.2 / driver compatibility
probe belong in the run receipt; optional-stage omissions remain explicit.
Air-gap mode requires these assets and the resolved image already staged.

Dataset layout is `TEXTURE/clean_image`, optional `TEXTURE/cad_mask`, training
`TEXTURE/anomaly_image/DEFECT` paired with `TEXTURE/mask/DEFECT/*_mask.png`, plus
`defect_spec.jsonl`. Read the native input contract for extension fallback and
`cad`, `cad_tht`, or `text` placement. Keep CAD semantic colors and text prompts.
Caller `--exclude-stems` is passed to preparation, generation and consumers.
Placement must operate on the filtered dataset view. With precomputed JSONL,
exclusion covers exact source/mask stems (including `_mask`); transformed donor
ancestry requires caller provenance or filtering before placement. Never infer
private dataset ancestry from a generated filename.

## Generation

The application runs `inference_only`; `prepare`, `full`, and `finetune_only`
remain explicit standalone/bootstrap modes. Use the shared structured action or:

```bash
python /bank/skills/data/tao-generate-anomalies/scripts/run_anomalygen.py inference_only \
  --name "iter${N}" --output-dir "$RUN_DIR" \
  --checkpoint "$CHECKPOINT" --recipe "$RECIPE" \
  --dataset-dir "$DATASET" --num-sdg "$NUM_SDG" \
  --exclude-stems "$EXCLUDE_STEMS"
```

Mount `/bank` read-only, data/model paths in their recorded compute namespace,
the selected checkpoint cache at `/workspace/cdf-anomalygen/checkpoints`, and
results writable. Use the selected platform's four verbs and durable job record.
For offline single-node Docker use `--network=none --hostname localhost` so
`torchrun --standalone` can resolve its rendezvous hostname.

The default quality policy runs native search for three rounds with no automatic
discard or top-up. Drop-ratio filtering is opt-in only when explicitly requested;
record its ratio and retained/dropped identities. Never replace blocked samples.
Bounded smoke explicitly adds `--skip-quality`; it is not production quality
acceptance. Keep training recipe `max_iter=5000` unchanged; the public adapter's
iter15000 is its provenance.

## Native accounting and ChangeNet emission

The completed `manifest.json` binds testcase identity/hash, selected output,
native timing and generated/blocked CSV hashes. The selected directory contains
`texture_ft_generation_result.csv`, optional `guardrail_blocked.csv`, and paired
`reconstructed_image/`, `original_image/`, `original_mask/`. CSV output names are
basenames. IDs may be sparse: generated and blocked identity sets must be
disjoint and cover every testcase exactly. Never top up a blocked identity.
A fully blocked run is COMPLETE with `training_ready=false`; it contributes zero
synthetic rows. Continue using eligible real data; never launch empty training.

```bash
python "$SKILL_ROOT/scripts/changenet_data_pair_prepare.py" \
  --manifest "$RUN_DIR/manifest.json" --exclude-stems "$EXCLUDE_STEMS" \
  --images-dir "$IMAGES_DIR" --subdir "sdg_iter${N}" --output "$SDG_CSV"
python "$SKILL_ROOT/scripts/validate_training_csv.py" \
  --csv "$TRAIN_CSV" --workspace-root "$IMAGES_DIR" \
  --validation-csv "$VALIDATION_CSV" --exclude-stems "$EXCLUDE_STEMS" \
  --state "$RESULTS_DIR/deft_state.json" --iter-label "iter${N}"
python "$SKILL_ROOT/scripts/commit_stage.py" \
  --results-dir "$RESULTS_DIR" --iter-label "iter${N}" --stage anomalygen \
  --anomalygen-manifest "$RUN_DIR/manifest.json" \
  --duration-sec "$STAGE_DURATION_SEC" --summary "native generation accounted"
```

ChangeNet keeps PASS uppercase and defect labels lowercase, with one paired
clean image per generated defect. Retain the `anomalygen_lineage` CSV column
through assembly. Save the immutable emission at
`iter${N}/mining_filter/sdg_rows_raw.csv` with its `.anomalygen.json` receipt.
Rewrite paths in a working copy, never the raw emission. The receipt binds both
converted image hashes, native lineage, object identity and label. Validation
with `--state` derives every contributing iteration's required raw CSV and
rejects stripped lineage, substituted pairs and missing synthetic rows. The
`data_merge` stage independently derives and verifies those same sources before
training. Standalone validation uses repeated `--synthetic-csv` arguments.
Keep receipts beside artifacts, including header-only all-blocked output.

Hard stop on missing/corrupt artifacts, stamp/recipe mismatch, source leakage,
disabled requested guardrails, or incomplete identity accounting. Keep native
`image_guardrail_enforcing` and `text_guardrail_enforcing` capabilities separate
from `guardrail_enabled`; the 1.2 image can report image enforcement false while
text screening and face blur are enabled. Do not report image screening passed
when the native capability does not enforce it.

## Direct Docker platform fallback

Map the submitting identity and verify writable results before submission:

```bash
probe="$RESULTS_DIR/.tao-write-probe.$$"
(umask 077 && : > "$probe" && rm -f "$probe") || exit 2
docker run --pull=never --rm --gpus all --ipc=host \
  --user $(id -u):$(id -g) -e USER="$(id -un)" -e LOGNAME="$(id -un)" -e HOME=/tmp \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -v "$TAO_SKILL_BANK_PATH:/bank:ro" -v "$WS:$WS" \
  -v "$COSMOS:/workspace/cdf-anomalygen/checkpoints:ro" \
  -w /workspace/cdf-anomalygen "$AG_IMAGE" \
  python /bank/skills/data/tao-generate-anomalies/scripts/run_anomalygen.py inference_only \
    --name "iter${N}" --output-dir "$RUN_DIR" --checkpoint "$CHECKPOINT" \
    --recipe "$RECIPE" --dataset-dir "$DATASET" --num-sdg "$NUM_SDG" \
    --exclude-stems "$EXCLUDE_STEMS"
```

Apply the recorded quality selection before launching, and add offline runtime
settings in air-gap mode. Default public assets use `hf download` as above.
