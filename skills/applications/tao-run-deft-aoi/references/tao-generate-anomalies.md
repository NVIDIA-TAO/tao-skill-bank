# Cosmos AnomalyGen — DEFT Loop Reference

Read this when the parent runs the `anomalygen` stage. The underlying skill
`tao-skill-bank:tao-generate-anomalies` (`data/tao-generate-anomalies/SKILL.md`) owns
the standalone 8-phase pipeline and parameter reference. This file is the
DEFT-loop overlay: what to pass, how mounts resolve, the few invariants
that gate the run, and the failure mode the loop has actually hit.

**Per-iteration**, the DEFT loop runs the underlying skill in
`mode=inference_only` and only needs Phases 2 (`prep_testcase.sh`) and 3
(`run_sdg.sh`). Phases 4–7 (eval / search / filter+regen) are SDG-quality
optimization and do not contribute to the loop's training pairs. Skip them
by setting `num_search_run=0` and `nn_threshold=0`, or invoke the two
wrappers directly (see *Per-iteration invocation* below).

**Once, before the loop starts**, the post-gate bootstrap (SKILL.md →
*Workflow* step 2) populates missing Cosmos base checkpoints and the PCB
reference dataset. The AnomalyGen fine-tuned checkpoint is never downloaded:
a PAIDF-compatible package must already be staged.
Pre-Flight only **probes** status and reports it — no side-effecting work
happens before the user gate.

This bootstrap applies only when network access is allowed. For air-gap mode,
apply `references/air-gap.md`: skip download helpers and validate every staged
input before running the per-iteration commands.

## Workspace Inputs

Three independent inputs under `<workspace>/augmentation/anomalygen/` plus
the Cosmos base-checkpoints cache.

| Input | Path (e.g. `<project>=nvpcb`) | Holds | Source policy |
|---|---|---|---|
| `checkpoint_dir` | `augmentation/anomalygen/checkpoints/<project>/` | A nested package such as `nvidia/Cosmos-AnomalyGen-PCB-2B/{ag_config.yaml,iter_000015000.pt}`; the flat model file is normalized below. | **BYO/pre-staged REQUIRED:** finetune the AnomalyGen LoRA with the container `texture_ft` recipe using the `uc1_pcb` layout, then stage `ag_config.yaml` and `iter_<step>.pt` as documented below. There is no checkpoint download fallback. |
| `dataset_dir` | `augmentation/anomalygen/datasets/<project>/` | PCB reference data + `semantic_segmentation_labels.json` + `defect_spec.jsonl`. Sibling to `checkpoints/`. | **BYO:** pre-stage. **Auto:** `python3 -m scripts.utilities.prepare_dataset_uc1 <dir>` (HF `nvidia/Cosmos-AnomalyGen-PCB-Dataset`). `uc1` is the `tao-generate-anomalies` skill's identifier for the PCB use-case — the script name is unrelated to `<project>`. |
| `defect_spec` | `${dataset_dir}/defect_spec.jsonl` | One entry per defect_type (`<T>+<A>`); `spatial_dependency ∈ {free, text, cad}` | Bundled with `dataset_dir` (either path). Template at `data/tao-generate-anomalies/assets/defect_spec_template.jsonl`. |
| `cosmos_models_dir` | `${COSMOS_MODELS_DIR}` (resolved by Pre-Flight) | The container-required 2B workflow assets | **BYO:** pre-stage. **Auto:** the container's `check.sh` checks its own contract before the explicit Text2Image 2B download. Post-gate bootstrap runs this once with a `:rw` mount; persists across runs. |

DEFT AOI is always a PCB workflow; the `<project>` placeholder is just the
directory label the user picked for this AnomalyGen project's checkpoint +
dataset (commonly `UC1` or `nvpcb`). The loop reads it from
`deft_state.json::config.anomalygen.project`.

`dataset_dir` and `clean_dir` resolve to the same path on this workspace —
clean images live under `<dataset_dir>/<T>/clean_image/` which is the
container's first probe hit. The container handles both flat and
split-by-texture layouts transparently via `validate_amp_inputs.py`; the
loop passes the workspace dir verbatim, no pre-staging.

## PAIDF checkpoint compatibility and staging

The fine-tuned checkpoint must match the **major.minor** version of the
container resolved from `images.metropolis_sdg.paidf_anomalygen`. PAIDF 1.0
and 1.1 checkpoints are not interchangeable: the GB-scale, full-finetune
`iter_000014000.pt` published in the 1.0-era Hugging Face repo
`nvidia/Cosmos-AnomalyGen-PCB-2B` is incompatible with a PAIDF 1.1 container.
PAIDF 1.1 uses the DEFT team's per-defect LoRA recipe on a Cosmos3 backbone.

Finetune the AnomalyGen LoRA with the container's `texture_ft` recipe using the
`uc1_pcb` layout (`dataset_path/<texture>/anomaly_image/` paired with
`dataset_path/<texture>/mask/<defect>/`), then stage the resulting
`ag_config.yaml` and `iter_<step>.pt` before Pre-Flight. The uc1 reference
dataset remains auto-downloadable, and the known PCB package uses this exact
layout:

```text
augmentation/anomalygen/checkpoints/nvpcb/
└── nvidia/Cosmos-AnomalyGen-PCB-2B/
    ├── ag_config.yaml
    └── iter_000015000.pt
```

Discover the package and derive its step from the single model file:

```bash
CKPT=$(find -L "$WS/augmentation/anomalygen/checkpoints/<project>" \
  -path '*/ag_config.yaml' -print -quit | xargs -r dirname)
: "${CKPT:?HARD STOP: PAIDF-compatible AnomalyGen checkpoint missing; finetune the AnomalyGen LoRA with the container texture_ft recipe using the uc1_pcb dataset layout to produce ag_config.yaml + iter_<step>.pt, then stage them under augmentation/anomalygen/checkpoints/<project>/...}"
mapfile -t staged_models < <(
  find -L "$CKPT" -maxdepth 1 -type f -name 'iter_[0-9]*.pt' \
    ! -name '*_reg_model.pt' | sort
)
[ "${#staged_models[@]}" -eq 1 ] || {
  echo "HARD STOP: expected exactly one PAIDF-compatible iter_<step>.pt in $CKPT" >&2
  exit 2
}
model_name=$(basename "${staged_models[0]}")
STEP=$(sed -n 's/^iter_0*\([0-9][0-9]*\)\.pt$/\1/p' <<<"$model_name")
: "${STEP:?HARD STOP: cannot parse AnomalyGen step from $model_name}"
```

For the staged example, discovery resolves the nested directory and produces
`STEP=15000`. A missing or ambiguous package is a hard stop in every network
mode; never substitute an HF checkpoint.

## Invariants

Verify these before invoking; the rest is up to the container.

1. **`cad_mask` preserves per-class RGB.** `cad2roi` looks up each pixel's
   RGB tuple in `semantic_segmentation_labels.json`. A flattened binary
   `(0,0,0)`/`(255,255,255)` cad_mask yields zero ROIs everywhere (see
   *AMP no-ROI failure mode* below). Verify with
   `Image.open(cad_mask).convert('RGB').getcolors(maxcolors=64)` —
   unique tuples must overlap the labels file.
2. **`defect_spec.jsonl` `text` entries have non-empty
   `roi_prompt_defect_location`.** `cad` and `free` entries don't need it.
3. **`<T>/cad_mask/` and `<T>/clean_image/` are non-empty and paired by
   stem.** Missing pair → record dropped silently.
4. **`semantic_segmentation_labels.json` exists at `datasets/<project>/`.**

Mask file format, image-size agreement, and channel mode do **not** gate
`mode=inference_only` — AMP processes each record at its native size. See
the underlying skill's `references/inference.md` if you need the full
list for `mode=full` / `mode=finetune_only`.

## AMP "no ROI candidates" failure mode

`run_auto_roi_amp.py` silently skips a sample when the cad_mask doesn't
have enough free area for the requested anomaly mask shape. The wrapper
does **not** propagate this — `num_SDG=N` quietly degrades to whatever
AMP could allocate, and the loop only notices via a smaller
`SDG_result.csv`.

Symptoms:

```
WARNING ... <stem>/<T>+<A>: no ROI candidates, skipping
INFO ... <T>+<A>: 0/N with ROI, 0/0 seeds OK
wrote 4 entries to testcase.jsonl       # <-- expected 20
```

Diagnose in this order:

1. **cad_mask class mapping** — invariant #1 above. Most common cause.
2. **Anomaly mask shape vs cad free area** — if the anomaly mask's
   bounding box exceeds every connected component in the cad_mask, AMP
   can't place it. Provide smaller anomaly masks or switch
   `spatial_dependency: free` to skip ROI placement entirely.
3. **Isolate the failing defect** — filter `defect_spec.jsonl` to just
   `<T>+<A>` and re-run `prep_testcase.sh --num-sdg 1`.

After Phase 2, parse `<output_dir>/allocation.json` to confirm per-defect
counts before launching Phase 3 — GPU + model load cost is fixed, so a
4-of-20 yield is worth aborting on.

## DEFT-Loop Parameters

The parent invokes `tao-skill-bank:tao-generate-anomalies` (or the wrappers
directly) with:

| Param | Value | Notes |
|---|---|---|
| `mode` | `inference_only` (or omit when calling wrappers directly) | DEFT loop never runs Phase 1 |
| `checkpoint_dir` | Directory containing `ag_config.yaml` plus the staged iteration checkpoint; resolve `checkpoints/<project>` or its single nested override such as `checkpoints/<project>/nvidia/Cosmos-AnomalyGen-PCB-2B`. The normal layout has `checkpoints/latest_checkpoint.txt` and `checkpoints/model/iter_<step>.pt`; normalize a flat staged repo into a run-local view as shown below. | |
| `step` | int parsed from `checkpoint_dir/checkpoints/latest_checkpoint.txt` | strip `iter_` prefix and `.pt` suffix |
| `dataset_dir` | `<workspace>/augmentation/anomalygen/datasets/<project>/` | passed verbatim |
| `clean_dir` | same as `dataset_dir` | |
| `defect_spec` | `${dataset_dir}/defect_spec.jsonl` | |
| `num_SDG` | per-iter budget from `deft_state.json` | proportionally allocated across defect types by mask count |
| `num_gpus` | `1` | |
| `model_size` | `2b` (pinned by the DEFT AOI bootstrap and invocation) | |
| `output_dir` | `${RESULTS_DIR}/iter${N}/anomalygen/sdg/` | receives `reconstructed_image/`, `original_image/`, `SDG_result.csv` |
| `cosmos_models_dir` | `${COSMOS_MODELS_DIR}` | resolved in Pre-Flight |
| `num_search_run` | `0` | skip Phase 5 search rounds |
| `nn_threshold` | `0` | skip Phase 7 filter+regen |

## Shared shell setup

Used by both the bootstrap and the per-iteration calls:

```bash
# Per-iteration calls use staged assets and require no credential.
WS=<workspace>
DS=$WS/augmentation/anomalygen/datasets/<project>
CKPT=$(find -L "$WS/augmentation/anomalygen/checkpoints/<project>" -path '*/ag_config.yaml' -print -quit | xargs -r dirname)  # -L follows a symlinked project checkpoint dir
: "${CKPT:?HARD STOP: PAIDF-compatible AnomalyGen checkpoint missing; finetune the AnomalyGen LoRA with the container texture_ft recipe using the uc1_pcb dataset layout to produce ag_config.yaml + iter_<step>.pt, then stage them under augmentation/anomalygen/checkpoints/<project>/...}"
COSMOS=$WS/augmentation/anomalygen/base_checkpoints
RUN_DIR=$WS/results/run_<TS>/iter${N}/anomalygen
TAO_SKILL_BANK_ROOT=<tao-skill-bank>
DEFT_AOI_SKILL_ROOT=$TAO_SKILL_BANK_ROOT/skills/applications/tao-run-deft-aoi
: "${AG_IMAGE:?AG_IMAGE unset — resolve images.metropolis_sdg.paidf_anomalygen from versions.yaml in Pre-Flight step 5}"
mkdir -p $COSMOS $DS $(dirname $CKPT) $RUN_DIR/amp $RUN_DIR/sdg
for p in "$COSMOS" "$DS" "$(dirname "$CKPT")" "$RUN_DIR"; do
  probe="$p/.tao-write-probe.$$"
  (umask 077 && : >"$probe" && rm -f "$probe") || {
    rm -f "$probe" 2>/dev/null || true
    echo "FATAL: $p is not writable by uid $(id -u)" >&2
    exit 2
  }
done
```

After approval, normalize the flat, pre-staged PAIDF 1.1 package without
modifying the source. The package stages `iter_000015000.pt` beside
`ag_config.yaml`, while the container loads
`checkpoints/model/iter_000015000.pt`. If the normal layout is already
complete, keep `$CKPT` unchanged. Otherwise require exactly one root iteration
file and build a lightweight view under the run directory:

```bash
if [ ! -f "$CKPT/checkpoints/latest_checkpoint.txt" ] ||
   ! latest_name=$(sed -n '1p' "$CKPT/checkpoints/latest_checkpoint.txt") ||
   [ ! -f "$CKPT/checkpoints/model/$(basename "$latest_name")" ]; then
  mapfile -t staged_models < <(
    find -L "$CKPT" -maxdepth 1 -type f -name 'iter_[0-9]*.pt' \
      ! -name '*_reg_model.pt' | sort
  )
  if [ "${#staged_models[@]}" -ne 1 ]; then
    echo "FATAL: expected one flat AnomalyGen iter checkpoint under $CKPT" >&2
    exit 2
  fi
  model_name=$(basename "${staged_models[0]}")
  CKPT_VIEW="$RUN_DIR/checkpoint_view"
  mkdir -p "$CKPT_VIEW/checkpoints/model"
  ln -sfn "$(realpath "$CKPT/ag_config.yaml")" "$CKPT_VIEW/ag_config.yaml"
  ln -sfn "$(realpath "${staged_models[0]}")" \
    "$CKPT_VIEW/checkpoints/model/$model_name"
  printf '%s\n' "$model_name" > "$CKPT_VIEW/checkpoints/latest_checkpoint.txt"
  CKPT="$CKPT_VIEW"
fi
```

This normalization is a path adapter, not a download. Record the resulting
`$CKPT` and derive `step` from its `latest_checkpoint.txt` before invoking
Phase 2 or Phase 3.

Required env for the per-iteration calls: `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, and
`PYTHONPATH=/workspace/paidf-anomalygen`. Required workdir:
`/workspace/paidf-anomalygen` (the `-m scripts.…` invocation needs CWD —
this matches the current container's `WORKDIR` / `ENV PYTHONPATH`).
`${ANOMALYGEN_SCRIPTS}` is preset inside the container — do not export it
on the host. Use **single quotes** around the inner `bash -lc` so the host
shell doesn't expand `${ANOMALYGEN_SCRIPTS}`; use **double quotes with
escaped `\$`** (`bash -lc "\${ANOMALYGEN_SCRIPTS}/..."`) when you also need
host-side variables (like `$DS`, `$NUM_SDG`) expanded in the same line.

## Post-gate bootstrap (one-time, SKILL.md → Workflow step 2)

Order matters — (a) populates the base checkpoints that (b) depends on.
Run only the dataset/base-model steps the Pre-Flight Summary flagged
`WILL_AUTO_FETCH`. Both
are idempotent — re-running a completed step exits quickly.

```bash
set -a; source /path/to/.env; set +a   # omit if already exported
SB="$BANK/skills/applications/tao-run-deft-aoi/scripts/stage_bundle.py"
EX="$BANK/skills/applications/tao-run-deft-aoi/scripts/deft_exec.py"

# (a) Container-owned Cosmos base checkpoints (~22 GB; Text2Image 2B only).
# The cache being populated is the stage's OUTPUT, so it is --results-dir:
# declared inputs are read-only on every platform, and a bootstrap must write.
python3 "$SB" anomalygen.bootstrap_cosmos --results-dir "$COSMOS" \
  --arg '${ANOMALYGEN_SCRIPTS}/check.sh || python -m scripts.download_checkpoints --model_types text2image --model_sizes 2B' \
  > "$RUN_DIR/cosmos.bundle.json"
python3 "$EX" --state "$STATE" --submit --bundle "$RUN_DIR/cosmos.bundle.json" \
  --platform "$PLATFORM" $PLATFORM_CTX --ctx env_passthrough=HF_TOKEN

# (b) PCB reference dataset — prepare_dataset_uc1.py is the `tao-generate-anomalies`
# skill's PCB-dataset fetcher (`uc1` = the skill's identifier for the PCB
# use-case; unrelated to the host-side <project> directory label).
if [ ! -f "$DS/defect_spec.jsonl" ]; then
  python3 "$SB" anomalygen.bootstrap_dataset --results-dir "$DS" \
    --arg 'python3 -m scripts.utilities.prepare_dataset_uc1 $TAO_RESULTS_ROOT' \
    > "$RUN_DIR/dataset.bundle.json"
  python3 "$EX" --state "$STATE" --submit --bundle "$RUN_DIR/dataset.bundle.json" \
    --platform "$PLATFORM" $PLATFORM_CTX --ctx env_passthrough=HF_TOKEN
fi
```

Each bootstrap is a recorded job, so a 22 GB download that stalls is visible
through `--logs` rather than being a silent hang inside a `docker run`.
`HF_TOKEN` is passed by name only — the value comes from the environment and is
never written into a bundle, a record, or a log. These touch the network, so the
launch gate refuses them under `--ctx airgap=1`; pre-stage the directories
instead. See `references/stage-execution.md`.

The fine-tuned checkpoint is not part of this bootstrap. It must pass the
pre-staged PAIDF compatibility and discovery checks above before either fetch
step is eligible to run.

The container's `check.sh` owns asset completeness and skips the downloader
when its contract is already satisfied. The fallback command explicitly
requests `--model_sizes 2B`; a future larger model is a separate workflow
opt-in, never an expansion of this default command.

## Flat BYO checkpoint layout — build a `checkpoint_view`

`run_sdg.sh` resolves the model file as
`<checkpoint_dir>/checkpoints/model/iter_<STEP>.pt`, but the PAIDF 1.1 package
is staged **flat** (`<dir>/iter_<STEP>.pt` + `ag_config.yaml`).
Passing the flat directory fails with
`FileNotFoundError: .../checkpoints/model/iter_<STEP>.pt`. Build a symlink
adapter once per iteration output dir and pass it as `--checkpoint_dir`:

```bash
CV=${RUN_DIR}/checkpoint_view
mkdir -p $CV/checkpoints/model
ln -sf $CKPT_DIR/iter_000015000.pt $CV/checkpoints/model/iter_000015000.pt
ln -sf $CKPT_DIR/ag_config.yaml   $CV/ag_config.yaml
echo "iter_000015000.pt" > $CV/checkpoints/latest_checkpoint.txt
```

## Per-iteration invocation (every loop iteration)

After bootstrap, the per-iteration AnomalyGen stage is two launches
calls — same image, READ-ONLY mount on the cosmos cache.

```bash
STEP=$(sed 's/^iter_0*\([0-9]*\)\.pt$/\1/' $CKPT/checkpoints/latest_checkpoint.txt)
SDG_LOG=$RUN_DIR/sdg.log

# Phase 2: AMP routing → testcase.jsonl  (~10s, declared gpus=0)
python3 "$SB" anomalygen.amp --results-dir "$RUN_DIR" \
  --param dataset_dir="$DS" \
  --param defect_spec="$DS/defect_spec.jsonl" \
  --param cosmos_models="$COSMOS" \
  --arg "\${ANOMALYGEN_SCRIPTS}/prep_testcase.sh \
    --name iter${N} --num-sdg $NUM_SDG \
    --dataset-dir \$TAO_INPUT_DATASET_DIR --clean-dir \$TAO_INPUT_DATASET_DIR \
    --defect-spec \$TAO_INPUT_DEFECT_SPEC \
    --amp-output-dir \$TAO_RESULTS_ROOT/amp \
    --output-jsonl \$TAO_RESULTS_ROOT/testcase.jsonl" \
  > "$RUN_DIR/amp.bundle.json"
AMP=$(python3 "$EX" --state "$STATE" --submit --bundle "$RUN_DIR/amp.bundle.json" \
        --platform "$PLATFORM" $PLATFORM_CTX)
python3 "$EX" --state "$STATE" --await-job "$AMP" $PLATFORM_CTX \
  || { echo "FATAL: AMP stage failed; guardrail=not_run" >&2; exit 2; }

# Phase 3: SDG diffusion → reconstructed_image/ + original_image/  (1-3 min)
test -s "$COSMOS/nvidia/Cosmos-Guardrail1/video_content_safety_filter/safety_filter.pt" || { echo "FATAL: AnomalyGen Guardrail checkpoint missing; guardrail=not_run" >&2; exit 2; }

python3 "$SB" anomalygen.sdg --results-dir "$RUN_DIR" \
  --param testcase_jsonl="$RUN_DIR/testcase.jsonl" \
  --param checkpoint_dir="$CKPT" \
  --param cosmos_models="$COSMOS" \
  --arg "\${ANOMALYGEN_SCRIPTS}/run_sdg.sh \
    --checkpoint_dir \$TAO_INPUT_CHECKPOINT_DIR --step $STEP \
    --input_jsonl \$TAO_INPUT_TESTCASE_JSONL \
    --output_dir \$TAO_RESULTS_ROOT/sdg \
    --model_size 2b --num_gpus 1" \
  > "$RUN_DIR/sdg.bundle.json"
SDG=$(python3 "$EX" --state "$STATE" --submit --bundle "$RUN_DIR/sdg.bundle.json" \
        --platform "$PLATFORM" $PLATFORM_CTX)
python3 "$EX" --state "$STATE" --await-job "$SDG" $PLATFORM_CTX
sdg_rc=$?

# The guardrail assertion below reads the container's output, so capture it from
# the PLATFORM rather than a local pipe: `tee` only works when the container
# runs on this host, and the point of the bundle is that it may not.
python3 "$EX" --state "$STATE" --logs "$SDG" --tail 2000 $PLATFORM_CTX > "$SDG_LOG" 2>&1 || true

if [ "$sdg_rc" -ne 0 ]; then
  echo "FATAL: SDG stage exited $sdg_rc; guardrail=not_run" >&2
  exit "$sdg_rc"
fi

test -s "$SDG_LOG" && ! grep -Fqi "post-generation image checks are DISABLED" "$SDG_LOG" || { echo "FATAL: AnomalyGen Guardrail log missing or screening disabled; guardrail=not_run" >&2; exit 2; }
```

Both one-line checks are hard gates. On failure, end the `anomalygen` stage as
`--status error`, include `guardrail=not_run` in the summary, and halt; never
commit the generated output downstream.

Required mounts (per-iteration): `<workspace>:<workspace>` (same path) +
`<cosmos_models_dir>:/workspace/paidf-anomalygen/checkpoints:ro`.
All bootstrap and per-iteration calls map the submitting UID:GID, so their
writable workspace/results artifacts remain updateable and removable without
sudo on resume.

## Output layout

```
<output_dir>/
├── allocation.json                         # Phase 2 defect -> AMP count proof
├── SDG_result.csv                          # one row per generated sample (image, mask, params, PSNR)
├── reconstructed_image/<T>+<A>_<idx>.png   # synthetic NG — ChangeNet input_path
├── original_image/<T>+<A>_<idx>.png        # paired OK — ChangeNet golden_path
├── original_mask/, cropped_image/, cropped_mask/, annotated_image/   # intermediates
└── timing_summary.json
```

After verifying `SDG_result.csv`, `reconstructed_image/`, and
`original_image/`, confirming the disabled-screening marker is absent, and
checking that every CSV Guardrail verdict passed, keep Phase 2's
`allocation.json` beside those outputs and commit:

```python
phase = state["iterations"][f"iter{N}"]
phase["anomalygen_sdg_csv"] = "<abs_path>/SDG_result.csv"
phase["anomalygen_allocation_json"] = "<abs_path>/allocation.json"
phase["anomalygen_amp_allocated"] = <sum of allocation.json counts>
phase["stage_completed"] = "anomalygen"
```

This snippet documents the schema only; never execute it as inline Python.

## Log Stage

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/commit_stage.py \
    --results-dir "${RESULTS_DIR}" \
    --iter-label iter${N} \
    --stage anomalygen \
    --anomalygen-sdg <absolute path to SDG_result.csv> \
    --anomalygen-allocation <absolute path to allocation.json> \
    --duration-sec "${STAGE_DURATION_SEC}" \
    --summary "SDG: requested=N, AMP-allocated=M, generated=K by type; guardrail=passed"
```

`commit_stage.py` derives `M` by summing the committed defect-to-count
`allocation.json` and audits that disk proof on resume. When `M < N` (AMP
yield gap), include both requested and allocated counts
— that's the signal a reviewer needs to spot allocation-vs-generation
bottlenecks.

## Guardrail tri-state and container follow-up

Interpret image screening as a tri-state, never a boolean default:

- `passed`: screening ran and accepted the generated row;
- `failed`: screening ran and rejected the row; fail the stage rather than
  emitting that row into training data;
- `not_run`: screening was disabled or failed to initialize; fail the stage,
  record `guardrail=not_run` (never `passed`), and treat the rows as unscreened.

The current container CSV schema cannot represent `not_run` and may write `1`
when initialization failed. The disabled-marker log check above blocks that
contradiction. The paired container follow-up must add the tri-state schema and
stop emitting safety-passed values for unscreened content.
