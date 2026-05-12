# Inference Reference — Phases 2–7

Detail for `cosmos-anomalygen` Phases 2–7. Read before executing any inference phase.
Operational summaries for each phase — deep detail is in the per-phase reference files
(`prep-testcase.md`, `sdg-inference.md`, `eval.md`, `sdg-refine.md`).
When updating, keep those files in sync.

## Contents
- [Pre-flight](#pre-flight-environment-and-checkpoint-validation)
- [Phase 2: AMP Routing and JSONL Preparation](#phase-2-amp-routing-and-jsonl-preparation)
- [Phase 3: SDG Inference](#phase-3-sdg-inference)
- [Eval (Phases 4 and 6)](#eval-phases-4-and-6)
- [Phase 5: Per-Sample Search](#phase-5-per-sample-search)
- [Phase 7: Threshold Filter](#phase-7-threshold-filter)

---

## Pre-flight: Environment and Checkpoint Validation

### Environment check

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'torch={torch.__version__}, CUDA={torch.cuda.is_available()}, devices={torch.cuda.device_count()}')"
```

Stop if CUDA unavailable (`conda activate cosmos-predict2`); stop if `num_gpus`
exceeds detected count. If compilation errors: `export CUDA_HOME=$CONDA_PREFIX`.

### Checkpoint validation (run before Phase 3)

`validate_checkpoint.py` ensures `ag_config.yaml` exists and is well-formed.
It prints the supported `TEXTURE+ANOMALY` set — use this as `DEFECTS` if not
already set from `defect_spec`.

```python
# What validate_checkpoint.py reads:
cfg["dataloader_train"]["dataset"]["anomaly_types"]
# Each entry is [texture, anomaly] → rendered as "texture+anomaly"
```

Stop if `ag_config.yaml` is missing — ask the user to verify `checkpoint_dir`.

---

## Phase 2: AMP Routing and JSONL Preparation

Read `references/prep-testcase.md` for the full parameter table, helper script
descriptions, pipeline invariant, two calling conventions (validation vs
inference), and AMP error diagnosis.

### AMP routing table

| `spatial_dependency` | AMP branch | Extra inputs required |
|---|---|---|
| `free` | Whole-image ROI | none |
| `text` | text2roi (Qwen VL + SAM2) | `roi_prompt_defect_location` in `defect_spec` |
| `cad` | cad2roi | `<dataset>/<TEXTURE>/cad_mask/<stem>.png` + `<dataset>/semantic_segmentation_labels.json` |

Unrecognized values (including legacy `"roi"`) default to `text`. Auto-cad
routing fires whenever a record has a non-null `cad_mask` regardless of
`spatial_dependency`.

### Clean image discovery

`prep_testcase.sh` probes these layouts in order:
1. `<clean_dir>/<TEXTURE>/clean_image/*`
2. `<clean_dir>/<TEXTURE>/*`
3. flat `<clean_dir>/*`

When clean images live at `<dataset_dir>/<TEXTURE>/clean_image/`, omit
`--clean-dir` (defaults to `dataset_dir`).

### Pairing strategy

Budget is **1 clean per submask** (not a clean × submask cartesian product).
Cleans rotate across submasks so every training submask is represented before
any repeats. JSONL defaults: `guidance=7.0`, `crop_ratio=2.0`, `seed=42` —
these are the per-sample starting values overridden by Phase 5 search.

### n_seeds sizing

Auto-computed from allocation — do NOT pass `--seeds`:

```
n_seeds = max_d ⌈allocation[d] / num_submasks[d]⌉
```

With proportional allocation this simplifies to `⌈num_SDG / total_training_masks⌉`.

### Submask handling

AMP's `preprocess_submask` defaults `submask_split_largest=True` — a
multi-component training submask is reduced to its largest connected component.
Accept this default unless preserving multiple components is required for a
specific defect. To override, set `submask_split_largest: false` on the
individual record in `amp_samples.json` before `build_jsonl.py` runs.

### Verification after prep_testcase.sh

- JSONL has `num_SDG` entries (minus logged AMP skips).
- `allocation.json` sums to `num_SDG`.
- `amp_samples.json` has exactly `num_SDG` records.

### Phase 2 errors

- Validator failure → stop with itemised report (missing submask/clean/cad/prompt).
- AMP output count < allocation → `build_jsonl.py` errors (check `run_auto_roi_amp.py` logs for NO_DETECTION / FAILED).
- Mask-size mismatch → `verify_jsonl.py` auto-resizes into `resized_masks/` cache.

---

## Phase 3: SDG Inference

Read `references/sdg-inference.md` for the full parameter table, step-by-step
detail, multi-GPU VRAM constraints, and verification checklist.

### JSONL validation against checkpoint

`validate_jsonl.py` cross-checks anomaly types in the JSONL against the
checkpoint's supported set. If any are unsupported, stop and show the mismatch:
```
Supported:  TEXTURE+TYPE_A, TEXTURE+TYPE_B
JSONL has:  TEXTURE+TYPE_C  ← unsupported
```
Options: retrain with extended defect set, use an isolated model for the new
defect, or trim `defect_spec` to supported types.

If many image/mask paths are missing → stop; ask the user to verify paths
before burning GPU time.

### run_sdg.sh flags

```bash
.claude/skills/cosmos-anomalygen/scripts/run_sdg.sh \
    --checkpoint_dir <CKPT> \
    --step <STEP> \
    --input_jsonl <JSONL> \
    --output_dir <OUTPUT_DIR> \
    --model_size <2b|14b> \
    [--num_gpus N] \
    [--seed 0]
```

### Multi-GPU caveats

| Config | Behavior |
|---|---|
| 14B + `num_gpus > 1` | FSDP auto-disabled at inference; each rank holds the full 14B model (~80 GB VRAM per GPU) |
| 14B + single GPU | FSDP enabled; fits on smaller GPUs |
| 2B + any num_gpus | Standard DDP; no special VRAM constraint |

NCCL hang controls (set as env vars before `run_sdg.sh` if needed):

| Var | Default | Effect |
|---|---|---|
| `ANOMALY_GEN_FINALIZE_BACKEND` | `gloo` | Set to `nccl` only if gloo finalization hangs |
| `TORCH_DISTRIBUTED_TIMEOUT_SEC` | `1800` | Process-group timeout in seconds |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | (alias) | Same effect as `TORCH_DISTRIBUTED_TIMEOUT_SEC`; set whichever is defined in your environment |
| `MASTER_PORT` | `12341` | Override if port is occupied |

### SDG output structure

```
<output_dir>/
├── reconstructed_image/   # final synthetic anomaly images
├── annotated_image/
├── cropped_image/
├── cropped_mask/
├── original_image/
├── original_mask/
└── SDG_result.csv
```

### Verify before eval

```bash
.claude/skills/cosmos-anomalygen/scripts/verify_output.sh ${JSONL} ${OUTPUT_DIR}
```

Counts in `reconstructed_image/` must match JSONL entry count. If SDG was
interrupted, do NOT eval the partial output — re-run SDG first.

### Phase 3 errors

- Missing `ag_config.yaml` → stop; verify `checkpoint_dir`.
- `step` not on a `save_iter` boundary → `torch.load` FileNotFoundError; run `ls ${CKPT}/checkpoints/model/iter_*.pt` and pick a valid step.
- OOM → reduce `num_gpus` or switch to `model_size=2b`.
- Non-binary masks → pipeline raises `ValueError`; masks must be 0 or 255 only.
- Port conflict → increment `MASTER_PORT`.

---

## Eval (Phases 4 and 6)

Read `references/eval.md` for the full parameter reference, example output
table (FID column comes first — surprises people), detailed pre-flight, and
all error cases. Summary of the essentials below.

### run_eval.sh flags

```bash
.claude/skills/cosmos-anomalygen/scripts/run_eval.sh \
    --real-path <real_path> \
    --generated-path <generated_path> \
    --anomaly-types <TEXTURE+TYPE> [<TEXTURE+TYPE> ...] \
    [--per-sample-csv <path>] \
    [--backbone cradio_v3_base]
```

`per_sample.csv` is always emitted to `<generated_path>/per_sample.csv` by
default. Columns: `anomaly_type`, `path`, `nn_score`, `mnn_score`.
`sdg-refine` and `filter_by_nn.py` both consume this file.

### Score interpretation

`nn_score` (DINOv2 nearest-neighbour, locked to `dinov2-large`) is the **key
KPI** — higher is better. It is **relative**: compare within the same dataset
and backbone, not across datasets. Scoring uses top-K=3 averaging
(`compute_correspondence_kpi`): each generated image is scored against every
real reference of its type and the three highest-NN values are averaged.

`mnn_score` (mutual nearest-neighbour) is a diagnostic for refinement internals.
`fid` (using `backbone` flag, default `cradio_v3_base`) is a secondary diagnostic.

If a type has < 2 real or generated samples → FID is skipped for that type
and logged as `None`; `nn_score`/`mnn_score` still run.

If a type has < 10 real images → warn that `nn_score` may be noisy. Interpret
results cautiously and consider acquiring more training data.

### Feature counts

The eval log reports feature counts that may **exceed** image count. This is
expected: `mask_crop_images` extracts individual defect instances via DBSCAN
clustering (eps=30) on mask contours. A single image with a multi-region mask
produces multiple feature crops.

### Eval pre-flight

Check before running:
1. `reconstructed_image/` exists and is non-empty.
2. No SDG process is still writing to the output directory.
3. Auto-detect anomaly types from generated filenames if not supplied:
   ```bash
   ls <generated_path>/reconstructed_image/ | sed 's/_[0-9]*\.\(png\|jpg\)$//' | sort -u
   ```

### Eval errors

- Empty `reconstructed_image/` → SDG may not have finished; stop.
- Anomaly type not found in real data → 0 real features; check `<TEXTURE>/anomaly_image/<TYPE>/` structure.
- SDG still running → wait for completion; partial output produces incorrect scores.

---

## Phase 5: Per-Sample Search

Read `references/sdg-refine.md` for the full inputs table, draws.json
alignment detail (`SDG_result.csv` index column), output layout, and
re-AMP heuristics.

### Claude's draw strategy

For each round `r`:
1. Read `per_sample.csv` from round `r-1` (or `original/per_sample.csv` for `r=1`).
2. For each sample, pick new `(guidance, crop_ratio)`. Focus search budget on
   low-scoring samples. Skip samples already scoring well to save inference time.
3. Write `draws.json` to `${ROUNDS}/round_${r}/draws.json`.

### draws.json format

```json
{
  "<sample_index>": {"guidance": 4.2, "crop_ratio": 2.7},
  "<sample_index>": {"guidance": 8.1, "crop_ratio": 1.8}
}
```

- Sample indices are 0-based JSONL line numbers.
- Omitted samples are not retried this round (stay at current best-seen).
- Ranges to consider: `guidance ∈ [1.5, 10.0]`, `crop_ratio ∈ [1.5, 10.0]`.
  Narrow or widen based on prior-round feedback.

`run_round.sh` produces `rounds/round_r/testcase.jsonl`, `sdg/`, and
`per_sample.csv`. The `sdg/SDG_result.csv.index` column carries the
base-JSONL `sample_index`, keeping `assemble_searched.py` aligned across rounds.

### Re-rolling AMP (optional)

By default the `(clean_image, mask)` pair per sample index is fixed across all
rounds — only `(guidance, crop_ratio)` change. If `nn_score` stays flat for a
sample across ≥ 2 rounds of `(g, c)` variation, the mask placement itself may
be the blocker. Add `--reamp-seed` to re-run AMP
with a fresh base seed on the same `(clean, submask)` pairs:

```bash
.claude/skills/cosmos-anomalygen/scripts/run_round.sh \
    ...standard args... \
    --reamp-seed $((1000 + r)) \
    --defect-spec ${DEFECT_DESC}
```

Use sparingly — each re-AMP is one extra `run_auto_roi_amp.py` pass. The
text2roi ROI cache is reused, so Qwen VL + SAM2 are not re-run.

### num_search_run = 0

Valid. Skip `run_round.sh` entirely; run `assemble_searched.py` with an empty
rounds dir — `searched/` will equal `original/`.

### search_summary.csv

After `assemble_searched.py`: `rounds_dir/search_summary.csv` has one row per
sample with `best_round`, `best_guidance`, `best_crop_ratio`, `best_nn_score`,
`attempts`.

---

## Phase 7: Threshold Filter

`filter_by_nn.py` reads `per_sample.csv` from the source bucket and copies
samples with `nn_score >= threshold` into `filtered/`. Samples below threshold
go to `filtered/drop/` (kept for audit, not included in the main output).

After filtering, `eval_filtered.log` provides the final quality report on
the kept set. The count in `filtered/reconstructed_image/` will be ≤ `num_SDG`.
