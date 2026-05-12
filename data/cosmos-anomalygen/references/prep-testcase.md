# Prep-Testcase Reference — Phase 2 JSONL Preparation

Full detail for `prep_testcase.sh`. Read when debugging AMP failures,
understanding allocation, or customising clean-image pairing.

---

## What it produces

A JSONL that pairs clean images with AMP-placed masks. Every defect routes to
one of three AMP branches based on `spatial_dependency` in `defect_spec`:

| spatial_dependency | AMP branch | Extra inputs |
|---|---|---|
| `free` | whole-image ROI | none |
| `text` | text2roi (Qwen VL + SAM2) | `roi_prompt_defect_location` (text, required) |
| `cad` | cad2roi | `<dataset>/<TEXTURE>/cad_mask/<stem>.png` + `<dataset>/semantic_segmentation_labels.json` |

`run_auto_roi_amp.py` defaults unrecognized values (including legacy `"roi"`)
to `text`; auto-cad routing fires whenever the per-sample record has a
non-null `cad_mask` regardless of `spatial_dependency`.

The JSONL contains no mask-augmentation fields — AMP handles placement.

---

## Pairing strategy and n_seeds

Pair budget is **1 clean per submask** (not the clean × submask cartesian)
so every training submask is represented before any repeats. Cleans rotate
across submasks to spread the load without inflating the pair space.

n_seeds is auto-computed so AMP produces enough placements per record to
cover the allocation:

```
n_seeds = max_d ⌈allocation[d] / num_submasks[d]⌉
```

With proportional allocation this reduces to `⌈num_SDG / total_training_masks⌉`.

**Pipeline invariant:**

```
num_SDG → allocation (proportional to mask counts)
       → ⌈alloc[d]/n_seeds⌉ AMP records per defect
       → ≥num_SDG AMP masks (n_seeds placements per record)
       → N JSONL rows (first alloc[d] per defect)
```

**Two calling conventions:**
- **Validation** (Phase 1 fine-tuning): `num_SDG` = total training mask count
  → n_seeds=1; every training submask appears once with one clean.
- **Inference** (Phase 2): `num_SDG` = user-supplied target → n_seeds scales
  with `num_SDG / total_masks`; each submask produces multiple placements with
  different random offsets within its ROI.

---

## Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `name` | yes | — | Experiment label. |
| `num_SDG` | yes | — | Total SDG entries across all defect types. `0` → stop. |
| `dataset_dir` | yes | — | Training dataset root. Drives allocation, submask source, cad_mask lookup (`<dataset>/<TEXTURE>/cad_mask/<stem>.png`), and cad labels (`<dataset>/semantic_segmentation_labels.json`). |
| `clean_dir` | no | `dataset_dir` | Clean images. Layouts probed in order: `<clean_dir>/<TEXTURE>/clean_image/*`, `<clean_dir>/<TEXTURE>/*`, flat `<clean_dir>/*`. Omit when clean images are at `<dataset_dir>/<TEXTURE>/clean_image/`. |
| `defect_spec` | yes | — | JSONL tagging each defect `free`/`text`/`cad`. `text` entries need `roi_prompt_defect_location`. Template at `.claude/skills/cosmos-anomalygen/assets/defect_spec_template.jsonl`. |
| `guidance` | no | `7.0` | Default guidance written to each JSONL entry (overridden per-sample in Phase 5). |
| `crop_ratio` | no | `2.0` | Default crop ratio. Matches `cosmos_predict2/data/anomaly_gen/anomaly_dataset.py` fallback. |
| `seed` | no | `42` | Base random seed for `run_auto_roi_amp.py`. |

Defect types are derived from `defect_spec` — no separate `--defect-types` arg.

---

## Invocation

```bash
.claude/skills/cosmos-anomalygen/scripts/prep_testcase.sh \
    --name <name> \
    --num-sdg <N> \
    --dataset-dir <dataset_dir> \
    --clean-dir <path> \
    --defect-spec <path> \
    --amp-output-dir ag_inference/<name>/amp \
    --output-jsonl ag_inference/<name>/testcase.jsonl \
    [--guidance 7.0] [--crop-ratio 2.0] [--seed 42]
```

**Do NOT pass `--seeds`** — it is not a recognized flag and the script will
halt with `unknown arg`. n_seeds is auto-computed internally.

---

## Helper scripts (each supports `--help`)

| Script | Role |
|---|---|
| `validate_amp_inputs.py` | Pre-flight: cross-check dataset layout, clean pool, cad masks, cad labels, and `roi_prompt_defect_location`. Runs automatically as step 1. |
| `allocate_samples.py` | Proportional allocation of `num_SDG` across defect types by training mask count. |
| `build_amp_samples.py` | Emit exactly `allocation[defect]` AMP input records per defect. |
| `build_jsonl.py` | Scan AMP output, pair with clean images, honor allocation ceiling. |
| `verify_jsonl.py` | Resize mismatched masks into `resized_masks/` cache; validate all paths. |

The AMP branching lives in `scripts/run_auto_roi_amp.py` (repo root).

---

## Submask handling

`build_amp_samples.py` does not override `preprocess_submask`'s
`submask_split_largest=True` default — a multi-component training submask
(e.g., two scratches on one image) becomes a single-component mask.
Set `submask_split_largest: false` on an individual record in
`amp_samples.json` before `build_jsonl.py` runs if preserving multiple
components matters for that defect.

---

## Verification

After `prep_testcase.sh` completes:
- Output JSONL has `num_SDG` entries (minus logged AMP skips).
- `allocation.json` sums to `num_SDG`.
- `amp_samples.json` has exactly `num_SDG` records.
- `<amp-output-dir>/<clean_stem>__<submask_stem>/<TEXTURE>+<ANOMALY>/seed0.png` exists per AMP record.

---

## Error handling

| Symptom | Action |
|---|---|
| Validator failure | Stop with itemised report (missing submask/clean/cad/prompt) |
| AMP output count < allocation | `build_jsonl.py` errors; check `run_auto_roi_amp.py` logs for `NO_DETECTION` / `FAILED` |
| Mask-size mismatch | `verify_jsonl.py` auto-resizes into `resized_masks/` cache |
| `num_SDG = 0` | Stop — nothing to generate |
