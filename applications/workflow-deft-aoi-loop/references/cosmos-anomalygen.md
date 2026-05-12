# Cosmos AnomalyGen — DEFT Loop Reference

Read this when the parent runs the `anomalygen` stage. The underlying skill
`tao-skill-bank:cosmos-anomalygen` (`data/cosmos-anomalygen/SKILL.md`) owns the
docker invocation, phase descriptions, parameter reference, and error patterns —
use `mode=inference_only` for DEFT loop iterations (checkpoint already exists).
This file only covers the DEFT-loop-specific overlay: pool layout, input hygiene,
required mounts/env, container-script substitutions, and logging.

## Dataset Layout

All reference data lives under `augmentation/anomalygen/checkpoints/<project>/dataset/`:

```
dataset/
├── semantic_segmentation_labels.json
└── <defect_type>/                          # folder name == defect_type from defect_spec.jsonl (e.g. IC+bridge)
    ├── anomaly_image/                      # defect reference images — primary source for pool staging
    ├── mask/                               # per-sample defect masks (paired 1-to-1 with anomaly_image/)
    ├── cad_mask/                           # CAD component masks — required for cad spatial_dependency
    └── clean_image/                        # clean reference images (AMP inpainting targets)
```

One folder per `defect_type` entry in `defect_spec.jsonl` — the folder name IS the `defect_type` string verbatim.
No sub-grouping by component type.

For `cad` spatial_dependency, AMP uses `cad_mask/<stem>.png` (same stem as the clean/anomaly image) to
locate the component ROI — no text prompt or SAM2 pass is needed.

## Pool Layout

> **Source of truth — single, fixed.** All pool inputs are copied **only** from
> `augmentation/anomalygen/checkpoints/<project>/`, namely:
> - `<project>/dataset/` (per-defect `mask/`, `cad_mask/`, `clean_image/` and `semantic_segmentation_labels.json`)
> - `<project>/defect_spec.jsonl`
> - `<project>/ag_config.yaml` (read for `texture`; not copied)
>
> **Nothing else is ever read or injected** — no `train/base/training_set.csv`,
> no KPI images, no prior-iteration SDG outputs, no real-sample injection from
> `routing_*_parquet`, no other PASS/NG sources. If a required subdirectory
> under `dataset/<T>+<A>/` is empty, **hard stop** — never substitute.

The parent stages a pool at `results/iter${N}/pool_anomalygen/inputs/` by mechanically restructuring the canonical input. `<T>` = `ag_config.yaml.texture` (e.g. `PCB`); `<A>` = anomaly suffix from `defect_type` (`<T>+<A>`).

| Source (under `<project>/`) | Pool destination (under `pool_anomalygen/inputs/`) |
|---|---|
| `dataset/semantic_segmentation_labels.json` | `masks_structured/semantic_segmentation_labels.json` |
| `dataset/<T>+<A>/mask/*.png` | `masks_structured/<T>/mask/<A>/*.png` |
| `dataset/<T>+<A>/cad_mask/*.png` | `masks_structured/<T>/cad_mask/*.png` (deduped across `<A>`) |
| `dataset/<T>+<A>/clean_image/*` | `clean/train_set/*` (flat) |
| `defect_spec.jsonl` | `defect_spec.jsonl` (verbatim; if any `defect_type` prefix differs from `<T>`, remap during copy — e.g. `IC+bridge` → `PCB+bridge`) |

`anomaly_image/` is **not** staged. AnomalyGen at inference time generates new defects from `mask/` + `clean_image/` + the fine-tuned checkpoint; it does not consume `anomaly_image/`.

`masks_structured/` is what `prep_testcase.sh` consumes as `--dataset-dir` — the underlying skill expects the `<T>/mask/<A>/` split, while the canonical dataset stores masks flat under `<T>+<A>/mask/`. The table above is the only bridge.

**Mask format.** `mask/` and `cad_mask/` files must be binary 0/255 PNG. JPEG fails SDG's binary check (`ValueError: Mask is not binary`). The canonical dataset is expected to comply.

**`cosmos_models_dir`** — absolute host path to the directory containing `Cosmos-Predict2-2B-Text2Image/`. Resolved by Pre-Flight as `COSMOS_MODELS_DIR`. Required mount; the AnomalyGen container does not bundle these weights (missing → `FileNotFoundError: ... tokenizer.pth`).

## Input hygiene (parent must do before invoking the skill)

- **Resize OK images and masks to `ag_config.yaml.dataloader_train.dataset.image_size`** (e.g. `512×512`). Mismatched pair sizes raise `AssertionError: Image filename ... 's size with mask filename ...` at `_prepare_diffusion_inference_data_batches`.
- **Masks must be binary PNG (0/255)**. JPEG masks fail with `ValueError: Mask is not binary` (raised in `mask_augmentation.augment_binary_mask` and again in `_load_image_and_mask`) because JPEG compression smears pixel values. Threshold at 128 and re-save as PNG.
- **Smoke-test with a 1-row JSONL before scaling.** Model load is ~5 min and the 2B diffusion holds ~50 GB VRAM — per-row failures after a successful load are cheap, but per-row failures during a fresh load are expensive. Validate JSONL fields, image/mask sizes, and mask binarity on one row first; only then submit the full `num_SDG`-row batch.
- **Strip training-only keys from `ag_config.yaml`** before pointing the SDG script at it: drop `scheduler`, `trainer`, `checkpoint`, `dataloader_train`, `dataloader_val`. The omegaconf struct under `cosmos_predict2/configs/base/ag_config.py` rejects unknown keys (e.g. `scheduler.warm_up_steps` exists in the lambdalinear scheduler config but the default is `constant`, which doesn't have it). Symlink the rest of the checkpoint dir into a shim and write the sanitized `ag_config.yaml` next to the symlinks.

## DEFT-Loop Parameters

```
mode=inference_only
checkpoint_dir=<workspace>/augmentation/anomalygen/checkpoints/<project>
step=<from checkpoints/latest_checkpoint.txt>
num_SDG=<per-iter budget>
num_gpus=1
cosmos_models_dir=<COSMOS_MODELS_DIR>   # resolved by parent in Pre-Flight
```

**Required docker mount for Cosmos base weights:**

```bash
-v <cosmos_models_dir>:/workspace/cosmos-anomalygen/checkpoints:ro
```

The container expects `Cosmos-Predict2-2B-Text2Image/` (and sibling model dirs) at `/workspace/cosmos-anomalygen/checkpoints/`. Without this mount, SDG exits with `FileNotFoundError: checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image/tokenizer/tokenizer.pth`.

**Required docker env:**

```bash
-e PYTHONPATH=/workspace/cosmos-anomalygen   # imaginaire/cosmos_predict2 import from CWD; not pip-installed
-e HF_HUB_DISABLE_XET=1                       # xet downloader hits permission errors when downloading missing weights
```

Without `PYTHONPATH`, SDG fails immediately with `ModuleNotFoundError: No module named 'imaginaire'`.

## Container script paths (note for `data/cosmos-anomalygen/SKILL.md` readers)

In container image `1.0.3-006434bb.main`, the shell wrappers `scripts/run_sdg.sh`, `scripts/prep_testcase.sh`, and `scripts/validate_checkpoint.py` referenced by the skill **do not exist**. The actual entry points are (`validate_jsonl.py` / `verify_output.sh` have no equivalent — rely on the smoke-test row and on the SDG script's own row-level errors):

| SKILL.md says | Actual script in container |
|---|---|
| `scripts/run_sdg.sh` | `torchrun --nproc_per_node=<N> scripts/anomaly_gen/synthetic_dataset_generation.py --ag_checkpoint_dir <CKPT> --step <STEP> --input_data_path <JSONL> --output_image_path <OUT>` |
| `scripts/prep_testcase.sh` | `scripts/anomaly_gen/create_testcase.py` (different flag set; takes `--OK_image_path`, `--NG_mask_path`, `--name`, `--SDG_RATIO`, etc.). Produces `ag_inference/<testcase>.jsonl`, which `synthetic_dataset_generation.py` consumes as `--input_data_path`. |
| `scripts/validate_checkpoint.py` | (no equivalent — verify `ag_config.yaml` and `checkpoints/model/iter_<step>.pt` exist manually) |

Output layout for `synthetic_dataset_generation.py`: `<output_image_path>/{reconstructed_image,original_image,cropped_image,annotated_image,original_mask,cropped_mask}/<anomaly_type>_<index>.png` plus `SDG_result.csv`. `reconstructed_image/` holds the synthetic NG output; `original_image/` is the OK input.

## Log Stage

```bash
python3 <skill_root>/scripts/log_stage.py \
    --log-path results/loop_log.jsonl \
    --iter-label iter${N} \
    --stage anomalygen --status ok \
    --summary "SDG: N samples by type; gates passed"
```
