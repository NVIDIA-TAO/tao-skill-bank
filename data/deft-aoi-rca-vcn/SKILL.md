---
name: deft-aoi-rca-vcn
description: Performs gap analysis on NVIDIA TAO VCN Classify (Visual Component Net) experiments by invoking the data-services container `nvcr.io/nvidian/iva/tao-toolkit-ds:aoi` directly via `docker run … gap_analysis vcn_aoi …` — picks the optimal decision threshold, ranks per-sample weakness, and emits a top-K weakest parquet expanded per-lighting for downstream augmentation. Use when analyzing VCN classification failures, picking SDA augmentation targets, or auditing PASS/NO_PASS boundary cases.
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit and a CUDA GPU. Pulls `nvcr.io/nvidian/iva/tao-toolkit-ds:aoi`.
metadata:
  author: NVIDIA Corporation
  version: '0.3'
allowed-tools: Read Bash
tags:
- data
- rca
- vcn
- aoi
---

# TAO VCN Classify Gap Analysis Skill

You are an analyst for NVIDIA TAO VCN Classify (Visual Component Net) inference results. Your job is to identify the **weakest samples per ground-truth label** by measuring signed distance from the decision threshold *in the wrong direction*, then surface them for downstream augmentation or relabeling.

This skill is intentionally lightweight. VCN's classify head is a single-score binary boundary (PASS vs NO_PASS by `siamese_score`), so the analysis is computational, not investigative. The whole computation lives behind one direct `docker run` invocation against `nvcr.io/nvidian/iva/tao-toolkit-ds:aoi`. The container's entrypoint takes `<category> <action> <args>`; we pass `gap_analysis vcn_aoi …`. (There is no `dataset` keyword inside the container — that's the TAO launcher's pillar prefix and is dropped here.) You do **not** need subagents, multi-phase image audits, or component-type clustering — VCN does not expose those dimensions. View only a small set of representative weak samples to qualify the gaps after the container returns.

---

## Inputs

1. **Experiment result directory** — contains `inference/inference.csv` from TAO VCN Classify inference. Required columns: `input_path`, `object_name`, `label`, `siamese_score`.
2. **Training code/config directory** — contains the VCN train YAML. The container reads `dataset.classify.input_map` (lighting condition list) and `dataset.classify.image_ext` from it to expand each weak sample into one row per lighting.
3. **Dataset directory** — image root prepended to the relative `input_path` from each row (`kpi_media_path`).
4. **Target KPI** — minimum NO_PASS-class recall the chosen threshold must satisfy (`min_recall`). Default to `1.0` (zero-miss) if the user does not specify.
5. **`top_k_per_label`** — augmentation budget per label. **Required.** Pass a positive integer (default 50). Omitting this argument flips the container into "below-threshold filter" mode, which at `min_recall=1.0` returns only PASS misclassifications and zero NO_PASS rows. See Common pitfalls.

---

## Setup

The threshold sweep, weakness ranking, and per-lighting expansion all run inside `nvcr.io/nvidian/iva/tao-toolkit-ds:aoi`. Confirm Docker, the NVIDIA container toolkit, and a GPU are present, then ensure the image is cached:

```bash
docker info > /dev/null && echo "OK: docker"
nvidia-smi > /dev/null && echo "OK: GPU"
docker image inspect nvcr.io/nvidian/iva/tao-toolkit-ds:aoi > /dev/null \
  || docker pull nvcr.io/nvidian/iva/tao-toolkit-ds:aoi
```

A GPU is required (the same image is used across the AOI loop and other actions assume CUDA is present). Aborting early on a GPU-less host saves a confusing late error.

**Path mounting.** Every host path the container reads or writes — `inference.csv`, the train YAML, the dataset image root, and the output dir — must be bind-mounted. The simplest pattern is to mount the workspace root with **identical paths** inside and outside the container so absolute paths in args resolve the same on both sides:

```bash
WORKSPACE=<absolute path that contains inference.csv, train YAML, dataset images, and the output dir>
DOCKER="docker run --gpus all --rm --ipc=host -v $WORKSPACE:$WORKSPACE -w $WORKSPACE nvcr.io/nvidian/iva/tao-toolkit-ds:aoi"
```

If `inference.csv`, the train YAML, and the dataset images live in different roots, pass multiple `-v` flags — but every absolute path you pass in args must resolve inside the container.

---

## Method

The whole skill is a single `docker run` invocation followed by a small visual spot-check. The container does Steps 1–4 internally (threshold sweep, weakness scoring, top-K selection, per-lighting expansion). You handle Step 5 (visual spot-check) directly with the Read tool.

### Step 1–4 — Run the container

```bash
$DOCKER gap_analysis vcn_aoi \
    inference_csv=<exp_dir>/inference/inference.csv \
    train_config=<exp_dir>/train.yaml \
    kpi_media_path=<dataset_root> \
    min_recall=<float> \
    top_k_per_label=<positive int — e.g. 50> \
    output_dir=<rca_results_dir>
```

> **Always pass `top_k_per_label`.** This is the argument that switches the container
> from the default "samples below threshold" filter into proper top-K-per-label
> ranking. At `min_recall=1.0` the threshold is by construction at-or-below every
> NO_PASS score, so the below-threshold filter returns ONLY misclassified PASS rows
> and zero NO_PASS rows — useless as an augmentation queue. With `top_k_per_label`
> set to a positive integer, the container computes signed weakness against the
> threshold for every row and surfaces the K weakest **per ground-truth label**,
> which is the per-label ranked output downstream steps consume.

Reads `inference.csv`, sweeps every unique `siamese_score` plus one value just below the minimum, keeps the candidates with NO_PASS-class recall ≥ `min_recall` (with `1e-12` tolerance), then picks the threshold with the best F1 (tie-break: precision, then threshold value). For every row, computes signed weakness from that threshold (positive = misclassified, negative = correct, magnitude = margin). Sorts by weakness descending and takes the top `top_k_per_label` per ground-truth label, then expands each weak row into one row per lighting condition using `dataset.classify.input_map` and `dataset.classify.image_ext` from the train YAML.

If **no** candidate threshold meets the recall target, the container exits non-zero and writes `unreachable_kpi.txt` into `output_dir` explaining which recall the model can actually achieve. In that case, stop the analysis after the docker call, write a one-section report explaining the model fundamentally cannot reach the KPI at any operating point, and recommend retraining or relabeling — skip the visual spot-check.

**Container writes into `output_dir`:**

| Artifact | Contents |
|----------|----------|
| `gaps.parquet` | Top-K weakest per label, expanded per lighting. Columns: `filepath`, `label`, `siamese_score`, `weakness`. |
| `threshold.txt` | Chosen decision threshold (single float, plain text). |
| `metrics.json` | At the chosen threshold: `precision`, `recall`, `f1`, confusion matrix `{tp, fp, tn, fn}`, plus per-label `{total, mean_weakness, median_weakness, max_weakness, n_misclassified}`. |
| `weak_samples_breakdown.txt` | Per-label kept-row breakdown: `<count>` total, `<%>` of all kept rows, `N` misclassified (weakness > 0), `N` marginal (weakness ≤ 0). |
| `unreachable_kpi.txt` | Only written when the recall target is unreachable. Presence of this file means: skip Step 5, write the abridged report, recommend retrain. |

Print the container's stdout summary (chosen threshold, kept-row counts, per-label breakdown) to your own stdout so the script-check hook can verify the run produced output.

> **About output ownership:** the container may run as root, leaving the artifacts owned by `root` on the host. After the docker call, run `sudo chown -R $(id -u):$(id -g) "$OUT"` if subsequent host-side steps (the visual spot-check copies, the report write) need to modify the directory.

### Step 5 — Visual spot check (small, fixed)

Skip this step if `unreachable_kpi.txt` exists in `output_dir` — there is nothing meaningful to spot-check when the model can't reach the KPI at any threshold.

Otherwise, use the Read tool to **view** the test images for:

- The 5 weakest PASS samples (the top of the "PASS misclassified as NO_PASS" pile) — pick by sorting `gaps.parquet` rows where `label == 'PASS'` by `weakness` descending.
- The 5 weakest NO_PASS samples (the top of the "NO_PASS misclassified as PASS" pile) — same, with `label != 'PASS'`.

`gaps.parquet` is already expanded per-lighting (multiple rows per sample). For the spot check, deduplicate to one row per (input_path, object_name) — pick the row whose `filepath` uses the FIRST lighting from the train YAML (one image per sample is enough — VCN's classify head sees all lightings stacked, but for human spot-check one is representative).

Classify each viewed sample as exactly one of:
- **mislabeled** — visual content disagrees with the CSV label
- **edge case** — genuinely ambiguous boundary case
- **data quality** — corrupted, dark, wrong crop, bad framing
- **systematic** — model has learned the wrong feature (the image looks "obviously PASS/NO_PASS" but the model disagrees)

Copy each viewed image (resized to 128×128 if PIL is available, otherwise just copy) into `<output_dir>/rca_images/` so it can be embedded inline in the report.

This is the **only** image inspection required. Do not view dozens of images, do not run failure mode clustering, do not audit goldens — VCN does not have golden images.

---

## Reference invocation

Paste-and-edit the workspace, the four paths, and the two numeric knobs; this runs end-to-end. Capture stdout so the script-check hook sees row counts.

```bash
WORKSPACE=<absolute path>            # mounted identically inside the container
EXP_DIR=<experiment_result_dir>      # contains inference/inference.csv and train.yaml; must be inside $WORKSPACE
DATASET_ROOT=<dataset_root>          # image root for inference.csv input_path entries; must be inside $WORKSPACE
MIN_RECALL=1.0                       # zero-miss default; lower if KPI relaxes
TOP_K=50                             # per-label augmentation budget
OUT="$EXP_DIR/rca_results/$(date +%Y-%m-%d_%H%M%S)"
IMG=nvcr.io/nvidian/iva/tao-toolkit-ds:aoi

mkdir -p "$OUT"

docker run --gpus all --rm --ipc=host \
    -v "$WORKSPACE:$WORKSPACE" -w "$WORKSPACE" \
    "$IMG" gap_analysis vcn_aoi \
    inference_csv="$EXP_DIR/inference/inference.csv" \
    train_config="$EXP_DIR/train.yaml" \
    kpi_media_path="$DATASET_ROOT" \
    min_recall="$MIN_RECALL" \
    top_k_per_label="$TOP_K" \
    output_dir="$OUT"

# Reclaim ownership in case the container ran as root
[ "$(stat -c %u "$OUT" 2>/dev/null)" = "0" ] && sudo chown -R "$(id -u):$(id -g)" "$OUT"

# Sanity print so the script-check hook sees real numbers
python3 - "$OUT" << 'PYEOF'
import json, os, sys
out = sys.argv[1]
unreachable = os.path.join(out, "unreachable_kpi.txt")
if os.path.isfile(unreachable):
    print("KPI UNREACHABLE — see", unreachable)
    sys.exit(0)
with open(os.path.join(out, "threshold.txt")) as f:
    print("threshold:", f.read().strip())
with open(os.path.join(out, "metrics.json")) as f:
    m = json.load(f)
print(f"precision={m['precision']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f}")
import pandas as pd
df = pd.read_parquet(os.path.join(out, "gaps.parquet"))
print(f"gaps.parquet: rows={len(df)}, cols={list(df.columns)}")
print(df['label'].value_counts())
PYEOF
```

---

## Outputs

Write everything into a timestamped folder under the experiment result directory. The container's outputs go straight there; the visual spot-check writes `rca_images/`; the packaging hook will add `rca_config/` and `claude_session.jsonl` automatically when `RCA_Report.md` is written.

```
<experiment_result_dir>/rca_results/YYYY-MM-DD_HHMMSS/
├── RCA_Report.md              # Full gap analysis report (you write this)
├── gaps.parquet               # Container: top-K weakest per label, expanded per lighting
├── threshold.txt              # Container: chosen decision threshold (single float)
├── metrics.json               # Container: confusion matrix + per-label distribution stats
├── weak_samples_breakdown.txt # Container: per-label count/misclassified/marginal counts
├── unreachable_kpi.txt        # Container: ONLY when no threshold meets min_recall
├── rca_images/                # You: thumbnails of the 10 viewed weak samples
├── rca_config/                # Auto-copied by hook
└── claude_session.jsonl       # Auto-copied by hook
```

At the start of the run, get the real timestamp by running `date +%Y-%m-%d_%H%M%S` in Bash. Do NOT hardcode or guess. If the user specifies a custom output path, use that instead but maintain the same internal structure.

---

## Common pitfalls

- **Forgetting `top_k_per_label` when `min_recall=1.0`** — the most consequential failure mode of this skill. At `min_recall=1.0` the chosen threshold sits at or below every NO_PASS sample's score (so recall=100% by construction means there are NO false negatives). Without `top_k_per_label`, the container falls back to a "samples below threshold" filter, which at this threshold matches ONLY misclassified PASS rows (false positives) — `gaps.parquet` ends up containing zero NO_PASS rows and the augmentation queue is broken. **Always pass an explicit positive `top_k_per_label`** (default 50) so the container ranks by signed weakness and returns the K weakest *per label*.
- **Image not pulled / wrong tag** — `docker pull nvcr.io/nvidian/iva/tao-toolkit-ds:aoi` before the run. The `:aoi` tag is required; the generic `:latest` does not contain the AOI gap-analysis entrypoint, and the docker run will fail with `gap_analysis: action not found` or similar.
- **Path-mount mismatch** — every absolute path passed in args (`inference_csv`, `train_config`, `kpi_media_path`, `output_dir`) must resolve inside the container. Use `-v $WORKSPACE:$WORKSPACE` so host and container paths match exactly. If you mount under a different in-container root, pass the in-container path in the args.
- **Output files owned by root** — the container runs as root by default, so `gaps.parquet`, `threshold.txt`, etc. are written as `root` on the host. The visual spot-check copies files into `rca_images/` and writes `RCA_Report.md` from the host — these will fail with `Permission denied` unless you `sudo chown -R $(id -u):$(id -g) "$OUT"` after the docker call.
- **`unreachable_kpi.txt` written** — the model fundamentally cannot reach the requested NO_PASS recall at any threshold. Do NOT proceed to the visual spot-check; write the abridged report and recommend retrain or relabeling.
- **`inference.csv` missing required columns** — container fails fast with a column-name error. Required: `input_path`, `object_name`, `label`, `siamese_score`. Re-run TAO VCN Classify inference if columns are absent.
- **Train YAML missing `dataset.classify.input_map` or `image_ext`** — per-lighting expansion fails. Confirm the train YAML actually came from the matching VCN Classify experiment.
- **`kpi_media_path` doesn't match `input_path` prefixes** — `gaps.parquet` ships with non-existent filepaths. Sanity-check a few rows on disk after the docker call returns and before the visual spot-check.
- **No GPU detected from inside the container** — confirm `nvidia-smi` works on the host AND that `--gpus all` was passed to `docker run`. Without it, the container errors late.

---

## Report Structure

Keep the report tight (1000–1800 words). This is a computational gap analysis, not a deep RCA — depth comes from accurate numbers and a clear action list, not narrative.

```
# VCN Gap Analysis Report: <Experiment Name>

## 1. Verdict
- Chosen threshold: <value>  (achieves precision=<p>, recall=<r>, F1=<f1> on NO_PASS at recall ≥ <KPI>)
- KPI reachability: <yes/no — and the recall it actually achieves>
- Total samples: <N>  |  Total weak samples kept: <K>  |  Misclassified: <M>
- Top-3 labels by misclassification share
- One-line headline: "<K> weak samples written to gaps.parquet for augmentation"

## 2. Threshold Selection
- Target NO_PASS recall: <KPI>
- Candidates evaluated: <count>; candidates meeting recall target: <count>
- Chosen threshold and tie-break reasoning (best F1 → precision → threshold)
- Confusion matrix at chosen threshold (from `metrics.json`):

| | Predicted NO_PASS | Predicted PASS |
|--|--|--|
| Actual NO_PASS | TP=… | FN=… |
| Actual PASS    | FP=… | TN=… |

## 3. Weakness Distribution
| Label | Total Samples | Mean Weakness | Median Weakness | Max Weakness | # Misclassified |
|-------|---------------|----------------|------------------|---------------|------------------|

(One row per ground-truth label across the FULL inference CSV — read directly from
`metrics.json` per-label stats — not just the kept K.)

## 4. Top-K Weakest Samples (per label)
| Label | object_name | input_path | siamese_score | weakness | misclassified? |
|-------|-------------|-------------|----------------|-----------|-----------------|

(Up to top_k_per_label rows per label group. Sorted by weakness descending within each group.
Read from gaps.parquet, deduplicated to one row per (input_path, object_name) — gaps.parquet
is per-lighting, but the table is per-sample.)

## 5. Visual Spot Check (10 samples)
| Label | object_name | siamese_score | weakness | Test Image | Verdict |
|-------|-------------|----------------|-----------|-------------|----------|

(5 weakest PASS + 5 weakest NO_PASS. `Test Image` column is `![](rca_images/<filename>)`. `Verdict` is one of: mislabeled / edge case / data quality / systematic.)

## 6. Per-Label Breakdown
(Render the contents of `weak_samples_breakdown.txt` here.)

## 7. Recommended Actions
1. **Relabel** — list every sample tagged `mislabeled` in section 5. Path is `{input_path}/{object_name}` in `inference.csv`.
2. **Augment** — `gaps.parquet` (`<K> rows × <L> lightings = <K*L> filepaths`) is the augmentation queue. Pass it to `deft-aoi-routing-vcn` next.
3. **Threshold action** — recommend whether to (a) retrain with current data and re-run this skill, (b) lower the recall target if the visual spot check shows the misclassified samples are genuinely ambiguous, or (c) ship at the current threshold if KPI is met.
4. **Systematic failures** — if any visual spot-check sample is tagged `systematic`, flag the failure mode (which lighting? which component family?) for model architecture review.
```

When `unreachable_kpi.txt` exists, replace sections 3–6 with a single short section quoting that file's contents and stating the model cannot meet the KPI at any threshold. Section 7 then collapses to one recommendation: retrain or relabel.

---

## Execution Order

1. Run `docker info`, `nvidia-smi`, and `docker image inspect nvcr.io/nvidian/iva/tao-toolkit-ds:aoi` (pulling if missing) once to confirm the environment. Abort with a clear message if any fail.
2. Run `date +%Y-%m-%d_%H%M%S` to get the timestamp; create `<experiment_result_dir>/rca_results/<timestamp>/`.
3. Run `docker run … nvcr.io/nvidian/iva/tao-toolkit-ds:aoi gap_analysis vcn_aoi …` with the inputs above. The container writes `gaps.parquet`, `threshold.txt`, `metrics.json`, `weak_samples_breakdown.txt` into `output_dir`. Print the chosen threshold and kept-row counts to stdout so the script-check hook can verify the run produced output. If the output dir is owned by root, `sudo chown -R $(id -u):$(id -g)` it.
4. If `unreachable_kpi.txt` exists, skip Step 5 and write the abridged report. Otherwise continue.
5. Pick 10 weak samples (5 weakest PASS + 5 weakest NO_PASS) from `gaps.parquet`, view each test image with Read, classify, and copy each into `rca_images/`.
6. Write `RCA_Report.md` last — writing it triggers the packaging hook, which copies session logs and skill config alongside.
