---
name: tao-mine-aoi-images
description: Runs the DEFT embed-then-mine workflow for VCN AOI iterations — embeds the gap-analysis target parquet, embeds
  a source pool, and mines nearest-neighbour source images for downstream augmentation. Use as the immediate next step after
  `tao-route-visual-changenet-samples` when expanding a real-image augmentation queue from the mining subset.
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit and a CUDA GPU. Pulls the `tao_toolkit.data_services` image declared in `versions.yaml` at the skill bank root.
metadata:
  author: NVIDIA Corporation
  version: '0.2'
allowed-tools: Read Bash
tags:
- data
- mining
- embedding
- vcn
- aoi
- sda
---

# DEFT Mining and Embedding Skill

You are the operator of the DEFT embed-then-mine workflow for VCN AOI. Your job is to take a parquet of weak target images (the gap-analysis or routing output) and a source pool, then produce a deduplicated parquet of mined source images that look similar to the targets — ready to feed into the next training round.

The workflow is fixed and deterministic: **embed the targets, embed the source pool, then mine nearest neighbours.** Each step's output parquet is the next step's input. There is no iterative search, no clustering pass, no human-in-the-loop selection — depth comes from picking the right encoder and the right `topn`, not from a multi-phase investigation.

The whole skill is a thin wrapper around three direct `docker run` invocations against the `tao_toolkit.data_services` image declared in `versions.yaml` (resolved at runtime — see Setup). The container's entrypoint takes `<category> <action> -e <spec.yaml> [hydra overrides...]` — pass `embedding image_embeddings -e <embedding_spec.yaml> …` for embedding and `tmm nearest_neighbors -e <mining_spec.yaml> …` for mining. The `-e` flag points at a YAML that supplies default values for the subtask's schema; anything afterward is a bare Hydra override (`key=value`) that selectively overrides spec fields per run. (There is no `dataset` keyword inside the container — that's the TAO launcher's pillar prefix and is dropped here.) Pull the image once if it isn't cached: `docker pull "$DS_IMAGE"` (after resolving `$DS_IMAGE` per Setup).

Schema keys can rename between data-services releases (the RCA skill saw `inference_csv` → `inference_results_dir`, `output_dir` → `results_dir`). When in doubt, introspect the actual schema once per image: `docker run --rm "$DS_IMAGE" embedding image_embeddings --cfg=job` and `... tmm nearest_neighbors --cfg=job`.

---

## Inputs

1. **Target parquet** — the gap-analysis output, typically `mining_gaps.parquet` from `tao-route-visual-changenet-samples` (or `gaps.parquet` from `tao-analyze-gaps-visual-changenet` if routing was skipped). Required column: `filepath`. If `label` is also present, label-aware filtering during mining is available; otherwise the mining task silently no-ops the filter.
2. **Source pool** — a parquet of candidate images to mine against, with a `filepath` column. If the user only has a CSV, convert it to a parquet **with the same columns** before Step 2. For label-aware filtering, the pool must also carry a `label` column.
3. **Embedding spec file** — a YAML containing `model`, `model_path`, `batch_size`, and (only when `model_path` is a TAO `.pth`/`.ckpt`) `model_config_path`. Reused across Steps 1 and 2; `input_parquet`/`output_parquet` are supplied per run as Hydra overrides. The **same** spec MUST drive both embedding steps — embeddings from different encoders are not comparable, and mismatched encoders are the most common cause of "the mined images look unrelated" reports.
4. **Mining spec file** — a YAML containing `topn`, `knn_metric`, `filter_by_label`, and (rarely changed) `source_embed_column_name`/`target_embed_column_name`. `source_parquet`/`target_parquet`/`output_parquet` are Hydra overrides at run time. SigLIP and CLIP embeddings should use `knn_metric: cosine`. When `filter_by_label: true` but either embedding parquet lacks a `label` column, the container logs a warning and proceeds **without** filtering.

---

## Setup

The mining and embedding tasks live inside the `tao_toolkit.data_services` image declared in `versions.yaml`. Resolve the concrete URI once at the top of the run, then confirm Docker, the NVIDIA container toolkit, and a GPU are present before doing anything else:

```bash
# Resolve tao_toolkit.data_services → concrete nvcr.io/... URI from versions.yaml
DS_IMAGE=$(python3 -c "import yaml,os; print(yaml.safe_load(open(os.environ['TAO_SKILL_BANK_PATH']+'/versions.yaml'))['images']['tao_toolkit']['data_services'])")
echo "DS_IMAGE=$DS_IMAGE"

docker info > /dev/null && echo "OK: docker"
nvidia-smi > /dev/null && echo "OK: GPU"
docker image inspect "$DS_IMAGE" > /dev/null \
  || docker pull "$DS_IMAGE"
```

`TAO_SKILL_BANK_PATH` is exported by the plugin's `session_start` hook. If it is unset (e.g. running outside the Claude Code plugin), point it at the skill-bank repo root before resolving.

A GPU is required for both the encoder forward pass and the cuML/cuDF k-NN search; both steps will fail without CUDA.

**Path mounting.** Every host path the container reads or writes — input parquets, output dirs, and the source-pool image root — must be bind-mounted. The simplest and most predictable approach is to mount the workspace root with **identical paths** inside and outside the container so the absolute paths in the parquet args resolve the same way on both sides:

```bash
WORKSPACE=<absolute path that contains all parquets, outputs, and the source-pool images>
DOCKER="docker run --gpus all --rm --ipc=host --user $(id -u):$(id -g) -v $WORKSPACE:$WORKSPACE -w $WORKSPACE $DS_IMAGE"
```

Reuse `$DOCKER` for the three invocations below.

If the source pool is provided only as a CSV, convert it to a parquet up front:

```python
import pandas as pd
pd.read_csv(source_pool_csv).to_parquet(source_pool_parquet, index=False)
```

The conversion must preserve the `filepath` column verbatim (and `label` if present). Do not add a path prefix — the container reads input parquets as-is, and the `$WORKSPACE` mount keeps host and container paths identical.

**Author the two spec files once per iteration.** Both files live under `$WORKSPACE` so the `-e` argument resolves on both sides of the mount. Per-run values stay out of the spec and are passed as Hydra overrides at invocation time.

```bash
cat > "$WORKSPACE/embedding_spec.yaml" <<'EOF'
model: SigLIP                                # CLIP, SigLIP, or a TAO checkpoint
model_path: google/siglip-base-patch16-224   # HF id, local HF dir, or .pth/.ckpt
# model_config_path: <train_spec.yaml>       # required only when model_path is a TAO checkpoint
batch_size: 64
EOF

cat > "$WORKSPACE/mining_spec.yaml" <<'EOF'
topn: 5
knn_metric: cosine                           # cosine for SigLIP/CLIP; euclidean/manhattan otherwise
filter_by_label: "false"                     # quoted — the schema reads it as a string
EOF
```

Any field in either spec can still be overridden inline at the CLI (e.g. `topn=10`) — Hydra applies CLI overrides on top of the spec.

---

## Method

Three commands, in order. Each command's output parquet is the next command's input. Run them as plain Bash; the `$DOCKER` alias from the Setup section handles the container, GPU, and mounts. Every invocation follows the same shape: `-e <spec>` for the baked-in defaults, then a handful of Hydra overrides for the run-specific paths.

### Step 1 — Embed the target images

```bash
$DOCKER embedding image_embeddings \
    -e <embedding_spec.yaml> \
    input_parquet=<target_parquet> \
    output_parquet=<target_embeddings_parquet>
```

Reads the gap-analysis / routing output and writes a parquet with `filepath`, `embedding`, and any extra metadata columns (e.g. `label`, `siamese_score`, `weakness`) carried forward verbatim from the input. Print the output schema (`pd.read_parquet(...).columns`) to stdout so the script-check hook can confirm the embedding column exists.

If you need to override `model` / `model_path` / `batch_size` for one run without editing the spec, append them as Hydra overrides (e.g. `model_path=...`).

### Step 2 — Embed the source pool

```bash
$DOCKER embedding image_embeddings \
    -e <embedding_spec.yaml> \
    input_parquet=<source_pool_parquet> \
    output_parquet=<source_embeddings_parquet>
```

Same command shape as Step 1, applied to the source pool. Use the **identical** `embedding_spec.yaml` as Step 1, and do not override `model` / `model_path` / `batch_size` differently here — mismatched encoder configs across the two steps produce non-comparable embeddings.

### Step 3 — Mine nearest neighbours

```bash
$DOCKER tmm nearest_neighbors \
    -e <mining_spec.yaml> \
    source_parquet=<source_embeddings_parquet> \
    target_parquet=<target_embeddings_parquet> \
    output_parquet=<mined_parquet>
```

For each target embedding, finds the `topn` closest source embeddings under the chosen metric, deduplicates across targets, and writes a single-column (`filepath`) parquet of unique mined source paths. The container also drops a `mining_summary.txt` next to the output parquet with: query count, neighbour count, duplicates removed, and (when label filtering is on) kept-vs-dropped pair counts. Tweak `topn`, `knn_metric`, or `filter_by_label` via inline Hydra override when sweeping (e.g. `topn=10`) — no need to rewrite the spec.

When `filter_by_label=true` but one of the embedding parquets is missing the `label` column, the container logs a warning and proceeds without filtering. If the mined output looks larger than expected or contains cross-label pairs, scan the docker log for that warning before assuming the task did the right thing.

---

## Reference invocation

This is the minimal end-to-end recipe — paste-and-edit the workspace, the three parquet paths, and the encoder, and it runs. Run as a single Bash block so the script-check hook sees one streamed log.

```bash
WORKSPACE=<absolute path>           # mounted identically inside the container
TARGETS=<target_parquet>            # e.g. .../routing_results/<ts>/mining_gaps.parquet
SOURCE_POOL=<source_pool_parquet>   # parquet with `filepath` (and optional `label`)
OUT="$WORKSPACE/mining_results/$(date +%Y-%m-%d_%H%M%S)"
EMBED_SPEC="$OUT/embedding_spec.yaml"
MINE_SPEC="$OUT/mining_spec.yaml"
MODEL=SigLIP                        # or CLIP, or a TAO checkpoint name
MODEL_PATH=google/siglip-base-patch16-224  # or a local checkpoint path
TOPN=5
METRIC=cosine
FILTER_BY_LABEL=false
IMG=$(python3 -c "import yaml,os; print(yaml.safe_load(open(os.environ['TAO_SKILL_BANK_PATH']+'/versions.yaml'))['images']['tao_toolkit']['data_services'])")

mkdir -p "$OUT"

# Write the two spec files for this iteration
cat > "$EMBED_SPEC" <<EOF
model: $MODEL
model_path: $MODEL_PATH
batch_size: 64
EOF

cat > "$MINE_SPEC" <<EOF
topn: $TOPN
knn_metric: $METRIC
filter_by_label: "$FILTER_BY_LABEL"
EOF

# Step 1: embed targets
docker run --gpus all --rm --ipc=host \
    --user "$(id -u):$(id -g)" \
    -v "$WORKSPACE:$WORKSPACE" -w "$WORKSPACE" \
    "$IMG" embedding image_embeddings \
    -e "$EMBED_SPEC" \
    input_parquet="$TARGETS" \
    output_parquet="$OUT/target_embeddings.parquet"

# Step 2: embed source pool (SAME embedding spec as Step 1)
docker run --gpus all --rm --ipc=host \
    --user "$(id -u):$(id -g)" \
    -v "$WORKSPACE:$WORKSPACE" -w "$WORKSPACE" \
    "$IMG" embedding image_embeddings \
    -e "$EMBED_SPEC" \
    input_parquet="$SOURCE_POOL" \
    output_parquet="$OUT/source_embeddings.parquet"

# Step 3: mine nearest neighbours
docker run --gpus all --rm --ipc=host \
    --user "$(id -u):$(id -g)" \
    -v "$WORKSPACE:$WORKSPACE" -w "$WORKSPACE" \
    "$IMG" tmm nearest_neighbors \
    -e "$MINE_SPEC" \
    source_parquet="$OUT/source_embeddings.parquet" \
    target_parquet="$OUT/target_embeddings.parquet" \
    output_parquet="$OUT/mined.parquet"

# Sanity print so the script-check hook sees row counts
python3 -c "
import pandas as pd
for name, p in [('target_embeddings', '$OUT/target_embeddings.parquet'),
                ('source_embeddings', '$OUT/source_embeddings.parquet'),
                ('mined',             '$OUT/mined.parquet')]:
    df = pd.read_parquet(p)
    print(f'{name}: rows={len(df)}, cols={list(df.columns)}')
"
```

Print the row counts and column lists at the end so the script-check hook can verify each step actually produced output.

---

## Outputs

Write everything into a timestamped folder under the experiment / iteration directory. The packaging hook will add `mining_config/` and `claude_session.jsonl` automatically when `Mining_Report.md` is written.

```
<output_dir>/mining_results/YYYY-MM-DD_HHMMSS/
├── Mining_Report.md            # Full mining report
├── embedding_spec.yaml         # The -e spec used for Steps 1 and 2
├── mining_spec.yaml            # The -e spec used for Step 3
├── target_embeddings.parquet   # Step 1 output (filepath, embedding, + carried metadata)
├── source_embeddings.parquet   # Step 2 output (filepath, embedding, + carried metadata)
├── mined.parquet               # Step 3 output — unique mined source filepaths
├── mining_summary.txt          # Auto-emitted next to mined.parquet by the container
├── mining_config/              # Auto-copied by hook
└── claude_session.jsonl        # Auto-copied by hook
```

At the start of the run, get the real timestamp by running `date +%Y-%m-%d_%H%M%S` in Bash. Do NOT hardcode or guess. If the user specifies a custom output path, use it directly but maintain the same internal layout.

The mined parquet is the artifact downstream training consumes. The two embedding parquets are intermediate but worth retaining: they are reusable across multiple mining runs against the same source pool, and they are the only place to look when a "looks unrelated" report needs encoder-level debugging.

---

## Common pitfalls

- **Mismatched encoders between target and source embeddings** — the single most common cause of garbage mining output. Both embedding steps must consume the **same** `embedding_spec.yaml`, and any Hydra override that changes `model` / `model_path` / `batch_size` must be applied to *both* invocations or to neither. The hook checks for this.
- **Skipping an embedding step** — the mining task requires both inputs to contain an embedding column; the raw filepath parquets cannot be fed to it directly.
- **Missing `label` column with `filter_by_label=true`** — the filter silently no-ops with a warning rather than erroring. If the mined output looks too large or contains cross-label pairs, grep the docker log for the warning and confirm both embedding parquets carry `label`.
- **Spec file outside `$WORKSPACE`** — `-e <path>` is resolved inside the container, so the spec must live under the bind-mounted workspace. Place `embedding_spec.yaml` and `mining_spec.yaml` next to the other run artifacts and pass absolute paths.
- **Spec file with unresolved `???` sentinels** — the bundled defaults under `experiment_specs/` mark required fields with `???`. Replace every `???` (e.g. `model`, `model_path`) before the run, or supply that field as a Hydra override on the CLI. Hydra rejects unresolved sentinels with a clear `MissingMandatoryValue` error.
- **TAO checkpoint without `model_config_path`** — when `model_path` points at a TAO `.pth` / `.ckpt`, the entrypoint cannot reconstruct the encoder without the matching train-spec YAML. Add `model_config_path: <spec.yaml>` to `embedding_spec.yaml` (it'll apply to both embedding steps).
- **Source pool provided as CSV** — convert to parquet **before** Step 2; the entrypoint only reads parquet. The conversion must preserve `filepath` (and `label` if present).
- **Path resolution mismatch between host and container** — every parquet path passed in args must be readable inside the container. The simplest fix is the `-v $WORKSPACE:$WORKSPACE` pattern from Setup so paths resolve identically on both sides. If you mount `<host>:<other-path>`, pass the in-container path in the args, not the host one.
- **No GPU available** — both steps need CUDA. Check `nvidia-smi` once at the top; the entrypoint's error is clear but it surfaces late in a long run.
- **Image not pulled / wrong tag** — resolve `tao_toolkit.data_services` from `versions.yaml` and `docker pull "$DS_IMAGE"` before the run. The data-services tag declared there is required; the generic `:latest` tag does not contain the AOI-specific embedding/mining entrypoints.
- **`topn` × N_targets ≫ source size** — the dedup pass will run out of unique source images and the mined parquet will be much smaller than `topn × N_targets`. This is expected, not a bug; report the actual mined count, not the requested one.

---

## Report Structure

Keep the report tight (600–1200 words). Mining is a deterministic pipeline; the value is making the encoder choice, the row counts, and any silent filter no-ops auditable — not narrative.

```
# Mining Report: <Iteration / Experiment Name>

## 1. Verdict
- Targets in: <N_targets> rows from `<target_parquet>`
- Source pool in: <N_source> rows from `<source_pool_parquet>`
- Mined out: <N_mined> unique source filepaths → `mined.parquet`
- Encoder: <model> @ <model_path>
- Mining params: topn=<topn>, knn_metric=<metric>, filter_by_label=<bool>
- One-line headline: "<N_mined> source images mined for <N_targets> targets, ready for the next training round."

## 2. Inputs
| Input | Path | Rows | Has `label`? | Notes |
|-------|------|------|---------------|-------|
| target_parquet     | … | … | yes/no | source: `tao-route-visual-changenet-samples` mining subset |
| source_pool_parquet | … | … | yes/no | converted from CSV? yes/no |

## 3. Encoder Consistency
- Step 1 model / model_path: …
- Step 2 model / model_path: …
- Match? <yes — required>
- (If a TAO checkpoint:) model_config_path: …

## 4. Mining Run
- Command: `docker run … "$DS_IMAGE" tmm nearest_neighbors …` (where `DS_IMAGE` = `tao_toolkit.data_services` from `versions.yaml`)
- topn=<topn>, knn_metric=<metric>, filter_by_label=<bool>
- Reported by `mining_summary.txt`:
  - queries: <N>
  - neighbours requested: <N × topn>
  - duplicates removed: <N>
  - kept pairs (label filter): <N or n/a>
  - dropped pairs (label filter): <N or n/a>
- Filter no-op warning in docker log? <yes/no — quote the line if yes>

## 5. Per-Label Breakdown (if `label` is present in target_parquet)
| Target Label | N_targets | N_mined source rows | Notes |
|--------------|-----------|----------------------|-------|

(One row per distinct target label. If the target parquet has no label column, write
"label column not present in target parquet — per-label breakdown skipped." and move on.)

## 6. Output Sanity
- mined.parquet schema: <columns>
- First 5 mined paths exist on disk? <yes/no — list any missing>
- Path-encoding sanity check: <pass/fail — see "Common pitfalls" if fail>

## 7. Recommended Actions
1. **Augment** — `mined.parquet` is the augmentation queue for the next training round.
   Concatenate it with the AnomalyGen SDG output (if any) before kicking off training.
2. **If `N_mined ≪ topn × N_targets`** — the source pool is exhausted; widen the pool
   or accept a smaller augmentation budget.
3. **If filter no-op fired** — backfill the missing `label` column on whichever embedding
   parquet lacked it, then re-run Step 3 only (Steps 1–2 do not need to repeat).
4. **If mined images "look unrelated"** — verify Steps 1 and 2 used the *same* `model` and
   `model_path`. The encoder consistency section above is the first thing to check.
```

---

## Execution Order

1. Resolve `DS_IMAGE` from `versions.yaml` (`images.tao_toolkit.data_services`), then run `docker info`, `nvidia-smi`, and `docker image inspect "$DS_IMAGE"` (pulling if missing) once to confirm the environment. Abort with a clear message if any fail.
2. Run `date +%Y-%m-%d_%H%M%S` to get the timestamp; create `<output_dir>/mining_results/<timestamp>/`.
3. Write `embedding_spec.yaml` and `mining_spec.yaml` into the timestamped dir, filling in the encoder choice and mining knobs. Keep these under `$WORKSPACE` so the `-e` path resolves inside the container.
4. If the source pool is a CSV, convert to parquet first (preserve `filepath` and `label`).
5. Run Step 1 (embed targets) via `docker run … embedding image_embeddings -e embedding_spec.yaml input_parquet=… output_parquet=…`. Print the output parquet's row count and columns to stdout.
6. Run Step 2 (embed source pool) with the **identical** `embedding_spec.yaml` as Step 1. Print output row count and columns.
7. Run Step 3 (mine nearest neighbours) via `docker run … tmm nearest_neighbors -e mining_spec.yaml source_parquet=… target_parquet=… output_parquet=…`. Confirm `mining_summary.txt` was written next to `mined.parquet`.
8. Compute the per-label breakdown (Section 5) by joining the target embeddings parquet with the mined output on filepath, if both carry `label`.
9. Write `Mining_Report.md` last — writing it triggers the packaging hook, which copies session logs and skill config alongside.
