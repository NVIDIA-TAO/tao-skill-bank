# DEFT OD — Source-Pool Prep (runs once, before baseline)

Prep turns a directory of raw unlabeled images into the two artifacts every iteration depends on:

| Artifact | Produced by | Consumed by |
|---|---|---|
| `source_pool/odvg/*_odvg.jsonl` + `*_odvg_labelmap.json` | Co-DETR pseudo-labeling (folding via `category_mapping`) → KITTI→COCO→ODVG | `stage` (annotation lookup by basename) |
| `source_pool/source_embeddings.parquet` | `embedding image_embeddings` over the pool | `mine` (the search corpus) |
| `source_pool/coco.json` | the KITTI→COCO step, retained | `mine` as `source_detection_file` (class_stratified only) |

**This runs exactly once, before the baseline.** Every iteration afterwards is a lookup against what prep produced — no labeler and no encoder runs inside the loop. That is the point: it keeps the per-iteration path deterministic and confines GPU work to train and inference.

**Prep is idempotent.** Skip any step whose output already exists. A user arriving with a pool that is already labeled and embedded pays nothing; a user with raw images pays for the whole chain once. Never re-label or re-embed a pool that already has current artifacts.

**Cost warning.** Labeling and embedding are proportional to *pool size*, not to what mining eventually selects. A large pool means a substantial one-time job that must finish before the baseline starts. Surface the pool image count and this expectation in the Pre-Flight Summary so it is not discovered mid-run.

## Target classes are a run-wide contract

The user supplies the classes they want to train on. That same list has to be consistent across four places, or the loop silently misbehaves:

| Where | Use |
|---|---|
| `classes.yaml` (this stage) | Folds Co-DETR's predicted vocabulary down to the target set |
| `weak_thresholds` in `gap_analysis` | Per-class AP50 gates — every target class listed explicitly |
| `rare_class_list` in `mine` | Which target classes drive class-stratified allocation |
| `kpi/mapping.yaml` in `kpi_analyze` | Class mapping for scoring |

Resolve the target class list once in Pre-Flight and derive all four from it. A class present in the KPI ground truth but absent from `classes.yaml` can never be pseudo-labeled, so mining will never find examples for it — the loop will look like it is working while being structurally unable to improve that class.

## Step 1 — Pseudo-label the pool with Co-DETR, folding as you go

Invoke `tao-skill-bank:tao-train-codetr` (read its `SKILL.md` first).

**The `codetr` console script is unregistered in every TAO PyTorch image checked so far** — `7.0.1-pyt` and the `2026.7.31-rc-12` nightly both answer `codetr: command not found` while `grounding_dino`, `dino`, and `rtdetr` are registered. The module ships, so this is a packaging gap, not a missing feature. Probe both forms and use whichever answers; only if *both* fail is Co-DETR genuinely absent:

```bash
docker run --rm "$TAO_PYT_IMAGE" codetr --help >/dev/null 2>&1 && CODETR="codetr"
[ -z "${CODETR:-}" ] && docker run --rm --entrypoint sh "$TAO_PYT_IMAGE" \
  -c 'python3 -c "import nvidia_tao_pytorch.cv.codetr"' >/dev/null 2>&1 \
  && CODETR="python3 -m nvidia_tao_pytorch.cv.codetr.entrypoint.codetr"
```

Both forms take identical arguments. Substitute `$CODETR` wherever this document writes `codetr`.

### The fold belongs here, not downstream

`inference.category_mapping` groups the detector's own classes into output categories **after the forward pass**, and it is the right place to fold COCO-80 down to the target set:

```yaml
inference:
  checkpoint: <co-detr checkpoint>
  conf_threshold: 0.3          # reference value; TAO defaults to 0.5
  category_mapping:            # emitted by scripts/build_fold_mapping.py
    bicycle:   ["bicycle", "motorcycle"]
    car:       ["car", "bus", "truck"]
    person:    ["person"]
    road_sign: ["road_sign", "traffic light", "stop sign"]
```

From `category_mapping.py`: unmapped originals are dropped, a name claimed by two groups keeps the first with a warning, names absent from the classmap are warned about and ignored, an empty remap raises, and output category IDs are assigned `0..K-1` **in the order the mapping is written**.

Then it runs `apply_category_mapping_groupnms` — per-output-category soft-NMS *after* the merge. That is the reason to fold here rather than by rewriting labels afterwards: one object detected as both `truck` and `car` becomes two boxes of the *same* class the moment those fold together, and only a post-fold NMS removes the duplicate. Renaming labels later cannot; it ships the duplicates into training.

```bash
docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" \
  -v "$WORKSPACE:$WORKSPACE" -w "$WORKSPACE" \
  "$TAO_PYT_IMAGE" \
  $CODETR inference -e "$CODETR_SPEC" \
    inference.checkpoint="$CODETR_CHECKPOINT" \
    dataset.infer_data_sources.image_dir="$POOL_IMAGES" \
    dataset.infer_data_sources.classmap="$CODETR_CLASSMAP" \
    results_dir="${PREP_DIR}" \
    inference.num_gpus="$NUM_GPUS"
```

`dataset.infer_data_sources.classmap` is the detector's **own** vocabulary — one class name per line in `category_id` order starting at 1, COCO-80 for a COCO-trained checkpoint. It is not the target list; `category_mapping` names are matched against it. `results_dir` auto-appends the action, so labels land in `${PREP_DIR}/inference/labels/` already carrying target class names.

`conf_threshold` is the main quality control on the pseudo-labels and is applied at write time.

Obtaining a checkpoint is outside this skill — see `tao-train-codetr`'s SKILL.md.

## Step 2 — Emit the two mappings

TAO does the folding; this step only writes the files that tell it how.

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/build_fold_mapping.py \
  --classes "$CLASSES_YAML" \
  --emit-codetr-category-mapping "${PREP_DIR}/codetr_category_mapping.yaml" \
  --emit-kitti-mapping           "${PREP_DIR}/kitti_mapping.yaml" \
  --emit-classmap                "${PREP_DIR}/classmap_target.txt"
```

Two accepted forms for `classes.yaml`. **Explicit folding** — several predicted classes collapse into one target:

```yaml
classes:
  car:     [car, truck, bus]
  person:  [person]
  bicycle: [bicycle]
```

This is the shipped ITS default (`references/example_classes_its.yaml`): trucks and buses count as `car`, because the target vocabulary has no separate heavy-vehicle class and dropping them would discard real road users the model must detect.

**Identity form** — when the target names already match the predicted names and you only want to drop the rest:

```yaml
classes: [car, person, bicycle]
```

**Confirm the fold rather than inventing one.** For a new dataset, whether two predicted classes belong together is a decision about what the metric should mean, so ask instead of guessing. Where a class sits ambiguously between targets — `motorcycle` is neither a car nor a bicycle — surface the choice, since folding it either way silently changes what the numbers report.

### Class names are matched exactly, including case

Both consumers do exact-string lookups — Co-DETR checks `orig_name not in name_to_id`, and `annotations convert` does `labels2cat.get(row_p[0])`. A source written `Car` against a detector that emits `car` matches nothing and every one of its boxes is dropped. Neither raises; Co-DETR logs a warning among many, and the converter says nothing at all.

Copy source names verbatim from the detector's classmap. `validate_pool_coco.py` (step 4) names case-only mismatches explicitly, because this is the most likely way a fold goes wrong and the least visible.

> **Improvement to make to `annotations convert`:** `construct_category_map` /
> `labels2cat` in `kitti_to_coco.py` should match class names case-insensitively, or
> at minimum warn when a label class differs from a mapping alias only by case. Today
> the reference mapping works around it by hand-enumerating variants — `Bicycle`,
> `bicycle`, `AutoMobile`, `Automobile` — which is fragile and silently incomplete for
> any spelling nobody thought of. The same function is imported by
> `analytics kpi_analyze`, so a fix there covers both.

### Target names must be single tokens

Targets become COCO categories, then ODVG labels, then the Grounding DINO caption list, and finally the class field of KITTI inference labels. KITTI is space-delimited, so a multi-word target makes those unparseable — the script rejects it. Source names may contain spaces; COCO has `traffic light`.

## Step 3 — KITTI → COCO

```yaml
# ${PREP_DIR}/kitti_to_coco.yaml
data:
  input_format: "KITTI"
  output_format: "COCO"
kitti:
  image_dir: <pool images>
  label_dir: ${PREP_DIR}/inference/labels     # Co-DETR's output, already folded
  mapping:   ${PREP_DIR}/kitti_mapping.yaml   # identity over the targets
  project:   coco                             # names the output file — see below
results_dir: <workspace>/source_pool          # NOT a scratch dir — see below
```

```bash
docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" -v "$WORKSPACE:$WORKSPACE" -w "$WORKSPACE" \
  "$TAO_DS_IMAGE" annotations convert -e "${PREP_DIR}/kitti_to_coco.yaml"
```

**The mapping is identity here, and still explicit.** Co-DETR already folded, so each target maps only to itself. Omitting `mapping` entirely would make the converter auto-derive one by scanning the label directory — but then COCO category IDs follow filesystem discovery order, and a target absent from this particular pool disappears from `categories`. Those IDs travel into the ODVG labelmap and then into training, so they have to be stable across runs.

**Every TAO task invocation needs `--gpus`, including the CPU-bound ones.** The launcher shells
out to `nvidia-smi -L` unconditionally before dispatching, so a container started without
`--gpus` dies with `FileNotFoundError: [Errno 2] No such file or directory: 'nvidia-smi'` —
`annotations convert` does no GPU work and still requires it. A bare `--help` is the exception;
it short-circuits before the probe, which is why the step 1 availability check works without one.

**`kitti.project` names the output file, and it is not optional here.** The converter writes `<results_dir>/<project>.json`, falling back to the second-to-last component of `kitti.image_dir` (`project = name or img_dir.split('/')[-2]`). Leave it unset and a pool at `.../my_pool/images` silently produces `source_pool/my_pool.json` — not the `source_pool/coco.json` that step 5, the `source_detection_file` gate, and `config.source_detection_file` all name.

**This COCO is not a throwaway.** It is step 5's input *and* the `source_detection_file` that `class_stratified` mining consumes on every iteration. Write it to `<workspace>/source_pool/coco.json` and record that path in state as `config.source_detection_file`.

**Empty label files must exist, not be omitted.** The converter calls `os.path.getsize()` on the label file for *every* image in `image_dir`, inside a multiprocessing worker, so a label file that is absent rather than empty raises `FileNotFoundError` and takes the whole conversion down with `Execution status: FAIL`. Co-DETR writes one file per image, so this holds by construction — it matters only if something prunes them in between.

Do not read that as "the image survives to training". It does not: this step skips images with no valid annotation (`kitti.no_skip` defaults to `False`) and step 5 skips zero-annotation images unconditionally, so an image whose boxes all dropped reaches neither `coco.json` nor the ODVG. It *does* stay in the embedding parquet, which covers the whole pool — the parquet/ODVG divergence `references/data-layout.md` describes.

## Step 4 — Verify the pool before anything trusts it

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/validate_pool_coco.py \
  --coco        "<workspace>/source_pool/coco.json" \
  --classes     "$CLASSES_YAML" \
  --labels-dir  "${PREP_DIR}/inference/labels" \
  --images-dir  "$POOL_IMAGES" \
  --report-json "${PREP_DIR}/pool_report.json"
```

Both TAO folds fail *quietly* when names do not line up: an unmatched name is a log warning, an unmapped detection is silently dropped, and the conversion still prints `Execution status: PASS` and exits 0 while writing a COCO with no annotations. Nothing downstream notices until training produces a model that cannot detect a class, several GPU-hours later. So verify the artifact.

| Check | Why it is a hard error |
|---|---|
| every target class carries annotations | The pool holds no examples of it. If it also appears in the KPI ground truth, gap analysis marks those images weak every iteration while mining cannot find anything — the loop runs its full course looking healthy and cannot improve it. |
| the COCO has any annotations at all | The classic mapping-shape failure: a string where a list belongs. |
| case-only class mismatches | `Car` vs `car` drops every box of that class, silently. |

`pool_report.json` also settles **`max_labels`** — one per target class. Grounding DINO caps
each training caption at that many class phrases: the classes present in the image plus
randomly sampled ones that are not, which is what teaches the model what a class *is not*. Any
larger value behaves identically, since there are no further negatives to sample; any smaller
one truncates them. Fixing it here means the train spec never carries a stale constant, which
is silently wrong the moment the target set changes size.

Reported as warnings, not errors: unmapped source classes with counts (a large `truck=1523` against an identity mapping is a missing fold), the pool-vs-COCO image shortfall, and an annotation count below the folded box count — the converter calls `drop_duplicates()` per image, so byte-identical KITTI lines collapse into one.

Pass `--allow-empty-classes` only when a class is listed defensively and its absence from this pool is expected.

## Step 5 — COCO → ODVG

```yaml
# ${PREP_DIR}/coco_to_odvg.yaml
data:
  input_format: "COCO"
  output_format: "ODVG"
coco:
  ann_file: <the COCO json written by step 3>
results_dir: <workspace>/source_pool/odvg
```

```bash
docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" -v "$WORKSPACE:$WORKSPACE" -w "$WORKSPACE" \
  "$TAO_DS_IMAGE" annotations convert -e "${PREP_DIR}/coco_to_odvg.yaml"
```

Output is named after the input: `<basename>_odvg.jsonl` and `<basename>_odvg_labelmap.json`. There is no direct KITTI → ODVG conversion, which is why steps 3 and 4 are separate.

Confirm the ODVG records key on `file_name` with the image **basename** — that is how `stage_mined_odvg.py` looks them up later.

## Step 6 — Embed the pool

First build the input list. Do not hand-write it:

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/build_pool_input_parquet.py \
  --images-dir "$POOL_IMAGES" \
  --out        "${PREP_DIR}/pool_input.parquet"
```

Then invoke `tao-skill-bank:tao-generate-image-embeddings` over it, writing
`source_pool/source_embeddings.parquet`, with the encoder resolved in Pre-Flight check 9.

**Embed the whole pool, not just what reached the COCO.** Step 3 skips images with no surviving
box — 35 of 5,000 on the reference pool — so `coco.json` is not the image list. Mining searches
the parquet, and an image absent from it can never be selected at all.

The `filepath` column must hold the **same absolute paths** the ODVG `file_name` basenames
resolve against. Mining selects by `filepath`; staging then looks up annotations by basename. If
those two disagree, mining succeeds and staging reports every image as missing an annotation.
The script resolves symlinks for this reason: a pool of links into an unmounted tree yields a
container that sees no images, and an error that blames the image list rather than the mount.

## Commit

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/commit_stage.py \
  --results-dir "${RESULTS_DIR}" --iter-label prep --stage prep \
  --pool-odvg "<workspace>/source_pool/odvg" \
  --pool-embeddings "<workspace>/source_pool/source_embeddings.parquet" \
  --pool-report "${PREP_DIR}/pool_report.json" \
  --summary "prep: labeled <N> pool images, <M> boxes kept across <K> classes"
```

## Gates

Hard-stop before the baseline when any of these hold:

- Co-DETR is unavailable in the resolved image. Probe **both** forms — the bare `codetr`
  console script *and* `python3 -m nvidia_tao_pytorch.cv.codetr.entrypoint.codetr` (step 1).
  A failing `codetr --help` on its own is not this gate: it fails in every image checked so
  far while the module works, so gating on it alone hard-stops every run.
- `validate_pool_coco.py` exits non-zero (step 4). It covers the three failures that are
  otherwise silent: a target class with no annotations, a COCO with no annotations at all,
  and a case-only class mismatch.
- The ODVG output is empty or absent.
- `class_stratified` is configured but `source_pool/coco.json` was not retained from step 3.
- The embedding parquet is empty, or its row count is wildly below the pool image count.

Read `pool_report.json`'s warnings before proceeding even when it exits zero — `dropped_by_class`
is where a missing fold shows up, and it is not an error because dropping classes is often
exactly what was intended.
