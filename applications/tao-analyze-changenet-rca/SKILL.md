---
name: tao-analyze-changenet-rca
description: Performs deep Root Cause Analysis (RCA) on NVIDIA TAO Visual ChangeNet classification experiments with
  image-evidence-driven investigation. Use when analyzing ChangeNet model failures, investigating poor recall / FAR / PASS-NO_PASS
  metrics, auditing visual inspection pipeline quality, or running an RCA report for an AOI defect-detection model.
  Trigger phrases include "RCA on my ChangeNet model", "why is my AOI model failing", "audit ChangeNet predictions",
  "investigate FAR regressions", "root cause analysis on visual-changenet".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit. Sub-skills declare additional requirements.
metadata:
  author: NVIDIA Corporation
  version: '0.1'
allowed-tools: Read Bash
tags:
- application
- rca
- changenet
---

# TAO ChangeNet Classification RCA Skill

You are an expert investigator for NVIDIA TAO Visual ChangeNet classification experiments. Your job is to find **why** the model fails, backed by **visual evidence from actual images**.

When the user provides an experiment result directory and training code directory, perform a deep Root Cause Analysis. The investigation must be **image-evidence-driven** — every major conclusion should trace back to specific images you viewed.

---

## Inputs

1. **Experiment result directory** — contains `train/` and `inference/`
2. **Training code directory** — the `visual_changenet/` source tree
3. **Dataset directory** — where CSV files and images reside (often in experiment.yaml)
4. **Target KPI** — default to **Recall-first** if not specified. Options: Recall-first (FAR at 100% recall), FAR-first (recall at target FAR), Balanced (F1), Custom.

---

## Visual Inspection Primer

The ChangeNet model compares a **test image** against a **golden image** (known-good reference) to detect differences. When viewing images, check these three things:

1. **Image quality**: Both images should be properly exposed with visible content. Watch for unusually dark images — but **do not use a fixed intensity threshold**. Some illumination types (e.g., SolderLight) produce systemically dark images where mean intensity < 30 is normal. Always establish a PASS golden baseline first and flag outliers relative to that baseline.
2. **Framing match**: Test and golden should show the same region at the same zoom and orientation. Mismatched framing (e.g., wide-field vs close-up) indicates a golden pipeline error.
3. **Defect visibility**: Can you see the difference between test and golden? Some defects are obvious at any resolution; others may be invisible after downscaling to the model's input size. Compare original image dimensions to model input size to assess information loss.

---

## Investigation Flow

The investigation has 5 phases. Phase 1 (numbers) gives you hypotheses. Phase 2 (images) proves or disproves them. Phase 3 (cross-dimensional) finds hidden patterns. Phase 4 (config) explains the mechanism. Phase 5 (counterfactual) quantifies fixes. **Phase 2 is the core — spend the most effort there. Phase 5 is the most actionable — never skip it.**

---

## Parallelization Strategy (USE SUBAGENTS)

**You MUST use the Agent tool to run independent investigation tracks in parallel.** This dramatically speeds up the RCA. Follow this execution plan:

### Step 1: Phase 1 — Run sequentially (everything depends on this)
Run Phase 1 yourself in the main thread. Save the results:
- Score statistics, tier, threshold sweep, per-defect-type table, drop-N analysis
- List of bottom 5 defects (for 2A), top 10 FP PASS samples (for 2C)
- All defect types found

### Step 2: Parallel Wave 1 — Launch 6 subagents simultaneously
After Phase 1 completes, launch ALL 6 agents **in a single message with multiple Agent tool calls**:

**Agent A — "Image Evidence: Critical Samples + Failure Clustering"**
- Phase 2A: Threshold-critical sample deep dive (bottom 5 defects, top 10 FPs)
- Phase 2B: Failure mode clustering (view ALL defect images, classify each)
- Provide: inference CSV path, image path construction rules, experiment.yaml path, Phase 1 results (bottom 5 defects list, score stats)

**Agent B — "Image Evidence: Golden Audit + FP Analysis"**
- Phase 2B: Systematic golden image audit (Python script + view flagged goldens)
- Phase 2C: False positive deep dive (top 10 highest-scoring PASS)
- Phase 2D: Comparative visual analysis
- Provide: inference CSV path, image path construction rules, top 10 FP sample IDs from Phase 1

**Agent C — "Data & Label Analysis"**
- Phase 2E: Label semantics & visual pattern alignment audit
- Phase 3C: Training image deep dive (view training defects, compare to test)
- Phase 4A: Data sufficiency analysis
- Provide: train CSV path, val CSV path, inference CSV path, image path construction rules

**Agent D — "Config & Cross-Dimensional Analysis"**
- Phase 3A: Component-type clustering
- Phase 3B: Board-level & positional analysis
- Phase 3D: Multi-light condition analysis
- Phase 4B: Training config audit
- Phase 4C: Training metrics
- Phase 4D: Loss function & decision boundary analysis
- Provide: inference CSV path, experiment.yaml path, status.json path

**Agent E — "Exploratory: Random Sampling & Anomaly Hunting"**
This agent has NO fixed checklist. Its job is to find what the structured agents miss.
- **Random image sampling**: Pick 20 random samples across the full score range (not just extremes). View test + golden for each. Look for anything unexpected — patterns not captured by the defect labels, images that "feel wrong" but aren't flagged, subtle systematic issues.
- **Score anomaly hunting**: Find statistical outliers — samples whose scores don't match their neighbors (e.g., a PASS sample with a score way above other PASS, or a defect with a suspiciously perfect score). View their images and explain the anomaly.
- **Golden-to-golden variance**: Pick 5 components that appear in multiple boards. View their golden images across boards. Are goldens consistent, or do they vary (= golden pipeline instability)?
- **Edge case search**: Find the samples closest to the decision boundary (scores near the optimal threshold). These are the model's hardest decisions. View them. What makes them ambiguous?
- **Correlation mining**: Run a Python script to compute correlations between score and every available metadata field (comp_type, object_name, board, position, image size, etc.). Report any unexpected strong correlations (r > 0.3).
- **Free-form observations**: Note anything surprising, unusual, or unexplained. No finding is too small — even "the naming convention changes after row 500" can be a clue.
- Provide: inference CSV path, train CSV path, image path construction rules, ALL file paths, Phase 1 results

**Agent F — "Exploratory: Cross-Validation & Stress Testing"**
This agent stress-tests the model's behavior and the data integrity.
- **Score consistency check**: If the same component appears on multiple test boards, does it get consistent scores? Large variance = the model is sensitive to non-defect factors. View the most inconsistent components.
- **Synthetic threshold analysis**: Beyond the global optimal threshold, compute per-component-type optimal thresholds. How much KPI improves with component-aware thresholds? This reveals if a single threshold is fundamentally wrong.
- **Data integrity audit**: Run a Python script to check for: duplicate rows, missing image files (test or golden), NaN/empty scores, mismatched column counts, inconsistent path formats, samples where test_path == golden_path (comparing image to itself).
- **Augmentation sensitivity probe**: If augmentation config is available, check if test-time conditions fall outside the augmentation range (e.g., model trained with ±10° rotation but test has ±30° offset from golden).
- **Score distribution shape analysis**: Beyond mean/std — fit score distributions to known shapes (bimodal, uniform, skewed). A bimodal PASS distribution suggests two populations (e.g., two board types with different baselines). Plot histograms if possible.
- **Misalignment between train and inference pipeline**: Compare how images are loaded in training code vs inference code. Check for: different normalization, different resize interpolation, different crop strategy, channel order mismatch (RGB vs BGR).
- Provide: inference CSV path, train CSV path, experiment.yaml path, training code directory, image path construction rules, Phase 1 results

### Step 3: Collect and synthesize — Run sequentially
Collect all 6 agent results. Pay special attention to Agents E and F — they may surface root causes that Agents A-D missed entirely. Cross-reference exploratory findings with structured findings:
- Do the random samples confirm or contradict the failure mode clustering?
- Did anomaly hunting find issues not in any defect type category?
- Does the data integrity audit invalidate any conclusions from other agents?

Then run Phase 5 (counterfactual) yourself, because it needs findings from ALL agents. Include any new root causes from E/F in the what-if simulations.

### Step 4: Write the report — Run sequentially

**BEFORE writing RCA_Report.md**, run `ls rca_images/` to inventory all available thumbnails. You need exact filenames for inline embedding.

**Image Embedding Protocol (MANDATORY)**:
Every visual evidence table row MUST have inline thumbnail columns using `![caption](rca_images/<filename>.jpg)` syntax. A report without per-row images is incomplete — the hook will reject it.

Rules:
- **Section 3.1 (Golden Audit)**: Every audited golden row gets a `![golden](rca_images/...)` column
- **Section 3.2 (Failure Mode Clustering)**: Every defect sample row gets BOTH a test thumbnail column AND a golden thumbnail column
- **Section 3.3 (False Positive Analysis)**: Every FP row gets BOTH test and golden thumbnail columns
- **Section 3.4 (Visual Detectability)**: Every comparison pair gets side-by-side test + golden thumbnails
- **Section 7.4 (Decision Boundary Cases)**: Each boundary sample gets test + golden thumbnails

To match thumbnails to samples: cross-reference `object_name` and `boardname` from each row against filenames in `rca_images/`. If a thumbnail was not generated for a sample, note `(no thumbnail)` in that cell.

Table format for image-heavy sections:
```
| Sample | Score | Test Image | Golden Image | Failure Mode | ... |
|--------|-------|------------|--------------|--------------|-----|
| <obj> | <score> | ![test](rca_images/<test_thumb>.jpg) | ![golden](rca_images/<golden_thumb>.jpg) | <mode> | ... |
```

Add a dedicated section for exploratory findings:
```
## 7. Exploratory Findings (Agents E & F)
- Unexpected patterns discovered
- Data integrity issues
- Cross-validation inconsistencies
- Anything that doesn't fit neatly into Phases 2-4
```

### Subagent Prompt Template

When launching each agent, include in the prompt:
1. The Visual Inspection Primer (copy it)
2. The image path construction rules
3. The specific Phase instructions for that agent
4. Phase 1 results (score stats, key sample IDs, defect types)
5. All file paths (experiment dir, CSV paths, image dir, config paths)
6. Instruction to return structured findings as markdown sections matching the report structure

**IMPORTANT**: Each agent must return:
- Markdown tables with all data (will be pasted into the report)
- List of all images viewed with verdicts
- Key findings and root causes identified
- **Thumbnail filename mapping**: A table mapping each sample (object_name + boardname) to exact thumbnail filenames generated in `rca_images/`. The main thread needs these exact filenames to embed inline images. Format:
  ```
  ## Thumbnail Map
  | object_name | boardname | test_thumbnail | golden_thumbnail |
  |-------------|-----------|----------------|------------------|
  | ... | ... | test_<name>.jpg | golden_<name>.jpg |
  ```

### PHASE 1: Score Analysis (establish hypotheses)

Read `inference/inference.csv` and compute:

1. **Score statistics**: Split by PASS vs all non-PASS. Compute min/max/mean/median/std for each. Score gap = mean(NO_PASS) - mean(PASS).
2. **Tier classification** from score gap:
   - Tier 1 (Dead): gap < 0.03 — near-random
   - Tier 2 (Weak): gap 0.03–0.10 — some signal, heavy overlap
   - Tier 3 (Moderate): gap 0.10–0.20 — partial separation
   - Tier 4 (Strong): gap > 0.20 — good separation
3. **Threshold sweep**: For 200 thresholds from min to max score, compute TP/FP/TN/FN/precision/recall/F1/FAR. Find: KPI-optimal threshold, best-F1 threshold, 100%-recall threshold. Build confusion matrices.
4. **Per-defect-type scores**: Table of each defect type with count, min/max/mean score. Sort by mean score ascending (hardest to detect first).
5. **KPI verdict**: Can the model meet the target? How far off? (e.g., "100% recall requires FAR = 99%")

This gives you hypotheses: which defect types fail, which PASS components are FP magnets, whether the model learned anything at all.

6. **Threshold-critical sample analysis**: The lowest-scoring defect sets the 100% recall threshold — a single bad sample can force FAR from 5% to 99%. Compute "drop-N" analysis: FAR at 100% recall if worst 1, 2, 3, 5 defects excluded. If dropping a few helps dramatically → data quality issue on those samples. If dropping 5+ barely helps → systemic model failure.

### PHASE 2: Deep Image Investigation (prove with visual evidence)

This is the most important phase. You must **view actual images** to understand why scores are what they are. Use the Read tool to view images — it renders them visually.

**Image path construction:**
- Test image: `{images_dir}/{input_path}/{object_name}_{light_condition}.{ext}`
- Golden image: `{images_dir}/{golden_path}/{object_name}_{light_condition}.{ext}`
- `light_condition` from `dataset.classify.input_map` keys
- `ext` from `dataset.classify.image_ext` (e.g., .jpg)
- `images_dir` from `dataset.classify.train_dataset.images_dir` (or infer_dataset)

#### 2A. Threshold-Critical Sample Deep Dive (MUST DO FIRST)

**Goal**: View the samples that directly set the KPI operating point — they have disproportionate impact. A single bad sample can shift FAR from 5% to 99%.

- **Recall-first**: View test + golden for the **bottom 5 lowest-scoring defects**. For each: is it a data issue (dark golden, framing mismatch, mislabel) or a genuine hard case?
- **FAR-first**: View the **top 10 highest-scoring PASS** samples similarly.
- Cross-reference with the drop-N analysis from Phase 1: would fixing these samples make the KPI achievable, or is the overlap systemic?

#### 2B. Systematic Golden Image Audit

**Goal**: Find corrupted/dark/misframed golden images that inject noise into scores.

Write and run a Python script that:
1. Loads every unique golden image path referenced by defect samples in inference.csv
2. Computes mean pixel intensity for each golden image
3. **First, establish a baseline**: sample ~20 random PASS golden images and compute
   their mean intensity. This determines what "normal" looks like for this imaging
   modality. Some illumination types (e.g., SolderLight) produce systemically dark
   images where 80%+ of goldens have mean intensity < 30 — this is normal, not
   corruption. Set the "dark/corrupted" threshold relative to the PASS baseline
   (e.g., flag images below the 5th percentile of PASS golden intensities).
4. Flag images below the adaptive threshold as potentially corrupted
5. **Thumbnail generation**: For every image viewed during the investigation (golden audit, failure mode clustering, FP analysis, detectability assessment), copy and resize it to 128×128 px into an `rca_images/` folder next to the report. Name thumbnails descriptively (e.g., `golden_<sample_id>.jpg`, `test_<sample_id>.jpg`). These will be embedded in the final report using `![caption](rca_images/<name>.jpg)` syntax.

Then **view every flagged golden image** with the Read tool to confirm. For each:
- Is it completely dark/black?
- Is it a board-level view instead of component crop?
- Is the component visible and properly framed?

**Report**: Table of golden quality findings with image paths, mean intensity, visual verdict, and inline thumbnail image.

#### 2B. Failure Mode Clustering (view ALL defect images)

**Goal**: Classify every test defect into a failure mode category by viewing images.

For **every defect sample** in inference.csv (or up to 50 if there are many):
1. View both the test image and golden image using the Read tool
2. classify each sample at two levels:
  - failure mode (dark golden, framing mismatch, subtle defect, etc.)
  - visual defect subtype (describe what you actually see in the image — do not assume categories, derive them from observation):

| Failure Mode | Defect Subtype | Description | Example |
|--------------|----------------|-------------|---------|

3. Record: sample_id, defect_type, score, failure_mode, visual_description, golden_quality

**This clustering is the key deliverable.** It tells you:
- What fraction of failures are data quality issues (dark golden, framing) vs genuine model limitations?
- Are "obvious" defects scoring low? (= model hasn't learned) vs only "subtle" ones? (= model learned basics but needs refinement)
- Which failure modes dominate? This determines the fix.

#### 2C. False Positive Deep Dive

**Goal**: Understand why specific PASS components score high.

1. Take the top 10 highest-scoring PASS samples
2. View both golden and test images for each
3. Classify the FP cause:

| FP Cause | Description |
|----------|-------------|
| **Surface Reflectance** | Reflective surfaces differ between golden and test due to material/angle variation |
| **Position Shift** | Subject slightly offset from golden reference |
| **Lighting Variation** | Different illumination intensity/angle |
| **Golden Quality** | Golden image has issues (dark, misframed) |
| **Background Difference** | Background pattern differs between test and golden |

4. Check if FPs cluster on specific `object_name` values (same component across boards)
5. Check if FPs cluster on specific `comp_type_2` values (component category)

**Report**: Table of top 10 FPs with scores, inline test/golden thumbnails, visual cause classification, and clustering analysis.

#### 2D. Comparative Visual Analysis

**Goal**: Establish whether defects are visually detectable at the model's input resolution.

View side-by-side pairs for:
1. A typical low-scoring PASS pair (score near PASS median) — what "normal similar" looks like
2. The training defect sample(s) — what the model was taught
3. Representative defects from each type in test — are they visually distinguishable from PASS?

For each pair, describe: what visual difference exists, how prominent it is, whether a human could detect it at the model's input resolution.

#### 2E. Label Semantics & Visual Pattern Alignment Audit

**Goal**: Determine whether the dataset labels correspond to consistent visual concepts, and whether train/validation/test are aligned at the visual-pattern level.

A label is not sufficient evidence by itself. The investigator must verify whether samples sharing the same label also share the same visible pattern. A single label may contain multiple unrelated visual patterns. If the training samples and test samples under the same label are visually different, the model may fail even when the label names match.

For each label in train, validation, and inference:
1. Sample representative rows and construct test/golden image paths
2. View the actual images
3. Assign a **visual subtype** based on what is visible, independent of the CSV label
4. Build a subtype distribution table per split
5. Compare train vs validation vs test subtype coverage and proportions

Required subtype checks:
- Does one label contain multiple unrelated visual patterns? → **Label impurity**
- Does test contain subtypes absent from training? → **Unseen subtype**
- Do train and test use the same label name but different visual meanings? → **Semantic mismatch**
- Do visually similar samples appear under different labels? → **Label inconsistency**

For each label, report:
- split counts
- subtype counts
- representative thumbnails
- purity verdict
- alignment verdict

Severity guidance:
- **High severity**: test subtype absent from train, or label contains unrelated visual mechanisms
- **Medium severity**: subtype exists in train but at very low frequency vs test
- **Low severity**: subtype mix differs slightly but main patterns overlap

### PHASE 3: Cross-Dimensional Analysis (find patterns the model can't see)

#### 3A. Component-Type Clustering

**Goal**: Determine if failures correlate with physical component characteristics, not just defect labels.

Write and run a Python script that:
1. Group all inference samples by `comp_type_2` (component category)
2. For each component type, compute: count, mean PASS score, mean defect score, score gap, FP rate, FN rate
3. Rank by FP rate descending — which component types are FP magnets?
4. Rank by FN rate descending — which component types hide defects?

Then **view representative images** from the worst 3 component types for FP and FN. Look for:
- Physical size (large objects lose detail when downscaled to model input size)
- Surface material (reflective vs matte surfaces)
- Subject complexity (multi-element vs simple subjects)

**Report**: Component-type heatmap table with score statistics, FP/FN rates, and visual explanation of why certain types fail.

#### 3B. Board-Level & Positional Analysis

**Goal**: Find systematic issues tied to board identity or component position rather than defect type.

1. If `board_id` or equivalent field exists in CSV: group scores by board. Do certain boards consistently produce higher FP rates? (= board-level golden quality issue)
2. If positional data exists (`object_name` often encodes location): do failures cluster spatially? (= lighting gradient, camera vignetting, or board warp)
3. Cross-tabulate: board × defect_type × score. Is the model failing on specific board+component combinations?

**Report**: Board-level score table. Flag any board where mean PASS score > overall 75th percentile (= systematic FP source).

#### 3C. Training Image Deep Dive

**Goal**: Understand what the model was actually taught — view the training data, not just test data.

1. Read the training CSV and **view all training defect samples** (test + golden pairs)
2. For each training defect, assign a visual subtype (same taxonomy as Phase 2B)
3. Compare training defect visual patterns vs test defect visual patterns:
   - Does training cover the visual diversity seen in test?
   - Are training defects more obvious/exaggerated than test defects?
   - Is training data from the same board type / lighting setup?
4. View 10 random training PASS pairs — are they truly defect-free? Mislabeled PASS samples poison the model.

**Report**: Training vs test visual pattern comparison table. Flag any test pattern not represented in training.

#### 3D. Multi-Light Condition Analysis

**Goal**: If multiple light conditions exist in `dataset.classify.input_map`, check if performance varies by lighting.

1. Check `dataset.classify.input_map` for all light conditions
2. If multiple exist: for each light condition, compute the score distribution separately
3. View the same component under different lights — which light makes defects most visible?
4. Check if the model uses all light conditions or only one

**Report**: Per-light-condition score statistics. Recommendation on which lights are informative vs noise.

### PHASE 4: Data & Training Config Analysis

#### 4A. Data Sufficiency

Read training CSV, validation CSV, and inference CSV. Report:
1. **Sample counts**: Total/PASS/per-defect-type for train, validation, test
2. **Defect type coverage matrix**: Which types appear in which splits
3. **Domain gap**: Check whether train and test come from different visual domains.
4. **Validation signal**: Does validation contain any defects? If not, checkpoint selection is blind.
5. **Class ratio analysis**: Compute PASS:defect ratio in train. If > 100:1, the model may never learn defect features. Cross-reference with sampler settings.

#### 4B. Training Config Audit

Read `train/experiment.yaml`. Compute and report:

1. **Sampler × class weight interaction**:
   - From code (`oi_dataset.py:get_sampler`): `fail_wt = (num_pass / num_fail) * fpratio_sampling`
   - Effective over-emphasis = fail_wt × cls_weight[1]
   - Flag if > 100x
2. **Learning rate at inference checkpoint**:
   - Linear policy: `effective_lr = lr * (1.0 - epoch / (num_epochs + 1))`
   - Compute at checkpoint epoch. Flag if < 1e-6.
3. **Key config table**: difference_module, loss, embed_dec, freeze_backbone, num_epochs, batch_size, image_size
4. **Model output type**: learnable → softmax P(defect), euclidean → distance
5. **Augmentation audit**: What augmentations are enabled? Are they appropriate for the domain? (e.g., color jitter may destroy color-based signals; aggressive crop can remove small defects)
6. **Image size vs component size**: Is 224x224 sufficient? Compute the pixel-per-mm ratio for the largest components — if original images are 1600+ px and the defect occupies < 5% of the area, 224x224 may discard the defect entirely.

#### 4C. Training Metrics

Read `train/status.json` (JSONL format — one JSON object per line). Extract epoch-level metrics if available. Look for:
- Did loss converge or oscillate?
- train_fpr = 0 throughout? (not challenged)
- val_acc = 100% on defect-free validation? (meaningless)
- **Overfitting signal**: train_acc >> val_acc? Loss divergence between train/val?
- **Early stopping**: Did the best checkpoint occur early (underfitting) or at the very end (may not have converged)?

#### 4D. Loss Function & Decision Boundary Analysis

**Goal**: Understand if the loss function and decision mechanism match the problem.

1. For **learnable** module: softmax outputs P(defect). Check if the score distribution is bimodal (good) or uniform (model uncertain).
2. For **euclidean** module: distance-based scores have no natural threshold. Check if distances are calibrated — is there a clear gap between PASS and defect distances?
3. Compute **score entropy**: `H = -p*log(p) - (1-p)*log(1-p)` for learnable scores. High entropy near the threshold = model is guessing.
4. **Calibration plot**: Bin scores into 10 buckets, compute actual defect rate per bucket. Is the model calibrated? (score 0.8 should mean ~80% chance of defect)

### PHASE 5: Counterfactual & Actionability Analysis

#### 5A. "What-If" Simulations

**Goal**: Quantify the impact of fixing each root cause to prioritize remediation.

For each root cause identified, simulate the fix:
1. **Dark golden fix**: Remove all samples with dark goldens from scoring → recompute FAR at 100% recall
2. **Mislabel fix**: Remove suspected mislabels → recompute metrics
3. **Component-type exclusion**: What if we exclude the worst FP component type? What's the KPI improvement?
4. **Threshold per component type**: Instead of one global threshold, compute optimal per-type thresholds → theoretical best KPI

**Report**: Impact table showing each fix, samples affected, KPI before, KPI after, delta.

#### 5B. Minimum Viable Fix Path

**Goal**: Give the user a concrete, prioritized action plan.

1. Rank all root causes by: (impact on KPI) × (1 / effort to fix)
2. For each fix, specify:
   - Exactly what to change (specific samples to relabel, golden images to reshoot, config values to modify)
   - Expected KPI improvement (from 5A simulations)
   - Risk (could this make other metrics worse?)
3. Identify the **minimum set of fixes** needed to reach the target KPI
4. Flag if the target KPI is **unreachable** even with all fixes — and explain why (e.g., defects are genuinely invisible at this resolution)

---

## Architecture Reference

- **Learnable module**: `softmax(model(img1, img2), dim=1)[:, 1]` → score = P(defect). Higher = more defective.
- **Euclidean module**: `F.pairwise_distance(embed1, embed2)` → score = distance. Higher = more different.
- **WeightedRandomSampler**: `fail_wt = (num_pass / num_fail) * fpratio_sampling`. Defects sampled at fail_wt:1 rate.
- **Image paths**: `{images_dir}/{input_path}/{object_name}_{light_condition}.{ext}`
- **LR linear**: `lr * (1.0 - epoch / (num_epochs + 1))`
- **Data loading**: `SiameseNetworkTRIDataset` for `num_golden=1`, `MultiGoldenDataset` for `num_golden>1`

---

## Report Structure

```
# Root Cause Analysis Report: <Experiment Name>

## 1. Verdict
- Tier (1-4), score gap, KPI result
- One-paragraph root cause summary
- Top 3 root causes ranked

## 2. Score Analysis
- Score distributions (PASS vs NO_PASS)
- Threshold analysis with confusion matrices
- Per-defect-type score table

## 3. Visual Evidence
For every table below, embed inline thumbnail images using Markdown image syntax:
`![caption](path/to/image)` — use relative paths from the report location.
Before writing the report, generate a thumbnail gallery: write and run a Python script
that copies relevant images into a `rca_images/` subfolder next to the report, resized
to 128×128 px (or original size if smaller). Use these thumbnails in the Markdown tables.

- 3.1 Golden Image Audit
     | Golden Path | Thumbnail | Mean Intensity | Visual Verdict |
     |-------------|-----------|----------------|----------------|
     (one row per audited golden image, thumbnail = `![golden](rca_images/<name>.jpg)`)

- 3.2 Failure Mode Clustering
     | Sample | Score | Defect Type | Test Image | Golden Image | Failure Mode | Description |
     |--------|-------|-------------|------------|--------------|--------------|-------------|
     (embed test + golden thumbnails side-by-side per row)
     - Summary: N obvious defects scoring low → model didn't learn
     - Summary: N dark goldens → data quality issue
     - Summary: N framing mismatches → golden pipeline issue

- 3.3 False Positive Analysis
     | Rank | Sample | Score | Test Image | Golden Image | FP Cause | Component/Type |
     |------|--------|-------|------------|--------------|----------|----------------|
     (top 10 FPs with inline thumbnails and visual cause)
     - Clustering: which components/types dominate FPs

- 3.4 Visual Detectability Assessment (can a human see it at 224x224?)
     Include side-by-side test vs golden thumbnail pairs for:
     - A typical low-scoring PASS pair
     - Representative defects from each type
     - The hardest cases (highest-scoring defects, lowest-scoring defects)

## 4. Cross-Dimensional Analysis
- 4.1 Component-Type Clustering
     | Component Type | Count | Mean PASS Score | Mean Defect Score | Gap | FP Rate | FN Rate |
     |----------------|-------|-----------------|-------------------|-----|---------|---------|
     (with visual explanation for worst types)
- 4.2 Board-Level Analysis (if board IDs available)
- 4.3 Training Image Deep Dive
     | Training Sample | Visual Subtype | Also in Test? | Difficulty vs Test |
     |-----------------|----------------|---------------|-------------------|
     - Training vs test pattern coverage verdict
- 4.4 Multi-Light Condition Analysis (if applicable)

## 5. Data Issues
- Sample counts table (train/val/test × PASS/defect types)
- Defect type coverage matrix
- Class ratio analysis
- Domain gap / board mismatch analysis
- Validation signal check

## 6. Training Config Issues
- Sampler × class weight computation
- LR at checkpoint epoch
- Augmentation audit
- Image size vs component size analysis
- Config parameter table with flags
- Loss function & calibration analysis

## 7. Exploratory Findings

- 7.1 Random Sampling Discoveries
     | Sample | Score | Expected? | Observation |
     |--------|-------|-----------|-------------|
     (20 random samples across full score range — anything unexpected)

- 7.2 Score Anomalies
     | Sample | Score | Why Anomalous | Visual Explanation |
     |--------|-------|---------------|-------------------|
     (outliers that don't match their neighbors)

- 7.3 Golden Consistency Check
     | Component | Board A Golden | Board B Golden | Consistent? |
     |-----------|---------------|---------------|-------------|
     (same component across boards — golden pipeline stability)

- 7.4 Decision Boundary Cases
     | Sample | Score | Label | Test Image | Golden Image | Why Ambiguous |
     |--------|-------|-------|------------|--------------|---------------|
     (samples closest to threshold — the model's hardest calls)

- 7.5 Metadata Correlations
     | Field | Correlation with Score | Interpretation |
     |-------|----------------------|----------------|
     (unexpected correlations found by mining)

- 7.6 Data Integrity Issues
     - Duplicate rows, missing files, NaN scores, path mismatches

- 7.7 Score Distribution Shape Analysis
     - Bimodal? Uniform? Skewed? What does the shape reveal?

- 7.8 Train vs Inference Pipeline Misalignment
     - Normalization, resize, crop, channel order differences

## 8. Counterfactual Impact Analysis
- 8.1 What-If Simulations
     | Root Cause | Fix | Samples Affected | KPI Before | KPI After | Delta |
     |------------|-----|------------------|------------|-----------|-------|
- 8.2 Minimum Viable Fix Path
     | Priority | Fix | Effort | KPI Impact | Risk |
     |----------|-----|--------|------------|------|
     - Is target KPI reachable? If not, why?

## 9. Recommended Fixes (prioritized by impact × feasibility)
```

## Output Location

Always save into a timestamped folder:
```
<experiment_result_dir>/rca_results/YYYY-MM-DD_HHMMSS/
├── RCA_Report.md          # The full report
├── rca_images/            # All thumbnails embedded in the report
├── rca_config/            # Auto-copied by hook: skill, commands, hooks, settings
│   ├── skills/
│   ├── commands/
│   ├── hooks/
│   └── settings.local.json
└── claude_session.jsonl   # Auto-copied by hook: conversation log
```

1. At the start of the investigation, get the real current timestamp by running `date +%Y-%m-%d_%H%M%S` in Bash, then create the output folder: `<experiment_dir>/rca_results/<timestamp>/`. Do NOT hardcode or guess the time — always use the shell command.
2. Write `rca_images/` thumbnails into that folder
3. Write `RCA_Report.md` into that folder (this triggers the packaging hook to copy config + logs)

If the user specifies a custom path, use that instead but maintain the same structure.
