# Cosmos3 Proxy RCCA Report

Use Proxy evidence only. Preserve these level-2 headings exactly; the stage
commit validates them.

## 1. Verdict

State whether Proxy evidence shows false accepts, false rejects, both, or
neither. Include counts and the metric artifact path.

## 2. False-Accept Breakdown

Summarize `NG -> OK` rows by defect or visible pattern. Cite sample IDs and
never import Benchmark sample evidence.

## 3. False-Reject Breakdown

Summarize `OK -> NG` rows by visible pattern. Cite sample IDs and keep this
separate from false accepts.

## 4. Top-K Worst Samples

List the highest-priority Proxy sample IDs, ground truth, parsed prediction,
confidence when present, and image path recovered by joining to the Proxy
annotation on `id`.

## 5. Per-Defect Analysis

Describe supported defect families and uncertainty. Mark missing evidence
explicitly; do not invent a defect label.

## 6. Recommended Actions

Name concrete real-image Mining targets for the next routing stage. Indicate
whether each target belongs to the OK or NG query. These actions may influence
mining only; they never alter the frozen Benchmark gate.
