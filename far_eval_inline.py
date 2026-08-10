"""FAR@100%recall — exact port of DEFT analyze_kpi.py semantics.

Rule (matches scripts/analyze_kpi.py):
  - predicted NO_PASS (defect)  <=>  siamese_score > threshold   (STRICT >)
  - candidate thresholds = nextafter(min_score, -inf) + every unique score
  - eligible = thresholds where recall == 1.0 (abs_tol 1e-12)
  - winner   = max over eligible by (f1, precision, threshold)
  - FAR      = fp / (fp + tn)

Usage: python3 far_eval_inline.py <inference.csv> <out_metric_result.json>
"""
import json, math, pathlib, sys

import numpy as np
import pandas as pd

csv_path, out_path = sys.argv[1], sys.argv[2]

df = pd.read_csv(csv_path)
labels = df["label"].astype(str).str.strip()
is_pass = (labels.str.upper() == "PASS").to_numpy()
scores = df["siamese_score"].astype(float).to_numpy()

n_nopass = int((~is_pass).sum())
n_pass = int(is_pass.sum())
if n_nopass == 0:
    sys.exit("No NO_PASS rows in inference.csv")

# Candidate thresholds: nextafter(min, -inf) + unique scores  (same as analyze_kpi)
unique_scores = np.unique(scores)
thresholds = np.concatenate(([np.nextafter(unique_scores[0], -np.inf)], unique_scores))

# Vectorized confusion counts for the strict `score > t` rule.
# For each t: flagged = #scores > t, computed via searchsorted on sorted arrays.
nopass_sorted = np.sort(scores[~is_pass])
pass_sorted = np.sort(scores[is_pass])
# side='right' gives #elements <= t, so #elements > t = n - that
tp = n_nopass - np.searchsorted(nopass_sorted, thresholds, side="right")
fp = n_pass - np.searchsorted(pass_sorted, thresholds, side="right")
fn = n_nopass - tp
tn = n_pass - fp

with np.errstate(divide="ignore", invalid="ignore"):
    recall = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
    precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
    f1 = np.where(
        (precision + recall) > 0, 2 * precision * recall / (precision + recall), np.nan
    )
    far = np.where(fp + tn > 0, fp / (fp + tn), np.nan)

# Eligible: recall == 1.0 exactly (abs_tol=1e-12, same as analyze_kpi)
eligible = np.isclose(recall, 1.0, rtol=0.0, atol=1e-12)
if not eligible.any():
    sys.exit("No threshold achieves 100pct recall")

# Selection: max by (f1, precision, threshold) — NaN treated as -inf (max_key port)
f1_k = np.where(np.isnan(f1), -np.inf, f1)
pr_k = np.where(np.isnan(precision), -np.inf, precision)
idx_pool = np.flatnonzero(eligible)
order = np.lexsort((thresholds[idx_pool], pr_k[idx_pool], f1_k[idx_pool]))
best = idx_pool[order[-1]]

result = {
    "name": "far_pct",
    "value": float(far[best] * 100.0),
    "unit": "%",
    "threshold": float(thresholds[best]),
    "constraints": {"recall_pct": float(recall[best] * 100.0)},
    "diagnostics": {
        "recall_pct": float(recall[best] * 100.0),
        "precision": float(precision[best]) if not math.isnan(precision[best]) else None,
        "f1": float(f1[best]) if not math.isnan(f1[best]) else None,
        "tp": int(tp[best]),
        "fp": int(fp[best]),
        "n_pass": n_pass,
        "n_nopass": n_nopass,
    },
}
pathlib.Path(out_path).write_text(json.dumps(result, indent=2))
print(f"FAR@100pctR={result['value']:.4f}% threshold={result['threshold']:.6f}")
