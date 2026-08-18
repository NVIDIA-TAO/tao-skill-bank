#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibrate FAR threshold on validation and apply it unchanged to KPI data.

The classification rule and candidate thresholds exactly match
``analyze_kpi.py``: ``score > threshold`` predicts a defect, and candidates are
``nextafter(min_score, -inf)`` plus every observed calibration score. The
highest candidate below the minimum defect score gives 100% calibration recall
while minimizing validation FAR.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _load(path: Path) -> tuple[list[float], list[bool]]:
    scores: list[float] = []
    is_pass: list[bool] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = {"label", "siamese_score"} - fields
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            raw_score = (row.get("siamese_score") or "").strip()
            if not raw_score:
                raise ValueError(f"{path}: empty siamese_score at line {line}")
            score = float(raw_score)
            if not math.isfinite(score):
                raise ValueError(f"{path}: non-finite siamese_score at line {line}")
            scores.append(score)
            is_pass.append((row.get("label") or "").strip().upper() == "PASS")
    if not scores:
        raise ValueError(f"{path}: no rows")
    return scores, is_pass


def calibrate_threshold(scores: list[float], is_pass: list[bool]) -> float:
    defects = [score for score, passed in zip(scores, is_pass) if not passed]
    passed = [score for score, is_ok in zip(scores, is_pass) if is_ok]
    if not defects or not passed:
        raise ValueError("calibration data requires both PASS and defect rows")
    candidates = [math.nextafter(min(scores), -math.inf), *sorted(set(scores))]
    eligible = [threshold for threshold in candidates if threshold < min(defects)]
    if not eligible:  # Defensive; nextafter(min(scores), -inf) is always eligible.
        raise ValueError("no threshold achieves 100% calibration recall")
    return max(eligible)


def evaluate_at_threshold(
    scores: list[float],
    is_pass: list[bool],
    threshold: float,
) -> dict[str, float | int]:
    tp = fp = tn = fn = 0
    for score, passed in zip(scores, is_pass):
        predicted_defect = score > threshold
        if not passed and predicted_defect:
            tp += 1
        elif passed and predicted_defect:
            fp += 1
        elif passed:
            tn += 1
        else:
            fn += 1
    recall = tp / (tp + fn) if tp + fn else math.nan
    far = fp / (fp + tn) if fp + tn else math.nan
    precision = tp / (tp + fp) if tp + fp else math.nan
    return {
        "far_pct": far * 100.0,
        "recall_pct": recall * 100.0,
        "precision": precision,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_pass": fp + tn,
        "n_defect": tp + fn,
    }


def calibrated_far(validation_csv: Path, kpi_csv: Path) -> dict:
    val_scores, val_is_pass = _load(validation_csv)
    kpi_scores, kpi_is_pass = _load(kpi_csv)
    threshold = calibrate_threshold(val_scores, val_is_pass)
    validation = evaluate_at_threshold(val_scores, val_is_pass, threshold)
    kpi = evaluate_at_threshold(kpi_scores, kpi_is_pass, threshold)
    return {
        "name": "far_pct",
        "value": kpi["far_pct"],
        "unit": "%",
        "threshold": threshold,
        "constraints": {"recall_pct": kpi["recall_pct"]},
        "diagnostics": {
            "protocol": "validation_threshold_applied_to_kpi",
            "validation": validation,
            "kpi": kpi,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation_csv", type=Path)
    parser.add_argument("kpi_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args(argv)

    result = calibrated_far(args.validation_csv, args.kpi_csv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"FAR@validation-threshold={result['value']:.4f}% "
        f"threshold={result['threshold']:.8f} "
        f"recall={result['constraints']['recall_pct']:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
