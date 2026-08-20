# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import importlib.util
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "far_eval_calibrated.py"
SPEC = importlib.util.spec_from_file_location("far_eval_calibrated", MODULE_PATH)
assert SPEC and SPEC.loader
metric = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metric)


def _write(path: Path, rows: list[tuple[str, float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "siamese_score"])
        writer.writerows(rows)


def test_calibration_threshold_is_frozen_for_kpi(tmp_path):
    validation = tmp_path / "validation.csv"
    kpi = tmp_path / "kpi.csv"
    _write(validation, [("PASS", 0.1), ("PASS", 0.6), ("missing", 0.5)])
    _write(
        kpi,
        [("PASS", 0.1), ("PASS", 0.4), ("PASS", 0.55), ("missing", 0.5)],
    )

    result = metric.calibrated_far(validation, kpi)

    assert result["diagnostics"]["validation"]["far_pct"] == 50.0
    assert math.isclose(result["value"], 200.0 / 3.0)
    assert result["constraints"]["recall_pct"] == 100.0
    assert result["diagnostics"]["protocol"] == "validation_threshold_applied_to_kpi"
