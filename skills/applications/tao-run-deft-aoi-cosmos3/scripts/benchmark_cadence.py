#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Validate Proxy KPI evidence and decide frozen-Benchmark cadence."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
from typing import Any


CADENCES = ("every", "final_and_best")


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Proxy metric {name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Proxy metric {name} must be finite")
    return result


def metric_rank(result: dict[str, Any]) -> tuple[float, float, float, float]:
    tie_breakers = result.get("tie_breakers")
    if not isinstance(tie_breakers, dict):
        raise ValueError("Proxy metric result requires tie_breakers")
    value = _number(result.get("value"), name="value")
    minimum_f1 = _number(tie_breakers.get("minimum_f1"), name="minimum_f1")
    mean_f1 = _number(tie_breakers.get("mean_f1"), name="mean_f1")
    coverage = _number(
        tie_breakers.get("coverage_failures"), name="coverage_failures"
    )
    if not 0.0 <= value <= 1.0 or not 0.0 <= minimum_f1 <= 1.0:
        raise ValueError("Proxy KPI value and minimum_f1 must be in [0, 1]")
    if not 0.0 <= mean_f1 <= 1.0 or coverage < 0.0:
        raise ValueError("Proxy mean_f1 must be in [0, 1] and coverage non-negative")
    return value, minimum_f1, mean_f1, -coverage


def is_proxy_best_so_far(
    result: dict[str, Any], previous_results: list[dict[str, Any]]
) -> tuple[bool, dict[str, Any]]:
    ranking = metric_rank(result)
    previous_rankings = [metric_rank(previous) for previous in previous_results]
    previous_best = max(previous_rankings) if previous_rankings else None
    is_best = previous_best is None or ranking >= previous_best
    return is_best, {
        "policy": "proxy_kpi_lexicographic_v1",
        "ranking_fields": [
            "value",
            "minimum_f1",
            "mean_f1",
            "negative_coverage_failures",
        ],
        "ranking": list(ranking),
        "comparison_count": len(previous_rankings),
        "previous_best_ranking": (
            list(previous_best) if previous_best is not None else None
        ),
        "is_best_so_far": is_best,
    }


def prior_proxy_results(
    state: dict[str, Any], *, current_label: str
) -> list[dict[str, Any]]:
    iterations = state.get("iterations")
    if not isinstance(iterations, dict):
        raise ValueError("state.iterations must be an object")

    def order(label: str) -> tuple[int, int]:
        if label == "baseline":
            return 0, 0
        if label.startswith("iter") and label[4:].isdigit():
            return 1, int(label[4:])
        return 2, 0

    output: list[dict[str, Any]] = []
    for label in sorted(iterations, key=order):
        if label == current_label:
            continue
        phase = iterations[label]
        result = phase.get("proxy_metric_result") if isinstance(phase, dict) else None
        if isinstance(result, dict):
            output.append(result)
    return output


def validate_proxy_metric_result(
    state: dict[str, Any],
    *,
    result_path: pathlib.Path,
    raw_report: pathlib.Path,
) -> dict[str, Any]:
    result_path = result_path.expanduser().resolve(strict=True)
    raw_report = raw_report.expanduser().resolve(strict=True)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Proxy metric result must be a JSON object")
    kpi = state.get("config", {}).get("kpi", {})
    if result.get("name") != "f1_cohort_balanced_v1":
        raise ValueError("Proxy metric result has the wrong KPI name")
    if result.get("evaluator_path") != kpi.get("evaluator"):
        raise ValueError("Proxy metric evaluator path differs from frozen state")
    if result.get("evaluator_sha256") != kpi.get("evaluator_sha256"):
        raise ValueError("Proxy metric evaluator hash differs from frozen state")
    if result.get("component_threshold") != kpi.get("component_threshold"):
        raise ValueError("Proxy metric threshold differs from frozen state")
    if pathlib.Path(str(result.get("raw_report_path", ""))).resolve() != raw_report:
        raise ValueError("Proxy metric raw-report path does not match the committed file")
    if result.get("raw_report_sha256") != _sha256(raw_report):
        raise ValueError("Proxy metric raw-report hash does not match the committed file")
    constraints = result.get("constraints")
    if not isinstance(constraints, dict) or any(
        type(constraints.get(name)) is not int or constraints[name] != 0
        for name in ("missing_evaluated_predictions", "unknown_prediction_ids")
    ):
        raise ValueError("Proxy metric requires exact prediction coverage")
    required = state.get("metric_contract", {}).get("required_components")
    if not isinstance(required, list) or result.get("required_components") != required:
        raise ValueError("Proxy metric required components differ from frozen state")
    metric_rank(result)
    result = dict(result)
    result["evidence_path"] = str(result_path)
    result["evaluation_role"] = "proxy"
    return result
