#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Invoke the canonical NVPAW evaluator and build the frozen five-part gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Any

from cfw_predictions import normalize_prediction, read_jsonl


COMPONENTS = (
    ("non_reference_based.tasks.BCQ.macro_f1", ("non_reference_based", "BCQ", "macro_f1")),
    ("non_reference_based.tasks.MCQ.macro_f1", ("non_reference_based", "MCQ", "macro_f1")),
    ("non_reference_based.tasks.DET.f1", ("non_reference_based", "DET", "f1")),
    ("reference_based.tasks.BCQ.macro_f1", ("reference_based", "BCQ", "macro_f1")),
    ("reference_based.tasks.DET.f1", ("reference_based", "DET", "f1")),
)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _component(report: dict[str, Any], path: tuple[str, str, str], label: str) -> float:
    cohort, task, metric = path
    try:
        value = report["tasks_by_reference_cohort"][cohort]["tasks"][task][metric]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"exact evaluator report is missing {label}") from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"exact evaluator component {label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"exact evaluator component {label} must be in [0, 1]")
    return result


def build_metric_result(
    report: dict[str, Any],
    *,
    threshold: float,
    evaluator_path: str,
    evaluator_sha256: str,
    raw_report_path: str,
    raw_report_sha256: str,
) -> dict[str, Any]:
    if not 0.0 < threshold <= 1.0:
        raise ValueError("F1 component threshold must be in (0, 1]")
    if not isinstance(report, dict):
        raise ValueError("exact evaluator report must be an object")
    if not pathlib.Path(evaluator_path).is_absolute():
        raise ValueError("evaluator_path must be absolute")
    if not pathlib.Path(raw_report_path).is_absolute():
        raise ValueError("raw_report_path must be absolute")
    if not isinstance(evaluator_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", evaluator_sha256):
        raise ValueError("evaluator_sha256 must be a SHA-256 digest")
    if not isinstance(raw_report_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", raw_report_sha256
    ):
        raise ValueError("raw_report_sha256 must be a SHA-256 digest")
    alignment = report.get("alignment")
    if not isinstance(alignment, dict):
        raise ValueError("exact evaluator report is missing alignment evidence")
    coverage_failures = {
        "missing_evaluated_predictions": alignment.get("missing_evaluated_predictions"),
        "unknown_prediction_ids": alignment.get("unknown_prediction_ids"),
    }
    for name, value in coverage_failures.items():
        if type(value) is not int or value < 0:
            raise ValueError(f"exact evaluator alignment.{name} must be a non-negative integer")
    values = {
        label: _component(report, path, label) for label, path in COMPONENTS
    }
    components = {
        label: {
            "f1": value,
            "threshold": threshold,
            "attainment": min(value / threshold, 1.0),
            "passed": value >= threshold,
        }
        for label, value in values.items()
    }
    minimum_f1 = min(values.values())
    minimum_attainment = min(item["attainment"] for item in components.values())
    coverage_ok = all(value == 0 for value in coverage_failures.values())
    return {
        "schema_version": 1,
        "name": "f1_cohort_balanced_v1",
        "kpi_profile": "f1_cohort_balanced_v1",
        "display_name": "Worst required cohort F1 attainment",
        "value": minimum_attainment,
        "unit": "",
        "operator": ">=",
        "target": 1.0,
        "component_threshold": threshold,
        "required_components": [label for label, _ in COMPONENTS],
        "minimum_f1": minimum_f1,
        "components": components,
        "constraints": coverage_failures,
        "tie_breakers": {
            "minimum_f1": minimum_f1,
            "mean_f1": sum(values.values()) / len(values),
            "coverage_failures": float(sum(coverage_failures.values())),
        },
        "passed": coverage_ok and all(item["passed"] for item in components.values()),
        "evaluator_path": evaluator_path,
        "evaluator_sha256": evaluator_sha256,
        "raw_report_path": raw_report_path,
        "raw_report_sha256": raw_report_sha256,
        "alignment": dict(alignment),
    }


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_exact_evaluator(
    *,
    evaluator: pathlib.Path,
    predictions: pathlib.Path,
    source: pathlib.Path,
    raw_output: pathlib.Path,
) -> tuple[dict[str, Any], str]:
    evaluator = evaluator.expanduser().resolve(strict=True)
    predictions = predictions.expanduser().resolve(strict=True)
    source = source.expanduser().resolve(strict=True)
    raw_output = raw_output.expanduser().resolve()
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{raw_output.name}.", suffix=".tmp", dir=raw_output.parent
    )
    os.close(descriptor)
    temporary = pathlib.Path(temporary_name)
    temporary.unlink()
    try:
        command = [
            sys.executable,
            str(evaluator),
            "--source",
            str(source),
            "--predictions",
            str(predictions),
            "--output-json",
            str(temporary),
        ]
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ValueError(
                f"canonical evaluator failed with exit code {completed.returncode}: {detail}"
            )
        try:
            report = json.loads(temporary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"canonical evaluator did not write valid JSON: {raw_output}"
            ) from exc
        if not isinstance(report, dict):
            raise ValueError("canonical evaluator report root must be an object")
        os.replace(temporary, raw_output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return report, sha256_file(evaluator)


def preflight_source_contract(
    *, evaluator: pathlib.Path, source: pathlib.Path
) -> dict[str, Any]:
    """Prove the frozen source is fully consumable by the recorded evaluator."""

    source = source.expanduser().resolve(strict=True)
    rows = read_jsonl(source)
    with tempfile.TemporaryDirectory(prefix="deft-exact-f1-preflight-") as temporary:
        root = pathlib.Path(temporary)
        predictions = root / "perfect_predictions.jsonl"
        with predictions.open("w", encoding="utf-8") as stream:
            for row in rows:
                normalized = normalize_prediction(row, {"prediction": "preflight"})
                normalized["raw_prediction"] = normalized["GT"]
                stream.write(
                    json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
        raw_output = root / "raw_report.json"
        report, evaluator_hash = run_exact_evaluator(
            evaluator=evaluator,
            predictions=predictions,
            source=source,
            raw_output=raw_output,
        )
        result = build_metric_result(
            report,
            threshold=1.0,
            evaluator_path=str(evaluator.expanduser().resolve(strict=True)),
            evaluator_sha256=evaluator_hash,
            raw_report_path=str(raw_output.resolve()),
            raw_report_sha256=sha256_file(raw_output),
        )
    if not result["passed"]:
        raise ValueError(
            "perfect-prediction evaluator preflight did not attain every required component"
        )
    return {
        "schema_version": 1,
        "compatible": True,
        "source": str(source),
        "rows": len(rows),
        "evaluator": result["evaluator_path"],
        "evaluator_sha256": evaluator_hash,
        "required_components": result["required_components"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator", type=pathlib.Path, required=True)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--predictions", type=pathlib.Path)
    parser.add_argument("--raw-output", type=pathlib.Path)
    parser.add_argument("--metric-output", type=pathlib.Path)
    parser.add_argument("--component-threshold", type=float)
    parser.add_argument("--preflight-source-contract", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.preflight_source_contract:
            result = preflight_source_contract(
                evaluator=args.evaluator,
                source=args.source,
            )
            print(json.dumps(result, sort_keys=True))
            return 0
        missing = [
            flag
            for flag, value in (
                ("--predictions", args.predictions),
                ("--raw-output", args.raw_output),
                ("--metric-output", args.metric_output),
                ("--component-threshold", args.component_threshold),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"normal evaluation requires: {', '.join(missing)}")
        raw_output = args.raw_output.expanduser().resolve()
        report, evaluator_hash = run_exact_evaluator(
            evaluator=args.evaluator,
            predictions=args.predictions,
            source=args.source,
            raw_output=raw_output,
        )
        evaluator = args.evaluator.expanduser().resolve(strict=True)
        result = build_metric_result(
            report,
            threshold=args.component_threshold,
            evaluator_path=str(evaluator),
            evaluator_sha256=evaluator_hash,
            raw_report_path=str(raw_output),
            raw_report_sha256=sha256_file(raw_output),
        )
        _atomic_json(args.metric_output.expanduser().resolve(), result)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"exact_f1_adapter: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
