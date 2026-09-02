#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Attach a replacement-cohort Benchmark result without changing loop progress."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import re
import sys
from typing import Any

from cfw_predictions import read_prediction_jsonl
from metric_contract import contract_from_state, result_from_iteration, result_passes
from record_metric_result import _required_file, _validate_exact_provenance, _write_atomic


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: pathlib.Path, *, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} root must be an object")
    return payload


def _line_count(path: pathlib.Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for line in stream if line.strip())


def _resolved_report_path(value: Any, *, name: str) -> pathlib.Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"exact evaluator report is missing {name}")
    path = pathlib.Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"exact evaluator report {name} must be absolute")
    return path.resolve()


def _validate_active_benchmark(
    state: dict[str, Any], *, benchmark_rows: int
) -> tuple[pathlib.Path, str]:
    config = state.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("state.config must be an object")
    annotations = config.get("annotations", {})
    if not isinstance(annotations, dict):
        raise ValueError("state.config.annotations must be an object")
    raw_path = annotations.get("benchmark")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("state.config.annotations.benchmark is required")
    benchmark = _required_file(pathlib.Path(raw_path), name="active Benchmark")
    actual_rows = _line_count(benchmark)
    if actual_rows != benchmark_rows:
        raise ValueError(
            f"active Benchmark row count changed: expected {benchmark_rows}, got {actual_rows}"
        )
    actual_sha = _sha256(benchmark)
    seals: list[tuple[str, Any]] = []
    annotation_sha = config.get("annotation_sha256", {})
    if isinstance(annotation_sha, dict):
        seals.append(("config.annotation_sha256.benchmark", annotation_sha.get("benchmark")))
    evaluation = config.get("evaluation", {})
    if isinstance(evaluation, dict) and isinstance(evaluation.get("benchmark"), dict):
        seals.append(("config.evaluation.benchmark.sha256", evaluation["benchmark"].get("sha256")))
    for name, expected in seals:
        if expected != actual_sha:
            raise ValueError(
                f"active Benchmark SHA-256 does not match {name}: expected {expected}, got {actual_sha}"
            )
    if not seals:
        raise ValueError("state has no active Benchmark SHA-256 seal")
    return benchmark, actual_sha


def _validate_coverage(
    *,
    raw_report: dict[str, Any],
    metric_result: dict[str, Any],
    benchmark: pathlib.Path,
    predictions: pathlib.Path,
    benchmark_rows: int,
) -> None:
    if _resolved_report_path(raw_report.get("source"), name="source") != benchmark:
        raise ValueError("exact evaluator report source is not the active Benchmark")
    reported_predictions = raw_report.get("predictions")
    if reported_predictions is not None and _resolved_report_path(
        reported_predictions, name="predictions"
    ) != predictions:
        raise ValueError("exact evaluator report predictions path does not match evidence")
    for owner, payload in (("raw report", raw_report), ("metric result", metric_result)):
        alignment = payload.get("alignment")
        if not isinstance(alignment, dict):
            raise ValueError(f"{owner} is missing alignment evidence")
        for field in ("source_rows", "evaluated_source_rows", "prediction_rows"):
            if alignment.get(field) != benchmark_rows:
                raise ValueError(
                    f"{owner} alignment.{field} must equal {benchmark_rows}"
                )
        for field in ("missing_evaluated_predictions", "unknown_prediction_ids"):
            if alignment.get(field) != 0:
                raise ValueError(f"{owner} alignment.{field} must be zero")


def _evaluated_model(
    state: dict[str, Any], *, iter_label: str, explicit: pathlib.Path | None
) -> str:
    if explicit is not None:
        resolved = explicit.expanduser().resolve(strict=True)
        if resolved.is_file() and resolved.stat().st_size == 0:
            raise ValueError("evaluated model file must not be empty")
        if resolved.is_dir() and not any(resolved.iterdir()):
            raise ValueError("evaluated model directory must not be empty")
        return str(resolved)
    if iter_label == "baseline":
        model = state.get("config", {}).get("base_model")
    else:
        phase = state.get("iterations", {}).get(iter_label, {})
        model = phase.get("best_ckpt_path") if isinstance(phase, dict) else None
    if not isinstance(model, str) or not model:
        raise ValueError(f"cannot resolve evaluated model for {iter_label}")
    return model


def commit(args: argparse.Namespace) -> dict[str, Any]:
    if not re.fullmatch(r"baseline|iter[1-9][0-9]*", args.iter_label):
        raise ValueError("iter label must be baseline or iterN (N >= 1)")
    if type(args.benchmark_rows) is not int or args.benchmark_rows <= 0:
        raise ValueError("benchmark rows must be a positive integer")
    state_path = _required_file(args.state_path, name="state path")
    result_path = _required_file(args.result_json, name="metric result JSON")
    predictions = _required_file(
        args.benchmark_results, name="Benchmark predictions JSONL"
    )
    if predictions.suffix != ".jsonl":
        raise ValueError("Benchmark predictions must be JSONL")
    prediction_rows = read_prediction_jsonl(predictions)
    if len(prediction_rows) != args.benchmark_rows:
        raise ValueError(
            f"Benchmark prediction rows must equal {args.benchmark_rows}, got {len(prediction_rows)}"
        )
    raw_path = _required_file(args.raw_f1_report, name="raw exact-evaluator report")
    training_spec = (
        _required_file(args.training_spec, name="training spec")
        if args.training_spec is not None
        else None
    )
    state = _json_object(state_path, name="deft_state.json")
    if state.get("version") != 7:
        raise ValueError("state schema is not the Cosmos Framework v7 contract")
    benchmark, benchmark_sha = _validate_active_benchmark(
        state, benchmark_rows=args.benchmark_rows
    )
    raw_result = _json_object(result_path, name="metric result JSON")
    raw_report = _json_object(raw_path, name="raw exact-evaluator report")
    _validate_exact_provenance(state, raw_result, raw_path)
    _validate_coverage(
        raw_report=raw_report,
        metric_result=raw_result,
        benchmark=benchmark,
        predictions=predictions,
        benchmark_rows=args.benchmark_rows,
    )
    contract = contract_from_state(state)
    normalized = result_from_iteration({"metric_result": raw_result}, contract)
    if normalized is None:  # pragma: no cover
        raise ValueError("metric result JSON is empty")
    passed, _ = result_passes(contract, normalized)
    normalized["passed"] = passed
    normalized["evidence_path"] = str(result_path)
    phase = state.get("iterations", {}).get(args.iter_label, {})
    if not isinstance(phase, dict):
        phase = {}
    supersedes = {
        key: phase[key]
        for key in ("benchmark_predictions_jsonl", "metric_result")
        if key in phase
    }
    entry = {
        "schema": "deft_replacement_benchmark_evaluation_v1",
        "iter_label": args.iter_label,
        "evaluated_model": _evaluated_model(
            state,
            iter_label=args.iter_label,
            explicit=getattr(args, "evaluated_model", None),
        ),
        "benchmark_predictions_jsonl": str(predictions),
        "benchmark_predictions_sha256": _sha256(predictions),
        "raw_f1_report": str(raw_path),
        "raw_f1_report_sha256": _sha256(raw_path),
        "metric_result": normalized,
        "supersedes": supersedes,
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
    }
    if training_spec is not None:
        entry["training_spec"] = str(training_spec)

    trajectory = state.setdefault("benchmark_trajectory", {})
    if not isinstance(trajectory, dict):
        raise ValueError("state.benchmark_trajectory must be an object")
    for field, expected in (
        ("cohort_path", str(benchmark)),
        ("cohort_sha256", benchmark_sha),
        ("cohort_rows", args.benchmark_rows),
    ):
        existing = trajectory.get(field)
        if existing is not None and existing != expected:
            raise ValueError(f"replacement Benchmark trajectory {field} changed")
        trajectory[field] = expected
    trajectory["reporting_semantics"] = (
        "supersedes retired-cohort metrics for trajectory reporting without "
        "mutating historical pipeline stages"
    )
    evaluations = trajectory.setdefault("evaluations", {})
    if not isinstance(evaluations, dict):
        raise ValueError("state.benchmark_trajectory.evaluations must be an object")
    existing = evaluations.get(args.iter_label)
    if isinstance(existing, dict):
        immutable_fields = (
            "benchmark_predictions_sha256",
            "raw_f1_report_sha256",
        )
        if all(existing.get(field) == entry[field] for field in immutable_fields):
            return existing
        raise ValueError(
            f"replacement Benchmark result for {args.iter_label} already exists with different evidence"
        )
    evaluations[args.iter_label] = entry
    _write_atomic(state_path, state)
    return entry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-path", required=True, type=pathlib.Path)
    parser.add_argument("--iter-label", required=True)
    parser.add_argument("--result-json", required=True, type=pathlib.Path)
    parser.add_argument("--benchmark-results", required=True, type=pathlib.Path)
    parser.add_argument("--raw-f1-report", required=True, type=pathlib.Path)
    parser.add_argument("--benchmark-rows", required=True, type=int)
    parser.add_argument("--evaluated-model", type=pathlib.Path)
    parser.add_argument("--training-spec", type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        entry = commit(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"record_replacement_benchmark_metric: {exc}", file=sys.stderr)
        return 2
    metric = entry["metric_result"]
    print(
        f"iter={args.iter_label} metric={metric['name']} "
        f"minimum_f1={metric['minimum_f1']:.6g} passed={str(metric['passed']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
