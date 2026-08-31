#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Build Proxy RCCA gap candidates with the recorded exact evaluator logic.

This script does not calculate an application KPI. It dynamically loads the
workspace evaluator recorded in state and reuses its parsers/matching helpers
to assign record-level correctness for mining selection.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from typing import Any

import yaml

from gap_analysis.config import load_profile, validate_config
from gap_analysis.runner import run_selection
from validate_sharegpt import image_paths, load_records, target_path


def _load_evaluator(path: pathlib.Path) -> Any:
    resolved = path.expanduser().resolve(strict=True)
    spec = importlib.util.spec_from_file_location("deft_recorded_exact_f1", resolved)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load exact evaluator: {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = (
        "build_classification_examples",
        "build_detection_examples",
        "parse_choice_labels",
        "parse_direct_bcq",
        "parse_boxes",
        "canonicalize_prediction_boxes",
        "one_to_one_detection_counts",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise ValueError(f"exact evaluator lacks required helpers: {missing}")
    return module


def _score_row(
    evaluator: Any,
    source: dict[str, Any],
    prediction: dict[str, Any],
) -> tuple[float, bool, str]:
    row_id = str(source["id"])
    raw = str(prediction.get("raw_prediction", ""))
    classification = evaluator.build_classification_examples({row_id: source})
    if classification:
        example = classification[0]
        if example["answer_format"] == "direct_bcq":
            labels, parse_ok = evaluator.parse_direct_bcq(raw)
        else:
            labels, parse_ok = evaluator.parse_choice_labels(
                raw, example["option_letters"], example["option_text"]
            )
        correct = bool(parse_ok and labels == example["gt_labels"])
        return float(correct), bool(parse_ok), "classification_mismatch"
    detection = evaluator.build_detection_examples({row_id: source}, 1.0)
    if not detection:
        raise ValueError(f"exact evaluator does not classify source row {row_id!r}")
    example = detection[0]
    native, parse_ok = evaluator.parse_boxes(raw)
    predicted = evaluator.canonicalize_prediction_boxes(native, "xyxy") if parse_ok else []
    if parse_ok:
        tp, fp, fn = evaluator.one_to_one_detection_counts(
            example["gt_boxes"], predicted, 0.5
        )
    else:
        tp, fp, fn = 0, 0, len(example["gt_boxes"])
    correct = bool(parse_ok and fp == 0 and fn == 0)
    return float(correct), bool(parse_ok), "detection_mismatch"


def build_candidates(
    evaluator_path: pathlib.Path,
    source_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evaluator = _load_evaluator(evaluator_path)
    predictions = {str(row.get("id", "")): row for row in prediction_rows}
    if "" in predictions or len(predictions) != len(prediction_rows):
        raise ValueError("prediction JSONL has missing or duplicate IDs")
    source_ids = {str(row.get("id", "")) for row in source_rows}
    if "" in source_ids or len(source_ids) != len(source_rows):
        raise ValueError("Proxy JSONL has missing or duplicate IDs")
    missing = sorted(source_ids - predictions.keys())
    unknown = sorted(predictions.keys() - source_ids)
    if missing or unknown:
        raise ValueError(f"prediction coverage mismatch: missing={missing[:10]}, unknown={unknown[:10]}")
    candidates: list[dict[str, Any]] = []
    for source in source_rows:
        row_id = str(source["id"])
        score, parse_ok, gap_type = _score_row(evaluator, source, predictions[row_id])
        paths = image_paths(source, context=row_id)
        target = target_path(source, context=row_id)
        candidates.append(
            {
                "id": row_id,
                "evaluation_role": "proxy",
                "task_type": str(source["task_type"]),
                "metric_family": (
                    "detection" if "detection" in str(source["task_type"]).casefold()
                    else "classification"
                ),
                "reference_cohort": (
                    "reference_based" if len(paths) == 2 else "non_reference_based"
                ),
                "dataset": str(source.get("dataset", "unknown")),
                # Several task rows may address the same physical target.
                # Default to the target path so routing embeds that image once.
                "target_id": str(source.get("target_id", target)),
                "target_path": target,
                "sample_score": score,
                "parse_ok": parse_ok,
                "gap_type": gap_type,
                "raw_prediction": predictions[row_id].get("raw_prediction"),
            }
        )
    return candidates


def _write_parquet(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("pyarrow is required for RCCA parquet artifacts") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator", required=True, type=pathlib.Path)
    parser.add_argument("--source", required=True, type=pathlib.Path)
    parser.add_argument("--predictions", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--gap-analysis-profile", default="deficit_weighted_round_robin")
    choice.add_argument("--gap-analysis-config", type=pathlib.Path)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    try:
        if args.gap_analysis_config:
            config = validate_config(yaml.safe_load(args.gap_analysis_config.read_text()))
        else:
            config = load_profile(args.gap_analysis_profile)
        if args.budget is not None:
            config["budget"] = args.budget
        if args.seed is not None:
            config["seed"] = args.seed
        config = validate_config(config)
        candidates = build_candidates(
            args.evaluator,
            load_records(args.source),
            load_records(args.predictions),
        )
        selected, selection = run_selection(candidates, config)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_parquet(args.output_dir / "gap_candidates.parquet", candidates)
        _write_parquet(args.output_dir / "selected_gaps.parquet", selected)
        summary = {
            "schema_version": 1,
            "evaluation_role": "proxy",
            "evaluator": str(args.evaluator.expanduser().resolve()),
            "samples": len(candidates),
            "incorrect_samples": sum(row["sample_score"] < 1.0 for row in candidates),
            "parse_failures": sum(not row["parse_ok"] for row in candidates),
            "selection": selection,
        }
        (args.output_dir / "gaps_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    except (ImportError, OSError, TypeError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"analyze_gaps: {exc}", file=sys.stderr)
        return 2
    print(
        f"analyze_gaps: samples={summary['samples']} "
        f"incorrect={summary['incorrect_samples']} selected={len(selected)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
