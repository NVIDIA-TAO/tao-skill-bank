#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Capability gate: decide per detection task whether the loop is in acquisition or refinement mode.

Rule (Phase 3, 2026-09-14). For every detection task compare the checkpoint's KPI-set F1
with the *trivial baseline* of the same set, i.e. the F1 a model gets by always answering
"no boxes" (empty ground truth + empty prediction counts as one true positive under the
authoritative evaluator, every ground-truth box is a false negative):

    trivial_f1 = 2 * E / (2 * E + B)      E = empty rows, B = ground-truth boxes

* F1 <= trivial_f1 + margin  ->  acquisition mode: the skill is absent, error-driven
  (residual / kNN) mining has nothing to refine; give the task a volume budget with broad
  dataset coverage (`--acquisition-task TASK=ROWS`).
* otherwise                   ->  refinement mode: residual mining, anchors, profile
  calibration as usual.

Inputs: the KPI set JSONL and the evaluator's raw F1 report for the checkpoint
(`raw_f1.json`, `tasks_by_reference_cohort` layout). Output: `capability_gate.json`.
Analysis-side: it changes nothing unless its recommendation is launch-recorded.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

from select_detection_calibration import DETECTION_TASKS, _json_answer
from validate_sharegpt import load_records, prompt_and_response

# evaluator cohort layout -> (cohort, family) per detection task
TASK_REPORT_KEY = {
    "Defect Detection": ("non_reference_based", "DET"),
    "Ref_based Defect Detection": ("reference_based", "DET"),
    "Component Detection": ("non_reference_based", "DET"),
}


def trivial_baselines(records) -> dict[str, dict[str, Any]]:
    out = {task: {"rows": 0, "empty_rows": 0, "boxes": 0} for task in sorted(DETECTION_TASKS)}
    for index, record in enumerate(records):
        task = str(record.get("task_type"))
        if task not in out:
            continue
        _, answer = prompt_and_response(record, context=f"gate record[{index}]")
        boxes = _json_answer(answer, context=f"gate record[{index}]")
        if not isinstance(boxes, list):
            raise ValueError(f"gate record[{index}]: detection answer must be a list")
        out[task]["rows"] += 1
        out[task]["empty_rows"] += not boxes
        out[task]["boxes"] += len(boxes)
    for task, payload in out.items():
        e, b = payload["empty_rows"], payload["boxes"]
        payload["trivial_f1"] = (2 * e / (2 * e + b)) if (e or b) else 0.0
    return out


def checkpoint_f1(report: dict[str, Any], task: str) -> float | None:
    cohort, family = TASK_REPORT_KEY[task]
    cohorts = report.get("tasks_by_reference_cohort") or {}
    tasks = cohorts.get(cohort, {}).get("tasks", cohorts.get(cohort, {}))
    value = tasks.get(family)
    if not isinstance(value, dict):
        return None
    f1 = value.get("macro_f1", value.get("f1"))
    return float(f1) if f1 is not None else None


def decide(baselines: dict[str, dict[str, Any]], report: dict[str, Any], *, margin: float,
           acquisition_rows: int, refinement_rows: dict[str, int] | None = None) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    for task, base in baselines.items():
        f1 = checkpoint_f1(report, task)
        if f1 is None or base["rows"] == 0:
            decisions[task] = {"mode": "unknown", "reason": "no KPI rows or no evaluator score"}
            continue
        # the evaluator report scores the whole cohort; Component Detection shares the
        # single-image DET cohort with Defect Detection, so its gate is informational only
        below = f1 <= base["trivial_f1"] + margin
        decisions[task] = {
            "checkpoint_f1": round(f1, 4),
            "trivial_f1": round(base["trivial_f1"], 4),
            "margin": margin,
            "mode": "acquisition" if below else "refinement",
            "recommended_rows": acquisition_rows if below else (refinement_rows or {}).get(task),
            "cohort_scored": TASK_REPORT_KEY[task][0],
        }
    return decisions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kpi-annotations", required=True, type=pathlib.Path)
    parser.add_argument("--raw-report", required=True, type=pathlib.Path, help="evaluator raw F1 report of the gating checkpoint (zero-shot for iteration 1)")
    parser.add_argument("--margin", type=float, default=0.0, help="F1 margin above the trivial baseline that still counts as acquisition")
    parser.add_argument("--acquisition-rows", type=int, default=3000, help="calibration rows per iteration recommended for a task in acquisition mode")
    parser.add_argument("--refinement-rows", action="append", default=[], metavar="TASK=ROWS")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        refinement = {}
        for value in args.refinement_rows:
            task, sep, rows = value.partition("=")
            if not sep:
                raise ValueError(f"invalid --refinement-rows {value!r}")
            refinement[task] = int(rows)
        baselines = trivial_baselines(load_records(args.kpi_annotations.expanduser().resolve(strict=True)))
        report = json.loads(args.raw_report.read_text(encoding="utf-8"))
        decisions = decide(baselines, report, margin=args.margin, acquisition_rows=args.acquisition_rows, refinement_rows=refinement)
        payload = {
            "schema_version": "capability_gate_v1",
            "rule": "acquisition when checkpoint F1 <= trivial always-empty F1 + margin; else refinement",
            "kpi_annotations": str(args.kpi_annotations.resolve()),
            "raw_report": str(args.raw_report.resolve()),
            "trivial_baselines": baselines,
            "decisions": decisions,
            "acquisition_tasks": sorted(t for t, d in decisions.items() if d.get("mode") == "acquisition"),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"capability_gate: {exc}", file=sys.stderr)
        return 2
    for task, d in payload["decisions"].items():
        print(f"capability_gate: {task}: {d.get('mode')} (f1={d.get('checkpoint_f1')} trivial={d.get('trivial_f1')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
