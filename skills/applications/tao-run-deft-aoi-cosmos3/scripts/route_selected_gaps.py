#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collapse record-level selected gaps into target-level mining queries."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from typing import Any


DEFECT_DETECTION_TASK = "Defect Detection"
DEFECT_DETECTION_MINING_EVIDENCE = {
    "hard_negative_proxy_false_positive",
    "hard_positive_best_overlap_0_lt_iou_lte_0p5",
    "hard_positive_proxy_false_negative",
}


def augment_defect_detection_targets(
    selected_rows: list[dict[str, Any]],
    all_gap_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add every proxy DD row with approved hard-mining evidence."""

    output = list(selected_rows)
    selected_ids = {str(row.get("id")) for row in selected_rows}
    eligible: list[dict[str, Any]] = []
    evidence_counts: Counter[str] = Counter()
    for row in all_gap_rows:
        if row.get("evaluation_role") != "proxy" or row.get("task_type") != DEFECT_DETECTION_TASK:
            continue
        evidence = row.get("defect_detection_evidence")
        evidence_types = evidence.get("evidence_types", []) if isinstance(evidence, dict) else []
        approved = sorted(DEFECT_DETECTION_MINING_EVIDENCE.intersection(evidence_types))
        if not approved:
            continue
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("every DD supplement row requires a non-empty id")
        eligible.append(row)
        evidence_counts.update(approved)
        if row["id"] not in selected_ids:
            output.append(row)
            selected_ids.add(row["id"])
    return output, {
        "schema_version": "defect_detection_routing_supplement_v1",
        "selected_input_rows": len(selected_rows),
        "eligible_defect_detection_rows": len(eligible),
        "supplemental_rows": len(output) - len(selected_rows),
        "output_rows": len(output),
        "approved_evidence_types": sorted(DEFECT_DETECTION_MINING_EVIDENCE),
        "eligible_evidence_counts": dict(sorted(evidence_counts.items())),
    }


def route(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        raise ValueError("selected gaps are empty")
    targets: dict[str, dict[str, Any]] = {}
    task_counts: Counter[str] = Counter()
    for row in rows:
        if row.get("evaluation_role") != "proxy":
            raise ValueError("routing accepts only Proxy selected gaps")
        record_id = row.get("id")
        target_id = row.get("target_id")
        target_path = row.get("target_path")
        task_type = row.get("task_type")
        if not all(isinstance(value, str) and value for value in (record_id, target_id, target_path, task_type)):
            raise ValueError("every selected gap requires id, target_id, target_path, and task_type")
        task_counts[task_type] += 1
        target = targets.setdefault(
            target_id,
            {
                "filepath": target_path,
                "target_id": target_id,
                "record_ids": [],
                "task_types": [],
                "datasets": [],
                "defect_detection_evidence": [],
                "mining_eligible": True,
            },
        )
        if target["filepath"] != target_path:
            raise ValueError(
                f"target_id {target_id!r} maps to conflicting target paths"
            )
        target["record_ids"].append(record_id)
        if task_type not in target["task_types"]:
            target["task_types"].append(task_type)
        dataset = str(row.get("dataset", "unknown"))
        if dataset not in target["datasets"]:
            target["datasets"].append(dataset)
        evidence = row.get("defect_detection_evidence", {})
        if isinstance(evidence, dict):
            for evidence_type in evidence.get("evidence_types", []):
                if evidence_type not in target["defect_detection_evidence"]:
                    target["defect_detection_evidence"].append(evidence_type)
    output = []
    for target in targets.values():
        target["record_ids"].sort()
        target["task_types"].sort()
        target["datasets"].sort()
        target["defect_detection_evidence"].sort()
        output.append(target)
    output.sort(key=lambda row: (row["filepath"], row["target_id"]))
    return output, {
        "schema_version": "nvpaw_routing_v1",
        "selected_records": len(rows),
        "unique_targets": len(output),
        "embedding_queries": len(output),
        "task_records": dict(sorted(task_counts.items())),
        "mining_eligible_records": len(rows),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-gaps", required=True, type=pathlib.Path)
    parser.add_argument("--defect-detection-supplement", type=pathlib.Path)
    parser.add_argument("--supplement-summary", type=pathlib.Path)
    parser.add_argument("--output-json", required=True, type=pathlib.Path)
    parser.add_argument("--output-parquet", required=True, type=pathlib.Path)
    parser.add_argument("--summary", required=True, type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if bool(args.defect_detection_supplement) != bool(args.supplement_summary):
            raise ValueError(
                "--defect-detection-supplement and --supplement-summary are required together"
            )
        rows = pq.read_table(args.selected_gaps).to_pylist()
        supplement_summary = None
        if args.defect_detection_supplement:
            rows, supplement_summary = augment_defect_detection_targets(
                rows,
                pq.read_table(args.defect_detection_supplement).to_pylist(),
            )
        targets, summary = route(rows)
        if supplement_summary is not None:
            summary["defect_detection_supplement"] = supplement_summary
        for path in (args.output_json, args.output_parquet, args.summary):
            path.parent.mkdir(parents=True, exist_ok=True)
        if args.supplement_summary:
            args.supplement_summary.parent.mkdir(parents=True, exist_ok=True)
            args.supplement_summary.write_text(
                json.dumps(supplement_summary, indent=2, sort_keys=True) + "\n"
            )
        args.output_json.write_text(json.dumps(targets, indent=2, sort_keys=True) + "\n")
        pq.write_table(pa.Table.from_pylist(targets), args.output_parquet)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    except (ImportError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"route_selected_gaps: {exc}", file=sys.stderr)
        return 2
    print(
        f"route_selected_gaps: records={summary['selected_records']} "
        f"targets={summary['unique_targets']} output={args.output_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
