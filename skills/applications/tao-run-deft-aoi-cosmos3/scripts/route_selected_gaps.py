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

from atomic_samples import embedding_filepath, sample_from_record

DEFECT_DETECTION_TASK = "Defect Detection"
DEFECT_DETECTION_MINING_EVIDENCE = {
    "hard_negative_proxy_false_positive",
    "hard_positive_best_overlap_0_lt_iou_lte_0p5",
    "hard_positive_proxy_false_negative",
}
DEFECT_DETECTION_ANCHOR_POLICIES = ("hard_only", "all_proxy_severity")
CORRECT_DEFECT_DETECTION_EVIDENCE = "proxy_correct"


def _defect_detection_severity(row: dict[str, Any]) -> tuple[int, int, int, int, str, str]:
    evidence = row.get("defect_detection_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    evidence_types = set(evidence.get("evidence_types", []))

    def count(name: str) -> int:
        value = evidence.get(name, 0)
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0

    false_negatives = count("false_negative_count")
    partial_overlaps = count("best_overlap_0_lt_iou_lte_0p5_count")
    false_positives = count("false_positive_count")
    if false_negatives or partial_overlaps or evidence_types.intersection(
        {
            "hard_positive_proxy_false_negative",
            "hard_positive_best_overlap_0_lt_iou_lte_0p5",
        }
    ):
        severity = "false_negative_or_partial_overlap"
        priority = 0
    elif false_positives or "hard_negative_proxy_false_positive" in evidence_types:
        severity = "false_positive"
        priority = 1
    else:
        severity = "correct"
        priority = 2
    return (
        priority,
        -false_negatives,
        -partial_overlaps,
        -false_positives,
        str(row.get("id", "")),
        severity,
    )


def augment_defect_detection_targets(
    selected_rows: list[dict[str, Any]],
    all_gap_rows: list[dict[str, Any]],
    *,
    anchor_policy: str = "hard_only",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add launch-selected Proxy DD anchors without weakening task isolation."""

    if anchor_policy not in DEFECT_DETECTION_ANCHOR_POLICIES:
        raise ValueError(
            f"unsupported Defect Detection anchor policy {anchor_policy!r}; "
            f"choose one of {DEFECT_DETECTION_ANCHOR_POLICIES}"
        )

    if anchor_policy == "all_proxy_severity":
        anchors: list[tuple[tuple[int, int, int, int, str, str], dict[str, Any]]] = []
        seen_ids: set[str] = set()
        for row in all_gap_rows:
            if row.get("evaluation_role") != "proxy" or row.get("task_type") != DEFECT_DETECTION_TASK:
                continue
            if not isinstance(row.get("id"), str) or not row["id"]:
                raise ValueError("every DD anchor row requires a non-empty id")
            if row["id"] in seen_ids:
                raise ValueError(f"duplicate DD anchor id: {row['id']!r}")
            seen_ids.add(row["id"])
            sort_key = _defect_detection_severity(row)
            anchor = dict(row)
            evidence = anchor.get("defect_detection_evidence")
            evidence = dict(evidence) if isinstance(evidence, dict) else {}
            evidence_types = list(evidence.get("evidence_types", []))
            if sort_key[-1] == "correct" and CORRECT_DEFECT_DETECTION_EVIDENCE not in evidence_types:
                evidence_types.append(CORRECT_DEFECT_DETECTION_EVIDENCE)
            evidence["evidence_types"] = sorted(set(evidence_types))
            anchor["defect_detection_evidence"] = evidence
            anchor["defect_detection_anchor_priority"] = sort_key[0]
            anchor["defect_detection_anchor_severity"] = sort_key[-1]
            anchors.append((sort_key, anchor))
        anchors.sort(key=lambda item: item[0][:-1])
        ordered_anchors = [row for _, row in anchors]
        maintenance = [
            row for row in selected_rows if row.get("task_type") != DEFECT_DETECTION_TASK
        ]
        severity_counts = Counter(
            row["defect_detection_anchor_severity"] for row in ordered_anchors
        )
        output = ordered_anchors + maintenance
        return output, {
            "schema_version": "defect_detection_routing_supplement_v1",
            "anchor_policy": anchor_policy,
            "selected_input_rows": len(selected_rows),
            "eligible_defect_detection_rows": len(ordered_anchors),
            "supplemental_rows": max(0, len(output) - len(selected_rows)),
            "output_rows": len(output),
            "severity_order": [
                "false_negative_or_partial_overlap",
                "false_positive",
                "correct",
            ],
            "severity_counts": dict(sorted(severity_counts.items())),
            "approved_evidence_types": sorted(
                {*DEFECT_DETECTION_MINING_EVIDENCE, CORRECT_DEFECT_DETECTION_EVIDENCE}
            ),
        }

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
        "anchor_policy": anchor_policy,
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
        paths = row.get("image_paths")
        if not isinstance(paths, list) or not all(
            isinstance(value, str) and value for value in paths
        ):
            reference_path = row.get("reference_path")
            paths = (
                [reference_path, target_path]
                if isinstance(reference_path, str) and reference_path
                else [target_path]
            )
        atomic_sample_id = row.get("atomic_sample_id")
        if not isinstance(atomic_sample_id, str) or not atomic_sample_id:
            atomic_sample_id = f"legacy_target:{target_id}"
        target = targets.setdefault(
            atomic_sample_id,
            {
                "filepath": target_path,
                "target_id": target_id,
                "atomic_sample_id": atomic_sample_id,
                "sample_kind": "reference_pair" if len(paths) == 2 else "single_image",
                "image_paths": list(paths),
                "reference_filepath": paths[0] if len(paths) == 2 else None,
                "target_filepath": target_path,
                "record_ids": [],
                "task_types": [],
                "datasets": [],
                "defect_detection_evidence": [],
                "defect_detection_anchor_priority": row.get(
                    "defect_detection_anchor_priority"
                ),
                "defect_detection_anchor_severity": row.get(
                    "defect_detection_anchor_severity"
                ),
                "mining_eligible": True,
            },
        )
        if target["image_paths"] != paths:
            raise ValueError(
                f"atomic_sample_id {atomic_sample_id!r} maps to conflicting image paths"
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
        priority = row.get("defect_detection_anchor_priority")
        if isinstance(priority, int) and (
            target["defect_detection_anchor_priority"] is None
            or priority < target["defect_detection_anchor_priority"]
        ):
            target["defect_detection_anchor_priority"] = priority
            target["defect_detection_anchor_severity"] = row.get(
                "defect_detection_anchor_severity"
            )
    output = []
    for target in targets.values():
        target["record_ids"].sort()
        target["task_types"].sort()
        target["datasets"].sort()
        target["defect_detection_evidence"].sort()
        output.append(target)
    return output, {
        "schema_version": "nvpaw_routing_v1",
        "selected_records": len(rows),
        "unique_targets": len(output),
        "embedding_queries": len(output),
        "task_records": dict(sorted(task_counts.items())),
        "mining_eligible_records": len(rows),
    }


def materialize_embedding_inputs(
    targets: list[dict[str, Any]],
    *,
    media_root: pathlib.Path,
    pair_assets_dir: pathlib.Path,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        record = {
            "task_type": target["task_types"][0],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        *(
                            {"type": "image", "image": value}
                            for value in target["image_paths"]
                        ),
                        {"type": "text", "text": "atomic embedding query"},
                    ],
                },
                {"role": "assistant", "content": "embedding-only"},
            ],
        }
        sample = sample_from_record(
            record, media_root=media_root, context=f"embedding target[{index}]"
        )
        materialized = dict(target)
        materialized.update(sample)
        materialized["filepath"] = embedding_filepath(
            sample, pair_assets_dir=pair_assets_dir
        )
        output.append(materialized)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-gaps", required=True, type=pathlib.Path)
    parser.add_argument("--defect-detection-supplement", type=pathlib.Path)
    parser.add_argument("--supplement-summary", type=pathlib.Path)
    parser.add_argument(
        "--defect-detection-anchor-policy",
        choices=DEFECT_DETECTION_ANCHOR_POLICIES,
        default="hard_only",
    )
    parser.add_argument("--output-json", required=True, type=pathlib.Path)
    parser.add_argument("--output-parquet", required=True, type=pathlib.Path)
    parser.add_argument("--summary", required=True, type=pathlib.Path)
    parser.add_argument("--media-root", required=True, type=pathlib.Path)
    parser.add_argument("--pair-assets-dir", required=True, type=pathlib.Path)
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
                anchor_policy=args.defect_detection_anchor_policy,
            )
        targets, summary = route(rows)
        targets = materialize_embedding_inputs(
            targets,
            media_root=args.media_root.expanduser().resolve(),
            pair_assets_dir=args.pair_assets_dir,
        )
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
