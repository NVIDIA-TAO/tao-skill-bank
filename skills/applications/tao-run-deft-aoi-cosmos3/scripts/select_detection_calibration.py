#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Prepend empty/few-box Mining examples to routed candidates for calibration."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections.abc import Iterable, Iterator
from typing import Any

from atomic_samples import embedding_filepath, sample_from_record
from validate_sharegpt import prompt_and_response, resolve_image, target_path


DETECTION_TASKS = {
    "Component Detection",
    "Defect Detection",
    "Ref_based Defect Detection",
}
DETECTION_COHORTS = {
    "non_reference_based": "Defect Detection",
    "reference_based": "Ref_based Defect Detection",
}
TASK_TO_COHORT = {task: cohort for cohort, task in DETECTION_COHORTS.items()}
CALIBRATION_EMPTY_EVIDENCE = "calibration_empty_ground_truth"
CALIBRATION_FEW_EVIDENCE = "calibration_few_box_ground_truth"
REFERENCE_NO_CHANGE_EVIDENCE = "calibration_reference_no_change_ground_truth"


def select_component_count_replay(
    records: Iterable[dict[str, Any]],
    *,
    media_root: pathlib.Path,
    max_count: int,
    excluded_identities: set[str] | None = None,
    allow_empty: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select bounded real Component Count rows independently of KPI gap routing."""

    if max_count <= 0:
        raise ValueError("max_count must be positive")
    media_root = media_root.expanduser().resolve()
    excluded = {
        str(pathlib.Path(value).expanduser().resolve())
        for value in (excluded_identities or set())
    }
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    excluded_previously_mined = 0
    examined = 0
    for index, record in enumerate(records):
        if record.get("task_type") != "Component Count":
            continue
        examined += 1
        prompt_and_response(record, context=f"component-count record[{index}]")
        filepath = target_path(record, context=f"component-count record[{index}]")
        identity = str(resolve_image(filepath, media_root))
        if identity in excluded:
            excluded_previously_mined += 1
            continue
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(
            {
                "filepath": filepath,
                "route_tier": "count_replay",
                "route_tiers": ["count_replay"],
                "routed_task_types": ["Component Count"],
                "count_record_id": record.get("id"),
            }
        )
        if len(selected) >= max_count:
            break
    if not selected and not allow_empty:
        raise ValueError("no eligible Component Count replay examples were found")
    summary = {
        "schema_version": "component_count_replay_v1",
        "policy": "bounded_real_mining_replay",
        "requested_component_count": max_count,
        "selected_component_count": len(selected),
        "examined_component_count_records": examined,
        "excluded_previously_mined": excluded_previously_mined,
    }
    if not selected:
        summary["empty_reason"] = "eligible_tier_exhausted"
    return selected, summary


def _json_answer(text: str, *, context: str) -> Any:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            value = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{context}: detection answer is not JSON") from exc


def derive_proxy_empty_rates(
    records: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Derive the exact empty-ground-truth rate for each detection cohort."""

    counts = {
        cohort: {"task_type": task, "empty_rows": 0, "total_rows": 0}
        for cohort, task in DETECTION_COHORTS.items()
    }
    for index, record in enumerate(records):
        cohort = TASK_TO_COHORT.get(str(record.get("task_type")))
        if cohort is None:
            continue
        _, answer = prompt_and_response(record, context=f"proxy record[{index}]")
        boxes = _json_answer(answer, context=f"proxy record[{index}]")
        if not isinstance(boxes, list):
            raise ValueError(f"proxy record[{index}]: detection answer must be a list")
        counts[cohort]["total_rows"] += 1
        counts[cohort]["empty_rows"] += not boxes
    missing = [
        DETECTION_COHORTS[cohort]
        for cohort, payload in counts.items()
        if payload["total_rows"] == 0
    ]
    if missing:
        raise ValueError(
            "Proxy has no detection rows for required task type(s): "
            + ", ".join(missing)
        )
    for payload in counts.values():
        payload["empty_rate"] = payload["empty_rows"] / payload["total_rows"]
    return counts


def _rounded_rate_quota(total: int, rate: float) -> int:
    if total < 0 or not 0.0 <= rate <= 1.0:
        raise ValueError("cohort totals must be non-negative and rates must be in [0, 1]")
    return math.floor(total * rate + 0.5)


def _excluded_identity(value: str) -> str:
    if value.startswith(("single_image:", "reference_pair:")):
        return value
    return str(pathlib.Path(value).expanduser().resolve())


def _calibration_row(
    record: dict[str, Any],
    *,
    sample: dict[str, Any],
    filepath: str,
    task_type: str,
    box_count: int,
) -> dict[str, Any]:
    evidence = [
        CALIBRATION_EMPTY_EVIDENCE
        if box_count == 0
        else CALIBRATION_FEW_EVIDENCE
    ]
    if sample["sample_kind"] == "reference_pair" and box_count == 0:
        evidence.append(REFERENCE_NO_CHANGE_EVIDENCE)
    return {
        "filepath": filepath,
        "atomic_sample_id": sample["atomic_sample_id"],
        "sample_kind": sample["sample_kind"],
        "source_image_paths": sample["image_paths"],
        "source_reference_filepath": sample["reference_filepath"],
        "source_target_filepath": sample["target_filepath"],
        "route_tier": "calibration",
        "route_tiers": ["calibration"],
        "route_tier_by_task": {task_type: "calibration"},
        "routed_task_types": [task_type],
        "defect_detection_evidence": evidence,
        "calibration_box_count": box_count,
        "calibration_record_id": record.get("id"),
    }


def select_calibration(
    records: Iterable[dict[str, Any]],
    *,
    media_root: pathlib.Path,
    max_empty: int | None = None,
    max_few: int | None = None,
    max_boxes: int = 2,
    excluded_identities: set[str] | None = None,
    cohort_quotas: dict[str, int] | None = None,
    cohort_bucket_quotas: dict[str, dict[str, int]] | None = None,
    cohort_rates: dict[str, dict[str, Any]] | None = None,
    pair_assets_dir: pathlib.Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_boxes < 1:
        raise ValueError("max_boxes must be positive")
    cohort_mode = (
        cohort_quotas is not None
        or cohort_bucket_quotas is not None
        or cohort_rates is not None
    )
    if cohort_mode:
        if cohort_rates is None or (cohort_quotas is None) == (cohort_bucket_quotas is None):
            raise ValueError(
                "cohort_rates and exactly one cohort quota contract must be supplied"
            )
        if max_empty is not None or max_few is not None:
            raise ValueError("legacy and per-cohort calibration quotas cannot be combined")
        if set(cohort_rates) != set(DETECTION_COHORTS):
            raise ValueError("cohort_rates must define both detection cohorts exactly")
        targets: dict[str, dict[str, int | float]] = {}
        if cohort_bucket_quotas is not None:
            if set(cohort_bucket_quotas) != set(DETECTION_COHORTS):
                raise ValueError(
                    "cohort_bucket_quotas must define both detection cohorts exactly"
                )
            for cohort in DETECTION_COHORTS:
                quota = cohort_bucket_quotas[cohort]
                if set(quota) != {"empty", "few"} or any(
                    type(value) is not int or value < 0 for value in quota.values()
                ):
                    raise ValueError(
                        "every cohort bucket quota requires non-negative integer empty/few values"
                    )
                rate = cohort_rates[cohort].get("empty_rate")
                if not isinstance(rate, (int, float)):
                    raise ValueError(f"{cohort} requires a numeric empty_rate")
                targets[cohort] = {
                    "total": quota["empty"] + quota["few"],
                    "empty": quota["empty"],
                    "few": quota["few"],
                    "rate": float(rate),
                }
        else:
            assert cohort_quotas is not None
            if set(cohort_quotas) != set(DETECTION_COHORTS):
                raise ValueError("cohort_quotas must define both detection cohorts exactly")
            for cohort in DETECTION_COHORTS:
                total = cohort_quotas[cohort]
                if type(total) is not int or total < 0:
                    raise ValueError("every per-cohort calibration quota must be non-negative")
                rate = cohort_rates[cohort].get("empty_rate")
                if not isinstance(rate, (int, float)):
                    raise ValueError(f"{cohort} requires a numeric empty_rate")
                empty_target = _rounded_rate_quota(total, float(rate))
                targets[cohort] = {
                    "total": total,
                    "empty": empty_target,
                    "few": total - empty_target,
                    "rate": float(rate),
                }
        if sum(item["total"] for item in targets.values()) <= 0:
            raise ValueError("at least one per-cohort calibration quota must be positive")
    else:
        if max_empty is None or max_few is None:
            raise ValueError("max_empty and max_few are required for legacy calibration")
        if min(max_empty, max_few) < 0:
            raise ValueError("calibration quotas must be non-negative")
        if max_empty + max_few <= 0:
            raise ValueError("at least one calibration quota must be positive")
    media_root = media_root.expanduser().resolve()
    excluded = {_excluded_identity(value) for value in (excluded_identities or set())}
    empty: list[dict[str, Any]] = []
    few: list[dict[str, Any]] = []
    cohort_selected = {
        cohort: {"empty": [], "few": []} for cohort in DETECTION_COHORTS
    }
    seen: set[str] = set()
    examined_detection = 0
    excluded_many = 0
    excluded_previously_mined = 0
    for index, record in enumerate(records):
        task_type = record.get("task_type")
        if task_type not in DETECTION_TASKS:
            continue
        examined_detection += 1
        _, answer = prompt_and_response(record, context=f"calibration record[{index}]")
        boxes = _json_answer(answer, context=f"calibration record[{index}]")
        if not isinstance(boxes, list):
            raise ValueError(f"calibration record[{index}]: detection answer must be a list")
        if len(boxes) > max_boxes:
            excluded_many += 1
            continue
        sample = sample_from_record(
            record, media_root=media_root, context=f"calibration record[{index}]"
        )
        identity = str(sample["atomic_sample_id"])
        target_identity = str(sample["target_filepath"])
        if identity in excluded or target_identity in excluded:
            excluded_previously_mined += 1
            continue
        if identity in seen:
            continue
        if cohort_mode:
            cohort = TASK_TO_COHORT.get(str(task_type))
            if cohort is None:
                continue
            bucket = "empty" if not boxes else "few"
            destination = cohort_selected[cohort][bucket]
            quota = int(targets[cohort][bucket])
        else:
            destination = empty if not boxes else few
            quota = max_empty if not boxes else max_few
        if len(destination) >= quota:
            continue
        filepath = embedding_filepath(sample, pair_assets_dir=pair_assets_dir)
        seen.add(identity)
        destination.append(
            _calibration_row(
                record,
                sample=sample,
                filepath=filepath,
                task_type=str(task_type),
                box_count=len(boxes),
            )
        )
        if cohort_mode and all(
            len(cohort_selected[cohort][bucket]) >= int(targets[cohort][bucket])
            for cohort in DETECTION_COHORTS
            for bucket in ("empty", "few")
        ):
            break
        if (
            not cohort_mode
            and len(empty) >= max_empty
            and len(few) >= max_few
        ):
            break
    if cohort_mode:
        selected = [
            row
            for cohort in DETECTION_COHORTS
            for bucket in ("empty", "few")
            for row in cohort_selected[cohort][bucket]
        ]
        shortages = {
            cohort: {
                bucket: int(targets[cohort][bucket])
                - len(cohort_selected[cohort][bucket])
                for bucket in ("empty", "few")
                if len(cohort_selected[cohort][bucket])
                < int(targets[cohort][bucket])
            }
            for cohort in DETECTION_COHORTS
        }
        shortages = {key: value for key, value in shortages.items() if value}
        if shortages:
            raise ValueError(f"per-cohort calibration quotas cannot be filled: {shortages}")
        hybrid_mode = cohort_bucket_quotas is not None
        summary = {
            "schema_version": "detection_calibration_v3" if hybrid_mode else "detection_calibration_v2",
            "policy": (
                "fixed_single_image_proxy_rate_reference"
                if hybrid_mode
                else "proxy_empty_rate_by_reference_cohort"
            ),
            "max_boxes": max_boxes,
            "selected_total": len(selected),
            "examined_detection_records": examined_detection,
            "excluded_many_box": excluded_many,
            "excluded_previously_mined": excluded_previously_mined,
            "cohorts": {
                cohort: {
                    "task_type": DETECTION_COHORTS[cohort],
                    "proxy_empty_rate": targets[cohort]["rate"],
                    "proxy_empty_rate_binding": (
                        not hybrid_mode or cohort == "reference_based"
                    ),
                    "requested_total": targets[cohort]["total"],
                    "requested_empty": targets[cohort]["empty"],
                    "requested_few_box": targets[cohort]["few"],
                    "selected_empty": len(cohort_selected[cohort]["empty"]),
                    "selected_few_box": len(cohort_selected[cohort]["few"]),
                    "selected_total": len(cohort_selected[cohort]["empty"])
                    + len(cohort_selected[cohort]["few"]),
                }
                for cohort in DETECTION_COHORTS
            },
        }
        return selected, summary
    selected = [*empty, *few]
    if not selected:
        raise ValueError("no empty or few-box Mining calibration examples were found")
    return selected, {
        "schema_version": "detection_calibration_v1",
        "policy": "empty_and_few_box_from_mining",
        "max_boxes": max_boxes,
        "requested_empty": max_empty,
        "requested_few_box": max_few,
        "selected_empty": len(empty),
        "selected_few_box": len(few),
        "selected_total": len(selected),
        "examined_detection_records": examined_detection,
        "excluded_many_box": excluded_many,
        "excluded_previously_mined": excluded_previously_mined,
    }


def _stream_records(path: pathlib.Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            yield value


def _routed_rows(path: pathlib.Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("pyarrow is required to merge routed candidates") from exc
    return pq.read_table(path).to_pylist()


def merge_candidates(
    calibration: list[dict[str, Any]],
    routed: list[dict[str, Any]],
    *,
    media_root: pathlib.Path,
) -> tuple[list[dict[str, Any]], int]:
    output: list[dict[str, Any]] = []
    indexes: dict[str, int] = {}
    duplicates = 0
    for row in [*calibration, *routed]:
        filepath = row.get("filepath")
        if not isinstance(filepath, str) or not filepath:
            raise ValueError("every calibration/routed candidate requires filepath")
        atomic_sample_id = row.get("atomic_sample_id")
        identity = (
            atomic_sample_id
            if isinstance(atomic_sample_id, str) and atomic_sample_id
            else str(resolve_image(filepath, media_root))
        )
        tasks = row.get("routed_task_types") or row.get("source_task_types")
        if not isinstance(tasks, (list, tuple)) or not tasks:
            raise ValueError(f"candidate {filepath!r} requires routed_task_types")
        tier = str(row.get("route_tier") or "strict")
        if identity not in indexes:
            indexes[identity] = len(output)
            merged_row = dict(row)
            merged_row["route_tier"] = tier
            merged_row["route_tiers"] = sorted(
                set(row.get("route_tiers") or [tier])
            )
            merged_row["routed_task_types"] = sorted(
                set(str(item) for item in tasks)
            )
            merged_row["route_tier_by_task"] = {
                str(task): str((row.get("route_tier_by_task") or {}).get(task, tier))
                for task in merged_row["routed_task_types"]
            }
            output.append(merged_row)
            continue
        duplicates += 1
        existing = output[indexes[identity]]
        existing["route_tiers"] = sorted(set(existing["route_tiers"]) | {tier})
        existing["routed_task_types"] = sorted(
            set(existing["routed_task_types"]) | {str(item) for item in tasks}
        )
        task_tiers = dict(existing.get("route_tier_by_task") or {})
        incoming_tiers = dict(row.get("route_tier_by_task") or {})
        for task in tasks:
            task = str(task)
            incoming_tier = str(incoming_tiers.get(task, tier))
            if task_tiers.get(task) == "calibration" or incoming_tier == "calibration":
                task_tiers[task] = "calibration"
            else:
                task_tiers[task] = incoming_tier
        existing["route_tier_by_task"] = task_tiers
        for field in ("defect_detection_evidence", "source_task_types"):
            existing[field] = sorted(
                set(existing.get(field) or []) | set(row.get(field) or [])
            )
        for field in ("sample_kind", "source_image_paths"):
            if field in row and field in existing and row[field] != existing[field]:
                raise ValueError(f"atomic candidate has conflicting {field}: {identity}")
        if "calibration" in existing["route_tiers"]:
            existing["route_tier"] = "calibration"
    return output, duplicates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-annotations", required=True, type=pathlib.Path)
    parser.add_argument("--media-root", required=True, type=pathlib.Path)
    parser.add_argument("--routed-candidates", type=pathlib.Path)
    parser.add_argument("--proxy-annotations", type=pathlib.Path)
    parser.add_argument("--pair-assets-dir", type=pathlib.Path)
    parser.add_argument("--single-image-total", type=int)
    parser.add_argument("--reference-total", type=int)
    parser.add_argument("--single-image-max-empty", type=int)
    parser.add_argument("--single-image-max-few", type=int)
    parser.add_argument("--max-empty", type=int)
    parser.add_argument("--max-few", type=int)
    parser.add_argument("--max-boxes", type=int, default=2)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--summary", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        cohort_mode = any(
            value is not None
            for value in (
                args.proxy_annotations,
                args.single_image_total,
                args.reference_total,
                args.single_image_max_empty,
                args.single_image_max_few,
            )
        )
        hybrid_mode = any(
            value is not None
            for value in (args.single_image_max_empty, args.single_image_max_few)
        )
        if hybrid_mode and args.single_image_total is not None:
            raise ValueError(
                "--single-image-total cannot be combined with fixed single-image caps"
            )
        required = [args.proxy_annotations, args.reference_total, args.pair_assets_dir]
        required.extend(
            [args.single_image_max_empty, args.single_image_max_few]
            if hybrid_mode
            else [args.single_image_total]
        )
        if cohort_mode and any(value is None for value in required):
            raise ValueError(
                "per-cohort calibration requires --proxy-annotations, "
                "--reference-total, --pair-assets-dir, and either "
                "--single-image-total or both fixed single-image caps"
            )
        rates = (
            derive_proxy_empty_rates(_stream_records(args.proxy_annotations))
            if cohort_mode
            else None
        )
        calibration, summary = select_calibration(
            _stream_records(args.source_annotations),
            media_root=args.media_root,
            max_empty=args.max_empty,
            max_few=args.max_few,
            max_boxes=args.max_boxes,
            cohort_quotas=(
                {
                    "non_reference_based": args.single_image_total,
                    "reference_based": args.reference_total,
                }
                if cohort_mode and not hybrid_mode
                else None
            ),
            cohort_bucket_quotas=(
                {
                    "non_reference_based": {
                        "empty": args.single_image_max_empty,
                        "few": args.single_image_max_few,
                    },
                    "reference_based": {
                        "empty": _rounded_rate_quota(
                            args.reference_total,
                            float(rates["reference_based"]["empty_rate"]),
                        ),
                        "few": args.reference_total
                        - _rounded_rate_quota(
                            args.reference_total,
                            float(rates["reference_based"]["empty_rate"]),
                        ),
                    },
                }
                if cohort_mode and hybrid_mode
                else None
            ),
            cohort_rates=rates,
            pair_assets_dir=args.pair_assets_dir,
        )
        merged, duplicates = merge_candidates(
            calibration,
            _routed_rows(args.routed_candidates),
            media_root=args.media_root,
        )
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ValueError("pyarrow is required to write calibration candidates") from exc
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(merged), args.output)
        summary.update(
            {
                "routed_input": str(args.routed_candidates) if args.routed_candidates else None,
                "routed_records": len(merged) - len(calibration) + duplicates,
                "combined_unique_candidates": len(merged),
                "duplicates_merged": duplicates,
                "output": str(args.output),
            }
        )
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"select_detection_calibration: {exc}", file=sys.stderr)
        return 2
    print(
        "select_detection_calibration: "
        f"calibration={summary['selected_total']} combined={summary['combined_unique_candidates']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
