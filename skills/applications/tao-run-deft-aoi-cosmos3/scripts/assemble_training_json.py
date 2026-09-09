#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Assemble monotonic, real-mining-only NVPAW training JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from collections import Counter
from typing import Any

from atomic_samples import logical_record_identity, sample_from_record
from defect_detection_ablation import bind_cumulative_manifest
from nvpaw_annotations import TASK_SPECS
from repetition_blend import (
    POLICIES as REPETITION_POLICIES,
    apply_repetition_blend,
    bind_repetition_manifest,
    deficit_weights_from_gap_summary,
    load_repetition_config,
    merge_repetition_config,
    parse_explicit_multipliers,
    validate_repetition_config,
)
from validate_sharegpt import load_records, target_path


def _fingerprint(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_set_sha256(fingerprints: set[str]) -> str:
    digest = hashlib.sha256()
    for fingerprint in sorted(fingerprints):
        digest.update(hashlib.sha256(fingerprint.encode("utf-8")).digest())
    return digest.hexdigest()


def _task_balanced_indices(
    indices: list[int], records: list[dict[str, Any]], limit: int
) -> list[int]:
    """Select deterministically across tasks while preserving order within each task."""

    if limit >= len(indices):
        return indices
    groups: dict[str, list[int]] = {}
    for index in indices:
        task = str(records[index].get("task_type") or "unknown")
        groups.setdefault(task, []).append(index)
    positions = {task: 0 for task in groups}
    selected: list[int] = []
    while len(selected) < limit:
        advanced = False
        for task in sorted(groups):
            position = positions[task]
            if position >= len(groups[task]):
                continue
            selected.append(groups[task][position])
            positions[task] = position + 1
            advanced = True
            if len(selected) == limit:
                break
        if not advanced:
            break
    if len(selected) != limit:
        raise ValueError(
            f"task-balanced materialization selected {len(selected)} rows, expected {limit}"
        )
    return selected


def assemble(
    previous_path: pathlib.Path | None,
    mined_path: pathlib.Path,
    *,
    previous_sha256: str | None = None,
    validation_paths: list[pathlib.Path],
    max_rows: int | None = None,
    row_multiple: int | None = None,
    repetition_config: dict[str, Any] | None = None,
    deficit_weights: dict[str, float] | None = None,
    repetition_seed: int | None = None,
    deficit_weight_source: str | None = None,
    media_root: pathlib.Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for name, value in (("max_rows", max_rows), ("row_multiple", row_multiple)):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{name} must be a positive integer")
    resolved_repetition = validate_repetition_config(repetition_config)
    configured_row_cap = resolved_repetition["row_cap"]
    if max_rows is None:
        max_rows = configured_row_cap
    elif configured_row_cap is not None and configured_row_cap != max_rows:
        raise ValueError("repetition row_cap must equal the materialization max_rows")
    if resolved_repetition["enabled"] and max_rows is None:
        raise ValueError("enabled repetition blend requires a materialization row_cap")
    if max_rows is not None:
        resolved_repetition["row_cap"] = max_rows
    if previous_path is None and previous_sha256 is not None:
        raise ValueError("previous_sha256 requires previous_path")
    if previous_path is not None:
        resolved_previous_path = previous_path.expanduser().resolve(strict=True)
        actual_previous_sha256 = sha256_file(resolved_previous_path)
        if (
            previous_sha256 is not None
            and previous_sha256 != actual_previous_sha256
        ):
            raise ValueError("previous training JSONL SHA-256 changed")
    else:
        resolved_previous_path = None
        actual_previous_sha256 = None
    resolved_mined_path = mined_path.expanduser().resolve(strict=True)
    mined = load_records(resolved_mined_path)
    if not mined:
        raise ValueError("the current iteration must contribute at least one mined record")
    previous = (
        load_records(resolved_previous_path)
        if resolved_previous_path is not None
        else []
    )
    resolved_media_root = media_root.expanduser().resolve() if media_root else None

    def identity(record: dict[str, Any], context: str) -> str:
        if resolved_media_root is not None:
            return str(
                sample_from_record(
                    record, media_root=resolved_media_root, context=context
                )["atomic_sample_id"]
            )
        return logical_record_identity(record, context=context)

    evaluation_targets: dict[str, str] = {}
    for path in validation_paths:
        for index, record in enumerate(load_records(path)):
            evaluation_targets[identity(record, f"{path}:{index}")] = str(path)

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_count = 0
    tasks: Counter[str] = Counter()
    provenance: list[dict[str, Any]] = []
    for source_kind, source_path, records in (
        ("previous_iteration", resolved_previous_path, previous),
        ("current_mining", resolved_mined_path, mined),
    ):
        if source_path is None:
            continue
        for index, record in enumerate(records):
            target = target_path(record, context=f"{source_path}:{index}")
            sample_identity = identity(record, f"{source_path}:{index}")
            if sample_identity in evaluation_targets:
                raise ValueError(
                    f"train/evaluation leakage: atomic sample containing target "
                    f"{target!r} also occurs in {evaluation_targets[sample_identity]}"
                )
            key = _fingerprint(record)
            if source_kind == "current_mining" and key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            merged.append(record)
            tasks[str(record.get("task_type", "unknown"))] += 1
            provenance.append(
                {
                    "source_kind": source_kind,
                    "source": str(source_path),
                    "source_index": index,
                    "id": record.get("id"),
                }
            )
    if not merged:
        raise ValueError("real-mining assembly produced no training records")
    uncapped_records = len(merged)
    repetition_manifest: dict[str, Any] | None = None
    selection_policy = "monotonic_current_fill_task_balanced_v1"
    if resolved_repetition["enabled"]:
        assert max_rows is not None
        materialized_rows = max_rows
        if row_multiple is not None:
            materialized_rows -= materialized_rows % row_multiple
            if materialized_rows < row_multiple:
                raise ValueError(
                    "training materialization cannot form one complete global batch: "
                    f"row_cap={max_rows}, row_multiple={row_multiple}"
                )
        current = [
            index
            for index, item in enumerate(provenance)
            if item["source_kind"] == "current_mining"
        ]
        prior = [
            index
            for index, item in enumerate(provenance)
            if item["source_kind"] == "previous_iteration"
        ]
        if not current:
            raise ValueError(
                "repetition blend requires at least one unique current Mining record"
            )
        if len(prior) + 1 > materialized_rows:
            raise ValueError(
                "training materialization cannot retain all previous iteration records "
                "and include current Mining data under the configured cap"
            )
        # Keep fresh corrective examples at the front of the first repetition
        # pass, matching the non-repetition materializer's ordering contract.
        available_indices = current + prior
        available_records = [merged[index] for index in available_indices]
        available_provenance = [provenance[index] for index in available_indices]
        first_prior = len(current)
        merged, repetition_manifest = apply_repetition_blend(
            available_records,
            row_cap=materialized_rows,
            config=resolved_repetition,
            deficit_weights=deficit_weights,
            seed=repetition_seed,
            deficit_weight_source=deficit_weight_source,
            mandatory_indices=set(range(first_prior, len(available_records))) | {0},
            row_multiple=row_multiple or 1,
        )
        provenance_by_fingerprint: dict[str, list[dict[str, Any]]] = {}
        for record, item in zip(available_records, available_provenance):
            provenance_by_fingerprint.setdefault(_fingerprint(record), []).append(item)
        provenance_positions = {key: 0 for key in provenance_by_fingerprint}
        materialized_provenance: list[dict[str, Any]] = []
        for record in merged:
            key = _fingerprint(record)
            choices = provenance_by_fingerprint[key]
            position = provenance_positions[key]
            materialized_provenance.append(
                choices[position] if position < len(choices) else choices[0]
            )
            provenance_positions[key] = position + 1
        provenance = materialized_provenance
        tasks = Counter(str(record.get("task_type", "unknown")) for record in merged)
        selection_policy = "monotonic_deficit_repetition_blend_v1"
    elif max_rows is not None or row_multiple is not None:
        materialized_rows = min(uncapped_records, max_rows or uncapped_records)
        if row_multiple is not None:
            materialized_rows -= materialized_rows % row_multiple
            if materialized_rows < row_multiple:
                raise ValueError(
                    "training materialization cannot form one complete global batch: "
                    f"available={uncapped_records}, row_multiple={row_multiple}"
                )
        # Retain the full prior iteration to make the training set monotonic.
        # Current Mining owns the remaining slots under the cap, selected in
        # task-balanced order and emitted first so fresh corrective examples
        # are not pushed to the tail of a deterministic epoch.
        current = [
            index
            for index, item in enumerate(provenance)
            if item["source_kind"] == "current_mining"
        ]
        prior = [
            index
            for index, item in enumerate(provenance)
            if item["source_kind"] == "previous_iteration"
        ]
        if len(prior) > materialized_rows:
            raise ValueError(
                "training materialization cannot retain all previous iteration records: "
                f"previous={len(prior)}, materialized={materialized_rows}"
            )
        current_limit = min(len(current), materialized_rows - len(prior))
        if current and current_limit == 0:
            raise ValueError(
                "training materialization cannot retain all previous iteration records "
                "and include current Mining data under the configured cap"
            )
        selected = _task_balanced_indices(current, merged, current_limit)
        selected.extend(prior)
        merged = [merged[index] for index in selected]
        provenance = [provenance[index] for index in selected]
        tasks = Counter(str(record.get("task_type", "unknown")) for record in merged)
    if repetition_manifest is None:
        merged, repetition_manifest = apply_repetition_blend(
            merged,
            row_cap=max_rows or len(merged),
            config=resolved_repetition,
            deficit_weights=deficit_weights,
            seed=repetition_seed,
            deficit_weight_source=deficit_weight_source,
            row_multiple=row_multiple or 1,
        )
    records_truncated = (
        repetition_manifest["totals"]["dropped_rows"]
        if resolved_repetition["enabled"]
        else uncapped_records - len(merged)
    )
    mined_fingerprints = {_fingerprint(record) for record in mined}
    if not any(_fingerprint(record) in mined_fingerprints for record in merged):
        raise ValueError("current mined records were all lost during assembly")
    previous_fingerprints = {_fingerprint(record) for record in previous}
    output_fingerprints = {_fingerprint(record) for record in merged}
    previous_fingerprints_subset = previous_fingerprints.issubset(output_fingerprints)
    retained_previous_records = len(
        {
            (item["source"], item["source_index"])
            for item in provenance
            if item["source_kind"] == "previous_iteration"
        }
    )
    if retained_previous_records != len(previous):
        raise ValueError(
            "training materialization must retain every previous record: "
            f"retained={retained_previous_records}, previous={len(previous)}"
        )
    if not previous_fingerprints_subset:
        raise ValueError("previous training fingerprints are not a subset of output")
    return merged, {
        "schema_version": 2,
        "format": "jsonl",
        "annotation_profile": "nvpaw_multitask_v1",
        "training_source": "mined_real_samples_only",
        "previous_iteration": (
            str(resolved_previous_path)
            if resolved_previous_path is not None
            else None
        ),
        "previous_sha256": actual_previous_sha256,
        "mined_input": str(resolved_mined_path),
        "validation_inputs": [str(path) for path in validation_paths],
        "previous_records": len(previous),
        "previous_fingerprint_count": len(previous_fingerprints),
        "previous_fingerprints_sha256": _fingerprint_set_sha256(
            previous_fingerprints
        ),
        "previous_fingerprints_subset": previous_fingerprints_subset,
        "output_fingerprint_count": len(output_fingerprints),
        "output_fingerprints_sha256": _fingerprint_set_sha256(output_fingerprints),
        "mined_records": len(mined),
        "output_records": len(merged),
        "uncapped_records": uncapped_records,
        "materialization_cap": max_rows,
        "row_multiple": row_multiple,
        "selection_policy": selection_policy,
        "records_truncated": records_truncated,
        "duplicates_skipped": duplicate_count,
        "retained_previous_records": retained_previous_records,
        "selected_current_records": len(
            {
                (item["source"], item["source_index"])
                for item in provenance
                if item["source_kind"] == "current_mining"
            }
        ),
        "materialized_previous_records": sum(
            item["source_kind"] == "previous_iteration" for item in provenance
        ),
        "materialized_current_records": sum(
            item["source_kind"] == "current_mining" for item in provenance
        ),
        "tasks": dict(sorted(tasks.items())),
        "warnings": list(repetition_manifest["warnings"]),
        "provenance": provenance,
        "repetition_blend": repetition_manifest,
    }


def bind_summary(
    summary: dict[str, Any], training_jsonl: pathlib.Path
) -> dict[str, Any]:
    path = training_jsonl.expanduser().resolve(strict=True)
    rows = load_records(path)
    bound = json.loads(json.dumps(summary))
    if bound.get("output_records") != len(rows):
        raise ValueError("training JSONL row count differs from assembly summary")
    fingerprints = {_fingerprint(record) for record in rows}
    if bound.get("output_fingerprints_sha256") != _fingerprint_set_sha256(
        fingerprints
    ):
        raise ValueError("training JSONL fingerprints differ from assembly summary")
    bound["training_jsonl"] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
    }
    return bound


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-jsonl", type=pathlib.Path)
    parser.add_argument("--previous-sha256")
    parser.add_argument("--mined-jsonl", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--summary", type=pathlib.Path)
    parser.add_argument("--current-quota-manifest", type=pathlib.Path)
    parser.add_argument("--quota-manifest", type=pathlib.Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--global-batch", type=int)
    parser.add_argument("--validation-jsonl", action="append", default=[], type=pathlib.Path)
    parser.add_argument("--media-root", required=True, type=pathlib.Path)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--row-multiple", type=int)
    parser.add_argument("--gap-analysis-summary", type=pathlib.Path)
    parser.add_argument("--repetition-config", type=pathlib.Path)
    repetition_toggle = parser.add_mutually_exclusive_group()
    repetition_toggle.add_argument(
        "--repetition-blend",
        dest="repetition_blend",
        action="store_true",
        default=None,
    )
    repetition_toggle.add_argument(
        "--no-repetition-blend",
        dest="repetition_blend",
        action="store_false",
    )
    parser.add_argument("--repetition-policy", choices=REPETITION_POLICIES)
    parser.add_argument("--repetition-rep-min", type=float)
    parser.add_argument("--repetition-rep-max", type=float)
    parser.add_argument("--repetition-budget-multiplier", type=float)
    parser.add_argument("--repetition-share-gap-tolerance", type=float)
    redistribution_toggle = parser.add_mutually_exclusive_group()
    redistribution_toggle.add_argument(
        "--repetition-redistribute",
        dest="repetition_redistribute",
        action="store_true",
        default=None,
    )
    redistribution_toggle.add_argument(
        "--no-repetition-redistribute",
        dest="repetition_redistribute",
        action="store_false",
    )
    empty_toggle = parser.add_mutually_exclusive_group()
    empty_toggle.add_argument(
        "--repetition-never-repeat-empty-gt",
        dest="repetition_never_repeat_empty_gt",
        action="store_true",
        default=None,
    )
    empty_toggle.add_argument(
        "--repetition-allow-empty-gt",
        dest="repetition_never_repeat_empty_gt",
        action="store_false",
    )
    parser.add_argument(
        "--repetition-explicit-multiplier",
        action="append",
        metavar="TASK=MULTIPLIER",
    )
    parser.add_argument("--repetition-seed", type=int)
    parser.add_argument(
        "--repetition-manifest",
        type=pathlib.Path,
        help="Defaults to repetition_blend_manifest.json beside --output.",
    )
    args = parser.parse_args(argv)
    try:
        if args.previous_jsonl is not None and args.previous_sha256 is None:
            raise ValueError("--previous-sha256 is required with --previous-jsonl")
        repetition_config = merge_repetition_config(
            load_repetition_config(args.repetition_config)
            if args.repetition_config is not None
            else None,
            {
                "enabled": args.repetition_blend,
                "policy": args.repetition_policy,
                "rep_min": args.repetition_rep_min,
                "rep_max": args.repetition_rep_max,
                "budget_multiplier": args.repetition_budget_multiplier,
                "share_gap_tolerance": args.repetition_share_gap_tolerance,
                "redistribute": args.repetition_redistribute,
                "never_repeat_empty_gt": args.repetition_never_repeat_empty_gt,
                "explicit_multipliers": parse_explicit_multipliers(
                    args.repetition_explicit_multiplier
                ),
                "seed": args.repetition_seed,
            },
        )
        gap_summary = (
            json.loads(args.gap_analysis_summary.read_text(encoding="utf-8"))
            if args.gap_analysis_summary is not None
            else None
        )
        deficit_weights, deficit_weight_source = deficit_weights_from_gap_summary(
            gap_summary, list(TASK_SPECS)
        )
        repetition_seed = args.repetition_seed
        if repetition_seed is None:
            repetition_seed = repetition_config["seed"]
        if repetition_seed is None and isinstance(gap_summary, dict):
            repetition_seed = gap_summary.get("seed")
        if repetition_seed is None:
            repetition_seed = 17
        rows, summary = assemble(
            args.previous_jsonl,
            args.mined_jsonl,
            previous_sha256=args.previous_sha256,
            validation_paths=args.validation_jsonl,
            max_rows=args.max_rows,
            row_multiple=args.row_multiple,
            repetition_config=repetition_config,
            deficit_weights=deficit_weights,
            repetition_seed=repetition_seed,
            deficit_weight_source=deficit_weight_source,
            media_root=args.media_root,
        )
        _write_jsonl(args.output, rows)
        summary = bind_summary(summary, args.output)
        repetition_manifest = bind_repetition_manifest(
            summary["repetition_blend"], args.output
        )
        summary["repetition_blend"] = repetition_manifest
        quota_values = (
            args.current_quota_manifest,
            args.quota_manifest,
            args.epochs,
            args.global_batch,
        )
        if any(value is not None for value in quota_values):
            if any(value is None for value in quota_values):
                raise ValueError(
                    "cumulative quota binding requires current/final manifests, "
                    "epochs, and global batch"
                )
            current_quota = json.loads(
                args.current_quota_manifest.read_text(encoding="utf-8")
            )
            final_quota = bind_cumulative_manifest(
                current_quota,
                current_jsonl=args.mined_jsonl,
                training_jsonl=args.output,
                assembly_summary=summary,
                epochs=args.epochs,
                global_batch=args.global_batch,
            )
            args.quota_manifest.parent.mkdir(parents=True, exist_ok=True)
            args.quota_manifest.write_text(
                json.dumps(final_quota, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        repetition_manifest_path = (
            args.repetition_manifest
            or args.output.with_name("repetition_blend_manifest.json")
        )
        repetition_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        repetition_manifest_path.write_text(
            json.dumps(repetition_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary_path = args.summary or args.output.with_name("assemble_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"assemble_training_json: {exc}", file=sys.stderr)
        return 2
    print(f"assemble_training_json: wrote {len(rows)} real records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
