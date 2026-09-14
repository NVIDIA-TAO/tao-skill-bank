#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Select single-image MCQ classification rows with a class answer as calibration rows.

Phase 4 step 4c-A. The anchors run's corpus taught the answer ``[]`` about
3,600 times against roughly 320 rows where a class had to be chosen, and the
model answered ``[]`` on 58% of the single-image defect MCQ of the new
benchmark. A probe without the "return []" clause removed the empty answers but
left accuracy near guessing, so the missing capability is defect-type
classification. This selector adds a per-iteration quota of real pool rows that
*do* choose a class: single image, lettered ``current possible classes`` MCQ,
non-empty ground truth, class shares taken from the KPI set's ground-truth
labels (semantic label text, not the letter, because letters differ between
prompts) with largest-remainder allocation over the classes the pool can
supply. Rows are copied unchanged plus the inert markers ``deft_calibration``
and ``deft_calibration_kind = classification``; the assembler protects them
from the anchor-cap trim exactly like detection calibration rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
from collections import Counter
from typing import Any, Iterable, Iterator

from answer_profile import image_count, mcq_labels
from assemble_training_json import marker_free_key, record_identity, sha256_file
from defect_detection_ablation import (
    CALIBRATION_KIND_MARK,
    CALIBRATION_MARK,
    CLASSIFICATION_CALIBRATION_KIND,
)
from nvpaw_annotations import TASK_SPECS
from select_detection_calibration import parse_task_totals
from validate_sharegpt import load_records


# Single-image classification tasks only: the eligibility rule is one image in
# the user message, so the two-image reference classification task is out.
CLASSIFICATION_CALIBRATION_TASKS = tuple(
    sorted(
        task
        for task, spec in TASK_SPECS.items()
        if spec["metric_family"] == "classification" and tuple(spec["image_roles"]) == ("target",)
    )
)
SCHEMA_VERSION = "classification_calibration_v1"
POLICY = "kpi_class_shares_largest_remainder"
DEFAULT_MIN_FILL_FRACTION = 0.9
DEFAULT_SEED = 17
MARKERS = {CALIBRATION_MARK: True, CALIBRATION_KIND_MARK: CLASSIFICATION_CALIBRATION_KIND}
ELIGIBILITY = {
    "tasks": list(CLASSIFICATION_CALIBRATION_TASKS),
    "images": 1,
    "format": "lettered MCQ (options block 'current possible classes'); the yes/no "
    "'Answer with the complete option text' form is rejected",
    "ground_truth": "non-empty option set (letters mapped to the prompt's option labels)",
    "exclusions": "atomic identities / ids listed in --exclude-identities-file; marker-free content duplicates",
}


def _rank(seed: int, record_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{record_id}".encode()).digest(), "big")


def _largest_remainder(weights: dict[str, float], total: int) -> dict[str, int]:
    if total <= 0 or not weights:
        return {key: 0 for key in weights}
    scale = sum(weights.values())
    raw = {key: total * value / scale for key, value in weights.items()}
    alloc = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(alloc.values())
    for key in sorted(weights, key=lambda k: (-(raw[k] - alloc[k]), k))[:remaining]:
        alloc[key] += 1
    return alloc


def validate_task_totals(task_totals: dict[str, int]) -> dict[str, int]:
    if not task_totals:
        raise ValueError("at least one --task-total TASK=ROWS is required")
    for task, total in task_totals.items():
        if task not in CLASSIFICATION_CALIBRATION_TASKS:
            raise ValueError(
                f"classification calibration accepts single-image classification tasks "
                f"{list(CLASSIFICATION_CALIBRATION_TASKS)}, not {task!r}"
            )
        if type(total) is not int or total < 0:
            raise ValueError(f"classification calibration total must be a non-negative integer: {task}={total!r}")
    if sum(task_totals.values()) <= 0:
        raise ValueError("at least one classification calibration total must be positive")
    return dict(task_totals)


def derive_class_shares(
    kpi_records: Iterable[dict[str, Any]], tasks: Iterable[str]
) -> dict[str, dict[str, Any]]:
    """Per task: KPI rows with a parsed non-empty MCQ answer and their label shares.

    A multi-label ground truth counts once per label. Rows in the yes/no form,
    unparsable or empty answers do not contribute (they carry no class label).
    """

    wanted = set(tasks)
    counts: dict[str, Counter[str]] = {task: Counter() for task in wanted}
    rows: Counter[str] = Counter()
    for index, record in enumerate(kpi_records):
        task = str(record.get("task_type"))
        if task not in wanted:
            continue
        status, labels = mcq_labels(record, context=f"kpi record[{index}]")
        if status != "ok" or not labels:
            continue
        rows[task] += 1
        counts[task].update(labels)
    result: dict[str, dict[str, Any]] = {}
    for task in sorted(wanted):
        total = sum(counts[task].values())
        result[task] = {
            "rows": rows[task],
            "label_rows": dict(sorted(counts[task].items())),
            "shares": {label: count / total for label, count in sorted(counts[task].items())} if total else {},
        }
    return result


def class_quotas(
    total: int, kpi_shares: dict[str, float], present: Iterable[str]
) -> tuple[dict[str, int], str, list[str]]:
    """Largest-remainder split of ``total`` over the classes present in the pool.

    KPI classes the pool cannot supply are dropped and their share is
    redistributed over the remaining KPI classes; when the KPI set gives no
    usable class for the task the quota is uniform over the pool's classes.
    """

    present_sorted = sorted(set(present))
    missing = sorted(label for label, share in kpi_shares.items() if share > 0 and label not in present_sorted)
    if not present_sorted:
        return {}, "no_eligible_rows", missing
    weights = {label: float(kpi_shares[label]) for label in present_sorted if kpi_shares.get(label, 0.0) > 0.0}
    source = "kpi"
    if not weights:
        weights = {label: 1.0 for label in present_sorted}
        source = "uniform_pool_fallback"
    alloc = _largest_remainder(weights, total)
    return {label: alloc.get(label, 0) for label in present_sorted}, source, missing


def load_exclusions(
    paths: Iterable[pathlib.Path], *, media_root: pathlib.Path | None
) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    """Atomic identities and ids to exclude, from JSONL records or one identity per line.

    ``.jsonl`` files are training/anchor rows: every row contributes its id and
    (for supported tasks) the same atomic identity the assembler uses; any
    other file lists identities verbatim.
    """

    identities: set[str] = set()
    ids: set[str] = set()
    inputs: list[dict[str, Any]] = []
    for path in paths:
        resolved = pathlib.Path(path).expanduser().resolve(strict=True)
        entry: dict[str, Any] = {"path": str(resolved), "sha256": sha256_file(resolved)}
        rows = 0
        with resolved.open(encoding="utf-8") as stream:
            if resolved.suffix.casefold() == ".jsonl":
                entry["kind"] = "records"
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{resolved}:{line_number}: invalid JSON") from exc
                    if not isinstance(record, dict):
                        raise ValueError(f"{resolved}:{line_number}: row must be an object")
                    rows += 1
                    if record.get("id") is not None:
                        ids.add(str(record["id"]))
                    if record.get("task_type") in TASK_SPECS:
                        identities.add(
                            record_identity(record, media_root=media_root, context=f"{resolved}:{line_number}")
                        )
            else:
                entry["kind"] = "identities"
                for line in stream:
                    value = line.strip()
                    if value:
                        identities.add(value)
                        rows += 1
        entry["rows"] = rows
        inputs.append(entry)
    return identities, ids, inputs


class _Bucket:
    """One class bucket: seed/id-ranked rows per dataset, filled round-robin across datasets."""

    def __init__(self, entries: Iterable[dict[str, Any]]) -> None:
        self.datasets: dict[str, list[dict[str, Any]]] = {}
        for entry in entries:
            self.datasets.setdefault(entry["dataset"], []).append(entry)
        for rows in self.datasets.values():
            rows.sort(key=lambda item: (item["rank"], item["id"]))
        self.positions = {dataset: 0 for dataset in self.datasets}
        self.taken: Counter[str] = Counter()
        self.size = sum(len(rows) for rows in self.datasets.values())

    def next(self, selected_ids: set[str]) -> dict[str, Any] | None:
        for dataset in sorted(self.datasets, key=lambda name: (self.taken[name], name)):
            rows = self.datasets[dataset]
            position = self.positions[dataset]
            while position < len(rows) and rows[position]["id"] in selected_ids:
                position += 1
            self.positions[dataset] = position
            if position < len(rows):
                self.positions[dataset] = position + 1
                self.taken[dataset] += 1
                return rows[position]
        return None


def select_classification_calibration(
    pool: Iterable[dict[str, Any]],
    *,
    kpi_records: Iterable[dict[str, Any]],
    task_totals: dict[str, int],
    excluded_identities: set[str] | None = None,
    excluded_ids: set[str] | None = None,
    media_root: pathlib.Path | None = None,
    seed: int = DEFAULT_SEED,
    min_fill_fraction: float = DEFAULT_MIN_FILL_FRACTION,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    totals = validate_task_totals(task_totals)
    if not 0.0 < float(min_fill_fraction) <= 1.0:
        raise ValueError("min_fill_fraction must be in (0, 1]")
    excluded_identities = excluded_identities or set()
    excluded_ids = excluded_ids or set()
    shares = derive_class_shares(kpi_records, totals)

    examined = 0
    rejected: Counter[str] = Counter()
    excluded_by_identity = 0
    deduplicated_content = 0
    deduplicated_identity = 0
    seen_content: set[str] = set()
    seen_ids: set[str] = set()
    seen_identities: set[str] = set()
    eligible: dict[str, list[dict[str, Any]]] = {task: [] for task in totals}
    for index, record in enumerate(pool):
        examined += 1
        context = f"pool record[{index}]"
        task = str(record.get("task_type"))
        if task not in CLASSIFICATION_CALIBRATION_TASKS:
            rejected["task_not_eligible"] += 1
            continue
        if task not in totals:
            rejected["task_not_requested"] += 1
            continue
        if image_count(record, context=context) != 1:
            rejected["multi_image"] += 1
            continue
        status, labels = mcq_labels(record, context=context)
        if status != "ok" or not labels:
            rejected[status] += 1
            continue
        record_id = str(record.get("id"))
        identity = record_identity(record, media_root=media_root, context=context)
        if identity in excluded_identities or record_id in excluded_ids:
            excluded_by_identity += 1
            continue
        key = marker_free_key(record)
        if key in seen_content or record_id in seen_ids:
            deduplicated_content += 1
            continue
        if identity in seen_identities:
            deduplicated_identity += 1
            continue
        seen_content.add(key)
        seen_ids.add(record_id)
        seen_identities.add(identity)
        eligible[task].append(
            {
                "record": record,
                "id": record_id,
                "labels": labels,
                "dataset": str(record.get("dataset") or "unknown"),
                "rank": _rank(seed, record_id),
            }
        )

    selected_rows: list[dict[str, Any]] = []
    tasks_summary: dict[str, Any] = {}
    shortfall_tasks: list[str] = []
    for task in sorted(totals):
        total = totals[task]
        entries = eligible[task]
        present = sorted({label for entry in entries for label in entry["labels"]})
        quotas, share_source, missing = class_quotas(total, shares[task]["shares"], present)
        buckets = {
            label: _Bucket(entry for entry in entries if label in entry["labels"]) for label in present
        }
        selected_ids: set[str] = set()
        picked: list[dict[str, Any]] = []
        per_class_selected: Counter[str] = Counter()
        datasets: Counter[str] = Counter()
        progress = True
        while progress and any(per_class_selected[label] < quotas[label] for label in quotas):
            progress = False
            for label in sorted(quotas):
                if per_class_selected[label] >= quotas[label]:
                    continue
                entry = buckets[label].next(selected_ids)
                if entry is None:
                    continue
                selected_ids.add(entry["id"])
                picked.append(entry)
                per_class_selected[label] += 1
                datasets[entry["dataset"]] += 1
                progress = True
        for entry in picked:
            selected_rows.append({**entry["record"], **MARKERS})
        weights_total = sum(shares[task]["shares"].get(label, 0.0) for label in quotas) if share_source == "kpi" else 0.0
        per_class: dict[str, Any] = {}
        for label in sorted(set(present) | set(shares[task]["label_rows"])):
            if label in quotas:
                if share_source == "kpi":
                    target_share = shares[task]["shares"].get(label, 0.0) / weights_total if weights_total else 0.0
                else:
                    target_share = 1.0 / len(quotas)
            else:
                target_share = 0.0
            per_class[label] = {
                "kpi_rows": int(shares[task]["label_rows"].get(label, 0)),
                "target_share": target_share,
                "target": int(quotas.get(label, 0)),
                "eligible": buckets[label].size if label in buckets else 0,
                "selected": int(per_class_selected[label]),
            }
        fill = len(picked) / total if total else 1.0
        if total and fill < float(min_fill_fraction):
            shortfall_tasks.append(task)
        tasks_summary[task] = {
            "requested": total,
            "eligible": len(entries),
            "selected": len(picked),
            "fill_fraction": round(fill, 4),
            "shortfall": max(0, total - len(picked)),
            "min_fill_fraction": float(min_fill_fraction),
            "kpi_rows": shares[task]["rows"],
            "class_share_source": share_source,
            "classes_missing_in_pool": missing,
            "per_class": per_class,
            "datasets": dict(sorted(datasets.items())),
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "policy": POLICY,
        "kind": CLASSIFICATION_CALIBRATION_KIND,
        "markers": dict(MARKERS),
        "eligibility": ELIGIBILITY,
        "seed": int(seed),
        "min_fill_fraction": float(min_fill_fraction),
        "allow_shortfall": False,
        "task_totals": totals,
        "selected_total": len(selected_rows),
        "examined_records": examined,
        "excluded_by_identity": excluded_by_identity,
        "excluded_identities": len(excluded_identities),
        "excluded_ids": len(excluded_ids),
        "deduplicated_content": deduplicated_content,
        "deduplicated_identity": deduplicated_identity,
        "rejected": dict(sorted(rejected.items())),
        "tasks": tasks_summary,
        "shortfall_tasks": shortfall_tasks,
        "fill_ok": not shortfall_tasks,
        "accepted": not shortfall_tasks,
    }
    return selected_rows, manifest


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


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True, type=pathlib.Path, help="Mining / calibration pool JSONL")
    parser.add_argument("--kpi", required=True, type=pathlib.Path, help="Steering set JSONL (class shares per task)")
    parser.add_argument(
        "--task-total", action="append", metavar="TASK=ROWS", required=True,
        help='Rows per single-image classification task, e.g. "Defect Classification=256" (repeatable).',
    )
    parser.add_argument(
        "--exclude-identities-file", action="append", default=[], type=pathlib.Path, metavar="TXT|JSONL",
        help="Rows (JSONL, e.g. the cumulative train.jsonl or anchor candidates) or one atomic identity per line to exclude (repeatable).",
    )
    parser.add_argument("--media-root", type=pathlib.Path, help="Resolve relative image paths like the assembler does.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-fill-fraction", type=float, default=DEFAULT_MIN_FILL_FRACTION)
    parser.add_argument("--allow-shortfall", action="store_true", help="Exit 0 below --min-fill-fraction (the shortfall is recorded either way).")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        totals = validate_task_totals(parse_task_totals(args.task_total))
        media_root = args.media_root.expanduser().resolve() if args.media_root is not None else None
        excluded_identities, excluded_ids, exclusion_inputs = load_exclusions(
            args.exclude_identities_file, media_root=media_root
        )
        pool_path = args.pool.expanduser().resolve(strict=True)
        kpi_path = args.kpi.expanduser().resolve(strict=True)
        rows, manifest = select_classification_calibration(
            _stream_records(pool_path),
            kpi_records=load_records(kpi_path),
            task_totals=totals,
            excluded_identities=excluded_identities,
            excluded_ids=excluded_ids,
            media_root=media_root,
            seed=args.seed,
            min_fill_fraction=args.min_fill_fraction,
        )
        manifest["allow_shortfall"] = bool(args.allow_shortfall)
        manifest["accepted"] = bool(manifest["fill_ok"] or args.allow_shortfall)
        manifest["inputs"] = {
            "pool": {"path": str(pool_path), "sha256": sha256_file(pool_path)},
            "kpi": {"path": str(kpi_path), "sha256": sha256_file(kpi_path)},
            "exclusions": exclusion_inputs,
            "media_root": str(media_root) if media_root is not None else None,
        }
        _write_jsonl(args.output, rows)
        output_path = args.output.expanduser().resolve()
        manifest["output"] = {"path": str(output_path), "rows": len(rows), "sha256": sha256_file(output_path)}
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not manifest["accepted"]:
            details = "; ".join(
                f"{task}: selected {manifest['tasks'][task]['selected']}/{manifest['tasks'][task]['requested']}"
                for task in manifest["shortfall_tasks"]
            )
            raise ValueError(
                f"classification calibration shortfall below min_fill_fraction={args.min_fill_fraction}: {details} "
                f"(manifest written to {args.manifest}; pass --allow-shortfall to accept)"
            )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"select_classification_calibration: {exc}", file=sys.stderr)
        return 2
    print(
        "select_classification_calibration: "
        f"selected={manifest['selected_total']} tasks="
        + ",".join(f"{task}:{payload['selected']}/{payload['requested']}" for task, payload in manifest["tasks"].items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
