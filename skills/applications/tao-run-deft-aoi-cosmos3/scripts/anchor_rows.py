#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Correct-row anchors for the cumulative training set.

An anchor is a mining-pool row the *current* (or zero-shot) model already
answers correctly. Anchors are added to the cumulative training set so that a
fixed share of the corpus keeps rehearsing behaviour the model must not lose
(single-image yes/no, clean -> [] , rare labels) while the mined residual rows
push the weak abilities. Selection is deterministic (seed/id hash), stratified
by task with a per-dataset cap inside each task, and never touches evaluation
targets or rows already present in the corpus.

The candidate file is produced offline by ``build_anchor_candidates.py`` from
the canonical Mining JSONL and a scored-pool id list; this module only reads
that file, so it needs no PyArrow.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable

from nvpaw_annotations import TASK_SPECS
from validate_sharegpt import load_records

ANCHOR_SOURCE_KIND = "anchor_correct"
# Top-level marker written on every anchor row so later iterations can count
# retained anchors exactly. The training runtime reads only id/task_type/messages.
ANCHOR_MARK = "deft_anchor"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_anchor_config(share: float | None, source: pathlib.Path | None,
                           task_shares: pathlib.Path | None, source_cap: float | None) -> dict[str, Any]:
    """Return a resolved anchor config; share 0 / None means the feature is off."""
    share = 0.0 if share is None else float(share)
    if not 0.0 <= share < 1.0:
        raise ValueError("anchor share must be in [0, 1)")
    cap = 0.35 if source_cap is None else float(source_cap)
    if not 0.0 < cap <= 1.0:
        raise ValueError("anchor source cap must be in (0, 1]")
    enabled = share > 0.0
    if enabled and source is None:
        raise ValueError("--anchor-share > 0 requires --anchor-source")
    if enabled and task_shares is None:
        raise ValueError("--anchor-share > 0 requires --anchor-task-shares")
    return {"enabled": enabled, "share": share, "source": str(source) if source else None,
            "task_shares": str(task_shares) if task_shares else None, "source_cap": cap,
            "unit": "rows"}


def task_shares_from_jsonl(path: pathlib.Path) -> dict[str, float]:
    """Task row shares of an evaluation set (the KPI set); six supported tasks only."""
    counts: Counter[str] = Counter()
    for record in load_records(path.expanduser().resolve(strict=True)):
        task = record.get("task_type")
        if task in TASK_SPECS:
            counts[str(task)] += 1
    total = sum(counts.values())
    if total == 0:
        raise ValueError(f"anchor task-share source has no supported task rows: {path}")
    return {task: counts[task] / total for task in sorted(counts)}


def anchor_target_rows(share: float, non_anchor_rows: int, existing_anchor_rows: int) -> tuple[int, int]:
    """Total anchors wanted so anchors are ``share`` of the corpus, and the new rows needed."""
    if non_anchor_rows <= 0 or share <= 0.0:
        return 0, 0
    total = int(round(non_anchor_rows * share / (1.0 - share)))
    return total, max(0, total - existing_anchor_rows)


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


def _rank(seed: int, record_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{record_id}".encode()).digest(), "big")


def select_anchors(candidates: Iterable[dict[str, Any]], *, new_rows: int, task_shares: dict[str, float],
                   source_cap: float, seed: int, is_excluded: Callable[[dict[str, Any]], str | None],
                   dataset_of: Callable[[dict[str, Any]], str] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pick ``new_rows`` anchors: task quotas by ``task_shares``, per-dataset cap inside a task.

    ``is_excluded`` returns a reason (str) when a candidate must be skipped
    (already in the corpus, evaluation target, unsupported task) or None.
    """
    dataset_of = dataset_of or (lambda r: str(r.get("dataset") or "unknown"))
    by_task: dict[str, dict[str, list[tuple[int, str, dict[str, Any]]]]] = defaultdict(lambda: defaultdict(list))
    skipped: Counter[str] = Counter()
    eligible = 0
    for record in candidates:
        task = str(record.get("task_type"))
        if task not in TASK_SPECS or task not in task_shares:
            skipped["task_not_in_shares"] += 1
            continue
        reason = is_excluded(record)
        if reason:
            skipped[reason] += 1
            continue
        eligible += 1
        by_task[task][dataset_of(record)].append((_rank(seed, str(record["id"])), str(record["id"]), record))
    quotas = _largest_remainder({t: s for t, s in task_shares.items() if s > 0}, new_rows)
    selected: list[dict[str, Any]] = []
    per_cell: dict[str, dict[str, int]] = {}
    shortage: dict[str, int] = {}
    for task, quota in sorted(quotas.items()):
        datasets = by_task.get(task, {})
        if quota <= 0 or not datasets:
            if quota > 0:
                shortage[task] = quota
            continue
        for ds in datasets:
            datasets[ds].sort(key=lambda item: (item[0], item[1]))
        # Relax the cap when there are too few datasets to fill the quota under it.
        n_ds = len(datasets)
        cap_rows = max(int(math.ceil(quota * source_cap - 1e-12)), int(math.ceil(quota / n_ds)))
        taken: dict[str, int] = {ds: 0 for ds in datasets}
        positions: dict[str, int] = {ds: 0 for ds in datasets}
        picked: list[dict[str, Any]] = []
        progress = True
        while len(picked) < quota and progress:
            progress = False
            for ds in sorted(datasets, key=lambda d: (taken[d], d)):
                if len(picked) >= quota:
                    break
                if taken[ds] >= cap_rows or positions[ds] >= len(datasets[ds]):
                    continue
                picked.append(datasets[ds][positions[ds]][2])
                positions[ds] += 1
                taken[ds] += 1
                progress = True
        per_cell[task] = {ds: n for ds, n in sorted(taken.items()) if n}
        if len(picked) < quota:
            shortage[task] = quota - len(picked)
        selected.extend(picked)
    report = {"requested_new_rows": new_rows, "selected_rows": len(selected), "eligible_candidates": eligible,
              "skipped": dict(sorted(skipped.items())), "task_quotas": quotas,
              "per_task_dataset": per_cell, "shortage": shortage, "source_cap": source_cap, "seed": seed}
    return selected, report


def anchor_ids(candidates: Iterable[dict[str, Any]]) -> set[str]:
    return {str(record["id"]) for record in candidates if record.get("id") is not None}


def write_manifest(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
