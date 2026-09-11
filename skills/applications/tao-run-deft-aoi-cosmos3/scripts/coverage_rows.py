#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Cross-dataset coverage blend for the cumulative training set.

The coverage slice is a small, fixed share of the materialized corpus drawn so
that every eligible (task, dataset) cell of the mining pool is represented,
independent of what the gap-driven miner selected. Two modes are launch
recorded so the arms are comparable:

* ``plain``    - rows sampled uniformly from the pool cell (no correctness filter);
* ``residual`` - rows the scored checkpoint got wrong (``is_residual``); a cell
  whose residual rows run out falls back to its correct rows, which are then
  tagged ``["coverage", "anchor"]`` and counted separately.

Allocation is a deterministic round-robin that always feeds the cell with the
fewest cumulative coverage rows (prior iterations + this one), so cells below
the per-cell floor ``K`` are filled first and the remainder spreads evenly.
The floor is a fail-closed check on the eventual budget under the row cap
(``cells x K <= round(cap x share)``), not on the first iteration's slice.

Candidates come from ``build_coverage_candidates.py``; rows carry
``deft_pool_status`` (``correct`` / ``residual`` / ``unscored``).
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable

from nvpaw_annotations import TASK_SPECS

COVERAGE_SOURCE_KIND = "coverage_blend"
COVERAGE_MARK = "deft_coverage"
COVERAGE_MODES = ("plain", "residual")
POOL_STATUS_KEY = "deft_pool_status"


def validate_coverage_config(share: float | None, mode: str | None, source: pathlib.Path | None,
                             min_rows_per_cell: int | None) -> dict[str, Any]:
    """Return a resolved coverage-blend config; share 0 / None means off."""
    share = 0.0 if share is None else float(share)
    if not 0.0 <= share < 1.0:
        raise ValueError("coverage blend share must be in [0, 1)")
    floor = 8 if min_rows_per_cell is None else int(min_rows_per_cell)
    if floor < 0:
        raise ValueError("coverage blend min rows per dataset must be >= 0")
    enabled = share > 0.0
    if enabled and mode not in COVERAGE_MODES:
        raise ValueError("--coverage-blend-share > 0 requires --coverage-blend-mode plain|residual")
    if enabled and source is None:
        raise ValueError("--coverage-blend-share > 0 requires --coverage-blend-source")
    return {"enabled": enabled, "share": share, "mode": mode if enabled else None,
            "source": str(source) if source else None, "min_rows_per_cell": floor, "unit": "rows"}


def joint_targets(base_rows: int, shares: dict[str, float], existing: dict[str, int]) -> dict[str, tuple[int, int]]:
    """Solve the slice totals so each named slice is its share of the whole corpus.

    ``base_rows`` are the rows that belong to none of the slices. Returns
    ``{name: (total_wanted, new_rows_needed)}``. With one slice this equals
    ``round(N0 * s / (1 - s))``.
    """
    active = {name: share for name, share in shares.items() if share > 0.0}
    if base_rows <= 0 or not active:
        return {name: (0, 0) for name in shares}
    remaining = 1.0 - sum(active.values())
    if remaining <= 0.0:
        raise ValueError("combined slice shares must be below 1")
    corpus = base_rows / remaining
    result: dict[str, tuple[int, int]] = {}
    for name in shares:
        total = int(round(corpus * active.get(name, 0.0)))
        result[name] = (total, max(0, total - int(existing.get(name, 0))))
    return result


def cell_of(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("task_type")), str(record.get("dataset") or "unknown")


def check_floor_budget(n_cells: int, min_rows_per_cell: int, budget_total: int | None) -> dict[str, Any]:
    """Fail closed when the eventual coverage budget cannot give every cell its floor."""
    needed = n_cells * min_rows_per_cell
    report = {"cells": n_cells, "min_rows_per_cell": min_rows_per_cell, "floor_rows_needed": needed,
              "budget_total": budget_total}
    if budget_total is not None and needed > budget_total:
        raise ValueError(
            "coverage blend floor cannot be met under the row cap: "
            f"{n_cells} cells x {min_rows_per_cell} rows = {needed} > budget {budget_total}"
        )
    return report


def _rank(seed: int, record_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{record_id}".encode()).digest(), "big")


def select_coverage(candidates: Iterable[dict[str, Any]], *, new_rows: int, mode: str, min_rows_per_cell: int,
                    seed: int, is_excluded: Callable[[dict[str, Any]], str | None],
                    prior_cell_counts: dict[tuple[str, str], int] | None = None
                    ) -> tuple[list[tuple[dict[str, Any], bool]], dict[str, Any]]:
    """Pick ``new_rows`` coverage rows; returns ``[(record, is_fallback_correct)]`` and a report."""
    if mode not in COVERAGE_MODES:
        raise ValueError(f"unknown coverage blend mode {mode!r}")
    prior = dict(prior_cell_counts or {})
    primary: dict[tuple[str, str], list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    fallback: dict[tuple[str, str], list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    skipped: Counter[str] = Counter()
    eligible = 0
    for record in candidates:
        task = str(record.get("task_type"))
        if task not in TASK_SPECS:
            skipped["unsupported_task"] += 1
            continue
        reason = is_excluded(record)
        if reason:
            skipped[reason] += 1
            continue
        eligible += 1
        status = str(record.get(POOL_STATUS_KEY) or "unscored")
        item = (_rank(seed, str(record["id"])), str(record["id"]), record)
        cell = cell_of(record)
        if mode == "plain":
            primary[cell].append(item)
        elif status == "residual":
            primary[cell].append(item)
        elif status == "correct":
            fallback[cell].append(item)
        else:
            skipped["unscored_in_residual_mode"] += 1
    cells = sorted(set(primary) | set(fallback) | set(prior))
    for pool in (primary, fallback):
        for cell in pool:
            pool[cell].sort(key=lambda entry: (entry[0], entry[1]))
    positions = {cell: [0, 0] for cell in cells}
    taken: Counter[tuple[str, str]] = Counter()
    fallback_taken: Counter[tuple[str, str]] = Counter()
    selected: list[tuple[dict[str, Any], bool]] = []

    def has_rows(cell: tuple[str, str]) -> bool:
        return positions[cell][0] < len(primary.get(cell, ())) or positions[cell][1] < len(fallback.get(cell, ()))

    while len(selected) < new_rows:
        open_cells = [cell for cell in cells if has_rows(cell)]
        if not open_cells:
            break
        # Feed the cell with the fewest cumulative coverage rows first (floor-first, then even spread).
        cell = min(open_cells, key=lambda c: (prior.get(c, 0) + taken[c], c))
        if positions[cell][0] < len(primary.get(cell, ())):
            record = primary[cell][positions[cell][0]][2]
            positions[cell][0] += 1
            selected.append((record, False))
        else:
            record = fallback[cell][positions[cell][1]][2]
            positions[cell][1] += 1
            fallback_taken[cell] += 1
            selected.append((record, True))
        taken[cell] += 1
    cumulative = {cell: prior.get(cell, 0) + taken[cell] for cell in cells}
    below_floor = {f"{cell[0]}|{cell[1]}": count for cell, count in sorted(cumulative.items()) if count < min_rows_per_cell}
    status_counts = Counter(str(record.get(POOL_STATUS_KEY) or "unscored") for record, _ in selected)
    report = {
        "mode": mode, "requested_new_rows": new_rows, "selected_rows": len(selected), "eligible_candidates": eligible,
        "skipped": dict(sorted(skipped.items())), "cells": len(cells), "min_rows_per_cell": min_rows_per_cell,
        "per_cell_new": {f"{cell[0]}|{cell[1]}": taken[cell] for cell in cells if taken[cell]},
        "per_cell_cumulative": {f"{cell[0]}|{cell[1]}": count for cell, count in sorted(cumulative.items())},
        "fallback_correct_rows": sum(fallback_taken.values()),
        "fallback_per_cell": {f"{cell[0]}|{cell[1]}": n for cell, n in sorted(fallback_taken.items()) if n},
        "cells_below_floor_after_selection": below_floor,
        "shortage": max(0, new_rows - len(selected)),
        "selected_pool_status_counts": dict(sorted(status_counts.items())),
        "seed": seed,
    }
    return selected, report


def write_manifest(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "COVERAGE_MARK", "COVERAGE_MODES", "COVERAGE_SOURCE_KIND", "POOL_STATUS_KEY", "cell_of", "check_floor_budget",
    "joint_targets", "select_coverage", "validate_coverage_config", "write_manifest",
]
