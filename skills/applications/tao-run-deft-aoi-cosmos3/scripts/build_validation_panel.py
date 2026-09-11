#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Build an analysis-only, benchmark/proxy-disjoint stratified validation panel."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
import math
import pathlib
import sys

from build_target_profile import (
    TASKS, build_profile, dataset_family, image_paths, integer_targets,
    normalize_path, read_rows, row_features, sha256_file, write_outputs,
)

STRATA = ("task_type", "dataset", "empty_status", "count_bin", "size_bin")


def _image_keys(record, media_root):
    keys = set()
    for path in image_paths(record, context=str(record.get("id"))):
        path = normalize_path(path)
        keys.add(path)
        if media_root is not None and not path.startswith("/"):
            keys.add(normalize_path(str(media_root) + "/" + path))
    return keys


def _bounded_allocate(capacities, weights, budget):
    """Allocate integer rows closest to target shares, respecting every cap."""
    allocated = dict.fromkeys(capacities, 0)
    total_weight = sum(weights[key] for key in capacities)
    for _ in range(budget):
        eligible = [key for key in capacities if allocated[key] < capacities[key]]
        if not eligible:
            raise ValueError("internal panel allocation capacity shortfall")
        key = min(eligible, key=lambda key: (
            -(budget * weights[key] / total_weight - allocated[key]), key,
        ))
        allocated[key] += 1
    return allocated


def select_panel(candidates, benchmark, proxy, *, per_task=500, seed=17, media_root=None, per_task_override=None):
    if type(per_task) is not int or per_task <= 0 or type(seed) is not int:
        raise ValueError("per_task must be positive and seed must be an integer")
    per_task_override = dict(per_task_override or {})
    for task, value in per_task_override.items():
        if task not in TASKS or type(value) is not int or value <= 0:
            raise ValueError(f"per_task_override must map a supported task to a positive integer: {task!r}={value!r}")
    def task_budget(task):
        return per_task_override.get(task, per_task)
    target = Counter()
    blocked = {}
    for name, records in (("benchmark", benchmark), ("proxy", proxy)):
        ids, paths = set(), set()
        for record in records:
            ids.add(str(record["id"]))
            paths.update(_image_keys(record, media_root))
            if name == "benchmark" and record.get("task_type") in TASKS:
                feature = row_features(record)
                target[tuple(feature[field] for field in STRATA)] += 1
        blocked[name] = (ids, paths)
    requested = {}
    for task in TASKS:
        counts = {key: count for key, count in target.items() if key[0] == task}
        if counts:
            requested.update(integer_targets(counts, task_budget(task)))
    reservoirs, available = defaultdict(list), Counter()
    exclusions, excluded_families = Counter(), Counter()
    seen = set()
    for record in candidates:
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("panel candidate requires a non-empty string id")
        if record_id in seen:
            raise ValueError(f"duplicate panel candidate id: {record_id}")
        seen.add(record_id)
        paths = _image_keys(record, media_root)
        reason = None
        for name, (ids, excluded_paths) in blocked.items():
            if record_id in ids:
                reason = name + "_id"
            elif paths & excluded_paths:
                reason = name + "_image"
            if reason:
                break
        if reason is None and record.get("task_type") not in TASKS:
            reason = "unsupported_task"
        if reason:
            exclusions[reason] += 1
            excluded_families[dataset_family(record)] += 1
            continue
        feature = row_features(record)
        key = tuple(feature[field] for field in STRATA)
        if key not in target:
            exclusions["non_target_stratum"] += 1
            continue
        available[key] += 1
        limit = requested[key]
        if limit:
            # Bounded, order-independent hash reservoir. No prediction inputs.
            rank = int.from_bytes(hashlib.sha256(f"{seed}\0{record_id}".encode()).digest(), "big")
            heapq.heappush(reservoirs[key], (-rank, record_id, record))
            if len(reservoirs[key]) > limit:
                heapq.heappop(reservoirs[key])
    selected, actual, task_reports = [], Counter(), {}
    for task in TASKS:
        capacities = {key: min(available[key], wanted) for key, wanted in requested.items() if key[0] == task}
        family_supply, family_weight = Counter(), Counter()
        for key, capacity in capacities.items():
            family_supply[key[1]] += capacity
            family_weight[key[1]] += target[key]
        # Keep shares relative to the full benchmark task, without renormalizing
        # away unavailable families. Only families with usable target-stratum
        # supply may relax the base cap.
        task_weight = sum(family_weight.values())
        eligible_shares = [family_weight[family] / task_weight
                           for family, count in family_supply.items() if count > 0]
        family_cap = max([0.35, *eligible_shares])
        cap_relaxed = family_cap > 0.35
        cap_reason = ("relaxed_to_largest_eligible_benchmark_share" if cap_relaxed else
                      "eligible_benchmark_shares_within_base_cap" if eligible_shares else
                      "no_eligible_benchmark_families")

        def row_cap(total):
            # A relaxed share must accommodate ordinary integer quota rounding
            # (e.g. 43.7% of 500 -> 219). Preserve the old floor path exactly for
            # unrelaxed tasks so their seed/id-hash selections are unchanged.
            if cap_relaxed:
                return math.ceil(total * family_cap - 1e-12)
            return math.floor(total * family_cap + 1e-12)

        accepted = 0
        for total in range(sum(capacities.values()), 0, -1):
            cap = row_cap(total)
            if sum(min(count, cap) for count in family_supply.values()) >= total:
                accepted = total
                break
        if accepted:
            cap = row_cap(accepted)
            families = _bounded_allocate(
                {key: min(count, cap) for key, count in family_supply.items()},
                family_weight, accepted,
            )
            for family, count in families.items():
                cells = {key: value for key, value in capacities.items() if key[1] == family}
                allocation = _bounded_allocate(cells, target, count)
                for key, amount in allocation.items():
                    actual[key] = amount
                    selected.extend(item[2] for item in sorted(reservoirs[key], key=lambda item: (-item[0], item[1]))[:amount])
        task_reports[task] = {"requested_rows": task_budget(task), "selected_rows": accepted,
            "shortage_rows": task_budget(task) - accepted,
            "family_cap_effective": family_cap, "family_cap_relaxed": cap_relaxed,
            "family_cap_reason": cap_reason, "family_cap_rounding": "ceil" if cap_relaxed else "floor",
            "reason": "no_benchmark_support" if not capacities else
                      "insufficient_stratum_supply_or_family_cap" if accepted < task_budget(task) else None}
    strata = []
    for key in sorted(target):
        task_total = sum(count for cell, count in target.items() if cell[0] == key[0])
        realized_total = task_reports[key[0]]["selected_rows"]
        strata.append({**dict(zip(STRATA, key)), "target_share": target[key] / task_total,
            "realized_share": actual[key] / realized_total if realized_total else 0.0,
            "requested_rows": requested[key], "available_rows": available[key],
            "selected_rows": actual[key], "shortage_rows": requested[key] - actual[key],
            "shortage_reason": "no_eligible_supply" if available[key] == 0 and requested[key] else
                "supply_shortage_or_family_cap" if actual[key] < requested[key] else None})
    selected.sort(key=lambda record: (record["task_type"], record["id"]))
    # Recheck disjointness against both exclusion sets at the output boundary.
    for record in selected:
        for ids, paths in blocked.values():
            if record["id"] in ids or _image_keys(record, media_root) & paths:
                raise ValueError("panel output is not benchmark/proxy disjoint")
    return selected, {"schema_version": "nvpaw_validation_panel_v1", "analysis_only": True,
        "seed": seed, "per_task": per_task, "per_task_override": dict(sorted(per_task_override.items())), "family_cap": 0.35,
        "rows": len(selected), "exclusions": dict(sorted(exclusions.items())),
        "excluded_by_family": dict(sorted(excluded_families.items())),
        "tasks": task_reports, "strata": strata,
        "disjointness": "id OR any normalized image path; benchmark first, then proxy",
        "allocation": "benchmark shares; capped realized family shares; no unrelated-stratum backfill"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark", type=pathlib.Path, required=True)
    parser.add_argument("--proxy", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--media-root", type=pathlib.Path)
    parser.add_argument("--per-task", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--per-task-override", action="append", default=[], metavar="TASK=ROWS",
                        help="Raise or lower the row target of one task, e.g. 'Ref_based Defect Detection=1500'. Repeatable.")
    args = parser.parse_args(argv)
    try:
        overrides = {}
        for item in args.per_task_override:
            task, sep, value = item.rpartition("=")
            if not sep or not task or not value.isdigit():
                raise ValueError(f"--per-task-override expects TASK=ROWS, got {item!r}")
            overrides[task] = int(value)
        selected, manifest = select_panel(read_rows(args.input), read_rows(args.benchmark), read_rows(args.proxy),
            per_task=args.per_task, seed=args.seed, per_task_override=overrides,
            media_root=args.media_root.expanduser().resolve() if args.media_root else None)
        manifest["inputs"] = {name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                              for name, path in (("manifest", args.input), ("benchmark", args.benchmark), ("proxy", args.proxy))}
        text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in selected)
        manifest["panel_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        payload = build_profile(selected)
        payload["panel_manifest"] = manifest
        lines = ["# Validation panel (analysis only)", "", f"Panel SHA-256: `{manifest['panel_sha256']}`", "",
                 "| Task | Requested | Selected | Shortage | family_cap_effective | family_cap_relaxed | family_cap_reason |",
                 "|---|---:|---:|---:|---:|---|---|"]
        lines.extend(f"| {task} | {item['requested_rows']} | {item['selected_rows']} | {item['shortage_rows']} | "
                     f"{item['family_cap_effective']:.6f} | {str(item['family_cap_relaxed']).lower()} | {item['family_cap_reason']} |"
                     for task, item in manifest["tasks"].items())
        lines.extend(["", "Families exhausted by exclusions are not silently backfilled. Full per-stratum shortages, input seals and exclusions:",
                      "", "```json", json.dumps(manifest, indent=2), "```", ""])
        write_outputs({args.output_dir / "validation_panel.jsonl": text,
                       args.output_dir / "validation_panel_profile.json": json.dumps(payload, indent=2) + "\n",
                       args.output_dir / "PANEL_MANIFEST.md": "\n".join(lines)})
        print(f"validation panel: {len(selected)} rows; analysis only")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"build_validation_panel: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
