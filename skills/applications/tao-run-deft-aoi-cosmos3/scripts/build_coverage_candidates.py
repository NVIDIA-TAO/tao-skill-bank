#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Build the coverage-blend candidate file for one mode (plain or residual).

Inputs
  --mining      canonical Mining JSONL (streamed; never loaded whole)
  --scored-ids  JSONL with one row per scored pool id: {id, task_type, dataset,
                is_residual, row_score, error_type} (exported from
                pool_residual.parquet by the operator)
Output
  <output-dir>/coverage_candidates_<mode>.jsonl
      plain    : six-task pool rows, at most --per-cell per (task, dataset),
                 chosen uniformly by a stable seed/id hash regardless of score
      residual : rows with is_residual true, at most --per-cell per cell, plus
                 at most --fallback-per-cell correct rows per cell so a cell
                 whose residual rows run out can fall back (tagged by the assembler)
      every row gains the inert top-level key ``deft_pool_status``
      (correct | residual | unscored)
  <output-dir>/coverage_candidates_<mode>_manifest.json  counts, caps, seals

Analysis-side tool: it does not change the loop unless the assembler is
launched with --coverage-blend-source pointing at its output.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import pathlib
import sys
from collections import Counter, defaultdict

from anchor_rows import sha256_file
from coverage_rows import COVERAGE_MODES, POOL_STATUS_KEY
from nvpaw_annotations import TASK_SPECS


def _rank(seed: int, record_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{record_id}".encode()).digest(), "big")


def load_pool_status(path: pathlib.Path) -> dict[str, str]:
    status: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("task_type")) not in TASK_SPECS:
                continue
            if row.get("is_residual"):
                status[str(row["id"])] = "residual"
            elif row.get("row_score") == 1.0:
                status[str(row["id"])] = "correct"
    return status


def build(mining: pathlib.Path, scored_ids: pathlib.Path, output_dir: pathlib.Path, *, mode: str,
          per_cell: int, fallback_per_cell: int, seed: int) -> dict:
    if mode not in COVERAGE_MODES:
        raise ValueError("--mode must be plain or residual")
    if per_cell <= 0 or fallback_per_cell < 0:
        raise ValueError("--per-cell must be positive and --fallback-per-cell >= 0")
    status_of = load_pool_status(scored_ids)
    reservoirs: dict[tuple[str, str, str], list] = defaultdict(list)
    caps = {"primary": per_cell, "fallback": fallback_per_cell}
    seen_pool = 0
    pool_cells: Counter = Counter()
    with mining.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task = str(row.get("task_type"))
            if task not in TASK_SPECS:
                continue
            seen_pool += 1
            record_id = str(row.get("id"))
            dataset = str(row.get("dataset") or "unknown")
            status = status_of.get(record_id, "unscored")
            pool_cells[(task, dataset, status)] += 1
            if mode == "plain":
                bucket = "primary"
            elif status == "residual":
                bucket = "primary"
            elif status == "correct" and fallback_per_cell > 0:
                bucket = "fallback"
            else:
                continue
            row[POOL_STATUS_KEY] = status
            heap = reservoirs[(task, dataset, bucket)]
            heapq.heappush(heap, (-_rank(seed, record_id), record_id, json.dumps(row, ensure_ascii=False, separators=(",", ":"))))
            if len(heap) > caps[bucket]:
                heapq.heappop(heap)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"coverage_candidates_{mode}.jsonl"
    counts: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    written = 0
    with out.open("w", encoding="utf-8") as w:
        for key in sorted(reservoirs):
            rows = sorted(reservoirs[key], key=lambda item: (-item[0], item[1]))
            counts[key[0]][key[1]][key[2]] = len(rows)
            for _, _, line in rows:
                w.write(line + "\n")
                written += 1
    manifest = {
        "schema_version": "nvpaw_coverage_candidates_v1",
        "mode": mode,
        "purpose": f"{mode} cross-dataset coverage blend for the cumulative training set (assembler --coverage-blend-source)",
        "inputs": {"mining": {"path": str(mining.resolve()), "sha256": sha256_file(mining)},
                   "scored_ids": {"path": str(scored_ids.resolve()), "sha256": sha256_file(scored_ids)}},
        "seed": seed, "per_cell_cap": per_cell, "fallback_per_cell_cap": fallback_per_cell if mode == "residual" else 0,
        "pool_rows_seen_six_tasks": seen_pool,
        "pool_cells_by_status": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in sorted(pool_cells.items())},
        "written_rows": written,
        "rows_by_task_dataset_bucket": {task: {ds: dict(sorted(b.items())) for ds, b in sorted(cells.items())}
                                        for task, cells in sorted(counts.items())},
        "output": {"path": str(out.resolve()), "sha256": sha256_file(out)},
    }
    (output_dir / f"coverage_candidates_{mode}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mining", type=pathlib.Path, required=True)
    parser.add_argument("--scored-ids", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--mode", choices=COVERAGE_MODES, required=True)
    parser.add_argument("--per-cell", type=int, default=3000, help="max primary rows per (task, dataset) cell")
    parser.add_argument("--fallback-per-cell", type=int, default=300, help="residual mode: max correct rows per cell kept for fallback")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    try:
        manifest = build(args.mining, args.scored_ids, args.output_dir, mode=args.mode, per_cell=args.per_cell,
                         fallback_per_cell=args.fallback_per_cell, seed=args.seed)
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        print(f"build_coverage_candidates: {exc}", file=sys.stderr)
        return 2
    print(f"coverage candidates ({manifest['mode']}): {manifest['written_rows']} rows from "
          f"{manifest['pool_rows_seen_six_tasks']} six-task pool rows -> {manifest['output']['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
