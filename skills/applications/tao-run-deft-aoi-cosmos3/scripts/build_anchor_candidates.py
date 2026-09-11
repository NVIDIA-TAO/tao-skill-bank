#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Build the anchor candidate file: pool rows the scored checkpoint already gets right.

Inputs
  --mining      canonical Mining JSONL (streamed; never loaded whole)
  --scored-ids  JSONL with one row per scored pool id: {id, task_type, dataset,
                is_residual, row_score, error_type} (exported from
                pool_residual.parquet by the operator)
Output
  <output-dir>/anchor_candidates.jsonl  correct rows of the six supported tasks,
                at most --per-cell rows per (task, dataset), chosen by a stable
                seed/id hash so the file is independent of input order
  <output-dir>/anchor_candidates_manifest.json  counts, caps, input seals

Analysis-side tool: it does not change the loop unless the assembler is
launched with --anchor-source pointing at its output.
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
from nvpaw_annotations import TASK_SPECS


def _rank(seed: int, record_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{record_id}".encode()).digest(), "big")


def load_correct_ids(path: pathlib.Path) -> tuple[dict[str, tuple[str, str]], Counter]:
    correct: dict[str, tuple[str, str]] = {}
    counts: Counter = Counter()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task = str(row.get("task_type"))
            counts[(task, bool(row.get("is_residual")))] += 1
            if task in TASK_SPECS and not row.get("is_residual") and row.get("row_score") == 1.0:
                correct[str(row["id"])] = (task, str(row.get("dataset") or "unknown"))
    return correct, counts


def build(mining: pathlib.Path, scored_ids: pathlib.Path, output_dir: pathlib.Path, *, per_cell: int, seed: int) -> dict:
    if per_cell <= 0:
        raise ValueError("--per-cell must be positive")
    correct, scored_counts = load_correct_ids(scored_ids)
    reservoirs: dict[tuple[str, str], list] = defaultdict(list)
    seen_pool = 0
    matched = 0
    with mining.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            seen_pool += 1
            # cheap pre-filter: find the id without parsing the whole row
            head = line.find('"id"')
            row = json.loads(line)
            record_id = str(row.get("id"))
            cell = correct.get(record_id)
            if cell is None:
                continue
            matched += 1
            heap = reservoirs[cell]
            heapq.heappush(heap, (-_rank(seed, record_id), record_id, line))
            if len(heap) > per_cell:
                heapq.heappop(heap)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "anchor_candidates.jsonl"
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    with out.open("w", encoding="utf-8") as w:
        for cell in sorted(reservoirs):
            rows = sorted(reservoirs[cell], key=lambda item: (-item[0], item[1]))
            counts[cell[0]][cell[1]] = len(rows)
            for _, _, line in rows:
                w.write(line + "\n")
    manifest = {
        "schema_version": "nvpaw_anchor_candidates_v1",
        "purpose": "correct-row anchors for the cumulative training set (assembler --anchor-source)",
        "inputs": {"mining": {"path": str(mining.resolve()), "sha256": sha256_file(mining)},
                   "scored_ids": {"path": str(scored_ids.resolve()), "sha256": sha256_file(scored_ids)}},
        "seed": seed, "per_cell_cap": per_cell,
        "pool_rows_seen": seen_pool, "correct_ids": len(correct), "correct_rows_matched": matched,
        "scored_counts_by_task_residual": {f"{k[0]}|residual={k[1]}": v for k, v in sorted(scored_counts.items())},
        "written_rows": sum(sum(v.values()) for v in counts.values()),
        "rows_by_task_dataset": {task: dict(sorted(ds.items())) for task, ds in sorted(counts.items())},
        "output": {"path": str(out.resolve()), "sha256": sha256_file(out)},
    }
    (output_dir / "anchor_candidates_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mining", type=pathlib.Path, required=True)
    parser.add_argument("--scored-ids", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--per-cell", type=int, default=3000, help="max rows per (task, dataset) cell")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    try:
        manifest = build(args.mining, args.scored_ids, args.output_dir, per_cell=args.per_cell, seed=args.seed)
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        print(f"build_anchor_candidates: {exc}", file=sys.stderr)
        return 2
    print(f"anchor candidates: {manifest['written_rows']} rows from {manifest['correct_rows_matched']} correct pool rows -> {manifest['output']['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
