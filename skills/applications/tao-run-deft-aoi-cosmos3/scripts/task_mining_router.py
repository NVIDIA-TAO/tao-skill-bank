#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Route image-embedding neighbours through an explicit multi-task policy."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import Counter
from typing import Any, Iterable

from nvpaw_annotations import TASK_SPECS
from validate_sharegpt import (
    image_paths,
    load_records,
    prompt_and_response,
    resolve_image,
    target_path,
)


MINING_ROUTER_MODES = ("image_only", "task_strict", "task_then_fallback")
_ROUTE_TIER_PRIORITY = {"strict": 0, "image_only": 1, "fallback": 2}


def _string_list(value: Any, *, context: str, allow_empty: bool = False) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{context}: expected a string list") from exc
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{context}: expected a string list")
    result = sorted(set(value))
    if not result and not allow_empty:
        raise ValueError(f"{context}: list must not be empty")
    return result


def _embedding(value: Any, *, context: str) -> list[float]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{context}: invalid embedding JSON") from exc
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{context}: embedding must be a non-empty list")
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: embedding contains a non-numeric value") from exc
    if not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{context}: embedding contains a non-finite value")
    norm = math.sqrt(sum(item * item for item in vector))
    if norm == 0.0:
        raise ValueError(f"{context}: embedding has zero norm")
    return [item / norm for item in vector]


def _path_keys(path_text: str, media_root: pathlib.Path) -> set[str]:
    normalized = path_text.replace("\\", "/").rstrip("/")
    return {
        normalized,
        str(resolve_image(path_text, media_root)),
        pathlib.PurePosixPath(normalized).name,
    }


def _source_catalog(
    records: list[dict[str, Any]], media_root: pathlib.Path
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]], set[str]]:
    catalog: dict[str, dict[str, Any]] = {}
    aliases: dict[str, set[str]] = {}
    ignored_aliases: set[str] = set()
    for index, record in enumerate(records):
        context = f"source annotation[{index}]"
        task_type = record.get("task_type")
        if task_type not in TASK_SPECS:
            paths = image_paths(record, context=context)
            if paths:
                ignored_aliases.update(_path_keys(paths[-1], media_root))
            continue
        prompt_and_response(record, context=context)
        record_id = record.get("id")
        source_target_path = target_path(record, context=context)
        target_id = record.get("target_id", source_target_path)
        if not all(
            isinstance(value, str) and value
            for value in (task_type, target_id, record_id)
        ):
            raise ValueError(f"{context}: id, target_id, and task_type are required")
        canonical = str(resolve_image(source_target_path, media_root))
        entry = catalog.setdefault(
            canonical,
            {
                "target_id": target_id,
                "task_types": set(),
                "record_ids": set(),
            },
        )
        if entry["target_id"] != target_id:
            raise ValueError(
                f"{context}: target path {source_target_path!r} maps to conflicting target_ids"
            )
        entry["task_types"].add(task_type)
        entry["record_ids"].add(record_id)
        for key in _path_keys(source_target_path, media_root):
            aliases.setdefault(key, set()).add(canonical)
    return catalog, aliases, ignored_aliases


def _source_metadata(
    filepath: str,
    *,
    media_root: pathlib.Path,
    catalog: dict[str, dict[str, Any]],
    aliases: dict[str, set[str]],
) -> dict[str, Any]:
    resolved = str(resolve_image(filepath, media_root))
    if resolved in catalog:
        return catalog[resolved]
    matches: set[str] = set()
    for key in _path_keys(filepath, media_root):
        matches.update(aliases.get(key, set()))
    if not matches:
        raise ValueError(f"source embedding path has no Mining annotation: {filepath!r}")
    if len(matches) > 1:
        raise ValueError(
            f"source embedding path is ambiguous by basename: {filepath!r}; "
            f"matches={sorted(matches)}"
        )
    return catalog[next(iter(matches))]


def _prepare_sources(
    rows: list[dict[str, Any]],
    *,
    media_root: pathlib.Path,
    catalog: dict[str, dict[str, Any]],
    aliases: dict[str, set[str]],
    ignored_aliases: set[str],
) -> tuple[list[dict[str, Any]], int, int]:
    prepared: list[dict[str, Any]] = []
    dimensions: set[int] = set()
    seen: set[str] = set()
    ignored = 0
    for index, row in enumerate(rows):
        filepath = row.get("filepath")
        if not isinstance(filepath, str) or not filepath:
            raise ValueError(f"source embedding[{index}]: filepath is required")
        path_key = str(resolve_image(filepath, media_root))
        if path_key in seen:
            raise ValueError(f"duplicate source embedding filepath: {filepath!r}")
        seen.add(path_key)
        vector = _embedding(row.get("embedding"), context=f"source embedding[{index}]")
        try:
            metadata = _source_metadata(
                filepath,
                media_root=media_root,
                catalog=catalog,
                aliases=aliases,
            )
        except ValueError as exc:
            if "has no Mining annotation" in str(exc) and (
                _path_keys(filepath, media_root) & ignored_aliases
            ):
                ignored += 1
                continue
            raise
        dimensions.add(len(vector))
        prepared.append(
            {
                "filepath": filepath,
                "source_target_id": metadata["target_id"],
                "source_task_types": sorted(metadata["task_types"]),
                "source_record_ids": sorted(metadata["record_ids"]),
                "embedding": vector,
            }
        )
    if not prepared:
        raise ValueError("source embeddings are empty")
    if len(dimensions) != 1:
        raise ValueError("source embedding dimensions are inconsistent")
    return prepared, next(iter(dimensions)), ignored


def _prepare_targets(
    rows: list[dict[str, Any]], *, expected_dimension: int
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        filepath = row.get("filepath")
        target_id = row.get("target_id")
        if not all(isinstance(value, str) and value for value in (filepath, target_id)):
            raise ValueError(f"target embedding[{index}]: filepath and target_id are required")
        if target_id in seen:
            raise ValueError(f"duplicate target embedding target_id: {target_id!r}")
        seen.add(target_id)
        vector = _embedding(row.get("embedding"), context=f"target embedding[{index}]")
        if len(vector) != expected_dimension:
            raise ValueError(
                f"target embedding[{index}]: dimension {len(vector)} does not match "
                f"source dimension {expected_dimension}"
            )
        prepared.append(
            {
                "filepath": filepath,
                "target_id": target_id,
                "task_types": _string_list(
                    row.get("task_types"), context=f"target embedding[{index}].task_types"
                ),
                "embedding": vector,
            }
        )
    if not prepared:
        raise ValueError("target embeddings are empty")
    return sorted(prepared, key=lambda row: (row["filepath"], row["target_id"]))


def _candidate(
    *,
    source: dict[str, Any],
    target: dict[str, Any],
    similarity: float,
    route_tier: str,
    query_task_types: Iterable[str],
    routed_task_types: Iterable[str],
    rank: int,
) -> dict[str, Any]:
    return {
        "filepath": source["filepath"],
        "source_target_id": source["source_target_id"],
        "source_task_types": source["source_task_types"],
        "source_record_ids": source["source_record_ids"],
        "matched_target_filepath": target["filepath"],
        "matched_target_id": target["target_id"],
        "matched_target_ids": [target["target_id"]],
        "query_task_types": sorted(set(query_task_types)),
        "routed_task_types": sorted(set(routed_task_types)),
        "route_tier": route_tier,
        "route_tiers": [route_tier],
        "max_cosine_similarity": similarity,
        "best_rank": rank,
    }


def _merge_candidate(existing: dict[str, Any], candidate: dict[str, Any]) -> None:
    for field in ("matched_target_ids", "query_task_types", "routed_task_types", "route_tiers"):
        existing[field] = sorted(set(existing[field]) | set(candidate[field]))
    existing["best_rank"] = min(existing["best_rank"], candidate["best_rank"])
    if candidate["max_cosine_similarity"] > existing["max_cosine_similarity"]:
        existing["max_cosine_similarity"] = candidate["max_cosine_similarity"]
        existing["matched_target_filepath"] = candidate["matched_target_filepath"]
        existing["matched_target_id"] = candidate["matched_target_id"]
    existing["route_tier"] = min(
        existing["route_tiers"], key=lambda tier: _ROUTE_TIER_PRIORITY[tier]
    )


def route_candidates(
    target_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    source_annotations: list[dict[str, Any]],
    *,
    media_root: pathlib.Path,
    mode: str,
    top_k_per_target: int,
    min_similarity: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return routed source candidates and auditable selection evidence."""

    if mode not in MINING_ROUTER_MODES:
        raise ValueError(
            f"unsupported mining router mode {mode!r}; choose one of {MINING_ROUTER_MODES}"
        )
    if type(top_k_per_target) is not int or top_k_per_target <= 0:
        raise ValueError("top_k_per_target must be a positive integer")
    if not -1.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be between -1 and 1")
    media_root = media_root.expanduser().resolve()
    catalog, aliases, ignored_aliases = _source_catalog(source_annotations, media_root)
    sources, dimension, ignored_sources = _prepare_sources(
        source_rows,
        media_root=media_root,
        catalog=catalog,
        aliases=aliases,
        ignored_aliases=ignored_aliases,
    )
    targets = _prepare_targets(target_rows, expected_dimension=dimension)
    try:
        import numpy as np
    except ImportError as exc:
        raise ValueError("numpy is required for batched mining similarity") from exc
    source_matrix = np.asarray(
        [source["embedding"] for source in sources], dtype=np.float64
    )
    # Bound each similarity block to roughly 64 MiB. This keeps large source
    # pools from materializing a target_count x source_count matrix while still
    # using vectorized BLAS instead of Python loops over embedding dimensions.
    score_bytes_per_target = max(1, len(sources) * 8)
    target_batch_size = max(1, min(256, (64 * 1024 * 1024) // score_bytes_per_target))

    selected: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    target_evidence: list[dict[str, Any]] = []
    raw_tiers: Counter[str] = Counter()
    target_scores: Iterable[tuple[dict[str, Any], Any]] = (
        (target, scores)
        for start in range(0, len(targets), target_batch_size)
        for target, scores in zip(
            targets[start : start + target_batch_size],
            np.asarray(
                [
                    target["embedding"]
                    for target in targets[start : start + target_batch_size]
                ],
                dtype=np.float64,
            )
            @ source_matrix.T,
            strict=True,
        )
    )
    for target, similarities in target_scores:
        target_tasks = set(target["task_types"])
        scored = [
            (
                max(-1.0, min(1.0, float(similarity))),
                source,
                sorted(target_tasks & set(source["source_task_types"])),
            )
            for source, similarity in zip(sources, similarities, strict=True)
            if float(similarity) >= min_similarity
        ]
        scored.sort(key=lambda item: (-item[0], item[1]["filepath"]))
        chosen: list[
            tuple[float, dict[str, Any], list[str], list[str], str]
        ]
        task_routes: dict[str, dict[str, int]] = {}
        if mode == "image_only":
            chosen = [
                (
                    similarity,
                    source,
                    target["task_types"],
                    source["source_task_types"],
                    "image_only",
                )
                for similarity, source, _ in scored[:top_k_per_target]
            ]
        else:
            # A multi-prompt physical target is one embedding but several task
            # routes. Allocate top-K independently per task so an easier task
            # cannot consume the neighborhood intended for a weaker task.
            chosen = []
            for task_type in target["task_types"]:
                strict_pool = [
                    item for item in scored if task_type in item[1]["source_task_types"]
                ]
                strict_chosen = strict_pool[:top_k_per_target]
                task_chosen = [
                    (similarity, source, [task_type], [task_type], "strict")
                    for similarity, source, _ in strict_chosen
                ]
                if mode == "task_then_fallback":
                    remaining = top_k_per_target - len(task_chosen)
                    if remaining:
                        strict_paths = {
                            source["filepath"] for _, source, _ in strict_chosen
                        }
                        fallback_pool = [
                            item
                            for item in scored
                            if item[1]["filepath"] not in strict_paths
                        ]
                        task_chosen.extend(
                            (
                                similarity,
                                source,
                                [task_type],
                                source["source_task_types"],
                                "fallback",
                            )
                            for similarity, source, _ in fallback_pool[:remaining]
                        )
                chosen.extend(task_chosen)
                strict_selected = sum(
                    tier == "strict" for _, _, _, _, tier in task_chosen
                )
                fallback_selected = sum(
                    tier == "fallback" for _, _, _, _, tier in task_chosen
                )
                task_routes[task_type] = {
                    "strict_eligible": len(strict_pool),
                    "strict_selected": strict_selected,
                    "fallback_selected": fallback_selected,
                    "selected": len(task_chosen),
                    "shortfall": top_k_per_target - len(task_chosen),
                }

        tier_counts: Counter[str] = Counter()
        for rank, (similarity, source, query_tasks, routed_tasks, tier) in enumerate(
            chosen, start=1
        ):
            tier_counts[tier] += 1
            raw_tiers[tier] += 1
            candidate = _candidate(
                source=source,
                target=target,
                similarity=similarity,
                route_tier=tier,
                query_task_types=query_tasks,
                routed_task_types=routed_tasks,
                rank=rank,
            )
            key = str(resolve_image(candidate["filepath"], media_root))
            if key not in selected:
                selected[key] = candidate
                order.append(key)
            else:
                _merge_candidate(selected[key], candidate)
        expected = (
            top_k_per_target
            if mode == "image_only"
            else top_k_per_target * len(target["task_types"])
        )
        strict_eligible = len(
            {source["filepath"] for _, source, intersection in scored if intersection}
        )
        target_evidence.append(
            {
                "target_id": target["target_id"],
                "filepath": target["filepath"],
                "task_types": target["task_types"],
                "similarity_qualified": len(scored),
                "strict_eligible": strict_eligible,
                "selected": len(chosen),
                "shortfall": expected - len(chosen),
                "route_tier_counts": dict(sorted(tier_counts.items())),
                "task_routes": task_routes,
            }
        )

    output = [selected[key] for key in order]
    final_tiers = Counter(row["route_tier"] for row in output)
    task_records: Counter[str] = Counter()
    for row in output:
        task_records.update(row["routed_task_types"])
    summary = {
        "schema_version": "task_mining_router_v1",
        "mode": mode,
        "top_k_per_target": top_k_per_target,
        "min_similarity": min_similarity,
        "target_queries": len(targets),
        "source_images": len(sources),
        "ignored_out_of_scope_source_images": ignored_sources,
        "embedding_dimension": dimension,
        "similarity_batch_size": target_batch_size,
        "raw_selections": sum(raw_tiers.values()),
        "unique_sources": len(output),
        "duplicates_collapsed": sum(raw_tiers.values()) - len(output),
        "raw_route_tier_counts": dict(sorted(raw_tiers.items())),
        "route_tier_counts": dict(sorted(final_tiers.items())),
        "routed_task_records": dict(sorted(task_records.items())),
        "targets": target_evidence,
    }
    return output, summary


def _read_parquet(path: pathlib.Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("pyarrow is required to read mining embeddings") from exc
    return pq.read_table(path).to_pylist()


def _write_parquet(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("mining router selected zero candidates")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("pyarrow is required to write routed mining candidates") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-embeddings", required=True, type=pathlib.Path)
    parser.add_argument("--source-embeddings", required=True, type=pathlib.Path)
    parser.add_argument("--source-annotations", required=True, type=pathlib.Path)
    parser.add_argument("--media-root", required=True, type=pathlib.Path)
    parser.add_argument("--mode", choices=MINING_ROUTER_MODES, default="image_only")
    parser.add_argument("--top-k-per-target", type=int, default=5)
    parser.add_argument("--min-similarity", type=float, default=0.9)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--summary", required=True, type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        rows, summary = route_candidates(
            _read_parquet(args.target_embeddings),
            _read_parquet(args.source_embeddings),
            load_records(args.source_annotations),
            media_root=args.media_root,
            mode=args.mode,
            top_k_per_target=args.top_k_per_target,
            min_similarity=args.min_similarity,
        )
        _write_parquet(args.output, rows)
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    except (ImportError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"task_mining_router: {exc}", file=sys.stderr)
        return 2
    print(
        f"task_mining_router: mode={args.mode} targets={summary['target_queries']} "
        f"sources={summary['unique_sources']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
