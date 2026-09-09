#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Route image-embedding neighbours through an explicit multi-task policy."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
from collections import Counter
from typing import Any, Iterable

from atomic_samples import (
    PAIR_SIMILARITIES,
    PAIR_SIMILARITY_COMBINES,
    embedding_filepath,
    sample_from_record,
)
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
    records: list[dict[str, Any]],
    media_root: pathlib.Path,
    pair_assets_dir: pathlib.Path | None,
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
        sample = sample_from_record(record, media_root=media_root, context=context)
        target_id = sample["atomic_sample_id"]
        if not all(
            isinstance(value, str) and value
            for value in (task_type, target_id, record_id)
        ):
            raise ValueError(f"{context}: id, target_id, and task_type are required")
        canonical = embedding_filepath(sample, pair_assets_dir=pair_assets_dir)
        entry = catalog.setdefault(
            canonical,
            {
                "target_id": target_id,
                "atomic_sample_id": sample["atomic_sample_id"],
                "sample_kind": sample["sample_kind"],
                "image_paths": sample["image_paths"],
                "reference_filepath": sample["reference_filepath"],
                "target_filepath": sample["target_filepath"],
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
        for key in _path_keys(canonical, media_root):
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
                "atomic_sample_id": metadata["atomic_sample_id"],
                "sample_kind": metadata["sample_kind"],
                "source_image_paths": metadata["image_paths"],
                "source_reference_filepath": metadata["reference_filepath"],
                "source_target_filepath": metadata["target_filepath"],
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
                "defect_detection_evidence": _string_list(
                    row.get("defect_detection_evidence", []),
                    context=f"target embedding[{index}].defect_detection_evidence",
                    allow_empty=True,
                ),
                "sample_kind": row.get("sample_kind", "single_image"),
                "image_paths": row.get("image_paths", [filepath]),
                "reference_filepath": row.get("reference_filepath"),
                "target_filepath": row.get("target_filepath", filepath),
                "embedding": vector,
            }
        )
    if not prepared:
        raise ValueError("target embeddings are empty")
    return prepared


def _atomic_source_catalog(
    records: list[dict[str, Any]], media_root: pathlib.Path
) -> tuple[list[dict[str, Any]], set[str]]:
    catalog: dict[str, dict[str, Any]] = {}
    ignored_paths: set[str] = set()
    resolved_path_cache: dict[str, str] = {}
    for index, record in enumerate(records):
        context = f"source annotation[{index}]"
        task_type = record.get("task_type")
        if task_type not in TASK_SPECS:
            paths = image_paths(record, context=context)
            ignored_paths.update(str(resolve_image(path, media_root)) for path in paths)
            continue
        prompt_and_response(record, context=context)
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{context}: id is required")
        sample = sample_from_record(
            record,
            media_root=media_root,
            context=context,
            resolved_path_cache=resolved_path_cache,
        )
        identity = str(sample["atomic_sample_id"])
        entry = catalog.setdefault(
            identity,
            {
                "filepath": str(sample["target_filepath"]),
                "source_target_id": identity,
                "source_task_types": set(),
                "source_record_ids": set(),
                "atomic_sample_id": identity,
                "sample_kind": sample["sample_kind"],
                "source_image_paths": sample["image_paths"],
                "source_reference_filepath": sample["reference_filepath"],
                "source_target_filepath": sample["target_filepath"],
            },
        )
        if entry["source_image_paths"] != sample["image_paths"]:
            raise ValueError(f"{context}: atomic sample maps to conflicting image paths")
        entry["source_task_types"].add(str(task_type))
        entry["source_record_ids"].add(record_id)
    output = []
    for entry in catalog.values():
        entry["source_task_types"] = sorted(entry["source_task_types"])
        entry["source_record_ids"] = sorted(entry["source_record_ids"])
        output.append(entry)
    return output, ignored_paths


def _component_embedding_index(
    rows: list[dict[str, Any]],
    *,
    media_root: pathlib.Path,
    context: str,
) -> tuple[dict[str, list[float]], int]:
    embeddings: dict[str, list[float]] = {}
    dimensions: set[int] = set()
    for index, row in enumerate(rows):
        filepath = row.get("filepath")
        if not isinstance(filepath, str) or not filepath:
            raise ValueError(f"{context}[{index}]: filepath is required")
        resolved = str(resolve_image(filepath, media_root))
        if resolved in embeddings:
            raise ValueError(f"duplicate {context} filepath: {filepath!r}")
        vector = _embedding(row.get("embedding"), context=f"{context}[{index}]")
        embeddings[resolved] = vector
        dimensions.add(len(vector))
    if not embeddings:
        raise ValueError(f"{context}s are empty")
    if len(dimensions) != 1:
        raise ValueError(f"{context} dimensions are inconsistent")
    return embeddings, next(iter(dimensions))


def _prepare_two_vector_sources(
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    media_root: pathlib.Path,
) -> tuple[list[dict[str, Any]], int, int]:
    embeddings, dimension = _component_embedding_index(
        rows, media_root=media_root, context="source embedding"
    )
    catalog, ignored_paths = _atomic_source_catalog(records, media_root)
    prepared: list[dict[str, Any]] = []
    used_paths: set[str] = set()
    for source in catalog:
        test_path = str(source["source_target_filepath"])
        test_embedding = embeddings.get(test_path)
        if test_embedding is None:
            raise ValueError(
                f"source atomic sample {source['atomic_sample_id']!r} has no cached "
                f"test embedding for {test_path!r}"
            )
        used_paths.add(test_path)
        item = {**source, "test_embedding": test_embedding}
        if source["sample_kind"] == "reference_pair":
            golden_path = str(source["source_reference_filepath"])
            golden_embedding = embeddings.get(golden_path)
            if golden_embedding is None:
                raise ValueError(
                    f"source atomic sample {source['atomic_sample_id']!r} has no "
                    f"golden embedding for {golden_path!r}"
                )
            used_paths.add(golden_path)
            item["golden_embedding"] = golden_embedding
        else:
            item["golden_embedding"] = None
        prepared.append(item)
    unknown = set(embeddings) - used_paths - ignored_paths
    if unknown:
        raise ValueError(
            "source component embedding has no Mining atomic sample: "
            f"{sorted(unknown)[:10]}"
        )
    return prepared, dimension, len(set(embeddings) & ignored_paths)


def _embedding_memberships(value: Any, *, context: str) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{context}: invalid embedding_memberships JSON") from exc
    if not isinstance(value, (list, tuple)) or not value or not all(
        isinstance(item, dict) for item in value
    ):
        raise ValueError(f"{context}: embedding_memberships must be an object list")
    return [dict(item) for item in value]


def _prepare_two_vector_targets(
    rows: list[dict[str, Any]],
    *,
    media_root: pathlib.Path,
    expected_dimension: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    seen_component_paths: set[str] = set()
    for index, row in enumerate(rows):
        filepath = row.get("filepath")
        if not isinstance(filepath, str) or not filepath:
            raise ValueError(f"target embedding[{index}]: filepath is required")
        resolved = str(resolve_image(filepath, media_root))
        if resolved in seen_component_paths:
            raise ValueError(f"duplicate target embedding filepath: {filepath!r}")
        seen_component_paths.add(resolved)
        vector = _embedding(row.get("embedding"), context=f"target embedding[{index}]")
        if len(vector) != expected_dimension:
            raise ValueError(
                f"target embedding[{index}]: dimension {len(vector)} does not match "
                f"source dimension {expected_dimension}"
            )
        memberships = _embedding_memberships(
            row.get("embedding_memberships"), context=f"target embedding[{index}]"
        )
        for membership_index, membership in enumerate(memberships):
            context = f"target embedding[{index}].membership[{membership_index}]"
            target_id = membership.get("target_id")
            role = membership.get("role")
            sample_kind = membership.get("sample_kind")
            if not isinstance(target_id, str) or not target_id:
                raise ValueError(f"{context}: target_id is required")
            if role not in {"golden", "test"}:
                raise ValueError(f"{context}: role must be golden or test")
            if sample_kind not in {"single_image", "reference_pair"}:
                raise ValueError(f"{context}: invalid sample_kind")
            image_paths_value = membership.get("image_paths")
            if not isinstance(image_paths_value, (list, tuple)) or len(
                image_paths_value
            ) != (2 if sample_kind == "reference_pair" else 1):
                raise ValueError(f"{context}: image_paths do not match sample_kind")
            image_paths_value = [str(value) for value in image_paths_value]
            metadata = {
                "filepath": str(membership.get("target_filepath") or image_paths_value[-1]),
                "target_id": target_id,
                "task_types": _string_list(
                    membership.get("task_types"), context=f"{context}.task_types"
                ),
                "defect_detection_evidence": _string_list(
                    membership.get("defect_detection_evidence", []),
                    context=f"{context}.defect_detection_evidence",
                    allow_empty=True,
                ),
                "sample_kind": sample_kind,
                "image_paths": image_paths_value,
                "reference_filepath": membership.get("reference_filepath"),
                "target_filepath": membership.get("target_filepath")
                or image_paths_value[-1],
            }
            if target_id not in grouped:
                grouped[target_id] = {**metadata, "golden_embedding": None, "test_embedding": None}
                order.append(target_id)
            target = grouped[target_id]
            for field, expected in metadata.items():
                if target[field] != expected:
                    raise ValueError(
                        f"{context}: target {target_id!r} has conflicting {field}"
                    )
            field = f"{role}_embedding"
            if target[field] is not None:
                raise ValueError(f"{context}: duplicate {role} embedding for {target_id!r}")
            target[field] = vector
    for target_id in order:
        target = grouped[target_id]
        if target["test_embedding"] is None:
            raise ValueError(f"target {target_id!r} has no test embedding")
        if target["sample_kind"] == "reference_pair" and target["golden_embedding"] is None:
            raise ValueError(f"target {target_id!r} has no golden embedding")
        if target["sample_kind"] == "single_image" and target["golden_embedding"] is not None:
            raise ValueError(f"single-image target {target_id!r} has a golden embedding")
    if not order:
        raise ValueError("target embeddings are empty")
    return [grouped[target_id] for target_id in order]


def _candidate(
    *,
    source: dict[str, Any],
    target: dict[str, Any],
    sim_golden: float | None,
    sim_test: float | None,
    sim_pair: float,
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
        "atomic_sample_id": source["atomic_sample_id"],
        "sample_kind": source["sample_kind"],
        "source_image_paths": source["source_image_paths"],
        "matched_target_filepath": target["filepath"],
        "matched_target_id": target["target_id"],
        "matched_target_ids": [target["target_id"]],
        "query_task_types": sorted(set(query_task_types)),
        "routed_task_types": sorted(set(routed_task_types)),
        "route_tier": route_tier,
        "route_tiers": [route_tier],
        "defect_detection_evidence": target["defect_detection_evidence"],
        "sim_golden": sim_golden,
        "sim_test": sim_test,
        "sim_pair": sim_pair,
        "max_cosine_similarity": sim_pair,
        "best_rank": rank,
    }


def _merge_candidate(existing: dict[str, Any], candidate: dict[str, Any]) -> None:
    for field in (
        "matched_target_ids",
        "query_task_types",
        "routed_task_types",
        "route_tiers",
        "defect_detection_evidence",
    ):
        existing[field] = sorted(set(existing[field]) | set(candidate[field]))
    existing["best_rank"] = min(existing["best_rank"], candidate["best_rank"])
    if candidate["sim_pair"] > existing["sim_pair"]:
        for field in ("sim_golden", "sim_test", "sim_pair", "max_cosine_similarity"):
            existing[field] = candidate[field]
        existing["matched_target_filepath"] = candidate["matched_target_filepath"]
        existing["matched_target_id"] = candidate["matched_target_id"]
    existing["route_tier"] = min(
        existing["route_tiers"], key=lambda tier: _ROUTE_TIER_PRIORITY[tier]
    )


def _board_id_prefix(path_text: str | None) -> str | None:
    if not isinstance(path_text, str) or not path_text:
        return None
    path = pathlib.PurePath(path_text.replace("\\", "/"))
    stem = path.stem
    matches = re.findall(r"[A-Za-z]+\d+(?:[@_-]\d+)?", stem)
    if matches:
        return matches[-1].casefold()
    if "__" in stem:
        prefix = stem.split("__", 1)[0].casefold()
        if prefix:
            return prefix
    ignored = {"golden", "images", "normal", "reference", "ref", "top10"}
    parent = path.parent.name.casefold()
    if parent and parent not in ignored:
        return parent
    tail = stem.split("__")[-1].casefold()
    tail = re.sub(r"(?:[_-](?:uniformlight|golden|normal|reference|ref).*)$", "", tail)
    return tail or None


def _same_board_type(
    target: dict[str, Any], source: dict[str, Any]
) -> tuple[bool, str | None]:
    query_path = target.get("reference_filepath")
    source_path = source.get("source_reference_filepath")
    if not all(isinstance(value, str) and value for value in (query_path, source_path)):
        return False, None
    if pathlib.PurePath(str(query_path)) == pathlib.PurePath(str(source_path)):
        return True, "golden_path"
    query_prefix = _board_id_prefix(str(query_path))
    source_prefix = _board_id_prefix(str(source_path))
    if query_prefix is not None and query_prefix == source_prefix:
        return True, "board_id_prefix"
    return False, None


def _clip_similarity(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def route_candidates(
    target_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    source_annotations: list[dict[str, Any]],
    *,
    media_root: pathlib.Path,
    pair_assets_dir: pathlib.Path | None = None,
    mode: str,
    top_k_per_target: int,
    top_k_by_task: dict[str, int] | None = None,
    min_similarity: float,
    pair_similarity: str = "canvas",
    pair_similarity_combine: str = "mean",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return routed source candidates and auditable selection evidence."""

    if mode not in MINING_ROUTER_MODES:
        raise ValueError(
            f"unsupported mining router mode {mode!r}; choose one of {MINING_ROUTER_MODES}"
        )
    if type(top_k_per_target) is not int or top_k_per_target <= 0:
        raise ValueError("top_k_per_target must be a positive integer")
    top_k_by_task = dict(top_k_by_task or {})
    invalid_top_k = {
        task: value
        for task, value in top_k_by_task.items()
        if task not in TASK_SPECS
        or type(value) is not int
        or value <= 0
    }
    if invalid_top_k:
        raise ValueError(f"invalid per-task top-K overrides: {invalid_top_k}")
    if not -1.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be between -1 and 1")
    if pair_similarity not in PAIR_SIMILARITIES:
        raise ValueError(
            f"unsupported pair similarity {pair_similarity!r}; "
            f"choose one of {PAIR_SIMILARITIES}"
        )
    if pair_similarity_combine not in PAIR_SIMILARITY_COMBINES:
        raise ValueError(
            f"unsupported pair similarity combine {pair_similarity_combine!r}; "
            f"choose one of {PAIR_SIMILARITY_COMBINES}"
        )
    media_root = media_root.expanduser().resolve()
    if pair_similarity == "canvas":
        catalog, aliases, ignored_aliases = _source_catalog(
            source_annotations, media_root, pair_assets_dir
        )
        sources, dimension, ignored_sources = _prepare_sources(
            source_rows,
            media_root=media_root,
            catalog=catalog,
            aliases=aliases,
            ignored_aliases=ignored_aliases,
        )
        targets = _prepare_targets(target_rows, expected_dimension=dimension)
    else:
        sources, dimension, ignored_sources = _prepare_two_vector_sources(
            source_rows, source_annotations, media_root=media_root
        )
        targets = _prepare_two_vector_targets(
            target_rows,
            media_root=media_root,
            expected_dimension=dimension,
        )
    try:
        import numpy as np
    except ImportError as exc:
        raise ValueError("numpy is required for batched mining similarity") from exc
    matrix_dtype = np.float64 if pair_similarity == "canvas" else np.float32
    source_matrix = np.asarray(
        [
            source["embedding"]
            if pair_similarity == "canvas"
            else source["test_embedding"]
            for source in sources
        ],
        dtype=matrix_dtype,
    )
    source_pair_mask = np.asarray(
        [source["sample_kind"] == "reference_pair" for source in sources],
        dtype=bool,
    )
    source_golden_matrix = None
    if pair_similarity == "two_vector":
        source_golden_matrix = np.zeros_like(source_matrix)
        for index, source in enumerate(sources):
            if source["golden_embedding"] is not None:
                source_golden_matrix[index] = source["golden_embedding"]
            source.pop("golden_embedding")
            source.pop("test_embedding")
    # Bound each similarity block to roughly 64 MiB. This keeps large source
    # pools from materializing a target_count x source_count matrix while still
    # using vectorized BLAS instead of Python loops over embedding dimensions.
    score_array_count = 1 if pair_similarity == "canvas" else 3
    score_bytes_per_target = max(
        1, len(sources) * np.dtype(matrix_dtype).itemsize * score_array_count
    )
    target_batch_size = max(1, min(256, (64 * 1024 * 1024) // score_bytes_per_target))

    selected: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    target_evidence: list[dict[str, Any]] = []
    raw_tiers: Counter[str] = Counter()
    def score_targets() -> Iterable[
        tuple[dict[str, Any], Any, Any | None, Any | None]
    ]:
        for start in range(0, len(targets), target_batch_size):
            batch = targets[start : start + target_batch_size]
            target_matrix = np.asarray(
                [
                    target["embedding"]
                    if pair_similarity == "canvas"
                    else target["test_embedding"]
                    for target in batch
                ],
                dtype=matrix_dtype,
            )
            test_scores = target_matrix @ source_matrix.T
            for target, target_test_scores in zip(batch, test_scores, strict=True):
                if pair_similarity == "canvas":
                    yield target, target_test_scores, None, None
                    continue
                target_golden_scores = None
                pair_scores = target_test_scores.copy()
                if target["sample_kind"] == "reference_pair":
                    assert source_golden_matrix is not None
                    target_golden_scores = (
                        np.asarray(target["golden_embedding"], dtype=matrix_dtype)
                        @ source_golden_matrix.T
                    )
                    combined = (
                        (target_golden_scores + target_test_scores) / 2.0
                        if pair_similarity_combine == "mean"
                        else np.minimum(target_golden_scores, target_test_scores)
                    )
                    pair_scores = np.where(source_pair_mask, combined, target_test_scores)
                yield target, pair_scores, target_golden_scores, target_test_scores

    for target, similarities, golden_similarities, test_similarities in score_targets():
        target_tasks = set(target["task_types"])
        scored = []
        for source_index, (source, similarity) in enumerate(
            zip(sources, similarities, strict=True)
        ):
            sim_pair = _clip_similarity(float(similarity))
            if sim_pair < min_similarity:
                continue
            sim_golden = (
                _clip_similarity(float(golden_similarities[source_index]))
                if golden_similarities is not None
                and source["sample_kind"] == "reference_pair"
                else None
            )
            sim_test = (
                _clip_similarity(float(test_similarities[source_index]))
                if test_similarities is not None
                else None
            )
            scored.append(
                (
                    {
                        "sim_golden": sim_golden,
                        "sim_test": sim_test,
                        "sim_pair": sim_pair,
                    },
                    source,
                    sorted(target_tasks & set(source["source_task_types"])),
                )
            )
        scored.sort(
            key=lambda item: (
                -item[0]["sim_pair"],
                item[1]["filepath"],
                item[1]["atomic_sample_id"],
            )
        )
        chosen: list[
            tuple[dict[str, float | None], dict[str, Any], list[str], list[str], str]
        ]
        task_routes: dict[str, dict[str, int]] = {}
        if mode == "image_only":
            target_top_k = max(
                [top_k_per_target]
                + [top_k_by_task.get(task, top_k_per_target) for task in target["task_types"]]
            )
            chosen = [
                (
                    similarity,
                    source,
                    target["task_types"],
                    source["source_task_types"],
                    "image_only",
                )
                for similarity, source, _ in scored[:target_top_k]
            ]
        else:
            # A multi-prompt physical target is one embedding but several task
            # routes. Allocate top-K independently per task so an easier task
            # cannot consume the neighborhood intended for a weaker task.
            chosen = []
            for task_type in target["task_types"]:
                task_top_k = top_k_by_task.get(task_type, top_k_per_target)
                strict_pool = [
                    item for item in scored if task_type in item[1]["source_task_types"]
                ]
                strict_chosen = strict_pool[:task_top_k]
                task_chosen = [
                    (similarity, source, [task_type], [task_type], "strict")
                    for similarity, source, _ in strict_chosen
                ]
                if mode == "task_then_fallback":
                    remaining = task_top_k - len(task_chosen)
                    if remaining:
                        strict_paths = {
                            source["atomic_sample_id"] for _, source, _ in strict_chosen
                        }
                        fallback_pool = [
                            item
                            for item in scored
                            if item[1]["atomic_sample_id"] not in strict_paths
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
                    "top_k": task_top_k,
                    "strict_eligible": len(strict_pool),
                    "strict_selected": strict_selected,
                    "fallback_selected": fallback_selected,
                    "selected": len(task_chosen),
                    "shortfall": task_top_k - len(task_chosen),
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
                sim_golden=similarity["sim_golden"],
                sim_test=similarity["sim_test"],
                sim_pair=float(similarity["sim_pair"]),
                route_tier=tier,
                query_task_types=query_tasks,
                routed_task_types=routed_tasks,
                rank=rank,
            )
            key = candidate["atomic_sample_id"]
            if key not in selected:
                selected[key] = candidate
                order.append(key)
            else:
                _merge_candidate(selected[key], candidate)
        expected = (
            target_top_k
            if mode == "image_only"
            else sum(
                top_k_by_task.get(task, top_k_per_target)
                for task in target["task_types"]
            )
        )
        strict_eligible = len(
            {
                source["atomic_sample_id"]
                for _, source, intersection in scored
                if intersection
            }
        )
        evidence = {
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
        if target.get("sample_kind") == "reference_pair":
            selected_pairs = {
                source["atomic_sample_id"]: source
                for _, source, _, _, _ in chosen
                if source["sample_kind"] == "reference_pair"
            }
            match_modes = Counter(
                match_mode
                for source in selected_pairs.values()
                for matched, match_mode in [_same_board_type(target, source)]
                if matched and match_mode is not None
            )
            same_board_hits = sum(match_modes.values())
            evidence.update(
                {
                    "query_golden_filepath": target.get("reference_filepath"),
                    "query_board_id_prefix": _board_id_prefix(
                        target.get("reference_filepath")
                    ),
                    "selected_reference_pairs": len(selected_pairs),
                    "same_golden_path_hits": match_modes["golden_path"],
                    "same_board_id_prefix_hits": match_modes["board_id_prefix"],
                    "same_board_type_hits": same_board_hits,
                    "same_board_type_hit_rate": (
                        same_board_hits / len(selected_pairs)
                        if selected_pairs
                        else None
                    ),
                }
            )
        target_evidence.append(evidence)

    output = [selected[key] for key in order]
    final_tiers = Counter(row["route_tier"] for row in output)
    task_records: Counter[str] = Counter()
    for row in output:
        task_records.update(row["routed_task_types"])
    summary = {
        "schema_version": "task_mining_router_v1",
        "mode": mode,
        "pair_similarity": pair_similarity,
        "pair_similarity_combine": pair_similarity_combine,
        "top_k_per_target": top_k_per_target,
        "top_k_by_task": dict(sorted(top_k_by_task.items())),
        "min_similarity": min_similarity,
        "target_queries": len(targets),
        "source_images": len(sources),
        "source_single_images": sum(
            source["sample_kind"] == "single_image" for source in sources
        ),
        "source_reference_pairs": sum(
            source["sample_kind"] == "reference_pair" for source in sources
        ),
        "source_embedding_inputs": len(source_rows),
        "target_embedding_inputs": len(target_rows),
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
    parser.add_argument("--pair-assets-dir", type=pathlib.Path)
    parser.add_argument("--pair-similarity", choices=PAIR_SIMILARITIES, default="canvas")
    parser.add_argument(
        "--pair-similarity-combine",
        choices=PAIR_SIMILARITY_COMBINES,
        default="mean",
    )
    parser.add_argument("--mode", choices=MINING_ROUTER_MODES, default="image_only")
    parser.add_argument("--top-k-per-target", type=int, default=5)
    parser.add_argument("--defect-detection-top-k-per-target", type=int)
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
            pair_assets_dir=args.pair_assets_dir,
            mode=args.mode,
            top_k_per_target=args.top_k_per_target,
            top_k_by_task=(
                {"Defect Detection": args.defect_detection_top_k_per_target}
                if args.defect_detection_top_k_per_target is not None
                else None
            ),
            min_similarity=args.min_similarity,
            pair_similarity=args.pair_similarity,
            pair_similarity_combine=args.pair_similarity_combine,
        )
        _write_parquet(args.output, rows)
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    except (ImportError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"task_mining_router: {exc}", file=sys.stderr)
        return 2
    print(
        f"task_mining_router: mode={args.mode} pair_similarity={args.pair_similarity} "
        f"targets={summary['target_queries']} "
        f"sources={summary['unique_sources']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
