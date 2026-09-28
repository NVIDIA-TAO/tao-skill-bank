#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic per-query round-robin similarity selection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def _vectors(values: pd.Series) -> np.ndarray:
    rows = [np.asarray(value, dtype=np.float32).reshape(-1) for value in values]
    if not rows or len({row.size for row in rows}) != 1:
        raise ValueError("embeddings are empty or width-mismatched")
    matrix = np.stack(rows)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or np.any(norms <= 0):
        raise ValueError("embeddings contain non-finite or zero-norm rows")
    return matrix / norms


def round_robin_rank(candidates: pd.DataFrame, queries: pd.DataFrame, *,
                     excluded: set[str], minimum: float, quota: int,
                     overfetch: int, audit_top_k: int,
                     excluded_candidates: set[str] | None = None
                     ) -> list[dict[str, Any]]:
    """Mirror the bounded per-query ranking used by historical commit 7ebdfbb."""
    if quota <= 0 or queries.empty or candidates.empty:
        return []
    required = {"candidate_id", "source_filepath", "embedding"}
    if not required.issubset(candidates) or not {"query_id", "embedding"}.issubset(queries):
        raise ValueError("round-robin selection inputs lack identity or embedding columns")
    if candidates.candidate_id.astype(str).duplicated().any():
        raise ValueError("candidate_id values must be unique")
    if queries.query_id.astype(str).duplicated().any():
        raise ValueError("query_id values must be unique")
    eligible = candidates[
        ~candidates.source_filepath.astype(str).isin(excluded)
    ].reset_index(drop=True)
    if excluded_candidates:
        if "filepath" not in eligible:
            raise ValueError("candidate exclusions require candidate filepath values")
        eligible = eligible[
            ~eligible.filepath.astype(str).map(
                lambda value: str(Path(value).expanduser().resolve())
            ).isin(excluded_candidates)
        ].reset_index(drop=True)
    if eligible.empty:
        return []
    candidate_vectors = _vectors(eligible.embedding)
    query_vectors = _vectors(queries.embedding)
    if candidate_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError("candidate/query embedding widths differ")
    scores = query_vectors @ candidate_vectors.T
    per_query_limit = min(
        scores.shape[1],
        max(audit_top_k, math.ceil(quota / len(queries)) * overfetch + 10),
    )
    orders: list[list[tuple[int, float]]] = []
    for row_scores in scores:
        if per_query_limit < len(row_scores):
            positions = np.argpartition(-row_scores, per_query_limit - 1)[:per_query_limit]
            positions = positions[np.argsort(-row_scores[positions], kind="stable")]
        else:
            positions = np.argsort(-row_scores, kind="stable")
        ranked, seen = [], set()
        for position in positions:
            score = float(row_scores[position])
            parent = str(eligible.iloc[position].source_filepath)
            if score < minimum or parent in seen:
                continue
            seen.add(parent)
            ranked.append((int(position), score))
        orders.append(ranked)
    target = max(quota * overfetch, quota + 25)
    output, selected, depth = [], set(), 0
    while len(output) < target and any(depth < len(order) for order in orders):
        for query_position, order in enumerate(orders):
            if depth >= len(order):
                continue
            candidate_position, score = order[depth]
            row = eligible.iloc[candidate_position].to_dict()
            parent = str(row["source_filepath"])
            if parent in selected:
                continue
            selected.add(parent)
            row.update(
                similarity=score,
                query_id=str(queries.iloc[query_position].query_id),
            )
            output.append(row)
            if len(output) >= target:
                break
        depth += 1
    return output


def select(candidates: dict[str, pd.DataFrame], queries: dict[str, pd.DataFrame],
           policy: dict[str, Any], excluded: dict[str, set[str]],
           prior_real: int = 0, prior_clean: int = 0,
           excluded_candidates: dict[str, set[str]] | None = None,
           ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    retrieval, routing = policy["retrieval"], policy["routing"]
    output: dict[str, list[dict[str, Any]]] = {"real": [], "clean": []}
    audit: dict[str, Any] = {"branches": []}
    used_real = set(excluded["real"])

    def pocket(row: pd.Series) -> tuple[str, str, str]:
        values = tuple(
            str(row.get(key, "")).strip()
            for key in ("benchmark", "texture", "defect_type")
        )
        if any(not value for value in values):
            raise ValueError(
                f"round-robin real query {row.get('query_id')!r} lacks pocket metadata"
            )
        return values  # type: ignore[return-value]

    real = queries.get("real", pd.DataFrame())
    for reason in ("fn", "near_miss_fp"):
        branch = real[real.reason.astype(str).eq(reason)] if not real.empty else real
        groups = branch.groupby(branch.apply(pocket, axis=1), sort=False) if not branch.empty else []
        for pocket_key, rows in groups:
            if reason == "fn":
                factors = {int(value) for value in rows.get(
                    "real_factor",
                    pd.Series([routing["real_mine_factor_min"]] * len(rows)),
                ).tolist()}
                if len(factors) != 1:
                    raise ValueError("strict queries in one pocket must share real_factor")
                quota = factors.pop() * len(rows)
            else:
                quota = min(
                    int(routing["near_miss_real_factor"]) * len(rows),
                    int(routing["near_miss_real_cap_per_pocket"]),
                )
            ranked = round_robin_rank(
                candidates["real"], rows, excluded=used_real,
                minimum=float(retrieval["minimum_similarity"]), quota=quota,
                overfetch=int(retrieval["candidate_overfetch"]),
                audit_top_k=int(retrieval["audit_top_k_per_query"]),
                excluded_candidates=(excluded_candidates or {}).get("real", set()),
            )
            chosen = ranked[:quota]
            output["real"].extend(chosen)
            used_real.update(str(row["source_filepath"]) for row in chosen)
            audit["branches"].append({
                "role": "real", "reason": reason, "pocket": list(pocket_key),
                "queries": len(rows), "requested": quota, "ranked": len(ranked),
                "admitted": len(chosen), "shortfall": quota - len(chosen),
            })
    clean = queries.get("clean", pd.DataFrame())
    if not clean.empty:
        clean_room = max(
            0,
            int((prior_real + len(output["real"]))
                * float(routing["clean_cumulative_cap_per_real"])) - prior_clean,
        )
        quota = min(int(routing["clean_factor"]) * len(clean), clean_room)
        ranked = round_robin_rank(
            candidates["clean"], clean, excluded=set(excluded["clean"]),
            minimum=float(retrieval["minimum_similarity"]), quota=quota,
            overfetch=int(retrieval["candidate_overfetch"]),
            audit_top_k=int(retrieval["audit_top_k_per_query"]),
            excluded_candidates=(excluded_candidates or {}).get("clean", set()),
        )
        output["clean"] = ranked[:quota]
        audit["branches"].append({
            "role": "clean", "reason": "background_fp", "pocket": None,
            "queries": len(clean), "requested": quota,
            "ranked": len(ranked), "admitted": len(output["clean"]),
            "shortfall": quota - len(output["clean"]),
        })
    return output, audit


def _previous(document: dict[str, Any], root: Path | None
              ) -> tuple[dict[str, set[str]], dict[str, int]]:
    excluded = {"real": set(), "clean": set()}
    counts = {"real": 0, "clean": 0}
    roles = {"real_defect": "real", "clean_negative": "clean"}
    for image in document.get("images", []):
        role = roles.get(str(image.get("deft_kind")))
        if not role:
            continue
        raw = image.get("source_path")
        path = Path(str(raw or image["file_name"]))
        if not path.is_absolute() and root:
            path = root / "images" / path.name
        excluded[role].add(str(path.resolve()))
        counts[role] += 1
    return excluded, counts


def _candidate_exclusions(role: str, retrieval_root: Path,
                          candidates: pd.DataFrame, manifest: dict[str, Any]) -> set[str]:
    path = retrieval_root / f"exclude_{role}_candidates.parquet"
    evidence = (manifest.get("role_status") or {}).get(role)
    if evidence:
        declared = evidence.get("exclusion_manifest")
        if declared and Path(str(declared)).resolve() != path.resolve():
            raise ValueError(f"{role} exclusion manifest path is not canonical")
        if not path.is_file():
            raise FileNotFoundError(f"{role} exclusion manifest is missing: {path}")
    elif not path.is_file():
        return set()
    frame = pd.read_parquet(path)
    if "filepath" not in frame:
        raise ValueError(f"{role} exclusion manifest lacks filepath")
    values = [str(Path(value).expanduser().resolve()) for value in frame.filepath.astype(str)]
    if len(values) != len(set(values)):
        raise ValueError(f"{role} exclusion manifest contains duplicate filepaths")
    if "filepath" not in candidates:
        raise ValueError(f"{role} candidates lack filepath")
    known = {
        str(Path(value).expanduser().resolve())
        for value in candidates.filepath.astype(str)
    }
    unmatched = set(values) - known
    if unmatched:
        raise ValueError(f"{role} exclusion manifest contains unknown candidates")
    if evidence and int(evidence.get("excluded_count", -1)) != len(values):
        raise ValueError(f"{role} exclusion count disagrees with its manifest")
    return set(values)


def materialize(policy_path: Path, candidate_root: Path, retrieval_root: Path,
                previous_path: Path | None = None) -> dict[str, Any]:
    """Write round-robin selections to the standard retrieval-stage artifacts."""
    policy = yaml.safe_load(policy_path.read_text())
    strategy = ((policy.get("retrieval") or {}).get("selection") or {}).get("strategy")
    if strategy != "round_robin_similarity":
        raise ValueError("round-robin materialization requires round_robin_similarity")
    manifest_path = retrieval_root / "query_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "COMPLETE":
        raise ValueError("query manifest is incomplete")
    enabled = list(manifest.get("enabled_roles") or [])
    if any(role not in {"real", "clean"} for role in enabled):
        raise ValueError("query manifest contains an unsupported role")
    previous = {} if previous_path is None else json.loads(previous_path.read_text())
    excluded, counts = _previous(previous, previous_path.parent if previous_path else None)
    candidates = {
        role: pd.read_parquet(candidate_root / f"{role}_candidate_embeddings.parquet")
        for role in enabled
    }
    queries = {
        role: pd.read_parquet(retrieval_root / f"{role}_query_embeddings.parquet")
        for role in enabled
    }
    candidate_exclusions = {
        role: _candidate_exclusions(role, retrieval_root, candidates[role], manifest)
        for role in enabled
    }
    selected, audit = select(
        candidates, queries, policy, excluded, counts["real"], counts["clean"],
        candidate_exclusions,
    )
    outputs = {
        role: retrieval_root / f"mine_{role}" / "final_unique_files.parquet"
        for role in enabled
    }
    report_path = retrieval_root / "round_robin_selection_report.json"
    existing = [path for path in (*outputs.values(), report_path) if path.exists()]
    if existing:
        raise FileExistsError(existing[0])
    for role, path in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(selected.get(role, [])).to_parquet(path, index=False)
    report = {
        "status": "COMPLETE", "iteration": int(manifest["iteration"]),
        "selection_strategy": strategy,
        "selected_counts": {role: len(selected.get(role, [])) for role in enabled},
        "outputs": {role: str(path) for role, path in outputs.items()},
        "audit": audit,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--retrieval-root", type=Path, required=True)
    parser.add_argument("--previous-coco", type=Path)
    args = parser.parse_args()
    result = materialize(
        args.policy.resolve(), args.candidate_root.resolve(), args.retrieval_root.resolve(),
        args.previous_coco.resolve() if args.previous_coco else None,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
