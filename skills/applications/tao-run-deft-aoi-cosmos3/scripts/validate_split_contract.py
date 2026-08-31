#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Validate canonical split isolation and monotonic real-mining lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from itertools import combinations
from typing import Any

from nvpaw_annotations import TASK_SPECS
from validate_sharegpt import load_records, resolve_image, target_path, validate_records


ROLE_PATHS = {
    "proxy": ("annotations", "proxy_kpi.jsonl"),
    "benchmark": ("annotations", "benchmark.jsonl"),
    "mining": ("annotations", "mining.jsonl"),
}


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records(
    path: pathlib.Path,
    media_root: pathlib.Path,
    *,
    skip_unsupported_tasks: bool = False,
) -> tuple[list[dict[str, Any]], set[str], dict[str, int]]:
    all_rows = load_records(path)
    summary = validate_records(
        all_rows,
        media_root=media_root,
        require_files=False,
        skip_unsupported_tasks=skip_unsupported_tasks,
    )
    rows = (
        [row for row in all_rows if row.get("task_type") in TASK_SPECS]
        if skip_unsupported_tasks
        else all_rows
    )
    targets = {
        str(resolve_image(target_path(row, context=f"{path}:{index}"), media_root))
        for index, row in enumerate(rows)
    }
    return rows, targets, summary["unsupported_tasks"]


def validate(
    role_paths: dict[str, pathlib.Path],
    *,
    media_root: pathlib.Path,
    expected_benchmark_sha256: str | None = None,
) -> dict[str, Any]:
    missing = set(ROLE_PATHS) - set(role_paths)
    if missing:
        raise ValueError(f"missing required roles: {sorted(missing)}")
    rows: dict[str, list[dict[str, Any]]] = {}
    targets: dict[str, set[str]] = {}
    ignored_unsupported_tasks: dict[str, dict[str, int]] = {}
    for role, path in role_paths.items():
        rows[role], targets[role], ignored = _records(
            path,
            media_root,
            skip_unsupported_tasks=role == "mining",
        )
        if ignored:
            ignored_unsupported_tasks[role] = ignored

    overlaps: dict[str, int] = {}
    for left, right in combinations(("proxy", "benchmark", "mining"), 2):
        shared = targets[left] & targets[right]
        overlaps[f"{left}:{right}"] = len(shared)
        if shared:
            raise ValueError(f"target leakage between {left} and {right}: {sorted(shared)[:5]}")

    for role in ("previous_train", "train"):
        if role not in targets:
            continue
        for evaluation_role in ("proxy", "benchmark"):
            shared = targets[role] & targets[evaluation_role]
            overlaps[f"{role}:{evaluation_role}"] = len(shared)
            if shared:
                raise ValueError(
                    f"target leakage between {role} and {evaluation_role}: {sorted(shared)[:5]}"
                )
    if "train" in targets:
        eligible = targets["mining"] | targets.get("previous_train", set())
        outside = targets["train"] - eligible
        if outside:
            raise ValueError(
                "generated Train targets must come from Mining or the previous "
                f"iteration: {sorted(outside)[:5]}"
            )
        overlaps["train:mining"] = len(targets["train"] & targets["mining"])
        if not overlaps["train:mining"]:
            raise ValueError("current Train must contain at least one real Mining target")
        if "previous_train" in targets:
            previous_rows = {
                json.dumps(row, sort_keys=True, separators=(",", ":"))
                for row in rows["previous_train"]
            }
            train_rows = {
                json.dumps(row, sort_keys=True, separators=(",", ":"))
                for row in rows["train"]
            }
            missing_rows = previous_rows - train_rows
            if missing_rows:
                raise ValueError(
                    f"generated Train must retain previous iteration records ({len(missing_rows)} missing)"
                )

    benchmark_hash = _sha256(role_paths["benchmark"])
    if expected_benchmark_sha256 and benchmark_hash != expected_benchmark_sha256:
        raise ValueError(
            f"frozen Benchmark hash mismatch: expected {expected_benchmark_sha256}, got {benchmark_hash}"
        )
    return {
        "schema_version": 1,
        "format": "jsonl",
        "training_source": "mined_real_samples_only",
        "roles": {role: str(path) for role, path in role_paths.items()},
        "records": {role: len(value) for role, value in rows.items()},
        "unique_targets": {role: len(value) for role, value in targets.items()},
        "ignored_unsupported_tasks": ignored_unsupported_tasks,
        "target_overlap": overlaps,
        "benchmark_sha256": benchmark_hash,
        "benchmark_hash_verified": bool(expected_benchmark_sha256),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=pathlib.Path)
    parser.add_argument("--train", type=pathlib.Path)
    parser.add_argument("--previous-train", type=pathlib.Path)
    for role in ROLE_PATHS:
        parser.add_argument(f"--{role}", type=pathlib.Path)
    parser.add_argument("--benchmark-sha256")
    parser.add_argument("--summary", type=pathlib.Path)
    args = parser.parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    role_paths = {
        role: (getattr(args, role) or workspace.joinpath(*parts)).expanduser().resolve()
        for role, parts in ROLE_PATHS.items()
    }
    if args.previous_train:
        role_paths["previous_train"] = args.previous_train.expanduser().resolve()
    if args.train:
        role_paths["train"] = args.train.expanduser().resolve()
    try:
        summary = validate(
            role_paths,
            media_root=workspace,
            expected_benchmark_sha256=args.benchmark_sha256,
        )
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"validate_split_contract: {exc}", file=sys.stderr)
        return 2
    print("validate_split_contract: OK " + " ".join(
        f"{role}={count}" for role, count in summary["records"].items()
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
