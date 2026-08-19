#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one gap-analysis profile against frozen candidate parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any

import yaml

from gap_analysis.config import PACKAGED_PROFILES, load_profile, validate_config
from gap_analysis.runner import run_selection


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_candidates(path: pathlib.Path) -> tuple[Any, list[dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("pyarrow is required to read gap candidate parquet") from exc
    table = pq.read_table(path)
    if table.num_rows == 0:
        raise ValueError("gap candidate parquet is empty")
    return table, table.to_pylist()


def write_selection(
    output_dir: pathlib.Path,
    selected: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("pyarrow is required to write selected gap parquet") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    table = (
        pa.Table.from_pylist(selected)
        if selected
        else pa.table({"id": pa.array([], type=pa.string())})
    )
    pq.write_table(table, output_dir / "selected_gaps.parquet")
    (output_dir / "gap_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.profile:
        config = load_profile(args.profile)
    else:
        try:
            payload = yaml.safe_load(args.config.read_text())
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid custom gap config: {exc}") from exc
        config = validate_config(payload)
    if args.budget is not None:
        config["budget"] = args.budget
    if args.seed is not None:
        config["seed"] = args.seed
    return validate_config(config)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=pathlib.Path)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--profile", choices=PACKAGED_PROFILES)
    choice.add_argument("--config", type=pathlib.Path)
    parser.add_argument("--task-metrics", type=pathlib.Path)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = resolve_config(args)
        _, candidates = read_candidates(args.candidates)
        selected, summary = run_selection(candidates, config)
        summary["candidate_file_sha256"] = file_sha256(args.candidates)
        summary["task_metrics_sha256"] = (
            file_sha256(args.task_metrics) if args.task_metrics is not None else None
        )
        summary["profile"] = args.profile or "custom"
        write_selection(args.output_dir, selected, summary)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"run_gap_analysis: {exc}", file=sys.stderr)
        return 2
    print(
        f"run_gap_analysis: profile={summary['profile']} "
        f"selected={summary['realized_budget']}/{summary['requested_budget']} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
