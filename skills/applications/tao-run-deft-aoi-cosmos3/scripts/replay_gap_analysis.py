#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay several gap profiles against one frozen candidate parquet."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from gap_analysis.config import PACKAGED_PROFILES, load_profile, validate_config
from gap_analysis.runner import run_selection
from run_gap_analysis import file_sha256, read_candidates, write_selection


def _csv_values(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("at least one comma-separated value is required")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=pathlib.Path)
    parser.add_argument("--profiles", required=True, type=_csv_values)
    parser.add_argument("--seeds", default=["17"], type=_csv_values)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        invalid = sorted(set(args.profiles) - set(PACKAGED_PROFILES))
        if invalid:
            raise ValueError(f"unknown replay profiles: {invalid}")
        try:
            seeds = [int(value) for value in args.seeds]
        except ValueError as exc:
            raise ValueError("--seeds must contain comma-separated integers") from exc
        _, candidates = read_candidates(args.candidates)
        runs: list[dict] = []
        selected_sets: dict[str, set[str]] = {}
        for profile in args.profiles:
            for seed in seeds:
                config = load_profile(profile)
                config["seed"] = seed
                if args.budget is not None:
                    config["budget"] = args.budget
                config = validate_config(config)
                selected, summary = run_selection(candidates, config)
                label = f"{profile}-seed{seed}"
                summary["profile"] = profile
                summary["candidate_file_sha256"] = file_sha256(args.candidates)
                output = args.output_dir / label
                write_selection(output, selected, summary)
                ids = [str(row["id"]) for row in selected]
                selected_sets[label] = set(ids)
                runs.append(
                    {
                        "label": label,
                        "profile": profile,
                        "seed": seed,
                        "selected_ids": ids,
                        "selected_ids_sha256": summary["selected_ids_sha256"],
                        "realized_budget": summary["realized_budget"],
                        "per_group_selected": summary["per_group_selected"],
                        "per_group_mean_weakness": summary["per_group_mean_weakness"],
                        "per_dataset_selected": summary["per_dataset_selected"],
                        "selected_unique_targets": summary["selected_unique_targets"],
                        "duplicate_target_rate": summary["duplicate_target_rate"],
                    }
                )
        pairwise: list[dict] = []
        labels = sorted(selected_sets)
        for left_index, left in enumerate(labels):
            for right in labels[left_index + 1 :]:
                union = selected_sets[left] | selected_sets[right]
                intersection = selected_sets[left] & selected_sets[right]
                pairwise.append(
                    {
                        "left": left,
                        "right": right,
                        "intersection": len(intersection),
                        "union": len(union),
                        "jaccard": len(intersection) / len(union) if union else 1.0,
                    }
                )
        report = {
            "schema_version": "gap_analysis_replay_v1",
            "candidate_file": str(args.candidates.resolve()),
            "candidate_file_sha256": file_sha256(args.candidates),
            "runs": runs,
            "pairwise_overlap": pairwise,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "replay_summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"replay_gap_analysis: {exc}", file=sys.stderr)
        return 2
    print(
        f"replay_gap_analysis: runs={len(runs)} comparisons={len(pairwise)} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
