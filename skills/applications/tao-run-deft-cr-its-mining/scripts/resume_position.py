#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report the next unfinished stage for a DEFT CR ITS mining run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from log_stage import read_valid_events, rebuild_state
from workflow_common import absolute_path


INITIAL_STAGES = (
    "baseline_evaluate",
    "setup_embeddings",
    "cosmos_embed",
    "convert_embeddings",
)
ITERATION_STAGES = (
    "gap_analysis",
    "build_mining_target",
    "mine_nearest_neighbors",
    "record_mined_paths",
    "build_llava_from_mining",
    "assemble_train_annotations",
    "train",
    "evaluate",
)


def latest_events(events: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Return the highest-sequence event for each iteration/stage pair."""
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for event in sorted(events, key=lambda item: item["seq"]):
        iter_label = event.get("iter")
        stage = event.get("stage")
        if isinstance(iter_label, str) and isinstance(stage, str):
            latest[(iter_label, stage)] = event
    return latest


def stage_is_complete(event: dict[str, Any] | None) -> bool:
    """Return whether a latest stage event represents completed work."""
    return event is not None and event.get("status") in {"ok", "skipped"}


def read_gap_status(run_dir: Path, iteration: int) -> dict[str, Any] | None:
    """Read one iteration's gap status, returning None when it was not written."""
    status_path = run_dir / f"iter_{iteration}" / "gaps" / "gap_status.json"
    if not status_path.is_file():
        return None
    with status_path.open("r", encoding="utf-8") as handle:
        status = json.load(handle)
    if not isinstance(status, dict) or not isinstance(status.get("has_gaps"), bool):
        raise ValueError(f"{status_path}: expected a JSON object with boolean 'has_gaps'")
    return status


def resume_position(
    state: dict[str, Any],
    events: list[dict[str, Any]],
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Derive the next workflow stage from static state and valid log events."""
    latest = latest_events(events)
    loop_stop_candidates = [event for (_, stage), event in latest.items() if stage == "loop_stop"]
    loop_stop = max(loop_stop_candidates, key=lambda event: event["seq"], default=None)
    if stage_is_complete(loop_stop):
        return {
            "workflow_status": "complete",
            "next_iteration": None,
            "next_stage": None,
            "reason": "loop_stop already logged",
        }

    for stage in INITIAL_STAGES:
        candidates = [event for (_, event_stage), event in latest.items() if event_stage == stage]
        event = max(candidates, key=lambda item: item["seq"], default=None)
        if not stage_is_complete(event):
            return {
                "workflow_status": "failed" if event and event.get("status") == "error" else "running",
                "next_iteration": 0,
                "next_stage": stage,
                "reason": "retry failed stage" if event else "stage has no completion event",
            }

    max_iterations = state.get("max_iterations")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("deft_state.json is missing a positive max_iterations")
    mine_unique_only = bool(state.get("mine_unique_only", True))
    iteration_stages = tuple(
        stage for stage in ITERATION_STAGES if mine_unique_only or stage != "record_mined_paths"
    )
    for iteration in range(1, max_iterations + 1):
        iter_label = f"iter_{iteration}"
        gap_event = latest.get((iter_label, "gap_analysis"))
        if stage_is_complete(gap_event) and run_dir is not None:
            gap_status = read_gap_status(run_dir, iteration)
            if gap_status is None:
                return {
                    "workflow_status": "running",
                    "next_iteration": iteration,
                    "next_stage": "gap_analysis",
                    "reason": "gap analysis has no gap_status.json completion artifact",
                }
            if not gap_status["has_gaps"]:
                return {
                    "workflow_status": "running",
                    "next_iteration": None,
                    "next_stage": "loop_stop",
                    "reason": f"iteration {iteration} has no weak samples",
                }
        for stage in iteration_stages:
            event = latest.get((iter_label, stage))
            if not stage_is_complete(event):
                return {
                    "workflow_status": "failed" if event and event.get("status") == "error" else "running",
                    "next_iteration": iteration,
                    "next_stage": stage,
                    "reason": "retry failed stage" if event else "stage has no completion event",
                }
    return {
        "workflow_status": "running",
        "next_iteration": None,
        "next_stage": "loop_stop",
        "reason": "all configured iterations are complete",
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    """Refresh state and print the next-stage record as JSON."""
    args = parse_args()
    run_dir = absolute_path(args.run_dir)
    state_path = run_dir / "deft_state.json"
    log_path = run_dir / "loop_log.jsonl"
    if not state_path.is_file():
        raise FileNotFoundError(f"state file does not exist: {state_path}")
    if log_path.is_file():
        state = rebuild_state(state_path, log_path)
        events = read_valid_events(log_path)
    else:
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        events = []
    print(json.dumps(resume_position(state, events, run_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
