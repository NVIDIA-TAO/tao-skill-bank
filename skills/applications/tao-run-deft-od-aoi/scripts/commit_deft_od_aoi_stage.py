#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atomically commit one verified stage to the DEFT OD AOI state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NEXT = {"candidate_cache": "baseline_measurement", "baseline_measurement": "baseline_gaps",
        "baseline_gaps": "iteration_retrieval", "iteration_retrieval": "iteration_admission",
        "iteration_admission": "iteration_training", "iteration_training": "iteration_measurement",
        "iteration_measurement": "iteration_gaps"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def commit(state_path: Path, stage: str, iteration: int, values: list[str]) -> dict[str, Any]:
    state = json.loads(state_path.read_text())
    if state.get("status") not in {"READY", "RUNNING"} or state.get("next_stage") != stage:
        raise ValueError(f"expected stage {state.get('next_stage')}, not {stage}")
    expected = int(state["current_iteration"])
    if stage == "iteration_retrieval" and state.get("last_stage") in {"baseline_gaps", "iteration_gaps"}:
        expected += 1
    if iteration != expected:
        raise ValueError(f"expected iteration {expected}, not {iteration}")
    artifacts = {}
    for value in values:
        name, separator, raw = value.partition("=")
        path = Path(raw).expanduser().resolve()
        if not separator or not name or not path.is_file():
            raise ValueError(f"artifact must be name=existing-file: {value}")
        artifacts[name] = {"path": str(path), "sha256": _sha(path), "bytes": path.stat().st_size}
    if not artifacts:
        raise ValueError("at least one completion artifact is required")
    next_stage, status = NEXT.get(stage), "RUNNING"
    if stage == "iteration_gaps":
        if iteration >= int(state["max_iterations"]):
            next_stage, status = None, "COMPLETE"
        else:
            next_stage = "iteration_retrieval"
    state.update(status=status, current_iteration=iteration, last_stage=stage,
                 next_stage=next_stage)
    event = {"stage": stage, "iteration": iteration, "artifacts": artifacts,
             "committed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    state.setdefault("events", []).append(event)
    temporary = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, state_path)
    with (state_path.parent / "loop_log.jsonl").open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(NEXT) + ("iteration_gaps",), required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--artifact", action="append", default=[])
    args = parser.parse_args()
    result = commit(args.state.resolve(), args.stage, args.iteration, args.artifact)
    print(json.dumps({"status": result["status"], "next_stage": result["next_stage"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
