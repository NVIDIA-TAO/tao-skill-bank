#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atomically commit one verified stage to the DEFT OD AOI state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


NEXT = {"synthesis_bootstrap": "candidate_cache", "candidate_cache": "baseline_measurement",
        "baseline_measurement": "baseline_gaps",
        "baseline_gaps": "iteration_retrieval", "iteration_retrieval": "iteration_admission",
        "iteration_admission": "iteration_training", "iteration_synthesis": "iteration_training",
        "iteration_training": "iteration_measurement",
        "iteration_measurement": "iteration_gaps"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(artifacts: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    if name not in artifacts:
        raise ValueError(f"stage commit requires artifact {name}")
    value = json.loads(Path(artifacts[name]["path"]).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"artifact {name} must be a JSON object")
    return value


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _nonzero_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError("query counts must be a JSON object")
    result = {}
    for key, count in value.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("query counts must be nonnegative integers")
        if count:
            result[str(key)] = count
    return result


def _validate_retrieval(iteration: int, artifacts: dict[str, dict[str, Any]]) -> None:
    manifest = _read_json(artifacts, "query_manifest")
    if manifest.get("status") != "COMPLETE" or int(manifest.get("iteration", -1)) != iteration:
        raise ValueError("query manifest is incomplete or for another iteration")
    counts = _nonzero_counts(manifest.get("query_counts"))
    if set(manifest.get("enabled_roles") or []) != set(counts):
        raise ValueError("enabled retrieval roles disagree with nonzero query counts")
    for role, count in counts.items():
        for suffix in ("queries", "query_embeddings", "mined"):
            name = f"{role}_{suffix}"
            if name not in artifacts:
                raise ValueError(f"iteration retrieval requires artifact {name}")
        if len(pd.read_parquet(artifacts[f"{role}_queries"]["path"])) != count:
            raise ValueError(f"{role} query count disagrees with its manifest")
        if len(pd.read_parquet(artifacts[f"{role}_query_embeddings"]["path"])) != count:
            raise ValueError(f"{role} embedding count disagrees with its manifest")
        if pd.read_parquet(artifacts[f"{role}_mined"]["path"]).empty:
            raise ValueError(f"{role} mining produced no selected candidates")


def _validate_admission(iteration: int, artifacts: dict[str, dict[str, Any]]) -> None:
    report = _read_json(artifacts, "admission_report")
    if report.get("status") != "COMPLETE" or int(report.get("iteration", -1)) != iteration:
        raise ValueError("admission report is incomplete or for another iteration")


def _validate_synthesis(iteration: int, artifacts: dict[str, dict[str, Any]]) -> None:
    generation = _read_json(artifacts, "generation_report")
    admission = _read_json(artifacts, "admission_report")
    if generation.get("status") != "COMPLETE":
        raise ValueError("generation report is incomplete")
    generated = blocked = requested = 0
    groups = generation.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("generation report has no dataset groups")
    for group in groups:
        group_requested = int(group.get("requested", -1))
        group_generated = int(group.get("generated", -1))
        group_blocked = int(group.get("guardrail_blocked", -1))
        if min(group_requested, group_generated, group_blocked) < 0:
            raise ValueError("generation report contains negative counts")
        if group_generated + group_blocked != group_requested:
            raise ValueError("generated and blocked counts do not reconcile")
        requested += group_requested
        generated += group_generated
        blocked += group_blocked
    if int(generation.get("generated", -1)) != generated or generated + blocked != requested:
        raise ValueError("generation totals disagree with dataset groups")
    _validate_admission(iteration, artifacts)
    admitted = int((admission.get("admitted") or {}).get("synthetic", -1))
    if admitted < 0 or admitted > generated:
        raise ValueError("synthetic admission count exceeds generated images")


def _validate_training(artifacts: dict[str, dict[str, Any]]) -> None:
    report = _read_json(artifacts, "checkpoint_selection")
    if report.get("status") != "COMPLETE" or report.get("action") != "select":
        raise ValueError("checkpoint selection is not final")
    try:
        epoch = int(report["best_epoch"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("checkpoint selection has no valid epoch") from error
    if epoch < 0 or _finite(report.get("best_kpi_mAP50"), "best_kpi_mAP50") < 0:
        raise ValueError("checkpoint selection has invalid KPI evidence")
    checkpoint = Path(str(report.get("selected_checkpoint") or "")).expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.name != f"model_epoch_{epoch:03d}.pth":
        raise ValueError("checkpoint selection does not bind the KPI-best checkpoint")
    statuses = report.get("status_files")
    if not isinstance(statuses, list) or not statuses:
        raise ValueError("checkpoint selection has no status evidence")
    if any(not Path(str(path)).is_file() for path in statuses):
        raise ValueError("checkpoint selection references missing status evidence")


def _committed_checkpoint(state: dict[str, Any], iteration: int) -> tuple[Path, str]:
    for event in reversed(state.get("events", [])):
        if event.get("stage") != "iteration_training" or event.get("iteration") != iteration:
            continue
        artifact = (event.get("artifacts") or {}).get("checkpoint_selection")
        if artifact and _sha(Path(artifact["path"])) == artifact["sha256"]:
            report = json.loads(Path(artifact["path"]).read_text())
            checkpoint = Path(str(report.get("selected_checkpoint") or "")).resolve()
            if checkpoint.is_file():
                return checkpoint, _sha(checkpoint)
        break
    raise ValueError("measurement has no valid committed checkpoint")


def _validate_measurement(state: dict[str, Any], iteration: int,
                          artifacts: dict[str, dict[str, Any]]) -> None:
    manifest = _read_json(artifacts, "measurement_manifest")
    if manifest.get("status") != "COMPLETE":
        raise ValueError("measurement manifest is incomplete")
    checkpoint, checkpoint_sha = _committed_checkpoint(state, iteration)
    if (Path(str(manifest.get("checkpoint") or "")).resolve() != checkpoint
            or manifest.get("checkpoint_sha256") != checkpoint_sha):
        raise ValueError("measurement does not bind the committed checkpoint")
    roles = manifest.get("inference_roles")
    if not isinstance(roles, dict) or set(roles) != {"kpi", "test"}:
        raise ValueError("measurement lacks KPI/test inference evidence")
    for role, evidence in roles.items():
        expected = int(evidence.get("expected_images", -1))
        predictions = Path(str(evidence.get("predictions") or ""))
        actual = len(list(predictions.glob("*.txt"))) if predictions.is_dir() else -1
        if expected < 1 or actual != expected:
            raise ValueError(f"{role} inference cardinality mismatch: expected {expected}, got {actual}")


def _gap_counts(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) - {"FP", "FN"}:
        raise ValueError(f"{label} must contain only FP/FN counts")
    result = {"FP": 0, "FN": 0}
    for key, count in value.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{label} must contain nonnegative integer counts")
        result[key] = count
    return result


def _validate_gaps(artifacts: dict[str, dict[str, Any]]) -> None:
    confidences = {}
    for kind in ("loose", "strict"):
        report = _read_json(artifacts, f"{kind}_gap_report")
        totals = _gap_counts(report.get("counts_by_type"), f"{kind}.counts_by_type")
        classes = report.get("counts_by_class")
        if report.get("kpi") != f"kpi_{kind}" or not isinstance(classes, dict):
            raise ValueError(f"{kind} gap report has invalid identity or class counts")
        aggregate = {"FP": 0, "FN": 0}
        for name, counts in classes.items():
            normalized = _gap_counts(counts, f"{kind}.counts_by_class.{name}")
            for gap_type in aggregate:
                aggregate[gap_type] += normalized[gap_type]
        if aggregate != totals:
            raise ValueError(f"{kind} gap report type and class counts disagree")
        boxes = f"{kind}_box_gaps"
        if boxes not in artifacts or len(pd.read_parquet(artifacts[boxes]["path"])) != sum(totals.values()):
            raise ValueError(f"{kind} gap parquet and report counts disagree")
        settings = report.get("settings") or {}
        confidences[kind] = _finite(settings.get("conf_threshold"), f"{kind}.conf_threshold")
        iou = _finite(settings.get("iou_threshold"), f"{kind}.iou_threshold")
        if not 0 <= confidences[kind] <= 1 or not 0 <= iou <= 1:
            raise ValueError(f"{kind} gap thresholds are out of range")
    if confidences["loose"] >= confidences["strict"]:
        raise ValueError("loose gap confidence must be lower than strict gap confidence")


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
    if stage == "iteration_retrieval":
        _validate_retrieval(iteration, artifacts)
    elif stage == "iteration_admission":
        _validate_admission(iteration, artifacts)
    elif stage == "iteration_synthesis":
        _validate_synthesis(iteration, artifacts)
    elif stage == "iteration_training":
        _validate_training(artifacts)
    elif stage == "iteration_measurement":
        _validate_measurement(state, iteration, artifacts)
    elif stage == "iteration_gaps":
        _validate_gaps(artifacts)
    next_stage, status = NEXT.get(stage), "RUNNING"
    if stage == "iteration_admission" and state.get("synthesis_enabled"):
        next_stage = "iteration_synthesis"
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
