# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomically commit one DEFT stage to ``deft_state.json``.

Use this instead of inline Python, jq, or hand-authored JSON.  The state file
contains both the resume snapshot and the ordered stage events, so a run has a
single durable source of truth.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import re
import sys
import tempfile
from typing import Any

from record_metric_result import commit as commit_metric_result
from render_report import render as render_html_report


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_text(path: pathlib.Path, text: str) -> None:
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _required_file(path: pathlib.Path | None, name: str) -> str:
    if path is None:
        raise ValueError(f"{name} is required")
    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise ValueError(f"{name} must be absolute: {path}")
    if not expanded.is_file() or expanded.stat().st_size == 0:
        raise ValueError(f"{name} must be an existing non-empty file: {path}")
    return str(expanded.resolve())


def _required_allocation(
    path: pathlib.Path | None, name: str
) -> tuple[str, int]:
    resolved = _required_file(path, name)
    try:
        payload = json.loads(pathlib.Path(resolved).read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be a JSON object: {resolved} ({exc})") from exc
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{name} must be a non-empty defect-to-count JSON object")
    invalid = {
        str(defect): count
        for defect, count in payload.items()
        if not isinstance(count, int) or isinstance(count, bool) or count < 0
    }
    if invalid:
        raise ValueError(f"{name} contains invalid allocation counts: {invalid}")
    allocated = sum(payload.values())
    if allocated <= 0:
        raise ValueError(f"{name} must allocate at least one sample")
    return resolved, allocated


def _require_within(path: str, root: pathlib.Path, name: str) -> str:
    resolved = pathlib.Path(path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{name} must be under {root}: {resolved}") from exc
    return str(resolved)


def _append_event(
    state: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    events = state.setdefault("events", [])
    if not isinstance(events, list):
        raise ValueError("state.events must be an array")
    sequence = max(
        (
            event.get("seq", 0)
            for event in events
            if isinstance(event, dict)
            and isinstance(event.get("seq"), int)
            and not isinstance(event.get("seq"), bool)
        ),
        default=0,
    ) + 1
    event = {
        "seq": sequence,
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "iter": args.iter_label,
        "stage": args.stage,
        "status": args.status,
        "summary": args.summary,
        "duration_sec": args.duration_sec,
        "context_tokens": 0,
    }
    events.append(event)
    return event


def _apply_success(
    phase: dict[str, Any],
    stage: str,
    args: argparse.Namespace,
    results_dir: pathlib.Path,
    iter_label: str,
) -> None:
    if stage == "train":
        checkpoint = _required_file(args.best_ckpt, "--best-ckpt")
        expected_train_dir = results_dir / iter_label / "train"
        phase["best_ckpt_path"] = _require_within(
            checkpoint, expected_train_dir, "--best-ckpt"
        )
        phase["training_spec"] = _required_file(
            args.training_spec, "--training-spec"
        )
        if args.best_ckpt_kind is not None:
            phase["best_ckpt_kind"] = args.best_ckpt_kind
        if args.val_loss is not None:
            phase["val_loss"] = args.val_loss
    elif stage == "evaluate":
        required = ("best_ckpt_path", "inference_csv", "metric_result")
        missing = [field for field in required if not phase.get(field)]
        if missing or phase.get("status") != "complete":
            raise ValueError(
                "evaluate metric commit is incomplete; missing "
                f"{missing or ['status=complete']}"
            )
    elif stage == "rca":
        phase["rca_gaps_parquet"] = _required_file(args.rca_gaps, "--rca-gaps")
        if args.rca_threshold is not None:
            phase["rca_threshold"] = args.rca_threshold
        if args.rca_target_defect:
            phase["rca_target_defects"] = args.rca_target_defect
    elif stage == "routing":
        phase["routing_mining_parquet"] = _required_file(
            args.routing_mining, "--routing-mining"
        )
        phase["routing_anomalygen_parquet"] = _required_file(
            args.routing_anomalygen, "--routing-anomalygen"
        )
    elif stage == "anomalygen":
        if args.skip:
            phase["anomalygen_skipped"] = True
        else:
            phase["anomalygen_sdg_csv"] = _required_file(
                args.anomalygen_sdg, "--anomalygen-sdg"
            )
            allocation, allocated = _required_allocation(
                args.anomalygen_allocation, "--anomalygen-allocation"
            )
            phase["anomalygen_allocation_json"] = allocation
            phase["anomalygen_amp_allocated"] = allocated
    elif stage == "data_mining":
        if args.skip:
            phase["data_mining_skipped"] = True
        else:
            phase_root = results_dir / iter_label
            mining_artifacts = {
                "mining_mined_parquet": (
                    args.mining_parquet,
                    "--mining-parquet",
                ),
                "mining_candidate_parquet": (
                    args.mining_candidates,
                    "--mining-candidates",
                ),
                "mining_summary": (args.mining_summary, "--mining-summary"),
                "mining_history_summary": (
                    args.mining_history_summary,
                    "--mining-history-summary",
                ),
                "mining_target_embeddings": (
                    args.mining_target_embeddings,
                    "--mining-target-embeddings",
                ),
                "mining_source_embeddings": (
                    args.mining_source_embeddings,
                    "--mining-source-embeddings",
                ),
                "mining_target_log": (
                    args.mining_target_log,
                    "--mining-target-log",
                ),
                "mining_source_log": (
                    args.mining_source_log,
                    "--mining-source-log",
                ),
                "mining_knn_log": (args.mining_knn_log, "--mining-knn-log"),
            }
            for field, (path, flag) in mining_artifacts.items():
                phase[field] = _require_within(
                    _required_file(path, flag), phase_root, flag
                )
            phase["mining_history"] = _require_within(
                _required_file(args.mining_history, "--mining-history"),
                results_dir,
                "--mining-history",
            )
            if args.mining_count is None or args.mining_count < 0:
                raise ValueError("--mining-count is required and must be >= 0")
            phase["mining_mined_count"] = args.mining_count
    elif stage == "data_merge":
        phase["combined_training_csv"] = _required_file(
            args.combined_csv, "--combined-csv"
        )
        phase["provenance_csv"] = _required_file(
            args.provenance_csv, "--provenance-csv"
        )
        phase["merge_validation_report"] = _required_file(
            args.merge_validation_report, "--merge-validation-report"
        )
    elif stage not in {"anomalygen_finetune", "loop_stop"}:
        raise ValueError(f"unsupported stage: {stage}")

    if stage != "loop_stop":
        phase["stage_completed"] = stage
    if stage != "evaluate" and phase.get("status") != "complete":
        phase["status"] = "in_progress"


def commit(args: argparse.Namespace) -> dict[str, Any]:
    if not re.fullmatch(r"baseline|iter[1-9][0-9]*", args.iter_label):
        raise ValueError("--iter-label must be baseline or iterN (N >= 1)")
    if (
        not isinstance(args.duration_sec, int)
        or isinstance(args.duration_sec, bool)
        or args.duration_sec <= 0
    ):
        raise ValueError(
            "--duration-sec is required and must be a positive measured duration"
        )
    if args.skip and args.stage not in {"anomalygen", "data_mining"}:
        raise ValueError("--skip is valid only for anomalygen or data_mining")

    results_dir = args.results_dir.expanduser().resolve()
    state_path = results_dir / "deft_state.json"
    if not state_path.is_file():
        raise ValueError(f"state file not found: {state_path}")
    original_state_text = state_path.read_text()
    state = json.loads(original_state_text)
    if not isinstance(state, dict):
        raise ValueError("deft_state.json root must be an object")
    try:
        if args.stage == "evaluate" and args.status == "ok":
            commit_metric_result(
                argparse.Namespace(
                    state_path=state_path,
                    iter_label=args.iter_label,
                    result_json=args.metric_result,
                    best_ckpt=args.best_ckpt,
                    inference_csv=args.inference_csv,
                    training_spec=args.training_spec,
                    threshold=args.threshold,
                )
            )
            state = json.loads(state_path.read_text())
        state["version"] = 3

        iterations = state.get("iterations")
        if not isinstance(iterations, dict):
            raise ValueError("state.iterations must be an object")
        existing = iterations.setdefault(args.iter_label, {"status": "in_progress"})
        if not isinstance(existing, dict):
            raise ValueError(
                f"state.iterations.{args.iter_label} must be an object"
            )
        if args.status == "error":
            existing["status"] = "failed"
            state["status"] = "failed"
            state["completed_at"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(timespec="seconds")
        else:
            if args.stage != "loop_stop":
                _apply_success(
                    existing, args.stage, args, results_dir, args.iter_label
                )
                state["status"] = "in_progress"
                state.pop("completed_at", None)
            else:
                baseline = iterations.get("baseline")
                if not isinstance(baseline, dict) or baseline.get("status") != "complete":
                    raise ValueError(
                        "loop_stop requires iterations.baseline.status=complete"
                    )
                if existing.get("status") != "complete":
                    raise ValueError(
                        f"loop_stop requires iterations.{args.iter_label}.status=complete"
                    )
                state["status"] = "complete"
                state["completed_at"] = datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(timespec="seconds")

        match = re.fullmatch(r"iter([1-9][0-9]*)", args.iter_label)
        if match:
            state["current_iteration"] = max(
                int(match.group(1)), int(state.get("current_iteration", 0))
            )

        event = _append_event(state, args)
        _atomic_json(state_path, state)
    except Exception:
        _atomic_text(state_path, original_state_text)
        raise
    report = {
        "status": str(state.get("status", "in_progress")).upper(),
        "terminal": state.get("status") in {"complete", "failed"},
        "last_committed": event,
    }
    # Report rendering is a deterministic post-commit hook.  A presentation
    # failure must be visible, but it must not roll back a valid GPU-stage
    # commit and leave callers unable to advance the state machine.
    try:
        output = render_html_report(results_dir)
        report["report_path"] = str(output)
    except Exception as exc:  # noqa: BLE001 - hook failures are non-transactional
        report["report_render_error"] = str(exc)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--iter-label", required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "train",
            "evaluate",
            "rca",
            "routing",
            "anomalygen_finetune",
            "anomalygen",
            "data_mining",
            "data_merge",
            "loop_stop",
        ),
    )
    parser.add_argument("--status", choices=("ok", "error"), default="ok")
    parser.add_argument("--summary", required=True)
    parser.add_argument(
        "--duration-sec",
        required=True,
        type=int,
        help="Measured wall-clock seconds for this stage; must be positive",
    )
    parser.add_argument("--skip", action="store_true")
    parser.add_argument("--best-ckpt", type=pathlib.Path)
    parser.add_argument("--best-ckpt-kind", choices=("best_val", "latest"))
    parser.add_argument("--training-spec", type=pathlib.Path)
    parser.add_argument("--val-loss", type=float)
    parser.add_argument("--metric-result", type=pathlib.Path)
    parser.add_argument("--inference-csv", type=pathlib.Path)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--rca-gaps", type=pathlib.Path)
    parser.add_argument("--rca-threshold", type=float)
    parser.add_argument("--rca-target-defect", action="append", default=[])
    parser.add_argument("--routing-mining", type=pathlib.Path)
    parser.add_argument("--routing-anomalygen", type=pathlib.Path)
    parser.add_argument("--anomalygen-sdg", type=pathlib.Path)
    parser.add_argument("--anomalygen-allocation", type=pathlib.Path)
    parser.add_argument("--mining-parquet", type=pathlib.Path)
    parser.add_argument("--mining-candidates", type=pathlib.Path)
    parser.add_argument("--mining-summary", type=pathlib.Path)
    parser.add_argument("--mining-history", type=pathlib.Path)
    parser.add_argument("--mining-history-summary", type=pathlib.Path)
    parser.add_argument("--mining-target-embeddings", type=pathlib.Path)
    parser.add_argument("--mining-source-embeddings", type=pathlib.Path)
    parser.add_argument("--mining-target-log", type=pathlib.Path)
    parser.add_argument("--mining-source-log", type=pathlib.Path)
    parser.add_argument("--mining-knn-log", type=pathlib.Path)
    parser.add_argument("--mining-count", type=int)
    parser.add_argument("--combined-csv", type=pathlib.Path)
    parser.add_argument("--provenance-csv", type=pathlib.Path)
    parser.add_argument("--merge-validation-report", type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = commit(args)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"commit_stage: {exc}", file=sys.stderr)
        return 2
    last = report["last_committed"]
    print(
        f"committed seq={last['seq']} {last['iter']}/{last['stage']} "
        f"status={last['status']} run={report['status']}"
    )
    if report.get("report_render_error"):
        print(
            f"commit_stage: report hook failed: {report['report_render_error']}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
