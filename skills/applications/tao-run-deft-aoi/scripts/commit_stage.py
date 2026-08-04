# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomically commit one DEFT stage to state and loop_log.

Use this instead of inline Python, jq, or hand-authored JSON. For evaluate, this
command validates and records the metric result before adding the ordered log
event, then rolls both files back if the combined audit fails.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import tempfile
from typing import Any

from audit_deft_run import _expected_next, audit
from log_stage import append_stage
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


def _require_within(path: str, root: pathlib.Path, name: str) -> str:
    resolved = pathlib.Path(path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{name} must be under {root}: {resolved}") from exc
    return str(resolved)


def _load_log(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"loop_log line {line_number} is invalid JSON: {exc}"
            ) from exc
        if not isinstance(entry, dict):
            raise ValueError(f"loop_log line {line_number} must be an object")
        entries.append(entry)
    return entries


def _validate_transition(
    entries: list[dict[str, Any]], iter_label: str, stage: str
) -> None:
    key = (iter_label, stage)
    if any((entry.get("iter"), entry.get("stage")) == key for entry in entries):
        raise ValueError(f"stage already committed: {iter_label}/{stage}")
    if not entries:
        if key not in {("baseline", "train"), ("baseline", "evaluate")}:
            raise ValueError("first stage must be baseline/train or baseline/evaluate")
        return
    allowed = _expected_next(entries[-1])
    if key not in allowed:
        rendered = ", ".join(f"{label}/{name}" for label, name in sorted(allowed))
        previous = entries[-1]
        raise ValueError(
            f"illegal transition {previous.get('iter')}/{previous.get('stage')} -> "
            f"{iter_label}/{stage}; expected one of [{rendered or 'end-of-log'}]"
        )


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
                "mining_summary": (args.mining_summary, "--mining-summary"),
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
    if args.skip and args.stage not in {"anomalygen", "data_mining"}:
        raise ValueError("--skip is valid only for anomalygen or data_mining")

    results_dir = args.results_dir.expanduser().resolve()
    state_path = results_dir / "deft_state.json"
    log_path = results_dir / "loop_log.jsonl"
    if not state_path.is_file():
        raise ValueError(f"state file not found: {state_path}")
    original_state_text = state_path.read_text()
    original_log = log_path.read_text() if log_path.exists() else None
    state = json.loads(original_state_text)
    if not isinstance(state, dict):
        raise ValueError("deft_state.json root must be an object")
    if pathlib.Path(str(state.get("results_dir", ""))).resolve() != results_dir:
        raise ValueError("state.results_dir does not match --results-dir")

    entries = _load_log(log_path)
    _validate_transition(entries, args.iter_label, args.stage)
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

        iterations = state.get("iterations")
        if not isinstance(iterations, dict):
            raise ValueError("state.iterations must be an object")
        if args.stage != "loop_stop":
            existing = iterations.setdefault(
                args.iter_label, {"status": "in_progress"}
            )
            if not isinstance(existing, dict):
                raise ValueError(
                    f"state.iterations.{args.iter_label} must be an object"
                )
            if args.status == "error":
                existing["status"] = "failed"
            else:
                _apply_success(
                    existing, args.stage, args, results_dir, args.iter_label
                )

        match = re.fullmatch(r"iter([1-9][0-9]*)", args.iter_label)
        if match:
            state["current_iteration"] = max(
                int(match.group(1)), int(state.get("current_iteration", 0))
            )

        _atomic_json(state_path, state)
        append_stage(
            log_path,
            iter_label=args.iter_label,
            stage=args.stage,
            status=args.status,
            summary=args.summary,
            duration_sec=args.duration_sec,
        )
        report = audit(results_dir)
        if report["status"] == "INVALID":
            raise ValueError("post-commit audit failed: " + "; ".join(report["errors"]))
    except Exception:
        _atomic_text(state_path, original_state_text)
        if original_log is None:
            try:
                log_path.unlink()
            except FileNotFoundError:
                pass
        else:
            _atomic_text(log_path, original_log)
        raise
    # Report rendering is a deterministic post-commit hook.  A presentation
    # failure must be visible, but it must not roll back a valid GPU-stage
    # commit and leave callers unable to advance the state machine.
    try:
        output = render_html_report(results_dir, audit_report=report)
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
    parser.add_argument("--duration-sec", type=int, default=0)
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
    parser.add_argument("--mining-parquet", type=pathlib.Path)
    parser.add_argument("--mining-summary", type=pathlib.Path)
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
