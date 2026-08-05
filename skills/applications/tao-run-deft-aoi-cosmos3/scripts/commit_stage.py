#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Atomically commit one Cosmos3 DEFT AOI stage and audit the result."""

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


STAGES = (
    "train",
    "evaluate_proxy",
    "proxy_rcca",
    "evaluate_benchmark",
    "benchmark_metrics",
    "routing",
    "anomalygen",
    "data_mining",
    "assemble_data",
    "validate_data",
    "loop_stop",
)
SKIPPABLE_STAGES = ("anomalygen",)


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
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
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _required_file(value: pathlib.Path | None, flag: str) -> str:
    if value is None:
        raise ValueError(f"{flag} is required")
    path = value.expanduser()
    if not path.is_absolute():
        raise ValueError(f"{flag} must be absolute: {value}")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"{flag} must be an existing non-empty file: {value}")
    return str(path.resolve())


def _required_json_file(value: pathlib.Path | None, flag: str) -> str:
    """Like _required_file, but say so plainly when the file is not JSON.

    Routing and mining both deal in parquet, so a parquet lands on a `*_json`
    flag easily. Without this the failure surfaces much later as a UTF-8 decode
    error on a parquet magic byte, which names neither the expected format nor
    the offending flag.
    """
    path = _required_file(value, flag)
    with open(path, "rb") as handle:
        head = handle.read(4)
    if head == b"PAR1":
        raise ValueError(
            f"{flag} expects a JSON array but got a parquet file: {path}. "
            "Routing produces both — the JSON for state, the parquet for the "
            "embedding container. Pass the JSON here."
        )
    try:
        payload = json.loads(pathlib.Path(path).read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{flag} must be a JSON array: {path} ({exc})") from exc
    if not isinstance(payload, list):
        raise ValueError(
            f"{flag} must be a JSON array, got {type(payload).__name__}: {path}"
        )
    return path


def _required_checkpoint(value: pathlib.Path | None, flag: str) -> str:
    if value is None:
        raise ValueError(f"{flag} is required")
    path = value.expanduser()
    if not path.is_absolute() or not path.exists():
        raise ValueError(f"{flag} must be an existing absolute path: {value}")
    if path.is_file() and path.stat().st_size == 0:
        raise ValueError(f"{flag} must not be empty: {value}")
    if path.is_dir() and not any(path.iterdir()):
        raise ValueError(f"{flag} directory must not be empty: {value}")
    return str(path.resolve())


def _within(path: str, root: pathlib.Path, flag: str) -> str:
    resolved = pathlib.Path(path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{flag} must be under {root}: {resolved}") from exc
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
        if key != ("baseline", "evaluate_benchmark"):
            raise ValueError(
                "the first stage must be baseline/evaluate_benchmark"
            )
        return
    allowed = _expected_next(entries[-1])
    if key not in allowed:
        rendered = ", ".join(
            f"{label}/{name}" for label, name in sorted(allowed)
        )
        previous = entries[-1]
        raise ValueError(
            f"illegal transition {previous.get('iter')}/{previous.get('stage')} "
            f"-> {iter_label}/{stage}; expected [{rendered or 'end-of-log'}]"
        )


def _apply_success(
    phase: dict[str, Any],
    args: argparse.Namespace,
    results_dir: pathlib.Path,
) -> None:
    stage = args.stage
    phase_root = results_dir / args.iter_label
    if stage == "train":
        phase["best_ckpt_path"] = _within(
            _required_checkpoint(args.best_ckpt, "--best-ckpt"),
            phase_root / "train",
            "--best-ckpt",
        )
        phase["training_spec"] = _required_file(
            args.training_spec, "--training-spec"
        )
    elif stage == "evaluate_proxy":
        phase["proxy_results_json"] = _within(
            _required_file(args.proxy_results, "--proxy-results"),
            phase_root,
            "--proxy-results",
        )
    elif stage == "proxy_rcca":
        phase["proxy_gaps_summary"] = _within(
            _required_file(args.proxy_gaps_summary, "--proxy-gaps-summary"),
            phase_root,
            "--proxy-gaps-summary",
        )
        phase["false_accepts_json"] = _within(
            _required_file(args.false_accepts, "--false-accepts"),
            phase_root,
            "--false-accepts",
        )
        phase["false_rejects_json"] = _within(
            _required_file(args.false_rejects, "--false-rejects"),
            phase_root,
            "--false-rejects",
        )
    elif stage == "evaluate_benchmark":
        phase["benchmark_results_json"] = _within(
            _required_file(args.benchmark_results, "--benchmark-results"),
            phase_root,
            "--benchmark-results",
        )
    elif stage == "benchmark_metrics":
        phase["benchmark_metrics_summary"] = _within(
            _required_file(
                args.benchmark_metrics_summary, "--benchmark-metrics-summary"
            ),
            phase_root,
            "--benchmark-metrics-summary",
        )
    elif stage == "routing":
        phase["mining_targets_json"] = _within(
            _required_json_file(args.mining_targets, "--mining-targets"),
            phase_root,
            "--mining-targets",
        )
    elif stage == "anomalygen":
        if args.skip:
            # Documented branch skip: the driving Proxy RCCA found no false
            # accepts, so there is no under-detection gap for synthetic
            # defects to close. The audit re-proves this against disk.
            phase["anomalygen_skipped"] = True
        else:
            phase["anomalygen_sdg_csv"] = _within(
                _required_file(args.anomalygen_sdg, "--anomalygen-sdg"),
                phase_root,
                "--anomalygen-sdg",
            )
            phase["anomalygen_sharegpt_json"] = _within(
                _required_json_file(
                    args.anomalygen_sharegpt, "--anomalygen-sharegpt"
                ),
                phase_root,
                "--anomalygen-sharegpt",
            )
    elif stage == "data_mining":
        artifacts = {
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
        }
        for field, (value, flag) in artifacts.items():
            phase[field] = _within(
                _required_file(value, flag), phase_root, flag
            )
        if args.mining_count is None or args.mining_count <= 0:
            raise ValueError("--mining-count must be > 0")
        phase["mining_mined_count"] = args.mining_count
    elif stage == "assemble_data":
        phase["mined_sharegpt_json"] = _within(
            _required_json_file(args.mined_sharegpt, "--mined-sharegpt"),
            phase_root,
            "--mined-sharegpt",
        )
        phase["combined_training_json"] = _within(
            _required_json_file(args.combined_training, "--combined-training"),
            phase_root,
            "--combined-training",
        )
        phase["assemble_summary"] = _within(
            _required_file(args.assemble_summary, "--assemble-summary"),
            phase_root,
            "--assemble-summary",
        )
    elif stage == "validate_data":
        phase["validation_report"] = _within(
            _required_file(args.validation_report, "--validation-report"),
            phase_root,
            "--validation-report",
        )
    elif stage != "loop_stop":
        raise ValueError(f"unsupported stage: {stage}")

    if stage != "loop_stop":
        phase["stage_completed"] = stage
        # benchmark_metrics completes a stopping iteration (record_metric_result
        # sets status=complete). A continuing iteration then runs Proxy to seed
        # the next round and completes at proxy_rcca, so neither stage may
        # demote the phase back to in_progress.
        if stage not in ("benchmark_metrics", "proxy_rcca"):
            phase["status"] = "in_progress"


def commit(args: argparse.Namespace) -> dict[str, Any]:
    if not re.fullmatch(r"baseline|iter[1-9][0-9]*", args.iter_label):
        raise ValueError("--iter-label must be baseline or iterN")
    if getattr(args, "skip", False) and args.stage not in SKIPPABLE_STAGES:
        raise ValueError(
            f"--skip is valid only for: {', '.join(SKIPPABLE_STAGES)}"
        )
    results_dir = args.results_dir.expanduser().resolve()
    state_path = results_dir / "deft_state.json"
    log_path = results_dir / "loop_log.jsonl"
    if not state_path.is_file():
        raise ValueError(f"state file not found: {state_path}")
    original_state = state_path.read_text()
    original_log = log_path.read_text() if log_path.exists() else None
    state = json.loads(original_state)
    if pathlib.Path(str(state.get("results_dir", ""))).resolve() != results_dir:
        raise ValueError("state.results_dir does not match --results-dir")
    entries = _load_log(log_path)
    _validate_transition(entries, args.iter_label, args.stage)

    try:
        iterations = state.get("iterations")
        if not isinstance(iterations, dict):
            raise ValueError("state.iterations must be an object")
        if args.stage != "loop_stop":
            phase = iterations.setdefault(
                args.iter_label, {"status": "in_progress"}
            )
            if not isinstance(phase, dict):
                raise ValueError(
                    f"state.iterations.{args.iter_label} must be an object"
                )
            if args.status == "error":
                phase["status"] = "failed"
            elif args.stage == "benchmark_metrics":
                commit_metric_result(
                    argparse.Namespace(
                        state_path=state_path,
                        iter_label=args.iter_label,
                        result_json=args.metric_result,
                        best_ckpt=(
                            pathlib.Path(phase["best_ckpt_path"])
                            if phase.get("best_ckpt_path")
                            else None
                        ),
                        benchmark_results=pathlib.Path(
                            phase["benchmark_results_json"]
                        ),
                        training_spec=(
                            pathlib.Path(phase["training_spec"])
                            if phase.get("training_spec")
                            else None
                        ),
                    )
                )
                state = json.loads(state_path.read_text())
                iterations = state["iterations"]
                phase = iterations[args.iter_label]
                _apply_success(phase, args, results_dir)
            else:
                _apply_success(phase, args, results_dir)

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
            raise ValueError(
                "post-commit audit failed: " + "; ".join(report["errors"])
            )
    except Exception:
        _atomic_text(state_path, original_state)
        if original_log is None:
            try:
                log_path.unlink()
            except FileNotFoundError:
                pass
        else:
            _atomic_text(log_path, original_log)
        raise
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--iter-label", required=True)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--status", choices=("ok", "error"), default="ok")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--duration-sec", type=int, default=0)
    parser.add_argument("--best-ckpt", type=pathlib.Path)
    parser.add_argument("--training-spec", type=pathlib.Path)
    parser.add_argument("--proxy-results", type=pathlib.Path)
    parser.add_argument("--proxy-gaps-summary", type=pathlib.Path)
    parser.add_argument("--false-accepts", type=pathlib.Path)
    parser.add_argument("--false-rejects", type=pathlib.Path)
    parser.add_argument("--benchmark-results", type=pathlib.Path)
    parser.add_argument("--benchmark-metrics-summary", type=pathlib.Path)
    parser.add_argument("--metric-result", type=pathlib.Path)
    parser.add_argument("--mining-targets", type=pathlib.Path)
    parser.add_argument("--anomalygen-sdg", type=pathlib.Path)
    parser.add_argument("--anomalygen-sharegpt", type=pathlib.Path)
    parser.add_argument(
        "--skip",
        action="store_true",
        help=(
            "Record a documented branch skip instead of artifacts. Valid only "
            f"for: {', '.join(SKIPPABLE_STAGES)}."
        ),
    )
    parser.add_argument("--mining-parquet", type=pathlib.Path)
    parser.add_argument("--mining-summary", type=pathlib.Path)
    parser.add_argument("--mining-target-embeddings", type=pathlib.Path)
    parser.add_argument("--mining-source-embeddings", type=pathlib.Path)
    parser.add_argument("--mining-count", type=int)
    parser.add_argument("--mined-sharegpt", type=pathlib.Path)
    parser.add_argument("--combined-training", type=pathlib.Path)
    parser.add_argument("--assemble-summary", type=pathlib.Path)
    parser.add_argument("--validation-report", type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = commit(args)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"commit_stage: {exc}", file=sys.stderr)
        return 2
    last = report["last_committed"]
    print(
        f"committed seq={last['seq']} {last['iter']}/{last['stage']} "
        f"status={last['status']} run={report['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
