#!/usr/bin/env python3
"""Promote every original-data AutoML brain winner to 2,000 epochs.

The three non-LLM BFBO promotions already owned by
``launch_full2000_winners.py`` are adopted into the combined manifest.  This
launcher submits the other nine brain winners and monitors all twelve without
overwriting the original six-finalist controller state.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from launch_full2000_winners import (
    IMAGE,
    LOCAL_ROOT,
    RECORD,
    REQUEUE_SAFE_CHECKPOINT_INTERVAL,
    build_full_spec,
    evaluation_metric,
    final_checkpoint,
    stage_spec,
    track_result,
)
from run_lam_track import REMOTE_ROOT, SOURCE_DATA


BACKBONES = ("fan_base", "fan_large", "mit_b5")
VARIANTS = ("bayesian", "bfbo", "bayesian_llm", "bfbo_llm")
EXISTING_MANIFEST = LOCAL_ROOT / "full2000_manifest.json"
EXISTING_STATUS = LOCAL_ROOT / "full2000_status.json"
MANIFEST = LOCAL_ROOT / "full2000_all_brains_manifest.json"
STATUS = LOCAL_ROOT / "full2000_all_brains_status.json"


def write_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def record_command(*args: str) -> str:
    return subprocess.check_output(
        [sys.executable, str(RECORD), *args], text=True
    ).strip()


def existing_bfbo_rows() -> dict[str, dict]:
    if not EXISTING_MANIFEST.is_file():
        raise RuntimeError(f"missing existing promotion manifest: {EXISTING_MANIFEST}")
    rows = json.loads(EXISTING_MANIFEST.read_text())
    found = {
        row["source_track"]: row
        for row in rows
        if row.get("kind") == "automl"
        and row.get("source_track", "").endswith("_bfbo")
    }
    expected = {f"automl_{backbone}_bfbo" for backbone in BACKBONES}
    if set(found) != expected:
        raise RuntimeError(
            f"existing BFBO promotions mismatch: expected={sorted(expected)} "
            f"found={sorted(found)}"
        )
    return found


def brain_rows() -> list[dict]:
    adopted = existing_bfbo_rows()
    rows: list[dict] = []
    for backbone in BACKBONES:
        for variant in VARIANTS:
            source_track = f"automl_{backbone}_{variant}"
            if variant == "bfbo":
                row = dict(adopted[source_track])
                row.update(
                    {
                        "brain": source_track,
                        "owned_by_this_launcher": False,
                    }
                )
            else:
                row = {
                    "name": f"full2000_{backbone}_{variant}",
                    "kind": "automl",
                    "backbone": backbone,
                    "dataset": "original",
                    "source_track": source_track,
                    "brain": source_track,
                    "selection_miou": evaluation_metric(source_track),
                    "hyperparameters": track_result(source_track, "automl")["best"]["specs"],
                    "owned_by_this_launcher": True,
                    "state": "PENDING",
                }
            rows.append(row)
    return rows


def preflight(rows: list[dict]) -> None:
    expected = {
        f"automl_{backbone}_{variant}"
        for backbone in BACKBONES
        for variant in VARIANTS
    }
    if len(rows) != 12 or {row["source_track"] for row in rows} != expected:
        raise RuntimeError("the combined promotion set is not exactly 12 AutoML brains")
    if sum(bool(row["owned_by_this_launcher"]) for row in rows) != 9:
        raise RuntimeError("the corrected launcher must own exactly nine missing promotions")
    for row in rows:
        if row["dataset"] != "original":
            raise RuntimeError(f"non-original dataset in AutoML promotion: {row}")
        spec = build_full_spec(row)
        segment = spec["dataset"]["segment"]
        train = spec["train"]
        checks = {
            "root_dir": segment["root_dir"] == str(SOURCE_DATA),
            "num_epochs": train["num_epochs"] == 2000,
            "checkpoint_interval": (
                train["checkpoint_interval"] == REQUEUE_SAFE_CHECKPOINT_INTERVAL
            ),
            "validation_interval": train["validation_interval"] == 10,
            "num_gpus": train["num_gpus"] == 8,
            "gpu_ids": train["gpu_ids"] == list(range(8)),
            "num_nodes": train["num_nodes"] == 1,
        }
        failed = [key for key, ok in checks.items() if not ok]
        if failed:
            raise RuntimeError(f"preflight failed for {row['source_track']}: {failed}")
        dotted = []
        stack = [spec]
        while stack:
            node = stack.pop()
            for key, value in node.items():
                if "." in key:
                    dotted.append(key)
                if isinstance(value, dict):
                    stack.append(value)
        if dotted:
            raise RuntimeError(f"dotted keys remain in {row['source_track']}: {dotted}")


def submit(sdk, row: dict, rows: list[dict]) -> None:
    spec = build_full_spec(row)
    remote_spec = stage_spec(row, spec)
    results_dir = spec["results_dir"]
    record_id = record_command(
        "open",
        "--platform",
        "slurm",
        "--image",
        IMAGE,
        "--network-arch",
        "segformer",
        "--action",
        "train",
        "--storage-tier",
        "A",
        "--results-dir",
        results_dir,
    )
    row.update(
        {
            "record_id": record_id,
            "spec": remote_spec,
            "results_dir": results_dir,
            "state": "PENDING",
        }
    )
    write_json(MANIFEST, rows)
    command = (
        "python3 "
        f"{REMOTE_ROOT}/controller/resume_training_entrypoint.py "
        f"--spec {remote_spec}"
    )
    # A local workspace path can never reach SLURM.
    expected_prefix = "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
    if not command.split()[1].startswith(expected_prefix):
        raise RuntimeError(f"invalid remote entrypoint in command: {command}")
    try:
        job = sdk.create_job(
            image=IMAGE,
            command=command,
            gpu_count=8,
            num_nodes=1,
            account=os.environ.get("SLURM_ACCOUNT") or None,
            env_vars={"PYTHONPATH": "/usr/local/lib/python3.12/dist-packages"},
        )
    except Exception:
        record_command(
            "mark",
            record_id,
            "--state",
            "ERROR",
            "--source",
            "backend-hook",
            "--err-class",
            "ERR_PROGRAM",
            "--message",
            f"2,000-epoch brain promotion submission failed for {row['brain']}",
        )
        row["state"] = "ERROR"
        write_json(STATUS, rows)
        raise
    row.update(
        {
            "job_id": job.id,
            "backend_ref": job.backend_job_id,
            "state": "RUNNING",
        }
    )
    record_command(
        "mark",
        record_id,
        "--state",
        "RUNNING",
        "--source",
        "backend-hook",
        "--backend-ref",
        job.backend_job_id,
        "--message",
        f"2,000-epoch original-data promotion submitted for {row['brain']}",
    )
    write_json(MANIFEST, rows)
    print(f"SUBMITTED {row['brain']} {job.id}", flush=True)


def sync_adopted(rows: list[dict]) -> None:
    if not EXISTING_STATUS.is_file():
        return
    existing = {
        row.get("source_track"): row
        for row in json.loads(EXISTING_STATUS.read_text())
        if row.get("kind") == "automl"
    }
    for row in rows:
        if row["owned_by_this_launcher"]:
            continue
        source = existing.get(row["source_track"])
        if not source:
            continue
        for key in ("state", "final_checkpoint", "error"):
            if key in source:
                row[key] = source[key]


def main() -> None:
    from tao_sdk.platforms.slurm import SlurmSDK

    os.environ["SLURM_TIME_HOURS"] = "4"
    os.environ["SLURM_TIMEOUT_HOURS"] = "3.8"
    os.environ["SLURM_USE_REQUEUE"] = "true"
    os.environ["TAO_SDK_STATE_DIR"] = str(LOCAL_ROOT / "sdk_state/full2000_all_brains")

    rows = brain_rows()
    preflight(rows)
    write_json(MANIFEST, rows)
    print("ALL_BRAINS_PREFLIGHT_OK total=12 adopted=3 submit=9", flush=True)
    sdk = SlurmSDK(poll_interval=30, epoch_milestone_interval=10)
    for row in rows:
        if row["owned_by_this_launcher"]:
            submit(sdk, row, rows)

    terminal = {"COMPLETE", "ERROR", "CANCELED"}
    while any(row.get("state") not in terminal for row in rows):
        sync_adopted(rows)
        for row in rows:
            if not row["owned_by_this_launcher"] or row.get("state") in terminal:
                continue
            observed = sdk.get_job_status(row["job_id"]).status.upper()
            state = "CANCELED" if observed == "CANCELLED" else observed
            row["state"] = state
            if state not in terminal:
                continue
            if state == "COMPLETE":
                checkpoint = final_checkpoint(row["results_dir"])
                if not checkpoint:
                    state = "ERROR"
                    row["state"] = state
                    row["error"] = "terminal job has no epoch-1999 checkpoint"
                else:
                    row["final_checkpoint"] = checkpoint
            mark = [
                "mark",
                row["record_id"],
                "--state",
                state,
                "--source",
                "backend-hook",
                "--message",
                f"2,000-epoch promotion terminal for {row['brain']}",
            ]
            if state == "ERROR":
                mark += ["--err-class", "ERR_PROGRAM"]
            record_command(*mark)
        write_json(STATUS, rows)
        counts = {state: sum(row.get("state") == state for row in rows) for state in terminal}
        running = len(rows) - sum(counts.values())
        print(
            f"ALL_BRAINS running={running} complete={counts['COMPLETE']} "
            f"error={counts['ERROR']} canceled={counts['CANCELED']}",
            flush=True,
        )
        if running:
            time.sleep(30)
    print("ALL_BRAINS_TERMINAL", flush=True)


if __name__ == "__main__":
    main()
