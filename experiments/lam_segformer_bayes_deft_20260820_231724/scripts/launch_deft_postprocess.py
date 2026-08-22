#!/usr/bin/env python3
"""Run DEFT postprocessing on SLURM, then launch standalone DEFT training."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from launch_final_evaluations import ssh
from run_lam_track import IMAGE, LOCAL_ROOT, REMOTE_ROOT


RECORD = LOCAL_ROOT / "skill_bank_snapshot/scripts/tao_job_record.py"
STATUS = LOCAL_ROOT / "deft_postprocess_status.json"
RESULTS_DIR = REMOTE_ROOT / "deft_oof_v1/postprocess"
MARKER = REMOTE_ROOT / "deft_oof_v1/postprocess_complete.json"
REMOTE_SCRIPT = REMOTE_ROOT / "controller/run_deft_postprocess.py"


def write_status(payload: dict) -> None:
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(STATUS)


def record_command(*args: str) -> str:
    return subprocess.check_output([sys.executable, str(RECORD), *args], text=True).strip()


def validate_marker() -> dict:
    payload = json.loads(ssh(f"jq -c . '{MARKER}'"))
    if payload.get("validation_used_for_selection") is not False:
        raise RuntimeError("postprocess marker does not guarantee validation exclusion")
    snapshots = payload.get("snapshots", {})
    if set(snapshots) != {"fan_base", "fan_large", "mit_b5"}:
        raise RuntimeError(f"unexpected postprocess snapshots: {sorted(snapshots)}")
    if any(row.get("train_count") != 395 for row in snapshots.values()):
        raise RuntimeError(f"invalid snapshot counts: {snapshots}")
    return payload


def main() -> None:
    from tao_sdk.platforms.slurm import SlurmSDK

    os.environ["SLURM_TIME_HOURS"] = "0.5"
    os.environ["SLURM_TIMEOUT_HOURS"] = "0.45"
    os.environ["SLURM_USE_REQUEUE"] = "false"
    os.environ["TAO_SDK_STATE_DIR"] = str(LOCAL_ROOT / "sdk_state/deft_postprocess_v1")
    ssh(f"test -f '{REMOTE_SCRIPT}' && test ! -e '{MARKER}' && mkdir -p '{RESULTS_DIR}'")
    record_id = record_command(
        "open", "--platform", "slurm", "--image", IMAGE,
        "--network-arch", "segformer", "--action", "inference",
        "--storage-tier", "A", "--results-dir", str(RESULTS_DIR),
    )
    state = {
        "record_id": record_id,
        "state": "PENDING",
        "results_dir": str(RESULTS_DIR),
        "gpus": 8,
        "time_limit_minutes": 30,
    }
    write_status(state)
    sdk = SlurmSDK(poll_interval=30, epoch_milestone_interval=5)
    try:
        job = sdk.create_job(
            image=IMAGE,
            command=f"python3 {REMOTE_SCRIPT}",
            gpu_count=8,
            num_nodes=1,
            account=os.environ.get("SLURM_ACCOUNT") or None,
            env_vars={"PYTHONPATH": "/usr/local/lib/python3.12/dist-packages"},
        )
    except Exception:
        record_command(
            "mark", record_id, "--state", "ERROR", "--source", "backend-hook",
            "--err-class", "ERR_PROGRAM", "--message", "DEFT postprocess submission failed",
        )
        state["state"] = "ERROR"
        write_status(state)
        raise
    state.update({"job_id": job.id, "backend_ref": job.backend_job_id, "state": "RUNNING"})
    write_status(state)
    record_command(
        "mark", record_id, "--state", "RUNNING", "--source", "backend-hook",
        "--backend-ref", job.backend_job_id, "--message", "DEFT postprocess submitted",
    )
    print(f"DEFT_POSTPROCESS_SUBMITTED {job.id}", flush=True)
    terminal = {"COMPLETE", "ERROR", "CANCELED"}
    while state["state"] not in terminal:
        observed = sdk.get_job_status(job.id).status.upper()
        state["state"] = "CANCELED" if observed == "CANCELLED" else observed
        write_status(state)
        print(f"DEFT_POSTPROCESS state={state['state']}", flush=True)
        if state["state"] not in terminal:
            time.sleep(30)
    if state["state"] == "COMPLETE":
        try:
            state["marker"] = validate_marker()
        except Exception as exc:
            state["state"] = "ERROR"
            state["error"] = f"{type(exc).__name__}: {exc}"
    mark = [
        "mark", record_id, "--state", state["state"], "--source", "backend-hook",
        "--message", "DEFT postprocess terminal",
    ]
    if state["state"] == "ERROR":
        mark += ["--err-class", "ERR_PROGRAM"]
    record_command(*mark)
    write_status(state)
    if state["state"] != "COMPLETE":
        raise RuntimeError(f"DEFT postprocess failed: {state}")
    print("DEFT_POSTPROCESS_VALIDATED launching_full2000", flush=True)
    subprocess.run([sys.executable, str(LOCAL_ROOT / "launch_deft_full2000.py")], check=True)


if __name__ == "__main__":
    main()
