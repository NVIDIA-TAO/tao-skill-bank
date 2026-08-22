#!/usr/bin/env python3
"""Launch a lower-LR, best-checkpoint-guarded FAN-base DEFT replicate."""

from __future__ import annotations

import json
import os
import time

from launch_deft_full2000 import (
    IMAGE,
    LOCAL_ROOT,
    REMOTE_ROOT,
    final_checkpoint,
    record_command,
    stage_spec,
    train_spec,
)
from launch_final_evaluations import ssh


MANIFEST = LOCAL_ROOT / "deft_fan_base_guard_manifest.json"
STATUS = LOCAL_ROOT / "deft_fan_base_guard_status.json"


def write_json(path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def row() -> dict:
    root = REMOTE_ROOT / "datasets/deft_model_v1_mix25/fan_base"
    return {
        "name": "deft_full2000_fan_base_mix25_guard_lr3e5",
        "backbone": "fan_base",
        "dataset": "model_driven_deft_mix25",
        "dataset_root": str(root),
        "selection_manifest": str(root / "manifest.json"),
        "lr_override": 3.0e-5,
        "state": "PENDING",
    }


def preflight(campaign: dict) -> None:
    spec = train_spec(campaign)
    checkpointer = spec["train"]["checkpointer"]
    if not (
        spec["train"]["num_epochs"] == 2000
        and spec["train"]["num_gpus"] == 8
        and spec["train"]["optim"]["lr"] == 3.0e-5
        and checkpointer["enable_topk"] is True
        and checkpointer["monitor"] == "val_miou"
        and checkpointer["mode"] == "max"
        and checkpointer["save_top_k"] == 3
    ):
        raise RuntimeError("invalid guarded FAN-base DEFT spec")
    raw = ssh(
        f"test -f '{campaign['selection_manifest']}' && "
        f"test \"$(jq -r '.validation_used_for_selection' '{campaign['selection_manifest']}')\" = false && "
        f"test \"$(jq -r '.method.resulting_train_count' '{campaign['selection_manifest']}')\" -eq 395 && "
        f"test ! -e '{spec['results_dir']}/train/status.json' && echo ok"
    )
    if raw != "ok":
        raise RuntimeError(f"guard preflight failed: {raw}")


def main() -> None:
    from tao_sdk.platforms.slurm import SlurmSDK

    os.environ["SLURM_TIME_HOURS"] = "4"
    os.environ["SLURM_TIMEOUT_HOURS"] = "3.8"
    os.environ["SLURM_USE_REQUEUE"] = "true"
    os.environ["TAO_SDK_STATE_DIR"] = str(LOCAL_ROOT / "sdk_state/deft_fan_base_guard_v1")
    campaign = row()
    preflight(campaign)
    spec = train_spec(campaign)
    remote_spec = stage_spec(campaign, spec)
    record_id = record_command(
        "open", "--platform", "slurm", "--image", IMAGE,
        "--network-arch", "segformer", "--action", "train",
        "--storage-tier", "A", "--results-dir", spec["results_dir"],
    )
    campaign.update(
        {"record_id": record_id, "spec": remote_spec, "results_dir": spec["results_dir"]}
    )
    write_json(MANIFEST, campaign)
    sdk = SlurmSDK(poll_interval=30, epoch_milestone_interval=10)
    try:
        job = sdk.create_job(
            image=IMAGE,
            command=f"python3 {REMOTE_ROOT}/controller/resume_training_entrypoint.py --spec {remote_spec}",
            gpu_count=8,
            num_nodes=1,
            account=os.environ.get("SLURM_ACCOUNT") or None,
            env_vars={"PYTHONPATH": "/usr/local/lib/python3.12/dist-packages"},
        )
    except Exception:
        record_command(
            "mark", record_id, "--state", "ERROR", "--source", "backend-hook",
            "--err-class", "ERR_PROGRAM", "--message", "guarded FAN-base DEFT submission failed",
        )
        raise
    campaign.update({"job_id": job.id, "backend_ref": job.backend_job_id, "state": "RUNNING"})
    write_json(MANIFEST, campaign)
    record_command(
        "mark", record_id, "--state", "RUNNING", "--source", "backend-hook",
        "--backend-ref", job.backend_job_id, "--message", "guarded FAN-base DEFT submitted",
    )
    print(f"DEFT_FAN_BASE_GUARD_SUBMITTED {job.id}", flush=True)
    terminal = {"COMPLETE", "ERROR", "CANCELED"}
    while campaign["state"] not in terminal:
        observed = sdk.get_job_status(job.id).status.upper()
        campaign["state"] = "CANCELED" if observed == "CANCELLED" else observed
        write_json(STATUS, campaign)
        print(f"DEFT_FAN_BASE_GUARD state={campaign['state']}", flush=True)
        if campaign["state"] not in terminal:
            time.sleep(30)
    if campaign["state"] == "COMPLETE":
        checkpoint = final_checkpoint(campaign["results_dir"])
        if checkpoint:
            campaign["final_checkpoint"] = checkpoint
        else:
            campaign["state"] = "ERROR"
            campaign["error"] = "terminal job has no epoch-1999 checkpoint"
    mark = [
        "mark", record_id, "--state", campaign["state"], "--source", "backend-hook",
        "--message", "guarded FAN-base DEFT terminal",
    ]
    if campaign["state"] == "ERROR":
        mark += ["--err-class", "ERR_PROGRAM"]
    record_command(*mark)
    write_json(STATUS, campaign)
    if campaign["state"] != "COMPLETE":
        raise RuntimeError(f"guarded FAN-base DEFT failed: {campaign}")


if __name__ == "__main__":
    main()
