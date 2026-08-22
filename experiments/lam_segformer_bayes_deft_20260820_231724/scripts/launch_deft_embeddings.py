#!/usr/bin/env python3
"""Launch eight-GPU SegFormer embedding extraction for DEFT mining."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from launch_final_evaluations import ssh
from run_lam_track import IMAGE, LOCAL_ROOT, REMOTE_ROOT


RECORD = LOCAL_ROOT / "skill_bank_snapshot/scripts/tao_job_record.py"
REMOTE_SCRIPT = REMOTE_ROOT / "controller/extract_segformer_embeddings.py"
DATA_ROOT = REMOTE_ROOT / "datasets/deft_embedding_v1"
OUTPUT_ROOT = REMOTE_ROOT / "deft_oof_v1/embeddings"
MANIFEST = LOCAL_ROOT / "deft_embeddings_manifest.json"
STATUS = LOCAL_ROOT / "deft_embeddings_status.json"

CONTROL_JOBS = {
    "fan_base": "11ba25bf-9b5f-4714-ae4e-d9f9742fc8df",
    "fan_large": "49051f66-e1fb-436c-89e4-3442a5790471",
    "mit_b5": "a7c29292-9d4d-4914-9268-ad111ba515c9",
}


def write_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def record_command(*args: str) -> str:
    return subprocess.check_output(
        [sys.executable, str(RECORD), *args], text=True
    ).strip()


def rows() -> list[dict]:
    result = []
    for backbone, job_id in CONTROL_JOBS.items():
        train_root = REMOTE_ROOT / f"results/{job_id}/results_dir/train"
        result.append(
            {
                "name": f"deft_embeddings_{backbone}",
                "backbone": backbone,
                "source_control_job": job_id,
                "spec": str(train_root / "experiment.yaml"),
                "checkpoint": str(train_root / "model_epoch_019_step_00800.pth"),
                "results_dir": str(OUTPUT_ROOT / backbone),
                "state": "PENDING",
            }
        )
    return result


def preflight(campaign_rows: list[dict]) -> None:
    if len(campaign_rows) != 3:
        raise RuntimeError("expected three backbone embedding jobs")
    commands = [f"test -f '{REMOTE_SCRIPT}'"]
    commands.append(
        f"test \"$(find -L '{DATA_ROOT}/images/embed' -maxdepth 1 -type f | wc -l)\" -eq 316"
    )
    for row in campaign_rows:
        commands.extend(
            [
                f"test -f '{row['spec']}'",
                f"test -f '{row['checkpoint']}'",
            ]
        )
    ssh(" && ".join(commands))


def submit(sdk, row: dict, campaign_rows: list[dict]) -> None:
    record_id = record_command(
        "open",
        "--platform",
        "slurm",
        "--image",
        IMAGE,
        "--network-arch",
        "segformer",
        "--action",
        "inference",
        "--storage-tier",
        "A",
        "--results-dir",
        row["results_dir"],
    )
    row["record_id"] = record_id
    write_json(MANIFEST, campaign_rows)
    command = (
        "torchrun --standalone --nproc_per_node=8 "
        f"{REMOTE_SCRIPT} --spec {row['spec']} --checkpoint {row['checkpoint']} "
        f"--dataset-root {DATA_ROOT} --split embed "
        f"--output-dir {row['results_dir']} --backbone {row['backbone']}"
    )
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
        row["state"] = "ERROR"
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
            f"DEFT embedding submission failed for {row['backbone']}",
        )
        write_json(STATUS, campaign_rows)
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
        f"DEFT embeddings submitted for {row['backbone']}",
    )
    write_json(MANIFEST, campaign_rows)
    print(f"SUBMITTED {row['name']} {job.id}", flush=True)


def metadata(row: dict) -> dict:
    raw = ssh(f"jq -c . '{row['results_dir']}/metadata.json'")
    result = json.loads(raw)
    if result.get("sample_count") != 316 or result.get("normalized") is not True:
        raise RuntimeError(f"invalid embedding metadata for {row['backbone']}: {result}")
    return result


def main() -> None:
    from tao_sdk.platforms.slurm import SlurmSDK

    os.environ["SLURM_TIME_HOURS"] = "4"
    os.environ["SLURM_TIMEOUT_HOURS"] = "3.8"
    os.environ["SLURM_USE_REQUEUE"] = "false"
    os.environ["TAO_SDK_STATE_DIR"] = str(LOCAL_ROOT / "sdk_state/deft_embeddings_v1")
    campaign_rows = rows()
    preflight(campaign_rows)
    write_json(MANIFEST, campaign_rows)
    print("DEFT_EMBEDDINGS_PREFLIGHT_OK jobs=3 samples=316 gpus=8", flush=True)
    sdk = SlurmSDK(poll_interval=30, epoch_milestone_interval=5)
    for row in campaign_rows:
        submit(sdk, row, campaign_rows)

    terminal = {"COMPLETE", "ERROR", "CANCELED"}
    while any(row["state"] not in terminal for row in campaign_rows):
        for row in campaign_rows:
            if row["state"] in terminal:
                continue
            observed = sdk.get_job_status(row["job_id"]).status.upper()
            state = "CANCELED" if observed == "CANCELLED" else observed
            row["state"] = state
            if state not in terminal:
                continue
            if state == "COMPLETE":
                try:
                    row["metadata"] = metadata(row)
                except Exception as exc:
                    row["state"] = state = "ERROR"
                    row["error"] = f"{type(exc).__name__}: {exc}"
            mark = [
                "mark",
                row["record_id"],
                "--state",
                state,
                "--source",
                "backend-hook",
                "--message",
                f"DEFT embeddings terminal for {row['backbone']}",
            ]
            if state == "ERROR":
                mark += ["--err-class", "ERR_PROGRAM"]
            record_command(*mark)
        write_json(STATUS, campaign_rows)
        counts = {state: sum(row["state"] == state for row in campaign_rows) for state in terminal}
        active = len(campaign_rows) - sum(counts.values())
        print(
            f"DEFT_EMBEDDINGS running={active} complete={counts['COMPLETE']} "
            f"error={counts['ERROR']} canceled={counts['CANCELED']}",
            flush=True,
        )
        if active:
            time.sleep(30)
    if any(row["state"] != "COMPLETE" for row in campaign_rows):
        raise RuntimeError("one or more DEFT embedding jobs failed")
    print("DEFT_EMBEDDINGS_TERMINAL", flush=True)


if __name__ == "__main__":
    main()
