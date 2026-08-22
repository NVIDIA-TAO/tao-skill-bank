#!/usr/bin/env python3
"""Launch train-only out-of-fold SegFormer jobs for DEFT error mining."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from launch_final_evaluations import ssh, ssh_host
from run_lam_track import IMAGE, LOCAL_ROOT, REMOTE_ROOT, build_spec


RECORD = LOCAL_ROOT / "skill_bank_snapshot/scripts/tao_job_record.py"
DATA_ROOT = REMOTE_ROOT / "datasets/deft_oof_v1"
SPEC_ROOT = REMOTE_ROOT / "deft_oof_v1/specs"
RESULT_ROOT = REMOTE_ROOT / "deft_oof_v1"
LOCAL_SPEC_ROOT = LOCAL_ROOT / "deft_oof_specs"
MANIFEST = LOCAL_ROOT / "deft_oof_manifest_jobs.json"
STATUS = LOCAL_ROOT / "deft_oof_status.json"
BACKBONES = ("fan_base", "fan_large", "mit_b5")
FOLDS = tuple(range(4))


def write_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def record_command(*args: str) -> str:
    return subprocess.check_output(
        [sys.executable, str(RECORD), *args], text=True
    ).strip()


def rows() -> list[dict]:
    return [
        {
            "name": f"deft_oof_{backbone}_fold{fold}",
            "backbone": backbone,
            "fold": fold,
            "dataset_root": str(DATA_ROOT / f"fold{fold}"),
            "state": "PENDING",
            "stage": "train",
        }
        for backbone in BACKBONES
        for fold in FOLDS
    ]


def train_spec(row: dict) -> dict:
    spec = build_spec(row["backbone"], "original")
    output = RESULT_ROOT / "training" / row["name"]
    spec["model_name"] = row["name"]
    spec["results_dir"] = str(output)
    spec["dataset"]["segment"].update(
        {
            "root_dir": row["dataset_root"],
            "train_split": "train",
            "validation_split": "val",
            "test_split": "val",
            "predict_split": "test",
        }
    )
    spec["train"].update(
        {
            "num_gpus": 8,
            "gpu_ids": list(range(8)),
            "num_nodes": 1,
            "num_epochs": 20,
            "checkpoint_interval": 20,
            "validation_interval": 10,
            "resume_training_checkpoint_path": "",
            "use_distributed_sampler": True,
            "sync_batchnorm": True,
        }
    )
    spec["train"]["results_dir"] = ""
    return spec


def eval_spec(row: dict, checkpoint: str) -> dict:
    spec = train_spec(row)
    output = RESULT_ROOT / "evaluations" / row["name"]
    spec["results_dir"] = str(output)
    spec["evaluate"] = {
        "checkpoint": checkpoint,
        "trt_engine": "",
        "num_gpus": 8,
        "gpu_ids": list(range(8)),
        "num_nodes": 1,
        "batch_size": 1,
        "vis_after_n_batches": 1,
        "results_dir": str(output),
    }
    return spec


def assert_nested(spec: dict) -> None:
    stack = [spec]
    while stack:
        node = stack.pop()
        for key, value in node.items():
            if "." in key:
                raise RuntimeError(f"dotted spec key remains: {key}")
            if isinstance(value, dict):
                stack.append(value)


def preflight(campaign_rows: list[dict]) -> None:
    if len(campaign_rows) != 12:
        raise RuntimeError(f"expected 12 OOF jobs, found {len(campaign_rows)}")
    if len({row["name"] for row in campaign_rows}) != 12:
        raise RuntimeError("OOF job names are not unique")
    counts = ssh(
        "for fold in 0 1 2 3; do "
        f"root='{DATA_ROOT}'/fold${{fold}}; "
        "printf '%s|' \"$fold\"; "
        "find -L \"$root/images/train\" -maxdepth 1 -type f | wc -l | tr '\\n' '|'; "
        "find -L \"$root/masks/train\" -maxdepth 1 -type f | wc -l | tr '\\n' '|'; "
        "find -L \"$root/images/val\" -maxdepth 1 -type f | wc -l | tr '\\n' '|'; "
        "find -L \"$root/masks/val\" -maxdepth 1 -type f | wc -l; done"
    )
    observed = [line.split("|") for line in counts.splitlines()]
    if observed != [[str(i), "237", "237", "79", "79"] for i in FOLDS]:
        raise RuntimeError(f"OOF dataset counts failed: {observed}")
    for row in campaign_rows:
        spec = train_spec(row)
        assert_nested(spec)
        if not row["dataset_root"].startswith(str(DATA_ROOT)):
            raise RuntimeError(f"unexpected dataset root: {row['dataset_root']}")
        train = spec["train"]
        expected = {
            "num_gpus": 8,
            "gpu_ids": list(range(8)),
            "num_nodes": 1,
            "num_epochs": 20,
            "checkpoint_interval": 20,
            "validation_interval": 10,
        }
        for key, value in expected.items():
            if train[key] != value:
                raise RuntimeError(f"{row['name']} {key}: {train[key]} != {value}")


def stage_spec(row: dict, spec: dict, stage: str) -> str:
    local_dir = LOCAL_SPEC_ROOT / stage
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / f"{row['name']}.yaml"
    local_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    remote_dir = SPEC_ROOT / stage
    remote_path = remote_dir / local_path.name
    ssh("mkdir", "-p", str(remote_dir), spec["results_dir"])
    user = os.environ["SLURM_USER"]
    subprocess.run(
        [
            "scp",
            "-q",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            str(local_path),
            f"{user}@{ssh_host()}:{remote_path}",
        ],
        check=True,
    )
    return str(remote_path)


def open_record(row: dict, action: str, results_dir: str) -> str:
    return record_command(
        "open",
        "--platform",
        "slurm",
        "--image",
        IMAGE,
        "--network-arch",
        "segformer",
        "--action",
        action,
        "--storage-tier",
        "A",
        "--results-dir",
        results_dir,
    )


def submit(sdk, row: dict, action: str, spec: dict) -> None:
    remote_spec = stage_spec(row, spec, action)
    results_dir = spec["results_dir"]
    record_id = open_record(row, action, results_dir)
    row.update(
        {
            "stage": action,
            f"{action}_spec": remote_spec,
            f"{action}_results_dir": results_dir,
            f"{action}_record_id": record_id,
            "state": "PENDING",
        }
    )
    write_json(MANIFEST, campaign_rows_public(CAMPAIGN_ROWS))
    command = (
        f"python3 {REMOTE_ROOT}/controller/segformer_entrypoint.py "
        f"{action} -e {remote_spec}"
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
            f"DEFT OOF {action} submission failed for {row['name']}",
        )
        write_json(STATUS, campaign_rows_public(CAMPAIGN_ROWS))
        raise
    row.update(
        {
            f"{action}_job_id": job.id,
            f"{action}_backend_ref": job.backend_job_id,
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
        f"DEFT OOF {action} submitted for {row['name']}",
    )
    write_json(MANIFEST, campaign_rows_public(CAMPAIGN_ROWS))
    print(f"SUBMITTED {action} {row['name']} {job.id}", flush=True)


def campaign_rows_public(campaign_rows: list[dict]) -> list[dict]:
    return [dict(row) for row in campaign_rows]


def train_checkpoint(row: dict) -> str:
    raw = ssh(
        f"find '{row['train_results_dir']}/train' -maxdepth 1 -type f "
        "-name 'model_epoch_019_step_*.pth' -size +1M -printf '%p\\n' "
        "| sort -V | tail -1"
    )
    if not raw:
        raise RuntimeError(f"missing epoch-19 OOF checkpoint for {row['name']}")
    return raw


def prediction_count(row: dict) -> int:
    raw = ssh(
        f"find '{row['evaluate_results_dir']}' -maxdepth 1 -type f "
        "-name '*.png' | wc -l"
    )
    return int(raw)


def monitor_stage(sdk, campaign_rows: list[dict], action: str) -> None:
    terminal = {"COMPLETE", "ERROR", "CANCELED"}
    while any(row["state"] not in terminal for row in campaign_rows):
        for row in campaign_rows:
            if row["state"] in terminal:
                continue
            observed = sdk.get_job_status(row[f"{action}_job_id"]).status.upper()
            state = "CANCELED" if observed == "CANCELLED" else observed
            row["state"] = state
            if state not in terminal:
                continue
            if state == "COMPLETE" and action == "train":
                row["checkpoint"] = train_checkpoint(row)
            if state == "COMPLETE" and action == "evaluate":
                count = prediction_count(row)
                row["prediction_count"] = count
                if count != 79:
                    row["state"] = state = "ERROR"
                    row["error"] = f"expected 79 held-out predictions, found {count}"
            mark = [
                "mark",
                row[f"{action}_record_id"],
                "--state",
                state,
                "--source",
                "backend-hook",
                "--message",
                f"DEFT OOF {action} terminal for {row['name']}",
            ]
            if state == "ERROR":
                mark += ["--err-class", "ERR_PROGRAM"]
            record_command(*mark)
        write_json(STATUS, campaign_rows_public(campaign_rows))
        counts = {state: sum(row["state"] == state for row in campaign_rows) for state in terminal}
        active = len(campaign_rows) - sum(counts.values())
        print(
            f"DEFT_OOF_{action.upper()} running={active} complete={counts['COMPLETE']} "
            f"error={counts['ERROR']} canceled={counts['CANCELED']}",
            flush=True,
        )
        if active:
            time.sleep(30)
    failed = [row["name"] for row in campaign_rows if row["state"] != "COMPLETE"]
    if failed:
        raise RuntimeError(f"DEFT OOF {action} failed: {failed}")


def main() -> None:
    global CAMPAIGN_ROWS
    from tao_sdk.platforms.slurm import SlurmSDK

    os.environ["SLURM_TIME_HOURS"] = "4"
    os.environ["SLURM_TIMEOUT_HOURS"] = "3.8"
    os.environ["SLURM_USE_REQUEUE"] = "false"
    os.environ["TAO_SDK_STATE_DIR"] = str(LOCAL_ROOT / "sdk_state/deft_oof_v1")
    CAMPAIGN_ROWS = rows()
    preflight(CAMPAIGN_ROWS)
    write_json(MANIFEST, campaign_rows_public(CAMPAIGN_ROWS))
    print("DEFT_OOF_PREFLIGHT_OK jobs=12 folds=4 train=237 held_out=79", flush=True)
    sdk = SlurmSDK(poll_interval=30, epoch_milestone_interval=5)

    for row in CAMPAIGN_ROWS:
        submit(sdk, row, "train", train_spec(row))
    monitor_stage(sdk, CAMPAIGN_ROWS, "train")

    for row in CAMPAIGN_ROWS:
        row["state"] = "PENDING"
        submit(sdk, row, "evaluate", eval_spec(row, row["checkpoint"]))
    monitor_stage(sdk, CAMPAIGN_ROWS, "evaluate")
    print("DEFT_OOF_TERMINAL predictions=948", flush=True)


CAMPAIGN_ROWS: list[dict] = []


if __name__ == "__main__":
    main()
