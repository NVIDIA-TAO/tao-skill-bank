#!/usr/bin/env python3
"""Launch one standalone 2,000-epoch model-driven DEFT run per backbone."""

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


BACKBONES = ("fan_base", "fan_large", "mit_b5")
DATA_ROOT = REMOTE_ROOT / "datasets/deft_model_v1_mix25"
RESULT_ROOT = REMOTE_ROOT / "deft_full2000"
REMOTE_SPEC_ROOT = REMOTE_ROOT / "deft_full2000_specs"
LOCAL_SPEC_ROOT = LOCAL_ROOT / "deft_full2000_specs"
MANIFEST = LOCAL_ROOT / "deft_full2000_manifest.json"
STATUS = LOCAL_ROOT / "deft_full2000_status.json"
RECORD = LOCAL_ROOT / "skill_bank_snapshot/scripts/tao_job_record.py"
REQUEUE_SAFE_CHECKPOINT_INTERVAL = 20


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def record_command(*args: str) -> str:
    return subprocess.check_output([sys.executable, str(RECORD), *args], text=True).strip()


def rows() -> list[dict]:
    return [
        {
            "name": f"deft_full2000_{backbone}_mix25",
            "backbone": backbone,
            "dataset": "model_driven_deft_mix25",
            "dataset_root": str(DATA_ROOT / backbone),
            "selection_manifest": str(DATA_ROOT / backbone / "manifest.json"),
            "state": "PENDING",
        }
        for backbone in BACKBONES
    ]


def train_spec(row: dict) -> dict:
    # build_spec supplies the fixed, non-AutoML control hyperparameters and the
    # requested PTM for each backbone. Only the train snapshot and horizon vary.
    spec = build_spec(row["backbone"], "original")
    output = RESULT_ROOT / row["name"]
    spec["model_name"] = f"lam_{row['name']}"
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
            "num_epochs": 2000,
            "checkpoint_interval": REQUEUE_SAFE_CHECKPOINT_INTERVAL,
            "validation_interval": 10,
            "resume_training_checkpoint_path": "",
            "use_distributed_sampler": True,
            "sync_batchnorm": True,
        }
    )
    spec["train"].setdefault("checkpointer", {}).update(
        {
            "enable_topk": True,
            "replace_periodic": False,
            "monitor": "val_miou",
            "mode": "max",
            "save_top_k": 3,
            "filename": "model_best_{epoch:03d}",
            "auto_insert_metric_name": False,
        }
    )
    if "lr_override" in row:
        spec["train"]["optim"]["lr"] = row["lr_override"]
    spec["train"]["results_dir"] = ""
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
    if len(campaign_rows) != 3 or {row["backbone"] for row in campaign_rows} != set(BACKBONES):
        raise RuntimeError("expected exactly one standalone DEFT run per backbone")
    for row in campaign_rows:
        spec = train_spec(row)
        assert_nested(spec)
        train = spec["train"]
        if not (
            train["num_epochs"] == 2000
            and train["checkpoint_interval"] == REQUEUE_SAFE_CHECKPOINT_INTERVAL
            and train["validation_interval"] == 10
            and train["num_gpus"] == 8
            and train["gpu_ids"] == list(range(8))
            and train["num_nodes"] == 1
        ):
            raise RuntimeError(f"invalid 2,000-epoch DEFT spec for {row['backbone']}")
        raw = ssh(
            f"test -f '{row['selection_manifest']}' && "
            f"test \"$(jq -r '.validation_used_for_selection' '{row['selection_manifest']}')\" = false && "
            f"test \"$(jq -r '.method.resulting_train_count' '{row['selection_manifest']}')\" -eq 395 && "
            f"test \"$(find -L '{row['dataset_root']}/images/train' -maxdepth 1 -type f | wc -l)\" -eq 395 && "
            f"test \"$(find -L '{row['dataset_root']}/masks/train' -maxdepth 1 -type f | wc -l)\" -eq 395 && "
            "echo ok"
        )
        if raw != "ok":
            raise RuntimeError(f"DEFT snapshot preflight failed for {row['backbone']}: {raw}")


def stage_spec(row: dict, spec: dict) -> str:
    LOCAL_SPEC_ROOT.mkdir(parents=True, exist_ok=True)
    local_path = LOCAL_SPEC_ROOT / f"{row['name']}.yaml"
    local_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    remote_path = REMOTE_SPEC_ROOT / local_path.name
    ssh("mkdir", "-p", str(REMOTE_SPEC_ROOT), spec["results_dir"])
    subprocess.run(
        [
            "scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            str(local_path), f"{os.environ['SLURM_USER']}@{ssh_host()}:{remote_path}",
        ],
        check=True,
    )
    return str(remote_path)


def submit(sdk, row: dict, campaign_rows: list[dict]) -> None:
    spec = train_spec(row)
    remote_spec = stage_spec(row, spec)
    record_id = record_command(
        "open", "--platform", "slurm", "--image", IMAGE,
        "--network-arch", "segformer", "--action", "train",
        "--storage-tier", "A", "--results-dir", spec["results_dir"],
    )
    row.update(
        {
            "record_id": record_id,
            "spec": remote_spec,
            "results_dir": spec["results_dir"],
            "state": "PENDING",
        }
    )
    write_json(MANIFEST, campaign_rows)
    command = (
        f"python3 {REMOTE_ROOT}/controller/resume_training_entrypoint.py "
        f"--spec {remote_spec}"
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
            "mark", record_id, "--state", "ERROR", "--source", "backend-hook",
            "--err-class", "ERR_PROGRAM", "--message",
            f"standalone DEFT submission failed for {row['backbone']}",
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
        "mark", record_id, "--state", "RUNNING", "--source", "backend-hook",
        "--backend-ref", job.backend_job_id, "--message",
        f"standalone model-driven DEFT submitted for {row['backbone']}",
    )
    write_json(MANIFEST, campaign_rows)
    print(f"SUBMITTED {row['name']} {job.id}", flush=True)


def final_checkpoint(results_dir: str) -> str:
    return ssh(
        f"find '{results_dir}/train' -maxdepth 1 -type f "
        "-name 'model_epoch_1999_step_*.pth' -size +1M -printf '%p\\n' "
        "| sort -V | tail -1"
    )


def main() -> None:
    from tao_sdk.platforms.slurm import SlurmSDK

    os.environ["SLURM_TIME_HOURS"] = "4"
    os.environ["SLURM_TIMEOUT_HOURS"] = "3.8"
    os.environ["SLURM_USE_REQUEUE"] = "true"
    os.environ["TAO_SDK_STATE_DIR"] = str(LOCAL_ROOT / "sdk_state/deft_full2000_v1")
    campaign_rows = rows()
    preflight(campaign_rows)
    write_json(MANIFEST, campaign_rows)
    print("DEFT_FULL2000_PREFLIGHT_OK jobs=3 epochs=2000 train=395 validation_selection=false", flush=True)
    sdk = SlurmSDK(poll_interval=30, epoch_milestone_interval=10)
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
                checkpoint = final_checkpoint(row["results_dir"])
                if checkpoint:
                    row["final_checkpoint"] = checkpoint
                else:
                    row["state"] = state = "ERROR"
                    row["error"] = "terminal job has no epoch-1999 checkpoint"
            mark = [
                "mark", row["record_id"], "--state", state, "--source", "backend-hook",
                "--message", f"standalone DEFT terminal for {row['backbone']}",
            ]
            if state == "ERROR":
                mark += ["--err-class", "ERR_PROGRAM"]
            record_command(*mark)
        write_json(STATUS, campaign_rows)
        counts = {state: sum(row["state"] == state for row in campaign_rows) for state in terminal}
        active = len(campaign_rows) - sum(counts.values())
        print(
            f"DEFT_FULL2000 running={active} complete={counts['COMPLETE']} "
            f"error={counts['ERROR']} canceled={counts['CANCELED']}",
            flush=True,
        )
        if active:
            time.sleep(30)
    if any(row["state"] != "COMPLETE" for row in campaign_rows):
        raise RuntimeError("one or more standalone DEFT jobs failed")
    print("DEFT_FULL2000_TERMINAL", flush=True)


if __name__ == "__main__":
    main()
