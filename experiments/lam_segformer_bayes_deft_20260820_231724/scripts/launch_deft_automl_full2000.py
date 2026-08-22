#!/usr/bin/env python3
"""Promote all 12 AutoML-on-DEFT brain winners to 2,000 epochs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from launch_final_evaluations import ssh, ssh_host
from run_lam_track import IMAGE, LOCAL_ROOT, REMOTE_ROOT, build_spec, dataset_root


BACKBONES = ("fan_base", "fan_large", "mit_b5")
VARIANTS = ("bayesian", "bfbo", "bayesian_llm", "bfbo_llm")
RESULT_ROOT = REMOTE_ROOT / "deft_automl_full2000"
REMOTE_SPEC_ROOT = REMOTE_ROOT / "deft_automl_full2000_specs"
LOCAL_SPEC_ROOT = LOCAL_ROOT / "deft_automl_full2000_specs"
MANIFEST = LOCAL_ROOT / "deft_automl_full2000_manifest.json"
STATUS = LOCAL_ROOT / "deft_automl_full2000_status.json"
RECORD = LOCAL_ROOT / "skill_bank_snapshot/scripts/tao_job_record.py"
LONG_HORIZON_LR_CAP = 6.0e-5
REQUEUE_SAFE_CHECKPOINT_INTERVAL = 20


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def record_command(*args: str) -> str:
    return subprocess.check_output([sys.executable, str(RECORD), *args], text=True).strip()


def set_nested(spec: dict, dotted_key: str, value) -> None:
    node = spec
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def track_result(backbone: str, variant: str) -> dict:
    path = LOCAL_ROOT / "workspaces" / f"deft_automl_{backbone}_{variant}/track_result.json"
    if not path.is_file():
        raise RuntimeError(f"missing DEFT AutoML result: {path}")
    result = json.loads(path.read_text())
    best = result.get("best") or {}
    if set(best.get("specs", {})) != {
        "dataset.segment.augmentation.random_color.brightness",
        "dataset.segment.augmentation.random_color.color_probability",
        "dataset.segment.augmentation.random_color.contrast",
        "dataset.segment.augmentation.random_color.hue",
        "dataset.segment.augmentation.random_color.saturation",
        "dataset.segment.augmentation.random_flip.hflip_probability",
        "train.optim.lr",
        "train.optim.weight_decay",
    }:
        raise RuntimeError(f"invalid best recommendation in {path}")
    if not isinstance(best.get("metric_value"), (int, float)):
        raise RuntimeError(f"missing best metric in {path}")
    return result


def rows() -> list[dict]:
    campaign = []
    for backbone in BACKBONES:
        for variant in VARIANTS:
            result = track_result(backbone, variant)
            best = result["best"]
            campaign.append(
                {
                    "name": f"deft_automl_full2000_{backbone}_{variant}",
                    "brain": f"deft_automl_{backbone}_{variant}",
                    "backbone": backbone,
                    "variant": variant,
                    "dataset": "deft25",
                    "dataset_root": str(dataset_root(backbone, "deft25")),
                    "selection_metric": float(best["metric_value"]),
                    "selected_rec_id": best.get("rec_id"),
                    "hyperparameters": best["specs"],
                    "search_lr": float(best["specs"]["train.optim.lr"]),
                    "promotion_lr": min(
                        float(best["specs"]["train.optim.lr"]),
                        LONG_HORIZON_LR_CAP,
                    ),
                    "lr_transfer_rule": "min(search_lr, 6e-5) for 20-to-2000 epoch transfer",
                    "state": "PENDING",
                }
            )
    return campaign


def train_spec(row: dict) -> dict:
    spec = build_spec(row["backbone"], "deft25")
    for key, value in row["hyperparameters"].items():
        set_nested(spec, key, value)
    # A learning rate selected for 20 epochs must not remain at that magnitude
    # across a 100x-longer linear schedule. Retain the other seven winning
    # parameters, but cap LR at the historical stable 2,000-epoch value.
    spec["train"]["optim"]["lr"] = row.get(
        "promotion_lr",
        min(float(row["hyperparameters"]["train.optim.lr"]), LONG_HORIZON_LR_CAP),
    )
    output = RESULT_ROOT / row["name"]
    spec["model_name"] = f"lam_{row['name']}"
    spec["results_dir"] = str(output)
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
    spec["train"]["results_dir"] = ""
    return spec


def assert_nested(spec: dict) -> None:
    stack = [spec]
    while stack:
        node = stack.pop()
        for key, value in node.items():
            if "." in key:
                raise RuntimeError(f"dotted key remains: {key}")
            if isinstance(value, dict):
                stack.append(value)


def preflight(campaign: list[dict]) -> None:
    expected = {
        f"deft_automl_{backbone}_{variant}"
        for backbone in BACKBONES
        for variant in VARIANTS
    }
    if len(campaign) != 12 or {row["brain"] for row in campaign} != expected:
        raise RuntimeError("promotion set is not exactly 12 DEFT AutoML brains")
    for row in campaign:
        spec = train_spec(row)
        assert_nested(spec)
        train = spec["train"]
        if not (
            spec["dataset"]["segment"]["root_dir"] == row["dataset_root"]
            and train["num_epochs"] == 2000
            and train["checkpoint_interval"] == REQUEUE_SAFE_CHECKPOINT_INTERVAL
            and train["validation_interval"] == 10
            and train["num_gpus"] == 8
            and train["gpu_ids"] == list(range(8))
            and train["num_nodes"] == 1
            and train["optim"]["lr"] <= LONG_HORIZON_LR_CAP
        ):
            raise RuntimeError(f"invalid promotion spec for {row['brain']}")


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


def submit(sdk, row: dict, campaign: list[dict]) -> None:
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
    write_json(MANIFEST, campaign)
    command = f"python3 {REMOTE_ROOT}/controller/resume_training_entrypoint.py --spec {remote_spec}"
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
            f"DEFT AutoML 2,000-epoch submission failed for {row['brain']}",
        )
        write_json(STATUS, campaign)
        raise
    row.update({"job_id": job.id, "backend_ref": job.backend_job_id, "state": "RUNNING"})
    record_command(
        "mark", record_id, "--state", "RUNNING", "--source", "backend-hook",
        "--backend-ref", job.backend_job_id, "--message",
        f"DEFT AutoML 2,000-epoch promotion submitted for {row['brain']}",
    )
    write_json(MANIFEST, campaign)
    print(f"SUBMITTED {row['brain']} {job.id}", flush=True)


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
    os.environ["TAO_SDK_STATE_DIR"] = str(LOCAL_ROOT / "sdk_state/deft_automl_full2000_v1")
    campaign = rows()
    preflight(campaign)
    write_json(MANIFEST, campaign)
    print("DEFT_AUTOML_FULL2000_PREFLIGHT_OK brains=12 epochs=2000 gpus=8", flush=True)
    sdk = SlurmSDK(poll_interval=30, epoch_milestone_interval=10)
    for row in campaign:
        submit(sdk, row, campaign)
    terminal = {"COMPLETE", "ERROR", "CANCELED"}
    while any(row["state"] not in terminal for row in campaign):
        for row in campaign:
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
                "--message", f"DEFT AutoML promotion terminal for {row['brain']}",
            ]
            if state == "ERROR":
                mark += ["--err-class", "ERR_PROGRAM"]
            record_command(*mark)
        write_json(STATUS, campaign)
        counts = {state: sum(row["state"] == state for row in campaign) for state in terminal}
        active = len(campaign) - sum(counts.values())
        print(
            f"DEFT_AUTOML_FULL2000 running={active} complete={counts['COMPLETE']} "
            f"error={counts['ERROR']} canceled={counts['CANCELED']}",
            flush=True,
        )
        if active:
            time.sleep(30)
    if any(row["state"] != "COMPLETE" for row in campaign):
        raise RuntimeError("one or more DEFT AutoML promotions failed")
    print("DEFT_AUTOML_FULL2000_TERMINAL", flush=True)


if __name__ == "__main__":
    main()
