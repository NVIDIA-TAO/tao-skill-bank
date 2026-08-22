#!/usr/bin/env python3
"""Train the best AutoML and DEFT-mix finalist per backbone to 2,000 epochs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from launch_final_evaluations import ssh
from run_lam_track import IMAGE, LOCAL_ROOT, REMOTE_ROOT, build_spec


RECORD = LOCAL_ROOT / "skill_bank_snapshot/scripts/tao_job_record.py"
EVAL_ROOT = REMOTE_ROOT / "evaluations_global_v2"
REMOTE_SPEC_ROOT = REMOTE_ROOT / "full2000_specs"
REMOTE_RESULT_ROOT = REMOTE_ROOT / "full2000"
MANIFEST = LOCAL_ROOT / "full2000_manifest.json"
STATUS = LOCAL_ROOT / "full2000_status.json"
FINAL_EPOCHS = 2000
REQUEUE_SAFE_CHECKPOINT_INTERVAL = 20


def write_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def record_command(*args: str) -> str:
    return subprocess.check_output(
        [sys.executable, str(RECORD), *args], text=True
    ).strip()


def evaluation_metric(name: str) -> float:
    status_path = EVAL_ROOT / name / "status.json"
    raw = ssh(
        f"tac '{status_path}' | jq -Rr "
        "'fromjson? | select(.kpi.test_miou != null) | .kpi.test_miou' | head -1"
    )
    if not raw:
        raise RuntimeError(f"global validation metric is missing for {name}")
    return float(raw)


def track_result(name: str, kind: str) -> dict:
    suffix = "_exact_v2" if kind == "control" else ""
    path = LOCAL_ROOT / "workspaces" / f"{name}{suffix}" / "track_result.json"
    if not path.is_file():
        raise RuntimeError(f"track result is missing: {path}")
    return json.loads(path.read_text())


def choose_finalists() -> list[dict]:
    finalists: list[dict] = []
    for backbone in ("fan_base", "fan_large", "mit_b5"):
        automl_names = [
            f"automl_{backbone}_{variant}"
            for variant in ("bayesian", "bfbo", "bayesian_llm", "bfbo_llm")
        ]
        automl_metric, automl_name = max(
            ((evaluation_metric(name), name) for name in automl_names),
            key=lambda item: (
                item[0],
                "_llm" not in item[1],
                "_bfbo" in item[1],
            ),
        )
        finalists.append(
            {
                "name": f"full2000_{backbone}_automl",
                "kind": "automl",
                "backbone": backbone,
                "dataset": "original",
                "source_track": automl_name,
                "selection_miou": automl_metric,
                "hyperparameters": track_result(automl_name, "automl")["best"]["specs"],
            }
        )

        control_names = [
            f"control_{backbone}_{dataset}"
            for dataset in ("original", "mix50", "mix100")
        ]
        control_metric, control_name = max(
            (evaluation_metric(name), name) for name in control_names
        )
        dataset = control_name.rsplit("_", 1)[-1]
        finalists.append(
            {
                "name": f"full2000_{backbone}_deft_{dataset}",
                "kind": "deft",
                "backbone": backbone,
                "dataset": dataset,
                "source_track": control_name,
                "selection_miou": control_metric,
                "hyperparameters": track_result(control_name, "control")["best"]["specs"],
            }
        )
    return finalists


def set_nested(spec: dict, dotted_key: str, value) -> None:
    node = spec
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def build_full_spec(row: dict) -> dict:
    spec = build_spec(row["backbone"], row["dataset"])
    for key, value in row["hyperparameters"].items():
        set_nested(spec, key, value)
    output_dir = REMOTE_RESULT_ROOT / row["name"]
    spec["model_name"] = f"lam_{row['name']}"
    spec["results_dir"] = str(output_dir)
    spec["train"].update(
        {
            "num_gpus": 8,
            "gpu_ids": list(range(8)),
            "num_nodes": 1,
            "num_epochs": FINAL_EPOCHS,
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


def stage_spec(row: dict, spec: dict) -> str:
    local_dir = LOCAL_ROOT / "full2000_specs"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / f"{row['name']}.yaml"
    local_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    remote_path = REMOTE_SPEC_ROOT / local_path.name
    ssh("mkdir", "-p", str(REMOTE_SPEC_ROOT), spec["results_dir"])
    user = os.environ["SLURM_USER"]
    host = os.environ["SLURM_HOSTNAME"].split(",", 1)[0]
    subprocess.run(
        [
            "scp",
            "-q",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            str(local_path),
            f"{user}@{host}:{remote_path}",
        ],
        check=True,
    )
    return str(remote_path)


def final_checkpoint(results_dir: str) -> str:
    return ssh(
        f"find '{results_dir}/train' -maxdepth 1 -type f "
        "-name 'model_epoch_1999_step_*.pth' -printf '%p\\n' | sort -V | tail -1"
    )


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
            f"2,000-epoch training submission failed for {row['name']}",
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
        f"2,000-epoch requeue-resumable training submitted for {row['name']}",
    )
    write_json(MANIFEST, rows)
    print(f"SUBMITTED {row['name']} {job.id}", flush=True)


def main() -> None:
    from tao_sdk.platforms.slurm import SlurmSDK

    # The partitions cap each allocation at four hours. At 3h48m, SLURM
    # requeues the same job; resume_training_entrypoint discovers the latest
    # periodic checkpoint while the spec and LR horizon remain 2,000 epochs.
    os.environ["SLURM_TIME_HOURS"] = "4"
    os.environ["SLURM_TIMEOUT_HOURS"] = "3.8"
    os.environ["SLURM_USE_REQUEUE"] = "true"
    os.environ["TAO_SDK_STATE_DIR"] = str(LOCAL_ROOT / "sdk_state/full2000")

    rows = choose_finalists()
    write_json(MANIFEST, rows)
    sdk = SlurmSDK(poll_interval=30, epoch_milestone_interval=10)
    for row in rows:
        submit(sdk, row, rows)

    terminal = {"COMPLETE", "ERROR", "CANCELED"}
    while any(row.get("state") not in terminal for row in rows):
        for row in rows:
            if row.get("state") in terminal:
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
                f"2,000-epoch training terminal for {row['name']}",
            ]
            if state == "ERROR":
                mark += ["--err-class", "ERR_PROGRAM"]
            record_command(*mark)
        write_json(STATUS, rows)
        counts = {state: sum(row.get("state") == state for row in rows) for state in terminal}
        running = len(rows) - sum(counts.values())
        print(
            f"FULL2000 running={running} complete={counts['COMPLETE']} "
            f"error={counts['ERROR']} canceled={counts['CANCELED']}",
            flush=True,
        )
        if running:
            time.sleep(30)
    print("FULL2000_TERMINAL", flush=True)


if __name__ == "__main__":
    main()
