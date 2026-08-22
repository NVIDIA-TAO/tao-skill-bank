#!/usr/bin/env python3
"""Wait for all campaign tracks, then evaluate every winner on validation."""

from __future__ import annotations

import json
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from run_lam_track import IMAGE, LOCAL_ROOT, REMOTE_ROOT, SKILL_DIR, build_spec


RECORD = LOCAL_ROOT / "skill_bank_snapshot/scripts/tao_job_record.py"
REMOTE_SPEC_ROOT = REMOTE_ROOT / "evaluation_specs"


def tracks() -> list[dict]:
    rows: list[dict] = []
    for backbone in ("fan_base", "fan_large", "mit_b5"):
        for variant in ("bayesian", "bfbo", "bayesian_llm", "bfbo_llm"):
            name = f"automl_{backbone}_{variant}"
            rows.append(
                {
                    "name": name,
                    "kind": "automl",
                    "backbone": backbone,
                    "dataset": "original",
                    "workspace": LOCAL_ROOT / "workspaces" / name,
                }
            )
        for dataset in ("original", "mix50", "mix100"):
            name = f"control_{backbone}_{dataset}"
            rows.append(
                {
                    "name": name,
                    "kind": "control",
                    "backbone": backbone,
                    "dataset": dataset,
                    "workspace": LOCAL_ROOT / "workspaces" / f"{name}_exact_v2",
                }
            )
    return rows


def write_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def record_command(*args: str) -> str:
    return subprocess.check_output(
        [sys.executable, str(RECORD), *args], text=True
    ).strip()


def ssh_host() -> str:
    hosts = os.environ.get("SLURM_HOSTNAME", "")
    if not hosts:
        raise RuntimeError("SLURM_HOSTNAME is unset")
    return hosts.split(",", 1)[0]


def ssh(*remote_args: str) -> str:
    user = os.environ.get("SLURM_USER", "")
    if not user:
        raise RuntimeError("SLURM_USER is unset")
    return subprocess.check_output(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            f"{user}@{ssh_host()}",
            *remote_args,
        ],
        text=True,
    ).strip()


def wait_for_track_results(rows: list[dict]) -> None:
    while True:
        missing = [row["name"] for row in rows if not (row["workspace"] / "track_result.json").is_file()]
        if not missing:
            print("ALL_TRACK_RESULTS_READY", flush=True)
            return
        print(f"WAITING_FOR_TRACK_RESULTS missing={len(missing)}", flush=True)
        time.sleep(30)


def best_job_id(row: dict) -> str:
    run_dirs = sorted(row["workspace"].glob("run_*"))
    if not run_dirs:
        raise RuntimeError(f"no run directory for {row['name']}")
    best_files = list((run_dirs[-1] / ".automl/best_rec").glob("*.json"))
    if len(best_files) != 1:
        raise RuntimeError(f"expected one best_rec for {row['name']}, found {len(best_files)}")
    payload = json.loads(best_files[0].read_text())
    job_id = payload.get("rec_data", {}).get("job_id", "")
    if not job_id:
        raise RuntimeError(f"best_rec has no job_id for {row['name']}")
    return str(job_id)


def checkpoint_for(job_id: str) -> str:
    train_dir = REMOTE_ROOT / "results" / job_id / "results_dir/train"
    # shell globbing happens remotely; the fixed UUID/path components contain
    # no shell metacharacters.
    output = ssh(
        f"find '{train_dir}' -maxdepth 1 -type f -name 'model_epoch_*.pth' "
        "-printf '%p\\n' | sort -V | tail -1"
    )
    if not output:
        raise RuntimeError(f"no checkpoint found for training job {job_id}")
    return output


def evaluation_spec(row: dict, checkpoint: str, output_root: Path) -> dict:
    spec = build_spec(row["backbone"], row["dataset"])
    output_dir = output_root / row["name"]
    spec["model_name"] = f"lam_eval_{row['name']}"
    spec["results_dir"] = str(output_dir)
    spec["wandb"]["enable"] = False
    spec["evaluate"] = {
        "checkpoint": checkpoint,
        "trt_engine": "",
        "num_gpus": 8,
        "gpu_ids": list(range(8)),
        "num_nodes": 1,
        "batch_size": 1,
        "vis_after_n_batches": 1,
        "results_dir": str(output_dir),
    }
    # Evaluation consumes only validation_split=val. The test split remains a
    # duplicate of val and is intentionally not used for selection or scoring.
    spec["dataset"]["segment"]["validation_split"] = "val"
    return spec


def stage_spec(row: dict, spec: dict, label: str, output_root: Path) -> str:
    local_dir = LOCAL_ROOT / "evaluation_specs" / label
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / f"{row['name']}.yaml"
    local_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    remote_spec_dir = REMOTE_SPEC_ROOT / label
    ssh("mkdir", "-p", str(remote_spec_dir), str(output_root / row["name"]))
    remote_path = str(remote_spec_dir / local_path.name)
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
    return remote_path


def public(rows: list[dict]) -> list[dict]:
    return [
        {key: (str(value) if isinstance(value, Path) else value) for key, value in row.items()}
        for row in rows
    ]


def main() -> None:
    from tao_sdk.platforms.slurm import SlurmSDK

    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="evaluations")
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()

    rows = tracks()
    if args.only:
        requested = set(args.only)
        rows = [row for row in rows if row["name"] in requested]
        missing = requested - {row["name"] for row in rows}
        if missing:
            raise ValueError(f"unknown evaluation tracks: {sorted(missing)}")
    output_root = REMOTE_ROOT / args.label
    manifest = LOCAL_ROOT / f"final_{args.label}_manifest.json"
    status_path = LOCAL_ROOT / f"final_{args.label}_status.json"
    wait_for_track_results(rows)
    os.environ["TAO_SDK_STATE_DIR"] = str(
        LOCAL_ROOT / "sdk_state" / f"final_{args.label}"
    )
    sdk = SlurmSDK(poll_interval=30, epoch_milestone_interval=5)

    for row in rows:
        train_job_id = best_job_id(row)
        checkpoint = checkpoint_for(train_job_id)
        spec = evaluation_spec(row, checkpoint, output_root)
        remote_spec = stage_spec(row, spec, args.label, output_root)
        results_dir = str(output_root / row["name"])
        record_id = record_command(
            "open",
            "--platform",
            "slurm",
            "--image",
            IMAGE,
            "--network-arch",
            "segformer",
            "--action",
            "evaluate",
            "--storage-tier",
            "A",
            "--results-dir",
            results_dir,
        )
        row.update(
            {
                "train_job_id": train_job_id,
                "checkpoint": checkpoint,
                "spec": remote_spec,
                "results_dir": results_dir,
                "record_id": record_id,
                "state": "PENDING",
            }
        )
        write_json(manifest, public(rows))
        try:
            command = (
                "python3 "
                f"{REMOTE_ROOT}/controller/segformer_entrypoint.py "
                f"evaluate -e {remote_spec}"
            )
            job = sdk.create_job(
                image=IMAGE,
                command=command,
                gpu_count=8,
                num_nodes=1,
                account=os.environ.get("SLURM_ACCOUNT") or None,
                env_vars={"PYTHONPATH": "/usr/local/lib/python3.12/dist-packages"},
            )
            row["job_id"] = job.id
            row["backend_ref"] = job.backend_job_id
            row["state"] = "RUNNING"
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
                f"validation evaluation submitted for {row['name']}",
            )
            print(f"SUBMITTED {row['name']} {job.id} {job.backend_job_id}", flush=True)
        except Exception as exc:
            row["state"] = "ERROR"
            row["error"] = f"{type(exc).__name__}: {exc}"
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
                f"evaluation submission failed for {row['name']}",
            )
            write_json(manifest, public(rows))
            raise
        write_json(manifest, public(rows))

    terminal = {"COMPLETE", "ERROR", "CANCELED"}
    while any(row.get("state") not in terminal for row in rows):
        for row in rows:
            if row.get("state") in terminal:
                continue
            status = sdk.get_job_status(row["job_id"])
            state = status.status.upper()
            if state == "CANCELLED":
                state = "CANCELED"
            row["state"] = state
            if state in terminal:
                mark = [
                    "mark",
                    row["record_id"],
                    "--state",
                    state,
                    "--source",
                    "backend-hook",
                    "--message",
                    f"validation evaluation terminal for {row['name']}",
                ]
                if state == "ERROR":
                    mark += ["--err-class", "ERR_PROGRAM"]
                record_command(*mark)
        write_json(status_path, public(rows))
        counts = {state: sum(row.get("state") == state for row in rows) for state in terminal}
        running = len(rows) - sum(counts.values())
        print(
            f"EVALUATIONS running={running} complete={counts['COMPLETE']} "
            f"error={counts['ERROR']} canceled={counts['CANCELED']}",
            flush=True,
        )
        if running:
            time.sleep(30)
    print("FINAL_EVALUATIONS_TERMINAL", flush=True)


if __name__ == "__main__":
    main()
