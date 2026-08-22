#!/usr/bin/env python3
"""Launch prediction/rank fusion and checkpoint soups for all three backbones."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml

from run_lam_track import IMAGE, LOCAL_ROOT, REMOTE_ROOT


RECORD = LOCAL_ROOT / "skill_bank_snapshot/scripts/tao_job_record.py"
LOCAL_MANIFEST_ROOT = LOCAL_ROOT / "downstream_fusion_soup_manifests"
REMOTE_MANIFEST_ROOT = REMOTE_ROOT / "downstream_fusion_soup_manifests"
REMOTE_OUTPUT_ROOT = REMOTE_ROOT / "downstream_fusion_soup"
BACKBONES = ("fan_base", "fan_large", "mit_b5")
EXPECTED_PARTITIONS = {"polar", "polar3", "polar4", "grizzly"}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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
    remote_command = " ".join(shlex.quote(value) for value in remote_args)
    return subprocess.check_output(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            f"{user}@{ssh_host()}",
            remote_command,
        ],
        text=True,
    ).strip()


def scp(local_path: Path, remote_path: Path) -> None:
    subprocess.run(
        [
            "scp",
            "-q",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            str(local_path),
            f"{os.environ['SLURM_USER']}@{ssh_host()}:{remote_path}",
        ],
        check=True,
    )


def train_sources(backbone: str) -> list[dict]:
    return [
        {
            "name": "original_automl_bfbo",
            "source": "original_data_automl",
            "backbone": backbone,
            "spec": str(
                REMOTE_ROOT
                / "full2000_specs"
                / f"full2000_{backbone}_automl.yaml"
            ),
            "local_spec": str(
                LOCAL_ROOT
                / "full2000_specs"
                / f"full2000_{backbone}_automl.yaml"
            ),
            "train_dir": str(
                REMOTE_ROOT
                / "full2000"
                / f"full2000_{backbone}_automl"
                / "train"
            ),
        },
        {
            "name": "standalone_deft_mix25",
            "source": "standalone_deft",
            "backbone": backbone,
            "spec": str(
                REMOTE_ROOT
                / "deft_full2000_specs"
                / f"deft_full2000_{backbone}_mix25.yaml"
            ),
            "local_spec": str(
                LOCAL_ROOT
                / "deft_full2000_specs"
                / f"deft_full2000_{backbone}_mix25.yaml"
            ),
            "train_dir": str(
                REMOTE_ROOT
                / "deft_full2000"
                / f"deft_full2000_{backbone}_mix25"
                / "train"
            ),
        },
        {
            "name": "deft_automl_bfbo",
            "source": "deft_snapshot_automl",
            "backbone": backbone,
            "spec": str(
                REMOTE_ROOT
                / "deft_automl_full2000_specs"
                / f"deft_automl_full2000_{backbone}_bfbo.yaml"
            ),
            "local_spec": str(
                LOCAL_ROOT
                / "deft_automl_full2000_specs"
                / f"deft_automl_full2000_{backbone}_bfbo.yaml"
            ),
            "train_dir": str(
                REMOTE_ROOT
                / "deft_automl_full2000"
                / f"deft_automl_full2000_{backbone}_bfbo"
                / "train"
            ),
        },
    ]


def nearest_saved_to_best(train_dir: str) -> dict:
    script = r'''import glob,json,os,sys
d=sys.argv[1]
values=[]
with open(os.path.join(d,"status.json"),errors="replace") as handle:
    for line in handle:
        try: row=json.loads(line)
        except Exception: continue
        epoch=row.get("epoch")
        kpi=row.get("kpi") or {}
        value=next((kpi.get(k) for k in ("val_miou","miou","mIoU") if isinstance(kpi.get(k),(int,float))),None)
        if isinstance(epoch,int) and value is not None:
            values.append((float(value),epoch))
if not values:
    raise RuntimeError("no validation metrics in "+d)
best_value,best_epoch=max(values)
checkpoints=[]
for path in glob.glob(os.path.join(d,"model_epoch_*_step_*.pth")):
    try: epoch=int(os.path.basename(path).split("_")[2])
    except Exception: continue
    size=os.path.getsize(path)
    if size>1024*1024:
        checkpoints.append((epoch,path,size))
if not checkpoints:
    raise RuntimeError("no valid checkpoint in "+d)
epoch,path,size=min(checkpoints,key=lambda row:(abs(row[0]-best_epoch),-row[0]))
print(json.dumps({"checkpoint":path,"checkpoint_epoch":epoch,"checkpoint_bytes":size,"retained_best_epoch":best_epoch,"retained_best_val_miou":best_value,"selection":"nearest durable checkpoint to retained best validation epoch"}))'''
    return json.loads(ssh("python3", "-c", script, train_dir))


def preflight_model_rows(rows: list[dict]) -> None:
    architecture = None
    for row in rows:
        local_spec = Path(row["local_spec"])
        if not local_spec.is_file():
            raise FileNotFoundError(local_spec)
        spec = yaml.safe_load(local_spec.read_text())
        if not isinstance(spec, dict) or not isinstance(spec.get("dataset"), dict):
            raise ValueError(f"spec is not a nested mapping: {local_spec}")
        if spec["dataset"]["segment"].get("num_classes") != 4:
            raise ValueError(f"expected four classes in {local_spec}")
        current = json.dumps(spec.get("model", {}), sort_keys=True)
        if architecture is None:
            architecture = current
        elif current != architecture:
            raise ValueError(f"model architecture mismatch within {row['backbone']}")


def stage_inputs(backbone: str, rows: list[dict]) -> tuple[str, str]:
    public_models = [
        {key: value for key, value in row.items() if key not in {"local_spec", "train_dir"}}
        for row in rows
    ]
    common = {
        "schema_version": 1,
        "experiment": f"lam_{backbone}_three_source",
        "backbone": backbone,
        "dataset_root": "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/lam_research",
        "selection_split": "val",
        "expected_samples": 1262,
        "num_classes": 4,
        "test_used_for_selection": False,
        "models": public_models,
    }
    local_fusion = LOCAL_MANIFEST_ROOT / f"prediction_fusion_{backbone}.json"
    local_soup = LOCAL_MANIFEST_ROOT / f"checkpoint_soup_{backbone}.json"
    write_json(local_fusion, {**common, "method": "probability_and_class_rank_fusion"})
    write_json(local_soup, {**common, "method": "same_backbone_checkpoint_interpolation"})
    remote_fusion = REMOTE_MANIFEST_ROOT / local_fusion.name
    remote_soup = REMOTE_MANIFEST_ROOT / local_soup.name
    scp(local_fusion, remote_fusion)
    scp(local_soup, remote_soup)
    return str(remote_fusion), str(remote_soup)


def public(rows: list[dict]) -> list[dict]:
    return [
        {key: (str(value) if isinstance(value, Path) else value) for key, value in row.items()}
        for row in rows
    ]


def valid_terminal_artifacts(row: dict) -> bool:
    script = r'''import json,os,sys
kind,root=sys.argv[1:3]
result=os.path.join(root,"results.json")
if not os.path.isfile(result) or os.path.getsize(result)<=100:
    raise SystemExit(1)
with open(result) as handle:
    payload=json.load(handle)
if payload.get("sample_count")!=1262 or "best" not in payload:
    raise SystemExit(1)
if kind=="checkpoint_soup":
    checkpoint=os.path.join(root,"best_soup.pth")
    if not os.path.isfile(checkpoint) or os.path.getsize(checkpoint)<=1024*1024:
        raise SystemExit(1)'''
    try:
        ssh("python3", "-c", script, row["kind"], row["results_dir"])
    except subprocess.CalledProcessError:
        return False
    return True


def main() -> None:
    from tao_sdk.platforms.slurm import SlurmSDK

    if not os.environ.get("SLURM_ACCOUNT"):
        raise RuntimeError("SLURM_ACCOUNT is unset")
    partitions = {
        value.strip()
        for value in os.environ.get("SLURM_PARTITION", "").split(",")
        if value.strip()
    }
    if partitions != EXPECTED_PARTITIONS:
        raise RuntimeError(
            f"SLURM_PARTITION must contain exactly {sorted(EXPECTED_PARTITIONS)}"
        )

    controller = REMOTE_ROOT / "controller"
    ssh(
        "mkdir",
        "-p",
        str(controller),
        str(REMOTE_MANIFEST_ROOT),
        str(REMOTE_OUTPUT_ROOT),
    )
    for script_name in ("score_prediction_fusions.py", "score_checkpoint_soups.py"):
        scp(LOCAL_ROOT / script_name, controller / script_name)

    jobs = []
    for backbone in BACKBONES:
        model_rows = train_sources(backbone)
        for row in model_rows:
            row.update(nearest_saved_to_best(row["train_dir"]))
        preflight_model_rows(model_rows)
        fusion_manifest, soup_manifest = stage_inputs(backbone, model_rows)
        fusion_root = REMOTE_OUTPUT_ROOT / "prediction_fusion" / backbone
        soup_root = REMOTE_OUTPUT_ROOT / "checkpoint_soup" / backbone
        jobs.extend(
            [
                {
                    "name": f"prediction_fusion_{backbone}",
                    "backbone": backbone,
                    "kind": "prediction_fusion",
                    "models": model_rows,
                    "results_dir": str(fusion_root),
                    "command": (
                        "torchrun --standalone --nproc_per_node=8 "
                        f"{controller}/score_prediction_fusions.py "
                        f"--manifest {fusion_manifest} "
                        f"--output {fusion_root}/results.json"
                    ),
                },
                {
                    "name": f"checkpoint_soup_{backbone}",
                    "backbone": backbone,
                    "kind": "checkpoint_soup",
                    "models": model_rows,
                    "results_dir": str(soup_root),
                    "command": (
                        "torchrun --standalone --nproc_per_node=8 "
                        f"{controller}/score_checkpoint_soups.py "
                        f"--manifest {soup_manifest} "
                        f"--output {soup_root}/results.json "
                        f"--best-checkpoint {soup_root}/best_soup.pth"
                    ),
                },
            ]
        )

    remote_preflight = r'''import json,os,sys
jobs=json.loads(sys.argv[1])
for job in jobs:
    os.makedirs(job["results_dir"],exist_ok=True)
    for model in job["models"]:
        for key in ("checkpoint","spec"):
            path=model[key]
            if not os.path.isfile(path):
                raise RuntimeError(f"missing {key}: {path}")
        if os.path.getsize(model["checkpoint"]) <= 1024*1024:
            raise RuntimeError("checkpoint too small: "+model["checkpoint"])
print("DOWNSTREAM_REMOTE_PREFLIGHT_OK jobs="+str(len(jobs)))'''
    ssh("python3", "-c", remote_preflight, json.dumps(jobs))

    manifest_path = LOCAL_ROOT / "downstream_fusion_soup_manifest.json"
    status_path = LOCAL_ROOT / "downstream_fusion_soup_status.json"
    write_json(manifest_path, public(jobs))
    os.environ["TAO_SDK_STATE_DIR"] = str(
        LOCAL_ROOT / "sdk_state" / "downstream_fusion_soup"
    )
    sdk = SlurmSDK(poll_interval=30, epoch_milestone_interval=5)

    for row in jobs:
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
            row["results_dir"],
        )
        row["record_id"] = record_id
        row["state"] = "PENDING"
        write_json(manifest_path, public(jobs))
        try:
            job = sdk.create_job(
                image=IMAGE,
                command=row["command"],
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
                f"submitted {row['name']} with 8 GPUs",
            )
            print(
                f"SUBMITTED {row['name']} {job.id} {job.backend_job_id}",
                flush=True,
            )
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
                f"submission failed for {row['name']}",
            )
            write_json(manifest_path, public(jobs))
            raise
        write_json(manifest_path, public(jobs))

    terminal = {"COMPLETE", "ERROR", "CANCELED"}
    while any(row.get("state") not in terminal for row in jobs):
        for row in jobs:
            if row.get("state") in terminal:
                continue
            state = sdk.get_job_status(row["job_id"]).status.upper()
            if state == "CANCELLED":
                state = "CANCELED"
            if state == "COMPLETE" and not valid_terminal_artifacts(row):
                state = "ERROR"
                row["error"] = "backend completed without valid result artifacts"
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
                    f"terminal state for {row['name']}",
                ]
                if state == "ERROR":
                    mark += ["--err-class", "ERR_PROGRAM"]
                record_command(*mark)
        write_json(status_path, public(jobs))
        counts = {state: sum(row.get("state") == state for row in jobs) for state in terminal}
        running = len(jobs) - sum(counts.values())
        print(
            f"DOWNSTREAM running={running} complete={counts['COMPLETE']} "
            f"error={counts['ERROR']} canceled={counts['CANCELED']}",
            flush=True,
        )
        if running:
            time.sleep(30)
    print("DOWNSTREAM_TERMINAL", flush=True)


if __name__ == "__main__":
    main()
