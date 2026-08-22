#!/usr/bin/env python3
"""Launch the four AutoML brains per backbone on model-driven DEFT data."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from launch_final_evaluations import ssh
from run_lam_track import IMAGE, LOCAL_ROOT, build_spec, dataset_root, validate_bundle


RUNNER = LOCAL_ROOT / "run_lam_track.py"
RECORD = LOCAL_ROOT / "skill_bank_snapshot/scripts/tao_job_record.py"
BACKBONES = ("fan_base", "fan_large", "mit_b5")
VARIANTS = ("bayesian", "bfbo", "bayesian_llm", "bfbo_llm")
MANIFEST = LOCAL_ROOT / "deft_automl_campaign_manifest.json"
STATUS = LOCAL_ROOT / "deft_automl_campaign_status.json"


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def record_command(*args: str) -> str:
    return subprocess.check_output([sys.executable, str(RECORD), *args], text=True).strip()


def tracks() -> list[dict]:
    return [
        {
            "name": f"deft_automl_{backbone}_{variant}",
            "kind": "deft_automl",
            "backbone": backbone,
            "variant": variant,
            "dataset": "deft25",
        }
        for backbone in BACKBONES
        for variant in VARIANTS
    ]


def preflight(campaign: list[dict]) -> None:
    expected = {
        f"deft_automl_{backbone}_{variant}"
        for backbone in BACKBONES
        for variant in VARIANTS
    }
    if len(campaign) != 12 or {row["name"] for row in campaign} != expected:
        raise RuntimeError("DEFT AutoML campaign is not exactly 12 brains")
    for backbone in BACKBONES:
        root = dataset_root(backbone, "deft25")
        raw = ssh(
            f"test -f '{root}/manifest.json' && "
            f"test \"$(jq -r '.validation_used_for_selection' '{root}/manifest.json')\" = false && "
            f"test \"$(jq -r '.method.resulting_train_count' '{root}/manifest.json')\" -eq 395 && "
            f"test \"$(find -L '{root}/images/train' -maxdepth 1 -type f | wc -l)\" -eq 395 && "
            f"test \"$(find -L '{root}/masks/train' -maxdepth 1 -type f | wc -l)\" -eq 395 && echo ok"
        )
        if raw != "ok":
            raise RuntimeError(f"invalid DEFT AutoML dataset for {backbone}: {raw}")
        spec = build_spec(backbone, "deft25")
        if spec["dataset"]["segment"]["root_dir"] != str(root):
            raise RuntimeError(f"wrong DEFT root in {backbone} spec")
        if spec["train"]["num_gpus"] != 8 or spec["train"]["num_epochs"] != 20:
            raise RuntimeError(f"wrong search-trial compute in {backbone} spec")
        bundle = LOCAL_ROOT / "deft_automl_bundles" / f"{backbone}.json"
        validate_bundle(backbone, "deft25", bundle)


def public_rows(live: list[dict]) -> list[dict]:
    return [
        {key: value for key, value in row.items() if key not in {"process", "log_handle"}}
        for row in live
    ]


def main() -> None:
    campaign = tracks()
    preflight(campaign)
    logs = LOCAL_ROOT / "logs/deft_automl"
    logs.mkdir(parents=True, exist_ok=True)
    print("DEFT_AUTOML_PREFLIGHT_OK brains=12 recommendations_per_brain=3 gpus=8", flush=True)
    live = []
    for track in campaign:
        workspace = LOCAL_ROOT / "workspaces" / track["name"]
        record_id = record_command(
            "open", "--platform", "slurm", "--image", IMAGE,
            "--network-arch", "segformer", "--action", "automl",
            "--storage-tier", "A", "--results-dir", str(workspace),
        )
        log_path = logs / f"{track['name']}.log"
        log_handle = log_path.open("a", buffering=1)
        process = subprocess.Popen(
            [
                sys.executable, str(RUNNER), "--kind", "deft_automl",
                "--backbone", track["backbone"], "--variant", track["variant"],
            ],
            cwd=LOCAL_ROOT,
            env=os.environ.copy(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        record_command(
            "mark", record_id, "--state", "RUNNING", "--source", "agent",
            "--backend-ref", f"automl-controller-pid:{process.pid}", "--message",
            f"DEFT AutoML controller launched for {track['name']}",
        )
        live.append(
            {
                **track,
                "record_id": record_id,
                "pid": process.pid,
                "log": str(log_path),
                "workspace": str(workspace),
                "process": process,
                "log_handle": log_handle,
                "returncode": None,
            }
        )
    write_json(MANIFEST, public_rows(live))
    print("DEFT_AUTOML_LAUNCHED controllers=12", flush=True)
    while any(row["returncode"] is None for row in live):
        for row in live:
            if row["returncode"] is not None:
                continue
            returncode = row["process"].poll()
            if returncode is None:
                continue
            row["returncode"] = returncode
            row["log_handle"].close()
            state = "COMPLETE" if returncode == 0 else "ERROR"
            mark = [
                "mark", row["record_id"], "--state", state, "--source", "backend-hook",
                "--message", f"DEFT AutoML controller exited rc={returncode}",
            ]
            if returncode:
                mark += ["--err-class", "ERR_PROGRAM"]
            record_command(*mark)
        write_json(STATUS, public_rows(live))
        counts = {
            "running": sum(row["returncode"] is None for row in live),
            "complete": sum(row["returncode"] == 0 for row in live),
            "failed": sum(row["returncode"] not in {None, 0} for row in live),
        }
        print(
            f"DEFT_AUTOML_CONTROLLERS running={counts['running']} "
            f"complete={counts['complete']} failed={counts['failed']}",
            flush=True,
        )
        if counts["running"]:
            time.sleep(30)
    if any(row["returncode"] != 0 for row in live):
        raise RuntimeError("one or more DEFT AutoML controllers failed")
    print("DEFT_AUTOML_TERMINAL", flush=True)


if __name__ == "__main__":
    main()
