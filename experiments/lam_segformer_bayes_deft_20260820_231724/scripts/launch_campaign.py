#!/usr/bin/env python3
"""Open durable TAO records, launch all approved controllers, and supervise."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(
    "/localhome/local-rarunachalam/workspace/"
    "lam_segformer_bayes_deft_20260820_231724"
)
RUNNER = ROOT / "run_lam_track.py"
BANK = ROOT / "skill_bank_snapshot"
RECORD = BANK / "scripts/tao_job_record.py"
IMAGE = "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt"


def tracks() -> list[dict]:
    result = []
    for backbone in ("fan_base", "fan_large", "mit_b5"):
        for variant in ("bayesian", "bfbo", "bayesian_llm", "bfbo_llm"):
            result.append(
                {
                    "name": f"automl_{backbone}_{variant}",
                    "kind": "automl",
                    "backbone": backbone,
                    "variant": variant,
                }
            )
        for dataset in ("original", "mix50", "mix100"):
            result.append(
                {
                    "name": f"control_{backbone}_{dataset}",
                    "kind": "control",
                    "backbone": backbone,
                    "variant": dataset,
                }
            )
    return result


def record_command(*args: str) -> str:
    return subprocess.check_output(
        [sys.executable, str(RECORD), *args], text=True
    ).strip()


def write_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("all", "automl", "control"), default="all")
    parser.add_argument("--retry-tag", default="")
    args = parser.parse_args()

    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (ROOT / "workspaces").mkdir(parents=True, exist_ok=True)
    live = []

    selected_tracks = [
        track for track in tracks()
        if args.kind == "all" or track["kind"] == args.kind
    ]
    for track in selected_tracks:
        workspace_name = track["name"]
        if args.retry_tag:
            workspace_name = f"{workspace_name}_{args.retry_tag}"
        workspace = ROOT / "workspaces" / workspace_name
        action = "automl" if track["kind"] == "automl" else "train"
        record_id = record_command(
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
            str(workspace),
        )
        log_path = logs / f"{track['name']}.log"
        log_handle = log_path.open("a", buffering=1)
        command = [
            sys.executable,
            str(RUNNER),
            "--kind",
            track["kind"],
            "--backbone",
            track["backbone"],
            "--variant",
            track["variant"],
        ]
        if args.retry_tag:
            command += ["--retry-tag", args.retry_tag]
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        record_command(
            "mark",
            record_id,
            "--state",
            "RUNNING",
            "--source",
            "agent",
            "--backend-ref",
            f"automl-controller-pid:{process.pid}",
            "--message",
            f"controller launched for {track['name']}",
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

    def public_rows() -> list[dict]:
        return [
            {k: value for k, value in item.items() if k not in {"process", "log_handle"}}
            for item in live
        ]

    output_stem = "campaign" if args.kind == "all" else f"campaign_{args.kind}_{args.retry_tag or 'retry'}"
    write_json(ROOT / f"{output_stem}_manifest.json", public_rows())
    print(f"LAUNCHED={len(live)}", flush=True)

    while any(item["returncode"] is None for item in live):
        for item in live:
            if item["returncode"] is not None:
                continue
            rc = item["process"].poll()
            if rc is None:
                continue
            item["returncode"] = rc
            item["log_handle"].close()
            state = "COMPLETE" if rc == 0 else "ERROR"
            mark = [
                "mark",
                item["record_id"],
                "--state",
                state,
                "--source",
                "backend-hook",
                "--message",
                f"controller exited rc={rc}",
            ]
            if rc != 0:
                mark += ["--err-class", "ERR_PROGRAM"]
            record_command(*mark)
        write_json(ROOT / f"{output_stem}_status.json", public_rows())
        running = sum(item["returncode"] is None for item in live)
        complete = sum(item["returncode"] == 0 for item in live)
        failed = sum(
            item["returncode"] not in {None, 0}
            for item in live
        )
        print(
            f"CONTROLLERS running={running} complete={complete} failed={failed}",
            flush=True,
        )
        if running:
            time.sleep(30)

    print("CAMPAIGN_CONTROLLERS_TERMINAL", flush=True)


if __name__ == "__main__":
    main()
