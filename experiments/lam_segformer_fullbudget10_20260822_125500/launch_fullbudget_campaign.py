#!/usr/bin/env python3
"""Launch the twelve reviewed full-budget AutoML controllers."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


CAMPAIGN = "lam_segformer_fullbudget10_20260822_125500"
LOCAL_ROOT = Path("/localhome/local-rarunachalam/workspace") / CAMPAIGN
REMOTE_ROOT = Path("/lustre/fsw/portfolios/edgeai/users/rarunachalam") / CAMPAIGN
SCRIPT = Path(__file__).with_name("run_fullbudget_track.py")
AUTOML_SRC = Path("/localhome/local-rarunachalam/github/tao-automl/src")
BACKBONES = ("fan_base", "fan_large", "mit_b5")
VARIANTS = ("bayesian", "bfbo", "bayesian_llm", "bfbo_llm")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def process_alive(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        os.kill(pid, 0)
        if stat_path.is_file() and stat_path.read_text().split()[2] == "Z":
            return False
    except (OSError, ValueError):
        return False
    return True


def controller_alive(row: dict) -> bool:
    pid = int(row["pid"])
    if not process_alive(pid):
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return False
    return (
        str(SCRIPT) in command
        and f"--backbone {row['backbone']}" in command
        and f"--variant {row['variant']}" in command
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only controllers that are not currently alive",
    )
    parser.add_argument(
        "--tracks",
        nargs="*",
        help="optional track names to launch or resume",
    )
    args = parser.parse_args()

    required = ("SLURM_HOSTNAME", "SLURM_USER", "NGC_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"required environment variables are unset: {missing}")
    if not SCRIPT.is_file() or not AUTOML_SRC.is_dir():
        raise RuntimeError("campaign controller or patched AutoML source is missing")

    for path in (LOCAL_ROOT / "logs", LOCAL_ROOT / "workspaces", LOCAL_ROOT / "sdk_state"):
        path.mkdir(parents=True, exist_ok=True)
    manifest_path = LOCAL_ROOT / "controller_manifest.json"
    previous = {"controllers": []}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        live = [row for row in previous.get("controllers", []) if controller_alive(row)]
        if live and not args.resume:
            raise RuntimeError(f"refusing duplicate launch; {len(live)} controllers are alive")

    host = os.environ["SLURM_HOSTNAME"].split(",", 1)[0]
    subprocess.run(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            f"{os.environ['SLURM_USER']}@{host}",
            "mkdir", "-p", str(REMOTE_ROOT),
        ],
        check=True,
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(AUTOML_SRC) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PYTHONHASHSEED"] = "0"
    env["SLURM_PARTITION"] = "polar,polar3,polar4,grizzly"
    env["SLURM_SQSH_CACHE_DIR"] = (
        "/lustre/fsw/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam"
    )

    previous_by_track = {
        row["track"]: row for row in previous.get("controllers", [])
    }
    all_tracks = [
        f"{backbone}_{variant}"
        for backbone in BACKBONES
        for variant in VARIANTS
    ]
    requested = set(args.tracks or all_tracks)
    unknown = requested.difference(all_tracks)
    if unknown:
        raise RuntimeError(f"unknown tracks: {sorted(unknown)}")

    launched = []
    for backbone in BACKBONES:
        for variant in VARIANTS:
            track = f"{backbone}_{variant}"
            if track not in requested:
                continue
            prior = previous_by_track.get(track)
            if args.resume and prior and controller_alive(prior):
                continue
            log_path = LOCAL_ROOT / "logs" / f"{track}.log"
            log_handle = log_path.open("a", encoding="utf-8")
            command = [
                sys.executable, "-u", str(SCRIPT),
                "--backbone", backbone, "--variant", variant,
            ]
            if args.resume:
                command.append("--resume")
            process = subprocess.Popen(
                command,
                cwd=str(SCRIPT.parent),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            log_handle.close()
            launched.append(
                {
                    "track": track,
                    "backbone": backbone,
                    "variant": variant,
                    "pid": process.pid,
                    "log": str(log_path),
                    "started_at": now(),
                    "resume": args.resume,
                    "previous_pid": prior.get("pid") if prior else None,
                    "recommendations": 10,
                    "epochs_per_recommendation": 2000,
                    "gpus_per_recommendation": 8,
                }
            )

    merged = dict(previous_by_track)
    merged.update({row["track"]: row for row in launched})
    manifest = {
        "campaign": CAMPAIGN,
        "remote_root": str(REMOTE_ROOT),
        "partitions": ["polar", "polar3", "polar4", "grizzly"],
        "launched_at": previous.get("launched_at", now()),
        "last_updated_at": now(),
        "controllers": [merged[track] for track in all_tracks if track in merged],
        "last_launch": launched,
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
