#!/usr/bin/env python3
"""Launch the twelve reviewed full-budget AutoML controllers."""

from __future__ import annotations

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
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def main() -> None:
    required = ("SLURM_HOSTNAME", "SLURM_USER", "NGC_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"required environment variables are unset: {missing}")
    if not SCRIPT.is_file() or not AUTOML_SRC.is_dir():
        raise RuntimeError("campaign controller or patched AutoML source is missing")

    for path in (LOCAL_ROOT / "logs", LOCAL_ROOT / "workspaces", LOCAL_ROOT / "sdk_state"):
        path.mkdir(parents=True, exist_ok=True)
    manifest_path = LOCAL_ROOT / "controller_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        live = [row for row in previous.get("controllers", []) if process_alive(int(row["pid"]))]
        if live:
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

    launched = []
    for backbone in BACKBONES:
        for variant in VARIANTS:
            track = f"{backbone}_{variant}"
            log_path = LOCAL_ROOT / "logs" / f"{track}.log"
            log_handle = log_path.open("a", encoding="utf-8")
            command = [
                sys.executable, "-u", str(SCRIPT),
                "--backbone", backbone, "--variant", variant,
            ]
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
                    "recommendations": 10,
                    "epochs_per_recommendation": 2000,
                    "gpus_per_recommendation": 8,
                }
            )

    manifest = {
        "campaign": CAMPAIGN,
        "remote_root": str(REMOTE_ROOT),
        "partitions": ["polar", "polar3", "polar4", "grizzly"],
        "launched_at": now(),
        "controllers": launched,
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
