#!/usr/bin/env python3
"""Resume a long SegFormer run after a SLURM timeout/requeue."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


TAO_SITE_PACKAGES = "/usr/local/lib/python3.12/dist-packages"
if TAO_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, TAO_SITE_PACKAGES)

import yaml


REMOTE_ROOT = Path(
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
    "lam_segformer_bayes_deft_20260820_231724"
)
SEGFORMER_ENTRYPOINT = REMOTE_ROOT / "controller/segformer_entrypoint.py"
CHECKPOINT_RE = re.compile(r"model_epoch_(\d+)_step_(\d+)\.pth$")


def checkpoint_key(path: Path) -> tuple[int, int]:
    match = CHECKPOINT_RE.search(path.name)
    if not match:
        return (-1, -1)
    return int(match.group(1)), int(match.group(2))


def latest_checkpoint(results_dir: Path) -> Path | None:
    candidates = [
        path
        for path in (results_dir / "train").glob("model_epoch_*_step_*.pth")
        if path.is_file() and path.stat().st_size > 1_000_000
    ]
    return max(candidates, key=checkpoint_key) if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()

    spec = yaml.safe_load(args.spec.read_text())
    results_dir = Path(spec["results_dir"])
    checkpoint = latest_checkpoint(results_dir)
    if checkpoint is not None:
        spec["train"]["resume_training_checkpoint_path"] = str(checkpoint)
        print(f"SLURM_REQUEUE_RESUME={checkpoint}", flush=True)
    else:
        spec["train"]["resume_training_checkpoint_path"] = ""
        print("SLURM_REQUEUE_RESUME=FRESH", flush=True)

    runtime_spec = Path("/tmp") / f"segformer_resume_{os.getpid()}.yaml"
    runtime_spec.write_text(yaml.safe_dump(spec, sort_keys=False))
    os.execv(
        sys.executable,
        [
            sys.executable,
            str(SEGFORMER_ENTRYPOINT),
            "train",
            "-e",
            str(runtime_spec),
        ],
    )


if __name__ == "__main__":
    main()
