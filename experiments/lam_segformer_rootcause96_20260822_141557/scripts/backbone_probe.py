#!/usr/bin/env python3
"""Run frozen then unfrozen one-epoch training probes for one backbone."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ENTRYPOINT = Path(
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
    "lam_segformer_rootcause96_20260822_141557/controller/scripts/"
    "segformer_entrypoint.py"
)


def run(spec: Path, mode: str) -> dict:
    print(f"BACKBONE_PROBE_START mode={mode} spec={spec}", flush=True)
    completed = subprocess.run(
        [sys.executable, str(ENTRYPOINT), "train", "-e", str(spec)],
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"{mode} backbone probe failed with exit code {completed.returncode}"
        )
    return {"mode": mode, "spec": str(spec), "returncode": completed.returncode}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--frozen-spec", type=Path, required=True)
    parser.add_argument("--unfrozen-spec", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    rows = [run(args.frozen_spec, "frozen"), run(args.unfrozen_spec, "unfrozen")]
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(
            {"schema_version": 1, "backbone": args.name, "probes": rows},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"BACKBONE_PROBE_OK name={args.name}", flush=True)


if __name__ == "__main__":
    main()
