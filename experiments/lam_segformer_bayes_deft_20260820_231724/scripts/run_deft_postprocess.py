#!/usr/bin/env python3
"""Run DEFT OOF scoring and snapshot construction inside the TAO container."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REMOTE_ROOT = Path(
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
    "lam_segformer_bayes_deft_20260820_231724"
)
CONTROLLER = REMOTE_ROOT / "controller"
OOF_ROOT = REMOTE_ROOT / "deft_oof_v1"
SCORES_ROOT = OOF_ROOT / "scores"
SNAPSHOT_ROOT = REMOTE_ROOT / "datasets/deft_model_v1_mix25"
SOURCE_DATA = Path(
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/lam_research"
)
BACKBONES = ("fan_base", "fan_large", "mit_b5")


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True)


def main() -> None:
    if SNAPSHOT_ROOT.exists():
        raise RuntimeError(f"refusing to overwrite existing snapshot: {SNAPSHOT_ROOT}")
    run(
        str(CONTROLLER / "score_deft_oof_predictions.py"),
        "--manifest", str(CONTROLLER / "deft_oof_manifest.json"),
        "--evaluations-root", str(OOF_ROOT / "evaluations"),
        "--fold-data-root", str(REMOTE_ROOT / "datasets/deft_oof_v1"),
        "--output-root", str(SCORES_ROOT),
    )
    for backbone in BACKBONES:
        payload = json.loads((SCORES_ROOT / f"{backbone}_oof_scores.json").read_text())
        if payload.get("sample_count") != 316 or payload.get("validation_used") is not False:
            raise RuntimeError(f"invalid OOF score payload for {backbone}")
    run(
        str(CONTROLLER / "build_deft_snapshots.py"),
        "--scores-root", str(SCORES_ROOT),
        "--embeddings-root", str(OOF_ROOT / "embeddings"),
        "--source-root", str(SOURCE_DATA),
        "--output-root", str(SNAPSHOT_ROOT),
        "--anchor-fraction", "0.20",
        "--duplicate-fraction", "0.25",
        "--neighbors-per-anchor", "3",
        "--expected-count", "316",
    )
    summaries = {}
    for backbone in BACKBONES:
        payload = json.loads((SNAPSHOT_ROOT / backbone / "manifest.json").read_text())
        if (
            payload.get("validation_used_for_selection") is not False
            or payload.get("method", {}).get("resulting_train_count") != 395
        ):
            raise RuntimeError(f"invalid DEFT snapshot for {backbone}")
        summaries[backbone] = {
            "mean_oof_miou": payload["provenance"]["mean_oof_miou"],
            "embedding_dimension": payload["provenance"]["embedding_dimension"],
            "anchor_count": payload["method"]["anchor_count"],
            "duplicate_count": payload["method"]["duplicate_count"],
            "train_count": payload["method"]["resulting_train_count"],
        }
    marker = OOF_ROOT / "postprocess_complete.json"
    temporary = marker.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "validation_used_for_selection": False,
                "snapshots": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary.replace(marker)
    print(f"DEFT_POSTPROCESS_COMPLETE marker={marker}", flush=True)


if __name__ == "__main__":
    main()
