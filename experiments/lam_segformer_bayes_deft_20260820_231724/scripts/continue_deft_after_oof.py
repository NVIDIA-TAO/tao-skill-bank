#!/usr/bin/env python3
"""Continue the DEFT pipeline automatically after all OOF evaluations finish."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from launch_final_evaluations import ssh
from run_lam_track import LOCAL_ROOT, REMOTE_ROOT, SOURCE_DATA


OOF_STATUS = LOCAL_ROOT / "deft_oof_status.json"
CONTINUATION_STATUS = LOCAL_ROOT / "deft_continuation_status.json"
REMOTE_CONTROLLER = REMOTE_ROOT / "controller"
OOF_ROOT = REMOTE_ROOT / "deft_oof_v1"
SNAPSHOT_ROOT = REMOTE_ROOT / "datasets/deft_model_v1_mix25"


def write_status(state: str, message: str) -> None:
    payload = {"state": state, "message": message, "updated_at": time.time()}
    temporary = CONTINUATION_STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(CONTINUATION_STATUS)


def oof_complete() -> bool:
    if not OOF_STATUS.is_file():
        return False
    try:
        rows = json.loads(OOF_STATUS.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        len(rows) == 12
        and all(row.get("stage") == "evaluate" for row in rows)
        and all(row.get("state") == "COMPLETE" for row in rows)
        and all(row.get("prediction_count") == 79 for row in rows)
    )


def postprocess() -> None:
    scores_root = OOF_ROOT / "scores"
    score_command = (
        f"python '{REMOTE_CONTROLLER}/score_deft_oof_predictions.py' "
        f"--manifest '{REMOTE_CONTROLLER}/deft_oof_manifest.json' "
        f"--evaluations-root '{OOF_ROOT}/evaluations' "
        f"--fold-data-root '{REMOTE_ROOT}/datasets/deft_oof_v1' "
        f"--output-root '{scores_root}'"
    )
    print("DEFT_CONTINUATION scoring_oof", flush=True)
    print(ssh(score_command), flush=True)
    ssh(
        " && ".join(
            f"test \"$(jq -r '.sample_count' '{scores_root}/{backbone}_oof_scores.json')\" -eq 316 && "
            f"test \"$(jq -r '.validation_used' '{scores_root}/{backbone}_oof_scores.json')\" = false"
            for backbone in ("fan_base", "fan_large", "mit_b5")
        )
    )

    if ssh(f"if [ -e '{SNAPSHOT_ROOT}' ]; then echo yes; else echo no; fi") == "yes":
        raise RuntimeError(f"refusing to overwrite existing DEFT snapshot root: {SNAPSHOT_ROOT}")
    snapshot_command = (
        f"python '{REMOTE_CONTROLLER}/build_deft_snapshots.py' "
        f"--scores-root '{scores_root}' "
        f"--embeddings-root '{OOF_ROOT}/embeddings' "
        f"--source-root '{SOURCE_DATA}' "
        f"--output-root '{SNAPSHOT_ROOT}' "
        "--anchor-fraction 0.20 --duplicate-fraction 0.25 "
        "--neighbors-per-anchor 3 --expected-count 316"
    )
    print("DEFT_CONTINUATION building_snapshots", flush=True)
    print(ssh(snapshot_command), flush=True)
    ssh(
        " && ".join(
            f"test \"$(jq -r '.method.resulting_train_count' '{SNAPSHOT_ROOT}/{backbone}/manifest.json')\" -eq 395 && "
            f"test \"$(jq -r '.validation_used_for_selection' '{SNAPSHOT_ROOT}/{backbone}/manifest.json')\" = false"
            for backbone in ("fan_base", "fan_large", "mit_b5")
        )
    )


def main() -> None:
    write_status("WAITING", "waiting for 12 complete OOF evaluations")
    while not oof_complete():
        print("DEFT_CONTINUATION waiting_for_oof", flush=True)
        time.sleep(30)
    write_status("POSTPROCESSING", "OOF complete; scoring and materializing snapshots")
    postprocess()
    write_status("LAUNCHING", "snapshots complete; launching 2,000-epoch DEFT runs")
    print("DEFT_CONTINUATION launching_full2000", flush=True)
    subprocess.run(
        [sys.executable, str(LOCAL_ROOT / "launch_deft_full2000.py")],
        check=True,
    )
    write_status("COMPLETE", "standalone DEFT controller reached terminal state")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_status("ERROR", f"{type(exc).__name__}: {exc}")
        raise
