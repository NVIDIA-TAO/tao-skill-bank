#!/usr/bin/env python3
"""Wait for DEFT AutoML search, then promote every brain automatically."""

from __future__ import annotations

import json
import subprocess
import sys
import time

from run_lam_track import LOCAL_ROOT


CAMPAIGN_STATUS = LOCAL_ROOT / "deft_automl_campaign_status.json"
CONTINUATION_STATUS = LOCAL_ROOT / "deft_automl_promotion_continuation.json"


def write_status(state: str, message: str) -> None:
    temporary = CONTINUATION_STATUS.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"state": state, "message": message, "updated_at": time.time()}, indent=2)
        + "\n"
    )
    temporary.replace(CONTINUATION_STATUS)


def campaign_state() -> str:
    if not CAMPAIGN_STATUS.is_file():
        return "RUNNING"
    try:
        rows = json.loads(CAMPAIGN_STATUS.read_text())
    except (OSError, json.JSONDecodeError):
        return "RUNNING"
    if len(rows) != 12:
        return "RUNNING"
    returncodes = [row.get("returncode") for row in rows]
    if any(code not in {None, 0} for code in returncodes):
        return "ERROR"
    if all(code == 0 for code in returncodes):
        return "COMPLETE"
    return "RUNNING"


def main() -> None:
    write_status("WAITING", "waiting for all 12 DEFT AutoML brains")
    while True:
        state = campaign_state()
        print(f"DEFT_AUTOML_PROMOTION_CONTINUATION search_state={state}", flush=True)
        if state == "COMPLETE":
            break
        if state == "ERROR":
            raise RuntimeError("DEFT AutoML search campaign failed")
        time.sleep(30)
    write_status("LAUNCHING", "search complete; launching all 12 promotions")
    subprocess.run(
        [sys.executable, str(LOCAL_ROOT / "launch_deft_automl_full2000.py")],
        check=True,
    )
    write_status("COMPLETE", "all DEFT AutoML promotions reached terminal state")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_status("ERROR", f"{type(exc).__name__}: {exc}")
        raise
