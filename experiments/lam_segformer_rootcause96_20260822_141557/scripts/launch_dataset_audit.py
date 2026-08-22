#!/usr/bin/env python3
"""Stage and submit the approved read-only LAM dataset audit."""

from __future__ import annotations

import json
import shlex

import launch_core_campaign as core


def main() -> None:
    core.stage_controller()
    manifest_path = core.LOCAL_ROOT / "dataset_audit_launch_manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"refusing duplicate dataset-audit launch: {manifest_path}")
    output = core.REMOTE_ROOT / "dataset_audit"
    command = (
        "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1; "
        f"python3 {core.REMOTE_CONTROLLER}/scripts/dataset_audit.py "
        f"--dataset {shlex.quote(str(core.DATA_ROOT))} "
        f"--output {shlex.quote(str(output))}"
    )
    submitted = core.submit_job(
        label="dataset_integrity_distribution_audit",
        action="evaluate",
        results_dir=str(output),
        command=command,
    )
    submitted.update(
        {
            "audit_id": "A01",
            "source_dataset_read_only": True,
            "expected_outputs": [
                "audit_summary.json",
                "per_image.csv",
                "exact_duplicates.csv",
                "near_duplicates_train_val.csv",
                "group_summary.csv",
                "annotation_suspects.csv",
            ],
        }
    )
    core.atomic_json(
        manifest_path,
        {"campaign": core.CAMPAIGN, "jobs": [submitted]},
    )
    print(json.dumps({"manifest": str(manifest_path), "job": submitted}, indent=2))


if __name__ == "__main__":
    main()
