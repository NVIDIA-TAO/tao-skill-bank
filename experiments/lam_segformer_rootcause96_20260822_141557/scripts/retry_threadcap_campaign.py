#!/usr/bin/env python3
"""Retry FAN causal/DEFT runs after restoring safe CPU thread caps."""

from __future__ import annotations

import json
import shlex

import launch_core_campaign as core


def main() -> None:
    core.stage_controller()
    previous_manifest = json.loads(
        (core.LOCAL_ROOT / "standard_retry_manifest.json").read_text()
    )
    previous = {row["label"]: row for row in previous_manifest["jobs"]}
    manifest_path = core.LOCAL_ROOT / "threadcap_retry_manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"refusing duplicate thread-cap retry: {manifest_path}")

    rows = [row for row in core.full_run_rows() if row["group"] in {"causal", "deft"}]
    launched = []
    core.atomic_json(manifest_path, {"campaign": core.CAMPAIGN, "jobs": launched})
    prefix = (
        "export SLURM_JOB_NAME=bash PYTHONFAULTHANDLER=1 HYDRA_FULL_ERROR=1 "
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1; "
        "unset SLURM_NTASKS SLURM_NNODES SLURM_PROCID SLURM_LOCALID "
        "SLURM_NTASKS_PER_NODE RANK LOCAL_RANK GROUP_RANK WORLD_SIZE NODE_RANK "
        "NUM_GPU_PER_NODE MASTER_ADDR MASTER_PORT NNODES NPROC_PER_NODE; "
    )
    for row in rows:
        spec_path = core.REMOTE_SPECS / f"{row['label']}.yaml"
        command = prefix + (
            f"python3 -X faulthandler {core.REMOTE_CONTROLLER}/scripts/"
            f"resume_training_entrypoint.py --spec {shlex.quote(str(spec_path))}"
        )
        submitted = core.submit_job(
            label=row["label"],
            action="train",
            results_dir=row["spec"]["results_dir"],
            command=command,
            retry_of=previous[row["label"]]["job_id"],
        )
        submitted.update(
            {"run_id": row["run_id"], "group": row["group"], "spec": str(spec_path)}
        )
        launched.append(submitted)
        core.atomic_json(manifest_path, {"campaign": core.CAMPAIGN, "jobs": launched})
        print(f"THREADCAP_RETRY_SUBMITTED {row['run_id']} {submitted['backend_ref']}", flush=True)


if __name__ == "__main__":
    main()
