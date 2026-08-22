#!/usr/bin/env python3
"""Validate the CPU-thread fix on FAN-L and DINOv3-L constructors."""

from __future__ import annotations

import json

import launch_core_campaign as core


def main() -> None:
    core.stage_controller()
    manifest_path = core.LOCAL_ROOT / "threadcap_diagnostic_manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"refusing duplicate thread-cap diagnostics: {manifest_path}")
    jobs = []
    prefix = (
        "export SLURM_JOB_NAME=bash PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 "
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1; "
        "unset SLURM_NTASKS SLURM_NNODES SLURM_PROCID SLURM_LOCALID "
        "SLURM_NTASKS_PER_NODE RANK LOCAL_RANK GROUP_RANK WORLD_SIZE NODE_RANK "
        "NUM_GPU_PER_NODE MASTER_ADDR MASTER_PORT NNODES NPROC_PER_NODE; "
    )
    for model_name, retry_of in (
        ("fan_large", None),
        ("dinov3_large", "segformer-evaluate-342357"),
    ):
        command = prefix + (
            f"python3 -X faulthandler -u {core.REMOTE_CONTROLLER}/scripts/"
            f"diagnose_backbones.py --only {model_name}"
        )
        submitted = core.submit_job(
            label=f"threadcap_{model_name}_constructor",
            action="evaluate",
            results_dir=str(core.REMOTE_ROOT / "probes" / f"threadcap_{model_name}"),
            command=command,
            retry_of=retry_of,
        )
        jobs.append(submitted)
        core.atomic_json(manifest_path, {"campaign": core.CAMPAIGN, "jobs": jobs})
        print(json.dumps(submitted, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
