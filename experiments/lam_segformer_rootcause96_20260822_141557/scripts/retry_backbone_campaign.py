#!/usr/bin/env python3
"""Retry all modern-backbone probes and queue their 2,000-epoch runs."""

from __future__ import annotations

import json
import shlex

import launch_core_campaign as core


def main() -> None:
    core.stage_controller()
    original_manifest = json.loads(
        (core.LOCAL_ROOT / "core_launch_manifest.json").read_text()
    )
    original = {row["label"]: row for row in original_manifest["jobs"]}
    manifest_path = core.LOCAL_ROOT / "backbone_retry_manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"refusing duplicate backbone retry: {manifest_path}")

    probes = core.probe_rows()
    full_rows = [row for row in core.full_run_rows() if row["group"] == "backbone"]
    core.stage_probe_specs(probes)
    core.stage_specs(full_rows)
    full_by_probe = {}
    for row in full_rows:
        full_by_probe.setdefault(row["probe_key"], []).append(row)

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
    for probe in probes:
        command = prefix + (
            f"python3 -X faulthandler {core.REMOTE_CONTROLLER}/scripts/backbone_probe.py "
            f"--name {shlex.quote(probe['probe_key'])} "
            f"--frozen-spec {shlex.quote(probe['frozen_spec_path'])} "
            f"--unfrozen-spec {shlex.quote(probe['unfrozen_spec_path'])} "
            f"--receipt {shlex.quote(probe['results_dir'] + '/probe_receipt.json')}"
        )
        previous = original[probe["label"]]
        submitted_probe = core.submit_job(
            label=probe["label"],
            action="train",
            results_dir=probe["results_dir"],
            command=command,
            retry_of=previous["job_id"],
        )
        submitted_probe["group"] = "backbone_probe"
        submitted_probe["probe_key"] = probe["probe_key"]
        launched.append(submitted_probe)
        core.atomic_json(manifest_path, {"campaign": core.CAMPAIGN, "jobs": launched})
        print(
            f"BACKBONE_PROBE_RETRY_SUBMITTED {probe['probe_key']} "
            f"{submitted_probe['backend_ref']}",
            flush=True,
        )

        # Queue each frozen/full 2,000-epoch pair immediately while its probe
        # is pending/running. SLURM releases it only if both probe stages pass.
        for row in full_by_probe[probe["probe_key"]]:
            full_command = prefix + (
                f"python3 -X faulthandler {core.REMOTE_CONTROLLER}/scripts/"
                f"resume_training_entrypoint.py --spec {shlex.quote(row['spec_path'])}"
            )
            submitted = core.submit_job(
                label=row["label"],
                action="train",
                results_dir=row["spec"]["results_dir"],
                command=full_command,
                dependency=[submitted_probe["backend_ref"]],
            )
            submitted.update(
                {
                    "run_id": row["run_id"],
                    "group": row["group"],
                    "probe_key": row["probe_key"],
                    "spec": row["spec_path"],
                }
            )
            launched.append(submitted)
            core.atomic_json(manifest_path, {"campaign": core.CAMPAIGN, "jobs": launched})
            print(
                f"BACKBONE_FULL_QUEUED {row['run_id']} {submitted['backend_ref']} "
                f"afterok={submitted_probe['backend_ref']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
