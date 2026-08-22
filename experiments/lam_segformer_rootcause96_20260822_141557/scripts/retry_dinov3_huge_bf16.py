#!/usr/bin/env python3
"""Recover DINOv3-H+: frozen FP32 plus full-backbone BF16 at 1024px."""

from __future__ import annotations

import json
import shlex

import launch_core_campaign as core


TERMINAL_PROBES = {
    "segformer-train-3eb3f9": ("COMPLETE", "modern-backbone probe passed"),
    "segformer-train-de1ffc": ("ERROR", "unfrozen FP32 stage exceeded 80 GB GPU memory"),
    "segformer-train-055972": ("COMPLETE", "modern-backbone probe passed"),
    "segformer-train-c1fe12": ("COMPLETE", "modern-backbone probe passed"),
    "segformer-train-131e66": ("COMPLETE", "modern-backbone probe passed"),
    "segformer-train-5e943e": ("COMPLETE", "modern-backbone probe passed"),
}


def mark_previous_records() -> None:
    for job_id, (state, message) in TERMINAL_PROBES.items():
        args = [
            "mark", job_id,
            "--state", state,
            "--source", "poller",
            "--message", message,
        ]
        if state == "ERROR":
            args.extend(("--err-class", "ERR_PROGRAM"))
        core.record(*args)
    for job_id in ("segformer-train-843ced", "segformer-train-83cb36"):
        core.record(
            "mark", job_id,
            "--state", "CANCELED",
            "--source", "poller",
            "--message", "SLURM canceled job after its failed probe dependency",
        )


def command_for(spec_path: str) -> str:
    return (
        "export SLURM_JOB_NAME=bash PYTHONFAULTHANDLER=1 HYDRA_FULL_ERROR=1 "
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1; "
        "unset SLURM_NTASKS SLURM_NNODES SLURM_PROCID SLURM_LOCALID "
        "SLURM_NTASKS_PER_NODE RANK LOCAL_RANK GROUP_RANK WORLD_SIZE NODE_RANK "
        "NUM_GPU_PER_NODE MASTER_ADDR MASTER_PORT NNODES NPROC_PER_NODE; "
        f"python3 -X faulthandler {core.REMOTE_CONTROLLER}/scripts/"
        f"resume_training_entrypoint.py --spec {shlex.quote(spec_path)}"
    )


def main() -> None:
    manifest_path = core.LOCAL_ROOT / "dinov3_huge_bf16_retry_manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"refusing duplicate DINOv3-H+ retry: {manifest_path}")

    mark_previous_records()
    core.stage_controller()

    originals = {
        row["run_id"]: row
        for row in json.loads(
            (core.LOCAL_ROOT / "backbone_retry_manifest.json").read_text()
        )["jobs"]
        if row.get("run_id") in {"B03", "B04"}
    }
    rows = {
        row["run_id"]: row
        for row in core.full_run_rows()
        if row.get("run_id") in {"B03", "B04"}
    }
    rows["B04"]["spec"]["train"]["precision"] = "bf16-mixed"

    probe_label = "probe_dinov3_huge_plus_bf16_unfrozen"
    probe_spec = core.build_spec(
        label=probe_label,
        backbone=core.BACKBONES["dinov3_huge_plus"],
        ptm=core.PTMS["dinov3_huge_plus"],
        epochs=1,
        freeze=False,
        lr=1.0e-5,
        precision="bf16-mixed",
    )
    probe_row = {
        "run_id": "P-H-BF16",
        "label": probe_label,
        "group": "backbone_probe",
        "spec": probe_spec,
    }
    core.stage_specs([probe_row, rows["B03"], rows["B04"]])

    launched: list[dict] = []
    core.atomic_json(manifest_path, {"campaign": core.CAMPAIGN, "jobs": launched})

    b03 = core.submit_job(
        label=rows["B03"]["label"],
        action="train",
        results_dir=rows["B03"]["spec"]["results_dir"],
        command=command_for(rows["B03"]["spec_path"]),
        retry_of=originals["B03"]["job_id"],
    )
    b03.update(
        run_id="B03",
        group="backbone",
        probe_key="dinov3_huge_plus",
        spec=rows["B03"]["spec_path"],
        precision="32-true",
    )
    launched.append(b03)
    core.atomic_json(manifest_path, {"campaign": core.CAMPAIGN, "jobs": launched})

    probe = core.submit_job(
        label=probe_label,
        action="train",
        results_dir=probe_spec["results_dir"],
        command=command_for(probe_row["spec_path"]),
        retry_of="segformer-train-de1ffc",
    )
    probe.update(
        run_id="P-H-BF16",
        group="backbone_probe",
        probe_key="dinov3_huge_plus",
        spec=probe_row["spec_path"],
        precision="bf16-mixed",
    )
    launched.append(probe)
    core.atomic_json(manifest_path, {"campaign": core.CAMPAIGN, "jobs": launched})

    b04 = core.submit_job(
        label=rows["B04"]["label"],
        action="train",
        results_dir=rows["B04"]["spec"]["results_dir"],
        command=command_for(rows["B04"]["spec_path"]),
        dependency=[probe["backend_ref"]],
        retry_of=originals["B04"]["job_id"],
    )
    b04.update(
        run_id="B04",
        group="backbone",
        probe_key="dinov3_huge_plus",
        spec=rows["B04"]["spec_path"],
        precision="bf16-mixed",
    )
    launched.append(b04)
    core.atomic_json(manifest_path, {"campaign": core.CAMPAIGN, "jobs": launched})

    print(json.dumps({"manifest": str(manifest_path), "jobs": launched}, indent=2))


if __name__ == "__main__":
    main()
