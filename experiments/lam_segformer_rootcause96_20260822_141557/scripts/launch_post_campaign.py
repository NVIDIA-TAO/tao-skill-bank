#!/usr/bin/env python3
"""Queue all seven validation-only fusion/post-processing jobs."""

from __future__ import annotations

import json
import shlex

import launch_core_campaign as core


def main() -> None:
    core.stage_controller()
    ddp = json.loads((core.LOCAL_ROOT / "ddp_retry_manifest.json").read_text())["jobs"]
    backbone = json.loads((core.LOCAL_ROOT / "backbone_retry_manifest.json").read_text())["jobs"]
    recovery_path = core.LOCAL_ROOT / "dinov3_huge_bf16_retry_manifest.json"
    replacement_by_run = {}
    if recovery_path.is_file():
        recovery = json.loads(recovery_path.read_text())["jobs"]
        replacement_by_run = {
            row["run_id"]: row
            for row in recovery
            if row.get("group") == "backbone" and row.get("run_id") in {"B03", "B04"}
        }
        if set(replacement_by_run) != {"B03", "B04"}:
            raise RuntimeError("DINOv3-H+ recovery exists but B03/B04 replacements are incomplete")
    full_backbones = [
        replacement_by_run.get(row["run_id"], row)
        for row in backbone
        if row.get("group") == "backbone"
    ]
    catalog = [
        {
            "run_id": row["run_id"],
            "label": row["label"],
            "group": row["group"],
            "spec": row["spec"],
            "results_dir": row["results_dir"],
            "backend_ref": row["backend_ref"],
        }
        for row in ddp + full_backbones
    ]
    if len(catalog) != 28:
        raise RuntimeError(f"expected 28 full-run catalog rows, found {len(catalog)}")
    if len({row["run_id"] for row in catalog}) != 28:
        raise RuntimeError("post model catalog contains duplicate run IDs")
    catalog_path = core.LOCAL_ROOT / "post_model_catalog.json"
    core.atomic_json(catalog_path, catalog)
    core.scp(catalog_path, core.REMOTE_CONTROLLER / "post_model_catalog.json")

    manifest_path = core.LOCAL_ROOT / "post_launch_manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"refusing duplicate post launch: {manifest_path}")
    launched = []
    core.atomic_json(manifest_path, {"campaign": core.CAMPAIGN, "jobs": launched})
    prefix = (
        "export SLURM_JOB_NAME=bash PYTHONFAULTHANDLER=1 HYDRA_FULL_ERROR=1 "
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 TAO_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; "
        "unset SLURM_NTASKS SLURM_NNODES SLURM_PROCID SLURM_LOCALID "
        "SLURM_NTASKS_PER_NODE RANK LOCAL_RANK GROUP_RANK WORLD_SIZE NODE_RANK "
        "NUM_GPU_PER_NODE MASTER_ADDR MASTER_PORT NNODES NPROC_PER_NODE; "
    )
    cache_command = prefix + (
        f"torchrun --standalone --nproc_per_node=8 {core.REMOTE_CONTROLLER}/scripts/"
        "fusion_postprocess.py cache"
    )
    cache = core.submit_job(
        label="post_cache_validation_logits",
        action="evaluate",
        results_dir=str(core.REMOTE_ROOT / "post/cache"),
        command=cache_command,
        dependency=[row["backend_ref"] for row in catalog],
        dependency_type="afterany",
    )
    cache["post_id"] = "P01"
    cache["mode"] = "cache"
    launched.append(cache)
    core.atomic_json(manifest_path, {"campaign": core.CAMPAIGN, "jobs": launched})
    print(f"POST_SUBMITTED P01 cache {cache['backend_ref']}", flush=True)

    modes = (
        ("P02", "basic", "basic_fusions"),
        ("P03", "global_cv", "global_nonnegative_cv"),
        ("P04", "class_cv", "class_specific_cv"),
        ("P05", "d4_tta", "d4_tta"),
        ("P06", "soup", "checkpoint_soups"),
        ("P07", "deft_all", "deft_all_fusions"),
    )
    for post_id, mode, directory in modes:
        command = prefix + (
            f"torchrun --standalone --nproc_per_node=8 {core.REMOTE_CONTROLLER}/scripts/"
            f"fusion_postprocess.py {shlex.quote(mode)}"
        )
        submitted = core.submit_job(
            label=f"post_{directory}",
            action="evaluate",
            results_dir=str(core.REMOTE_ROOT / "post" / directory),
            command=command,
            dependency=[cache["backend_ref"]],
        )
        submitted["post_id"] = post_id
        submitted["mode"] = mode
        launched.append(submitted)
        core.atomic_json(manifest_path, {"campaign": core.CAMPAIGN, "jobs": launched})
        print(f"POST_SUBMITTED {post_id} {mode} {submitted['backend_ref']}", flush=True)


if __name__ == "__main__":
    main()
