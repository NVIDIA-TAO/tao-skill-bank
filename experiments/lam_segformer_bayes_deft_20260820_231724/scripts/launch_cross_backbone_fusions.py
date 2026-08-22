#!/usr/bin/env python3
"""Launch best-checkpoint and hierarchical cross-backbone fusion jobs."""

from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path

from tao_sdk.platforms.slurm import SlurmSDK

from launch_downstream_fusion_soups import (
    EXPECTED_PARTITIONS,
    IMAGE,
    LOCAL_MANIFEST_ROOT,
    LOCAL_ROOT,
    REMOTE_MANIFEST_ROOT,
    REMOTE_OUTPUT_ROOT,
    REMOTE_ROOT,
    public,
    record_command,
    scp,
    ssh,
    valid_terminal_artifacts,
    write_json,
)


BACKBONES = ("fan_base", "fan_large", "mit_b5")
LOCAL_RESULT_ROOT = LOCAL_ROOT / "downstream_fusion_soup_results"
CROSS_ROOT = REMOTE_OUTPUT_ROOT / "cross_backbone"


def probability_rows(backbone: str) -> list[dict]:
    payload = json.loads(
        (LOCAL_RESULT_ROOT / f"prediction_fusion_{backbone}.json").read_text()
    )
    return [row for row in payload["results"] if row["scheme"] == "probability"]


def best_single_index(backbone: str) -> int:
    rows = []
    for row in probability_rows(backbone):
        nonzero = [index for index, value in enumerate(row["weights"]) if value]
        if len(nonzero) == 1 and abs(row["weights"][nonzero[0]] - 1.0) < 1.0e-8:
            rows.append((row["miou"], nonzero[0]))
    if len(rows) != 3:
        raise RuntimeError(f"expected three single baselines for {backbone}")
    return max(rows)[1]


def best_internal_weights(backbone: str) -> list[float]:
    return list(max(probability_rows(backbone), key=lambda row: row["miou"])["weights"])


def model_rows(backbone: str) -> list[dict]:
    payload = json.loads(
        (LOCAL_MANIFEST_ROOT / f"prediction_fusion_{backbone}.json").read_text()
    )
    return payload["models"]


def group_candidates(internal: dict[str, list[float]]) -> list[dict]:
    groups: dict[tuple[int, int, int], str] = {}
    for left, right in itertools.combinations(range(3), 2):
        for tenth in range(11):
            weights = [0, 0, 0]
            weights[left] = tenth * 10
            weights[right] = 100 - weights[left]
            groups.setdefault(tuple(weights), f"backbone_pair_{left}_{right}_tenths")
    for left in range(5):
        for middle in range(5 - left):
            right = 4 - left - middle
            groups.setdefault(
                (left * 25, middle * 25, right * 25),
                "backbone_quarter_simplex",
            )
    groups[(34, 33, 33)] = "near_uniform_backbones"

    candidates = []
    for key in sorted(groups):
        group_weights = [value / 100.0 for value in key]
        flat = []
        for group_weight, backbone in zip(group_weights, BACKBONES):
            flat.extend(group_weight * value for value in internal[backbone])
        total = sum(flat)
        flat = [value / total for value in flat]
        candidates.append(
            {
                "name": "group_" + "_".join(f"{value:03d}" for value in key),
                "weights": flat,
                "sources": [groups[key]],
            }
        )
    candidates.append(
        {
            "name": "uniform_nine_models",
            "weights": [1.0 / 9.0] * 9,
            "sources": ["uniform_all_sources_and_backbones"],
        }
    )
    return candidates


def stage_manifest(name: str, models: list[dict], candidates: list[dict] | None) -> str:
    payload = {
        "schema_version": 1,
        "experiment": name,
        "dataset_root": "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/lam_research",
        "selection_split": "val",
        "expected_samples": 1262,
        "num_classes": 4,
        "test_used_for_selection": False,
        "models": models,
    }
    if candidates:
        payload["weight_candidates"] = candidates
    local = LOCAL_MANIFEST_ROOT / f"{name}.json"
    remote = REMOTE_MANIFEST_ROOT / local.name
    write_json(local, payload)
    scp(local, remote)
    return str(remote)


def main() -> None:
    partitions = {
        value.strip()
        for value in os.environ.get("SLURM_PARTITION", "").split(",")
        if value.strip()
    }
    if partitions != EXPECTED_PARTITIONS:
        raise RuntimeError(
            f"SLURM_PARTITION must contain exactly {sorted(EXPECTED_PARTITIONS)}"
        )

    all_rows = {backbone: model_rows(backbone) for backbone in BACKBONES}
    top_models = [
        all_rows[backbone][best_single_index(backbone)] for backbone in BACKBONES
    ]
    internal = {backbone: best_internal_weights(backbone) for backbone in BACKBONES}
    nine_models = [row for backbone in BACKBONES for row in all_rows[backbone]]

    top_manifest = stage_manifest(
        "cross_backbone_top_checkpoints", top_models, None
    )
    hierarchy_manifest = stage_manifest(
        "hierarchical_nine_model_fusion",
        nine_models,
        group_candidates(internal),
    )

    controller = REMOTE_ROOT / "controller"
    ssh("mkdir", "-p", str(controller), str(CROSS_ROOT))
    scp(LOCAL_ROOT / "score_prediction_fusions.py", controller / "score_prediction_fusions.py")

    jobs = []
    for name, manifest in (
        ("cross_backbone_top_checkpoints", top_manifest),
        ("hierarchical_nine_model_fusion", hierarchy_manifest),
    ):
        root = CROSS_ROOT / name
        ssh("mkdir", "-p", str(root))
        jobs.append(
            {
                "name": name,
                "kind": "prediction_fusion",
                "results_dir": str(root),
                "manifest": manifest,
                "gpu_count": 8,
                "command": (
                    "torchrun --standalone --nproc_per_node=8 "
                    f"{controller}/score_prediction_fusions.py "
                    f"--manifest {manifest} --output {root}/results.json"
                ),
            }
        )

    manifest_path = LOCAL_ROOT / "cross_backbone_fusion_manifest.json"
    status_path = LOCAL_ROOT / "cross_backbone_fusion_status.json"
    write_json(manifest_path, public(jobs))
    os.environ["TAO_SDK_STATE_DIR"] = str(
        LOCAL_ROOT / "sdk_state" / "cross_backbone_fusion"
    )
    sdk = SlurmSDK(poll_interval=30, epoch_milestone_interval=5)

    for row in jobs:
        record_id = record_command(
            "open",
            "--platform",
            "slurm",
            "--image",
            IMAGE,
            "--network-arch",
            "segformer",
            "--action",
            "evaluate",
            "--storage-tier",
            "A",
            "--results-dir",
            row["results_dir"],
        )
        row["record_id"] = record_id
        row["state"] = "PENDING"
        try:
            job = sdk.create_job(
                image=IMAGE,
                command=row["command"],
                gpu_count=8,
                num_nodes=1,
                account=os.environ.get("SLURM_ACCOUNT") or None,
                env_vars={"PYTHONPATH": "/usr/local/lib/python3.12/dist-packages"},
            )
            row["job_id"] = job.id
            row["backend_ref"] = job.backend_job_id
            row["state"] = "RUNNING"
            record_command(
                "mark",
                record_id,
                "--state",
                "RUNNING",
                "--source",
                "backend-hook",
                "--backend-ref",
                job.backend_job_id,
                "--message",
                f"submitted {row['name']} with 8 GPUs",
            )
            print(
                f"SUBMITTED {row['name']} {job.id} {job.backend_job_id}",
                flush=True,
            )
        except Exception as exc:
            row["state"] = "ERROR"
            row["error"] = f"{type(exc).__name__}: {exc}"
            record_command(
                "mark",
                record_id,
                "--state",
                "ERROR",
                "--source",
                "backend-hook",
                "--err-class",
                "ERR_PROGRAM",
                "--message",
                f"submission failed for {row['name']}",
            )
            write_json(manifest_path, public(jobs))
            raise
        write_json(manifest_path, public(jobs))

    terminal = {"COMPLETE", "ERROR", "CANCELED"}
    while any(row.get("state") not in terminal for row in jobs):
        for row in jobs:
            if row.get("state") in terminal:
                continue
            state = sdk.get_job_status(row["job_id"]).status.upper()
            if state == "CANCELLED":
                state = "CANCELED"
            if state == "COMPLETE" and not valid_terminal_artifacts(row):
                state = "ERROR"
                row["error"] = "backend completed without valid result artifacts"
            row["state"] = state
            if state in terminal:
                mark = [
                    "mark",
                    row["record_id"],
                    "--state",
                    state,
                    "--source",
                    "backend-hook",
                    "--message",
                    f"terminal state for {row['name']}",
                ]
                if state == "ERROR":
                    mark += ["--err-class", "ERR_PROGRAM"]
                record_command(*mark)
        write_json(status_path, public(jobs))
        counts = {state: sum(row.get("state") == state for row in jobs) for state in terminal}
        running = len(jobs) - sum(counts.values())
        print(
            f"CROSS_FUSION running={running} complete={counts['COMPLETE']} "
            f"error={counts['ERROR']} canceled={counts['CANCELED']}",
            flush=True,
        )
        if running:
            time.sleep(30)
    print("CROSS_FUSION_TERMINAL", flush=True)


if __name__ == "__main__":
    main()
