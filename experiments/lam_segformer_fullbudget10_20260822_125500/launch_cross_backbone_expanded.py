#!/usr/bin/env python3
"""Launch fully parallel cross-backbone fusion over expanded source pools."""

from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path

from tao_sdk.platforms.slurm import SlurmSDK

from launch_expanded_fusions import (
    BACKBONES,
    EXPECTED_PARTITIONS,
    HERE,
    IMAGE,
    LOCAL_ROOT,
    REMOTE_CONTROLLER,
    REMOTE_MANIFESTS,
    REMOTE_OUTPUTS,
    record,
    scp,
    ssh,
    valid,
    write_json,
)


SCHEMES = (
    "probability",
    "geometric_probability",
    "raw_logit",
    "class_rank_probability_tiebreak",
    "hard_vote_probability_tiebreak",
)
CROSS_ROOT = REMOTE_OUTPUTS / "cross_backbone_expanded"


def remote_json(path: Path) -> dict:
    script = "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))))"
    return json.loads(ssh("python3", "-c", script, str(path)))


def one_hot_index(weights: list[float]) -> int | None:
    nonzero = [index for index, value in enumerate(weights) if abs(value) > 1.0e-10]
    if len(nonzero) == 1 and abs(weights[nonzero[0]] - 1.0) < 1.0e-8:
        return nonzero[0]
    return None


def outer_candidates(inner: dict[str, list[float]]) -> list[dict]:
    rows = []
    seen = set()

    def add(name: str, outer: list[float], source: str) -> None:
        flat = []
        for backbone, weight in zip(BACKBONES, outer):
            flat.extend(weight * value for value in inner[backbone])
        total = sum(flat)
        flat = [value / total for value in flat]
        key = tuple(round(value, 12) for value in flat)
        if key not in seen:
            seen.add(key)
            rows.append({"name": name, "weights": flat, "sources": [source]})

    for left, right in itertools.combinations(range(3), 2):
        for tenth in range(11):
            outer = [0.0, 0.0, 0.0]
            outer[left] = tenth / 10.0
            outer[right] = 1.0 - outer[left]
            add(f"backbone_pair_{left}_{right}_{tenth}", outer, "backbone_pair_tenths")
    for left in range(5):
        for middle in range(5 - left):
            right = 4 - left - middle
            add(
                f"backbone_simplex_{left}_{middle}_{right}",
                [left / 4.0, middle / 4.0, right / 4.0],
                "backbone_quarter_simplex",
            )
    add("near_uniform_backbones", [0.34, 0.33, 0.33], "near_uniform_backbones")
    # Separate control: equal mass per model, not per backbone or inner winner.
    uniform = [1.0 / 27.0] * 27
    key = tuple(round(value, 12) for value in uniform)
    if key not in seen:
        rows.append({"name": "uniform_27_models", "weights": uniform, "sources": ["uniform_all_models"]})
    return rows


def main() -> None:
    required = ("SLURM_USER", "SLURM_HOSTNAME", "SLURM_ACCOUNT", "NGC_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"unset required variables: {missing}")
    partitions = {value.strip() for value in os.environ.get("SLURM_PARTITION", "").split(",") if value.strip()}
    if partitions != EXPECTED_PARTITIONS:
        raise RuntimeError("unexpected SLURM partition set")

    scorer = HERE / "score_prediction_fusions_expanded.py"
    scp(scorer, REMOTE_CONTROLLER / scorer.name)
    ssh("mkdir", "-p", str(CROSS_ROOT), str(REMOTE_MANIFESTS))
    model_groups = {
        backbone: json.loads((LOCAL_ROOT / "fusion9_manifests" / f"prediction_fusion_{backbone}.json").read_text())["models"]
        for backbone in BACKBONES
    }
    all_models = [row for backbone in BACKBONES for row in model_groups[backbone]]
    jobs = []

    for scheme in SCHEMES:
        per_backbone = {
            backbone: remote_json(
                REMOTE_OUTPUTS / "prediction_fusion" / scheme / backbone / "results.json"
            )
            for backbone in BACKBONES
        }
        inner = {backbone: list(per_backbone[backbone]["best"]["weights"]) for backbone in BACKBONES}
        top_models = []
        for backbone in BACKBONES:
            singles = []
            for row in per_backbone[backbone]["results"]:
                index = one_hot_index(row["weights"])
                if index is not None:
                    singles.append((float(row["miou"]), index))
            if len(singles) != 9:
                raise RuntimeError(f"{scheme}/{backbone}: expected nine single controls")
            top_models.append(model_groups[backbone][max(singles)[1]])

        common = {
            "schema_version": 1,
            "dataset_root": "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/lam_research",
            "selection_split": "val",
            "expected_samples": 1262,
            "num_classes": 4,
            "test_used_for_selection": False,
        }
        payloads = {
            "cross3": {
                **common,
                "experiment": f"cross3_top_checkpoints_{scheme}",
                "models": top_models,
            },
            "cross27": {
                **common,
                "experiment": f"cross27_hierarchical_{scheme}",
                "models": all_models,
                "weight_candidates": outer_candidates(inner),
            },
        }
        for kind, payload in payloads.items():
            local = LOCAL_ROOT / "fusion9_manifests" / f"{kind}_{scheme}.json"
            remote = REMOTE_MANIFESTS / local.name
            write_json(local, payload)
            scp(local, remote)
            output = CROSS_ROOT / kind / scheme
            command = (
                f"torchrun --standalone --nproc_per_node=8 {REMOTE_CONTROLLER}/{scorer.name} "
                f"--manifest {remote} --scheme {scheme} --output {output}/results.json"
            )
            jobs.append(
                {
                    "name": f"{kind}_{scheme}",
                    "kind": "prediction_fusion",
                    "scheme": scheme,
                    "model_count": len(payload["models"]),
                    "results_dir": str(output),
                    "command": command,
                    "state": "PENDING",
                }
            )
    if len(jobs) != 10:
        raise RuntimeError(f"expected ten cross-backbone jobs, got {len(jobs)}")
    ssh(
        "python3", "-c",
        "import json,os,sys; [os.makedirs(x['results_dir'],exist_ok=True) for x in json.loads(sys.argv[1])]",
        json.dumps(jobs),
    )

    manifest = LOCAL_ROOT / "cross_backbone_expanded_manifest.json"
    status = LOCAL_ROOT / "cross_backbone_expanded_status.json"
    write_json(manifest, jobs)
    os.environ["TAO_SDK_STATE_DIR"] = str(LOCAL_ROOT / "sdk_state/cross_backbone_expanded")
    os.environ["SLURM_SQSH_CACHE_DIR"] = (
        "/lustre/fsw/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam"
    )
    sdk = SlurmSDK(poll_interval=30, epoch_milestone_interval=5)
    for row in jobs:
        rid = record(
            "open", "--platform", "slurm", "--image", IMAGE,
            "--network-arch", "segformer", "--action", "evaluate",
            "--storage-tier", "A", "--results-dir", row["results_dir"],
        )
        row["record_id"] = rid
        write_json(manifest, jobs)
        try:
            job = sdk.create_job(
                image=IMAGE,
                command=row["command"],
                gpu_count=8,
                num_nodes=1,
                account=os.environ.get("SLURM_ACCOUNT") or None,
                env_vars={"PYTHONPATH": "/usr/local/lib/python3.12/dist-packages"},
            )
            row.update({"job_id": job.id, "backend_ref": job.backend_job_id, "state": "RUNNING"})
            record(
                "mark", rid, "--state", "RUNNING", "--source", "backend-hook",
                "--backend-ref", job.backend_job_id,
                "--message", f"submitted {row['name']} with 8 GPUs",
            )
            print(f"SUBMITTED {row['name']} {job.id} {job.backend_job_id}", flush=True)
        except Exception as exc:
            row.update({"state": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
            record(
                "mark", rid, "--state", "ERROR", "--source", "backend-hook",
                "--err-class", "ERR_PROGRAM", "--message", f"submission failed for {row['name']}",
            )
            write_json(manifest, jobs)
            raise
        write_json(manifest, jobs)

    terminal = {"COMPLETE", "ERROR", "CANCELED"}
    while any(row["state"] not in terminal for row in jobs):
        for row in jobs:
            if row["state"] in terminal:
                continue
            state = sdk.get_job_status(row["job_id"]).status.upper()
            if state == "CANCELLED":
                state = "CANCELED"
            if state == "COMPLETE" and not valid(row):
                row["error"] = "backend completed without valid result artifact"
                state = "ERROR"
            row["state"] = state
            if state in terminal:
                args = ["mark", row["record_id"], "--state", state, "--source", "backend-hook", "--message", f"terminal {row['name']}"]
                if state == "ERROR":
                    args.extend(["--err-class", "ERR_PROGRAM"])
                record(*args)
        write_json(status, jobs)
        counts = {state: sum(row["state"] == state for row in jobs) for state in terminal}
        print(
            f"CROSS_EXPANDED running={len(jobs)-sum(counts.values())} "
            f"complete={counts['COMPLETE']} error={counts['ERROR']} canceled={counts['CANCELED']}",
            flush=True,
        )
        if any(row["state"] not in terminal for row in jobs):
            time.sleep(30)


if __name__ == "__main__":
    main()
