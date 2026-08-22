#!/usr/bin/env python3
"""Launch the required bounded single-node/eight-GPU NCCL preflight."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(
    "/localhome/local-rarunachalam/workspace/"
    "lam_segformer_bayes_deft_20260820_231724"
)
REMOTE_ROOT = Path(
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
    "lam_segformer_bayes_deft_20260820_231724"
)
BANK = ROOT / "skill_bank_snapshot"
IMAGE = REMOTE_ROOT / "inputs/images/nvcr.io_nvidia_tao_tao-toolkit_7.1.0-pyt.sqsh"


def run(*args: str, capture: bool = False) -> str:
    result = subprocess.run(
        list(args),
        check=True,
        text=True,
        capture_output=capture,
    )
    return result.stdout.strip() if capture else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2p-disable", action="store_true")
    args = parser.parse_args()

    user = os.environ["SLURM_USER"]
    hostname = os.environ["SLURM_HOSTNAME"].split(",", 1)[0]
    login = f"{user}@{hostname}"
    account = os.environ["SLURM_ACCOUNT"]
    partitions = os.environ.get(
        "SLURM_PARTITION", "polar,polar3,polar4,grizzly"
    )
    record_script = BANK / "scripts/tao_job_record.py"
    job_id = run(
        sys.executable,
        str(record_script),
        "open",
        "--platform",
        "slurm",
        "--image",
        str(IMAGE),
        "--network-arch",
        "segformer",
        "--action",
        "nccl_probe",
        "--storage-tier",
        "A",
        "--results-root",
        str(REMOTE_ROOT / "probes"),
        capture=True,
    )

    local_sbatch_dir = ROOT / "probes/sbatch"
    local_sbatch_dir.mkdir(parents=True, exist_ok=True)
    rendered_path = local_sbatch_dir / f"job_{job_id}.sbatch"
    remote_sbatch = REMOTE_ROOT / f"probes/sbatch/job_{job_id}.sbatch"
    log_dir = REMOTE_ROOT / "probes/slurm-logs"
    probe = REMOTE_ROOT / "controller/nccl_allreduce_probe.py"
    model_probe = REMOTE_ROOT / "controller/model_load_probe.py"
    full_probe = REMOTE_ROOT / "controller/run_full_probe.sh"
    command = f"bash {full_probe}"
    replacements = {
        "JOB_NAME": job_id,
        "NUM_GPUS": "8",
        "CPUS_PER_TASK": "16",
        "TIME": "00:10:00",
        "LOG_DIR": str(log_dir),
        "REQUEUE_DIRECTIVE": "#SBATCH --no-requeue",
        "SBATCH_EXTRA": (
            f"#SBATCH --partition={partitions}\n#SBATCH --account={account}"
        ),
        "ENV_FILE": "",
        "EXTRA_ENV": (
            "export NCCL_P2P_DISABLE=1" if args.p2p_disable else ":"
        ),
        "IMAGE": str(IMAGE),
        "CONTAINER_MOUNTS": "/lustre:/lustre",
        "COMMAND": command,
    }
    text = (BANK / "templates/slurm/singlenode.sbatch.tmpl").read_text()
    for key, value in replacements.items():
        text = text.replace(f"@@{key}@@", value)
    if re.search(r"@@[A-Z0-9_]+@@", text):
        raise RuntimeError("unresolved template placeholder")
    rendered_path.write_text(text)

    ssh = ["ssh", "-o", "BatchMode=yes"]
    if os.environ.get("SSH_KEY_PATH"):
        ssh += ["-i", os.environ["SSH_KEY_PATH"]]
    run(*ssh, login, f"mkdir -p {REMOTE_ROOT}/probes/sbatch {log_dir}")
    submit = run(
        sys.executable,
        str(
            BANK
            / "skills/platform/tao-run-on-slurm/scripts/slurm_submit_action.py"
        ),
        "--login",
        login,
        "--job-id",
        job_id,
        "--rendered-script",
        str(rendered_path),
        "--remote-script",
        str(remote_sbatch),
        capture=True,
    )
    payload = json.loads(submit)
    backend_ref = str(payload["backend_ref"])
    run(
        sys.executable,
        str(record_script),
        "mark",
        job_id,
        "--state",
        "RUNNING",
        "--source",
        "agent",
        "--backend-ref",
        backend_ref,
        "--message",
        "bounded 8-GPU NCCL probe submitted",
    )
    output = {
        "job_id": job_id,
        "backend_ref": backend_ref,
        "p2p_disabled": args.p2p_disable,
        "log_dir": str(log_dir),
    }
    (ROOT / "probes").mkdir(parents=True, exist_ok=True)
    (ROOT / "probes/nccl_submission.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(output))


if __name__ == "__main__":
    main()
