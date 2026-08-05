# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render + rendezvous tests for the multi-node templates (SLURM + k8s).

The load-bearing invariant everywhere: WORLD_SIZE is the NODE COUNT (TAO's
misnamed convention), never the GPU count — every case renders with nodes != gpus
and asserts WORLD_SIZE == nodes. A subtly-wrong rendezvous hangs silently for the
whole distributed timeout, so these are vendored, tested templates, not freehand.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import redact_secrets  # noqa: E402

SLURM = REPO / "templates/slurm/multinode.sbatch.tmpl"
K8S = REPO / "templates/k8s/indexed-job.yaml.tmpl"

# nodes (2) deliberately != gpus-per-node (8) so WORLD_SIZE=2 can't be confused with GPUs
SLURM_VALS = {
    "JOB_NAME": "dino-train-a1b2c3", "NUM_NODES": "2", "GPUS_PER_NODE": "8",
    "CPUS_PER_TASK": "16", "TIME": "04:00:00",
    "LOG_DIR": "/lustre/fsw/users/me/results/dino-train-a1b2c3/slurm-logs",
    "SBATCH_EXTRA": "#SBATCH --account=edgeai\n#SBATCH --partition=batch",
    "ENV_FILE": "", "EXTRA_ENV": "export NCCL_P2P_DISABLE=1",
    "IMAGE": "/lustre/sqsh/tao.sqsh", "CONTAINER_MOUNTS": "/lustre",
    "COMMAND": "dino train -e /lustre/specs/spec.yaml",
}
K8S_VALS = {
    "JOB_NAME": "dino-train-a1b2c3", "NUM_NODES": "2", "GPUS_PER_NODE": "8",
    "TTL_SECONDS": "3600", "IMAGE_PULL_SECRET": "ngc-pull-secret",
    "IMAGE": "nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt",  # unpinned: test fixture
    "CRED_SECRET": "tao-creds-dino-train-a1b2c3", "RESULTS_DIR": "/data/results/dino-train-a1b2c3",
    "MOUNT_PATH": "/data", "SHM_SIZE": "16Gi", "PVC_CLAIM": "edgeai-datasets",
    "COMMAND": "dino train -e /data/specs/spec.yaml",
}


def render(tmpl, vals):
    text = tmpl.read_text()
    for k, v in vals.items():
        text = text.replace(f"@@{k}@@", v)
    return text


# --------------------------------------------------------------------------- #
# SLURM multi-node
# --------------------------------------------------------------------------- #

def test_slurm_all_markers_substituted_and_valid_bash():
    t = render(SLURM, SLURM_VALS)
    assert re.findall(r"@@[A-Z_]+@@", t) == []
    assert subprocess.run(["bash", "-n", "/dev/stdin"], input=t, text=True, capture_output=True).returncode == 0


def test_slurm_world_size_is_node_count_not_gpus():
    t = render(SLURM, SLURM_VALS)
    assert "export WORLD_SIZE=2" in t           # nodes, not 8 GPUs
    assert "export WORLD_SIZE=8" not in t
    assert "export NUM_GPU_PER_NODE=8" in t


def test_slurm_rendezvous_and_directives():
    t = render(SLURM, SLURM_VALS)
    for needle in ("#SBATCH --nodes=2", "#SBATCH --wait-all-nodes=1", "#SBATCH --gres=gpu:8",
                   "#SBATCH --requeue", "scontrol show hostname", "export NODE_RANK=$SLURM_NODEID",
                   "export MASTER_PORT=29500", "export NCCL_P2P_DISABLE=1"):
        assert needle in t, f"missing: {needle}"


def test_slurm_lints_clean_and_uses_sidecar():
    t = render(SLURM, SLURM_VALS)
    assert redact_secrets.scan(t) == []
    assert "trap 'shred -u" in t


# --------------------------------------------------------------------------- #
# k8s Indexed Job + headless Service
# --------------------------------------------------------------------------- #

def k8s_docs():
    return list(yaml.safe_load_all(render(K8S, K8S_VALS)))


def test_k8s_all_markers_substituted():
    assert re.findall(r"@@[A-Z_]+@@", render(K8S, K8S_VALS)) == []


def test_k8s_two_docs_service_then_job():
    svc, job = k8s_docs()
    assert svc["kind"] == "Service" and svc["spec"]["clusterIP"] == "None"     # headless
    assert svc["spec"]["selector"]["job-name"] == "dino-train-a1b2c3"
    assert job["kind"] == "Job"


def test_k8s_indexed_completions_parallelism_subdomain():
    _, job = k8s_docs()
    assert job["spec"]["completionMode"] == "Indexed"
    assert job["spec"]["completions"] == 2 and job["spec"]["parallelism"] == 2   # = nodes
    assert job["spec"]["template"]["spec"]["subdomain"] == "dino-train-a1b2c3"


def test_k8s_world_size_node_count_and_master_addr():
    _, job = k8s_docs()
    env = {e["name"]: e["value"] for e in job["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["WORLD_SIZE"] == "2"                       # nodes, not the 8 GPUs
    assert env["NUM_GPU_PER_NODE"] == "8"
    assert env["MASTER_ADDR"] == "dino-train-a1b2c3-0.dino-train-a1b2c3"   # pod-0 . headless-svc
    assert env["MASTER_PORT"] == "29500"


def test_k8s_node_rank_from_completion_index():
    _, job = k8s_docs()
    cmd = job["spec"]["template"]["spec"]["containers"][0]["command"][2]
    assert "NODE_RANK=${JOB_COMPLETION_INDEX:-0}" in cmd


def test_k8s_gpu_limit_and_shm_and_secretref():
    _, job = k8s_docs()
    c = job["spec"]["template"]["spec"]["containers"][0]
    assert c["resources"]["limits"]["nvidia.com/gpu"] == "8"     # per node
    assert c["envFrom"][0]["secretRef"]["name"] == "tao-creds-dino-train-a1b2c3"
    vols = {v["name"]: v for v in job["spec"]["template"]["spec"]["volumes"]}
    assert vols["dshm"]["emptyDir"]["sizeLimit"] == "16Gi"


def test_k8s_no_inline_creds():
    _, job = k8s_docs()
    env = {e["name"] for e in job["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert not any(k in env for k in ("AWS_SECRET_ACCESS_KEY", "NGC_KEY", "HF_TOKEN"))
    assert redact_secrets.scan(render(K8S, K8S_VALS)) == []
