# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render + security tests for the single-node SLURM sbatch template.

A vendored launch template exists so the exact #SBATCH directives, the
sidecar-cred security control, and `--requeue` are faithful rather than authored
freehand per run. These tests assert a rendered instance is valid bash, leaves
no unsubstituted markers, never inlines a secret, and carries the load-bearing
directives.
"""

import re
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "templates/slurm/singlenode.sbatch.tmpl"
IAA_CLIP_TRAIN_WRAPPER = (
    REPO
    / "skills"
    / "applications"
    / "tao-run-deft-iaa"
    / "patches"
    / "run_clip_train_slurm.sh"
)
sys.path.insert(0, str(REPO / "scripts"))
import redact_secrets  # noqa: E402

BASE = {
    "JOB_NAME": "dino-train-a1b2c3",
    "NUM_GPUS": "1",
    "CPUS_PER_TASK": "16",
    "TIME": "04:00:00",
    "LOG_DIR": "/lustre/fsw/portfolios/edgeai/users/me/results/dino-train-a1b2c3/slurm-logs",
    "SBATCH_EXTRA": "#SBATCH --account=edgeai\n#SBATCH --partition=polar,polar3",
    "REQUEUE_DIRECTIVE": "#SBATCH --requeue",
    "ENV_FILE": "",
    "EXTRA_ENV": "",
    "IMAGE": "/lustre/fsw/sqsh/tao-toolkit-6.26.3-pyt.sqsh",
    "CONTAINER_MOUNTS": "/lustre",
    "COMMAND": "dino train -e /lustre/fsw/.../specs/dino-train-a1b2c3/spec.yaml",
}


def render(overrides=None):
    values = {**BASE, **(overrides or {})}
    text = TEMPLATE.read_text()
    for key, val in values.items():
        text = text.replace(f"@@{key}@@", val)
    return text


def bash_syntax_ok(text):
    return subprocess.run(["bash", "-n", "/dev/stdin"], input=text,
                          text=True, capture_output=True).returncode == 0


def test_all_markers_substituted():
    rendered = render()
    leftover = re.findall(r"@@[A-Z_]+@@", rendered)
    assert leftover == [], f"unsubstituted markers: {leftover}"


def test_rendered_is_valid_bash():
    assert bash_syntax_ok(render())


def test_env_file_present_case_is_valid_bash():
    rendered = render({"ENV_FILE": "/lustre/.../job_dino-train-a1b2c3.env"})
    assert bash_syntax_ok(rendered)
    assert "trap 'shred -u" in rendered            # sidecar shredded on exit
    assert "source \"/lustre" in rendered


def test_load_bearing_directives_present():
    rendered = render()
    for needle in ("#SBATCH --requeue", "#SBATCH --nodes=1",
                   "#SBATCH --gres=gpu:1", "srun --container-image=",
                   "--container-mounts=/lustre", "#SBATCH --account=edgeai"):
        assert needle in rendered, f"missing: {needle}"


def test_logs_use_precreated_directory_without_uncreatable_job_subdirectory():
    rendered = render()
    assert "#SBATCH --output=" in rendered and "/%x-%j.out" in rendered
    assert "#SBATCH --error=" in rendered and "/%x-%j.err" in rendered
    assert "/%x-%j/" not in rendered


def test_requeue_can_be_disabled_without_freehand_template_edits():
    rendered = render({"REQUEUE_DIRECTIVE": "#SBATCH --no-requeue"})
    assert "#SBATCH --no-requeue" in rendered
    assert "#SBATCH --requeue" not in rendered
    assert bash_syntax_ok(rendered)


def test_no_cross_node_rendezvous_in_singlenode():
    # single-node must NOT set the multi-node rendezvous env (that is M7)
    rendered = render()
    assert "MASTER_ADDR" not in rendered
    assert "WORLD_SIZE" not in rendered


def test_secrets_never_inlined_lints_clean():
    # the sidecar pattern means the rendered script carries no literal creds,
    # even if the agent (wrongly) put a secret-shaped value in EXTRA_ENV we catch it
    rendered = render()
    assert rendered.count("export ") >= 1                 # NCCL_DEBUG etc. are fine
    assert redact_secrets.scan(rendered) == []            # no literal credential

    # a deliberately bad render (inline secret) MUST be caught by the lint gate
    bad = render({"EXTRA_ENV": "export NGC_KEY=nvapi-LEAKEDsecret1234567890"})
    assert redact_secrets.scan(bad), "lint must flag an inlined credential"


def test_extra_env_nccl_knob_renders():
    rendered = render({"EXTRA_ENV": "export NCCL_P2P_DISABLE=1"})
    assert bash_syntax_ok(rendered)
    assert "NCCL_P2P_DISABLE=1" in rendered


def test_pyxis_passes_only_fixed_nonsecret_nccl_runtime_names():
    rendered = render(
        {
            "EXTRA_ENV": (
                "export NCCL_P2P_DISABLE=1\n"
                "export NCCL_IB_DISABLE=1\n"
                "export NCCL_SOCKET_IFNAME=eth0\n"
                "export NCCL_NET=Socket\n"
                "export UNAPPROVED_VALUE=must-stay-host-only"
            )
        }
    )
    expected = (
        "--container-env=NCCL_DEBUG,LOGLEVEL,NCCL_P2P_DISABLE,NCCL_IB_DISABLE,"
        "NCCL_SOCKET_IFNAME,NCCL_IB_HCA,NCCL_NET"
    )
    assert expected in rendered
    container_names = rendered.split("--container-env=", 1)[1].split()[0].split(",")
    assert container_names == [
        "NCCL_DEBUG",
        "LOGLEVEL",
        "NCCL_P2P_DISABLE",
        "NCCL_IB_DISABLE",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_HCA",
        "NCCL_NET",
    ]
    assert "UNAPPROVED_VALUE" not in container_names
    assert "NGC_KEY" not in container_names


def test_rendered_srun_exposes_approved_nccl_values_to_container_boundary(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "capture.json"
    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "arg = next(x for x in sys.argv if x.startswith('--container-env='))\n"
        "names = arg.split('=', 1)[1].split(',')\n"
        "json.dump({'names': names, 'values': {n: os.environ.get(n) for n in names}, "
        "'unapproved': os.environ.get('UNAPPROVED_VALUE')}, open(os.environ['CAPTURE'], 'w'))\n",
        encoding="utf-8",
    )
    fake_srun.chmod(0o755)
    rendered = render(
        {
            "EXTRA_ENV": (
                "export NCCL_P2P_DISABLE=1\n"
                "export NCCL_IB_DISABLE=1\n"
                "export NCCL_SOCKET_IFNAME=eth0\n"
                "export NCCL_NET=Socket\n"
                "export UNAPPROVED_VALUE=must-stay-host-only"
            ),
            "COMMAND": "true",
        }
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CAPTURE": str(capture),
    }
    subprocess.run(["bash"], input=rendered, text=True, env=env, check=True)
    evidence = json.loads(capture.read_text())
    assert evidence["values"]["NCCL_P2P_DISABLE"] == "1"
    assert evidence["values"]["NCCL_IB_DISABLE"] == "1"
    assert evidence["values"]["NCCL_SOCKET_IFNAME"] == "eth0"
    assert evidence["values"]["NCCL_NET"] == "Socket"
    assert "UNAPPROVED_VALUE" not in evidence["names"]


def test_iaa_clip_train_wrapper_removes_only_lightning_topology(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "capture.json"
    fake_clip = fake_bin / "clip"
    fake_clip.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "keys = ['SLURM_NTASKS', 'SLURM_NTASKS_PER_NODE', 'SLURM_PROCID', "
        "'SLURM_LOCALID', 'SLURM_NODEID', 'WORLD_SIZE', 'RANK', 'LOCAL_RANK', "
        "'NODE_RANK', 'MASTER_ADDR', 'MASTER_PORT', 'NUM_GPU_PER_NODE', "
        "'SLURM_JOB_ID', 'SLURM_JOB_ACCOUNT', 'CUDA_VISIBLE_DEVICES']\n"
        "json.dump({'argv': sys.argv, 'env': {k: os.environ.get(k) for k in keys}}, "
        "open(os.environ['CAPTURE'], 'w'))\n",
        encoding="utf-8",
    )
    fake_clip.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CAPTURE": str(capture),
        "SLURM_NTASKS": "1",
        "SLURM_NTASKS_PER_NODE": "1",
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
        "SLURM_NODEID": "0",
        "WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "NODE_RANK": "0",
        "MASTER_ADDR": "stale-master",
        "MASTER_PORT": "29500",
        "NUM_GPU_PER_NODE": "2",
        "SLURM_JOB_ID": "12345",
        "SLURM_JOB_ACCOUNT": "approved-account",
        "CUDA_VISIBLE_DEVICES": "0,1",
    }
    subprocess.run(
        [str(IAA_CLIP_TRAIN_WRAPPER), "clip", "train", "-e", "/results/spec.yaml"],
        env=env,
        check=True,
    )
    evidence = json.loads(capture.read_text())
    assert evidence["argv"] == [str(fake_clip), "train", "-e", "/results/spec.yaml"]
    for key in (
        "SLURM_NTASKS", "SLURM_NTASKS_PER_NODE", "SLURM_PROCID", "SLURM_LOCALID",
        "SLURM_NODEID", "WORLD_SIZE", "RANK", "LOCAL_RANK", "NODE_RANK",
        "MASTER_ADDR", "MASTER_PORT", "NUM_GPU_PER_NODE",
    ):
        assert evidence["env"][key] is None
    assert evidence["env"]["SLURM_JOB_ID"] == "12345"
    assert evidence["env"]["SLURM_JOB_ACCOUNT"] == "approved-account"
    assert evidence["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_iaa_clip_train_wrapper_rejects_non_train_actions():
    rejected = subprocess.run(
        [str(IAA_CLIP_TRAIN_WRAPPER), "clip", "evaluate"], capture_output=True, text=True
    )
    assert rejected.returncode == 64
