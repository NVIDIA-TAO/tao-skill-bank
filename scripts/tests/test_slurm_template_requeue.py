#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wall-time self-requeue and auto-resume, at the TEMPLATE layer.

test_slurm_walltime_requeue.py covers the same behaviour through our
renderer. This one renders the shipped .tmpl directly, so it still holds if
the rendering is done by a different renderer or by the agent by hand --
which is a live possibility while a parallel action-request renderer exists.

Ported from tao-sdk. `#SBATCH --requeue` covers NODE_FAIL and pre-emption only;
a job that hits its TimeLimit ends in TIMEOUT and is NOT requeued. The SDK
wrapped srun in `timeout` set below the wall limit and requeued on exit 124, so
the job never reaches TIMEOUT, and pointed empty resume keys at a checkpoint
from a previous attempt, "so a resumable job would otherwise restart from epoch
0" (tao_sdk/script_runner.py::_apply_auto_resume).

Both templates carried the passive halves (`--requeue`, `--open-mode=append`)
without the active ones, and SLURM_TIMEOUT_HOURS stayed documented and
validated in preflight while nothing consumed it.

These render the templates directly rather than through any renderer, so they
hold the shipped artifact regardless of who does the substituting.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
TEMPLATES = {
    "singlenode": REPO / "templates/slurm/singlenode.sbatch.tmpl",
    "multinode": REPO / "templates/slurm/multinode.sbatch.tmpl",
}
RESUME_KEY = "train.resume_training_checkpoint_path"


def _render(template: pathlib.Path, **overrides) -> str:
    text = template.read_text(encoding="utf-8")
    values = {
        "TIMEOUT_MINUTES": "228", "RESUME_KEY": RESUME_KEY,
        "RESULTS_DIR": "/lustre/results/job-1",
    }
    values.update(overrides)
    for key, value in values.items():
        text = text.replace(f"@@{key}@@", value)
    return re.sub(r"@@[A-Z_]+@@", "x", text)


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_requeues_on_wall_time(name):
    body = TEMPLATES[name].read_text(encoding="utf-8")
    assert "scontrol requeue" in body, "a wall-time kill would be terminal"
    assert 'timeout "@@TIMEOUT_MINUTES@@m"' in body


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_exit_code_survives_set_e(name):
    """`set -e` aborts before the check without `|| rc=$?`.

    Copying the SDK's bare `if [[ $? == 124 ]]` into a `set -euo pipefail`
    script makes the requeue unreachable.
    """
    body = TEMPLATES[name].read_text(encoding="utf-8")
    assert "|| rc=$?" in body
    assert body.index("|| rc=$?") < body.index("scontrol requeue")


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_requeue_pairs_with_append_logging(name):
    """Without append mode a requeued attempt truncates the earlier logs."""
    body = TEMPLATES[name].read_text(encoding="utf-8")
    assert "--open-mode=append" in body and "#SBATCH --requeue" in body


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_renders_to_valid_bash(name, tmp_path):
    script = tmp_path / f"{name}.sbatch"
    script.write_text(_render(TEMPLATES[name]))
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_no_gnu_only_predicates(name):
    """stderr is discarded in the resume block, so a GNU-only predicate would
    fail silently on a BSD find and disable auto-resume without a word."""
    code = "\n".join(
        line for line in TEMPLATES[name].read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "-printf" not in code
    assert "-print0" in code and "xargs -0" in code


# ── Executed behaviour ──────────────────────────────────────────────────────
# Text matching passed while three separate bugs were live (GNU-only -printf, a
# grep that did not filter, word-splitting that zsh does not do), so these run
# the block under bash and assert what it actually selects.

def _run_resume_block(results_dir, env=None):
    import os

    body = _render(TEMPLATES["singlenode"], RESULTS_DIR=str(results_dir))
    block = re.search(r'^RESUME_OVERRIDE=""$.*?^fi$', body, re.S | re.M).group(0)
    out = subprocess.run(
        ["bash", "-c", f'{block}\necho "${{RESUME_OVERRIDE:-}}"'],
        capture_output=True, text=True, env={**os.environ, **(env or {})},
    )
    lines = out.stdout.strip().splitlines()
    return lines[-1] if lines else ""


def _mk(tmp_path, *names):
    root = tmp_path / "results"
    (root / "ptm").mkdir(parents=True)
    (root / "train").mkdir(parents=True)
    for name in names:
        target = (root / "ptm" / name[4:]) if name.startswith("ptm/") else (root / "train" / name)
        target.write_text("x")
        time.sleep(0.02)
    return root


def test_prefers_epoch_checkpoint_over_best_and_latest(tmp_path):
    """best/latest are usually written LAST, so newest-by-time alone resumes
    from a metric snapshot and loses optimizer/scheduler position."""
    root = _mk(tmp_path, "model_epoch_007.pth", "best_model.pth", "latest.ckpt")
    assert _run_resume_block(root).endswith("model_epoch_007.pth")


def test_falls_back_to_best_when_nothing_else_exists(tmp_path):
    root = _mk(tmp_path, "best_model.pth")
    assert _run_resume_block(root).endswith("best_model.pth")


def test_never_resumes_from_a_staged_ptm(tmp_path):
    """The failure this prevents: resuming from a pretrained backbone."""
    root = _mk(tmp_path, "ptm/pretrained.pth")
    assert _run_resume_block(root) == ""


def test_ignores_zero_byte_checkpoints(tmp_path):
    root = tmp_path / "results"
    (root / "train").mkdir(parents=True)
    (root / "train" / "model_epoch_1.pth").write_text("")
    assert _run_resume_block(root) == ""


def test_disabled_by_env(tmp_path):
    """TAO_AUTO_RESUME=0, the switch the SDK exposed."""
    root = _mk(tmp_path, "model_epoch_007.pth")
    assert _run_resume_block(root, {"TAO_AUTO_RESUME": "0"}) == ""


def test_resume_is_off_without_a_key():
    """The key is network-specific; a workflow that cannot resume omits it."""
    body = _render(TEMPLATES["singlenode"], RESUME_KEY="")
    assert 'if [ -n "" ]' in body


# ── The mount source must exist before the mount ────────────────────────────
# Found end-to-end. The job scheduled, ran, and died inside pyxis:
#
#   enroot-mount: failed to mount: /lustre/.../results/<job> at /raid/...:
#   No such file or directory
#
# results_dir is bound into the record BEFORE launch, but nothing created it on
# the cluster. Docker's -v auto-creates a missing source, so every docker run
# hid this; Pyxis/Enroot refuses and fails the whole container start, naming
# the mount rather than the missing directory.

def _rendered(tmp_path, **over):
    """Render a template with real paths so it can be EXECUTED, not grepped."""
    import pathlib as _p
    template = _p.Path(__file__).resolve().parents[2] / "templates/slurm/singlenode.sbatch.tmpl"
    body = template.read_text(encoding="utf-8")
    values = {
        "RESULTS_DIR": str(tmp_path / "results" / "job-1"),
        "RESUME_KEY": "", "TIMEOUT_MINUTES": "1", "IMAGE": "img.sqsh",
        "CONTAINER_MOUNTS": "/a:/a", "COMMAND": "true", "JOB_NAME": "j",
        "NUM_GPUS": "0", "CPUS_PER_TASK": "1", "TIME": "00:01:00",
        "LOG_DIR": str(tmp_path / "logs"), "SBATCH_EXTRA": "",
        "ENV_FILE": "", "NUM_NODES": "1",
    }
    values.update(over)
    for key, value in values.items():
        body = body.replace(f"@@{key}@@", value)
    # Blank any placeholder this fixture does not name. The renderer always
    # substitutes every one, and hard-coding the list here means a template
    # gaining a placeholder breaks these tests for an unrelated reason.
    return re.sub(r"@@[A-Z_]+@@", "", body)


def test_results_dir_is_created_before_the_container_mount(tmp_path):
    """Execute the prologue: the directory must exist by the time srun runs."""
    import re
    import subprocess

    body = _rendered(tmp_path)
    prologue = body[: body.index("timeout ")]
    # Drop #SBATCH directives and the trap/env lines that need a live job.
    script = "\n".join(
        line for line in prologue.splitlines()
        if not line.startswith("#SBATCH") and "scontrol" not in line
    )
    subprocess.run(["bash", "-c", script], check=True, capture_output=True,
                   text=True, timeout=30)
    assert (tmp_path / "results" / "job-1").is_dir(), (
        "the prologue did not create results_dir; pyxis will refuse to mount "
        "it and fail the container start"
    )


def test_results_dir_creation_is_not_conditional_on_auto_resume(tmp_path):
    """The mount happens either way, so the mkdir cannot sit inside that if.

    It did, on the first attempt at this fix: the comment-block walkback landed
    it after `if [ -n "@@RESUME_KEY@@" ]`, so a bundle with no resume key --
    every CPU-only glue stage -- would still have failed to mount.
    """
    import subprocess

    body = _rendered(tmp_path, RESUME_KEY="")
    prologue = body[: body.index("timeout ")]
    script = "\n".join(
        line for line in prologue.splitlines()
        if not line.startswith("#SBATCH") and "scontrol" not in line
    )
    subprocess.run(["bash", "-c", script], check=True, capture_output=True,
                   text=True, timeout=30)
    assert (tmp_path / "results" / "job-1").is_dir(), (
        "results_dir is only created when auto-resume is on"
    )
