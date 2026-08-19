#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wall-time self-requeue and auto-resume, ported from tao-sdk.

`#SBATCH --requeue` covers NODE_FAIL and pre-emption only; a job that hits its
TimeLimit ends in TIMEOUT and is NOT requeued. The SDK handled this by wrapping
srun in `timeout` set below the wall limit and requeueing on exit 124, so the
job never reaches TIMEOUT — and by pointing empty resume keys at a checkpoint
left by a previous attempt, since a requeue re-runs the same script with the
spec frozen at submit time.

Both templates carried the passive halves (`--requeue`, `--open-mode=append`)
without the active ones, and `SLURM_TIMEOUT_HOURS` was still documented and
validated in preflight while nothing consumed it. These tests pin the ported
behaviour so the halves cannot drift apart again.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
TEMPLATES = [REPO / "templates/slurm/singlenode.sbatch.tmpl",
             REPO / "templates/slurm/multinode.sbatch.tmpl"]


@pytest.fixture(scope="module")
def render():
    path = REPO / "skills/platform/tao-run-on-slurm/references/render.py"
    spec = importlib.util.spec_from_file_location("slurm_render", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUNDLE = {
    "network_arch": "vcn", "action": "iter1.train",
    "image": "/lustre/img/tao.sqsh", "mode": "args",
    "command": "visual_changenet train", "args": ["-e", "/w/spec.yaml"],
    "declared_inputs": [{"spec_key": "d", "type": "folder", "uri": "/lustre/data"}],
    "declared_outputs": [{"spec_key": "o", "type": "folder"}],
    "compute_shape": {"gpus": 8, "nodes": 1},
}


def _ctx(**extra):
    base = {"job_id": "vcn-train-1", "results_dir": "/lustre/results/vcn-train-1",
            "bank": str(REPO), "login": "me@login", "sqsh_dir": "/lustre/img",
            "time_limit": "04:00:00"}
    base.update(extra)
    return base


def _body(render, **ctx):
    return list(render.render(BUNDLE, _ctx(**ctx))["files"].values())[0]


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.stem)
def test_template_requeues_on_timeout(template):
    body = template.read_text(encoding="utf-8")
    assert "scontrol requeue" in body, "no self-requeue: a wall-time kill is terminal"
    assert 'timeout "@@TIMEOUT_MINUTES@@m"' in body


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.stem)
def test_exit_code_survives_set_e(template):
    """`set -e` would abort before the 124 check without `|| rc=$?`.

    Copying the SDK's bare `if [[ $? == 124 ]]` into a `set -euo pipefail`
    script would make the requeue unreachable.
    """
    body = template.read_text(encoding="utf-8")
    assert "|| rc=$?" in body
    assert body.index("|| rc=$?") < body.index("scontrol requeue")


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.stem)
def test_requeue_pairs_with_append_logging(template):
    """Without append mode a requeued attempt truncates the previous logs."""
    body = template.read_text(encoding="utf-8")
    assert "--open-mode=append" in body and "#SBATCH --requeue" in body


def test_rendered_script_is_valid_bash(render, tmp_path):
    script = tmp_path / "job.sbatch"
    script.write_text(_body(render, resume_key="train.resume_training_checkpoint_path"))
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0


def test_timeout_is_under_the_wall_limit(render):
    body = _body(render)
    minutes = int(re.search(r'timeout "(\d+)m"', body).group(1))
    assert minutes < 240, "timeout must fire before SLURM's --time=04:00:00"


def test_timeout_over_the_wall_limit_is_refused(render):
    """Otherwise the job dies in TIMEOUT and is never requeued — silently."""
    with pytest.raises(ValueError, match="not under the wall limit"):
        render.render(BUNDLE, _ctx(timeout_minutes=240))


@pytest.mark.parametrize("limit,expected", [
    ("04:00:00", 240), ("1-00:00:00", 1440), ("7-00:00:00", 10080), ("00:31:00", 31),
])
def test_wall_limit_parsing(render, limit, expected):
    assert render._limit_minutes(limit) == expected


def test_auto_resume_is_off_without_a_resume_key(render):
    """The key is network-specific; a bundle that cannot resume omits it."""
    body = _body(render)
    assert 'if [ -n "" ]' in body.replace("@@RESUME_KEY@@", "")


def test_auto_resume_searches_only_the_job_results_dir(render):
    """A checkpoint may only come from a previous attempt of THIS job."""
    body = _body(render, resume_key="train.resume_training_checkpoint_path")
    assert '"/lustre/results/vcn-train-1"' in body


def test_auto_resume_prunes_staged_model_directories(render):
    """A staged PTM or downloaded input must never look like training state."""
    body = _body(render, resume_key="train.resume_training_checkpoint_path")
    assert "-name ptm" in body and "-name inputs" in body and "-prune" in body


def test_auto_resume_ignores_empty_checkpoints(render):
    body = _body(render, resume_key="train.resume_training_checkpoint_path")
    assert "-size +0c" in body


def test_auto_resume_avoids_gnu_only_printf(render):
    """stderr is discarded here, so a GNU-only predicate would fail silently
    on a BSD find and disable auto-resume without a word."""
    body = _body(render, resume_key="train.resume_training_checkpoint_path")
    code = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("#"))
    assert "-printf" not in code, "GNU-only -printf fails silently under 2>/dev/null"
    assert "-print0" in code and "xargs -0" in code


# ── Executed behaviour ──────────────────────────────────────────────────────
# Grepping the template proves the text; these run the block under bash and
# assert what it actually selects.

def _run_resume_block(render, results_dir, env=None):
    import os
    import re as _re
    # render() substitutes @@RESULTS_DIR@@, so point it at the fixture via ctx
    # rather than trying to patch the rendered text afterwards.
    body = _body(render, resume_key="train.resume_training_checkpoint_path",
                 results_dir=str(results_dir))
    block = _re.search(r'^RESUME_OVERRIDE=""$.*?^fi$', body, _re.S | _re.M).group(0)
    script = f'{block}\necho "${{RESUME_OVERRIDE:-}}"\n'
    environ = {**os.environ, **(env or {})}
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                         env=environ)
    return out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""


def _mk(tmp_path, *names):
    import time
    root = tmp_path / "results"
    (root / "ptm").mkdir(parents=True)
    (root / "train").mkdir(parents=True)
    for name in names:
        path = (root / "ptm" / name[4:]) if name.startswith("ptm/") else (root / "train" / name)
        path.write_text("" if name.endswith("!empty") else "x")
        time.sleep(0.02)
    return root


def test_resume_prefers_epoch_checkpoint_over_best_and_latest(render, tmp_path):
    """best/latest are often written LAST, so newest-by-time alone is wrong."""
    root = _mk(tmp_path, "model_epoch_007.pth", "best_model.pth", "latest.ckpt")
    assert _run_resume_block(render, root).endswith("model_epoch_007.pth")


def test_resume_falls_back_to_best_when_nothing_else_exists(render, tmp_path):
    root = _mk(tmp_path, "best_model.pth")
    assert _run_resume_block(render, root).endswith("best_model.pth")


def test_resume_never_picks_a_staged_ptm(render, tmp_path):
    """The failure this prevents: resuming from a pretrained backbone."""
    root = _mk(tmp_path, "ptm/pretrained.pth")
    assert _run_resume_block(render, root) == ""


def test_resume_ignores_zero_byte_checkpoints(render, tmp_path):
    root = tmp_path / "results"
    (root / "train").mkdir(parents=True)
    (root / "train" / "model_epoch_1.pth").write_text("")
    assert _run_resume_block(render, root) == ""


def test_resume_is_disabled_by_env(render, tmp_path):
    root = _mk(tmp_path, "model_epoch_007.pth")
    assert _run_resume_block(render, root, {"TAO_AUTO_RESUME": "0"}) == ""


def test_auto_resume_is_disablable(render):
    """TAO_AUTO_RESUME=0, the same switch the SDK exposed."""
    body = _body(render, resume_key="train.resume_training_checkpoint_path")
    assert "TAO_AUTO_RESUME:-1" in body
