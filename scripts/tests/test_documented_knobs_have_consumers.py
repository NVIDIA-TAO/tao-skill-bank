#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A documented control must actually control something.

This is the shape of the tao-sdk port regression. `SLURM_TIMEOUT_HOURS` came
across into the SKILL.md, into platform-preflight.md, and into
check_tao_launch_preflight.py -- which validates that it is smaller than
SLURM_TIME_HOURS -- while the behaviour it configured (wrap srun in `timeout`,
requeue on exit 124) did not. Preflight carefully checked a value nothing read,
so a 6-hour stage on a 4-hour partition died instead of requeueing, and nothing
in the repo could have told you.

The generalisable rule: if the bank documents or validates a knob, something
must consume it. A knob with no consumer is either a lie or a half-finished
port, and both are worth failing on.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

# Knobs the bank documents AND is expected to act on. Each maps to the marker
# that proves a consumer exists.
KNOBS = {
    "SLURM_TIMEOUT_HOURS": "timeout",           # the srun wrapper it sizes
    "SLURM_USE_REQUEUE": "--requeue",
    "TAO_AUTO_RESUME": "TAO_AUTO_RESUME",
    "TAO_RESULTS_ROOT": "TAO_RESULTS_ROOT",
}

SEARCH_ROOTS = ["skills", "templates", "scripts"]


def _grep(pattern: str) -> list[str]:
    result = subprocess.run(
        # `-e` is required: a pattern like "--requeue" would otherwise be
        # parsed as a grep option and silently match nothing.
        ["grep", "-rl", "--include=*.py", "--include=*.sh", "--include=*.tmpl",
         "--include=*.yaml", "--include=*.md", "-e", pattern, *SEARCH_ROOTS],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    return [p for p in result.stdout.splitlines() if "/tests/" not in p]


@pytest.mark.parametrize("knob,marker", sorted(KNOBS.items()))
def test_documented_knob_has_a_consumer(knob, marker):
    mentions = _grep(knob)
    if not mentions:
        pytest.skip(f"{knob} is not documented anywhere")
    consumers = _grep(marker)
    executable = [
        p for p in consumers
        if p.endswith((".py", ".sh", ".tmpl")) and not p.startswith("docs/")
    ]
    assert executable, (
        f"{knob} is documented in {mentions[:3]} but nothing executable "
        f"consumes it (no file contains {marker!r}). A validated-but-unread "
        "knob is how the tao-sdk wall-time requeue went missing."
    )


def test_slurm_timeout_hours_reaches_the_template():
    """The specific case that regressed: preflight validates it, so the
    template must actually wrap srun in a timeout."""
    template = (REPO / "templates/slurm/singlenode.sbatch.tmpl").read_text()
    assert re.search(r'timeout "@@TIMEOUT_MINUTES@@m"', template), (
        "check_tao_launch_preflight.py enforces SLURM_TIMEOUT_HOURS < "
        "SLURM_TIME_HOURS, but the template never applies a timeout"
    )
