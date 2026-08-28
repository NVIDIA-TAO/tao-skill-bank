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

# The dict above is hand-maintained, so it only ever catches knobs somebody
# remembered to add -- and `sqsh_conversion_memory_gb` was documented in
# skill_info.yaml, never rendered, and missed here for exactly that reason. The
# conversion step then ran with the partition's default memory and was
# OOM-killed while extracting layers it had already downloaded.
#
# So discover these rather than enumerate them: every `sqsh_conversion_*` key a
# platform ships is a promise that something reads it.
CONVERSION_KNOB_MARKERS = {
    "sqsh_conversion_partition": "conversion_partition",
    "sqsh_conversion_timeout_minutes": "conversion_minutes",
    "sqsh_conversion_memory_mb": "conversion_memory_mb",
    "sqsh_conversion_cpus_per_task": "conversion_cpus_per_task",
}


def _shipped_conversion_knobs() -> list[tuple[str, str]]:
    import yaml

    found = []
    for info in (REPO / "skills/platform").glob("*/references/skill_info.yaml"):
        data = yaml.safe_load(info.read_text(encoding="utf-8")) or {}
        for key in (data.get("resource_defaults") or {}):
            if key.startswith("sqsh_conversion_"):
                found.append((str(info.relative_to(REPO)), key))
    return found


def test_conversion_knobs_are_known_to_this_test():
    """A new sqsh_conversion_* key must be given a consumer marker here.

    Without this the discovery below would silently skip an unrecognised knob,
    reintroducing the allowlist gap it exists to close.
    """
    unknown = sorted({k for _, k in _shipped_conversion_knobs()
                      if k not in CONVERSION_KNOB_MARKERS})
    assert not unknown, (
        f"new conversion knob(s) {unknown} ship in skill_info.yaml; add the "
        "marker proving a renderer consumes each one"
    )


def test_every_shipped_conversion_knob_is_consumed():
    """Documented conversion resources must reach the rendered command."""
    render = (REPO / "skills/platform/tao-run-on-slurm/references/render.py").read_text(
        encoding="utf-8"
    )
    for source, key in _shipped_conversion_knobs():
        marker = CONVERSION_KNOB_MARKERS.get(key)
        if marker is None:
            continue
        assert marker in render, (
            f"{source} ships {key} but no renderer reads it; the allocation "
            "then silently takes the partition default"
        )


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
