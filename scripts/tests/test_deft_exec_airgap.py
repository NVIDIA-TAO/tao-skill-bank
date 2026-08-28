#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Air-gap policy must fail CLOSED on commands `deft_exec` cannot reason about.

`_reject_airgap` recognises a container run by scanning argv for `docker` or
`podman`. Before this suite existed, anything else — `srun`, `sbatch`,
`kubectl`, `enroot import`, `ssh` — skipped the registry/pull/offline-env block
entirely and was permitted unexamined, so an air-gapped run could pull from a
registry or install over the network with no error. These tests pin the
fail-closed behaviour, and pin that the networked default is unaffected.

Both DEFT forks that ship `deft_exec.py` are checked, because the file is
duplicated per skill and the two copies have drifted before.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
FORKS = {
    "aoi": REPO / "skills/applications/tao-run-deft-aoi/scripts/deft_exec.py",
    "cosmos3": REPO / "skills/applications/tao-run-deft-aoi-cosmos3/scripts/deft_exec.py",
}

AIRGAP = {"network_mode": "airgap"}
NETWORKED = {"network_mode": "network-enabled"}

# Commands an air-gapped run must refuse. Each one reached the network before.
REFUSED = [
    pytest.param(["srun", "--container-image=nvcr.io#nvidia/tao:1", "x"], id="srun-pyxis-import"),
    pytest.param(["sbatch", "job.sbatch"], id="sbatch"),
    pytest.param(["kubectl", "apply", "-f", "job.yaml"], id="kubectl-apply"),
    pytest.param(["enroot", "import", "docker://nvcr.io#img"], id="enroot-import"),
    pytest.param(["singularity", "pull", "docker://img"], id="singularity-pull"),
    pytest.param(["ssh", "gpu-node", "pip install torch"], id="ssh-remote-install"),
    pytest.param(["aws", "s3", "cp", "s3://b/k", "."], id="aws-s3"),
    pytest.param(["rsync", "-a", "host:/src", "/dst"], id="rsync"),
    pytest.param(["docker", "pull", "img"], id="docker-pull"),
    pytest.param(["docker", "run", "--pull=always", "img"], id="docker-pull-always"),
    # The payload, not the wrapper, is what reaches the network.
    pytest.param(["srun", "bash", "-c", "curl http://example/x"], id="nested-under-srun"),
    pytest.param(["bash", "-lc", "wget http://example/x"], id="nested-login-shell"),
]

# Must keep working: air-gap is a restriction, not a prohibition on running.
PERMITTED_AIRGAP = [
    pytest.param(["docker", "run", "--pull=never", "img", "train"], id="docker-run-no-pull"),
    pytest.param(["bash", "-c", "docker run --pull=never img train"], id="nested-legit-run"),
    pytest.param(["python", "train.py"], id="plain-python"),
]


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(f"deft_exec_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=sorted(FORKS), ids=sorted(FORKS))
def deft_exec(request):
    path = FORKS[request.param]
    if not path.is_file():
        pytest.skip(f"{path} not present")
    return _load(request.param, path)


@pytest.mark.parametrize("command", REFUSED)
def test_airgap_refuses(deft_exec, command):
    """Air-gap mode raises rather than silently permitting."""
    with pytest.raises(ValueError):
        deft_exec._reject_airgap(command, AIRGAP)


@pytest.mark.parametrize("command", PERMITTED_AIRGAP)
def test_airgap_permits_legitimate_work(deft_exec, command):
    """Fail-closed must not become fail-on-everything."""
    deft_exec._reject_airgap(command, AIRGAP)


@pytest.mark.parametrize("command", REFUSED)
def test_networked_mode_is_unaffected(deft_exec, command):
    """network-enabled is the default; the policy applies only to air-gap."""
    deft_exec._reject_airgap(command, NETWORKED)


def test_forks_agree_on_policy_sets(deft_exec):
    """The two forks drift; the policy vocabularies must not."""
    aoi = _load("aoi_ref", FORKS["aoi"])
    assert deft_exec.NETWORK_TOOLS == aoi.NETWORK_TOOLS
    assert deft_exec.PACKAGE_TOOLS == aoi.PACKAGE_TOOLS
    assert deft_exec.REMOTE_LAUNCHERS == aoi.REMOTE_LAUNCHERS
    assert deft_exec.OFFLINE_ENV == aoi.OFFLINE_ENV


def test_unparseable_payload_fails_closed(deft_exec):
    """A shell payload we cannot tokenize must not be waved through."""
    with pytest.raises(ValueError):
        deft_exec._reject_airgap(["bash", "-c", "curl 'unbalanced"], AIRGAP)
