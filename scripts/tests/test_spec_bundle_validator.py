#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`tao_spec_bundle.py validate` — the producer-side lint for a spec-bundle.

The spec-bundle is the bank's producer -> consumer interface, and it is already
consumed by the docker, slurm and kubernetes skills. Nothing checked one before
it was handed over, so a malformed bundle failed inside a GPU allocation rather
than on the laptop.

The validator runs stdlib-only checks always, and full schema validation when
`jsonschema` is importable. Both layers must agree — the parity test below is
the one that matters, because a stdlib check that silently disagrees with the
shipped schema is worse than no check.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
MODULE = REPO / "scripts/tao_spec_bundle.py"
SCHEMA = REPO / "skills/core/tao-artifacts/references/spec_bundle.schema.json"

VALID = {
    "network_arch": "visual_changenet",
    "action": "train",
    "image": "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt",
    "mode": "config",
    "command": "visual_changenet train -e {config_path}",
    "config_format": "yaml",
    "spec": {"train": {"num_epochs": 12, "results_dir": "/results"}},
    "declared_inputs": [
        {"spec_key": "dataset.train.image_dir", "type": "folder", "uri": "/w/images"}
    ],
    "declared_outputs": [{"spec_key": "train.results_dir", "type": "folder"}],
    "compute_shape": {"gpus": 1, "nodes": 1},
}


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("tao_spec_bundle", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mutate(**changes):
    bundle = copy.deepcopy(VALID)
    for key, value in changes.items():
        if value is None:
            bundle.pop(key, None)
        else:
            bundle[key] = value
    return bundle


def test_valid_bundle_passes(mod):
    assert mod.validate(VALID) == []


@pytest.mark.parametrize("field", sorted(VALID))
def test_required_fields_are_required(mod, field):
    optional = {"config_format", "spec"}  # mode-conditional, reported differently
    problems = mod.validate(_mutate(**{field: None}))
    assert problems, f"dropping {field} produced no error"


def test_dotted_spec_key_is_rejected_with_the_fix(mod):
    """The schema calls this the #1 mistake; the message must show the fix."""
    problems = mod.validate(_mutate(spec={"train.num_epochs": 12}))
    assert any("dotted spec key" in p and "NESTED" in p for p in problems)


def test_dotted_key_found_at_depth(mod):
    problems = mod.validate(_mutate(spec={"train": {"opt": {"lr.warmup": 1}}}))
    assert any("lr.warmup" in p for p in problems)


def test_unresolved_versions_key_is_rejected(mod):
    """`image` is data, never a versions.yaml key — resolve before submit."""
    problems = mod.validate(_mutate(image="tao_toolkit.pyt"))
    assert any("versions.yaml key" in p and "resolve_tao_image" in p for p in problems)


def test_producer_cannot_declare_storage_tier(mod):
    """The tier is tao-data-io's decision at stage time, not the producer's."""
    bundle = copy.deepcopy(VALID)
    bundle["declared_inputs"][0]["storage_tier"] = "A"
    assert any("storage_tier" in p for p in mod.validate(bundle))


def test_config_mode_requires_config_path_placeholder(mod):
    """Without it the consumer's spec file is written and never read."""
    problems = mod.validate(_mutate(command="visual_changenet train"))
    assert any("{config_path}" in p for p in problems)


def test_config_mode_rejects_args(mod):
    assert any("args" in p for p in mod.validate(_mutate(args=["x"])))


def test_args_mode_rejects_spec(mod):
    bundle = _mutate(mode="args", args=["train"], command="run {config_path}")
    assert any("spec" in p for p in mod.validate(bundle))


def test_declared_outputs_cannot_be_empty(mod):
    assert any("declared_outputs" in p for p in mod.validate(_mutate(declared_outputs=[])))


@pytest.mark.parametrize("shape", [
    {"gpus": -1, "nodes": 1},
    {"gpus": 1, "nodes": 0},
    {"gpus": True, "nodes": 1},
    {"nodes": 1},
])
def test_compute_shape_is_checked(mod, shape):
    assert mod.validate(_mutate(compute_shape=shape))


def test_gpus_zero_is_legal(mod):
    """A CPU-only glue stage is a real case: zero GPUs, one node."""
    assert mod.validate(_mutate(compute_shape={"gpus": 0, "nodes": 1})) == []


BAD_CASES = [
    pytest.param(_mutate(spec={"train.num_epochs": 12}), id="dotted-key"),
    pytest.param(_mutate(image="tao_toolkit.pyt"), id="versions-key"),
    pytest.param(_mutate(command="visual_changenet train"), id="no-config-path"),
    pytest.param(_mutate(args=["x"]), id="config-with-args"),
    pytest.param(_mutate(declared_outputs=[]), id="no-outputs"),
    pytest.param(_mutate(compute_shape={"gpus": 1}), id="no-nodes"),
]


@pytest.mark.parametrize("bundle", BAD_CASES)
def test_stdlib_and_schema_agree_on_rejection(mod, bundle):
    """Parity: a stdlib check that disagrees with the shipped schema is a trap.

    Both layers must reject the same bundles. `jsonschema` is not a runtime
    dependency of scripts/, so the stdlib layer is what most callers get.
    """
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    schema_errors = list(jsonschema.Draft202012Validator(schema).iter_errors(bundle))
    assert schema_errors, "fixture is not actually schema-invalid"
    assert mod.validate(bundle), "stdlib layer accepted what the schema rejects"


def test_stdlib_and_schema_agree_on_the_valid_bundle(mod):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert list(jsonschema.Draft202012Validator(schema).iter_errors(VALID)) == []
    assert mod.validate(VALID) == []
