# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify model-owned schema exclusions preserve a consistent public contract."""
import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "schema_generator", Path(__file__).resolve().parents[1] / "generate_dataclass_schemas.py"
)
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


def test_exclusions_remove_nested_defaults_and_search_metadata_without_mutation():
    original = {
        "properties": {
            "model": {
                "properties": {
                    "keep": {"default": 7},
                    "private": {"properties": {"weight": {"default": 2}}},
                    "private_neighbor": {"default": 3},
                },
                "required": ["keep", "private", "private_neighbor"],
                "default": {"keep": 7, "private": {"weight": 2}, "private_neighbor": 3},
            }
        },
        "default": {"model": {"keep": 7, "private": {"weight": 2}, "private_neighbor": 3}},
        "popular": {"model": {"private": {"weight": 4}, "keep": 8}},
        "automl_default_parameters": ["model.keep", "model.private.weight", "model.private_neighbor"],
        "automl_disabled_parameters": ["model.private", "model.keep"],
    }
    snapshot = copy.deepcopy(original)
    result = GENERATOR.filter_schema_excluded_fields(original, ["model.private"])
    assert original == snapshot
    assert result["properties"]["model"]["required"] == ["keep", "private_neighbor"]
    assert set(result["properties"]["model"]["properties"]) == {"keep", "private_neighbor"}
    assert result["properties"]["model"]["default"] == {"keep": 7, "private_neighbor": 3}
    assert result["default"] == {"model": {"keep": 7, "private_neighbor": 3}}
    assert result["popular"] == {"model": {"keep": 8}}
    assert result["automl_default_parameters"] == ["model.keep", "model.private_neighbor"]
    assert result["automl_disabled_parameters"] == ["model.keep"]
    assert GENERATOR.filter_schema_excluded_fields(original, []) == original


@pytest.mark.parametrize("invalid", ["model.private", [None], [""], ["model..private"]])
def test_malformed_exclusion_contract_is_rejected(invalid):
    with pytest.raises(ValueError, match="schema_excluded_fields"):
        GENERATOR.filter_schema_excluded_fields({}, invalid)
