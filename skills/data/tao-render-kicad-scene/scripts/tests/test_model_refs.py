# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model reference resolution against self-made files; no KiCad needed."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model_refs  # noqa: E402


@pytest.fixture()
def tree(tmp_path):
    library = tmp_path / "3dmodels" / "Parts.3dshapes"
    library.mkdir(parents=True)
    (library / "shared_body.step").write_text("ISO-10303-21; synthetic\n")
    project = tmp_path / "project"
    (project / "models").mkdir(parents=True)
    (project / "models" / "local.stp").write_text("ISO-10303-21; synthetic local\n")
    (project / "models" / "mesh.wrl").write_text("#VRML V2.0 utf8\n")
    (project / "models" / "scan.obj").write_text("o synthetic\n")
    return tmp_path / "3dmodels", project


def resolve(reference, tree, **kwargs):
    return model_refs.resolve(reference, tree[0], tree[1], environ=kwargs.pop("environ", {}), **kwargs)


def test_library_variable_and_name_unrelated_to_any_footprint(tree):
    row = resolve("${KICAD10_3DMODEL_DIR}/Parts.3dshapes/shared_body.step", tree)
    assert row["problem"] is None and row["format"] == "STEP"
    assert row["resolved_path"] == str((tree[0] / "Parts.3dshapes/shared_body.step").resolve())
    assert "runtime model library" in row["via"]


def test_legacy_variables_alias_the_current_library(tree):
    for name in ("KICAD6_3DMODEL_DIR", "KICAD9_3DMODEL_DIR", "KISYS3DMOD"):
        row = resolve("${%s}/Parts.3dshapes/shared_body.step" % name, tree)
        assert row["problem"] is None and "legacy alias" in row["via"]
    assert resolve("$(KISYS3DMOD)/Parts.3dshapes/shared_body.step", tree)["problem"] is None


def test_given_variable_beats_environment_beats_builtin(tree):
    other = tree[1] / "models"
    reference = "${KICAD10_3DMODEL_DIR}/local.stp"
    assert "(given)" in resolve(reference, tree, variables={"KICAD10_3DMODEL_DIR": other})["via"]
    assert "(environment)" in resolve(reference, tree, environ={"KICAD10_3DMODEL_DIR": str(other)})["via"]
    assert resolve(reference, tree)["problem"].startswith("file not found")


def test_project_variable_relative_and_absolute_paths(tree):
    by_variable = resolve("${KIPRJMOD}/models/local.stp", tree)
    relative = resolve("models/local.stp", tree)
    absolute = resolve(str(tree[1] / "models/local.stp"), tree)
    assert by_variable["via"] == "variable KIPRJMOD (project directory)"
    assert relative["via"] == "relative to project directory" and absolute["via"] == "absolute path"
    assert by_variable["sha256"] == relative["sha256"] == absolute["sha256"]  # one file, three references
    assert resolve("Parts.3dshapes/shared_body.step", tree)["via"] == "relative to runtime model library"


def test_missing_file_undefined_variable_and_alias_are_problems_not_substitutions(tree):
    missing = resolve("${KICAD10_3DMODEL_DIR}/Parts.3dshapes/absent.step", tree)
    assert missing["resolved_path"] is None and missing["problem"].startswith("file not found")
    assert resolve("${NO_SUCH_VAR}/x.step", tree)["problem"] == "undefined variable: NO_SUCH_VAR"
    assert "alias" in resolve(":vendor:x.step", tree)["problem"]
    assert resolve("", tree)["problem"] == "empty model reference"
    assert resolve("x.step", (None, None))["problem"].startswith("relative path but no")


def test_format_must_have_been_checked_with_the_exporter(tree):
    assert resolve("models/mesh.wrl", tree)["problem"] is None
    unchecked = resolve("models/scan.obj", tree)
    assert unchecked["resolved_path"] and "has not been checked" in unchecked["problem"]
