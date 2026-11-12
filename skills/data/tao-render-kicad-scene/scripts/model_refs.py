# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve the 3D model references a KiCad footprint declares. No pcbnew needed.

A footprint names its models itself: any number of them, each with its own
offset, rotation and scale, under a file name that need not match the
footprint and may be shared with other footprints. This module turns one
declared reference into a file on disk, or into an explicit problem. It never
substitutes another file.
"""
import hashlib
import os
from pathlib import Path
import re

# Formats seen to load in `kicad-cli pcb export glb` of KiCad 10.0.5. Others are refused until checked.
EXPORTABLE = {".step": "STEP", ".stp": "STEP", ".wrl": "VRML (units of 0.1 inch; mesh only)"}
CURRENT_VAR = "KICAD10_3DMODEL_DIR"
LEGACY_VARS = re.compile(r"^(KICAD\d+_3DMODEL_DIR|KISYS3DMOD)$")
VARIABLE = re.compile(r"\$\{([^}]+)\}|\$\(([^)]+)\)")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(reference, model_dir=None, project_dir=None, variables=None, environ=None):
    """Resolve one declared reference.

    Lookup order for `${NAME}` / `$(NAME)`: `variables`, the process
    environment, then KiCad's own names (`KICAD10_3DMODEL_DIR` -> `model_dir`,
    older `KICADn_3DMODEL_DIR`/`KISYS3DMOD` as aliases of it, `KIPRJMOD` ->
    `project_dir`). A relative path is tried under `project_dir`, then `model_dir`.
    Returns a dict; `problem` is None only when the file exists in an exportable format.
    """
    environ = os.environ if environ is None else environ
    variables = dict(variables or {})
    out = {"declared": reference, "resolved_path": None, "via": None, "format": None, "sha256": None, "problem": None}
    if not reference or not reference.strip():
        out["problem"] = "empty model reference"
        return out
    if reference.startswith(":"):
        out["problem"] = "KiCad path alias (':name:path') is not resolved by this helper; use a variable or a path"
        return out
    used, undefined = [], []

    def expand(match):
        name = match.group(1) or match.group(2)
        if name in variables:
            used.append(f"variable {name} (given)")
            return str(variables[name])
        if name in environ:
            used.append(f"variable {name} (environment)")
            return environ[name]
        if name == CURRENT_VAR and model_dir:
            used.append(f"variable {name} (runtime model library)")
            return str(model_dir)
        if LEGACY_VARS.match(name) and model_dir:
            used.append(f"variable {name} (legacy alias of {CURRENT_VAR})")
            return str(model_dir)
        if name == "KIPRJMOD" and project_dir:
            used.append("variable KIPRJMOD (project directory)")
            return str(project_dir)
        undefined.append(name)
        return match.group(0)

    text = VARIABLE.sub(expand, reference).replace("\\", "/")
    if undefined:
        out["problem"] = "undefined variable: " + ", ".join(sorted(set(undefined)))
        return out
    path = Path(text)
    if path.is_absolute():
        out["via"] = "; ".join(used) or "absolute path"
        candidates = [path]
    else:
        bases = [(project_dir, "relative to project directory"), (model_dir, "relative to runtime model library")]
        candidates = [Path(base) / path for base, _ in bases if base]
        labels = [label for base, label in bases if base]
        if not candidates:
            out["problem"] = "relative path but no project or model directory given"
            return out
    for index, candidate in enumerate(candidates):
        if candidate.is_file():
            out["resolved_path"] = str(candidate.resolve())
            out["via"] = out["via"] or labels[index]
            break
    else:
        out["problem"] = "file not found: " + " | ".join(str(c) for c in candidates)
        return out
    suffix = Path(out["resolved_path"]).suffix.lower()
    out["format"] = EXPORTABLE.get(suffix)
    out["sha256"] = sha256(out["resolved_path"])
    if out["format"] is None:
        out["problem"] = (f"format '{suffix}' has not been checked with this exporter; "
                          f"checked: {', '.join(sorted(EXPORTABLE))}")
    return out
