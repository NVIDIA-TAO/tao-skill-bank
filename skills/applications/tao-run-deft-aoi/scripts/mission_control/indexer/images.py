# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image path resolution for TAO 7.0.1 Visual ChangeNet (siamese) DEFT runs.

The dataloader reconstructs each component crop as::

    {base}/{input_path}/{object_name}_{light}{ext}

``light`` and ``ext`` come from the run's own spec (``dataset.classify.input_map``
+ ``image_ext``), falling back to the AOI defaults ``SolderLight`` / ``.jpg``.

``base`` depends on the source table: KPI and inference rows are relative to
``images_dir``; train and mining-pool rows are relative to the workspace root
(synthetic rows already carry ``results/.../synthetic_*``); embedding and RCA
parquets carry an absolute ``filepath`` to use verbatim.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_LIGHT = "SolderLight"
DEFAULT_EXT = ".jpg"


def read_lights_ext(spec: dict | None) -> tuple[list[str], str]:
    """Return a run's lighting names ordered by channel index, and its file extension.

    Lighting conditions come from ``dataset.classify.input_map``. A component's
    captures are stacked into one training sample, so extra lights never change
    how many components there are. Falls back to the AOI defaults when the spec
    is missing or malformed.
    """
    try:
        dc = spec["dataset"]["classify"]
        ext = dc.get("image_ext", DEFAULT_EXT) or DEFAULT_EXT
        input_map = dc.get("input_map") or {}
        if not input_map:
            return [DEFAULT_LIGHT], str(ext)
        lights = [str(name) for name, _ in sorted(input_map.items(), key=lambda kv: kv[1])]
        return lights, str(ext)
    except (AttributeError, KeyError, TypeError, ValueError):
        return [DEFAULT_LIGHT], DEFAULT_EXT


def read_light_ext(spec: dict | None) -> tuple[str, str]:
    """Return the channel-0 lighting name and extension, e.g. ``("SolderLight", ".jpg")``.

    Channel 0 is the capture a point's key is built from, so point identity is
    the same however many lights a run has.
    """
    lights, ext = read_lights_ext(spec)
    return lights[0], ext


def swap_light(path: str | os.PathLike, from_light: str, to_light: str,
               ext: str = DEFAULT_EXT) -> str:
    """Return the same component's capture under a different lighting condition.

    Paths are built by concatenation, so a sibling capture is the same path with
    the light suffix swapped. Returns ``path`` unchanged when the suffix does not
    match — a synthetic or mined image that does not follow the convention.
    """
    p = str(path)
    suffix = f"_{from_light}{ext}"
    if not p.endswith(suffix):
        return p
    return p[: -len(suffix)] + f"_{to_light}{ext}"


def component_file(
    base: str | os.PathLike,
    input_path: str,
    object_name: str,
    light: str = DEFAULT_LIGHT,
    ext: str = DEFAULT_EXT,
) -> str:
    """Build one image's absolute path from its scattered pieces.

    Glues ``{base}/{input_path}/{object_name}_{light}{ext}`` and normalizes it
    (resolves symlinks + ``..``) so the same image always maps to one path.
    ``base`` is the run's ``images_dir`` or the workspace root, depending on
    which table the row came from.
    """
    fname = f"{object_name}_{light}{ext}"
    return os.path.realpath(os.path.join(str(base), str(input_path).strip("/"), fname))


def from_filepath(filepath: str | os.PathLike) -> str:
    """Normalize a path a table already stores in full.

    Resolves symlinks and ``..`` so it matches paths built by ``component_file``.
    """
    return os.path.realpath(str(filepath))


def exists(path: str | os.PathLike) -> bool:
    """Return True when the path is a regular file on disk."""
    return os.path.isfile(path)
