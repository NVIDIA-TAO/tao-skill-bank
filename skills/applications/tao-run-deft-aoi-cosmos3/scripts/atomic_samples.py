#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Canonical identities and embedding assets for single images and reference pairs."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
from typing import Any, Iterable

from nvpaw_annotations import TASK_SPECS
from validate_sharegpt import image_paths, resolve_image


PAIR_ASSET_SCHEMA = "nvpaw_reference_pair_embedding_v1"
PAIR_CONTENT_IDENTITY = f"{PAIR_ASSET_SCHEMA}:ordered_constituent_bytes_sha256"
PAIR_CANVAS_SIZE = (1024, 512)
PAIR_DIVIDER_PIXELS = 4
PAIR_SIMILARITIES = ("canvas", "two_vector")
PAIR_SIMILARITY_COMBINES = ("mean", "min")


def identity_for_paths(kind: str, paths: Iterable[str]) -> str:
    payload = json.dumps(
        {"schema": PAIR_ASSET_SCHEMA, "kind": kind, "image_paths": list(paths)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{kind}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def content_identity_for_paths(kind: str, paths: Iterable[str]) -> str:
    """Hash the ordered bytes of every image in one atomic visual sample."""

    source_paths = [pathlib.Path(value) for value in paths]
    if kind == "single_image" and len(source_paths) != 1:
        raise ValueError("single-image content identity requires exactly one path")
    if kind == "reference_pair" and len(source_paths) != 2:
        raise ValueError("reference-pair content identity requires exactly two paths")
    if kind not in {"single_image", "reference_pair"}:
        raise ValueError(f"unsupported atomic content kind: {kind!r}")
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise ValueError(f"atomic content image files are missing: {missing}")
    digest = hashlib.sha256()
    digest.update(f"{PAIR_ASSET_SCHEMA}\0{kind}\0".encode("utf-8"))
    for index, path in enumerate(source_paths):
        item_digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                item_digest.update(chunk)
        digest.update(index.to_bytes(4, "big"))
        digest.update(item_digest.digest())
    return digest.hexdigest()


def sample_from_record(
    record: dict[str, Any],
    *,
    media_root: pathlib.Path,
    context: str,
    resolved_path_cache: dict[str, str] | None = None,
) -> dict[str, Any]:
    task_type = record.get("task_type")
    if task_type not in TASK_SPECS:
        raise ValueError(f"{context}: unsupported task_type {task_type!r}")
    raw_paths = image_paths(record, context=context)
    roles = list(TASK_SPECS[str(task_type)]["image_roles"])
    if len(raw_paths) != len(roles):
        raise ValueError(f"{context}: {task_type} requires image roles {roles}")
    resolved: list[str] = []
    for value in raw_paths:
        cached = (
            resolved_path_cache.get(value)
            if resolved_path_cache is not None
            else None
        )
        if cached is None:
            cached = str(resolve_image(value, media_root))
            if resolved_path_cache is not None:
                resolved_path_cache[value] = cached
        resolved.append(cached)
    if roles == ["target"]:
        kind = "single_image"
        reference = None
        target = resolved[0]
    elif roles == ["golden", "target"]:
        kind = "reference_pair"
        reference, target = resolved
    else:  # pragma: no cover - guarded by the sealed TASK_SPECS contract
        raise ValueError(f"{context}: unsupported image-role contract {roles}")
    return {
        "atomic_sample_id": identity_for_paths(kind, resolved),
        "sample_kind": kind,
        "image_paths": resolved,
        "reference_filepath": reference,
        "target_filepath": target,
    }


def logical_record_identity(record: dict[str, Any], *, context: str) -> str:
    task_type = record.get("task_type")
    if task_type not in TASK_SPECS:
        raise ValueError(f"{context}: unsupported task_type {task_type!r}")
    paths = [
        pathlib.PurePosixPath(value.replace("\\", "/")).as_posix()
        for value in image_paths(record, context=context)
    ]
    roles = list(TASK_SPECS[str(task_type)]["image_roles"])
    kind = "reference_pair" if roles == ["golden", "target"] else "single_image"
    return identity_for_paths(kind, paths)


def pair_asset_path(pair_assets_dir: pathlib.Path, atomic_sample_id: str) -> pathlib.Path:
    prefix, separator, digest = atomic_sample_id.partition(":")
    if prefix != "reference_pair" or separator != ":" or len(digest) != 64:
        raise ValueError(f"invalid reference-pair identity: {atomic_sample_id!r}")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError(f"invalid reference-pair identity: {atomic_sample_id!r}") from exc
    return pair_assets_dir.expanduser().resolve() / digest[:2] / f"{digest}.png"


def materialize_pair_asset(
    sample: dict[str, Any], *, pair_assets_dir: pathlib.Path
) -> pathlib.Path:
    if sample.get("sample_kind") != "reference_pair":
        raise ValueError("pair asset materialization requires a reference-pair sample")
    paths = sample.get("image_paths")
    if not isinstance(paths, list) or len(paths) != 2:
        raise ValueError("reference-pair sample requires exactly two ordered image paths")
    output = pair_asset_path(pair_assets_dir, str(sample.get("atomic_sample_id")))
    if output.is_file():
        return output
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - runtime dependency contract
        raise ValueError("Pillow is required to materialize pair embedding assets") from exc

    source_paths = [pathlib.Path(str(value)) for value in paths]
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise ValueError(f"reference-pair image files are missing: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".png", dir=output.parent
    )
    os.close(descriptor)
    temporary = pathlib.Path(temporary_name)
    try:
        canvas = Image.new("RGB", PAIR_CANVAS_SIZE, color=(0, 0, 0))
        side_width = (PAIR_CANVAS_SIZE[0] - PAIR_DIVIDER_PIXELS) // 2
        for index, source_path in enumerate(source_paths):
            with Image.open(source_path) as source:
                fitted = ImageOps.contain(
                    ImageOps.exif_transpose(source).convert("RGB"),
                    (side_width, PAIR_CANVAS_SIZE[1]),
                    method=Image.Resampling.LANCZOS,
                )
            offset_x = index * (side_width + PAIR_DIVIDER_PIXELS)
            offset = (
                offset_x + (side_width - fitted.width) // 2,
                (PAIR_CANVAS_SIZE[1] - fitted.height) // 2,
            )
            canvas.paste(fitted, offset)
        divider_x = side_width
        canvas.paste(
            (127, 127, 127),
            (divider_x, 0, divider_x + PAIR_DIVIDER_PIXELS, PAIR_CANVAS_SIZE[1]),
        )
        canvas.save(temporary, format="PNG", optimize=False)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def lookup_embedding_filepath(
    sample: dict[str, Any], *, pair_assets_dir: pathlib.Path | None
) -> str:
    """Return a cached embedding's path key without reading or rendering images."""

    if sample.get("sample_kind") == "single_image":
        return str(sample["target_filepath"])
    if pair_assets_dir is None:
        raise ValueError("pair_assets_dir is required for reference-pair embedding")
    return str(pair_asset_path(pair_assets_dir, str(sample.get("atomic_sample_id"))))


def embedding_filepath(
    sample: dict[str, Any], *, pair_assets_dir: pathlib.Path | None
) -> str:
    """Prepare an image for embedding, rendering a missing pair canvas if needed."""

    if sample.get("sample_kind") == "single_image":
        return str(sample["target_filepath"])
    if pair_assets_dir is None:
        raise ValueError("pair_assets_dir is required for reference-pair embedding")
    return str(materialize_pair_asset(sample, pair_assets_dir=pair_assets_dir))


def single_image_embedding_sample(
    filepath: str, *, embedding_roles: Iterable[str] = ()
) -> dict[str, Any]:
    """Return one cache-compatible full-resolution image embedding input."""

    path = str(pathlib.Path(filepath).expanduser().resolve())
    identity = identity_for_paths("single_image", [path])
    return {
        "atomic_sample_id": identity,
        "embedding_cache_key": identity,
        "sample_kind": "single_image",
        "image_paths": [path],
        "reference_filepath": None,
        "target_filepath": path,
        "filepath": path,
        "embedding_roles": sorted(set(embedding_roles)),
    }
