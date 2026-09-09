# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable input-archive contract for the IAA/PAS DEFT workflow."""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import re
import stat
import sys
from collections.abc import Mapping
from typing import Any


ARCHIVE_BINDINGS = (
    ("images_archive", "images_archive_sha256"),
    ("metadata_archive", "metadata_archive_sha256"),
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def approved_sha256(value: object, name: str) -> str:
    """Return one canonical SHA-256 digest or reject the approval value."""
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be exactly 64 lowercase hexadecimal characters")
    return value


def archive_path(value: object, name: str) -> pathlib.Path:
    """Resolve one approved regular, non-empty archive without path aliases."""
    if not isinstance(value, (str, pathlib.Path)) or not str(value):
        raise ValueError(f"{name} must be an existing absolute non-empty file")
    raw = pathlib.Path(value).expanduser()
    absolute = pathlib.Path(raw) if raw.is_absolute() else raw.absolute()
    if (
        not raw.is_absolute()
        or absolute != raw
        or not absolute.is_file()
        or absolute.stat().st_size == 0
        or absolute.is_symlink()
        or absolute.resolve() != absolute
    ):
        raise ValueError(
            f"{name} must be an existing normalized absolute non-empty regular file: "
            f"{value}"
        )
    return absolute


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def file_sha256(path: pathlib.Path) -> str:
    """Stream one stable regular file without loading the archive into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size == 0:
            raise ValueError(f"archive is not a non-empty regular file: {path}")
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(handle.fileno())
    if _stat_identity(before) != _stat_identity(after):
        raise ValueError(f"archive changed while its SHA-256 was being computed: {path}")
    return digest.hexdigest()


def verify_archive(
    path_value: object, digest_value: object, path_name: str, digest_name: str
) -> tuple[pathlib.Path, str]:
    """Verify an archive's current bytes against its approved content identity."""
    path = archive_path(path_value, path_name)
    expected = approved_sha256(digest_value, digest_name)
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{path_name} changed after approval: SHA-256 mismatch for {path}"
        )
    return path, actual


def verify_archive_bindings(
    config: Mapping[str, Any], *, verify_content: bool = True
) -> dict[str, str]:
    """Validate both required archive bindings from approval or run state."""
    verified: dict[str, str] = {}
    for path_field, digest_field in ARCHIVE_BINDINGS:
        path = archive_path(config.get(path_field), f"state.config.{path_field}")
        expected = approved_sha256(
            config.get(digest_field), f"state.config.{digest_field}"
        )
        if verify_content:
            actual = file_sha256(path)
            if actual != expected:
                raise ValueError(
                    f"state.config.{path_field} changed after initialization: "
                    f"SHA-256 mismatch for {path}"
                )
        verified[digest_field] = expected
    return verified


def main(argv: list[str] | None = None) -> int:
    """Print the canonical content identity used in read-only preflight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        path = archive_path(args.archive, "--archive")
        digest = file_sha256(path)
    except (OSError, ValueError) as exc:
        print(f"archive_contract: {exc}", file=sys.stderr)
        return 2
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
