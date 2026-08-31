"""Standard-library validation for immutable NVPAW JSONL offset indexes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


INDEX_FORMAT = "native_uint64_byte_offsets_v2"
PIXEL_POLICY = "per_image_jsonl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_metadata_path(index_path: Path) -> Path:
    return index_path.with_suffix(index_path.suffix + ".json")


def source_fingerprint(jsonl_path: Path) -> dict[str, Any]:
    source = jsonl_path.expanduser().resolve(strict=True)
    stat = source.stat()
    return {
        "source_path": str(source),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


def load_index_metadata(index_path: Path) -> dict[str, Any] | None:
    path = index_metadata_path(index_path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def index_status(
    jsonl_path: Path,
    index_path: Path,
    expected_rows: int,
    expected_sha256: str,
    expected_image_items: int,
) -> tuple[bool, str, dict[str, Any] | None]:
    source = jsonl_path.expanduser()
    target = index_path.expanduser()
    if not source.is_file():
        return False, f"source JSONL is missing: {source}", None
    if not target.is_file():
        return False, f"offset index is missing: {target}", None
    metadata = load_index_metadata(target)
    if metadata is None:
        return False, f"index metadata is missing or invalid: {index_metadata_path(target)}", None
    expected = {
        **source_fingerprint(source),
        "row_count": int(expected_rows),
        "sha256": expected_sha256,
        "image_items": int(expected_image_items),
        "index_format": INDEX_FORMAT,
        "pixel_policy": PIXEL_POLICY,
    }
    mismatches = {
        key: {"expected": value, "observed": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        return False, f"index metadata mismatch: {mismatches}", metadata
    if target.stat().st_size != expected_rows * 8:
        return False, "offset index byte size does not match row count", metadata
    return True, "reusing validated JSONL offset index", metadata
