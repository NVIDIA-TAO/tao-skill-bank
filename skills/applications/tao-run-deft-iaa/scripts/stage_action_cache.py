#!/usr/bin/env python3
"""Materialize and verify the exact TAO cache subset bound to an action request."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import tempfile


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_manifest(request: dict) -> tuple[pathlib.Path, list[dict]]:
    manifest = request.get("cache_subset")
    if not isinstance(manifest, dict):
        raise ValueError(
            "action request has no TAO cache subset; adapter actions must skip cache staging"
        )
    unsigned = {"root": manifest["root"], "entries": manifest["entries"]}
    if sha256_json(unsigned) != manifest["sha256"]:
        raise ValueError("cache subset manifest digest mismatch")
    root = pathlib.Path(manifest["root"])
    seen: set[str] = set()
    for entry in manifest["entries"]:
        rel = pathlib.PurePosixPath(entry["path"])
        if rel.is_absolute() or ".." in rel.parts or str(rel) in seen:
            raise ValueError(f"unsafe or duplicate cache path: {entry['path']}")
        seen.add(str(rel))
        source = root / pathlib.Path(*rel.parts)
        resolved = source.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
        if not resolved.is_file() or resolved.stat().st_size != entry["size"]:
            raise ValueError(f"cache entry missing or changed: {entry['path']}")
        if sha256_file(resolved) != entry["sha256"]:
            raise ValueError(f"cache entry digest mismatch: {entry['path']}")
    return root, manifest["entries"]


def validate_tree(root: pathlib.Path, entries: list[dict], label: str) -> None:
    """Verify an exact materialized tree, including every byte copied."""
    expected = {entry["path"]: entry for entry in entries}
    actual: dict[str, pathlib.Path] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"{label} contains a symlink: {path.relative_to(root)}")
        if path.is_file():
            actual[path.relative_to(root).as_posix()] = path
    if set(actual) != set(expected):
        raise ValueError(f"{label} contains missing or unrelated files")
    for relative, path in actual.items():
        entry = expected[relative]
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"{label} size mismatch: {relative}")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{label} digest mismatch: {relative}")


def stage(request_path: pathlib.Path, destination: pathlib.Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    root, entries = validate_manifest(request)
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tao-cache-", dir=destination.parent) as tmp:
        staged = pathlib.Path(tmp) / "cache"
        for entry in entries:
            rel = pathlib.PurePosixPath(entry["path"])
            source = (root / pathlib.Path(*rel.parts)).resolve(strict=True)
            target = staged / pathlib.Path(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            os.chmod(target, 0o600)
        validate_tree(staged, entries, "staged cache")
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staged, destination)
    validate_tree(destination, entries, "promoted cache")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=pathlib.Path, required=True)
    parser.add_argument("--destination", type=pathlib.Path, required=True)
    args = parser.parse_args()
    stage(args.request.resolve(strict=True), args.destination)
    print("IAA_CACHE_SUBSET=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
