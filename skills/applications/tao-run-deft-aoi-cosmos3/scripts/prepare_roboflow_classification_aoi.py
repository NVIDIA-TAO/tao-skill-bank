#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Convert an expanded Roboflow good/defect classification export to DEFT AOI.

The converter expects the usual Roboflow Classification directory layout::

    dataset_root/
      train/good/*    train/defect/*
      valid/good/*    valid/defect/*
      test/good/*     test/defect/*

It maps ``good`` to ``OK`` and ``defect`` to ``NG``.  The upstream ``valid``
split becomes Proxy, ``test`` becomes frozen Benchmark, and ``train`` becomes
the Mining Pool.  Since a classification export has no physical golden image
pairs, every target receives a deterministic *demo pseudo-reference*: a
different good image from its own upstream split where one is available.  This
creates structurally valid two-image records for an end-to-end demonstration,
but the manifest makes the limitation explicit.  It must never be relabelled
as a production matched-golden AOI dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import tempfile
from collections import Counter
from typing import Any


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
SPLIT_ROLES = {"train": "mining", "valid": "proxy", "test": "benchmark"}
ROLE_FILES = {
    "mining": "mining_pool.json",
    "proxy": "proxy_kpi.json",
    "benchmark": "benchmark_kpi.json",
}
DEFAULT_PROMPT = (
    "Compare the inspection image with the reference image. "
    "Return exactly OK if the inspected PCB is good; otherwise return exactly NG."
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _split_dir(root: pathlib.Path, upstream_split: str) -> pathlib.Path:
    aliases = {
        "train": ("train",),
        "valid": ("valid", "validation", "val"),
        "test": ("test",),
    }[upstream_split]
    for alias in aliases:
        candidate = root / alias
        if candidate.is_dir():
            return candidate
    raise ValueError(
        f"missing {upstream_split!r} split beneath {root}; expected one of {aliases}"
    )


def _label_dirs(split_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    found: dict[str, pathlib.Path] = {}
    for child in split_dir.iterdir():
        if not child.is_dir():
            continue
        normalized = child.name.casefold().replace("_", "-").replace(" ", "-")
        if normalized in {"good", "ok"}:
            label = "OK"
        elif normalized in {"defect", "defective", "ng", "not-good"}:
            label = "NG"
        else:
            continue
        if label in found:
            raise ValueError(
                f"{split_dir}: multiple directories map to label {label}: "
                f"{found[label].name!r}, {child.name!r}"
            )
        found[label] = child
    missing = {"OK", "NG"}.difference(found)
    if missing:
        raise ValueError(
            f"{split_dir}: expected good/defect class directories; missing {sorted(missing)}"
        )
    return found


def _source_images(directory: pathlib.Path) -> list[pathlib.Path]:
    images = sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"no supported image files under {directory}")
    return images


def _destination_for(
    *, source: pathlib.Path, class_dir: pathlib.Path, workspace: pathlib.Path, upstream_split: str
) -> pathlib.Path:
    relative = source.relative_to(class_dir)
    return workspace / "images" / "roboflow-poc-ongfq-v2" / upstream_split / class_dir.name / relative


def _copy_image(source: pathlib.Path, destination: pathlib.Path) -> None:
    if destination.exists():
        if _sha256(source) != _sha256(destination):
            raise ValueError(f"refusing to replace non-identical existing image: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _relative(path: pathlib.Path, workspace: pathlib.Path) -> str:
    return path.relative_to(workspace).as_posix()


def _choose_reference(target: str, good_targets: list[str]) -> str:
    candidates = [candidate for candidate in good_targets if candidate != target] or good_targets
    if not candidates:
        raise ValueError("a demo pseudo-reference requires at least one good image")
    index = int(hashlib.sha256(target.encode("utf-8")).hexdigest(), 16) % len(candidates)
    return candidates[index]


def prepare(
    *, dataset_root: pathlib.Path, workspace: pathlib.Path, prompt: str, provenance_url: str
) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    workspace = workspace.expanduser().resolve()
    if not dataset_root.is_dir():
        raise ValueError(f"dataset root is not a directory: {dataset_root}")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    prepared: dict[str, list[dict[str, str]]] = {}
    source_hashes: dict[str, str] = {}
    for upstream_split, role in SPLIT_ROLES.items():
        split_dir = _split_dir(dataset_root, upstream_split)
        classes = _label_dirs(split_dir)
        rows: list[dict[str, str]] = []
        for label, class_dir in classes.items():
            for source in _source_images(class_dir):
                destination = _destination_for(
                    source=source,
                    class_dir=class_dir,
                    workspace=workspace,
                    upstream_split=upstream_split,
                )
                _copy_image(source, destination)
                target = _relative(destination, workspace)
                rows.append({"target": target, "label": label})
                source_hashes[target] = _sha256(destination)
        if not rows:
            raise ValueError(f"{upstream_split}: no images were prepared")
        prepared[role] = sorted(rows, key=lambda row: row["target"])

    annotations: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, Any] = {"roles": {}, "reference_policy": "demo_pseudo_reference"}
    for role, rows in prepared.items():
        good_targets = [row["target"] for row in rows if row["label"] == "OK"]
        if not good_targets:
            raise ValueError(f"{role}: no good images are available for pseudo-references")
        records: list[dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            reference = _choose_reference(row["target"], good_targets)
            record: dict[str, Any] = {
                "images": [row["target"], reference],
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": row["label"]},
                ],
            }
            if role in {"proxy", "benchmark"}:
                record["id"] = f"roboflow-poc-v2-{role}-{index:05d}"
            records.append(record)
        annotations[role] = records
        summary["roles"][role] = dict(Counter(row["label"] for row in rows))

    annotation_paths = {role: workspace / "annotations" / filename for role, filename in ROLE_FILES.items()}
    existing = [str(path) for path in annotation_paths.values() if path.exists()]
    if existing:
        raise ValueError("refusing to overwrite existing annotations: " + ", ".join(existing))
    for role, path in annotation_paths.items():
        _atomic_json(path, annotations[role])
    benchmark_hash = _sha256(annotation_paths["benchmark"])
    manifest = {
        "version": 1,
        "dataset": {
            "name": "Roboflow poc classification",
            "source_url": provenance_url,
            "dataset_version": 2,
            "source_root": str(dataset_root),
            "license": "CC BY 4.0",
        },
        "label_mapping": {
            "good": "OK",
            "defect": "NG",
        },
        "reference_policy": {
            "name": "demo_pseudo_reference",
            "method": (
                "A deterministic good image from the same upstream classification split; "
                "not a physical matched golden reference."
            ),
            "production_eligible": False,
        },
        "source_image_sha256": source_hashes,
        "evaluation_contract": {
            "benchmark": {
                "annotations": str(annotation_paths["benchmark"]),
                "annotations_sha256": benchmark_hash,
                "frozen": True,
            }
        },
        "summary": summary,
    }
    manifest_path = workspace / "provenance" / "roboflow_poc_v2_manifest.json"
    _atomic_json(manifest_path, manifest)
    return {
        "annotation_paths": {role: str(path) for role, path in annotation_paths.items()},
        "manifest": str(manifest_path),
        "benchmark_sha256": benchmark_hash,
        "summary": summary,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=pathlib.Path)
    parser.add_argument("--workspace", required=True, type=pathlib.Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--provenance-url",
        default="https://universe.roboflow.com/s-workspace-a82wo/poc-ongfq/dataset/2",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = prepare(
            dataset_root=args.dataset_root,
            workspace=args.workspace,
            prompt=args.prompt,
            provenance_url=args.provenance_url,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"prepare_roboflow_classification_aoi: {exc}")
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
