# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind a SLURM dataset subset manifest to one immutable IAA action request."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import tempfile
from collections.abc import Sequence
from typing import Any

import yaml
import pyarrow.parquet as pq


def _digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_request(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("action request must be an absolute regular file")
    request = json.loads(path.read_text(encoding="utf-8"))
    expected = request.get("request_sha256")
    unsigned = dict(request)
    unsigned.pop("request_sha256", None)
    if not isinstance(expected, str) or _digest(unsigned) != expected:
        raise ValueError("action request digest mismatch")
    return request


def _mount_source(request: dict[str, Any], target: str) -> pathlib.Path:
    matches = [item for item in request.get("mounts", []) if item.get("target") == target]
    if len(matches) != 1:
        raise ValueError(f"action request must contain exactly one {target} mount")
    source = pathlib.Path(matches[0]["source"])
    if not source.is_absolute() or source.is_symlink() or not source.is_dir():
        raise ValueError(f"{target} mount source is missing or unsafe")
    return source.resolve(strict=True)


def _inside(path: pathlib.Path, parent: pathlib.Path, name: str) -> pathlib.Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(parent)
    except ValueError as exc:
        raise ValueError(f"{name} escapes the approved dataset parent") from exc
    return resolved


def _relative_leaf(path: pathlib.Path, parent: pathlib.Path, name: str) -> str:
    if path.is_symlink():
        _inside(path, parent, name)
    elif not path.is_file():
        raise ValueError(f"{name} is not a regular file or symlink: {path}")
    try:
        relative = path.relative_to(parent)
    except ValueError as exc:
        raise ValueError(f"{name} is outside the approved dataset parent") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{name} has an unsafe relative path")
    return relative.as_posix()


def build_evaluate_manifest(request_path: pathlib.Path) -> dict[str, Any]:
    request = _load_request(request_path)
    bundle = request.get("spec_bundle", {})
    if request.get("platform") != "slurm" or request.get("name") != "evaluate":
        raise ValueError("subset mapping supports only a SLURM evaluate action")
    if bundle.get("command") != "clip" or bundle.get("args", [])[:2] != ["evaluate", "-e"]:
        raise ValueError("unexpected evaluate action argv")
    if len(bundle.get("args", [])) != 3:
        raise ValueError("evaluate action must name exactly one generated config")

    results_dir = _mount_source(request, "/results")
    dataset_parent = _mount_source(request, "/data")
    config_arg = bundle["args"][2]
    if not isinstance(config_arg, str) or not config_arg.startswith("/results/"):
        raise ValueError("evaluate config must be under /results")
    config_path = results_dir / config_arg.removeprefix("/results/")
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("generated evaluate config is missing or unsafe")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    dataset_specs = []
    for section in ("train", "val"):
        dataset_specs.extend(config.get("dataset", {}).get(section, {}).get("datasets", []))
    dataset_specs.extend(config.get("evaluate", {}).get("datasets", []))
    if not dataset_specs:
        raise ValueError("evaluate config contains no dataset inputs")
    normalized = {
        (
            item.get("image_dir"),
            item.get("caption_dir"),
            item.get("image_list_file"),
            item.get("caption_file_suffix"),
        )
        for item in dataset_specs
        if isinstance(item, dict)
    }
    if len(normalized) != 1 or len(dataset_specs) != len(
        [item for item in dataset_specs if isinstance(item, dict)]
    ):
        raise ValueError("evaluate train/val/evaluate inputs must be one exact dataset mapping")
    image_raw, caption_raw, list_raw, suffix = normalized.pop()
    if suffix != ".txt":
        raise ValueError("evaluate caption suffix must be .txt")

    image_dir = pathlib.Path(image_raw)
    caption_dir = pathlib.Path(caption_raw)
    if image_dir.name != "images" or caption_dir.name != "captions":
        raise ValueError("evaluate image/caption directories have unexpected names")
    dataset_root = image_dir.parent
    if caption_dir.parent != dataset_root or dataset_root.parent.resolve() != dataset_parent:
        raise ValueError("evaluate dataset directories do not bind to the /data mount")
    if image_dir.is_symlink() or caption_dir.is_symlink():
        raise ValueError("evaluate image/caption directories must not be symlinks")

    if not isinstance(list_raw, str) or not list_raw.startswith("/results/iaa_splits/"):
        raise ValueError("evaluate image list must be a canonical /results/iaa_splits input")
    list_path = results_dir / list_raw.removeprefix("/results/")
    if list_path.is_symlink() or not list_path.is_file():
        raise ValueError("evaluate image list is missing or unsafe")
    names = list_path.read_text(encoding="utf-8").splitlines()
    if not names or len(names) != len(set(names)):
        raise ValueError("evaluate image list must be non-empty and unique")

    entries: set[str] = set()
    for name in names:
        leaf = pathlib.PurePosixPath(name)
        if len(leaf.parts) != 1 or leaf.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
        }:
            raise ValueError(f"unsafe or unsupported evaluate image name: {name!r}")
        image = image_dir / name
        caption = caption_dir / f"{leaf.stem}.txt"
        entries.add(_relative_leaf(image, dataset_parent, "evaluate image"))
        entries.add(_relative_leaf(caption, dataset_parent, "evaluate caption"))
        for candidate, label in ((image, "evaluate image target"), (caption, "evaluate caption target")):
            if candidate.is_symlink():
                resolved = _inside(candidate, dataset_parent, label)
                entries.add(_relative_leaf(resolved, dataset_parent, label))

    ordered = sorted(entries)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "iaa-evaluate-dataset-subset",
        "request_sha256": request["request_sha256"],
        "source": str(dataset_parent),
        "dataset_root": str(dataset_root),
        "image_list": str(list_path),
        "image_list_count": len(names),
        "entries": ordered,
        "entry_count": len(ordered),
    }
    manifest["manifest_sha256"] = _digest(manifest)
    return manifest


def build_image_embedding_manifest(request_path: pathlib.Path) -> dict[str, Any]:
    """Select only image leaves named by an immutable visualization request."""
    request = _load_request(request_path)
    bundle = request.get("spec_bundle", {})
    name = request.get("name")
    if request.get("platform") != "slurm" or name not in {
        "viz_weak_embed", "viz_mined_embed", "viz_previous_embed",
    }:
        raise ValueError("image subset mapping supports only SLURM visualization embeddings")
    args = bundle.get("args", [])
    if bundle.get("command") != "embedding" or args[:2] != ["image_embeddings", "-e"]:
        raise ValueError("unexpected image embedding action argv")
    inputs = [item.split("=", 1)[1] for item in args if isinstance(item, str) and item.startswith("input_parquet=")]
    expected = {
        "viz_weak_embed": r"/results/iter_[1-9][0-9]*/embeddings/viz_weak/input\.parquet",
        "viz_mined_embed": r"/results/iter_[1-9][0-9]*/mining/mined_unique_images\.parquet",
        "viz_previous_embed": r"/results/iter_[1-9][0-9]*/embeddings/previous/prev_pool\.parquet",
    }[name]
    if len(inputs) != 1 or re.fullmatch(expected, inputs[0]) is None:
        raise ValueError("image embedding input parquet is not the canonical visualization input")

    results_dir = _mount_source(request, "/results")
    dataset_parent = _mount_source(request, "/data")
    input_path = results_dir / inputs[0].removeprefix("/results/")
    if input_path.is_symlink() or not input_path.is_file():
        raise ValueError("image embedding input parquet is missing or unsafe")
    table = pq.read_table(input_path, columns=["filepath"])
    filepaths = table.column("filepath").to_pylist()
    if not filepaths or any(not isinstance(item, str) or not item for item in filepaths):
        raise ValueError("image embedding filepath column must be non-empty strings")

    entries: set[str] = set()
    for raw in filepaths:
        image = pathlib.Path(raw)
        if raw.startswith("/results/"):
            image = results_dir / raw.removeprefix("/results/")
        elif raw.startswith("/data/"):
            image = dataset_parent / raw.removeprefix("/data/")
        if not image.is_absolute():
            raise ValueError("image embedding filepath must be absolute")
        try:
            result_relative = image.relative_to(results_dir)
        except ValueError:
            result_relative = None
        if result_relative is not None:
            if re.fullmatch(
                r"iter_[1-9][0-9]*/datagen/dataset/images/[^/]+\.(?:jpg|jpeg|png|bmp)",
                result_relative.as_posix(),
                flags=re.IGNORECASE,
            ) is None:
                raise ValueError("previous-pool result image is outside generated-image scope")
            if image.is_symlink():
                resolved = image.resolve(strict=True)
                try:
                    resolved.relative_to(results_dir)
                except ValueError as exc:
                    raise ValueError("previous-pool result image escapes /results") from exc
            elif not image.is_file():
                raise ValueError(f"previous-pool result image is missing: {image}")
            # The mutable results tree is staged separately with mirror semantics.
            continue
        entries.add(_relative_leaf(image, dataset_parent, "image embedding input"))
        if image.is_symlink():
            resolved = _inside(image, dataset_parent, "image embedding target")
            entries.add(_relative_leaf(resolved, dataset_parent, "image embedding target"))

    ordered = sorted(entries)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "iaa-visualization-image-subset",
        "request_sha256": request["request_sha256"],
        "source": str(dataset_parent),
        "input_parquet": str(input_path),
        "image_count": len(filepaths),
        "entries": ordered,
        "entry_count": len(ordered),
    }
    manifest["manifest_sha256"] = _digest(manifest)
    return manifest


def _canonical_results_file(
    results_dir: pathlib.Path, raw: Any, pattern: str, name: str
) -> pathlib.Path:
    if not isinstance(raw, str) or re.fullmatch(pattern, raw) is None:
        raise ValueError(f"{name} is not a canonical /results input")
    path = results_dir / raw.removeprefix("/results/")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} is missing or unsafe")
    return path


def _add_named_dataset_leaves(
    *,
    entries: set[str],
    names: list[str],
    image_dir: pathlib.Path,
    caption_dir: pathlib.Path,
    dataset_parent: pathlib.Path,
    label: str,
) -> None:
    if not names or len(names) != len(set(names)):
        raise ValueError(f"{label} image list must be non-empty and unique")
    for name in names:
        leaf = pathlib.PurePosixPath(name)
        if len(leaf.parts) != 1 or leaf.suffix.lower() not in {
            ".jpg", ".jpeg", ".png", ".bmp",
        }:
            raise ValueError(f"unsafe or unsupported {label} image name: {name!r}")
        image = image_dir / name
        caption = caption_dir / f"{leaf.stem}.txt"
        for candidate, kind in ((image, "image"), (caption, "caption")):
            item_label = f"{label} {kind}"
            entries.add(_relative_leaf(candidate, dataset_parent, item_label))
            if candidate.is_symlink():
                target = _inside(candidate, dataset_parent, f"{item_label} target")
                entries.add(
                    _relative_leaf(target, dataset_parent, f"{item_label} target")
                )


def build_train_manifest(request_path: pathlib.Path) -> dict[str, Any]:
    """Select only original mined and validation leaves required by CLIP train."""
    request = _load_request(request_path)
    bundle = request.get("spec_bundle", {})
    args = bundle.get("args", [])
    if request.get("platform") != "slurm" or request.get("name") != "train":
        raise ValueError("train subset mapping supports only a SLURM train action")
    if bundle.get("command") != "clip" or args[:2] != ["train", "-e"] or len(args) != 3:
        raise ValueError("unexpected train action argv")

    results_dir = _mount_source(request, "/results")
    dataset_parent = _mount_source(request, "/data")
    config_path = _canonical_results_file(
        results_dir,
        args[2],
        r"/results/iter_([1-9][0-9]*)/specs/train_config\.yaml",
        "train config",
    )
    match = re.fullmatch(
        r"/results/iter_([1-9][0-9]*)/specs/train_config\.yaml", args[2]
    )
    assert match is not None
    iteration = int(match.group(1))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train_specs = config.get("dataset", {}).get("train", {}).get("datasets", [])
    val_specs = config.get("dataset", {}).get("val", {}).get("datasets", [])
    if not isinstance(train_specs, list) or not train_specs:
        raise ValueError("train config contains no training datasets")
    if not isinstance(val_specs, list) or len(val_specs) != 1:
        raise ValueError("train config must contain one validation dataset")

    entries: set[str] = set()
    lists: list[str] = []
    image_count = 0
    dataset_root: pathlib.Path | None = None
    for index, item in enumerate(train_specs):
        if not isinstance(item, dict):
            raise ValueError("train dataset mapping must be an object")
        image_raw = item.get("image_dir")
        caption_raw = item.get("caption_dir")
        list_raw = item.get("image_list_file")
        pairs_raw = item.get("train_pairs_file")
        if isinstance(image_raw, str) and image_raw.startswith("/results/"):
            generated_match = re.fullmatch(
                r"/results/iter_([1-9][0-9]*)/datagen/dataset/images", image_raw
            )
            generated_iter = int(generated_match.group(1)) if generated_match else 0
            expected_base = rf"/results/iter_{generated_iter}/datagen/dataset"
            if (
                generated_match is None
                or generated_iter > iteration
                or caption_raw != expected_base + "/captions"
                or list_raw != expected_base + "/sdg_image_list.txt"
                or pairs_raw != expected_base + "/sdg_pairs.json"
                or item.get("caption_file_suffix") != ".txt"
            ):
                raise ValueError("generated train dataset mapping is not canonical")
            generated_images = results_dir / image_raw.removeprefix("/results/")
            generated_captions = results_dir / caption_raw.removeprefix("/results/")
            if (
                generated_images.is_symlink()
                or not generated_images.is_dir()
                or generated_captions.is_symlink()
                or not generated_captions.is_dir()
            ):
                raise ValueError("generated train dataset directories are missing or unsafe")
            _canonical_results_file(
                results_dir, list_raw, re.escape(list_raw), "generated train image list"
            )
            _canonical_results_file(
                results_dir, pairs_raw, re.escape(pairs_raw), "generated train pairs"
            )
            # Generated inputs are already inside the staged results tree.
            continue

        image_dir = pathlib.Path(image_raw) if isinstance(image_raw, str) else pathlib.Path()
        caption_dir = pathlib.Path(caption_raw) if isinstance(caption_raw, str) else pathlib.Path()
        if image_dir.name != "images" or caption_dir.name != "captions":
            raise ValueError("original train image/caption directories have unexpected names")
        current_root = image_dir.parent
        if (
            caption_dir.parent != current_root
            or current_root.parent.resolve() != dataset_parent
            or image_dir.is_symlink()
            or not image_dir.is_dir()
            or caption_dir.is_symlink()
            or not caption_dir.is_dir()
            or item.get("caption_file_suffix") != ".txt"
        ):
            raise ValueError("original train dataset does not bind to the /data mount")
        if dataset_root is None:
            dataset_root = current_root
        elif dataset_root != current_root:
            raise ValueError("original train datasets must share one dataset root")
        list_match = re.fullmatch(
            r"/results/iter_([1-9][0-9]*)/mining/mined_image_list\.txt",
            list_raw or "",
        )
        list_iter = int(list_match.group(1)) if list_match else 0
        if list_match is None or list_iter > iteration:
            raise ValueError(f"train dataset {index} image list is not canonical")
        expected_pairs = f"/results/iter_{list_iter}/mining/mined_pairs.json"
        if pairs_raw != expected_pairs:
            raise ValueError(f"train dataset {index} pairs are not canonical")
        list_path = _canonical_results_file(
            results_dir, list_raw, re.escape(list_raw), f"train dataset {index} image list"
        )
        _canonical_results_file(
            results_dir, pairs_raw, re.escape(expected_pairs), f"train dataset {index} pairs"
        )
        names = list_path.read_text(encoding="utf-8").splitlines()
        _add_named_dataset_leaves(
            entries=entries,
            names=names,
            image_dir=image_dir,
            caption_dir=caption_dir,
            dataset_parent=dataset_parent,
            label="training",
        )
        image_count += len(names)
        lists.append(str(list_path))

    if dataset_root is None:
        raise ValueError("train config contains no original mined dataset")

    val = val_specs[0]
    if not isinstance(val, dict):
        raise ValueError("validation dataset mapping must be an object")
    val_image_dir = pathlib.Path(val.get("image_dir", ""))
    val_caption_dir = pathlib.Path(val.get("caption_dir", ""))
    if (
        val_image_dir != dataset_root / "images"
        or val_caption_dir != dataset_root / "captions"
        or val.get("caption_file_suffix") not in {None, ".txt"}
    ):
        raise ValueError("validation dataset mapping is not canonical")
    val_list = _canonical_results_file(
        results_dir,
        val.get("image_list_file"),
        r"/results/iaa_splits/val_list\.txt",
        "training validation image list",
    )
    val_names = val_list.read_text(encoding="utf-8").splitlines()
    _add_named_dataset_leaves(
        entries=entries,
        names=val_names,
        image_dir=val_image_dir,
        caption_dir=val_caption_dir,
        dataset_parent=dataset_parent,
        label="training validation",
    )

    ordered = sorted(entries)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "iaa-train-dataset-subset",
        "request_sha256": request["request_sha256"],
        "source": str(dataset_parent),
        "dataset_root": str(dataset_root),
        "iteration": iteration,
        "training_lists": lists,
        "training_image_count": image_count,
        "validation_list": str(val_list),
        "validation_image_count": len(val_names),
        "entries": ordered,
        "entry_count": len(ordered),
    }
    manifest["manifest_sha256"] = _digest(manifest)
    return manifest


def build_manifest(request_path: pathlib.Path) -> dict[str, Any]:
    request = _load_request(request_path)
    if request.get("name") == "evaluate":
        return build_evaluate_manifest(request_path)
    if request.get("name") == "train":
        return build_train_manifest(request_path)
    return build_image_embedding_manifest(request_path)


def _atomic_write(path: pathlib.Path, value: dict[str, Any]) -> None:
    if not path.is_absolute():
        raise ValueError("output must be an absolute path")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", mode="w", encoding="utf-8", delete=False
    ) as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        temp = pathlib.Path(stream.name)
    os.replace(temp, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        manifest = build_manifest(pathlib.Path(args.request))
        _atomic_write(pathlib.Path(args.output), manifest)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.exit(1, f"prepare_slurm_dataset_subset: {exc}\n")
    print(json.dumps({key: manifest[key] for key in ("entry_count", "manifest_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
