# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pytest
import yaml


SCRIPT = Path(__file__).parents[1] / "prepare_deft_od_aoi_sources.py"
SPEC = importlib.util.spec_from_file_location("prepare_deft_od_aoi_sources", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)
INIT_SCRIPT = Path(__file__).parents[1] / "init_deft_od_aoi.py"
INIT_SPEC = importlib.util.spec_from_file_location("init_deft_od_aoi_for_sources", INIT_SCRIPT)
INIT_MODULE = importlib.util.module_from_spec(INIT_SPEC)
assert INIT_SPEC.loader
INIT_SPEC.loader.exec_module(INIT_MODULE)


def _source(root: Path, name: str, *, boxed: bool, category: int = 7) -> dict:
    directory = root / name / "images"
    directory.mkdir(parents=True)
    image = directory / f"{name}.png"
    image.write_bytes(b"image")
    annotations = ([{"id": 9, "image_id": 4, "category_id": category,
                     "bbox": [1, 2, 3, 4], "label": "scratch"}] if boxed else [])
    coco = root / name / "source.json"
    coco.write_text(json.dumps({
        "images": [{"id": 4, "file_name": image.name, "width": 12, "height": 10,
                    "customer_field": name}],
        "annotations": annotations,
        "categories": [{"id": category, "name": "customer-defect"}],
    }))
    return {"coco": str(coco), "images_dir": str(directory)}


def _manifest(root: Path) -> Path:
    path = root / "dataset_sources.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "inputs": {
            "kpi": [_source(root, "kpi", boxed=True)],
            "test": [_source(root, "test", boxed=False)],
            "mining": [_source(root, "mine", boxed=True)],
            "clean": [_source(root, "clean", boxed=False)],
        },
    }))
    return path


def test_prepares_binary_roles_and_customer_handoff(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    documents, report = MODULE.prepare(manifest)

    assert set(documents) == {"kpi", "test", "real", "clean"}
    assert documents["real"]["categories"] == [{"id": 1, "name": "defect"}]
    assert documents["real"]["annotations"][0]["category_id"] == 1
    assert documents["real"]["images"][0]["customer_field"] == "mine"
    assert not documents["clean"]["annotations"]
    assert report["roles"]["real"] == {"images": 1, "annotations": 1}

    output = tmp_path / "normalized"
    handoff = MODULE.materialize(manifest, documents, report, output, "copy")
    assert json.loads((output / "sources.json").read_text()) == handoff
    assert set(handoff["sources"]) == {"kpi", "test", "real", "clean"}
    for role, source in handoff["sources"].items():
        coco = json.loads(Path(source["coco"]).read_text())
        image = Path(source["images"]) / coco["images"][0]["file_name"]
        assert image.read_bytes() == b"image", role
        assert Path(coco["images"][0]["source_path"]) == image
        assert Path(coco["images"][0]["original_source_path"]).is_file()

    checkpoint = tmp_path / "base.pth"
    checkpoint.write_bytes(b"checkpoint")
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({"platform": "slurm", "max_iterations": 1,
                                      "base_checkpoint": str(checkpoint),
                                      "sources": handoff["sources"]}))
    state = INIT_MODULE.initialize(policy, tmp_path / "contract")
    assert state["roles"]["real"]["annotation_count"] == 1
    assert state["roles"]["clean"]["annotation_count"] == 0


def test_accepts_multiple_coco_shards_in_one_role(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    value = json.loads(manifest.read_text())
    second = _source(tmp_path, "mine_second", boxed=True)
    value["inputs"]["mining"][0]["coco"] = [
        value["inputs"]["mining"][0]["coco"], second["coco"]]
    manifest.write_text(json.dumps(value))

    documents, report = MODULE.prepare(manifest)

    assert len(documents["real"]["images"]) == 2
    assert len(documents["real"]["annotations"]) == 2
    assert len(report["sources"]) == 5


def test_rejects_boxless_mining_until_it_is_explicitly_routed(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    value = json.loads(manifest.read_text())
    value["inputs"]["mining"] = [_source(tmp_path, "boxless_mine", boxed=False)]
    manifest.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="only boxed images"):
        MODULE.prepare(manifest)


def test_rejects_annotations_in_verified_clean_source(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    value = json.loads(manifest.read_text())
    value["inputs"]["clean"] = [_source(tmp_path, "boxed_clean", boxed=True)]
    manifest.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="zero annotations"):
        MODULE.prepare(manifest)


def test_rejects_cross_role_image_overlap(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    value = json.loads(manifest.read_text())
    value["inputs"]["test"] = value["inputs"]["kpi"]
    manifest.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="overlaps kpi and test"):
        MODULE.prepare(manifest)


def test_refuses_to_overwrite_materialized_output(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    documents, report = MODULE.prepare(manifest)
    output = tmp_path / "normalized"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        MODULE.materialize(manifest, documents, report, output, "symlink")
