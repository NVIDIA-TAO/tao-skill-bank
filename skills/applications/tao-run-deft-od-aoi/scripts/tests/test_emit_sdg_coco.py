# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path
import pytest

SCRIPT = Path(__file__).parents[1] / "emit_sdg_coco.py"


def module():
    spec = importlib.util.spec_from_file_location("emit_sdg_coco", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def result(tmp_path):
    image = tmp_path / "defect.png"
    image.write_bytes(b"image")
    return {
        "group": {
            "dataset_id": "metal",
            "requested_rows": 2,
            "anomaly_types": ["metal+scratch"],
        },
        "generated": 1,
        "blocked": 1,
        "image_root": tmp_path,
        "coco": {
            "images": [{"id": 8, "file_name": str(image), "width": 20, "height": 10}],
            "categories": [{"id": 4, "name": "metal+scratch"}],
            "annotations": [
                {"id": 20, "image_id": 8, "category_id": 4, "bbox": [2, 1, 8, 3]}
            ],
        },
    }


def test_merge_preserves_native_classes_and_binary_projection(tmp_path):
    mod = module()
    report = mod._merge([result(tmp_path)], tmp_path / "output", guardrail_enabled=True)
    native = json.loads(
        (tmp_path / "output/pseudo_labels/coco_annotations.json").read_text()
    )
    binary = json.loads(
        (tmp_path / "output/pseudo_labels/coco_annotations_od_defect.json").read_text()
    )
    assert native["categories"] == [{"id": 1, "name": "metal+scratch"}]
    assert binary["categories"] == [{"id": 1, "name": "defect"}]
    assert report["groups"][0]["guardrail_blocked"] == 1
    assert report["training_pool_mutated"] is False


@pytest.mark.parametrize(
    "fault", ["duplicate_id", "duplicate_file", "missing_annotation", "bbox", "count"]
)
def test_merge_rejects_invalid_training_labels(tmp_path, fault):
    value = result(tmp_path)
    if fault == "duplicate_id":
        value["coco"]["images"].append(dict(value["coco"]["images"][0]))
    if fault == "duplicate_file":
        value["coco"]["images"].append({**value["coco"]["images"][0], "id": 9})
    if fault == "missing_annotation":
        value["coco"]["annotations"] = []
    if fault == "bbox":
        value["coco"]["annotations"][0]["bbox"] = [19, 1, 8, 3]
    if fault == "count":
        value["generated"] = 2
    with pytest.raises(ValueError):
        module()._merge([value], tmp_path / "output", guardrail_enabled=True)


def test_publish_all_blocked_run_is_complete_but_not_training_ready(tmp_path):
    import csv
    import hashlib

    row = {
        "image_filename": str(tmp_path / "clean.png"),
        "mask_filename": str(tmp_path / "mask.png"),
        "anomaly_type": "metal+scratch",
    }
    for key in ("image_filename", "mask_filename"):
        Path(row[key]).write_bytes(b"image")
    testcase = tmp_path / "testcase.jsonl"
    testcase.write_text(json.dumps(row) + "\n")
    bucket = tmp_path / "sdg"
    bucket.mkdir()
    with (bucket / "texture_ft_generation_result.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["index", "output_filename", *row])
        writer.writeheader()
    with (bucket / "guardrail_blocked.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["index", "output_idx", "guardrail", *row]
        )
        writer.writeheader()
        writer.writerow({"index": 0, "output_idx": -1, "guardrail": "text", **row})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "guardrail_requested": True,
                "native_timing": {"guardrail_enabled": True},
                "generation": __import__("anomalygen_contract").verify_generation(testcase, bucket),
                "testcase": str(testcase),
                "testcase_sha256": hashlib.sha256(testcase.read_bytes()).hexdigest(),
                "selected_output": str(bucket),
            }
        )
    )
    entries = tmp_path / "runs.json"
    entries.write_text(json.dumps([{"dataset_id": "metal", "manifest": str(manifest)}]))
    report = module().publish(entries, tmp_path / "output")
    assert report["status"] == "COMPLETE" and not report["training_ready"]
    assert report["groups"][0]["guardrail_blocked"] == 1


def test_merge_rejects_coco_image_outside_verified_generation_bucket(tmp_path):
    value = result(tmp_path)
    foreign = tmp_path / "unrelated.png"
    foreign.write_bytes(b"foreign")
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "defect.png").write_bytes(b"image")
    value["image_root"] = generated
    value["coco"]["images"][0]["file_name"] = str(foreign)
    with pytest.raises(ValueError, match="generation|bucket"):
        module()._merge([value], tmp_path / "output", guardrail_enabled=True)
