# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST ONLY: CPU native contract fixture, no model behavior is simulated."""
import csv
import json
from pathlib import Path
from anomalygen_contract import sha256, verify_generation
from anomalygen_consumer import load_generation


def write_native(root, names, blocked=0):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    bucket = root / "sdg"
    bucket.mkdir(exist_ok=True)
    cases, generated, denied = [], [], []
    for index in range(len(names) + blocked):
        source, mask = root / f"source{index}.png", root / f"donor{index}_mask.png"
        source.write_bytes(b"TEST_ONLY_IMAGE")
        mask.write_bytes(b"TEST_ONLY_MASK")
        case = {"index": index, "image_filename": str(source), "mask_filename": str(mask), "anomaly_type": "PCB+bridge"}
        cases.append(case)
        if index >= len(names):
            denied.append({**case, "output_idx": 0, "guardrail": "text"})
        else:
            name = names[index]
            for part in ("reconstructed_image", "original_image", "original_mask"):
                target = bucket / part / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"TEST_ONLY_IMAGE")
            generated.append({**case, "output_filename": name})
    testcase = root / "testcase.jsonl"
    testcase.write_text("".join(json.dumps(row) + "\n" for row in cases))
    for name, rows, extra in (("texture_ft_generation_result.csv", generated, ["output_filename"]), ("guardrail_blocked.csv", denied, ["output_idx", "guardrail"])):
        with (bucket / name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(cases[0]) + extra)
            writer.writeheader(); writer.writerows(rows)
    checkpoint, recipe = root / "iter_000015000.pt", root / "recipe.yaml"
    checkpoint.write_bytes(b"TEST_ONLY_ADAPTER"); recipe.write_text("anomaly_types: [[PCB, bridge]]\n")
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"status": "COMPLETE", "guardrail_requested": True, "native_timing": {"guardrail_enabled": True}, "testcase": str(testcase), "testcase_sha256": sha256(testcase), "selected_output": str(bucket), "generation": verify_generation(testcase, bucket), "checkpoint": str(checkpoint), "recipe": str(recipe)}))
    return manifest


def prepared_fixture(manifest):
    value = json.loads(Path(manifest).read_text())
    cases = [json.loads(line) for line in Path(value["testcase"]).read_text().splitlines()]
    provenance = Path(manifest).parent / "provenance.jsonl"
    provenance.write_text("".join(json.dumps({"generation_index": i, "source_mask": c["mask_filename"], "source_mask_sha256": sha256(c["mask_filename"]), "fn_filepath": c["image_filename"], "clean_filepath": c["image_filename"], "aligned_mask": c["mask_filename"]}) + "\n" for i, c in enumerate(cases)))
    group = {"dataset_id": "fixture", "testcase": value["testcase"], "checkpoint": value["checkpoint"], "recipe": value["recipe"], "requested_rows": len(cases), "provenance": str(provenance)}
    prepared = Path(manifest).parent / "prepared.json"
    prepared.write_text(json.dumps({"status": "COMPLETE", "generation_ready": True, "generator_groups": [group], "artifacts": [{"path": str(p), "sha256": sha256(p)} for p in [Path(value["testcase"]), provenance]]}))
    return prepared


def bind_coco(path, images_root, *, only_synthetic=False):
    path = Path(path)
    doc = json.loads(path.read_text())
    selected = [row for row in doc["images"] if not only_synthetic or row.get("deft_kind") == "synthetic_defect"]
    if not selected or all("anomalygen_lineage" in row for row in selected):
        return
    manifest = write_native(path.parent / (path.stem + "-native"), [Path(row["file_name"]).name for row in selected])
    _, rows = load_generation(manifest)
    prepared = prepared_fixture(manifest)
    for image, row in zip(selected, rows):
        image["file_name"] = row["generated_image"]
        if only_synthetic:
            image["source_path"] = row["generated_image"]
        image["anomalygen_lineage"] = row["lineage"]
        image["anomalygen_prepared"] = {"manifest": str(prepared), "manifest_sha256": sha256(prepared), "dataset_id": "fixture"}
    path.write_text(json.dumps(doc))


def write_generation_report(path, generated, blocked):
    path = Path(path)
    manifest = write_native(path.parent / (path.stem + "-native"), [f"PCB+bridge_{i:05}.png" for i in range(generated)], blocked)
    value = json.loads(manifest.read_text())
    labels = manifest.parent / "labels.json"
    labels.write_text(json.dumps({"images": [{"id": i, "file_name": f"PCB+bridge_{i:05}.png", "width": 16, "height": 16} for i in range(generated)], "categories": [{"id": 1, "name": "PCB+bridge"}], "annotations": []}))
    value["pseudo_labels"] = str(labels)
    manifest.write_text(json.dumps(value))
    prepared = prepared_fixture(manifest)
    runs = manifest.parent / "runs.json"
    runs.write_text(json.dumps([{"dataset_id": "fixture", "manifest": str(manifest)}]))
    path.write_text(json.dumps({"status": "COMPLETE", "generated": generated, "groups": [{"dataset_id": "fixture", "requested": generated + blocked, "generated": generated, "guardrail_blocked": blocked}], "runs_manifest": str(runs), "runs_manifest_sha256": sha256(runs), "prepared_manifest": str(prepared), "prepared_manifest_sha256": sha256(prepared)}))
