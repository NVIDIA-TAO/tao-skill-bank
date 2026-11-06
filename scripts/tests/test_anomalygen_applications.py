# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native generation consumer contracts; CPU fixtures never run a model."""
import csv
import importlib.util
import json
import sys
from pathlib import Path
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "skills/data/tao-generate-anomalies/scripts"
sys.path.insert(0, str(SHARED))
from anomalygen_contract import sha256, verify_generation


def module(name, application):
    path = ROOT / "skills/applications" / application / "scripts" / (name + ".py")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("ag12_" + name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def native_run(root, blocked=()):
    root.mkdir(parents=True, exist_ok=True)
    output = root / "sdg"
    output.mkdir()
    cases, generated, denied = [], [], []
    for index in (4, 19):
        source, mask = root / f"source{index}.png", root / f"donor{index}_mask.png"
        for path in (source, mask):
            Image.new("RGB", (8, 8), "white").save(path)
        case = dict(index=index, image_filename=str(source), mask_filename=str(mask), anomaly_type="IC+bridge")
        cases.append(case)
        if index in blocked:
            denied.append(dict(case, output_idx=0, guardrail="text"))
        else:
            name = f"IC+bridge_{index:05}.png"
            for bucket in ("reconstructed_image", "original_image", "original_mask"):
                target = output / bucket / name
                target.parent.mkdir(exist_ok=True)
                Image.new("RGB", (8, 8), "white").save(target)
            generated.append(dict(case, output_filename=name))
    testcase = root / "testcase.jsonl"
    testcase.write_text("".join(json.dumps(row) + "\n" for row in cases))
    for name, rows, cols in (("texture_ft_generation_result.csv", generated, list(cases[0]) + ["output_filename"]), ("guardrail_blocked.csv", denied, list(cases[0]) + ["output_idx", "guardrail"])):
        with (output / name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
    report = dict(schema="anomalygen-run/1", status="COMPLETE", guardrail_requested=True, native_timing={"guardrail_enabled": True, "image_guardrail_enforcing": False}, testcase=str(testcase), testcase_sha256=sha256(testcase), selected_output=str(output), generation=verify_generation(testcase, output))
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(report))
    return manifest


@pytest.mark.parametrize("blocked", [(), (19,), (4, 19)])
def test_native_consumers_complete_with_sparse_and_blocked_identities(tmp_path, blocked):
    manifest = native_run(tmp_path / "run", blocked)
    emitter = module("emit_sdg_sharegpt", "tao-run-deft-aoi-cosmos3")
    records, report = emitter.emit_native(manifest, media_root=tmp_path, prompt="Inspect exactly.", relative=False)
    assert len(records) == 2 - len(blocked)
    assert report["guardrail_blocked"] == len(blocked)
    assert report["training_ready"] == bool(records)
    assert {r["anomalygen_lineage"]["index"] for r in records} == {4, 19} - set(blocked)
    assert all(r["conversations"][-1]["value"] == "NG" for r in records)
    assert all("original_image" in r["images"][1] for r in records)


def test_native_emitter_rejects_excluded_mask_donor(tmp_path):
    manifest = native_run(tmp_path / "run")
    excluded = tmp_path / "excluded.txt"
    excluded.write_text("donor19\n")
    emitter = module("emit_sdg_sharegpt", "tao-run-deft-aoi-cosmos3")
    with pytest.raises(ValueError, match="excluded"):
        emitter.emit_native(manifest, media_root=tmp_path, prompt="Inspect", relative=False, exclude_stems=excluded)


def test_manifest_rejects_mutated_source_and_missing_identity(tmp_path):
    manifest = native_run(tmp_path / "run")
    from anomalygen_consumer import load_generation
    value = json.loads(manifest.read_text())
    testcase = Path(value["testcase"])
    testcase.write_text(testcase.read_text().replace("source4", "source99"))
    with pytest.raises(ValueError, match="testcase changed"):
        load_generation(manifest)


def test_native_pair_requires_paired_original(tmp_path):
    manifest = native_run(tmp_path / "run")
    (tmp_path / "run/sdg/original_image/IC+bridge_00004.png").unlink()
    emitter = module("emit_sdg_sharegpt", "tao-run-deft-aoi-cosmos3")
    with pytest.raises(ValueError, match="paired"):
        emitter.emit_native(manifest, media_root=tmp_path, prompt="Inspect", relative=False)


def test_changenet_emission_keeps_labels_and_lineage(tmp_path):
    manifest = native_run(tmp_path / "run", (19,))
    emitter = module("changenet_data_pair_prepare", "tao-run-deft-aoi")
    destination = tmp_path / "train.csv"
    report = emitter.generate_native(manifest, destination, images_dir=tmp_path / "images", subdirname="sdg")
    rows = list(csv.DictReader(destination.open()))
    assert len(rows) == 1 and rows[0]["label"] == "bridge"
    assert json.loads(rows[0]["anomalygen_lineage"])["index"] == 4
    assert report["guardrail_blocked"] == 1


def test_split_rejects_generated_descendant_of_evaluation_target(tmp_path):
    manifest = native_run(tmp_path / "run")
    emitter = module("emit_sdg_sharegpt", "tao-run-deft-aoi-cosmos3")
    records, _ = emitter.emit_native(manifest, media_root=tmp_path, prompt="Inspect", relative=False)
    def record(target):
        return {"images": [str(target), str(tmp_path / "golden.png")], "conversations": [{"from": "human", "value": "Inspect"}, {"from": "gpt", "value": "NG"}]}
    paths = {}
    for role, rows in {"proxy": [record(tmp_path / "run/source4.png")], "benchmark": [record(tmp_path / "benchmark.png")], "mining": [record(tmp_path / "mining.png")], "synthetic": records}.items():
        paths[role] = tmp_path / (role + ".json")
        paths[role].write_text(json.dumps(rows))
    validator = module("validate_split_contract", "tao-run-deft-aoi-cosmos3")
    with pytest.raises(ValueError, match="lineage"):
        validator.validate(paths, media_root=tmp_path)


def test_assembly_ignores_valid_empty_generated_batch(tmp_path):
    manifest = native_run(tmp_path / "run", (4, 19))
    emitter = module("emit_sdg_sharegpt", "tao-run-deft-aoi-cosmos3")
    output = tmp_path / "synthetic.json"
    assert emitter.main(["--manifest", str(manifest), "--media-root", str(tmp_path), "--prompt", "Inspect", "--output", str(output)]) == 0
    assembly = module("assemble_training_json", "tao-run-deft-aoi-cosmos3")
    rows, report = assembly.assemble(None, [output], dedupe=True, validation_paths=[])
    assert rows == [] and report["output_records"] == 0


def test_checkpoint_binding_uses_actual_public_adapter_and_original_recipe(tmp_path):
    from anomalygen_consumer import checkpoint_binding
    config = checkpoint_binding(tmp_path, "nvpcb")
    assert config["checkpoint"].endswith("nvpcb/model/iter_000015000.pt")
    assert config["recipe"].endswith("nvpcb/model/exp_texture_ft_pcb.yaml")
    assert config["checkpoint_repo"] == "nvidia/Cosmos-AnomalyGen-PCB-Nano"
    with pytest.raises(ValueError, match="recipe"):
        checkpoint_binding(tmp_path, "nvpcb", checkpoint=tmp_path / "byo.pt")


def test_native_stage_rejects_wrong_generated_sharegpt(tmp_path):
    from anomalygen_consumer import validate_sharegpt_emission
    manifest = native_run(tmp_path / "run")
    artifact = tmp_path / "output.json"
    artifact.write_text("[]")
    with pytest.raises(ValueError, match="count"):
        validate_sharegpt_emission(artifact, manifest)


def test_changenet_validator_rejects_excluded_lineage(tmp_path):
    manifest = native_run(tmp_path / "run")
    emitter = module("changenet_data_pair_prepare", "tao-run-deft-aoi")
    destination = tmp_path / "train.csv"
    emitter.generate_native(manifest, destination)
    exclude = tmp_path / "excluded.txt"
    exclude.write_text("source4\n")
    validator = module("validate_training_csv", "tao-run-deft-aoi")
    assert any("excluded" in error for error in validator.validate(destination, tmp_path, exclude_stems=exclude))


def test_od_native_coco_preserves_source_lineage_and_blocks(tmp_path):
    manifest = native_run(tmp_path / "run", (19,))
    value = json.loads(manifest.read_text())
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"images": [{"id": 5, "file_name": "IC+bridge_00004.png", "width": 8, "height": 8}], "annotations": [{"id": 1, "image_id": 5, "category_id": 1, "bbox": [1, 1, 4, 4]}], "categories": [{"id": 1, "name": "IC+bridge"}]}))
    value["pseudo_labels"] = str(labels)
    manifest.write_text(json.dumps(value))
    runs = tmp_path / "runs.json"
    runs.write_text(json.dumps([{"dataset_id": "pcb", "manifest": str(manifest)}]))
    emitter = module("emit_sdg_coco", "tao-run-deft-od-aoi")
    report = emitter.publish(runs, tmp_path / "emitted")
    coco = json.loads((tmp_path / "emitted/pseudo_labels/coco_annotations_od_defect.json").read_text())
    assert coco["images"][0]["anomalygen_lineage"]["index"] == 4
    assert report["guardrail_blocked"] == 1
    assert report["runs_manifest_sha256"] == sha256(runs)


def test_od_emitter_rejects_changed_prepared_manifest(tmp_path):
    emitter = module("emit_sdg_coco", "tao-run-deft-od-aoi")
    prepared = tmp_path / "prepared.json"
    prepared.write_text(json.dumps({"status": "COMPLETE", "generation_ready": True, "generator_groups": [], "artifacts": [{"path": str(prepared), "sha256": "bad"}]}))
    with pytest.raises(ValueError, match="prepared"):
        emitter.validate_prepared(prepared)


def test_all_blocked_route_records_zero_actual_quality_searches(tmp_path, monkeypatch):
    import shutil
    import run_anomalygen
    manifest = native_run(tmp_path / "fixture", (4, 19))
    value = json.loads(manifest.read_text())
    checkpoint, recipe = tmp_path / "adapter.pt", tmp_path / "recipe.yaml"
    checkpoint.write_bytes(b"TEST_ONLY_ADAPTER")
    recipe.write_text("anomaly_types: [[IC, bridge]]\n")
    output = tmp_path / "run"
    args = run_anomalygen.parser().parse_args(["inference_only", "--name", "blocked", "--repo", str(tmp_path), "--output-dir", str(output), "--checkpoint", str(checkpoint), "--recipe", str(recipe), "--dataset-dir", str(tmp_path), "--input-data-path", value["testcase"]])
    commands = []
    def execute(command, args, **kwargs):
        commands.append(command)
        shutil.copytree(Path(value["selected_output"]), output / "sdg")
        (output / "sdg/timing_summary.json").write_text(json.dumps({"guardrail_enabled": True}))
        return 0
    monkeypatch.setattr(run_anomalygen, "execute", execute)
    report = run_anomalygen.run(args)
    assert len(commands) == 1 and report["quality"]["skipped"]
    assert report["quality"]["search_rounds"] == 0


def test_stage_rejects_images_substituted_under_valid_lineage(tmp_path):
    from anomalygen_consumer import validate_sharegpt_emission
    manifest = native_run(tmp_path / "run")
    emitter = module("emit_sdg_sharegpt", "tao-run-deft-aoi-cosmos3")
    records, _ = emitter.emit_native(manifest, media_root=tmp_path, prompt="Inspect", relative=False)
    records[0]["images"][0] = str(tmp_path / "foreign.png")
    output = tmp_path / "records.json"
    output.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="pair"):
        validate_sharegpt_emission(output, manifest)


def test_split_rejects_pair_substitution_with_retained_native_lineage(tmp_path):
    manifest = native_run(tmp_path / "run")
    emitter = module("emit_sdg_sharegpt", "tao-run-deft-aoi-cosmos3")
    records, _ = emitter.emit_native(manifest, media_root=tmp_path, prompt="Inspect", relative=False)
    records[0]["images"][0] = str(tmp_path / "substituted.png")
    paths = {}
    for role in ("proxy", "benchmark", "mining", "synthetic"):
        path = tmp_path / (role + ".json")
        rows = records if role == "synthetic" else [{"images": [str(tmp_path / (role + ".png")), "golden.png"], "conversations": [{"from": "human", "value": "Inspect"}, {"from": "gpt", "value": "OK"}]}]
        path.write_text(json.dumps(rows)); paths[role] = path
    validator = module("validate_split_contract", "tao-run-deft-aoi-cosmos3")
    with pytest.raises(ValueError, match="pair"):
        validator.validate(paths, media_root=tmp_path)


def test_od_admission_rejects_substituted_image_under_native_lineage(tmp_path):
    from anomalygen_consumer import load_generation
    manifest = native_run(tmp_path / "run")
    _, rows = load_generation(manifest)
    image = tmp_path / "foreign.png"
    image.write_bytes(b"TEST_ONLY")
    document = {"images": [{"id": 1, "file_name": str(image), "width": 8, "height": 8, "anomalygen_lineage": rows[0]["lineage"]}], "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4]}], "categories": [{"id": 1, "name": "defect"}]}
    admission = module("admit_deft_od_aoi_coco", "tao-run-deft-od-aoi")
    with pytest.raises(ValueError, match="lineage"):
        admission._synthetic_candidates(document, tmp_path, 1, 25)


def test_cosmos_report_distinguishes_all_blocked_from_missing_run(tmp_path):
    report = module("render_report", "tao-run-deft-aoi-cosmos3")
    html = report._augmentation_rows({"config": {"anomalygen": {"num_SDG": 2}}, "iterations": {"iter1": {"anomalygen_accounting": {"generated": 0, "guardrail_blocked": 2, "training_ready": False}, "anomalygen_amp_allocated": 2}}})
    assert "blocked: 2" in html
    assert "no synthetic training data" in html


def _write_rows(path, rows, fields=None):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


@pytest.mark.parametrize("change", ["input_path", "golden_path", "lineage", "label"])
@pytest.mark.parametrize("converted", [False, True])
def test_changenet_assembled_synthetic_membership_is_authoritative(tmp_path, change, converted):
    manifest = native_run(tmp_path / "run")
    emitter = module("changenet_data_pair_prepare", "tao-run-deft-aoi")
    validator = module("validate_training_csv", "tao-run-deft-aoi")
    source = tmp_path / "synthetic.csv"
    emitter.generate_native(manifest, source, images_dir=tmp_path / "images" if converted else None)
    rows = list(csv.DictReader(source.open()))
    if converted:
        for row in rows:
            for key in ("input_path", "golden_path"):
                row[key] = "images/" + row[key]
    assembled = tmp_path / "train.csv"
    _write_rows(assembled, rows)
    assert validator.validate(assembled, tmp_path, synthetic_csvs=[source]) == []
    if change in ("input_path", "golden_path"):
        foreign = tmp_path / "protected_evaluation.png"
        Image.new("RGB", (8, 8), "red").save(foreign)
        rows[0][change] = str(foreign)
    elif change == "lineage":
        for row in rows:
            row.pop("anomalygen_lineage")
    else:
        rows[0]["label"] = "PASS"
    _write_rows(assembled, rows)
    assert validator.validate(assembled, tmp_path, synthetic_csvs=[source])


def test_changenet_direct_emission_rejects_stripped_lineage(tmp_path):
    manifest = native_run(tmp_path / "run")
    emitter = module("changenet_data_pair_prepare", "tao-run-deft-aoi")
    validator = module("validate_training_csv", "tao-run-deft-aoi")
    source = tmp_path / "synthetic.csv"; emitter.generate_native(manifest, source)
    rows = list(csv.DictReader(source.open()))
    for row in rows: row["anomalygen_lineage"] = ""
    _write_rows(source, rows)
    assert validator.validate(source, tmp_path)


@pytest.mark.parametrize("change", ["lineage", "golden", "label"])
def test_cosmos_assembled_synthetic_records_must_match_verified_source(tmp_path, change):
    import copy
    manifest = native_run(tmp_path / "run")
    emitter = module("emit_sdg_sharegpt", "tao-run-deft-aoi-cosmos3")
    records, _ = emitter.emit_native(manifest, media_root=tmp_path, prompt="Inspect", relative=False)
    paths = {}
    for role in ("proxy", "benchmark", "mining", "synthetic", "train"):
        path = tmp_path / (role + ".json"); paths[role] = path
        rows = copy.deepcopy(records) if role in ("synthetic", "train") else [{"images": [str(tmp_path / (role + ".png")), "golden.png"], "conversations": [{"from": "human", "value": "Inspect"}, {"from": "gpt", "value": "OK"}]}]
        path.write_text(json.dumps(rows))
    validator = module("validate_split_contract", "tao-run-deft-aoi-cosmos3")
    validator.validate(paths, media_root=tmp_path)
    rows = json.loads(paths["train"].read_text())
    if change == "lineage": rows[0].pop("anomalygen_lineage")
    elif change == "golden": rows[0]["images"][1] = "foreign.png"
    else: rows[0]["conversations"][-1]["value"] = "OK"
    paths["train"].write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="synthetic"):
        validator.validate(paths, media_root=tmp_path)


def _prepared_od(tmp_path):
    manifest = native_run(tmp_path / "run")
    value = json.loads(manifest.read_text())
    checkpoint, recipe = tmp_path / "adapter.pt", tmp_path / "recipe.yaml"
    checkpoint.write_bytes(b"TEST_ONLY"); recipe.write_text("anomaly_types: [[IC, bridge]]\n")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"images": [{"id": i, "file_name": f"IC+bridge_{index:05}.png", "width": 8, "height": 8} for i, index in enumerate((4, 19))], "annotations": [{"id": i, "image_id": i, "category_id": 1, "bbox": [1, 1, 4, 4]} for i in range(2)], "categories": [{"id": 1, "name": "IC+bridge"}]}))
    value.update(checkpoint=str(checkpoint), recipe=str(recipe), pseudo_labels=str(labels)); manifest.write_text(json.dumps(value))
    donor = tmp_path / "protected_donor_mask.png"; donor.write_bytes(b"TEST_ONLY_MASK")
    provenance = tmp_path / "provenance.jsonl"
    provenance.write_text("".join(json.dumps({"generation_index": i, "source_mask": str(donor), "source_mask_sha256": sha256(donor), "fn_filepath": str(tmp_path / "protected_fn.png"), "clean_filepath": str(manifest.parent / f"source{index}.png"), "aligned_mask": str(manifest.parent / f"donor{index}_mask.png")}) + "\n" for i, index in enumerate((4, 19))))
    prepared = tmp_path / "prepared.json"
    prepared.write_text(json.dumps({"status": "COMPLETE", "generation_ready": True, "generator_groups": [{"dataset_id": "fixture", "testcase": value["testcase"], "checkpoint": str(checkpoint), "recipe": str(recipe), "requested_rows": 2, "provenance": str(provenance)}], "artifacts": [{"path": str(p), "sha256": sha256(p)} for p in [Path(value["testcase"]), provenance, donor]]}))
    runs = tmp_path / "runs.json"; runs.write_text(json.dumps([{"dataset_id": "fixture", "manifest": str(manifest)}]))
    return runs, prepared


@pytest.mark.parametrize("stem", ["protected_donor", "protected_fn"])
def test_od_conversion_enforces_explicit_prepared_ancestry(tmp_path, stem):
    runs, prepared = _prepared_od(tmp_path)
    excluded = tmp_path / "excluded.txt"; excluded.write_text(stem + "\n")
    emitter = module("emit_sdg_coco", "tao-run-deft-od-aoi")
    with pytest.raises(ValueError, match="excluded"):
        emitter.publish(runs, tmp_path / "emitted", prepared, excluded)


def test_od_admission_rechecks_prepared_ancestry_with_current_exclusions(tmp_path):
    runs, prepared = _prepared_od(tmp_path)
    emitter = module("emit_sdg_coco", "tao-run-deft-od-aoi")
    emitter.publish(runs, tmp_path / "emitted", prepared)
    document = json.loads((tmp_path / "emitted/pseudo_labels/coco_annotations_od_defect.json").read_text())
    admission = module("admit_deft_od_aoi_coco", "tao-run-deft-od-aoi")
    admission.validate_synthetic_lineage(document)
    excluded = tmp_path / "excluded.txt"; excluded.write_text("protected_donor\n")
    with pytest.raises(ValueError, match="excluded"):
        admission.validate_synthetic_lineage(document, excluded)
    document["images"][0].pop("anomalygen_prepared")
    with pytest.raises(ValueError, match="prepared"):
        admission.validate_synthetic_lineage(document)


def test_changenet_state_gate_requires_sources_even_after_all_lineage_is_stripped(tmp_path):
    import argparse
    emitter = module("changenet_data_pair_prepare", "tao-run-deft-aoi")
    validator = module("validate_training_csv", "tao-run-deft-aoi")
    commit = module("commit_stage", "tao-run-deft-aoi")
    results = tmp_path / "results"; manifest = native_run(results / "iter1/anomalygen")
    source = results / "iter1/mining_filter/sdg_rows_raw.csv"
    emitter.generate_native(manifest, source, images_dir=tmp_path / "images")
    state = results / "deft_state.json"
    state.write_text(json.dumps({"iterations": {"iter1": {"anomalygen_manifest": str(manifest)}}}))
    rows = list(csv.DictReader(source.open()))
    for row in rows:
        for key in ("input_path", "golden_path"): row[key] = "images/" + row[key]
    assembled = results / "train.csv"; _write_rows(assembled, rows)
    proof = results / "proof.json"; provenance = results / "provenance.csv"; provenance.write_text("TEST_ONLY")
    args = argparse.Namespace(combined_csv=assembled, provenance_csv=provenance, merge_validation_report=proof)
    assert validator.main(["--csv", str(assembled), "--workspace-root", str(tmp_path), "--state", str(state), "--iter-label", "iter1", "--report-json", str(proof)]) == 0
    commit._apply_success({}, "data_merge", args, results, "iter1", {})
    for row in rows: row.pop("anomalygen_lineage")
    _write_rows(assembled, rows)
    # Omitting every synthetic option can validate an ordinary-looking CSV, but
    # the actual stage gate derives sources from state and must reject its proof.
    assert validator.main(["--csv", str(assembled), "--workspace-root", str(tmp_path), "--report-json", str(proof)]) == 0
    with pytest.raises(ValueError, match="synthetic sources"):
        commit._apply_success({}, "data_merge", args, results, "iter1", {})
    assert validator.main(["--csv", str(assembled), "--workspace-root", str(tmp_path), "--state", str(state), "--iter-label", "iter1", "--report-json", str(proof)]) == 2


def test_changenet_converted_pair_content_cannot_change_after_emission(tmp_path):
    manifest = native_run(tmp_path / "run")
    emitter = module("changenet_data_pair_prepare", "tao-run-deft-aoi")
    validator = module("validate_training_csv", "tao-run-deft-aoi")
    source = tmp_path / "synthetic.csv"; emitter.generate_native(manifest, source, images_dir=tmp_path / "images")
    Image.new("RGB", (8, 8), "red").save(tmp_path / "images/sdg_ng/IC+bridge_00004_SolderLight.jpg")
    assert validator.validate(source, tmp_path / "images")


def test_od_cumulative_admission_rechecks_retained_original_donor(tmp_path):
    import yaml
    runs, prepared = _prepared_od(tmp_path)
    emitter = module("emit_sdg_coco", "tao-run-deft-od-aoi")
    emitter.publish(runs, tmp_path / "emitted", prepared)
    coco = tmp_path / "emitted/pseudo_labels/coco_annotations_od_defect.json"
    document = json.loads(coco.read_text())
    for image in document["images"]:
        image["deft_kind"] = "synthetic_defect"
        image["source_path"] = image["file_name"]
    previous = tmp_path / "previous.json"; previous.write_text(json.dumps(document))
    fixtures = module("tests/test_admit_deft_od_aoi_coco", "tao-run-deft-od-aoi")
    policy, candidates, retrieval = fixtures._fixture(tmp_path)
    excluded = tmp_path / "excluded.txt"; excluded.write_text("protected_donor\n")
    value = yaml.safe_load(policy.read_text()); value["synthesis"]["exclude_stems"] = str(excluded); policy.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="excluded"):
        fixtures.MODULE.admit(policy, candidates, retrieval, tmp_path / "out", previous, "copy")
