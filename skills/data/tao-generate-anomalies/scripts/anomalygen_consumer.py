#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Application admission of native runs and source lineage, without model execution."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from anomalygen_contract import sha256, verify_generation, load_exclusions, _excluded, read_jsonl


def load_generation(manifest, exclude_stems=None):
    manifest = Path(manifest).resolve()
    value = json.loads(manifest.read_text())
    if value.get("status") != "COMPLETE" or value.get("guardrail_requested") is not True:
        raise ValueError("generation is not COMPLETE with guardrails requested")
    if value.get("native_timing", {}).get("guardrail_enabled") is not True:
        raise ValueError("native guardrail receipt is missing or disabled")
    testcase = Path(value["testcase"])
    if sha256(testcase) != value.get("testcase_sha256"):
        raise ValueError("generation testcase changed after completion")
    bucket = Path(value["selected_output"]).resolve()
    report = verify_generation(testcase, bucket, exclude_stems)
    recorded = value.get("refined", value.get("generation", {}))
    for key in ("testcase_sha256", "generation_csv_sha256", "blocked_csv_sha256", "generated", "guardrail_blocked", "testcases"):
        if recorded.get(key) != report[key]:
            raise ValueError(f"generation receipt changed: {key}")
    with (bucket / "texture_ft_generation_result.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["generated_image"] = str(bucket / "reconstructed_image" / row["output_filename"])
        row["paired_original"] = str(bucket / "original_image" / row["output_filename"])
        row["paired_mask"] = str(bucket / "original_mask" / row["output_filename"])
        row["lineage"] = {
            "manifest": str(manifest), "manifest_sha256": sha256(manifest),
            "index": int(row["index"]), "image_filename": row["image_filename"],
            "mask_filename": row["mask_filename"], "anomaly_type": row["anomaly_type"],
            "generated_image": row["generated_image"],
        }
    return {**report, "manifest": str(manifest), "native_timing": value["native_timing"]}, rows


def validate_lineage(lineage, forbidden=(), exclude_stems=None):
    if not isinstance(lineage, dict):
        raise ValueError("synthetic record lacks native source lineage")
    manifest = Path(lineage.get("manifest", ""))
    if not manifest.is_file() or sha256(manifest) != lineage.get("manifest_sha256"):
        raise ValueError("synthetic lineage manifest changed or missing")
    _, rows = load_generation(manifest, exclude_stems)
    if not any(row["lineage"] == lineage for row in rows):
        raise ValueError("synthetic lineage does not match a generated testcase")
    excluded = load_exclusions(exclude_stems)
    forbidden_paths = {str(Path(path).resolve()) for path in forbidden}
    forbidden_stems = {Path(path).stem for path in forbidden}
    for key in ("image_filename", "mask_filename", "generated_image"):
        path = lineage[key]
        if str(Path(path).resolve()) in forbidden_paths or _excluded(path, excluded | forbidden_stems):
            raise ValueError(f"synthetic source lineage overlaps forbidden evaluation data: {path}")


def write_emission_receipt(output, manifest, report):
    output = Path(output)
    receipt = {**report, "manifest": str(Path(manifest).resolve()), "output_sha256": sha256(output)}
    output.with_name(output.name + ".anomalygen.json").write_text(json.dumps(receipt, indent=2) + "\n")


def validate_empty_emission(path):
    path = Path(path)
    receipt = json.loads(path.with_name(path.name + ".anomalygen.json").read_text())
    report, _ = load_generation(receipt["manifest"])
    if report["generated"] or receipt.get("output_sha256") != sha256(path):
        raise ValueError("empty synthetic artifact lacks complete all-blocked evidence")
    return []


def checkpoint_binding(workspace, project, checkpoint=None, recipe=None, exclude_stems=None):
    if bool(checkpoint) != bool(recipe):
        raise ValueError("BYO checkpoint requires its exact training recipe")
    root = Path(workspace) / "augmentation/anomalygen/checkpoints" / project / "model"
    return {"checkpoint": str(Path(checkpoint or root / "iter_000015000.pt").resolve()),
            "recipe": str(Path(recipe or root / "exp_texture_ft_pcb.yaml").resolve()),
            "checkpoint_repo": None if checkpoint else "nvidia/Cosmos-AnomalyGen-PCB-Nano",
            "exclude_stems": str(Path(exclude_stems).resolve()) if exclude_stems else None,
            "quality_policy": "native_search_3_no_automatic_discard"}


def validate_sharegpt_emission(path, manifest, exclude_stems=None, media_root=None):
    report, rows = load_generation(manifest, exclude_stems)
    records = json.loads(Path(path).read_text())
    if not isinstance(records, list) or len(records) != len(rows):
        raise ValueError("ShareGPT generated count differs from native run")
    expected = {json.dumps(row["lineage"], sort_keys=True) for row in rows}
    actual = set()
    for record in records:
        lineage = record.get("anomalygen_lineage")
        validate_lineage(lineage, exclude_stems=exclude_stems)
        actual.add(json.dumps(lineage, sort_keys=True))
        row = next(row for row in rows if row["lineage"] == lineage)
        pair = record.get("images", [])
        if len(pair) != 2:
            raise ValueError("synthetic pair must contain exactly two images")
        resolved = [str((Path(media_root or Path.cwd()) / image).resolve()) for image in pair]
        if resolved != [row["generated_image"], row["paired_original"]]:
            raise ValueError("synthetic pair differs from native generated/original images")
        if record.get("conversations", [{}])[-1].get("value") != "NG":
            raise ValueError("synthetic label must remain bare NG")
    if actual != expected:
        raise ValueError("ShareGPT lineage does not cover generated identities")
    if not records:
        validate_empty_emission(path)
    return report


def validate_pair(record, media_root):
    lineage = record["anomalygen_lineage"]
    generated = Path(lineage["generated_image"])
    expected = [str(generated), str(generated.parent.parent / "original_image" / generated.name)]
    actual = [str((Path(media_root) / value).resolve()) for value in record.get("images", [])]
    if actual != expected:
        raise ValueError("synthetic pair differs from its native source lineage")


def validate_prepared_ancestry(binding, lineage=None, exclude_stems=None):
    """Recheck explicit original ancestry in the immutable OD prepared bundle."""
    if not isinstance(binding, dict):
        raise ValueError("OD synthetic image lacks prepared ancestry binding")
    prepared = Path(binding.get("manifest", ""))
    if not prepared.is_file() or sha256(prepared) != binding.get("manifest_sha256"):
        raise ValueError("prepared ancestry manifest changed or missing")
    value = json.loads(prepared.read_text())
    if value.get("status") != "COMPLETE" or value.get("generation_ready") is not True:
        raise ValueError("prepared ancestry is incomplete")
    artifacts = {str(Path(a["path"]).resolve()): a["sha256"] for a in value.get("artifacts", [])}
    for path, digest in artifacts.items():
        if not Path(path).is_file() or sha256(path) != digest:
            raise ValueError("prepared ancestry artifact changed")
    groups = [g for g in value.get("generator_groups", []) if g["dataset_id"] == binding.get("dataset_id")]
    if len(groups) != 1:
        raise ValueError("prepared ancestry group is missing or ambiguous")
    group = groups[0]
    for key in ("testcase", "provenance"):
        if str(Path(group.get(key, "")).resolve()) not in artifacts:
            raise ValueError("prepared ancestry lacks hash-bound testcase/provenance")
    cases, provenance = read_jsonl(group["testcase"]), read_jsonl(group["provenance"])
    if len(provenance) != len(cases) or {p["generation_index"] for p in provenance} != set(range(len(cases))):
        raise ValueError("prepared ancestry does not cover testcase identities")
    excluded = load_exclusions(exclude_stems)
    matched = lineage is None
    for row in provenance:
        ordinal = row["generation_index"]
        case = cases[ordinal]
        if str(Path(row["clean_filepath"]).resolve()) != str(Path(case["image_filename"]).resolve()) or str(Path(row["aligned_mask"]).resolve()) != str(Path(case["mask_filename"]).resolve()):
            raise ValueError("prepared ancestry does not match testcase source/mask")
        ancestry_keys = ("source_mask", "fn_filepath", "clean_filepath", "aligned_mask") + (("copied_mask",) if "copied_mask" in row else ())
        for key in ancestry_keys:
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError("prepared ancestry lacks explicit source/donor identity")
            if _excluded(row[key], excluded):
                raise ValueError(f"prepared ancestry uses excluded source/donor: {row[key]}")
        if not row.get("source_mask_sha256") or ("copied_mask" in row and not row.get("copied_mask_sha256")):
            raise ValueError("prepared ancestry lacks original/copied mask hashes")
        for key in ("source_mask", "aligned_mask", *(("copied_mask",) if "copied_mask" in row else ())):
            if row.get(key + "_sha256") and sha256(row[key]) != row[key + "_sha256"]:
                raise ValueError("prepared ancestry source/mask changed")
        if lineage is not None and case.get("index", ordinal) == lineage["index"]:
            if any(case[key] != lineage[key] for key in ("image_filename", "mask_filename", "anomaly_type")):
                raise ValueError("prepared ancestry differs from native lineage")
            matched = True
    if not matched:
        raise ValueError("native lineage has no prepared ancestry")
    if lineage is not None:
        native = json.loads(Path(lineage["manifest"]).read_text())
        if read_jsonl(native["testcase"]) != cases or any(str(Path(native[key]).resolve()) != str(Path(group[key]).resolve()) for key in ("checkpoint", "recipe")):
            raise ValueError("native run differs from prepared ancestry group")
    return binding
