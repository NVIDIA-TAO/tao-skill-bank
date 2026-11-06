# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import csv
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]


def module():
    path = SCRIPTS / "anomalygen_contract.py"
    assert path.exists(), "The unified AnomalyGen input/output contract must exist"
    spec = importlib.util.spec_from_file_location("ag_contract", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def inputs(tmp_path):
    rows = []
    for i in range(3):
        image = tmp_path / f"clean{i}.png"
        image.write_bytes(b"image")
        mask = tmp_path / f"mask{i}.png"
        mask.write_bytes(b"mask")
        rows.append(
            {
                "image_filename": str(image),
                "mask_filename": str(mask),
                "anomaly_type": "Board+scratch",
                "num_generated_images": 1,
            }
        )
    path = tmp_path / "testcase.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path, rows


def csv_file(path, rows, header):
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def output(tmp_path, rows):
    root = tmp_path / "output"
    (root / "reconstructed_image").mkdir(parents=True)
    generated = []
    for i in [0, 2]:
        name = f"Board+scratch_{i}.png"
        (root / "reconstructed_image" / name).write_bytes(b"image")
        generated.append({"output_filename": name, **rows[i], "index": i})
    csv_file(
        root / "texture_ft_generation_result.csv",
        generated,
        ["output_filename", *rows[0], "index"],
    )
    csv_file(
        root / "guardrail_blocked.csv",
        [
            {
                "index": 1,
                "output_idx": 0,
                **{
                    k: rows[1][k]
                    for k in ("anomaly_type", "image_filename", "mask_filename")
                },
                "guardrail": "text",
                "message": "blocked",
            }
        ],
        [
            "index",
            "output_idx",
            "anomaly_type",
            "image_filename",
            "mask_filename",
            "guardrail",
            "message",
        ],
    )
    return root


def test_exact_accounting_accepts_a_native_blocked_row(tmp_path):
    m = module()
    source, rows = inputs(tmp_path)
    root = output(tmp_path, rows)
    result = m.verify_generation(source, root)
    assert (
        result["generated"] == 2
        and result["guardrail_blocked"] == 1
        and result["testcases"] == 3
    )
    assert result["status"] == "COMPLETE"


def test_matching_total_cannot_hide_duplicate_and_missing_indices(tmp_path):
    m = module()
    source, rows = inputs(tmp_path)
    root = output(tmp_path, rows)
    path = root / "guardrail_blocked.csv"
    path.write_text(path.read_text().replace("\n1,0,", "\n0,0,"))
    with pytest.raises(ValueError, match="duplicate|overlap|identity"):
        m.verify_generation(source, root)


def test_missing_image_is_not_a_guardrail_block(tmp_path):
    m = module()
    source, rows = inputs(tmp_path)
    root = output(tmp_path, rows)
    (root / "reconstructed_image/Board+scratch_2.png").unlink()
    with pytest.raises(ValueError, match="image|manifest"):
        m.verify_generation(source, root)


def test_extra_image_and_path_escape_are_refused(tmp_path):
    m = module()
    source, rows = inputs(tmp_path)
    root = output(tmp_path, rows)
    (root / "reconstructed_image/extra.png").write_bytes(b"image")
    with pytest.raises(ValueError, match="image|manifest"):
        m.verify_generation(source, root)


def test_wrong_source_identity_is_refused_even_with_matching_counts(tmp_path):
    m = module()
    source, rows = inputs(tmp_path)
    root = output(tmp_path, rows)
    path = root / "guardrail_blocked.csv"
    path.write_text(
        path.read_text().replace(rows[1]["image_filename"], rows[0]["image_filename"])
    )
    with pytest.raises(ValueError, match="identity|source"):
        m.verify_generation(source, root)


def test_exclusion_filters_native_testcases_and_records_source_indices(tmp_path):
    m = module()
    source, rows = inputs(tmp_path)
    exclude = tmp_path / "exclude.txt"
    exclude.write_text("clean1\n")
    dest = tmp_path / "filtered.jsonl"
    report = m.filter_testcases(source, dest, exclude)
    assert [r["image_filename"] for r in m.read_jsonl(dest)] == [
        rows[0]["image_filename"],
        rows[2]["image_filename"],
    ]
    assert report["excluded"] == 1 and report["retained_source_indices"] == [0, 2]
    assert report["exclude_stems_sha256"]


def test_exclusions_are_enforced_at_generated_output_boundary(tmp_path):
    m = module()
    source, rows = inputs(tmp_path)
    root = output(tmp_path, rows)
    exclude = tmp_path / "exclude.txt"
    exclude.write_text("clean2\n")
    with pytest.raises(ValueError, match="excluded"):
        m.verify_generation(source, root, exclude)


def test_dataset_view_removes_clean_cad_anomaly_and_paired_mask_without_mutation(
    tmp_path,
):
    m = module()
    source = tmp_path / "dataset"
    keep = source / "Board/clean_image/keep.png"
    paths = [
        "Board/clean_image/hold.png",
        "Board/cad_mask/hold.png",
        "Board/anomaly_image/scratch/hold.jpg",
        "Board/mask/scratch/hold_mask.png",
        "Board/clean_image/keep.png",
    ]
    for name in paths:
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"image")
    exclude = tmp_path / "exclude.txt"
    exclude.write_text("hold\n")
    dest = tmp_path / "view"
    report = m.prepare_dataset_view(source, dest, exclude)
    assert list(p.relative_to(dest).as_posix() for p in dest.rglob("*.png")) == [
        "Board/clean_image/keep.png"
    ]
    assert all((source / p).exists() for p in paths) and report["excluded_files"] == 4


def test_all_blocked_batch_is_complete_but_not_a_training_dataset(tmp_path):
    m = module()
    source, rows = inputs(tmp_path)
    root = tmp_path / "out"
    (root / "reconstructed_image").mkdir(parents=True)
    csv_file(
        root / "texture_ft_generation_result.csv",
        [],
        ["output_filename", "image_filename", "mask_filename", "anomaly_type", "index"],
    )
    blocked = [
        {"index": i, "output_idx": 0, **r, "guardrail": "text", "message": "blocked"}
        for i, r in enumerate(rows)
    ]
    csv_file(
        root / "guardrail_blocked.csv",
        blocked,
        ["index", "output_idx", *rows[0], "guardrail", "message"],
    )
    result = m.verify_generation(source, root)
    assert (
        result["generated"] == 0
        and result["guardrail_blocked"] == 3
        and result["training_ready"] is False
    )


def test_native_text_block_uses_output_idx_minus_one(tmp_path):
    m = module()
    source, rows = inputs(tmp_path)
    root = output(tmp_path, rows)
    path = root / "guardrail_blocked.csv"
    path.write_text(path.read_text().replace("\n1,0,", "\n1,-1,"))
    assert m.verify_generation(source, root)["guardrail_blocked"] == 1


def test_native_preset_indices_need_not_be_contiguous(tmp_path):
    m = module()
    source, rows = inputs(tmp_path)
    root = output(tmp_path, rows)
    for i, row in enumerate(rows):
        row["index"] = 10 + i
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    for name in ("texture_ft_generation_result.csv", "guardrail_blocked.csv"):
        path = root / name
        with path.open() as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames
            records = list(reader)
        for row in records:
            row["index"] = int(row["index"]) + 10
        csv_file(path, records, header)
    assert m.verify_generation(source, root)["generated"] == 2


def test_relative_testcase_paths_resolve_against_native_repo(tmp_path):
    m = module()
    repo = tmp_path / "native"
    repo.mkdir()
    source, rows = inputs(repo)
    for row in rows:
        row["image_filename"] = Path(row["image_filename"]).name
        row["mask_filename"] = Path(row["mask_filename"]).name
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    target = tmp_path / "canonical.jsonl"
    m.filter_testcases(source, target, base_dir=repo)
    actual = m.read_jsonl(target)
    assert actual[0]["image_filename"] == str(repo / "clean0.png")
    assert actual[0]["mask_filename"] == str(repo / "mask0.png")


def test_mask_donors_are_also_excluded_from_testcases(tmp_path):
    m = module()
    source, rows = inputs(tmp_path)
    excluded = tmp_path / "excluded.txt"
    excluded.write_text("mask1\n")
    target = tmp_path / "filtered.jsonl"
    report = m.filter_testcases(source, target, excluded)
    assert report["retained"] == 2 and report["retained_source_indices"] == [0, 2]
    with pytest.raises(ValueError, match="excluded"):
        m.validate_testcases(rows, {"mask1"})


def test_refinement_preserves_initial_blocks_without_retrying_them(tmp_path):
    import shutil

    m = module()
    source, rows = inputs(tmp_path)
    original = output(tmp_path, rows)
    selected = tmp_path / "searched"
    shutil.copytree(original, selected)
    (selected / "guardrail_blocked.csv").unlink()
    eligible = tmp_path / "search-testcase.jsonl"
    report = m.prepare_refinement(source, original, eligible)
    assert [r["index"] for r in m.read_jsonl(eligible)] == [0, 2]
    assert report["eligible"] == 2 and report["blocked_not_retried"] == 1
    result = m.finalize_refinement(source, original, selected)
    assert result["generated"] == 2 and result["guardrail_blocked"] == 1
    assert (selected / "guardrail_blocked.csv").read_bytes() == (
        original / "guardrail_blocked.csv"
    ).read_bytes()
