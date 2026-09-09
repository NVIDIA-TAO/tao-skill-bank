# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

import yaml


SCRIPT = Path(__file__).parents[1] / "prepare_finetune_recipe.py"


def _fixture(root: Path) -> list[str]:
    dataset = root / "dataset"
    for path in (dataset / "texture" / "anomaly_image" / "defect" / "a.png",
                 dataset / "texture" / "mask" / "defect" / "a_mask.png",
                 dataset / "texture" / "clean_image" / "clean.png"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
    (dataset / "defect_spec.jsonl").write_text(
        json.dumps({"defect_type": "texture+defect", "spatial_dependency": "text",
                    "roi_prompt_defect_location": "on the surface"}) + "\n"
    )
    validation = root / "validation.jsonl"
    validation.write_text("".join(json.dumps({
        "image_filename": str(dataset / "texture" / "clean_image" / "clean.png"),
        "mask_filename": str(dataset / "texture" / "mask" / "defect" / "a_mask.png"),
        "anomaly_type": "texture+defect"}) + "\n" for _ in range(3)))
    base, nn = root / "Cosmos3-Nano", root / "dinov2-large"
    base.mkdir()
    nn.mkdir()
    vae = root / "Wan2.2_VAE.pth"
    vae.write_bytes(b"model")
    return ["--dataset-root", str(dataset), "--validation-testcase", str(validation),
            "--base-checkpoint", str(base), "--vae-path", str(vae),
            "--nn-backbone", str(nn), "--dataset-name", "fixture",
            "--output", str(root / "out" / "recipe.yaml")]


def test_prepare_preserves_custom_knobs_and_freezes_identities(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump({"anomaly_types": [["texture", "defect"]],
                                        "batch_size": 7, "custom_knob": "kept"}))
    result = subprocess.run([sys.executable, str(SCRIPT), *args,
                             "--recipe-template", str(template), "--max-iter", "1000"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    recipe = yaml.safe_load((tmp_path / "out" / "recipe.yaml").read_text())
    assert recipe["batch_size"] == 7 and recipe["custom_knob"] == "kept"
    assert recipe["dataset_name"] == "fixture"
    assert recipe["max_iter"] == recipe["save_iter"] == recipe["validation_iter"] == 1000
    assert recipe["run_validation_on_start"] is True
    rows = [json.loads(line) for line in
            (tmp_path / "out" / "recipe.validation.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert all(Path(row["image_filename"]).is_absolute() for row in rows)
    metadata = json.loads((tmp_path / "out" / "recipe.metadata.json").read_text())
    assert metadata["anomaly_types"] == ["texture+defect"]


def test_prepare_rejects_undercovered_validation(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    validation = Path(args[args.index("--validation-testcase") + 1])
    validation.write_text(validation.read_text().splitlines()[0] + "\n")
    result = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)
    assert result.returncode != 0
    assert "validation coverage mismatch" in result.stderr
