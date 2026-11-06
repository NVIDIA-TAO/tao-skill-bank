# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]


def module():
    path = SCRIPTS / "run_anomalygen.py"
    assert path.exists(), "Unified four-mode runner is required"
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("ag_run", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_generation_forwards_checkpoint_recipe_seed_and_native_guardrail(tmp_path):
    m = module()
    a = SimpleNamespace(
        repo=tmp_path,
        checkpoint=tmp_path / "nested/model/iter_000015000.pt",
        recipe=tmp_path / "training-copy.yaml",
        num_gpus=2,
        base_seed=7,
        base_checkpoint=None,
    )
    cmd = m.generation_command(a, tmp_path / "input.jsonl", tmp_path / "out")
    assert cmd[:3] == ["torchrun", "--standalone", "--nproc_per_node=2"]
    assert cmd[cmd.index("--checkpoint") + 1] == str(a.checkpoint)
    assert cmd[cmd.index("--recipe") + 1] == str(a.recipe)
    assert (
        cmd[cmd.index("--base_seed") + 1] == "7"
        and "--guardrail" in cmd
        and "--no-guardrail" not in cmd
    )


def test_training_uses_native_lifecycle_and_has_bounded_wait_slice(tmp_path):
    m = module()
    a = SimpleNamespace(repo=tmp_path, name="job", num_gpus=2)
    start, wait = m.training_commands(a, tmp_path / "recipe.yaml")
    assert start[0] == "bash" and "train_control.sh" in start[1] and "start" in start
    assert "--num_gpus" in start and start[start.index("--num_gpus") + 1] == "2"
    assert "wait" in wait and wait[wait.index("--timeout") + 1] == "30"


def test_all_four_modes_parse_and_source_is_explicit():
    m = module()
    for mode in ("prepare", "full", "finetune_only", "inference_only"):
        a = m.parser().parse_args([mode, "--name", "job", "--output-dir", "/out"])
        assert a.mode == mode and a.mask_backend == "amp"
    a = m.parser().parse_args(
        [
            "prepare",
            "--name",
            "job",
            "--output-dir",
            "/out",
            "--source",
            "od-gaps",
            "--prepare-stage",
            "finalize",
        ]
    )
    assert a.source == "od-gaps" and a.prepare_stage == "finalize"


def test_inference_rejects_missing_recipe_before_native_execution(tmp_path):
    m = module()
    a = m.parser().parse_args(
        [
            "inference_only",
            "--name",
            "job",
            "--output-dir",
            str(tmp_path / "out"),
            "--checkpoint",
            str(tmp_path),
        ]
    )
    with pytest.raises(ValueError, match="recipe"):
        m.validate_args(a)


def test_full_cannot_silently_reuse_an_inference_adapter(tmp_path):
    m = module()
    a = m.parser().parse_args(
        [
            "full",
            "--name",
            "job",
            "--output-dir",
            str(tmp_path / "out"),
            "--checkpoint",
            str(tmp_path),
        ]
    )
    with pytest.raises(ValueError, match="checkpoint|adapter"):
        m.validate_args(a)


def test_training_runtime_environment_preserves_parallelism_and_compile(tmp_path):
    m = module()
    a = m.parser().parse_args(
        [
            "finetune_only",
            "--name",
            "job",
            "--output-dir",
            str(tmp_path / "out"),
            "--parallelism",
            "fsdp",
            "--compile",
            "false",
        ]
    )
    assert m.training_environment(a) == {
        "ANOMALYGEN_PARALLELISM": "fsdp",
        "ANOMALYGEN_COMPILE": "0",
    }


@pytest.mark.parametrize("mode", ["prepare", "full", "finetune_only", "inference_only"])
@pytest.mark.parametrize("enabled", [False, True])
def test_structured_optional_flags_render_into_real_cli(mode, enabled):
    import shlex
    import yaml

    m = module()
    info = yaml.safe_load((SCRIPTS.parent / "references/skill_info.yaml").read_text())
    action = info["actions"][mode]
    values = {**action["defaults"], "name": "fixture", "results_dir": "/output/fixture"}
    for name in ("pseudo_label", "skip_quality"):
        assert name in action["args"], f"structured action omits {name}"
        if enabled:
            values[f"{name}_flag"] = "--" + name.replace("_", "-")
    argv = shlex.split(action["command"])[2:]
    for template in action["args"].values():
        try:
            rendered = template.format_map(values)
        except KeyError:
            continue  # Omitted optional input; its whole template is not supplied.
        argv.extend(shlex.split(rendered))
    parsed = m.parser().parse_args(argv)
    assert parsed.pseudo_label is enabled
    assert parsed.skip_quality is enabled
    assert parsed.mode == mode
    assert argv.count("--pseudo-label") == int(enabled)
    assert argv.count("--skip-quality") == int(enabled)


@pytest.mark.parametrize("mutation", ["before-copy", "copied-view", "before-evaluate"])
def test_frozen_reference_inventory_stops_before_quality(tmp_path, monkeypatch, mutation):
    import shutil
    from anomalygen_contract import dataset_inventory
    m = module()
    dataset = tmp_path / "dataset"
    image = dataset / "widget/clean_image/source.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fixture reference bytes")
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(dataset_inventory(dataset)))
    recipe = tmp_path / "recipe.yaml"; recipe.write_text("fixture")
    testcase = tmp_path / "testcase.jsonl"; testcase.write_text("fixture")
    exclusions = tmp_path / "exclude.txt"; exclusions.write_text("")
    args = m.parser().parse_args([
        "inference_only", "--name", "fixture", "--repo", str(tmp_path),
        "--output-dir", str(tmp_path / "out"), "--dataset-dir", str(dataset),
        "--recipe", str(recipe), "--checkpoint", str(tmp_path),
        "--input-data-path", str(testcase), "--exclude-stems", str(exclusions),
        "--expected-dataset-inventory", str(inventory)])
    original_copy = m.prepare_dataset_view
    def copy(*values):
        report = original_copy(*values)
        if mutation == "copied-view":
            (Path(values[1]) / "extra.png").write_bytes(b"extra")
        return report
    monkeypatch.setattr(m, "prepare_dataset_view", copy)
    monkeypatch.setattr(m, "filter_testcases", lambda source, target, *a, **kw: shutil.copyfile(source, target) and {})
    monkeypatch.setattr(m, "verify_generation", lambda *a: {"generated": 1})
    calls = []
    def execute(command, args):
        calls.append(command)
        assert len(calls) == 1, "changed references must never reach evaluate/refine"
        output = args.output_dir / "sdg"; output.mkdir()
        (output / "timing_summary.json").write_text(json.dumps({"guardrail_enabled": True}))
        (args.output_dir / "dataset-view/widget/clean_image/source.png").write_bytes(b"changed after generation")
    monkeypatch.setattr(m, "execute", execute)
    if mutation == "before-copy": image.write_bytes(b"changed original")
    with pytest.raises(ValueError, match="reference dataset"):
        m.run(args)
    assert len(calls) == int(mutation == "before-evaluate")
