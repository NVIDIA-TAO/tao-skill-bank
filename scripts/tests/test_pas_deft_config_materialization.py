# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for immutable PAS configuration materialization."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PAS_ROOT = REPO_ROOT / "skills" / "applications" / "tao-run-deft-pas"
PAS_SCRIPTS = PAS_ROOT / "scripts"
sys.path.insert(0, str(PAS_SCRIPTS))

import prepare_deft_config as prepare  # noqa: E402
import run_pas_stage  # noqa: E402
from command_contract import expected_fresh_outputs  # noqa: E402
from pas_deft.config import (  # noqa: E402
    DeftExperimentConfig,
    PasDeftConfig,
    config_field_metadata,
)
from pas_deft.pas_artifacts import (  # noqa: E402
    PAS_METRICS_AGGREGATE_FILENAME,
    PAS_METRICS_FILENAME,
)


def _base_argv(tmp_path: Path) -> tuple[list[str], Path, Path]:
    workspace = tmp_path / "workspace"
    dataset = workspace / "data" / "pas_v31_tao_ft"
    results = workspace / "results" / "run"
    dataset.mkdir(parents=True)
    images_archive = tmp_path / "images_raw.tar"
    metadata_archive = tmp_path / "meta.tar.gz"
    images_archive.write_bytes(b"images")
    metadata_archive.write_bytes(b"metadata")
    return (
        [
            "--workspace",
            str(workspace),
            "--results-dir",
            str(results),
            "--dataset-root",
            str(dataset),
            "--images-archive",
            str(images_archive),
            "--images-archive-sha256",
            hashlib.sha256(images_archive.read_bytes()).hexdigest(),
            "--metadata-archive",
            str(metadata_archive),
            "--metadata-archive-sha256",
            hashlib.sha256(metadata_archive.read_bytes()).hexdigest(),
            "--platform",
            "docker",
            "--max-iterations",
            "10",
        ],
        results,
        dataset,
    )


def _materialize(tmp_path: Path, *extra: str) -> tuple[dict, Path, Path]:
    argv, results, dataset = _base_argv(tmp_path)
    args = prepare._parser().parse_args([*argv, *extra])  # noqa: SLF001
    return prepare.materialize(args), results, dataset


def _yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text())
    assert isinstance(payload, dict)
    return payload


def _init_command(results: Path, dataset: Path, approval: dict) -> list[str]:
    return [
        sys.executable,
        str(PAS_SCRIPTS / "init_deft_state.py"),
        "--results-dir",
        str(results),
        "--workspace",
        approval["workspace"],
        "--dataset-root",
        str(dataset),
        "--images-archive",
        approval["images_archive"],
        "--images-archive-sha256",
        approval["images_archive_sha256"],
        "--metadata-archive",
        approval["metadata_archive"],
        "--metadata-archive-sha256",
        approval["metadata_archive_sha256"],
        "--max-iterations",
        "10",
        "--metric-name",
        "Rank-1",
        "--metric-query-type",
        "medium",
        "--metric-op",
        ">=",
        "--platform",
        "docker",
        "--pyt-image",
        approval["pyt_image"],
        "--ds-image",
        approval["ds_image"],
        "--deft-config",
        str(results / "config" / "deft_config.yaml"),
        "--tao-spec",
        str(results / "config" / "tao_spec.yaml"),
    ]


def test_pas_notebook_controls_are_materialized_without_semantic_drift(tmp_path):
    report, results, dataset = _materialize(
        tmp_path,
        "--mining-topn",
        "25",
        "--knn-metric",
        "cosine",
        "--target-query-count",
        "600",
        "--queries-per-slice",
        "50",
        "--gap-query-types",
        "easy",
        "--eval-split",
        "test",
        "--vision-lr",
        "4.5e-7",
        "--text-lr",
        "4.5e-7",
        "--train-batch-size",
        "32",
        "--val-batch-size",
        "32",
        "--eval-batch-size",
        "32",
        "--text-embed-model",
        "SigLIP",
    )

    config = results / "config"
    deft = _yaml(config / "deft_config.yaml")
    tao = _yaml(config / "tao_spec.yaml")
    text_embed = _yaml(config / "text_embed_spec.yaml")
    image_embed = _yaml(config / "image_embed_spec.yaml")
    mining = _yaml(config / "mining_spec.yaml")

    assert deft["pas"]["eval_pairs_source_file"] == str(dataset / "test_pairs.json")
    assert deft["gap_analysis"]["queries_per_slice"] == 50
    assert deft["gap_analysis"]["query_types"] == "easy"
    assert tao["train"]["optim"]["vision_lr"] == pytest.approx(4.5e-7)
    assert tao["train"]["optim"]["text_lr"] == pytest.approx(4.5e-7)
    assert tao["dataset"]["train"]["batch_size"] == 32
    assert tao["dataset"]["val"]["batch_size"] == 32
    assert tao["evaluate"]["batch_size"] == 32
    assert text_embed["model"] == "SigLIP"
    assert image_embed["model"] == "SigLIP"
    assert mining["topn"] == 25
    assert mining["knn_metric"] == "cosine"
    assert "topn" not in deft["mining"]
    assert "knn_metric" not in deft["mining"]
    assert report["eval_split"] == "test"
    assert report["text_embed_model"] == "SigLIP"


def test_mining_spec_is_the_only_materialized_mining_parameter_authority(tmp_path):
    _, results, _ = _materialize(tmp_path)
    deft = _yaml(results / "config" / "deft_config.yaml")
    mining = _yaml(results / "config" / "mining_spec.yaml")
    assert (mining["topn"], mining["knn_metric"]) == (25, "cosine")
    assert "topn" not in deft["mining"]
    assert "knn_metric" not in deft["mining"]


def test_pas_uses_workflow_scoped_tao_7_2_image_contract():
    versions = _yaml(REPO_ROOT / "versions.yaml")
    tao_images = versions["images"]["tao_toolkit"]
    assert prepare.PINNED_PYT_IMAGE == tao_images["deft_pas_pyt"]
    assert prepare.PINNED_DS_IMAGE == tao_images["deft_pas_data_services"]
    assert ":7.2.0-" in prepare.PINNED_PYT_IMAGE
    assert ":7.2.0-" in prepare.PINNED_DS_IMAGE


def test_tao_7_2_pas_evaluation_artifacts_are_the_runtime_contract(tmp_path):
    expected = expected_fresh_outputs("evaluate", "baseline", tmp_path)
    assert expected == [
        tmp_path / "zs" / "evaluate" / PAS_METRICS_AGGREGATE_FILENAME,
        tmp_path / "zs" / "evaluate" / "status.json",
    ]
    assert PAS_METRICS_FILENAME == "nvidia_pas_metrics.csv"
    assert PAS_METRICS_AGGREGATE_FILENAME == "nvidia_pas_metrics_aggregate.csv"
    assert all("nvidia_iaa_metrics" not in str(path) for path in expected)


@pytest.mark.parametrize(
    "extra",
    [
        ("--gap-query-types", "easy,easy"),
        ("--gap-query-types", "unknown"),
        ("--vision-lr", "0"),
        ("--queries-per-slice", "0"),
    ],
)
def test_invalid_new_controls_are_rejected(tmp_path, extra):
    argv, _, _ = _base_argv(tmp_path)
    args = prepare._parser().parse_args([*argv, *extra])  # noqa: SLF001
    with pytest.raises(ValueError):
        prepare.materialize(args)


def test_test_split_survives_state_initialization_and_audit(tmp_path):
    report, results, dataset = _materialize(
        tmp_path,
        "--eval-split",
        "test",
        "--queries-per-slice",
        "50",
        "--gap-query-types",
        "easy",
        "--vision-lr",
        "4.5e-7",
        "--text-lr",
        "4.5e-7",
        "--val-batch-size",
        "32",
        "--text-embed-model",
        "SigLIP",
    )
    config_path = results / "config" / "deft_config.yaml"
    payload = _yaml(config_path)
    payload["visualization"] = {"enabled": False, "embeddings": False}
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    approval = json.loads((results / "config" / "approval.json").read_text())
    init = subprocess.run(
        _init_command(results, dataset, approval),
        check=False,
        capture_output=True,
        text=True,
    )
    assert init.returncode == 0, init.stderr
    state = json.loads((results / "deft_state.json").read_text())
    assert state["config"]["eval_split"] == "test"
    assert state["config"]["queries_per_slice"] == 50
    assert state["config"]["text_embed_model"] == "SigLIP"
    assert state["config"]["visualize"] is False
    assert state["config"]["visualize_embeddings"] is False

    audit = subprocess.run(
        [
            sys.executable,
            str(PAS_SCRIPTS / "audit_deft_run.py"),
            "--results-dir",
            str(results),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert audit.returncode == 0, audit.stderr
    assert "DEFT_RUN_STATUS=IN_PROGRESS" in audit.stdout

    typed = run_pas_stage._config(  # noqa: SLF001
        config_path,
        results,
    )
    assert isinstance(typed, PasDeftConfig)
    assert isinstance(typed.cfg, DeftExperimentConfig)
    assert dataclasses.is_dataclass(typed.cfg)
    assert typed.pas.eval_pairs_source_file == str(dataset / "test_pairs.json")
    assert typed.gap_analysis.queries_per_slice == 50
    assert typed.mining.topn == 25
    assert typed.mining.knn_metric == "cosine"


def test_schema_v3_approval_remains_valid_as_local_docker(tmp_path):
    _, results, dataset = _materialize(tmp_path)
    approval_path = results / "config" / "approval.json"
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    approval["schema_version"] = "3"
    approval.pop("platform")
    approval.pop("docker_remote")
    approval.pop("virtualenvs")
    approval_path.write_text(json.dumps(approval, indent=2) + "\n", encoding="utf-8")

    init = subprocess.run(
        _init_command(results, dataset, approval),
        check=False,
        capture_output=True,
        text=True,
    )

    assert init.returncode == 0, init.stderr
    state = json.loads((results / "deft_state.json").read_text(encoding="utf-8"))
    assert state["config"]["platform"] == "docker"
    assert state["config"]["docker_remote"] is False
    audit = subprocess.run(
        [
            sys.executable,
            str(PAS_SCRIPTS / "audit_deft_run.py"),
            "--results-dir",
            str(results),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert audit.returncode == 0, audit.stderr
    assert "DEFT_RUN_STATUS=IN_PROGRESS" in audit.stdout


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("model", "SigLIP2", "image_embed_spec.model must match"),
        ("model_path", "different/checkpoint", "model_path values must be the same"),
    ],
)
def test_image_and_text_embedding_specs_cannot_drift(
    tmp_path, field, replacement, message
):
    _, results, dataset = _materialize(tmp_path)
    config = results / "config"
    text_embed = _yaml(config / "text_embed_spec.yaml")
    image_embed = _yaml(config / "image_embed_spec.yaml")
    assert text_embed["model"] == "SigLIP"
    assert image_embed["model"] == text_embed["model"]

    image_embed[field] = replacement
    (config / "image_embed_spec.yaml").write_text(
        yaml.safe_dump(image_embed, sort_keys=False)
    )
    approval = json.loads((config / "approval.json").read_text())
    init = subprocess.run(
        _init_command(results, dataset, approval),
        check=False,
        capture_output=True,
        text=True,
    )
    assert init.returncode != 0
    assert message in init.stderr


def test_materializer_rejects_text_only_model_token_for_shared_embeddings(tmp_path):
    argv, _, _ = _base_argv(tmp_path)
    with pytest.raises(SystemExit):
        prepare._parser().parse_args(  # noqa: SLF001
            [*argv, "--text-embed-model", "SigLIP2"]
        )


def test_dataclass_schema_documents_all_runtime_fields_and_constraints():
    metadata = config_field_metadata()
    assert len(metadata) >= 70
    assert all(values["description"] for values in metadata.values())
    assert metadata["iteration.start"]["valid_min"] == 1
    assert metadata["mining.history_aware.replay_fraction"]["valid_max"] == 1.0
    assert metadata["mining.knn_metric"]["valid_options"] == "cosine,euclidean"
    assert metadata["gap_analysis.metric_name"]["valid_options"] == (
        "mAP,Rank-1,Rank-5,Separability,Match@5,Zero@5"
    )
    assert metadata["gap_analysis.queries_per_slice"]["valid_min"] == 0
    assert metadata["gap_analysis.weak_attribute_topk"]["valid_min"] == 0
    assert metadata["gap_analysis.target_query_count"]["valid_min"] == 0
    assert metadata["gap_analysis.caption_diversity.coverage_target"]["valid_max"] == 1.0
    assert (
        metadata["gap_analysis.caption_diversity.history_file"]["valid_options"]
        == "caption_selection_history.json"
    )
    assert (
        metadata["pas.mining_pool_mode"]["valid_options"]
        == "real,augmented,real_and_augmented"
    )


def test_typed_runtime_does_not_require_omegaconf(tmp_path):
    _, results, _ = _materialize(tmp_path)
    config_path = results / "config" / "deft_config.yaml"
    probe = """
import importlib.abc
import sys

class BlockOmegaConf(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "omegaconf" or fullname.startswith("omegaconf."):
            raise ModuleNotFoundError("omegaconf intentionally unavailable")
        return None

sys.meta_path.insert(0, BlockOmegaConf())
sys.path.insert(0, sys.argv[1])
from pas_deft.config import PasDeftConfig

config = PasDeftConfig(sys.argv[2])
assert config.iteration.start == 1
assert config.mining.knn_metric == "cosine"
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(PAS_SCRIPTS), str(config_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_runtime_interpreter_probe_does_not_require_omegaconf(tmp_path, monkeypatch):
    stubs = tmp_path / "runtime-dependencies"
    stubs.mkdir()
    for module_name in (
        "pandas",
        "numpy",
        "pyarrow",
        "PIL",
        "yaml",
        "matplotlib",
        "sklearn",
        "torch",
    ):
        (stubs / f"{module_name}.py").write_text("")
    (stubs / "omegaconf.py").write_text(
        'raise ModuleNotFoundError("omegaconf intentionally unavailable")\n'
    )
    monkeypatch.setenv("DEFT_PYTHON", sys.executable)
    monkeypatch.setenv("PYTHONPATH", str(stubs))

    completed = subprocess.run(
        [
            str(PAS_SCRIPTS / "deft_python.sh"),
            "--runtime",
            "--workspace",
            str(tmp_path),
            "-c",
            "print('RUNTIME_INTERPRETER_SELECTED')",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "RUNTIME_INTERPRETER_SELECTED"


@pytest.mark.parametrize(
    ("section", "field", "value", "expected"),
    [
        ("iteration", "start", 1.5, "cannot be converted to int"),
        ("training", "continual_model", "sometimes", "cannot be converted to bool"),
        ("gap_analysis", "metric_name", None, "cannot be null"),
    ],
)
def test_typed_runtime_rejects_values_outside_declared_scalar_types(
    tmp_path,
    section,
    field,
    value,
    expected,
):
    _, results, _ = _materialize(tmp_path)
    config_path = results / "config" / "deft_config.yaml"
    payload = _yaml(config_path)
    payload[section][field] = value
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match=expected):
        PasDeftConfig(str(config_path))


@pytest.mark.parametrize(
    ("section_path", "key"),
    [
        ((), "unexpected_root"),
        (("experiment",), "unexpected_experiment"),
        (("mining", "recovery"), "unexpected_recovery"),
        (("gap_analysis", "caption_diversity"), "unexpected_diversity"),
    ],
)
def test_typed_runtime_rejects_unknown_keys(tmp_path, section_path, key):
    _, results, _ = _materialize(tmp_path)
    config_path = results / "config" / "deft_config.yaml"
    payload = _yaml(config_path)
    section = payload
    for part in section_path:
        section = section[part]
    section[key] = "silently ignored before dataclass validation"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match="unknown key"):
        PasDeftConfig(str(config_path))


@pytest.mark.parametrize(
    ("deft_path", "mining_value", "expected"),
    [
        (("gap_analysis", "queries_per_slice"), -1, "minimum allowed value"),
        (("gap_analysis", "metric_name"), "accuracy", "must be one of"),
        (("mining", "topn"), 0, "minimum allowed value"),
        (("mining", "knn_metric"), "manhattan", "must be one of"),
        (("pas", "mining_pool_mode"), "automatic", "must be one of"),
    ],
)
def test_typed_runtime_enforces_declared_bounds_and_options(
    tmp_path,
    deft_path,
    mining_value,
    expected,
):
    _, results, _ = _materialize(tmp_path)
    config_path = results / "config" / "deft_config.yaml"
    payload = _yaml(config_path)
    section = payload
    for part in deft_path[:-1]:
        section = section[part]
    value = mining_value
    section[deft_path[-1]] = value
    if deft_path[:1] == ("mining",) and deft_path[-1] in {"topn", "knn_metric"}:
        mining_spec_path = results / "config" / "mining_spec.yaml"
        mining_spec = _yaml(mining_spec_path)
        mining_spec[deft_path[-1]] = value
        mining_spec_path.write_text(yaml.safe_dump(mining_spec, sort_keys=False))
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match=expected):
        PasDeftConfig(str(config_path))


def test_typed_runtime_rejects_cross_spec_mining_conflicts(tmp_path):
    _, results, _ = _materialize(tmp_path)
    config_path = results / "config" / "deft_config.yaml"
    payload = _yaml(config_path)
    payload["mining"]["topn"] = 25
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))

    mining_spec_path = results / "config" / "mining_spec.yaml"
    mining_spec = _yaml(mining_spec_path)
    mining_spec["topn"] = mining_spec["topn"] + 1
    mining_spec_path.write_text(yaml.safe_dump(mining_spec, sort_keys=False))

    with pytest.raises(ValueError, match="conflicts with mining_spec"):
        PasDeftConfig(str(config_path))


@pytest.mark.parametrize(
    ("canonical", "legacy", "canonical_value", "legacy_value"),
    [
        ("total_queries_map", "total_queries_mAP", 768, 512),
        ("analyze_by_map", "analyze_by_mAP", False, True),
    ],
)
def test_typed_runtime_rejects_conflicting_gap_aliases(
    tmp_path,
    canonical,
    legacy,
    canonical_value,
    legacy_value,
):
    _, results, _ = _materialize(tmp_path)
    config_path = results / "config" / "deft_config.yaml"
    payload = _yaml(config_path)
    payload["gap_analysis"][canonical] = canonical_value
    payload["gap_analysis"][legacy] = legacy_value
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match="conflicts with legacy"):
        PasDeftConfig(str(config_path))


def test_typed_runtime_accepts_equal_gap_aliases(tmp_path):
    _, results, _ = _materialize(tmp_path)
    config_path = results / "config" / "deft_config.yaml"
    payload = _yaml(config_path)
    payload["gap_analysis"].update(
        {
            "total_queries_map": 512,
            "total_queries_mAP": 512,
            "analyze_by_map": True,
            "analyze_by_mAP": True,
        }
    )
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))

    typed = PasDeftConfig(str(config_path))
    assert typed.gap_analysis.total_queries_map == 512
    assert typed.gap_analysis.analyze_by_map is True


def test_typed_runtime_rejects_missing_required_pas_path(tmp_path):
    _, results, _ = _materialize(tmp_path)
    config_path = results / "config" / "deft_config.yaml"
    payload = _yaml(config_path)
    del payload["pas"]["eval_pairs_source_file"]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match="missing|Missing|mandatory"):
        PasDeftConfig(str(config_path))


def test_typed_runtime_preserves_boolean_semantics_during_normalization(tmp_path):
    _, results, _ = _materialize(tmp_path)
    config_path = results / "config" / "deft_config.yaml"
    payload = _yaml(config_path)
    payload["experiment"]["visualize"] = "false"
    payload["experiment"]["visualize_embeddings"] = "false"
    payload.pop("visualization", None)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))

    typed = PasDeftConfig(str(config_path))
    assert typed.experiment.visualize is False
    assert typed.experiment.visualize_embeddings is False
    assert typed.visualization.enabled is False
    assert typed.visualization.embeddings is False


def test_typed_runtime_retains_notebook_zero_means_unlimited_controls(tmp_path):
    _, results, _ = _materialize(tmp_path)
    config_path = results / "config" / "deft_config.yaml"
    payload = _yaml(config_path)
    payload["gap_analysis"].update(
        {
            "queries_per_slice": 0,
            "weak_attribute_topk": 0,
            "target_query_count": 0,
        }
    )
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))

    typed = PasDeftConfig(str(config_path))
    assert typed.gap_analysis.queries_per_slice == 0
    assert typed.gap_analysis.weak_attribute_topk == 0
    assert typed.gap_analysis.target_query_count == 0


def test_gap_stage_consumes_typed_caption_history_path(tmp_path, monkeypatch):
    _, results, _ = _materialize(tmp_path)
    config_path = results / "config" / "deft_config.yaml"
    typed = PasDeftConfig(str(config_path))

    import pas_deft.analyze_gaps as gap_module
    import pas_deft.utils as utils_module

    captured = {}

    def fake_analyze(**kwargs):
        captured.update(kwargs)
        gaps = Path(kwargs["gaps_parquet"])
        gaps.parent.mkdir(parents=True, exist_ok=True)
        gaps.write_bytes(b"parquet")

    monkeypatch.setattr(run_pas_stage, "_config", lambda *_: typed)
    monkeypatch.setattr(run_pas_stage, "_state", lambda *_: {"max_iterations": 10})
    monkeypatch.setattr(gap_module, "analyze_clip_inference_gaps", fake_analyze)
    monkeypatch.setattr(utils_module, "resolve_prev_eval_dir", lambda **_: "previous-eval")

    run_pas_stage.gap_analysis(
        types.SimpleNamespace(
            results_dir=results,
            deft_config=config_path,
            iter_num=1,
        )
    )

    assert captured["caption_history_file"] == str(
        results / "caption_selection_history.json"
    )


@pytest.mark.parametrize("history_file", ["", "../outside.json", "/tmp/outside.json"])
def test_caption_history_path_cannot_escape_results(tmp_path, history_file):
    _, results, _ = _materialize(tmp_path)
    config_path = results / "config" / "deft_config.yaml"
    payload = _yaml(config_path)
    payload["gap_analysis"]["caption_diversity"]["history_file"] = history_file
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match="history_file"):
        PasDeftConfig(str(config_path))
