# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the portable Docker test handoff."""

import argparse
import importlib.util
import json
import io
import tarfile
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

MODULE = Path(__file__).resolve().parents[1] / "deft_container_handoff.py"
SPEC = importlib.util.spec_from_file_location("deft_container_handoff", MODULE)
handoff = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(handoff)


def args(tmp_path, dry_run=False):
    return argparse.Namespace(skill_bank=tmp_path / "bank", pytorch=tmp_path / "pyt",
                              data_services=tmp_path / "ds", output=tmp_path / "output",
                              pytorch_image="registry/pyt:reviewed", data_services_image="registry/ds:reviewed",
                              suite="source", gpus="device=1", timeout=10, dry_run=dry_run, core=None, require_gpu_faiss=False, base_ref="origin/main")


def test_dry_run_needs_no_docker_or_writes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(handoff, "source_info", lambda *_: {"commit": "abc", "base": "def", "changed_files": ["x.py"]})
    monkeypatch.setattr(handoff, "capture", lambda *_: pytest.fail("dry-run invoked Docker"))
    assert handoff.run(args(tmp_path, True)) == 0
    assert not (tmp_path / "output").exists()
    assert json.loads(capsys.readouterr().out)["gpus"] == "device=1"


def test_source_info_uses_full_feature_merge_base(tmp_path, monkeypatch):
    calls = []

    def capture(argv, cwd=None):
        calls.append(argv)
        if argv[1:3] == ["status", "--porcelain"]:
            return ""
        if argv[1:3] == ["rev-parse", "HEAD"]:
            return "feature-tip"
        if argv[1:3] == ["merge-base", "origin/main"]:
            return "main-base"
        if argv[1:3] == ["diff", "--name-only"]:
            assert argv[-2:] == ["main-base", "feature-tip"]
            return "first.py\nsecond.py"
        raise AssertionError(argv)

    monkeypatch.setattr(handoff, "capture", capture)
    assert handoff.source_info(tmp_path) == {
        "commit": "feature-tip",
        "base": "main-base",
        "changed_files": ["first.py", "second.py"],
    }
    assert any(call[1:3] == ["merge-base", "origin/main"] for call in calls)



def test_dirty_checkout_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(handoff, "capture", lambda *_: " M source.py")
    with pytest.raises(ValueError, match="Commit or stash"):
        handoff.source_info(tmp_path)


def test_regression_selection_includes_changed_pytorch_surfaces():
    command = handoff.regression_commands("tao-pytorch", "python")[0]
    assert "tests/ssl_unit_test/dinov3" in command
    assert "tests/core" in command
    assert "tests/distributed" in command
    assert "tests/ssl_unit_test/nvdinov2/test_model.py" in command
    assert "tests/ssl_unit_test/nvdinov2/test_dataloader.py" in command


def test_data_services_source_tests_prefer_candidate_tao_dependencies(tmp_path):
    (tmp_path / "tao-core").mkdir()
    paths = handoff.source_pythonpath("tao-data-services", tmp_path).split(
        handoff.os.pathsep
    )
    assert paths == [
        str(tmp_path / "tao-data-services"),
        str(tmp_path / "tao-pytorch"),
        str(tmp_path / "tao-core"),
    ]
    command = handoff.regression_commands("tao-data-services", "python")[0]
    assert "../tao-skill-bank/scripts/tests/test_dinov3_cross_repo_smoke.py" in command


def test_only_data_services_requires_cross_repo_smoke_imports(
    tmp_path, monkeypatch
):
    """Candidate TAO import failures must fail the DS handoff, not skip."""
    monkeypatch.setenv("TAO_DEFT_REQUIRE_CROSS_REPO_SMOKE", "stale")
    venv = tmp_path / "venv"

    data_services = handoff.source_environment(
        "tao-data-services", tmp_path, venv
    )
    skill_bank = handoff.source_environment("tao-skill-bank", tmp_path, venv)

    assert data_services["TAO_DEFT_REQUIRE_CROSS_REPO_SMOKE"] == "1"
    assert "TAO_DEFT_REQUIRE_CROSS_REPO_SMOKE" not in skill_bank


def preflight_report(calls, contracts):
    """Fake DS preflight that records its argv and prints its JSON report."""
    def main(argv):
        calls.append(argv)
        print(json.dumps({"contracts": contracts, "cuda_verified": True}, indent=2))
        return 0
    return main


@pytest.mark.parametrize("repo", ["tao-pytorch", "tao-data-services"])
def test_packaged_default_does_not_require_optional_faiss(monkeypatch, repo):
    preflight_calls = []

    def installed(name):
        return SimpleNamespace(
            __file__="/installed/module.py",
            main=preflight_report(preflight_calls, {"benchmark_isolation": 1}),
            initialize_train_experiment=lambda cfg, **_: (None, {}),
        )

    isolation_checks = []
    monkeypatch.setattr(handoff.importlib, "import_module", installed)
    monkeypatch.setattr(handoff, "benchmark_isolation_check", lambda: isolation_checks.append(True))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setitem(sys.modules, "faiss", None)
    monkeypatch.setenv("TAO_VISIBLE_DEVICES", "0")
    assert handoff.packaged_check(repo) == 0
    expected_calls = (
        [["preflight", "--gpu"]]
        if repo in {"tao-data-services", "tao-skill-bank"} else []
    )
    assert preflight_calls == expected_calls
    assert isolation_checks == [True] * len(expected_calls)
    with pytest.raises(ModuleNotFoundError, match="faiss"):
        handoff.packaged_check(repo, require_gpu_faiss=True)


@pytest.mark.parametrize("contracts", [{}, {"benchmark_isolation": 0}])
def test_packaged_check_rejects_image_without_benchmark_isolation(monkeypatch, capsys, contracts):
    """An older DS image accepts the sidecar but never screens the source pool."""
    calls = []
    monkeypatch.setattr(handoff.importlib, "import_module", lambda _: SimpleNamespace(
        __file__="/installed/module.py", main=preflight_report(calls, contracts),
    ))
    with pytest.raises(RuntimeError, match="predates held-out benchmark isolation"):
        handoff.packaged_check("tao-data-services")
    assert calls == [["preflight", "--gpu"]]
    assert '"cuda_verified": true' in capsys.readouterr().out


def installed_materialize(seal_renamed=False, seal_rejected=False, skip_clean_seal=False):
    """Fake DS data action; JSON stands in for parquet so the test needs no pandas."""
    def main(argv):
        options = dict(zip(argv[1::2], argv[2::2]))
        delta = json.loads(Path(options["--delta"]).read_text())
        held_out = json.loads(Path(options["--benchmark-acquisition-units"]).read_text())
        output = Path(options["--output-dir"])
        output.mkdir()
        if delta["content_sha256"] == held_out["content_sha256"] and not seal_renamed:
            if seal_rejected:
                (output / "_SUCCESS").touch()
            raise ValueError("Cumulative training manifest overlaps the sealed benchmark: count=1")
        if not skip_clean_seal:
            (output / "_SUCCESS").touch()
        return 0

    class DataFrame(dict):
        def to_parquet(self, path, index):
            Path(path).write_text(json.dumps({key: value[0] for key, value in self.items()}))

    modules = {"pandas": SimpleNamespace(DataFrame=DataFrame),
               "nvidia_tao_ds.mining.dinov3.internal.refinement": SimpleNamespace(main=main)}
    return modules.__getitem__


def test_benchmark_isolation_check_accepts_guarded_image(monkeypatch):
    monkeypatch.setattr(handoff.importlib, "import_module", installed_materialize())
    handoff.benchmark_isolation_check()


def test_benchmark_isolation_check_rejects_image_without_the_guard_flag(monkeypatch):
    def main(argv):
        raise SystemExit(2)  # argparse: unrecognized --benchmark-acquisition-units
    modules = installed_materialize()
    monkeypatch.setattr(handoff.importlib, "import_module", lambda name: (
        SimpleNamespace(main=main) if name.endswith("refinement") else modules(name)))
    with pytest.raises(SystemExit):
        handoff.benchmark_isolation_check()


@pytest.mark.parametrize("image,error", [
    ({"seal_renamed": True}, "materialized a renamed benchmark copy"),
    ({"seal_rejected": True}, "sealed a rejected manifest"),
    ({"skip_clean_seal": True}, "did not seal a manifest disjoint"),
])
def test_benchmark_isolation_check_rejects_unguarded_image(monkeypatch, image, error):
    """A renamed copy shares only its content hash with the held-out sample."""
    monkeypatch.setattr(handoff.importlib, "import_module", installed_materialize(**image))
    with pytest.raises(RuntimeError, match=error):
        handoff.benchmark_isolation_check()


def test_packaged_check_rejects_image_that_resumes_from_empty_path(monkeypatch):
    """An image built before the fresh-start fix must not pass release checks."""
    def installed(name):
        return SimpleNamespace(
            __file__="/installed/module.py",
            # Requiring the keyword proves the check disables directory discovery.
            initialize_train_experiment=lambda cfg, *, auto_resume: (
                None if auto_resume else cfg["train"]["resume_training_checkpoint_path"], {}
            ),
        )

    monkeypatch.setattr(handoff.importlib, "import_module", installed)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setenv("TAO_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="instead of starting fresh"):
        handoff.packaged_check("tao-pytorch")


def test_packaged_check_still_rejects_missing_cuda(monkeypatch):
    monkeypatch.setattr(handoff.importlib, "import_module", lambda _: SimpleNamespace(__file__="/installed/module.py"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        handoff.packaged_check("tao-pytorch")


def test_packaged_core_check_rejects_partial_deft_schema(monkeypatch):
    from dataclasses import dataclass

    @dataclass
    class Empty:
        pass

    @dataclass
    class Manifest:
        train_manifest: str = ""

    def installed(name):
        return SimpleNamespace(
            ExperimentConfig=lambda: SimpleNamespace(grit_score=Empty(), dataset=Manifest(), train=Empty()),
            GRITScoreConfig=Empty,
        )

    monkeypatch.setattr(handoff.importlib, "import_module", installed)
    with pytest.raises(RuntimeError, match="complete DINOv3 refinement schema"):
        handoff.packaged_check("tao-core")


def test_packaged_data_services_propagates_gpu_preflight_failure(monkeypatch):
    calls = []

    def installed(name):
        if name == "nvidia_tao_ds.mining.dinov3.workflow.cli":
            def fail(argv):
                calls.append(argv)
                raise RuntimeError("mixed precision convolution failed")
            return SimpleNamespace(__file__="/installed/cli.py", main=fail)
        return SimpleNamespace(__file__="/installed/module.py")

    monkeypatch.setattr(handoff.importlib, "import_module", installed)
    with pytest.raises(RuntimeError, match="mixed precision convolution failed"):
        handoff.packaged_check("tao-data-services")
    assert calls == [["preflight", "--gpu"]]


@pytest.mark.parametrize("timeout", [False, True])
@pytest.mark.parametrize("include_core", [False, True])
@pytest.mark.parametrize("require_gpu_faiss", [False, True])
def test_execution_pins_images_preserves_sources_and_cleans_up(tmp_path, monkeypatch, timeout, include_core, require_gpu_faiss):
    monkeypatch.setattr(handoff, "source_info", lambda *_: {"commit": "abc", "base": "def", "changed_files": ["x.py"]})
    monkeypatch.setattr(handoff, "capture", lambda argv: "sha256:" + ("1" if "pyt" in argv[-1] else "2") * 64)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["docker", "run"] and timeout:
            raise subprocess.TimeoutExpired(argv, 10)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(handoff.subprocess, "run", fake_run)
    options = args(tmp_path)
    options.require_gpu_faiss = require_gpu_faiss
    if include_core:
        options.core = tmp_path / "core"
    if require_gpu_faiss:
        with pytest.raises(ValueError, match="requires --suite packaged"):
            handoff.run(options)
        return
    assert handoff.run(options) == int(timeout)
    runs = [argv for argv in calls if argv[:2] == ["docker", "run"]]
    removals = [argv for argv in calls if argv[:2] == ["docker", "rm"]]
    assert len(runs) == len(removals) == (4 if include_core else 3)
    assert not any("build" in argv or "pull" in argv for argv in calls)
    for argv, cleanup in zip(runs, removals):
        assert "--pull=never" in argv
        assert argv[argv.index("--gpus") + 1] == "device=1"
        assert any(part.startswith("sha256:") for part in argv)
        assert cleanup[-1] == argv[argv.index("--name") + 1]
        assert argv[argv.index("--user") + 1] == f"{handoff.os.getuid()}:{handoff.os.getgid()}"
        # The mapped UID may have no passwd entry in the image.
        environment = {argv[index + 1] for index, part in enumerate(argv) if part == "-e"}
        assert {"USER=tao", "LOGNAME=tao", "HOME=/results/.home"} <= environment
        assert ("--require-gpu-faiss" in argv) == require_gpu_faiss
    summary = json.loads((tmp_path / "output/summary.json").read_text())
    assert all((tmp_path / "output" / repo / ".home").is_dir() for repo in summary["repos"])
    assert set(summary["exit_codes"].values()) == ({124} if timeout else {0})
    assert ("tao-core" in summary["repos"]) == include_core
    assert summary["require_gpu_faiss"] == require_gpu_faiss
    assert summary["images"]["tao-skill-bank"] == options.data_services_image
    if include_core:
        assert summary["images"]["tao-core"] == options.data_services_image


def test_inside_reconstructs_real_git_diff_from_archives(tmp_path, monkeypatch):
    """Exercise archive extraction and synthetic commits, mocking only test tools."""
    inputs, results = tmp_path / "inputs", tmp_path / "results"
    inputs.mkdir()
    results.mkdir()
    monkeypatch.setattr(handoff, "INPUTS", inputs)
    monkeypatch.setattr(handoff, "RESULTS", results)
    name = "tao-pytorch"
    for suffix, content in (("-base", b"old\n"), ("", b"new\n")):
        with tarfile.open(inputs / f"{name}{suffix}.tar", "w") as archive:
            entry = tarfile.TarInfo("feature.py")
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))
    (inputs / "manifest.json").write_text(json.dumps({
        "repos": {name: {"changed_files": ["feature.py"]}}
    }))
    original = subprocess.run
    calls = []

    def run(argv, **kwargs):
        if argv[0] == "git":
            return original(argv, **kwargs)
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(handoff.subprocess, "run", run)
    assert handoff.inside(name, "source") == 0
    assert any("pre_commit" in argv for argv in calls)
    assert any("pytest" in argv for argv in calls)
    assert json.loads((results / "checks.json").read_text())["exit_codes"] == [0, 0]


def test_source_mode_rejects_packaged_only_flag(tmp_path):
    with pytest.raises(ValueError, match="requires --suite packaged"):
        handoff.inside("tao-pytorch", "source", require_gpu_faiss=True)
