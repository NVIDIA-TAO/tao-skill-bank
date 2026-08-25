# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for IAA's production virtualenv contract."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.parse
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
IAA_SCRIPTS = REPO / "skills" / "applications" / "tao-run-deft-iaa" / "scripts"
sys.path.insert(0, str(IAA_SCRIPTS))
import manage_iaa_virtualenv as manager  # noqa: E402
import virtualenv_runtime as runtime  # noqa: E402


def _profile_shell(tmp_path: Path, profile: str, cli: str) -> Path:
    venv = tmp_path / profile
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    python = bin_dir / "python"
    os.symlink(sys.executable, python)
    entrypoint = bin_dir / cli
    entrypoint.write_text(f"#!{python}\n", encoding="utf-8")
    entrypoint.chmod(0o755)
    return venv


def test_manifest_is_schema_valid_and_hash_locks_are_installable():
    manifest = runtime.load_runtime_manifest()
    assert manifest["python"]["major_minor"] == "3.12"
    assert manifest["cuda"] == {
        "torch_version": "2.11.0+cu130",
        "torch_cuda_build": "13.0",
        "pytorch_index_url": "https://download.pytorch.org/whl/cu130",
    }
    for profile in runtime.PROFILE_NAMES:
        status = runtime.lock_status(profile)
        assert status["declared_status"] == "complete"
        assert status["prebuilt_verification_available"] is True
        assert status["ready_to_install"] is True
        assert status["blocker"] is None
        assert status["actual_sha256"] == status["declared_sha256"]


def test_combined_lock_uses_only_public_https_artifacts_with_hashes():
    lock = runtime.REFERENCE_DIR / "virtualenv-combined-py312-cu130.lock"
    text = lock.read_text(encoding="utf-8")
    urls = [token for token in text.split() if token.startswith("https://")]
    assert len(urls) == 274
    assert text.count("--hash=sha256:") == len(urls)
    assert {
        urllib.parse.urlsplit(url).hostname for url in urls
    } <= {
        "download-r2.pytorch.org",
        "download.pytorch.org",
        "files.pythonhosted.org",
        "github.com",
        "pypi.nvidia.com",
    }
    for forbidden in ("file:", "gitlab", "nvstaging", "/localhome/"):
        assert forbidden not in text
    for required in (
        "accelerate-1.13.0",
        "braceexpand-0.1.7",
        "colorama-0.4.6",
        "contourpy-1.3.3",
        "cudf_cu13-26.4.0",
        "cuml_cu13-26.4.0",
        "cycler-0.12.1",
        "einops-0.8.2",
        "fonttools-4.63.0",
        "ftfy-6.3.1",
        "fvcore-0.1.5.post20221221",
        "huggingface_hub-0.36.2",
        "hydra_core-1.3.2",
        "h5py-3.16.0",
        "iopath-0.1.10",
        "lightning_fabric-2.6.1",
        "lightning_utilities-0.15.3",
        "matplotlib-3.11.1",
        "nvidia_eff-0.6.6",
        "onnx-1.21.0",
        "open_clip_torch-2.30.0",
        "portalocker-3.2.0",
        "pyparsing-3.3.2",
        "pytorch_lightning-2.6.1",
        "safetensors-0.7.0",
        "sentencepiece-0.2.1",
        "tabulate-0.10.0",
        "tensorboardx-2.6.5",
        "termcolor-3.3.0",
        "timm-1.0.26",
        "torchmetrics-1.9.0",
        "transformers-4.57.5",
        "webdataset-1.0.2",
        "yacs-0.1.8",
    ):
        assert required in text


def test_ds_profile_probes_eagerly_loaded_action_modules():
    profile = runtime.load_runtime_manifest()["profiles"]["ds"]
    imports = profile["imports"]
    for module in ("matplotlib", "numpy", "pandas", "PIL", "pyarrow", "sklearn"):
        assert module in imports
    assert profile["distributions"] | {
        "contourpy": "1.3.3",
        "cycler": "0.12.1",
        "fonttools": "4.63.0",
        "kiwisolver": "1.5.0",
        "matplotlib": "3.11.1",
        "pyparsing": "3.3.2",
    } == profile["distributions"]
    assert "nvidia_tao_ds.mining.embedding.scripts.image_embeddings" in imports
    assert "nvidia_tao_ds.mining.embedding.scripts.text_embeddings" in imports
    assert "nvidia_tao_ds.mining.tmm.scripts.nearest_neighbors" in imports


def test_ds_preflight_fails_when_matplotlib_import_is_missing(tmp_path, monkeypatch):
    venv = _profile_shell(tmp_path, "ds", "embedding")
    tmm = venv / "bin" / "tmm"
    tmm.write_text(f"#!{venv / 'bin' / 'python'}\n", encoding="utf-8")
    tmm.chmod(0o755)
    facts = {
        "implementation": "cpython",
        "python_major_minor": "3.12",
        "machine": "x86_64",
        "glibc": "2.39",
        "prefix": str(venv.resolve()),
        "base_prefix": "/usr",
        "torch_version": "2.11.0+cu130",
        "torch_cuda_build": "13.0",
    }

    def completed(command, **kwargs):
        contract = json.loads(command[-1])
        assert "matplotlib" in contract["imports"]
        return subprocess.CompletedProcess(
            command,
            0,
            "IAA_VENV_CONTRACT=" + json.dumps(
                {
                    "errors": [
                        "import matplotlib failed: ModuleNotFoundError: "
                        "No module named 'matplotlib'"
                    ],
                    "facts": facts,
                }
            ),
            "",
        )

    monkeypatch.setattr(runtime.subprocess, "run", completed)
    with pytest.raises(ValueError, match="import matplotlib failed"):
        runtime.validate_tao_virtualenv(venv, profile="ds", probe_imports=True)


def test_repair_synchronizes_existing_profile_from_hash_lock(tmp_path, monkeypatch):
    target = _profile_shell(tmp_path, "ds", "embedding")
    commands: list[list[str]] = []

    def completed(command, **kwargs):
        commands.append([str(item) for item in command])
        return subprocess.CompletedProcess(command, 0, "", "")

    validated = []
    monkeypatch.setattr(manager.subprocess, "run", completed)
    monkeypatch.setattr(
        manager,
        "apply_local_runtime_compatibility",
        lambda path: Path(path).resolve(),
    )
    monkeypatch.setattr(
        manager,
        "validate_tao_virtualenv",
        lambda path, **kwargs: validated.append((Path(path).resolve(), kwargs)),
    )

    assert manager.main(
        [
            "repair",
            "--profile",
            "ds",
            "--virtualenv",
            str(target),
            "--approve-repair",
        ]
    ) == 0
    assert commands[1][0] == str(target.resolve() / "bin" / "python")
    assert commands[1][1:7] == [
        "-m", "pip", "install", "--require-hashes", "--no-deps", "-r"
    ]
    assert commands[1][-1] == str(
        runtime.REFERENCE_DIR / "virtualenv-combined-py312-cu130.lock"
    )
    assert validated == [
        (target.resolve(), {"profile": "ds", "probe_imports": True})
    ]


def test_pyt_profile_probes_real_clip_subtask_initialization():
    profile = runtime.load_runtime_manifest()["profiles"]["pyt"]
    assert profile["initialization_probes"] == ["clip:evaluate", "clip:train"]
    for distribution, version in {
        "braceexpand": "0.1.7",
        "webdataset": "1.0.2",
        "onnx": "1.21.0",
        "h5py": "3.16.0",
        "tensorboardX": "2.6.5",
    }.items():
        assert profile["distributions"][distribution] == version
    assert "tensorboardX" in profile["imports"]
    tensorboard_artifact = next(
        item
        for item in profile["primary_artifacts"]
        if item["distribution"] == "tensorboardX"
    )
    assert tensorboard_artifact == {
        "distribution": "tensorboardX",
        "version": "2.6.5",
        "index_url": "https://pypi.org/simple",
        "sha256": "c10b891d00af306537cb8b58a039b2ba41571f0da06f433a41c4ca8d6abe1373",
        "size": 87510,
    }
    assert "get_subtask_list" in runtime._ENV_PROBE  # noqa: SLF001
    assert "TensorBoardLogger" in runtime._ENV_PROBE  # noqa: SLF001
    assert "TemporaryDirectory" in runtime._ENV_PROBE  # noqa: SLF001
    assert 'logger.finalize("success")' in runtime._ENV_PROBE  # noqa: SLF001


def test_install_builds_at_final_non_relocatable_path(tmp_path, monkeypatch):
    target = tmp_path / "pyt-runtime"
    commands = []
    monkeypatch.setattr(
        manager,
        "lock_status",
        lambda profile: {
            "ready_to_install": True,
            "blocker": None,
            "lock_file": str(tmp_path / "complete.lock"),
        },
    )
    monkeypatch.setattr(manager.shutil, "which", lambda name: f"/tools/{name}")
    monkeypatch.setattr(
        manager.subprocess,
        "run",
        lambda command, **kwargs: commands.append([str(item) for item in command]),
    )
    monkeypatch.setattr(
        manager,
        "validate_tao_virtualenv",
        lambda path, **kwargs: Path(path).resolve(),
    )
    monkeypatch.setattr(
        manager,
        "apply_local_runtime_compatibility",
        lambda path: Path(path).resolve(),
    )
    assert manager.main(
        [
            "install",
            "--profile",
            "pyt",
            "--virtualenv",
            str(target),
            "--approve-install",
        ]
    ) == 0
    assert target.is_dir()
    assert commands[0][-1] == str(target)
    assert commands[1][0] == str(target / "bin" / "python")
    assert "--require-hashes" in commands[1]
    assert "--no-deps" in commands[1]
    assert "--only-binary=:all:" not in commands[1]
    assert commands[2][-3:-1] == ["--no-deps", "--no-build-isolation"]
    assert commands[2][-1] == manager.APEX_SOURCE


def test_local_runtime_compatibility_is_hash_gated_and_idempotent(
    tmp_path, monkeypatch
):
    target = tmp_path / "runtime"
    destination = target / manager.TAO_CORE_HANDLER_INIT
    destination.parent.mkdir(parents=True)
    original = b"approved original\n"
    destination.write_bytes(original)
    monkeypatch.setattr(
        manager,
        "TAO_CORE_HANDLER_INIT_ORIGINAL_SHA256",
        __import__("hashlib").sha256(original).hexdigest(),
    )
    replacement = manager.pathlib.Path(manager.__file__).resolve().parent.parent / (
        "patches/tao_core_execution_handlers_init.py"
    )
    expected = replacement.read_bytes()

    assert manager.apply_local_runtime_compatibility(target) == destination
    assert destination.read_bytes() == expected
    assert manager.apply_local_runtime_compatibility(target) == destination
    assert destination.read_bytes() == expected


def test_local_runtime_compatibility_rejects_unknown_package_build(tmp_path):
    target = tmp_path / "runtime"
    destination = target / manager.TAO_CORE_HANDLER_INIT
    destination.parent.mkdir(parents=True)
    destination.write_text("unknown build\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the approved 7.1.0 build"):
        manager.apply_local_runtime_compatibility(target)


def test_failed_install_removes_only_reserved_target(tmp_path, monkeypatch):
    target = tmp_path / "ds-runtime"
    monkeypatch.setattr(
        manager,
        "lock_status",
        lambda profile: {
            "ready_to_install": True,
            "blocker": None,
            "lock_file": str(tmp_path / "complete.lock"),
        },
    )
    monkeypatch.setattr(manager.shutil, "which", lambda name: f"/tools/{name}")

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(manager.subprocess, "run", fail)
    assert manager.main(
        [
            "install",
            "--profile",
            "ds",
            "--virtualenv",
            str(target),
            "--approve-install",
        ]
    ) == 2
    assert not target.exists()


def test_prebuilt_verification_is_independent_of_install_lock(tmp_path):
    with pytest.raises(ValueError, match="virtualenv is missing or invalid"):
        runtime.validate_tao_virtualenv(
            tmp_path / "untrusted-prebuilt",
            profile="pyt",
            probe_imports=True,
        )


def test_full_ds_profile_verification_checks_metadata_pip_and_real_cuda_probe(
    tmp_path, monkeypatch
):
    venv = _profile_shell(tmp_path, "ds", "embedding")
    calls: list[list[str]] = []
    environments: list[dict[str, str] | None] = []
    facts = {
        "implementation": "cpython",
        "python_major_minor": "3.12",
        "machine": "x86_64",
        "glibc": "2.39",
        "prefix": str(venv.resolve()),
        "base_prefix": "/usr",
        "distributions": {},
        "entrypoints": {},
        "imports": [],
        "torch_version": "2.11.0+cu130",
        "torch_cuda_build": "13.0",
    }

    def completed(command, **kwargs):
        calls.append([str(item) for item in command])
        environments.append(kwargs.get("env"))
        if "-c" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                "import warning\nIAA_VENV_CONTRACT="
                + json.dumps({"errors": [], "facts": facts}),
                "",
            )
        if command[-2:] == ["pip", "check"]:
            return subprocess.CompletedProcess(command, 0, "No broken requirements found.\n", "")
        return subprocess.CompletedProcess(command, 0, "IAA_CUDA_PROBE=PASS {}\n", "")

    monkeypatch.setattr(runtime.subprocess, "run", completed)
    assert runtime.validate_tao_virtualenv(
        venv,
        profile="ds",
        probe_imports=True,
        required_cli="embedding",
        minimum_gpus=2,
        gpu_ids=[6, 7],
    ) == venv.resolve()
    assert len(calls) == 3
    assert calls[1][-3:] == ["-m", "pip", "check"]
    assert "check_iaa_cuda_runtime.py" in calls[2][1]
    assert calls[2][-4:] == ["--min-gpus", "2", "--require-cli", "embedding"]
    assert "tmm" not in calls[2]
    assert environments[2]["CUDA_VISIBLE_DEVICES"] == "6,7"


def test_pip_check_exception_is_narrow_to_eff_metadata(tmp_path, monkeypatch):
    venv = _profile_shell(tmp_path, "pyt", "clip")
    facts = {
        "implementation": "cpython",
        "python_major_minor": "3.12",
        "machine": "x86_64",
        "glibc": "2.39",
        "prefix": str(venv.resolve()),
        "base_prefix": "/usr",
        "distributions": {},
        "entrypoints": {},
        "imports": [],
        "torch_version": "2.11.0+cu130",
        "torch_cuda_build": "13.0",
    }

    def completed(command, **kwargs):
        if "-c" in command:
            contract = json.loads(command[-1])
            assert contract["initialization_probes"] == [
                "clip:evaluate",
                "clip:train",
            ]
            return subprocess.CompletedProcess(
                command,
                0,
                "IAA_VENV_CONTRACT=" + json.dumps({"errors": [], "facts": facts}),
                "",
            )
        return subprocess.CompletedProcess(
            command,
            1,
            "\n".join(
                [
                    *sorted(runtime._EFF_PIP_CHECK_EXCEPTIONS),
                    "nvidia-tao-core 7.1.0 requires unused-hosted-client, "
                    "which is not installed.",
                ]
            )
            + "\n",
            "",
        )

    monkeypatch.setattr(runtime.subprocess, "run", completed)
    assert runtime.validate_tao_virtualenv(
        venv, profile="pyt", probe_imports=True
    ) == venv.resolve()

    def broken(command, **kwargs):
        result = completed(command, **kwargs)
        if "-c" not in command:
            result.stdout += "unrelated 1.0 requires missing, which is not installed.\n"
        return result

    monkeypatch.setattr(runtime.subprocess, "run", broken)
    with pytest.raises(ValueError, match="unrelated 1.0 requires missing"):
        runtime.validate_tao_virtualenv(venv, profile="pyt", probe_imports=True)


def test_console_script_must_be_bound_to_profile_python(tmp_path, monkeypatch):
    venv = _profile_shell(tmp_path, "pyt", "clip")
    clip = venv / "bin" / "clip"
    clip.write_text("#!/usr/bin/python3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not bound"):
        runtime.validate_tao_virtualenv(
            venv, profile="pyt", probe_imports=False
        )


def test_legacy_shared_environment_is_verified_as_both_profiles(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    observed = []

    def validate(
        path, *, profile, probe_imports, required_cli=None, minimum_gpus=None,
        gpu_ids=None
    ):
        observed.append((Path(path), profile, probe_imports))
        return Path(path).resolve()

    monkeypatch.setattr(runtime, "validate_tao_virtualenv", validate)
    profiles = runtime.resolve_virtualenv_profiles(
        platform="virtualenv",
        legacy=shared,
        pyt=None,
        ds=None,
        probe_imports=True,
    )
    assert profiles == {"pyt": shared.resolve(), "ds": shared.resolve()}
    assert observed == [(shared, "pyt", True), (shared, "ds", True)]
