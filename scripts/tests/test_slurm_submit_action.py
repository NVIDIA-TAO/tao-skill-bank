# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import subprocess

import pytest


REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "skills/platform/tao-run-on-slurm/scripts/slurm_submit_action.py"
SPEC = importlib.util.spec_from_file_location("slurm_submit_action", SCRIPT)
assert SPEC and SPEC.loader
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)


def _completed(argv, rc=0, stdout=b"", stderr=b""):
    return subprocess.CompletedProcess(argv, rc, stdout, stderr)


def _script(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "job.sbatch"
    path.write_text("#!/usr/bin/env bash\ntrue\n", encoding="utf-8")
    return path


def _container_script(tmp_path: pathlib.Path, container_env: str | None) -> pathlib.Path:
    option = "" if container_env is None else f" --container-env={container_env}"
    path = tmp_path / "container.sbatch"
    path.write_text(
        "#!/usr/bin/env bash\n"
        f"srun --container-image=/lustre/tao.sqsh{option} true\n",
        encoding="utf-8",
    )
    return path


def _mounted_container_script(tmp_path: pathlib.Path, mounts: str) -> pathlib.Path:
    path = tmp_path / "mounted-container.sbatch"
    path.write_text(
        "#!/usr/bin/env bash\n"
        "srun --container-image=/lustre/tao.sqsh \\\n"
        f"  --container-env={','.join(submit.CONTAINER_ENV_NAMES)} \\\n"
        f"  --container-mounts={mounts} true\n",
        encoding="utf-8",
    )
    return path


def _typed_model_script(
    tmp_path: pathlib.Path, request: dict, *, gres: int | None = None
) -> pathlib.Path:
    environment = request["environment"]
    names = tuple(
        dict.fromkeys((*submit.CONTAINER_ENV_NAMES, *sorted(environment)))
    )
    gpu_count = request["spec_bundle"]["compute_shape"]["gpus"] if gres is None else gres
    path = tmp_path / "typed-model.sbatch"
    path.write_text(
        "#!/usr/bin/env bash\n"
        f"#SBATCH --gres=gpu:{gpu_count}\n"
        "export NCCL_DEBUG=INFO\n"
        "export LOGLEVEL=INFO\n"
        + "".join(f"export {name}={value}\n" for name, value in environment.items())
        + "srun --container-image=/lustre/tao.sqsh "
        + f"--container-env={','.join(names)} true\n",
        encoding="utf-8",
    )
    return path


def _iaa_train_script(
    tmp_path: pathlib.Path, *, wrapper: bool, quoted: bool = False
) -> pathlib.Path:
    names = ",".join(submit.CONTAINER_ENV_NAMES)
    if wrapper:
        prefix = (
            "'/patches/run_clip_train_slurm.sh' "
            if quoted
            else "/patches/run_clip_train_slurm.sh "
        )
    else:
        prefix = ""
    path = tmp_path / "iaa-train.sbatch"
    path.write_text(
        "#!/usr/bin/env bash\n"
        "#SBATCH --gres=gpu:2\n"
        "srun --container-image=/lustre/tao.sqsh \\\n"
        f"  --container-env={names} \\\n"
        f"  {prefix}clip train -e /results/train.yaml\n",
        encoding="utf-8",
    )
    return path


def _adapter_script(
    tmp_path: pathlib.Path, *, container_env: str | None = None, gres: bool = False
) -> pathlib.Path:
    names = container_env or ",".join(submit.ADAPTER_CONTAINER_ENV_NAMES)
    path = tmp_path / "adapter.sbatch"
    path.write_text(
        "#!/usr/bin/env bash\n"
        + ("#SBATCH --gres=gpu:1\n" if gres else "")
        + "".join(f"export {name}={value}\n" for name, value in submit.ADAPTER_ENVIRONMENT.items())
        + f"srun --container-image=/lustre/iaa.sqsh --container-env={names} true\n",
        encoding="utf-8",
    )
    return path


def _mock_run(monkeypatch, local: pathlib.Path, *, remote_hash=None, submit_out=b"12345\n"):
    digest = remote_hash or hashlib.sha256(local.read_bytes()).hexdigest()
    calls = []
    exact_queries = iter((b"", b"", b""))

    def run(argv, *, input_bytes=None):
        calls.append(list(argv))
        command = argv[-1] if argv and argv[0] == "ssh" else ""
        if "squeue -h --name" in command:
            return _completed(argv, stdout=next(exact_queries))
        if "sha256sum" in command:
            return _completed(argv, stdout=f"{digest}  remote\n".encode())
        if "sbatch --parsable" in command:
            return _completed(argv, stdout=submit_out)
        return _completed(argv)

    monkeypatch.setattr(submit, "_run", run)
    return calls


def test_submit_validates_copy_hash_test_only_and_parses_handle(tmp_path, monkeypatch):
    local = _script(tmp_path)
    calls = _mock_run(monkeypatch, local)
    result = submit.submit_action(
        login="user@login", job_id="iaa-job-1", rendered_script=local,
        remote_script=pathlib.Path("/lustre/run/sbatch/iaa-job-1.sbatch"),
    )
    assert result["backend_ref"] == "12345"
    assert result["reconciled"] is False
    flattened = "\n".join(" ".join(call) for call in calls)
    assert "scp -q --" in flattened
    assert "bash -n" in flattened
    assert "sbatch --test-only" in flattened
    assert "sbatch --parsable" in flattened
    assert flattened.count("squeue -h --name") == 2


def test_empty_rendered_script_fails_before_transport(tmp_path, monkeypatch):
    local = tmp_path / "empty.sbatch"
    local.touch()
    monkeypatch.setattr(submit, "_run", lambda *args, **kwargs: pytest.fail("transport called"))
    with pytest.raises(ValueError, match="nonempty regular file"):
        submit.submit_action(
            login="user@login", job_id="iaa-job-1", rendered_script=local,
            remote_script=pathlib.Path("/lustre/run/sbatch/job.sbatch"),
        )


def test_container_launch_accepts_only_fixed_nonsecret_env_allowlist(tmp_path):
    names = ",".join(submit.CONTAINER_ENV_NAMES)
    local = _container_script(tmp_path, names)
    _, _, digest = submit._validate_inputs(  # noqa: SLF001
        login="user@login",
        job_id="iaa-job-1",
        rendered_script=local,
        remote_script=pathlib.Path("/lustre/run/sbatch/job.sbatch"),
    )
    assert digest == hashlib.sha256(local.read_bytes()).hexdigest()
    assert not any("TOKEN" in name or "KEY" in name for name in submit.CONTAINER_ENV_NAMES)


@pytest.mark.parametrize("quoted", (False, True))
def test_iaa_clip_train_requires_and_accepts_exact_topology_wrapper(tmp_path, quoted):
    local = _iaa_train_script(tmp_path, wrapper=True, quoted=quoted)
    submit._validate_inputs(  # noqa: SLF001
        login="user@login",
        job_id="clip-deft-iaa-train-0123456789abcdef-a1b2c3",
        rendered_script=local,
        remote_script=pathlib.Path("/lustre/run/sbatch/job.sbatch"),
    )


def test_iaa_clip_train_without_topology_wrapper_fails_before_transport(
    tmp_path, monkeypatch
):
    local = _iaa_train_script(tmp_path, wrapper=False)
    monkeypatch.setattr(
        submit, "_run", lambda *args, **kwargs: pytest.fail("transport called")
    )
    with pytest.raises(ValueError, match="run_clip_train_slurm.sh"):
        submit.submit_action(
            login="user@login",
            job_id="clip-deft-iaa-train-0123456789abcdef-a1b2c3",
            rendered_script=local,
            remote_script=pathlib.Path("/lustre/run/sbatch/job.sbatch"),
        )


def test_typed_iaa_submit_requires_binding_evidence_before_transport(
    tmp_path, monkeypatch
):
    local = _iaa_train_script(tmp_path, wrapper=True)
    monkeypatch.setattr(
        submit, "_run", lambda *args, **kwargs: pytest.fail("transport called")
    )
    with pytest.raises(ValueError, match="--request and --job-binding"):
        submit.submit_action(
            login="user@login",
            job_id="clip-deft-iaa-train-0123456789abcdef-a1b2c3",
            rendered_script=local,
            remote_script=pathlib.Path("/lustre/run/sbatch/job.sbatch"),
        )


def test_typed_iaa_binding_validation_accepts_exact_pending_ownership(monkeypatch):
    request_path = pathlib.Path("/lustre/run/metric_parse.action.json")
    binding_path = pathlib.Path("/lustre/run/metric_parse.job-binding.json")
    job_id = "iaa-adapter-deft-iaa-metric_parse-0123456789abcdef-a1b2c3"
    job_path = pathlib.Path(f"/home/user/.tao/jobs/{job_id}.json")
    request = {
        "schema_version": "1",
        "workflow": "tao-run-deft-iaa",
        "platform": "slurm",
        "freshness_contract": "remote-mirror-with-delete-before-submit",
        "staging_receipt_path": "/lustre/run/metric_parse.staged.json",
        "job_binding_path": str(binding_path),
        "job_state_dir": "/home/user/.tao",
        "spec_bundle": {"action": "deft-iaa-metric_parse-0123456789abcdef"},
    }
    request["request_sha256"] = submit._canonical_sha256(request)  # noqa: SLF001
    job = {
        "schema_version": 1,
        "id": job_id,
        "platform": "slurm",
        "image": "nvcr.io/test/data-services:1",
        "network_arch": "iaa-adapter",
        "action": request["spec_bundle"]["action"],
        "results_dir": "/lustre/run",
        "storage_tier": "A",
        "upload_excludes": [".tao-runtime/"],
        "submitted_at": "2026-08-21T00:00:00+00:00",
        "backend_ref": None,
        "terminal_state": None,
        "transitions": [{"state": "PENDING"}],
    }
    identity = {field: job.get(field) for field in submit.JOB_IDENTITY_FIELDS}
    staging = {
        "schema_version": "1",
        "workflow": "tao-run-deft-iaa",
        "platform": "slurm",
        "request_path": str(request_path),
        "request_sha256": request["request_sha256"],
        "backend_scope": "/lustre/run",
        "checked_paths_absent": ["/local/output", "/local/log"],
        "checked_at": "2026-08-21T00:00:00+00:00",
    }
    staging["receipt_sha256"] = submit._canonical_sha256(staging)  # noqa: SLF001
    binding = {
        "schema_version": "1",
        "workflow": "tao-run-deft-iaa",
        "platform": "slurm",
        "request_path": str(request_path),
        "request_sha256": request["request_sha256"],
        "job_record_path": str(job_path),
        "job_id": job_id,
        "job_identity_sha256": submit._canonical_sha256(identity),  # noqa: SLF001
        "results_scope": job["results_dir"],
        "staging_receipt_sha256": staging["receipt_sha256"],
        "bound_at": "2026-08-21T00:00:01+00:00",
    }
    binding["binding_sha256"] = submit._canonical_sha256(binding)  # noqa: SLF001
    payloads = {
        str(request_path): request,
        str(binding_path): binding,
        str(job_path): job,
        request["staging_receipt_path"]: staging,
    }
    monkeypatch.setattr(
        submit,
        "_local_json",
        lambda path, _name: payloads[str(path)],
    )
    submit._validate_iaa_job_binding(  # noqa: SLF001
        login="user@login",
        job_id=job_id,
        request_path=request_path,
        binding_path=binding_path,
    )

    job["backend_ref"] = "32650279"
    with pytest.raises(ValueError, match="fresh PENDING"):
        submit._validate_iaa_job_binding(  # noqa: SLF001
            login="user@login",
            job_id=job_id,
            request_path=request_path,
            binding_path=binding_path,
        )


def test_typed_iaa_rendered_mounts_require_exact_ordered_request_mapping(tmp_path):
    request = {
        "mounts": [
            {"source": "/lustre/run", "target": "/results", "read_only": False},
            {"source": "/lustre/run/config", "target": "/specs", "read_only": True},
            {"source": "/lustre/cache", "target": "/cache", "read_only": False},
        ]
    }
    exact = _mounted_container_script(
        tmp_path,
        "/lustre/run:/results:rw,/lustre/run/config:/specs:ro,/lustre/cache:/cache:rw",
    )
    submit._validate_iaa_rendered_mounts(request, exact)  # noqa: SLF001

    stale = _mounted_container_script(
        tmp_path,
        "/lustre/run:/results:rw,/lustre/run/config:/specs:ro,"
        "/lustre/old-attempt/staged-cache:/cache:ro",
    )
    with pytest.raises(ValueError, match="differ from immutable IAA request"):
        submit._validate_iaa_rendered_mounts(request, stale)  # noqa: SLF001


def test_typed_iaa_submit_rejects_stale_mount_before_transport(tmp_path, monkeypatch):
    request = {
        "mounts": [
            {"source": "/lustre/run", "target": "/results", "read_only": False},
            {"source": "/lustre/cache", "target": "/cache", "read_only": False},
        ]
    }
    stale = _mounted_container_script(
        tmp_path, "/lustre/run:/results:rw,/lustre/old-cache:/cache:ro"
    )
    monkeypatch.setattr(
        submit, "_validate_iaa_job_binding", lambda **_kwargs: (request, None)
    )
    monkeypatch.setattr(
        submit, "_run", lambda *args, **kwargs: pytest.fail("transport called")
    )
    with pytest.raises(ValueError, match="differ from immutable IAA request"):
        submit.submit_action(
            login="user@login",
            job_id="data-services-deft-iaa-target_embed-0123456789abcdef-a1b2c3",
            rendered_script=stale,
            remote_script=pathlib.Path("/lustre/run/sbatch/job.sbatch"),
            request_path=pathlib.Path("/lustre/run/target_embed.action.json"),
            binding_path=pathlib.Path("/lustre/run/target_embed.job-binding.json"),
        )


def test_typed_iaa_model_compute_uses_signed_network_environment_and_gpu_shape(tmp_path):
    request = {
        "spec_bundle": {
            "network_arch": "data-services",
            "compute_shape": {"gpus": 2, "nodes": 1},
        },
        "gpu_ids": [0, 1],
        "environment": {
            "HOME": "/tmp",
            "PYTHONPATH": "/patches",
            "HF_HOME": "/cache/huggingface",
            "XDG_CACHE_HOME": "/cache",
        },
    }
    exact = _typed_model_script(tmp_path, request)
    submit._validate_iaa_rendered_compute(request, exact)  # noqa: SLF001

    wrong_gpu = _typed_model_script(tmp_path, request, gres=8)
    with pytest.raises(ValueError, match="GPU request differs"):
        submit._validate_iaa_rendered_compute(request, wrong_gpu)  # noqa: SLF001

    exact = _typed_model_script(tmp_path, request)
    text = exact.read_text(encoding="utf-8")
    text = text.replace(
        "export XDG_CACHE_HOME=/cache\n",
        "export XDG_CACHE_HOME=/cache\nexport IAA_COMPUTE_FRAME=slurm\n",
    ).replace("XDG_CACHE_HOME true", "XDG_CACHE_HOME,IAA_COMPUTE_FRAME true")
    exact.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="allowlist|unrequested"):
        submit._validate_iaa_rendered_compute(request, exact)  # noqa: SLF001


def test_typed_iaa_adapter_compute_requires_zero_gpu_shape(tmp_path):
    request = {
        "spec_bundle": {
            "network_arch": "iaa-adapter",
            "compute_shape": {"gpus": 0, "nodes": 1},
        },
        "gpu_ids": [],
        "environment": dict(submit.ADAPTER_ENVIRONMENT),
    }
    exact = _adapter_script(tmp_path)
    submit._validate_iaa_rendered_compute(request, exact)  # noqa: SLF001

    wrong_gpu = _adapter_script(tmp_path, gres=True)
    with pytest.raises(ValueError, match="GPU request differs"):
        submit._validate_iaa_rendered_compute(request, wrong_gpu)  # noqa: SLF001


def test_non_iaa_clip_job_is_not_subject_to_iaa_topology_wrapper(tmp_path):
    local = _iaa_train_script(tmp_path, wrapper=False)
    submit._validate_inputs(  # noqa: SLF001
        login="user@login",
        job_id="clip-train-unrelated-a1b2c3",
        rendered_script=local,
        remote_script=pathlib.Path("/lustre/run/sbatch/job.sbatch"),
    )


def test_adapter_cpu_launch_accepts_fixed_compute_frame_allowlist(tmp_path):
    local = _adapter_script(tmp_path)
    submit._validate_inputs(  # noqa: SLF001
        login="user@login", job_id="iaa-job-1", rendered_script=local,
        remote_script=pathlib.Path("/lustre/run/sbatch/job.sbatch"),
    )
    assert submit.ADAPTER_CONTAINER_ENV_NAMES == (
        "HF_HOME", "HOME", "IAA_COMPUTE_FRAME", "PYTHONPATH", "XDG_CACHE_HOME",
    )


@pytest.mark.parametrize(
    "path_factory,match",
    (
        (lambda path: _adapter_script(path, container_env=",".join(submit.CONTAINER_ENV_NAMES)), "allowlist"),
        (lambda path: _adapter_script(path, gres=True), "must not request GPUs"),
    ),
)
def test_adapter_launch_rejects_missing_marker_forwarding_or_gpu(tmp_path, path_factory, match):
    with pytest.raises(ValueError, match=match):
        submit._validate_inputs(  # noqa: SLF001
            login="user@login", job_id="iaa-job-1", rendered_script=path_factory(tmp_path),
            remote_script=pathlib.Path("/lustre/run/sbatch/job.sbatch"),
        )


def test_adapter_launch_rejects_changed_or_extra_producer_environment(tmp_path):
    changed = _adapter_script(tmp_path).read_text().replace("export HOME=/tmp", "export HOME=/root")
    path = tmp_path / "changed.sbatch"
    path.write_text(changed)
    with pytest.raises(ValueError, match="HOME=/tmp"):
        submit._validate_inputs(  # noqa: SLF001
            login="user@login", job_id="iaa-job-1", rendered_script=path,
            remote_script=pathlib.Path("/lustre/run/sbatch/job.sbatch"),
        )


def test_visualize_finish_accepts_fixed_native_math_thread_caps(tmp_path):
    environment = {**submit.ADAPTER_ENVIRONMENT, **submit.VISUALIZE_THREAD_CAPS}
    names = ",".join(environment)
    path = tmp_path / "visualize-finish.sbatch"
    path.write_text(
        "#!/usr/bin/env bash\n"
        + "".join(f"export {name}={value}\n" for name, value in environment.items())
        + "srun --container-image=/lustre/iaa.sqsh "
        + f"--container-env={names} python3 "
        + "/iaa-runtime/run_iaa_compute.py visualize_finish\n",
        encoding="utf-8",
    )

    submit._validate_inputs(  # noqa: SLF001
        login="user@login",
        job_id="iaa-visualize-finish",
        rendered_script=path,
        remote_script=pathlib.Path("/lustre/run/sbatch/visualize.sbatch"),
    )

    expected = ",".join(submit.ADAPTER_CONTAINER_ENV_NAMES)
    extra = _adapter_script(tmp_path).read_text().replace(
        f"--container-env={expected}", f"--container-env={expected},NGC_KEY"
    )
    path.write_text(extra)
    with pytest.raises(ValueError, match="allowlist"):
        submit._validate_inputs(  # noqa: SLF001
            login="user@login", job_id="iaa-job-1", rendered_script=path,
            remote_script=pathlib.Path("/lustre/run/sbatch/job.sbatch"),
        )


@pytest.mark.parametrize(
    "container_env",
    (
        None,
        "NCCL_P2P_DISABLE",
        ",".join((*submit.CONTAINER_ENV_NAMES, "NGC_KEY")),
        ",".join((*submit.CONTAINER_ENV_NAMES, "UNAPPROVED_VALUE")),
    ),
)
def test_container_launch_rejects_missing_secret_or_unapproved_env(
    tmp_path, container_env
):
    local = _container_script(tmp_path, container_env)
    with pytest.raises(ValueError, match="container-env|NCCL allowlist"):
        submit._validate_inputs(  # noqa: SLF001
            login="user@login",
            job_id="iaa-job-1",
            rendered_script=local,
            remote_script=pathlib.Path("/lustre/run/sbatch/job.sbatch"),
        )


def test_hash_mismatch_fails_before_test_only_or_submit(tmp_path, monkeypatch):
    local = _script(tmp_path)
    calls = _mock_run(monkeypatch, local, remote_hash="0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        submit.submit_action(
            login="user@login", job_id="iaa-job-1", rendered_script=local,
            remote_script=pathlib.Path("/lustre/run/sbatch/job.sbatch"),
        )
    assert not any("sbatch --test-only" in " ".join(call) for call in calls)
    assert not any("sbatch --parsable" in " ".join(call) for call in calls)


def test_existing_exact_job_name_fails_before_copy(tmp_path, monkeypatch):
    local = _script(tmp_path)
    calls = []

    def run(argv, *, input_bytes=None):
        calls.append(list(argv))
        if argv[0] == "ssh" and "squeue -h --name" in argv[-1]:
            return _completed(argv, stdout=b"999\n")
        return _completed(argv)

    monkeypatch.setattr(submit, "_run", run)
    with pytest.raises(ValueError, match="already exists"):
        submit.submit_action(
            login="user@login", job_id="iaa-job-1", rendered_script=local,
            remote_script=pathlib.Path("/lustre/run/sbatch/job.sbatch"),
        )
    assert not any(call[0] == "scp" for call in calls)


def test_ambiguous_submit_reconciles_one_exact_job(tmp_path, monkeypatch):
    local = _script(tmp_path)
    digest = hashlib.sha256(local.read_bytes()).hexdigest()
    queries = iter((b"", b"", b"777\n"))

    def run(argv, *, input_bytes=None):
        command = argv[-1] if argv and argv[0] == "ssh" else ""
        if "squeue -h --name" in command:
            return _completed(argv, stdout=next(queries))
        if "sha256sum" in command:
            return _completed(argv, stdout=f"{digest}  remote\n".encode())
        if "sbatch --parsable" in command:
            return _completed(argv, rc=255, stderr=b"connection closed")
        return _completed(argv)

    monkeypatch.setattr(submit, "_run", run)
    result = submit.submit_action(
        login="user@login", job_id="iaa-job-1", rendered_script=local,
        remote_script=pathlib.Path("/lustre/run/sbatch/job.sbatch"),
    )
    assert result["backend_ref"] == "777"
    assert result["reconciled"] is True


def test_lost_submit_response_polls_bounded_until_exact_job_is_visible(
    monkeypatch,
):
    replies = iter(([], [], ["777"]))
    clock = [0.0]
    sleeps = []
    monkeypatch.setattr(submit, "_exact_job_ids", lambda *_args: next(replies))
    monkeypatch.setattr(submit.time, "monotonic", lambda: clock[0])

    def advance(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(submit.time, "sleep", advance)
    assert submit._reconcile_submitted_job("user@login", "iaa-job-1") == ["777"]  # noqa: SLF001
    assert sleeps == [2.0, 2.0]
    assert clock[0] < submit.SUBMIT_RECONCILE_TIMEOUT_SECONDS


def test_lost_submit_response_reconciliation_has_finite_deadline(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(submit, "_exact_job_ids", lambda *_args: [])
    monkeypatch.setattr(submit.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        submit.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    assert submit._reconcile_submitted_job("user@login", "iaa-job-1") == []  # noqa: SLF001
    assert clock[0] == submit.SUBMIT_RECONCILE_TIMEOUT_SECONDS


def test_ambiguous_submit_without_exact_job_fails_closed(tmp_path, monkeypatch):
    local = _script(tmp_path)
    _mock_run(monkeypatch, local, submit_out=b"not-a-handle\n")
    monkeypatch.setattr(submit, "_reconcile_submitted_job", lambda *_args: [])
    with pytest.raises(ValueError, match="ambiguous sbatch result"):
        submit.submit_action(
            login="user@login", job_id="iaa-job-1", rendered_script=local,
            remote_script=pathlib.Path("/lustre/run/sbatch/job.sbatch"),
        )
