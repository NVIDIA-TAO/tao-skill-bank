# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the Windows Cosmos-3 path — runnable on Linux.

Nothing here may depend on Windows behaviour: the suite has to keep passing on
the Linux and macOS machines the skills are written on, and only the dedicated
job in .github/workflows/windows-tests.yml runs it on Windows. So these assert
on FILE CONTENT and on module structure instead — the machine-readable
contract, the two platform skills' metadata, and the places in the shared
scripts where the Windows platforms had to be registered.

That restriction is the point rather than a compromise. The values these guard
are the ones whose loss produces a run that EXITS 0 with a wrong answer, and
those are precisely the failures no exit code will ever report:

  * an old ``attn_implementation=sdpa`` override is reintroduced without the
    W14 attention-mask gate, and the run exits successfully while learning the
    label prior;
  * ``answer_type`` drifts back to ``"letter"``, which clamps max_tokens to 10
    inside the evaluator, truncates the reasoner mid-``<think>``, and reports
    ``Accuracy: 0.0000`` at exit 0;
  * ``--base-model-path-or-uri`` is dropped from the export step, the output
    still carries LoRA tensors, and the server answers with the BASELINE model.

Nothing below needs a GPU, a Windows host, a container, or credentials.
"""

from __future__ import annotations

import importlib.util
import inspect
import re
import sys
import tomllib
import types
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
MODEL_SKILL = REPO / "skills/models/tao-finetune-cosmos-reason"
CONTRACT_PATH = MODEL_SKILL / "references/cosmos3-windows-contract.toml"
PLATFORM_DIR = REPO / "skills/platform"
WINDOWS_PLATFORMS = ("tao-run-on-windows-docker", "tao-run-on-windows-venv")

sys.path.insert(0, str(REPO / "scripts"))
import check_tao_launch_preflight as preflight  # noqa: E402


@pytest.fixture(scope="module")
def contract() -> dict:
    """Parse with tomllib only — the contract promises stdlib-only readers."""
    return tomllib.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Shape
# --------------------------------------------------------------------------- #

def test_contract_parses_with_the_standard_library_only():
    """No third-party TOML parser, on any consumer.

    The file documents ``tomllib.loads(...)`` as its reader, so it must stay
    inside what tomllib accepts — the whole point of shipping the values as
    data is that a skill, a script or a downstream tool can read them without
    taking on a dependency.
    """
    assert CONTRACT_PATH.is_file(), f"{CONTRACT_PATH} is the single source of truth"
    parsed = tomllib.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert re.fullmatch(r"\d+\.\d+\.\d+", parsed["schema_version"]), (
        "schema_version must be present and semver-shaped: consumers key their "
        "compatibility off it")


REQUIRED_TABLES = (
    "image", "runtime", "actions", "overrides", "lora", "evaluate",
    "export", "inference", "checkpoint", "env", "limits",
)


@pytest.mark.parametrize("table", REQUIRED_TABLES)
def test_contract_carries_every_required_table(contract, table):
    """A consumer reads these by key; a missing table is a KeyError at runtime."""
    assert isinstance(contract.get(table), dict), (
        f"[{table}] is missing from {CONTRACT_PATH.name}. Skills and downstream "
        f"tools read this file instead of hardcoding values, so a dropped table "
        f"breaks them with a KeyError rather than a diff")


def test_contract_declares_all_five_actions_and_their_packages(contract):
    """Four of the five entrypoints resolve into cosmos-rl, not cosmos-framework.

    A reader that assumes one package builds an image that cannot serve export,
    evaluate, inference or the microservice.
    """
    names = contract["actions"]["names"]
    assert names == ["train", "export", "evaluate", "inference",
                     "inference-microservice"]
    packages = {name: contract["actions"][name]["package"] for name in names}
    assert packages["train"] == "cosmos-framework"
    assert [packages[n] for n in names if n != "train"] == ["cosmos-rl"] * 4
    for name in names:
        assert contract["actions"][name]["entrypoint"] == (
            contract["actions"]["entrypoint_prefix"] + name)


# --------------------------------------------------------------------------- #
# The two-policy attention trap, asserted as data
# --------------------------------------------------------------------------- #

def test_attention_debug_overrides_are_not_in_the_default_route(contract):
    """Native cosmos attention is the validated default; retain debug history.

    SDPA/eager needs the W14 attention-mask gate. When it is selected for a
    controlled debug run, both policy configs still have to change together
    because VLMModel reads the nested one.
    """
    train = contract["overrides"]["train"]
    pair = [
        "policy.attn_implementation=sdpa",
        "model.config.policy.attn_implementation=sdpa",
    ]
    assert not any(value in train for value in pair)
    debug = contract["overrides"]["optional_attention_debug"]
    assert debug["requires_w14_mask_gate"] is True
    assert debug["values"] == pair
    assert "VLMModel" in debug["reason"]


def test_override_count_matches_the_list(contract):
    overrides = contract["overrides"]
    assert overrides["count"] == len(overrides["train"]) == 11
    assert overrides["separator"] == "--"
    # torchrun's shipped launchers are bash and hardcode --nproc_per_node=8.
    assert overrides["launcher"] == "python"


def test_command_line_only_overrides_are_still_marked(contract):
    """Nine of the eleven cannot be folded into a recipe TOML.

    SFTExperimentConfig rejects ``model.config.*`` and ``dataloader_*`` with
    "Extra inputs are not permitted", and TOML has no null literal for
    prefetch_factor. If that count is edited down, someone will try to move
    them into the recipe and the run will refuse to start.
    """
    train = contract["overrides"]["train"]
    assert contract["overrides"]["not_foldable_into_recipe_toml"] == 9
    assert 0 < contract["overrides"]["not_foldable_into_recipe_toml"] < len(train)
    # The prefetch_factor pair is the unarguable part: TOML has no null literal,
    # so those two can only ever be passed on the command line.
    assert [o for o in train if o.endswith("prefetch_factor=null")] == [
        "dataloader_train.prefetch_factor=null",
        "dataloader_val.prefetch_factor=null",
    ]


@pytest.mark.parametrize("override", [
    "optimizer=adamw",                       # fusedadamw imports transformer_engine
    "trainer.distributed_parallelism=ddp",
    "model.config.policy.lora_enabled=true",  # the only 8.8B config that fits
    "dataloader_train.num_workers=0",         # Windows spawns rather than forks
    "dataloader_train.prefetch_factor=null",
    "dataloader_val.num_workers=0",
])
def test_named_overrides_survive(contract, override):
    assert override in contract["overrides"]["train"]


def test_every_override_has_a_recorded_reason(contract):
    """An override with no reason is the one a future editor deletes."""
    keys = {o.split("=", 1)[0] for o in contract["overrides"]["train"]}
    assert keys == set(contract["overrides"]["why"])


# --------------------------------------------------------------------------- #
# evaluate: the settings that stop a silent wrong answer
# --------------------------------------------------------------------------- #

def test_evaluate_is_freeform_not_letter(contract):
    """``answer_type = "letter"`` clamps max_tokens to 10 inside the evaluator.

    It discards the configured value, truncates a reasoner mid-``<think>`` so
    every reply is byte-identical, reports "Accuracy: 0.0000" — and exits 0.
    Freeform loses nothing: MCQ letter extraction keys on task.type, and the
    scorer strips ``</think>`` before matching.
    """
    evaluate = contract["evaluate"]
    assert evaluate["answer_type"] == "freeform"
    assert evaluate["answer_type_letter_clamps_max_tokens_to"] == 10
    assert evaluate["max_tokens"] > 10, (
        "max_tokens must leave room for the <think> block to close")


def test_evaluate_uses_torchcodec(contract):
    """The default decoder cannot initialize where NVENC is absent.

    PyNvVideoCodec probes for the NVENC *encoder* at init although it only
    decodes, and the GPU device class projects nvcuvid.dll (NVDEC) but not
    nvEncodeAPI64.dll: "RuntimeError: NVENC library file is not found".
    """
    evaluate = contract["evaluate"]
    assert evaluate["video_decoder"] == "torchcodec"
    assert evaluate["default_video_decoder"] != "torchcodec", (
        "the contract must keep naming the DEFAULT decoder, or the override "
        "reads as redundant and gets dropped")


def test_evaluate_never_sets_num_frames(contract):
    """num_frames is inert — it must not appear as a setting anyone can copy.

    runtime.py propagates (fps, num_frames, min_pixels, max_pixels,
    total_pixels) but qwen-vl-utils keys off "nframes", so the cap is accepted,
    logged and ignored: a 77 s clip decodes whole and asks for 376.52 GiB. The
    work has to be bounded with fps AND total_pixels instead.
    """
    evaluate = contract["evaluate"]
    assert "num_frames" not in evaluate, (
        "[evaluate] sets num_frames — it does not reach the decoder, so it "
        "reads as a frame budget while enforcing nothing")
    assert evaluate["num_frames_is_inert"] is True
    assert set(evaluate["frame_budget_keys"]) == {"fps", "total_pixels"}
    assert evaluate["fps"] and evaluate["total_pixels"], (
        "both bounds must carry a real value; either one alone is not a budget")
    assert evaluate["max_pixels_is_per_image"] is True, (
        "max_pixels is per-image; total_pixels is the video-wide cap and is "
        "the one that decides whether the decode fits in memory")


# --------------------------------------------------------------------------- #
# export: the post-condition that catches a baseline-serving model
# --------------------------------------------------------------------------- #

def test_export_names_the_base_model_flag_and_the_zero_lora_postcondition(contract):
    """Without --base-model-path-or-uri a LoRA export still succeeds.

    The output keeps its lora tensors and the server then answers with the
    BASELINE model. Nothing in the exit code or the logs says so, which is why
    the contract carries a post-condition to check every time: zero keys
    matching 'lora' in the safetensors index.
    """
    export = contract["export"]
    assert export["base_model_arg"] == "--base-model-path-or-uri"
    assert export["base_model_arg_required_for_lora"] is True
    assert export["expected_lora_keys"] == 0
    assert export["verify_index"] == "model.safetensors.index.json"
    assert export["verify_index"] in export["verify_command"]
    assert "lora" in export["verify_command"].lower(), (
        "the verify command must actually count lora keys, not just open the index")
    assert export["input_format"] == "dcp" and "safetensors" in export["output_format"]


# --------------------------------------------------------------------------- #
# Release management: the image is stamped, not hand-written
# --------------------------------------------------------------------------- #

def test_contract_image_matches_versions_yaml(contract):
    """[image].uri must equal what versions.yaml resolves for its versions-key.

    The contract is the single source of truth for the skill, but versions.yaml
    is the single source of truth for the tag. Drift between them means a skill
    that pulls an image the release never published.
    """
    key = contract["image"]["versions_key"]
    versions = yaml.safe_load((REPO / "versions.yaml").read_text(encoding="utf-8"))
    resolved = versions
    for part in key.split("."):
        assert isinstance(resolved, dict) and part in resolved, (
            f"versions.yaml has no key {key} (stopped at {part!r}); the "
            f"contract's versions_key must resolve or the stamper cannot "
            f"maintain it")
        resolved = resolved[part]
    assert contract["image"]["uri"] == resolved, (
        f"[image].uri drifted from versions.yaml:{key}. Edit versions.yaml and "
        f"run scripts/stamp_versions.py; never hand-edit the stamped value")


def test_contract_image_line_carries_the_stamp_marker():
    """The stamper only rewrites lines it can see.

    Without the marker comment the URI is invisible to both stamping and
    `--check`, and it silently rots across a release bump.
    """
    line = next(
        raw for raw in CONTRACT_PATH.read_text(encoding="utf-8").splitlines()
        if raw.strip().startswith("uri =")
    )
    assert "versions-key: images.tao_toolkit.cosmos_framework_windows" in line


def test_device_class_guid_is_identical_in_the_contract_and_the_preflight(contract):
    """One GUID, two files. A copy that drifts attaches no GPU at all."""
    assert contract["runtime"]["device_class"] == preflight.WINDOWS_GPU_DEVICE_CLASS
    assert any(preflight.WINDOWS_GPU_DEVICE_CLASS in flag
               for flag in contract["runtime"]["required_run_flags"])
    assert "--isolation=process" in contract["runtime"]["required_run_flags"], (
        "the daemon default is hyperv isolation, which has NO GPU path")


def test_contract_records_the_structural_single_gpu_limit(contract):
    """Multi-GPU is impossible here, not untested.

    gloo cannot run a collective on a CUDA tensor on Windows and NCCL has no
    Windows build, so the process dies with 0xC0000005 and no traceback. A
    router that reads max_gpus > 1 from this file would plan a run that cannot
    start.
    """
    limits = contract["limits"]
    assert limits["multi_gpu"] is False and limits["multi_node"] is False
    assert limits["max_gpus"] == limits["max_nodes"] == limits["world_size"] == 1
    assert "gloo" in limits["multi_gpu_reason"] and "NCCL" in limits["multi_gpu_reason"]


def test_contract_requires_both_tao_api_env_vars_together(contract):
    """TAO_API_JOB_ID alone is fatal for training at iteration 0.

    Every action refuses to start without TAO_API_JOB_ID, and training then
    dies with "TAO_API_RESULTS_DIR is required when TAO_API_JOB_ID is set".
    The two requirements are opposite, so one shared env block must set both.
    """
    env = contract["env"]
    assert env["required_always"] == ["TAO_API_JOB_ID"]
    assert set(env["required_for_train"]) == {"TAO_API_JOB_ID", "TAO_API_RESULTS_DIR"}
    assert env["set_both_when_sharing_one_env_block"] is True


# --------------------------------------------------------------------------- #
# Registration of the two platforms in the shared scripts
# --------------------------------------------------------------------------- #

PLATFORM_SLUGS = ("windows-docker", "windows-venv")


@pytest.mark.parametrize("slug", PLATFORM_SLUGS)
def test_slug_is_registered_in_the_mount_proof_exemption(slug):
    """Site 1: has_unverified_remote_mounts.

    windows-docker bind-mounts a host directory and windows-venv runs on the
    host itself, so a plain `C:\\data\\...` path needs no mounted-volume proof.
    Omitting either slug rejects every Windows input path with "use s3://",
    which neither platform can act on: there is no object-store client on the
    Windows image and no sidecar stage for a single `docker run`.

    The exemption is only sound because each slug's own branch verifies those
    paths against the real filesystem — check_windows_docker() and
    check_windows_venv(). Test site 2 pins that half; exempting a slug whose
    branch does NOT check is how the existence check gets deleted outright.
    """
    source = inspect.getsource(preflight.has_unverified_remote_mounts)
    assert f'"{slug}"' in source, (
        f"{slug} is not exempt in has_unverified_remote_mounts, so every host "
        f"path it is given is refused with unactionable advice")


@pytest.mark.parametrize("slug", PLATFORM_SLUGS)
def test_slug_is_accounted_for_in_the_main_dispatch(slug):
    """Site 2: main()'s platform dispatch.

    Each slug gets its OWN branch. windows-venv used to fall through to
    check_env_only, which can only print "verify it is mounted in the job" —
    paired with the site-1 exemption that left it with no path check at all,
    and `--path train=C:\\does\\not\\exist` passed preflight. Both Windows
    platforms run the job on the machine preflight runs on, so both branches
    verify their paths locally.
    """
    source = inspect.getsource(preflight.main)
    assert slug in source, (
        f"{slug} appears nowhere in main()'s dispatch; a platform that is "
        f"neither routed nor documented as a fall-through is routed by accident")
    checker = getattr(preflight, f"check_{slug.replace('-', '_')}", None)
    assert checker is not None and f"{checker.__name__}(" in source, (
        f"{slug} is not routed to its own checker in main(); an exempted slug "
        f"without a branch that verifies its paths has no existence check left")


@pytest.mark.parametrize("slug", PLATFORM_SLUGS)
def test_platform_resolves_to_its_own_self_describing_record(slug):
    """The slug must not be aliased away before the dispatch sees it."""
    record = preflight.resolve_platform(REPO, slug)
    assert record["name"] == slug
    info = PLATFORM_DIR / f"tao-run-on-{slug}/references/skill_info.yaml"
    assert info.is_file(), f"{info} is what makes the platform self-describing"
    assert yaml.safe_load(info.read_text(encoding="utf-8"))["platform_slug"] == slug


@pytest.mark.parametrize(
    ("skill", "slug"),
    list(zip(WINDOWS_PLATFORMS, PLATFORM_SLUGS)),
    ids=PLATFORM_SLUGS,
)
def test_documented_open_uses_the_declared_slug(skill, slug):
    """The `--platform` value in the docs must be the slug the metadata declares.

    A job record opened under a different name is untrackable: nothing else
    knows to look for it.
    """
    text = (PLATFORM_DIR / skill / "SKILL.md").read_text(encoding="utf-8")
    assert f"--platform {slug}" in text


# --------------------------------------------------------------------------- #
# The job-record writer has to survive a Windows interpreter
# --------------------------------------------------------------------------- #

def _load_job_record_as_on_windows(monkeypatch):
    """Import tao_job_record with the module set a Windows interpreter offers.

    fcntl is removed and a msvcrt stand-in supplied. ``os.name`` is left alone
    on purpose: pathlib picks WindowsPath vs PosixPath from it, so setting it
    to "nt" on Linux makes ``Path(...)`` raise NotImplementedError and the test
    would measure pathlib rather than this module. What actually decides here
    is which modules exist, so that is what the simulation changes.
    """
    fake_msvcrt = types.ModuleType("msvcrt")
    fake_msvcrt.LK_NBLCK, fake_msvcrt.LK_UNLCK = 1, 0
    fake_msvcrt.calls = []
    fake_msvcrt.locking = lambda fd, mode, nbytes: fake_msvcrt.calls.append(mode)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setitem(sys.modules, "fcntl", None)  # -> ModuleNotFoundError

    spec = importlib.util.spec_from_file_location(
        "tao_job_record_windows_sim", REPO / "scripts/tao_job_record.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, fake_msvcrt


def test_job_record_imports_where_fcntl_does_not_exist(monkeypatch):
    """`import fcntl` at module scope made the writer unusable on Windows.

    Both Windows platforms document `tao_job_record.py open` as step one of
    submit, and the record-then-launch invariant means a submit with no id
    cannot launch at all — so an unguarded POSIX-only import does not degrade
    the platform, it stops every job on it.
    """
    module, _ = _load_job_record_as_on_windows(monkeypatch)
    assert module.fcntl is None
    assert module._POSIX is False


def test_windows_lock_really_locks(tmp_path, monkeypatch):
    """The fallback must take a lock, not quietly become a no-op.

    A no-op would pass every single-writer test and lose transitions only when
    an agent and a detached poller mark the same record at the same time.
    """
    module, fake_msvcrt = _load_job_record_as_on_windows(monkeypatch)
    with module._Locked(tmp_path / "job.json"):
        assert fake_msvcrt.calls == [fake_msvcrt.LK_NBLCK], (
            "the Windows lock must be acquired non-blocking and retried; "
            "msvcrt's blocking LK_LOCK mode retries once a second and then "
            "gives up, which turns a normal poll into a stall and an error")
    assert fake_msvcrt.calls == [fake_msvcrt.LK_NBLCK, fake_msvcrt.LK_UNLCK], (
        "the lock is not released on exit; the next mark, in any process, "
        "would block until the timeout")


@pytest.mark.parametrize("skill", WINDOWS_PLATFORMS)
def test_documented_open_is_never_wrapped_with_a_powershell_backtick(skill):
    """A backtick-continued `open` example would defeat the shared flags check.

    test_platform_contract.py verifies that every documented
    ``tao_job_record.py open`` carries all five required flags, and its capture
    understands only the POSIX ``\\``-newline continuation. PowerShell continues
    a line with a backtick, so a wrapped invocation truncates at the first line:
    the missing flags become invisible to CI and `open` exits 2 at runtime,
    which reads as "the platform is broken" rather than "the example is
    incomplete". Keep these invocations on one line.
    """
    text = (PLATFORM_DIR / skill / "SKILL.md").read_text(encoding="utf-8")
    wrapped = re.findall(r"tao_job_record\.py[\"']?\s+open\b[^\n]*`[ \t]*$",
                         text, re.M)
    assert not wrapped, (
        f"{skill}/SKILL.md wraps a `tao_job_record.py open` example with a "
        f"PowerShell backtick, which hides the rest of its flags from the "
        f"platform-contract check: {wrapped}")


def test_every_action_claimed_verified_has_evidence(contract):
    """`actions_verified_native` must be backed by a passing sub-table.

    The list and the evidence drifted apart once already: export and inference
    passed after the list was written, and for a while the contract said they
    were unverified while the sub-tables said PASS. Either direction of that
    skew is worse than no claim at all, because a reader trusts the list.
    """
    validation = contract["validation"]
    for action in validation["actions_verified_native"]:
        sub = validation.get(action)
        assert isinstance(sub, dict), (
            f"[validation] claims '{action}' is verified natively but there is "
            f"no [validation.{action}] table recording what was run")
        assert sub.get("status") == "PASS", (
            f"[validation.{action}].status is {sub.get('status')!r}, so the "
            f"action must not appear in actions_verified_native")


def test_verified_and_unverified_are_disjoint_and_cover_every_action(contract):
    """No action may be silently absent from both lists."""
    validation = contract["validation"]
    verified = set(validation["actions_verified_native"])
    unverified = set(validation["actions_unverified_native"])
    assert not (verified & unverified), (
        f"actions in both lists: {sorted(verified & unverified)}")
    assert verified | unverified == set(contract["actions"]["names"]), (
        "every action in [actions].names must be classified as verified or "
        "unverified; an action in neither list reads as supported by default")


def test_serving_stack_is_not_conflated_with_the_microservice_action(contract):
    """vLLM serving PASSED; the `inference-microservice` action did not run.

    They are different things — the former is the engine underneath the latter —
    and collapsing them would let "5 of 5 actions" be claimed off work that only
    proved the serving layer.
    """
    validation = contract["validation"]
    assert "inference-microservice" in validation["actions_unverified_native"]
    assert validation["vllm_serving"]["status"] == "PASS"
    assert validation["vllm_serving"]["entrypoint"].startswith("vllm."), (
        "[validation.vllm_serving].entrypoint must name vLLM's own server, not "
        "a cosmos-framework-* console script")


def test_export_records_the_zero_lora_postcondition(contract):
    """A non-zero count means the export silently serves the BASELINE."""
    assert contract["validation"]["export"]["leftover_lora_keys"] == 0
