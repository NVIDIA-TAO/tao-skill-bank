# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for prepare.py — the build step server.py depends on."""

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare  # noqa: E402

LIGHT, EXT = "SolderLight", ".jpg"


@pytest.fixture
def run(tmp_path):
    """A workspace holding one KPI image, and its results/run_X/ dir."""
    ws = tmp_path / "ws"
    (ws / "kpi" / "images" / "board_A").mkdir(parents=True)
    (ws / "kpi" / "images" / "board_A" / f"C1_{LIGHT}{EXT}").write_bytes(b"x")
    pd.DataFrame([["board_A", "golden/", "PASS", "C1"]],
                 columns=["input_path", "golden_path", "label", "object_name"]
                 ).to_csv(ws / "kpi" / "testing_set.csv", index=False)
    rd = ws / "results" / "run_X"
    rd.mkdir(parents=True)
    (rd / "deft_state.json").write_text('{"run_id": "run_X", "config": {}}')
    (rd / "mission_control").mkdir()
    return rd


@pytest.fixture(autouse=True)
def no_inherited_image_env(monkeypatch):
    """Never let the developer's own exports decide a test's outcome."""
    monkeypatch.delenv("TAO_DS_IMAGE", raising=False)
    monkeypatch.delenv("TAO_SKILL_BANK_PATH", raising=False)


@pytest.fixture
def skill_bank(tmp_path, monkeypatch):
    """A TAO_SKILL_BANK_PATH whose versions.yaml names a data_services image."""
    sb = tmp_path / "bank"
    sb.mkdir()
    (sb / "versions.yaml").write_text(yaml.safe_dump(
        {"images": {"tao_toolkit": {"data_services": "nvcr.io/fake/ds:1.0"}}}))
    monkeypatch.setenv("TAO_SKILL_BANK_PATH", str(sb))
    return sb


@pytest.fixture
def docker(monkeypatch):
    """Capture the docker argv instead of running it; opt in to writing output."""
    calls = []

    def fake_sh(cmd, **kw):
        calls.append([str(c) for c in cmd])
        if getattr(fake_sh, "writes", True):
            out = next(c.split("=", 1)[1] for c in map(str, cmd)
                       if c.startswith("output_parquet="))
            pd.DataFrame({"filepath": ["x"]}).to_parquet(out, index=False)

    monkeypatch.setattr(prepare, "sh", fake_sh)
    fake_sh.calls = calls
    return fake_sh


def _spec(rd, input_map):
    """Write the run's baseline_spec.yaml with a given lighting layout."""
    (rd / "baseline_spec.yaml").write_text(yaml.safe_dump(
        {"dataset": {"classify": {"input_map": input_map, "image_ext": EXT}}}))


# --------------------------------------------------------------------------- #
# _redact — secrets must not reach the printed command line
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tok, expected", [
    ("HF_TOKEN=hunter2", "HF_TOKEN=***"),
    ("NGC_KEY=abc", "NGC_KEY=***"),
    ("MY_SECRET=abc", "MY_SECRET=***"),
    ("DB_PASSWORD=abc", "DB_PASSWORD=***"),
    ("hf_token=abc", "hf_token=***"),          # matched case-insensitively
])
def test_credential_assignments_are_masked(tok, expected):
    assert prepare._redact(tok) == expected


@pytest.mark.parametrize("tok", [
    "docker", "--gpus", "all",
    "USER=athouta",                            # not a credential name
    "HOME=/tmp",
    "input_parquet=/ws/embed_input.parquet",   # paths must stay readable
])
def test_ordinary_arguments_pass_through_unchanged(tok):
    assert prepare._redact(tok) == tok


def test_only_the_value_is_masked_not_the_name():
    # the name has to survive, or the printed command is undebuggable
    assert prepare._redact("HF_TOKEN=hunter2").startswith("HF_TOKEN=")


def test_an_empty_credential_is_still_masked():
    # HF_TOKEN= is what an unset var expands to; do not leak whether it was set
    assert prepare._redact("HF_TOKEN=") == "HF_TOKEN=***"


# --------------------------------------------------------------------------- #
# resolve_ds_image — the container image comes from the skill bank
# --------------------------------------------------------------------------- #

def test_the_preflight_exported_var_is_used_when_set(monkeypatch):
    monkeypatch.setenv("TAO_DS_IMAGE", "nvcr.io/pinned/ds:7.0.1")
    assert prepare.resolve_ds_image() == "nvcr.io/pinned/ds:7.0.1"


def test_the_exported_var_wins_over_the_skill_bank(skill_bank, monkeypatch):
    # Pre-Flight pins the image for the whole run; the bank must not override it
    monkeypatch.setenv("TAO_DS_IMAGE", "nvcr.io/pinned/ds:7.0.1")
    assert prepare.resolve_ds_image() == "nvcr.io/pinned/ds:7.0.1"


def test_the_skill_bank_supplies_the_image_when_the_var_is_absent(skill_bank):
    assert prepare.resolve_ds_image() == "nvcr.io/fake/ds:1.0"


def test_the_bank_is_consulted_for_a_plain_clone(skill_bank, monkeypatch):
    # no Pre-Flight has run, so only the plugin hook's path is available
    monkeypatch.delenv("TAO_DS_IMAGE", raising=False)
    assert prepare.resolve_ds_image() == "nvcr.io/fake/ds:1.0"


def test_neither_source_returns_none():
    assert prepare.resolve_ds_image() is None


@pytest.mark.parametrize("var", ["TAO_DS_IMAGE", "TAO_SKILL_BANK_PATH"])
def test_an_empty_value_is_treated_as_unset(monkeypatch, var):
    # bash substitutes "" for an unset var; that must not read as configured
    monkeypatch.setenv(var, "")
    assert prepare.resolve_ds_image() is None


def test_a_bad_skill_bank_path_raises_rather_than_returning_none(monkeypatch, tmp_path):
    # None means "unset" to the caller; a broken bank must not masquerade as that
    monkeypatch.setenv("TAO_SKILL_BANK_PATH", str(tmp_path / "nope"))
    with pytest.raises(OSError):
        prepare.resolve_ds_image()


# --------------------------------------------------------------------------- #
# mount_roots — one -v per top-level root, image paths valid inside
# --------------------------------------------------------------------------- #

def test_paths_under_one_root_produce_one_mount():
    assert prepare.mount_roots(["/home/a/x.jpg", "/home/b/y.jpg"],
                               Path("/home/cache")) == ["-v", "/home:/home"]


def test_each_distinct_root_is_mounted_once():
    got = prepare.mount_roots(["/home/a/x.jpg", "/data/b/y.jpg", "/data/c/z.jpg"],
                              Path("/home/cache"))
    assert got == ["-v", "/data:/data", "-v", "/home:/home"]


def test_the_cache_root_is_mounted_even_when_no_image_shares_it():
    # the container writes the parquet there; an unmounted cache loses the output
    got = prepare.mount_roots(["/data/x.jpg"], Path("/home/run/mission_control"))
    assert got == ["-v", "/data:/data", "-v", "/home:/home"]


def test_mounts_map_each_root_onto_itself():
    # host paths are written into the parquet verbatim, so they must resolve
    # to the same path inside the container
    for src, dst in [tuple(m.split(":")) for m in
                     prepare.mount_roots(["/home/x.jpg", "/data/y.jpg"], Path("/tmp"))[1::2]]:
        assert src == dst


def test_mount_order_is_deterministic():
    a = prepare.mount_roots(["/home/x.jpg", "/data/y.jpg"], Path("/tmp"))
    b = prepare.mount_roots(["/data/y.jpg", "/home/x.jpg"], Path("/tmp"))
    assert a == b


def test_a_relative_cache_is_resolved_before_its_root_is_taken(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    got = prepare.mount_roots([], Path("mission_control"))
    assert got == ["-v", f"/{tmp_path.resolve().parts[1]}:/{tmp_path.resolve().parts[1]}"]


# --------------------------------------------------------------------------- #
# embed_all — idempotency
# --------------------------------------------------------------------------- #

def test_a_cached_parquet_short_circuits_the_container(run, skill_bank, docker):
    cache = run / "mission_control"
    pd.DataFrame({"filepath": ["a", "b"]}).to_parquet(cache / "embeddings.parquet")
    assert prepare.embed_all(run, run.parent.parent, cache) is None
    assert docker.calls == []


def test_force_re_embeds_over_a_cached_parquet(run, skill_bank, docker):
    cache = run / "mission_control"
    pd.DataFrame({"filepath": ["a"]}).to_parquet(cache / "embeddings.parquet")
    assert prepare.embed_all(run, run.parent.parent, cache, force=True) is None
    assert len(docker.calls) == 1


def test_a_successful_embed_leaves_the_cache_a_later_run_reuses(run, skill_bank, docker):
    cache = run / "mission_control"
    assert prepare.embed_all(run, run.parent.parent, cache) is None
    assert prepare.embed_all(run, run.parent.parent, cache) is None
    assert len(docker.calls) == 1          # second call served from cache


# --------------------------------------------------------------------------- #
# embed_all — multi-lighting: channel 0 is what gets embedded
# --------------------------------------------------------------------------- #

def test_a_single_light_run_embeds_that_light(run, skill_bank, docker):
    _spec(run, {LIGHT: 0})
    prepare.embed_all(run, run.parent.parent, run / "mission_control")
    assert len(docker.calls) == 1


def test_multi_light_embeds_channel_zero_not_dict_order(run, skill_bank, docker, capsys):
    # captures are stacked channels of one sample, so channel 0 keys the map
    _spec(run, {"WhiteLight": 1, LIGHT: 0, "UVLight": 2})
    prepare.embed_all(run, run.parent.parent, run / "mission_control")
    assert f"embedding channel 0 ({LIGHT})" in capsys.readouterr().out


def test_multi_light_yields_one_row_per_component_not_per_capture(run, skill_bank, docker):
    _spec(run, {LIGHT: 0, "WhiteLight": 1, "UVLight": 2})
    cache = run / "mission_control"
    prepare.embed_all(run, run.parent.parent, cache)
    assert len(pd.read_parquet(cache / "embed_input.parquet")) == 1


def test_a_missing_spec_falls_back_to_the_default_light(run, skill_bank, docker):
    # no baseline_spec.yaml written; the fixture image uses the default naming
    prepare.embed_all(run, run.parent.parent, run / "mission_control")
    assert len(docker.calls) == 1


# --------------------------------------------------------------------------- #
# embed_all — every failure returns a reason, never a silent success
# --------------------------------------------------------------------------- #

def test_no_collectable_images_reports_why(tmp_path, skill_bank, docker):
    rd = tmp_path / "ws" / "results" / "run_Y"
    rd.mkdir(parents=True)
    (rd / "deft_state.json").write_text("{}")
    cache = rd / "mission_control"
    cache.mkdir()
    problem = prepare.embed_all(rd, tmp_path / "ws", cache)
    assert problem and "no images collected" in problem
    assert docker.calls == []


def test_no_resolvable_image_reports_both_ways_to_supply_one(run, docker):
    problem = prepare.embed_all(run, run.parent.parent, run / "mission_control")
    assert problem and "TAO_DS_IMAGE" in problem and "TAO_SKILL_BANK_PATH" in problem
    assert docker.calls == []


def test_the_resolved_image_is_the_one_launched(run, docker, monkeypatch):
    monkeypatch.setenv("TAO_DS_IMAGE", "nvcr.io/pinned/ds:7.0.1")
    prepare.embed_all(run, run.parent.parent, run / "mission_control")
    assert "nvcr.io/pinned/ds:7.0.1" in docker.calls[0]


# --------------------------------------------------------------------------- #
# embed_all — no credential may reach the container's argv
# --------------------------------------------------------------------------- #

def test_the_token_is_inherited_not_written_into_argv(run, skill_bank, docker, monkeypatch):
    # argv is world-readable in /proc for the life of the container
    monkeypatch.setenv("HF_TOKEN", "hunter2")
    prepare.embed_all(run, run.parent.parent, run / "mission_control")
    argv = docker.calls[0]
    assert "hunter2" not in " ".join(argv)
    assert argv[argv.index("HF_TOKEN") - 1] == "-e"


def test_an_unset_token_does_not_become_an_empty_assignment(run, skill_bank, docker, monkeypatch):
    # -e HF_TOKEN= would override a value baked into the image with nothing
    monkeypatch.delenv("HF_TOKEN", raising=False)
    prepare.embed_all(run, run.parent.parent, run / "mission_control")
    assert not [a for a in docker.calls[0] if a.startswith("HF_TOKEN=")]


def test_no_launch_argument_carries_a_credential_value(run, skill_bank, docker, monkeypatch):
    for var in ("HF_TOKEN", "NGC_KEY"):
        monkeypatch.setenv(var, f"secret-{var}")
    prepare.embed_all(run, run.parent.parent, run / "mission_control")
    for arg in docker.calls[0]:
        assert "secret-" not in arg


def test_a_container_that_exits_zero_without_writing_is_a_failure(run, skill_bank, docker):
    # exit status alone does not prove the parquet exists
    docker.writes = False
    problem = prepare.embed_all(run, run.parent.parent, run / "mission_control")
    assert problem and "wrote no" in problem


def test_success_returns_none_and_writes_the_parquet(run, skill_bank, docker):
    cache = run / "mission_control"
    assert prepare.embed_all(run, run.parent.parent, cache) is None
    assert (cache / "embeddings.parquet").is_file()


# --------------------------------------------------------------------------- #
# main — the exit code is the contract: 0 means server.py can start
# --------------------------------------------------------------------------- #

def _argv(monkeypatch, run):
    monkeypatch.setattr(sys, "argv", ["prepare.py", "--run", str(run)])


def test_a_run_that_is_not_a_directory_exits_nonzero(monkeypatch, tmp_path):
    _argv(monkeypatch, tmp_path / "nope")
    with pytest.raises(SystemExit) as e:
        prepare.main()
    assert e.value.code != 0


def test_a_directory_without_deft_state_is_rejected(monkeypatch, tmp_path):
    _argv(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as e:
        prepare.main()
    assert "not a DEFT results dir" in str(e.value.code)


def test_a_failed_embed_exits_nonzero_instead_of_printing_serve(
        monkeypatch, run, docker, capsys):
    monkeypatch.delenv("TAO_SKILL_BANK_PATH", raising=False)
    _argv(monkeypatch, run)
    with pytest.raises(SystemExit) as e:
        prepare.main()
    assert "cannot start without" in str(e.value.code)
    assert "Serve the map" not in capsys.readouterr().out


def test_an_unindexable_run_exits_nonzero_rather_than_deferring_to_serve(
        monkeypatch, run, skill_bank, docker):
    # embeddings land, but the index cannot be built — server.py would fail too
    _argv(monkeypatch, run)
    with pytest.raises(SystemExit) as e:
        prepare.main()
    assert "would fail the same way" in str(e.value.code)


# --------------------------------------------------------------------------- #
# air-gap — declared, never inferred; no network action may be attempted
# --------------------------------------------------------------------------- #

@pytest.fixture
def siglip(tmp_path, monkeypatch):
    """A staged google/siglip-base-patch16-224 snapshot, as Pre-Flight resolves it."""
    snap = tmp_path / "hf" / "models--google--siglip-base-patch16-224" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    monkeypatch.setenv("SIGLIP_MODEL_PATH", str(snap))
    return snap


def test_air_gap_is_off_unless_declared(monkeypatch):
    monkeypatch.delenv("AIR_GAPPED", raising=False)
    assert prepare.air_gapped() is False


@pytest.mark.parametrize("val, expected", [("1", True), ("0", False), ("", False), ("true", False)])
def test_only_an_explicit_1_activates_air_gap(monkeypatch, val, expected):
    monkeypatch.setenv("AIR_GAPPED", val)
    assert prepare.air_gapped() is expected


def test_the_staged_snapshot_is_used_instead_of_the_hub_id(run, skill_bank, docker, siglip):
    prepare.embed_all(run, run.parent.parent, run / "mission_control")
    spec = (run / "mission_control" / "embedding_spec.yaml").read_text()
    assert str(siglip) in spec
    assert "google/siglip-base-patch16-224" not in spec.split("model_path:")[1]


def test_the_snapshot_root_is_mounted_so_the_spec_path_resolves(run, skill_bank, docker, siglip, tmp_path):
    prepare.embed_all(run, run.parent.parent, run / "mission_control")
    argv = docker.calls[0]
    root = f"/{siglip.resolve().parts[1]}"
    assert f"{root}:{root}" in argv


def test_the_hub_id_is_still_used_when_nothing_is_staged(run, skill_bank, docker, monkeypatch):
    monkeypatch.delenv("SIGLIP_MODEL_PATH", raising=False)
    prepare.embed_all(run, run.parent.parent, run / "mission_control")
    assert "google/siglip-base-patch16-224" in (run / "mission_control" / "embedding_spec.yaml").read_text()


def test_air_gap_forces_the_libraries_offline(run, skill_bank, docker, siglip, monkeypatch):
    monkeypatch.setenv("AIR_GAPPED", "1")
    prepare.embed_all(run, run.parent.parent, run / "mission_control")
    argv = docker.calls[0]
    assert "HF_HUB_OFFLINE=1" in argv and "TRANSFORMERS_OFFLINE=1" in argv


def test_a_networked_run_does_not_force_offline(run, skill_bank, docker, siglip):
    prepare.embed_all(run, run.parent.parent, run / "mission_control")
    assert "HF_HUB_OFFLINE=1" not in docker.calls[0]


def test_air_gap_without_a_staged_snapshot_refuses_to_launch(run, skill_bank, docker, monkeypatch):
    # the container would reach for HuggingFace and hang or fail obscurely
    monkeypatch.setenv("AIR_GAPPED", "1")
    monkeypatch.delenv("SIGLIP_MODEL_PATH", raising=False)
    problem = prepare.embed_all(run, run.parent.parent, run / "mission_control")
    assert problem and "SIGLIP_MODEL_PATH" in problem
    assert docker.calls == []
