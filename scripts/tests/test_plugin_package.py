# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the plugin package that scripts/build_plugin_package.py generates.

The plugin-level evaluation suite that the package carries has its own tests in
test_plugin_eval_suite.py.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_plugin_package as builder  # noqa: E402
import skill_layers  # noqa: E402
import test_session_start_hook as hook_checks  # noqa: E402

# The package is built from the files git tracks, so an extracted archive cannot build it.
pytestmark = pytest.mark.skipif(not (REPO_ROOT / ".git").exists(), reason="needs a git checkout")

BUILDER = REPO_ROOT / "scripts" / "build_plugin_package.py"
SUITE = REPO_ROOT / "evals" / "evals.json"
FLAT_BANK = builder.DEFAULT_OUTPUT.as_posix()
LAYER = "|".join(skill_layers.LAYERS)
# Every way a shipped file could still describe the layer layout: a path into a
# layer folder, a layer folder in a drawing of the tree, the repository root as
# the place to run from.
LAYER_REFERENCE = re.compile(
    rf"(?<![\w-])skills/(?:{LAYER})(?![\w-])|\.\./(?:{LAYER})/|`(?:{LAYER})/`"
    rf"|[├└]──[ \t]*(?:{LAYER})/(?=[ \t]*(?:#[^\n]*)?$)"
    rf"|`?tao-skill-bank`?[ \t]+repo(?:sitory)?[ \t]+root",
    re.M,
)
# A bank in the home folder that is not the plugin folder inside a checkout.
HOME_BANK = re.compile(
    r"(?:(?:~|\$HOME|\$\{HOME\})/|home\(\)\s*/\s*[\"'])tao-skill-bank(?!/"
    + re.escape(FLAT_BANK) + r")(?![\w-])"
)
# Shipped text that names one all the same: (file in the package, the match) -> why.
HOME_BANK_ALLOWED = {
    ("skills/tao-run-deft-aoi/scripts/align_token_usage.py", "~/tao-skill-bank"):
        "example of the working directory of an agent session, used to find its transcript",
    ("skills/tao-run-deft-aoi-cosmos3/scripts/align_token_usage.py", "~/tao-skill-bank"):
        "example of the working directory of an agent session, used to find its transcript",
    ("skills/tao-run-deft-aoi/scripts/resolve_versions_key.py", 'home() / "tao-skill-bank'):
        "last place searched for versions.yaml, which the root of a checkout carries as well; "
        "no path into skills/ is built from it",
}
# A placeholder for the bank that names a clone of the repository; the flat paths
# that follow such a placeholder do not exist below the root of a clone.
CLONE_PLACEHOLDER = re.compile(
    r"/path/to/tao-skill-bank(?!/" + re.escape(FLAT_BANK) + r")(?![\w.-])"
    r"|<[\w-]*tao-skill-bank>"
)
# Shipped text that uses one all the same: (file in the package, the match) -> why.
CLONE_PLACEHOLDER_ALLOWED = {}
# The limits of the verification pipeline for one plugin folder, written out
# here so that a wrong constant in the generator does not pass.
PIPELINE_MAX_FILES = 4096
PIPELINE_MAX_TOTAL_BYTES = 67_108_864   # 64 MiB
PIPELINE_MAX_FILE_BYTES = 16_777_216    # 16 MiB
PIPELINE_MAX_DEPTH = 64
PIPELINE_JUNK = (".DS_Store", "__pycache__", "__MACOSX")
# Agent Plugins 1.0.0 (https://agent-plugins.org/schemas/1.0.0/plugin.schema.json):
# these keys and no others, "$schema" and "name" required.
AGENT_PLUGINS_KEYS = {
    "$schema", "name", "version", "description", "author", "homepage", "repository", "license",
    "keywords", "extensions",
}
AGENT_PLUGINS_NAME = re.compile(r"^(?!.*(?:--|\.\.))[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")
TOP_LEVEL = {
    ".claude-plugin", ".codex-plugin", ".cursor-plugin", "AGENTS.md", "LICENSE", "catalog", "evals",
    "hooks", "plugin.json", "scripts", "skills", "templates", "versions.yaml",
}
REPOSITORY_ONLY = (
    "scripts/tests/", "scripts/ci/", "templates/skill-skeleton/", "templates/tests/", "docs/",
    "integrations/", ".github/", ".agents/",
)
HOOK_OUTPUT_CAP = 10_000


def _repository_skills():
    return sorted(path.parent.name for path in (REPO_ROOT / "skills").glob("*/*/SKILL.md"))


def _files(folder):
    return sorted(path for path in folder.rglob("*") if not path.is_dir())


def _text(path):
    return path.read_text(encoding="utf-8")


def _json(path):
    return json.loads(_text(path))


def _clean_environment(**extra):
    """The environment of a session without the plugin's variables."""
    unset = ("TAO_SKILL_BANK_PATH", "CLAUDE_PLUGIN_ROOT", "CURSOR_PLUGIN_ROOT", "PLUGIN_ROOT",
             "CLAUDE_ENV_FILE", "CODEX_HOME")
    environment = {key: value for key, value in os.environ.items() if key not in unset}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.update({key: str(value) for key, value in extra.items()})
    return environment


def _run(args, cwd, **extra):
    return subprocess.run(
        args, cwd=cwd, env=_clean_environment(**extra), capture_output=True, text=True, check=False
    )


def _shipped_texts(folder):
    for path in _files(folder):
        try:
            yield path.relative_to(folder).as_posix(), _text(path)
        except UnicodeDecodeError:
            continue


def _listing(folder):
    """Every file below a folder with its bytes and mode, .git included."""
    return {path.relative_to(folder).as_posix(): (path.read_bytes(), path.stat().st_mode)
            for path in _files(folder)}


def _add(folder, *relatives):
    for relative in relatives:
        (folder / relative).parent.mkdir(parents=True, exist_ok=True)
        (folder / relative).write_text("added after the build\n", encoding="utf-8")


@pytest.fixture(scope="module")
def package():
    return builder.build(REPO_ROOT)


@pytest.fixture(scope="module")
def folder(package, tmp_path_factory):
    target = tmp_path_factory.mktemp("package") / "plugins" / builder.PLUGIN_NAME
    builder.write(package, target)
    return target


# ─────────────────────────────── shape of the package ───────────────────────────────
def test_every_skill_is_one_folder_directly_under_skills(folder):
    skills = _repository_skills()
    assert len(skills) == len(set(skills)) > 0
    assert sorted(path.name for path in (folder / "skills").iterdir()) == skills
    for name in skills:
        frontmatter = _text(folder / "skills" / name / "SKILL.md").split("---", 2)[1]
        declared = rf"^name:\s*[\"']?{re.escape(name)}[\"']?\s*$"
        assert re.search(declared, frontmatter, re.M), name
    assert sorted(path.parent.name for path in folder.rglob("SKILL.md")) == skills


def test_membership_file_records_the_layer_of_every_skill(folder):
    layers = _json(folder / builder.MEMBERSHIP_PATH)["layers"]
    assert list(layers) == list(skill_layers.LAYERS)
    for layer, names in layers.items():
        in_repository = (REPO_ROOT / "skills" / layer).glob("*/SKILL.md")
        assert names == sorted(path.parent.name for path in in_repository)


def test_required_root_files_are_present(folder):
    for relative in (
        "plugin.json", ".claude-plugin/plugin.json", ".cursor-plugin/plugin.json",
        ".codex-plugin/plugin.json", "evals/evals.json", "LICENSE", "AGENTS.md", "versions.yaml",
        "hooks/hooks.json", "hooks/session_start.sh", "scripts/skill_layers.py",
        "scripts/tao_agent_identity.py",
        "catalog/schemas.manifest.json", "catalog/automl_support.json",
    ):
        assert (folder / relative).is_file(), relative


def test_nothing_that_must_stay_out_is_shipped(folder):
    shipped = [path.relative_to(folder).as_posix() for path in _files(folder)]
    assert {relative.split("/")[0] for relative in shipped} == TOP_LEVEL
    assert not [relative for relative in shipped if relative.endswith("marketplace.json")]
    assert not [relative for relative in shipped if relative.startswith(REPOSITORY_ONLY)]
    for relative in shipped:
        parts = relative.split("/")
        if parts[0] == "skills":
            assert builder.left_out(parts[1], "/".join(parts[2:])) is None, relative
        if parts[0] == "scripts":
            assert relative in builder.SUPPORT_FILES, f"{relative} needs a reason in SUPPORT_FILES"
        assert not builder.is_verification_file(relative), relative


def test_tests_fixtures_and_evaluation_configs_stay_in_the_repository(folder):
    kept = {f"skills/{skill}/{inside}" for skill, inside in builder.SKILL_FILES_KEPT}
    for path in _files(folder / "skills"):
        relative = path.relative_to(folder).as_posix()
        inside = "/".join(relative.split("/")[2:])
        if relative in kept:
            continue
        assert "tests" not in inside.split("/")[:-1], relative
        assert not builder.TEST_LIKE_NAME.search(inside), relative
        assert not re.fullmatch(r"eval[^/]*\.config", inside), relative
        assert path.suffix != ".jpg" or "fixtures" not in inside.split("/"), relative
    # The per-skill suites stay: a single skill can still be evaluated from its folder.
    for name in _repository_skills():
        source = next((REPO_ROOT / "skills").glob(f"*/{name}/evals/evals.json"))
        packaged = _json(folder / "skills" / name / "evals" / "evals.json")
        assert [task["id"] for task in packaged] == [task["id"] for task in _json(source)], name


def test_codex_installer_is_not_shipped_and_no_shipped_file_names_it(folder):
    assert not list(folder.rglob("install-codex-agents.sh"))
    assert not [relative for relative, text in _shipped_texts(folder)
                if "install-codex-agents" in text]
    # The identity and the helper that adds it to a project ship; the setup skill names
    # the helper, and nothing shipped tells the reader to copy the identity over a file.
    assert (folder / "AGENTS.md").is_file()
    helper = folder / "scripts" / "tao_agent_identity.py"
    assert helper.read_bytes() == (REPO_ROOT / "scripts" / "tao_agent_identity.py").read_bytes()
    assert helper.stat().st_mode & 0o777 == 0o755
    setup = _text(folder / "skills/tao-setup/SKILL.md")
    assert "/scripts/tao_agent_identity.py" in setup
    for relative, text in _shipped_texts(folder):
        assert not re.search(r"\bcp\b[^\n]*AGENTS\.md", text), relative
        assert "AGENTS.md.bak" not in text, relative


def test_every_listed_exception_still_applies(package, folder):
    """An entry that no longer matches anything would hide the next real case."""
    shipped = builder.with_folders(package.files)
    for skill, inside in builder.SKILL_FILES_KEPT:
        assert f"skills/{skill}/{inside}" in package.files, (skill, inside)
    for path, target in builder.LEFT_OUT_MENTIONS:
        assert path in package.files and target not in shipped, (path, target)
        assert Path(target).name in _text(folder / path), (path, target)
    for path, helper in builder.UNSHIPPED_SCRIPT_MENTIONS:
        assert helper in _text(folder / path) and helper not in package.files, (path, helper)
    for path, words in builder.LAYER_WORDS_ALLOWED:
        assert words in _text(folder / path), (path, words)
    for skill, inside in builder.ONE_SKILL_FILES_LEFT_OUT:
        assert f"skills/{skill}/{inside}" not in package.files, (skill, inside)


def test_pipeline_limits_hold_on_disk(folder):
    limits = (builder.MAX_FILES, builder.MAX_TOTAL_BYTES, builder.MAX_FILE_BYTES,
              builder.MAX_DEPTH)
    assert limits == (PIPELINE_MAX_FILES, PIPELINE_MAX_TOTAL_BYTES, PIPELINE_MAX_FILE_BYTES,
                      PIPELINE_MAX_DEPTH)
    assert builder.JUNK_NAMES == PIPELINE_JUNK and builder.JUNK_SUFFIXES == (".pyc",)
    files = _files(folder)
    assert len(files) <= PIPELINE_MAX_FILES
    assert sum(path.stat().st_size for path in files) <= PIPELINE_MAX_TOTAL_BYTES
    for path in files:
        relative = path.relative_to(folder)
        status = path.lstat()
        assert not path.is_symlink() and path.is_file(), relative
        assert status.st_size <= PIPELINE_MAX_FILE_BYTES, relative
        assert status.st_mode & 0o777 in (0o644, 0o755), relative
        assert status.st_nlink == 1, relative
        assert len(relative.parts) - 1 <= PIPELINE_MAX_DEPTH, relative
        assert not any(part in PIPELINE_JUNK or part.endswith(".pyc")
                       for part in relative.parts), relative
        if path.suffix in builder.TEXT_SUFFIXES:
            path.read_text(encoding="utf-8")  # every text file decodes
    assert not [path for path in folder.rglob("*") if path.is_symlink()]
    for relative in ("hooks/session_start.sh", "scripts/check_tao_launch_preflight.py"):
        assert (folder / relative).stat().st_mode & 0o777 == 0o755, relative


def test_folders_get_mode_755_whatever_the_umask(package, tmp_path):
    target = tmp_path / "strict" / builder.PLUGIN_NAME
    previous = os.umask(0o077)
    try:
        builder.write(package, target)
    finally:
        os.umask(previous)
    folders = [target, *(path for path in target.rglob("*") if path.is_dir())]
    assert {path.stat().st_mode & 0o777 for path in folders} == {0o755}
    assert {path.stat().st_mode & 0o777 for path in _files(target)} == {0o644, 0o755}


def test_no_shipped_text_file_points_into_a_layer_folder(folder):
    offenders = {}
    for relative, text in _shipped_texts(folder):
        found = sorted(set(LAYER_REFERENCE.findall(text)))
        found = [words for words in found if (relative, words) not in builder.LAYER_WORDS_ALLOWED]
        if found:
            offenders[relative] = found
    assert not offenders


def test_package_carries_the_repository_evaluation_suite_unchanged(folder):
    assert (folder / "evals" / "evals.json").read_bytes() == SUITE.read_bytes()


def test_git_ignores_results_at_the_root_and_nothing_in_the_package(package):
    prefix = f"plugins/{builder.PLUGIN_NAME}/"
    results = ".eval-results/run-1/eval-result.json"
    paths = [prefix + path for path in package.files] + [prefix + results, results]
    # Only the rules of this repository count, not the excludes file of the developer.
    ignored = subprocess.run(
        ["git", "-c", "core.excludesFile=/dev/null", "check-ignore", "--no-index", "--stdin"],
        cwd=REPO_ROOT, input="\n".join(paths), capture_output=True, text=True, check=False,
    ).stdout.split()
    assert ignored == [results]


# ─────────────────────────────── manifests ───────────────────────────────
def test_root_manifest_follows_agent_plugins_1_0_0(folder):
    manifest = _json(folder / "plugin.json")
    assert set(manifest) <= AGENT_PLUGINS_KEYS
    assert manifest["$schema"] == builder.AGENT_PLUGINS_SCHEMA
    assert manifest["name"] == builder.PLUGIN_NAME and len(manifest["name"]) <= 64
    assert AGENT_PLUGINS_NAME.match(manifest["name"])
    assert set(manifest["author"]) <= {"name", "email", "url"}
    assert all(isinstance(item, str) for item in manifest["keywords"])
    source = _json(REPO_ROOT / ".claude-plugin" / "plugin.json")
    for key in ("version", "description", "author", "repository", "license", "keywords"):
        assert manifest[key] == source[key], key


def test_manifests_agree_and_describe_the_flat_layout(folder):
    root = _json(folder / "plugin.json")
    claude = _json(folder / ".claude-plugin" / "plugin.json")
    cursor = _json(folder / ".cursor-plugin" / "plugin.json")
    codex = _json(folder / ".codex-plugin" / "plugin.json")
    for manifest in (claude, cursor, codex):
        assert (manifest["name"], manifest["version"]) == (root["name"], root["version"])
    assert "skills" not in claude
    assert cursor["skills"] == [f"./skills/{name}" for name in _repository_skills()]
    assert codex["skills"] == "./skills/"


def test_codex_manifest_can_be_left_out(package):
    without = builder.build(REPO_ROOT, codex_manifest=False)
    assert set(package.files) - set(without.files) == {builder.CODEX_MANIFEST}
    assert all(without.files[path] == package.files[path] for path in without.files)


# ─────────────────────────────── build behaviour ───────────────────────────────
def test_two_builds_are_byte_identical(package, folder, tmp_path):
    again = builder.build(REPO_ROOT)
    assert again.files == package.files
    second = tmp_path / builder.PLUGIN_NAME
    builder.write(again, second)
    first_files, second_files = _files(folder), _files(second)
    assert ([path.relative_to(second) for path in second_files]
            == [path.relative_to(folder) for path in first_files])
    for first, twin in zip(first_files, second_files):
        assert first.read_bytes() == twin.read_bytes()
        assert first.stat().st_mode == twin.stat().st_mode


def test_check_passes_on_a_fresh_build_and_names_every_difference(package, tmp_path):
    target = tmp_path / builder.PLUGIN_NAME
    builder.write(package, target)
    check = [sys.executable, str(BUILDER), "--check", "--output", str(target)]
    assert _run(check, REPO_ROOT).returncode == 0

    skill_md = target / "skills" / _repository_skills()[0] / "SKILL.md"
    skill_md.write_text(_text(skill_md) + "\nedited by hand\n", encoding="utf-8")
    (target / "LICENSE").chmod(0o755)
    (target / "hooks" / "hooks.json").unlink()
    (target / "notes.txt").write_text("stray\n", encoding="utf-8")
    result = _run(check, REPO_ROOT)
    assert result.returncode == 1
    for expected in (
        f"content differs: {skill_md.relative_to(target).as_posix()}",
        "mode differs: LICENSE",
        "missing: hooks/hooks.json",
        "not generated: notes.txt",
    ):
        assert expected in result.stderr


def test_verification_output_is_recognised_at_the_root_and_in_skill_folders():
    added = (
        # at the folder root, in both naming families
        "BENCHMARK.md", "plugin-card.json", "plugin-card.md", "skill-card.md", "plugin.sig.oms",
        "plugin.oms.sig", "skill.oms.sig", "eval-result.json", "plugin-card-review-needed.md",
        ".eval-results/run-1/eval-result.json", ".eval-results/notes.txt",
        # directly inside a skill folder
        "skills/tao-setup/BENCHMARK.md", "skills/tao-setup/skill-card.md",
        "skills/tao-setup/skill.oms.sig", "skills/tao-setup/skill-review-needed.md",
        # a signature, wherever it sits
        "skills/tao-setup/references/skill.oms.sig", "scripts/plugin.sig.oms",
    )
    not_added = (
        "notes.md", "docs/x-review-needed.md", "evals/eval-result.json", "scripts/skill-card.md",
        "skills/BENCHMARK.md", "skills/tao-setup/plugin-card.md",
        "skills/tao-setup/eval-result.json", "skills/tao-setup/references/BENCHMARK.md",
        "skills/tao-setup/references/x-review-needed.md",
    )
    assert all(builder.is_verification_file(path) for path in added)
    assert not any(builder.is_verification_file(path) for path in not_added)
    assert [path for path in added if builder.is_signature(path)] == [
        "plugin.sig.oms", "plugin.oms.sig", "skill.oms.sig", "skills/tao-setup/skill.oms.sig",
        "skills/tao-setup/references/skill.oms.sig", "scripts/plugin.sig.oms",
    ]


def test_check_ignores_and_a_rebuild_keeps_what_verification_adds(package, tmp_path):
    target = tmp_path / builder.PLUGIN_NAME
    builder.write(package, target)
    added = (
        "BENCHMARK.md", "plugin-card.json", "plugin-card.md", "skill-card.md", "eval-result.json",
        "plugin-card-review-needed.md", ".eval-results/run-1/eval-result.json",
        "skills/tao-setup/BENCHMARK.md", "skills/tao-setup/skill-card.md",
        "skills/tao-setup/skill-review-needed.md",
    )
    _add(target, *added)
    assert builder.compare(package, target) == []
    before = _listing(target)
    done = builder.write(package, target)
    assert sorted(done.kept) == sorted(added) and done.changed == 0 and done.removed == []
    assert _listing(target) == before
    # A packaged file that differs is written again; what verification added stays.
    (target / "LICENSE").write_text("edited by hand\n", encoding="utf-8")
    done = builder.write(package, target)
    assert done.changed == 1 and sorted(done.kept) == sorted(added) and done.removed == []
    assert _listing(target) == before


SIGNATURES = (
    "plugin.sig.oms", "plugin.oms.sig", "skill.oms.sig", "skills/tao-setup/skill.oms.sig",
    "scripts/skill.oms.sig",
)


@pytest.mark.parametrize("signature", SIGNATURES)
def test_a_signed_folder_is_rebuilt_only_with_force(package, tmp_path, signature):
    target = tmp_path / builder.PLUGIN_NAME
    builder.write(package, target)
    _add(target, signature, "plugin-card.md")
    signed = _listing(target)
    # Nothing would change: no force is needed and nothing is written.
    done = builder.write(package, target)
    assert done.changed == 0 and done.removed == [] and _listing(target) == signed

    (target / "LICENSE").write_text("edited after signing\n", encoding="utf-8")
    edited = _listing(target)
    refusal = rf"(?s)is signed \({re.escape(signature)}\).*content differs: LICENSE.*--force"
    with pytest.raises(builder.PackageError, match=refusal):
        builder.write(package, target)
    assert _listing(target) == edited

    # Forced: the packaged file is restored and the signature, now stale, goes. It is
    # announced while it is still there.
    announced = []
    done = builder.write(
        package, target, force=True,
        announce=lambda plan: announced.extend(
            (path, reason, (target / path).is_file()) for path, reason in plan.removed),
    )
    assert announced == [(signature, "signature of the earlier content", True)]
    assert done.stale == [signature] and done.kept == ["plugin-card.md"] and done.changed == 1
    assert not (target / signature).exists() and (target / "plugin-card.md").is_file()
    assert (target / "LICENSE").read_bytes() == package.files["LICENSE"][0]
    assert builder.compare(package, target) == []


def test_unknown_files_stop_a_rebuild_and_junk_is_removed(package, tmp_path):
    target = tmp_path / builder.PLUGIN_NAME
    builder.write(package, target)
    junk = (".DS_Store", "__MACOSX/._plugin.json",
            "scripts/__pycache__/skill_layers.cpython-312.pyc", "skills/tao-setup/.DS_Store")
    _add(target, *junk)
    done = builder.write(package, target)
    assert done.junk == sorted(junk) and done.changed == 0 and done.unknown == []
    assert not any((target / path).exists() for path in junk)
    assert not (target / "__MACOSX").exists() and not (target / "scripts/__pycache__").exists()

    unknown = ["notes.txt", "skills/tao-setup/extra.md"]
    _add(target, ".DS_Store", *unknown)
    before = _listing(target)
    refusal = (r"(?s)2 file\(s\) that no build writes.*notes\.txt.*skills/tao-setup/extra\.md"
               r".*--force")
    with pytest.raises(builder.PackageError, match=refusal):
        builder.write(package, target)
    assert _listing(target) == before  # the junk stays too: a refusal touches nothing

    still_there = []
    done = builder.write(
        package, target, force=True,
        announce=lambda plan: still_there.extend(
            path for path, _ in plan.removed if (target / path).is_file()),
    )
    assert still_there == [".DS_Store", *unknown]
    assert done.junk == [".DS_Store"] and done.unknown == unknown and done.changed == 0
    assert builder.compare(package, target) == []


def test_git_folder_is_never_touched_and_a_work_tree_root_is_refused(package, tmp_path):
    target = tmp_path / "mirror"
    builder.write(package, target)
    subprocess.run(["git", "init", "-q", str(target)], check=True, capture_output=True)
    (target / ".git" / "marker").write_text("history\n", encoding="utf-8")
    history = _listing(target / ".git")
    assert history and not [path for path in builder.read_folder(target) if ".git" in path]
    assert builder.compare(package, target) == []

    (target / "LICENSE").write_text("edited\n", encoding="utf-8")
    for force in (False, True):
        with pytest.raises(builder.PackageError, match="root of a git work tree"):
            builder.write(package, target, force=force)
    assert _text(target / "LICENSE") == "edited\n" and _listing(target / ".git") == history
    # Inside the source checkout the same folder is written, and its .git stays as it is.
    done = builder.write(package, target, source=tmp_path)
    assert done.changed == 1 and done.removed == []
    assert _listing(target / ".git") == history and builder.compare(package, target) == []


def test_write_refuses_a_folder_it_did_not_build(package, tmp_path):
    (tmp_path / "keep.txt").write_text("not a package\n", encoding="utf-8")
    with pytest.raises(builder.PackageError, match="refusing to replace"):
        builder.write(package, tmp_path)
    assert [path.name for path in tmp_path.iterdir()] == ["keep.txt"]


def test_left_out_patterns_do_not_cross_folders():
    cases = {
        "skill.oms.sig": "skill.oms.sig",
        "skill-card-review-needed.md": "*-review-needed.md",
        "config/scan-baseline.yaml": "config/*-baseline.yaml",
        "tests/test_a.py": "**/tests/**",
        "scripts/tests/fixtures/a/b.jpg": "**/tests/**",
        "scripts/test_a.py": "**/test_*.py",
        "test_a.py": "**/test_*.py",
        "eval.config": "eval*.config",
        "eval.slow-manual.config": "eval*.config",
    }
    for path, pattern in cases.items():
        assert builder.left_out("tao-any", path) == pattern, path
    for path in (
        "config/models/dino-baseline.yaml",   # "*" stays within one name
        "references/skill.oms.sig", "references/eval.config", "evals/evals.json",
        "references/x-review-needed.md", "review-needed.md",
        "scripts/contest.py", "scripts/latest_tests.py", "references/testing.md",
        "scripts/smoke_test.sh",
    ):
        assert builder.left_out("tao-any", path) is None, path
    for (skill, inside) in builder.SKILL_FILES_KEPT:
        assert builder.left_out(skill, inside) is None
    for (skill, inside) in builder.ONE_SKILL_FILES_LEFT_OUT:
        assert builder.left_out(skill, inside) and builder.left_out("tao-other", inside) is None


def test_rewriter_flattens_every_form_of_a_layer_reference(package):
    rewriter = builder.LayoutRewriter(package.membership, {"schemas.manifest.json": "x"})
    unrelated = "my-skills/models/x and ../../tao-setup/SKILL.md"
    clone = "tao-skill-bank/" + FLAT_BANK
    for before, after in (
        ("`skills/platform/tao-run-on-docker/SKILL.md`", "`skills/tao-run-on-docker/SKILL.md`"),
        ("$BANK/skills/models/<network>/schemas/", "$BANK/skills/<network>/schemas/"),
        ('"$BANK"/skills/applications/*/cards', '"$BANK"/skills/*/cards'),
        ("skills/models/schemas.manifest.json", "catalog/schemas.manifest.json"),
        ("](../../../platform/tao-run-on-docker/SKILL.md)", "](../../tao-run-on-docker/SKILL.md)"),
        ('bank / "skills" / "models" / name', 'bank / "skills" / name'),
        ('os.path.join(bank, "skills", "models", name)', 'os.path.join(bank, "skills", name)'),
        # A clone of the repository holds the flat bank in the package folder.
        ("${TAO_SKILL_BANK_PATH:-~/tao-skill-bank}/scripts/x.py",
         "${TAO_SKILL_BANK_PATH:-~/" + clone + "}/scripts/x.py"),
        ('"${TAO_SKILL_BANK_PATH:-$HOME/tao-skill-bank}"',
         '"${TAO_SKILL_BANK_PATH:-$HOME/' + clone + '}"'),
        ('Path.home() / "tao-sdk/tao-skill-bank/skills/models/tao-train-dino"',
         'Path.home() / "tao-sdk/' + clone + '/skills/tao-train-dino"'),
        ("under `skills/models/`.", "under `skills/`."),
        (unrelated, unrelated),
        # A skill is addressed by the plugin it ships in.
        ("invoke `tao-skill-bank:tao-run-automl`", "invoke `nvidia-tao-toolkit:tao-run-automl`"),
        ("through tao-skill-bank:tao-run-automl.", "through nvidia-tao-toolkit:tao-run-automl."),
        ("[`tao-skill-bank:tao-setup-nvidia-gpu-host`](../x)",
         "[`nvidia-tao-toolkit:tao-setup-nvidia-gpu-host`](../x)"),
        # and all skills of the plugin together
        ("the underlying `tao-skill-bank:*` skills",
         "the underlying `nvidia-tao-toolkit:*` skills"),
        ("invoke the matching tao-skill-bank:* skill.",
         "invoke the matching nvidia-tao-toolkit:* skill."),
        # A placeholder for the bank stands for the plugin folder, not for a clone.
        ("export TAO_SKILL_BANK_PATH=/path/to/tao-skill-bank",
         "export TAO_SKILL_BANK_PATH=/path/to/nvidia-tao-toolkit"),
        ('skill_bank = Path("/path/to/tao-skill-bank")',
         'skill_bank = Path("/path/to/nvidia-tao-toolkit")'),
        ("/abs/path/to/tao-skill-bank/skills/models/tao-train-dino",
         "/abs/path/to/" + clone + "/skills/tao-train-dino"),
    ):
        assert rewriter.refused(before) == [], before
        assert rewriter.rewrite(before) == after
        assert rewriter.leftovers(after) == []
    # Only an exact skill name after the old plugin name is a skill reference.
    for kept in (
        "tao-skill-bank:tao-setup-extra",                 # a longer word
        "tao-skill-bank:tao-setupx", "tao-skill-bank:tao-train", "tao-skill-bank: tao-setup",
        "tao-skill-bank:latest", "registry.example.com/x/tao-skill-bank:tao-setup",   # image tags
        "https://github.com/NVIDIA-TAO/tao-skill-bank:tao-setup",             # inside a URL
        "git@github.com:NVIDIA-TAO/tao-skill-bank.git",
        "tao-skill-bank:tao-setup/SKILL.md",                                  # a path in a ref
        "nvidia-tao-toolkit@tao-skill-bank", "tao-skills@tao-skill-bank:tao-setup",
        "my-tao-skill-bank:tao-setup", "x.tao-skill-bank:tao-setup",
        "registry.example.com/x/tao-skill-bank:*", "tao-skills@tao-skill-bank:*",
        "tao-skill-bank:**", "tao-skill-bank:*/SKILL.md", "tao-skill-bank:*-rc",
        # a placeholder that names something else, or a path that goes through a clone
        "/path/to/tao-skill-bank-fork", "/path/to/tao-skill-bank.git",
        "/path/to/tao-skill-bank/" + FLAT_BANK, "~/path/to/tao-skill-bank",
    ):
        assert rewriter.rewrite(kept) == kept and rewriter.old_names(kept) == 0, kept
    assert rewriter.old_names("tao-skill-bank:tao-setup, tao-skill-bank:tao-run-automl") == 2
    for survivor in (
        "ls skills/models skills/platform",
        'root / "skills" / "models"',
        '"core/tao-artifacts/references/x.json"',
        '/ "models"\n    / "tao-train-dino"',
        "every workflow under `applications/`",
        "for skill in skills/*/*/SKILL.md",
        "see [the layer](../../models/README.md)",
        # a drawing of the repository tree, and the repository as the place to run from
        "└── skills/\n    ├── models/               # per-network skill packages\n",
        "    └── platform/",
        "|-- data/",
        "Run from the `tao-skill-bank` repo root.",
        "from the repository root of tao-skill-bank",
    ):
        assert rewriter.leftovers(rewriter.rewrite(survivor)), survivor
    for fine in ("├── models.yaml", "├── data/train/", "the repo root of the paper's code",
                 "| models/ | weights |"):
        assert rewriter.leftovers(fine) == [], fine


def test_rewriter_refuses_what_is_not_a_path_in_a_bank(package):
    rewriter = builder.LayoutRewriter(package.membership, {})
    refused = {
        "https://github.com/NVIDIA-TAO/tao-skill-bank/blob/main/skills/models/tao-train-dino":
            "inside a URL",
        "[raw](https://raw.githubusercontent.com/o/r/main/skills/platform/x.md)": "inside a URL",
        "s3://bucket/skills/models/ckpt.pt": "inside a URL",
        "cat ~/.claude/skills/data/notes.md": "under another folder",
        "/datasets/skills/data/train.json": "under another folder",
        "see docs/skills/models/overview.md": "under another folder",
        "The layers are skills/applications/, skills/core/, skills/data/, skills/models/ and "
        "skills/platform/.": "a list of layer folders",
        "- `skills/models/` holds models\n- `skills/data/` holds data": "a list of layer folders",
        'KEYS = ["skills", "models", "hooks"]': "a word list",
        'names = ("skills", "data", "other")': "a word list",
        # a path at the root of a filesystem: a dataset, the target of a volume mount
        "ls /skills/data/train": "at the root of a filesystem",
        "-v $PWD/ds:/skills/data/train": "at the root of a filesystem",
        'os.listdir(bank + "/skills/models/x")': "at the root of a filesystem",
        # a path inside a git ref, or behind any other colon
        "git show origin/main:skills/models/tao-train-dino/SKILL.md": "after a colon",
        "docker run -v cache:skills/data/x image": "after a colon",
        # a variable that may hold a URL or a ref
        'curl -fsSL "$RAW_BASE/skills/models/tao-train-dino/SKILL.md"': "fetches from a URL",
        "wget ${MIRROR}/skills/models/x.tgz": "fetches from a URL",
        "see https://example.com/{branch}/skills/models/x": "inside a URL",
        "requests.get(f\"http://{host}\" + f\"{base}/skills/models/{name}\")": "fetches from a URL",
        'git show "$REF"/skills/models/x.md': "fetches from a URL",
        # relative to a clone, which keeps its layer folders
        "cd tao-skill-bank\npython3 skills/models/tao-train-dino/scripts/run.py":
            "after cd tao-skill-bank",
        "git clone <url> && cd tao-skill-bank && ls ./skills/models/tao-train-dino":
            "after cd tao-skill-bank",
        'cd "$HOME/tao-skill-bank"; cat skills/core/tao-setup/SKILL.md': "after cd tao-skill-bank",
    }
    for text, reason in refused.items():
        found = rewriter.refused(text)
        assert found and all(reason in item for item in found), (text, found)
    # In code, a layer folder that is listed is a different list in the package.
    for path, text in (
        ("scripts/install.sh", 'for pack in "$BANK"/skills/applications/*/cards; do'),
        ("scripts/list.py", 'sorted(Path(bank, "skills/models/").iterdir())'),
        ("scripts/list.py", "glob.glob(f'{bank}/skills/models/*')"),
    ):
        assert any("lists one layer folder" in item for item in rewriter.refused(text, path)), text
        assert rewriter.refused(text, "references/notes.md") == [], text
    # Not refused: the same words where they are a path in a bank, or not a layer path at all.
    for text in (
        "see `skills/models/tao-train-dino/SKILL.md`",
        "<bank-root>/skills/models/tao-train-dino",
        "https://example.com/docs and skills/models/tao-train-dino/SKILL.md",
        "agent_skills/data/x and my-skills/models/x",
        'root.joinpath("skills", "models", name)',
        # a bank variable stays one next to a download, any variable does where nothing is fetched
        'curl -fsSL https://example.com/x.tgz | tar -xz -C "$TAO_SKILL_BANK_PATH/skills/models/x"',
        'python3 "$WORKDIR/skills/models/tao-train-dino/scripts/run.py"',
        "python3 {root}/skills/models/tao-train-dino/scripts/run.py",
        "see https://example.com first\nthen read $CHECKOUT/skills/core/tao-setup/SKILL.md",
        # a clone that is entered and left again, and the package folder inside a clone
        "```bash\ncd tao-skill-bank\n```\n\nThen read `skills/core/tao-setup/SKILL.md`.",
        "cd tao-skill-bank/plugins/nvidia-tao-toolkit && ls skills/models/tao-train-dino",
        "cd tao-skill-bank && python3 $BANK/skills/models/tao-train-dino/scripts/run.py",
    ):
        assert rewriter.refused(text) == [], text
        assert rewriter.refused(text, "scripts/run.sh") == [], text


def test_code_that_climbs_out_of_its_skill_is_noticed():
    climbing = (
        ("scripts/a.py", "ROOT = pathlib.Path(__file__).resolve().parents[4]"),
        ("scripts/a.py", "skills = Path(__file__).resolve().parent.parent.parent.parent"),
        ("tests/test_a.py", "REPO_ROOT = SKILL_ROOT.parents[2]"),
        ("scripts/run.sh", 'BANK=$(cd "$script_dir/../../.." && pwd)'),
    )
    staying = (
        ("scripts/a.py", "SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]"),
        ("scripts/a.py", "assets = Path(__file__).resolve().parent.parent / 'assets'"),
        ("scripts/a.py", "sibling = SKILL_ROOT.parent / name"),
        ("hooks/check.sh", '"$report_dir/../../inference/inference.csv"'),
    )
    for path, line in climbing:
        assert builder.climbs_out(path, line) == {line}, line
    for path, line in staying:
        assert builder.climbs_out(path, line) == set(), line


def test_a_shipped_file_that_names_a_left_out_file_is_noticed():
    gone = set()
    for skill, inside in (
        ("tao-a", "tests/test_a.py"), ("tao-a", "tests/fixtures/x/one.json"),
        ("tao-a", "references/tests/test_b.py"), ("tao-a", "config/scan-baseline.yaml"),
        ("tao-b", "eval.config"),
    ):
        pattern = builder.left_out(skill, inside)
        folders = builder.left_out_folders(inside, pattern)
        gone.update(f"skills/{skill}/{name}" for name in [inside, *folders])
    assert "skills/tao-a/tests/fixtures" in gone and "skills/tao-a/config" not in gone
    shipped = builder.with_folders({
        "skills/tao-a/SKILL.md", "skills/tao-a/references/b.py", "skills/tao-a/scripts/run.py",
        "skills/tao-b/SKILL.md", "scripts/x.py",
    })

    def named(path, text):
        return builder.left_out_mentions(path, text, gone, shipped)

    skill_md, tests = "skills/tao-a/SKILL.md", "skills/tao-a/tests"
    assert named(skill_md, "run `python -m unittest tests.test_a`") == [tests + "/test_a.py"]
    assert named(skill_md, "a fixture is under `tests/fixtures/x/`") == [tests + "/fixtures/x"]
    assert named(skill_md, "see test_a.py") == [tests + "/test_a.py"]
    # pytest and unittest name the folder without a slash
    assert named(skill_md, "run `python -m pytest tests -q`") == [tests]
    assert named(skill_md, "pytest -q -x tests") == [tests]
    assert named(skill_md, "python -m unittest discover -s tests -p 'test_*.py'") == [tests]
    assert named(skill_md, "python -m pytest references -q") == []
    assert named(skill_md, "unit tests are in `references/tests/`.") == [
        "skills/tao-a/references/tests"]
    assert named("skills/tao-a/references/b.py", "tested by ``tests/test_b.py``") == [
        "skills/tao-a/references/tests/test_b.py"]
    assert named("AGENTS.md", "read skills/tao-b/eval.config") == ["skills/tao-b/eval.config"]
    assert named("skills/tao-b/SKILL.md", "$BANK/skills/tao-a/tests") == ["skills/tao-a/tests"]
    # Not a left-out file of this package: another project's tests, the bank's own scripts/.
    # Nor the folder of a single left-out file: config/ here is a folder of a run.
    for text in ("`tao-pytorch/tests/test_lora.py`", "the shared `scripts/` helpers",
                 "the tests pass", "writes `test/gt_new.txt`", "one.json",
                 "specs are copied to `config/`"):
        assert named(skill_md, text) == [], text


# ─────────────────────────────── the package works ───────────────────────────────
def test_layer_lookup_answers_the_same_in_both_layouts(folder):
    for layer in skill_layers.LAYERS:
        nested = skill_layers.layer_skill_dirs(REPO_ROOT, layer)
        flat = skill_layers.layer_skill_dirs(folder, layer)
        assert [path.name for path in flat] == [path.name for path in nested]
        assert all(path.parent == folder / "skills" for path in flat)
    model = skill_layers.layer_skill_dirs(folder, "models")[0].name
    manifest = "schemas.manifest.json"
    assert skill_layers.skill_dir(folder, "models", model) == folder / "skills" / model
    assert not skill_layers.skill_dir(folder, "applications", model).exists()
    assert skill_layers.layer_file(folder, "models", manifest) == folder / "catalog" / manifest
    assert skill_layers.layer_file(REPO_ROOT, "models", manifest).is_file()
    assert skill_layers.layer_root(folder, "models") == folder / "skills"
    assert skill_layers.layer_root(REPO_ROOT, "models") == REPO_ROOT / "skills" / "models"


def test_packaged_helper_scripts_start_cleanly_and_leave_no_bytecode(folder, tmp_path):
    """The checks of test_helper_scripts_start.py on the scripts folder of the package."""
    import test_helper_scripts_start as helper_start  # the test module beside this one

    scripts = tmp_path / "plugin" / "scripts"
    shutil.copytree(folder / "scripts", scripts)
    assert sorted(path.name for path in scripts.iterdir()) == helper_start.HELPERS
    helper_start.check_helpers_start(scripts, tmp_path)
    helper_start.check_no_bytecode_is_written(scripts, tmp_path)


def test_helper_scripts_answer_from_the_package_as_from_the_repository(folder):
    def names(root, script, *keys):
        result = _run([sys.executable, f"scripts/{script}", "--format", "json"], root)
        assert result.returncode == 0, result.stderr
        value = json.loads(result.stdout)
        for key in keys:
            value = value[key]
        return [item.get("name") or item.get("model") for item in value]

    for script, keys in (
        ("list_tao_models.py", ("models",)),
        ("list_tao_capabilities.py", ("applications",)),
        ("list_tao_capabilities.py", ("data_workflows",)),
        ("list_tao_capabilities.py", ("platforms", "supported")),
    ):
        from_package = names(folder, script, *keys)
        assert from_package and from_package == names(REPO_ROOT, script, *keys), (script, keys)

    model = skill_layers.layer_skill_dirs(folder, "models")[0].name
    resolve = [sys.executable, "scripts/resolve_tao_model.py", "--model", model, "--format", "json"]
    resolved = _run(resolve, folder)
    assert resolved.returncode == 0, resolved.stderr
    skill_md = (folder / "skills" / model / "SKILL.md").resolve()
    assert json.loads(resolved.stdout)["skill_path"] == str(skill_md)
    # A model is not an application in the flat layout either.
    image = [sys.executable, "scripts/resolve_tao_image.py", "--application", model,
             "--action", "train", "--format", "json"]
    assert _run(image, folder).returncode != 0


def test_session_start_hook_runs_from_the_package(folder):
    """The hook of the package prints the same JSON form and names only what the package has."""
    hook = folder / "hooks" / "session_start.sh"
    for part in ("identity", "context"):
        output = hook_checks.run_hook(part, hook=hook)
        text = hook_checks.context_of(output)
        assert 0 < len(text) < len(output) < HOOK_OUTPUT_CAP, part
    line = hook_checks.folder_line(text)
    assert line.startswith(f"TAO plugin folder: `{folder}`. It is not exported here")
    assert f"`{folder}/skills`" in line and f"`{folder}/scripts`" in line
    # The Docker hint names the check script where the flat layout has it.
    script = folder / "skills" / hook_checks.SETUP_SCRIPT
    assert script.is_file()
    assert hook_checks.check_command(text) == [
        "bash", str(script), "--backend", "docker", "--check-only"]
    # A README is named only when the package ships one.
    assert ("README.md" in text) == (folder / "README.md").is_file()
    # Card packs are found one level below skills/ as they are two levels below it.
    card_folders = (REPO_ROOT / "skills").glob("*/*/cards")
    packs = sorted(path.parent.name for path in card_folders if path.is_dir())
    announcement = f"card packs exist for: {' '.join(packs)}."
    with_pi = hook_checks.hook_path("pi")
    assert packs and announcement in hook_checks.context_of(
        hook_checks.run_hook("context", hook=hook, path=with_pi))
    assert announcement in hook_checks.context_of(hook_checks.run_hook("context", path=with_pi))
    assert "ard pack" not in text


def test_identity_text_is_true_of_the_repository_and_of_the_package(folder):
    """AGENTS.md is printed into sessions that work in other projects: what it names must exist."""
    named = re.compile(r"`((?:skills|scripts)/[\w./-]+\.(?:py|sh|md|yaml|json))`")
    for root in (REPO_ROOT, folder):
        text = _text(root / "AGENTS.md")
        paths = named.findall(text)
        assert any(path.endswith("/cosmos_workflow.py") for path in paths)
        for path in paths:
            assert (root / path).is_file(), f"{root.name}/AGENTS.md names {path}"
        # The bank is the plugin folder, not the project a session works in.
        assert "this repo" not in text and "the installed plugin folder" in text
        # versions.yaml pins nvidia-tao-sdk for AutoML, so the text must not deny the package.
        assert "nvidia-tao-sdk==" in _text(root / "versions.yaml")
        assert not re.search(r"(?i)there is no\s+`?nvidia-tao-sdk", text)
        assert text.count("nothing needs `nvidia-tao-sdk`") == 2


def test_skill_code_finds_the_package_root_and_its_neighbours(folder):
    root = folder.resolve()
    for relative in (
        "skills/tao-run-on-kubernetes/scripts/render_action_job.py",
        "skills/paidf-cosmos-predict/scripts/paidf_common.py",
        "skills/tao-run-deft-pas/scripts/run_deft_docker_action.py",
    ):
        script = folder / relative
        levels = re.search(r"REPO_ROOT = \S+\.resolve\(\)\.parents\[(\d)\]", _text(script))
        assert script.resolve().parents[int(levels.group(1))] == root, relative
    for relative in (
        "skills/tao-run-automl/cards/driver.sh",
        "skills/tao-run-deft-aoi/cards/driver.sh",
    ):
        climb = re.search(r'BANK=\$\(cd "\$PACK/([./]+)" && pwd\)', _text(folder / relative))
        assert ((folder / relative).parent / climb.group(1)).resolve() == root, relative
    adapter = folder / "skills/tao-run-automl/references/best_rec_adapter.py"
    schema = re.search(r'parents\[(\d)\]\s*/ "([^"]+)"', _text(adapter))
    assert (adapter.resolve().parents[int(schema.group(1))] / schema.group(2)).is_file()


# ── the bank check a reader is told to run, followed exactly as the package prints it ──
BANK_CHECKS = (
    "skills/tao-setup/SKILL.md",
    "skills/tao-launch-workflow/references/platform-preflight.md",
    "skills/tao-finetune-clip/references/standalone-setup.md",
)
PLACEHOLDER = "<this-skill-folder>"
HINT = "plugin folder that contains this skill"
VARIABLE_UNSET = "# TAO_SKILL_BANK_PATH not set? Set it to the " + HINT


def _bank_check(document):
    """The commands of the bank check: its fenced block up to a blank line or the closing fence."""
    start = r"(?:# TAO_SKILL_BANK_PATH not set\?|export TAO_SKILL_BANK_PATH=)"
    block = re.search(rf"```bash\n({start}.*?)\n(?:\n|```)", _text(document), re.S)
    assert block, document
    return block.group(1)


def _follow(installed, document, home):
    """Do what the packaged text says, with no variable exported: (exit code, bank, stderr).

    ``installed`` is the folder that holds skills/<skill>/ as the reader has it.
    The reader knows where the skill's folder is, puts it into the one
    placeholder of the block and runs the block as printed.
    """
    skill_folder = installed / "/".join(document.split("/")[:2])
    commands = _bank_check(installed / document)
    assert commands.count(PLACEHOLDER) == 1
    commands = commands.replace(PLACEHOLDER, str(skill_folder))
    result = _run(["bash", "-c", commands + '\nprintf %s "$TAO_SKILL_BANK_PATH"'], home, HOME=home)
    return result.returncode, result.stdout, result.stderr


def _clone_without_package(repository, clone):
    """Copy what `git clone` of ``repository`` gives to ``clone``, minus the package folder.

    Once the package folder is committed, git tracks it like every other file,
    and the committed copy may be signed and older than the source beside it. A
    test that needs a clone with the folder writes the build it tests into the
    clone; a test that needs a clone without it gets exactly that.
    """
    subprocess.run(
        ["git", "checkout-index", "--all", f"--prefix={clone}/"], cwd=repository, check=True
    )
    shutil.rmtree(clone / FLAT_BANK, ignore_errors=True)
    return clone


@pytest.fixture(scope="module")
def checkout(package, tmp_path_factory):
    """What `git clone` of the repository gives once this build of the package is committed."""
    bank = _clone_without_package(REPO_ROOT, tmp_path_factory.mktemp("clone") / "tao-skill-bank")
    builder.write(package, bank / FLAT_BANK)
    return bank


@pytest.fixture(scope="module")
def bare_checkout(folder, tmp_path_factory):
    """A home folder with a clone that lacks the package folder, and skills installed one by one."""
    home = tmp_path_factory.mktemp("bare")
    _clone_without_package(REPO_ROOT, home / "tao-skill-bank")
    for name in sorted({document.split("/")[1] for document in BANK_CHECKS}):
        shutil.copytree(folder / "skills" / name, home / ".agents" / "skills" / name)
    return home


PLUGIN_CACHES = (".claude/plugins/cache", ".codex/plugins/cache")


@pytest.fixture(scope="module")
def installed_home(package, tmp_path_factory):
    """A home folder with the plugin installed as Claude Code and as Codex lay it out.

    Both copy a plugin to <cache>/<marketplace>/<plugin>/<version>/ below their own folder.
    """
    home = tmp_path_factory.mktemp("installed")
    for cache in PLUGIN_CACHES:
        builder.write(package, _installed(home, cache, package))
    return home


def _installed(home, cache, package):
    return home / cache / "catalog" / builder.PLUGIN_NAME / package.manifest["version"]


@pytest.mark.parametrize("document", BANK_CHECKS)
@pytest.mark.parametrize("cache", PLUGIN_CACHES)
def test_bank_check_finds_the_installed_plugin_with_no_variable_exported(
        document, cache, package, installed_home):
    home = installed_home
    installed = _installed(home, cache, package)
    code, bank, error = _follow(installed, document, home)
    assert (code, bank) == (0, str(installed.resolve())), error
    # Read but not filled in, the block falls back to the default, which is not there.
    unfilled = _run(["bash", "-c", _bank_check(installed / document)], home, HOME=home)
    assert unfilled.returncode == 1 and HINT in unfilled.stderr


@pytest.mark.parametrize("document", BANK_CHECKS)
def test_bank_check_follows_a_skill_folder_that_is_a_link_into_the_plugin(
        document, package, installed_home, tmp_path):
    """A skill folder may be a link into the plugin folder; the bank is then behind the link."""
    installed = _installed(installed_home, PLUGIN_CACHES[0], package)
    name = document.split("/")[1]
    home = tmp_path / "home"
    (home / ".agents" / "skills").mkdir(parents=True)
    (home / ".agents" / "skills" / name).symlink_to(
        installed / "skills" / name, target_is_directory=True)
    code, bank, error = _follow(home / ".agents", document, home)
    # Not the folder two levels above the link, which holds nothing but the link.
    assert (code, bank) == (0, str(installed.resolve())), error


@pytest.mark.parametrize("document", BANK_CHECKS)
def test_bank_check_finds_the_package_folder_of_a_checkout(document, folder, checkout, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    inside = checkout / FLAT_BANK
    # read from the checkout, wherever it was cloned to
    code, bank, error = _follow(inside, document, home)
    assert (code, bank) == (0, str(inside.resolve())), error
    # a variable that a session hook exported wins, and the block needs no editing then
    commands = ["bash", "-c", _bank_check(folder / document) + '\nprintf %s "$TAO_SKILL_BANK_PATH"']
    hooked = _run(commands, home, TAO_SKILL_BANK_PATH=folder, HOME=home)
    assert (hooked.returncode, hooked.stdout) == (0, str(folder)), hooked.stderr
    # the printed default is the package folder of a clone in the home folder
    (home / "tao-skill-bank").symlink_to(checkout, target_is_directory=True)
    default = _run(commands, home, HOME=home)
    assert default.returncode == 0, default.stderr
    assert default.stdout == str(home / "tao-skill-bank" / FLAT_BANK)


@pytest.mark.parametrize("document", BANK_CHECKS)
def test_bank_check_fails_clearly_on_a_checkout_without_the_package_folder(
        document, bare_checkout):
    home = bare_checkout
    clone = home / "tao-skill-bank"
    assert (clone / "versions.yaml").is_file() and not (clone / FLAT_BANK).exists()
    code, bank, error = _follow(home / ".agents", document, home)
    # No bank is chosen: not the clone, not the folder above the installed skills.
    assert (code, bank) == (1, "")
    assert HINT in error and f"{FLAT_BANK} is not a bank" in error
    assert str(clone / FLAT_BANK) in error
    # Pointed at the root of the clone on purpose, the check still says no.
    at_root = _run(["bash", "-c", _bank_check(home / ".agents" / document)], home,
                   TAO_SKILL_BANK_PATH=clone, HOME=home)
    assert at_root.returncode == 1 and "Incomplete TAO skill bank" in at_root.stderr, document
    # In the repository the same check passes on a clone: the two layouts differ on purpose.
    source = next((REPO_ROOT / "skills").glob("*/" + document.split("/", 1)[1]))
    assert _run(["bash", "-c", _bank_check(source)], home,
                TAO_SKILL_BANK_PATH=clone, HOME=home).returncode == 0, document


def test_bank_check_text_says_where_the_bank_is_and_promises_no_clone(folder):
    for document in BANK_CHECKS:
        text = _text(folder / document)
        block = _bank_check(folder / document)
        # the instruction before the check, the first line of the block, the error text
        before = text[:text.index(block)]
        assert re.search(r"When\s+(?:`TAO_SKILL_BANK_PATH`|the\s+variable)\s+is\s+not\s+already"
                         r"\s+set", before), document
        assert "two levels" in before and "`catalog/`" in before and PLACEHOLDER in before
        assert block.startswith(VARIABLE_UNSET), document
        assert re.search(rf"(?:echo|printf) [\"'][^\n]*{HINT}", block), document
        # a checkout is a bank only when it has the folder; no ref is promised to have it
        assert re.search(rf"only\s+when\s+it\s+contains\s+`{FLAT_BANK}`", text), document
        for promise in ("git clone", "release ref", "<installed-release-ref>", "<bank-dir>"):
            assert promise not in text, (document, promise)
    assert "install the complete\n`nvidia-tao-toolkit` plugin" in _text(folder / BANK_CHECKS[0])


def test_no_shipped_file_names_a_bank_in_the_home_folder_that_is_not_the_package(folder):
    offenders = {}
    seen = set()
    for relative, text in _shipped_texts(folder):
        found = {match.group(0) for match in HOME_BANK.finditer(text)}
        seen |= {(relative, words) for words in found}
        found -= {words for path, words in HOME_BANK_ALLOWED if path == relative}
        if found:
            offenders[relative] = sorted(found)
    assert not offenders
    assert set(HOME_BANK_ALLOWED) <= seen  # an entry that matches nothing hides the next case
    # what the pattern is for
    for wrong in ("${TAO_SKILL_BANK_PATH:-~/tao-skill-bank}", "cd $HOME/tao-skill-bank",
                  "`${HOME}/tao-skill-bank/skills`", 'Path.home() / "tao-skill-bank"',
                  "~/tao-skill-bank/plugins/other"):
        assert HOME_BANK.search(wrong), wrong
    for fine in (f"~/tao-skill-bank/{FLAT_BANK}/scripts", "~/tao-skill-bank-fork",
                 "~/.claude/projects/-home-user-tao-skill-bank"):
        assert not HOME_BANK.search(fine), fine


def test_no_shipped_placeholder_for_the_bank_names_a_clone(folder):
    offenders = {}
    seen = set()
    for relative, text in _shipped_texts(folder):
        found = {match.group(0) for match in CLONE_PLACEHOLDER.finditer(text)}
        seen |= {(relative, words) for words in found}
        found -= {words for path, words in CLONE_PLACEHOLDER_ALLOWED if path == relative}
        if found:
            offenders[relative] = sorted(found)
    assert not offenders
    assert set(CLONE_PLACEHOLDER_ALLOWED) <= seen
    for wrong in ("export TAO_SKILL_BANK_PATH=/path/to/tao-skill-bank",
                  'Path("/absolute/path/to/tao-skill-bank")',
                  "/path/to/tao-skill-bank/scripts/x.py",
                  "TAO_SKILL_BANK_ROOT=<tao-skill-bank>", 'Path("<absolute-tao-skill-bank>")'):
        assert CLONE_PLACEHOLDER.search(wrong), wrong
    for fine in (f"/path/to/tao-skill-bank/{FLAT_BANK}/skills", "/path/to/tao-skill-bank-fork",
                 "/path/to/nvidia-tao-toolkit", "<bank-root>", "<tao-skill-bank-url>"):
        assert not CLONE_PLACEHOLDER.search(fine), fine


def test_capability_listing_says_where_the_bank_is_before_its_first_command(folder):
    """No hook exports the variable in Codex, and the commands of this skill start with it."""
    text = _text(folder / "skills/tao-list-capabilities/SKILL.md")
    before = text[:text.index("```bash")]
    assert re.search(r"When\s+it\s+is\s+not\s+set.*set\s+it\s+first\s+to\s+the\s+plugin\s+folder"
                     r".*two\s+levels\s+above", before, re.S)
    assert "${TAO_SKILL_BANK_PATH:-" in text[len(before):]


IDENTITY_SECTION = "## Optional: agent identity for Codex and Cursor"
BEGIN_MARKER = "<!-- BEGIN nvidia-tao-toolkit agent identity"
END_MARKER = "<!-- END nvidia-tao-toolkit agent identity -->"


def _identity_commands(setup):
    """The commands of the identity section of a tao-setup SKILL.md."""
    section = _text(setup).split(IDENTITY_SECTION, 1)[1]
    return re.search(r"```bash\n(.*?)```", section, re.S).group(1)


def test_repository_setup_skill_adds_the_identity_with_the_helper_of_the_bank(tmp_path):
    """The section in the repository: one command, the bank from the variable Quick Start sets."""
    commands = _identity_commands(REPO_ROOT / "skills/core/tao-setup/SKILL.md")
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    unset = _run(["bash", "-c", commands], project, HOME=home)
    assert unset.returncode != 0 and "TAO_SKILL_BANK_PATH" in unset.stderr
    assert list(project.iterdir()) == []
    added = _run(["bash", "-c", commands], project, HOME=home, TAO_SKILL_BANK_PATH=REPO_ROOT)
    assert added.returncode == 0, added.stderr
    assert _text(project / "AGENTS.md").startswith(BEGIN_MARKER)
    assert list(home.iterdir()) == []


def test_agent_identity_is_added_to_a_project_as_the_packaged_setup_skill_says(
        folder, package, installed_home, tmp_path):
    """No copy over a file in the Codex home: a block in the AGENTS.md of the project."""
    setup = folder / "skills/tao-setup/SKILL.md"
    before = _text(setup).split(IDENTITY_SECTION, 1)[1].split("```bash", 1)[0]
    assert re.search(r"With\s+user\s+approval", before) and "marked block" in before
    assert re.search(r"When\s+`TAO_SKILL_BANK_PATH`\s+is\s+not\s+set", before)
    assert PLACEHOLDER in before
    commands = _identity_commands(setup)
    assert commands.count(PLACEHOLDER) == 1
    identity = _text(folder / "AGENTS.md")
    own = "# My rules\n\n- keep answers short\n"
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    (project / "AGENTS.md").write_text(own, encoding="utf-8")
    block = f"{identity.rstrip()}\n{END_MARKER}\n"

    # Read but not filled in, with no variable exported: the block stops, nothing is written.
    unfilled = _run(["bash", "-c", commands], project, HOME=home)
    assert unfilled.returncode == 1
    assert "Set TAO_SKILL_BANK_PATH to the plugin folder first" in unfilled.stderr
    assert _text(project / "AGENTS.md") == own

    # Codex: no hook exported the variable; the reader fills in the folder of the skill.
    for cache in PLUGIN_CACHES:
        installed = _installed(installed_home, cache, package)
        filled = commands.replace(PLACEHOLDER, str(installed / "skills" / "tao-setup"))
        (project / "AGENTS.md").write_text(own, encoding="utf-8")
        added = _run(["bash", "-c", filled], project, HOME=home)
        assert added.returncode == 0, added.stderr
        written = _text(project / "AGENTS.md")
        assert written.startswith(own + "\n" + BEGIN_MARKER) and written.endswith(block), cache
    assert sorted(path.name for path in project.iterdir()) == ["AGENTS.md"]
    assert list(home.iterdir()) == []            # no Codex home, no backup, no copy

    # A skill folder that is a link into the plugin folder leads to the same helper.
    (home / "skills").mkdir()
    (home / "skills" / "tao-setup").symlink_to(
        installed / "skills" / "tao-setup", target_is_directory=True)
    linked = commands.replace(PLACEHOLDER, str(home / "skills" / "tao-setup"))
    again = _run(["bash", "-c", linked], project, HOME=home)
    assert again.returncode == 0 and "Nothing to do" in again.stdout, again.stderr

    # A variable that a session hook exported wins, and the block needs no editing then.
    hooked = _run(["bash", "-c", commands], project, HOME=home, TAO_SKILL_BANK_PATH=folder)
    assert hooked.returncode == 0 and "Nothing to do" in hooked.stdout, hooked.stderr
    assert _text(project / "AGENTS.md") == written

    # "remove" in place of "add", as the text after the block says, gives the file back.
    after = _text(setup).split(IDENTITY_SECTION, 1)[1].split("```", 2)[2]
    assert "`remove`" in after and "`status`" in after and "Do not copy the identity" in after
    removal = commands.replace('"$helper" add', '"$helper" remove')
    assert removal != commands
    removed = _run(["bash", "-c", removal], project, HOME=home, TAO_SKILL_BANK_PATH=folder)
    assert removed.returncode == 0, removed.stderr
    assert _text(project / "AGENTS.md") == own


# ─────────────────────────── failure paths, on a small checkout ───────────────────────────
ALPHA = """---
name: tao-alpha
---
# Alpha

Read [beta](../../models/tao-beta/SKILL.md) and `skills/models/tao-beta/references/notes.md`.
"""
BETA = "---\nname: tao-beta\n---\n# Beta\n"


class SmallCheckout:
    """A git checkout with two tiny skills that the generator can package."""

    def __init__(self, root, capsys):
        self.root, self.capsys = root / "source", capsys
        self.output = root / "out" / builder.PLUGIN_NAME
        self.manifest()
        self.write("AGENTS.md", "# Agents\n")
        self.write("docs/guide.md", "# Guide\n")
        self.write("scripts/skill_layers.py", "LAYERS = ()\n")
        self.write("scripts/repository_only.py", "print('not shipped')\n")
        self.write("skills/core/tao-alpha/SKILL.md", ALPHA)
        self.write("skills/core/tao-alpha/scripts/run.sh", "#!/bin/bash\necho run\n", 0o755)
        self.write("skills/models/tao-beta/SKILL.md", BETA)
        self.write("skills/models/tao-beta/references/notes.md", "notes\n")
        self.write("skills/models/tao-beta/tests/test_beta.py", "def test_beta():\n    pass\n")
        self.suite([{"id": "alpha-basic", "expected_skill": "tao-alpha"},
                    {"id": "beta-basic", "expected_skill": "tao-beta"}])
        self.git("init", "-q")
        self.commit()

    def write(self, relative, text, mode=0o644):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        path.chmod(mode)

    def suite(self, tasks):
        self.write("evals/evals.json", tasks if isinstance(tasks, str) else json.dumps(tasks))

    def manifest(self, **fields):
        manifest = {"name": builder.PLUGIN_NAME, "version": "0.0.1", "description": "fixture"}
        manifest.update(fields)
        self.write(".claude-plugin/plugin.json", json.dumps({**manifest, "skills": ["./skills/"]}))
        self.write(".codex-plugin/plugin.json", json.dumps({**manifest, "skills": "./skills/"}))

    def git(self, *args):
        identity = ["-c", "user.name=test", "-c", "user.email=test@example.com",
                    "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"]
        subprocess.run(["git", *identity, *args], cwd=self.root, check=True, capture_output=True)

    def commit(self):
        self.git("add", "-A", "-f")  # -f: a global ignore file must not hide .DS_Store here
        self.git("commit", "-q", "--allow-empty", "-m", "fixture")

    def main(self, *options):
        """Run the generator on this checkout: (exit code, stdout, stderr)."""
        self.commit()
        arguments = ["--source", str(self.root), "--output", str(self.output), *options]
        code = builder.main(arguments)
        captured = self.capsys.readouterr()
        return code, captured.out, captured.err

    def fails_with(self, message):
        """The build stops with exit code 2 and ``message``, and writes nothing."""
        existed = self.output.exists()
        code, _, error = self.main()
        assert code == 2 and message in error, error
        assert self.output.exists() == existed
        return error


@pytest.fixture
def small(tmp_path, capsys, monkeypatch):
    """The generator's tables, cut down to what a checkout of two tiny skills has."""
    monkeypatch.setattr(builder, "SUPPORT_FILES", {
        "AGENTS.md": "agent guidance", builder.EVAL_SUITE: "suite",
        "scripts/skill_layers.py": "layer lookup",
    })
    for table in ("PACKAGE_EDITS", "LAYOUT_NEUTRAL"):
        monkeypatch.setattr(builder, table, ())
    for table in ("ONE_SKILL_FILES_LEFT_OUT", "SKILL_FILES_KEPT", "LEFT_OUT_MENTIONS",
                  "UNSHIPPED_SCRIPT_MENTIONS", "LAYER_WORDS_ALLOWED"):
        monkeypatch.setattr(builder, table, {})
    return SmallCheckout(tmp_path, capsys)


def test_small_checkout_builds_and_exit_codes_are_0_1_and_2(small):
    code, out, error = small.main()
    assert (code, error) == (0, "") and f"written: {small.output}" in out
    shipped = sorted(path.relative_to(small.output).as_posix() for path in _files(small.output))
    assert shipped == [
        ".claude-plugin/plugin.json", ".codex-plugin/plugin.json", "AGENTS.md",
        "catalog/skill-layers.json", "evals/evals.json", "plugin.json", "scripts/skill_layers.py",
        "skills/tao-alpha/SKILL.md", "skills/tao-alpha/scripts/run.sh", "skills/tao-beta/SKILL.md",
        "skills/tao-beta/references/notes.md",
    ]
    alpha = _text(small.output / "skills/tao-alpha/SKILL.md")
    assert "(../tao-beta/SKILL.md)" in alpha and "`skills/tao-beta/references/notes.md`" in alpha
    assert (small.output / "skills/tao-alpha/scripts/run.sh").stat().st_mode & 0o777 == 0o755

    code, out, error = small.main("--check")
    assert (code, error) == (0, "") and f"up to date: {small.output}" in out

    (small.output / "AGENTS.md").write_text("# Agents!\n", encoding="utf-8")
    code, _, error = small.main("--check")
    assert code == 1 and "content differs: AGENTS.md" in error

    small.suite("{not json")
    for options in ((), ("--check",)):
        code, _, error = small.main(*options)
        assert code == 2 and "evals/evals.json is not valid JSON" in error
    assert _text(small.output / "AGENTS.md") == "# Agents!\n"


def test_build_stops_when_a_skill_has_no_task_in_the_root_suite(small):
    small.suite([{"id": "alpha-basic", "expected_skill": "tao-alpha"}])
    small.fails_with("evals/evals.json has no task for tao-beta")
    small.suite([{"id": "alpha-basic", "expected_skill": "tao-alpha"},
                 {"id": "beta-basic", "expected_skill": "tao-beta"},
                 {"id": "gamma-basic", "expected_skill": "tao-gamma"}])
    small.fails_with("evals/evals.json expects a skill the package lacks: tao-gamma")


@pytest.mark.parametrize("content", ('{"tasks": []}', '["alpha-basic"]', '[{"id": "alpha-basic"}]'))
def test_build_stops_on_a_suite_of_the_wrong_shape(small, content):
    small.suite(content)
    small.fails_with("evals/evals.json must be a JSON array of task objects")


def test_build_stops_on_a_malformed_or_misshapen_manifest(small):
    small.write(".claude-plugin/plugin.json", "{")
    small.fails_with(".claude-plugin/plugin.json is not valid JSON")
    small.write(".claude-plugin/plugin.json", "[]")
    small.fails_with(".claude-plugin/plugin.json must hold a JSON object")


def test_build_stops_on_a_manifest_field_of_the_wrong_type(small):
    """The root plugin.json has to satisfy the Agent Plugins 1.0.0 schema."""
    for fields, message in (
        ({"author": "NVIDIA"}, "author must be an object"),
        ({"author": {"name": "NVIDIA", "url": 1}}, "every author field must be a string"),
        ({"keywords": "tao"}, "keywords must be a list"),
        ({"keywords": ["tao", 7]}, "every keyword must be a string"),
        ({"description": ["fixture"]}, "description must be a string"),
        ({"repository": {"type": "git", "url": "https://example.com"}},
         "repository must be a string"),
        ({"license": None}, "license must be a string"),
    ):
        small.manifest(**fields)
        small.fails_with("plugin.json: " + message)
    small.manifest(author={"name": "NVIDIA", "url": "https://example.com", "team": "dropped"},
                   keywords=["tao"], homepage="https://example.com")
    assert small.main()[0] == 0
    assert _json(small.output / "plugin.json")["author"] == {
        "name": "NVIDIA", "url": "https://example.com"}


def test_build_stops_when_a_listed_edit_no_longer_matches(small, monkeypatch):
    alpha = "skills/core/tao-alpha/SKILL.md"
    monkeypatch.setattr(builder, "PACKAGE_EDITS", ((alpha, "# Alpha\n", "# Alpha, packaged\n"),))
    assert small.main()[0] == 0
    assert "# Alpha, packaged" in _text(small.output / "skills/tao-alpha/SKILL.md")
    small.write(alpha, ALPHA.replace("# Alpha\n", "# Alpha skill\n"))
    small.fails_with(f"PACKAGE_EDITS: expected one match in {alpha}, found 0")
    monkeypatch.setattr(builder, "PACKAGE_EDITS", (("skills/core/tao-alpha/GONE.md", "a", "b"),))
    small.fails_with("PACKAGE_EDITS names a file the source does not have")
    left_out = "skills/models/tao-beta/tests/test_beta.py"
    monkeypatch.setattr(builder, "PACKAGE_EDITS", ((left_out, "pass", "return"),))
    small.fails_with(f"PACKAGE_EDITS names a file the package leaves out: {left_out}")


def test_build_stops_on_a_layer_reference_that_survives_or_must_not_be_rewritten(small):
    alpha = "skills/core/tao-alpha/SKILL.md"
    small.write(alpha, ALPHA + "\nRun `ls skills/models` first.\n")
    small.fails_with(f"{alpha} still refers to the layer layout: skills/models")
    small.write(alpha, ALPHA + "\nEvery model under `models/`.\n")
    small.fails_with(f"{alpha} still refers to the layer layout: `models/`")
    small.write(alpha, ALPHA + "\nSee https://example.com/tree/skills/models/tao-beta/SKILL.md.\n")
    small.fails_with("(inside a URL)")
    small.write(alpha, ALPHA + "\nRun from the `tao-skill-bank` repo root.\n")
    small.fails_with(f"{alpha} still refers to the layer layout: `tao-skill-bank` repo root")
    small.write(alpha, ALPHA + "\n```\nskills/\n├── core/\n└── models/   # one per network\n```\n")
    small.fails_with(f"{alpha} still refers to the layer layout: ├── core/")
    small.write(alpha, ALPHA + "\n```bash\ncd tao-skill-bank\n"
                               "cat skills/models/tao-beta/SKILL.md\n```\n")
    small.fails_with("(after cd tao-skill-bank: a clone keeps its skills in layer folders)")
    small.write(alpha, ALPHA + "\n`git show main:skills/models/tao-beta/SKILL.md`\n")
    small.fails_with("(after a colon: a path inside a git ref or a mount target, not in a bank)")
    small.write(alpha, ALPHA)
    small.write("skills/core/tao-alpha/scripts/run.sh",
                '#!/bin/bash\nfor skill in "$BANK"/skills/models/*/; do echo "$skill"; done\n',
                0o755)
    small.fails_with("(code that lists one layer folder; in the package it would list every "
                     "skill)")
    small.write("skills/core/tao-alpha/scripts/run.sh", "#!/bin/bash\necho run\n", 0o755)
    small.write("skills/core/tao-alpha/scripts/up.py",
                "from pathlib import Path\nROOT = Path(__file__).resolve().parents[4]\n")
    small.write(alpha, ALPHA)
    small.fails_with("climbs above its skill folder")


def test_build_stops_when_a_shipped_file_names_what_the_package_lacks(small):
    alpha = "skills/core/tao-alpha/SKILL.md"
    small.write(alpha, ALPHA + "\nThen run `scripts/repository_only.py`.\n")
    small.fails_with("skills/tao-alpha/SKILL.md names scripts/repository_only.py, which the "
                     "package leaves out")
    small.write(alpha, ALPHA)
    small.write("skills/models/tao-beta/SKILL.md", BETA + "\nRun `python -m pytest tests/`.\n")
    small.fails_with("skills/tao-beta/SKILL.md names skills/tao-beta/tests, which the package "
                     "leaves out")
    for command in ("python -m pytest tests -q", "python -m unittest discover -s tests"):
        small.write("skills/models/tao-beta/SKILL.md", BETA + f"\nRun `{command}`.\n")
        small.fails_with("skills/tao-beta/SKILL.md names skills/tao-beta/tests, which the "
                         "package leaves out")
    small.write("skills/models/tao-beta/SKILL.md", BETA + "\nIts signature is skill.oms.sig.\n")
    small.fails_with("skills/tao-beta/SKILL.md names skill.oms.sig, which the package leaves out")
    small.write("skills/models/tao-beta/SKILL.md", BETA)
    small.write("skills/models/tao-beta/scripts/test_tool.sh", "echo looks like a test\n")
    small.fails_with("skills/models/tao-beta/scripts/test_tool.sh looks like a test")


def test_build_stops_on_a_link_that_would_break(small):
    alpha = "skills/core/tao-alpha/SKILL.md"
    small.write(alpha, ALPHA + "\nSee the [guide](../../../docs/guide.md).\n")
    small.fails_with(f"{alpha} has relative links that break in the package: "
                     "../../../docs/guide.md")
    # A link that is broken in the repository already is not this build's doing.
    small.write(alpha, ALPHA + "\nSee the [guide](../../../docs/missing.md).\n")
    assert small.main()[0] == 0


def test_build_stops_on_a_tracked_symlink_and_on_a_junk_file(small):
    link = small.root / "skills/core/tao-alpha/link.md"
    link.symlink_to("SKILL.md")
    small.fails_with("skills/core/tao-alpha/link.md is a symlink or a submodule")
    link.unlink()
    small.write("skills/core/tao-alpha/.DS_Store", "junk")
    small.fails_with("skills/tao-alpha/.DS_Store is a junk file the pipeline rejects")
    (small.root / "skills/core/tao-alpha/.DS_Store").unlink()
    small.write("skills/core/tao-alpha/docs/SKILL.md", "---\nname: nested\n---\n")
    small.fails_with("a SKILL.md is allowed only at skills/<skill>/SKILL.md")


def test_build_stops_on_a_text_file_that_is_not_utf_8_and_on_a_reserved_name(small):
    notes = small.root / "skills/models/tao-beta/references/notes.md"
    notes.write_bytes("caf\u00e9 in skills/models/tao-beta\n".encode("latin-1"))
    small.fails_with("skills/models/tao-beta/references/notes.md has a text extension and is not "
                     "UTF-8 text")
    notes.write_bytes("see skills/models/tao-beta\n".encode("utf-16"))
    small.fails_with("has a text extension and is not UTF-8 text")
    notes.write_text("notes\n", encoding="utf-8")
    # Binary content under another extension is copied as it is.
    (small.root / "skills/models/tao-beta/references/weights.bin").write_bytes(b"\x00\xff\x00")
    assert small.main()[0] == 0
    assert (small.output / "skills/tao-beta/references/weights.bin").read_bytes() == b"\x00\xff\x00"
    # A SKILL.md that the rewrite pushes over the size that signing reads stops the build.
    beta = "skills/models/tao-beta/SKILL.md"
    link = "Use `tao-skill-bank:tao-alpha`.\n"
    filler = "x" * (builder.SKILL_MD_MAX_CHARS - len(BETA) - len(link) - 1) + "\n"
    small.write(beta, BETA + filler + link)
    assert len(_text(small.root / beta)) == builder.SKILL_MD_MAX_CHARS
    small.fails_with(f"skills/tao-beta/SKILL.md has {builder.SKILL_MD_MAX_CHARS + 4} characters "
                     f"in the package; the limit is {builder.SKILL_MD_MAX_CHARS}")
    small.write(beta, BETA)
    # What per-skill signing left in a skill folder stays out; a name that verification
    # writes into the package is never generated.
    small.write("skills/models/tao-beta/skill-card-review-needed.md", "left by a signing run\n")
    assert small.main()[0] == 0
    assert not (small.output / "skills/tao-beta/skill-card-review-needed.md").exists()
    small.write("skills/models/tao-beta/references/skill.oms.sig", "in the wrong place\n")
    small.fails_with("skills/tao-beta/references/skill.oms.sig is a name reserved for the files "
                     "verification adds")


def test_operating_system_errors_end_with_exit_code_2(small, monkeypatch):
    assert small.main()[0] == 0
    blocker = small.output.parent / "file.txt"
    blocker.write_text("a file where a folder is needed\n", encoding="utf-8")
    code = builder.main(["--source", str(small.root), "--output", str(blocker / "package")])
    assert code == 2 and "was not written completely" in small.capsys.readouterr().err

    def unreadable(*_):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(builder, "read_file", unreadable)
    code, _, error = small.main()
    assert code == 2 and "PermissionError" in error
    code = builder.main(["--source", str(small.root.parent / "missing")])
    assert code == 2 and "is not a git checkout" in small.capsys.readouterr().err


def test_rebuild_lists_what_it_removes_first_and_stops_on_a_file_it_does_not_know(small):
    small.write("skills/models/tao-beta/archive/old.md", "superseded\n")
    assert small.main()[0] == 0
    stale = small.output / "skills/tao-beta/archive/old.md"
    assert stale.is_file()
    (small.root / "skills/models/tao-beta/archive/old.md").unlink()
    (small.output / "unknown.txt").write_text("not written by any build\n", encoding="utf-8")
    code, out, error = small.main()
    assert code == 2 and out == "" and "2 file(s) that no build writes" in error
    assert "  skills/tao-beta/archive/old.md" in error and "  unknown.txt" in error
    assert "Nothing was changed" in error and "--force" in error
    assert stale.is_file() and (small.output / "unknown.txt").is_file()

    code, out, _ = small.main("--force")
    assert code == 0 and not stale.exists() and not (small.output / "unknown.txt").exists()
    assert not stale.parent.exists()  # the folder it left empty is gone as well
    assert out.index("removing 2 file(s) from") < out.index("cleaned: ")
    assert "  skills/tao-beta/archive/old.md  (not part of this build)" in out
    assert "  unknown.txt  (not part of this build)" in out
    code, out, _ = small.main()
    assert code == 0 and f"up to date: {small.output}" in out and "removing" not in out


def test_rebuild_of_a_signed_folder_needs_force_only_when_something_changes(small):
    assert small.main()[0] == 0
    _add(small.output, "plugin.oms.sig", "plugin-card.md", ".eval-results/run-1/eval-result.json")
    code, out, _ = small.main()
    assert code == 0 and f"up to date: {small.output}" in out

    small.write("AGENTS.md", "# Agents, changed\n")
    code, _, error = small.main()
    assert code == 2 and "is signed (plugin.oms.sig)" in error
    assert "content differs: AGENTS.md" in error and "--force" in error
    assert _text(small.output / "AGENTS.md") == "# Agents\n"
    assert (small.output / "plugin.oms.sig").is_file()
    code, _, error = small.main("--check")
    assert code == 1 and "content differs: AGENTS.md" in error
    assert "plugin.oms.sig" not in error and "plugin-card.md" not in error

    code, out, _ = small.main("--force")
    assert code == 0 and _text(small.output / "AGENTS.md") == "# Agents, changed\n"
    assert out.index("removing 1 file(s) from") < out.index("written: ")
    assert "  plugin.oms.sig  (signature of the earlier content)" in out
    assert "removed 1 signature file(s) that the rebuild made stale (plugin.oms.sig)" in out
    assert "cards, benchmark and evaluation results describe the earlier content" in out
    assert not (small.output / "plugin.oms.sig").exists()
    assert (small.output / "plugin-card.md").is_file()
    assert (small.output / ".eval-results/run-1/eval-result.json").is_file()

    # A file that no build writes, in a signed folder that is otherwise up to date: the
    # refusal says that --force deletes it, and the signature, which lists it, goes too.
    _add(small.output, "plugin.oms.sig", "notes.txt")
    signed = _listing(small.output)
    code, _, error = small.main()
    assert code == 2 and "is signed (plugin.oms.sig)" in error
    assert "  not generated: notes.txt" in error
    assert "the 1 file(s) listed as not generated are deleted" in error
    assert _listing(small.output) == signed
    code, out, _ = small.main("--force")
    assert code == 0 and f"cleaned: {small.output} (no packaged file changed)" in out
    assert "  notes.txt  (not part of this build)" in out
    assert "  plugin.oms.sig  (signature of the earlier content)" in out
    assert "removed 1 signature file(s) that the rebuild made stale (plugin.oms.sig)" in out
    assert "cards, benchmark and evaluation results describe the earlier content" in out
    assert not (small.output / "plugin.oms.sig").exists()
    assert not (small.output / "notes.txt").exists()
    assert (small.output / "plugin-card.md").is_file()

    # The same holds for a file that an earlier build wrote and this one does not.
    _add(small.output, "plugin.oms.sig")
    code, _, error = small.main("--no-codex-manifest")
    assert code == 2 and "  not generated: .codex-plugin/plugin.json" in error
    code, out, _ = small.main("--no-codex-manifest", "--force")
    assert code == 0 and "  .codex-plugin/plugin.json  (not part of this build)" in out
    assert "removed 1 signature file(s) that the rebuild made stale (plugin.oms.sig)" in out
    assert not (small.output / "plugin.oms.sig").exists()
    assert not (small.output / ".codex-plugin").exists()

    # Junk alone is cleaned from a signed folder without --force, and the signature stays.
    _add(small.output, "plugin.oms.sig", ".DS_Store")
    code, out, _ = small.main("--no-codex-manifest")
    assert code == 0 and "  .DS_Store  (junk the pipeline rejects)" in out
    assert f"cleaned: {small.output} (no packaged file changed)" in out
    assert "signature file(s)" not in out and "describe the earlier content" not in out
    assert (small.output / "plugin.oms.sig").is_file()
    assert not (small.output / ".DS_Store").exists()


def test_check_reports_junk_separately_and_does_not_count_what_git_ignores(
        small, monkeypatch, tmp_path):
    # Only the ignore rules of the checkout count here, not those of the developer.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "no-config"))
    small.write(".gitignore", "__pycache__/\n*.pyc\n")
    small.output = small.root / "plugins" / builder.PLUGIN_NAME
    assert small.main()[0] == 0
    # A helper that was run from the folder leaves its cache behind; git ignores it.
    cache = "scripts/__pycache__/skill_layers.cpython-312.pyc"
    _add(small.output, cache)
    arguments = ["--source", str(small.root), "--output", str(small.output)]
    assert builder.main([*arguments, "--check"]) == 0
    out = small.capsys.readouterr().out
    assert "up to date" in out and "1 junk file(s) that git ignores are not counted" in out
    # Junk that git would commit is a difference, under its own label.
    _add(small.output, "__MACOSX/._plugin.json")
    assert builder.main([*arguments, "--check"]) == 1
    error = small.capsys.readouterr().err
    assert "  junk: __MACOSX/._plugin.json" in error and cache not in error
    assert "not generated" not in error
    # A rebuild removes both kinds without --force and says so first.
    assert builder.main(arguments) == 0
    out = small.capsys.readouterr().out
    assert out.index("removing 2 file(s) from") < out.index("cleaned: ")
    assert f"  {cache}  (junk the pipeline rejects)" in out
    assert not (small.output / "scripts/__pycache__").exists()
    assert not (small.output / "__MACOSX").exists()
    # Outside a work tree nothing is ignored, so the same cache file is a difference.
    outside = small.root.parent / "outside" / builder.PLUGIN_NAME
    assert builder.main(["--source", str(small.root), "--output", str(outside)]) == 0
    _add(outside, cache)
    assert builder.main(["--source", str(small.root), "--output", str(outside), "--check"]) == 1
    assert f"  junk: {cache}" in small.capsys.readouterr().err


def test_a_committed_package_folder_stays_out_of_the_clones_the_bank_checks_use(small):
    """The bank checks run in copies of the tracked files; a committed folder is tracked too."""
    small.output = small.root / FLAT_BANK
    assert small.main()[0] == 0
    _add(small.output, "plugin.oms.sig")
    small.commit()                                    # the folder is committed, and signed
    small.write("AGENTS.md", "# Agents, changed\n")   # the source moves on: the folder is stale
    small.commit()
    tracked = builder.tracked_files(small.root)
    assert {f"{FLAT_BANK}/plugin.oms.sig", f"{FLAT_BANK}/AGENTS.md"} <= set(tracked)

    clone = _clone_without_package(small.root, small.root.parent / "clone")
    assert _text(clone / "AGENTS.md") == "# Agents, changed\n"
    assert not (clone / FLAT_BANK).exists()
    # The build under test goes into such a clone as into an empty folder: nothing that
    # is signed or stale is in its way.
    package = builder.build(small.root)
    done = builder.write(package, clone / FLAT_BANK)
    assert done.removed == [] and builder.compare(package, clone / FLAT_BANK) == []
    assert _text(clone / FLAT_BANK / "AGENTS.md") == "# Agents, changed\n"
    assert not (clone / FLAT_BANK / "plugin.oms.sig").exists()


def test_output_that_is_a_git_work_tree_is_refused_unless_it_is_inside_the_source(small):
    assert small.main()[0] == 0
    subprocess.run(["git", "init", "-q", str(small.output)], check=True, capture_output=True)
    small.write("AGENTS.md", "# Agents, changed\n")
    for options in ((), ("--force",)):
        code, _, error = small.main(*options)
        assert code == 2 and "is the root of a git work tree" in error
    assert _text(small.output / "AGENTS.md") == "# Agents\n"
    assert (small.output / ".git" / "HEAD").is_file()
    # --check reads such a folder, without its .git
    code, _, error = small.main("--check")
    assert code == 1 and "content differs: AGENTS.md" in error and ".git" not in error
