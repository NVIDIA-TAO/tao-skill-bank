# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the release delivered by the default marketplace."""

import json
import os
import re
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_NAME = "nvidia-tao"
# Every file that carries the plugin version; they must always agree.
VERSIONED_MANIFESTS = (
    ".claude-plugin/plugin.json",
    ".claude-plugin/marketplace.json",
    ".codex-plugin/plugin.json",
    ".cursor-plugin/plugin.json",
    ".cursor-plugin/marketplace.json",
)


def _load(relative_path):
    return json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def _manifest_version(relative_path):
    document = _load(relative_path)
    if relative_path.endswith("marketplace.json"):
        return document["metadata"]["version"]
    return document["version"]


def _skills_on_disk():
    return {p.parent.resolve() for p in (REPO_ROOT / "skills").rglob("SKILL.md")}


def test_default_branch_publishes_7_3_images_and_new_plugin_version():
    versions = yaml.safe_load((REPO_ROOT / "versions.yaml").read_text())
    images = versions["images"]["tao_toolkit"]
    assert "7.3.0" in images["pyt"]
    assert "7.3.0" in images["data_services"]

    # The plugin version is Claude Code's install cache key: it only fetches
    # new content when the version changes. It is the plugin's own version, not
    # the TAO release number, and every manifest has to carry the same value.
    versions = {path: _manifest_version(path) for path in VERSIONED_MANIFESTS}
    assert len(set(versions.values())) == 1, versions
    version = next(iter(versions.values()))
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), version
    assert tuple(int(part) for part in version.split(".")) > (0, 1, 16), (
        "0.1.16 shipped with TAO 7.3.0; later content needs a newer plugin version"
    )


def test_default_marketplace_exposes_the_canonical_codex_plugin_name():
    """A root marketplace add must expose the name Codex installs.

    The shared marketplace historically listed only the Claude-facing
    ``tao-skills`` alias. Codex reads its canonical name from plugin.json and
    then could not find that name in the marketplace it had just registered.
    Keep the alias for existing Claude installs, but require the canonical
    cross-client entry too.
    """
    codex_manifest = json.loads(
        (REPO_ROOT / ".codex-plugin/plugin.json").read_text()
    )
    shared_marketplace = json.loads(
        (REPO_ROOT / ".claude-plugin/marketplace.json").read_text()
    )
    names = {entry["name"] for entry in shared_marketplace["plugins"]}

    assert codex_manifest["name"] in names
    assert "tao-skills" in names


def test_one_plugin_name_across_every_harness():
    """The plugin is ``nvidia-tao`` in each manifest and each marketplace.

    Codex refuses to install a plugin whose manifest name differs from its
    marketplace entry, and a user should be able to type the same
    ``nvidia-tao@tao-skill-bank`` in every client.
    """
    for path in (
        ".claude-plugin/plugin.json",
        ".codex-plugin/plugin.json",
        ".cursor-plugin/plugin.json",
    ):
        assert _load(path)["name"] == PLUGIN_NAME, path

    claude = _load(".claude-plugin/marketplace.json")
    codex = _load(".agents/plugins/marketplace.json")
    cursor = _load(".cursor-plugin/marketplace.json")
    assert {claude["name"], codex["name"], cursor["name"]} == {"tao-skill-bank"}
    assert claude["plugins"][0]["name"] == PLUGIN_NAME
    assert [entry["name"] for entry in codex["plugins"]] == [PLUGIN_NAME]
    assert [entry["name"] for entry in cursor["plugins"]] == [PLUGIN_NAME]


def test_retired_claude_entries_are_renamed_not_dropped():
    """Removing a marketplace entry breaks every install that used it.

    Claude Code fails to load a plugin whose entry disappeared unless the
    marketplace maps the old name to a current one. ``renames`` moves installs
    that already exist; it does not make the old name installable again. Treat
    it as append-only.
    """
    marketplace = _load(".claude-plugin/marketplace.json")
    names = {entry["name"] for entry in marketplace["plugins"]}
    renames = marketplace.get("renames", {})

    for retired in ("tao-skill-bank", "tao-daft-process"):
        assert retired not in names
        assert renames.get(retired) == PLUGIN_NAME, retired
    assert set(renames.values()) <= names
    assert not set(renames) & names


def test_claude_manifest_declares_no_custom_skill_selector():
    """Skill paths belong in the marketplace entry, not in the Claude manifest.

    NVIDIA's plugin evaluation reads ``.claude-plugin/plugin.json`` and rejects
    any ``skills`` value other than the plugin's own ``./skills`` directory as
    an unsupported custom skill selector. Listing the layer directories there
    makes Claude Code load the nested skills but makes the plugin impossible to
    evaluate.
    """
    manifest = _load(".claude-plugin/plugin.json")
    declared = manifest.get("skills")
    if declared is None:
        return
    if isinstance(declared, list):
        assert len(declared) == 1, declared
        declared = declared[0]
    assert declared.rstrip("/") in {"skills", "./skills"}, declared


def test_no_agent_plugins_manifest_while_skills_are_nested():
    """A root ``plugin.json`` switches Codex to one-level skill discovery.

    When that file declares the Agent Plugins schema, the ``skills`` path in
    ``.codex-plugin/plugin.json`` no longer applies and Codex loads only
    ``skills/<skill>/SKILL.md``. The bank nests skills one level deeper, so
    adding that manifest silently takes Codex from every skill to none. It may
    be added only together with a flat ``skills/`` tree.
    """
    if not (REPO_ROOT / "plugin.json").exists():
        return
    nested = sorted(
        str(p.relative_to(REPO_ROOT))
        for p in _skills_on_disk()
        if p.parent != (REPO_ROOT / "skills").resolve()
    )
    assert not nested, f"root plugin.json with nested skills hides them from Codex: {nested[:5]}"


def _symlinks_in_the_repository():
    """Symlinks the repository ships: tracked ones, or all of them without git."""
    try:
        staged = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "--stage"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        staged = None
    if staged:
        return [
            line.split("\t", 1)[1]
            for line in staged.splitlines()
            if line.startswith("120000 ")
        ]

    # An extracted archive has no index to ask, so walk the tree.
    links = []
    for directory, subdirs, files in os.walk(REPO_ROOT):
        subdirs[:] = [d for d in subdirs if d not in {".git", "__pycache__", ".pytest_cache"}]
        for name in (*subdirs, *files):
            path = Path(directory, name)
            if path.is_symlink():
                links.append(str(path.relative_to(REPO_ROOT)))
    return links


def test_plugin_tree_has_no_symlinks():
    """Codex drops symlinks when it copies a plugin into its install cache.

    Checkouts without symlink support and zip extraction turn them into small
    text files instead. Either way the linked content silently goes missing.
    The plugin is the whole repository root, so no directory is exempt.
    """
    links = _symlinks_in_the_repository()
    assert not links, f"symlinks in the plugin tree: {links[:5]}"


def test_pas_resolves_published_versions_without_embedding_image_uris():
    versions = yaml.safe_load((REPO_ROOT / "versions.yaml").read_text())
    images = versions["images"]["tao_toolkit"]
    skill = REPO_ROOT / "skills/applications/tao-run-deft-pas"
    preflight = (skill / "references/preflight.md").read_text()

    assert "images.tao_toolkit.deft_pas_pyt" in preflight
    assert "images.tao_toolkit.deft_pas_data_services" in preflight
    for path in (*skill.rglob("*.py"), *skill.rglob("*.md")):
        text = path.read_text()
        assert images["deft_pas_pyt"] not in text
        assert images["deft_pas_data_services"] not in text


def test_codex_manifest_skills_root_covers_every_skill_on_disk():
    """The README promises Codex parity with Claude Code; the manifest must deliver it.

    Codex loads only what sits under the single ``skills`` root in
    ``.codex-plugin/plugin.json`` — it recurses, so one root is enough, but that
    root has to contain every ``SKILL.md`` in the bank. Pointing it at
    ``./skills/core/`` exposed 4 skills of 76.
    """
    codex_manifest = json.loads(
        (REPO_ROOT / ".codex-plugin/plugin.json").read_text()
    )
    root = (REPO_ROOT / codex_manifest["skills"]).resolve()
    assert root.is_dir(), f"Codex skills root does not exist: {root}"

    all_skills = {p.parent.resolve() for p in (REPO_ROOT / "skills").rglob("SKILL.md")}
    outside = sorted(
        str(p.relative_to(REPO_ROOT)) for p in all_skills if root not in p.parents
    )
    assert not outside, (
        f"{len(outside)} skills sit outside the Codex manifest root {codex_manifest['skills']!r} "
        f"and are invisible to Codex: {outside[:5]}{' ...' if len(outside) > 5 else ''}"
    )


def _assert_lists_every_skill(listed_paths, owner):
    listed = {(REPO_ROOT / s).resolve() for s in listed_paths}
    on_disk = _skills_on_disk()
    missing = sorted(str(p.relative_to(REPO_ROOT)) for p in on_disk - listed)
    stale = sorted(str(p.relative_to(REPO_ROOT)) for p in listed - on_disk)
    assert len(listed_paths) == len(set(listed_paths)), f"{owner} lists a skill twice"
    assert not missing, f"skills on disk but not installed by {owner}: {missing}"
    assert not stale, f"{owner} lists skills that do not exist: {stale}"


def test_documented_claude_plugin_lists_every_skill_on_disk():
    """Each Claude entry must list every skill, core included.

    Claude Code scans ``skills/`` one level deep, so it loads exactly the paths a
    marketplace entry lists. ``tao-skills`` listed the 77 layer skills and omitted
    ``skills/core/``, so the documented Claude install and the fixed Codex install
    disagreed by five skills. ``nvidia-tao`` is the plugin; ``tao-skills`` is its
    alias for existing installs and must stay identical.
    """
    marketplace = _load(".claude-plugin/marketplace.json")
    entries = {p["name"]: p for p in marketplace["plugins"]}
    for name in (PLUGIN_NAME, "tao-skills"):
        _assert_lists_every_skill(entries[name]["skills"], name)


def test_cursor_manifest_lists_every_skill_on_disk():
    """Cursor also scans ``skills/`` one level deep.

    The bank therefore names every skill directory in its Cursor manifest. A
    skill missing here is invisible in Cursor even though Claude Code and Codex
    load it.
    """
    manifest = _load(".cursor-plugin/plugin.json")
    _assert_lists_every_skill(manifest["skills"], ".cursor-plugin/plugin.json")


def test_readme_install_commands_name_what_the_pinned_tag_ships():
    """The README install commands run against the pinned tag, not against main.

    ``7.3.0`` is the last tag that ships the old plugin names. A pin bump that
    changes only the tag leaves commands that fail against the new release.
    """
    readme = (REPO_ROOT / "README.md").read_text()
    pinned = re.search(
        r"^/plugin marketplace add NVIDIA-TAO/tao-skill-bank@(\S+)$", readme, re.MULTILINE
    ).group(1)
    claude_install = re.search(r"^/plugin install (\S+)$", readme, re.MULTILINE).group(1)
    codex_install = re.search(r"^codex plugin add (\S+)", readme, re.MULTILINE).group(1)

    if pinned == "7.3.0":
        expected = ("tao-skills@tao-skill-bank", "tao-skill-bank@tao-local-plugins")
    else:
        expected = (f"{PLUGIN_NAME}@tao-skill-bank", f"{PLUGIN_NAME}@tao-skill-bank")
    assert (claude_install, codex_install) == expected, f"README pins {pinned}"


def test_readme_does_not_hardcode_stale_skill_counts():
    """README once said skills/core/ held 2 skills while it held 4.

    The repository-structure tree listed a count per layer; every one of them
    had drifted. Counts are not maintained, so the tree must not carry them.
    """
    readme = (REPO_ROOT / "README.md").read_text()
    layer_lines = [
        line for line in readme.splitlines()
        if re.search(r"[├└]── (applications|data|models|platform|core)/", line)
    ]
    assert len(layer_lines) == 5, layer_lines
    stale = [line.strip() for line in layer_lines if re.search(r"#\s*\d+\s", line)]
    assert not stale, f"README hardcodes skill counts that will drift: {stale}"
