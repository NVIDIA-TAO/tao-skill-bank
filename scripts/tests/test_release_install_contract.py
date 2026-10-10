# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the release delivered by the default marketplace."""

import json
import re
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_default_branch_publishes_7_2_images_and_new_plugin_version():
    versions = yaml.safe_load((REPO_ROOT / "versions.yaml").read_text())
    images = versions["images"]["tao_toolkit"]
    assert "7.2.0" in images["pyt"]
    assert "7.2.0" in images["data_services"]

    manifests = [
        json.loads((REPO_ROOT / ".claude-plugin/plugin.json").read_text()),
        json.loads((REPO_ROOT / ".codex-plugin/plugin.json").read_text()),
    ]
    marketplace = json.loads(
        (REPO_ROOT / ".claude-plugin/marketplace.json").read_text()
    )
    advertised = marketplace["metadata"]["version"]
    assert advertised == "0.1.14"
    assert {manifest["version"] for manifest in manifests} == {advertised}


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
    ``./skills/core/`` exposed 4 skills of 76 (NVBug 6777460).
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


def test_documented_claude_plugin_lists_every_skill_on_disk():
    """The README installs ``tao-skills``; it must list every skill, core included.

    Claude Code loads exactly the paths a marketplace entry lists. ``tao-skills``
    listed the 77 layer skills and omitted ``skills/core/``, so the documented
    Claude install and the fixed Codex install disagreed by five skills
    (NVBug 6777460 review).
    """
    marketplace = json.loads(
        (REPO_ROOT / ".claude-plugin/marketplace.json").read_text()
    )
    entry = next(p for p in marketplace["plugins"] if p["name"] == "tao-skills")
    listed = {(REPO_ROOT / s).resolve() for s in entry["skills"]}
    on_disk = {p.parent.resolve() for p in (REPO_ROOT / "skills").rglob("SKILL.md")}
    missing = sorted(str(p.relative_to(REPO_ROOT)) for p in on_disk - listed)
    stale = sorted(str(p.relative_to(REPO_ROOT)) for p in listed - on_disk)
    assert not missing, f"skills on disk but not installed by tao-skills: {missing}"
    assert not stale, f"tao-skills lists skills that do not exist: {stale}"


def test_readme_does_not_hardcode_stale_skill_counts():
    """README once said skills/core/ held 2 skills while it held 4 (NVBug 6777460).

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
