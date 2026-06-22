#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bulk-add DAFT-style frontmatter fields to every SKILL.md.

Mechanical fields (no per-skill judgment required):
  - license: Apache-2.0          (always — required by validator)
  - metadata.author              (from `git log` — top contributors to the file)
  - metadata.version: "0.1"      (default starting version)
  - compatibility:               (heuristic per layer + presence of container_image)
  - allowed-tools:               (heuristic — default 'Read Bash'; orchestrators add 'Write')
  - tags:                        (migrated from references/skill_info.yaml; falls back
                                  to layer + skill-name heuristic when absent)

Also removes `tags:` from `references/skill_info.yaml` after migration so SKILL.md
becomes the single source of truth for tags.

Preserves existing fields verbatim. Idempotent — running twice is a no-op for
fields already present.

Usage:
  python3 scripts/migrate-frontmatter.py             # dry-run, prints actions
  python3 scripts/migrate-frontmatter.py --apply     # writes changes
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml


def git_authors(path: str) -> str:
    """Return top git authors for a file as a comma-separated string."""
    try:
        out = subprocess.check_output(
            ["git", "log", "--format=%an", "--", path],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return ""
    seen = []
    for line in out.splitlines():
        name = line.strip()
        if name and name not in seen:
            seen.append(name)
        if len(seen) >= 3:
            break
    return ", ".join(seen)


def compatibility_for(skill_path: str) -> str:
    """Heuristic compatibility line based on layer + skill_info.yaml content."""
    layer = skill_path.split("/", 1)[0] if "/" in skill_path else ""
    name = os.path.basename(os.path.dirname(skill_path))

    # Platform-specific known values
    platform_compat = {
        "docker": "Requires docker + nvidia-container-toolkit.",
        "brev": "Requires the brev CLI (https://github.com/brevdev/brev-cli) and an active brev login.",
        "tao-sdk": "Requires Python 3.10+ and the tao-sdk package (pip install tao-sdk).",
    }
    if layer == "platform" and name in platform_compat:
        return platform_compat[name]

    # Look for container_image in references/skill_info.yaml
    skill_dir = os.path.dirname(skill_path)
    info_path = os.path.join(skill_dir, "references", "skill_info.yaml")
    has_container = False
    if os.path.isfile(info_path):
        try:
            with open(info_path) as f:
                info = yaml.safe_load(f) or {}
            has_container = bool(info.get("container_image"))
        except yaml.YAMLError:
            pass

    if has_container:
        # Containerized model/data/application
        if "ngc" in (info.get("container_image", "") if isinstance(info, dict) else ""):
            return "Requires docker + nvidia-container-toolkit + NGC API key."
        return "Requires docker + nvidia-container-toolkit."

    # Application/workflow without container_image — orchestrator
    if layer == "applications":
        return "Requires docker + nvidia-container-toolkit. Sub-skills declare additional requirements."

    # Pure agent-prompt-driven or local-script skills
    return "Standalone — no external runtime requirements."


# Skills that orchestrate workflows and persist state files — need `Write`.
# Most other skills only read inputs and shell out (Bash for docker / python / aws).
_ORCHESTRATOR_PATHS = {
    "skills/applications/tao-run-deft-aoi",
    "skills/applications/deft-vcn-aoi",
    "skills/applications/tao-run-automl",
    "skills/applications/tao-train-single-step",
}


def allowed_tools_for(skill_path: str) -> str:
    """Heuristic allowed-tools value for a skill.

    Default: 'Read Bash' (covers 90% of skills — Read inputs, Bash to run
    docker / python / CLI tools).

    Workflow orchestrators add 'Write' for state files / generated configs.
    """
    skill_dir = os.path.dirname(skill_path)
    if skill_dir in _ORCHESTRATOR_PATHS:
        return "Read Bash Write"
    return "Read Bash"


def tags_for_skill(skill_path: str, fm: dict) -> tuple[list, str]:
    """Return (tags_list, source) for a skill.

    Source priority:
      1. Existing `tags:` in SKILL.md frontmatter (idempotent — already migrated).
      2. `tags:` in references/skill_info.yaml (migrate it out).
      3. Heuristic — layer + skill name.

    Returns the source string for action logging:
      'frontmatter' | 'skill_info' | 'heuristic'
    """
    # 1. Already in frontmatter
    if isinstance(fm.get("tags"), list) and fm["tags"]:
        return fm["tags"], "frontmatter"

    # 2. From skill_info.yaml
    skill_dir = os.path.dirname(skill_path)
    info_path = os.path.join(skill_dir, "references", "skill_info.yaml")
    if os.path.isfile(info_path):
        try:
            with open(info_path) as f:
                info = yaml.safe_load(f) or {}
            if isinstance(info.get("tags"), list) and info["tags"]:
                return info["tags"], "skill_info"
        except yaml.YAMLError:
            pass

    # 3. Heuristic — layer + skill name
    parts = skill_path.split("/")
    if len(parts) >= 2:
        layer = parts[0].rstrip("s")  # 'models' → 'model'
        name = parts[1]  # 'visual-changenet'
        # Split skill name on hyphens, drop "deft-aoi-" prefix if present
        name_parts = name.split("-")
        if name_parts[:2] == ["deft", "aoi"]:
            name_parts = name_parts[2:]
        tags = [layer] + [p for p in name_parts if len(p) > 1]
        # Dedupe, preserve order
        seen = set()
        out = []
        for t in tags:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out, "heuristic"

    return [], "heuristic"


def remove_tags_from_skill_info(skill_path: str, apply: bool) -> bool:
    """Remove `tags:` key from references/skill_info.yaml. Returns True if removed."""
    skill_dir = os.path.dirname(skill_path)
    info_path = os.path.join(skill_dir, "references", "skill_info.yaml")
    if not os.path.isfile(info_path):
        return False
    try:
        with open(info_path) as f:
            info = yaml.safe_load(f) or {}
    except yaml.YAMLError:
        return False
    if "tags" not in info:
        return False
    if apply:
        del info["tags"]
        with open(info_path, "w") as f:
            yaml.safe_dump(info, f, sort_keys=False, default_flow_style=False, width=120, allow_unicode=True)
    return True


def parse_frontmatter(content: str):
    m = re.match(r"^---\n(.*?)\n---\n(.*)", content, re.DOTALL)
    if not m:
        return None, content
    fm_text = m.group(1)
    body = m.group(2)
    fm = yaml.safe_load(fm_text) or {}
    return fm, body


def render_frontmatter(fm: dict) -> str:
    """Render frontmatter as YAML, preserving common ordering."""
    # Conventional field order
    order = [
        "name",
        "description",
        "license",
        "compatibility",
        "metadata",
        "allowed-tools",
        "tags",
        "hooks",
    ]
    ordered = {}
    for key in order:
        if key in fm:
            ordered[key] = fm[key]
    # Anything else not in order
    for key, val in fm.items():
        if key not in ordered:
            ordered[key] = val
    return yaml.safe_dump(ordered, sort_keys=False, default_flow_style=False, width=120, allow_unicode=True).rstrip() + "\n"


def migrate_one(skill_md: str, apply: bool) -> str | None:
    """Migrate one SKILL.md. Returns description of action taken, or None if nothing to do."""
    with open(skill_md) as f:
        content = f.read()

    fm, body = parse_frontmatter(content)
    if fm is None:
        return f"SKIP: {skill_md} — no frontmatter (manual fix needed)"

    actions = []

    # license (required)
    if "license" not in fm:
        fm["license"] = "Apache-2.0"
        actions.append("+license")

    # compatibility
    if "compatibility" not in fm:
        fm["compatibility"] = compatibility_for(skill_md)
        actions.append("+compatibility")

    # metadata.{author, version}
    metadata = fm.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    if "author" not in metadata:
        author = git_authors(skill_md)
        if author:
            metadata["author"] = author
            actions.append("+metadata.author")
    if "version" not in metadata:
        metadata["version"] = "0.1"
        actions.append("+metadata.version")
    if metadata:
        fm["metadata"] = metadata

    # allowed-tools
    if "allowed-tools" not in fm:
        fm["allowed-tools"] = allowed_tools_for(skill_md)
        actions.append("+allowed-tools")

    # tags — migrate from skill_info.yaml or fall back to heuristic
    if "tags" not in fm:
        tags, source = tags_for_skill(skill_md, fm)
        if tags:
            fm["tags"] = tags
            actions.append(f"+tags(from-{source})")
    # Always remove tags from skill_info.yaml if present (single source of truth in SKILL.md)
    if remove_tags_from_skill_info(skill_md, apply):
        actions.append("-skill_info.tags")

    if not actions:
        return None  # already migrated

    if apply:
        new_fm = render_frontmatter(fm)
        new_content = f"---\n{new_fm}---\n{body}"
        with open(skill_md, "w") as f:
            f.write(new_content)

    return f"{skill_md}: {', '.join(actions)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply changes (otherwise dry-run)")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)

    skill_mds = []
    for path in Path(".").rglob("SKILL.md"):
        spath = str(path)
        if "templates/skill-skeleton" in spath:
            continue
        if ".git/" in spath:
            continue
        skill_mds.append(spath)

    if not skill_mds:
        print("No SKILL.md files found", file=sys.stderr)
        return 1

    actions_taken = []
    for skill_md in sorted(skill_mds):
        result = migrate_one(skill_md, apply=args.apply)
        if result:
            actions_taken.append(result)

    if not actions_taken:
        print(f"All {len(skill_mds)} skills already migrated. No action needed.")
        return 0

    print(f"{'Applied' if args.apply else 'Would apply'} {len(actions_taken)} migrations:")
    for line in actions_taken[:60]:
        print(f"  {line}")
    if len(actions_taken) > 60:
        print(f"  …and {len(actions_taken) - 60} more")

    if not args.apply:
        print("\nRe-run with --apply to write changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
