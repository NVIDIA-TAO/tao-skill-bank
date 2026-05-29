#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""List supported TAO execution platforms from the packaged platform manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_SKILL_BANK = Path(
    os.environ.get("TAO_SKILL_BANK_PATH", Path.home() / "tao-skills-external")
)
MANIFEST_REL = Path("platform") / "platforms.manifest.json"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skill-bank",
        type=Path,
        default=DEFAULT_SKILL_BANK,
        help="Path to the packaged TAO skill bank.",
    )
    parser.add_argument(
        "--platform",
        help="Optional platform name or alias to resolve.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="text",
        help="Output format.",
    )
    return parser.parse_args()


def load_platform_manifest(skill_bank: Path) -> dict[str, Any]:
    """Load packaged platform support metadata."""
    manifest_path = skill_bank.expanduser() / MANIFEST_REL
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object")
    return data


def supported_platforms(skill_bank: Path) -> list[dict[str, Any]]:
    """Return supported platforms without scanning platform folders or docs."""
    manifest = load_platform_manifest(skill_bank)
    platforms = manifest.get("platforms", [])
    if not isinstance(platforms, list):
        raise ValueError("platform/platforms.manifest.json is missing a platforms list")
    return [item for item in platforms if isinstance(item, dict)]


def prompt_defaults(skill_bank: Path) -> dict[str, Any]:
    """Return workflow prompt defaults from the platform manifest."""
    manifest = load_platform_manifest(skill_bank)
    return {
        "long_running_enabled": manifest.get("default_long_running_enabled", True),
        "status_interval_minutes": manifest.get("default_status_interval_minutes", 5),
    }


def resolve_platform(skill_bank: Path, requested: str) -> dict[str, Any]:
    """Resolve a platform name or alias to its platform record."""
    normalized = requested.strip().lower()
    for platform in supported_platforms(skill_bank):
        names = [platform.get("name", "")]
        names.extend(platform.get("aliases", []))
        if normalized in {str(name).lower() for name in names}:
            return platform
    known = ", ".join(platform["name"] for platform in supported_platforms(skill_bank))
    raise ValueError(f"Unknown TAO platform '{requested}'. Supported platforms: {known}")


def enrich_credential(
    item: dict[str, Any],
    definitions: dict[str, Any],
) -> dict[str, Any]:
    """Merge centralized credential descriptions into a platform record."""
    name = item.get("name")
    merged = dict(definitions.get(name, {}))
    merged.update(item)
    return merged


def credentials_for_platform(skill_bank: Path, requested: str) -> dict[str, Any]:
    """Return only the credentials relevant to the selected platform."""
    manifest = load_platform_manifest(skill_bank)
    definitions = manifest.get("credential_definitions", {})
    platform = resolve_platform(skill_bank, requested)
    return {
        "platform": platform["name"],
        "display_name": platform.get("display_name", platform["name"]),
        "required_credentials": [
            enrich_credential(item, definitions)
            for item in platform.get("required_credentials", [])
        ],
        "credential_groups": platform.get("credential_groups", []),
        "optional_credentials": [
            enrich_credential(item, definitions)
            for item in platform.get("optional_credentials", [])
        ],
        "resource_defaults": platform.get("resource_defaults", {}),
        "storage": platform.get("storage", {}),
        "dataset_examples": platform.get("dataset_examples", []),
        "preflight_checks": platform.get("preflight_checks", []),
    }


def format_credential(item: dict[str, Any]) -> str:
    """Format a credential record."""
    text = item.get("name", "")
    if item.get("only_when"):
        text += f" ({item['only_when']})"
    details = []
    if item.get("description"):
        details.append(str(item["description"]))
    if item.get("how_to_get"):
        details.append(f"How to get it: {item['how_to_get']}")
    if details:
        text += " - " + " ".join(details)
    return text


def format_credential_group(item: dict[str, Any]) -> str:
    """Format a required one-of credential group."""
    choices = ", ".join(item.get("require_one_of", []))
    preferred = item.get("preferred")
    if preferred:
        text = f"{item.get('name', 'credential_group')}: {preferred} preferred"
    else:
        text = f"{item.get('name', 'credential_group')}: one of [{choices}]"
    details = []
    if item.get("description"):
        details.append(str(item["description"]))
    if item.get("how_to_get"):
        details.append(f"How to get it: {item['how_to_get']}")
    if details:
        text += " - " + " ".join(details)
    return text


def format_resource_value(value: Any) -> str:
    """Format resource defaults in a shell/config-friendly way."""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def format_platform_list_text(skill_bank: Path) -> str:
    """Format supported platforms for an initial workflow prompt."""
    defaults = prompt_defaults(skill_bank)
    lines = [
        "Supported TAO execution platforms:",
    ]
    for platform in supported_platforms(skill_bank):
        aliases = platform.get("aliases", [])
        alias_text = f" aliases: {', '.join(aliases)}" if aliases else ""
        lines.append(
            f"- {platform['name']} ({platform.get('display_name', platform['name'])}): "
            f"{platform.get('description', '')}{alias_text}"
        )
    lines.extend(
        [
            "",
            "Prompt defaults:",
            f"- long_running_enabled: {str(defaults['long_running_enabled']).lower()}",
            f"- status_interval_minutes: {defaults['status_interval_minutes']}",
        ]
    )
    return "\n".join(lines)


def format_platform_detail_text(skill_bank: Path, requested: str) -> str:
    """Format the selected platform's credentials and storage hints."""
    detail = credentials_for_platform(skill_bank, requested)
    required = detail["required_credentials"]
    groups = detail["credential_groups"]
    optional = detail["optional_credentials"]
    resource_defaults = detail["resource_defaults"]
    storage = detail["storage"]
    dataset_examples = detail["dataset_examples"]
    preflight_checks = detail["preflight_checks"]

    lines = [
        f"Platform: {detail['platform']} ({detail['display_name']})",
        "Required credentials:",
    ]
    if required:
        lines.extend(f"- {format_credential(item)}" for item in required)
    else:
        lines.append("- None")

    if groups:
        lines.append("Required credential groups:")
        lines.extend(f"- {format_credential_group(item)}" for item in groups)

    lines.append("Optional credentials/settings (do not request during initial intake unless the condition applies):")
    if optional:
        lines.extend(f"- {format_credential(item)}" for item in optional)
    else:
        lines.append("- None")

    if resource_defaults:
        lines.append("Resource defaults:")
        for key, value in resource_defaults.items():
            lines.append(f"- {key}: {format_resource_value(value)}")

    if storage:
        lines.extend(
            [
                "Storage:",
                f"- protocol: {storage.get('protocol', '')}",
                f"- URI format: {storage.get('uri_format', '')}",
            ]
        )
    if dataset_examples:
        lines.append("Dataset examples:")
        lines.extend(f"- {example}" for example in dataset_examples)
    if preflight_checks:
        lines.append("Preflight checks:")
        lines.extend(f"- {check}" for check in preflight_checks)
    return "\n".join(lines)


def main() -> int:
    """Run the platform support helper."""
    args = parse_args()
    skill_bank = args.skill_bank.expanduser()

    if args.platform:
        data = credentials_for_platform(skill_bank, args.platform)
        if args.format == "json":
            print(json.dumps(data, indent=2, sort_keys=True))
        else:
            print(format_platform_detail_text(skill_bank, args.platform))
        return 0

    data = {
        "schema_version": load_platform_manifest(skill_bank).get("schema_version", 1),
        "prompt_defaults": prompt_defaults(skill_bank),
        "platforms": supported_platforms(skill_bank),
    }
    if args.format == "json":
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(format_platform_list_text(skill_bank))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
