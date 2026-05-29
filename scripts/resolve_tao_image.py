#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the default TAO container image for a model action.

The helper reads packaged model metadata instead of relying on hand-written
prompts. Launch workflows should show this image to the user and accept an
explicit override before generating runner artifacts or submitting jobs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SKILL_BANK = Path(
    os.environ.get("TAO_SKILL_BANK_PATH", Path.home() / "tao-skills-external")
)


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
        "--model",
        "--network",
        dest="model",
        required=True,
        help="Packaged model/network name, for example cosmos-rl.",
    )
    parser.add_argument(
        "--action",
        default="train",
        help="Model action to resolve, for example train, evaluate, inference, or export.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="text",
        help="Output format.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML object from disk."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML object")
    return data


def resolve_image_key(skill_bank: Path, image: str) -> tuple[str, str]:
    """Resolve a versions.yaml image key to a URI when possible."""
    image = image.strip()
    if "/" in image or ":" in image:
        return image, "absolute"

    versions_path = skill_bank.expanduser() / "versions.yaml"
    versions = load_yaml(versions_path)
    cursor: Any = versions.get("images", {})
    for part in image.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return image, "unresolved_key"
        cursor = cursor[part]
    if not isinstance(cursor, str) or not cursor.strip():
        return image, "unresolved_key"
    return cursor.strip(), "versions.yaml"


def resolve_image(skill_bank: Path, model: str, action: str) -> dict[str, Any]:
    """Resolve action-level image first, then model-level image."""
    metadata_path = (
        skill_bank.expanduser() / "models" / model / "references" / "skill_info.yaml"
    )
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")

    skill_info = load_yaml(metadata_path)
    actions = skill_info.get("actions", {})
    if not isinstance(actions, dict):
        actions = {}

    action_config = actions.get(action)
    if action_config is None:
        available = ", ".join(sorted(actions)) if actions else "none"
        raise ValueError(
            f"Action '{action}' is not packaged for model '{model}'. "
            f"Available actions: {available}"
        )
    if not isinstance(action_config, dict):
        raise ValueError(
            f"models/{model}/references/skill_info.yaml actions.{action} must be an object"
        )

    candidates = [
        ("action.container_image", action_config.get("container_image")),
        ("action.image", action_config.get("image")),
        ("model.container_image", skill_info.get("container_image")),
        ("model.image", skill_info.get("image")),
    ]
    for source, image in candidates:
        if isinstance(image, str) and image.strip():
            resolved_image, resolved_from = resolve_image_key(skill_bank, image)
            return {
                "schema_version": 2,
                "model": model,
                "network_arch": skill_info.get("network_arch", model),
                "action": action,
                "image": resolved_image,
                "declared_image": image.strip(),
                "resolved_from": resolved_from,
                "source": source,
                "metadata_path": str(metadata_path),
                "confirmation_required": True,
                "override_key": "image",
            }

    raise ValueError(
        f"No container image found for model '{model}' action '{action}' in {metadata_path}"
    )


def format_text(data: dict[str, Any]) -> str:
    """Format resolved image metadata for launch prompts."""
    return "\n".join(
        [
            "TAO container image resolution:",
            f"- model: {data['model']} ({data['network_arch']})",
            f"- action: {data['action']}",
            f"- default image: {data['image']}",
            f"- declared image: {data['declared_image']}",
            f"- source: {data['source']} in {data['metadata_path']}",
            f"- resolved from: {data['resolved_from']}",
            "- confirmation: ask the user to use this image or provide image=<override> before launch",
        ]
    )


def main() -> int:
    """Run the image resolver."""
    args = parse_args()
    data = resolve_image(args.skill_bank, args.model, args.action)
    if args.format == "json":
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(format_text(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
