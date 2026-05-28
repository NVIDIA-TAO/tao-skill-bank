#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""List packaged TAO model capabilities from shipped model metadata.

AutoML enablement is model-level metadata (`automl_enabled: true` in
models/<network>/references/skill_info.yaml). Runnable AutoML support is then
gated by the exact packaged train dataclass schema for each model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_SKILL_BANK = Path(
    os.environ.get("TAO_SKILL_BANK_PATH", Path.home() / "tao-skills-external")
)
TRAIN_SCHEMA_REL = Path("schemas") / "train.schema.json"
SUPPORT_RULE = (
    "AutoML is enabled at model level; runnable AutoML also requires "
    "models/<network>/schemas/train.schema.json to be packaged and valid."
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
        "--scope",
        choices=("all", "automl"),
        default="all",
        help="List all model actions or AutoML-capable models.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="text",
        help="Output format.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_schema_manifest(skill_bank: Path) -> dict[str, Any]:
    """Load the packaged model action manifest."""
    return load_json(skill_bank.expanduser() / "models" / "schemas.manifest.json")


def load_automl_manifest(skill_bank: Path) -> dict[str, Any]:
    """Load the packaged AutoML compatibility manifest if present."""
    path = skill_bank.expanduser() / "models" / "automl_support.json"
    if not path.exists():
        return {"supported": [], "unsupported": []}
    return load_json(path)


def parse_bool(value: Any, default: bool = True) -> bool:
    """Parse a bool-ish metadata value."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_scalar(value: str) -> Any:
    """Parse the small scalar subset used by references/skill_info.yaml."""
    text = value.strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    return text.strip("\"'")


def load_skill_info(skill_bank: Path, model: str) -> dict[str, Any]:
    """Load top-level model metadata from references/skill_info.yaml."""
    path = skill_bank.expanduser() / "models" / model / "references" / "skill_info.yaml"
    if not path.exists():
        return {}

    data: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith((" ", "\t", "#")) or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = parse_scalar(value)
    return data


def skill_info_actions(skill_bank: Path, model: str) -> list[str]:
    """Read action names from a model skill_info.yaml without a YAML dependency."""
    path = skill_bank.expanduser() / "models" / model / "references" / "skill_info.yaml"
    if not path.exists():
        return []

    actions: list[str] = []
    in_actions = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("actions:"):
            in_actions = True
            continue
        if not in_actions:
            continue
        if line and not line.startswith((" ", "\t")):
            break
        if line.startswith("  ") and not line.startswith("    "):
            name = line.strip()
            if name.endswith(":"):
                actions.append(name[:-1])
    return sorted(actions)


def load_model_schema_manifest(skill_bank: Path, model: str) -> dict[str, Any]:
    """Load models/<model>/schemas/manifest.json when shipped."""
    path = skill_bank.expanduser() / "models" / model / "schemas" / "manifest.json"
    if not path.exists():
        return {}
    return load_json(path)


def train_schema_status(skill_bank: Path, model: str) -> tuple[bool, str]:
    """Return whether a model has a valid packaged train dataclass schema."""
    schema_path = skill_bank.expanduser() / "models" / model / TRAIN_SCHEMA_REL
    if not schema_path.exists():
        return False, f"{TRAIN_SCHEMA_REL.as_posix()} is not packaged"

    try:
        load_json(schema_path)
    except json.JSONDecodeError as exc:
        return False, f"{TRAIN_SCHEMA_REL.as_posix()} is invalid JSON: {exc.msg}"
    except OSError as exc:
        return False, f"{TRAIN_SCHEMA_REL.as_posix()} cannot be read: {exc}"
    except ValueError as exc:
        return False, str(exc)

    return True, f"{TRAIN_SCHEMA_REL.as_posix()} is packaged and valid"


def build_model_records(skill_bank: Path) -> list[dict[str, Any]]:
    """Build sorted model records from model-level metadata and schema manifests."""
    skill_bank = skill_bank.expanduser()
    global_manifest = load_schema_manifest(skill_bank)
    manifest_models = global_manifest.get("models", {})
    if not isinstance(manifest_models, dict):
        raise ValueError("models/schemas.manifest.json is missing a models object")

    models_root = skill_bank / "models"
    model_names = set(manifest_models)
    model_names.update(
        item.name
        for item in models_root.iterdir()
        if item.is_dir() and (item / "references" / "skill_info.yaml").exists()
    )

    records: list[dict[str, Any]] = []
    for model in sorted(model_names):
        metadata = manifest_models.get(model, {})
        if not isinstance(metadata, dict):
            metadata = {}
        skill_info = load_skill_info(skill_bank, model)
        schema_manifest = load_model_schema_manifest(skill_bank, model)
        has_train_schema, train_schema_reason = train_schema_status(skill_bank, model)
        schema_actions = schema_manifest.get("actions", {})
        actions = metadata.get("actions", [])
        if not isinstance(actions, list):
            actions = []
        if not actions and isinstance(schema_actions, dict):
            actions = sorted(schema_actions)
        if not actions:
            actions = skill_info_actions(skill_bank, model)
        failures = metadata.get("failures", {})
        if not isinstance(failures, dict):
            failures = {}
        if not failures:
            schema_failures = schema_manifest.get("failures", {})
            if isinstance(schema_failures, dict):
                failures = schema_failures

        records.append(
            {
                "model": model,
                "network_arch": (
                    skill_info.get("network_arch")
                    or metadata.get("network_arch")
                    or schema_manifest.get("network_arch")
                    or model
                ),
                "actions": actions,
                "failures": failures,
                "automl_enabled": parse_bool(
                    skill_info.get("automl_enabled", metadata.get("automl_enabled")),
                    default=True,
                ),
                "has_train_schema": has_train_schema,
                "train_schema": f"models/{model}/{TRAIN_SCHEMA_REL.as_posix()}",
                "train_schema_status": train_schema_reason,
            }
        )
    return records


def build_all_models(skill_bank: Path) -> dict[str, Any]:
    """Return packaged model/action support."""
    return {
        "schema_version": 1,
        "source": "models/schemas.manifest.json",
        "models": build_model_records(skill_bank),
    }


def build_automl_support(skill_bank: Path) -> dict[str, Any]:
    """Return model-level AutoML support, validated against packaged train schemas."""
    automl_manifest = load_automl_manifest(skill_bank)
    model_records = {item["model"]: item for item in build_model_records(skill_bank)}

    supported_manifest = {
        item.get("model"): item
        for item in automl_manifest.get("supported", [])
        if isinstance(item, dict) and item.get("model")
    }

    supported: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []

    for model in sorted(model_records):
        record = model_records[model]

        if not record["automl_enabled"]:
            unsupported.append(
                {
                    "model": model,
                    "network_arch": record["network_arch"],
                    "automl_enabled": False,
                    "reason": "automl_enabled is false in model metadata",
                    "train_schema_status": record["train_schema_status"],
                }
            )
            continue

        if record["has_train_schema"]:
            manifest_item = supported_manifest.get(model, {})
            item = {
                "model": model,
                "network_arch": record["network_arch"],
                "automl_enabled": True,
                "train_schema": TRAIN_SCHEMA_REL.as_posix(),
                "train_schema_status": record["train_schema_status"],
                "train_spec_template": (
                    manifest_item.get("train_spec_template")
                    or train_action_metadata(skill_bank, model).get("train_spec_template")
                ),
                "actions": record["actions"],
                "automl_default_parameters": (
                    train_action_metadata(skill_bank, model).get("automl_default_parameters")
                    or manifest_item.get("automl_default_parameters", [])
                ),
            }
            supported.append(item)
            continue

        unsupported.append(
            {
                "model": model,
                "network_arch": record["network_arch"],
                "automl_enabled": True,
                "reason": record["train_schema_status"],
                "train_schema_status": record["train_schema_status"],
            }
        )

    return {
        "schema_version": 1,
        "source": [
            "models/<network>/references/skill_info.yaml",
            "models/<network>/schemas/manifest.json",
            "models/<network>/schemas/train.schema.json",
        ],
        "support_rule": SUPPORT_RULE,
        "supported": supported,
        "unsupported": unsupported,
    }


def train_action_metadata(skill_bank: Path, model: str) -> dict[str, Any]:
    """Return AutoML parameter metadata for a model train action."""
    schema_manifest = load_model_schema_manifest(skill_bank, model)
    actions = schema_manifest.get("actions", {})
    train = actions.get("train") if isinstance(actions, dict) else None
    if isinstance(train, dict):
        return {
            "train_spec_template": train.get("spec_template"),
            "automl_default_parameters": train.get("automl_default_parameters", []),
        }

    train_schema = skill_bank.expanduser() / "models" / model / TRAIN_SCHEMA_REL
    if train_schema.exists():
        schema = load_json(train_schema)
        return {
            "train_spec_template": "references/spec_template_train.yaml",
            "automl_default_parameters": schema.get("automl_default_parameters", []),
        }
    return {"train_spec_template": None, "automl_default_parameters": []}


def action_text(actions: list[str]) -> str:
    """Format action names for compact text output."""
    return ", ".join(actions) if actions else "no packaged action schemas"


def format_all_models_text(data: dict[str, Any]) -> str:
    """Format packaged model/action support for a human."""
    lines = ["Packaged TAO models and action schemas:"]
    for item in data["models"]:
        lines.append(
            "- {model} ({network_arch}): {actions}; train schema: {schema}".format(
                model=item["model"],
                network_arch=item["network_arch"],
                actions=action_text(item["actions"]),
                schema="valid" if item["has_train_schema"] else item["train_schema_status"],
            )
        )
    return "\n".join(lines)


def format_automl_text(data: dict[str, Any]) -> str:
    """Format AutoML support for a human."""
    lines = [data["support_rule"], "", "Supported AutoML models:"]
    if data["supported"]:
        for item in data["supported"]:
            params = item.get("automl_default_parameters", [])
            params_text = ", ".join(params) if params else "schema-defined defaults"
            lines.append(
                f"- {item['model']} ({item['network_arch']}): automl_enabled=true; train schema valid; "
                f"AutoML parameters: {params_text}"
            )
    else:
        lines.append("- None")

    lines.append("")
    lines.append("Not supported:")
    if data["unsupported"]:
        lines.extend(
            f"- {item['model']} ({item['network_arch']}): "
            f"automl_enabled={str(item.get('automl_enabled', False)).lower()}; {item['reason']}"
            for item in data["unsupported"]
        )
    else:
        lines.append("- None")
    return "\n".join(lines)


def main() -> int:
    """Run the model listing helper."""
    args = parse_args()
    skill_bank = args.skill_bank.expanduser()
    data = build_automl_support(skill_bank) if args.scope == "automl" else build_all_models(skill_bank)

    if args.format == "json":
        print(json.dumps(data, indent=2, sort_keys=True))
    elif args.scope == "automl":
        print(format_automl_text(data))
    else:
        print(format_all_models_text(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
