#!/usr/bin/env python3
"""List packaged TAO model capabilities from shipped manifests.

This helper intentionally reads packaged metadata instead of discovering models
by walking the model directories. AutoML eligibility is then gated by the exact
packaged train dataclass schema for each model.
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
    "AutoML is supported only when models/<network>/schemas/train.schema.json "
    "is packaged and valid."
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
    """Load the packaged AutoML support manifest."""
    return load_json(skill_bank.expanduser() / "models" / "automl_support.json")


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
    """Build sorted model records from models/schemas.manifest.json."""
    manifest = load_schema_manifest(skill_bank)
    models = manifest.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("models/schemas.manifest.json is missing a models object")

    records: list[dict[str, Any]] = []
    for model, metadata in sorted(models.items()):
        if not isinstance(metadata, dict):
            metadata = {}
        has_train_schema, train_schema_reason = train_schema_status(skill_bank, model)
        actions = metadata.get("actions", [])
        if not isinstance(actions, list):
            actions = []
        failures = metadata.get("failures", {})
        if not isinstance(failures, dict):
            failures = {}

        records.append(
            {
                "model": model,
                "network_arch": metadata.get("network_arch", model),
                "actions": actions,
                "failures": failures,
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
    """Return AutoML support, validated against packaged train schemas."""
    automl_manifest = load_automl_manifest(skill_bank)
    model_records = {item["model"]: item for item in build_model_records(skill_bank)}

    supported_manifest = {
        item.get("model"): item
        for item in automl_manifest.get("supported", [])
        if isinstance(item, dict) and item.get("model")
    }
    unsupported_manifest = {
        item.get("model"): item
        for item in automl_manifest.get("unsupported", [])
        if isinstance(item, dict) and item.get("model")
    }

    supported: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []

    for model in sorted(model_records):
        record = model_records[model]
        manifest_item = supported_manifest.get(model)

        if manifest_item and record["has_train_schema"]:
            item = dict(manifest_item)
            item.setdefault("model", model)
            item.setdefault("network_arch", record["network_arch"])
            item.setdefault("train_schema", TRAIN_SCHEMA_REL.as_posix())
            item["train_schema_status"] = record["train_schema_status"]
            item["actions"] = record["actions"]
            supported.append(item)
            continue

        reason = record["train_schema_status"]
        if manifest_item and not record["has_train_schema"]:
            reason = record["train_schema_status"]
        elif model in unsupported_manifest:
            reason = unsupported_manifest[model].get("reason", reason)
        elif not manifest_item:
            reason = "not listed as AutoML-supported in models/automl_support.json"

        unsupported.append(
            {
                "model": model,
                "network_arch": record["network_arch"],
                "reason": reason,
                "train_schema_status": record["train_schema_status"],
            }
        )

    for model, manifest_item in sorted(supported_manifest.items()):
        if model in model_records:
            continue
        unsupported.append(
            {
                "model": model,
                "network_arch": manifest_item.get("network_arch", model),
                "reason": "listed in models/automl_support.json but missing from schemas.manifest.json",
                "train_schema_status": f"{TRAIN_SCHEMA_REL.as_posix()} was not checked",
            }
        )

    return {
        "schema_version": 1,
        "source": [
            "models/schemas.manifest.json",
            "models/automl_support.json",
        ],
        "support_rule": SUPPORT_RULE,
        "supported": supported,
        "unsupported": unsupported,
    }


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
                f"- {item['model']} ({item['network_arch']}): train schema valid; "
                f"AutoML parameters: {params_text}"
            )
    else:
        lines.append("- None")

    lines.append("")
    lines.append("Not supported:")
    if data["unsupported"]:
        lines.extend(
            f"- {item['model']} ({item['network_arch']}): {item['reason']}"
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
