#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Maintenance tool: generate TAO Core dataclass schemas into model packages.

The skill bank treats these schemas as the source of truth for AutoML
parameter metadata: defaults, ranges, categorical options, option weights,
popular parameters, and the `automl_enabled` flag.
For each generated action schema, the script also writes
`references/spec_template_<action>.yaml` from the schema's top-level `default`
field.

This script is for skill-bank maintainers before packaging the plugin. The
plugin workflow must not require a `tao-core` checkout at runtime.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml


CORE_MODULE_ALIASES = {
    "depth_net_mono": "depth_net",
    "depth_net_stereo": "depth_net",
    "visual-changenet": "visual_changenet",
    "visual_changenet": "visual_changenet",
}

ACTION_ALIASES = {
    "segment_train": "train",
    "segment_evaluate": "evaluate",
    "segment_inference": "inference",
}

COMMON_ACTION_KEYS = {
    "calibration_tensorfile",
    "dataset_convert",
    "deploy",
    "distill",
    "evaluate",
    "export",
    "generate",
    "gen_trt_engine",
    "inference",
    "prune",
    "quantize",
    "retrain",
    "segment_evaluate",
    "segment_inference",
    "segment_train",
    "train",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skill-bank",
        type=Path,
        default=Path.home() / "tao-skills-external",
        help="Path to the TAO skill bank.",
    )
    parser.add_argument(
        "--tao-core",
        type=Path,
        default=Path.home() / "tao-core",
        help="Path to the tao-core repository.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Generate only this skill-bank model folder. Can be passed more than once.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing *.schema.json files before regenerating selected models.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Read JSON from path."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write a YAML spec template."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, default_flow_style=False, sort_keys=False)


def candidate_core_modules(network_arch: str, model_name: str) -> list[str]:
    """Return candidate TAO Core config module names for a skill-bank model."""
    candidates = [
        CORE_MODULE_ALIASES.get(network_arch, network_arch),
        CORE_MODULE_ALIASES.get(model_name, model_name),
        network_arch.replace("-", "_"),
        model_name.replace("-", "_"),
    ]
    if network_arch.startswith("depth_net_") or model_name.startswith("depth-net-"):
        candidates.append("depth_net")
    return list(dict.fromkeys(candidates))


def import_config_module(core_module: str, action: str):
    """Import the TAO Core config module for a network/action."""
    if core_module == "cosmos-rl":
        return importlib.import_module(f"nvidia_tao_core.config.{core_module}.{action}")
    return importlib.import_module(f"nvidia_tao_core.config.{core_module}.default_config")


def instantiate_experiment(core_module: str, module, action: str):
    """Instantiate the root dataclass expected by TAO Core schema conversion."""
    if core_module == "bevfusion" and action == "dataset_convert":
        return module.BEVFusionDataConvertExpConfig()
    if core_module == "stylegan_xl" and action == "dataset_convert":
        dataset_module = importlib.import_module("nvidia_tao_core.config.stylegan_xl.dataset")
        return dataset_module.DataConvertExpConfig()
    if core_module == "clip":
        return module.CLIPExperimentConfig()
    return module.ExperimentConfig()


def get_valid_action_keys(skill_config: dict[str, Any], core_module: str) -> set[str]:
    """Build the action-key set used to filter action-specific schemas."""
    actions = set(COMMON_ACTION_KEYS)
    actions.update((skill_config.get("actions") or {}).keys())

    try:
        from nvidia_tao_core.microservices import enum_constants

        actions.update(enum_constants._get_valid_config_json_param_for_network(core_module, "actions"))
        network_arch = skill_config.get("network_arch")
        if network_arch:
            actions.update(enum_constants._get_valid_config_json_param_for_network(network_arch, "actions"))
    except Exception as exc:  # pragma: no cover - defensive around local tao-core variants.
        logging.debug("Could not load TAO Core valid actions for %s: %s", core_module, exc)

    return actions


def filter_schema(schema: dict[str, Any], valid_actions: set[str], current_action: str) -> dict[str, Any]:
    """Keep train/current action plus non-action top-level keys.

    This mirrors TAO Core's public generator behavior, with a larger action-key
    vocabulary so alias packages such as visual-changenet and depth-net do not
    accidentally keep unrelated actions as "non-action" keys.
    """
    allowed_keys = {"train", "distill", "quantize", current_action}
    properties = schema.get("properties", {})
    allowed_keys.update(key for key in properties if key not in valid_actions)
    schema["properties"] = {key: value for key, value in properties.items() if key in allowed_keys}
    schema["default"] = {
        key: value for key, value in schema.get("default", {}).items() if key in allowed_keys
    }
    return schema


def generate_schema_for_action(
    skill_config: dict[str, Any],
    model_name: str,
    action: str,
) -> tuple[dict[str, Any], str, str]:
    """Generate a schema for one skill-bank model/action."""
    from nvidia_tao_core.api_utils import dataclass2json_converter

    network_arch = skill_config.get("network_arch", model_name)
    schema_action = ACTION_ALIASES.get(action, action)
    errors = []

    for core_module in candidate_core_modules(network_arch, model_name):
        try:
            module = import_config_module(core_module, schema_action)
            exp_config = instantiate_experiment(core_module, module, schema_action)
            json_with_meta = dataclass2json_converter.dataclass_to_json(exp_config)
            schema = dataclass2json_converter.create_json_schema(json_with_meta)
            schema = filter_schema(schema, get_valid_action_keys(skill_config, core_module), schema_action)
            schema["x_tao_schema"] = {
                "schema_version": 1,
                "model": model_name,
                "network_arch": network_arch,
                "action": action,
                "schema_action": schema_action,
                "core_module": core_module,
                "source": "tao-core dataclass config",
            }
            return schema, core_module, schema_action
        except Exception as exc:  # noqa: BLE001 - record all candidates for manifest diagnostics.
            errors.append(f"{core_module}: {type(exc).__name__}: {exc}")

    raise RuntimeError("; ".join(errors))


def generate_for_model(model_dir: Path, clean: bool) -> dict[str, Any]:
    """Generate all declared action schemas for one model directory."""
    config_path = model_dir / "config.json"
    skill_config = load_json(config_path)
    actions = sorted((skill_config.get("actions") or {}).keys())
    schema_dir = model_dir / "schemas"
    references_dir = model_dir / "references"
    if clean and schema_dir.exists():
        for existing in schema_dir.glob("*.schema.json"):
            existing.unlink()
    if clean and references_dir.exists():
        for action in actions:
            existing = references_dir / f"spec_template_{action}.yaml"
            if existing.exists():
                existing.unlink()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "model": model_dir.name,
        "network_arch": skill_config.get("network_arch", model_dir.name),
        "automl_enabled": True,
        "actions": {},
        "failures": {},
    }

    for action in actions:
        try:
            schema, core_module, schema_action = generate_schema_for_action(skill_config, model_dir.name, action)
            schema_path = schema_dir / f"{action}.schema.json"
            spec_template_path = references_dir / f"spec_template_{action}.yaml"
            dump_json(schema_path, schema)
            dump_yaml(spec_template_path, schema.get("default", {}))
            manifest["actions"][action] = {
                "path": f"schemas/{action}.schema.json",
                "spec_template": f"references/spec_template_{action}.yaml",
                "core_module": core_module,
                "schema_action": schema_action,
                "automl_default_parameters": sorted(schema.get("automl_default_parameters", [])),
                "automl_disabled_parameters": sorted(schema.get("automl_disabled_parameters", [])),
                "popular": schema.get("popular", {}),
            }
        except Exception as exc:  # noqa: BLE001 - keep generating other models/actions.
            manifest["failures"][action] = str(exc)

    dump_json(schema_dir / "manifest.json", manifest)
    return manifest


def build_support_summary(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the quick AutoML support summary from generated manifests."""
    supported = []
    unsupported = []
    for manifest in manifests:
        model = manifest["model"]
        actions = manifest.get("actions", {})
        failures = manifest.get("failures", {})
        if "train" in actions:
            supported.append(
                {
                    "model": model,
                    "network_arch": manifest.get("network_arch", model),
                    "automl_enabled": True,
                    "train_schema": actions["train"]["path"],
                    "train_spec_template": actions["train"].get("spec_template"),
                    "automl_default_parameters": actions["train"].get("automl_default_parameters", []),
                }
            )
        else:
            reason = failures.get("train") or "schemas/train.schema.json is not packaged"
            unsupported.append(
                {
                    "model": model,
                    "network_arch": manifest.get("network_arch", model),
                    "automl_enabled": True,
                    "reason": reason,
                }
            )

    return {
        "schema_version": 1,
        "support_rule": "AutoML is enabled at model level; runnable AutoML also requires models/<network>/schemas/train.schema.json to be packaged and valid.",
        "supported": sorted(supported, key=lambda item: item["model"]),
        "unsupported": sorted(unsupported, key=lambda item: item["model"]),
    }


def main() -> int:
    """Generate schemas for selected skill-bank models."""
    args = parse_args()
    tao_core = args.tao_core.expanduser().resolve()
    skill_bank = args.skill_bank.expanduser().resolve()
    sys.path.insert(0, str(tao_core))

    models_root = skill_bank / "models"
    selected = set(args.model)
    manifests = []

    for model_dir in sorted(path for path in models_root.iterdir() if path.is_dir()):
        if selected and model_dir.name not in selected:
            continue
        if not (model_dir / "config.json").exists():
            continue
        manifest = generate_for_model(model_dir, args.clean)
        manifests.append(manifest)

    summary = {
        "schema_version": 1,
        "models": {
            manifest["model"]: {
                "network_arch": manifest["network_arch"],
                "automl_enabled": manifest.get("automl_enabled", True),
                "actions": sorted(manifest["actions"].keys()),
                "failures": manifest["failures"],
            }
            for manifest in manifests
        },
    }
    dump_json(skill_bank / "models" / "schemas.manifest.json", summary)
    dump_json(skill_bank / "models" / "automl_support.json", build_support_summary(manifests))

    generated_actions = sum(len(manifest["actions"]) for manifest in manifests)
    failed_actions = sum(len(manifest["failures"]) for manifest in manifests)
    print(
        f"Generated {generated_actions} action schema(s) across {len(manifests)} model(s); "
        f"{failed_actions} action(s) failed."
    )
    if failed_actions:
        print(f"See {(skill_bank / 'models' / 'schemas.manifest.json')}")
    return 0 if generated_actions else 1


if __name__ == "__main__":
    raise SystemExit(main())
