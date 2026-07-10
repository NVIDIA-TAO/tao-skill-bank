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

COSMOS_METROPOLIS_SYSTEM_PROMPT_OPTIONS = [
    (
        "You are a helpful assistant that can answer questions about a "
        "street-view CCTV footage. The vehicles that need attention are "
        "marked with bounding boxes and IDs."
    ),
    (
        "You are a careful video event verification assistant for traffic "
        "and security CCTV. Answer only from visible evidence, track object "
        "IDs and bounding boxes, and state yes/no when the question asks "
        "whether an event occurred."
    ),
    (
        "You are a Metropolis VLM assistant for surveillance video QA. Focus "
        "on temporal order, object interactions, near misses, direction of "
        "travel, and safety-relevant evidence. Prefer concise factual answers."
    ),
    (
        "You are an event-verification assistant. Inspect the full clip before "
        "answering, compare the question with observed actions, ignore "
        "unsupported assumptions, and return the most specific answer allowed "
        "by the evidence."
    ),
]

COSMOS_EVALUATE_AUTOML_DEFAULT_PARAMETERS = [
    "dataset.system_prompt",
    "vision.nframes",
    "generation.max_tokens",
    "generation.temperature",
    "generation.repetition_penalty",
    "generation.presence_penalty",
    "generation.frequency_penalty",
]


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


def unwrap_cosmos_non_train_action_schema(schema: dict[str, Any], action: str) -> dict[str, Any]:
    """Keep Cosmos non-train schemas in the flat TOML shape consumed by runtime."""
    properties = schema.get("properties", {})
    defaults = schema.get("default", {})
    action_properties = properties.get(action)
    action_defaults = defaults.get(action)
    if not isinstance(action_properties, dict) or not isinstance(action_defaults, dict):
        return schema

    nested_properties = action_properties.get("properties")
    if not isinstance(nested_properties, dict):
        return schema

    properties = {key: value for key, value in properties.items() if key != action}
    defaults = {key: value for key, value in defaults.items() if key != action}
    properties.update(nested_properties)
    defaults.update(action_defaults)
    schema["properties"] = properties
    schema["default"] = defaults
    return schema


def apply_cosmos_evaluate_autoprompt_metadata(schema: dict[str, Any]) -> dict[str, Any]:
    """Add bounded Auto-Prompter-style search metadata for Cosmos evaluate."""
    schema["automl_default_parameters"] = list(COSMOS_EVALUATE_AUTOML_DEFAULT_PARAMETERS)

    properties = schema.setdefault("properties", {})
    dataset = properties.get("dataset", {}).get("properties", {})
    vision = properties.get("vision", {}).get("properties", {})
    generation = properties.get("generation", {}).get("properties", {})

    if "system_prompt" in dataset:
        dataset["system_prompt"].update(
            {
                "automl_enabled": True,
                "description": (
                    "System prompt for the evaluation tasks. AutoML treats this as a "
                    "bounded prompt-candidate search for zero-shot Metropolis/VSS "
                    "evaluation."
                ),
                "enum": list(COSMOS_METROPOLIS_SYSTEM_PROMPT_OPTIONS),
                "option_weights": [0.4, 0.25, 0.2, 0.15],
                "type": "categorical",
            }
        )
    if "nframes" in vision:
        vision["nframes"].update(
            {
                "automl_enabled": True,
                "enum": [4, 8],
                "type": "ordered_int",
            }
        )
    if "max_tokens" in generation:
        generation["max_tokens"].update(
            {
                "automl_enabled": True,
                "enum": [256, 512, 1024, 2048],
                "type": "ordered_int",
            }
        )

    ranges = {
        "temperature": (0.0, 0.4),
        "repetition_penalty": (0.8, 1.2),
        "presence_penalty": (-0.5, 0.5),
        "frequency_penalty": (-0.5, 0.5),
    }
    for parameter, (minimum, maximum) in ranges.items():
        if parameter in generation:
            generation[parameter].update(
                {
                    "automl_enabled": True,
                    "minimum": minimum,
                    "maximum": maximum,
                }
            )

    return schema


def generate_schema_for_action(
    skill_config: dict[str, Any],
    model_name: str,
    action: str,
) -> tuple[dict[str, Any], str, str]:
    """Generate a schema for one skill-bank model/action."""
    from nvidia_tao_core.api_utils import dataclass2json_converter

    network_arch = skill_config.get("network_arch", model_name)
    if network_arch == "sparse4d" and action == "dataset_convert":
        module = importlib.import_module("nvidia_tao_ds.config.annotations.default_config")
        exp_config = module.ExperimentConfig()
        json_with_meta = dataclass2json_converter.dataclass_to_json(exp_config)
        schema = dataclass2json_converter.create_json_schema(json_with_meta)
        schema["default"]["data"]["input_format"] = "AICITY"
        schema["default"]["data"]["output_format"] = "OVPKL"
        schema["default"]["aicity"]["root"] = "???"
        schema["properties"]["data"]["default"]["input_format"] = "AICITY"
        schema["properties"]["data"]["default"]["output_format"] = "OVPKL"
        schema["properties"]["data"]["properties"]["input_format"]["default"] = "AICITY"
        schema["properties"]["data"]["properties"]["output_format"]["default"] = "OVPKL"
        schema["properties"]["aicity"]["default"]["root"] = "???"
        schema["properties"]["aicity"]["properties"]["root"]["default"] = "???"
        schema["x_tao_schema"] = {
            "schema_version": 1,
            "model": model_name,
            "network_arch": network_arch,
            "action": action,
            "schema_action": "convert",
            "core_module": "annotations",
            "source": "tao-dataservices annotations dataclass config",
        }
        return schema, "annotations", "convert"

    schema_action = ACTION_ALIASES.get(action, action)
    errors = []

    for core_module in candidate_core_modules(network_arch, model_name):
        try:
            module = import_config_module(core_module, schema_action)
            exp_config = instantiate_experiment(core_module, module, schema_action)
            json_with_meta = dataclass2json_converter.dataclass_to_json(exp_config)
            schema = dataclass2json_converter.create_json_schema(json_with_meta)
            schema = filter_schema(schema, get_valid_action_keys(skill_config, core_module), schema_action)
            if core_module == "cosmos-rl" and schema_action in {"evaluate", "inference"}:
                schema = unwrap_cosmos_non_train_action_schema(schema, schema_action)
            if core_module == "cosmos-rl" and schema_action == "evaluate":
                schema = apply_cosmos_evaluate_autoprompt_metadata(schema)
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
        supported_actions = []
        for action, metadata in sorted(actions.items()):
            supported_actions.append(
                {
                    "action": action,
                    "schema": metadata["path"],
                    "schema_status": f"{metadata['path']} is packaged and valid",
                    "spec_template": metadata.get("spec_template"),
                    "automl_default_parameters": metadata.get("automl_default_parameters", []),
                }
            )

        if supported_actions:
            train_action = actions.get("train", {})
            supported.append(
                {
                    "model": model,
                    "network_arch": manifest.get("network_arch", model),
                    "automl_enabled": True,
                    "train_schema": train_action.get("path", "schemas/train.schema.json"),
                    "train_spec_template": train_action.get("spec_template"),
                    "automl_default_parameters": train_action.get("automl_default_parameters", []),
                    "supported_actions": supported_actions,
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
        "support_rule": "AutoML is enabled at model level; runnable AutoML for an action also requires skills/models/<network>/schemas/<action>.schema.json to be packaged and valid.",
        "supported": sorted(supported, key=lambda item: item["model"]),
        "unsupported": sorted(unsupported, key=lambda item: item["model"]),
    }


def main() -> int:
    """Generate schemas for selected skill-bank models."""
    args = parse_args()
    tao_core = args.tao_core.expanduser().resolve()
    skill_bank = args.skill_bank.expanduser().resolve()
    sys.path.insert(0, str(tao_core))

    models_root = skill_bank / "skills" / "models"
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
    dump_json(skill_bank / "skills" / "models" / "schemas.manifest.json", summary)
    dump_json(skill_bank / "skills" / "models" / "automl_support.json", build_support_summary(manifests))

    generated_actions = sum(len(manifest["actions"]) for manifest in manifests)
    failed_actions = sum(len(manifest["failures"]) for manifest in manifests)
    print(
        f"Generated {generated_actions} action schema(s) across {len(manifests)} model(s); "
        f"{failed_actions} action(s) failed."
    )
    if failed_actions:
        print(f"See {(skill_bank / 'skills' / 'models' / 'schemas.manifest.json')}")
    return 0 if generated_actions else 1


if __name__ == "__main__":
    raise SystemExit(main())
