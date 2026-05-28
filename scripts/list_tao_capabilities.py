#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize TAO Skill Bank capabilities for plugin capability answers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from list_tao_models import build_all_models, build_automl_support
from list_tao_platforms import prompt_defaults, supported_platforms


DEFAULT_SKILL_BANK = Path(
    os.environ.get("TAO_SKILL_BANK_PATH", Path.home() / "tao-skills-external")
)
FLOW_ACTIONS = ("train", "evaluate", "inference", "export", "gen_trt_engine")


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
        "--format",
        choices=("json", "text"),
        default="text",
        help="Output format.",
    )
    return parser.parse_args()


def parse_frontmatter(path: Path) -> dict[str, str]:
    """Parse the simple YAML frontmatter shape used by bundled skills."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    frontmatter: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        frontmatter.append(line)

    data: dict[str, str] = {}
    index = 0
    while index < len(frontmatter):
        line = frontmatter[index]
        if not line.strip() or line.startswith(" "):
            index += 1
            continue
        if ":" not in line:
            index += 1
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value in {">", ">-", "|", "|-"}:
            block: list[str] = []
            index += 1
            while index < len(frontmatter):
                next_line = frontmatter[index]
                if next_line and not next_line.startswith(" "):
                    break
                if next_line.strip():
                    block.append(next_line.strip())
                index += 1
            data[key] = " ".join(block)
            continue

        data[key] = value.strip("\"'")
        index += 1

    return data


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def clean_description(text: str) -> str:
    """Normalize markdown/frontmatter prose for compact capability output."""
    description = " ".join(text.split()).strip()
    for marker in (
        " Use when ",
        " Use this skill ",
        " Use this skill whenever ",
        " Trigger when ",
        " Also trigger ",
    ):
        if marker in description:
            description = description.split(marker, 1)[0].rstrip()

    if description and description[-1] not in ".!?":
        description += "."
    return description


def skill_capability_record(skill_bank: Path, skill_md: Path) -> dict[str, Any]:
    """Build a capability record from a skill doc and optional config.json."""
    skill_bank = skill_bank.expanduser()
    skill_dir = skill_md.parent
    metadata = parse_frontmatter(skill_md)
    config_path = skill_dir / "config.json"
    config: dict[str, Any] = {}
    if config_path.exists():
        config = load_json_object(config_path)

    actions = config.get("actions", [])
    if isinstance(actions, dict):
        action_names = sorted(actions)
    elif isinstance(actions, list):
        action_names = sorted(str(action) for action in actions)
    else:
        action_names = []

    tags = config.get("tags", [])
    if not isinstance(tags, list):
        tags = []

    name = str(config.get("name") or metadata.get("name") or skill_dir.name)
    description = clean_description(
        str(
            config.get("description")
            or metadata.get("description")
            or f"Use the {skill_dir.name} skill."
        )
    )

    return {
        "name": name,
        "path": str(skill_md.relative_to(skill_bank)),
        "actions": action_names,
        "tags": [str(tag) for tag in tags],
        "capability": description,
    }


def application_capabilities(skill_bank: Path) -> list[dict[str, Any]]:
    """Read top-level application skills and turn them into capability records."""
    application_root = skill_bank.expanduser() / "applications"
    records: list[dict[str, Any]] = []

    for skill_md in sorted(application_root.glob("*/SKILL.md")):
        record = skill_capability_record(skill_bank, skill_md)
        record["files"] = sorted(
            item.name for item in skill_md.parent.iterdir() if item.is_file()
        )
        records.append(record)
    return records


def data_capabilities(skill_bank: Path) -> list[dict[str, Any]]:
    """Read data skills and turn them into capability records."""
    data_root = skill_bank.expanduser() / "data"
    records: list[dict[str, Any]] = []

    for skill_md in sorted(data_root.glob("*/SKILL.md")):
        records.append(skill_capability_record(skill_bank, skill_md))
    return records


def build_capabilities(skill_bank: Path) -> dict[str, Any]:
    """Build a capability summary from application skills and model manifests."""
    all_models = build_all_models(skill_bank)
    automl = build_automl_support(skill_bank)
    model_records = all_models["models"]

    training_models = [
        item["model"] for item in model_records if "train" in item.get("actions", [])
    ]
    full_finetune_models = [
        item["model"]
        for item in model_records
        if all(action in item.get("actions", []) for action in FLOW_ACTIONS)
    ]

    return {
        "schema_version": 1,
        "applications": application_capabilities(skill_bank),
        "data_workflows": data_capabilities(skill_bank),
        "platforms": {
            "source": "platform/platforms.manifest.json",
            "prompt_defaults": prompt_defaults(skill_bank),
            "supported": supported_platforms(skill_bank),
        },
        "model_workflows": {
            "source": "models/schemas.manifest.json",
            "actions": list(FLOW_ACTIONS),
            "training_capable_models": training_models,
            "full_train_eval_infer_export_trt_models": full_finetune_models,
            "models": model_records,
        },
        "automl": automl,
    }


def csv(items: list[str]) -> str:
    """Format compact comma-separated text."""
    return ", ".join(items) if items else "none"


def action_set(model_records: list[dict[str, Any]]) -> list[str]:
    """Return all action names declared by packaged model schemas."""
    actions: set[str] = set()
    for record in model_records:
        actions.update(str(action) for action in record.get("actions", []))
    return sorted(actions)


def format_capabilities_text(data: dict[str, Any]) -> str:
    """Format the capability summary for a plugin intro/capability answer."""
    model_workflows = data["model_workflows"]
    models = model_workflows["models"]
    platforms = data["platforms"]
    defaults = platforms["prompt_defaults"]
    automl = data["automl"]

    all_actions = action_set(models)
    full_flow_models = model_workflows["full_train_eval_infer_export_trt_models"]
    training_models = model_workflows["training_capable_models"]
    automl_models = [item["model"] for item in automl["supported"]]

    lines = [
        "Hi, I'm the TAO Skill Bank.",
        "",
        "I turn a TAO model goal, dataset, KPI, and compute target into runnable "
        "NVIDIA TAO workflows for training, AutoML/HPO, evaluation, inference, "
        "export, TensorRT engine generation, data improvement, and deployment "
        "handoff.",
        "",
        "Application workflows I can drive:",
    ]
    for app in data["applications"]:
        lines.append(f"- {app['name']}: {app['capability']}")

    lines.extend(
        [
            "",
            "Dataset and data-improvement skills I can use:",
        ]
    )
    for data_skill in data["data_workflows"]:
        action_text = (
            f" Actions: {csv(data_skill['actions'])}."
            if data_skill["actions"]
            else ""
        )
        lines.append(
            f"- {data_skill['name']}: {data_skill['capability']}{action_text}"
        )

    lines.extend(
        [
            "",
            "Model families and action coverage:",
            "- Packaged model schemas declare these action types: " + csv(all_actions),
            "- Full train/evaluate/inference/export/TensorRT flow: "
            + csv(full_flow_models),
            "- Training-capable model families: " + csv(training_models),
            "",
            "Per-model action support:",
        ]
    )
    for model in models:
        actions = csv(model["actions"])
        lines.append(f"- {model['model']}: {actions}")

    lines.extend(
        [
            "",
            "AutoML/HPO support:",
            "- AutoML is enabled from model metadata, so workflows that train a "
            "model should route through AutoMLRunner unless their run settings "
            "set automl_policy=off or the user explicitly asks for a plain "
            "single run.",
            "- Runnable AutoML still requires a valid packaged train schema. "
            f"Runnable models: {csv(automl_models)}",
            f"- Rule: {automl['support_rule']}",
        ]
    )
    if automl["unsupported"]:
        lines.append(
            "- AutoML-enabled models waiting on train schema packaging: "
            + "; ".join(
                f"{item['model']} ({item['reason']})" for item in automl["unsupported"]
            )
        )
    else:
        lines.append("- AutoML-enabled models waiting on train schema packaging: none")

    lines.extend(
        [
            "",
            "Where I can run and monitor workflows:",
        ]
    )
    for platform in platforms["supported"]:
        lines.append(
            "- {name} ({display_name}): {description}".format(
                name=platform["name"],
                display_name=platform.get("display_name", platform["name"]),
                description=platform.get("description", ""),
            )
        )
    lines.append(
        "- Long-running workflow monitoring is enabled by default, with status "
        f"updates every {defaults['status_interval_minutes']} minutes."
    )

    lines.extend(
        [
            "",
            "Tell me the target task or model family, dataset location and format, "
            "KPI, and execution platform. I can recommend the right TAO workflow, "
            "generate specs/configs/commands, launch or monitor the run, and help "
            "debug failures until the model is ready for the next iteration.",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    """Run the TAO capability summary helper."""
    args = parse_args()
    data = build_capabilities(args.skill_bank.expanduser())

    if args.format == "json":
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(format_capabilities_text(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
