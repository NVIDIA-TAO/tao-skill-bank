#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit a DEFT Cosmos3 stage as a spec-bundle, so any platform can render it.

Same purpose as the DEFT AOI emitter, and deliberately NOT a copy of it. Two
things differ, and both come from what Cosmos3 actually is.

**Most of this loop is host-side.** Of its eleven stages only four families run
a container -- Train, Proxy/Benchmark evaluate, AnomalyGen, and Mining. Proxy
RCCA, benchmark metrics, routing, assemble and validate are bundled Python
(`analyze_gaps.py`, `assemble_training_json.py`, `validate_sharegpt.py`, ...)
that runs on the host and needs no platform at all. Only the container stages
belong here; putting the others in a bundle would wrap a local script in a
scheduler for nothing.

**Commands are RESOLVED, not restated.** The AOI table copies each command from
the model skill and a test compares them. That test exists because the copy was
wrong -- it carried `visual_changenet classify train`, a subcommand that does
not exist, so all three of its VCN stages would have failed on argument
parsing. Here the command, its mode, and its config_format are read from
`tao-finetune-cosmos-reason/references/skill_info.yaml` at build time. The
model skill OWNS them; there is no second copy to drift.

That matters more for Cosmos3 than it would for AOI: `cosmos-rl` train is not a
tidy CLI but a multi-line shell hook that computes a path inside the container,
and any hand-transcription of it would be wrong within one release.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any

# The model skill that owns the Cosmos-Reason action contract.
MODEL_SKILL = "skills/models/tao-finetune-cosmos-reason"
# Backend whose container_image the loop uses. cosmos-rl is the DEFT default;
# `backend_contracts` in the model skill maps each to a concrete image.
DEFAULT_BACKEND = "cosmos-rl"

# Non-model images, resolved through versions.yaml like everything else.
ANOMALYGEN = "metropolis_sdg.paidf_anomalygen"
DATA_SERVICES = "tao_toolkit.data_services"

# Cosmos-RL addresses EVERY path in its spec under this root, and its own
# template warns: "Mount the workspace at /tao-workspace -- NEVER over
# /workspace, which is where cosmos-rl itself is installed." Mounting the
# workspace at its host path instead does not fail; the spec's paths simply do
# not resolve.
COSMOS_WORKSPACE = "/tao-workspace"
AG_CHECKPOINTS = "/workspace/paidf-anomalygen/checkpoints"


def _stage(*, action=None, image=None, gpus, mode="args", command=None,
           config_format=None, inputs=(), outputs=(), workdir=None, targets=None):
    """One stage. `action` resolves through the model skill; `image`+`command`
    declare a stage the model skill does not own."""
    return {"action": action, "image": image, "command": command,
            "config_format": config_format, "gpus": gpus, "mode": mode,
            "inputs": tuple(inputs), "outputs": tuple(outputs),
            "workdir": workdir, "targets": dict(targets or {})}


STAGES: dict[str, dict[str, Any]] = {
    # ---- Cosmos-Reason: train and the two evaluates ----------------------
    # command/mode/config_format come from the model skill; only the DEFT-loop
    # facts (what it reads, what it writes, where the workspace must appear)
    # are stated here.
    "train": _stage(
        action="train", gpus=1,
        inputs=("workspace", "annotations"), outputs=("checkpoint",),
        targets={"workspace": COSMOS_WORKSPACE},
    ),
    # Proxy and Benchmark are the SAME action against different annotation
    # sets. They are separate entries because the loop records them as separate
    # stages and their outputs must not collide.
    "evaluate_proxy": _stage(
        action="evaluate", gpus=1,
        inputs=("workspace", "checkpoint", "proxy_annotations"),
        outputs=("proxy_results",),
        targets={"workspace": COSMOS_WORKSPACE},
    ),
    "evaluate_benchmark": _stage(
        action="evaluate", gpus=1,
        inputs=("workspace", "checkpoint", "benchmark_annotations"),
        outputs=("benchmark_results",),
        targets={"workspace": COSMOS_WORKSPACE},
    ),

    # ---- AnomalyGen: AMP routing (CPU) then SDG diffusion (GPU) ----------
    # gpus=0 on AMP for the same reason as AOI: it is seconds of routing, and a
    # scheduler bills an allocation from the moment it starts.
    "anomalygen.amp": _stage(
        image=ANOMALYGEN, command="bash -lc", gpus=0,
        inputs=("dataset_dir", "defect_spec", "cosmos_models"),
        outputs=("testcase_jsonl", "allocation_json"),
        workdir="/workspace/paidf-anomalygen",
        targets={"cosmos_models": AG_CHECKPOINTS},
    ),
    "anomalygen.sdg": _stage(
        image=ANOMALYGEN, command="bash -lc", gpus=1,
        inputs=("testcase_jsonl", "checkpoint_dir", "cosmos_models"),
        outputs=("sdg_dir",),
        workdir="/workspace/paidf-anomalygen",
        targets={"cosmos_models": AG_CHECKPOINTS},
    ),

    # ---- Mining: embed then k-NN ----------------------------------------
    # Named `data_mining.*` to match the stage this loop RECORDS. AOI calls the
    # same family `mining.*`; each table follows its own workflow's vocabulary,
    # because that is what commit_stage.py and deft_state.json use.
    "data_mining.embed_target": _stage(
        image=DATA_SERVICES, command="embedding image_embeddings -e {config_path}",
        gpus=1, mode="config", config_format="yaml",
        inputs=("target_parquet",), outputs=("target_embeddings",),
    ),
    "data_mining.embed_pool": _stage(
        image=DATA_SERVICES, command="embedding image_embeddings -e {config_path}",
        gpus=1, mode="config", config_format="yaml",
        inputs=("mining_pool",), outputs=("pool_embeddings",),
    ),
    "data_mining.knn": _stage(
        image=DATA_SERVICES, command="tmm nearest_neighbors -e {config_path}",
        gpus=1, mode="config", config_format="yaml",
        inputs=("target_embeddings", "pool_embeddings"), outputs=("mined_parquet",),
    ),
}

# Stages the loop records that run on the HOST. Named so `--list` can say why
# they are absent, rather than looking like an oversight.
HOST_SIDE_STAGES = {
    "proxy_rcca": "scripts/analyze_gaps.py",
    "benchmark_metrics": "scripts/analyze_gaps.py",
    "routing": "bundled routing over the Proxy gaps",
    "assemble_data": "scripts/assemble_training_json.py",
    "validate_data": "scripts/validate_sharegpt.py",
    "loop_stop": "scripts/finalize_run.py",
}


def _model_skill_info(bank: pathlib.Path) -> dict[str, Any]:
    import yaml

    path = bank / MODEL_SKILL / "references/skill_info.yaml"
    if not path.is_file():
        raise ValueError(
            f"{MODEL_SKILL} is not installed at {path}; its skill_info.yaml owns "
            "the Cosmos-Reason commands this workflow launches"
        )
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_action(action: str, bank: pathlib.Path,
                   backend: str = DEFAULT_BACKEND) -> dict[str, Any]:
    """Command, mode, config_format and image for a model-skill action.

    Read rather than copied. `cosmos-rl` train is a multi-line shell hook that
    computes a path inside the container; a transcription of it would be stale
    within a release, and a wrong one fails at argument parsing after the job
    has already been scheduled.
    """
    info = _model_skill_info(bank)
    actions = info.get("actions") or {}
    if action not in actions:
        raise ValueError(
            f"{MODEL_SKILL} declares no action {action!r}; it has "
            f"{', '.join(sorted(actions))}"
        )
    entry = actions[action]
    contracts = info.get("backend_contracts") or {}
    if backend not in contracts:
        raise ValueError(
            f"{MODEL_SKILL} declares no backend {backend!r}; it has "
            f"{', '.join(sorted(contracts))}"
        )
    image = contracts[backend].get("container_image")
    if not image:
        raise ValueError(f"backend {backend!r} declares no container_image")
    return {"command": entry.get("command"), "mode": entry.get("mode", "config"),
            "config_format": entry.get("config_format"), "image": image}


def resolve_image(key: str, bank: pathlib.Path) -> str:
    """Dotted versions.yaml key -> URI. A resolved URI passes through."""
    if "/" in key:
        return key
    import yaml

    data = yaml.safe_load((bank / "versions.yaml").read_text(encoding="utf-8"))
    node: Any = data.get("images", data)
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"versions.yaml has no image key {key!r} (at {part!r})")
        node = node[part]
    return str(node)


def build(stage: str, params: dict[str, str], *, results_dir: str,
          bank: pathlib.Path, network_arch: str = "cosmos-reason",
          args: list[str] | None = None, spec: dict[str, Any] | None = None,
          image: str | None = None, backend: str = DEFAULT_BACKEND) -> dict[str, Any]:
    """Assemble one Cosmos3 stage's spec-bundle."""
    if stage in HOST_SIDE_STAGES:
        raise ValueError(
            f"{stage!r} runs on the host ({HOST_SIDE_STAGES[stage]}), not in a "
            "container; there is nothing for a platform to render"
        )
    if stage not in STAGES:
        raise ValueError(
            f"unknown stage {stage!r}; known: {', '.join(sorted(STAGES))}"
        )
    entry = STAGES[stage]
    missing = [name for name in entry["inputs"] if not params.get(name)]
    if missing:
        raise ValueError(f"stage {stage!r} needs --param for: {', '.join(missing)}")

    if entry["action"]:
        owned = resolve_action(entry["action"], bank, backend)
        command, mode = owned["command"], owned["mode"]
        config_format = owned["config_format"]
        resolved_image = image or owned["image"]
    else:
        command, mode = entry["command"], entry["mode"]
        config_format = entry["config_format"]
        resolved_image = image or resolve_image(entry["image"], bank)

    declared_inputs = []
    for name in entry["inputs"]:
        item: dict[str, Any] = {
            "spec_key": name,
            "type": "file" if name.endswith(("_spec", "_jsonl", "annotations")) else "folder",
            "uri": params[name],
        }
        target = entry["targets"].get(name)
        if target:
            item["target"] = target
        declared_inputs.append(item)

    bundle: dict[str, Any] = {
        "network_arch": network_arch,
        "action": stage,
        "image": resolved_image,
        "mode": mode,
        "command": command,
        "declared_inputs": declared_inputs,
        "declared_outputs": [{"spec_key": n, "type": "folder"} for n in entry["outputs"]],
        "compute_shape": {"gpus": int(entry["gpus"]), "nodes": 1},
    }
    if mode == "config":
        if spec is None:
            raise ValueError(f"stage {stage!r} is mode=config and needs --spec-file")
        if args:
            raise ValueError(
                f"stage {stage!r} is mode=config and cannot take --arg; put the "
                "setting in the spec file"
            )
        bundle["spec"] = spec
        bundle["config_format"] = config_format or "toml"
    elif args:
        bundle["args"] = list(args)
    if entry["workdir"]:
        bundle["workdir"] = entry["workdir"]
    return bundle


def _parse_params(pairs: list[str] | None) -> dict[str, str]:
    params: dict[str, str] = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"--param expects KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        params[key.strip()] = value
    return params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", nargs="?", help=f"one of: {', '.join(sorted(STAGES))}")
    parser.add_argument("--results-dir")
    parser.add_argument("--param", action="append", metavar="KEY=VALUE")
    parser.add_argument("--arg", action="append", dest="args")
    parser.add_argument("--spec-file", type=pathlib.Path)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--image", help="smoke-test override; production images "
                                        "come from the model skill or versions.yaml")
    parser.add_argument("--network-arch", default="cosmos-reason")
    parser.add_argument("--bank", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parents[4])
    parser.add_argument("--list", action="store_true")
    parsed = parser.parse_args(argv)

    if parsed.list:
        for name in sorted(STAGES):
            entry = STAGES[name]
            owner = f"model skill action `{entry['action']}`" if entry["action"] else entry["image"]
            print(f"{name:22} gpus={entry['gpus']}  {owner}")
            print(f"{'':22} inputs: {', '.join(entry['inputs']) or '-'}")
        print("\nhost-side (no container, nothing to render):")
        for name, owner in sorted(HOST_SIDE_STAGES.items()):
            print(f"  {name:20} {owner}")
        return 0

    if not parsed.stage or not parsed.results_dir:
        parser.error("stage and --results-dir are required unless --list is given")
    try:
        spec = None
        if parsed.spec_file:
            import yaml
            spec = yaml.safe_load(parsed.spec_file.read_text(encoding="utf-8"))
        bundle = build(parsed.stage, _parse_params(parsed.param),
                       results_dir=parsed.results_dir, bank=parsed.bank,
                       network_arch=parsed.network_arch, args=parsed.args,
                       spec=spec, image=parsed.image, backend=parsed.backend)
    except ValueError as exc:
        print(f"stage_bundle: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(bundle, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
