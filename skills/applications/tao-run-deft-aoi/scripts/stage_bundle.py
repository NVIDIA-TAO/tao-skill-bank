#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit a DEFT stage as a spec-bundle, so any platform can render it.

Every DEFT stage was documented as a literal `docker run` line:

    docker run --gpus all --rm --ipc=host --shm-size=16g -v $WS:$WS \\
      -w /workspace/paidf-anomalygen $AG_IMAGE bash -lc "..."

That string IS the coupling. It names a runtime (`docker`), a GPU flag spelled
docker's way, mount syntax docker's way, and three flags that have no meaning
anywhere else. A workflow written this way cannot move platforms, and every new
DEFT workflow forks the same lines again -- which is why `commit_stage.py`
exists three times over in this bank.

A stage is really only ever six facts: which image, what to run, what it reads,
what it writes, how much compute, and whether the command takes a config file
or arguments. Those are exactly the spec-bundle fields, and the platform skills
already turn a bundle into `docker run` / `srun --container-image` /
`kubectl apply` without this file knowing which.

So the docker-isms drop rather than translate:

  --gpus all        -> compute_shape.gpus (the scheduler decides the spelling)
  -v $WS:$WS        -> declared_inputs / results_dir
  --ipc=host        -> nothing; enroot exposes the host tmpfs (89G measured on
  --shm-size=16g       CS-OCI-ORD), and it is a docker-only workaround for
                       docker's 64MB /dev/shm default
  --user $(id -u)   -> nothing; enroot is rootless and already runs as you
  -v /etc/passwd:ro -> nothing; same reason
  --rm              -> never: it deletes the exit code `status()` reads

STAGES below is data, not code. Adding a stage is a table entry, and a second
DEFT workflow supplies its own table rather than forking this module.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any

# Image keys resolve through versions.yaml, never pinned here: a URI written
# into this file would drift from the bank's single source at the next bump.
PYT = "tao_toolkit.pyt"
DATA_SERVICES = "tao_toolkit.data_services"
ANOMALYGEN = "metropolis_sdg.paidf_anomalygen"


def _stage(image, command, *, gpus, mode="args", inputs=(), outputs=(), workdir=None):
    return {"image": image, "command": command, "gpus": gpus, "mode": mode,
            "inputs": tuple(inputs), "outputs": tuple(outputs), "workdir": workdir}


# Each entry's `inputs` names the PARAMS a caller must supply as paths. They
# become declared_inputs, which every platform mounts read-only and exports as
# TAO_INPUT_<SPEC_KEY> -- so a stage command never names a host path directly
# and never has to guess the platform's mount layout.
STAGES: dict[str, dict[str, Any]] = {
    # ---- Visual ChangeNet: train / evaluate / inference -------------------
    "train": _stage(
        PYT, "visual_changenet classify train -e {config_path}", gpus=1, mode="config",
        inputs=("dataset_dir", "backbone"), outputs=("checkpoint",),
    ),
    "evaluate": _stage(
        PYT, "visual_changenet classify evaluate -e {config_path}", gpus=1, mode="config",
        inputs=("dataset_dir", "checkpoint"), outputs=("metrics",),
    ),
    "inference": _stage(
        PYT, "visual_changenet classify inference -e {config_path}", gpus=1, mode="config",
        inputs=("dataset_dir", "checkpoint"), outputs=("inference_csv",),
    ),

    # ---- Root-cause analysis on baseline inference ------------------------
    "rca": _stage(
        DATA_SERVICES, "gap_analysis vcn_aoi -e {config_path}", gpus=1, mode="config",
        inputs=("inference_results_dir",), outputs=("kpi_gaps",),
    ),

    # ---- Mining: embed targets, embed pool, then k-NN ---------------------
    # Three allocations, not one: each output is the next input, and splitting
    # them lets a platform schedule and retry them independently.
    "mining.embed_target": _stage(
        DATA_SERVICES, "embedding image_embeddings -e {config_path}", gpus=1, mode="config",
        inputs=("target_parquet",), outputs=("target_embeddings",),
    ),
    "mining.embed_pool": _stage(
        DATA_SERVICES, "embedding image_embeddings -e {config_path}", gpus=1, mode="config",
        inputs=("mining_pool",), outputs=("pool_embeddings",),
    ),
    "mining.knn": _stage(
        DATA_SERVICES, "tmm nearest_neighbors -e {config_path}", gpus=1, mode="config",
        inputs=("target_embeddings", "pool_embeddings"),
        outputs=("mined_parquet",),
    ),

    # ---- AnomalyGen: AMP routing (CPU) then SDG diffusion (GPU) ----------
    # gpus=0 on the AMP phase is deliberate. Its docker recipe passes
    # `--gpus all` because that is how the image is always launched, but the
    # phase is ~10s of routing with no GPU work -- and on a scheduler an
    # allocation is billed from the moment it starts, not from first compute.
    "anomalygen.amp": _stage(
        ANOMALYGEN, "bash -lc", gpus=0, mode="args",
        inputs=("dataset_dir", "defect_spec", "cosmos_models"),
        outputs=("testcase_jsonl", "allocation_json"),
        workdir="/workspace/paidf-anomalygen",
    ),
    # Bootstrap: one-time asset population. These WRITE to their target, so
    # the cache being populated is passed as --results-dir -- the only path a
    # bundle may write to. Declared inputs are read-only on every platform, so
    # a bootstrap cannot express its target that way, and should not: the
    # populated cache genuinely is this stage's output.
    #
    # They need network and HF_TOKEN, so they are refused under --ctx airgap=1
    # by the launch gate like any other registry-touching work.
    "anomalygen.bootstrap_cosmos": _stage(
        ANOMALYGEN, "bash -lc", gpus=0, mode="args",
        inputs=(), outputs=("cosmos_models",),
        workdir="/workspace/paidf-anomalygen",
    ),
    "anomalygen.bootstrap_dataset": _stage(
        ANOMALYGEN, "bash -lc", gpus=0, mode="args",
        inputs=(), outputs=("dataset_dir",),
        workdir="/workspace/paidf-anomalygen",
    ),
    "anomalygen.bootstrap_checkpoint": _stage(
        ANOMALYGEN, "bash -lc", gpus=0, mode="args",
        inputs=(), outputs=("checkpoint_dir",),
        workdir="/workspace/paidf-anomalygen",
    ),

    "anomalygen.sdg": _stage(
        ANOMALYGEN, "bash -lc", gpus=1, mode="args",
        inputs=("testcase_jsonl", "checkpoint_dir", "cosmos_models"),
        outputs=("sdg_dir",),
        workdir="/workspace/paidf-anomalygen",
    ),
}


def resolve_image(key: str, bank: pathlib.Path) -> str:
    """Dotted versions.yaml key -> concrete URI. Absolute URIs pass through."""
    if "/" in key or key.startswith(("docker://", "nvcr.io")):
        return key
    import yaml

    data = yaml.safe_load((bank / "versions.yaml").read_text(encoding="utf-8"))
    node: Any = data.get("images", data)
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ValueError(
                f"versions.yaml has no image key {key!r} (failed at {part!r})"
            )
        node = node[part]
    if not isinstance(node, str):
        raise ValueError(f"image key {key!r} resolves to {type(node).__name__}, not a URI")
    return node


def spec_key_env(spec_key: str) -> str:
    """The TAO_INPUT_* name a stage command uses to find this input."""
    return "TAO_INPUT_" + re.sub(r"[^A-Za-z0-9]+", "_", spec_key).strip("_").upper()


def build(stage: str, params: dict[str, str], *, results_dir: str,
          bank: pathlib.Path, network_arch: str = "visual-changenet",
          args: list[str] | None = None,
          spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble one stage's spec-bundle.

    Fails closed on a missing input rather than emitting a bundle that would
    run and read nothing: an absent mount is not an error at runtime, so the
    stage would exit 0 having processed no data and every downstream stage
    would treat the empty result as real.
    """
    if stage not in STAGES:
        raise ValueError(
            f"unknown stage {stage!r}; known: {', '.join(sorted(STAGES))}"
        )
    entry = STAGES[stage]
    missing = [name for name in entry["inputs"] if not params.get(name)]
    if missing:
        raise ValueError(
            f"stage {stage!r} needs --param for: {', '.join(missing)}"
        )

    declared_inputs = []
    for name in entry["inputs"]:
        uri = params[name]
        declared_inputs.append({
            "spec_key": name,
            # A spec file is a file; everything else this table declares is a
            # directory tree. Guessing from the path suffix would misclassify
            # an extensionless directory.
            "type": "file" if name == "defect_spec" else "folder",
            "uri": uri,
        })

    bundle: dict[str, Any] = {
        "network_arch": network_arch,
        "action": stage,
        "image": resolve_image(entry["image"], bank),
        "mode": entry["mode"],
        "command": entry["command"],
        "declared_inputs": declared_inputs,
        "declared_outputs": [{"spec_key": name, "type": "folder"}
                             for name in entry["outputs"]],
        "compute_shape": {"gpus": int(entry["gpus"]), "nodes": 1},
    }
    if entry["mode"] == "config":
        # The spec travels as CONTENT, not as a mounted path: the consumer
        # writes it into the staged inputs and substitutes {config_path} with
        # wherever it landed on that platform's compute frame. A bundle that
        # named a host path here would be naming a layout it cannot know.
        if spec is None:
            raise ValueError(
                f"stage {stage!r} is mode=config and needs --spec-file"
            )
        bundle["spec"] = spec
        bundle["config_format"] = "yaml"
        if args:
            # The contract forbids both: a config-mode command reads its
            # settings from the spec file, so anything passed here would be
            # dropped by the consumer rather than reaching the container.
            # Put it in the spec instead.
            raise ValueError(
                f"stage {stage!r} is mode=config and cannot take --arg; "
                "put the setting in the spec file"
            )
    if args:
        bundle["args"] = list(args)
    if entry["workdir"]:
        # Carried in the bundle rather than as a docker `-w`: kubernetes and
        # slurm each spell working directory differently, and the renderer owns
        # that translation.
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
    # Both optional so `--list` works on its own; required-ness is enforced
    # below, where the error can say which mode the caller is in.
    parser.add_argument("stage", nargs="?", help=f"one of: {', '.join(sorted(STAGES))}")
    parser.add_argument("--results-dir")
    parser.add_argument("--param", action="append", metavar="KEY=VALUE",
                        help="input path for a spec_key this stage declares")
    parser.add_argument("--arg", action="append", dest="args",
                        help="argument appended to the container command")
    parser.add_argument("--spec-file", type=pathlib.Path,
                        help="YAML spec for a mode=config stage; travels as "
                             "bundle content, not as a mount")
    parser.add_argument("--network-arch", default="visual-changenet")
    parser.add_argument("--bank", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parents[4])
    parser.add_argument("--list", action="store_true",
                        help="print the stage table and exit")
    parsed = parser.parse_args(argv)

    if parsed.list:
        for name in sorted(STAGES):
            entry = STAGES[name]
            print(f"{name:24} gpus={entry['gpus']}  {entry['command']}")
            print(f"{'':24} inputs: {', '.join(entry['inputs']) or '-'}")
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
                       spec=spec)
    except ValueError as exc:
        print(f"stage_bundle: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(bundle, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
