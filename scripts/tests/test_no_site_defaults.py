# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skills ship no value that only one site can use as a default.

A dataset, a registry image, a model endpoint, or a cluster's queue names belong
to the user's site. A skill names the input, says what it must contain, and
asks; it never fills in a value that an outside user cannot reach.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import build_plugin_package as builder  # noqa: E402

SKILLS = ROOT / "skills"
DINO = ROOT / "skills/models/tao-train-dino"
AUTOML_INTENT = ROOT / "skills/applications/tao-run-automl/references/automl-intent-algorithms.md"
SLURM = ROOT / "skills/platform/tao-run-on-slurm"

# An s3:// URI in skill text is an example only when its bucket is a placeholder.
S3_URI = re.compile(r"s3://[^\s\"'`)|,]*")
PLACEHOLDER_BUCKET = re.compile(r"s3://(?:bucket(?:/|$)|<[^>]+>|\{[^}]+\}|\$\{?\w+\}?|\.\.\.|$)")


# nvcr.io organisations that anyone with an NGC account can pull from.
PUBLIC_REGISTRY_ORGS = {"nvidia"}
REGISTRY_ORG = re.compile(r"nvcr\.io/([A-Za-z0-9][A-Za-z0-9_.-]*)/")
# Hosts under nvidia.com that skill text may name: documentation, downloads, NGC
# and the public model catalog. A host that is missing here needs a check that
# an outside user can reach it before it is added.
PUBLIC_NVIDIA_HOSTS = {
    "authn.nvidia.com", "build.nvidia.com", "catalog.ngc.nvidia.com", "developer.download.nvidia.com",
    "developer.nvidia.com", "docs.nvidia.com", "helm.ngc.nvidia.com", "ngc.nvidia.com",
}
NVIDIA_HOST = re.compile(r"(?<![\w.-])((?:[a-z0-9-]+\.)+nvidia\.com)\b")


def skill_dirs():
    return sorted(path.parent for path in SKILLS.glob("*/*/SKILL.md"))


def shipped_texts(skill_dir: Path):
    """(path, text) of every text file of a skill that the plugin package ships."""
    for path in sorted(skill_dir.rglob("*")):
        inside = path.relative_to(skill_dir).as_posix()
        if not path.is_file() or builder.left_out(skill_dir.name, inside):
            continue
        text = builder.as_text(path.read_bytes())
        if text is not None:
            yield path, text


def test_dino_offers_no_default_dataset():
    for path, text in shipped_texts(DINO):
        concrete = [uri for uri in S3_URI.findall(text) if not PLACEHOLDER_BUCKET.match(uri)]
        assert not concrete, f"{path} names a bucket instead of a placeholder: {concrete}"
        assert not re.search(r"\b\w+_PROFILE\b", text), f"{path} still names a packaged profile"
        assert "default profile" not in text, path
    for name in ("SKILL.md", "references/dino-data-specs.md"):
        text = (DINO / name).read_text(encoding="utf-8")
        assert "ships no default dataset" in text, name
        assert "look in the\nworking directory for a COCO-format dataset" in text, name
        assert "S3 path" not in text, name
    # tao-run-automl asks the model skill for a dataset; it must not expect a default there.
    assert "default profile" not in AUTOML_INTENT.read_text(encoding="utf-8")


def test_dino_tuning_note_agrees_with_the_shipped_schema():
    schema = json.loads((DINO / "schemas/train.schema.json").read_text(encoding="utf-8"))
    assert "train.optim.weight_decay" in schema["automl_default_parameters"]
    note = (DINO / "references/dino-tuning-multigpu.md").read_text(encoding="utf-8")
    assert "`train.optim.weight_decay`" in note          # still a suggested knob
    assert not re.search(r"weight_decay`? is not in", note)
    assert "`weight_decay`\nbehavior" not in (DINO / "SKILL.md").read_text(encoding="utf-8")


def test_shipped_skill_files_name_only_public_registries_and_hosts():
    """A private registry organisation or an internal endpoint is never a default or an example."""
    problems = []
    for skill_dir in skill_dirs():
        for path, text in shipped_texts(skill_dir):
            where = path.relative_to(ROOT).as_posix()
            problems += [f"{where}: nvcr.io/{org}/" for org in set(REGISTRY_ORG.findall(text))
                         if org not in PUBLIC_REGISTRY_ORGS]
            problems += [f"{where}: {host}" for host in set(NVIDIA_HOST.findall(text))
                         if host not in PUBLIC_NVIDIA_HOSTS]
    assert not problems, "\n".join(sorted(problems))


KIT = SKILLS / "core/tao-token-efficient-execution"
AUTOML_REFERENCES = SKILLS / "applications/tao-run-automl/references"


def test_nim_provider_builds_in_no_endpoint_and_no_model_id():
    """Static, so it also holds where Node cannot run the provider tests."""
    text = (KIT / "adapters/pi/nvidia-provider.ts").read_text(encoding="utf-8")
    code = "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith(("/*", "*", "//", "import ")))
    literals = re.findall(r'"([^"\n]*)"', code)
    assert "PI_KIT_NIM_MODELS" in literals and "PI_KIT_NIM_BASE_URL" in literals
    # A URL or an <org>/<model> id in the code would be a built-in endpoint or model.
    assert not [literal for literal in literals if "/" in literal], literals
    assert "if (!BASE_URL || MODEL_IDS.length === 0) return;" in code


def test_kit_and_pack_texts_describe_no_site_default():
    """The model, the data image and the endpoint are the user's: required, never an internal default."""
    for path, text in list(shipped_texts(KIT)) + [
            (cards, cards.read_text(encoding="utf-8")) for cards in sorted(SKILLS.glob("*/*/cards/README.md"))]:
        assert not re.search(r"(?i)nvidia-internal|internal default|\bthe default (?:MODEL|`nim/` endpoint)", text), path
    install = " ".join((KIT / "scripts/install.sh").read_text(encoding="utf-8").split())
    assert "MODEL and the nim/ endpoint have no default." in install
    required = {
        "applications/tao-run-automl/cards/README.md": "`WS`, `VENV`, `MODEL` required, with no default",
        "applications/tao-run-deft-aoi/cards/README.md": "`WS`, `MODEL` and `DS_IMG` required, with no default",
    }
    for name, sentence in required.items():
        flat = " ".join((SKILLS / name).read_text(encoding="utf-8").split())
        assert sentence in flat, name
        optional = " ".join(re.findall(r"[^.;:]* optional\b", flat))
        assert optional and "`MODEL`" not in optional and "`DS_IMG`" not in optional, (name, optional)
    deft = " ".join((SKILLS / "applications/tao-run-deft-aoi/cards/README.md").read_text(encoding="utf-8").split())
    assert ("Ask the user for the image; never guess or substitute one, not even as an example: "
            "show the placeholder `<data-services-image-uri>`.") in deft


def test_automl_texts_do_not_claim_the_runner_has_no_llm_default():
    """The runner package falls back to a built-in endpoint and model. The skill always passes
    both, and its text says so instead of saying that no default exists."""
    config = (AUTOML_REFERENCES / "automl-runner-configuration.md").read_text(encoding="utf-8")
    rows = dict(re.findall(r"(?m)^\| `(llm_endpoint|llm_model)` \| str \| ([^|]*) \|", config))
    assert set(rows) == {"llm_endpoint", "llm_model"}, rows
    for name, default in rows.items():
        assert "**required by this skill**: always pass it" in default, (name, default)
        assert "built-in" in default and "none" not in default.lower(), (name, default)
    # Where the agent resolves the two settings, the text names the runner's fallback.
    intent = (AUTOML_REFERENCES / "automl-intent-algorithms.md").read_text(encoding="utf-8")
    for needle in ("-> ask the user for the base URL of an OpenAI-compatible endpoint they can reach. "
                   "Never leave it out: the runner package then uses its own built-in endpoint",
                   "-> ask the user for a model id that endpoint serves. "
                   "Never leave it out: the runner package then uses a model id of its built-in endpoint"):
        assert needle in intent, needle
    assert intent.count("when it is left out, the runner package uses") == 2
    for path in sorted(AUTOML_REFERENCES.glob("automl-*.md")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if re.search(r"(?i)llm_endpoint|llm_model|llm endpoint|llm model", line):
                bare = re.sub(r"(?i)this skill (?:uses|has) no default", "", line)
                assert not re.search(r"(?i)\bno default\b", bare), f"{path.name}: {line[:120]}"


def test_pack_drivers_require_the_image_and_model_instead_of_defaulting():
    """The value comes from the user: no ${VAR:-<something>} for the model or the data image."""
    packs = sorted(SKILLS.glob("*/*/cards/driver.sh"))
    assert len(packs) == 2, packs
    kit = SKILLS / "core/tao-token-efficient-execution"
    for path in packs + [kit / "scripts/smoke_test.sh"]:
        text = path.read_text(encoding="utf-8")
        for name in ("MODEL", "SMOKE_MODEL", "DS_IMG", "PI_KIT_NIM_BASE_URL"):
            defaults = re.findall(r"\b%s=\$\{%s:-([^}]*)\}" % (name, name), text)
            assert not any(defaults), f"{path}: {name} defaults to {defaults}"
    template = (kit / "templates/kit.env.template").read_text(encoding="utf-8")
    assert "#MODEL=<provider>/<model-id>" in template
    assert "#DS_IMG=<data-services-image-uri>" in template


def test_slurm_metadata_names_inputs_and_no_partition_or_wall_time_cap():
    info = yaml.safe_load((SLURM / "references/skill_info.yaml").read_text(encoding="utf-8"))
    required = {item["name"] for item in info["required_credentials"]}
    optional = {item["name"]: item["only_when"] for item in info["optional_credentials"]}
    assert "SLURM_PARTITION" in required
    assert {"SLURM_CONVERSION_PARTITION", "SLURM_MAX_TIME_HOURS"} <= set(optional)
    defaults = info["resource_defaults"]
    # Partition names and a wall-time limit are properties of the user's cluster.
    assert not {"partition", "sqsh_conversion_partition", "max_time_hours"} & set(defaults)
    # The request the skill makes by default, and the conversion profile, stay: numbers only.
    assert (defaults["time_hours"], defaults["timeout_hours"]) == (4, 3.8)
    assert (defaults["sqsh_conversion_cpus_per_task"], defaults["sqsh_conversion_memory_mb"],
            defaults["sqsh_conversion_timeout_minutes"]) == (4, 7200, 120)


# SHA-256 of the partition names of one cluster that the SLURM skill used to ship. They are
# digests so that the names do not come back through this file.
REMOVED_PARTITION_NAMES = {
    "f712d322b8f5a7e4431bbabfd7f4e5ec4a761cc130b36a2e6a3bad54a246425a",
    "6d1c0bd78874baa5efa332e9b3ed5c6c174becb7c8303d0a640381bf580dc023",
    "25fa656624451b189f4b051baf830b7324ce5d4775ab003f37d17af5782e6a8c",
    "7d87a3f2a6abb5f836eca15337fee30d868d11f77d049e12c9723685f27f2714",
    "fe0eb49a0a391491e21425b2e961e64a0d47ed2b6c10a55d29d95dd895f9a895",
}


def test_slurm_text_asks_for_partitions_and_says_what_the_cluster_must_provide():
    skill = (SLURM / "SKILL.md").read_text(encoding="utf-8")
    section = skill.split("## What the cluster must provide", 1)[1].split("\n## ", 1)[0]
    flat = " ".join(section.split())
    for needle in ("`SLURM_PARTITION`", "`SLURM_CONVERSION_PARTITION`", "4 CPUs", "7200M",
                   "120 minutes", "Pyxis and Enroot", "shared file system"):
        assert needle in flat, needle
    assert "**SLURM_PARTITION** (required, no default)" in skill
    for path, text in shipped_texts(SLURM):
        assert "packaged default is" not in text.lower(), path
        # A partition in a command is the user's variable or a placeholder, never a name.
        commands = [line for line in text.splitlines() if "mkdir -p" not in line]
        for partition in re.findall(r"(?:\s-p|--partition[= ])\s*([^\s\\]+)", "\n".join(commands)):
            assert partition[0] in "<$@", (path, partition)
    # No removed name comes back in prose either, in the skill, in the launch preflight
    # paragraph on SLURM, or in the sbatch templates.
    preflight = ROOT / "skills/core/tao-launch-workflow/references/platform-preflight.md"
    texts = list(shipped_texts(SLURM)) + [
        (path, path.read_text(encoding="utf-8"))
        for path in [preflight, *sorted((ROOT / "templates/slurm").iterdir())]]
    assert len(texts) > 10, len(texts)
    for path, text in texts:
        words = set(re.findall(r"[a-z][a-z0-9_]*", text.lower()))
        back = sorted(word[:2] + "..." for word in words
                      if hashlib.sha256(word.encode()).hexdigest() in REMOVED_PARTITION_NAMES)
        assert not back, f"{path} names a partition of one cluster again: {back}"
    flat = " ".join(preflight.read_text(encoding="utf-8").split())
    assert "partition list" not in flat
    assert "A longer wall time is the user's choice and must fit the limit of their partition." in flat


@pytest.fixture()
def slurm_runtime(monkeypatch):
    """check_slurm_runtime of the launch preflight, with the shipped SLURM metadata."""
    import check_tao_launch_preflight as preflight

    for name in ("SLURM_TIME_HOURS", "SLURM_TIMEOUT_HOURS", "SLURM_MAX_TIME_HOURS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_PARTITION", "gpu-a")
    platform = yaml.safe_load((SLURM / "references/skill_info.yaml").read_text(encoding="utf-8"))
    return lambda: preflight.check_slurm_runtime(platform)


def test_slurm_preflight_keeps_the_default_request(slurm_runtime, capsys):
    assert slurm_runtime()
    out = capsys.readouterr().out
    assert "partition=gpu-a, SLURM_TIME_HOURS=4, SLURM_TIMEOUT_HOURS=3.8" in out


def test_slurm_preflight_has_no_wall_time_cap_of_its_own(slurm_runtime, monkeypatch, capsys):
    """Eight hours on the user's partition used to be refused with a 4-hour limit."""
    monkeypatch.setenv("SLURM_TIME_HOURS", "8")
    monkeypatch.setenv("SLURM_TIMEOUT_HOURS", "7.5")
    assert slurm_runtime()
    out = capsys.readouterr().out
    assert "partition=gpu-a, SLURM_TIME_HOURS=8" in out and "limit" not in out


def test_slurm_preflight_enforces_the_limit_the_user_sets(slurm_runtime, monkeypatch, capsys):
    monkeypatch.setenv("SLURM_MAX_TIME_HOURS", "6")
    monkeypatch.setenv("SLURM_TIME_HOURS", "8")
    assert not slurm_runtime()
    out = capsys.readouterr().out
    assert "requested 8h > max 6h" in out and "SLURM_MAX_TIME_HOURS" in out
    monkeypatch.setenv("SLURM_TIME_HOURS", "6")
    assert slurm_runtime()
    # The child timeout must still end before the wall time does.
    monkeypatch.setenv("SLURM_TIMEOUT_HOURS", "6")
    assert not slurm_runtime()
    monkeypatch.setenv("SLURM_MAX_TIME_HOURS", "soon")
    with pytest.raises(SystemExit, match="SLURM_MAX_TIME_HOURS must be a number of hours"):
        slurm_runtime()
