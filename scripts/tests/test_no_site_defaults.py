# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skills ship no value that only one site can use as a default.

A dataset, a registry image, a model endpoint, or a cluster's queue names belong
to the user's site. A skill names the input, says what it must contain, and
asks; it never fills in a value that an outside user cannot reach.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import build_plugin_package as builder  # noqa: E402

SKILLS = ROOT / "skills"
DINO = ROOT / "skills/models/tao-train-dino"
AUTOML_INTENT = ROOT / "skills/applications/tao-run-automl/references/automl-intent-algorithms.md"

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
