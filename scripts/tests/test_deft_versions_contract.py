# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard DEFT image references against release-tag drift."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFT_SKILLS = {
    REPO_ROOT / "skills/applications/tao-run-deft-aoi": (
        "images.tao_toolkit.pyt",
        "images.tao_toolkit.data_services",
        "images.metropolis_sdg.anomalygen",
    ),
    REPO_ROOT / "skills/applications/tao-run-deft-aoi-cosmos3": (
        "images.tao_toolkit.data_services",
        "images.metropolis_sdg.anomalygen",
    ),
    REPO_ROOT / "skills/applications/tao-run-deft-pas": (
        "images.tao_toolkit.deft_pas_pyt",
        "images.tao_toolkit.deft_pas_data_services",
    ),
}


@pytest.mark.parametrize(
    ("skill_root", "image_keys"),
    DEFT_SKILLS.items(),
    ids=("changenet", "cosmos3", "pas"),
)
def test_preflight_resolves_images_from_versions_yaml(
    skill_root: Path, image_keys: tuple[str, ...]
) -> None:
    text = (skill_root / "references/preflight.md").read_text(encoding="utf-8")

    assert text.count("resolve_versions_key.py") >= len(image_keys)
    for key in image_keys:
        assert key in text
    for variable in (
        "TAO_PYT_IMAGE",
        "TAO_DS_IMAGE",
        "AG_IMAGE",
        "PAS_PYT_IMAGE",
        "PAS_DS_IMAGE",
    ):
        assert not re.search(rf"(?:export\s+)?{variable}=nvcr\.io/", text)


def test_cosmos3_deft_resolves_model_image_from_skill_info() -> None:
    skill_root = REPO_ROOT / "skills/applications/tao-run-deft-aoi-cosmos3"
    text = (skill_root / "references/preflight.md").read_text(encoding="utf-8")

    assert "scripts/resolve_tao_image.py" in text
    assert 'COSMOS_MODEL_ID="${COSMOS_MODEL_ID:-nvidia/Cosmos3-Nano}"' in text
    assert '--model "$COSMOS_MODEL_ID"' in text
    assert "--backend cosmos-rl" in text
    assert "images.tao_toolkit." + "cosmos_rl" not in text


def test_active_deft_references_do_not_pin_anomalygen_release() -> None:
    pinned = re.compile(r"nvcr\.io/nvidia/(?:paidf|cdf)-anomalygen:[^\s`'\"]+")
    offenders: list[str] = []
    for skill_root in DEFT_SKILLS:
        for path in sorted((skill_root / "references").glob("*.md")):
            if pinned.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, (
        "DEFT references must resolve images.metropolis_sdg.anomalygen "
        "from versions.yaml instead of pinning a tag: " + ", ".join(offenders)
    )


def test_anomalygen_single_version_key_and_unified_actions() -> None:
    import subprocess
    import sys
    import yaml
    versions = yaml.safe_load((REPO_ROOT / 'versions.yaml').read_text())
    images = versions['images']['metropolis_sdg']
    assert images['anomalygen'] == 'nvcr.io/nvidia/cdf-anomalygen:1.2.0'  # versions-key: images.metropolis_sdg.anomalygen
    assert 'paidf_anomalygen' not in images and 'anomalygen_next' not in images
    resolved = subprocess.run([sys.executable, str(REPO_ROOT / 'scripts/resolve_versions_key.py'),
                               'images.metropolis_sdg.anomalygen'], capture_output=True, text=True, check=True)
    assert resolved.stdout.strip() == images['anomalygen']
    info = yaml.safe_load((REPO_ROOT / 'skills/data/tao-generate-anomalies/references/skill_info.yaml').read_text())
    assert info['container_image'] == images['anomalygen']
    assert set(info['actions']) == {'prepare', 'full', 'finetune_only', 'inference_only'}
