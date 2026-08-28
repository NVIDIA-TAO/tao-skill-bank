# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for PAS-specific release image bindings."""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pas_stamped_pins_match_published_versions():
    versions = yaml.safe_load((REPO_ROOT / "versions.yaml").read_text())
    images = versions["images"]["tao_toolkit"]
    scripts = REPO_ROOT / "skills/applications/tao-run-deft-iaa/scripts"
    for path in (
        scripts / "prepare_deft_config.py",
        scripts / "init_deft_state.py",
        scripts / "audit_deft_run.py",
        scripts / "run_deft_container.py",
    ):
        text = path.read_text()
        assert images["deft_pas_pyt"] in text
        assert images["deft_pas_data_services"] in text
