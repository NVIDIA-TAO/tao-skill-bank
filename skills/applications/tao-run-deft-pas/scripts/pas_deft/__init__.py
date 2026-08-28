# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Bundled and adapted from the Apache-2.0 NVIDIA TAO Tutorials PAS DEFT utilities so the
# customer workflow does not depend on an external source checkout.

"""Bundled CLIP DEFT pipeline utilities used by this skill."""

<<<<<<< HEAD:skills/applications/tao-run-deft-iaa/scripts/iaa_deft/__init__.py
from iaa_deft.config import IaaDeftConfig

__all__ = ["IaaDeftConfig"]
=======
from pas_deft.config import PasDeftConfig, config_field_metadata

__all__ = ["PasDeftConfig", "config_field_metadata"]
>>>>>>> 0ea1223 ([TAO-6655434][Bugfix] Rename DEFT workflow from IAA to PAS (#194)):skills/applications/tao-run-deft-pas/scripts/pas_deft/__init__.py
