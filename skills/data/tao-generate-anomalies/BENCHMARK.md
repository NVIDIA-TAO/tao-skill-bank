<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Validation scope

Unit regressions cover exact generated/blocked identity accounting, file and
class provenance, exclusion propagation, OD mask pairing and COCO validity,
recipe preparation and native best-checkpoint selection. They do not establish
image quality or performance. GPU image/runtime and public PCB-Nano generation
must be verified separately in the actual pinned container, with phase timing,
VRAM, manifests and visual review retained outside the public repository.
