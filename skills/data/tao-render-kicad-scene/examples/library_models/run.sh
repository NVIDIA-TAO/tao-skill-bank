#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Synthetic model-reference check: shared and differently named models, several models per
# footprint with transforms, VRML, a refused missing file -> export -> render -> check -> replay.
# Usage: run.sh <new-run-dir> [controller-python]
set -euo pipefail
RUN=${1:?usage: run.sh <new-run-dir> [python]}
PY=${2:-python3}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LIGHT="$PY $HERE/../../scripts/kicad_light.py"
EX='{skill}/examples/library_models'
step() { local name=$1 tool=$2; shift 2; $LIGHT step --run "$RUN" --name "$name" --tool "$tool" -- "$@"; }

step project python "$EX/make_project.py" --output '{run}/00_project'
step create kicad-python "$EX/create_board.py" --project '{run}/00_project' --output '{run}/01_create'
step export kicad-cli pcb export glb --force --subst-models --include-tracks --include-pads --include-zones \
  --include-silkscreen --include-soldermask --user-origin 0x0mm \
  --output '{run}/01_create/board.glb' '{run}/01_create/board.kicad_pcb'
step render blender --python '{skill}/scripts/blender_scene.py' -- --glb '{run}/01_create/board.glb' \
  --cad '{run}/01_create/cad_metadata.json' --scene "$EX/scene.json" --output '{run}/02_render'
step check python "$EX/check_models.py" --cad '{run}/01_create/cad_metadata.json' \
  --receipt '{run}/02_render/render_receipt.json' --out-json '{run}/03_check/model_report.json'
$LIGHT finish --run "$RUN"
$LIGHT replay --receipt "$RUN/receipt.json" --run "$RUN-replay"
