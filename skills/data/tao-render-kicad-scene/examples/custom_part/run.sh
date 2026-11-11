#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Synthetic custom part: SDK-built two-pad footprint + box body -> render -> misplaced variant -> replay.
# Usage: run.sh <new-run-dir> [controller-python]
set -euo pipefail
RUN=${1:?usage: run.sh <new-run-dir> [python]}
PY=${2:-python3}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LIGHT="$PY $HERE/../../scripts/kicad_light.py"
EX='{skill}/examples/custom_part'
step() { local name=$1 tool=$2; shift 2; $LIGHT step --run "$RUN" --name "$name" --tool "$tool" -- "$@"; }
render() {  # name, scene json, output dir
  step "$1" blender --python '{skill}/scripts/blender_scene.py' -- --glb '{run}/01_create/board.glb' \
    --cad '{run}/01_create/cad_metadata.json' --scene "$EX/$2" --output "{run}/$3"
}

step create kicad-python "$EX/create_board.py" --output '{run}/01_create' --board-mm 6.0 4.0 --at-mm 3.0 2.0 \
  --pad 1 -1.5 0 1.2 1.5 --pad 2 1.5 0 1.2 1.5
step export kicad-cli pcb export glb --force --subst-models --include-tracks --include-pads --include-zones \
  --include-silkscreen --include-soldermask --user-origin 0x0mm \
  --output '{run}/01_create/board.glb' '{run}/01_create/board.kicad_pcb'
render render_parent scene.json 02_render_parent
render render_misplaced scene_variant_misplaced.json 03_render_misplaced
$LIGHT finish --run "$RUN"
$LIGHT replay --receipt "$RUN/receipt.json" --run "$RUN-replay"
