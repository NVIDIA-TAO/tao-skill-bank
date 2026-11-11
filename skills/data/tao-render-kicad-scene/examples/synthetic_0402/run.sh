#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Synthetic create -> modify -> render -> replay. Usage: run.sh <new-run-dir> [controller-python]
# Needs KICAD_APPDIR and BLENDER_BIN; the run directory must be outside the checkout.
set -euo pipefail
RUN=${1:?usage: run.sh <new-run-dir> [python]}
PY=${2:-python3}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LIGHT="$PY $HERE/../../scripts/kicad_light.py"
EX='{skill}/examples/synthetic_0402'
GLB=(pcb export glb --force --subst-models --include-tracks --include-pads --include-zones
     --include-silkscreen --include-soldermask --cut-vias-in-body --user-origin 0x0mm)
render() {  # name, board dir, scene json, output dir
  $LIGHT step --run "$RUN" --name "$1" --tool blender -- --python '{skill}/scripts/blender_scene.py' -- \
    --glb "{run}/$2/board.glb" --cad "{run}/$2/cad_metadata.json" --scene "$EX/$3" --output "{run}/$4"
}

$LIGHT step --run "$RUN" --name create --tool kicad-python -- "$EX/create_board.py" --output '{run}/01_create'
$LIGHT step --run "$RUN" --name export_create --tool kicad-cli -- "${GLB[@]}" \
  --output '{run}/01_create/board.glb' '{run}/01_create/board.kicad_pcb'
render render_create 01_create scene.json 02_render_create
$LIGHT step --run "$RUN" --name modify --tool kicad-python -- "$EX/modify_board.py" \
  --parent '{run}/01_create/board.kicad_pcb' --output '{run}/03_modify'
$LIGHT step --run "$RUN" --name export_modify --tool kicad-cli -- "${GLB[@]}" \
  --output '{run}/03_modify/board.glb' '{run}/03_modify/board.kicad_pcb'
render render_modify 03_modify scene.json 04_render_modify
render render_variant_starved 03_modify scene_variant_starved.json 05_render_variant_starved
$LIGHT finish --run "$RUN"
$LIGHT replay --receipt "$RUN/receipt.json" --run "$RUN-replay"
