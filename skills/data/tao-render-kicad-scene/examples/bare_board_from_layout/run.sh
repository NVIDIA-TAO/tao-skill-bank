#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Synthetic bare board: draw a reference -> trace -> create -> render -> compare -> cut an open
# -> render -> compare with an independently drawn defect image -> replay.
# Usage: run.sh <new-run-dir> <python-with-numpy-and-pillow>
set -euo pipefail
RUN=${1:?usage: run.sh <new-run-dir> <python>}
PY=${2:?a python with numpy and Pillow}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LIGHT="$PY $HERE/../../scripts/kicad_light.py"
EX='{skill}/examples/bare_board_from_layout'
IMG='{skill}/scripts/offline_image.py'
GLB=(pcb export glb --force --include-tracks --include-pads --include-zones --user-origin 0x0mm)
step() { local name=$1 tool=$2; shift 2; $LIGHT step --run "$RUN" --name "$name" --tool "$tool" -- "$@"; }
render() {  # name, board dir, output dir
  step "$1" blender --python '{skill}/scripts/blender_scene.py' -- --glb "{run}/$2/board.glb" \
    --cad "{run}/$2/cad_metadata.json" --scene "$EX/scene.json" --output "{run}/$3"
}

step draw_golden python "$EX/make_synthetic_reference.py" --output '{run}/00_reference/golden.png'
step draw_open python "$EX/make_synthetic_reference.py" --defect open --output '{run}/00_reference/open.png'
step trace python "$IMG" layout '{run}/00_reference/golden.png' --out '{run}/00_reference/golden.layout.json' \
  --mm-per-px 0.05 --scale-basis 'synthetic fixture: defined as 0.05 mm per pixel' \
  --copper bright --copper-basis 'synthetic fixture: copper is drawn bright'
step create kicad-python "$EX/build_board.py" --layout '{run}/00_reference/golden.layout.json' --output '{run}/01_create'
step export_create kicad-cli "${GLB[@]}" --output '{run}/01_create/board.glb' '{run}/01_create/board.kicad_pcb'
render render_create 01_create 02_render_create
step compare_create python "$IMG" compare '{run}/00_reference/golden.png' --layout '{run}/00_reference/golden.layout.json' \
  --render '{run}/02_render_create' --entity-prefix board:board_copper --out-json '{run}/02_render_create/compare.json'
# The open is drawn at pixels x 80..92, y 100..120 (inclusive) = 4.0..4.65 mm, 5.0..6.05 mm.
step modify kicad-python "$EX/modify_copper.py" --parent '{run}/01_create/board.kicad_pcb' \
  --output '{run}/03_modify' --cut-mm 4.0 5.0 4.65 6.05
step export_modify kicad-cli "${GLB[@]}" --output '{run}/03_modify/board.glb' '{run}/03_modify/board.kicad_pcb'
render render_modify 03_modify 04_render_modify
step compare_modify python "$IMG" compare '{run}/00_reference/open.png' --layout '{run}/00_reference/golden.layout.json' \
  --render '{run}/04_render_modify' --entity-prefix board:board_copper --out-json '{run}/04_render_modify/compare.json'
step diff_boxes python "$IMG" diff --render-a '{run}/02_render_create' --render-b '{run}/04_render_modify' \
  --entity-prefix board:board_copper --boxes '[[250, 312.5, 290.625, 378.125]]' --out-json '{run}/04_render_modify/diff.json'
$LIGHT finish --run "$RUN"
$LIGHT replay --receipt "$RUN/receipt.json" --run "$RUN-replay"
