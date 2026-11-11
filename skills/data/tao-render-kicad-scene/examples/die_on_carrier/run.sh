#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Synthetic die on a carrier: create -> render -> two single-factor variants (die missing,
# broken corner) -> replay. Usage: run.sh <new-run-dir> [controller-python]
set -euo pipefail
RUN=${1:?usage: run.sh <new-run-dir> [python]}
PY=${2:-python3}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LIGHT="$PY $HERE/../../scripts/kicad_light.py"
EX='{skill}/examples/die_on_carrier'
step() { local name=$1 tool=$2; shift 2; $LIGHT step --run "$RUN" --name "$name" --tool "$tool" -- "$@"; }
render() {  # name, scene json, output dir
  step "$1" blender --python '{skill}/scripts/blender_scene.py' -- --glb '{run}/01_create/board.glb' \
    --cad '{run}/01_create/cad_metadata.json' --scene "$EX/$2" --output "{run}/$3"
}

step create kicad-python "$EX/create_carrier.py" --output '{run}/01_create' --width-mm 2.4 --height-mm 2.0 \
  --attach-pad-mm 0.4 0.45 1.9 1.65
step export kicad-cli pcb export glb --force --include-tracks --include-pads --include-zones --user-origin 0x0mm \
  --output '{run}/01_create/board.glb' '{run}/01_create/board.kicad_pcb'
render render_parent scene.json 02_render_parent
render render_missing scene_variant_missing.json 03_render_missing
render render_broken_corner scene_variant_broken_corner.json 04_render_broken_corner
$LIGHT finish --run "$RUN"
$LIGHT replay --receipt "$RUN/receipt.json" --run "$RUN-replay"
