#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Synthetic appearance search: render a reference with a known look, then search for it from another
# look on the same frozen geometry and camera, within a plan written beforehand -> check -> replay.
# Usage: run.sh <new-run-dir> [controller-python]   (the search directory is <new-run-dir>-search)
set -euo pipefail
RUN=${1:?usage: run.sh <new-run-dir> [python]}
PY=${2:-python3}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPTS="$HERE/../../scripts"
LIGHT="$PY $SCRIPTS/kicad_light.py"
EX='{skill}/examples/appearance_fit'
step() { local name=$1 tool=$2; shift 2; $LIGHT step --run "$RUN" --name "$name" --tool "$tool" -- "$@"; }

step create kicad-python '{skill}/examples/synthetic_0402/create_board.py' --output '{run}/01_create'
step export kicad-cli pcb export glb --force --subst-models --include-tracks --include-pads --include-zones \
  --include-silkscreen --include-soldermask --cut-vias-in-body --user-origin 0x0mm \
  --output '{run}/01_create/board.glb' '{run}/01_create/board.kicad_pcb'
step reference blender --python '{skill}/scripts/blender_scene.py' -- --glb '{run}/01_create/board.glb' \
  --cad '{run}/01_create/cad_metadata.json' --scene "$EX/camera.json" --scene "$EX/look_target.json" --output '{run}/02_reference'
$PY "$SCRIPTS/appearance_search.py" --plan "$HERE/plan.json" --run "$RUN" --search-dir "$RUN-search"
step check python "$EX/check_fit.py" --plan "$EX/plan.json" --search-dir "$RUN-search" --target "$EX/look_target.json" \
  --out-json '{run}/03_check/fit_report.json'
$LIGHT finish --run "$RUN"
$LIGHT replay --receipt "$RUN/receipt.json" --run "$RUN-replay"
