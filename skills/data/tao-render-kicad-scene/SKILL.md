---
name: tao-render-kicad-scene
description: >-
  Build or modify a small PCB scene with the KiCad pcbnew SDK, export it with
  kicad-cli, add parametric solder and AOI lighting in Blender, render it, and
  replay the run from its JSON receipt. Use to reconstruct a board, component
  or solder-joint scene from a description or reference measurements and to
  make controlled variants. Not for training, dataset export, or full CAD recovery.
license: Apache-2.0
compatibility: Requires an extracted KiCad 10.0.5 AppDir and Blender 4.2.1 installed outside the repository. Linux x64, CPU only. Python 3.10+ for the controller; numpy and Pillow enable pixel comparison on replay.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash Write
tags: [tao, data, kicad, blender, pcb, synthetic-data, rendering, aoi]
---

# Render a KiCad Scene

You write short KiCad SDK scripts that create or edit a board, then export,
render and replay them. The skill adds a few thin helpers and four small
worked examples; KiCad, its libraries and Blender are external runtimes you reuse.

One image cannot recover the original CAD or a unique physical truth. Keep
what you observed apart from what you assumed, and record unknowns (pitch,
height, material) instead of guessing them from a reference designator.

## Quick Start

```bash
export KICAD_APPDIR=/external/runtime/kicad-10.0.5/AppDir
export BLENDER_BIN=/external/runtime/blender-4.2.1-linux-x64/blender
python3 scripts/kicad_light.py check                      # versions and hashes
bash examples/synthetic_0402/run.sh /external/runs/demo-001   # create, modify, render, replay
```

The example takes about 45 s per pass on 8 CPU threads (three 640x480 renders
at 64 samples) and ends with `"replay_pass": true`. The run directory holds
`receipt.json`, two editable `.kicad_pcb` boards, three `scene.blend` files and
their `rgb.png`. Open the boards in KiCad and the scenes in Blender to edit them.

Runtimes, inputs and run directories must be outside the checkout; the
controller refuses paths inside it. See `references/runtime.md` to install the
pinned runtimes and for what pcbnew, the IPC API and kicad-cli can each do.

## Pick a representation

Use the smallest one that fits what is visible. Each has a replayable
synthetic example under `examples/`; every `run.sh` takes a new run directory
and a Python that has numpy and Pillow.

| Scene | KiCad owns | Custom asset | Example |
|---|---|---|---|
| Library part, pads, solder | Footprints, pads, tracks, STEP models | `solder_fillet` per pad | `synthetic_0402` |
| Part with no library match | SDK-built SMD pads (`smd_footprint`) | `box` body | `custom_part` |
| Bare-board copper, opens and shorts | F.Cu polygons traced from a two-level image | none | `bare_board_from_layout` |
| Die or other object on a carrier | Carrier outline, optional attach pad | `box` die | `die_on_carrier` |

## Workflow

1. **Freeze the inputs.** Decide which reference image, pair image and ground
   truth you may use, and say so in the run notes. Source files are read-only.
2. **Inventory the scene.** List board, components, pads, joints and traces
   that are visible; the camera and lighting; and each assumption. Pixel size
   is not millimetres until a known dimension fixes the scale.
3. **Create the board** with an SDK script (start from
   `examples/synthetic_0402/create_board.py`). Prefer a standard-library
   footprint and its STEP model; add a custom asset only when the library has none.
4. **Export** with `kicad-cli pcb export glb` (flags in `run.sh`).
5. **Render** with `scripts/blender_scene.py` and a scene JSON.
6. **Read `render_receipt.json`** before looking at the picture: pad alignment
   must be `pass`, each solder asset `built` with a positive volume, and the
   entities you expect must have pixels in `entity_ids.pixel_counts`.
7. **Modify one factor** in a new script or scene JSON that reads the parent and
   writes a new directory. Compare receipts to confirm only that factor moved.
8. **Seal and replay**: `finish`, then `replay` into a new directory.

## Recording steps

Run every native command through the recorder so the receipt can replay it:

```bash
python3 scripts/kicad_light.py step --run RUN --name create --tool kicad-python -- \
  '{skill}/examples/synthetic_0402/create_board.py' --output '{run}/01_create'
python3 scripts/kicad_light.py step --run RUN --name render --tool blender -- \
  --python '{skill}/scripts/blender_scene.py' -- --glb '{run}/01_create/board.glb' \
  --cad '{run}/01_create/cad_metadata.json' --scene SCENE.json --output '{run}/02_render'
python3 scripts/kicad_light.py finish --run RUN
python3 scripts/kicad_light.py replay --receipt RUN/receipt.json --run RUN-replay
```

`--tool` is `kicad-python` (KiCad's bundled interpreter), `kicad-cli`,
`blender` or `python`. Quote `{run}` and `{skill}`: the receipt stores them
unexpanded so a replay can target another directory. Each step records its
argv, input and output SHA-256, wall and CPU seconds, peak memory and log. A
step fails if it exits non-zero or changes a file it was given as input.

Keep your own SDK scripts and scene JSON files next to the run (outside the
checkout) and pass them by absolute path; the receipt hashes them.

## Writing SDK scripts

`scripts/kicad_sdk.py` wraps only the parts that are easy to get wrong. Use
pcbnew directly for everything else.

| Helper | Purpose |
|---|---|
| `seed_ids(seed)` | Call first. KiCad saves items in UUID order, so unseeded IDs reorder the board and the GLB between runs. |
| `new_board(outline_mm)` | Board with a closed Edge.Cuts outline. |
| `place(board, library, name, ref, x, y, angle)` | Standard footprint plus its STEP model by absolute path. |
| `connect(board, net, pads, width_mm, via_points_mm)` | One net with straight F.Cu segments. |
| `smd_footprint(board, ref, pads, x, y, angle)` | Footprint built from scratch with rectangular SMD pads. |
| `add_copper(board, polys)` / `take_copper(board)` | Put a polygon set on F.Cu, or take the existing copper off for boolean edits. |
| `save(board, out_dir, extra=...)` | Writes `board.kicad_pcb`, reloads it, writes `cad_metadata.json`. Refuses to overwrite. |

Lengths are millimetres in KiCad's board frame, +Y down. To edit, load the
parent with `pcbnew.LoadBoard`, change it, and `save` to a new directory.

## Scene JSON

Unset keys take the defaults at the top of `scripts/blender_scene.py`; the
resolved values are written to the receipt.

| Key | Meaning |
|---|---|
| `render` | `resolution`, `samples`, `seed`, `view_transform`, `exposure`. |
| `camera` | `type` (`ORTHO`/`PERSP`), `target_mm`, `ortho_scale_mm` (image width), `elevation_deg` (90 is top-down), `azimuth_deg`, `distance_mm`. |
| `lighting.area` | White area lights: `position_mm` (x, y, height above the board), `power_w`, `size_mm`. |
| `lighting.rgb_dome` | AOI dome as emissive bands: `radius_mm` and per band `color`, `elevation_deg`, `width_deg`, `strength`. |
| `assets` | `solder_fillet` entries: `pads` (`"all"` or IDs such as `"C1:2"`), `amount`, `wetting`. Later entries override earlier ones per pad; `amount: 0` omits the joint. `box` entries: `name`, `center_mm`, `size_mm`, `height_mm`, `rotation_deg` (counter-clockwise on screen), `corner_cut_mm`, `color`, `roughness`, `specular`. |
| `materials.solder` | `color`, `roughness`. |
| `materials.overrides` | Appearance for exported KiCad meshes by entity-ID prefix, for example `"board:board_soldermask"`: `color`, `roughness`, `metallic`, `specular`. |
| `pad_alignment` | `mode` `require` (default) or `report`, `tolerance_mm`. |

`amount` scales fillet height against the terminal height (1.0 nominal, below
1 starved, above 1 convex and capped at the terminal top). `wetting` scales
how much of the pad toe the solder covers. Both are appearance parameters,
not measured solder volume.

Colours are linear RGB. With `view_transform: "Standard"`, a white world of
strength 1 and `specular: 0`, a flat surface renders at about its own colour,
which makes a measured mean colour a usable starting point.

## Reading a reference offline

`scripts/offline_image.py` is the only way the agent reads a private
reference. It prints numbers, never pixels.

| Command | Use |
|---|---|
| `stats` | Size, value distribution, whether the image is two-level. |
| `segments` | Colour clusters with size, position, orientation and mean colour of their largest regions. |
| `layout` | Trace a two-level bare-board image into pixel-exact copper rectangles. State which level is copper and the scale, each with its basis. |
| `compare` | Copper mask of a render against the reference mask: IoU, mismatches, registration shift. |
| `region` | One rendered entity against a thresholded reference region: IoU, centroid, size, axis, mean colour, SSIM. |
| `diff` | Change between two renders, tested against ground-truth boxes. |

Render at the reference's native pixel size; nothing here resizes for scoring.
Files traced from a reference (`layout` output, boards built from it, their
renders) are derived private data and follow the data boundary below.

Treat the scores for what they are. A traced layout that matches its own
source proves the trace-to-CAD-to-render path is exact, not that the scene
was understood. A thresholded region is a segmentation, not labelled truth.
Polarity, scale, height and material stay assumptions until something
independent fixes them.

## Judging a run

- `finish` reports `complete` only when every step succeeded.
- `replay` passes when boards and metadata match, entity-ID arrays are equal
  in dtype, shape and values, and each PNG is within the stated pixel
  tolerance. `.blend` and `.glb` bytes embed non-semantic data and are
  reported but do not gate.
- Pad alignment checks the XY centre of each CAD pad against the exported
  copper. It does not prove solder contact, stack-up height or optical realism.
- A render that looks plausible is not a reconstruction result. Compare
  against the reference with fixed regions and metrics, and report failed
  attempts with the best one.

## Data boundary

Reference photographs, ground truth and anything derived from their pixels
(crops, thumbnails, side-by-side panels, textures, packed scenes) stay outside
every repository, container image and CI artifact, and are never sent to a
cloud model or image tool. Read them only with offline tools on the approved
host and pass numbers, not pixels, back to the agent. Repository examples are
self-generated synthetic data only. Before committing from a checkout that
has handled private data, run `scripts/privacy_guard.py` (see
`references/runtime.md`).

## Limits

- Solder fillets are supported for two-terminal chip parts whose body outline
  is on F.Fab. Other packages report `unsupported` and need a custom asset.
- `layout` handles two-level images only. Photographs need a measured
  description (`segments`) and your own SDK script.
- There is no automatic image understanding or appearance fitting. Without a
  person or an offline vision model describing a photograph, a reconstruction
  from cluster statistics is a first candidate to be reviewed, not a result.
- Top-side components only; bottom views are untested.
- CPU Cycles only. GPU rendering, optical fitting, consumer masks and dataset
  export are out of scope for this version.
