---
name: tao-render-kicad-scene
description: >-
  Build or modify a small PCB scene with the KiCad pcbnew SDK, reusing library
  footprints and their 3D models, export it with kicad-cli, add parametric
  solder and lighting in Blender, render it, and replay the run from its JSON
  receipt. Use to reconstruct a board, component or solder-joint scene from a
  description or reference measurements, fit its appearance within fixed
  bounds, and make controlled variants. Not for training, dataset export, or
  full CAD recovery.
license: Apache-2.0
compatibility: Requires an extracted KiCad 10.0.5 AppDir and Blender 4.2.1 installed outside the repository. Linux x64, CPU only. Python 3.10+ for the controller; numpy and Pillow enable pixel comparison on replay.
metadata:
  author: NVIDIA Corporation
  version: "0.2.0"
allowed-tools: Read Bash Write
tags: [tao, data, kicad, blender, pcb, synthetic-data, rendering, aoi]
---

# Render a KiCad Scene

You write short KiCad SDK scripts that create or edit a board, then export,
render and replay them. The skill adds a few thin helpers and six small
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
| Library part with shared, renamed or several models | The models each footprint declares, with their transforms | none | `library_models` |
| Part with no library match | SDK-built SMD pads (`smd_footprint`) | `box` body | `custom_part` |
| Bare-board copper, opens and shorts | F.Cu polygons traced from a two-level image | none | `bare_board_from_layout` |
| Die or other object on a carrier | Carrier outline, optional attach pad | `box` die | `die_on_carrier` |

`examples/appearance_fit` is not a scene type: it renders a reference with a
known look and then searches for it on the same geometry, as a check of the
appearance search itself.

## Workflow

1. **Freeze the inputs.** Decide which reference image, pair image and ground
   truth you may use, and say so in the run notes. Source files are read-only.
2. **Inventory the scene.** List board, components, pads, joints and traces
   that are visible; the camera and lighting; and each assumption. Pixel size
   is not millimetres until a known dimension fixes the scale.
3. **Create the board** with an SDK script (start from
   `examples/synthetic_0402/create_board.py`). Search the library first (see
   "Library first"); add a custom asset only for what the library lacks, and
   say what you searched and why it was rejected.
4. **Export** with `kicad-cli pcb export glb` (flags in `run.sh`).
5. **Render** with `scripts/blender_scene.py` and a scene JSON.
6. **Read `render_receipt.json`** before looking at the picture: pad alignment
   must be `pass`, `model_check` must show every resolved model in the export,
   each solder asset `built` with a positive volume, and the entities you
   expect must have pixels in `entity_ids.pixel_counts`.
7. **Fit appearance only after geometry and camera are accepted** (see
   "Appearance search"). Keep the initial render, the best one and the history.
8. **Modify one factor** in a new script or scene JSON that reads the parent and
   writes a new directory. Compare receipts to confirm only that factor moved.
9. **Seal and replay**: `finish`, then `replay` into a new directory.

## Library first

Reuse what KiCad already has before modelling anything yourself.

1. **Describe and measure.** A photograph needs someone or something to say
   what the objects are. `scripts/offline_describe.py` asks a vision-language
   model stored on the host (no network) and returns text only; treat its
   answers as provisional and check each against another question or a
   measurement. `offline_image.py profile` gives ordered colour runs along an
   axis, which is how body, termination and pad edges are separated. Write
   down which statements you used, which you rejected and why.
2. **Search.** `scripts/library_search.py` lists footprints by pad count,
   mounting style and body proportions and shows whether each one's models
   resolve in this install. Proportions are all a picture gives without a
   known dimension, so several sizes of one family usually survive.
3. **Compare at most three candidates.** Build each with `place`, render at
   the reference's pixel size, and score regions with `region` and `rois`.
   Record what fits and what does not for each; a silhouette alone often
   cannot separate neighbouring sizes.
4. **Calibrate what the library cannot know.** Lands and silkscreen belong to
   the board, not to the part: resize pads or add graphics with pcbnew and
   list every change as a deviation from the library footprint. Agreement of
   those items afterwards is by construction, not evidence.

`place` reads the models the footprint itself declares. A footprint may name
none, one or several, each with its own offset, rotation and scale, under a
file name unrelated to the footprint and shared with others. References may
use `${KICAD10_3DMODEL_DIR}` (older `KICADn_3DMODEL_DIR` and `KISYS3DMOD` are
aliases), `${KIPRJMOD}`, your own variables, or relative and absolute paths.
A file that is missing, or in a format not yet checked with this exporter, is
an error; `missing="drop"` removes that model on purpose and records it.
Nothing is ever replaced by a box.

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
| `place(board, library, name, ref, x, y, angle)` | Footprint from a library name or `.pretty` path, with every model it declares resolved and bound. |
| `footprint_models(fp)` / `bind_models(fp)` / `add_model(fp, ref, offset, rotation, scale)` | Inspect, resolve or declare model references yourself; transforms are kept. |
| `connect(board, net, pads, width_mm, via_points_mm)` | One net with straight F.Cu segments. |
| `smd_footprint(board, ref, pads, x, y, angle)` | Footprint built from scratch with rectangular SMD pads. |
| `add_copper(board, polys)` / `take_copper(board)` | Put a polygon set on F.Cu, or take the existing copper off for boolean edits. |
| `save(board, out_dir, extra=...)` | Writes `board.kicad_pcb`, reloads it, writes `cad_metadata.json`. Refuses to overwrite. |

Lengths are millimetres in KiCad's board frame, +Y down. To edit, load the
parent with `pcbnew.LoadBoard`, change it, and `save` to a new directory.

## Scene JSON

Unset keys take the defaults at the top of `scripts/blender_scene.py`; the
resolved values are written to the receipt. `--scene` may be repeated; later
files override earlier keys, so camera, look and render settings can live in
separate files.

| Key | Meaning |
|---|---|
| `render` | `resolution`, `samples`, `seed`, `view_transform`, `exposure`. |
| `camera` | `type` (`ORTHO`/`PERSP`), `target_mm`, `ortho_scale_mm` (image width), `elevation_deg` (90 is top-down), `azimuth_deg`, `distance_mm`. |
| `lighting.area` | White area lights: `position_mm` (x, y, height above the board), `power_w`, `size_mm`. |
| `lighting.rgb_dome` | AOI dome as emissive bands: `radius_mm` and per band `color`, `elevation_deg`, `width_deg`, `strength`. |
| `assets` | `solder_fillet` entries: `pads` (`"all"` or IDs such as `"C1:2"`), `amount`, `wetting`. Later entries override earlier ones per pad; `amount: 0` omits the joint. `box` entries: `name`, `center_mm`, `size_mm`, `height_mm`, `rotation_deg` (counter-clockwise on screen), `corner_cut_mm`, `color`, `roughness`, `specular`. |
| `materials.solder` | `color`, `roughness`. |
| `materials.overrides` | Appearance by entity-ID prefix, for example `"board:board_soldermask"` or `"component:C1:part0"`: `color`, `roughness`, `metallic`, `specular`. Only the keys given change; the rest stays as exported. |
| `id_mask` | `false` skips the entity mask (reuse the one from another render of the same geometry). |
| `pad_alignment` | `mode` `require` (default) or `report`, `tolerance_mm`. |

`amount` scales fillet height against the terminal height (1.0 nominal, below
1 starved, above 1 convex and capped at the terminal top). `wetting` scales
how much of the pad toe the solder covers. Both are appearance parameters,
not measured solder volume.

A library model arrives as one entity per material group:
`component:<ref>:part<n>`. For a chip part these are the terminations and the
body; `component_parts` in the receipt gives each group's exported colour and
bounds so you can tell which is which. A prefix such as `component:C1` matches
every group and paints them alike, so name the part. KiCad exports model
materials without a metallic value, which glTF reads as fully metallic: set
`metallic` yourself for ceramic or plastic bodies.

Colours are linear RGB. With `view_transform: "Standard"`, a white world of
strength 1 and `specular: 0`, a flat surface renders at about its own colour,
which makes a measured mean colour a usable starting point.

## Reading a reference offline

`scripts/offline_image.py` is the only way the agent reads a private
reference. It prints numbers, never pixels.

| Command | Use |
|---|---|
| `stats` | Size, value distribution, whether the image is two-level. |
| `segments` | Colour clusters with size, position, orientation and mean colour of their largest regions; `--crop` limits it to a window. |
| `profile` | Ordered colour runs along one axis of a window, each with its extent across the axis: where one material ends and the next begins. |
| `layout` | Trace a two-level bare-board image into pixel-exact copper rectangles. State which level is copper and the scale, each with its basis. |
| `compare` | Copper mask of a render against the reference mask: IoU, mismatches, registration shift. |
| `region` | Rendered entities against a reference region chosen by gray range or by colour cluster: IoU, centroid, size, axis, mean colour, SSIM. |
| `rois` | Appearance inside the render's own entity regions, reference against render: mean colour, 5/50/95 gray percentiles, contrast, highlight share, and one loss. |
| `diff` | Change between two renders, tested against ground-truth boxes. |

Render at the reference's native pixel size; nothing here resizes for scoring.
Files traced from a reference (`layout` output, boards built from it, their
renders) are derived private data and follow the data boundary below.

Treat the scores for what they are. A traced layout that matches its own
source proves the trace-to-CAD-to-render path is exact, not that the scene
was understood. A thresholded region is a segmentation, not labelled truth.
Polarity, scale, height and material stay assumptions until something
independent fixes them.

## Appearance search

Fit materials and lighting only on geometry and a camera you have stopped
changing. `scripts/appearance_search.py` runs a bounded search from a plan
written beforehand:

```bash
python3 scripts/appearance_search.py --plan plan.json --run RUN --search-dir NEW_DIR
```

The plan fixes the seed, the evaluation budget, the stages with each
parameter's bounds, and the stop rule; see the script header for its keys.
It may only vary keys under `materials/`, `lighting/` and `render/exposure`;
anything else, or a base look that carries a camera, is refused. Once the
search has started the plan is final: if a stop rule ends it early, report
that, and do not start a second search to use up the rest of the budget.
`lhs` stages sample inside the bounds; `ratio` stages rescale a base colour by
the reference/render colour of its region. Each evaluation is a recorded
render and a recorded `rois` score, so the receipt replays the search. The
search directory keeps a look file per evaluation, `history.json` with every
result, `initial.json` and `best.json`.

Tune parts separately (board, body, terminations, solder, light) and keep
metals metallic and dielectrics specular: a scene with every reflection
removed can match mean colours and still look wrong. Never texture the scene
with the reference or blur or warp the reference. The loss compares regional
statistics; it is not a likeness score, and regions with content you did not
model (text, neighbouring parts) set a floor on it.

## Judging a run

- `finish` reports `complete` only when every step succeeded.
- `replay` passes when boards and metadata match, entity-ID arrays are equal
  in dtype, shape and values, and each PNG is within the stated pixel
  tolerance. `.blend` and `.glb` bytes embed non-semantic data and are
  reported but do not gate. Paths inside the run directory are read as
  `{run}`. An input from outside the run (a reference, a scene file) whose
  hash changed fails the replay; a changed skill script is reported. A score
  that names the hash of a reference rendered in the same run is accepted
  only through a pair that already matched pixel for pixel.
- `kicad_light.py recompare` repeats the comparison of an existing replay
  without running anything and writes the result to a separate file.
- Pad alignment checks the XY centre of each CAD pad against the exported
  copper. It does not prove solder contact, stack-up height or optical realism.
- `model_check` fails the render when the export holds fewer or more models
  than the CAD metadata resolved; kicad-cli itself drops a model it cannot
  load and still exits 0.
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
  description (`segments`, `profile`) and your own SDK script.
- Colour clustering starts from randomly drawn pixels with a fixed seed. An
  image with a few flat colours can lose a small cluster; raise `--k`.
- Choosing a footprint from a photograph is not automatic. The offline reader
  and the measurements narrow the family and proportions; without a known
  dimension the case size stays an assumption, and the reader contradicts
  itself on fine detail.
- The appearance search matches regional statistics within its bounds. It
  does not recover measured material properties or the real light positions.
- Library models carry nominal dimensions and heights, not those of the part
  in the picture.
- Top-side components only; bottom views are untested.
- CPU Cycles only. GPU rendering, optical fitting, consumer masks and dataset
  export are out of scope for this version.
