# Runtime, SDK capabilities and the privacy guard

## Pinned runtimes

Install both outside the repository and point `KICAD_APPDIR` and `BLENDER_BIN`
at them. Do not mix their interpreters: KiCad scripts run in KiCad's bundled
Python, Blender scripts in Blender's, and the controller in any Python 3.10+.

| Runtime | Official download | SHA-256 |
|---|---|---|
| KiCad 10.0.5 Full AppImage tar | https://mirrors.mit.edu/kicad/appimage/stable/kicad-10.0.5-x86_64.AppImage.tar | `af65bb1fd5ee2730df860bc2a8c49f507a64c83c15c2ce13927eec74d38eba8f` |
| Blender 4.2.1 LTS Linux x64 | https://download.blender.org/release/Blender4.2/blender-4.2.1-linux-x64.tar.xz | `be0fbaa0c1e52d4552023220b4c67351efbb707cac49bb381fcbee2182447005` |

Extract the KiCad tar, make the AppImage executable, and run it with
`--appimage-extract`; this release creates `AppDir` directly. Extract the
Blender tar separately. `kicad_light.py check` prints the versions and the
hashes of `AppRun`, `bin/kicad-cli` and the Blender binary; every receipt
stores the same block, and `replay` reports any drift from the original run.

The controller compares replayed PNG files by decoded pixels and entity-ID
arrays by value when numpy and Pillow are importable; without them a file
whose bytes differ is a failure. `offline_image.py` needs both. Blender PNG
files usually differ in container bytes between runs while every decoded
pixel is equal; replay reports that as `pixel_identical`.

## What each KiCad interface can do in 10.0.5

Checked on the pinned AppDir; re-check before relying on any of this with
another version.

| Interface | Status here | Notes |
|---|---|---|
| `pcbnew` SWIG bindings via `AppRun python3.11` | Used. Headless. | Create, load, edit and save boards; load library footprints; pads, tracks, nets, shapes, 3D model references; seedable UUIDs through `KIID.SeedGenerator`. KiCad classes this API as legacy. |
| `kicad-cli pcb export` | Used for `glb`. Headless. | Also offers step, stl, ply, vrml, svg, gerbers, drill, pos and stats. |
| `kicad-cli pcb render` | Available, not used. | KiCad's own raytraced PNG or JPEG. Useful as a quick preview; it has no solder, dome lighting or entity IDs. |
| IPC API with the `kicad-python` client | Not used. | The client module is not in the AppDir, the CLI has no API-server subcommand, and in KiCad 10 the IPC API needs a running GUI. Headless IPC is a later-version capability; do not assume it here. |

Two behaviours of the 10.0.5 bindings that the helpers work around:

- Item UUIDs are random and KiCad writes items in UUID order, so two runs of
  the same script give differently ordered boards and GLB meshes. Seed the
  generator first (`kicad_sdk.seed_ids`).
- After `board.Remove(item)`, freeing that item's Python wrapper makes
  `SHAPE_POLY_SET` methods such as `UnitSet()` return opaque objects for the
  rest of the process. `kicad_sdk.take_copper` keeps the wrappers alive; do
  the same if you remove items yourself.

`kicad_sdk.py` binds a footprint's STEP model by absolute path into the
AppDir, so a board file is tied to that install location. `cad_metadata.json`
records the model path and hash.

## Coordinates and units

- JSON and SDK helpers: millimetres, KiCad board frame, +Y down.
- Blender world: metres, `world = (x, -y, z) / 1000`. The Y flip happens once,
  at import; nothing is mirrored afterwards.
- Top-down renders match KiCad's on-screen orientation (image up is KiCad -Y).
- Board top copper sits near z = 1.55 mm for the default 1.6 mm board; the
  render receipt records the measured `board_top_z_mm`.

## Privacy guard

Run before a checkout that has handled private references commits or pushes.

```bash
# 1. Hash every private source file into a denylist kept outside the repository.
python3 scripts/privacy_guard.py build --root /external/private-data --out /external/guard/denylist.sha256
# 2. Install pre-commit and pre-push hooks for this repository.
python3 scripts/privacy_guard.py install --denylist /external/guard/denylist.sha256
# 3. Or check by hand.
python3 scripts/privacy_guard.py check --denylist /external/guard/denylist.sha256 --staged
python3 scripts/privacy_guard.py check --denylist /external/guard/denylist.sha256 --range 'HEAD ^origin/main'
```

It blocks a denylisted SHA-256 (also as a Git LFS oid), restricted file names,
any binary or image payload without an explicit `--allow` glob, embedded
base64 images and files over 1 MB. It fails closed when the denylist is
missing or empty. Rebuild the denylist whenever private roots change.

A hash cannot recognise a cropped or re-encoded derivative. The binary rule is
what stops those, so add an `--allow` glob only for a file you generated from
synthetic inputs and have looked at. Hooks are local and can be bypassed with
`--no-verify`; treat that as a policy violation, not an option.
