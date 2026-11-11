# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small pcbnew conveniences for the error-prone parts: units, library lookup, metadata.

Import from a script run by KiCad's bundled interpreter (`AppRun python3.11`).
Everything else is plain pcbnew; call the SDK directly for anything not here.
"""
import hashlib
import json
import math
import os
from pathlib import Path

import pcbnew

SCHEMA = "tao.kicad_light.cad.v1"
_REMOVED = []  # wrappers of items taken off a board; see take_copper


def mm(value):
    return pcbnew.FromMM(float(value))


def point(x, y):
    return pcbnew.VECTOR2I(mm(x), mm(y))


def appdir():
    return Path(os.environ["KICAD_APPDIR"])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_ids(seed):
    """Make KiCad's UUID stream deterministic so a replay writes byte-identical boards.

    KiCad saves items in UUID order, so random IDs also reorder the file and the GLB.
    Call once, before any item is created or loaded from a library.
    """
    pcbnew.KIID.SeedGenerator(int(seed))


def new_board(outline_mm, thickness_mm=1.6):
    """Empty board with a closed Edge.Cuts polygon given as [(x, y), ...] in mm."""
    board = pcbnew.BOARD()
    board.GetDesignSettings().SetBoardThickness(mm(thickness_mm))
    for start, end in zip(outline_mm, outline_mm[1:] + outline_mm[:1]):
        edge = pcbnew.PCB_SHAPE(board)
        edge.SetShape(pcbnew.SHAPE_T_SEGMENT)
        edge.SetStart(point(*start))
        edge.SetEnd(point(*end))
        edge.SetLayer(pcbnew.Edge_Cuts)
        edge.SetWidth(mm(0.05))
        board.Add(edge)
    return board


def place(board, library, name, ref, x, y, angle_deg=0.0, model=True):
    """Load a standard-library footprint, bind its STEP model by absolute path, place it."""
    lib = appdir() / "share/kicad/footprints" / (library + ".pretty")
    fp = pcbnew.FootprintLoad(str(lib), name)
    if fp is None:
        raise RuntimeError(f"Footprint not found: {library}:{name}")
    fp.SetReference(ref)
    fp.SetPosition(point(x, y))
    fp.SetOrientationDegrees(angle_deg)
    fp.Reference().SetVisible(False)
    fp.Value().SetVisible(False)
    if model:
        step = appdir() / "share/kicad/3dmodels" / (library + ".3dshapes") / (name + ".step")
        models = list(fp.Models())
        if not step.is_file() or len(models) != 1:
            raise RuntimeError(f"No single standard 3D model for {library}:{name}; build a custom asset instead")
        models[0].m_Filename = str(step)
    board.Add(fp)
    return fp


def smd_footprint(board, ref, pads, x=0.0, y=0.0, angle_deg=0.0):
    """Build a footprint from scratch when no library part fits.

    `pads` is [(number, dx, dy, width, height), ...] in mm relative to the footprint
    origin, before rotation. Pads are rectangular SMD pads on F.Cu with mask and paste.
    The footprint has no 3D model: add the body as a custom asset in the scene JSON.
    """
    fp = pcbnew.FOOTPRINT(board)
    fp.SetReference(ref)
    fp.Reference().SetVisible(False)
    fp.Value().SetVisible(False)
    for number, dx, dy, width, height in pads:
        pad = pcbnew.PAD(fp)
        pad.SetNumber(str(number))
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        pad.SetShape(pcbnew.F_Cu, pcbnew.PAD_SHAPE_RECT)
        pad.SetSize(pcbnew.F_Cu, pcbnew.VECTOR2I(mm(width), mm(height)))
        pad.SetPosition(point(dx, dy))
        fp.Add(pad)
    board.Add(fp)
    fp.SetPosition(point(x, y))
    fp.SetOrientationDegrees(angle_deg)
    return fp


def find(board, ref):
    fp = board.FindFootprintByReference(ref)
    if fp is None:
        raise RuntimeError("No footprint " + ref)
    return fp


def connect(board, net_name, pads, width_mm=0.2, via_points_mm=()):
    """Put pads on one net and join them with straight F.Cu segments through optional waypoints."""
    net = pcbnew.NETINFO_ITEM(board, net_name)
    board.Add(net)
    stops = [(pcbnew.ToMM(p.GetPosition().x), pcbnew.ToMM(p.GetPosition().y)) for p in pads]
    for pad in pads:
        pad.SetNet(net)
    route = [stops[0], *via_points_mm, *stops[1:]]
    for start, end in zip(route, route[1:]):
        track = pcbnew.PCB_TRACK(board)
        track.SetStart(point(*start))
        track.SetEnd(point(*end))
        track.SetWidth(mm(width_mm))
        track.SetLayer(pcbnew.F_Cu)
        track.SetNet(net)
        board.Add(track)


def add_copper(board, polys):
    """Add a SHAPE_POLY_SET to F.Cu as one filled polygon per island; returns the island count.

    The board file stores a single outline per polygon, so holes are fractured into it.
    """
    for index in range(polys.OutlineCount()):
        island = polys.UnitSet(index)
        island.Fracture()
        shape = pcbnew.PCB_SHAPE(board)
        shape.SetShape(pcbnew.SHAPE_T_POLY)
        shape.SetPolyShape(island)
        shape.SetFilled(True)
        shape.SetWidth(0)
        shape.SetLayer(pcbnew.F_Cu)
        board.Add(shape)
    return polys.OutlineCount()


def copper_shapes(board):
    return [d for d in board.GetDrawings() if d.GetClass() == "PCB_SHAPE" and d.GetLayer() == pcbnew.F_Cu
            and d.GetShape() == pcbnew.SHAPE_T_POLY]


def take_copper(board):
    """Remove the F.Cu polygons from a board and return (their union, area in mm2)."""
    union = pcbnew.SHAPE_POLY_SET()
    for shape in copper_shapes(board):
        union.BooleanAdd(shape.GetPolyShape())
        board.Remove(shape)
        # pcbnew 10.0.5: once the Python wrapper of a removed item is freed, SHAPE_POLY_SET
        # methods such as UnitSet() return opaque objects. Keep the wrapper alive instead.
        _REMOVED.append(shape)
    union.Simplify()
    return union, union.Area() / pcbnew.FromMM(1) ** 2


def _local(fp, pos):
    """Board point -> footprint frame (KiCad Y is down; orientation is CCW on screen)."""
    theta = math.radians(fp.GetOrientationDegrees())
    dx = pcbnew.ToMM(pos.x - fp.GetPosition().x)
    dy = pcbnew.ToMM(pos.y - fp.GetPosition().y)
    return (dx * math.cos(theta) - dy * math.sin(theta), dx * math.sin(theta) + dy * math.cos(theta))


def _body_bbox(fp):
    """Body outline from straight F.Fab graphics in the footprint frame, or None if absent."""
    points = []
    for item in fp.GraphicalItems():
        if item.GetLayer() != pcbnew.F_Fab or item.GetClass() != "PCB_SHAPE":
            continue
        if item.GetShape() in (pcbnew.SHAPE_T_SEGMENT, pcbnew.SHAPE_T_RECTANGLE):
            points += [_local(fp, item.GetStart()), _local(fp, item.GetEnd())]
        elif item.GetShape() == pcbnew.SHAPE_T_POLY:  # rectangles become polygons once rotated
            poly = item.GetPolyShape()
            for index in range(poly.OutlineCount()):
                outline = poly.Outline(index)
                points += [_local(fp, outline.CPoint(j)) for j in range(outline.PointCount())]
    if not points:
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return [round(min(xs), 6), round(min(ys), 6), round(max(xs), 6), round(max(ys), 6)]


def describe(board):
    """Read back what the SDK actually stored: the facts later steps align against."""
    footprints, pads, tracks = [], [], []
    for fp in board.GetFootprints():
        ref = fp.GetReference()
        body = _body_bbox(fp)
        models = [m.m_Filename for m in fp.Models()]
        footprints.append({
            "refdes": ref, "footprint": str(fp.GetFPID().GetUniStringLibItemName()),
            "position_mm": [pcbnew.ToMM(fp.GetPosition().x), pcbnew.ToMM(fp.GetPosition().y)],
            "angle_deg": fp.GetOrientationDegrees(), "side": "bottom" if fp.IsFlipped() else "top",
            "body_bbox_local_mm": body, "body_source": "F.Fab straight graphics" if body else None,
            "models": [{"path": m, "sha256": sha(m) if Path(m).is_file() else None} for m in models],
            "uuid": fp.m_Uuid.AsString()})
        for pad in fp.Pads():
            cx, cy = pcbnew.ToMM(pad.GetPosition().x), pcbnew.ToMM(pad.GetPosition().y)
            lx, ly = _local(fp, pad.GetPosition())
            dx, dy = cx - footprints[-1]["position_mm"][0], cy - footprints[-1]["position_mm"][1]
            norm = math.hypot(dx, dy)
            along_x = abs(lx) >= abs(ly)
            reach = None
            if body:
                half = max(abs(body[0]), abs(body[2])) if along_x else max(abs(body[1]), abs(body[3]))
                reach = round(half - (abs(lx) if along_x else abs(ly)), 6)
            pads.append({
                "id": f"{ref}:{pad.GetNumber()}", "refdes": ref, "number": pad.GetNumber(),
                "center_mm": [cx, cy], "size_mm": [pcbnew.ToMM(pad.GetSize().x), pcbnew.ToMM(pad.GetSize().y)],
                "angle_deg": pad.GetOrientationDegrees(), "net": pad.GetNetname(),
                "outward_unit": [round(dx / norm, 9), round(dy / norm, 9)] if norm else None,
                # Signed distance from pad centre to the body end face along `outward_unit`.
                "body_end_offset_mm": reach, "local_axis": "x" if along_x else "y",
                "uuid": pad.m_Uuid.AsString()})
    for track in board.GetTracks():
        tracks.append({"net": track.GetNetname(), "layer": board.GetLayerName(track.GetLayer()),
                       "start_mm": [pcbnew.ToMM(track.GetStart().x), pcbnew.ToMM(track.GetStart().y)],
                       "end_mm": [pcbnew.ToMM(track.GetEnd().x), pcbnew.ToMM(track.GetEnd().y)],
                       "width_mm": pcbnew.ToMM(track.GetWidth())})
    shapes = copper_shapes(board)
    copper_union = pcbnew.SHAPE_POLY_SET()
    for shape in shapes:
        copper_union.BooleanAdd(shape.GetPolyShape())
    copper = {"layer": "F.Cu", "polygons": len(shapes),
              "vertices": sum(s.GetPolyShape().FullPointCount() for s in shapes),
              "area_mm2": round(copper_union.Area() / pcbnew.FromMM(1) ** 2, 6)} if shapes else None
    box = board.GetBoardEdgesBoundingBox()
    return {"schema_version": SCHEMA, "kicad_version": pcbnew.GetBuildVersion(),
            "sdk": "pcbnew SWIG bindings", "units": "mm; KiCad board frame, +Y down",
            "board_thickness_mm": pcbnew.ToMM(board.GetDesignSettings().GetBoardThickness()),
            "outline_bbox_mm": [pcbnew.ToMM(box.GetLeft()), pcbnew.ToMM(box.GetTop()),
                                pcbnew.ToMM(box.GetRight()), pcbnew.ToMM(box.GetBottom())],
            "footprints": sorted(footprints, key=lambda f: f["refdes"]),
            "pads": sorted(pads, key=lambda p: p["id"]), "copper": copper,
            "tracks": sorted(tracks, key=lambda t: (t["net"], t["start_mm"], t["end_mm"]))}


def save(board, out_dir, name="board", extra=None):
    """Write the editable board, reload it, and store the read-back metadata beside it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (name + ".kicad_pcb")
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite {path}; write each revision to a new directory")
    pcbnew.SaveBoard(str(path), board)
    meta = describe(pcbnew.LoadBoard(str(path)))
    meta["board_file"] = path.name
    meta["board_sha256"] = sha(path)
    meta.update(extra or {})
    (out_dir / "cad_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return path, meta
