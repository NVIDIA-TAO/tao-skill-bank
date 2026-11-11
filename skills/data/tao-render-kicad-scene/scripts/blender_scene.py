#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build an editable Blender scene from a KiCad GLB export plus a scene JSON, then render.

Run with Blender's bundled Python. KiCad owns board/footprint/pad/track geometry;
this script adds what KiCad does not provide (solder, dome lighting, camera) from
explicit parameters, checks CAD/GLB pad alignment, and writes scene.blend, rgb.png,
entity IDs and render_receipt.json. Lengths in the JSON are millimetres in the
KiCad board frame (+Y down); Blender world is metres with Y flipped exactly once.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector

SCHEMA = "tao.kicad_light.render.v1"
MM = 0.001
DEFAULTS = {
    "render": {"resolution": [640, 480], "samples": 64, "seed": 17, "view_transform": "AgX", "exposure": 0.0},
    "camera": {"type": "ORTHO", "target_mm": [0.0, 0.0], "ortho_scale_mm": 8.0, "lens_mm": 50.0,
               "elevation_deg": 90.0, "azimuth_deg": 0.0, "distance_mm": 20.0},
    "lighting": {"world_strength": 0.25, "world_color": [0.5, 0.5, 0.5], "area": [], "rgb_dome": None},
    "assets": [],
    "materials": {},
    "pad_alignment": {"mode": "require", "tolerance_mm": 0.001},
    "id_mask": True,
}


def world_xy(point_mm):
    return Vector((point_mm[0] * MM, -point_mm[1] * MM))


def merged(base, override):
    out = dict(base)
    for key, value in (override or {}).items():
        out[key] = merged(base[key], value) if isinstance(value, dict) and isinstance(base.get(key), dict) else value
    return out


def bounds(obj):
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    return (Vector((min(p[i] for p in pts) for i in range(3))), Vector((max(p[i] for p in pts) for i in range(3))))


def islands(obj):
    """Disconnected mesh islands by shared vertex positions; independent of CAD pad centres."""
    parent = list(range(len(obj.data.vertices)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    seen = {}
    for vertex in obj.data.vertices:
        key = tuple(round(float(x) * 1e9) for x in vertex.co)
        if key in seen:
            parent[find(vertex.index)] = find(seen[key])
        else:
            seen[key] = vertex.index
    for face in obj.data.polygons:
        for index in face.vertices[1:]:
            parent[find(face.vertices[0])] = find(index)
    groups = {}
    for face in obj.data.polygons:
        groups.setdefault(find(face.vertices[0]), set()).update(face.vertices)
    out = []
    for indices in groups.values():
        xyz = np.array([list(obj.matrix_world @ obj.data.vertices[i].co) for i in indices])
        out.append({"min": xyz.min(axis=0), "max": xyz.max(axis=0)})
    return out


def check_pads(scene, cad, policy):
    """Each CAD pad centre must coincide with the bbox centre of one exported pad island."""
    if not cad["pads"]:
        return {"status": "not_applicable", "reason": "board has no pads", "pads": [], "max_error_mm": None}
    meshes = [o for o in scene.objects if o.type == "MESH" and o.data.name.split(".")[0].endswith("_pad")]
    if len(meshes) != 1:
        raise RuntimeError("Expected one exported pad mesh, got " + repr([o.data.name for o in meshes]))
    found = islands(meshes[0])
    centers = np.array([(i["min"] + i["max"]) / 2 for i in found])
    rows, used = [], set()
    for pad in cad["pads"]:
        want = np.array(world_xy(pad["center_mm"]))
        dist = np.linalg.norm(centers[:, :2] - want, axis=1)
        index = int(dist.argmin())
        rows.append({"id": pad["id"], "error_mm": round(float(dist[index] / MM), 9),
                     "top_z_mm": round(float(found[index]["max"][2] / MM), 6)})
        used.add(index)
    worst = max(r["error_mm"] for r in rows)
    ok = worst <= policy["tolerance_mm"] and len(used) == len(rows) == len(found)
    result = {"status": "pass" if ok else "fail", "max_error_mm": worst, "tolerance_mm": policy["tolerance_mm"],
              "cad_pads": len(rows), "glb_pad_islands": len(found), "pads": rows,
              "scope": "XY centre alignment of CAD pads vs exported copper; not solder contact or stackup Z"}
    if not ok and policy["mode"] == "require":
        raise RuntimeError("CAD/GLB pad alignment failed: " + json.dumps({k: result[k] for k in
                           ("max_error_mm", "cad_pads", "glb_pad_islands")}))
    return result


def surface(name, color, roughness, metallic=0.0, specular=0.5):
    """Principled material; `specular` 0 gives a matte surface with no white-world sheen."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Specular IOR Level"].default_value = specular
    return mat


def finish_asset(scene, name, verts, faces, material, facts):
    """Link a generated mesh, require a closed positive volume, and record its geometry hash."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], faces)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    closed = all(len(e.link_faces) == 2 for e in bm.edges)
    volume = bm.calc_volume(signed=True) / MM ** 3
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(material)
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    obj["entity_id"] = name
    for key, value in facts.items():
        obj["recipe_" + key] = json.dumps(value) if isinstance(value, (list, dict)) else value
    flat = [i for face in faces for i in (*face, -1)]  # faces may mix quads and n-gons
    digest = hashlib.sha256(np.round(np.array([tuple(v) for v in verts]) / MM, 5).tobytes()
                            + np.array(flat, dtype=np.int32).tobytes()).hexdigest()
    facts.update(status="built", closed_manifold=closed, volume_mm3=round(volume, 7), geometry_sha256=digest,
                 units="mm", vertices=len(verts), faces=len(faces))
    if not closed or volume <= 0:
        raise RuntimeError("Asset is not a closed positive volume: " + name)
    return obj, facts


def box_asset(scene, spec, board_top):
    """Custom rectangular body (die, block, foreign object) for parts KiCad has no model for.

    `rotation_deg` is counter-clockwise as seen on screen; `corner_cut_mm` chamfers the
    top-left corner in the body's own frame to model a broken edge.
    """
    name = "asset:" + spec["name"]
    w, h, t = spec["size_mm"][0], spec["size_mm"][1], spec["height_mm"]
    cut = float(spec.get("corner_cut_mm", 0.0))
    outline = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    if cut > 0:  # KiCad frame, +Y down: (-w/2, -h/2) is the top-left corner on screen
        cut = min(cut, 0.9 * min(w, h))
        outline = [(-w / 2 + cut, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2), (-w / 2, -h / 2 + cut)]
    angle = math.radians(spec.get("rotation_deg", 0.0))
    centre = world_xy(spec["center_mm"])
    base = board_top + spec.get("z_offset_mm", 0.0) * MM
    ring = []
    for x, y in outline:  # rotate in the KiCad frame (screen CCW), then flip Y into the world
        rx, ry = x * math.cos(angle) + y * math.sin(angle), -x * math.sin(angle) + y * math.cos(angle)
        ring.append((centre.x + rx * MM, centre.y - ry * MM))
    n = len(ring)
    verts = [(x, y, base) for x, y in ring] + [(x, y, base + t * MM) for x, y in ring]
    faces = [tuple(range(n)), tuple(range(n, 2 * n))] + [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    facts = {"type": "box", "name": spec["name"], "center_mm": spec["center_mm"], "size_mm": [w, h],
             "height_mm": t, "rotation_deg": spec.get("rotation_deg", 0.0), "corner_cut_mm": cut,
             "generator": "blender_scene.box_asset/v1"}
    material = surface(name, spec.get("color", [0.8, 0.8, 0.8]), spec.get("roughness", 0.5),
                       spec.get("metallic", 0.0), spec.get("specular", 0.5))
    return finish_asset(scene, name, verts, faces, material, facts)


def solder_fillet(scene, pad, top_z, body, params, material):
    """Closed height-field solid on one pad, wetting the component end face.

    amount scales the toe height against the terminal height (1.0 = nominal, <1 starved,
    >1 convex/excess, capped at the terminal top); wetting scales how far the toe
    spreads toward the outer pad edge. Returns (object, facts) or (None, facts).
    """
    amount, wetting = float(params.get("amount", 1.0)), float(params.get("wetting", 1.0))
    facts = {"type": "solder_fillet", "pad": pad["id"], "amount": amount, "wetting": wetting,
             "generator": "blender_scene.solder_fillet/v1"}
    if amount <= 0 or wetting <= 0:
        return None, dict(facts, status="omitted", reason="amount or wetting is zero")
    along_x = pad["local_axis"] == "x"
    half_a = (pad["size_mm"][0] if along_x else pad["size_mm"][1]) / 2
    half_b = (pad["size_mm"][1] if along_x else pad["size_mm"][0]) / 2
    end = pad["body_end_offset_mm"] if pad["body_end_offset_mm"] is not None else 0.0
    end = max(-half_a * 0.9, min(half_a * 0.9, end))
    term_h = (body[1].z - top_z) / MM          # pad top to terminal top
    gap = max((body[0].z - top_z) / MM, 0.0)    # pad top to terminal underside
    peak = min(amount, 1.0) * 0.75 * term_h + max(amount - 1.0, 0.0) * 0.5 * term_h
    peak = min(peak, term_h)
    power = 2.0 if amount <= 1.0 else max(0.6, 2.0 - 2.8 * (amount - 1.0))
    toe = (half_a - end) * min(wetting, 1.0)
    skin = 0.004
    na, nb = int(params.get("grid", 28)), int(params.get("grid", 28))
    a_dir = Vector((pad["outward_unit"][0], -pad["outward_unit"][1], 0)).normalized()
    b_dir = Vector((-a_dir.y, a_dir.x, 0))
    origin = Vector((*world_xy(pad["center_mm"]), top_z))
    verts, faces = [], []
    s_min, s_max = -half_a, min(half_a, end + toe)  # solder covers only the wetted part of the pad
    s_mid, s_half = (s_min + s_max) / 2, (s_max - s_min) / 2
    for layer in (1, 0):  # top sheet first, then bottom sheet
        for i in range(na + 1):
            s = s_min + 2 * s_half * i / na
            for j in range(nb + 1):
                t = -half_b + 2 * half_b * j / nb
                edge = (1 - abs(t / half_b) ** 4) * (1 - abs((s - s_mid) / s_half) ** 8)
                if s < end:
                    height = min(peak, gap) * edge
                else:
                    r = (s - end) / toe if toe > 0 else 2.0
                    height = peak * (1 - r) ** power * edge if r < 1 else 0.0
                z = (skin + height) if layer else -0.001
                verts.append(origin + a_dir * (s * MM) + b_dir * (t * MM) + Vector((0, 0, z * MM)))
    stride, count = nb + 1, (na + 1) * (nb + 1)
    for i in range(na):
        for j in range(nb):
            v = i * stride + j
            faces.append((v, v + stride, v + stride + 1, v + 1))
            faces.append((count + v, count + v + 1, count + v + stride + 1, count + v + stride))
    rim = [i * stride for i in range(na + 1)] + [na * stride + j for j in range(1, nb + 1)] + \
          [i * stride + nb for i in range(na - 1, -1, -1)] + [j for j in range(nb - 1, 0, -1)]
    for v, w in zip(rim, rim[1:] + rim[:1]):
        faces.append((v, w, count + w, count + v))
    obj, facts = finish_asset(scene, "solder:" + pad["id"], verts, faces, material, facts)
    for poly in obj.data.polygons:
        poly.use_smooth = True
    facts.update(peak_height_mm=round(peak, 5), terminal_height_mm=round(term_h, 5), toe_length_mm=round(toe, 5),
                 wetted_length_mm=round(s_max - s_min, 5), pad_length_mm=round(2 * half_a, 5),
                 touches_terminal_face=bool(peak > 0 and end < half_a))
    return obj, facts


def add_dome(scene, target, dome):
    """RGB AOI dome as emissive spherical bands (elevation measured from the board plane)."""
    made = []
    radius = dome.get("radius_mm", 30.0) * MM
    for band in dome["bands"]:
        lo = math.radians(band["elevation_deg"] - band["width_deg"] / 2)
        hi = math.radians(band["elevation_deg"] + band["width_deg"] / 2)
        lo, hi = max(lo, math.radians(1)), min(hi, math.radians(89.5))
        seg, verts, faces = 96, [], []
        for elev in (lo, hi):
            for k in range(seg):
                az = 2 * math.pi * k / seg
                verts.append((target.x + radius * math.cos(elev) * math.cos(az),
                              target.y + radius * math.cos(elev) * math.sin(az), target.z + radius * math.sin(elev)))
        for k in range(seg):
            faces.append((k, (k + 1) % seg, seg + (k + 1) % seg, seg + k))
        mesh = bpy.data.meshes.new("dome:" + band["name"])
        mesh.from_pydata(verts, [], faces)
        mat = bpy.data.materials.new("dome:" + band["name"])
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        emit, out = nodes.new("ShaderNodeEmission"), nodes.new("ShaderNodeOutputMaterial")
        emit.inputs["Color"].default_value = (*band["color"], 1)
        emit.inputs["Strength"].default_value = band["strength"]
        mat.node_tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
        mesh.materials.append(mat)
        obj = bpy.data.objects.new("dome:" + band["name"], mesh)
        scene.collection.objects.link(obj)
        obj.visible_camera = False
        made.append(obj.name)
    return made


def id_mask(scene, camera, width, height, out_dir):
    """Unfiltered entity ID at each pixel centre (nearest hit; material alpha ignored)."""
    entities = sorted((o for o in scene.objects if o.type == "MESH" and "entity_id" in o), key=lambda o: o["entity_id"])
    for index, obj in enumerate(entities, 1):
        obj.pass_index = index
    depsgraph = bpy.context.evaluated_depsgraph_get()
    frame = [camera.matrix_world @ v for v in camera.data.view_frame(scene=scene)]  # TR, BR, BL, TL
    tr, br, bl, tl = frame
    forward = (camera.matrix_world.to_3x3() @ Vector((0, 0, -1))).normalized()
    eye = camera.matrix_world.translation.copy()
    ids = np.zeros((height, width), dtype=np.uint16)
    for y in range(height):
        fy = (y + 0.5) / height
        left, right = tl.lerp(bl, fy), tr.lerp(br, fy)
        for x in range(width):
            p = left.lerp(right, (x + 0.5) / width)
            if camera.data.type == "ORTHO":  # view_frame sits on the far clip plane: project back to the camera
                origin, direction = p - forward * (p - eye).dot(forward), forward
            else:
                origin, direction = eye, (p - eye).normalized()
            hit, _, _, _, obj, _ = scene.ray_cast(depsgraph, origin, direction, distance=1.0)
            if hit and "entity_id" in obj.original:
                ids[y, x] = obj.original.pass_index
    np.save(out_dir / "entity_ids.npy", ids, allow_pickle=False)
    names = {str(o.pass_index): o["entity_id"] for o in entities}
    counts = {names[str(i)]: int(c) for i, c in zip(*np.unique(ids[ids > 0], return_counts=True))}
    if not counts:
        raise RuntimeError("ID mask is empty: no camera ray hit a tagged entity")
    return {"file": "entity_ids.npy", "dtype": "uint16", "background": 0, "entities": names, "pixel_counts": counts,
            "policy": "nearest geometry at pixel centre; alpha ignored; no AA; not a validated consumer mask"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--cad", type=Path, required=True, help="cad_metadata.json written by kicad_sdk.save")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "scene.blend").exists():
        raise RuntimeError("Refusing to overwrite an existing scene; use a new output directory")
    cad = json.loads(args.cad.read_text())
    spec = merged(DEFAULTS, json.loads(args.scene.read_text()))
    started = time.perf_counter()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.glb.resolve()))
    scene = bpy.context.scene
    refs = {f["refdes"] for f in cad["footprints"]}
    bodies = {}
    for obj in [o for o in scene.objects if o.type == "MESH"]:
        if obj.parent and obj.parent.name in refs:
            obj["entity_id"] = "component:" + obj.parent.name
            bodies[obj.parent.name] = bounds(obj)
        else:
            obj["entity_id"] = "board:" + obj.data.name
    alignment = check_pads(scene, cad, spec["pad_alignment"])
    pad_top = {r["id"]: r["top_z_mm"] * MM for r in alignment["pads"]}
    board_top = max(pad_top.values()) if pad_top else max(
        bounds(o)[1].z for o in scene.objects if o.type == "MESH" and o["entity_id"].startswith("board:"))

    # Custom assets: later entries override earlier ones for the same pad.
    solder_spec = spec["materials"].get("solder", {})
    solder_mat = surface("solder", solder_spec.get("color", [0.78, 0.78, 0.8]), solder_spec.get("roughness", 0.22), 1.0)
    plan, boxes = {}, []
    for entry in spec["assets"]:
        if entry["type"] == "box":
            boxes.append(entry)
            continue
        if entry["type"] != "solder_fillet":
            raise RuntimeError("Unsupported asset type: " + entry["type"])
        ids = [p["id"] for p in cad["pads"]] if entry.get("pads", "all") == "all" else entry["pads"]
        for pad_id in ids:
            plan[pad_id] = merged(plan.get(pad_id, {}), {k: v for k, v in entry.items() if k not in ("type", "pads")})
    pads = {p["id"]: p for p in cad["pads"]}
    assets = []
    for pad_id in sorted(plan):
        pad = pads[pad_id]
        if pad["refdes"] not in bodies:
            assets.append({"type": "solder_fillet", "pad": pad_id, "status": "unsupported", "reason": "no 3D body"})
            continue
        _, facts = solder_fillet(scene, pad, pad_top[pad_id], bodies[pad["refdes"]], plan[pad_id], solder_mat)
        assets.append(facts)
    for entry in boxes:
        assets.append(box_asset(scene, entry, board_top)[1])
    # Appearance overrides for exported KiCad meshes, keyed by entity-ID prefix.
    overrides = []
    for prefix, look in sorted(spec["materials"].get("overrides", {}).items()):
        mat = surface("override:" + prefix, look["color"], look.get("roughness", 0.6), look.get("metallic", 0.0),
                      look.get("specular", 0.5))
        hits = [o for o in scene.objects if o.type == "MESH" and str(o.get("entity_id", "")).startswith(prefix)]
        for obj in hits:
            obj.data.materials.clear()
            obj.data.materials.append(mat)
        overrides.append({"prefix": prefix, "objects": len(hits), **look})

    render, cam = spec["render"], spec["camera"]
    width, height = render["resolution"]
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = render["samples"]
    scene.cycles.seed = render["seed"]
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = width, height, 100
    scene.render.film_transparent = False
    scene.view_settings.view_transform = render["view_transform"]
    scene.view_settings.look = "None"
    scene.view_settings.exposure = render["exposure"]
    scene.view_settings.gamma = 1
    for prop in scene.render.bl_rna.properties:  # no date/host metadata: PNG bytes stay reproducible
        if prop.identifier.startswith("use_stamp"):
            setattr(scene.render, prop.identifier, False)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.compression = 15
    scene.render.filepath = str((args.output / "rgb.png").resolve())

    target = Vector((*world_xy(cam["target_mm"]), board_top))
    elev, azim = math.radians(cam["elevation_deg"]), math.radians(cam["azimuth_deg"])
    offset = Vector((math.cos(elev) * math.cos(azim), math.cos(elev) * math.sin(azim), math.sin(elev)))
    cam_data = bpy.data.cameras.new("camera")
    camera = bpy.data.objects.new("camera", cam_data)
    scene.collection.objects.link(camera)
    camera.location = target + offset * cam["distance_mm"] * MM
    # Explicit look-at: top-down keeps image-up = world +Y = KiCad -Y, i.e. KiCad's on-screen view.
    forward = -offset
    hint = Vector((0, 1, 0)) if abs(cam["elevation_deg"]) > 89.9 else Vector((0, 0, 1))
    right = forward.cross(hint).normalized()
    up = right.cross(forward).normalized()
    camera.rotation_euler = Matrix((right, up, -forward)).transposed().to_euler()
    cam_data.type = cam["type"]
    cam_data.sensor_fit = "HORIZONTAL"  # ortho_scale_mm is always the image width, also for portrait frames
    cam_data.ortho_scale = cam["ortho_scale_mm"] * MM
    cam_data.lens = cam["lens_mm"]
    cam_data.clip_start, cam_data.clip_end = 1e-5, 1.0
    scene.camera = camera

    light = spec["lighting"]
    scene.world = bpy.data.worlds.new("world")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (*light["world_color"], 1)
    background.inputs["Strength"].default_value = light["world_strength"]
    for index, area in enumerate(light["area"]):
        data = bpy.data.lights.new(area.get("name", f"area{index}"), "AREA")
        data.energy, data.size = area["power_w"], area["size_mm"] * MM
        data.color = area.get("color", [1, 1, 1])
        obj = bpy.data.objects.new(data.name, data)
        scene.collection.objects.link(obj)
        obj.location = Vector((*world_xy(area["position_mm"][:2]), board_top + area["position_mm"][2] * MM))
        obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()
    dome = add_dome(scene, target, light["rgb_dome"]) if light["rgb_dome"] else []
    bpy.context.view_layer.update()
    setup_seconds = time.perf_counter() - started

    bpy.ops.wm.save_as_mainfile(filepath=str((args.output / "scene.blend").resolve()))
    t0 = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    render_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    mask = id_mask(scene, camera, width, height, args.output) if spec["id_mask"] else None
    receipt = {
        "schema_version": SCHEMA, "blender_version": bpy.app.version_string,
        "build_hash": bpy.app.build_hash.decode(), "engine": "CYCLES", "device": "CPU", "gpu_used": False,
        "threads": scene.render.threads, "resolved_scene": spec,
        "transform_world_from_kicad_mm": [[MM, 0, 0, 0], [0, -MM, 0, 0], [0, 0, MM, 0], [0, 0, 0, 1]],
        "camera_world": {"location_m": list(camera.location), "rotation_euler_rad": list(camera.rotation_euler)},
        "board_top_z_mm": board_top / MM, "pad_alignment": alignment, "assets": assets,
        "material_overrides": overrides, "dome_objects": dome,
        "entity_ids": mask, "outputs": ["rgb.png", "scene.blend"] + (["entity_ids.npy"] if mask else []),
        "timing_seconds": {"import_and_build": round(setup_seconds, 3), "cycles_render": round(render_seconds, 3),
                           "id_mask": round(time.perf_counter() - t0, 3)},
    }
    (args.output / "render_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"assets": [(a.get("pad") or a.get("name"), a["status"], a.get("volume_mm3")) for a in assets],
                      "pad_alignment": alignment["status"], "max_error_mm": alignment["max_error_mm"],
                      "timing_seconds": receipt["timing_seconds"]}))


if __name__ == "__main__":
    main()
