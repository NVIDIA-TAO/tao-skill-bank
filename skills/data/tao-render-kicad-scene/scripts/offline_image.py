#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline reader: turn a reference image into numbers the agent may see.

Runs on the host with numpy and Pillow only. It prints measurements and scores
as JSON and never prints, uploads or re-encodes pixels. Files it writes
(layouts traced from a reference) are derived private data: keep them outside
every repository.

  stats    size, value distribution and two-level check
  segments colour clusters with the size, position and orientation of their largest region
  layout   trace a two-level bare-board image into pixel-exact copper rectangles
  compare  score a render's entity-ID mask against the reference mask
  region   score one rendered entity against a thresholded reference region (shape, place, gray, SSIM)
  diff     compare two renders and test the change against ground-truth boxes
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gray(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float64), image.mode, image.size


def otsu(values):
    hist = np.bincount(values.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    total, levels = hist.sum(), np.arange(256)
    w0 = np.cumsum(hist)
    m0 = np.cumsum(hist * levels)
    w1 = total - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (m0[-1] * w0 - m0 * total) ** 2 / (w0 * w1)
    return int(np.nanargmax(between))


def runs(mask):
    """Horizontal runs per row as (y, x0, x1) with x1 exclusive."""
    padded = np.zeros((mask.shape[0], mask.shape[1] + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    delta = np.diff(padded, axis=1)
    out = []
    for y in range(mask.shape[0]):
        starts, ends = np.flatnonzero(delta[y] == 1), np.flatnonzero(delta[y] == -1)
        out.extend((y, int(a), int(b)) for a, b in zip(starts, ends))
    return out


def components(mask):
    """4-connected components from runs: list of {area, bbox, touches_border}."""
    items = runs(mask)
    parent = list(range(len(items)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    by_row = {}
    for index, (y, _, _) in enumerate(items):
        by_row.setdefault(y, []).append(index)
    for y, row in by_row.items():
        for i in row:
            for j in by_row.get(y - 1, ()):
                if items[i][1] < items[j][2] and items[j][1] < items[i][2]:
                    parent[find(i)] = find(j)
    height, width = mask.shape
    groups = {}
    for index, (y, x0, x1) in enumerate(items):
        g = groups.setdefault(find(index), {"area": 0, "bbox": [x0, y, x1, y + 1], "touches_border": False,
                                            "runs": []})
        g["runs"].append((y, x0, x1))
        g["area"] += x1 - x0
        b = g["bbox"]
        g["bbox"] = [min(b[0], x0), min(b[1], y), max(b[2], x1), max(b[3], y + 1)]
        g["touches_border"] |= x0 == 0 or x1 == width or y == 0 or y == height - 1
    return list(groups.values())


def rectangles(mask):
    """Merge identical runs on consecutive rows into [x0, y0, x1, y1) rectangles."""
    open_rects, done = {}, []
    for y in range(mask.shape[0] + 1):
        row = {(x0, x1) for yy, x0, x1 in runs(mask[y:y + 1])} if y < mask.shape[0] else set()
        for key in [k for k in open_rects if k not in row]:
            done.append([key[0], open_rects.pop(key), key[1], y])
        for key in row:
            open_rects.setdefault(key, y)
    return sorted(done, key=lambda r: (r[1], r[0]))


def enclosed(mask):
    """Small compact islands of `mask` that do not touch the border (drill-hole-like)."""
    parts = [c for c in components(mask) if not c["touches_border"]]
    compact = 0
    for c in parts:
        w, h = c["bbox"][2] - c["bbox"][0], c["bbox"][3] - c["bbox"][1]
        if c["area"] >= 12 and 0.6 <= w / h <= 1.67 and c["area"] / (w * h) >= 0.6:
            compact += 1
    return {"count": len(parts), "compact": compact}


def reference_mask(path, layout):
    values, _, _ = gray(path)
    bright = values > layout["threshold"]
    return bright if layout["copper"] == "bright" else ~bright


def render_mask(render_dir, prefixes):
    render_dir = Path(render_dir)
    receipt = json.loads((render_dir / "render_receipt.json").read_text())
    ids = np.load(render_dir / "entity_ids.npy")
    wanted = [int(i) for i, name in receipt["entity_ids"]["entities"].items() if name.startswith(tuple(prefixes))]
    if not wanted:
        raise SystemExit("No entity matches " + repr(prefixes) + "; have " +
                         repr(sorted(receipt["entity_ids"]["entities"].values())))
    return np.isin(ids, wanted)


def score(reference, candidate):
    inter, union = int((reference & candidate).sum()), int((reference | candidate).sum())
    wrong = reference != candidate
    edge = np.zeros_like(reference)
    edge[:, 1:] |= reference[:, 1:] != reference[:, :-1]
    edge[:, :-1] |= reference[:, 1:] != reference[:, :-1]
    edge[1:, :] |= reference[1:, :] != reference[:-1, :]
    edge[:-1, :] |= reference[1:, :] != reference[:-1, :]
    return {"iou": inter / union if union else 1.0, "pixel_agreement": float(1 - wrong.mean()),
            "mismatched_pixels": int(wrong.sum()), "mismatches_on_reference_edge": int((wrong & edge).sum()),
            "reference_fraction": float(reference.mean()), "candidate_fraction": float(candidate.mean())}


def emit(args, result):
    text = json.dumps(result, indent=2)
    if getattr(args, "out_json", None):
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text + "\n")
    print(text)


def cmd_stats(args):
    values, mode, size = gray(args.image)
    threshold = otsu(values)
    two_level = float(((values > 16) & (values < 240)).mean())
    emit(args, {"sha256": sha(args.image), "size_px": list(size), "mode": mode, "otsu_threshold": threshold,
                "bright_fraction": float((values > threshold).mean()), "mid_gray_fraction": two_level,
                "is_two_level": two_level < 0.01,
                "percentiles_p1_p50_p99": [float(np.percentile(values, p)) for p in (1, 50, 99)]})


def kmeans(points, k, seed=0, rounds=25):
    rng = np.random.default_rng(seed)
    centers = points[rng.choice(len(points), k, replace=False)].astype(np.float64)
    for _ in range(rounds):
        labels = ((points[:, None, :] - centers[None]) ** 2).sum(axis=2).argmin(axis=1)
        moved = np.array([points[labels == i].mean(axis=0) if (labels == i).any() else centers[i] for i in range(k)])
        if np.allclose(moved, centers, atol=0.25):
            break
        centers = moved
    return centers


def region_facts(mask, rank=0, parts=None):
    """The rank-th largest 4-connected part of a mask: area, bbox, fill, and PCA axes in pixels."""
    parts = components(mask) if parts is None else parts
    if len(parts) <= rank:
        return None
    top = sorted(parts, key=lambda c: -c["area"])[rank]
    x0, y0, x1, y1 = top["bbox"]
    pts = np.array([(x, y) for y, a, b in top["runs"] for x in range(a, b)], dtype=np.float64)
    centre = pts.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov((pts - centre).T)) if len(pts) > 2 else (np.zeros(2), np.eye(2))
    major = vectors[:, 1]
    return {"components": len(parts), "largest_area_px": top["area"], "largest_bbox_px": top["bbox"],
            "largest_bbox_fill": round(top["area"] / ((x1 - x0) * (y1 - y0)), 4),
            "largest_touches_border": top["touches_border"], "centroid_px": [round(float(c), 2) for c in centre],
            "major_axis_deg": round(float(np.degrees(np.arctan2(major[1], major[0]))) % 180, 2),
            # A filled rectangle of length L has standard deviation L / sqrt(12) along that side.
            "rect_equivalent_size_px": [round(float(np.sqrt(12 * max(v, 0))), 2) for v in values[::-1]]}


def cmd_segments(args):
    with Image.open(args.image) as image:
        native = image.size
        if args.crop:  # window in native pixels; reported boxes are relative to its corner
            image = image.convert("RGB").crop(tuple(args.crop))
        if args.max_side and max(image.size) > args.max_side:  # measure on a reduced copy held in memory only
            ratio = args.max_side / max(image.size)
            image = image.convert("RGB").resize((round(image.size[0] * ratio), round(image.size[1] * ratio)), Image.BOX)
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    height, width = rgb.shape[:2]
    flat = rgb.reshape(-1, 3)
    sample = flat[np.random.default_rng(0).choice(len(flat), min(len(flat), 40000), replace=False)]
    centers = kmeans(sample, args.k)
    labels = np.concatenate([((chunk[:, None, :] - centers[None]) ** 2).sum(axis=2).argmin(axis=1)
                             for chunk in np.array_split(flat, max(1, len(flat) // 200000))]).reshape(height, width)
    rows = []
    for index in np.argsort(-np.bincount(labels.ravel(), minlength=args.k)):
        mask = labels == index
        if not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        parts = components(mask)
        rows.append({"cluster": int(index), "fraction": round(float(mask.mean()), 4),
                     "mean_rgb": [round(float(c), 1) for c in rgb[mask].mean(axis=0)],
                     "bbox_px": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                     "largest_region": region_facts(mask, 0, parts),
                     "next_regions": [region_facts(mask, rank, parts) for rank in range(1, min(args.top, len(parts)))]})
    emit(args, {"sha256": sha(args.image), "native_size_px": list(native), "size_px": [width, height],
                "crop_px": args.crop, "k": args.k, "clusters": rows,
                "note": "k-means colour clusters; aggregated statistics only"})


def cmd_profile(args):
    """Ordered colour segments along one axis of a window, each with its extent across the axis."""
    x0, y0, x1, y1 = args.window
    with Image.open(args.image) as image:
        rgb = np.asarray(image.convert("RGB").crop((x0, y0, x1, y1)), dtype=np.float64)
    if args.axis == "y":
        rgb, (x0, y0) = rgb.transpose(1, 0, 2), (y0, x0)  # work along the first image axis as "x"
    height, width = rgb.shape[:2]
    flat = rgb.reshape(-1, 3)
    centers = kmeans(flat[np.random.default_rng(0).choice(len(flat), min(len(flat), 40000), replace=False)], args.k)
    labels = np.concatenate([((chunk[:, None, :] - centers[None]) ** 2).sum(axis=2).argmin(axis=1)
                             for chunk in np.array_split(flat, max(1, len(flat) // 200000))]).reshape(height, width)
    b0, b1 = (args.band[0] - y0, args.band[1] - y0) if args.band else (height // 2 - height // 16, height // 2 + height // 16)
    band = labels[b0:b1]
    dominant = np.array([np.bincount(band[:, x], minlength=args.k).argmax() for x in range(width)])
    spans, start = [], 0
    for x in range(1, width + 1):
        if x == width or dominant[x] != dominant[start]:
            spans.append([start, x, int(dominant[start])])
            start = x
    merged_spans = []
    for span in spans:  # fold runs shorter than --min-run into the run before them
        if merged_spans and (span[1] - span[0] < args.min_run or span[2] == merged_spans[-1][2]):
            merged_spans[-1][1] = span[1]
        else:
            merged_spans.append(span)
    gray_values = rgb.mean(axis=2)
    rows = []
    for a, b, cluster in merged_spans:
        occupancy = (labels[:, a:b] == cluster).mean(axis=1) >= 0.5
        lo = hi = (b0 + b1) // 2
        while lo > 0 and occupancy[lo - 1]:
            lo -= 1
        while hi < height and occupancy[hi]:
            hi += 1
        inside = rgb[b0:b1, a:b].reshape(-1, 3)
        rows.append({"along_px": [a + x0, b + x0], "length_px": b - a, "cluster": cluster,
                     "across_px": [lo + y0, hi + y0] if occupancy[(b0 + b1) // 2] else None,
                     "across_length_px": hi - lo if occupancy[(b0 + b1) // 2] else None,
                     "band_mean_rgb": [round(float(c), 1) for c in inside.mean(axis=0)],
                     "band_gray_std": round(float(gray_values[b0:b1, a:b].std()), 1),
                     "band_fraction_above_240": round(float((gray_values[b0:b1, a:b] > 240).mean()), 4)})
    emit(args, {"sha256": sha(args.image), "window_px": args.window, "axis": args.axis,
                "band_px": [b0 + y0, b1 + y0], "k": args.k,
                "clusters": [{"cluster": i, "mean_rgb": [round(float(c), 1) for c in centers[i]],
                              "fraction": round(float((labels == i).mean()), 4)} for i in range(args.k)],
                "segments": rows,
                "note": "dominant k-means colour cluster per position inside the band; `across_px` is where that "
                        "cluster fills at least half of the segment. Colour runs, not labelled parts."})


def box_mean(values, size):
    """Mean over a size x size window (valid region) using an integral image."""
    total = np.pad(values, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    return (total[size:, size:] - total[:-size, size:] - total[size:, :-size] + total[:-size, :-size]) / size ** 2


def ssim(a, b, size=7):
    """Mean structural similarity of two 8-bit grayscale arrays (uniform window)."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ma, mb = box_mean(a, size), box_mean(b, size)
    va, vb = box_mean(a * a, size) - ma ** 2, box_mean(b * b, size) - mb ** 2
    cov = box_mean(a * b, size) - ma * mb
    return float((((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))).mean())


def largest_part(mask):
    """Boolean mask of the largest 4-connected part."""
    parts = components(mask)
    if not parts:
        return np.zeros_like(mask)
    out = np.zeros_like(mask)
    for y, x0, x1 in max(parts, key=lambda c: c["area"])["runs"]:
        out[y, x0:x1] = True
    return out


def mean_rgb(reference_path, render_path, reference, candidate):
    out = {}
    for name, path, mask in (("reference", reference_path, reference), ("render", render_path, candidate)):
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
        out[name + "_inside"] = [round(float(c), 1) for c in rgb[mask].mean(axis=0)]
        out[name + "_outside"] = [round(float(c), 1) for c in rgb[~mask].mean(axis=0)]
    return out


def cmd_region(args):
    """Score one rendered entity against a reference region chosen by gray range or colour cluster."""
    values, _, _ = gray(args.image)
    x0, y0, x1, y1 = args.within or (0, 0, values.shape[1], values.shape[0])
    if (args.gray_min is None) == (not args.cluster_rgb):
        raise SystemExit("Give either --gray-min or --cluster-rgb")
    wanted = np.zeros(values.shape, dtype=bool)
    if args.cluster_rgb:  # k-means inside the window; keep the clusters nearest the given colours
        with Image.open(args.image) as image:
            rgb = np.asarray(image.convert("RGB").crop((x0, y0, x1, y1)), dtype=np.float64)
        flat = rgb.reshape(-1, 3)
        centers = kmeans(flat[np.random.default_rng(0).choice(len(flat), min(len(flat), 40000), replace=False)], args.k)
        labels = np.concatenate([((chunk[:, None, :] - centers[None]) ** 2).sum(axis=2).argmin(axis=1)
                                 for chunk in np.array_split(flat, max(1, len(flat) // 200000))]).reshape(rgb.shape[:2])
        chosen = sorted({int(((centers - np.array(c)) ** 2).sum(axis=1).argmin()) for c in args.cluster_rgb})
        wanted[y0:y1, x0:x1] = np.isin(labels, chosen)
        definition = {"cluster_rgb": args.cluster_rgb, "k": args.k,
                      "cluster_centres_used": [[round(float(v), 1) for v in centers[i]] for i in chosen]}
    else:
        wanted[y0:y1, x0:x1] = ((values >= args.gray_min) & (values <= args.gray_max))[y0:y1, x0:x1]
        definition = {"gray_min": args.gray_min, "gray_max": args.gray_max}
    reference = wanted if args.all_parts else largest_part(wanted)
    candidate = render_mask(args.render, args.entity_prefix)
    if reference.shape != candidate.shape:
        raise SystemExit(f"Shape mismatch: reference {reference.shape} vs render {candidate.shape}")
    with Image.open(Path(args.render) / "rgb.png") as image:
        rendered = np.asarray(image.convert("L"), dtype=np.float64)
    ref, cand = region_facts(reference), region_facts(candidate)
    result = score(reference, candidate)
    size_error = [round(abs(c - r) / r, 4) for c, r in zip(cand["rect_equivalent_size_px"], ref["rect_equivalent_size_px"])]
    angle = abs(cand["major_axis_deg"] - ref["major_axis_deg"])
    result.update(
        reference_region=ref, rendered_region=cand,
        centroid_offset_px=[round(c - r, 2) for c, r in zip(cand["centroid_px"], ref["centroid_px"])],
        size_relative_error=size_error, axis_difference_deg=round(min(angle, 180 - angle), 2),
        mean_gray={"reference_inside": round(float(values[reference].mean()), 1),
                   "reference_outside": round(float(values[~reference].mean()), 1),
                   "render_inside": round(float(rendered[candidate].mean()), 1),
                   "render_outside": round(float(rendered[~candidate].mean()), 1)},
        mean_rgb=mean_rgb(args.image, Path(args.render) / "rgb.png", reference, candidate),
        ssim_gray_full_frame=round(ssim(values, rendered), 4),
        reference_definition={**definition, "within": args.within,
                              "rule": "every selected pixel" if args.all_parts
                              else "largest 4-connected region of the selected pixels"},
        reference_sha256=sha(args.image), entity_prefixes=args.entity_prefix,
        caveat="the reference region is a threshold or colour-cluster segmentation, not human-labelled truth")
    emit(args, result)


def erode(mask, steps):
    for _ in range(steps):
        inner = mask.copy()
        inner[1:, :] &= mask[:-1, :]
        inner[:-1, :] &= mask[1:, :]
        inner[:, 1:] &= mask[:, :-1]
        inner[:, :-1] &= mask[:, 1:]
        mask = inner
    return mask


def look(rgb, mask):
    """Appearance statistics of one region: colour, tonal range, contrast and highlight share."""
    pixels = rgb[mask]
    tone = pixels.mean(axis=1)
    return {"mean_rgb": [round(float(c), 1) for c in pixels.mean(axis=0)],
            "gray_p5_p50_p95": [round(float(np.percentile(tone, q)), 1) for q in (5, 50, 95)],
            "gray_std": round(float(tone.std()), 1), "fraction_above_240": round(float((tone > 240).mean()), 4)}


def cmd_rois(args):
    """Compare appearance inside regions taken from the render's own entity mask.

    The regions come from the rendered geometry, so they are only meaningful
    once that geometry is registered to the reference; they are not labels of
    the photograph. Each region is eroded to keep boundary pixels out.
    """
    render_dir = Path(args.render)
    with Image.open(args.image) as image:
        reference = np.asarray(image.convert("RGB"), dtype=np.float64)
    with Image.open(render_dir / "rgb.png") as image:
        rendered = np.asarray(image.convert("RGB"), dtype=np.float64)
    if reference.shape != rendered.shape:
        raise SystemExit(f"Shape mismatch: reference {reference.shape} vs render {rendered.shape}")
    rows, total, weight_sum = [], 0.0, 0.0
    for item in args.roi:
        name, _, rest = item.partition("=")
        prefixes, _, weight = rest.partition("@")
        mask = erode(render_mask(args.mask_render or render_dir, prefixes.split(",")), args.erode)
        if args.within:
            x0, y0, x1, y1 = args.within
            keep = np.zeros_like(mask)
            keep[y0:y1, x0:x1] = True
            mask &= keep
        if mask.sum() < args.min_pixels:
            rows.append({"roi": name, "prefixes": prefixes.split(","), "pixels": int(mask.sum()), "status": "too small"})
            continue
        ref, got = look(reference, mask), look(rendered, mask)
        colour = float(np.sqrt(np.mean((np.array(ref["mean_rgb"]) - np.array(got["mean_rgb"])) ** 2))) / 255
        tone = float(np.mean(np.abs(np.array(ref["gray_p5_p50_p95"]) - np.array(got["gray_p5_p50_p95"])))) / 255
        contrast = abs(ref["gray_std"] - got["gray_std"]) / 255
        loss = colour + tone + contrast
        weight = float(weight) if weight else 1.0
        total, weight_sum = total + weight * loss, weight_sum + weight
        rows.append({"roi": name, "prefixes": prefixes.split(","), "pixels": int(mask.sum()), "weight": weight,
                     "reference": ref, "render": got, "colour_term": round(colour, 4), "tone_term": round(tone, 4),
                     "contrast_term": round(contrast, 4), "loss": round(loss, 4)})
    emit(args, {"reference_sha256": sha(args.image), "erode_px": args.erode, "within": args.within,
                "mask_from": "another render of the same geometry" if args.mask_render else "this render", "rois": rows,
                "loss": round(total / weight_sum, 5) if weight_sum else None,
                "loss_definition": "weighted mean over regions of: RMS mean-RGB difference + mean absolute difference "
                                   "of the 5th/50th/95th gray percentiles + gray standard-deviation difference, each / 255",
                "caveat": "regions are the render's entity mask, not labels of the reference; statistics, not likeness"})


def cmd_layout(args):
    values, mode, size = gray(args.image)
    mid = float(((values > 16) & (values < 240)).mean())
    # Two-level input: split halfway between the two levels (Otsu drifts into compression ringing).
    threshold = int((np.percentile(values, 1) + np.percentile(values, 99)) / 2) if mid < 0.01 else otsu(values)
    if mid >= 0.01 and not args.force:
        raise SystemExit(f"Not a two-level image (mid-gray fraction {mid:.3f}); tracing would invent edges")
    bright = values > threshold
    holes = {"bright_is_copper": enclosed(~bright), "dark_is_copper": enclosed(bright)}
    copper = args.copper
    mask = bright if copper == "bright" else ~bright
    rects = rectangles(mask)
    parts = components(mask)
    out = Path(args.out)
    layout = {
        "schema_version": "tao.kicad_light.layout.v1", "derived_from_private_reference": True,
        "source": {"name": Path(args.image).name, "sha256": sha(args.image), "mode": mode},
        "size_px": list(size), "threshold": threshold, "copper": copper,
        "copper_basis": args.copper_basis, "enclosed_islands_if": holes,
        "enclosed_islands_note": "informational only: topology cannot decide polarity (isolated pads and "
                                 "drill holes are both compact enclosed islands)", "mm_per_px": args.mm_per_px, "mm_per_px_basis": args.scale_basis,
        "copper_fraction": float(mask.mean()), "copper_components": len(parts), "rect_count": len(rects),
        "rects_px": rects,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(layout) + "\n")
    print(json.dumps({k: v for k, v in layout.items() if k != "rects_px"} | {"layout_sha256": sha(out)}, indent=2))


def cmd_compare(args):
    layout = json.loads(Path(args.layout).read_text())
    reference = reference_mask(args.image, layout)
    candidate = render_mask(args.render, args.entity_prefix)
    if reference.shape != candidate.shape:
        raise SystemExit(f"Shape mismatch: reference {reference.shape} vs render {candidate.shape}; "
                         "render at the reference's native size, do not resize")
    result = score(reference, candidate)
    best = max(((score(reference, np.roll(candidate, (dy, dx), axis=(0, 1)))["iou"], dx, dy)
                for dx in range(-2, 3) for dy in range(-2, 3)))
    result.update(best_shift_px={"dx": best[1], "dy": best[2], "iou": best[0]},
                  registered=best[1] == 0 and best[2] == 0, reference_sha256=sha(args.image),
                  entity_prefixes=args.entity_prefix)
    emit(args, result)


def cmd_diff(args):
    a, b = render_mask(args.render_a, args.entity_prefix), render_mask(args.render_b, args.entity_prefix)
    changed = a != b
    height, width = changed.shape
    boxes = json.loads(args.boxes)
    scale = args.box_range
    inside = np.zeros_like(changed)
    rows = []
    for x0, y0, x1, y1 in boxes:
        px = [int(np.floor(x0 / scale * width)), int(np.floor(y0 / scale * height)),
              int(np.ceil(x1 / scale * width)), int(np.ceil(y1 / scale * height))]
        region = changed[px[1]:px[3], px[0]:px[2]]
        inside[px[1]:px[3], px[0]:px[2]] = True
        rows.append({"box": [x0, y0, x1, y1], "box_px": px, "changed_pixels": int(region.sum()),
                     "added": int((b & ~a)[px[1]:px[3], px[0]:px[2]].sum()),
                     "removed": int((a & ~b)[px[1]:px[3], px[0]:px[2]].sum())})
    total = int(changed.sum())
    outside = [c for c in components(changed & ~inside)]
    emit(args, {"changed_pixels": total, "changed_inside_boxes": int((changed & inside).sum()),
                "fraction_inside_boxes": float((changed & inside).sum() / total) if total else None,
                "boxes_with_change": sum(1 for r in rows if r["changed_pixels"]), "boxes": rows,
                "components_outside_boxes": len(outside),
                "largest_outside_component_px": max((c["area"] for c in outside), default=0)})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    stats = sub.add_parser("stats")
    stats.add_argument("image")
    stats.add_argument("--out-json", help="also save the printed numbers here")
    stats.set_defaults(func=cmd_stats)
    segments = sub.add_parser("segments")
    segments.add_argument("image")
    segments.add_argument("--k", type=int, default=4)
    segments.add_argument("--max-side", type=int, help="measure on a copy reduced to this longest side")
    segments.add_argument("--top", type=int, default=1, help="regions to describe per cluster")
    segments.add_argument("--crop", type=int, nargs=4, help="x0 y0 x1 y1 window in native pixels")
    segments.add_argument("--out-json", help="also save the printed numbers here")
    segments.set_defaults(func=cmd_segments)
    profile = sub.add_parser("profile")
    profile.add_argument("image")
    profile.add_argument("--window", type=int, nargs=4, required=True, help="x0 y0 x1 y1 in pixels")
    profile.add_argument("--axis", choices=("x", "y"), default="x", help="direction the segments are ordered along")
    profile.add_argument("--band", type=int, nargs=2, help="range across the axis that decides each segment")
    profile.add_argument("--k", type=int, default=5)
    profile.add_argument("--min-run", type=int, default=6)
    profile.add_argument("--out-json", help="also save the printed numbers here")
    profile.set_defaults(func=cmd_profile)

    layout = sub.add_parser("layout")
    layout.add_argument("image")
    layout.add_argument("--out", required=True)
    layout.add_argument("--mm-per-px", type=float, required=True)
    layout.add_argument("--scale-basis", required=True, help="where the scale comes from; say 'assumed' if it is")
    layout.add_argument("--copper", choices=("bright", "dark"), required=True, help="which level is copper")
    layout.add_argument("--copper-basis", required=True, help="evidence for the polarity; say 'assumed' if it is")
    layout.add_argument("--force", action="store_true", help="trace even if the image is not two-level")
    layout.set_defaults(func=cmd_layout)
    compare = sub.add_parser("compare")
    compare.add_argument("image")
    compare.add_argument("--layout", required=True)
    compare.add_argument("--render", required=True, help="directory with entity_ids.npy and render_receipt.json")
    compare.add_argument("--entity-prefix", action="append", required=True)
    compare.add_argument("--out-json", help="also save the printed numbers here")
    compare.set_defaults(func=cmd_compare)
    region = sub.add_parser("region")
    region.add_argument("image")
    region.add_argument("--render", required=True)
    region.add_argument("--entity-prefix", action="append", required=True)
    region.add_argument("--gray-min", type=float)
    region.add_argument("--cluster-rgb", type=float, nargs=3, action="append", metavar=("R", "G", "B"),
                        help="instead of a gray range: the k-means cluster nearest this colour (repeatable)")
    region.add_argument("--k", type=int, default=6)
    region.add_argument("--gray-max", type=float, default=255.0)
    region.add_argument("--within", type=int, nargs=4, help="x0 y0 x1 y1 window in pixels")
    region.add_argument("--all-parts", action="store_true", help="use every pixel in range, not the largest region")
    region.add_argument("--out-json", help="also save the printed numbers here")
    region.set_defaults(func=cmd_region)
    rois = sub.add_parser("rois")
    rois.add_argument("image")
    rois.add_argument("--render", required=True)
    rois.add_argument("--roi", action="append", required=True, metavar="NAME=PREFIX[,PREFIX][@WEIGHT]")
    rois.add_argument("--mask-render", help="take entity_ids.npy from this render directory instead "
                                            "(same geometry and camera, rendered once)")
    rois.add_argument("--erode", type=int, default=3, help="pixels removed from each region boundary")
    rois.add_argument("--min-pixels", type=int, default=50)
    rois.add_argument("--within", type=int, nargs=4, help="x0 y0 x1 y1 window in pixels")
    rois.add_argument("--out-json", help="also save the printed numbers here")
    rois.set_defaults(func=cmd_rois)

    diff = sub.add_parser("diff")
    diff.add_argument("--render-a", required=True)
    diff.add_argument("--render-b", required=True)
    diff.add_argument("--entity-prefix", action="append", required=True)
    diff.add_argument("--boxes", required=True, help="JSON list of [x0, y0, x1, y1]")
    diff.add_argument("--box-range", type=float, default=1000.0)
    diff.add_argument("--out-json", help="also save the printed numbers here")
    diff.set_defaults(func=cmd_diff)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
