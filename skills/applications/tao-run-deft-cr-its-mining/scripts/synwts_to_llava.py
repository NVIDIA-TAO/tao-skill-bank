#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SynWTS -> DEFT ITS adapter: 33-frame overhead-view windows, four binary questions, LLaVA annotations, media cuts.

One entry point driven by scripts/sim2real_spec.json (liftable into a tao-generate-sim2real-video data skill later):
  python synwts_to_llava.py build  [--spec sim2real_spec.json]   windows + labels + LLaVA annotations + manifest (no media)
  python synwts_to_llava.py media  [--spec ...] [--sets train,kpi_sim] [--ids w1,w2]   cut the clips + per-window GT boxes
  python synwts_to_llava.py report [--spec ...]                  print the label / skip table from the manifest

Windows. One scenario = one SynWTS val scenario folder (e.g. 20230707_18_CY23_T1) or, for the 'normal_trimmed' clips, one recording
(prefix up to '_normal', several cameras / events). For every scenario-camera video the five caption phases (0 pre-recognition,
1 recognition, 2 judgement, 3 action, 4 avoidance) are located frame-accurately from the `phase_number` field of the per-frame
bbox_annotated pedestrian / vehicle files (fallback: caption start / end seconds x 30 fps). A window is 33 frames centred on the
phase centre, clamped to the video; one window per scenario-camera-phase is a candidate, at most `max_per_scenario` are kept per
scenario (seeded; distinct phases first, then distinct cameras, windows that lie fully inside their phase preferred).
Ten scenarios (seeded) are held out entirely as KPI-sim; the rest is the train pool.

Labels. Derived per window from (a) the annotated pedestrian / vehicle boxes of the window frames, (b) the caption of the window's
phase (caption_pedestrian + caption_vehicle) and (c) the overhead-view VQA conversations of that phase. Every rule returns
"yes", "no" or None (skip: the question is left out for that window rather than guessed); the manifest records which clause fired.
  q1 "Is there a pedestrian on the road?"
      yes: VQA action in {crossing, collision, thrown back}; VQA fine-grained action says crossing / rushing out / traveling in a
           car lane; caption says crossing the road|street, on|in the road, middle of the road, car lane, roadway;
           or VQA position "in front of the vehicle" with distance close|near (in the vehicle's lane at close range) unless the
           boxes are > 2 vehicle lengths apart (conflict -> skip).
      no:  no pedestrian box in the window (vehicle-only clip); caption places the pedestrian on the sidewalk / pavement /
           roadside / shoulder; or VQA distance far with action standing still|about to cross and no road wording.
  q2 "Is there a pedestrian within one car length of a moving or stopped vehicle?"
      box geometry first: g = median over window frames of (gap between the pedestrian box and the vehicle box) / (vehicle box
      long side) - overlapping boxes have gap 0; g <= 1.0 -> yes, g >= 2.0 -> no (a VQA distance that contradicts the boxes -> skip);
      in between the VQA distance decides
      (close -> yes, far -> no, near -> skip). VQA collision / thrown back -> yes. No pedestrian box -> no. No boxes at all -> skip.
  q3 "Is a vehicle stopped or braking in this clip?"
      caption_vehicle speed "<n> km/h" and words. yes: speed 0, or braking|brakes|decelerat*|slow(s|ing) down|stopp*|stationary|
      halt*|comes to a stop. no: speed >= 5 km/h and none of those words and the vehicle box moves >= 0.25 of its long side over
      the window. Vehicle box motion < 0.10 of its long side over the window -> yes unless the caption speed is >= 10 km/h (conflict ->
      skip). Otherwise skip.
  q4 "Is a pedestrian crossing in front of a vehicle?"
      yes: VQA position in front AND (action crossing|collision, fine-grained crossing*|rushing out, or action going straight ahead
           with body perpendicular to the vehicle), or caption "crossing ... in front of the vehicle".
      no:  no pedestrian box; VQA position behind / beside (not in front); action standing still|squatting|about to *;
           action going straight ahead with body in the same / opposite direction as the vehicle and no crossing wording.
Outputs (spec "workspace" / "media_root"): <workspace>/data/{train,kpi}/annotations.json (LLaVA: one record per answered
question, id "<window>__q<k>", video "<window>.mp4" relative to the media dir, conversations human "<video>\n<question> Answer
with yes or no." / gpt "yes|no"), media junction targets under <media_root>/<set>/media, GT boxes <media_root>/<set>/gt/<window>.json
(geometry_metrics --ref-boxes format), and the committed manifest evidence/synwts_windows_manifest.json (ids, frame ranges,
labels, rule clauses, skip reasons; no captions, boxes or media).
"""
import argparse
import collections
import json
import os
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")  # on PATH, or set FFMPEG / FFPROBE to full paths
FFPROBE = os.environ.get("FFPROBE", str(Path(FFMPEG).with_name("ffprobe" + Path(FFMPEG).suffix)) if os.sep in FFMPEG or "/" in FFMPEG else "ffprobe")
PHASE_NAMES = {"0": "prerecognition", "1": "recognition", "2": "judgement", "3": "action", "4": "avoidance"}


# ----------------------------------------------------------------------------------------------------------------- data access
def load_spec(path):
    return json.load(open(path, encoding="utf-8"))


def ann_root(spec):
    return Path(spec["dataset"]["annotations"])


def video_root(spec):
    return Path(spec["dataset"]["videos"])


def list_videos(spec):
    """Return [(scenario_key, camera_name, video_path, is_normal)] for every overhead-view val video on disk."""
    out = []
    vr = video_root(spec)
    for scen_dir in sorted(vr.iterdir()):
        if scen_dir.name == "normal_trimmed":
            for rec in sorted(scen_dir.iterdir()):
                for mp4 in sorted((rec / "overhead_view").glob("*.mp4")):
                    key = rec.name.split("_normal")[0] + "_normal"
                    out.append((key, mp4.stem, mp4, True))
        elif scen_dir.is_dir():
            for mp4 in sorted((scen_dir / "overhead_view").glob("*.mp4")):
                out.append((scen_dir.name, mp4.stem, mp4, False))
    return out


def bbox_file(spec, kind, camera, is_normal):
    base = ann_root(spec) / "bbox_annotated" / kind / spec["dataset"]["split"]
    hits = list(base.glob(f"{'normal_trimmed/' if is_normal else ''}*/overhead_view/{camera}_bbox.json"))
    return hits[0] if hits else None


def load_boxes(path):
    """{frame(0-based): [x1,y1,x2,y2]} and {frame: phase} from a SynWTS bbox file (COCO xywh, one actor)."""
    boxes, phases = {}, {}
    if path is None:
        return boxes, phases
    for a in json.load(open(path))["annotations"]:
        x, y, w, h = a["bbox"]
        f = int(a["image_id"])
        boxes[f] = [x, y, x + w, y + h]
        if a.get("phase_number") is not None:
            phases[f] = str(a["phase_number"])
    return boxes, phases


def caption_for(spec, scenario_dirname, is_normal):
    base = ann_root(spec) / "caption" / spec["dataset"]["split"]
    p = base / ("normal_trimmed" if is_normal else "") / scenario_dirname / "overhead_view"
    hits = list(p.glob("*_caption.json"))
    if not hits:
        return {}
    d = json.load(open(hits[0], encoding="utf-8"))
    return {ph["labels"][0]: ph for ph in d["event_phase"]}


def vqa_for(spec, scenario_dirname, is_normal):
    base = ann_root(spec) / "vqa" / spec["dataset"]["split"]
    p = base / ("normal_trimmed" if is_normal else "") / scenario_dirname / "overhead_view"
    hits = list(p.glob("*.json"))
    if not hits:
        return {}
    d = json.load(open(hits[0], encoding="utf-8"))
    inv = {v: k for k, v in spec["windows"]["phase_labels"].items()}
    out = {}
    for item in d:
        for ph in item.get("event_phase", []):
            lab = ph["labels"][0]
            num = spec["windows"]["phase_labels"].get(lab, lab)
            qa = {}
            for c in ph.get("conversations", []):
                ans = c.get(c.get("correct", ""), "")
                qa[norm_q(c["question"])] = ans.strip().lower()
            out[num] = qa
    return out


def norm_q(q):
    q = q.lower()
    if "orientation" in q:
        return "orientation"
    if "position of the pedestrian" in q:
        return "position"
    if "relative distance" in q:
        return "distance"
    if "fine-grained" in q:
        return "fine_action"
    if "pedestrian's action" in q:
        return "action"
    if "direction of travel" in q:
        return "direction"
    if "speed" in q:
        return "ped_speed"
    return q


def nframes(video, cache):
    key = str(video)
    if key not in cache:
        out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                              "stream=nb_read_frames,width,height", "-of", "csv=p=0", str(video)],
                             capture_output=True, text=True, check=True).stdout.strip().split(",")
        cache[key] = {"w": int(out[0]), "h": int(out[1]), "n": int(out[2])}
    return cache[key]


# ----------------------------------------------------------------------------------------------------------------- geometry
def gap(a, b):
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return (dx * dx + dy * dy) ** 0.5


def long_side(b):
    return max(b[2] - b[0], b[3] - b[1])


def window_geometry(ped, veh, f0, n):
    frames = range(f0, f0 + n)
    ped_in = [f for f in frames if f in ped]
    veh_in = [f for f in frames if f in veh]
    g = {"ped_frames": len(ped_in), "veh_frames": len(veh_in), "gap_ratio": None, "veh_motion_ratio": None,
         "ped_h_med": None, "veh_long_med": None}
    if ped_in:
        g["ped_h_med"] = statistics.median(ped[f][3] - ped[f][1] for f in ped_in)
    if veh_in:
        g["veh_long_med"] = statistics.median(long_side(veh[f]) for f in veh_in)
        first, last = veh[veh_in[0]], veh[veh_in[-1]]
        c0 = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
        c1 = ((last[0] + last[2]) / 2, (last[1] + last[3]) / 2)
        if g["veh_long_med"] > 0 and len(veh_in) >= 2:
            span = (veh_in[-1] - veh_in[0]) / max(1, n - 1)
            disp = ((c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2) ** 0.5
            g["veh_motion_ratio"] = round(disp / g["veh_long_med"] / max(span, 1e-6), 3)
    both = [f for f in frames if f in ped and f in veh]
    if both:
        g["gap_ratio"] = round(statistics.median(gap(ped[f], veh[f]) / max(long_side(veh[f]), 1.0) for f in both), 3)
    return g


# ----------------------------------------------------------------------------------------------------------------- rules
ROAD_RE = re.compile(r"crossing the (road|street)|crosses the (road|street)|\bon the road\b|\bin the road\b|middle of the road|"
                     r"\bcar lane\b|\broadway\b|\bin the lane\b|\bcarriageway\b|crossing in front|walking (across|along) the (road|street)", re.I)
SIDEWALK_RE = re.compile(r"\bsidewalk\b|\bpavement\b|\broadside\b|\bshoulder\b|\bfootpath\b|\bkerb\b|\bcurb\b", re.I)
BRAKE_RE = re.compile(r"\bbrak(e|es|ing)\b|\bdecelerat\w*|\bslow(s|ed|ing)? down\b|\bstopp?(ed|ing|s)?\b|\bstationary\b|\bhalt\w*|"
                      r"comes? to a stop|came to a stop|\bat a standstill\b|\bnot moving\b|\bparked\b", re.I)
SPEED_RE = re.compile(r"(\d+)\s*km/h", re.I)
CROSS_FRONT_RE = re.compile(r"cross\w*[^.]{0,60}in front of the vehicle|in front of the vehicle[^.]{0,60}cross\w*", re.I)


def rule_q1(vqa, cap_p, cap_v, geo):
    if geo["ped_frames"] == 0:
        return "no", "no pedestrian box in window"
    act, fine, pos, dist = vqa.get("action", ""), vqa.get("fine_action", ""), vqa.get("position", ""), vqa.get("distance", "")
    if act in ("crossing", "collision", "thrown back"):
        return "yes", f"vqa action={act}"
    if fine and (fine.startswith("crossing") or fine.startswith("rushing out") or "traveling in a car lane" in fine):
        return "yes", f"vqa fine_action={fine}"
    m = ROAD_RE.search(cap_p)
    if m:
        return "yes", f"caption road wording '{m.group(0)}'"
    if "in front" in pos and dist in ("close", "near"):
        if geo["gap_ratio"] is not None and geo["gap_ratio"] > 2.0:
            return None, f"vqa position={pos}, distance={dist} but box gap {geo['gap_ratio']} vehicle lengths (conflict)"
        return "yes", f"vqa position={pos}, distance={dist}"
    m = SIDEWALK_RE.search(cap_p)
    if m and not ROAD_RE.search(cap_p):
        return "no", f"caption off-road wording '{m.group(0)}'"
    if dist == "far" and act in ("standing still", "about to cross", "about to stand still") and not ROAD_RE.search(cap_p):
        return "no", f"vqa distance=far, action={act}, no road wording"
    return None, "undecided"


def rule_q2(vqa, cap_p, cap_v, geo):
    act = vqa.get("action", "")
    if act in ("collision", "thrown back"):
        return "yes", f"vqa action={act}"
    if geo["ped_frames"] == 0:
        return "no", "no pedestrian box in window"
    g = geo["gap_ratio"]
    dist = vqa.get("distance", "")
    if g is not None:
        if g <= 1.0 and dist == "far":
            return None, f"box gap {g} but vqa distance=far (conflict)"
        if g >= 2.0 and dist == "close":
            return None, f"box gap {g} but vqa distance=close (conflict)"
        if g <= 1.0:
            return "yes", f"box gap {g} vehicle lengths"
        if g >= 2.0:
            return "no", f"box gap {g} vehicle lengths"
        if dist == "close":
            return "yes", f"box gap {g} in (1,2), vqa distance=close"
        if dist == "far":
            return "no", f"box gap {g} in (1,2), vqa distance=far"
        return None, f"box gap {g} in (1,2), vqa distance={dist or 'n/a'}"
    if geo["veh_frames"] == 0:
        return None, "no vehicle box in window"
    return None, "boxes never co-visible"


def rule_q3(vqa, cap_p, cap_v, geo):
    speeds = [int(x) for x in SPEED_RE.findall(cap_v)]
    words = BRAKE_RE.findall(cap_v)
    word_hit = BRAKE_RE.search(cap_v)
    motion = geo["veh_motion_ratio"]
    if speeds and min(speeds) == 0:
        return "yes", f"caption speed {min(speeds)} km/h"
    if word_hit:
        return "yes", f"caption word '{word_hit.group(0)}'"
    if motion is not None and motion < 0.10 and geo["veh_frames"] >= 17:
        if speeds and min(speeds) >= 10:
            return None, f"vehicle box moved {motion} lengths but caption speed {min(speeds)} km/h (conflict)"
        return "yes", f"vehicle box moved {motion} lengths over the window"
    if speeds and min(speeds) >= 5 and not words and (motion is None or motion >= 0.25):
        return "no", f"caption speed {min(speeds)} km/h, no braking word, box motion {motion}"
    return None, f"speed={speeds or 'n/a'}, words={bool(words)}, motion={motion}"


def rule_q4(vqa, cap_p, cap_v, geo):
    if geo["ped_frames"] == 0:
        return "no", "no pedestrian box in window"
    act, fine, pos, ori = vqa.get("action", ""), vqa.get("fine_action", ""), vqa.get("position", ""), vqa.get("orientation", "")
    front = "in front" in pos
    if front and act in ("crossing", "collision"):
        return "yes", f"vqa position={pos}, action={act}"
    if front and fine and (fine.startswith("crossing") or fine.startswith("rushing out")):
        return "yes", f"vqa position={pos}, fine_action={fine}"
    if front and act == "going straight ahead" and "perpendicular" in ori:
        return "yes", f"vqa position={pos}, action={act}, orientation={ori}"
    m = CROSS_FRONT_RE.search(cap_p)
    if m and (front or not pos):
        return "yes", f"caption '{m.group(0)[:60]}'"
    if pos and not front:
        return "no", f"vqa position={pos}"
    if act in ("standing still", "squatting", "about to cross", "about to stand still", "about to squat"):
        return "no", f"vqa action={act}"
    if act == "going straight ahead" and ("same direction" in ori or "opposite direction" in ori) and "diagonal" not in ori \
            and not re.search(r"cross", cap_p, re.I):
        return "no", f"vqa action={act}, orientation={ori}, no crossing wording"
    return None, "undecided"


RULES = {"q1_ped_on_road": rule_q1, "q2_ped_near_vehicle": rule_q2, "q3_vehicle_stopped": rule_q3, "q4_ped_crossing_front": rule_q4}


# ----------------------------------------------------------------------------------------------------------------- build
def phase_ranges(ped_ph, veh_ph, caption, fps, n):
    """{phase: (first, last)} frame ranges, from bbox phase numbers, else caption seconds."""
    src = ped_ph or veh_ph
    rng = {}
    if src:
        by = collections.defaultdict(list)
        for f, p in src.items():
            by[p].append(f)
        for p, fs in by.items():
            rng[p] = (min(fs), max(fs), "bbox")
    for p, ph in caption.items():
        if p not in rng:
            a, b = int(round(float(ph["start_time"]) * fps)), int(round(float(ph["end_time"]) * fps)) - 1
            if 0 <= a < n:
                rng[p] = (a, min(b, n - 1), "caption")
    return rng


def build(spec, manifest_path):
    fps = spec["dataset"]["fps"]
    N = spec["windows"]["frames"]
    cache_path = Path(spec["dataset"]["root"]) / "video_probe_cache.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    vids = list_videos(spec)
    scenarios = sorted({v[0] for v in vids})
    rng = random.Random(spec["windows"]["seed"])
    holdout = sorted(rng.sample(scenarios, spec["windows"]["holdout_scenarios"]))
    cands_by_scen = collections.defaultdict(list)
    for scen_key, cam, mp4, is_normal in vids:
        scen_dir = mp4.parents[1].name
        ped, ped_ph = load_boxes(bbox_file(spec, "pedestrian", cam, is_normal))
        veh, veh_ph = load_boxes(bbox_file(spec, "vehicle", cam, is_normal))
        cap = caption_for(spec, scen_dir, is_normal)
        vqa = vqa_for(spec, scen_dir, is_normal)
        info = nframes(mp4, cache)
        n = info["n"]
        for p, (a, b, src) in sorted(phase_ranges(ped_ph, veh_ph, cap, fps, n).items()):
            centre = (a + b) / 2.0
            f0 = int(round(centre - (N - 1) / 2.0))
            f0 = max(0, min(f0, n - N))
            if n < N:
                continue
            inside = sum(1 for f in range(f0, f0 + N) if a <= f <= b) / N
            geo = window_geometry(ped, veh, f0, N)
            ph_cap = cap.get(p, {})
            cap_p, cap_v = ph_cap.get("caption_pedestrian", ""), ph_cap.get("caption_vehicle", "")
            labels, clauses = {}, {}
            for q in spec["questions"]:
                ans, why = RULES[q["key"]](vqa.get(p, {}), cap_p, cap_v, geo)
                labels[q["key"]] = ans
                clauses[q["key"]] = why
            wid = f"{cam}__p{p}_f{f0:04d}"
            cands_by_scen[scen_key].append({
                "id": wid, "scenario": scen_key, "scenario_dir": scen_dir, "normal": is_normal, "camera": cam,
                "video": str(mp4.relative_to(video_root(spec))).replace("\\", "/"), "video_frames": n, "width": info["w"],
                "height": info["h"], "phase": p, "phase_name": PHASE_NAMES.get(p, p), "phase_frames": [a, b], "phase_source": src,
                "start_frame": f0, "frames": N, "fraction_in_phase": round(inside, 3), "has_vqa": bool(vqa.get(p)),
                "has_caption": bool(ph_cap), "geometry": geo, "labels": labels, "clauses": clauses})
    json.dump(cache, open(cache_path, "w"), indent=0)
    # selection: <= max_per_scenario windows per scenario; seeded; distinct phases, then distinct cameras; full-phase windows first
    sel = random.Random(spec["windows"]["seed"] + 1)
    chosen = []
    for scen in scenarios:
        cands = cands_by_scen[scen]
        sel.shuffle(cands)
        cands.sort(key=lambda c: -int(c["fraction_in_phase"] >= 0.999))  # stable: full-phase windows first, shuffled within
        picked, phases, cams = [], set(), set()
        for pref in ("new_phase_new_cam", "new_phase", "new_cam", "any"):
            for c in cands:
                if len(picked) >= spec["windows"]["max_per_scenario"]:
                    break
                if c in picked or not any(v for v in c["labels"].values()):
                    continue
                np_, nc = c["phase"] not in phases, c["camera"] not in cams
                if (pref == "new_phase_new_cam" and np_ and nc) or (pref == "new_phase" and np_) or (pref == "new_cam" and nc) or pref == "any":
                    picked.append(c)
                    phases.add(c["phase"])
                    cams.add(c["camera"])
        for c in picked:
            c["set"] = "kpi_sim" if scen in holdout else "train"
            chosen.append(c)
    chosen.sort(key=lambda c: (c["set"], c["id"]))
    all_cands = sum(len(v) for v in cands_by_scen.values())
    manifest = {"spec": os.path.relpath(spec["_path"], ROOT).replace("\\", "/"), "dataset": spec["dataset"]["name"],
                "split": spec["dataset"]["split"], "view": spec["dataset"]["view"], "window_frames": N, "fps": fps,
                "seed": spec["windows"]["seed"], "scenarios": len(scenarios), "scenario_list": scenarios, "holdout_scenarios": holdout,
                "videos": len(vids), "candidate_windows": all_cands, "selected_windows": len(chosen),
                "questions": spec["questions"], "windows": chosen}
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(manifest_path, "w", encoding="utf-8"), indent=1)
    write_llava(spec, manifest)
    report(spec, manifest)
    return manifest


def write_llava(spec, manifest):
    ws = Path(spec["workspace"])
    suffix = spec["answer_suffix"]
    qtext = {q["key"]: q["text"] for q in spec["questions"]}
    for set_name in ("train", "kpi_sim"):
        recs = []
        for w in manifest["windows"]:
            if w["set"] != set_name:
                continue
            for k, (qkey, ans) in enumerate(w["labels"].items(), start=1):
                if ans is None:
                    continue
                recs.append({"id": f"{w['id']}__q{k}", "video": f"{w['id']}.mp4",
                             "conversations": [{"from": "human", "value": f"<video>\n{qtext[qkey]}{suffix}"},
                                               {"from": "gpt", "value": ans}]})
        out = ws / spec["sets"][set_name]["annotations"]
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(recs, open(out, "w", encoding="utf-8"), indent=2)
        print(f"{set_name}: {len(recs)} LLaVA records -> {out}")


def report(spec, manifest):
    ws = manifest["windows"]
    print(f"scenarios {manifest['scenarios']} (holdout {len(manifest['holdout_scenarios'])}: {', '.join(manifest['holdout_scenarios'])})")
    print(f"videos {manifest['videos']}, candidate windows {manifest['candidate_windows']}, selected {manifest['selected_windows']} "
          f"(train {sum(1 for w in ws if w['set'] == 'train')}, kpi_sim {sum(1 for w in ws if w['set'] == 'kpi_sim')})")
    print(f"phases of selected windows: {dict(sorted(collections.Counter(w['phase'] for w in ws).items()))}; "
          f"fully inside phase: {sum(1 for w in ws if w['fraction_in_phase'] >= 0.999)}; with VQA: {sum(1 for w in ws if w['has_vqa'])}")
    rows = []
    for q in spec["questions"]:
        k = q["key"]
        for set_name in ("train", "kpi_sim"):
            sub = [w for w in ws if w["set"] == set_name]
            c = collections.Counter(w["labels"][k] for w in sub)
            rows.append((k, set_name, c.get("yes", 0), c.get("no", 0), c.get(None, 0), len(sub)))
    print("\n| Question | Set | yes | no | skipped | windows |\n|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    print("\nTop skip reasons:")
    for q in spec["questions"]:
        k = q["key"]
        c = collections.Counter(w["clauses"][k] for w in ws if w["labels"][k] is None)
        for why, n in c.most_common(3):
            print(f"  {k}: {n} x {why}")


# ----------------------------------------------------------------------------------------------------------------- media
def cut_clip(src, f0, n, dst, crf="10"):
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = f"trim=start_frame={f0}:end_frame={f0 + n},setpts=PTS-STARTPTS"
    cmd = [FFMPEG, "-y", "-v", "error", "-i", str(src), "-vf", vf, "-an", "-c:v", "libx264", "-preset", "medium", "-crf", crf,
           "-pix_fmt", "yuv420p", "-r", "30", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True)


def media(spec, manifest, sets, ids):
    mr = Path(spec["media_root"])
    ws = Path(spec["workspace"])
    probe = {}
    done = 0
    for w in manifest["windows"]:
        if w["set"] not in sets or (ids and w["id"] not in ids):
            continue
        set_dir = mr / w["set"]
        dst = set_dir / "media" / f"{w['id']}.mp4"
        src = video_root(spec) / w["video"]
        if not dst.exists():
            cut_clip(src, w["start_frame"], w["frames"], dst)
            info = nframes(dst, probe)
            if info["n"] != w["frames"]:
                raise SystemExit(f"{dst}: {info['n']} frames, expected {w['frames']}")
        ped, _ = load_boxes(bbox_file(spec, "pedestrian", w["camera"], w["normal"]))
        veh, _ = load_boxes(bbox_file(spec, "vehicle", w["camera"], w["normal"]))
        frames = {}
        for k in range(w["frames"]):
            f = w["start_frame"] + k
            items = []
            if f in ped:
                items.append({"label": "person", "box": [round(v, 1) for v in ped[f]]})
            if f in veh:
                items.append({"label": "car", "box": [round(v, 1) for v in veh[f]]})
            if items:
                frames[str(k + 1)] = items
        gt = set_dir / "gt" / f"{w['id']}.json"
        gt.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"hw": [w["height"], w["width"]], "frames": frames, "source": w["video"], "start_frame": w["start_frame"]},
                  open(gt, "w"), indent=0)
        done += 1
    # junction from the workspace media dir to the media root (the workspace holds no media bytes itself)
    for set_name in sets:
        link = ws / spec["sets"][set_name]["media"]
        target = mr / set_name / "media"
        target.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
            print(f"junction {link} -> {target}")
    print(f"media: {done} windows cut / refreshed under {mr}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["build", "media", "report"])
    ap.add_argument("--spec", default=str(HERE / "sim2real_spec.json"))
    ap.add_argument("--manifest", default=str(ROOT / "evidence" / "synwts_windows_manifest.json"))
    ap.add_argument("--sets", default="train,kpi_sim")
    ap.add_argument("--ids", default=None)
    a = ap.parse_args()
    spec = load_spec(a.spec)
    spec["_path"] = os.path.abspath(a.spec)
    if a.stage == "build":
        build(spec, a.manifest)
    else:
        manifest = json.load(open(a.manifest, encoding="utf-8"))
        if a.stage == "report":
            report(spec, manifest)
        else:
            media(spec, manifest, set(a.sets.split(",")), set(a.ids.split(",")) if a.ids else None)


if __name__ == "__main__":
    main()
