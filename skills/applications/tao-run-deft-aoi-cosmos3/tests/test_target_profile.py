# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import build_target_profile as profile


def row(key, *, task="Defect Detection", boxes=None, answer=None, paths=None):
    return {"id": key, "task_type": task, "messages": [
        {"role": "user", "content": [
            *[{"type": "image", "image": path, "min_pixels": 1048576, "max_pixels": 1048576}
              for path in (paths or [f"/data/datasets/family/{key}.png"])],
            {"type": "text", "text": "Which?\nA. None\nB. Bridge\nC. Open"},
        ]},
        {"role": "assistant", "content": answer if answer is not None else json.dumps(boxes or [])},
    ]}


def box(side, label="bridge"):
    return {"bbox_2d": [0, 0, side, side], "label": label}


def write_rows(path, rows):
    path.write_text("".join(json.dumps(item) + "\n" for item in rows))
    return path


class TargetProfileTests(unittest.TestCase):
    def test_exact_geometry_bins_and_unscaled_relative_area(self):
        rows = [row("empty"), row("sizes", boxes=[box(s) for s in (15, 16, 33, 66, 130, 260)])]
        result = profile.build_profile(rows)
        task = result.get("tasks", {}).get("Defect Detection", {})
        self.assertEqual(task.get("rows"), 2)
        self.assertEqual(task["empty_rate"], 0.5)
        self.assertEqual(task["box_size_bins"], dict.fromkeys(profile.SIZE_LABELS, 1))
        self.assertEqual(task["count_bins"], {"0": 1, "1": 0, "2-3": 0, "4-9": 1, "10+": 0})
        self.assertEqual(task["nonempty_count_share"]["4-9"], 1.0)
        self.assertAlmostEqual(task["relative_area_quantiles"]["50"], (33**2 + 66**2) / 2e6)
        self.assertEqual(task["aspect_ratio_median"], 1.0)
        self.assertEqual(task["label_histogram"], {"bridge": 6})
        self.assertEqual(sum(cell["rows"] for cell in result["cells"]), 2)

    def test_pair_kinds_dataset_precedence_and_classification_rows(self):
        rows = [row("edit", task="Ref_based Defect Classification", answer="B", paths=[
            "/data/NVPAW_pair/edit/gen_Qwen/same.png", "/data/NVPAW_pair/edit/gen_Qwen/same.png"]),
            row("aug", task="Ref_based Defect Detection", paths=["/data/g.png", "/data/gen_aug/light.png"]),
            row("same", task="Ref_based Defect Detection", paths=["/data/a.png", "/data/a.png"]),
            row("photo", task="Ref_based Defect Detection", paths=["/data/a.png", "/data/b.png"])]
        result = profile.build_profile(rows)
        self.assertEqual(result.get("rows"), 4)
        self.assertEqual(result["tasks"]["Ref_based Defect Classification"]["dataset_histogram"], {"pair:edit": 1})
        self.assertEqual(result["tasks"]["Ref_based Defect Classification"]["pair_kind"], {"generated_edit": 1})
        self.assertEqual(result["tasks"]["Ref_based Defect Detection"]["pair_kind"], {
            "augmented_view": 1, "identical_path": 1, "different_photo": 1})

    def test_compare_allocates_integer_budget_and_names_shortage(self):
        target = profile.build_profile([row("a"), row("b", boxes=[box(33)])])
        supply = profile.build_profile([row("c")])
        comparison = profile.compare_profiles(target, supply, budget=10)
        self.assertEqual(sum(cell["requested_rows"] for cell in comparison), 10)
        positive = next(cell for cell in comparison if cell["empty_status"] == "non_empty")
        self.assertEqual((positive["target_share"], positive["supply_share"]), (0.5, 0.0))
        self.assertEqual((positive["achievable_rows"], positive["shortage_rows"]), (0, 5))

    def test_invalid_gt_is_not_silently_counted_as_empty(self):
        with self.assertRaisesRegex(ValueError, "ground truth"):
            profile.build_profile([row("bad", answer="not json")])
        with self.assertRaisesRegex(ValueError, "1000"):
            profile.build_profile([row("bad-coordinate", boxes=[box(1024)])])

    def test_cli_profiles_training_repetitions_without_mutating_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = write_rows(root / "train.jsonl", [row("same"), row("same")])
            original = source.read_bytes()
            self.assertEqual(profile.main(["--input", str(source), "--output-dir", str(root), "--name", "train"]), 0)
            self.assertEqual(json.loads((root / "train_profile.json").read_text())["rows"], 2)
            self.assertIn("Defect Detection", (root / "train_profile.md").read_text())
            self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
