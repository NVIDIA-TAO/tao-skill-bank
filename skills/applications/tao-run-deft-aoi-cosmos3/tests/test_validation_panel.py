# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from collections import Counter
import hashlib
import json
import pathlib
import tempfile
import unittest

from test_target_profile import row, box, write_rows
import build_validation_panel as panel
import build_target_profile as profile


def fixture(prefix, copies):
    return [row(f"{prefix}-{family}-{empty}-{index}", boxes=[] if empty else [box(33)],
                paths=[f"/data/NVPAW_pair/{family}/{prefix}-{empty}-{index}.png"])
            for family in ("a", "b", "c", "d") for empty in (False, True) for index in range(copies)]


class ValidationPanelTests(unittest.TestCase):
    def test_two_family_benchmark_relaxes_cap_and_fills_500(self):
        task = "Component Detection"
        benchmark = [row(f"benchmark-{family}-{i}", task=task, boxes=[box(33)],
                         paths=[f"/data/datasets/{family}/benchmark-{i}.png"])
                     for family, count in (("FPIC_Component", 217), ("FICS-PCB", 33)) for i in range(count)]
        candidates = [row(f"pool-{family}-{i}", task=task, boxes=[box(33)],
                          paths=[f"/data/datasets/{family}/pool-{i}.png"])
                      for family in ("FPIC_Component", "FICS-PCB") for i in range(600)]
        selected, manifest = panel.select_panel(candidates, benchmark, [], per_task=500, seed=17)
        self.assertEqual(len(selected), 500)
        self.assertEqual(Counter(profile.dataset_family(item) for item in selected),
                         {"FPIC_Component": 434, "FICS-PCB": 66})
        report = manifest["tasks"][task]
        self.assertAlmostEqual(report["family_cap_effective"], 0.868)
        self.assertIs(report["family_cap_relaxed"], True)
        self.assertIn("benchmark", report["family_cap_reason"])
        self.assertEqual(report["shortage_rows"], 0)

    def test_relaxed_cap_keeps_rounded_benchmark_quota(self):
        task = "Component Classification"
        benchmark = [row(f"benchmark-{family}-{i}", task=task, answer="B",
                         paths=[f"/data/datasets/{family}/benchmark-{i}.png"])
                     for family, count in (("a", 437), ("b", 300), ("c", 263)) for i in range(count)]
        candidates = [row(f"pool-{family}-{i}", task=task, answer="B",
                          paths=[f"/data/datasets/{family}/pool-{i}.png"])
                      for family in ("a", "b", "c") for i in range(600)]
        selected, manifest = panel.select_panel(candidates, benchmark, [], per_task=500, seed=17)
        self.assertEqual(len(selected), 500)
        self.assertEqual(Counter(profile.dataset_family(item) for item in selected), {"a": 219, "b": 150, "c": 131})
        self.assertAlmostEqual(manifest["tasks"][task]["family_cap_effective"], 0.437)

    def test_feasible_cap_preserves_seeded_selection_bytes(self):
        selected, manifest = panel.select_panel(fixture("pool", 5), fixture("benchmark", 2),
                                                fixture("proxy", 1), per_task=24, seed=17)
        text = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in selected)
        # Characterized on pre-fix 9c1140b, using the production JSONL serializer.
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),
                         "57210e36249ef89c0f92964461e5df1a948b1a6060622440a9a89731fcc6d995")
        report = manifest["tasks"]["Defect Detection"]
        self.assertEqual(report.get("family_cap_effective"), 0.35)
        self.assertIs(report.get("family_cap_relaxed"), False)

    def test_stratified_panel_is_deterministic_disjoint_and_message_exact(self):
        benchmark, proxy = fixture("benchmark", 2), fixture("proxy", 1)
        candidates = fixture("pool", 5) + [
            {**benchmark[0], "id": "different-id"},
            {**fixture("pool-extra", 1)[0], "id": benchmark[1]["id"]},
            {**proxy[0], "id": "proxy-alias"},
            # Either pair side is excluded, even with a novel target.
            row("golden-overlap", task="Ref_based Defect Detection", paths=[
                benchmark[2]["messages"][0]["content"][0]["image"], "/data/novel.png"]),
        ]
        selected, manifest = panel.select_panel(candidates, benchmark, proxy, per_task=40, seed=17)
        self.assertEqual(len(selected), 40)
        again, _ = panel.select_panel(list(reversed(candidates)), benchmark, proxy, per_task=40, seed=17)
        self.assertEqual(selected, again)
        counts = Counter((profile.row_features(item)["dataset"], profile.row_features(item)["empty_status"]) for item in selected)
        self.assertEqual(set(counts.values()), {5})
        blocked_ids = {item["id"] for item in benchmark + proxy}
        blocked_paths = {path for item in benchmark + proxy for path in profile.image_paths(item, context="test")}
        for item in selected:
            self.assertNotIn(item["id"], blocked_ids)
            self.assertFalse(blocked_paths.intersection(profile.image_paths(item, context="test")))
            self.assertIn(item, candidates)
        self.assertEqual(sum(manifest["exclusions"].values()), 4)

    def test_shortage_names_exhausted_family_and_caps_realized_share(self):
        benchmark, candidates = fixture("benchmark", 1), fixture("pool", 8)
        candidates = [item for item in candidates if profile.dataset_family(item) != "pair:d"]
        selected, manifest = panel.select_panel(candidates, benchmark, [], per_task=40, seed=9)
        self.assertGreater(len(selected), 0)
        families = Counter(profile.dataset_family(item) for item in selected)
        self.assertTrue(all(count / len(selected) <= 0.35 for count in families.values()))
        shortages = [cell for cell in manifest["strata"] if cell["dataset"] == "pair:d"]
        self.assertTrue(shortages)
        self.assertTrue(all(cell["shortage_rows"] > 0 and cell["available_rows"] == 0 for cell in shortages))
        self.assertEqual(manifest["tasks"]["Defect Detection"]["shortage_rows"], 40 - len(selected))

    def test_unachievable_family_cap_reports_empty_task_not_dominant_fill(self):
        benchmark = fixture("benchmark", 1)
        candidates = [item for item in fixture("pool", 3) if profile.dataset_family(item) == "pair:a"]
        selected, manifest = panel.select_panel(candidates, benchmark, [], per_task=40, seed=9)
        self.assertEqual(selected, [])
        self.assertEqual(manifest.get("tasks", {}).get("Defect Detection", {}).get("shortage_rows"), 40)

    def test_cli_seals_inputs_and_writes_panel_profile_and_manifest_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = write_rows(root / "pool.jsonl", fixture("pool", 4))
            benchmark = write_rows(root / "benchmark.jsonl", fixture("benchmark", 1))
            proxy = write_rows(root / "proxy.jsonl", [])
            before = benchmark.read_bytes()
            args = ["--input", str(source), "--benchmark", str(benchmark), "--proxy", str(proxy),
                    "--output-dir", str(root / "output"), "--per-task", "24", "--seed", "17"]
            self.assertEqual(panel.main(args), 0)
            output = root / "output"
            self.assertEqual(set(p.name for p in output.iterdir()), {
                "validation_panel.jsonl", "validation_panel_profile.json", "PANEL_MANIFEST.md"})
            self.assertEqual(json.loads((output / "validation_panel_profile.json").read_text())["rows"], 24)
            payload = json.loads((output / "validation_panel_profile.json").read_text())
            for report in payload["panel_manifest"]["tasks"].values():
                self.assertTrue({"family_cap_effective", "family_cap_relaxed", "family_cap_reason"} <= report.keys())
                self.assertIsInstance(report["family_cap_relaxed"], bool)
                self.assertTrue(report["family_cap_reason"])
            markdown = (output / "PANEL_MANIFEST.md").read_text()
            self.assertIn(profile.sha256_file(benchmark), markdown)
            header = next(line for line in markdown.splitlines() if line.startswith("| Task |"))
            for field in ("family_cap_effective", "family_cap_relaxed", "family_cap_reason"):
                self.assertIn(field, header)
            self.assertEqual(benchmark.read_bytes(), before)
            self.assertEqual(panel.main(args), 2)


if __name__ == "__main__":
    unittest.main()
