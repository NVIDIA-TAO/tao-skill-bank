# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import atomic_samples  # noqa: E402
import emit_mined_sharegpt  # noqa: E402
import task_mining_router  # noqa: E402


def _cached_pairs(root: pathlib.Path) -> tuple[list[dict], list[dict]]:
    records = [
        {
            "id": f"pair-{index}",
            "task_type": "Ref_based Defect Detection",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"images/golden-{index}.png"},
                        {"type": "image", "image": "images/shared-target.png"},
                        {"type": "text", "text": "detect"},
                    ],
                },
                {"role": "assistant", "content": f'[{index}, 2, 3, 4]'},
            ],
        }
        for index in range(2)
    ]
    # Only cached vector metadata is needed; no source images or canvases exist.
    samples = [
        atomic_samples.sample_from_record(record, media_root=root, context=record["id"])
        for record in records
    ]
    cached = [
        {
            "filepath": str(atomic_samples.pair_asset_path(
                root / "old-source-cache", sample["atomic_sample_id"]
            )),
            "embedding": [float(index == 0), float(index == 1)],
            "atomic_sample_id": sample["atomic_sample_id"],
            "sample_kind": "reference_pair",
            "routed_task_types": ["Ref_based Defect Detection"],
            "route_tier": "strict",
        }
        for index, sample in enumerate(samples)
    ]
    (root / "new-query-cache").mkdir()
    return records, cached


class Cosmos3CachedPairLookupTests(unittest.TestCase):
    def test_router_reuses_cached_vectors_without_rendering_and_preserves_dedup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            records, cached = _cached_pairs(root)
            # Do not rely on an atomic-id column in legacy source parquet files.
            source_rows = [
                {"filepath": row["filepath"], "embedding": row["embedding"]}
                for row in [cached[0], cached[0], cached[1]]
            ]
            target_rows = [{
                "filepath": "query.png", "target_id": "query",
                "task_types": ["Ref_based Defect Detection"],
                "defect_detection_evidence": [], "embedding": [1.0, 0.5],
            }]
            with mock.patch.object(
                atomic_samples, "materialize_pair_asset",
                side_effect=AssertionError("cached routing must never render"),
            ) as render:
                selected, summary = task_mining_router.route_candidates(
                    target_rows, source_rows, records, media_root=root,
                    pair_assets_dir=root / "new-query-cache", mode="task_strict",
                    top_k_per_target=2, min_similarity=-1.0,
                )
                self.assertEqual(
                    [row["atomic_sample_id"] for row in selected],
                    [row["atomic_sample_id"] for row in cached],
                )
                self.assertEqual(summary["source_images"], 2)
                self.assertEqual(summary["duplicate_source_embeddings_collapsed"], 1)
                emitted, _ = emit_mined_sharegpt.emit_records(
                    selected, records, media_root=root, relative=True,
                    pair_assets_dir=root / "new-query-cache",
                )
                self.assertEqual(emitted, records)
                # Conflicting vectors for the same atomic source must still fail.
                conflicting = source_rows + [{**source_rows[0], "embedding": [0.0, 1.0]}]
                with self.assertRaisesRegex(ValueError, "conflicting embeddings"):
                    task_mining_router.route_candidates(
                        target_rows, conflicting, records, media_root=root,
                        pair_assets_dir=root / "new-query-cache", mode="task_strict",
                        top_k_per_target=2, min_similarity=-1.0,
                    )
                render.assert_not_called()
            self.assertEqual(list((root / "new-query-cache").iterdir()), [])

    def test_emitter_reuses_cached_pair_ids_and_hash_aliases_without_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            records, cached = _cached_pairs(root)
            original = copy.deepcopy(records)
            with mock.patch.object(
                atomic_samples, "materialize_pair_asset",
                side_effect=AssertionError("cached emission must never render"),
            ) as render:
                # Exercise both modern atomic matching and old-cache hash aliases.
                for atomic_ids in (True, False):
                    with self.subTest(atomic_ids=atomic_ids):
                        mined = [
                            copy.deepcopy(row) for row in [cached[0], cached[0], cached[1]]
                        ]
                        if not atomic_ids:
                            for row in mined:
                                del row["atomic_sample_id"]
                        emitted, summary = emit_mined_sharegpt.emit_records(
                            mined, records, media_root=root, relative=True,
                            pair_assets_dir=root / "new-query-cache",
                        )
                        self.assertEqual(emitted, original)
                        self.assertEqual(records, original)
                        self.assertEqual(summary["output_records"], 2)
                        self.assertEqual(summary["duplicates_skipped"], 1)
                        self.assertEqual(summary["atomic_reference_pairs"], 2)
                        self.assertEqual(
                            summary["match_modes"], {"atomic" if atomic_ids else "name": 2}
                        )
                        render.assert_not_called()
            self.assertEqual(list((root / "new-query-cache").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
