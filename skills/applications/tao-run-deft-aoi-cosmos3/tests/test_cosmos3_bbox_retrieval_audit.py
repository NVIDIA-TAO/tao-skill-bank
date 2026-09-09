# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import audit_bbox_retrieval  # noqa: E402


def _detection_row(record_id: str, image_path: str, boxes: list[dict], dataset: str) -> dict:
    return {
        "id": record_id,
        "dataset": dataset,
        "task_type": "Defect Detection",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "detect defects"},
                ],
            },
            {
                "role": "assistant",
                "content": "```json\n" + json.dumps(boxes) + "\n```",
            },
        ],
    }


class _ExactEvaluatorFixture:
    @staticmethod
    def parse_boxes(text: str) -> tuple[list[tuple[float, float, float, float]], bool]:
        value = json.loads(text)
        return [tuple(float(x) for x in item["bbox_2d"]) for item in value], True

    @staticmethod
    def canonicalize_prediction_boxes(
        boxes: list[tuple[float, float, float, float]], coordinate_order: str
    ) -> list[tuple[float, float, float, float]]:
        if coordinate_order != "xyxy":
            raise AssertionError("fixture accepts xyxy only")
        return boxes

    @staticmethod
    def one_to_one_detection_counts(
        ground_truth: list[tuple[float, float, float, float]],
        predictions: list[tuple[float, float, float, float]],
        threshold: float,
    ) -> tuple[int, int, int]:
        return audit_bbox_retrieval.unmatched_ground_truth_indices(
            ground_truth, predictions, threshold=threshold
        )[1]


class Cosmos3BBoxRetrievalAuditTests(unittest.TestCase):
    def test_preserves_labels_and_maps_canonical_phenotypes(self) -> None:
        answer = "```json\n" + json.dumps(
            [
                {
                    "bbox_2d": [10, 20, 30, 40],
                    "label": "PCB Conductor Open / Copper Loss",
                },
                {
                    "bbox_2d": [50, 60, 70, 80],
                    "label": "Tombstoning",
                },
            ]
        ) + "\n```"

        objects = audit_bbox_retrieval.extract_labeled_boxes(answer, context="fixture")

        self.assertEqual([item["bbox_2d"] for item in objects], [[10, 20, 30, 40], [50, 60, 70, 80]])
        self.assertEqual(
            [item["canonical_phenotype"] for item in objects],
            ["conductor_open", "component_orientation"],
        )
        self.assertEqual(
            audit_bbox_retrieval.canonical_phenotype("Open / Poor Solder Joint"),
            "solder_open",
        )
        with self.assertRaisesRegex(ValueError, "unmapped defect label"):
            audit_bbox_retrieval.canonical_phenotype("invented label")

    def test_hungarian_fn_extraction_uses_strict_iou_boundary(self) -> None:
        ground_truth = [(0.0, 0.0, 2.0, 1.0)]
        boundary_prediction = [(0.0, 0.0, 1.0, 1.0)]

        unmatched, counts = audit_bbox_retrieval.unmatched_ground_truth_indices(
            ground_truth, boundary_prediction, threshold=0.5
        )

        self.assertEqual(unmatched, [0])
        self.assertEqual(counts, (0, 1, 1))
        unmatched, counts = audit_bbox_retrieval.unmatched_ground_truth_indices(
            [(0.0, 0.0, 1.0, 1.0), (2.0, 2.0, 3.0, 3.0)],
            [(0.0, 0.0, 1.0, 1.0)],
            threshold=0.5,
        )
        self.assertEqual(unmatched, [1])
        self.assertEqual(counts, (1, 0, 1))

    def test_context_crop_is_exif_aware_and_mean_pads_edges(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            oriented = root / "oriented.jpg"
            image = Image.new("RGB", (4, 2), color=(20, 40, 60))
            exif = image.getexif()
            exif[274] = 6
            image.save(oriented, exif=exif)

            displayed = audit_bbox_retrieval.load_display_image(oriented)
            self.assertEqual(displayed.size, (2, 4))
            crop, metadata = audit_bbox_retrieval.render_context_crop(
                displayed,
                [0, 0, 500, 500],
                scale=3.0,
            )

            self.assertEqual(crop.size, (224, 224))
            self.assertGreater(metadata["padding_fraction"], 0.0)
            corner = crop.getpixel((0, 0))
            self.assertTrue(all(abs(value - expected) <= 2 for value, expected in zip(corner, (20, 40, 60))))
            with self.assertRaisesRegex(ValueError, "positive area"):
                audit_bbox_retrieval.render_context_crop(
                    displayed, [10, 10, 10, 20], scale=1.5
                )

    def test_parent_dedup_is_deterministic_and_never_emits_crop_paths(self) -> None:
        candidates = [
            {
                "candidate_id": "c-b",
                "parent_id": "parent-1",
                "parent_filepath": "/parents/one.png",
                "crop_filepath_1p5": "/crops/b.png",
                "canonical_phenotype": "short",
                "dataset": "source-a",
            },
            {
                "candidate_id": "c-a",
                "parent_id": "parent-1",
                "parent_filepath": "/parents/one.png",
                "crop_filepath_1p5": "/crops/a.png",
                "canonical_phenotype": "open",
                "dataset": "source-a",
            },
            {
                "candidate_id": "c-c",
                "parent_id": "parent-2",
                "parent_filepath": "/parents/two.png",
                "crop_filepath_1p5": "/crops/c.png",
                "canonical_phenotype": "open",
                "dataset": "source-b",
            },
        ]

        ranked = audit_bbox_retrieval.parent_topk(
            candidates, np.asarray([0.9, 0.9, 0.8]), top_k=20
        )
        emitted = audit_bbox_retrieval.parent_only_rows(ranked)

        self.assertEqual([item["candidate_id"] for item in ranked], ["c-a", "c-c"])
        self.assertEqual([item["parent_id"] for item in ranked], ["parent-1", "parent-2"])
        self.assertEqual(
            [item["filepath"] for item in emitted],
            ["/parents/one.png", "/parents/two.png"],
        )
        self.assertTrue(all("/crops/" not in item["filepath"] for item in emitted))

    def test_metrics_bootstrap_and_sealed_gates_are_query_paired(self) -> None:
        queries = [
            {"query_id": "q-a", "dataset": "target-a", "canonical_phenotype": "open"},
            {"query_id": "q-b", "dataset": "target-b", "canonical_phenotype": "short"},
        ]

        def neighbors(relevant_counts: tuple[int, int]) -> dict[str, list[dict]]:
            output: dict[str, list[dict]] = {}
            for query, count in zip(queries, relevant_counts, strict=True):
                rows = []
                for index in range(10):
                    rows.append(
                        {
                            "parent_id": f"{query['query_id']}-p{index}",
                            "candidate_id": f"{query['query_id']}-c{index}",
                            "candidate_dataset": f"source-{index % 2}",
                            "candidate_phenotype": (
                                query["canonical_phenotype"] if index < count else "other"
                            ),
                            "score": 1.0 - index / 100.0,
                        }
                    )
                output[query["query_id"]] = rows
            return output

        a0 = audit_bbox_retrieval.compute_arm_metrics(queries, neighbors((1, 1)))
        a1 = audit_bbox_retrieval.compute_arm_metrics(queries, neighbors((3, 3)))
        bootstrap = audit_bbox_retrieval.paired_bootstrap_p10(
            queries,
            a1["per_query"],
            a0["per_query"],
            resamples=1000,
            seed=20260908,
        )
        lineage = {
            "lineage_rate": 1.0,
            "shared_image_leaks": 0,
            "crop_emitted_count": 0,
        }
        gate_a1 = {**a1, "unique_relevant_parents": 2}
        gates = audit_bbox_retrieval.evaluate_gates(
            a0,
            gate_a1,
            bootstrap,
            lineage,
        )

        self.assertAlmostEqual(a0["dataset_balanced_macro"]["p_at_10"], 0.1)
        self.assertAlmostEqual(a1["dataset_balanced_macro"]["p_at_10"], 0.3)
        self.assertAlmostEqual(bootstrap["delta"], 0.2)
        self.assertGreater(bootstrap["ci95_lower"], 0.0)
        self.assertTrue(gates["primary_precision"])
        self.assertTrue(gates["lineage"])
        self.assertFalse(gates["unique_relevant_parents"])
        self.assertFalse(gates["go"])

    def test_lineage_gate_detects_same_bytes_under_a_different_path(self) -> None:
        queries = [
            {
                "query_id": "q",
                "parent_id": "proxy-parent",
                "parent_filepath": "/proxy/renamed.png",
                "parent_sha256": "a" * 64,
            }
        ]
        candidates = [
            {
                "candidate_id": "c",
                "parent_id": "mining-parent",
                "parent_filepath": "/mining/original.png",
                "parent_sha256": "a" * 64,
            }
        ]

        summary = audit_bbox_retrieval.audit_lineage(
            queries,
            candidates,
            benchmark_attestation={"benchmark:mining": 0},
        )

        self.assertEqual(summary["proxy_mining_content_sha_leaks"], 1)
        self.assertEqual(summary["shared_image_leaks"], 1)
        self.assertEqual(summary["lineage_rate"], 1.0)

    def test_object_builders_keep_every_box_and_reuse_parent_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            image_path = root / "images" / "board.png"
            image_path.parent.mkdir()
            Image.new("RGB", (40, 20), color=(80, 100, 120)).save(image_path)
            boxes = [
                {"bbox_2d": [0, 0, 250, 500], "label": "Missing Component"},
                {
                    "bbox_2d": [500, 250, 1000, 1000],
                    "label": "Excess Solder / Solder Bridge",
                },
            ]
            record = _detection_row("row", "images/board.png", boxes, "fixture")

            candidates = audit_bbox_retrieval.build_candidate_objects(
                [record],
                media_root=root,
                crop_root=root / "crops",
                context_scales=(1.5, 3.0),
            )
            parallel_candidates = audit_bbox_retrieval.build_candidate_objects(
                [record],
                media_root=root,
                crop_root=root / "parallel-crops",
                context_scales=(1.5, 3.0),
                workers=2,
            )
            queries, reconciliation = audit_bbox_retrieval.extract_fn_objects(
                [record],
                [{"id": "row", "raw_prediction": "[]"}],
                evaluator=_ExactEvaluatorFixture(),
                variant="reference",
                media_root=root,
                crop_root=root / "query-crops",
                context_scales=(1.5, 3.0),
            )

            self.assertEqual(len(candidates), 2)
            self.assertEqual(
                [item["candidate_id"] for item in candidates],
                [item["candidate_id"] for item in parallel_candidates],
            )
            self.assertEqual(len({item["parent_id"] for item in candidates}), 1)
            self.assertEqual(len(queries), 2)
            self.assertEqual(len({item["query_id"] for item in queries}), 2)
            self.assertEqual(reconciliation["false_negatives"], 2)
            self.assertEqual(reconciliation["evaluator_count_mismatches"], 0)
            self.assertTrue(
                all(
                    pathlib.Path(item["crops"][scale]["filepath"]).is_file()
                    for item in candidates + queries
                    for scale in ("1.5", "3.0")
                )
            )

    def test_source_group_normalizes_known_augmentation_prefixes(self) -> None:
        original = "/data/Tiny/images/val/short_07_2_600.jpg"
        augmented = "/data/Tiny/images/val/rotation_90_light_05_short_07_2_600.jpg"

        self.assertEqual(
            audit_bbox_retrieval.derive_source_group_id("Tiny", original),
            audit_bbox_retrieval.derive_source_group_id("Tiny", augmented),
        )

    def test_query_parent_staging_preserves_parent_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            parent = root / "foreign-user" / "board.JPG"
            parent.parent.mkdir()
            Image.new("RGB", (12, 8), color=(10, 20, 30)).save(parent)
            parent_sha256 = audit_bbox_retrieval._sha256(parent)
            queries = [
                {"parent_filepath": str(parent), "parent_sha256": parent_sha256},
                {"parent_filepath": str(parent), "parent_sha256": parent_sha256},
            ]

            count = audit_bbox_retrieval.stage_query_parent_images(
                queries,
                stage_root=root / "audit" / "embedding_sources" / "query_parents",
            )

            staged = pathlib.Path(queries[0]["embedding_filepath"])
            self.assertEqual(count, 1)
            self.assertEqual(queries[0]["parent_filepath"], str(parent))
            self.assertEqual(queries[0]["embedding_filepath"], queries[1]["embedding_filepath"])
            self.assertTrue(staged.is_relative_to((root / "audit").resolve()))
            self.assertEqual(audit_bbox_retrieval._sha256(staged), parent_sha256)

    def test_embedding_loader_filters_cache_and_all_arms_rank_parents(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            parquet = root / "embeddings.parquet"
            pq.write_table(
                pa.table(
                    {
                        "filepath": ["unused", "query-parent", "c1", "c2", "c3", "q15", "q30"],
                        "embedding": [
                            [0.5, 0.5],
                            [1.0, 0.0],
                            [0.0, 1.0],
                            [1.0, 0.0],
                            [-1.0, 0.0],
                            [0.0, 1.0],
                            [-1.0, 0.0],
                        ],
                    }
                ),
                parquet,
                row_group_size=2,
            )
            filtered = audit_bbox_retrieval._load_required_embeddings(
                parquet, {"query-parent", "q15", "q30", "c1", "c2", "c3"}
            )
            self.assertNotIn("unused", filtered)

            candidates = [
                {
                    "candidate_id": f"candidate-{index}",
                    "parent_id": f"parent-{index}",
                    "parent_filepath": f"parent-path-{index}",
                    "dataset": "source",
                    "canonical_phenotype": "open" if index != 2 else "short",
                    "bbox_2d": [0, 0, 500, 500],
                    "source_group_id": f"group-{index}",
                    "crops": {
                        "1.5": {"filepath": f"c{index}"},
                        "3.0": {"filepath": f"c{index}"},
                    },
                }
                for index in (1, 2, 3)
            ]
            query = {
                "query_id": "query",
                "parent_filepath": "query-parent",
                "embedding_filepath": "query-parent",
                "crops": {
                    "1.5": {"filepath": "q15"},
                    "3.0": {"filepath": "q30"},
                },
            }
            a0 = {
                "parent-path-1": np.asarray([1.0, 0.0]),
                "parent-path-2": np.asarray([0.0, 1.0]),
                "parent-path-3": np.asarray([-1.0, 0.0]),
            }
            ranked = audit_bbox_retrieval.rank_retrieval_arms(
                candidates,
                [query],
                a0_embeddings=a0,
                crop_embeddings=filtered,
                top_k=2,
                query_batch_size=1,
            )

            self.assertEqual(
                ranked["whole_image_current"]["query"][0]["candidate_id"],
                "candidate-1",
            )
            self.assertEqual(
                ranked["bbox_context_1p5"]["query"][0]["candidate_id"],
                "candidate-1",
            )
            self.assertTrue(
                all(
                    "crops" not in neighbor
                    for arm in audit_bbox_retrieval.ARMS
                    for neighbor in ranked[arm]["query"]
                )
            )

    def test_human_selection_is_deterministic_and_stratified(self) -> None:
        queries = [
            {
                "query_id": f"q-{index}",
                "dataset": f"dataset-{index % 2}",
                "canonical_phenotype": f"class-{index % 2}",
                "bbox_area_quartile": f"Q{index % 4 + 1}",
            }
            for index in range(12)
        ]

        first = audit_bbox_retrieval._stratified_human_query_ids(
            queries, count=8, seed=20260908
        )
        second = audit_bbox_retrieval._stratified_human_query_ids(
            queries, count=8, seed=20260908
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        self.assertEqual(len(first), 8)

    def test_prepare_compute_integration_keeps_report_gates_before_results(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            media = root / "media"
            media.mkdir()
            proxy_image = media / "proxy.png"
            Image.new("RGB", (32, 32), color=(120, 80, 40)).save(proxy_image)
            candidate_rows = []
            candidate_paths = []
            for index in range(3):
                image_path = media / f"candidate-{index}.png"
                Image.new("RGB", (32, 32), color=(20 + 20 * index, 50, 90)).save(image_path)
                candidate_paths.append(str(image_path.resolve()))
                candidate_rows.append(
                    _detection_row(
                        f"candidate-{index}",
                        image_path.name,
                        [
                            {
                                "bbox_2d": [100, 100, 700, 700],
                                "label": "Missing Component" if index < 2 else "Other",
                            }
                        ],
                        f"source-{index}",
                    )
                )
            proxy_row = _detection_row(
                "proxy",
                proxy_image.name,
                [{"bbox_2d": [0, 0, 400, 400], "label": "Missing Component"}],
                "target",
            )

            def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )

            proxy_path = root / "proxy.jsonl"
            mining_path = root / "mining.jsonl"
            reference_path = root / "reference.jsonl"
            zero_path = root / "zero.jsonl"
            write_jsonl(proxy_path, [proxy_row])
            write_jsonl(mining_path, candidate_rows)
            write_jsonl(reference_path, [{"id": "proxy", "raw_prediction": "[]"}])
            write_jsonl(zero_path, [{"id": "proxy", "raw_prediction": "[]"}])
            evaluator_path = root / "evaluator.py"
            evaluator_path.write_text(
                "import json\n"
                "def parse_boxes(text):\n"
                "    return [tuple(x['bbox_2d']) for x in json.loads(text)], True\n"
                "def canonicalize_prediction_boxes(boxes, coordinate_order):\n"
                "    assert coordinate_order == 'xyxy'\n"
                "    return boxes\n"
                "def one_to_one_detection_counts(ground_truth, predictions, threshold):\n"
                "    assert threshold == 0.5\n"
                "    return (0, len(predictions), len(ground_truth))\n",
                encoding="utf-8",
            )
            a0_path = root / "a0.parquet"
            pq.write_table(
                pa.table(
                    {
                        "filepath": candidate_paths,
                        "embedding": [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
                    }
                ),
                a0_path,
            )
            encoder_manifest = root / "encoder.json"
            encoder_manifest.write_text(
                json.dumps(
                    {
                        "encoder": {
                            "model_id": "fixture/siglip",
                            "snapshot_revision": "fixed",
                        }
                    }
                ),
                encoding="utf-8",
            )
            split_attestation = root / "split.json"
            split_attestation.write_text(
                json.dumps(
                    {"target_overlap": {"proxy:mining": 0, "benchmark:mining": 0}}
                ),
                encoding="utf-8",
            )
            output = root / "audit"
            output.mkdir()
            report = output / "REPORT.md"
            report.write_text(
                "# report\n\n## Sealed gates\n\nGates were fixed first.\n\n"
                "## Results\n\n_Pending computation._\n\n"
                "## Decision\n\n_Pending computation._\n",
                encoding="utf-8",
            )
            report_sha = audit_bbox_retrieval._sha256(report)
            prepared = audit_bbox_retrieval.prepare_audit(
                proxy=proxy_path,
                reference_predictions=reference_path,
                zero_shot_predictions=zero_path,
                mining=mining_path,
                evaluator_path=evaluator_path,
                media_root=media,
                a0_source_embeddings=a0_path,
                encoder_manifest=encoder_manifest,
                split_attestation=split_attestation,
                output_dir=output,
                preregistered_report_sha256=report_sha,
            )
            embedding_inputs = pq.read_table(
                prepared["artifacts"]["embedding_inputs"]["path"]
            ).column("filepath").to_pylist()
            crop_embeddings = root / "crop-embeddings.parquet"
            pq.write_table(
                pa.table(
                    {
                        "filepath": embedding_inputs,
                        "embedding": [
                            [1.0, float(index % 3)] for index in range(len(embedding_inputs))
                        ],
                    }
                ),
                crop_embeddings,
            )
            snapshot = root / "models" / "snapshots" / "fixed"
            snapshot.mkdir(parents=True)
            crop_spec = root / "image_embeddings.yaml"
            crop_spec.write_text(
                f"input_parquet: {prepared['artifacts']['embedding_inputs']['path']}\n"
                f"output_parquet: {crop_embeddings}\n"
                "model: SigLIP\n"
                f"model_path: {snapshot}\n"
                "batch_size: 8\n",
                encoding="utf-8",
            )
            manifest = audit_bbox_retrieval.compute_audit(
                output_dir=output,
                crop_embeddings=crop_embeddings,
                crop_embedding_spec=crop_spec,
                job_id="fixture-job",
                preregistered_report_sha256=report_sha,
                human_pair_count=1,
            )

            report_text = report.read_text(encoding="utf-8")
            self.assertLess(report_text.index("## Sealed gates"), report_text.index("## Results"))
            self.assertNotIn("Pending computation", report_text)
            self.assertEqual(manifest["schema_version"], "audit_bbox_retrieval_v1")
            self.assertFalse(manifest["benchmark_read"])
            self.assertEqual(manifest["a0"]["candidate_parent_rows_reembedded"], 0)
            self.assertTrue((output / "human_pairs" / "pair_001.png").is_file())
            self.assertTrue((output / "metrics.json").is_file())
            self.assertTrue((output / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
