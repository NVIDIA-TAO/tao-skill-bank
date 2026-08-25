# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import tempfile
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import repair_sdg_normalize_freshness as repair  # noqa: E402
import run_sdg_stage as stage_runner  # noqa: E402


class SdgFreshnessRepairTests(unittest.TestCase):
    def _fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        results = pathlib.Path(temporary.name).resolve() / "run"
        datagen = results / "iter_1" / "datagen"
        dataset = datagen / "dataset"
        runtime = datagen / ".tao-runtime"
        status_dir = datagen / "status"
        for directory in (dataset, runtime, status_dir):
            directory.mkdir(parents=True, exist_ok=True)
        dataset_root = results.parent / "data" / "iaa"
        dataset_root.mkdir(parents=True)
        (dataset_root / "attribute_vocab.json").write_text('{"attributes": []}\n')
        (results / "config").mkdir()
        (results / "config" / "sdg_config.yaml").write_text("generation:\n  caption_policy: all\n")
        (results / "iaa_splits").mkdir()
        (results / "iaa_splits" / "eval_list.txt").write_text("eval.jpg\n")
        (results / "deft_state.json").write_text(json.dumps({
            "workflow": "tao-run-deft-iaa", "current_iteration": 1,
            "results_dir": str(results),
            "config": {"platform": "slurm", "dataset_root": str(dataset_root)},
        }))
        accepted_image = datagen / "accepted" / "source" / "generated.jpg"
        accepted_metadata = datagen / "accepted" / "source" / "output_metadata.json"
        label = datagen / "labels" / "source" / "task" / "open_qa.json"
        for path, content in (
            (accepted_image, b"generated-image"),
            (accepted_metadata, b'{"attribute_verification":{"passed":true}}\n'),
            (label, b'{"items":[]}\n'),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (datagen / "accepted_crop_manifest.json").write_text(json.dumps({
            "accepted": [{
                "source_key": "source", "attempt": 1,
                "image": str(accepted_image), "metadata": str(accepted_metadata),
            }]
        }) + "\n")
        for relative, content in (
            ("images/a.jpg", b"normalized-image"),
            ("captions/a.txt", b"person\n"),
            ("attribute_vocab.json", b'{"attributes": []}\n'),
        ):
            path = dataset / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        outputs = [dataset / name for name in repair.OUTPUTS]
        outputs[0].write_text(json.dumps({
            "num_pairs": 1, "num_source_images": 1, "rejected_samples_included": 0,
            "image_dir": str(dataset / "images"),
            "caption_dir": str(dataset / "captions"),
            "image_list_file": str(dataset / "sdg_image_list.txt"),
            "pairs_file": str(dataset / "sdg_pairs.json"),
            "attribute_vocab_file": str(dataset / "attribute_vocab.json"),
        }, indent=2, sort_keys=True) + "\n")
        outputs[1].write_text(json.dumps([{"image": "a.jpg", "caption": "person"}]) + "\n")
        outputs[2].write_text("a.jpg\n")
        second = 1_787_265_638_000_000_000
        for output in outputs:
            os.utime(output, ns=(second, second))
        status = {
            "schema_version": "1", "workflow": "tao-run-deft-iaa",
            "name": "sdg-normalize", "status": "ok", "exit_code": 0,
            "execution_platform": "slurm", "attempt": 1,
            "started_ns": second + 69_633_996, "finished_ns": second + 674_509_548,
            "finished_at": "2026-08-20T22:40:38.674510+00:00",
            "inputs": [], "fresh_outputs": [str(path) for path in outputs],
            "log_path": str(datagen / "logs" / "sdg-normalize.slurm.log"),
        }
        status_path = status_dir / "sdg-normalize.slurm.status.json"
        status_path.write_text(json.dumps(status))
        job_id, action_id = "iaa-sdg-test-123", "deft-iaa-sdg-test"
        request = {
            "schema_version": "1", "kind": "slurm_sdg_action",
            "workflow": "tao-run-deft-iaa", "platform": "slurm",
            "action_id": action_id, "attempt": 1,
            "expected_outputs": [*map(str, outputs), str(datagen / "sdg_execution_manifest.json")],
        }
        request["request_sha256"] = hashlib.sha256(
            json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        (runtime / f"sdg.action.{job_id}.json").write_text(json.dumps(request))
        (datagen / "endpoint_manifest.json").write_text(json.dumps({
            "job_id": job_id, "request_sha256": request["request_sha256"],
            "action_id": action_id, "attempt": 1,
        }))
        (datagen / f"slurm_sdg_terminal.{job_id}.json").write_text(json.dumps({
            "status": "ok", "job_id": job_id,
            "request_sha256": request["request_sha256"],
            "action_id": action_id, "attempt": 1,
            "expected_outputs": request["expected_outputs"],
        }))
        sentinels = []
        for relative in ("sdg_progress.json", "endpoint_pool.json"):
            path = datagen / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((relative + "-unchanged").encode())
            sentinels.append((path, path.read_bytes()))
        return results, datagen, outputs, sentinels

    def test_prepare_is_idempotent_and_touches_only_three_derived_outputs(self):
        results, _, outputs, sentinels = self._fixture()
        first = repair.prepare(results, 1)
        second = repair.prepare(results, 1)
        self.assertEqual(first["journal"], second["journal"])
        self.assertTrue(all(not path.exists() for path in outputs))
        self.assertTrue(all(path.read_bytes() == before for path, before in sentinels))
        source = (ROOT / "scripts" / "repair_sdg_normalize_freshness.py").read_text()
        self.assertNotIn("subprocess", source)
        self.assertNotIn("component_executor", source)

    def test_normalization_only_recompute_verifies_without_inference(self):
        results, datagen, outputs, sentinels = self._fixture()
        repair.prepare(results, 1)
        calls = []

        def normalize_only(_accepted, _labels, destination, _vocab, _eval, _policy):
            calls.append("normalize")
            destination.mkdir(parents=True)
            source = datagen / "dataset"
            for relative in ("images/a.jpg", "captions/a.txt", "attribute_vocab.json"):
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((source / relative).read_bytes())
            paths = repair._paths(results, 1)
            for output, backup in zip(
                (destination / name for name in repair.OUTPUTS), paths["backups"]
            ):
                output.write_bytes(backup.read_bytes())
            return destination / "sdg_manifest.json"

        original_normalize = repair.normalize_generated_pairs
        original_load_config = repair._load_config
        repair.normalize_generated_pairs = normalize_only
        repair._load_config = lambda _path: {"generation": {"caption_policy": "all"}}
        self.addCleanup(setattr, repair, "normalize_generated_pairs", original_normalize)
        self.addCleanup(setattr, repair, "_load_config", original_load_config)
        result = repair.recompute(results, 1)
        self.assertEqual(result["attempt"], 2)
        verified = repair.verify(results, 1)
        self.assertEqual(calls, ["normalize"])
        self.assertTrue(verified["byte_identical"])
        self.assertTrue(all(path.read_bytes() == before for path, before in sentinels))

    def test_prepared_repair_rejects_changed_label_input(self):
        results, datagen, _, _ = self._fixture()
        repair.prepare(results, 1)
        (datagen / "labels/source/task/open_qa.json").write_text('{"items":[{"answer":"changed"}]}\n')
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            repair.validate_prepared(results, 1)

    def test_repair_adapter_rejects_missing_prepared_journal(self):
        results, _, _, _ = self._fixture()
        with self.assertRaisesRegex(ValueError, "journal"):
            repair.validate_prepared(results, 1)

    def test_host_and_container_aliases_have_one_canonical_identity(self):
        canonical = pathlib.Path("/lustre/workspace/results/run")
        runtime = pathlib.Path("/results")
        allowed = canonical / "iter_1/datagen"
        host = allowed / "accepted/source/output.jpg"
        container = pathlib.Path("/results/iter_1/datagen/accepted/source/output.jpg")
        self.assertEqual(
            repair._canonical_mount_path(
                host, runtime_root=runtime, canonical_root=canonical,
                allowed_root=allowed, name="accepted image",
            ),
            host,
        )
        self.assertEqual(
            repair._canonical_mount_path(
                container, runtime_root=runtime, canonical_root=canonical,
                allowed_root=allowed, name="accepted image",
            ),
            host,
        )

    def test_unrelated_mount_alias_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "escapes"):
            repair._canonical_mount_path(
                pathlib.Path("/other/run/output.jpg"),
                runtime_root=pathlib.Path("/results"),
                canonical_root=pathlib.Path("/lustre/workspace/results/run"),
                allowed_root=pathlib.Path("/lustre/workspace/results/run/iter_1/datagen"),
                name="accepted image",
            )

    def test_failed_recompute_restores_exact_backups(self):
        results, datagen, outputs, sentinels = self._fixture()
        originals = [path.read_bytes() for path in outputs]
        repair.prepare(results, 1)
        outputs[0].write_text("corrupt recompute\n")
        restored = repair.restore(results, 1)
        self.assertEqual(restored["status"], "restored")
        self.assertEqual([path.read_bytes() for path in outputs], originals)
        failed = datagen / "freshness_repair" / "sdg-normalize-attempt-1" / "failed-recompute"
        self.assertEqual((failed / outputs[0].name).read_text(), "corrupt recompute\n")
        self.assertTrue(all(path.read_bytes() == before for path, before in sentinels))


if __name__ == "__main__":
    unittest.main()
