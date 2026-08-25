# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "coarse_freshness_commit", ROOT / "scripts" / "commit_stage.py"
)
commit = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(commit)


class CoarseSdgFreshnessTests(unittest.TestCase):
    def _fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        scope = pathlib.Path(temporary.name).resolve() / "iter_1"
        datagen = scope / "datagen"
        dataset = datagen / "dataset"
        status_dir = datagen / "status"
        runtime = datagen / ".tao-runtime"
        for directory in (dataset, status_dir, runtime):
            directory.mkdir(parents=True, exist_ok=True)
        outputs = [
            dataset / "sdg_manifest.json",
            dataset / "sdg_pairs.json",
            dataset / "sdg_image_list.txt",
        ]
        for index, path in enumerate(outputs):
            path.write_text(json.dumps({"index": index}) + "\n")
        second = 1_787_265_638_000_000_000
        for path in outputs:
            os.utime(path, ns=(second, second))
        started_ns = second + 69_633_996
        finished_ns = second + 674_509_548
        inputs = []
        pre_path = status_dir / "sdg-normalize.slurm.pre-action.json"
        pre = {
            "schema_version": "1", "workflow": "tao-run-deft-iaa",
            "name": "sdg-normalize", "execution_platform": "slurm",
            "attempt": 2, "started_ns": started_ns, "inputs": inputs,
            "outputs": [
                {"path": str(path), "absent": True} for path in outputs
            ],
        }
        pre_path.write_text(json.dumps(pre, sort_keys=True) + "\n")
        job_id = "iaa-sdg-test-123"
        action_id = "deft-iaa-sdg-test"
        request = {
            "schema_version": "1", "kind": "slurm_sdg_action",
            "workflow": "tao-run-deft-iaa", "platform": "slurm",
            "job_placeholder": True, "action_id": action_id, "attempt": 1,
            "expected_outputs": [*map(str, outputs), str(datagen / "sdg_execution_manifest.json")],
        }
        request["request_sha256"] = hashlib.sha256(
            json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        request_path = runtime / f"sdg.action.{job_id}.json"
        request_path.write_text(json.dumps(request))
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
        status_path = status_dir / "sdg-normalize.slurm.status.json"
        payload = {
            "schema_version": "1", "workflow": "tao-run-deft-iaa",
            "name": "sdg-normalize", "execution_platform": "slurm",
            "attempt": 2, "started_ns": started_ns, "finished_ns": finished_ns,
            "inputs": inputs, "fresh_outputs": [str(path) for path in outputs],
            "pre_action": {
                "path": str(pre_path),
                "sha256": hashlib.sha256(pre_path.read_bytes()).hexdigest(),
            },
            "output_evidence": [{
                "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
            } for path in outputs],
        }
        return scope, datagen, status_path, outputs, pre_path, pre, payload

    def test_exact_same_second_lustre_evidence_is_accepted(self):
        scope, _, status, outputs, _, _, payload = self._fixture()
        self.assertTrue(commit._coarse_slurm_sdg_freshness_attested(
            payload, outputs, scope=scope, status_path=status,
        ))

    def test_same_second_stale_output_without_absence_or_digest_is_rejected(self):
        scope, _, status, outputs, pre_path, pre, payload = self._fixture()
        cases = ("absence", "digest")
        for case in cases:
            with self.subTest(case=case):
                candidate = json.loads(json.dumps(payload))
                if case == "absence":
                    changed = json.loads(json.dumps(pre))
                    changed["outputs"][0]["absent"] = False
                    pre_path.write_text(json.dumps(changed, sort_keys=True) + "\n")
                    candidate["pre_action"]["sha256"] = hashlib.sha256(
                        pre_path.read_bytes()
                    ).hexdigest()
                else:
                    pre_path.write_text(json.dumps(pre, sort_keys=True) + "\n")
                    candidate["pre_action"]["sha256"] = hashlib.sha256(
                        pre_path.read_bytes()
                    ).hexdigest()
                    candidate["output_evidence"][0]["sha256"] = "0" * 64
                self.assertFalse(commit._coarse_slurm_sdg_freshness_attested(
                    candidate, outputs, scope=scope, status_path=status,
                ))

    def test_remote_absolute_paths_map_to_exact_synchronized_artifacts(self):
        scope, datagen, status, outputs, pre_path, pre, payload = self._fixture()
        local_root = scope.parent
        remote_root = pathlib.PurePosixPath("/lustre/workspace/results") / local_root.name

        def remote(path: pathlib.Path) -> str:
            return str(remote_root.joinpath(*path.relative_to(local_root).parts))

        pre["outputs"] = [
            {"path": remote(path), "absent": True} for path in outputs
        ]
        pre_path.write_text(json.dumps(pre, sort_keys=True) + "\n")
        payload["pre_action"] = {
            "path": remote(pre_path),
            "sha256": hashlib.sha256(pre_path.read_bytes()).hexdigest(),
        }
        payload["fresh_outputs"] = [remote(path) for path in outputs]
        for path, record in zip(outputs, payload["output_evidence"]):
            record["path"] = remote(path)
        job_id = json.loads((datagen / "endpoint_manifest.json").read_text())["job_id"]
        request_path = datagen / ".tao-runtime" / f"sdg.action.{job_id}.json"
        request = json.loads(request_path.read_text())
        request["expected_outputs"] = [
            *[remote(path) for path in outputs], remote(datagen / "sdg_execution_manifest.json")
        ]
        request.pop("request_sha256")
        request["request_sha256"] = hashlib.sha256(
            json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        request_path.write_text(json.dumps(request))
        endpoint = json.loads((datagen / "endpoint_manifest.json").read_text())
        endpoint["request_sha256"] = request["request_sha256"]
        (datagen / "endpoint_manifest.json").write_text(json.dumps(endpoint))
        terminal_path = datagen / f"slurm_sdg_terminal.{job_id}.json"
        terminal = json.loads(terminal_path.read_text())
        terminal["request_sha256"] = request["request_sha256"]
        terminal["expected_outputs"] = request["expected_outputs"]
        terminal_path.write_text(json.dumps(terminal))
        self.assertTrue(commit._coarse_slurm_sdg_freshness_attested(
            payload, outputs, scope=scope, status_path=status,
        ))


if __name__ == "__main__":
    unittest.main()
