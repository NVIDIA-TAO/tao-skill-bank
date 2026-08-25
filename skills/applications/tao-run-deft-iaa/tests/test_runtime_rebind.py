# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rebind_iaa_runtime as rebind_module  # noqa: E402
from runtime_binding import validate_runtime_lineage  # noqa: E402


class RuntimeRebindTests(unittest.TestCase):
    def _run_boundary(self, platform: str, committed_sequence: int) -> tuple[dict, pathlib.Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        results = pathlib.Path(temporary.name).resolve()
        old = "a" * 64
        state = {
            "schema_version": "3", "workflow": "tao-run-deft-iaa",
            "results_dir": str(results),
            "config": {"iaa_deft_bundle_sha256": old, "platform": platform},
            "test_committed_sequence": committed_sequence,
        }
        (results / "deft_state.json").write_text(json.dumps(state))
        if platform == "slurm" and committed_sequence == 5:
            action_dir = results / "iter_1" / "embeddings" / "target"
            action_dir.mkdir(parents=True)
            canonical_status = action_dir / "target_embed.status.json"
            (action_dir / "target_embed.action.json").write_text(json.dumps({
                "attempt": 1, "request_sha256": "1" * 64,
                "status_path": str(canonical_status),
            }))
            (action_dir / "target_embed.attempt-2.action.json").write_text(json.dumps({
                "attempt": 2, "request_sha256": "2" * 64,
                "status_path": str(canonical_status),
            }))
            (action_dir / "target_embed.attempt-1.status.json").write_text(json.dumps({
                "status": "error", "request_sha256": "1" * 64,
            }))
            canonical_status.write_text(json.dumps({
                "status": "ok", "request_sha256": "2" * 64,
            }))

        def evidence(root: pathlib.Path, sequence: int) -> pathlib.Path:
            path = root / "runtime_rebind" / f"validation-{sequence}.json"
            path.parent.mkdir()
            path.write_text(json.dumps({"result": "PASS"}))
            return path

        missing = mock.Mock(returncode=1, stdout="", stderr="not found")
        with (
            mock.patch.object(rebind_module, "audit", return_value={
                "errors": ["bundled IAA runtime changed after initialization"]
            }),
            mock.patch.object(rebind_module, "_validate_current_tree", side_effect=evidence),
            mock.patch.object(rebind_module, "_plugin_versions", return_value=("0.1.12", "0.4.0")),
        ):
            record = rebind_module.rebind(
                results, f"approved {platform} QA refresh", inspect_command=mock.Mock(return_value=missing)
            )
        return record, results

    def test_slurm_seq5_resume_rebinds_atomically(self):
        record, results = self._run_boundary("slurm", 5)
        state = json.loads((results / "deft_state.json").read_text())
        self.assertEqual(record["sequence"], 1)
        self.assertEqual(state["active_runtime_sha256"], record["new_sha256"])
        self.assertEqual(validate_runtime_lineage(state, results), [record])
        self.assertEqual(state["config"]["iaa_deft_bundle_sha256"], "a" * 64)

    def test_virtualenv_seq0_resume_rebinds_atomically(self):
        record, results = self._run_boundary("virtualenv", 0)
        state = json.loads((results / "deft_state.json").read_text())
        self.assertEqual(state["runtime_lineage"][0], record)
        self.assertEqual(validate_runtime_lineage(state, results)[0]["plugin_base_version"], "0.1.12")

    def test_created_endpoint_blocks_rebind(self):
        with tempfile.TemporaryDirectory() as raw:
            results = pathlib.Path(raw).resolve()
            state = {
                "schema_version": "3", "workflow": "tao-run-deft-iaa",
                "results_dir": str(results),
                "config": {"iaa_deft_bundle_sha256": "a" * 64, "platform": "docker"},
            }
            (results / "deft_state.json").write_text(json.dumps(state))
            created = mock.Mock(returncode=0, stdout=json.dumps([{"State": {"Status": "created"}}]), stderr="")
            with mock.patch.object(rebind_module, "audit", return_value={
                "errors": ["bundled IAA runtime changed after initialization"]
            }):
                with self.assertRaisesRegex(ValueError, "created endpoint blocks"):
                    rebind_module.rebind(results, "approved fix", inspect_command=mock.Mock(return_value=created))

    def test_remote_platform_rebind_never_invokes_local_docker(self):
        for platform in ("slurm", "brev", "kubernetes"):
            with self.subTest(platform=platform):
                state = {
                    "results_dir": "/tmp/run", "config": {"platform": platform},
                }
                inspect = mock.Mock(side_effect=FileNotFoundError("docker"))
                rebind_module._clean_endpoints(state, inspect_command=inspect)
                inspect.assert_not_called()

    def test_virtualenv_rebind_still_blocks_active_local_endpoint(self):
        state = {
            "results_dir": "/tmp/run", "config": {"platform": "virtualenv"},
        }
        created = mock.Mock(
            returncode=0, stdout=json.dumps([{"State": {"Status": "created"}}]), stderr=""
        )
        with self.assertRaisesRegex(ValueError, "created endpoint blocks"):
            rebind_module._clean_endpoints(state, inspect_command=mock.Mock(return_value=created))

    def test_clean_actions_distinguishes_slurm_preparation_from_generic_action(self):
        with tempfile.TemporaryDirectory() as raw:
            results = pathlib.Path(raw).resolve()
            runtime = results / "iter_1" / "datagen" / ".tao-runtime"
            runtime.mkdir(parents=True)
            prepared = runtime / "sdg.action.json"
            payload = {
                "schema_version": "1", "kind": "slurm_sdg_action",
                "workflow": "tao-run-deft-iaa", "platform": "slurm",
                "results_dir": str(results), "stage_dir": str(runtime.parent),
                "action_id": "sdg-test", "iteration": 1,
            }
            payload["request_sha256"] = __import__("hashlib").sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            prepared.write_text(json.dumps(payload))
            rebind_module._clean_actions(results)
            self.assertTrue(prepared.is_file())

            generic = runtime / "train.action.json"
            generic.write_text(json.dumps({
                "attempt": 1, "request_sha256": "b" * 64,
                "status_path": str(runtime / "missing.status.json"),
            }))
            with self.assertRaisesRegex(ValueError, "unfinalized action request"):
                rebind_module._clean_actions(results)

    def test_clean_actions_accepts_only_terminal_remote_slurm_mapping(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            results = root / "controller" / "run_1"
            runtime = results / "iter_1" / "datagen" / ".tao-runtime" / "controller"
            runtime.mkdir(parents=True)
            backend_results = pathlib.Path("/lustre/approved/run_1")
            backend_stage = backend_results / "iter_1" / "datagen"
            prepared = runtime / "sdg.action.json"
            payload = {
                "schema_version": "1", "kind": "slurm_sdg_action",
                "workflow": "tao-run-deft-iaa", "platform": "slurm",
                "results_dir": str(backend_results), "stage_dir": str(backend_stage),
                "action_id": "sdg-remote-test", "iteration": 1,
            }
            payload["request_sha256"] = __import__("hashlib").sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            prepared.write_text(json.dumps(payload))
            state_root = root / "state"
            jobs = state_root / "jobs"
            jobs.mkdir(parents=True)
            record_path = jobs / "job-1.json"
            record = {
                "id": "job-1", "platform": "slurm", "action": "sdg-remote-test",
                "results_dir": str(backend_stage), "terminal_state": "ERROR",
            }
            record_path.write_text(json.dumps(record))
            with mock.patch.dict("os.environ", {"TAO_STATE_DIR": str(state_root)}):
                rebind_module._clean_actions(results)
                record["terminal_state"] = None
                record_path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "active or mismatched job record"):
                    rebind_module._clean_actions(results)

            record["terminal_state"] = "ERROR"
            record_path.write_text(json.dumps(record))
            payload["results_dir"] = "/lustre/approved/other-run"
            payload["stage_dir"] = "/lustre/approved/other-run/iter_1/datagen"
            payload["request_sha256"] = __import__("hashlib").sha256(
                json.dumps(
                    {key: value for key, value in payload.items() if key != "request_sha256"},
                    sort_keys=True, separators=(",", ":"),
                ).encode()
            ).hexdigest()
            prepared.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "invalid backend run mapping"):
                rebind_module._clean_actions(results)

    def test_lineage_rejects_tampered_evidence_and_noop(self):
        record, results = self._run_boundary("slurm", 5)
        state = json.loads((results / "deft_state.json").read_text())
        pathlib.Path(record["evidence_path"]).write_text(json.dumps({"result": "FAIL"}))
        with self.assertRaisesRegex(ValueError, "evidence digest mismatch"):
            validate_runtime_lineage(state, results)
        state["runtime_lineage"][0]["new_sha256"] = state["runtime_lineage"][0]["old_sha256"]
        state["active_runtime_sha256"] = state["runtime_lineage"][0]["new_sha256"]
        state["runtime_lineage"][0]["evidence_sha256"] = __import__("hashlib").sha256(
            pathlib.Path(record["evidence_path"]).read_bytes()
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "no-op or digest downgrade"):
            validate_runtime_lineage(state, results)


if __name__ == "__main__":
    unittest.main()
