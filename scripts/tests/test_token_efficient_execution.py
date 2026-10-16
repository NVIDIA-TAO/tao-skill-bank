# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline driver regressions; never invoke Pi, Docker, or a model endpoint."""

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
APPS = ROOT / "skills/applications"
KIT = ROOT / "skills/core/tao-token-efficient-execution"


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rd = self.root / "ws/results/run_fixture"
        self.rd.mkdir(parents=True)
        self.run_home = self.root / "driver"
        self.run_home.mkdir()
        self.calls = self.root / "calls"
        self.shell = self.root / "mock-tools.sh"
        self.shell.write_text("""
sleep() { :; }
working() { return 1; }
docker() { return 0; }
pgrep() { return 1; }
find() {
  case "$*" in
    *"-maxdepth 1"*) printf '1 %s\\n' "$TEST_RD" ;;
  esac
}
timeout() {
  echo session >> "$TEST_CALLS"
  case "$TEST_PACK" in
    deft) echo '{"status":"ok","stage":"evaluate","iter":"baseline"}' >> "$TEST_RD/loop_log.jsonl" ;;
    *) echo 'preflight ok' >> "$TEST_RD/progress.log" ;;
  esac
}
""")
        scripts = self.root / "skill/scripts"
        scripts.mkdir(parents=True)
        wrapper = scripts / "deft_python.sh"
        wrapper.write_text("""#!/bin/bash
echo "$(basename "$1")" >> "$TEST_CALLS"
case "$1" in
  *prepare_inference_spec.py) exit "$TEST_PREPARE_EXIT" ;;
  *audit_deft_run.py) exit "$TEST_AUDIT_EXIT" ;;
esac
exit 99
""")
        wrapper.chmod(0o755)
        (self.rd / "progress.log").touch()
        self.env = {
            "PATH": os.environ["PATH"],
            "HOME": os.environ["HOME"],
            "BASH_ENV": str(self.shell),
            "KIT_ENV": str(self.root / "no-config"),
            "WS": str(self.root / "ws"),
            "RUN_HOME": str(self.run_home),
            "SKILL_ROOT": str(scripts.parent),
            "CARDS": str(APPS / "tao-run-deft-aoi/cards"),
            "VENV": sys.prefix,
            "MODEL": "test/no-network",
            "TEST_RD": str(self.rd),
            "TEST_CALLS": str(self.calls),
            "TEST_PREPARE_EXIT": "0",
            "TEST_AUDIT_EXIT": "0",
        }

    def run_driver(self, pack="deft"):
        self.env["TEST_PACK"] = pack
        name = "tao-run-deft-aoi" if pack == "deft" else "tao-run-automl"
        self.env["CARDS"] = str(APPS / name / "cards")
        return subprocess.run(
            ["bash", str(APPS / name / "cards/driver.sh")],
            env=self.env, capture_output=True, text=True, timeout=20,
        )

    def terminal_log(self, error=False):
        records = [{"status": "ok", "stage": "loop_stop", "iter": "iter3"}]
        if error:
            records.append({"status": "error", "stage": "loop_stop", "iter": "iter3"})
        (self.rd / "loop_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))

    def test_deft_error_takes_precedence_over_loop_stop(self):
        self.terminal_log(error=True)
        result = self.run_driver()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(self.calls.exists())

    def test_deft_interrupted_terminal_card_gets_finalized(self):
        self.terminal_log()
        result = self.run_driver()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls.read_text().splitlines(),
                         ["prepare_inference_spec.py", "audit_deft_run.py"])

    def test_deft_handoff_failure_is_not_completion(self):
        self.terminal_log()
        self.env["TEST_PREPARE_EXIT"] = "1"
        result = self.run_driver()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(self.calls.read_text().splitlines(), ["prepare_inference_spec.py"])

    def test_deft_audit_failure_is_not_completion(self):
        self.terminal_log()
        self.env["TEST_AUDIT_EXIT"] = "1"
        self.assertEqual(self.run_driver().returncode, 2)
        self.assertNotIn(" - DONE", (self.run_home / "driver.log").read_text())

    def test_round_caps_fail_for_both_packs(self):
        for pack, cap in [("deft", 80), ("automl", 40)]:
            with self.subTest(pack=pack):
                self.calls.write_text("")
                result = self.run_driver(pack)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(self.calls.read_text().count("session\n"), cap)

    def test_automl_error_takes_precedence_over_done(self):
        (self.rd / "progress.log").write_text("done ok\nrunner_finished FAIL\n")
        (self.rd / "AUTOML_DONE.marker").touch()
        self.assertEqual(self.run_driver("automl").returncode, 2)
        self.assertFalse(self.calls.exists())

    def test_completion_on_last_round_is_not_misreported_as_cap_failure(self):
        for pack, cap in [("deft", 80), ("automl", 40)]:
            with self.subTest(pack=pack):
                self.env["TEST_CAP"] = str(cap)
                self.calls.write_text("")
                (self.rd / "loop_log.jsonl").write_text("")
                with self.shell.open("a") as handle:
                    handle.write('''
timeout() {
  echo session >> "$TEST_CALLS"
  if [ "$TEST_PACK" = deft ]; then
    if [ "$round" -eq "$TEST_CAP" ]; then stage=loop_stop; else stage=evaluate; fi
    printf '{"status":"ok","stage":"%s","iter":"baseline"}\\n' "$stage" >> "$TEST_RD/loop_log.jsonl"
  else
    echo 'preflight ok' >> "$TEST_RD/progress.log"
    if [ "$round" -eq "$TEST_CAP" ]; then touch "$TEST_RD/AUTOML_DONE.marker"; fi
  fi
}
''')
                result = self.run_driver(pack)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.calls.read_text().count("session\n"), cap)

    def test_automl_default_image_uses_selected_bank(self):
        bank = self.root / "selected-bank"
        (bank / "scripts").mkdir(parents=True)
        (bank / "versions.yaml").write_text("images:\n  tao_toolkit:\n    pyt: fixture:current\n")
        (bank / "scripts/resolve_versions_key.py").symlink_to(ROOT / "scripts/resolve_versions_key.py")
        self.env["SB"] = str(bank)
        # Capture the real resolved value before any fake session.
        with self.shell.open("a") as handle:
            handle.write('\ntimeout() { echo "$TRAIN_IMG" >> "$TEST_CALLS"; echo "done ok" >> "$TEST_RD/progress.log"; touch "$TEST_RD/AUTOML_DONE.marker"; }\n')
        self.assertEqual(self.run_driver("automl").returncode, 0)
        self.assertEqual(self.calls.read_text().strip(), "fixture:current")

    def test_automl_explicit_image_override_is_preserved(self):
        self.env["TRAIN_IMG"] = "fixture:override"
        with self.shell.open("a") as handle:
            handle.write('\ntimeout() { echo "$TRAIN_IMG" >> "$TEST_CALLS"; touch "$TEST_RD/AUTOML_DONE.marker"; }\n')
        self.assertEqual(self.run_driver("automl").returncode, 0)
        self.assertEqual(self.calls.read_text().strip(), "fixture:override")

    def test_generic_template_halts_on_failure(self):
        cards = self.root / "cards"
        cards.mkdir()
        (cards / "00-first-stage.md").write_text("fixture")
        (self.rd / "progress.log").write_text("first_stage FAIL\n")
        # Generic template's find does not request timestamp output.
        with self.shell.open("a") as handle:
            handle.write('\nfind() { echo "$TEST_RD"; }\n')
        self.env["KIT"] = str(self.root)
        result = subprocess.run(
            ["bash", str(KIT / "templates/driver.template.sh")],
            env=self.env, capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 2, result.stderr)


if __name__ == "__main__":
    unittest.main()
