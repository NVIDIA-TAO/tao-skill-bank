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
[ -n "${TEST_NO_PI:-}" ] || pi() { :; }
sleep() { :; }
working() { return 1; }
docker() { return 0; }
pgrep() { return 1; }
find() {
  case "$*" in
    *"-maxdepth 1"*) printf '1 %s\\n' "$TEST_RD" ;;
  esac
}
add_event() {
  jq '.events = ((.events // []) + [{"seq": ((.events // []) | length + 1), "status": "ok"}])' \\
    "$TEST_RD/deft_state.json" > "$TEST_RD/state.tmp" && mv "$TEST_RD/state.tmp" "$TEST_RD/deft_state.json"
}
timeout() {
  echo session >> "$TEST_CALLS"
  case "$TEST_PACK" in
    deft) add_event ;;
    *) echo 'preflight ok' >> "$TEST_RD/progress.log" ;;
  esac
}
""")
        scripts = self.root / "skill/scripts"
        scripts.mkdir(parents=True)
        # Real deft_context.py decides routing; finalize_run.py is simulated.
        wrapper = scripts / "deft_python.sh"
        wrapper.write_text(f"""#!/bin/bash
case "$1" in
  *deft_context.py) shift; exec {sys.executable} {str(APPS / "tao-run-deft-aoi/scripts/deft_context.py")!r} "$@" ;;
  *finalize_run.py)
    echo "finalize_run.py $*" | sed 's|[^ ]*/finalize_run.py ||' >> "$TEST_CALLS"
    [ "$TEST_FINALIZE_EXIT" = 0 ] || exit "$TEST_FINALIZE_EXIT"
    [ -n "${{TEST_FINALIZE_NOOP:-}}" ] && exit 0
    jq '.status = "complete"' "$TEST_RD/deft_state.json" > "$TEST_RD/state.tmp" && mv "$TEST_RD/state.tmp" "$TEST_RD/deft_state.json"
    exit 0 ;;
esac
echo "$(basename "$1")" >> "$TEST_CALLS"
exit 99
""")
        wrapper.chmod(0o755)
        (self.rd / "progress.log").touch()
        self.write_state({"baseline": {"status": "in_progress", "stage_completed": "train"}})
        backbone = self.root / "ws/augmentation/backbone/c_radio_v2_b.safetensors"
        backbone.parent.mkdir(parents=True)
        backbone.write_bytes(b"fixture")
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
            "GPU_MODEL": "fixture GPU",
            "TEST_RD": str(self.rd),
            "TEST_CALLS": str(self.calls),
            "TEST_FINALIZE_EXIT": "0",
        }

    def run_driver(self, pack="deft"):
        self.env["TEST_PACK"] = pack
        name = "tao-run-deft-aoi" if pack == "deft" else "tao-run-automl"
        self.env["CARDS"] = str(APPS / name / "cards")
        return subprocess.run(
            ["bash", str(APPS / name / "cards/driver.sh")],
            env=self.env, capture_output=True, text=True, timeout=20,
        )

    def write_state(self, iterations, status="in_progress", current=0, events=()):
        (self.rd / "deft_state.json").write_text(json.dumps({
            "status": status, "current_iteration": current, "max_iterations": 3,
            "iterations": iterations, "events": list(events),
        }))

    def evaluated(self, passed):
        return {"status": "complete", "stage_completed": "evaluate", "metric_result": {"passed": passed}}

    def driver_log(self):
        return (self.run_home / "driver.log").read_text()

    def test_deft_committed_error_halts_without_a_session(self):
        self.write_state({"iter2": {"status": "failed", "stage_completed": "routing"}}, status="failed",
                         current=2, events=[{"seq": 1, "iter": "iter2", "stage": "data_mining",
                                             "status": "error", "summary": "leakage"}])
        result = self.run_driver()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("committed error at iter2/data_mining: leakage", self.driver_log())
        self.assertFalse(self.calls.exists())

    def test_deft_met_metric_is_finalized_by_the_driver(self):
        self.write_state({"baseline": self.evaluated(True)})
        result = self.run_driver()
        self.assertEqual(result.returncode, 0, result.stderr)
        call = self.calls.read_text().splitlines()
        self.assertEqual(len(call), 1)
        self.assertIn("--iter-label baseline --stop-reason metric_met", call[0])
        self.assertIn("status=complete", self.driver_log())

    def test_deft_last_iteration_is_finalized_as_max_iterations(self):
        self.write_state({"baseline": self.evaluated(False), "iter3": self.evaluated(False)}, current=3)
        self.assertEqual(self.run_driver().returncode, 0)
        self.assertIn("--iter-label iter3 --stop-reason max_iterations", self.calls.read_text())

    def test_deft_finalize_failure_is_not_completion(self):
        self.write_state({"baseline": self.evaluated(True)})
        self.env["TEST_FINALIZE_EXIT"] = "2"
        self.assertEqual(self.run_driver().returncode, 2)
        self.assertNotIn("DONE", self.driver_log())

    def test_deft_done_requires_complete_status_not_finalize_exit_code(self):
        self.write_state({"baseline": self.evaluated(True)})
        self.env["TEST_FINALIZE_NOOP"] = "1"
        self.assertEqual(self.run_driver().returncode, 2)
        self.assertIn("status is not complete", self.driver_log())

    def test_deft_already_complete_run_is_done_without_finalizing_again(self):
        self.write_state({"baseline": self.evaluated(True)}, status="complete")
        self.assertEqual(self.run_driver().returncode, 0)
        self.assertFalse(self.calls.exists())

    def test_deft_routes_on_deft_context_next_stage(self):
        for iterations, current, card in [
            ({"baseline": self.evaluated(False)}, 0, "30-post-evaluate.md (baseline)"),
            ({"baseline": {"stage_completed": "rca"}}, 0, "40-routing.md (iter1)"),
            ({"iter1": {"stage_completed": "routing"}}, 1, "40-routing.md (iter1)"),
            ({"iter1": {"stage_completed": "anomalygen"}}, 1, "50-mining.md (iter1)"),
            ({"iter1": {"stage_completed": "data_mining"}}, 1, "60-merge-train.md (iter1)"),
            ({"iter1": {"stage_completed": "data_merge"}}, 1, "10-post-train.md (iter1)"),
            ({"baseline": {"stage_completed": "train"}}, 0, "20-evaluate.md (baseline)"),
        ]:
            with self.subTest(card=card):
                self.write_state(iterations, current=current)
                (self.run_home / "driver.log").unlink(missing_ok=True)
                self.run_driver()
                self.assertIn(f"-> {card}", self.driver_log())

    def test_deft_baseline_without_train_log_reenters_init_card(self):
        self.write_state({})
        self.run_driver()
        self.assertIn("-> 00-init-baseline-train.md (baseline)", self.driver_log())

    def test_missing_pi_aborts_before_side_effects(self):
        self.env["TEST_NO_PI"] = "1"
        self.env["PATH"] = path_without("pi")
        self.env["HOME"] = str(self.root / "home")
        for pack in ["deft", "automl"]:
            with self.subTest(pack=pack):
                result = self.run_driver(pack)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("pi not on PATH", result.stderr)
                self.assertFalse((self.run_home / ".launch_marker").exists())
                self.assertFalse((self.run_home / "driver.log").exists())
                self.assertFalse(self.calls.exists())

    def test_missing_backbone_aborts_before_side_effects(self):
        (self.root / "ws/augmentation/backbone/c_radio_v2_b.safetensors").unlink()
        self.env["TRAIN_IMG"] = "fixture:image"
        for pack in ["deft", "automl"]:
            with self.subTest(pack=pack):
                result = self.run_driver(pack)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("backbone not staged", result.stderr)
                self.assertIn("stage_backbone.py", result.stderr)
                self.assertFalse((self.run_home / ".launch_marker").exists())
                self.assertFalse(self.calls.exists())

    def test_legacy_backbone_name_is_accepted(self):
        backbone_dir = self.root / "ws/augmentation/backbone"
        (backbone_dir / "c_radio_v2_b.safetensors").rename(backbone_dir / "model.safetensors")
        self.env["TRAIN_IMG"] = "fixture:image"
        with self.shell.open("a") as handle:
            handle.write('\ntimeout() { echo "$BACKBONE" >> "$TEST_CALLS"; touch "$TEST_RD/AUTOML_DONE.marker"; }\n')
        self.assertEqual(self.run_driver("automl").returncode, 0)
        self.assertEqual(self.calls.read_text().strip(), str(backbone_dir / "model.safetensors"))

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
                self.write_state({"baseline": {"status": "in_progress", "stage_completed": "train"}})
                with self.shell.open("a") as handle:
                    handle.write('''
timeout() {
  echo session >> "$TEST_CALLS"
  if [ "$TEST_PACK" = deft ]; then
    add_event
    if [ "$round" -eq "$TEST_CAP" ]; then
      jq '.iterations.baseline = {"status": "complete", "stage_completed": "evaluate", "metric_result": {"passed": true}}' \\
        "$TEST_RD/deft_state.json" > "$TEST_RD/state.tmp" && mv "$TEST_RD/state.tmp" "$TEST_RD/deft_state.json"
    fi
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

    def test_missing_provider_key_aborts_before_any_session(self):
        for pack in ("deft", "automl"):
            with self.subTest(pack=pack):
                self.env["MODEL"] = "nvidia/qwen/qwen3.6-35b-a3b"
                result = self.run_driver(pack)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("export NVIDIA_API_KEY", result.stderr)
                self.assertFalse(self.calls.exists())

    def test_unknown_provider_skips_key_preflight_explicitly(self):
        with self.shell.open("a") as handle:
            handle.write('\ntimeout() { touch "$TEST_RD/AUTOML_DONE.marker"; }\n')
        result = self.run_driver("automl")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("key preflight skipped for provider 'test'", result.stderr)

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


def path_without(*names):
    """PATH with every directory that provides one of ``names`` removed."""
    return os.pathsep.join(
        d for d in os.environ["PATH"].split(os.pathsep)
        if d and not any(os.access(os.path.join(d, n), os.X_OK) for n in names)
    )


def preflight(model, **env):
    """Run model_preflight.sh for ``model``; return (exit code, stderr, exported nim models)."""
    script = (f'. "{KIT}/scripts/model_preflight.sh"; MODEL="$1"; kit_prepare_model_env; '
              'kit_check_model_key "[t]"; rc=$?; printenv PI_KIT_NIM_MODELS; exit $rc')
    result = subprocess.run(["bash", "-c", script, "_", model],
                            env={"PATH": os.environ["PATH"], **env},
                            capture_output=True, text=True, timeout=20)
    return result.returncode, result.stderr, result.stdout.strip()


class ModelPreflightTests(unittest.TestCase):
    def test_provider_key_vars(self):
        cases = {
            "nim/nvidia/qwen/qwen3.6-35b-a3b:off": "NVIDIA_INFERENCE_API_KEY",
            "nvidia/qwen/qwen3.6-35b-a3b": "NVIDIA_API_KEY",
            "anthropic/claude-fable-5": "ANTHROPIC_API_KEY",
            "openai/gpt-5": "OPENAI_API_KEY",
        }
        for model, var in cases.items():
            with self.subTest(model=model):
                rc, err, _ = preflight(model)
                self.assertEqual(rc, 1)
                self.assertIn(f"export {var} ", err)
                self.assertEqual(preflight(model, **{var: "x"})[0], 0)

    def test_nim_key_var_is_configurable(self):
        rc, err, _ = preflight("nim/m:off", PI_KIT_NIM_API_KEY_VAR="MY_GATEWAY_KEY")
        self.assertEqual(rc, 1)
        self.assertIn("export MY_GATEWAY_KEY ", err)
        self.assertEqual(preflight("nim/m:off", PI_KIT_NIM_API_KEY_VAR="MY_GATEWAY_KEY", MY_GATEWAY_KEY="x")[0], 0)

    def test_nim_model_id_is_registered_from_model(self):
        _, _, models = preflight("nim/meta/llama-4-70b:off", NVIDIA_INFERENCE_API_KEY="x", PI_KIT_NIM_MODELS="a/b")
        self.assertEqual(models, "a/b,meta/llama-4-70b")

    def test_unknown_provider_and_opt_out_skip_explicitly(self):
        rc, err, _ = preflight("vllm/my-model")
        self.assertEqual(rc, 0)
        self.assertIn("key preflight skipped for provider 'vllm'", err)
        rc, err, _ = preflight("nvidia/x", PI_KIT_SKIP_KEY_PREFLIGHT="1")
        self.assertEqual(rc, 0)
        self.assertIn("skipped for NVIDIA_API_KEY", err)

    def test_malformed_model_ref_aborts(self):
        rc, err, _ = preflight("qwen3")
        self.assertEqual(rc, 1)
        self.assertIn("MODEL must be <provider>/<model-id>", err)


def node_strips_types():
    try:
        out = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return out.startswith("v") and int(out[1:].split(".")[0]) >= 23


@unittest.skipUnless(node_strips_types(), "needs node >= 23 (native TypeScript type stripping)")
class NimProviderTests(unittest.TestCase):
    def load(self, **env):
        script = f"""
const mod = await import({json.dumps((KIT / "adapters/pi/nvidia-provider.ts").as_uri())});
const out = {{}}; let hook;
mod.default({{
  on: (name, fn) => {{ if (name === "before_provider_request") hook = fn; }},
  registerProvider: (name, cfg) => {{ out.name = name; out.cfg = cfg; }},
}});
const payload = {{ model: "x", messages: [] }};
out.nim = hook({{ payload }}, {{ model: {{ provider: "nim" }} }});
out.other = hook({{ payload }}, {{ model: {{ provider: "nvidia" }} }});
console.log(JSON.stringify(out));
"""
        result = subprocess.run(["node", "--input-type=module", "-e", script],
                                env={"PATH": os.environ["PATH"], **env},
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_defaults(self):
        out = self.load()
        cfg = out["cfg"]
        self.assertEqual(out["name"], "nim")
        self.assertEqual(cfg["baseUrl"], "https://inference-api.nvidia.com/v1")
        self.assertEqual(cfg["apiKey"], "$NVIDIA_INFERENCE_API_KEY")
        self.assertEqual(len(cfg["models"]), 4)
        self.assertTrue(all(m["compat"] == cfg["models"][0]["compat"] for m in cfg["models"]))
        self.assertEqual(out["nim"]["temperature"], 0)
        self.assertIsNone(out.get("other"))

    def test_environment_overrides(self):
        cfg = self.load(PI_KIT_NIM_BASE_URL="http://localhost:8000/v1/", PI_KIT_NIM_API_KEY_VAR="MY_KEY",
                        PI_KIT_NIM_MODELS="meta/llama-4-70b, nvidia/qwen/qwen3.6-35b-a3b",
                        PI_KIT_NIM_MAX_TOKENS="4096")["cfg"]
        self.assertEqual(cfg["baseUrl"], "http://localhost:8000/v1")
        self.assertEqual(cfg["apiKey"], "$MY_KEY")
        ids = [m["id"] for m in cfg["models"]]
        self.assertEqual(len(ids), 5)
        self.assertEqual(ids[-1], "meta/llama-4-70b")
        self.assertTrue(all(m["maxTokens"] == 4096 for m in cfg["models"]))

    def test_legacy_base_url_alias_and_invalid_key_var(self):
        cfg = self.load(NVIDIA_INFERENCE_BASE_URL="https://gw.example/v1",
                        PI_KIT_NIM_API_KEY_VAR="not a var")["cfg"]
        self.assertEqual(cfg["baseUrl"], "https://gw.example/v1")
        self.assertEqual(cfg["apiKey"], "$NVIDIA_INFERENCE_API_KEY")


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for tool in ["jq", "python3", "docker", "claude"]:
            stub = self.bin / tool
            stub.write_text("#!/bin/sh\nexit 0\n")
            stub.chmod(0o755)

    def run_install(self):
        env = {
            "PATH": os.pathsep.join([str(self.bin), path_without("pi")]),
            "HOME": str(self.root / "home"),
            "KIT_HOME": str(self.root / "kit"),
        }
        return subprocess.run(
            ["bash", str(KIT / "scripts/install.sh")],
            env=env, capture_output=True, text=True, timeout=20,
        )

    def test_claude_without_pi_is_not_ready(self):
        result = self.run_install()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("MISSING: pi", result.stdout)
        self.assertNotIn("RESULT: ready", result.stdout)

    def test_pi_is_ready(self):
        stub = self.bin / "pi"
        stub.write_text("#!/bin/sh\necho 0.85.1\n")
        stub.chmod(0o755)
        result = self.run_install()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("RESULT: ready", result.stdout)


if __name__ == "__main__":
    unittest.main()
