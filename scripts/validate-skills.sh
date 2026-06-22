#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Validate the skill bank.
#
# Required:
#   1. Every skill path in .claude-plugin/marketplace.json resolves to a dir with SKILL.md.
#   2. The Codex-facing skills/ directory has no symlink mirror of canonical skills.
#   3. Every SKILL.md has valid YAML frontmatter with `name` and `description`.
#   4. Each SKILL.md body contains enough info to run the skill (heuristic: a Quick Start
#      section, a docker run code block, OR a references/skill_info.yaml link).
#   5. No SDK symbols leak into model/data/application SKILL.md (platform/* exempt).
#   6. Hook paths in skill frontmatter resolve to existing scripts.
#   7. AutoML guidance keeps the automatic post-preflight baseline eval gate.
#
# Optional (validated only if the file exists):
#   8. Any skill_info.yaml parses, including deploy/skill_info.yaml files.
#   9. Container image keys resolve through versions.yaml, including action-level overrides.
#  10. Model/data action contracts declare command, mode, inputs, outputs, and upload_excludes.
#  11. references/model_info.yaml (legacy name) parses if present — same rules.
#
# Exit status = number of errors found.
#
# Usage:
#   ./scripts/validate-skills.sh                  # full validation
#   ./scripts/validate-skills.sh --quick          # skip optional structured-metadata checks

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MARKETPLACE=".claude-plugin/marketplace.json"
errors=0

ok()  { echo "  OK: $*"; }

# ─── 1. marketplace paths ───────────────────────────────────────────────────
echo "=== 1. marketplace.json skill paths ==="
[ -f "$MARKETPLACE" ] || { echo "ERROR: $MARKETPLACE not found"; exit 1; }

python3 - <<'PY'
import json, os, sys
with open('.claude-plugin/marketplace.json') as f:
    mp = json.load(f)
errs = 0
for plugin in mp.get('plugins', []):
    for path in plugin.get('skills', []):
        real = path.lstrip('./')
        skill_md = os.path.join(real, 'SKILL.md')
        if not os.path.isfile(skill_md):
            print(f"ERROR: {plugin['name']} → {path} (no SKILL.md at {skill_md})", file=sys.stderr)
            errs += 1
sys.exit(errs)
PY
[ $? -eq 0 ] && ok "all marketplace paths resolve" || errors=$((errors + $?))

# ─── 1b. Codex skills/ should not mirror canonical skills ──────────────────
echo
echo "=== 1b. Codex skills/ has no mirror symlinks ==="
codex_skill_symlinks="$(find skills -mindepth 1 -maxdepth 1 -type l -print | sort || true)"
if [ -n "$codex_skill_symlinks" ]; then
  codex_skill_symlink_errors=0
  while IFS= read -r path; do
    [ -z "$path" ] && continue
    echo "ERROR: $path — do not mirror skills via symlinks under skills/. Real skills live under skills/{applications,data,models,platform,core}/." >&2
    codex_skill_symlink_errors=$((codex_skill_symlink_errors + 1))
  done <<< "$codex_skill_symlinks"
  errors=$((errors + codex_skill_symlink_errors))
else
  ok "skills/ contains only Codex helper skill directories"
fi

# ─── 2. SKILL.md frontmatter (errors) + DAFT-style optional fields (warnings) ─
echo
echo "=== 2. SKILL.md frontmatter ==="
python3 - <<'PY'
import os, sys, yaml, re
errs = 0
warns = 0

def iter_skill_files():
    for root, dirs, files in os.walk('.', followlinks=False):
        dirs[:] = [
            d for d in dirs
            if d not in ('.git', 'plugins')
            and 'templates/skill-skeleton' not in os.path.join(root, d)
            and not os.path.islink(os.path.join(root, d))
        ]
        if 'SKILL.md' in files:
            yield os.path.join(root, 'SKILL.md').lstrip('./')

for skill_md in iter_skill_files():
    with open(skill_md) as f:
        content = f.read()
    m = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
    if not m:
        print(f"ERROR: {skill_md} — missing or malformed frontmatter", file=sys.stderr); errs += 1; continue
    try:
        fm = yaml.safe_load(m.group(1))
    except yaml.YAMLError as e:
        print(f"ERROR: {skill_md} — YAML parse error: {e}", file=sys.stderr); errs += 1; continue
    if not isinstance(fm, dict):
        print(f"ERROR: {skill_md} — frontmatter is not a mapping", file=sys.stderr); errs += 1; continue
    # Required fields
    if 'name' not in fm:
        print(f"ERROR: {skill_md} — missing `name`", file=sys.stderr); errs += 1
    if 'license' not in fm:
        print(f"ERROR: {skill_md} — missing `license`. Add `license: Apache-2.0` (see docs/authoring.md).", file=sys.stderr); errs += 1
    # Optional fields — warn but don't fail
    if 'compatibility' not in fm:
        print(f"WARN: {skill_md} — missing `compatibility:` (runtime requirements). See docs/authoring.md for examples.", file=sys.stderr); warns += 1
    if not isinstance(fm.get('metadata'), dict) or 'author' not in fm.get('metadata', {}):
        print(f"WARN: {skill_md} — missing `metadata.author`. Add `author: NVIDIA Corporation`.", file=sys.stderr); warns += 1
    elif fm['metadata'].get('author') != 'NVIDIA Corporation':
        print(f"ERROR: {skill_md} — `metadata.author` must be exactly 'NVIDIA Corporation' (found: {fm['metadata'].get('author')!r}).", file=sys.stderr); errs += 1
    if not isinstance(fm.get('metadata'), dict) or 'version' not in fm.get('metadata', {}):
        print(f"WARN: {skill_md} — missing `metadata.version`. Add e.g. `version: \"0.1\"`.", file=sys.stderr); warns += 1
    if 'allowed-tools' not in fm:
        print(f"WARN: {skill_md} — missing `allowed-tools`. Set if the skill uses Read/Bash/Write frequently.", file=sys.stderr); warns += 1
    if 'description' not in fm:
        print(f"ERROR: {skill_md} — missing `description`", file=sys.stderr); errs += 1
if warns > 0:
    print(f"  ({warns} warning(s) — see docs/authoring.md to address)", file=sys.stderr)
sys.exit(errs)
PY
[ $? -eq 0 ] && ok "all SKILL.md frontmatter valid" || errors=$((errors + $?))

# ─── 3. SKILL.md body has runnable info ─────────────────────────────────────
echo
echo "=== 3. SKILL.md body has runnable info ==="
python3 - <<'PY'
import os, sys, re
# A SKILL.md is "runnable" if any of:
#   - body has a "## Quick Start" or "## Quick start" heading
#   - body has a `docker run` code block
#   - body has a Python `sdk.create_job` call (for SDK-driven skills)
#   - the skill dir has references/skill_info.yaml or references/model_info.yaml on disk
# Skips templates/.
errs = 0

def iter_skill_files():
    for root, dirs, files in os.walk('.', followlinks=False):
        dirs[:] = [
            d for d in dirs
            if d not in ('.git', 'plugins')
            and 'templates/skill-skeleton' not in os.path.join(root, d)
            and not os.path.islink(os.path.join(root, d))
        ]
        if 'SKILL.md' in files:
            yield os.path.join(root, 'SKILL.md').lstrip('./')

for skill_md in iter_skill_files():
    skill_dir = os.path.dirname(skill_md)
    with open(skill_md) as f:
        content = f.read()
    has_qs = re.search(r'^##\s+quick ?start', content, re.IGNORECASE | re.MULTILINE)
    has_dr = 'docker run' in content
    has_sdk = re.search(r'sdk\.create_job|BrevSDK', content)
    has_refs = (os.path.isfile(os.path.join(skill_dir, 'references/skill_info.yaml'))
                or os.path.isfile(os.path.join(skill_dir, 'references/model_info.yaml')))
    # Local-Python or agent-prompt-driven skills: presence of scripts/ or hooks/ counts as runnable.
    has_scripts = os.path.isdir(os.path.join(skill_dir, 'scripts'))
    has_hooks = os.path.isdir(os.path.join(skill_dir, 'hooks'))
    if not (has_qs or has_dr or has_sdk or has_refs or has_scripts or has_hooks):
        print(f"ERROR: {skill_md} — no runnable info found. Add a Quick Start, docker run block, SDK call, references/skill_info.yaml, scripts/, or hooks/.", file=sys.stderr)
        errs += 1
sys.exit(errs)
PY
[ $? -eq 0 ] && ok "all SKILL.md bodies have runnable info" || errors=$((errors + $?))

# ─── 4. no SDK leaks in model/data/application skills ───────────────────────
echo
echo "=== 4. no SDK leaks in model/data/application skills ==="
python3 - <<'PY'
import re, os, sys
leak_re = re.compile(r'tao_sdk|TaoExecutionSDK|sdk\.create_job|sdk\.list_path|sdk\.check_path|execute_step|agent_runner|script_runner')
errs = 0
for root, dirs, files in os.walk('.'):
    if any(x in root for x in ('.git', 'templates/skill-skeleton')):
        continue
    if 'SKILL.md' in files:
        path = os.path.join(root, 'SKILL.md')
        # Platform skills legitimately document the SDK
        if path.startswith('./skills/platform/'):
            continue
        # Application skills that are SDK-orchestrated (AutoML, etc.) are exempt.
        # Add new ones here only after confirming they cannot run without the SDK.
        if path in ('./skills/applications/tao-run-automl/SKILL.md',):
            continue
        # Models may have an "Optional: running via the TAO SDK" section
        is_model = path.startswith('./skills/models/')
        with open(path) as f:
            content = f.read()
        matches = leak_re.findall(content)
        if not matches:
            continue
        if is_model:
            opt = re.search(r'##\s*Optional:.*?(?=\n##\s|\Z)', content, re.DOTALL | re.IGNORECASE)
            if opt:
                outside = leak_re.findall(content.replace(opt.group(0), ''))
                if outside:
                    print(f"ERROR: {path} — SDK symbols outside Optional SDK section: {outside[:3]}", file=sys.stderr); errs += 1
                continue
            print(f"ERROR: {path} — SDK symbols found: {matches[:3]}. Wrap in an 'Optional: running via the TAO SDK' section or remove.", file=sys.stderr); errs += 1
        else:
            print(f"ERROR: {path} — SDK symbols in non-model skill: {matches[:3]}", file=sys.stderr); errs += 1
sys.exit(errs)
PY
[ $? -eq 0 ] && ok "no SDK symbol leaks" || errors=$((errors + $?))

# ─── 5. hook paths resolve ──────────────────────────────────────────────────
echo
echo "=== 5. hook paths resolve ==="
python3 - <<'PY'
import re, os, sys, yaml
errs = 0
for root, dirs, files in os.walk('.'):
    if any(x in root for x in ('.git', 'templates/skill-skeleton')):
        continue
    if 'SKILL.md' not in files: continue
    path = os.path.join(root, 'SKILL.md')
    with open(path) as f:
        content = f.read()
    m = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
    if not m: continue
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError: continue
    hooks = fm.get('hooks') or {}
    if not isinstance(hooks, dict): continue
    for event, entries in hooks.items():
        for entry in (entries or []):
            for hook in (entry.get('hooks') or []):
                cmd = hook.get('command', '')
                for m2 in re.finditer(r'\$\{CLAUDE_SKILL_DIR\}/([^\s"\']+)', cmd):
                    rel = m2.group(1)
                    full = os.path.join(root, rel)
                    if not os.path.exists(full):
                        print(f"ERROR: {path} — hook references missing file: {rel}", file=sys.stderr); errs += 1
sys.exit(errs)
PY
[ $? -eq 0 ] && ok "all hook paths resolve" || errors=$((errors + $?))

# ─── 6. AutoML baseline eval guardrail ───────────────────────────────────────
echo
echo "=== 6. AutoML baseline eval guardrail ==="
python3 - <<'PY'
from pathlib import Path
import sys

required = {
    "skills/applications/tao-run-automl/SKILL.md": [
        "## Automatic Baseline Eval Job",
        "post-preflight eval job",
        "eval metric number",
    ],
    "skills/applications/tao-run-automl/references/automl-intent-algorithms.md": [
        "automatic baseline eval job",
        "job id, result path, and metric value",
    ],
    "skills/models/tao-finetune-cosmos-reason/SKILL.md": [
        "run the model's evaluate",
        "action once after preflight",
        "Report that eval job id, result path, and accuracy",
    ],
}
stale_phrases = (
    "baseline/pretrained evaluation",
    "pretrained evaluation before AutoML",
    "baseline-eval plan",
    "unless the user explicitly declines it",
)

errs = 0
for rel, needles in required.items():
    text = Path(rel).read_text(encoding="utf-8")
    for needle in needles:
        if needle not in text:
            print(f"ERROR: {rel} — missing AutoML baseline eval guardrail text: {needle}", file=sys.stderr)
            errs += 1
    for phrase in stale_phrases:
        if phrase in text:
            print(f"ERROR: {rel} — stale optional baseline wording remains: {phrase}", file=sys.stderr)
            errs += 1
sys.exit(errs)
PY
[ $? -eq 0 ] && ok "AutoML baseline eval guidance is guarded" || errors=$((errors + $?))

# ─── 7. optional structured metadata ────────────────────────────────────────
if [ "${1:-}" != "--quick" ]; then
  echo
  echo "=== 7. skill_info.yaml + legacy model_info.yaml (when present) ==="
  python3 - <<'PY'
import os, sys, yaml
errs = 0
VALID_MODES = {'config', 'args', 'passthrough'}
VALID_CONFIG_FORMATS = {'yaml', 'toml', 'json'}

try:
    with open('versions.yaml') as vf:
        manifest = yaml.safe_load(vf) or {}
except FileNotFoundError:
    manifest = {}

def iter_metadata_files():
    for root, dirs, files in os.walk('.'):
        dirs[:] = [
            d for d in dirs
            if d not in ('.git', 'templates', '.claude-plugin', '.codex-plugin')
        ]
        for fname in ('skill_info.yaml', 'model_info.yaml'):
            if fname in files:
                yield os.path.join(root, fname)

def skill_dir_for(path):
    parts = path.split(os.sep)
    if 'references' in parts:
        idx = parts.index('references')
        return os.sep.join(parts[:idx])
    if len(parts) >= 3 and parts[-2] == 'deploy':
        return os.sep.join(parts[:-2])
    return os.path.dirname(path)

def validate_image(path, img, context):
    global errs
    if not isinstance(img, str):
        print(f"ERROR: {path} — {context} must be a string", file=sys.stderr); errs += 1
        return
    # Absolute path heuristic: contains '/' or ':' (registry URI shape).
    if '/' in img or ':' in img:
        return
    if not manifest:
        print(f"ERROR: {path} — {context} '{img}' looks like a key but versions.yaml is missing at repo root", file=sys.stderr); errs += 1
        return
    node = manifest.get('images', {})
    try:
        for part in img.split('.'):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"key '{img}' missing from versions.yaml images tree")
            node = node[part]
        if not isinstance(node, str):
            print(f"ERROR: {path} — {context} key '{img}' resolves to non-string in versions.yaml", file=sys.stderr); errs += 1
    except KeyError as e:
        print(f"ERROR: {path} — {context} key '{img}' not found in versions.yaml ({e})", file=sys.stderr); errs += 1

for path in iter_metadata_files():
    try:
        with open(path) as f:
            info = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"ERROR: {path} — YAML parse error: {e}", file=sys.stderr); errs += 1; continue
    if not isinstance(info, dict):
        print(f"ERROR: {path} — metadata file must contain a YAML mapping", file=sys.stderr); errs += 1; continue

    skill_dir = skill_dir_for(path)
    is_model_or_data = skill_dir.startswith('./skills/models/') or skill_dir.startswith('./skills/data/')

    if isinstance(info.get('container_image'), str):
        validate_image(path, info['container_image'], 'container_image')
    elif is_model_or_data and 'actions' in info:
        print(f"WARN: {path} — has actions but no top-level container_image", file=sys.stderr)

    actions = info.get('actions') or {}
    if actions and not isinstance(actions, dict):
        print(f"ERROR: {path} — actions must be a mapping", file=sys.stderr); errs += 1; continue

    for name, spec in actions.items():
        if not isinstance(spec, dict):
            print(f"ERROR: {path} — actions.{name} must be a mapping", file=sys.stderr); errs += 1; continue

        if isinstance(spec.get('container_image'), str):
            validate_image(path, spec['container_image'], f'actions.{name}.container_image')

        command = spec.get('command')
        if is_model_or_data and not command:
            print(f"ERROR: {path} — actions.{name} missing `command`", file=sys.stderr); errs += 1
            continue
        if not command:
            continue

        mode = spec.get('mode')
        if mode not in VALID_MODES:
            print(f"ERROR: {path} — actions.{name}.mode must be one of {sorted(VALID_MODES)}", file=sys.stderr); errs += 1
        if mode == 'config':
            config_format = spec.get('config_format')
            if config_format not in VALID_CONFIG_FORMATS:
                print(f"ERROR: {path} — actions.{name}.config_format must be one of {sorted(VALID_CONFIG_FORMATS)} when mode is config", file=sys.stderr); errs += 1
            if '{config_path}' not in str(command):
                print(f"ERROR: {path} — actions.{name}.command must include {{config_path}} when mode is config", file=sys.stderr); errs += 1
        if mode == 'args' and not isinstance(spec.get('args'), dict):
            print(f"ERROR: {path} — actions.{name}.args must be a mapping when mode is args", file=sys.stderr); errs += 1

        if is_model_or_data:
            for field in ('inputs', 'outputs', 'upload_excludes'):
                if field not in spec:
                    print(f"ERROR: {path} — actions.{name} missing `{field}`", file=sys.stderr); errs += 1
sys.exit(errs)
PY
  [ $? -eq 0 ] && ok "skill_info.yaml / model_info.yaml validation passed" || errors=$((errors + $?))
fi

echo
if [ $errors -eq 0 ]; then
  echo "✓ validate-skills passed"
  exit 0
else
  echo "✗ validate-skills failed: $errors error(s)"
  exit $errors
fi
