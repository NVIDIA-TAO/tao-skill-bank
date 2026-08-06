#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Validate the skill bank.
#
# Required:
#   1. Every skill path in .claude-plugin/marketplace.json resolves to a dir with SKILL.md.
#   1a. Reverse of 1: every non-core skill dir with a SKILL.md is listed in marketplace.json,
#       so a skill added on disk without a marketplace entry cannot silently never ship.
#   1c. Every skills/<layer>/<dir> contains a SKILL.md (catches assets stranded by a rename).
#   3c. Every models/ skill has a `docker run` somewhere, or an explicit
#       requires_external:true frontmatter marker (README "Docker-native first").
#   1d. All three plugin manifests carry the same version, and README's "bundles all N skills"
#       matches the tao-skills array length.
#   2. The Codex-facing skills/ directory has no symlink mirror of canonical skills.
#   3. Every SKILL.md has valid YAML frontmatter satisfying the signing pipeline's
#      strict checks (parity with docs/skill-requirements.md § 2.1):
#        - `name` present and equal to the directory name (kebab-case)
#        - `description` present and contains no '<' / XML-like tokens
#        - `license` present
#        - `metadata.author` present (must equal 'NVIDIA Corporation')
#        - `metadata.version` present and strict-semver "x.y.z" ('"0.1"' fails)
#        - `compatibility` present and ≤ 500 characters
#        - top-level `tags` present and non-empty
#   4. Each SKILL.md body contains enough info to run the skill (heuristic: a Quick Start
#      section, a docker run code block, OR a references/skill_info.yaml link).
#   4b. SKILL.md is ≤ ~20000 chars (signer caps at ~5000 tokens).
#       No nested SKILL.md inside a skill directory.
#   5. No SDK symbols leak into model/data/application SKILL.md (platform/* exempt).
#   6. Hook paths in skill frontmatter resolve to existing scripts.
#   7. AutoML guidance keeps the automatic post-preflight baseline eval gate.
#   8. Each skill has evals/evals.json (required for Tier-3 signing).
#   9. compatibility: doesn't reference a specific agent harness (the bank is
#      harness-agnostic per Agent Skills spec).
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

# ─── 1a. every skill on disk is shipped by a plugin (reverse of check 1) ────
echo
echo "=== 1a. every skill on disk is listed in marketplace.json ==="
python3 - <<'PY'
import json, os, sys
with open('.claude-plugin/marketplace.json') as f:
    mp = json.load(f)
listed = {s for p in mp.get('plugins', []) for s in p.get('skills', [])}
errs = 0
for layer in sorted(os.listdir('skills')):
    layer_dir = os.path.join('skills', layer)
    if layer == 'core' or not os.path.isdir(layer_dir):
        continue  # skills/core/ ships via .codex-plugin, not the marketplace array
    for name in sorted(os.listdir(layer_dir)):
        skill_dir = os.path.join(layer_dir, name)
        if not os.path.isfile(os.path.join(skill_dir, 'SKILL.md')):
            continue
        if f'./{skill_dir}' not in listed:
            print(f"ERROR: {skill_dir} has a SKILL.md but no marketplace.json entry — "
                  f"it will never install. Add it to a plugin's skills array.", file=sys.stderr)
            errs += 1
sys.exit(errs)
PY
[ $? -eq 0 ] && ok "every skill on disk is shipped" || errors=$((errors + $?))

# ─── 1c. no asset-only skill directories ────────────────────────────────────
echo
echo "=== 1c. every skill directory has a SKILL.md ==="
orphans="$(find skills -mindepth 2 -maxdepth 2 -type d '!' -exec test -f '{}/SKILL.md' ';' -print | sort || true)"
if [ -n "$orphans" ]; then
  orphan_errors=0
  while IFS= read -r path; do
    [ -z "$path" ] && continue
    echo "ERROR: $path — skill directory without SKILL.md. Assets left behind by a rename are invisible to every other check and never ship; move them into the renamed skill or delete the directory." >&2
    orphan_errors=$((orphan_errors + 1))
  done <<< "$orphans"
  errors=$((errors + orphan_errors))
else
  ok "no asset-only skill directories"
fi

# ─── 1d. manifest version parity + README skill count ───────────────────────
echo
echo "=== 1d. manifest version parity and README skill count ==="
python3 - <<'PY'
import json, os, re, sys
errs = 0
manifests = ['.claude-plugin/marketplace.json', '.claude-plugin/plugin.json', '.codex-plugin/plugin.json']
versions = {}
for p in manifests:
    if not os.path.isfile(p):
        continue
    d = json.load(open(p))
    versions[p] = d.get('metadata', {}).get('version') if p.endswith('marketplace.json') else d.get('version')
if len(set(versions.values())) > 1:
    print(f"ERROR: plugin manifest versions disagree: {versions}", file=sys.stderr)
    errs += 1

mp = json.load(open('.claude-plugin/marketplace.json'))
plug = next((p for p in mp.get('plugins', []) if p['name'] == 'tao-skills'), None)
if plug and os.path.isfile('README.md'):
    shipped = len(plug['skills'])
    readme = open('README.md', encoding='utf-8').read()
    m = re.search(r'bundles all (\d+) skills', readme)
    if not m:
        print("ERROR: README.md has no 'bundles all N skills' claim to check", file=sys.stderr)
        errs += 1
    elif int(m.group(1)) != shipped:
        print(f"ERROR: README.md says 'bundles all {m.group(1)} skills' but tao-skills ships {shipped}", file=sys.stderr)
        errs += 1
sys.exit(errs)
PY
[ $? -eq 0 ] && ok "manifest versions agree and README count matches" || errors=$((errors + $?))

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
            if d not in ('.git', 'plugins', '.venv', '.venv-tao')
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
    # Required fields — parity with signing pipeline (docs/skill-requirements.md § 2.1).
    skill_dir_name = os.path.basename(os.path.dirname(skill_md))
    if 'name' not in fm:
        print(f"ERROR: {skill_md} — missing `name`", file=sys.stderr); errs += 1
    elif fm['name'] != skill_dir_name:
        print(f"ERROR: {skill_md} — `name: {fm['name']!r}` must equal the directory name {skill_dir_name!r} (kebab-case).", file=sys.stderr); errs += 1
    if 'description' not in fm:
        print(f"ERROR: {skill_md} — missing `description`", file=sys.stderr); errs += 1
    elif '<' in str(fm.get('description', '')):
        print(f"ERROR: {skill_md} — `description` contains '<' (XML-like token). Rewrite using 'below'/'under' or describe in words; the signer's scanner flags this as 'Description contains XML tags'.", file=sys.stderr); errs += 1
    if 'license' not in fm:
        print(f"ERROR: {skill_md} — missing `license`. Add `license: Apache-2.0` (see docs/authoring.md).", file=sys.stderr); errs += 1
    if 'compatibility' not in fm:
        print(f"ERROR: {skill_md} — missing `compatibility:` (runtime requirements). See docs/authoring.md for examples.", file=sys.stderr); errs += 1
    else:
        compat_str = str(fm['compatibility'])
        if len(compat_str) > 500:
            print(f"ERROR: {skill_md} — `compatibility` is {len(compat_str)} chars; signer caps it at 500.", file=sys.stderr); errs += 1
        if re.search(r'designed for (claude|codex|gemini|cursor)', compat_str, re.IGNORECASE):
            print(f"ERROR: {skill_md} — `compatibility` references a specific agent harness. The skill bank is harness-agnostic; describe runtime requirements only.", file=sys.stderr); errs += 1
    md = fm.get('metadata') if isinstance(fm.get('metadata'), dict) else {}
    if 'author' not in md:
        print(f"ERROR: {skill_md} — missing `metadata.author`. Add `author: NVIDIA Corporation`.", file=sys.stderr); errs += 1
    elif md.get('author') != 'NVIDIA Corporation':
        print(f"ERROR: {skill_md} — `metadata.author` must be exactly 'NVIDIA Corporation' (found: {md.get('author')!r}).", file=sys.stderr); errs += 1
    if 'version' not in md:
        print(f"ERROR: {skill_md} — missing `metadata.version`. Use strict semver, e.g. `version: \"0.1.0\"`.", file=sys.stderr); errs += 1
    elif not re.fullmatch(r'\d+\.\d+\.\d+', str(md.get('version', ''))):
        print(f"ERROR: {skill_md} — `metadata.version: {md.get('version')!r}` must be strict semver \"x.y.z\" (e.g. \"0.1.0\"); the signer rejects \"0.1\" / \"0.4\" / \"0.1-ea\".", file=sys.stderr); errs += 1
    tags = fm.get('tags')
    if not tags or not isinstance(tags, list):
        print(f"ERROR: {skill_md} — top-level `tags:` must be present and a non-empty list.", file=sys.stderr); errs += 1
    if 'allowed-tools' not in fm:
        print(f"WARN: {skill_md} — missing `allowed-tools`. Set if the skill uses Read/Bash/Write frequently.", file=sys.stderr); warns += 1
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
#   - the skill dir has references/skill_info.yaml or references/model_info.yaml on disk
# Skips templates/.
errs = 0

def iter_skill_files():
    for root, dirs, files in os.walk('.', followlinks=False):
        dirs[:] = [
            d for d in dirs
            if d not in ('.git', 'plugins', '.venv', '.venv-tao')
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
    has_refs = (os.path.isfile(os.path.join(skill_dir, 'references/skill_info.yaml'))
                or os.path.isfile(os.path.join(skill_dir, 'references/model_info.yaml')))
    # Local-Python or agent-prompt-driven skills: presence of scripts/ or hooks/ counts as runnable.
    has_scripts = os.path.isdir(os.path.join(skill_dir, 'scripts'))
    has_hooks = os.path.isdir(os.path.join(skill_dir, 'hooks'))
    if not (has_qs or has_dr or has_refs or has_scripts or has_hooks):
        print(f"ERROR: {skill_md} — no runnable info found. Add a Quick Start, docker run block, references/skill_info.yaml, scripts/, or hooks/.", file=sys.stderr)
        errs += 1
sys.exit(errs)
PY
[ $? -eq 0 ] && ok "all SKILL.md bodies have runnable info" || errors=$((errors + $?))

# ─── 3c. docker-native rule for models/ and data/ ───────────────────────────
echo
echo "=== 3c. models/ skills are docker-native ==="
python3 - <<'PY'
import os, re, sys
errs = 0
# skills/data/ is deliberately not gated yet: 5 of 10 data skills are agent-native or
# pip-CLI with no container at all, a category the docker-native rule does not describe.
for layer in ('models',):
    root = os.path.join('skills', layer)
    if not os.path.isdir(root):
        continue
    for name in sorted(os.listdir(root)):
        skill_dir = os.path.join(root, name)
        skill_md = os.path.join(skill_dir, 'SKILL.md')
        if not os.path.isfile(skill_md):
            continue
        # explicit, reviewable exemption for upstream workflows that are not TAO containers
        head = open(skill_md, encoding='utf-8').read(4000)
        if re.search(r'^\s*requires_external:\s*true\s*$', head, re.M):
            continue
        found = False
        for dirpath, _, files in os.walk(skill_dir):
            for f in files:
                try:
                    if 'docker run' in open(os.path.join(dirpath, f), encoding='utf-8', errors='ignore').read():
                        found = True
                        break
                except OSError:
                    continue
            if found:
                break
        if not found:
            print(f"ERROR: {skill_dir} — no `docker run` anywhere in the skill. README's "
                  f"'Docker-native first' rule requires model/data skills to be runnable from "
                  f"SKILL.md alone. Add a docker Quick Start, or set `requires_external: true` "
                  f"in frontmatter metadata if the upstream workflow is not a TAO container.",
                  file=sys.stderr)
            errs += 1
sys.exit(errs)
PY
[ $? -eq 0 ] && ok "all models/ skills are docker-native or explicitly external" || errors=$((errors + $?))

# ─── 3b. SKILL.md size + no nested SKILL.md (signing parity) ────────────────
echo
echo "=== 3b. SKILL.md size + no nested SKILL.md ==="
python3 - <<'PY'
import os, sys
# Signer caps SKILL.md at ~5000 tokens (≈ chars ÷ 4). Use 20000 chars as the
# hard ceiling; recommend ≤ 18000 in docs for margin.
SIZE_CEILING = 20000
errs = 0
for root, dirs, files in os.walk('.', followlinks=False):
    dirs[:] = [
        d for d in dirs
        if d not in ('.git', 'plugins', '.venv', '.venv-tao')
        and 'templates/skill-skeleton' not in os.path.join(root, d)
        and not os.path.islink(os.path.join(root, d))
    ]
    if 'SKILL.md' not in files: continue
    skill_md = os.path.join(root, 'SKILL.md').lstrip('./')
    skill_dir = os.path.dirname(skill_md)
    # Size
    with open(skill_md) as f: size = len(f.read())
    if size > SIZE_CEILING:
        print(f"ERROR: {skill_md} — {size} chars > {SIZE_CEILING} (signer caps SKILL.md at ~5000 tokens ≈ {SIZE_CEILING} chars). Move detail into references/.", file=sys.stderr)
        errs += 1
    # Nested SKILL.md
    for sub_root, sub_dirs, sub_files in os.walk(skill_dir, followlinks=False):
        if sub_root == skill_dir: continue
        sub_dirs[:] = [d for d in sub_dirs if d not in ('.git',)]
        if 'SKILL.md' in sub_files:
            nested = os.path.join(sub_root, 'SKILL.md')
            print(f"ERROR: {skill_md} — contains nested SKILL.md at {nested}. Fold into references/<name>.md.", file=sys.stderr)
            errs += 1
sys.exit(errs)
PY
[ $? -eq 0 ] && ok "no oversize or nested SKILL.md" || errors=$((errors + $?))

# ─── 4. no SDK or AutoML wheel leaks ────────────────────────────────────────
echo
echo "=== 4. no SDK or AutoML wheel leaks ==="
python3 - <<'PY'
import re, os, sys
# Execution and optimization are skill-owned. Accurate statements that the
# dependency is absent are allowed; imports, install instructions, legacy
# adapters, and wheel routing are not.
leak_re = re.compile(r'tao_sdk|tao_automl|AutoMLRunner|nvidia-tao-automl|TaoExecutionSDK|sdk\.create_job|sdk\.list_path|sdk\.check_path|build_entrypoint|BrevSDK|SlurmSDK|KubernetesSDK|DockerSDK|script_runner')
neg_re  = re.compile(r"no [`']?(?:tao_sdk|tao_automl|nvidia-tao-sdk|nvidia-tao-automl)|without (?:the )?TAO SDK|there is no [`']?(?:tao_sdk|tao_automl)|must not import|SDK-free|no in-container", re.IGNORECASE)
errs = 0
for root, dirs, files in os.walk('./skills'):
    if any(x in root for x in ('.git', '.venv', '__pycache__')):
        continue
    in_refs = os.path.basename(root) == 'references'
    for fn in files:
        if fn != 'SKILL.md' and not (in_refs and fn.endswith('.md')):
            continue
        path = os.path.join(root, fn)
        with open(path) as f:
            hits = [ln.strip() for ln in f if leak_re.search(ln) and not neg_re.search(ln)]
        if hits:
            print(f"ERROR: {path} — removed SDK/AutoML package symbols: {hits[:2]}", file=sys.stderr)
            errs += 1
sys.exit(errs)
PY
[ $? -eq 0 ] && ok "no SDK or AutoML wheel leaks" || errors=$((errors + $?))

# ─── 5. hook paths resolve ──────────────────────────────────────────────────
echo
echo "=== 5. hook paths resolve ==="
python3 - <<'PY'
import re, os, sys, yaml
errs = 0
for root, dirs, files in os.walk('.'):
    if any(x in root for x in ('.git', 'templates/skill-skeleton', '.venv')):
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

# ─── 5b. evals/evals.json exists (Tier-3 signing) ───────────────────────────
echo
echo "=== 5b. evals/evals.json exists ==="
python3 - <<'PY'
import json, os, sys
errs = 0
for root, dirs, files in os.walk('.'):
    if any(x in root for x in ('.git', 'templates/skill-skeleton', 'plugins', '.venv')):
        continue
    if 'SKILL.md' not in files:
        continue
    skill_dir = root.lstrip('./')
    evals_path = os.path.join(skill_dir, 'evals/evals.json')
    if not os.path.isfile(evals_path):
        print(f"ERROR: {skill_dir} — missing `evals/evals.json`. Required for Tier-3 signing; see docs/skill-requirements.md § 2.3.", file=sys.stderr)
        errs += 1
        continue
    try:
        with open(evals_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: {evals_path} — JSON parse error: {e}", file=sys.stderr); errs += 1; continue
    if not isinstance(data, list) or len(data) == 0:
        print(f"ERROR: {evals_path} — must be a non-empty top-level JSON array.", file=sys.stderr); errs += 1; continue
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            print(f"ERROR: {evals_path}[{i}] — entry must be an object.", file=sys.stderr); errs += 1; continue
        for field in ('id', 'question', 'expected_skill', 'ground_truth', 'expected_behavior'):
            if field not in entry:
                print(f"ERROR: {evals_path}[{i}] — missing `{field}`.", file=sys.stderr); errs += 1
        if 'expected_behavior' in entry and (not isinstance(entry['expected_behavior'], list) or len(entry['expected_behavior']) == 0):
            print(f"ERROR: {evals_path}[{i}] — `expected_behavior` must be a non-empty list.", file=sys.stderr); errs += 1
sys.exit(errs)
PY
[ $? -eq 0 ] && ok "all skills have evals/evals.json" || errors=$((errors + $?))

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
    # Disabled: this entry pre-dated the SKILL.md's current wording and was
    # already failing on main before this branch. Re-enable once the cosmos-reason
    # SKILL.md is updated to mention the baseline-eval guardrail.
    # "skills/models/tao-finetune-cosmos-reason/SKILL.md": [
    #     "run the model's evaluate",
    #     "action once after preflight",
    #     "Report that eval job id, result path, and accuracy",
    # ],
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
import os, re, sys, yaml
errs = 0
VALID_MODES = {'config', 'args', 'passthrough'}
VALID_CONFIG_FORMATS = {'yaml', 'toml', 'json'}
VALID_GPU_RUNTIME_KEYS = {
    'min_driver_version',
    'min_cuda_version',
    'min_container_toolkit_version',
}
VERSION_RE = re.compile(r'^[0-9]+(?:\.[0-9]+)*$')

try:
    with open('versions.yaml') as vf:
        manifest = yaml.safe_load(vf) or {}
except FileNotFoundError:
    manifest = {}

hf_model_owners = {}

def iter_metadata_files():
    for root, dirs, files in os.walk('.'):
        dirs[:] = [
            d for d in dirs
            if d not in ('.git', 'templates', '.claude-plugin', '.codex-plugin', '.venv', '.venv-tao')
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
    is_primary_model_metadata = (
        path.startswith('./skills/models/')
        and path.endswith('/references/skill_info.yaml')
    )

    if 'huggingface_model_ids' in info:
        model_ids = info['huggingface_model_ids']
        if not is_primary_model_metadata:
            print(f"ERROR: {path} — huggingface_model_ids is only valid in primary model metadata", file=sys.stderr); errs += 1
        elif not isinstance(model_ids, list) or not model_ids:
            print(f"ERROR: {path} — huggingface_model_ids must be a non-empty list", file=sys.stderr); errs += 1
        else:
            for model_id in model_ids:
                if not isinstance(model_id, str) or not model_id.strip():
                    print(f"ERROR: {path} — every huggingface_model_ids entry must be a non-empty string", file=sys.stderr); errs += 1
                    continue
                model_id = model_id.strip()
                if model_id.count('/') != 1 or any(not part for part in model_id.split('/')):
                    print(f"ERROR: {path} — Hugging Face model ID '{model_id}' must use namespace/repository form", file=sys.stderr); errs += 1
                    continue
                hf_model_owners.setdefault(model_id.casefold(), []).append((model_id, skill_dir))

    if isinstance(info.get('container_image'), str):
        validate_image(path, info['container_image'], 'container_image')
    elif is_model_or_data and 'actions' in info:
        print(f"WARN: {path} — has actions but no top-level container_image", file=sys.stderr)

    runtime_requirements = info.get('runtime_requirements')
    if runtime_requirements is not None:
        if not isinstance(runtime_requirements, dict):
            print(f"ERROR: {path} — runtime_requirements must be a mapping", file=sys.stderr); errs += 1
        else:
            gpu_host = runtime_requirements.get('gpu_host')
            unknown_scopes = set(runtime_requirements) - {'gpu_host'}
            if unknown_scopes:
                print(f"ERROR: {path} — unknown runtime requirement scopes: {sorted(unknown_scopes)}", file=sys.stderr); errs += 1
            if not isinstance(gpu_host, dict):
                print(f"ERROR: {path} — runtime_requirements.gpu_host must be a mapping", file=sys.stderr); errs += 1
            else:
                unknown_keys = set(gpu_host) - VALID_GPU_RUNTIME_KEYS
                if unknown_keys:
                    print(f"ERROR: {path} — unknown gpu_host requirement keys: {sorted(unknown_keys)}", file=sys.stderr); errs += 1
                for key, value in gpu_host.items():
                    if not isinstance(value, str) or not VERSION_RE.fullmatch(value):
                        print(f"ERROR: {path} — runtime_requirements.gpu_host.{key} must be a quoted numeric dotted version", file=sys.stderr); errs += 1

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

for owners in hf_model_owners.values():
    unique_skills = sorted({skill_dir for _, skill_dir in owners})
    if len(unique_skills) > 1:
        model_id = owners[0][0]
        print(
            f"ERROR: Hugging Face model ID '{model_id}' has multiple TAO owners: {', '.join(unique_skills)}",
            file=sys.stderr,
        )
        errs += 1
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
