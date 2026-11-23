#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the flat plugin package that NVIDIA's plugin verification signs.

The repository root is the plugin that users install from this repository's own
marketplace files; its skills sit in layer folders, skills/<layer>/<skill>/.
NVIDIA's plugin verification signs something else: one self-contained folder,
plugins/nvidia-tao-toolkit/, with a root plugin.json (Agent Plugins 1.0.0), an
evals/ folder and every skill directly under skills/. This script generates that
folder from the repository. The folder is never edited by hand: change the
repository and build it again.

    python3 scripts/build_plugin_package.py            # write plugins/nvidia-tao-toolkit/
    python3 scripts/build_plugin_package.py --check    # fail when that folder is stale
    python3 scripts/build_plugin_package.py --force    # rebuild a signed folder, drop unknown files

What a build does:
  * copies every tracked file of every skill to skills/<skill>/, except what
    SKILL_FILES_LEFT_OUT and ONE_SKILL_FILES_LEFT_OUT name: per-skill signing
    files, tests and their fixtures, and per-skill evaluation configs;
  * copies SUPPORT_FILES: the hook, the helper scripts and templates the skills
    call, AGENTS.md, versions.yaml, LICENSE, the plugin-level evaluation suite
    and the README of the package, which the repository keeps under docs/;
  * rewrites skills/<layer>/<skill> references in every text file it copies, and
    applies PACKAGE_EDITS, the hand-written edits for code that counts directory
    levels or joins a layer-relative path and for text that is only true of the
    repository;
  * names skills by the plugin they ship in, nvidia-tao-toolkit:<skill>, and the
    bank by the plugin folder where a placeholder named a clone;
  * writes catalog/skill-layers.json, which scripts/skill_layers.py reads when the
    layer folders are absent, and moves the layer-level manifests to catalog/;
  * writes the manifests: plugin.json, .claude-plugin/, .cursor-plugin/ and,
    unless --no-codex-manifest is given, .codex-plugin/;
  * checks the result against the verification pipeline's limits and the rules
    of this package, and writes nothing when one is broken.

Only files tracked by git are read, so local files (env files, results,
virtualenvs) cannot end up in the package. The output depends only on those
files and the options: sorted iteration, fixed JSON formatting, no timestamps.

An existing folder is treated with care, because verification signs it. Files
that verification and evaluation add (VERIFICATION_* below) are never written,
compared or removed without notice. A file the build does not know stops the
build unless --force is given, and so does a signature when the build would
change anything; a forced rebuild that changes or removes a file also removes
the signature. Junk the pipeline forbids (.DS_Store, __pycache__, *.pyc,
__MACOSX) is removed. Whatever is removed is listed before it is removed, and
.git is never read or touched.

Exit codes: 0 the folder was written or is up to date, 1 --check found
differences, 2 the package cannot be built or written (the message says why).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import posixpath
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from skill_layers import CATALOG_DIR, LAYERS, MEMBERSHIP_FILE

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_NAME = "nvidia-tao-toolkit"
DEFAULT_OUTPUT = PurePosixPath("plugins") / PLUGIN_NAME
MEMBERSHIP_PATH = f"{CATALOG_DIR}/{MEMBERSHIP_FILE}"

AGENT_PLUGINS_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
# The fields Agent Plugins 1.0.0 defines besides "$schema". Anything else in a
# client manifest (displayName, skills, interface) stays in that client's file.
AGENT_PLUGINS_FIELDS = (
    "name", "version", "description", "author", "homepage", "repository", "license", "keywords",
)
AGENT_PLUGINS_AUTHOR_FIELDS = ("name", "email", "url")
# The JSON type of each field; every author field and every keyword is a string.
AGENT_PLUGINS_TYPES = {
    "name": str, "version": str, "description": str, "author": dict, "homepage": str,
    "repository": str, "license": str, "keywords": list,
}
AGENT_PLUGINS_NAME = re.compile(r"(?!.*(?:--|\.\.))[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?")
CLAUDE_MANIFEST = ".claude-plugin/plugin.json"
CODEX_MANIFEST = ".codex-plugin/plugin.json"
CURSOR_MANIFEST = ".cursor-plugin/plugin.json"
EVAL_SUITE = "evals/evals.json"

# Limits of the verification pipeline for one plugin folder.
MAX_FILES = 4096
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_DEPTH = 64
JUNK_NAMES = (".DS_Store", "__pycache__", "__MACOSX")
JUNK_SUFFIXES = (".pyc",)
# Signing reads about 5,000 tokens of a SKILL.md; scripts/validate-skills.sh holds
# the repository to the same number of characters. A rewrite can add a few.
SKILL_MD_MAX_CHARS = 20000
# A file with one of these extensions is text: it must decode as UTF-8, because
# the rewrite and every check below read it as text.
TEXT_SUFFIXES = (
    ".cfg", ".csv", ".example", ".html", ".ini", ".js", ".json", ".jsonl", ".md", ".mjs", ".py",
    ".sh", ".template", ".tmpl", ".toml", ".ts", ".txt", ".yaml", ".yml",
)

# What the package carries from outside skills/, and what needs it. A path that
# ends in "/" takes every tracked file below it. Tests, docs, CI files,
# marketplace files, integrations/ and templates/skill-skeleton/ stay out: no
# skill or hook reads them, and a SKILL.md outside skills/ is not allowed.
# scripts/install-codex-agents.sh stays out as well: it installs the repository
# root as a plugin, and next to this package Codex would list every skill twice.
SUPPORT_FILES = {
    "AGENTS.md": "agent guidance; the session-start hook prints it, and "
                 "scripts/tao_agent_identity.py adds it to a project",
    "LICENSE": "licence of everything in the package",
    "README.md": "what the plugin is and needs, for people and for plugin validators; the "
                 "session-start hook names its Credentials section",
    "versions.yaml": "image and wheel pins that skills and scripts/resolve_*.py read",
    EVAL_SUITE: "plugin-level evaluation suite",
    "hooks/hooks.json": "registers the session-start hook",
    "hooks/session_start.sh": "session-start hook",
    "scripts/check_tao_launch_preflight.py": "launch preflight of the launch and platform skills",
    "scripts/list_automl_support.py": "AutoML support listing of tao-run-automl",
    "scripts/list_tao_capabilities.py": "capability listing of tao-list-capabilities",
    "scripts/list_tao_models.py": "model listing; imported by the two listings above",
    "scripts/nccl_allreduce_probe.py": "multi-node probe of the SLURM and Kubernetes skills",
    "scripts/redact_secrets.py": "secret lint every platform skill runs before a launch",
    "scripts/resolve_tao_image.py": "container image resolver of model and application skills",
    "scripts/resolve_tao_model.py": "model alias resolver; imported by resolve_tao_image.py",
    "scripts/resolve_versions_key.py": "versions.yaml key resolver of data and application skills",
    "scripts/skill_layers.py": "layer lookup the listing and resolver scripts import",
    "scripts/tao_agent_identity.py": "identity block in a project's AGENTS.md; tao-setup runs it",
    "scripts/tao_job_record.py": "job record of the four-verb launch contract",
    "scripts/tao_slurm_ssh.py": "SSH helper the launch preflight and tao-run-deft-pas import",
    "templates/k8s/": "job templates tao-run-on-kubernetes renders",
    "templates/slurm/": "sbatch templates tao-run-on-slurm renders",
}

# Support files that the repository keeps at another path, because the path they
# have in the package is taken there: path in the package -> path in the repository.
SUPPORT_FILE_SOURCES = {
    "README.md": "docs/plugin-package-readme.md",  # README.md is the repository's own
}

# Shipped files that name a file under scripts/ of the repository which the
# package leaves out: (file in the package, the path it names) -> why that is fine.
UNSHIPPED_SCRIPT_MENTIONS = {
    ("scripts/skill_layers.py", "scripts/build_plugin_package.py"):
        "docstring that says which script generates the flat layout",
    ("versions.yaml", "scripts/stamp_versions.py"):
        "comment that names the tool which stamps this file in the repository",
    ("hooks/session_start.sh", "scripts/tests/test_session_start_hook.py"):
        "comment that names the repository test of the hook's output size",
    ("skills/tao-run-dinov3-ssl-deft/SKILL.md", "scripts/deft_container_handoff.py"):
        "described there as a repository-only release QA tool",
}

# Files inside a skill folder that the package leaves out, by path inside the
# skill, and why. "*" matches within one name and never crosses "/"; a leading
# "**/" matches any number of folders and a trailing "/**" everything below one.
#
# Verification evaluates, describes and signs the package as one plugin, so
# nothing that vouches for a single skill at its layer path is copied, and a
# build from an unsigned tree and one from the signed tree have the same file
# list. Tests and evaluation configs are left out because no agent needs them at
# run time and every file in the package is scanned.
SKILL_FILES_LEFT_OUT = {
    "skill.oms.sig": "signature of the skill as it sits in its layer folder; it cannot be valid "
                     "for the rewritten copy, and the package gets one signature of its own",
    "BENCHMARK.md": "report of the per-skill evaluation that quotes layer paths; rewriting a tool "
                    "report would falsify it, and the package gets a benchmark of its own",
    "skill-card.md": "card that signing regenerates with the agents and tasks of the per-skill "
                     "evaluation; the package gets a card of its own",
    "*-review-needed.md": "request for review that per-skill signing leaves beside the card; it "
                          "is about the skill at its layer path",
    "config/*-baseline.yaml": "scanner baseline of per-skill signing; nothing reads it at run "
                              "time and plugin verification accepts no baseline",
    ".gitleaks.toml": "scanner allow-list of per-skill signing; same reason as the baseline",
    ".gitleaksignore": "scanner allow-list of per-skill signing; same reason as the baseline",
    "**/tests/**": "tests of the skill's own code and their fixtures; they run in the repository",
    "**/test_*.py": "unit-test module kept beside the script it tests",
    "eval*.config": "configuration of the per-skill evaluation harness; the package is evaluated "
                    "with evals/evals.json, and each skill keeps its evals/ folder",
}

# Single files left out of one skill: (skill, path inside the skill) -> why.
ONE_SKILL_FILES_LEFT_OUT = {
    ("tao-setup", "scripts/install-codex-agents.sh"):
        "copy of the repository's Codex installer; see SUPPORT_FILES",
}

# Files that a pattern in SKILL_FILES_LEFT_OUT covers, or whose name looks like a
# test, and that the package keeps: (skill, path inside the skill) -> why.
SKILL_FILES_KEPT = {
    ("tao-run-deft-aoi-cosmos3", "tests/test_cosmos3_bare.py"):
        "the bundled validation that the Installation section of the skill tells the user to run",
    ("tao-token-efficient-execution", "scripts/smoke_test.sh"):
        "adapter check that scripts/install.sh prints as a next step and the skill lists",
}

# Shipped files that still name a left-out file: (file in the package, what it
# names) -> why that is fine. Instructions to run or read a left-out file are
# not listed here; they are rewritten in PACKAGE_EDITS.
_DATA_IO = "skills/tao-data-io/references/"
LEFT_OUT_MENTIONS = {
    (_DATA_IO + "selective_download.py", _DATA_IO + "tests/test_selective_download.py"):
        "docstring that names the unit test of this module; nothing is run from it",
    ("skills/tao-token-efficient-execution/references/CARD_AUTHORING.md",
     "skills/tao-train-visual-changenet/eval.config"):
        "record of the files a past authoring session read in the repository, not an instruction",
}

# What verification and evaluation put into the folder after a build. The build
# never writes these names, --check ignores them, and a rebuild never removes one
# without saying so. Two naming families are in use and no real run has settled
# which one applies to a plugin, so both count; the list is provisional until
# then, see docs/maintenance.md.
#   at the folder root: the report, the cards, the signature, the evaluation result;
VERIFICATION_ROOT_FILES = (
    "BENCHMARK.md", "plugin-card.json", "plugin-card.md", "skill-card.md", "eval-result.json",
)
#   directly inside skills/<skill>/: what per-skill signing writes there;
VERIFICATION_SKILL_FILES = ("BENCHMARK.md", "skill-card.md")
#   at both of these levels: a request for review;
VERIFICATION_PATTERNS = ("*-review-needed.md",)
#   anywhere: a signature. It lists the files of the folder, so a rebuild that
#   changes one makes it stale;
SIGNATURE_FILES = ("skill.oms.sig", "plugin.sig.oms", "plugin.oms.sig")
#   and everything below these folders: the evaluation results.
VERIFICATION_DIRS = (".eval-results/",)

# Text that stays although the leftover check matches it: (file in the package,
# the match) -> why it is not a reference to a layer folder.
LAYER_WORDS_ALLOWED = {
    ("hooks/session_start.sh", "skills/*/*/"):
        "the hook scans one and two levels below skills/ so that it works in both layouts",
    ("skills/tao-finetune-huggingface-model/references/deliverables.md", "`data/`"):
        "the data folder of the project this skill generates",
    ("skills/tao-finetune-huggingface-model/references/workflow-generate-train.md", "`data/`"):
        "the data folder of the project this skill generates",
    ("skills/tao-finetune-cosmos-embed/references/data-staging.md", "`data/`"):
        "the mounted dataset folder",
    ("skills/tao-train-nvpanoptix3d/SKILL.md", "`data/`"):
        "the dataset folder the archive is extracted into",
    ("skills/tao-run-deft-aoi-cosmos3/references/data-layout.md", "`models/`"):
        "the models tree of a run workspace",
    ("skills/tao-run-deft-aoi-cosmos3/references/data-layout.md", "├── models/"):
        "the models tree in the drawing of a run workspace",
    ("skills/tao-run-deft-cr-its-mining/SKILL.md", "├── data/"):
        "the data folder in the drawing of the DEFT workspace",
    ("skills/tao-finetune-cosmos-embed/SKILL.md", "├── data/"):
        "the data folder in the drawing of the local workspace",
    ("skills/tao-finetune-video-clip/SKILL.md", "├── data/"):
        "the data folder in the drawing of the workspace",
}

_APPS = "skills/applications/"
_PROBE = "scripts/redact_secrets.py scripts/tao_job_record.py "
_FLAT_BANK = f"`{DEFAULT_OUTPUT}`"
# The bank check of tao-setup, the launch preflight and the CLIP setup. Its first
# lines find the plugin folder from the folder of the skill when nothing exported
# TAO_SKILL_BANK_PATH; the reader fills in that one folder. "cd -P" and "pwd -P"
# follow links as the file test does: a skill folder that is a link into a plugin
# folder leads to that plugin folder, not to the folder above the link.
_BANK_DEFAULT = 'export TAO_SKILL_BANK_PATH="${TAO_SKILL_BANK_PATH:-$HOME/tao-skill-bank}"\n'
_BANK_LINES = (
    "# TAO_SKILL_BANK_PATH not set? Set it to the plugin folder that contains this skill: "
    "two levels above this skill's folder.\n"
    'skill_dir="<this-skill-folder>"   # the folder that holds this skill\'s SKILL.md\n'
    'if [ -z "${TAO_SKILL_BANK_PATH:-}" ] && [ -f "$skill_dir/../../' + MEMBERSHIP_PATH
    + '" ]; then\n'
    '  TAO_SKILL_BANK_PATH="$(cd -P "$skill_dir/../.." && pwd -P)"\n'
    "fi\n"
)
_INCOMPLETE = '"Incomplete TAO skill bank: missing $item under $TAO_SKILL_BANK_PATH"'
_BANK_HINT = (
    ". Set TAO_SKILL_BANK_PATH to the " + PLUGIN_NAME + " plugin folder that contains this "
    "skill (two levels above this skill's folder); a repository checkout without "
    + str(DEFAULT_OUTPUT) + " is not a bank."
)
_REPO_ROOT = "Run from the `tao-skill-bank` repo root."
_BANK_ROOT = "Run from the bank root, the plugin folder that `$TAO_SKILL_BANK_PATH` names."
# Hand-written edits: (file in the repository, text there, text in the package).
# Each must match exactly once. They cover what no path rewrite can see: code that
# counts directory levels up from its own file, a "<layer>/<skill>" fragment joined
# onto the skills folder, prose that names a bare layer folder, and instructions
# that are only true of the repository (where the bank and its identity helper
# are, tests and evaluation configs the package leaves out).
PACKAGE_EDITS = (
    # shell: climb from a skill sub-folder to the package root, one level fewer
    (_APPS + "tao-run-automl/cards/driver.sh",
     'BANK=$(cd "$PACK/../../../.." && pwd)               # skill-bank root',
     'BANK=$(cd "$PACK/../../.." && pwd)                  # skill-bank root'),
    (_APPS + "tao-run-deft-aoi/cards/driver.sh",
     'BANK=$(cd "$PACK/../../../.." && pwd)               # skill-bank root',
     'BANK=$(cd "$PACK/../../.." && pwd)                  # skill-bank root'),
    (_APPS + "tao-run-deft-aoi-cosmos3/scripts/deft_python.sh",
     '"$script_dir/../../../../.."', '"$script_dir/../../../.."'),
    (_APPS + "tao-run-deft-aoi/scripts/deft_python.sh",
     '"$script_dir/../../../../.."', '"$script_dir/../../../.."'),
    (_APPS + "tao-run-deft-object-detection/scripts/deft_python.sh",
     '"$script_dir/../../../../.."', '"$script_dir/../../../.."'),
    (_APPS + "tao-run-deft-pas/scripts/deft_python.sh",
     '"$script_dir/../../../../.."', '"$script_dir/../../../.."'),
    ("skills/core/tao-token-efficient-execution/scripts/install.sh",
     'BANK=$(cd "$SKILL/../../.." && pwd)', 'BANK=$(cd "$SKILL/../.." && pwd)'),
    # python: parents[N] that reached the package root
    (_APPS + "tao-run-deft-aoi-cosmos3/scripts/submit_cfw_train.py",
     "else pathlib.Path(__file__).resolve().parents[4]",
     "else pathlib.Path(__file__).resolve().parents[3]"),
    (_APPS + "tao-run-deft-pas/scripts/classify_pas_image_acquisition.py",
     'Path(__file__).resolve().parents[4] / "scripts"',
     'Path(__file__).resolve().parents[3] / "scripts"'),
    (_APPS + "tao-run-deft-pas/scripts/run_deft_action.py",
     'pathlib.Path(__file__).resolve().parents[4] / "scripts" / "redact_secrets.py"',
     'pathlib.Path(__file__).resolve().parents[3] / "scripts" / "redact_secrets.py"'),
    (_APPS + "tao-run-deft-pas/scripts/run_deft_docker_action.py",
     "REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]",
     "REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]"),
    ("skills/data/paidf-cosmos-predict/scripts/paidf_common.py",
     "REPO_ROOT = Path(__file__).resolve().parents[4]",
     "REPO_ROOT = Path(__file__).resolve().parents[3]"),
    ("skills/platform/tao-run-on-kubernetes/scripts/render_action_job.py",
     "REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]",
     "REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]"),
    # python: parents[N] that reached skills/ and then named "<layer>/<skill>"
    (_APPS + "tao-run-automl/references/best_rec_adapter.py",
     'Path(__file__).resolve().parents[3]\n'
     '               / "core/tao-artifacts/references/best_rec.schema.json"',
     'Path(__file__).resolve().parents[2]\n'
     '               / "tao-artifacts/references/best_rec.schema.json"'),
    (_APPS + "tao-run-deft-aoi/scripts/prepare_card_mining.py",
     'parents[3] / "data/tao-mine-aoi-images/scripts"',
     'parents[2] / "tao-mine-aoi-images/scripts"'),
    (_APPS + "tao-run-deft-cr-its-mining/scripts/prepare_cosmos_reason_model.py",
     '    skills_root = Path(__file__).resolve().parents[3]\n'
     '    return (\n        skills_root\n'
     '        / "models"\n        / "tao-finetune-cosmos-reason"\n',
     '    skills_root = Path(__file__).resolve().parents[2]\n'
     '    return (\n        skills_root\n        / "tao-finetune-cosmos-reason"\n'),
    (_APPS + "tao-run-deft-pas/scripts/run_deft_action.py",
     '        pathlib.Path(__file__).resolve().parents[3]\n'
     '        / "core"\n        / "tao-artifacts"\n',
     '        pathlib.Path(__file__).resolve().parents[2]\n'
     '        / "tao-artifacts"\n'),
    # prose: a path that named a layer without the skills/ prefix, or a bare layer folder
    (_APPS + "tao-run-deft-aoi/references/tao-generate-anomalies.md",
     "(`data/tao-generate-anomalies/SKILL.md`)", "(`skills/tao-generate-anomalies/SKILL.md`)"),
    (_APPS + "tao-run-deft-aoi/references/tao-generate-anomalies.md",
     "Template at `data/tao-generate-anomalies/assets/",
     "Template at `skills/tao-generate-anomalies/assets/"),
    (_APPS + "tao-run-automl-deft-pipeline/references/phase-handoffs.md",
     "${TAO_SKILL_BANK_PATH}/applications/tao-run-deft-aoi/scripts/",
     "${TAO_SKILL_BANK_PATH}/skills/tao-run-deft-aoi/scripts/"),
    (_APPS + "tao-run-deft-object-detection/SKILL.md",
     "`skills/<layer>/<name>/SKILL.md`", "`skills/<name>/SKILL.md`"),
    (_APPS + "tao-finetune-huggingface-model/SKILL.md",
     "dedicated `skills/models/*` skill;", "dedicated model skill;"),
    ("AGENTS.md",
     "`skills/models/<arch>/SKILL.md` (network specifics),\n"
     "   `skills/data/<name>/SKILL.md` (transforms), or `skills/applications/<name>/SKILL.md`\n",
     "`skills/<name>/SKILL.md` of the model skill (network specifics),\n"
     "   the data skill (transforms), or the application skill\n"),
    ("skills/core/tao-list-capabilities/SKILL.md",
     "- Every top-level application workflow under `applications/` and what it can do.",
     "- Every application workflow the helper lists and what it can do."),
    ("skills/core/tao-list-capabilities/SKILL.md",
     "- Every top-level data workflow under `data/` and what it can do.",
     "- Every data workflow the helper lists and what it can do."),
    ("skills/core/tao-list-capabilities/SKILL.md",
     "coverage for models under `models/`: train,", "coverage for the model skills: train,"),
    ("skills/core/tao-list-capabilities/SKILL.md",
     "`scripts/list_tao_models.py` for model/action and AutoML support questions.\n",
     "`scripts/list_tao_models.py` for model/action and AutoML support questions.\n"
     "The commands below read `TAO_SKILL_BANK_PATH`. When it is not set (no session\n"
     "hook exported it, as in Codex), set it first to the plugin folder that contains\n"
     "this skill, the folder two levels above this skill's folder.\n"),
    # where the bank is: the package is flat and a checkout of the repository is
    # not, so the text names the plugin folder that contains the skill. No hook
    # exports TAO_SKILL_BANK_PATH in Codex, so each check finds that folder itself.
    ("skills/core/tao-setup/SKILL.md",
     "Individual skill installs do not include the shared `scripts/` helpers. Obtain\n"
     "a complete TAO skill bank checkout at the same release ref as the installed\n"
     "skills, then verify it before using any shared helper. If the checkout is\n"
     "elsewhere, set `TAO_SKILL_BANK_PATH` to its root. Reuse a plugin-provided path\n"
     "when present. If no checkout exists, obtain it with user approval, for example\n"
     "`git clone --branch <installed-release-ref> --single-branch "
     "https://github.com/NVIDIA-TAO/tao-skill-bank.git <bank-dir>`.\n"
     "Point `TAO_SKILL_BANK_PATH` at that checkout and stop if the helper check fails.\n",
     "Individual skill installs do not include the shared `scripts/` helpers. The\n"
     "bank that matches these skills is the `" + PLUGIN_NAME + "` plugin folder: the\n"
     "folder that holds `skills/`, `scripts/`, `catalog/` and `versions.yaml`, with\n"
     "every skill directly under `skills/`. Reuse a plugin-provided\n"
     "`TAO_SKILL_BANK_PATH` when present. When `TAO_SKILL_BANK_PATH` is not already\n"
     "set (no session hook exported it, as in Codex), set it before the check to the\n"
     "plugin folder that contains this skill, the folder two levels above this\n"
     "skill's folder, when that folder passes the check. The first lines of the\n"
     "block do that once `<this-skill-folder>` is replaced with the folder that\n"
     "holds this `SKILL.md`.\n\n"
     "A checkout of https://github.com/NVIDIA-TAO/tao-skill-bank is a bank only when\n"
     "it contains " + _FLAT_BANK + "; the bank is then that folder, never the\n"
     "root of the checkout, which keeps its skills in layer folders. When a checkout\n"
     "has no such folder, do not use it: with user approval, install the complete\n"
     "`" + PLUGIN_NAME + "` plugin from the catalog these skills came from. Verify the\n"
     "bank before using any shared helper, and stop if the check fails.\n"),
    ("skills/core/tao-setup/SKILL.md", _BANK_DEFAULT, _BANK_LINES + _BANK_DEFAULT),
    ("skills/core/tao-setup/SKILL.md",
     _PROBE + "skills/models skills/platform; do",
     _PROBE + "scripts/skill_layers.py " + MEMBERSHIP_PATH + "; do"),
    ("skills/core/tao-setup/SKILL.md", _INCOMPLETE, _INCOMPLETE[:-1] + _BANK_HINT + '"'),
    ("skills/core/tao-launch-workflow/references/platform-preflight.md",
     "`TAO_SKILL_BANK_PATH` when set. Otherwise obtain a complete checkout at the\n"
     "installed skills' release ref with user approval, as described in `tao-setup`.\n"
     "Individual skill installs do not include shared helper scripts. Set\n"
     "`TAO_SKILL_BANK_PATH` to another checkout root if needed, then run:\n",
     "`TAO_SKILL_BANK_PATH` when set. The bank is the `" + PLUGIN_NAME + "` plugin\n"
     "folder, which holds `skills/`, `scripts/`, `catalog/` and `versions.yaml`. When\n"
     "`TAO_SKILL_BANK_PATH` is not already set (no session hook exported it, as in\n"
     "Codex), set it before the check to the plugin folder that contains this skill,\n"
     "the folder two levels above this skill's folder, when that folder passes the\n"
     "check. The first lines of the block do that once `<this-skill-folder>` is\n"
     "replaced with the folder that holds this skill's `SKILL.md`. A checkout of the\n"
     "repository is a bank only when it contains " + _FLAT_BANK + ", and\n"
     "the bank is then that folder, never the root of the checkout. When it has\n"
     "none, install the plugin instead, as described in `tao-setup`. Individual\n"
     "skill installs do not include shared helper scripts. Run:\n"),
    ("skills/core/tao-launch-workflow/references/platform-preflight.md",
     _BANK_DEFAULT, _BANK_LINES + _BANK_DEFAULT),
    ("skills/core/tao-launch-workflow/references/platform-preflight.md",
     _PROBE + "skills/models skills/platform; do",
     _PROBE + "scripts/skill_layers.py " + MEMBERSHIP_PATH + "; do"),
    ("skills/core/tao-launch-workflow/references/platform-preflight.md",
     _INCOMPLETE, _INCOMPLETE[:-1] + _BANK_HINT + '"'),
    ("skills/models/tao-finetune-clip/references/standalone-setup.md",
     "`TAO_SKILL_BANK_PATH` to a complete TAO skill-bank checkout, not to the\n"
     "individual installed skill directory. Verify it with:\n",
     "`TAO_SKILL_BANK_PATH` to the `" + PLUGIN_NAME + "` plugin folder, which holds\n"
     "`skills/`, `scripts/`, `catalog/` and `versions.yaml`, not to the individual\n"
     "installed skill directory and not to the root of a repository checkout. When\n"
     "the variable is not already set (no session hook exported it, as in Codex),\n"
     "set it to the plugin folder that contains this skill, the folder two levels\n"
     "above this skill's folder, when that folder passes the check. The first lines\n"
     "of the block do that once `<this-skill-folder>` is replaced with the folder\n"
     "that holds this skill's `SKILL.md`. Verify it with:\n"),
    ("skills/models/tao-finetune-clip/references/standalone-setup.md",
     _BANK_DEFAULT, _BANK_LINES + _BANK_DEFAULT),
    ("skills/models/tao-finetune-clip/references/standalone-setup.md",
     "%s/scripts/%s. CLIP requires a complete bank checkout.\\n'",
     "%s/scripts/%s. Set TAO_SKILL_BANK_PATH to the " + PLUGIN_NAME + " plugin folder "
     "that contains this skill (two levels above the skill folder); a repository checkout "
     "without " + str(DEFAULT_OUTPUT) + " is not a bank.\\n'"),
    ("skills/models/tao-finetune-clip/references/standalone-setup.md",
     '[ -f "$TAO_SKILL_BANK_PATH/versions.yaml" ] && [ -d "$TAO_SKILL_BANK_PATH/skills" ] || {\n'
     "  echo 'Incomplete TAO skill-bank checkout; stop before CLIP launch.' >&2\n",
     '[ -f "$TAO_SKILL_BANK_PATH/versions.yaml" ] && [ -f "$TAO_SKILL_BANK_PATH/'
     + MEMBERSHIP_PATH + '" ] || {\n'
     "  echo 'Incomplete TAO skill bank; a repository checkout without "
     + str(DEFAULT_OUTPUT) + " is not a bank. Stop before CLIP launch.' >&2\n"),
    ("skills/models/tao-finetune-clip/references/standalone-setup.md",
     "If no complete checkout exists, obtain approval to clone\n"
     "`https://github.com/NVIDIA-TAO/tao-skill-bank.git` into a user-selected\n"
     "location, then set the variable and rerun the check.",
     "If no such folder exists, obtain approval to install the complete\n"
     "`" + PLUGIN_NAME + "` plugin, then set the variable to the installed plugin\n"
     "folder and rerun the check. A checkout of the repository is a bank only when\n"
     "it contains " + _FLAT_BANK + ",\n"
     "and the bank is then that folder."),
    (_APPS + "tao-run-automl/references/automl-preflight-concepts.md",
     "   - cloned standalone: `~/tao-skill-bank/` (or wherever the user cloned).\n"
     "   - Installed skill-bank cache: `<agent-cache>/tao-skill-bank/<version>/`.\n"
     "   - Codex plugin: `~/.codex/plugins/cache/<marketplace>/tao-skill-bank/<version>/`.\n"
     "   - submodule inside a cloned SDK: `<sdk>/tao-skill-bank/`.\n",
     "   - installed plugin: the plugin folder that contains this skill, two levels\n"
     "     above this skill's folder, in the agent's plugin cache.\n"
     "   - checkout of the repository: " + _FLAT_BANK + " inside it, when the\n"
     "     checkout has that folder. The root of a checkout is not the bank.\n"),
    (_APPS + "tao-run-automl/references/automl-preflight-concepts.md",
     "   tao-skill-bank/\n"
     "   └── skills/\n"
     "       ├── applications/         # workflow configs (this skill)\n"
     "       ├── models/               # per-network skill packages\n"
     "       │   ├── <model_skill>/\n"
     "       │   │   ├── SKILL.md\n"
     "       │   │   ├── schemas/\n"
     "       │   │   │   └── <action>.schema.json       # REQUIRED AutoML gate\n"
     "       │   │   └── references/\n"
     "       │   │       ├── skill_info.yaml             # actions, data_sources, "
     "container image\n"
     "       │   │       └── spec_template_train.yaml    # default training spec "
     "(recommended)\n"
     "       │   └── ...\n"
     "       ├── data/\n"
     "       └── platform/\n",
     "   <bank-root>/\n"
     "   ├── catalog/                  # layer of every skill, model action manifests\n"
     "   ├── scripts/                  # shared helpers\n"
     "   ├── versions.yaml\n"
     "   └── skills/                   # every skill directly below, in one folder each\n"
     "       ├── tao-run-automl/       # workflow configs (this skill)\n"
     "       ├── <model_skill>/        # per-network skill package\n"
     "       │   ├── SKILL.md\n"
     "       │   ├── schemas/\n"
     "       │   │   └── <action>.schema.json       # REQUIRED AutoML gate\n"
     "       │   └── references/\n"
     "       │       ├── skill_info.yaml             # actions, data_sources, "
     "container image\n"
     "       │       └── spec_template_train.yaml    # default training spec "
     "(recommended)\n"
     "       └── ...\n"),
    (_APPS + "tao-run-automl/references/automl-preflight-concepts.md",
     "(or via the `tao-skills` plugin)", "(or via the `" + PLUGIN_NAME + "` plugin)"),
    (_APPS + "tao-run-automl/SKILL.md",
     'skill_bank = Path("<absolute-tao-skill-bank>")', 'skill_bank = Path("<absolute-bank-root>")'),
    (_APPS + "tao-run-deft-aoi/references/tao-generate-anomalies.md",
     "TAO_SKILL_BANK_ROOT=<tao-skill-bank>", "TAO_SKILL_BANK_ROOT=<bank-root>"),
    ("skills/models/tao-finetune-clip/SKILL.md",
     "and verify the bank checkout before running", "and verify the bank before running"),
    (_APPS + "tao-run-dinov3-ssl-deft/references/tao-container-integration.md",
     "Run it from the Skill Bank checkout; without `--skill-bank` the resolver reads\n"
     "`$TAO_SKILL_BANK_PATH` or `~/tao-skill-bank`, which can be a different bank.\n",
     "Run it from the bank root, the plugin folder; without `--skill-bank` the resolver\n"
     "reads `$TAO_SKILL_BANK_PATH` or the bank it ships in, which can be a different bank.\n"),
    # commands that are run from the bank root: that is the plugin folder, not a clone
    ("skills/data/tao-analyze-detection-kpi/SKILL.md", _REPO_ROOT, _BANK_ROOT),
    ("skills/data/tao-analyze-gaps-od-map/SKILL.md", _REPO_ROOT, _BANK_ROOT),
    ("skills/data/tao-generate-image-embeddings/SKILL.md", _REPO_ROOT, _BANK_ROOT),
    ("skills/data/tao-mine-nearest-neighbors/SKILL.md", _REPO_ROOT, _BANK_ROOT),
    ("skills/data/tao-mine-od-images/SKILL.md", _REPO_ROOT, _BANK_ROOT),
    # python: a bank path that starts with a slash because it is added to a variable
    (_APPS + "tao-run-automl/cards/20-launch-recs.md",
     'os.environ["SB"] + "/skills/models/tao-train-visual-changenet"',
     'os.environ["SB"] + "/skills/tao-train-visual-changenet"'),
    # shell: a loop over one layer folder is a loop over skills/ in the package
    ("skills/core/tao-token-efficient-execution/scripts/install.sh",
     'for pack in "$BANK"/skills/applications/*/cards; do',
     'for pack in "$BANK"/skills/*/cards; do'),
    # Codex and Cursor: the helper that adds the identity to a project's AGENTS.md is
    # in the plugin folder, and no hook exports TAO_SKILL_BANK_PATH in Codex, so the
    # command finds the helper from the folder of the skill, as the bank check does.
    ("skills/core/tao-setup/SKILL.md",
     "With user approval, add the TAO identity to the project's `AGENTS.md` as a\n"
     "marked block (`add` again updates it, `remove` deletes it). In the project:\n\n"
     "```bash\n"
     'python3 "${TAO_SKILL_BANK_PATH:?}/scripts/tao_agent_identity.py" add\n'
     "```\n",
     "In Claude Code the session-start hook of this plugin supplies the TAO agent\n"
     "identity, and nothing more is needed. Codex and Cursor do not load the\n"
     "`AGENTS.md` at the root of the plugin folder; they read the `AGENTS.md` at the\n"
     "root of the project the user works in. With user approval, add the identity\n"
     "to that file as a marked block. The helper creates the file when it is\n"
     "missing, leaves every other byte of an existing file alone, and never copies\n"
     "over a file or makes a backup. Run it in the project. When\n"
     "`TAO_SKILL_BANK_PATH` is not set (no session hook exported it, as in Codex),\n"
     "replace `<this-skill-folder>` first, as in Quick Start:\n\n"
     "```bash\n"
     'skill_dir="<this-skill-folder>"   # the folder that holds this skill\'s SKILL.md\n'
     'helper="${TAO_SKILL_BANK_PATH:-$skill_dir/../..}/scripts/tao_agent_identity.py"\n'
     '[ -f "$helper" ] || { echo "No $helper. Set TAO_SKILL_BANK_PATH to the plugin folder '
     'first, see Quick Start." >&2; exit 1; }\n'
     'python3 "$helper" add\n'
     "```\n\n"
     "`add` again updates the block in place, `status` reports it, and `remove`\n"
     "deletes it; `--dry-run` shows what would change, and `--project DIR` names a\n"
     "project other than the current one. Do not copy the identity over\n"
     "`~/.codex/AGENTS.md` or any other file. Do not register another TAO\n"
     "marketplace or install another TAO plugin for this either: the skills are\n"
     "already installed, and a second copy of the plugin makes Codex list every\n"
     "skill twice.\n"),
    # tests, fixtures and evaluation configs that the package leaves out
    (_APPS + "tao-run-deft-aoi-cosmos3/SKILL.md",
     "Requires the companion TAO skill-bank skills from `eval.config`, host Python",
     "Requires the companion TAO skills, which ship in the same plugin, host Python"),
    ("skills/models/tao-train-optical-inspection/SKILL.md",
     "CSV row number. A two-row path-stub fixture is under\n"
     "`tests/fixtures/dataset/valid/`; it verifies the contract and is not PCB\n"
     "training data.\n",
     "CSV row number.\n"),
    ("skills/platform/tao-data-io/SKILL.md",
     " (boto3 + pandas). Unit tests live in `references/tests/`; run with "
     "`python -m pytest`.\n",
     " (boto3 + pandas).\n"),
    ("skills/core/tao-token-efficient-execution/adapters/pi/README.md",
     "Install `@earendil-works/pi-coding-agent@0.85.1` into a disposable npm prefix,\n"
     "then run `PI_TEST_RUNTIME=/that/prefix node --test "
     "scripts/tests/token_efficient_guard.test.mjs`\n"
     "from the bank root with a clean environment. Driver tests run with\n"
     "`python -m unittest discover -s scripts/tests -p test_token_efficient_execution.py`.\n"
     "Mining tests run with\n"
     "`python -m unittest discover -s skills/applications/tao-run-deft-aoi/tests "
     "-p test_card_mining.py`\n"
     "and need pandas, pyarrow, and PyYAML. These tests use fixtures only; the separate\n"
     "`smoke_test.sh` calls a real model API.\n",
     "The regression tests of the guard, the pack drivers, and card mining use\n"
     "fixtures only and run from a checkout of the repository,\n"
     "https://github.com/NVIDIA-TAO/tao-skill-bank; this plugin package does not carry\n"
     "them. The separate `smoke_test.sh` calls a real model API.\n"),
)

# Lines that climb above their own skill folder and already work in both layouts,
# so they need no entry in PACKAGE_EDITS: (file in the repository, the line).
LAYOUT_NEUTRAL = {
    (_APPS + "tao-run-deft-aoi-cosmos3/tests/test_cosmos3_bare.py",
     "SKILL_ROOT.parents[1] / category / name,"),
}

# A file name that looks like a test; such a file ships only through SKILL_FILES_KEPT.
TEST_LIKE_NAME = re.compile(
    r"(?:^|/)(?:test_[^/]*|[^/]*_test\.[^/]+|[^/]*\.test\.[^/]+|conftest\.py)$"
)
# Code is named in prose by its bare file name; data files (fixtures) are not.
CODE_SUFFIXES = (".py", ".sh", ".mjs", ".js", ".ts")


class PackageError(Exception):
    """The source or the built package breaks a rule; the message lists every case."""


@dataclass
class Package:
    """A built package: its files and the numbers the summary prints."""

    files: dict[str, tuple[bytes, int]]
    membership: dict[str, list[str]]
    manifest: dict
    source: str = ""
    rewritten: list[str] = field(default_factory=list)
    left_out: dict[str, int] = field(default_factory=dict)
    renamed: dict[str, int] = field(default_factory=dict)
    eval_tasks: int = 0


@dataclass
class Found:
    """How an existing folder differs from a build; every list holds package paths."""

    differences: list[str] = field(default_factory=list)  # "missing: x", "content differs: x"
    unknown: list[str] = field(default_factory=list)      # no build writes them, nor verification
    junk: list[str] = field(default_factory=list)         # names the pipeline forbids
    added: list[str] = field(default_factory=list)        # what verification and evaluation add

    @property
    def signatures(self) -> list[str]:
        return [path for path in self.added if is_signature(path)]


@dataclass
class Written:
    """What write() did to a folder: what it removed, what it left, whether content changed."""

    changed: int = 0                                      # packaged files written or replaced
    junk: list[str] = field(default_factory=list)         # removed: junk names
    unknown: list[str] = field(default_factory=list)      # removed: files no build writes
    stale: list[str] = field(default_factory=list)        # removed: signatures of other content
    kept: list[str] = field(default_factory=list)         # left: verification output

    @property
    def removed(self) -> list[tuple[str, str]]:
        """Every removed path with the reason, in the order it is listed."""
        return ([(path, "junk the pipeline rejects") for path in self.junk]
                + [(path, "not part of this build") for path in self.unknown]
                + [(path, "signature of the earlier content") for path in self.stale])


# ─────────────────────────────── source ───────────────────────────────
def _git(source: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(source), *args], capture_output=True, check=False
        )
    except FileNotFoundError as exc:
        raise PackageError("git is required: only tracked files are packaged") from exc
    if result.returncode != 0:
        reason = result.stderr.decode(errors="replace").strip()
        raise PackageError(f"{source} is not a git checkout: {reason}")
    return result.stdout


def tracked_files(source: Path) -> dict[str, str]:
    """Path -> git mode of every tracked file in the source checkout."""
    tracked: dict[str, str] = {}
    for row in _git(source, "ls-files", "--stage", "-z").split(b"\0"):
        if not row:
            continue
        meta, path = row.split(b"\t", 1)
        mode, _sha, stage = meta.decode().split()
        if stage != "0":
            raise PackageError(f"unmerged path in the source checkout: {path.decode()}")
        tracked[path.decode()] = mode
    return tracked


def describe_source(source: Path) -> str:
    """For the summary: the commit that was packaged and whether it had local changes."""
    commit = _git(source, "rev-parse", "--short=8", "HEAD").decode().strip()
    dirty = _git(source, "status", "--porcelain", "--untracked-files=no").strip()
    return f"{source} (commit {commit}{', with uncommitted changes' if dirty else ''})"


def read_file(source: Path, path: str, mode: str) -> tuple[bytes, int]:
    """Content and package mode of one tracked regular file."""
    if mode not in ("100644", "100755"):
        raise PackageError(f"{path} is a symlink or a submodule; only regular files are packaged")
    target = source / path
    if target.is_symlink() or not target.is_file():
        raise PackageError(f"{path} is tracked but is not a regular file in the checkout")
    return target.read_bytes(), 0o755 if mode == "100755" else 0o644


def find_skills(tracked: dict[str, str]) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Return ({layer: [skill, ...]}, {layer-level file name: its repository path})."""
    membership: dict[str, list[str]] = {layer: [] for layer in LAYERS}
    for path in tracked:
        parts = path.split("/")
        if len(parts) == 4 and parts[0] == "skills" and parts[3] == "SKILL.md":
            if parts[1] not in LAYERS:
                raise PackageError(f"skill outside the known layer folders: {path}")
            membership[parts[1]].append(parts[2])
    owner: dict[str, str] = {}
    for layer, names in membership.items():
        names.sort()
        for name in names:
            if name in owner or name in LAYERS:
                raise PackageError(f"skill name {name!r} is not unique; the package is flat")
            owner[name] = layer
    layer_files: dict[str, str] = {}
    for path in sorted(tracked):
        parts = path.split("/")
        if parts[0] != "skills":
            continue
        if len(parts) == 3 and parts[1] in LAYERS:
            if parts[2] in layer_files or parts[2] == MEMBERSHIP_FILE:
                raise PackageError(f"layer-level file name is not unique: {path}")
            layer_files[parts[2]] = path
        elif len(parts) < 4 or owner.get(parts[2]) != parts[1]:
            raise PackageError(f"file under skills/ that belongs to no skill: {path}")
    return membership, layer_files


def _name_regex(name: str) -> str:
    """One path component of a pattern as a regular expression; "*" stays within the name."""
    return "[^/]*".join(re.escape(piece) for piece in name.split("*"))


def pattern_regex(pattern: str) -> re.Pattern:
    """Compile a SKILL_FILES_LEFT_OUT pattern; see the comment on that table."""
    before = after = ""
    if pattern.startswith("**/"):
        before, pattern = "(?:.*/)?", pattern[3:]
    if pattern.endswith("/**"):
        after, pattern = "/.*", pattern[:-3]
    return re.compile(before + "/".join(_name_regex(name) for name in pattern.split("/")) + after)


def left_out(skill: str, path_in_skill: str) -> str | None:
    """Why a file of a skill is not packaged: the table entry that covers it, or None."""
    if (skill, path_in_skill) in SKILL_FILES_KEPT:
        return None
    if (skill, path_in_skill) in ONE_SKILL_FILES_LEFT_OUT:
        return f"{skill}/{path_in_skill}"
    for pattern in SKILL_FILES_LEFT_OUT:
        if pattern_regex(pattern).fullmatch(path_in_skill):
            return pattern
    return None


def left_out_folders(path_in_skill: str, pattern: str) -> list[str]:
    """The folders above a left-out file that a "/**" pattern removes as a whole.

    Only such a folder counts as left out. The folder of a single left-out file,
    such as config/ with nothing but a scanner baseline in it, does not: text
    that says ``config/`` means something else.
    """
    if not pattern.endswith("/**"):
        return []
    names = path_in_skill.split("/")
    whole = pattern_regex(pattern[:-3])
    for count in range(1, len(names)):
        if whole.fullmatch("/".join(names[:count])):
            return ["/".join(names[:index]) for index in range(count, len(names))]
    return []


# ─────────────────────────────── rewriting ───────────────────────────────
def as_text(data: bytes) -> str | None:
    """Decode a file the rewrite may touch; None for binary content."""
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


# What may stand directly before "skills/<layer>/": the root of a bank, written
# as a variable, a placeholder, a relative step or a clone of the repository.
BANK_ROOT_BEFORE = re.compile(
    r"(?:\$(?P<name>\w+)|\$\{(?P<braced>\w*)[^{}]*\}|\{(?P<field>[^{}\s]*)\}|<[^<>\s]+>"
    r"|(?<![\w.-])tao-skill-bank|(?<![\w.-])\.\.?)"
    r"[\"']?/$"
)
# Variables that name a bank root wherever they stand. Any other variable counts
# as one only on a line that fetches nothing: "$RAW_BASE/skills/models/..." after
# curl is part of a URL, and a URL keeps the repository layout.
BANK_ROOT_VARIABLES = (
    "TAO_SKILL_BANK_PATH", "TAO_SKILL_BANK_ROOT", "SKILL_BANK", "BANK", "BANK_ROOT", "SB",
    "TAO_ROOT",
)
FETCHES = re.compile(r"\bhttps?\b|\bcurl\b|\bwget\b|\bgit\s+show\b")
PATH_CALLS = ("join", "joinpath", "Path", "PurePath", "PurePosixPath", "PosixPath")


def _enclosing_call(text: str, position: int) -> str:
    """The name of the call whose arguments hold ``position``; "[" or "{" for a literal."""
    depth = 0
    for index in range(position - 1, max(position - 400, -1), -1):
        char = text[index]
        if char in ")]}":
            depth += 1
        elif char in "([{":
            if depth:
                depth -= 1
            elif char == "(":
                return re.search(r"(\w*)$", text[:index]).group(1)
            else:
                return char
    return ""


class LayoutRewriter:
    """Turn references to the layer layout into references to the flat one."""

    def __init__(self, membership: dict[str, list[str]], layer_files: dict[str, str]):
        layer = "|".join(LAYERS)
        names = [name for members in membership.values() for name in members]
        skill = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
        quote = "[\"']"
        home = r"(?:~|\$HOME|\$\{HOME\})"
        self.layer_path = re.compile(rf"(?<![\w-])skills/({layer})/")
        # "cd tao-skill-bank": what follows is relative to a clone, which has layer folders.
        self.into_clone = re.compile(
            r"(?m)(?:^|&&|;)[ \t]*(?:[$>][ \t]+)?cd[ \t]+[\"']?(?:[^\s\"';&|]*/)?tao-skill-bank/?"
            r"[\"']?[ \t]*(?:&&|;|$)"
        )
        self.quoted_join = re.compile(
            rf"({quote})skills\1(\s*[/,]\s*)({quote})(?:{layer})\3\s*[/,]\s*"
        )
        self.rules = [
            # A clone of the repository holds the flat bank in DEFAULT_OUTPUT: the
            # default of TAO_SKILL_BANK_PATH, and a path that goes through a clone.
            (re.compile(rf"(\$\{{TAO_SKILL_BANK_PATH:-{home}/tao-skill-bank)\}}"),
             rf"\1/{DEFAULT_OUTPUT}}}"),
            (re.compile(rf"(?<![\w.-])tao-skill-bank/(?=skills/(?:{layer})/)"),
             f"tao-skill-bank/{DEFAULT_OUTPUT}/"),
            # A placeholder for the bank that is named like a clone: the bank is the
            # plugin folder, and the flat paths that follow do not exist in a clone.
            (re.compile(r"(?<![\w.~-])/path/to/tao-skill-bank(?![\w./-])"),
             f"/path/to/{PLUGIN_NAME}"),
            # "skills" / "<layer>" / next  and  "skills", "<layer>", next
            (self.quoted_join, r"\1skills\1\2"),
            # ../../<layer>/<skill> from inside a skill: one level and the layer go
            (re.compile(rf"(?<=\.\./)\.\./(?:{layer})/(?=(?:{skill})(?![\w-]))"), ""),
            # skills/<layer>/<anything>
            (re.compile(rf"(?<![\w-])skills/(?:{layer})/"), "skills/"),
        ]
        # A skill is addressed as <plugin>:<skill>. The skills still say
        # tao-skill-bank:<skill>, the name the plugin had before it was renamed.
        # Only an exact skill name counts, and "tao-skill-bank:*" for all of them:
        # an image tag, a URL and a marketplace reference (<plugin>@tao-skill-bank)
        # stay.
        self.old_namespace = re.compile(rf"(?<![\w.@/-])tao-skill-bank:({skill})(?![\w/-])")
        self.rules.append((self.old_namespace, rf"{PLUGIN_NAME}:\1"))
        self.rules.append((re.compile(r"(?<![\w.@/-])tao-skill-bank:\*(?![\w*/-])"),
                           f"{PLUGIN_NAME}:*"))
        if layer_files:
            files = "|".join(re.escape(name) for name in sorted(layer_files, key=len, reverse=True))
            layer_file = re.compile(rf"(?<![\w-])skills/(?:{layer})/({files})(?![\w-])")
            self.rules.insert(2, (layer_file, CATALOG_DIR + r"/\1"))
        # What must not survive: a layer folder, a quoted layer join, "<layer>/<skill>",
        # a step up into a layer folder, a bare layer folder in backticks, a glob
        # over the two-level layout, a layer folder in a drawing of the tree, and
        # "the tao-skill-bank repo root" as the place to run a command from.
        repo_root = r"repo(?:sitory)?[ \t]+root"
        self.leftover = re.compile(
            rf"(?<![\w-])skills/(?:{layer})(?![\w-])"
            rf"|{quote}skills{quote}\s*[/,]\s*{quote}(?:{layer}){quote}"
            rf"|{quote}(?:{layer}){quote}\s*/\s*{quote}(?:{skill})[\"'/]"
            rf"|(?<![\w-])(?:{layer})/(?:{skill})(?![\w-])"
            rf"|\.\./(?:{layer})/"
            rf"|`(?:{layer})/`"
            rf"|(?<![\w-])skills/\*/\*/"
            rf"|(?:[├└]──|[|+`\\]--)[ \t]*(?:{layer})/(?=[ \t]*(?:#[^\n]*)?$)"
            rf"|`?tao-skill-bank`?[ \t]+{repo_root}|{repo_root}[ \t]+of[ \t]+(?:the[ \t]+)?"
            rf"`?tao-skill-bank\b",
            re.M,
        )

    def refused(self, text: str, path: str = "") -> list[str]:
        """Layer references that a rewrite would turn into something false, with the reason.

        The rewrite is meant for paths inside a bank. A match in a URL, in a git
        ref or a mount target, at the root of a filesystem, in a path under some
        other folder, behind ``cd tao-skill-bank``, in a list of the layer folders
        themselves, in a quoted word list or, in the code file ``path``, in a
        loop over one layer folder is none of those. The build stops, and the
        line gets an entry in PACKAGE_EDITS instead.
        """
        found: list[str] = []
        named_alone: set[str] = set()
        is_code = path.endswith((".py", ".sh"))
        clones = [match.end() for match in self.into_clone.finditer(text)]
        for match in self.layer_path.finditer(text):
            start = match.start()
            while start and not text[start - 1].isspace():
                start -= 1
            before = text[start:match.start()]
            shown = (before + re.match(r"\S*", text[match.start():]).group(0))[:120]
            root = BANK_ROOT_BEFORE.search(before)
            relative = (not before.endswith("/")
                        or bool(root and root.group(0).lstrip("\"'").startswith(".")))
            following = text[match.end():match.end() + 1]
            if "://" in before:
                found.append(f"{shown} (inside a URL)")
            elif before.endswith(":"):
                found.append(f"{shown} (after a colon: a path inside a git ref or a mount "
                             f"target, not in a bank)")
            elif before.endswith("/") and not root:
                if re.search(r"[\w.~@+%-]/$", before):
                    found.append(f"{shown} (a path under another folder, not under a bank root)")
                else:
                    found.append(f"{shown} (a path at the root of a filesystem, not in a bank)")
            elif root and self._fetched(text, match.start(), root):
                found.append(f"{shown} (behind a variable on a line that fetches from a URL or "
                             f"a git ref; only {', '.join(BANK_ROOT_VARIABLES)} count as a "
                             f"bank root there)")
            elif relative and self._in_clone(text, clones, start):
                found.append(f"{shown} (after cd tao-skill-bank: a clone keeps its skills in "
                             f"layer folders)")
            if is_code and following in ("*", '"', "'"):
                found.append(f"{shown} (code that lists one layer folder; in the package it "
                             f"would list every skill)")
            if not re.match(r"[\w<{$*\[.-]", following):
                named_alone.add(match.group(1))
        if len(named_alone) > 1:
            folders = ", ".join(f"skills/{name}/" for name in sorted(named_alone))
            found.append(f"{folders} (a list of layer folders; all of them would read skills/)")
        for match in self.quoted_join.finditer(text):
            joined_by_slash = "/" in match.group(2)
            if not joined_by_slash and _enclosing_call(text, match.start()) not in PATH_CALLS:
                found.append(f"{match.group(0).strip()} (a word list, not a path join)")
        return found

    @staticmethod
    def _fetched(text: str, position: int, root: re.Match) -> bool:
        """Whether the variable before a layer path may be a URL or a git ref."""
        name = root.group("name") or root.group("braced") or root.group("field")
        if name is None or name.split(".")[0].upper() in BANK_ROOT_VARIABLES:
            return False
        line_start = text.rfind("\n", 0, position) + 1
        line_end = text.find("\n", position)
        return bool(FETCHES.search(text[line_start:len(text) if line_end < 0 else line_end]))

    @staticmethod
    def _in_clone(text: str, clones: list[int], position: int) -> bool:
        """Whether ``position`` follows a ``cd tao-skill-bank`` in the same block of commands."""
        for end in clones:
            if end <= position and "```" not in text[end:position]:
                return True
        return False

    def rewrite(self, text: str) -> str:
        for pattern, replacement in self.rules:
            text = pattern.sub(replacement, text)
        return text

    def old_names(self, text: str) -> int:
        """How many skills ``text`` addresses by the plugin's former name."""
        return len(self.old_namespace.findall(text))

    def leftovers(self, text: str) -> list[str]:
        return sorted({match.group(0) for match in self.leftover.finditer(text)})


def climbs_out(path_in_skill: str, text: str) -> set[str]:
    """Lines of skill code that reach skills/ or higher by counting levels from the file.

    A file ``depth`` folders below its skill reaches the skill with ``parents[depth]``
    and the layer folder with one more; anything beyond that lands somewhere else
    once the layer folder is gone.
    """
    depth = len(PurePosixPath(path_in_skill).parts) - 1
    is_shell = path_in_skill.endswith(".sh")
    found: set[str] = set()
    for line in text.splitlines():
        levels = [int(n) for n in re.findall(r"__file__[^\n]*?\.parents\[(\d+)\]", line)]
        chains = re.findall(r"__file__\)(?:\.resolve\(\))?((?:\.parent\b(?!s))+)", line)
        levels += [chain.count(".parent") - 1 for chain in chains]
        from_skill = re.findall(r"\bSKILLS?_(?:ROOT|DIR)\.parents\[(\d+)\]", line)
        ups = [chain.count("..") for chain in re.findall(r"\$\{?\w+\}?((?:/\.\.)+)", line)]
        if (any(n >= depth + 2 for n in levels) or any(int(n) >= 1 for n in from_skill)
                or (is_shell and any(n >= depth + 2 for n in ups))):
            found.add(line.strip())
    return found


def with_folders(files) -> set[str]:
    """The given file paths plus every folder above them."""
    folders = {parent.as_posix() for name in files for parent in PurePosixPath(name).parents}
    return set(files) | folders


def relative_links(path: str, text: str, existing: set[str]) -> list[tuple[str, bool]]:
    """Every relative Markdown link target in ``text`` and whether it names something."""
    links = []
    for target in re.findall(r"\]\(([^)#\s]+)", text):
        if re.match(r"[A-Za-z][\w+.-]*:", target) or target[0] in "/<${":
            continue
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(path), target))
        links.append((target, resolved in existing))
    return links


def newly_broken_links(before: list[tuple[str, bool]], after: list[tuple[str, bool]]) -> list[str]:
    """Links that resolve in the repository and not in the package.

    The two lists are compared link by link. When an edit changed how many links
    a file has, a broken link passes only if the same target is broken in the
    repository as well.
    """
    if len(before) == len(after):
        return [target for (_, was_fine), (target, fine) in zip(before, after)
                if was_fine and not fine]
    already = {target for target, fine in before if not fine}
    return [target for target, fine in after if not fine and target not in already]


def left_out_mentions(path: str, text: str, gone: set[str], shipped: set[str]) -> list[str]:
    """Left-out files and folders (as package paths) that a shipped file names.

    A name counts when it resolves to a left-out file, or to a folder that was
    left out as a whole, from the package root or, for a file of a skill,
    from the skill folder or the folder of the file itself. In a skill, a module
    written with dots (``tests.test_x``), the bare file name of left-out code and
    the bare folder that pytest or ``unittest discover -s`` is pointed at count
    as well.
    """
    parts = path.split("/")
    in_skill = parts[0] == "skills" and len(parts) > 2
    skill_root = "/".join(parts[:2]) if in_skill else ""
    bases = (skill_root, posixpath.dirname(path)) if in_skill else ()
    code_names = {
        posixpath.basename(item): item for item in gone
        if in_skill and item.startswith(skill_root + "/") and item.endswith(CODE_SUFFIXES)
    }
    found: set[str] = set()

    def check(candidate: str) -> None:
        candidate = posixpath.normpath(candidate)
        if candidate in gone and candidate not in shipped:
            found.add(candidate)

    def check_from_skill(name: str) -> None:
        # A name that also exists at the package root (scripts/, templates/) means that one.
        if posixpath.normpath(name) not in shipped:
            for base in bases:
                check(f"{base}/{name}")

    # "pytest tests" and "unittest discover -s tests" name a folder without a slash.
    for name in re.findall(
            r"(?:\bpytest|\bdiscover[ \t]+-s)[ \t]+(?:-[\w-]+[ \t]+)*([\w.@+-]+)(?![\w./@+-])",
            text):
        check_from_skill(name)
    for token in re.findall(r"(?<![\w./@+-])[\w.@+-]+(?:/[\w.@+-]+)*/?", text):
        names = token.rstrip("/").rstrip(".").split("/")
        if len(names) == 1:
            name = names[0]
            if name in code_names:
                found.add(code_names[name])
            if token.endswith("/"):
                check_from_skill(name)
            elif re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", name):
                check_from_skill(name.replace(".", "/") + ".py")
            continue
        for index in range(len(names) - 1):
            tail = "/".join(names[index:])
            if names[index] == "skills":
                check(tail)
            check_from_skill(tail)
    return sorted(found)


# ─────────────────────────────── manifests ───────────────────────────────
def dump_json(value) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def load_json(source: Path, tracked: dict[str, str], path: str):
    """Parse a tracked JSON file; a missing or malformed one is a PackageError."""
    if path not in tracked:
        raise PackageError(f"the source checkout has no {path}")
    try:
        return json.loads(read_file(source, path, tracked[path])[0].decode("utf-8"))
    except ValueError as exc:
        raise PackageError(f"{path} is not valid JSON: {exc}") from exc


def load_manifest(source: Path, tracked: dict[str, str], path: str) -> dict:
    manifest = load_json(source, tracked, path)
    if not isinstance(manifest, dict):
        raise PackageError(f"{path} must hold a JSON object")
    return manifest


def agent_plugins_manifest(claude: dict) -> dict:
    """The root plugin.json: what the Claude manifest holds that Agent Plugins 1.0.0 defines."""
    manifest: dict = {"$schema": AGENT_PLUGINS_SCHEMA}
    for key in AGENT_PLUGINS_FIELDS:
        if key in claude:
            manifest[key] = claude[key]
    author = manifest.get("author")
    if isinstance(author, dict):
        manifest["author"] = {k: v for k, v in author.items() if k in AGENT_PLUGINS_AUTHOR_FIELDS}
    return manifest


def manifest_problems(manifest: dict) -> list[str]:
    """Where the root plugin.json breaks the Agent Plugins 1.0.0 schema."""
    problems = []
    for key, kind in AGENT_PLUGINS_TYPES.items():
        if key in manifest and not isinstance(manifest[key], kind):
            wanted = {str: "a string", dict: "an object", list: "a list"}[kind]
            problems.append(f"plugin.json: {key} must be {wanted} (Agent Plugins 1.0.0)")
    name = manifest.get("name")
    if isinstance(name, str) and not (len(name) <= 64 and AGENT_PLUGINS_NAME.fullmatch(name)):
        problems.append("plugin.json: name must be at most 64 lower-case letters, digits, "
                        "dots and hyphens")
    author, keywords = manifest.get("author"), manifest.get("keywords")
    if isinstance(author, dict) and not all(isinstance(value, str) for value in author.values()):
        problems.append("plugin.json: every author field must be a string")
    if isinstance(keywords, list) and not all(isinstance(item, str) for item in keywords):
        problems.append("plugin.json: every keyword must be a string")
    return problems


# ─────────────────────────────── build ───────────────────────────────
def build(source: Path = REPO_ROOT, *, codex_manifest: bool = True) -> Package:
    """Build the package from a source checkout, in memory. Raises PackageError."""
    source = Path(source).resolve()
    tracked = tracked_files(source)
    membership, layer_files = find_skills(tracked)
    skills = sorted(name for names in membership.values() for name in names)
    rewriter = LayoutRewriter(membership, layer_files)
    problems: list[str] = []

    edits: dict[str, list[tuple[str, str]]] = {}
    for path, old, new in PACKAGE_EDITS:
        edits.setdefault(path, []).append((old, new))
    for path in edits:
        if path not in tracked:
            problems.append(f"PACKAGE_EDITS names a file the source does not have: {path}")

    files: dict[str, tuple[bytes, int]] = {}
    documents: dict[str, tuple[str, str, str]] = {}
    rewritten: list[str] = []
    renamed: dict[str, int] = {}
    gone: set[str] = set()
    counts = {name: 0 for name in SKILL_FILES_LEFT_OUT}

    def add(source_path: str, package_path: str) -> None:
        data, mode = read_file(source, source_path, tracked[source_path])
        text = as_text(data)
        if text is None and source_path.lower().endswith(TEXT_SUFFIXES):
            problems.append(f"{source_path} has a text extension and is not UTF-8 text; the "
                            f"rewrite and the checks cannot read it")
        if text is not None:
            original = text
            for old, new in edits.pop(source_path, []):
                if text.count(old) != 1:
                    problems.append(f"PACKAGE_EDITS: expected one match in {source_path}, "
                                    f"found {text.count(old)}: {old!r}")
                text = text.replace(old, new)
            parts = source_path.split("/")
            if parts[0] == "skills" and source_path.endswith((".py", ".sh")):
                inside = "/".join(parts[3:])
                for line in sorted(climbs_out(inside, original) & climbs_out(inside, text)):
                    if (source_path, line) not in LAYOUT_NEUTRAL:
                        problems.append(f"{source_path} climbs above its skill folder and has no "
                                        f"entry in PACKAGE_EDITS or LAYOUT_NEUTRAL: {line}")
            for found in rewriter.refused(text, source_path):
                problems.append(f"{source_path} has a layer reference the rewrite must not "
                                f"touch; give the line an entry in PACKAGE_EDITS: {found}")
            if rewriter.old_names(text):
                renamed[package_path] = rewriter.old_names(text)
            text = rewriter.rewrite(text)
            for found in rewriter.leftovers(text):
                if (package_path, found) not in LAYER_WORDS_ALLOWED:
                    problems.append(f"{source_path} still refers to the layer layout: {found}")
            if package_path.endswith(".md"):
                documents[package_path] = (source_path, original, text)
            if text != original:
                rewritten.append(package_path)
                data = text.encode("utf-8")
        files[package_path] = (data, mode)

    for path in sorted(tracked):
        parts = path.split("/")
        if parts[0] != "skills":
            continue
        if len(parts) == 3:
            add(path, f"{CATALOG_DIR}/{parts[2]}")
            continue
        inside = "/".join(parts[3:])
        package_path = f"skills/{parts[2]}/{inside}"
        reason = left_out(parts[2], inside)
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
            gone.add(package_path)
            gone.update(f"skills/{parts[2]}/{folder}"
                        for folder in left_out_folders(inside, reason))
            continue
        if TEST_LIKE_NAME.search(inside) and (parts[2], inside) not in SKILL_FILES_KEPT:
            problems.append(f"{path} looks like a test; leave it out through "
                            f"SKILL_FILES_LEFT_OUT or give it a reason in SKILL_FILES_KEPT")
        add(path, package_path)

    for entry in SUPPORT_FILES:
        kept_at = SUPPORT_FILE_SOURCES.get(entry, entry)
        if entry.endswith("/"):
            matches = [path for path in sorted(tracked) if path.startswith(entry)]
        else:
            matches = [kept_at] if kept_at in tracked else []
        if not matches:
            hint = ""
            if entry == "scripts/skill_layers.py":
                hint = "; its helper scripts predate the flat package and cannot run in it"
            problems.append(f"the source checkout has no {kept_at}{hint}")
        for path in matches:
            add(path, entry if path == kept_at != entry else path)

    for path in sorted(edits):
        if path in tracked:
            problems.append(f"PACKAGE_EDITS names a file the package leaves out: {path}")

    # A link that resolves in the repository must resolve in the package too.
    in_package, in_source = with_folders(files), with_folders(tracked)
    for package_path, (source_path, original, text) in documents.items():
        broken = newly_broken_links(relative_links(source_path, original, in_source),
                                    relative_links(package_path, text, in_package))
        if broken:
            problems.append(f"{source_path} has relative links that break in the package: "
                            f"{', '.join(broken)}")

    claude = load_manifest(source, tracked, CLAUDE_MANIFEST)
    claude.pop("skills", None)  # the default, skills/, is what the flat package has
    manifest = agent_plugins_manifest(claude)
    files["plugin.json"] = (dump_json(manifest), 0o644)
    files[CLAUDE_MANIFEST] = (dump_json(claude), 0o644)
    if CURSOR_MANIFEST in tracked:
        cursor = load_manifest(source, tracked, CURSOR_MANIFEST)
        cursor["skills"] = [f"./skills/{name}" for name in skills]
        files[CURSOR_MANIFEST] = (dump_json(cursor), 0o644)
    if codex_manifest:
        files[CODEX_MANIFEST] = (dump_json(load_manifest(source, tracked, CODEX_MANIFEST)), 0o644)
    files[MEMBERSHIP_PATH] = (dump_json({"schema_version": 1, "layers": membership}), 0o644)

    package = Package(
        files=dict(sorted(files.items())), membership=membership, manifest=manifest,
        source=describe_source(source), rewritten=sorted(rewritten), left_out=counts,
        renamed=dict(sorted(renamed.items())),
    )
    repository_scripts = {path for path in tracked if path.startswith("scripts/")}
    problems += check_rules(package, repository_scripts, gone)
    if problems:
        raise PackageError("\n".join(problems))
    return package


def check_suite(package: Package, skills: list[str]) -> list[str]:
    """Problems of the plugin-level evaluation suite: its shape and what it covers."""
    try:
        tasks = json.loads(package.files[EVAL_SUITE][0].decode("utf-8"))
    except ValueError as exc:
        return [f"{EVAL_SUITE} is not valid JSON: {exc}"]
    shaped = isinstance(tasks, list) and all(
        isinstance(task, dict) and isinstance(task.get("id"), str)
        and isinstance(task.get("expected_skill"), str) for task in tasks
    )
    if not shaped:
        return [f"{EVAL_SUITE} must be a JSON array of task objects, each with an id and an "
                f"expected_skill"]
    package.eval_tasks = len(tasks)
    problems = []
    ids = [task["id"] for task in tasks]
    if len(set(ids)) != len(ids):
        problems.append(f"{EVAL_SUITE} has duplicate task ids")
    expected = {task["expected_skill"] for task in tasks}
    for missing in sorted(set(skills) - expected):
        problems.append(f"{EVAL_SUITE} has no task for {missing}")
    for unknown in sorted(expected - set(skills)):
        problems.append(f"{EVAL_SUITE} expects a skill the package lacks: {unknown}")
    return problems


def check_rules(package: Package, repository_scripts: set[str], gone: set[str]) -> list[str]:
    """Problems of a built package: the pipeline's limits and this package's own rules.

    ``gone`` holds the package paths of the skill files that were left out and
    of the folders that were left out as a whole.
    """
    files = package.files
    shipped = with_folders(files)
    problems: list[str] = []
    skills = sorted(name for names in package.membership.values() for name in names)

    if len(files) > MAX_FILES:
        problems.append(f"{len(files)} files; the limit is {MAX_FILES}")
    total = sum(len(data) for data, _ in files.values())
    if total > MAX_TOTAL_BYTES:
        problems.append(f"{total} bytes in total; the limit is {MAX_TOTAL_BYTES}")
    for path, (data, mode) in files.items():
        parts = path.split("/")
        if len(data) > MAX_FILE_BYTES:
            problems.append(f"{path} has {len(data)} bytes; the limit is {MAX_FILE_BYTES}")
        if len(parts) - 1 > MAX_DEPTH:
            problems.append(f"{path} is {len(parts) - 1} folders deep; the limit is {MAX_DEPTH}")
        if mode not in (0o644, 0o755):
            problems.append(f"{path} has mode {mode:o}; only 644 and 755 are allowed")
        if is_junk(path):
            problems.append(f"{path} is a junk file the pipeline rejects")
        if parts[-1] == "SKILL.md" and not (len(parts) == 3 and parts[0] == "skills"):
            problems.append(f"{path}: a SKILL.md is allowed only at skills/<skill>/SKILL.md")
        if parts[0] == "skills" and (len(parts) < 3 or parts[1] not in skills):
            problems.append(f"{path} is not inside a skill folder")
        if is_verification_file(path):
            problems.append(f"{path} is a name reserved for the files verification adds")

    for name in skills:
        skill_md = as_text(files[f"skills/{name}/SKILL.md"][0]) or ""
        frontmatter = re.match(r"---\n(.*?)\n---", skill_md, re.S)
        declared = frontmatter and re.search(
            r"^name:[ \t]*[\"']?([^\"'\n]+?)[\"']?[ \t]*$", frontmatter.group(1), re.M
        )
        if not declared or declared.group(1) != name:
            problems.append(f"skills/{name}/SKILL.md must declare name: {name}")
        if len(skill_md) > SKILL_MD_MAX_CHARS:
            problems.append(f"skills/{name}/SKILL.md has {len(skill_md)} characters in the "
                            f"package; the limit is {SKILL_MD_MAX_CHARS}")

    name, version = package.manifest.get("name"), package.manifest.get("version")
    if name != PLUGIN_NAME or not version:
        problems.append(f"plugin.json must carry the name {PLUGIN_NAME} and a version")
    problems += manifest_problems(package.manifest)
    for path in (CLAUDE_MANIFEST, CODEX_MANIFEST, CURSOR_MANIFEST):
        if path in files:
            other = json.loads(files[path][0].decode("utf-8"))
            if (other.get("name"), other.get("version")) != (name, version):
                problems.append(f"{path} disagrees with plugin.json on name or version")
    if "skills" in json.loads(files[CLAUDE_MANIFEST][0].decode("utf-8")):
        problems.append(f"{CLAUDE_MANIFEST} must not carry a skills key")

    if EVAL_SUITE in files:
        problems += check_suite(package, skills)

    # A shipped file must not send its reader to a file the package lacks.
    mention = re.compile(r"(?<![\w-])scripts/(?:[\w.-]+/)*[\w.-]+\.(?:py|sh|mjs)\b")
    fixed_names = [name for name in SKILL_FILES_LEFT_OUT if "/" not in name]
    fixed_names += [PurePosixPath(name).name for name in SKILL_FILES_LEFT_OUT
                    if "/" in name and not name.startswith("**/") and not name.endswith("/**")]
    by_name = re.compile(r"(?<![\w-])(?:%s)(?![\w-])" % "|".join(
        _name_regex(name).replace("[^/]*", r"[\w.-]*") for name in sorted(fixed_names)
    ))
    for path, (data, _) in files.items():
        text = as_text(data)
        if text is None:
            continue
        parts = path.split("/")
        own_folder = f"skills/{parts[1]}/" if parts[0] == "skills" else None
        for helper in sorted(set(mention.findall(text))):
            if (helper in repository_scripts and helper not in files
                    and (path, helper) not in UNSHIPPED_SCRIPT_MENTIONS
                    and not (own_folder and own_folder + helper in shipped)):
                problems.append(f"{path} names {helper}, which the package leaves out; add it to "
                                f"SUPPORT_FILES, rewrite the text in PACKAGE_EDITS or give the "
                                f"mention a reason in UNSHIPPED_SCRIPT_MENTIONS")
        named = set(left_out_mentions(path, text, gone, shipped))
        if own_folder:
            for found in set(by_name.findall(text)):
                by_path = any(item.endswith("/" + found) for item in named)
                if not by_path and own_folder + found not in shipped:
                    named.add(found)
        for found in sorted(named):
            if (path, found) not in LEFT_OUT_MENTIONS:
                problems.append(f"{path} names {found}, which the package leaves out; rewrite "
                                f"the text in PACKAGE_EDITS, keep the file through "
                                f"SKILL_FILES_KEPT or give the mention a reason in "
                                f"LEFT_OUT_MENTIONS")
    return problems


# ─────────────────────────────── output ───────────────────────────────
def is_signature(path: str) -> bool:
    """Whether ``path`` is a signature file, wherever it sits in the folder."""
    return path.rsplit("/", 1)[-1] in SIGNATURE_FILES


def is_verification_file(path: str) -> bool:
    """Whether verification or evaluation, not this script, puts ``path`` into the folder."""
    if path.startswith(VERIFICATION_DIRS) or is_signature(path):
        return True
    parts = path.split("/")
    if len(parts) == 1:
        names = VERIFICATION_ROOT_FILES
    elif len(parts) == 3 and parts[0] == "skills":
        names = VERIFICATION_SKILL_FILES
    else:
        return False
    return parts[-1] in names or any(
        fnmatch.fnmatchcase(parts[-1], pattern) for pattern in VERIFICATION_PATTERNS
    )


def is_junk(path: str) -> bool:
    """Whether the pipeline rejects ``path`` for its name."""
    return any(part in JUNK_NAMES or part.endswith(JUNK_SUFFIXES) for part in path.split("/"))


def read_folder(folder: Path) -> dict[str, tuple[bytes, int] | None]:
    """Files of an existing package folder; None marks what is not a regular file.

    ``.git`` is skipped at every level: a folder that is a work tree of its own
    keeps its history, whatever a build does to the files beside it.
    """
    found: dict[str, tuple[bytes, int] | None] = {}
    for parent, directories, names in os.walk(folder):
        directories[:] = sorted(name for name in directories if name != ".git")
        links = [name for name in directories if (Path(parent) / name).is_symlink()]
        for name in sorted(names + links):
            if name == ".git":
                continue
            path = Path(parent) / name
            relative = path.relative_to(folder).as_posix()
            if path.is_symlink() or not path.is_file():
                found[relative] = None
            else:
                found[relative] = (path.read_bytes(), path.stat().st_mode & 0o777)
    return found


def inspect(package: Package, folder: Path) -> Found:
    """Sort what an existing folder holds into what the build, verification and nobody wrote."""
    on_disk = read_folder(folder) if folder.is_dir() else {}
    found = Found()
    for path in sorted(set(on_disk) - set(package.files)):
        if is_verification_file(path):
            found.added.append(path)
        elif is_junk(path):
            found.junk.append(path)
        else:
            found.unknown.append(path)
    for path, (data, mode) in package.files.items():
        if path not in on_disk:
            found.differences.append(f"missing: {path}")
            continue
        entry = on_disk[path]
        if entry is None:
            found.differences.append(f"not a regular file: {path}")
        elif entry[0] != data:
            found.differences.append(f"content differs: {path}")
        elif entry[1] != mode:
            found.differences.append(f"mode differs: {path} is {entry[1]:o}, built {mode:o}")
    return found


def git_ignored(folder: Path, paths: list[str]) -> set[str]:
    """Those of ``paths`` below ``folder`` that git ignores; none outside a work tree."""
    if not paths:
        return set()
    try:
        result = subprocess.run(
            ["git", "-C", str(folder), "check-ignore", "-z", "--stdin"],
            input="\0".join(paths).encode(), capture_output=True, check=False,
        )
    except OSError:
        return set()
    if result.returncode != 0:  # 1: nothing is ignored, 128: not in a work tree
        return set()
    return {item.decode() for item in result.stdout.split(b"\0") if item}


def check(package: Package, folder: Path) -> tuple[list[str], list[str]]:
    """(differences between a folder and the build, junk in it that git ignores).

    Files that verification adds are not differences. Junk is reported under
    its own label, and junk that git ignores is no difference either: it cannot
    be committed, so it reaches neither a signature nor the catalog.
    """
    if not (folder / MEMBERSHIP_PATH).is_file():
        return [f"not a package this script built (no {MEMBERSHIP_PATH}): {folder}"], []
    found = inspect(package, folder)
    ignored = git_ignored(folder, found.junk)
    lines = found.differences + [f"not generated: {path}" for path in found.unknown]
    lines += [f"junk: {path}" for path in found.junk if path not in ignored]
    return lines, sorted(ignored)


def compare(package: Package, folder: Path) -> list[str]:
    """Differences between a folder and the build; see check()."""
    return check(package, folder)[0]


def _remove(folder: Path, path: str) -> None:
    """Delete one file and the folders it leaves empty, up to but not including ``folder``."""
    target = folder / path
    target.unlink()
    for parent in target.parents:
        if parent == folder or folder not in parent.parents or os.listdir(parent):
            break
        parent.rmdir()


def write(package: Package, folder: Path, *, force: bool = False, source: Path | None = None,
          announce=None) -> Written:
    """Make ``folder`` equal to the build, or raise PackageError and leave it untouched.

    The folder may hold more than the build wrote, and it may be signed:

    * verification output (is_verification_file) stays where it is;
    * junk the pipeline forbids is removed;
    * any other unknown file stops the build unless ``force`` is set, and is
      removed when it is;
    * a signature stops the build when anything but junk would change, unless
      ``force`` is set. A forced rebuild that changes a packaged file or removes
      an unknown one removes the signatures, which no longer describe the folder.

    ``announce`` is called with the result before the first file is removed, so
    the caller can list what goes. A folder that already equals the build is
    not written at all. ``.git`` is never touched, and a folder that is the root
    of a git work tree is refused unless it lies inside ``source``.
    """
    if folder.is_symlink() or (folder.exists() and not folder.is_dir()):
        raise PackageError(f"{folder} is not a folder")
    inside_source = source is not None and Path(source).resolve() in folder.resolve().parents
    if (folder / ".git").exists() and not inside_source:
        raise PackageError(f"{folder} is the root of a git work tree. The package is written "
                           f"into a folder of its own, not over a checkout; choose a folder "
                           f"inside it.")
    if folder.is_dir() and any(folder.iterdir()) and not (folder / MEMBERSHIP_PATH).is_file():
        raise PackageError(f"{folder} is not empty and is not a package this script built; "
                           f"refusing to replace it")
    found = inspect(package, folder)
    done = Written(changed=len(found.differences), junk=found.junk, kept=found.added)
    if not (found.differences or found.unknown or found.junk):
        return done
    signatures = found.signatures
    if signatures and (found.differences or found.unknown) and not force:
        # What --force would delete comes first, so that the short list always shows it.
        would = [f"not generated: {path}" for path in found.unknown] + found.differences
        more = f"\n  ... and {len(would) - 10} more" if len(would) > 10 else ""
        deleted = (f"the {len(found.unknown)} file(s) listed as not generated are deleted, "
                   if found.unknown else "")
        raise PackageError(
            f"{folder} is signed ({', '.join(signatures)}), and a rebuild would change it:\n  "
            + "\n  ".join(would[:10]) + more
            + "\nNothing was changed. Run with --check to see every difference. Pass --force "
              f"to rebuild anyway: {deleted}the signature files are removed, and the folder "
              "has to be verified and signed again."
        )
    if found.unknown and not force:
        raise PackageError(
            f"{folder} holds {len(found.unknown)} file(s) that no build writes and that "
            f"verification is not known to add:\n  " + "\n  ".join(found.unknown)
            + "\nNothing was changed. Move them out of the folder, or pass --force to have "
              "them removed."
        )
    done.unknown = found.unknown
    # A signature lists every file of the folder: it is stale once a packaged file
    # changes or an unknown file goes. Junk alone leaves it valid, because the
    # pipeline signs no folder that holds junk.
    done.stale = signatures if (found.differences or found.unknown) else []
    done.kept = [path for path in found.added if path not in done.stale]
    if announce:
        announce(done)
    for path, _reason in done.removed:
        _remove(folder, path)
    folder.mkdir(parents=True, exist_ok=True)
    folder.chmod(0o755)
    # Folders get mode 755 whatever the umask of the caller is, as files get 644 or 755.
    for relative in sorted(with_folders(package.files) - set(package.files) - {"."}):
        (folder / relative).mkdir(exist_ok=True)
        (folder / relative).chmod(0o755)
    for path, (data, mode) in package.files.items():
        target = folder / path
        if target.is_symlink():
            target.unlink()
        target.write_bytes(data)
        target.chmod(mode)
    differences = compare(package, folder)
    if differences:
        raise PackageError(f"{folder} does not match the build after writing:\n"
                           + "\n".join(differences[:20]))
    return done


def summary(package: Package) -> str:
    files = package.files
    megabytes = sum(len(data) for data, _ in files.values()) / 1048576
    largest = max(len(data) for data, _ in files.values()) / 1048576
    layers = ", ".join(f"{layer} {len(names)}" for layer, names in package.membership.items())
    skills = sum(len(names) for names in package.membership.values())
    in_skills = [path for path in package.rewritten if path.startswith("skills/")]
    left = ", ".join(f"{count} {name}" for name, count in package.left_out.items() if count)
    clients = (CLAUDE_MANIFEST, CURSOR_MANIFEST, CODEX_MANIFEST)
    manifests = [path for path in clients if path in files]
    return "\n".join([
        f"{package.manifest.get('name')} {package.manifest.get('version')}",
        f"  source      {package.source}",
        f"  skills      {skills} ({layers})",
        f"  files       {len(files)} of {MAX_FILES} allowed, {megabytes:.1f} MiB of "
        f"{MAX_TOTAL_BYTES // 1048576} MiB, largest {largest:.1f} MiB, "
        f"deepest {max(path.count('/') for path in files)} folders",
        f"  rewritten   {len(package.rewritten)} files for the package: {len(in_skills)} in "
        f"{len({path.split('/')[1] for path in in_skills})} skills, "
        f"{len(package.rewritten) - len(in_skills)} outside skills/; "
        f"{len(PACKAGE_EDITS)} hand-written edits",
        f"  skill names {sum(package.renamed.values())} references in {len(package.renamed)} "
        f"files now read {PLUGIN_NAME}:<skill>",
        f"  left out    from skill folders: {left or 'nothing'}",
        f"  eval suite  {package.eval_tasks} tasks in {EVAL_SUITE}",
        f"  manifests   plugin.json, {', '.join(manifests)}",
    ])


def _fail(headline: str, details: str) -> int:
    print(f"error: {headline}", file=sys.stderr)
    for line in details.splitlines():
        print(f"  {line}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, default=REPO_ROOT,
                        help="git checkout to package (default: the repository of this script)")
    parser.add_argument("--output", type=Path, default=None,
                        help=f"package folder to write or check (default: {DEFAULT_OUTPUT} in "
                             f"the source checkout)")
    parser.add_argument("--check", action="store_true",
                        help="build in memory and compare with the existing folder; exit 1 and "
                             "list the differences when they differ")
    parser.add_argument("--no-codex-manifest", action="store_true",
                        help=f"leave {CODEX_MANIFEST} out of the package")
    parser.add_argument("--force", action="store_true",
                        help="rebuild although the folder is signed or holds files that no "
                             "build writes: those files are removed, and so are the signature "
                             "files when a packaged file changes or such a file is removed")
    args = parser.parse_args(argv)

    source = args.source.resolve()
    if args.output is None:
        output = (source / DEFAULT_OUTPUT).resolve()
        if source not in output.parents:
            return _fail(f"{source / DEFAULT_OUTPUT} resolves outside the source checkout",
                         "pass --output to write there")
    else:
        output = args.output.resolve()

    try:
        package = build(source, codex_manifest=not args.no_codex_manifest)
    except PackageError as exc:
        return _fail(f"the package cannot be built from {source}:", str(exc))
    except (OSError, ValueError) as exc:
        return _fail(f"the package cannot be built from {source}:",
                     f"{type(exc).__name__}: {exc}")

    if args.check:
        try:
            differences, ignored = check(package, output)
        except OSError as exc:
            return _fail(f"{output} cannot be read:", f"{type(exc).__name__}: {exc}")
        if differences:
            print(f"{output} differs from a build of {package.source}:", file=sys.stderr)
            for line in differences[:40]:
                print(f"  {line}", file=sys.stderr)
            if len(differences) > 40:
                print(f"  ... and {len(differences) - 40} more", file=sys.stderr)
            print("Rebuild it with scripts/build_plugin_package.py; never edit the folder by "
                  "hand.", file=sys.stderr)
            return 1
        print(summary(package))
        print(f"up to date: {output}")
        if ignored:
            print(f"note: {len(ignored)} junk file(s) that git ignores are not counted "
                  f"({', '.join(ignored[:4])}{', ...' if len(ignored) > 4 else ''}); a rebuild "
                  f"removes them")
        return 0

    def announce(done: Written) -> None:
        if done.removed:
            print(f"removing {len(done.removed)} file(s) from {output}:")
            for path, reason in done.removed:
                print(f"  {path}  ({reason})")

    try:
        done = write(package, output, force=args.force, source=source, announce=announce)
    except PackageError as exc:
        return _fail("the package was not written:", str(exc))
    except OSError as exc:
        return _fail(f"{output} was not written completely:", f"{type(exc).__name__}: {exc}")
    print(summary(package))
    if done.changed:
        print(f"written: {output}")
    elif done.removed:
        print(f"cleaned: {output} (no packaged file changed)")
    else:
        print(f"up to date: {output}")
    if done.stale:
        print(f"removed {len(done.stale)} signature file(s) that the rebuild made stale "
              f"({', '.join(done.stale)}); the folder has to be verified and signed again")
    if done.kept and (done.changed or done.stale):
        print(f"note: left {len(done.kept)} file(s) that verification added "
              f"({', '.join(done.kept[:6])}{', ...' if len(done.kept) > 6 else ''}); cards, "
              f"benchmark and evaluation results describe the earlier content")
    return 0


if __name__ == "__main__":
    sys.exit(main())
