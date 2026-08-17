#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Screen which execution paths of a TAO skill could run with no network.

This is one half of a two-source verdict. It is deliberately not the whole
answer, and it is written so that it cannot pretend to be.

Air-gap compatibility is not a property of a skill. It is a property of a
(skill, execution path) pair: a skill reachable by docker, kubernetes, slurm and
brev is packageable via docker even though its brev path can never work offline.
A skill is unpackageable only when every one of its paths is.

Why a screen rather than a decision. The bank declares no machine-readable
platform list -- two skills carry ``features:``, fifteen name a platform in
prose, and only fifty-four of seventy-seven have a ``skill_info.yaml`` at all. A
pattern scan therefore fails in both directions, and both failures were observed
while writing this: matching the word "serve" disqualified two model skills over
"rebuild the engine for each (H, W) you serve", and requiring a declared
``container_image`` disqualified two data skills whose image is written in the
body of their SKILL.md. One error hides a workload the customer could have had;
the other ships a bundle that cannot run.

So this script reports three verdicts, not two: ``eligible``, ``ineligible``,
and ``undetermined``. It returns ``ineligible`` only on positive evidence, and
``undetermined`` whenever the tree is silent -- never a confident "no" from an
absent declaration. Everything it marks ``undetermined`` or ``presumed`` is
handed to the agent to adjudicate against the skill's own documentation, and a
disagreement between the two is a hard stop for a human rather than something
either side resolves alone.

What the code is for: reproducibility (the same tree yields the same verdict,
diffable across TAO releases), mechanical provenance, breadth, and the ability
to run in a pipeline with no model available.

On provenance: a verdict drawn from a line cites ``path:line``; a verdict drawn
from *absence* cites the file that was read and no line, because no line says a
thing is missing. A citation without a line is a place to look, not a fact
already established, and the two are not interchangeable.

Usage:
    probe_airgap_eligibility.py --skill tao-train-dinov3 --format json
    probe_airgap_eligibility.py --all --format text
    probe_airgap_eligibility.py --all --needs-review
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

# The bank this script is part of, so resolution works from a git clone, a
# plugin install, or anywhere else the tree is unpacked. $TAO_SKILL_BANK_PATH
# overrides it for the case where the caller means a different bank.
_BUNDLED_SKILL_BANK = Path(__file__).resolve().parents[4]
DEFAULT_SKILL_BANK = Path(os.environ.get("TAO_SKILL_BANK_PATH") or _BUNDLED_SKILL_BANK)

SCHEMA_VERSION = 1
UNMATCHED_EXIT = 3

ELIGIBLE = "eligible"
INELIGIBLE = "ineligible"
UNDETERMINED = "undetermined"

LAYERS = ("applications", "core", "data", "models", "platform")

# Layers that never describe a customer workload. A platform skill is how a job
# runs, a core skill routes and answers questions; neither is something to
# package. Reported as out-of-scope rather than ineligible, because "ineligible"
# would imply someone might have wanted it.
NON_WORKLOAD_LAYERS = ("core", "platform")

# Only authored files are evidence. BENCHMARK.md and skill-card.md are written
# by the signing pipeline and paraphrase the skill, so a verdict citing one
# reports a machine's summary as though a maintainer had written it.
GENERATED_FILES = ("BENCHMARK.md", "skill-card.md")

# A control-plane platform has no offline form: provisioning is an API call to a
# service that cannot be carried across the gap. Every other platform runs on
# hardware the customer already owns, so its compute side can sit in an enclave.
CONTROL_PLANE_PLATFORMS = ("brev",)

# E-SVC reads declared surfaces only -- the frontmatter description and the
# action commands in skill_info.yaml -- for the reason given in the docstring.
SERVICE_DESCRIPTION_MARKERS = (
    (r"inference[- ]?(micro)?service", "the deliverable is a served inference endpoint"),
    (r"model serving|serving service|serving stack", "the deliverable is a serving stack"),
)
SERVICE_COMMAND_MARKERS = (
    (r"(^|\s)serve(\s|$)", "a declared action starts a server"),
    (r"--port(\s|=)|--host(\s|=)", "a declared action binds a network socket"),
)

# A skill has a virtualenv path only when it documents one. Absence of a
# container image does not imply it -- an orchestrating skill has neither.
VIRTUALENV_MARKERS = (
    r"tao-run-on-virtualenv",
    r"uv run|uv sync|python -m venv",
)

# Fetches that happen while the run is in flight. These never remove a path --
# they are the asset-closure problem -- but they must be surfaced, because a
# path that looks clean and downloads a checkpoint on first use is the exact
# failure this skill exists to prevent.
RUNTIME_FETCH_MARKERS = (
    r"from_pretrained\(",
    r"hf_hub_download\(",
    r"snapshot_download\(",
    r"huggingface-cli\s+download",
    r"\bhf\s+download\b",
    r"\bwget\b|\bcurl\b",
    r"git\s+clone",
)

# An image URI written into the body rather than declared in skill_info.yaml.
# Matching the registry host keeps this independent of which org is pinned.
IMAGE_IN_BODY = r"nvcr\.io/[A-Za-z0-9._/-]+:[A-Za-z0-9._-]+"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--skill", help="Skill name, e.g. tao-train-dinov3.")
    target.add_argument(
        "--all", action="store_true", help="Screen every skill in the bank."
    )
    parser.add_argument(
        "--skill-bank",
        type=Path,
        default=DEFAULT_SKILL_BANK,
        help="Path to the packaged TAO skill bank.",
    )
    parser.add_argument(
        "--needs-review",
        action="store_true",
        help="List only what the agent must adjudicate, and why.",
    )
    parser.add_argument(
        "--format", choices=("json", "text"), default="text", help="Output format."
    )
    return parser.parse_args(argv)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML object, returning an empty dict when the file is absent."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def find_skill_dir(bank: Path, name: str) -> Path | None:
    """Locate a skill directory by name across every layer."""
    for layer in LAYERS:
        candidate = bank / "skills" / layer / name
        if (candidate / "SKILL.md").exists():
            return candidate
    return None


def platform_names(bank: Path) -> list[str]:
    """Discover execution platforms by globbing the installed platform skills.

    There is no central platform registry -- a platform skill present in the
    tree self-registers. Hardcoding the list would make a platform added in a
    later release invisible, which is the failure this scan exists to avoid.
    """
    return sorted(
        path.parent.name.removeprefix("tao-run-on-")
        for path in bank.glob("skills/platform/tao-run-on-*/SKILL.md")
    )


def cite(path: Path, bank: Path, pattern: str) -> str | None:
    """Return ``relative/path:line`` for the first line matching ``pattern``."""
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    rel = path.relative_to(bank)
    for number, line in enumerate(text.splitlines(), start=1):
        if re.search(pattern, line, re.IGNORECASE):
            return f"{rel}:{number}"
    return None


def evidence_files(skill_dir: Path) -> list[Path]:
    """Return the authored files under a skill, in a stable order."""
    return [
        path
        for path in sorted(skill_dir.rglob("*"))
        if path.is_file()
        and path.suffix in (".md", ".yaml", ".yml", ".py")
        and path.name not in GENERATED_FILES
        and "evals" not in path.relative_to(skill_dir).parts
    ]


def version_keys(bank: Path) -> dict[str, str]:
    """Flatten versions.yaml into dotted key to image URI.

    Both the full key and the key with the leading ``images.`` dropped are
    indexed, because skills cite it both ways.
    """
    data = load_yaml(bank / "versions.yaml").get("images")
    if not isinstance(data, dict):
        return {}
    flat: dict[str, str] = {}
    for group, entries in data.items():
        if not isinstance(entries, dict):
            continue
        for leaf, uri in entries.items():
            if isinstance(uri, str) and uri.strip():
                flat[f"images.{group}.{leaf}"] = uri.strip()
                flat[f"{group}.{leaf}"] = uri.strip()
    return flat


def find_image(skill_dir: Path, bank: Path, info: dict[str, Any]) -> dict[str, Any]:
    """Locate the skill's container image in any of the three forms it takes.

    Skills name an image three ways: declared in ``skill_info.yaml``, written as
    a literal URI in the body, or cited as a ``versions.yaml`` key. Only the
    first is a contract, and only forty-three of seventy-seven skills declare a
    usable image. Five name a single URI in the body and three cite a key;
    treating those eight as "no image" marked them unpackageable. Six more name
    several distinct images and are reported ambiguous rather than guessed at.
    """
    declared = info.get("container_image")
    if isinstance(declared, str) and declared.strip():
        return {
            "image": declared.strip(),
            "source": cite(skill_dir / "references" / "skill_info.yaml", bank, r"^container_image:"),
            "form": "declared",
        }

    # Same rule as the versions-key branch below, and for the same reason: a
    # skill that writes several distinct URIs into its body has not told us
    # which one its actions run. Returning the first match found is a confident
    # wrong answer, and three skills name four, four and two URIs respectively.
    files = evidence_files(skill_dir)
    body: dict[str, str] = {}
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for uri in re.findall(IMAGE_IN_BODY, text):
            body.setdefault(uri, cite(path, bank, re.escape(uri)) or str(path.relative_to(bank)))
    if len(body) == 1:
        uri, source = next(iter(body.items()))
        return {"image": uri, "source": source, "form": "body-uri"}
    if len(body) > 1:
        return {
            "image": None,
            "source": None,
            "form": "ambiguous",
            "candidates": sorted(body),
        }

    # A versions.yaml key is only the skill's own image when the skill mentions
    # exactly one. An orchestrating skill names the key of every image it
    # coordinates; picking one of those is a confident wrong answer, and it is
    # worse than no answer because nothing downstream questions it.
    keys = version_keys(bank)
    matched: dict[str, str] = {}
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for key, uri in keys.items():
            if key in text and uri not in matched:
                matched[uri] = cite(path, bank, re.escape(key)) or str(path.relative_to(bank))
    if len(matched) == 1:
        uri, source = next(iter(matched.items()))
        return {"image": uri, "source": source, "form": "versions-key"}
    if len(matched) > 1:
        return {
            "image": None,
            "source": None,
            "form": "ambiguous",
            "candidates": sorted(matched),
        }
    return {"image": None, "source": None, "form": "none"}


def frontmatter_description(skill_md: Path) -> tuple[str, int]:
    """Return the frontmatter description and the line it starts on.

    Matching the whole body instead of this field is how an earlier version
    disqualified three data skills: each names another skill in a sentence about
    delegating to it, and the phrase that names that skill also matches a
    service marker. A cross-reference is not a dependency.
    """
    if not skill_md.exists():
        return "", 0
    lines = skill_md.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or lines[0].strip() != "---":
        return "", 0
    end = next((i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---"), None)
    if end is None:
        return "", 0
    try:
        block = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError:
        return "", 0
    description = (block or {}).get("description") if isinstance(block, dict) else None
    if not isinstance(description, str):
        return "", 0
    start = next(
        (i for i, line in enumerate(lines[1:end], start=2) if line.startswith("description:")),
        1,
    )
    return description, start


def service_evidence(
    skill_dir: Path, bank: Path, info: dict[str, Any]
) -> list[dict[str, str]]:
    """Return E-SVC findings drawn from declared surfaces only.

    "Declared surfaces" means the frontmatter description value and the action
    commands -- not the body. The body is where a skill explains how it hands
    off to other skills, and naming one is not depending on one.
    """
    hits: list[dict[str, str]] = []
    skill_md = skill_dir / "SKILL.md"
    description, line = frontmatter_description(skill_md)
    for pattern, meaning in SERVICE_DESCRIPTION_MARKERS:
        if description and re.search(pattern, description, re.IGNORECASE):
            hits.append(
                {
                    "finding": meaning,
                    "source": f"{skill_md.relative_to(bank)}:{line}",
                }
            )
            break

    info_path = skill_dir / "references" / "skill_info.yaml"
    actions = info.get("actions")
    if isinstance(actions, dict):
        for name, action in actions.items():
            command = (action or {}).get("command") if isinstance(action, dict) else None
            if not isinstance(command, str):
                continue
            for pattern, meaning in SERVICE_COMMAND_MARKERS:
                if re.search(pattern, command):
                    hits.append(
                        {
                            "finding": f"{meaning} (action '{name}')",
                            "source": cite(info_path, bank, re.escape(command.split()[0]))
                            or str(info_path.relative_to(bank)),
                        }
                    )
                    break
    return hits


def has_virtualenv_path(skill_dir: Path, bank: Path) -> str | None:
    """Return provenance for a documented non-container Python path, if any."""
    for pattern in VIRTUALENV_MARKERS:
        source = cite(skill_dir / "SKILL.md", bank, pattern)
        if source:
            return source
    return None


def scan_runtime_fetches(skill_dir: Path, bank: Path) -> list[str]:
    """Return provenance for run-time asset fetches found under a skill."""
    files = evidence_files(skill_dir)
    found: list[str] = []
    for pattern in RUNTIME_FETCH_MARKERS:
        for path in files:
            source = cite(path, bank, pattern)
            if source:
                found.append(source)
                break
    return found


def probe_skill(bank: Path, name: str) -> dict[str, Any]:
    """Return the mechanical per-path screen for one skill."""
    skill_dir = find_skill_dir(bank, name)
    if skill_dir is None:
        raise KeyError(f"skill '{name}' not found under {bank / 'skills'}")

    info = load_yaml(skill_dir / "references" / "skill_info.yaml")
    skill_md = skill_dir / "SKILL.md"
    rel_dir = skill_dir.relative_to(bank)
    layer = rel_dir.parts[1]

    image = find_image(skill_dir, bank, info)
    service_hits = service_evidence(skill_dir, bank, info)
    venv_source = has_virtualenv_path(skill_dir, bank)

    named = {
        platform: cite(skill_md, bank, rf"tao-run-on-{platform}\b")
        for platform in platform_names(bank)
    }
    named_platforms = {p: src for p, src in named.items() if src}
    non_control = [p for p in named_platforms if p not in CONTROL_PLANE_PLATFORMS]
    control_only = bool(named_platforms) and not non_control

    # An orchestrating skill declares no image of its own: its assets are the
    # union of the underlying skills it runs. Deciding it here would report the
    # bank's most useful workloads as unpackageable.
    requires_expansion = (
        not image["image"] and layer == "applications" and not service_hits
    )
    out_of_scope = layer in NON_WORKLOAD_LAYERS

    paths: dict[str, dict[str, Any]] = {}
    for platform in platform_names(bank):
        if out_of_scope:
            paths[platform] = {
                "verdict": INELIGIBLE,
                "basis": "declared",
                "reason": f"a {layer} skill is not a customer workload to package",
                "source": f"{rel_dir}/SKILL.md",
            }
        elif platform in CONTROL_PLANE_PLATFORMS:
            # The verdict is a fact about the platform, so it holds whether or
            # not this skill mentions it -- but calling that "declared" when the
            # skill never named the platform attributes it to a citation that
            # does not exist.
            paths[platform] = {
                "verdict": INELIGIBLE,
                "basis": "declared" if platform in named_platforms else "presumed",
                "reason": (
                    "provisioning is a call to the platform's control plane; no "
                    "offline form of that exists"
                ),
                "source": named_platforms.get(platform)
                or f"skills/platform/tao-run-on-{platform}/SKILL.md",
            }
        elif service_hits:
            paths[platform] = {
                "verdict": INELIGIBLE,
                "basis": "evidence",
                "reason": f"E-SVC: {service_hits[0]['finding']}",
                "source": service_hits[0]["source"],
            }
        elif control_only:
            paths[platform] = {
                "verdict": UNDETERMINED,
                "basis": "absent-declaration",
                "reason": (
                    "the skill names only a control-plane platform; whether an "
                    "on-premises path exists is not stated either way"
                ),
                "source": next(iter(named_platforms.values()), None),
            }
        elif platform == "virtualenv":
            # Three cases, and collapsing the last two is how this branch once
            # removed the platform from most of the bank without review. A
            # declared image is positive evidence the path is containerised; no
            # image and no documented Python path is silence, and silence is
            # never a refusal.
            if venv_source:
                paths[platform] = {
                    "verdict": ELIGIBLE,
                    "basis": "declared",
                    "reason": (
                        "documents a non-container Python path, so its wheel "
                        "closure is staged instead of an image"
                    ),
                    "source": venv_source,
                }
            elif image["image"]:
                paths[platform] = {
                    "verdict": INELIGIBLE,
                    "basis": "evidence",
                    "reason": (
                        "runs in a container and documents no non-container "
                        "path; say so if the destination wants one anyway"
                    ),
                    "source": image["source"] or f"{rel_dir}/SKILL.md",
                }
            else:
                paths[platform] = {
                    "verdict": UNDETERMINED,
                    "basis": "absent-declaration",
                    "reason": "neither a container image nor a Python path is documented",
                    "source": f"{rel_dir}/SKILL.md",
                }
        elif requires_expansion:
            paths[platform] = {
                "verdict": UNDETERMINED,
                "basis": "absent-declaration",
                "reason": (
                    "orchestrating skill: its assets are the union of the "
                    "underlying skills it runs, so the verdict follows expansion"
                ),
                "source": f"{rel_dir}/SKILL.md",
            }
        elif not image["image"]:
            paths[platform] = {
                "verdict": UNDETERMINED,
                "basis": "absent-declaration",
                "reason": (
                    "no container image found in skill_info.yaml or the body; "
                    "absent contract is not absent workload"
                ),
                "source": f"{rel_dir}/SKILL.md",
            }
        else:
            paths[platform] = {
                "verdict": ELIGIBLE,
                "basis": "declared" if platform in named_platforms else "presumed",
                "reason": (
                    "carries a container image and shows no live-service "
                    "dependency"
                ),
                "source": named_platforms.get(platform) or image["source"],
            }

    review: list[str] = []
    for platform, verdict in sorted(paths.items()):
        # A control-plane platform's verdict is a fact about that platform, so
        # it holds whether or not this skill mentions it. Its basis is
        # "presumed" only in the sense that no citation exists here -- asking
        # the agent to adjudicate it is noise, and the generic presumption
        # message would explain it wrongly.
        if platform in CONTROL_PLANE_PLATFORMS:
            continue
        if verdict["verdict"] == UNDETERMINED:
            review.append(f"{platform}: {verdict['reason']}")
        elif verdict["basis"] == "presumed":
            review.append(
                f"{platform}: presumed from the container-image default, not "
                f"stated by the skill"
            )
    for hit in service_hits:
        review.append(
            f"E-SVC removed every path on the strength of one line "
            f"({hit['source']}): confirm the skill's deliverable really is a "
            f"live service before dropping the workload"
        )
    if image["form"] == "ambiguous":
        review.append(
            "several versions.yaml images are named and none is declared as this "
            f"skill's own ({', '.join(image.get('candidates', []))}) -- identify "
            "which the actions run, or expand the workload"
        )
    if image["image"] and image["form"] != "declared":
        review.append(
            f"container image was read as {image['form']}, not declared in "
            f"skill_info.yaml -- confirm it is the image the actions run"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "skill": name,
        "skill_path": str(rel_dir),
        "layer": layer,
        "container_image": image["image"],
        "container_image_source": image["source"],
        "container_image_form": image["form"],
        "paths": paths,
        "eligible_platforms": sorted(
            p for p, v in paths.items() if v["verdict"] == ELIGIBLE
        ),
        "undetermined_platforms": sorted(
            p for p, v in paths.items() if v["verdict"] == UNDETERMINED
        ),
        "requires_expansion": requires_expansion,
        "out_of_scope": out_of_scope,
        "service_findings": service_hits,
        "runtime_fetch_sites": scan_runtime_fetches(skill_dir, bank),
        "needs_agent_review": review,
        "verdict_is_provisional": bool(review),
        "closure_note": (
            "E-CLOSURE is not decided here. Run-time fetch sites are candidates "
            "only; the asset list is closed by observation, not by reading."
        ),
    }


def probe_all(bank: Path) -> list[dict[str, Any]]:
    """Screen every skill in the bank."""
    return [
        probe_skill(bank, skill_md.parent.name)
        for layer in LAYERS
        for skill_md in sorted(bank.glob(f"skills/{layer}/*/SKILL.md"))
    ]


def format_one(result: dict[str, Any]) -> str:
    """Format a single screen as agent-readable text."""
    lines = [f"Air-gap screen: {result['skill']}  ({result['layer']} layer)"]
    lines.append(f"- path: {result['skill_path']}")
    for platform, verdict in sorted(result["paths"].items()):
        source = verdict.get("source") or "no citation"
        lines.append(
            f"  - {platform}: {verdict['verdict']} ({verdict['basis']}) "
            f"-- {verdict['reason']} [{source}]"
        )
    if result["requires_expansion"]:
        lines.append("- orchestrating skill: expand the underlying skills, then screen each")
    if result["runtime_fetch_sites"]:
        lines.append("- run-time fetch sites to close by observation:")
        lines.extend(f"  - {site}" for site in result["runtime_fetch_sites"])
    if result["needs_agent_review"]:
        lines.append("- ADJUDICATE (this screen is provisional):")
        lines.extend(f"  - {item}" for item in result["needs_agent_review"])
    else:
        lines.append("- no adjudication outstanding; confirm against the skill anyway")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Print the screen or exit non-zero with a diagnostic."""
    args = parse_args(argv)
    bank = args.skill_bank.expanduser()
    if not (bank / "skills").is_dir():
        print(f"ERROR: no skills/ directory under {bank}", file=sys.stderr)
        return 2
    try:
        payload: Any = probe_all(bank) if args.all else probe_skill(bank, args.skill)
    except KeyError as error:
        print(f"UNMATCHED: {error}", file=sys.stderr)
        return UNMATCHED_EXIT
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    many = isinstance(payload, list)
    rows = payload if many else [payload]
    if args.needs_review:
        rows = [row for row in rows if row["needs_agent_review"] and not row["out_of_scope"]]

    # --needs-review can legitimately filter a single skill down to nothing.
    # That is the good outcome, not an error, and it must not index off the end.
    if not rows:
        if args.format == "json":
            print(json.dumps([] if many else {}, indent=2, sort_keys=True))
        else:
            print("Nothing to adjudicate: every verdict is declared or evidenced.")
        return 0

    if args.format == "json":
        print(json.dumps(rows if many else rows[0], indent=2, sort_keys=True))
        return 0

    if isinstance(payload, list):
        packageable = [r for r in rows if r["eligible_platforms"]]
        pending = [r for r in rows if r["needs_agent_review"] and not r["out_of_scope"]]
        unknown = [r for r in rows if r["undetermined_platforms"]]
        print(
            f"Screened {len(rows)} skills: {len(packageable)} with at least one "
            f"eligible path, {len(pending)} with something to adjudicate "
            f"({len(unknown)} of them undetermined on a path)"
        )
        for row in rows:
            paths = ",".join(row["eligible_platforms"]) or "-"
            flag = "?" if row["verdict_is_provisional"] else " "
            print(f" {flag} {row['skill']}: {paths}")
        print("\n'?' means the screen is provisional; adjudicate before packaging.")
    else:
        print(format_one(rows[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
