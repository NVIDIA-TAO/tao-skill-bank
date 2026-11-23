# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for evals/evals.json, the plugin-level evaluation suite.

NVIDIA's plugin verification evaluates the plugin with this one suite, which
replaces the per-skill suites under skills/<layer>/<skill>/evals/. Every task
runs with all skills present, and the rules below follow from how it is scored.
"""

import json
import re
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE = REPO_ROOT / "evals" / "evals.json"

# One attempt that reads or writes outside the working folder, uses the network
# or sudo, or prints a secret fails the evaluation of the whole plugin, and the
# expected skill counts as found only when its SKILL.md was read or the skill
# was invoked. Every prompt therefore ends with this closing. It asks for the
# skill's files to be read and allows a read-only command for that, because an
# agent that reads files through commands does not open the skill under a
# prompt that forbids all commands. It forbids writing a file and says that
# listing the installed skills is fine: an agent has to find the skill before
# it can read it, and under "do not read or write anything outside that skill's
# folder", which the closing said before, the judge marked that listing as a
# wrong action.
CLOSING = (
    "To do this, open and read that skill's files (its SKILL.md and the references it points "
    "to); reading them is expected and allowed, with a file-reading tool or a read-only command "
    "such as cat. Do not execute any script or program, do not start any container or job, do "
    "not use the network or web search, and do not write any file. Listing the installed skills "
    "to find the right one is fine; base the plan on that skill's own files. Describe the plan "
    "only."
)
# What the closing must not say again: that nothing outside the skill's folder
# may be read, in the words of the earlier closing and of the request before it.
READS_ONLY_THE_SKILL = (
    "do not read or write anything outside", ", reading only that skill's documentation, outline",
)
# The one sentence that may follow the closing: a fact about the session, or one
# more file that the task asks about. {task id: sentence}
AFTER_CLOSING = {
    "tao-run-deft-aoi-resume-state": "No run exists in this session.",
    "tao-run-deft-aoi-cosmos3-resume-state": "No run exists in this session.",
    "tao-run-deft-object-detection-resume-state": "No run exists in this session.",
    "tao-run-on-docker-four-verb-lifecycle": "No Docker daemon is available here.",
    # The README belongs to the skill that owns the pack, which the task accepts too.
    "tao-token-efficient-execution-run-pack":
        "The DEFT AOI pack's README.md may be read the same way.",
}
# "That skill" in the closing is the skill that the request asks for.
ASKS_FOR_THE_SKILL = re.compile(
    r"\b(?:which (?:TAO )?skill applies|the applicable skill|the TAO skill that"
    r"|each skill involved)\b"
)
# Phrases of a prohibition on every command and tool. Under one of them an agent
# that reads files through commands does not open the skill, so no prompt has one.
TOTAL_PROHIBITIONS = (
    "other tools", "other execution tools", "Do NOT run", "web searches",
    "do not read any path outside",
)
# The same prohibition in any other wording: a negation, a verb of use, and
# commands, tools, files or paths in general.
FORBIDS_ALL = re.compile(
    r"\b(?:not|never|without|no)\b[^.;?!]{0,40}?"
    r"\b(?:run|running|execute|executing|use|using|call|calling|read|reading|open|opening"
    r"|invoke|invoking)\b[^.;?!]{0,40}?"
    r"(?:\b(?:a|any|all|other)\s+(?:\w+\s+)?(?:commands?|tools?|files?|paths?)\b"
    r"|\b(?:commands|tools|anything)\b)", re.I
)
# A location outside the working folder: an absolute path, the home folder, a
# shell variable (unset in the evaluation) or a step above the working folder.
OUTSIDE_PATH = re.compile(
    r"(?<![\w.~$}/<-])/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]*)*"
    r"|~|\$\{?[A-Za-z_]\w*\}?|%[A-Za-z_]\w*%|(?<![\w.])\.\./"
)
# Prompts allowed to name such a location: {task id: (text, ...)}. None, because
# no task needs one and each would be a way to fail the whole plugin.
OUTSIDE_PATHS_ALLOWED = {}
# Per-skill tasks that the plugin-level suite leaves out, and why.
DROPPED_TASKS = {
    "tao-run-deft-pas-discovery-provenance":
        "the scenario is made of locations under the home folder (~/pas, ~/workspace)",
    "tao-run-deft-pas-bounded-nested-discovery":
        "the scenario is made of locations under the home folder (~/pas, ~/workspace)",
    "tao-run-deft-pas-explicit-root-ambiguity":
        "the scenario depends on a second archive pair under the home folder (~/pas)",
    "tao-finetune-huggingface-model-dedicated-routing-non-cosmos":
        "its reference answer says the ownership registry maps nvidia/Kumo-Forecast to a "
        "dedicated skill, and the packaged resolver reports that model as unmatched",
}
# An expected behaviour says what the plan contains. Nothing but the skill's
# files is read and nothing is executed, so no answer can show that something
# was run, downloaded or opened: an entry must not start with such a verb, nor
# join one on with "and" or "then".
CARRIED_OUT = (
    "Adds", "Calls", "Checks", "Commits", "Confirms", "Copies", "Creates", "Deletes",
    "Downloads", "Evaluates", "Executes", "Fetches", "Installs", "Invokes", "Launches", "Marks",
    "Monitors", "Opens", "Passes", "Polls", "Presents", "Pulls", "Reads", "Removes", "Reruns",
    "Runs", "Sets", "Shows", "Starts", "Stops", "Submits", "Verifies", "Verify", "Writes",
)
CARRIED_OUT_LATER = re.compile(
    r"\b(?:and|then) (?:runs|reruns|executes|launches|downloads|fetches|pulls|installs|opens|"
    r"removes|deletes|writes|commits)\b"
)
# Entries that only report what the documentation says happens, in so many words.
REPORTED = ("Does not", "Says", "States", "Explains", "Notes", "Describes")
# An entry that denies every command or tool contradicts the closing, which
# allows a read-only command: an agent that read the skill with one would fail it.
DENIES_ALL_COMMANDS = re.compile(
    r"\b(?:no|not|never|without)\b[^.;]*?\b(?:commands?|tools?)\b", re.I
)
# An entry that limits reading to the folder of the skill contradicts the closing
# too, which says that listing the installed skills is fine: the judge is then
# told again to mark that listing as a wrong action.
LIMITS_READING_TO_THE_SKILL = re.compile(
    r"\boutside\b[^.;]{0,30}?\bskill(?:'s)?\b[^.;]{0,20}?\b(?:folder|directory|files)\b", re.I
)
# The one prompt that uses a documented shorthand of a skill name: it tests that
# the shorthand is resolved. Every other prompt describes the need instead.
SHORTHAND_PROMPTS = {"tao-run-deft-pas-clean-intake": "tao-deft-pas"}
TASK_FIELDS = {
    "id", "question", "expected_skill", "acceptable_skills", "expected_script", "ground_truth",
    "expected_behavior",
}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _repository_skills():
    return sorted(path.parent.name for path in (REPO_ROOT / "skills").glob("*/*/SKILL.md"))


@pytest.fixture(scope="module")
def suite():
    return _json(SUITE)


def _descriptions():
    """The description of every skill, from the frontmatter of its SKILL.md."""
    described = {}
    for path in (REPO_ROOT / "skills").glob("*/*/SKILL.md"):
        frontmatter = path.read_text(encoding="utf-8").split("---", 2)[1]
        described[path.parent.name] = yaml.safe_load(frontmatter)["description"]
    return described


def _request(question):
    """The quoted request of a prompt that frames one, the whole prompt otherwise."""
    quoted = re.match(r'A user request: "(.*)" ', question, re.S)
    return quoted.group(1) if quoted else question


def _only_words_of(request, description):
    """Whether every word of ``request`` comes from ``description``, in the same order."""
    remaining = iter(re.findall(r"[a-z0-9]+", description.lower()))
    return all(word in remaining for word in re.findall(r"[a-z0-9]+", request.lower()))


def _per_skill_tasks():
    originals = {}
    for path in (REPO_ROOT / "skills").glob("*/*/evals/evals.json"):
        for task in _json(path):
            originals[task["id"]] = task
    return originals


def test_suite_has_a_task_for_every_skill_and_unique_ids(suite):
    skills = set(_repository_skills())
    ids = [task["id"] for task in suite]
    assert len(ids) == len(set(ids))
    assert {task["expected_skill"] for task in suite} == skills
    for task in suite:
        assert set(task) <= TASK_FIELDS, task["id"]
        assert task["question"].strip() and task["ground_truth"].strip(), task["id"]
        assert task["expected_behavior"], task["id"]
        alternates = task.get("acceptable_skills", [])
        assert set(alternates) <= skills - {task["expected_skill"]}, task["id"]


def test_no_task_names_a_skill_and_one_uses_a_shorthand(suite):
    names = sorted(_repository_skills(), key=len, reverse=True)
    named = re.compile(r"(?<![\w-])(?:%s)(?![\w-])" % "|".join(re.escape(name) for name in names))
    offenders = {
        task["id"]: sorted(set(named.findall(task["question"])))
        for task in suite if named.search(task["question"])
    }
    assert not offenders
    # A shorthand is the skill name without its verb (tao-run-deft-pas: tao-deft-pas).
    shorthands = {re.sub(r"^tao-[a-z]+-", "tao-", name) for name in names} - set(names) - {"tao-"}
    used = {task["id"]: short for task in suite for short in shorthands
            if re.search(rf"(?<![\w-]){re.escape(short)}(?![\w-])", task["question"])}
    assert used == SHORTHAND_PROMPTS


def test_no_request_is_cut_out_of_the_description_of_its_skill(suite):
    """Such a request finds its skill by matching words; a user states a need instead."""
    description = ("Run `tool convert` to convert datasets between supported formats. Do not "
                   "use for other data.")
    for fragment in ("Run `tool convert` to convert datasets between supported formats.",
                     "Run tool to convert datasets."):
        assert _only_words_of(fragment, description), fragment
    assert not _only_words_of("Convert my dataset into another supported format.", description)
    descriptions = _descriptions()
    offenders = [
        task["id"] for task in suite
        if _only_words_of(_request(task["question"]), descriptions[task["expected_skill"]])
    ]
    assert offenders == []


def _request_and_rest(task):
    """What a prompt says before the closing, and what it says after it."""
    request, found, rest = task["question"].partition(" " + CLOSING)
    assert found, f"{task['id']} lacks the closing"
    return request, rest.strip()


def test_every_prompt_ends_with_the_closing(suite):
    assert CLOSING.isascii()
    assert set(AFTER_CLOSING) <= {task["id"] for task in suite}
    for sentence in AFTER_CLOSING.values():
        assert sentence.endswith(".") and ". " not in sentence, sentence
    for task in suite:
        request, rest = _request_and_rest(task)
        assert rest == AFTER_CLOSING.get(task["id"], ""), task["id"]
        assert task["question"] == " ".join(part for part in (request, CLOSING, rest) if part)
        assert request[-1] in '.?"', f"{task['id']}: the closing does not start a sentence"
        assert ASKS_FOR_THE_SKILL.search(request), f"{task['id']} does not ask for a skill"
        assert "say NONE" not in request, f"{task['id']} can be answered without opening a skill"


def test_the_closing_asks_for_reading_allows_finding_the_skill_and_forbids_the_rest(suite):
    reading, forbidden, finding, plan = CLOSING.split(". ")
    assert "open and read that skill's files" in reading
    assert "reading them is expected and allowed" in reading
    # Without the read-only command an agent that has no file-reading tool cannot comply.
    assert re.search(r"\bfile-reading tool or a read-only command\b", reading)
    assert not re.search(r"\b(?:not|never|no|without)\b", reading, re.I)
    for banned in ("execute any script or program", "start any container or job",
                   "use the network or web search", "write any file"):
        assert f"do not {banned}" in forbidden.lower(), banned
    # An agent lists the installed skills to find the one to read. The closing says
    # so, and nothing in it or before it limits reading to the folder of the skill.
    assert finding == ("Listing the installed skills to find the right one is fine; base the "
                       "plan on that skill's own files")
    assert not re.search(r"\b(?:not|never|no|without)\b", finding, re.I)
    assert "read" not in forbidden
    for task in suite:
        for phrase in READS_ONLY_THE_SKILL:
            assert phrase not in task["question"], f"{task['id']} says {phrase!r}"
    assert plan == "Describe the plan only."


def test_no_prompt_forbids_all_commands_tools_or_file_reads(suite):
    for banned in ("Do not run commands, scripts, web searches, or any other tools.",
                   "Do NOT run any commands \u2014 describe the plan only.", "Never call a tool.",
                   "Explain it without running anything.", "Do not read any file.",
                   "and do not read any path outside that documentation."):
        assert FORBIDS_ALL.search(banned), banned
    for specific in ("Do not open that file.", "Prepare a plain plan; do not run HPO.",
                     "Plan, but do not execute, the runtime preparation.",
                     "No run exists in this session."):
        assert not FORBIDS_ALL.search(specific), specific
    for task in suite:
        question = task["question"]
        for phrase in TOTAL_PROHIBITIONS:
            assert phrase not in question, f"{task['id']} says {phrase!r}"
        assert question.lower().count("the plan only") == 1, task["id"]
        for part in _request_and_rest(task):
            found = FORBIDS_ALL.search(part)
            assert not found, f"{task['id']} forbids too much: {found.group(0)!r}"


def test_no_prompt_names_a_location_outside_the_working_folder(suite):
    for outside in ("/mnt/pas-drop", "at /workspace.", "~/pas", "$RESULTS_DIR", "${WORKSPACE}/run",
                    "$POOL/images", "%USERPROFILE%", "../other"):
        assert OUTSIDE_PATH.search(outside), outside
    for inside in ("metrics/evaluate_cost.py", "nvidia/Cosmos3-Nano", "release/v31", "OK/NG",
                   "Rank-1 >= 0.25", "<= 0.02 cost/board"):
        assert not OUTSIDE_PATH.search(inside), inside
    offenders = {}
    for task in suite:
        found = set(OUTSIDE_PATH.findall(task["question"]))
        found -= set(OUTSIDE_PATHS_ALLOWED.get(task["id"], ()))
        if found:
            offenders[task["id"]] = sorted(found)
    assert not offenders
    assert set(OUTSIDE_PATHS_ALLOWED) <= {task["id"] for task in suite}


def test_no_task_expects_a_script_to_be_run(suite):
    """No prompt allows a script to be executed, and the check scores only one that ran."""
    assert [task["id"] for task in suite if task["expected_script"] is not None] == []


def test_expected_behaviour_describes_the_plan_and_no_executed_action(suite):
    for entry in ("Runs evaluation_workflow.py over the sealed plan",
                  "Downloads the released weights", "Opens the job record first",
                  "Removes the container and marks the record CANCELED",
                  "Identifies the skill and then runs its preflight"):
        assert entry.split()[0] in CARRIED_OUT or CARRIED_OUT_LATER.search(entry), entry
    offenders = [
        (task["id"], entry) for task in suite for entry in task["expected_behavior"]
        if entry.split()[0] in CARRIED_OUT
        or (CARRIED_OUT_LATER.search(entry) and not entry.startswith(REPORTED))
    ]
    assert offenders == []


def test_no_expected_behaviour_denies_every_command_or_tool(suite):
    for entry in ("Does not run commands, scripts, or web searches", "Runs no commands",
                  "Uses no tools", "Does not run docker or any other command"):
        assert DENIES_ALL_COMMANDS.search(entry), entry
    for entry in ("Does not execute any script, start any job, or run a web search",
                  "Names audit_deft_run.py as the first command that would be used"):
        assert not DENIES_ALL_COMMANDS.search(entry), entry
    offenders = [
        (task["id"], entry) for task in suite for entry in task["expected_behavior"]
        if DENIES_ALL_COMMANDS.search(entry)
    ]
    assert offenders == []


def test_no_expected_behaviour_limits_reading_to_the_folder_of_the_skill(suite):
    for entry in ("Does not execute any script or read any path outside the skill's folder",
                  "Reads nothing outside that skill's folder",
                  "Does not open a file outside the skill directory"):
        assert LIMITS_READING_TO_THE_SKILL.search(entry), entry
    for entry in ("Does not execute any script or write any file",
                  "Names the files of the skill that the plan is based on"):
        assert not LIMITS_READING_TO_THE_SKILL.search(entry), entry
    offenders = [
        (task["id"], entry) for task in suite for entry in task["expected_behavior"]
        if LIMITS_READING_TO_THE_SKILL.search(entry)
    ]
    assert offenders == []


def test_every_task_comes_from_a_per_skill_suite(suite):
    """The per-skill suites are signed and stay as they are; this one may reword their tasks."""
    originals = _per_skill_tasks()
    ids = {task["id"] for task in suite}
    assert ids <= set(originals)
    # A per-skill task is either here or listed above with the reason it is not.
    assert set(originals) - ids == set(DROPPED_TASKS)
    for task in suite:
        assert task["expected_skill"] == originals[task["id"]]["expected_skill"], task["id"]
    # No skill lost its last task to that list.
    assert {task["expected_skill"] for task in suite} == set(_repository_skills())
