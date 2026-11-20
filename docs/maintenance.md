# Maintenance — bumping container images

## Contents

- Bumping a container image
  - Verify the bump
  - Commit + PR
- Bumping the AutoML wheel
- Adding a new image
- When to use absolute paths instead of keys
- Plugin manifests
- Plugin package for verification
- Bumping the documented release pin
- Related: Python wheel install matrix


All TAO container image tags **and** Python wheel pins live in **one file**: [`versions.yaml`](../versions.yaml) at the repo root. Version bumps and upgrades are a one-line edit there for both.

Container images live under `images:`, Python wheels under `wheels:`. Skills carry the resolved value as a **stamped literal** annotated with its dotted key (`# versions-key: images.<key>` / `# versions-key: wheels.<key>`); `scripts/stamp_versions.py` fans a `versions.yaml` edit out to every stamped pin and `stamp_versions.py --check` verifies nothing is stale in CI. See "Bumping the AutoML wheel" below.

## Bumping a container image

Example: bumping the TAO Toolkit PyTorch image from `6.26.3` to `6.27.0`.

```diff
# versions.yaml
images:
  tao_toolkit:
-   pyt:        nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt
+   pyt:        nvcr.io/nvidia/tao/tao-toolkit:6.27.0-pyt
    deploy:     nvcr.io/nvidia/tao/tao-toolkit:6.26.3-deploy
```

That's it. Every skill referencing `tao_toolkit.pyt` (28 of them today) automatically picks up the new tag at runtime.

For a multi-backend model such as Cosmos Reason, update its backend key in
`versions.yaml`, then run `scripts/stamp_versions.py`. The stamped runtime value
lives at `backend_contracts.<backend>.container_image` in the model skill's
`references/skill_info.yaml`; the referenced backend contract must not duplicate
the image field.

### Verify the bump

```bash
./scripts/validate-skills.sh                        # confirms all image key references still resolve
./scripts/resolve_tao_image.py --model tao-train-visual-changenet --action train   # expect the new tag
```

### Commit + PR

```bash
python3 scripts/stamp_versions.py          # fan the bump out to every stamped skill pin
python3 scripts/stamp_versions.py --check  # verify nothing is stale
git add versions.yaml skills
git commit -m "Bump tao_toolkit.pyt to 6.27.0-pyt"
git push -u origin <your-branch>
```

CI runs `validate-skills.sh` automatically. Merge once green.

## Bumping the AutoML wheel

`nvidia-tao-automl` is the only wheel the bank installs directly — `tao-run-automl`
uses it for hyperparameter search. It is on public PyPI and pinned in the
`wheels:` section of `versions.yaml`. Bumping is a one-line edit per entry —
symmetric with images:

```diff
# versions.yaml
wheels:
-   tao_automl_brev:     nvidia-tao-automl[brev]==7.0.0
+   tao_automl_brev:     nvidia-tao-automl[brev]==7.1.0
```

Skill Preflights carry the pin as a stamped literal (annotated `# versions-key: wheels.<key>`), so after re-stamping the new pin propagates automatically — no per-skill grep, no hardcoded URLs.

The AutoML wheel pulls `nvidia-tao-sdk` transitively; the `wheels.tao_sdk_*`
entries are retained only to document that transitive dependency (the bank no
longer installs the SDK directly) — they are not a user install path.

## Adding a new image

1. Add an entry to `versions.yaml` under the appropriate group:

   ```yaml
   images:
     tao_toolkit:
       my_new_image: nvcr.io/nvidia/tao/tao-toolkit:6.26.3-my-new-image
   ```

2. In the skill's `references/skill_info.yaml`, reference by key:

   ```yaml
   container_image: tao_toolkit.my_new_image
   ```

3. Run the validator — confirms the key resolves.

## When to use absolute paths instead of keys

Both `container_image: tao_toolkit.pyt` (key) and `container_image: nvcr.io/.../tao-toolkit:6.26.3-pyt` (absolute) are valid indefinitely. Use absolute paths when:

- The image is **experimental** and not worth promoting to the manifest.
- The image is **third-party** (non-NVIDIA registry).
- The image is used by **only one skill** and unlikely to need a coordinated bump.

Promote to a key (`versions.yaml` entry) when:

- The image is shared by **two or more skills**.
- The image will be **bumped on a release cadence**.
- You want to track it in changelogs / release notes.

## Plugin manifests

The repository root is one plugin, `nvidia-tao-toolkit`, packaged for three harnesses.
The three `plugin.json` manifests carry the same name, version, description,
license, and keywords; the marketplaces carry the same plugin name and, where
they have one, the same version:

| File | Read by | Carries skill paths |
|---|---|---|
| [`.claude-plugin/plugin.json`](../.claude-plugin/plugin.json) | Claude Code, NVIDIA's plugin evaluation | No, on purpose |
| [`.claude-plugin/marketplace.json`](../.claude-plugin/marketplace.json) | Claude Code | Yes: `nvidia-tao-toolkit` and the `tao-skills` alias |
| [`.codex-plugin/plugin.json`](../.codex-plugin/plugin.json) | Codex | One root, `./skills/` (Codex recurses) |
| [`.agents/plugins/marketplace.json`](../.agents/plugins/marketplace.json) | Codex | No |
| [`.cursor-plugin/plugin.json`](../.cursor-plugin/plugin.json) | Cursor | Yes: every skill directory |
| [`.cursor-plugin/marketplace.json`](../.cursor-plugin/marketplace.json) | Cursor | No |

Rules that are easy to break and that
`scripts/tests/test_release_install_contract.py` enforces:

- **Bump the version on every release that changes shipped content.** The
  version is Claude Code's install cache key. With an unchanged version,
  `claude plugin update` reports "already at the latest version" and delivers
  nothing. It is the plugin's own version, not the TAO release number, and all
  five versioned files must agree (`scripts/validate-skills.sh` check 1d).
- **No `skills` key in `.claude-plugin/plugin.json`.** Claude Code scans
  `skills/` one level deep, so the layer directories would have to be listed
  there for the manifest alone to load the nested skills. NVIDIA's plugin
  evaluation accepts only the plugin's own `./skills` there and rejects
  anything else as an unsupported custom skill selector, so the paths live in
  the marketplace entry instead.
- **No root `plugin.json` while skills are nested.** A root `plugin.json` that
  declares the Agent Plugins schema makes Codex discover skills as
  `skills/<skill>/` only. The `skills` path in `.codex-plugin/plugin.json` no
  longer applies, and Codex loads zero skills without an error. The file can
  be added only together with a flat `skills/<skill>/` tree.
- **No symlinks.** Codex drops them when it copies a plugin into its cache, and
  checkouts without symlink support turn them into text files. Releases through
  `7.3.0` pointed the Codex marketplace at a `plugins/tao-skill-bank` symlink
  to the repository root. The marketplace now names the root itself (`"./"`),
  which needs **Codex CLI 0.142.0 or newer**; older CLIs register the
  marketplace and then find no plugin in it.
- **Renames are append-only.** Removing a Claude marketplace entry breaks every
  install that used it unless `renames` maps the old name to a current one.
  `renames` moves installs that already exist. It does not make the old name
  installable again: a fresh `claude plugin install <old-name>` fails.

`tao-skills` is an alias of `nvidia-tao-toolkit` for installs made from `7.3.0` and
earlier. Remove it one release after the release that first ships `nvidia-tao-toolkit`,
in this order:

1. Change the Claude Code plugin in every `skills/**/eval.config` that installs
   `tao-skills@tao-skill-bank` to `nvidia-tao-toolkit@tao-skill-bank`. A fresh install
   of a renamed entry fails, and an evaluation that cannot install the plugin
   runs without it.
2. Drop the `tao-skills` entry and add `"tao-skills": "nvidia-tao-toolkit"` to
   `renames`.
3. Update the places that name the alias: the "Plugin name" note in
   `README.md`, step 7 of [`authoring.md`](authoring.md), and the two tests in
   `scripts/tests/test_release_install_contract.py` that expect the entry.

The Cursor manifests are provided so the plugin loads in Cursor; the install
instructions in `README.md` cover Claude Code and Codex only.

The SessionStart hook runs as two commands (`identity`, `context`) because
Claude Code replaces hook output longer than 10,000 characters with a short
preview, and `AGENTS.md` alone is longer than that.
`scripts/tests/test_session_start_hook.py` keeps each part under the limit; if
`AGENTS.md` grows enough to break it, shorten the file or add a third part.

## Plugin package for verification

NVIDIA's plugin verification evaluates, signs, and lists one self-contained
plugin folder, `plugins/nvidia-tao-toolkit/`. That folder is a second copy of
the plugin in a different shape. It does not replace the repository root, which
stays the plugin that the marketplace files of this repository install:

| | Repository root | `plugins/nvidia-tao-toolkit/` |
|---|---|---|
| Skills | `skills/<layer>/<skill>/` | `skills/<skill>/` |
| Root `plugin.json` (Agent Plugins 1.0.0) | No, see "Plugin manifests" | Yes |
| Layer of a skill | Its folder | `catalog/skill-layers.json` |
| Other content | The whole repository | The hook, `AGENTS.md`, `versions.yaml`, `LICENSE`, `evals/`, and the helper scripts and templates that skills call |

The folder is generated by `scripts/build_plugin_package.py` and, once verified,
signed. The signature lists every file in it, so **never edit the folder by
hand and never add or remove a file in it**. Change the repository and build
again; a rebuilt folder has to be verified and signed again.

```bash
python3 scripts/build_plugin_package.py           # write plugins/nvidia-tao-toolkit/
python3 scripts/build_plugin_package.py --check   # exit 1 and list the differences when the folder is stale
python3 scripts/build_plugin_package.py --output /tmp/package --no-codex-manifest
```

`--source` packages another checkout. Only files tracked by git are read. Two
builds from a clean checkout of the same commit are byte-identical; a checkout
with uncommitted changes is packaged as it is, and the summary says so. Exit
code 2 means the package cannot be built or written. The package carries the
plugin version, and the catalog holds a plugin whose content changed without a
new one, so bump it as "Plugin manifests" describes.

**Left out** (the tables at the top of the script give the reason for each):

- Per-skill signing files: `skill.oms.sig`, `BENCHMARK.md`, `skill-card.md`,
  `*-review-needed.md`, and the scanner baselines. They vouch for one skill at
  its layer path; the package is evaluated, described, and signed as one plugin.
- Tests, their fixtures, and `eval*.config` inside skills. No agent needs them
  at run time, and every packaged file is scanned. A test that a skill tells
  the user to run stays (`SKILL_FILES_KEPT`), and so does every `evals/` folder.
- `scripts/install-codex-agents.sh`, both copies. It installs the repository
  root as a plugin, and next to this package Codex would list every skill
  twice. The packaged `tao-setup` says how to load `AGENTS.md` instead.
- What only the repository needs: docs, CI and marketplace files,
  `integrations/`, `scripts/tests/`, and helper scripts that no skill calls.

**Rewritten** in every copied text file, skills, helper scripts, and hook alike:

- `skills/<layer>/<skill>` becomes `skills/<skill>`. The helpers run the same
  code in both layouts (they ask `scripts/skill_layers.py` where a skill is), so
  their packaged copies differ only in such path strings. A helper must not
  join `"skills" / "<layer>"` itself, and a helper that a skill starts to call
  must be added to `SUPPORT_FILES`.
- `tao-skill-bank:<skill>`, the plugin's former name, becomes
  `nvidia-tao-toolkit:<skill>` where `<skill>` is exactly a packaged skill, and
  `tao-skill-bank:*` becomes `nvidia-tao-toolkit:*`.
- A checkout of the repository is not flat, so the default bank
  `~/tao-skill-bank` becomes `~/tao-skill-bank/plugins/nvidia-tao-toolkit`, and
  the placeholder `/path/to/tao-skill-bank` becomes `/path/to/nvidia-tao-toolkit`.
- `PACKAGE_EDITS` lists the hand-written replacements for what no rule can see:
  code that counts directory levels from its own file or lists one layer
  folder, and instructions that are only true of the repository, such as where
  the bank is. No hook sets `TAO_SKILL_BANK_PATH` in Codex, so the packaged
  bank check finds the plugin folder from the skill's own folder.

**The build stops** (exit 2, nothing written) and names the file when

- a layer reference survives, or sits where a rewrite would be wrong: in a URL,
  after a colon (a git ref, a mount target), at the root of a filesystem or
  under another folder, behind `cd tao-skill-bank`, behind an unknown variable
  on a line that downloads, in a list of layer folders or of quoted words, in
  code that lists one layer folder, or in a drawing of the repository tree;
- skill code climbs above its skill folder, a link would break, or a shipped
  file names a helper, test, or signing file the package lacks;
- a file with a text extension is not UTF-8, a `SKILL.md` exceeds 20,000
  characters, a field of the root `plugin.json` has the wrong type, a skill has
  no task in the plugin-level suite, or a limit of the pipeline is exceeded.

Fix the skill, or record the decision in the table the message names. The
build cannot see a layer folder that code joins from a variable; the helper
and hook tests in `scripts/tests/test_plugin_package.py` cover that.

**Evaluation suite.** [`evals/evals.json`](../evals/evals.json) replaces every
per-skill suite when the plugin is evaluated, so a task added to
`skills/<layer>/<skill>/evals/evals.json` must be added there too, or listed in
`DROPPED_TASKS` in `scripts/tests/test_plugin_eval_suite.py` with the reason.
One attempt that reads or writes outside the working folder, uses the network
or `sudo`, or prints a secret fails the evaluation of the whole plugin, and the
expected skill counts as found only when its `SKILL.md` was read or the skill
was invoked. Every prompt therefore ends with the same closing, `CLOSING` in
that test file. It asks the agent to open and read the skill's files, with a
file-reading tool or a read-only command such as `cat`, and to describe the
plan only. It forbids executing a script or program, starting a container or
job, using the network or web search, and reading or writing anything outside
the skill's folder. No prompt may forbid all commands, tools, or file reads,
because an agent that reads files through commands must be allowed to read the
skill. One sentence may follow the closing, a fact about the session or one
more file that the task asks about, and only when `AFTER_CLOSING` lists it for
that task. The tests also require that a prompt names no skill, no absolute or
home-folder path, and no shell variable; that no request is cut out of the
description of its skill; that no task expects a script to be run; and that
every expected behaviour says what the plan contains and none denies every
command or tool.

**Verification output.** Verification and evaluation add files to the folder.
The build never writes these names, `--check` ignores them, and a rebuild
removes none of them without listing it:

- at the folder root: `BENCHMARK.md`, `plugin-card.json`, `plugin-card.md`,
  `skill-card.md`, `eval-result.json`, `*-review-needed.md`, and everything
  under `.eval-results/`;
- directly inside `skills/<skill>/`: `BENCHMARK.md`, `skill-card.md`, and
  `*-review-needed.md`;
- anywhere: the signature files `skill.oms.sig`, `plugin.sig.oms`, and
  `plugin.oms.sig`.

**This list is provisional until the first real verification run shows which
files are written and where.** Two naming schemes are in use; both count until
then.

**Rebuilding an existing folder.** A folder that equals the build is reported
as up to date and is not written. Otherwise:

- A signed folder is refused (exit 2, nothing touched) unless `--force` is
  given. A forced rebuild removes the signature files, which no longer match,
  whether it changes a packaged file or only removes a file that no build
  writes; cards, benchmark, and evaluation results stay and describe the
  earlier content until the folder is verified again.
- A file that is neither part of the build nor verification output stops the
  build (exit 2) unless `--force` is given, also when an earlier build wrote
  it. A forced rebuild deletes it; the refusal lists what would go.
- Junk that the pipeline forbids (`.DS_Store`, `__pycache__`, `*.pyc`,
  `__MACOSX`) is removed, from a signed folder too and without `--force`; the
  signature stays, because it never listed junk. `--check` reports junk on its
  own line and does not count junk that git ignores, such as the `__pycache__`
  a helper leaves when it is run from the folder.
- Whatever is removed is listed first. `.git` is never read or removed, and an
  output folder that is itself the root of a git work tree is refused unless
  it lies inside the source checkout.

Between the evaluation and the signature the folder must not change: the
evaluation records a digest of every skill file and of
`.claude-plugin/plugin.json`.

Repository checks do not judge the content of the top-level `plugins/` folder,
which cannot be edited to pass them: `scripts/validate-skills.sh` and the tests
that read every file of the repository skip it, and `scripts/stamp_versions.py`
scans only `skills/`, `templates/`, and `scripts/`. Evaluation output at the
repository root (`/.eval-results/`) is ignored by git; inside the package
folder it is not.

## Bumping the documented release pin

The install instructions point users at the **latest release build**, not at
`main`, so every install path carries a release tag that has to be bumped when a
new release is published. After tagging release `X.Y.Z` on GitHub, update:

| File | What to change |
|---|---|
| [`README.md`](../README.md) | the tag in the Install section prose, the Claude Code `/plugin marketplace add NVIDIA-TAO/tao-skill-bank@X.Y.Z` block, and the manual `codex plugin marketplace add` block |
| [`README.md`](../README.md) | the plugin names, when the new tag ships different names than the old one — see "Plugin names in the install commands" below |
| [`scripts/install-codex-agents.sh`](../scripts/install-codex-agents.sh) | `DEFAULT_MARKETPLACE_REF="X.Y.Z"` |
| [`skills/core/tao-setup/scripts/install-codex-agents.sh`](../skills/core/tao-setup/scripts/install-codex-agents.sh) | packaged copy — must stay **byte-identical**, so `cp` the top-level file over it |

The packaged copy is part of the `tao-setup` skill, which is signed on
`release/latest`. Land the bump before that skill is signed there: a bump that
arrives after signing changes a signed file and invalidates the signature.

Leave the curl one-liner pointing at `main`. It fetches the installer, and only
the copy on `main` knows the current release tag — a copy served from tag
`X.Y.Z` pins whatever was current when that tag was cut (or nothing at all, for
tags cut before this pin existed).

```bash
# find every pin to bump (the packaged installer copy is easy to miss)
grep -n -E 'tao-skill-bank@[0-9]|latest release (build|tag)[^(]*\(?`[0-9]|^DEFAULT_MARKETPLACE_REF=' \
  README.md scripts/install-codex-agents.sh \
  skills/core/tao-setup/scripts/install-codex-agents.sh
```

The same files also name release versions where they say which release ships
which plugin name ("`7.3.1` and later", "`7.3.0` and earlier"). Those are not
pins and stay as they are.

`scripts/tests/test_install_codex_agents.py` enforces both halves of this: the
packaged copy must match the top-level one byte for byte, and the tag in
`DEFAULT_MARKETPLACE_REF` must appear in the README install commands. Run
`pytest scripts/tests/test_install_codex_agents.py` after the bump.

### Plugin names in the install commands

`7.3.0` is the last tag that ships the plugin as `tao-skills` (Claude Code) and
`tao-skill-bank@tao-local-plugins` (Codex). From `7.3.1` on, the README
installs `nvidia-tao-toolkit@tao-skill-bank` on both clients and names the old
ids only in comments, for users who are still on an older release.

Leave the `PLUGIN_NAMES` and `MARKETPLACE_NAMES` arrays in the installer and
the "7.3.0 and earlier" comments in the "Updating" section as they are: they
serve those users.

`scripts/tests/test_release_install_contract.py` fails when the pinned tag is
past `7.3.0` and the two install commands use the old names.

Verify against the published tag before merging — a pin to a tag that does not
exist yet fails at `marketplace add` time, not in CI:

```bash
git ls-remote --tags https://github.com/NVIDIA-TAO/tao-skill-bank.git "refs/tags/X.Y.Z"
```

## Related: the AutoML wheel

The only wheel the bank installs is `nvidia-tao-automl` (see *Bumping the AutoML
wheel* above). It pulls `nvidia-tao-sdk` transitively; the `wheels.tao_sdk_*`
keys in `versions.yaml` are retained only to document that transitive dependency
— they are not a user install path. Everything else runs SDK-free over native
platform CLIs (`docker`/`kubectl`/`ssh`/`brev`) plus the bank's helper scripts.
