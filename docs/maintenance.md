# Maintenance — bumping container images

## Contents

- Bumping a container image
  - Verify the bump
  - Commit + PR
- Bumping the AutoML wheel
- Adding a new image
- When to use absolute paths instead of keys
- Bumping the documented release pin
- Related: Python wheel install matrix


All TAO container image tags **and** Python wheel pins live in **one file**: [`versions.yaml`](../versions.yaml) at the repo root. RC bumps and upgrades are a one-line edit there for both.

Container images live under `images:`, Python wheels under `wheels:`. Skills carry the resolved value as a **stamped literal** annotated with its dotted key (`# versions-key: images.<key>` / `# versions-key: wheels.<key>`); `scripts/stamp_versions.py` fans a `versions.yaml` edit out to every stamped pin and `stamp_versions.py --check` verifies nothing is stale in CI. See "Bumping the AutoML wheel" below.

## Bumping a container image

Example: bumping the TAO Toolkit PyTorch image from `6.26.3` to `6.27.0`.

```diff
# versions.yaml
images:
  tao_toolkit:
-   pyt:        nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt
+   pyt:        nvcr.io/nvidia/tao/tao-toolkit:6.27.0-pyt
    vila:       nvcr.io/nvidia/tao/tao-toolkit:6.26.3-vila
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
+   tao_automl_brev:     nvidia-tao-automl[brev]==7.1.0rc1
```

Skill Preflights carry the pin as a stamped literal (annotated `# versions-key: wheels.<key>`), so after re-stamping the new pin propagates automatically — no per-skill grep, no hardcoded URLs.

The AutoML wheel pulls `nvidia-tao-sdk` transitively; the `wheels.tao_sdk_*`
entries are retained only to document that transitive dependency (the bank no
longer installs the SDK directly) — they are not a user install path.

### Internal RC versions

To stage an RC internally before the public release:

1. Publish the RC wheel to the index pip is pointed at — an internal PyPI mirror, or `--extra-index-url` / `--index-url` supplied via pip config or `PIP_*` env. Index selection is an environment concern; the skill bank never bakes in a registry.
2. Pin the **exact** RC version in `versions.yaml` (e.g. `==7.1.0rc1`). pip installs an exact pre-release pin without `--pre`; a non-exact specifier like `>=7.1.0` would skip pre-releases unless `--pre` is passed.

That's the whole change: one line in `versions.yaml`, exactly like a container RC bump.

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
- You want to track it in changelogs / RC notes.

## Bumping the documented release pin

The install instructions point users at the **latest release build**, not at
`main`, so every install path carries a release tag that has to be bumped when a
new release is published. After tagging release `X.Y.Z` on GitHub, update:

| File | What to change |
|---|---|
| [`README.md`](../README.md) | the tag in the Install section prose, the Claude Code `/plugin marketplace add NVIDIA-TAO/tao-skill-bank@X.Y.Z` block, and the manual `codex plugin marketplace add` block |
| [`scripts/install-codex-agents.sh`](../scripts/install-codex-agents.sh) | `DEFAULT_MARKETPLACE_REF="X.Y.Z"` |

Leave the curl one-liner pointing at `main`. It fetches the installer, and only
the copy on `main` knows the current release tag — a copy served from tag
`X.Y.Z` pins whatever was current when that tag was cut (or nothing at all, for
tags cut before this pin existed).

```bash
grep -rn "7\.1\.0" README.md scripts/install-codex-agents.sh   # find every pin to bump
```

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
