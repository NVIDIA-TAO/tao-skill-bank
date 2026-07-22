# Maintenance — bumping container images

## Contents

- Bumping a container image
  - Verify the bump
  - Commit + PR
- Adding a new image
- When to use absolute paths instead of keys


All TAO container image tags live in **one file**:
[`versions.yaml`](../versions.yaml) at the repo root. Skills carry the resolved
value as a stamped literal annotated with `# versions-key: images.<key>`;
`scripts/stamp_versions.py` fans out an edit and `--check` verifies CI parity.

## Bumping a container image

Example: bumping the TAO Toolkit PyTorch image from `6.26.3` to `6.27.0`.

```diff
# versions.yaml
images:
  tao_toolkit:
-   pyt:        nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt
+   pyt:        nvcr.io/nvidia/tao/tao-toolkit:6.27.0-pyt
    cosmos_rl:  nvcr.io/nvidia/tao/tao-toolkit:6.26.3-cosmos-rl
    vila:       nvcr.io/nvidia/tao/tao-toolkit:6.26.3-vila
```

That's it. Every skill referencing `tao_toolkit.pyt` (28 of them today) automatically picks up the new tag at runtime.

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
