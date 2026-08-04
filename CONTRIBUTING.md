# Contributing to TAO Skill Bank

Thanks for contributing! Two requirements are enforced by CI on every pull request and must pass before it can merge:

1. **SPDX license header** on source files — checked by the `license header` hook in the **Static Tests** workflow.
2. **DCO sign-off** on every commit — checked by the **DCO** workflow.

Both run automatically on your PR. See [Running the checks locally](#running-the-checks-locally) to catch failures before you push.

## License headers

Every source file must carry an **SPDX license header** near the top. Add these two lines, **below** any shebang (`#!`) line, replacing `<year>` with the current year:

```
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

Use the comment style for the file's language. The check passes as long as **both** a copyright line (`SPDX-FileCopyrightText:` or `Copyright`) **and** an `SPDX-License-Identifier:` line appear within the **first 10 lines** of the file.

**Python / shell / YAML** (an optional shebang may sit above the header):

```python
#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

**C / C++ / JavaScript / Go** (`//` comments):

```c
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
```

Files that legitimately cannot carry a header (e.g. generated files or data) may be listed, one repo-relative path per line, in `.github/hooks/license_header_exclude.txt`.

## Signing your work (DCO)

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Already committed without a sign-off? Add one with:
  ```bash
  git commit --amend -s --no-edit      # fixes the most recent commit
  git rebase --signoff origin/main     # signs off a range of commits
  ```

* Full text of the DCO (https://developercertificate.org/):

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```

## Running the checks locally

Both checks also run in CI, but you can catch failures before pushing:

```bash
pip install pre-commit

# check only the files your branch changed vs main:
pre-commit run --from-ref origin/main --to-ref HEAD

# or check everything:
pre-commit run --all-files
```

The `license header` and `dco-signoff` hooks (plus the lint/docstring checks) are defined in `.pre-commit-config.yaml`. Run `pre-commit install` once to have them run automatically on every `git commit`.
