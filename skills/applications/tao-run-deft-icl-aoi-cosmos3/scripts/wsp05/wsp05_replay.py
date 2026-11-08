# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hash-bound gradient-free replacement replay, including authorized head edits."""
from pathlib import Path
import json


def apply_candidate(model, directory):
    import torch
    from safetensors import safe_open
    from wsp03_kernel import digest_tensor
    from wsp03_worker import digest
    root = Path(directory).resolve(strict=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    if not manifest.get('valid') or manifest.get('changed_elements', 0) <= 0:
        raise ValueError('nonempty valid replacement required')
    names, total = set(), 0
    with torch.inference_mode():
        for item in manifest['tensors']:
            name = item['name']
            if name in names:
                raise ValueError('duplicate tensor')
            names.add(name)
            parameter = model.get_parameter(name)
            if digest_tensor(parameter) != item['before_sha256']:
                raise ValueError('parent digest mismatch')
            raw = root / item['file']
            path = raw.resolve(strict=True)
            if raw.is_symlink() or path.parent != root:
                raise ValueError('unsafe replacement path')
            with safe_open(path, framework='pt', device='cpu') as file:
                if set(file.keys()) != {'value'}:
                    raise ValueError('only value tensor permitted')
                edited = file.get_tensor('value')
            if parameter.shape != edited.shape or parameter.dtype != edited.dtype:
                raise ValueError('shape or dtype mismatch')
            if not bool(torch.isfinite(edited).all()) or digest_tensor(edited) != item['after_sha256']:
                raise ValueError('nonfinite or changed replacement')
            changed = int((parameter != edited).sum())
            if changed != item['changed_elements']:
                raise ValueError('changed count mismatch')
            total += changed
            parameter.copy_(edited)
    if total != manifest['changed_elements'] or not names:
        raise ValueError('total changed count mismatch')
    return digest(manifest)


if __name__ == '__main__':
    import runpy
    import wsp03_worker
    wsp03_worker.apply_candidate = apply_candidate
    # Invoke the existing CLI with the versioned replay substituted in its globals.
    source = Path(wsp03_worker.__file__).read_text()
    body = source[source.rfind("if __name__=='__main__':"):]
    exec(compile(body, wsp03_worker.__file__, 'exec'), dict(vars(wsp03_worker), __name__='__main__'))
