# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Existing light launcher plus explicit read-only pristine model mounts."""
import json, sys
from pathlib import Path
import wsp03_light

BaseDocker = wsp03_light.Docker
class ModelDocker(BaseDocker):
    def submit(self, *args, **kwargs):
        config = json.loads(Path(sys.argv[2]).read_text())
        if set(config['extra_models']) != {'D','G','F'}:
            raise ValueError('explicit pristine extra model set required')
        kwargs['mounts'] = list(kwargs.get('mounts', [])) + [
            (Path(path).resolve(strict=True), '/models/'+arm, True)
            for arm,path in config['extra_models'].items()]
        return super().submit(*args, **kwargs)

if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        wsp03_light.Docker = ModelDocker
    wsp03_light.main()
