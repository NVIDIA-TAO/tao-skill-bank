# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Approved content identity and immutable metadata guards for trusted inputs."""
import json,os,time
from pathlib import Path
from wsp03_bindings import pinned,file_hash

class InputGuard:
    def __init__(self,asset,*,recipe,editor,final_inputs):
        started=time.monotonic();self.expected=json.loads(pinned(**asset).read_text());self.metadata={}
        for section,directory in [('base_model',recipe.config['model_directory']),('siglip',final_inputs['siglip_path'])]:
            root=Path(directory).resolve(strict=True);files=self.expected[section]['files']
            if section=='base_model' and {p.name for p in root.iterdir() if p.is_file()}!=set(files):raise ValueError('base17-file inventory changed')
            for relative,item in files.items():
                path=root/relative
                if path.resolve(strict=True)!=Path(item['path']).resolve(strict=True) or path.is_symlink() or path.stat().st_size!=item['bytes']:raise ValueError('model path/size differs from approved input')
                before=self.stat(path)
                if file_hash(path)!=item['sha256'] or before!=self.stat(path):raise ValueError('model bytes changed from approved identity')
                self.metadata[str(path)]=before
        binary=self.expected['codex_binary']
        if Path(editor['binary']).resolve()!=Path(binary['path']).resolve() or file_hash(editor['binary'])!=binary['sha256']:raise ValueError('Codex binary differs from approved identity')
        for name,item in self.expected['evaluator'].items():
            if Path(item['path']).resolve()!=Path(recipe.config['evaluator_directory']).resolve()/name:raise ValueError('evaluator path differs')
            pinned(**item);self.metadata[item['path']]=self.stat(Path(item['path']))
        self.receipt={'expected_inputs_sha256':asset['sha256'],'verified_paths':self.metadata,'verification_wall_seconds':time.monotonic()-started,'inference_calls':0,'withheld_access':False}
    @staticmethod
    def stat(path):
        s=path.stat()
        if s.st_uid!=os.getuid() or s.st_mode&0o002 or s.st_mode&0o020 and s.st_gid!=os.getgid():raise ValueError('input write access must remain within trusted supervisor owner/group')
        return {'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
    def check(self):
        for name,before in self.metadata.items():
            path=Path(name)
            if path.is_symlink() or self.stat(path)!=before:raise RuntimeError('frozen input metadata changed; stop before use')
