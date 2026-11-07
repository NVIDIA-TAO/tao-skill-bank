# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, non-symlink reads across the child-to-supervisor boundary."""
from contextlib import contextmanager
import json,os,stat
from pathlib import Path

@contextmanager
def child_file(root,relative,*,owner,max_bytes):
    relative=Path(relative)
    if relative.is_absolute() or not relative.parts or any(p in ('..','.') for p in relative.parts):raise ValueError('unsafe child artifact path')
    fds=[]
    try:
        fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);fds.append(fd)
        for part in relative.parts[:-1]:
            fd=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);fds.append(fd)
        leaf=os.open(relative.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd);fds.append(leaf)
        info=os.fstat(leaf)
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=owner or info.st_nlink!=1 or info.st_size>max_bytes:raise ValueError('child artifact type/owner/size violation')
        with os.fdopen(os.dup(leaf),'rb') as stream:yield stream
    finally:
        for fd in reversed(fds):os.close(fd)

def child_bytes(root,relative,*,owner,max_bytes):
    with child_file(root,relative,owner=owner,max_bytes=max_bytes) as stream:
        data=stream.read(max_bytes+1)
        if len(data)>max_bytes:raise ValueError('child artifact grew beyond size limit')
        return data

def child_json(root,relative,*,owner,max_bytes=4*1024**2):
    return json.loads(child_bytes(root,relative,owner=owner,max_bytes=max_bytes))


def stage_candidate(root,candidate,destination,*,owner,group,expected_manifest):
    """Copy only manifest-named regular patch files into a trusted-owned view."""
    import shutil,subprocess
    manifest=child_json(root,str(Path(candidate)/'manifest.json'),owner=owner)
    if manifest!=expected_manifest:raise ValueError('candidate differs from registered immutable manifest')
    if manifest.get('id')!=candidate or not manifest.get('valid') or not 0<len(manifest.get('tensors',[]))<=32:raise ValueError('invalid candidate manifest')
    destination=Path(destination);destination.mkdir(mode=0o750,parents=True,exist_ok=False)
    names=set()
    for row in manifest['tensors']:
        name=row['file']
        if Path(name).name!=name or name in names or not name.endswith('.safetensors'):raise ValueError('invalid candidate tensor filename')
        names.add(name)
        with child_file(root,str(Path(candidate)/name),owner=owner,max_bytes=16*1024**3) as source, (destination/name).open('xb') as target:bounded_copy(source,target,max_bytes=16*1024**3)
    from wsp03_io import json_write
    json_write(destination/'manifest.json',manifest,mode=0o640)
    for path in [destination,*destination.iterdir()]:
        subprocess.run(['sudo','-n','chgrp',str(group),str(path)],check=True)
        path.chmod(0o750 if path.is_dir() else 0o440)
    return manifest


def bounded_copy(source,target,*,max_bytes):
    total=0;before=os.fstat(source.fileno())
    while True:
        data=source.read(min(8*1024**2,max_bytes-total+1))
        if not data:break
        total+=len(data)
        if total>max_bytes:raise ValueError('child artifact grew beyond copy bound')
        target.write(data)
    after=os.fstat(source.fileno())
    if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns) or total!=before.st_size:raise ValueError('child artifact changed during staging')
    return total


def stage_export(root,candidate,receipt,destination,*,owner,group):
    import hashlib,subprocess
    from wsp03_worker import digest
    from wsp03_io import json_write
    if receipt.get('candidate')!=candidate or not receipt.get('replay_equivalence_verified') or digest(receipt['files'])!=receipt['inventory_sha256']:raise ValueError('export provenance/equivalence missing')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    for item in receipt['files']:
        relative=Path(item['file'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('export file escape')
        path=destination/relative;path.parent.mkdir(parents=True,exist_ok=True)
        with child_file(root,str(Path('exports')/candidate/relative),owner=owner,max_bytes=item['size']) as source,path.open('xb') as target:bounded_copy(source,target,max_bytes=item['size'])
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
        if h.hexdigest()!=item['sha256']:raise ValueError('export file differs from frozen receipt')
    json_write(destination/'export.json',receipt,mode=0o640)
    for path in [destination,*destination.rglob('*')]:
        subprocess.run(['sudo','-n','chgrp',str(group),str(path)],check=True);path.chmod(0o750 if path.is_dir() else 0o440)
    return receipt['inventory_sha256']
