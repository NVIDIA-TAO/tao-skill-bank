# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted mechanical partitioner. Prints counts/hashes only, never row data."""
from pathlib import Path
import os
import hashlib,json,random,re,shutil,time

ROOT=Path(os.environ['WSP05_RUN_ROOT']).expanduser().resolve()
PREVIOUS=Path(os.environ['WSP05_PREVIOUS_REQUEST_CONFIG']).expanduser().resolve()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def read_asset(a):
    assert sha(a['path'])==a['sha256'],'pinned source differs'
    return json.loads(Path(a['path']).read_text())
def split_groups(identities):
    groups=sorted(set(tuple(v) for v in identities.values()))
    random.Random(42).shuffle(groups)
    a=set(groups[:len(groups)//2])
    return {s:sorted(q for q,k in identities.items() if (tuple(k) in a)==(s=='A')) for s in ['A','B']}
def main():
    private=ROOT/'private';private.mkdir(mode=0o700,exist_ok=False)
    cfg=json.loads(PREVIOUS.read_text())
    single=read_asset(cfg['single_requests']['requests'])
    icl=read_asset(cfg['icl_requests']['requests'])
    assert len(single)==len(icl)==271 and {r['id'] for r in single}=={r['id'] for r in icl}
    ids=[r['id'] for r in single];assert len(set(ids))==271
    pattern=re.compile(r'SCN_(.*?)_(MBO_.*?)_(A\d+)(?:_[0-9a-f]{12})?\.[^.]+$')
    identities={};failed=0
    for q in ids:
        m=pattern.search(q)
        if m is None:failed+=1
        else:identities[q]=[m.group(1),m.group(2)]
    assert failed==0,'joint parser failure count='+str(failed)
    assert len(set(tuple(v) for v in identities.values()))==255,'joint count differs'
    partitions=split_groups(identities)
    assert not set(partitions['A'])&set(partitions['B'])
    assert not {tuple(identities[q]) for q in partitions['A']}&{tuple(identities[q]) for q in partitions['B']}
    write(private/'partition.json',{'seed':42,'method':'sort joint tuples, Python Random(42).shuffle, first floor(groups/2) to A; no label-based balancing','ids':partitions,'joints':identities})
    write(private/'upstream-bindings.json',{'source_config_sha256':sha(PREVIOUS),
          'gold':cfg['gold'],'evaluator_directory':cfg['evaluator_directory']})
    public={'schema':'wsp05-private-split/1','seed':42,'group_count':255,'n':271,
            'partition_sha256':sha(private/'partition.json'),'created_at':time.time(),
            'method':'joint-grouped; sorted joint tuples then Python Random(42).shuffle; no label-based stratification',
            'historical_predictions_opened':False,'splits':{}}
    for split in ['A','B']:
        d=private/split;d.mkdir();members=set(partitions[split]);counts={}
        for profile,rows in [('single',single),('icl',icl)]:
            selected=[r for r in rows if r['id'] in members]
            assert len(selected)==len(members)
            assert all(len(r['references'])==(profile=='icl') for r in selected)
            folder=d/profile;folder.mkdir();images=folder/'input';images.mkdir()
            asset=cfg[profile+'_requests'];source=Path(asset['images']).resolve()
            for row in selected:
                for message in row['messages']:
                    for part in message['content']:
                        if part.get('type')!='image':continue
                        path=Path(part['image']);assert path.parent==Path('/input')
                        src=source/path.name;assert not src.is_symlink() and src.resolve().parent==source
                        assert sha(src)==asset['image_sha256'][path.name],'image digest differs'
                        if not (images/path.name).exists():shutil.copyfile(src,images/path.name)
            write(folder/'requests.json',selected)
            counts[profile+'_requests_sha256']=sha(folder/'requests.json')
        public['splits'][split]={'n':len(members),'group_count':len({tuple(identities[q]) for q in members}),**counts}
    write(ROOT/'setup/split-public.json',public)
    print(json.dumps(public))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        # Deliberately no trace/local variables/row-bearing paths in public output.
        print(json.dumps({'status':'FAILED','error_type':type(e).__name__,'stage':'private partition; inspect source/aggregate evidence only'}))
        raise SystemExit(1)
