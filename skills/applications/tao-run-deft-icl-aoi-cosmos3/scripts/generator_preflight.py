# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scoped, read-only preflight. Never opens Benchmark gold or starts a model."""
import copy
import hashlib
import json
import math
import os
import sqlite3
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from generator_contract import file_binding, verify_file, verify_prepared, pixel_identity, within
from generator_policy import POLICY
from suite_runtime import BANK
from suite_preflight import EXPECTED_COUNTS, PROMPT_SHA
from suite_data import validate_identity_grammar
from tht_contract import CLASSES, external_evaluator, json_hash, load_jsonl, sha256, validate_queries

TRAINING = {'rank':16,'alpha':32,'dropout':0.05,'lr':1e-4,'weight_decay':0.01,
            'targets':'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
            'seed':42,'effective_batch':1,'precision':'bfloat16'}
STATISTICS = {'method':'paired_joint_percentile','grouping':'object_name+normalized_scan_id',
              'seed':42,'replicates':2000,'confidence':0.95}
D_GATES = ('external_scorer_parity','native_training','checkpoint_resume','native_export',
           'loss_mask','runtime_isolation','embedding_and_mining','paired_compatibility','new_round_resume')


def tree_binding(root, aliases=None, container_targets=None):
    """Seal every effective directory/file and every link in its resolution chain."""
    root = Path(root); root = root if root.is_absolute() else Path.cwd()/root
    aliases=aliases or {}; container_targets=container_targets or {}
    files={}; links={}; directories={}

    def translate(target):
        for destination, source in sorted(aliases.items(),key=lambda kv:-len(kv[0])):
            if target.is_relative_to(destination):
                relative=target.relative_to(destination)
                if '..' in relative.parts: raise ValueError('native alias escapes its mounted target')
                return Path(source)/relative
        return target

    def resolve(path, active=()):
        path=Path(path); path=path if path.is_absolute() else Path.cwd()/path
        current=Path(path.anchor)
        for index, part in enumerate(path.parts[1:],1):
            if part=='..':
                current=current.parent;continue
            current=current/part
            if current.is_symlink():
                name=str(current)
                if name in active: raise ValueError('native symlink cycle')
                link=os.readlink(current)
                target=Path(link) if Path(link).is_absolute() else current.parent/link
                target=translate(target)
                links[name]={'path':name,'link':link,'translated_target':str(target)}
                return resolve(target.joinpath(*path.parts[index+1:]),(*active,name))
        if not current.exists():
            declared=container_targets.get(str(current))
            if not declared: raise ValueError('unbound dangling native symlink: '+str(current))
            verify_file(declared['artifact']);files[declared['artifact']['path']]=declared['artifact']
            return current, declared
        return current, None

    def visit(path, ancestors=()):
        target, container=resolve(path)
        if container: return
        name=str(target)
        if target.is_file():
            if name not in files: files[name]=file_binding(target)
        elif target.is_dir():
            if name in ancestors: raise ValueError('native directory symlink cycle')
            children=sorted(target.iterdir())
            directories[name]=[p.name for p in children]
            for child in children: visit(child,(*ancestors,name))
        else: raise ValueError('unsupported native tree entry')

    visit(root)
    if not files: raise ValueError('empty input tree: '+str(root))
    return {'root':str(root),'files':[files[k] for k in sorted(files)],
            'symlinks':[links[k] for k in sorted(links)],'directories':directories,
            'aliases':aliases,'container_targets':container_targets}


def verify_tree(tree):
    try:
        rebuilt=tree_binding(tree['root'],tree['aliases'],tree['container_targets'])
    except (OSError,RuntimeError) as error:
        raise ValueError('sealed native tree resolution changed') from error
    if rebuilt!=tree: raise ValueError('sealed native tree bytes, membership or resolution changed')


def validate_cache_provenance(task):
    cache=task['retrieval_cache'];receipt=json.loads(verify_file(cache['provenance']).read_text())
    original=json.loads(verify_file(cache['approved_preflight']).read_text())
    approved=original['effective']
    if original['preflight_sha256']!=json_hash(approved) or original['preflight_sha256']!=receipt['original_preflight_semantic_sha256']:
        raise ValueError('original approved embedding provenance changed')
    for k in ('siglip_revision','siglip_path','data_services_image','framework_image','public_package','private_package'):
        if task[k]!=approved[k]: raise ValueError('same-D model/runtime identity differs: '+k)
    public=Path(task['public_package'])
    if sha256(public/'SHA256SUMS')!=approved['provenance']['public_checksums_sha256'] or sha256(public/'bundle.json')!=approved['provenance']['bundle_sha256']:
        raise ValueError('frozen public package differs from original D')
    base=json.loads((Path(task['model_path'])/'tao_conversion_provenance.json').read_text())
    if base['prepared']['fingerprint']!=approved['base_fingerprint']:
        raise ValueError('fresh classifier base differs from original D')
    for name,digest in approved['provenance']['siglip_files'].items():
        if sha256(Path(task['siglip_path'])/name)!=digest: raise ValueError('SigLIP differs from the approved original model bytes')
    for binding in receipt['inventory']: verify_file(binding)
    database=Path(receipt['readonly_original_stage_query']['db']).resolve(strict=True)
    # Fixed read-only query: never reads Benchmark rows or gold.
    with sqlite3.connect('file:'+str(database)+'?mode=ro',uri=True) as db:
        stages={name:{'name':name,'status':status,'result':json.loads(result)}
                for name,status,result in db.execute('SELECT name,status,result FROM stages WHERE name IN (?,?)',
                                                    ('retrieval/embed-mining','retrieval/embed-proxy'))}
    for role in ('mining','proxy'):
        name='retrieval/embed-'+role;stage=stages.get(name)
        recorded=next((r for r in receipt['completed_original_producer_stages'] if r['name']==name),None)
        if not stage or not recorded or any(stage[k]!=recorded[k] for k in ('name','status','result')) or stage['status']!='complete':
            raise ValueError('original native embedding producer completion differs')
        vector=cache[role]['vectors'];artifact={'path':vector['path'],'sha256':vector['sha256']}
        if artifact not in stage['result']['artifacts']: raise ValueError('embedding output differs from original completed producer')
        job=stage['result']['job'];terminal=verify_file(cache[role]['terminal'])
        if terminal.parent!=Path(job['results_dir']) or Path(cache[role]['launch']['path']).parent!=terminal.parent:
            raise ValueError('native embedding receipt job identity differs')
        state=json.loads(terminal.read_text())
        if state.get('ExitCode')!=0 or state.get('Status')!='exited' or any(state.get(k)!=v for k,v in recorded['terminal'].items()):
            raise ValueError('original native embedding job did not complete successfully')
    return [cache['provenance'],cache['approved_preflight'],*receipt['inventory']]


def validate_cache(task, role, expected):
    import pyarrow.parquet as pq
    import yaml
    entry=task['retrieval_cache'][role]
    paths={k:verify_file(entry[k]) for k in ('vectors','input','spec','launch','terminal')}
    spec=yaml.safe_load(paths['spec'].read_text())
    if spec != {'input_parquet':str(paths['input']), 'output_parquet':str(paths['vectors']),
                'model':'SigLIP','model_path':task['siglip_path'],'model_config_path':'','batch_size':32}:
        raise ValueError('cache does not match original native embedding spec')
    launch=json.loads(paths['launch'].read_text()); argv=launch['argv']
    if task['data_services_image'] not in argv or 'embedding' not in argv or 'image_embeddings' not in argv or str(paths['spec']) not in argv:
        raise ValueError('embedding cache lacks native launch provenance')
    inputs=pq.read_table(paths['input']).to_pylist(); vectors=pq.read_table(paths['vectors']).to_pylist()
    by={r['id']:r for r in expected}
    if len(inputs)!=len(by) or len(vectors)!=len(by) or {r['id'] for r in inputs}!=set(by) or {r['id'] for r in vectors}!=set(by):
        raise ValueError('embedding cache role membership differs')
    input_by={r['id']:r for r in inputs}; dimensions=set()
    public=Path(task['public_package'])
    for row in vectors:
        original=by[row['id']]; inp=input_by[row['id']]
        if row.get('image_sha256')!=original['image_sha256'] or inp.get('image_sha256')!=original['image_sha256']:
            raise ValueError('embedding cache source fingerprint mismatch')
        source=public/original['image']; view=Path(inp['filepath'])
        if sha256(source)!=original['image_sha256'] or sha256(view)!=inp['embedding_view_sha256']:
            raise ValueError('embedding source or preprocessing view changed')
        if pixel_identity(source,training=True)!=pixel_identity(view): raise ValueError('embedding preprocessing mismatch')
        v=row['embedding']; dimensions.add(len(v))
        if not v or any(not math.isfinite(float(x)) for x in v) or not any(v): raise ValueError('invalid native embedding')
        if 'labels' in inp or 'labels' in row: raise ValueError('embedding native worker must not receive labels')
    if len(dimensions)!=1: raise ValueError('inconsistent embedding dimensions')
    return {'count':len(vectors),'dimension':dimensions.pop(),'files':list(entry.values())}


def check_results_root(root, *, resume=False):
    root=Path(root)
    if not root.exists(): return
    if root.is_symlink() or not root.is_dir(): raise ValueError('results root must be a plain directory')
    entries=list(root.iterdir())
    if resume and (root/'suite.sqlite').is_file():
        try:
            with sqlite3.connect('file:'+str(root/'suite.sqlite')+'?mode=ro',uri=True) as db:
                row=db.execute("SELECT value FROM meta WHERE key='protocol'").fetchone()
            if row and json.loads(row[0])=='generator-addendum-v1': return
        except (sqlite3.Error,ValueError): pass
        raise ValueError('results journal is not an initialized generator run')
    allowed={'driver.lock'} if resume else set()
    if any(p.name not in allowed or p.is_symlink() or not p.is_file() or p.stat().st_size for p in entries):
        raise ValueError('fresh results root contains unowned artifacts')


def validate(task):
    task=copy.deepcopy(task)
    required=('public_package','private_package','model_path','framework_image','data_services_image',
              'direct_profile','siglip_path','siglip_revision','results_dir','budgets','m3_evidence',
              'environment_manifest','generator','generator_evidence','epoch_selection','retrieval_cache')
    missing=[k for k in required if k not in task]
    if missing: raise ValueError('missing addendum inputs: '+', '.join(missing))
    if task.get('mode')!='tht_single' or task.get('platform')!='docker': raise ValueError('explicit single-image Docker mode required')
    if task.get('skip_sdg',False): raise ValueError('generator addendum cannot skip SDG')
    public=Path(task['public_package']).resolve(strict=True); private=Path(task['private_package']).resolve(strict=True)
    root=Path(task['results_dir']).resolve()
    if root.is_relative_to(BANK) or root in (public,private) or public.is_relative_to(private) or private.is_relative_to(public):
        raise ValueError('separate code, results, public and private roots required')
    check_results_root(Path(task['results_dir']))
    if not root.parent.is_dir(): raise ValueError('results parent must exist')
    bundle=json.loads((public/'bundle.json').read_text())
    expected={'role_counts':EXPECTED_COUNTS,'excluded_unique':310,'metadata_repaired':0,
              'known_source_joint_overlaps':2,'split_policy':'preserve_original_Maxwell_splits'}
    if any(bundle.get(k)!=v for k,v in expected.items()): raise ValueError('frozen role membership policy changed')
    external_evaluator(public/'evaluator')
    mining=load_jsonl(public/'mining/samples.jsonl')
    if len(mining)!=3642 or len({r['id'] for r in mining})!=3642: raise ValueError('Mining membership mismatch')
    roles={r:load_jsonl(public/r/'queries.jsonl') for r in ('proxy','benchmark')}
    for role,rows in roles.items():
        validate_queries(rows,count=EXPECTED_COUNTS[role]); validate_identity_grammar(mining,rows)
    proxy_gold=load_jsonl(private/'evaluation/proxy-gold.jsonl')
    if len(proxy_gold)!=74 or {r['id'] for r in proxy_gold}!={r['id'] for r in roles['proxy']} or any(len(r['labels'])!=1 or r['labels'][0] not in CLASSES for r in proxy_gold):
        raise ValueError('Proxy gold membership or single-label support changed')
    task['proxy_class_counts']={c:sum(r['labels']==[c] for r in proxy_gold) for c in CLASSES}
    if any(n<=0 for n in task['proxy_class_counts'].values()): raise ValueError('frozen experiment requires every Proxy class support')

    profile=json.loads(Path(task['direct_profile']).read_text())
    if hashlib.sha256(profile['prompt'].encode()).hexdigest()!=PROMPT_SHA or profile['prompt_sha256']!=PROMPT_SHA:
        raise ValueError('exact baseline prompt mismatch')
    epoch=task['epoch_selection']; epoch_path=Path(epoch['path'])
    if sha256(epoch_path)!=epoch['sha256']: raise ValueError('epoch selection artifact changed')
    chosen=json.loads(epoch_path.read_text())
    if chosen.get('epoch')!=7 or chosen.get('phase')!='proxy' or chosen.get('status')!='complete' or chosen.get('n')!=74:
        raise ValueError('requires the frozen seven-epoch full-Proxy F selection')
    if task.get('max_deft_rounds',5)!=5 or task.get('epochs',7)!=7: raise ValueError('same-D round/epoch controls changed')
    if task.get('training',TRAINING)!=TRAINING: raise ValueError('same-D training controls changed')
    evidence=json.loads(Path(task['m3_evidence']).read_text())
    failed=[g for g in D_GATES if evidence.get(g,{}).get('status')!='passed']
    if failed or evidence.get('training_settings')!=TRAINING: raise ValueError('D native evidence gates incomplete: '+', '.join(failed))
    acceptance=json.loads(Path(task['generator_evidence']).read_text())
    acceptance_files=[]
    for gate in ('native_byo_quality','amended_source_selection','source_data_review'):
        value=acceptance.get(gate,{})
        if value.get('status')!='passed' or not value.get('artifacts'): raise ValueError('generator acceptance incomplete: '+gate)
        for item in value['artifacts']: verify_file(item);acceptance_files.append(item)
    g=task['generator']
    if g.get('policy')!=POLICY: raise ValueError('generator policy must exactly match the frozen amendment')
    prepared=verify_prepared(g['prepared_package'],public)
    if prepared['source_counts']!=g.get('source_counts'): raise ValueError('frozen prepared source population changed')
    import yaml
    recipe=yaml.safe_load(Path(g['recipe']).read_text())
    if set('+'.join(p) for p in recipe['anomaly_types'])!=set(prepared['mapping']): raise ValueError('recipe and external class map differ')
    model=Path(task['model_path']).resolve(strict=True)
    config=json.loads((model/'config.json').read_text())
    provenance=json.loads((model/'tao_conversion_provenance.json').read_text())
    if config['model_type']!='qwen3_vl': raise ValueError('shared converted base required')
    for record in provenance['prepared']['files']:
        p=model/record['name']
        if p.stat().st_size!=record['size'] or sha256(p)!=record['sha256']: raise ValueError('fresh base checkpoint changed')
    for image in (task['framework_image'],task['data_services_image'],g['image']):
        if '@sha256:' not in image: raise ValueError('all images must be digest pinned')
        subprocess.run(['docker','image','inspect',image],check=True,stdout=subprocess.DEVNULL)
    b=task['budgets']
    frozen={'suite_wall_seconds':345600,'stage_wall_seconds':172800,'generator_wall_seconds':7200,'max_attempts':2}
    if b!=frozen: raise ValueError('run/stage/generator/retry envelopes must match the approved protocol')
    storage=task.get('storage_required_bytes')
    if type(storage) is not int or storage<=0: raise ValueError('measured checkpoint and generation retention estimate required')
    if shutil.disk_usage(root.parent).free<storage: raise ValueError('insufficient available space for retained native attempts')
    # Gold content is not opened here. The private checksum manifest is sealed;
    # each role's gold is verified by the trusted evaluator only when admissible.
    files=[file_binding(public/name) for name in ('bundle.json','SHA256SUMS','mining/samples.jsonl','proxy/queries.jsonl','benchmark/queries.jsonl')]
    for line in (public/'SHA256SUMS').read_text().splitlines():
        digest,relative=line.split(maxsplit=1);path=within(public,relative.lstrip('*'))
        binding=file_binding(path)
        if binding['sha256']!=digest: raise ValueError('frozen public package asset changed')
        files.append(binding)
    for record in provenance['prepared']['files']:
        files.append({'path':str((model/record['name']).resolve()),'sha256':record['sha256'],'bytes':record['size']})
    files += [file_binding(private/'SHA256SUMS'),file_binding(private/'evaluation/proxy-gold.jsonl'),file_binding(task['direct_profile']),file_binding(epoch_path),
              file_binding(task['m3_evidence']),file_binding(task['generator_evidence']),*acceptance_files,file_binding(task['environment_manifest']),
              file_binding(g['checkpoint']),file_binding(g['recipe']),file_binding(model/'tao_conversion_provenance.json')]
    for key in ('base_checkpoint','checkpoint_root','base_source'):
        Path(g[key]).resolve(strict=True)
    container_targets=g.get('container_symlink_targets',{})
    for target,value in container_targets.items():
        if not Path(target).is_absolute() or value.get('image')!=g['image']:
            raise ValueError('container-owned link must bind exact absolute path and native image')
        verify_file(value['artifact']);files.append(value['artifact'])
    aliases={'/base-source':g['base_source'],'/workspace/cdf-anomalygen/checkpoints':g['checkpoint_root']}
    aliases.update({m['target']:m['source'] for m in g.get('runtime_mounts',[])})
    trees=[tree_binding(task['model_path']),tree_binding(g['checkpoint']),tree_binding(g['recipe']),tree_binding(task['siglip_path']),tree_binding(g['checkpoint_root'],aliases,container_targets),tree_binding(g['base_source'],aliases,container_targets)]
    if not Path(g['base_checkpoint']).is_relative_to(Path(g['checkpoint_root'])):
        if Path(g['base_checkpoint']).is_dir(): trees.append(tree_binding(g['base_checkpoint'],aliases,container_targets))
        else: trees.append(tree_binding(g['base_checkpoint'],aliases,container_targets))
    for mount in g.get('runtime_mounts',[]):
        source=Path(mount['source']).resolve(strict=True)
        if not Path(mount['target']).is_absolute() or str(source) in ('/','/home','/home/ubuntu') or source.is_relative_to(private) or private.is_relative_to(source):
            raise ValueError('invalid native runtime mount')
        trees.append(tree_binding(mount['source'],aliases,container_targets))
    if set(g.get('environment',{}))-{'IMAGINAIRE_HF_CLI_ENVIRONMENT'}: raise ValueError('unrecognized native environment setting')
    files.append(prepared['inventory'])
    files.extend(validate_cache_provenance(task))
    cache={role:validate_cache(task,role,mining if role=='mining' else roles['proxy']) for role in ('mining','proxy')}
    for info in cache.values(): files.extend(info['files'])
    code={'commit':subprocess.check_output(['git','-C',str(BANK),'rev-parse','HEAD'],text=True).strip(),
          'branch':subprocess.check_output(['git','-C',str(BANK),'branch','--show-current'],text=True).strip()}
    if subprocess.check_output(['git','-C',str(BANK),'status','--porcelain'],text=True).strip(): raise ValueError('formal preflight requires committed clean code')
    import inspect
    from retrieval import cosine
    task['cosine_implementation']={'file':file_binding(inspect.getsourcefile(cosine)),
                                   'function_sha256':hashlib.sha256(inspect.getsource(cosine).encode()).hexdigest()}
    files.append(task['cosine_implementation']['file'])
    environment=json.loads(Path(task['environment_manifest']).read_text())
    if not isinstance(environment.get('host'),str) or not environment['host']: raise ValueError('environment host identity required')
    task['host_python']={'version':sys.version,'executable':str(Path(sys.executable).resolve())}
    task.update(protocol='generator-addendum-v1',max_deft_rounds=5,epochs=7,training=TRAINING,
                statistics=STATISTICS,deft_neighbors=4,icl_reference_capacity=4,
                max_new_tokens=4096,do_sample=False,prompt_sha256=PROMPT_SHA,
                base_fingerprint=provenance['prepared']['fingerprint'],code=code,
                environment=environment,
                mining_class_counts={c:sum(r['labels']==[c] for r in mining) for c in CLASSES if c!='Good'},
                effective_counts=EXPECTED_COUNTS,prepared_inventory=prepared['inventory'],
                class_mapping=prepared['mapping'],class_order=list(prepared['mapping']),bindings={'files':files,'trees':trees},
                scope='engineering-addendum',generator_training_benchmark_exposure='unknown',
                overlap_disclosure='Two original Mining/Benchmark physical joints remain under source alignment policy.')
    return {'ready':True,'effective':task,'preflight_sha256':json_hash(task)}


def verify_resume(task):
    if 'host_python' in task and task['host_python']!={'version':sys.version,'executable':str(Path(sys.executable).resolve())}:
        raise ValueError('approved host Python arithmetic/runtime changed')
    for binding in task['bindings']['files']: verify_file(binding)
    for tree in task['bindings']['trees']: verify_tree(tree)
    verify_prepared(task['generator']['prepared_package'],task['public_package'],expected_inventory=task['prepared_inventory'])
    if subprocess.check_output(['git','-C',str(BANK),'rev-parse','HEAD'],text=True).strip()!=task['code']['commit']:
        raise ValueError('approved code commit changed')
    if subprocess.check_output(['git','-C',str(BANK),'status','--porcelain'],text=True).strip(): raise ValueError('approved code checkout changed')
