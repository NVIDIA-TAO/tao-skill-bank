# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Read-only H intake. All scientific settings originate in the frozen E task.
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from astra_isolation import generic_mounts
from generator_contract import file_binding, verify_file, within
from generator_preflight import tree_binding, verify_tree, STATISTICS
from icl_store import PROTOCOL, check_root
from suite_data import validate_identity_grammar, identity_from_query
from suite_preflight import EXPECTED_COUNTS, PROMPT_SHA
from suite_runtime import BANK
from tht_contract import external_evaluator, json_hash, load_jsonl, sha256, validate_queries

BUDGETS = {'suite_wall_seconds':345600, 'stage_wall_seconds':172800,
           'icl_wall_seconds':21600, 'icl_agent_tokens':1000000,
           'query_wall_seconds':180, 'query_agent_tokens':100000, 'trial_admission_seconds':1400}


def validate_controls(task):
    if task['budgets'] != BUDGETS:
        raise ValueError('H must retain every frozen E budget')
    expected = {'icl_reference_capacity':4, 'max_icl_trials':12, 'max_new_tokens':4096, 'do_sample':False}
    if any(task.get(k, v) != v for k,v in expected.items()) or task['training']['seed'] != 42:
        raise ValueError('H inference/search controls differ from E')
    if task.get('statistics', STATISTICS) != STATISTICS:
        raise ValueError('paired statistics protocol changed')


def baseline_metrics(binding):
    context = json.loads(verify_file(binding).read_text())
    if (context.get('arm') != 'E' or context.get('trials') != [] or
            context.get('max_trials') != 12 or context.get('reference_capacity') != 4):
        raise ValueError('baseline must come from the frozen first E proposal context')
    metrics = context['baseline_metrics']
    if metrics.get('n') != 74 or 'predictions' in metrics or 'policy' in metrics:
        raise ValueError('only the original B Proxy74 aggregate is permitted')
    return metrics


def verify_checkpoint(source, base):
    selection = json.loads(verify_file(source['selection']).read_text())
    model = Path(source['checkpoint']).resolve(strict=True)
    if (selection.get('checkpoint') != str(model) or selection.get('phase') != 'proxy' or
            selection.get('status') != 'complete' or selection.get('n') != 74):
        raise ValueError('fixed checkpoint differs from the sealed complete Proxy selection')
    manifest = json.loads(verify_file(source['model_manifest']).read_text())
    if (not manifest or {r['path'] for r in manifest} != {str(p) for p in model.iterdir() if p.is_file()}
            or any(not p.is_file() or p.is_symlink() for p in model.iterdir())):
        raise ValueError('fixed model manifest membership differs')
    for row in manifest:
        path = Path(row['path'])
        if (path.parent != model or path.stat().st_size != row['size'] or
                path.stat().st_mtime_ns != row['mtime_ns'] or sha256(path) != row['sha256']):
            raise ValueError('fixed checkpoint bytes differ from original model manifest')
    exported = json.loads((model / 'export_manifest.json').read_text())
    lineage = exported.get('base_model_fingerprint', {}).get('files', {})
    if not lineage:
        raise ValueError('native exported model lacks original base lineage')
    for relative,digest in lineage.items():
        if sha256(within(base, relative)) != digest:
            raise ValueError('fixed model original base lineage differs')
    return str(model)


def validate_output_root(root, protected):
    root = Path(root)
    if not root.is_absolute() or root.is_symlink() or root.resolve() != root:
        raise ValueError('results must be an absolute canonical unsymlinked path')
    for value in protected:
        source = Path(value).resolve()
        if root == source or root.is_relative_to(source) or source.is_relative_to(root):
            raise ValueError('results must be disjoint from source and historical results')
    if not root.parent.is_dir():
        raise ValueError('results parent must already exist')


def code_identity():
    if subprocess.check_output(['git','-C',str(BANK),'status','--porcelain'],text=True).strip():
        raise ValueError('formal H preflight requires clean committed code')
    return {key:subprocess.check_output(['git','-C',str(BANK),*args],text=True).strip()
            for key,args in {'commit':['rev-parse','HEAD'],'branch':['branch','--show-current']}.items()}


def validate(request):
    required = {'protocol','reference_task','reference_preflight','baseline_context','source_model',
                'results_dir','environment_manifest','storage_required_bytes','comparison_runs'}
    if required - request.keys() or request.get('protocol') != PROTOCOL:
        raise ValueError('explicit icl-addendum-v1 intake bindings are required')
    if set(request) - required - {'setup_token_exports','task_binding'}:
        raise ValueError('unrecognized H overrides; scientific controls are inherited from E')
    original = json.loads(verify_file(request['reference_task']).read_text())
    preflight = json.loads(verify_file(request['reference_preflight']).read_text())
    old = preflight['effective']
    if not preflight.get('ready') or preflight['preflight_sha256'] != json_hash(old):
        raise ValueError('original E preflight hash differs')
    for key in ('mode','platform','skip_sdg','public_package','private_package','model_path','framework_image',
                'data_services_image','direct_profile','siglip_path','siglip_revision','budgets','training',
                'icl_reference_capacity','max_icl_trials','m3_evidence'):
        if original[key] != old[key]:
            raise ValueError('frozen E task/preflight mismatch: ' + key)
    if any(old['astra'].get(k) != v for k,v in original['astra'].items()):
        raise ValueError('E policy runtime differs from frozen task')
    validate_controls(old)
    if old['mode'] != 'tht_single' or old['platform'] != 'docker' or not old['skip_sdg']:
        raise ValueError('only fixed single-image Docker ICL is supported')
    if set(request['comparison_runs']) != {'G','E','F'}:
        raise ValueError('freeze G primary and E/F secondary comparator roots')
    public = Path(old['public_package']).resolve(strict=True)
    private = Path(old['private_package']).resolve(strict=True)
    model = Path(old['model_path']).resolve(strict=True)
    source = request['source_model']
    protected = [BANK, public, private, model, old['siglip_path'], old['results_dir'],
                 Path(source['model_manifest']['path']).parent, *request['comparison_runs'].values()]
    validate_output_root(Path(request['results_dir']), protected)
    check_root(Path(request['results_dir']))
    if public.is_relative_to(private) or private.is_relative_to(public):
        raise ValueError('public/private role isolation changed')
    provenance = old['provenance']
    paths = {public/'bundle.json':'bundle_sha256', public/'SHA256SUMS':'public_checksums_sha256',
             Path(old['direct_profile']):'direct_profile_sha256', Path(old['m3_evidence']):'m3_evidence_sha256',
             model/'tao_conversion_provenance.json':'model_conversion_sha256'}
    files = [request['reference_task'],request['reference_preflight'],request['baseline_context'],
             source['selection'],source['model_manifest']]
    for path,key in paths.items():
        if sha256(path) != provenance[key]:
            raise ValueError('original E source drift: ' + key)
        files.append(file_binding(path))
    for name,digest in provenance['siglip_files'].items():
        path = within(old['siglip_path'],name)
        if sha256(path) != digest:
            raise ValueError('original E encoder drift')
        files.append(file_binding(path))
    # This public checksum inventory is frozen by the original E preflight.
    # Benchmark queries are label-free; private Benchmark gold is never opened.
    for line in (public/'SHA256SUMS').read_text().splitlines():
        digest,relative = line.split(maxsplit=1)
        path = within(public,relative.lstrip('*'))
        if sha256(path) != digest:
            raise ValueError('frozen public data asset changed')
        files.append(file_binding(path))
    bundle = json.loads((public/'bundle.json').read_text())
    if bundle['role_counts'] != EXPECTED_COUNTS or bundle['known_source_joint_overlaps'] != 2:
        raise ValueError('frozen source population changed')
    mining = load_jsonl(public/'mining/samples.jsonl')
    if len(mining) != 3642 or len({r['id'] for r in mining}) != 3642:
        raise ValueError('only the exact real Mining3642 pool is permitted')
    for role in ('proxy','benchmark'):
        queries = load_jsonl(public/role/'queries.jsonl')
        validate_queries(queries,count=EXPECTED_COUNTS[role])
        validate_identity_grammar(mining,queries)
        files.append(file_binding(public/role/'queries.jsonl'))
    identities = [identity_from_query(q) for q in queries]
    if len({(r['object_name'],r['scan_id']) for r in identities}) != 255:
        raise ValueError('Benchmark physical joint grouping differs from frozen255')
    files += [file_binding(public/'mining/samples.jsonl'),file_binding(private/'SHA256SUMS'),
              file_binding(private/'evaluation/proxy-gold.jsonl')]
    external_evaluator(public/'evaluator')
    profile = json.loads(Path(old['direct_profile']).read_text())
    if hashlib.sha256(profile['prompt'].encode()).hexdigest() != PROMPT_SHA:
        raise ValueError('original query prompt drift')
    conversion = json.loads((model/'tao_conversion_provenance.json').read_text())
    if conversion['prepared']['fingerprint'] != old['base_fingerprint']:
        raise ValueError('original model lineage drift')
    for row in conversion['prepared']['files']:
        path = within(model,row['name'])
        if path.stat().st_size != row['size'] or sha256(path) != row['sha256']:
            raise ValueError('original base bytes changed')
        files.append(file_binding(path))
    checkpoint = verify_checkpoint(source,model)
    astra = old['astra']
    if (subprocess.check_output([astra['codex_binary'],'--version'],text=True).strip() != 'codex-cli 0.166.0'
            or sha256(astra['codex_binary']) != astra['codex_binary_sha256']):
        raise ValueError('policy Codex runtime differs from original E')
    if not Path(astra['auth_file']).is_file():
        raise ValueError('approved private auth mount unavailable')
    images = [old['framework_image'],old['data_services_image'],astra['image']]
    for image in images:
        if '@sha256:' not in image:
            raise ValueError('immutable image required')
        subprocess.run(['docker','image','inspect',image],check=True,stdout=subprocess.DEVNULL)
    mounts = generic_mounts(astra['codex_binary'],astra['python_binary'],astra['python_stdlib'],
                            astra['system_libraries'],astra.get('site_packages',[]))
    trees = [tree_binding(path) for path,_,_ in mounts]
    trees += [tree_binding(checkpoint),tree_binding(old['siglip_path'])]
    environment = json.loads(verify_file(request['environment_manifest']).read_text())
    if not environment.get('host'):
        raise ValueError('current environment host identity required')
    files.append(request['environment_manifest'])
    if request.get('task_binding'):
        verify_file(request['task_binding']);files.append(request['task_binding'])
    required_bytes = request['storage_required_bytes']
    if type(required_bytes) is not int or required_bytes <= 0:
        raise ValueError('measured H retention storage bound required')
    free = shutil.disk_usage(Path(request['results_dir']).parent).free
    if free < required_bytes:
        raise ValueError('insufficient H retention storage')
    # Keep approved scientific controls, but replace old run scope and costs.
    task = {k:copy.deepcopy(old[k]) for k in ('mode','platform','skip_sdg','public_package','private_package',
            'model_path','framework_image','data_services_image','direct_profile','siglip_path','siglip_revision',
            'astra','training','budgets','icl_reference_capacity','max_icl_trials','max_new_tokens','do_sample',
            'base_fingerprint','prompt_sha256','statistics','budget_policy','provenance','overlap_disclosure')}
    task.update(protocol=PROTOCOL,results_dir=request['results_dir'],fixed_checkpoint=checkpoint,
                source_model=source,baseline_metrics=baseline_metrics(request['baseline_context']),
                baseline_context=request['baseline_context'],reference_task=request['reference_task'],
                reference_preflight=request['reference_preflight'],code=code_identity(),environment=environment,
                environment_manifest=request['environment_manifest'],effective_counts=EXPECTED_COUNTS,benchmark_count=271,
                comparison_runs=request['comparison_runs'],setup_token_exports=request.get('setup_token_exports',[]),
                bindings={'files':files,'trees':trees},protected_roots=[str(p) for p in protected],
                host_python={'version':sys.version,'executable':str(Path(sys.executable).resolve())},
                resource_estimates={'required_bytes':required_bytes,'available_bytes':free,
                    'retrieval':'fresh H cache; cold setup and all retrieval allocations included in actual H costs'},
                scope='engineering-addendum',generator_training_benchmark_exposure='unknown',
                retrieval_cache={'mode':'fresh','historical_cache_reuse':False})
    return {'ready':True,'effective':task,'preflight_sha256':json_hash(task)}


def verify_resume(task):
    validate_controls(task)
    validate_output_root(Path(task['results_dir']),task['protected_roots'])
    check_root(Path(task['results_dir']),resume=True)
    if task['host_python'] != {'version':sys.version,'executable':str(Path(sys.executable).resolve())}:
        raise ValueError('host Python changed since approval')
    if code_identity() != task['code']:
        raise ValueError('code changed since approval')
    for binding in task['bindings']['files']:
        verify_file(binding)
    for tree in task['bindings']['trees']:
        verify_tree(tree)
    verify_checkpoint(task['source_model'],task['model_path'])
    if baseline_metrics(task['baseline_context']) != task['baseline_metrics']:
        raise ValueError('permitted B Proxy context changed')


def verify_gold(task, phase):
    # Called only by trusted execution, after complete inference and the final
    # freeze gate for Benchmark. Never copied into role-limited model mounts.
    if phase not in ('proxy','benchmark'):
        raise ValueError('unknown trusted scoring role')
    private = Path(task['private_package'])
    for binding in task.get('bindings',{}).get('files',[]):
        if Path(binding['path']) == (private/'SHA256SUMS').resolve():
            verify_file(binding)
    relative = f'evaluation/{phase}-gold.jsonl'
    checksums = {parts[1].lstrip('*'):parts[0] for line in (private/'SHA256SUMS').read_text().splitlines()
                 if (parts := line.split(maxsplit=1))}
    if relative not in checksums or sha256(within(private,relative)) != checksums[relative]:
        raise ValueError('trusted scoring gold differs from the sealed private manifest')
