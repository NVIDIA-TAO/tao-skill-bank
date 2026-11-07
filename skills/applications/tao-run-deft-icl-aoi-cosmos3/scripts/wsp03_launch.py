# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concrete reviewed-config entrypoint for the two-track frozen experiment.

`preflight` binds code, private development recipe, context and all route config
without scientific work. `run` additionally requires an external approval whose
config/code fingerprints match. It never silently promotes a preflight record.
"""
from __future__ import annotations
import argparse,json,os,subprocess,time
from pathlib import Path
from contextlib import ExitStack
from wsp03_bindings import Recipe,TrustedGrader,NativeWorker,pinned,file_hash,PROFILES
from wsp03_worker import digest
from wsp03_contract import manifest
from wsp03_editor import run_editor,PROTOCOL
from wsp03_supervisor import Supervisor,canonical
from wsp03_state import Ledger
from wsp03_service import Service
from wsp03_runtime import DedicatedDocker,IsolatedKernel,compute_mode,set_compute_mode,APP
from wsp03_resources import GPULease
from wsp03_limits import AggregateLimits
from wsp03_final import finalize_development,register_reference_profiles,evaluate_final_arms
from wsp03_final_builder import FinalInputBuilder
from wsp03_artifacts import stage_candidate,stage_export,child_json
from wsp03_io import json_write


def register_lease_storage(limits,lease):
    """Charge only this lease's concrete shared files, never older coordination."""
    paths=[lease.path,lease.receipt,lease.trace,Path(str(lease.archive)+'.lineage.jsonl'),Path(str(lease.archive)+'.receipt.json')]
    for account in ('R3','R4'):limits.add_roots(account,paths)
    return list(map(str,paths))


def load_context(asset,track):
    path=pinned(**asset);value=json.loads(path.read_text());assets=value['context_assets']
    names=value['initial_files_by_track'][track];texts={name:pinned(**assets[name]).read_text() for name in names}
    task_key=next(name for name in names if name.startswith('editor-'+track))
    task=texts.pop(task_key)
    for name in list(texts):
        if name.startswith('editor-protocol'):texts.pop(name)
    contract_name=next(n for n in names if n.startswith('tool_contract') or n.startswith('tool-contract'))
    if json.loads(pinned(**assets[contract_name]).read_text())['contract_sha256']!=manifest()['contract_sha256']:raise ValueError('runtime tool contract differs from frozen context')
    # Counter-size differences are accounted again using the exact live payload.
    prototype={'immutable_task':task,'context':texts,'protocol':PROTOCOL,'prior_handoff':'','recent_dialogue':[],'trusted_counters':{},'global_round':1,'handoff_required':False}
    if len(canonical(prototype).encode())>46000:raise ValueError('context leaves insufficient serialized counter margin within48k')
    return task,texts,assets,{'manifest_sha256':asset['sha256'],'initial_payload_without_counters_bytes':len(canonical(prototype).encode()),'native_envelope_allowance_bytes':32768}


def load_config(path,expected):
    config=json.loads(pinned(path,expected).read_text())
    if config.get('schema')!='wsp03-run/1' or config['editor']['output_reservation']!=128000:raise ValueError('unsupported run contract/output reservation')
    if config['uid'] in (0,os.getuid()) or config['track_limits']!={'memory_bytes':128*1024**3,'storage_bytes':250*1024**3}:raise ValueError('dedicated UID/exact aggregate limits required')
    required={p.name for p in (APP/'scripts').glob('wsp03*.py')}|{'wsp03_operation_reference.json','suite_runtime.py','suite_data.py','tht_contract.py','suite_analysis.py','suite_costs.py'}
    if not required<=set(config['source_files']):raise ValueError('required runtime source inventory incomplete')
    for name in required:
        asset=config['source_files'][name]
        if Path(asset['path']).resolve()!=(APP/'scripts'/name).resolve():raise ValueError('source receipt does not bind actually loaded checkout')
        pinned(**asset)
    for field in ('access_policy','editing_policy'):pinned(**config[field])
    components=config['schedule_components']
    import math
    if set(components)!={'generation_seconds_per_query','setup_seconds_per_pass','reference_cpu_seconds_per_query','final_overhead_seconds'} or any(not math.isfinite(v) or v<0 for v in components.values()) or components['final_overhead_seconds']!=600:raise ValueError('explicit frozen additive timing components required')
    pinned(**config['schedule_evidence'])
    contexts={track:load_context(config['context'],track) for track in ('R3','R4')}
    recipe=Recipe(config['recipe']['path'],config['recipe']['sha256'])
    from wsp03_inputs import InputGuard
    input_guard=InputGuard(config['expected_inputs'],recipe=recipe,editor=config['editor'],final_inputs=config['final_inputs'])
    if config['architecture']['query_ids_sha256']!=digest(sorted(recipe.members['explore'])[:64]):raise ValueError('diagnostic subset differs')
    return config,recipe,contexts,input_guard


def preflight(path,expected,output):
    config,recipe,contexts,input_guard=load_config(path,expected)
    record={'schema':'wsp03-preflight-binding/1','run_config_sha256':expected,'source_files':config['source_files'],'context':{k:v[3] for k,v in contexts.items()},'tool_contract_sha256':manifest()['contract_sha256'],'development_count':len(recipe.members['development']),'explore_count':len(recipe.members['explore']),'final_routes':'lazy after both successful freezes, stopped kernels and live joint reserve','editor_output_reservation':128000,'scientific_launch_approved':False,'forward_calls':0,'provider_calls':0,'input_binding':input_guard.receipt}
    json_write(output,record);return record


def run(path,expected,approval_path,approval_sha256):
    config,recipe3,contexts,input_guard=load_config(path,expected);approval=json.loads(pinned(approval_path,approval_sha256).read_text())
    if approval.get('scientific_launch_approved') is not True or approval.get('run_config_sha256')!=expected or approval.get('source_files')!=config['source_files']:raise RuntimeError('matching external preflight review PASS required')
    root=Path(config['output']);root.mkdir(mode=0o700,parents=True,exist_ok=False)
    ledgers={};kernels=[];services={};lease=None;lease_entered=False;limits=None;previous=None;status='incomplete';timeline={};cleanup_errors=[]
    try:
        limits=AggregateLimits(root/'resources',accounts={track:[root/track,root/'frozen-r3',root/'resources',root/'run-terminal.json',root/'r3-freeze.json',root/'termination.json',root/'final-arm-results.json',Path(config['editor']['tracks'][track]['codex_home']),Path(config['editor']['tracks'][track]['workdir']),*([root/'frozen-hf-export'] if track=='R3' else []),*map(Path,config['accounted_immutable_roots'])] for track in ('R3','R4')},**config['track_limits'])
        limits.enter()
        lease=GPULease(config['lease_path'],'wsp03-science-'+root.name,gpu=config['gpu']);lease.__enter__();lease_entered=True
        register_lease_storage(limits,lease)
        previous=compute_mode(config['gpu'])
        if previous!='Exclusive_Process':set_compute_mode(config['gpu'],'Exclusive_Process')
        def guard():
            input_guard.check()
            if lease.violation:raise RuntimeError('GPU lease violation; stop owned work')
            if limits.error:raise RuntimeError(limits.error)
        started=time.time();joint_deadline=started+86400-config['probe_debit']['wall_seconds']
        frozen_manifest=None;frozen_root=None;helpfulness=None
        for track in ('R3','R4'):
            timeline[track]={'started_at':time.time()}
            tr=root/track;tr.mkdir(mode=0o700);spec=config['tracks'][track]
            recipe=recipe3 if track=='R3' else Recipe(config['recipe']['path'],config['recipe']['sha256'])
            ledger=Ledger(tr/'ledger.db',track=track,caps=spec['caps'],probe_debit=config['probe_debit'] if track=='R3' else None);ledgers[track]=ledger
            ledger.start_clock();ledger.set_finalist_reserve(spec['full_reserve_seconds'],spec['explore_reserve_seconds'])
            ledger.bind_joint_schedule(deadline=joint_deadline,downstream_seconds=spec['downstream_seconds'],evidence=config['schedule_evidence'])
            final_count=config['final_inputs']['prepare']['holdout_count']+config['final_inputs']['prepare']['benchmark_count']
            full_size=len(recipe.members['development']);explore_size=len(recipe.members['explore'])
            cpu_final=2*config['final_inputs']['prepare']['holdout_count']+2*config['final_inputs']['prepare']['benchmark_count']*config['final_inputs']['prepare']['mining_count']/full_size
            r4_development=12*explore_size+3*full_size if track=='R3' else 0
            components=config['schedule_components']
            ledger.set_schedule_model(queries=3*final_count+r4_development,cpu_queries=cpu_final+r4_development,passes=6+(15 if track=='R3' else 0),fixed_seconds=components['final_overhead_seconds'],initial_gpu_rate=components['generation_seconds_per_query'],initial_setup_seconds=components['setup_seconds_per_pass'],cpu_seconds_per_query=components['reference_cpu_seconds_per_query'])
            if track=='R4':ledger.inherit_timing(ledgers['R3'])
            runtime=DedicatedDocker(tr/'jobs',config['image'],uid=config['uid'],gid=config['gid'],gpus='device='+str(config['gpu']),resources=limits,resource_account=track)
            read_root=tr/'read-artifacts';read_root.mkdir(mode=0o750);subprocess.run(['sudo','-n','chgrp',str(config['gid']),str(read_root)],check=True)
            kernel=IsolatedKernel(runtime=runtime,base=recipe.config['model_directory'],output=tr/'edits',exchange=tr/'exchange',captures=read_root,access_policy=config['access_policy']['path'],editing_policy=config['editing_policy']['path']);kernels.append(kernel)
            contract=recipe.contract();contract['context_assets']=contexts[track][2]
            architecture=config['architecture']
            if track=='R3':contract['diagnostic_contract']={'query_ids':sorted(recipe.members['explore'])[:64],**{k:architecture[k] for k in ('layers','head_modules','head_count','head_dimension')}}
            if track=='R4':
                from wsp03_bindings import ReferenceSelector
                recipe.selector=ReferenceSelector(recipe.rows,recipe.selector.embeddings,helpfulness=helpfulness['helpfulness'],atlas=helpfulness['atlas'],pool=recipe.members['development']);recipe.helpfulness_frozen=True
                contract.update(weights_frozen=True,profile_registry=list(PROFILES),trusted_ablation_profiles=['P01'],fixed_worker_candidate=frozen_manifest['id'],active_checkpoint=digest(frozen_manifest))
                # R4 tensor reads need the frozen patch in its isolated edit mount.
                import shutil
                with limits.reserve(sum(item['elements']*2 for item in frozen_manifest['tensors'])+1024**2,'R4 readonly tensor view','R4'):
                    stage_candidate(frozen_root,frozen_manifest['id'],tr/'edits'/frozen_manifest['id'],owner=os.getuid(),group=config['gid'],expected_manifest=frozen_manifest)
            edit_root=tr/'edits' if track=='R3' else frozen_root
            worker=NativeWorker(recipe,runtime,lease,ledger,edit_root,tr/'forwards',artifact_owner=config['uid'] if track=='R3' else os.getuid(),read_root=read_root,frozen_manifest=frozen_manifest)
            service=Service(ledger=ledger,contract=contract,kernel=kernel,request_factory=recipe,worker=worker,grader=TrustedGrader(recipe),edit_root=tr/'edits',approval={'scientific_launch_approved':True,'contract_sha256':digest(contract)})
            service.guard=guard;services[track]=service
            runtime_context=dict(contexts[track][1])
            if track=='R4':
                service.result_assets=dict(services['R3'].result_assets)
                runtime_context['development-evidence']={'baseline_summary_asset':'phase0-baselines.json','available_via':'context.read','scope':'new frozen development evidence'}
                service.reuse_source=services['R3']
                register_reference_profiles(ledger,checkpoint_identity=worker.checkpoint_identity(frozen_manifest['id']))
            else:
                for profile in ('P00','P01','P02','P03','B04'):
                    service.forward('baseline',{'candidate':'base','profile':profile,'split':'explore','ids':recipe.members['explore']},trusted=True)
            if track=='R4':
                import shutil
                source_auth=Path(config['editor']['tracks']['R3']['codex_home'])/'auth.json';target_auth=Path(config['editor']['tracks']['R4']['codex_home'])/'auth.json'
                if source_auth.is_symlink() or target_auth.is_symlink():raise ValueError('private auth symlink forbidden')
                shutil.copyfile(source_auth,target_auth);target_auth.chmod(0o600)
            if track=='R3':
                from wsp03_dev_evidence import publish_baselines
                runtime_context['development-evidence']=publish_baselines(service,tr/'baseline-evidence')
            editor=dict(config['editor']['tracks'][track]);editor.update(binary=config['editor']['binary'],model=config['editor']['model'],effort=config['editor']['effort'],output_reservation=128000,output_bound_evidence=config['editor']['output_bound_evidence'],expected_binary_sha256=config['editor']['expected_binary_sha256'],expected_bound_files=config['editor']['expected_bound_files'],output=tr/'editor')
            supervisor=Supervisor(**editor,guard=guard)
            run_editor(service=service,supervisor=supervisor,immutable_task=contexts[track][0],context=runtime_context)
            selected=finalize_development(service)
            if selected is None:
                status='no_valid_patch';json_write(root/'termination.json',{'track':track,'status':status,'ledger':ledger.status()});return {'status':status}
            if track=='R3':
                helpfulness=recipe.freeze_helpfulness_from_ledger(ledger,tr/'helpfulness.json')
                frozen_manifest=json.loads(ledger.db.execute('SELECT payload FROM candidates WHERE id=?',(selected,)).fetchone()[0])
                # Export is trusted post-selection serialization, not another edit/search.
                base_bytes=sum(p.stat().st_size for p in Path(recipe.config['model_directory']).iterdir() if p.is_file())
                patch_bytes=sum(item['elements']*2 for item in frozen_manifest['tensors'])
                with limits.reserve(int(2*base_bytes+3*patch_bytes+64*1024**2),'trusted HF export and frozen copies','R3'):
                    exported=kernel.call({'op':'atomic','operation':'export.checkpoint','parameters':{'candidate':selected}})
                    if not exported.get('ok'):raise RuntimeError('trusted checkpoint export failed')
                    kernel.close()
                    frozen_root=root/'frozen-r3';frozen_root.mkdir(mode=0o750);subprocess.run(['sudo','-n','chgrp',str(config['gid']),str(frozen_root)],check=True)
                    stage_candidate(tr/'edits',selected,frozen_root/selected,owner=config['uid'],group=config['gid'],expected_manifest=frozen_manifest)
                    stage_export(tr/'edits',selected,exported['result']['result'],root/'frozen-hf-export',owner=config['uid'],group=config['gid'])
                service.worker.edit_root=frozen_root;service.worker.artifact_owner=os.getuid();service.worker.frozen_manifest=frozen_manifest
                json_write(root/'r3-freeze.json',{'selected':selected,'manifest':frozen_manifest,'checkpoint_identity':worker.checkpoint_identity(selected),'export':exported,'helpfulness_sha256':file_hash(tr/'helpfulness.json')})
            else:kernel.close()
            timeline[track]['frozen_at']=time.time()
            json_write(tr/'freeze.json',{'selected':selected,'status':ledger.status(),'active_checkpoint':worker.checkpoint_identity(frozen_manifest['id'])})
        data_runtime=DedicatedDocker(root/'R4/final-data-jobs',config['data_image'],uid=config['uid'],gid=config['gid'],gpus='device='+str(config['gpu']),resources=limits,resource_account='R4')
        builder=FinalInputBuilder(config['final_inputs'],runtime=services['R4'].worker.runtime,data_runtime=data_runtime,lease=lease,output=root/'R4/final-inputs')
        results=evaluate_final_arms(services['R3'],services['R4'],builder,root,required_seconds=config['final_inputs']['minimum_remaining_seconds'])
        from wsp03_scoring import final_statistics
        statistics=final_statistics(services['R3'],services['R4'],results,config['final_statistics'],root/'R4/statistics')
        status='evaluated';return {'status':status,'arms':list(results),'primary_outcomes':statistics['primary_outcomes']}
    finally:
        for kernel in kernels:
            try:kernel.close()
            except BaseException as error:cleanup_errors.append({'stage':'kernel_close','type':type(error).__name__})
        if limits is not None:
            try:limits.cancel_owned()
            except BaseException as error:cleanup_errors.append({'stage':'owned_job_cancel','type':type(error).__name__})
        lease_receipt=None
        if lease is not None and lease_entered:
            try:
                from wsp03_resources import gpu_processes
                if previous is not None:
                    if gpu_processes(config['gpu']):cleanup_errors.append({'stage':'compute_mode_restore','reason':'GPU remains occupied; do not interfere with foreign process'})
                    else:set_compute_mode(config['gpu'],previous)
            except BaseException as error:cleanup_errors.append({'stage':'compute_mode_restore','type':type(error).__name__})
            try:
                lease.__exit__(None,None,None)
            except BaseException as error:cleanup_errors.append({'stage':'lease_release','type':type(error).__name__})
            finally:
                if hasattr(lease,'archive'):lease_receipt=str(lease.archive)+'.receipt.json'
        if limits is not None:
            try:limits.close()
            except BaseException as error:cleanup_errors.append({'stage':'resource_close','type':type(error).__name__})
        waiting=None
        if timeline.get('R3',{}).get('frozen_at') and timeline.get('R4',{}).get('frozen_at'):waiting={'started_at':timeline['R3']['frozen_at'],'ended_at':timeline['R4']['frozen_at'],'seconds':timeline['R4']['frozen_at']-timeline['R3']['frozen_at'],'charged_to_continuous_R3_clock':True}
        json_write(root/'run-terminal.json',{'status':status,'cleanup_errors':cleanup_errors,'ledgers':{k:v.status() for k,v in ledgers.items()},'track_timeline':timeline,'R3_waiting_for_R4':waiting,'gpu_lease_receipt':lease_receipt,'aggregate_resources':str(root/'resources/aggregate-resources.json'),'ended_at':time.time()})
        for ledger in ledgers.values():ledger.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['preflight','run']);p.add_argument('--config',required=True);p.add_argument('--config-sha256',required=True);p.add_argument('--output');p.add_argument('--approval');p.add_argument('--approval-sha256');a=p.parse_args()
    result=preflight(a.config,a.config_sha256,a.output) if a.action=='preflight' else run(a.config,a.config_sha256,a.approval,a.approval_sha256)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
