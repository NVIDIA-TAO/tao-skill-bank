# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted dispatcher binding tool requests to frozen membership and durable debit.

The editor-facing dispatch exposes no trusted phase/usage/approval mutation.
Request rendering, generation grading and worker launching are trusted adapters;
scientific launch requires an explicit review receipt matching the contract.
"""
from __future__ import annotations
import json
from pathlib import Path
import uuid
import time
from wsp03_state import AdmissionError
from wsp03_worker import digest,validate_requests


def read_evidence_records(payload):
    """Content identities independent of request IDs, order or batch partition."""
    records=[]
    for row in payload.get('raw',[]):
        record={k:row[k] for k in ('references','forced_text','correct_class','teacher_forced','capture_sha256','diagnostic_only') if k in row}
        record.update(query_id=row.get('query_id',row['id']))
        records.append(digest(record))
    return sorted(set(records))


class Service:
    def __init__(self, *, ledger, contract, kernel, request_factory, worker, grader, edit_root, approval=None):
        self.ledger=ledger;self.contract=contract;self.kernel=kernel;self.request_factory=request_factory
        self.worker=worker;self.grader=grader;self.edit_root=Path(edit_root)
        self.approved=bool(approval and approval.get('scientific_launch_approved') is True and approval.get('contract_sha256')==digest(contract))
        with ledger.tx():
            old=ledger.get('service_contract_sha256')
            if old is not None and old!=digest(contract):raise ValueError('frozen service contract changed')
            ledger.set('service_contract_sha256',digest(contract))
        self.members={name:set(ids) for name,ids in contract['splits'].items()}
        if self.members['development'] & self.members['holdout']:raise ValueError('split leakage')
        if not self.members['explore'] <= self.members['development']:raise ValueError('explore outside development')

    def require_launch(self):
        if getattr(self,'guard',None):self.guard()
        if not self.approved:raise AdmissionError('scientific launch remains gated until matching preflight approval')

    def dispatch(self, action, params):
        self.require_launch()
        from wsp03_contract import OPERATIONS
        if action in OPERATIONS:
            schema=OPERATIONS[action]
            if not isinstance(params,dict) or set(params)-set(schema['required']+schema['optional']) or set(schema['required'])-set(params):raise ValueError('atomic operation arguments differ from frozen schema')
        if action in ('data.list','context.read','reference.candidates'):return self.metadata(action,params)
        if action=='status':
            from wsp03_progress import progress
            return dict(self.ledger.status(),science_progress=progress(self),active_checkpoint=self.contract.get('active_checkpoint'),baseline_evidence=self.ledger.get('baseline_evidence'))
        if action=='help':
            from wsp03_contract import manifest
            reference=json.loads((Path(__file__).with_name('wsp03_operation_reference.json')).read_text())
            return reference['operations'][params['operation']] if params.get('operation') else manifest()
        if action in ('python','tensor.read','reference.attribute','edit.family','edit.direction','edit.rank1','edit.memit','edit.head_scale','edit.commit','edit.replay','edit.revert','export.checkpoint'):
            if self.contract.get('weights_frozen') and action not in ('tensor.read','reference.attribute','edit.replay','export.checkpoint'):raise AdmissionError('track weights are immutable')
            return self.numeric(action,params)
        if action in ('activation.read','activation.patch'):return self.forward('read',params)
        if action=='eval.dev':return self.forward('explore',params)
        if action=='read':return self.forward('read',params)
        if action=='explore':return self.forward('explore',params)
        raise AdmissionError('editor action not exposed')

    def metadata(self,action,params):
        if action=='data.list':
            split=params['split']
            if split not in ('development','explore'):raise AdmissionError('only development metadata is editor-visible')
            start=params.get('offset',0);limit=params.get('limit',32)
            if type(start)is not int or start<0 or type(limit)is not int or not 1<=limit<=64:raise ValueError('bounded metadata page required')
            ids=self.contract['splits'][split];selected=ids[start:start+limit]
            return {'rows':[{'id':qid,'labels':self.request_factory.rows[qid]['labels']} for qid in selected],'total':len(ids),'next_offset':start+len(selected) if start+len(selected)<len(ids) else None}
        if action=='reference.candidates':
            qid=params['query_id']
            if qid not in self.contract['splits']['explore']:raise ValueError('attribution query outside explore')
            return {'query_id':qid,'references':self.request_factory.attribution_candidates(qid)}
        assets=dict(self.contract.get('context_assets',{}),**getattr(self,'result_assets',{}))
        if not params.get('name'):return {'names':sorted(assets)}
        if params['name'] not in assets:raise ValueError('context file outside frozen allowlist')
        from wsp03_bindings import pinned
        asset=assets[params['name']];original=Path(asset['path'])
        if original.is_symlink():raise ValueError('context symlink forbidden')
        path=pinned(**asset);start=params.get('offset',0);limit=params.get('limit',4096)
        if type(start)is not int or start<0 or type(limit)is not int or not 1<=limit<=8192:raise ValueError('bounded context slice required')
        raw=path.read_bytes();return {'name':params['name'],'text':raw[start:start+limit].decode('utf-8',errors='replace'),'total_bytes':len(raw),'next_offset':start+limit if start+limit<len(raw) else None,'sha256':asset['sha256']}

    def numeric(self, action, params):
        self.require_launch();rid=uuid.uuid4().hex
        import ast
        edits_in_python=action=='python' and any(isinstance(node,ast.Name) and node.id in {'register_edit_family','select_edit_configuration','save_edit','commit_candidate'} for node in ast.walk(ast.parse(params['code'])))
        if (edits_in_python or action.startswith('edit.') and action not in ('edit.replay','edit.revert')) and self.contract.get('diagnostic_contract'):
            from wsp03_progress import progress
            if not progress(self).get('editing_ready'):raise AdmissionError('complete frozen attribution and mechanism diagnostics before editing')
        if self.contract.get('weights_frozen') and action=='tensor.read':params=dict(params,candidate=self.contract['fixed_worker_candidate'])
        numeric_identity={'operation':action,'parameters_sha256':digest(params)}
        if action=='reference.attribute':
            try:params=self.attribution(params)
            except Exception as error:
                raise
        from contextlib import nullcontext
        import math
        resources=getattr(self.kernel.runtime,'resources',None)
        budget=None
        if resources:
            account=self.kernel.runtime.resource_account
            budget=max(0,resources.available(account)-64*1024**2)
            if budget<=0:raise AdmissionError('retained storage admission closed')
        reservation=resources.reserve(budget,'numeric cell artifacts',account) if resources else nullcontext()
        with reservation:
            self.ledger.admit('python',rid,numeric_identity)
            if resources:
                used=resources.size_roots([self.edit_root])
                limit=min(250*1024**3,used+budget)
                result=self.kernel.call({'op':'limits','storage_bytes':int(limit),'cell_seconds':max(1,min(600,int(self.ledger.editing_seconds())))})
                if not result.get('ok'):raise RuntimeError('trusted kernel limits failed')
            if action in ('python','edit.family') and self.contract.get('diagnostic_contract'):
                evidence={}
                for read_id,payload in self.ledger.db.execute("SELECT id,payload FROM requests WHERE kind='read' AND state='complete'"):
                    value=json.loads(payload)
                    if value.get('split') not in ('explore','development'):continue
                    evidence[read_id]=read_evidence_records(value)
                bound=self.kernel.call({'op':'evidence','lineage':evidence})
                if not bound.get('ok'):raise RuntimeError('trusted mechanism evidence binding failed')
            # An uncertain transport failure deliberately leaves the debit running.
            response=self.kernel.call({'op':'python','code':params['code']} if action=='python' else {'op':'atomic','operation':action,'parameters':params})
            self.ledger.finish(rid,{'complete':response.get('ok') is True,'response':response})
            for path in ([] if self.contract.get('weights_frozen') else sorted(self.edit_root.glob('*/manifest.json'))):
                from wsp03_artifacts import child_json
                manifest=child_json(self.edit_root,path.relative_to(self.edit_root),owner=self.kernel.runtime.uid);cid=manifest['id']
                if self.ledger.db.execute('SELECT 1 FROM candidates WHERE id=?',(cid,)).fetchone():continue
                proof=self.kernel.call({'op':'verify','candidate':cid})
                if proof.get('ok') is not True:raise ValueError('candidate replay proof failed')
                self.ledger.candidate(cid,manifest)
            return response
    def attribution(self,params):
        converted=[];observations=self.ledger.get('attribution_pairs') or {}
        for entry in params['records']:
            qid,ref=entry['query_id'],entry['reference_id']
            if qid not in self.members['explore'] or ref not in self.members['development']:raise ValueError('attribution must use frozen explore and dev references')
            if hasattr(self.request_factory,'attribution_candidates') and ref not in self.request_factory.attribution_candidates(qid):raise ValueError('reference outside preregistered attribution candidates')
            matched=[]
            for request_id,wanted_refs in ((entry['baseline_request'],[]),(entry['reference_request'],[ref])):
                row=self.ledger.db.execute('SELECT kind,state,payload FROM requests WHERE id=?',(request_id,)).fetchone()
                if not row or row[0]!='read' or row[1]!='complete':raise ValueError('attribution source is not a complete read')
                payload=json.loads(row[2])
                if payload.get('candidate')!='base' or payload.get('profile','P00')!='P00' or payload.get('split') not in ('explore','development'):raise ValueError('attribution only from base dev reads')
                values=[r for r in payload['raw'] if r.get('query_id',r['id'])==qid and r.get('references',[])==wanted_refs and not r.get('diagnostic_only')]
                if len(values)!=1 or not values[0].get('correct_class'):raise ValueError('unique measured unpatched correct-class read required')
                matched.append(values[0])
            if matched[0]['forced_text']!=matched[1]['forced_text']:raise ValueError('paired canonical class targets differ')
            z,r=matched[0]['correct_class']['mean_logprob'],matched[1]['correct_class']['mean_logprob']
            measured={'query_id':qid,'reference_id':ref,'baseline_request':entry['baseline_request'],'reference_request':entry['reference_request'],'delta':r-z}
            key=digest({'query_id':qid,'reference_id':ref})
            if key in observations and observations[key]!=measured:raise ValueError('attribution pair already bound; cannot substitute another result')
            observations[key]=measured
            converted.append({'query':qid,'reference':ref,'zero_shot_logprob':z,'reference_logprob':r,'read_ids':[entry['baseline_request'],entry['reference_request']]})
        with self.ledger.tx():self.ledger.set('attribution_pairs',observations);self.ledger.event('attribution_bound',{'count':len(converted),'observations_sha256':digest(observations)})
        return {'records':converted}

    def requests(self, mode, params, *, trusted=False):
        split=params.get('split','explore');records=params.get('records')
        if records is not None:
            if mode!='read' or not records or len(records)>32:raise ValueError('only reads accept up to32 condition records')
            ids=[r['query_id'] for r in records]
        else:
            ids=params.get('ids',[])
            records=[{'query_id':qid,'references':params.get('references',{}).get(qid,[]),'capture':params.get('capture',{})} for qid in ids]
            if len(set(ids))!=len(ids):raise ValueError('duplicate IDs require explicit read condition records')
        if split not in self.members or not ids or not set(ids)<=self.members[split]:raise ValueError('invalid split membership')
        if mode=='read':
            if split not in ('development','explore') or len(records)>32:raise AdmissionError('bounded dev-only read')
        elif ids!=self.contract['splits'][split]:raise ValueError('generation must cover exact ordered frozen membership')
        if split in ('holdout','benchmark') and not trusted:raise AdmissionError('trusted split inaccessible to editor')
        frozen_refs=self.contract.get('generation_references')
        pool=self.members['development'] if split!='benchmark' else set(self.contract['benchmark_reference_ids'])
        rows=[]
        for index,record in enumerate(records):
            qid=record['query_id'];refs=record.get('references',[])
            if mode!='read' and hasattr(self.request_factory,'references'):
                expected=self.request_factory.references(qid,params.get('profile','P00'))
                if refs and refs!=expected:raise ValueError('caller reference override differs from bound strategy')
                refs=expected
            if mode!='read' and frozen_refs is not None:
                expected=frozen_refs.get(qid,[])
                if refs and refs!=expected:raise ValueError('generation references differ from frozen policy')
                refs=expected
            if len(refs)!=len(set(refs)) or not set(refs)<=pool or len(refs)>self.contract['max_references']:raise ValueError('reference pool/count violation')
            for field in ('joints','source_ids','image_hashes','components'):
                identities=self.contract.get(field)
                if identities and any(identities[qid]==identities[ref] for ref in refs):raise ValueError(field+' reference exclusion')
            row=(self.request_factory.render(qid,refs,mode,params.get('profile','P00')) if hasattr(self.request_factory,'render') else self.request_factory(qid,refs,mode))
            if row['id']!=qid or row.get('references',[])!=refs:raise ValueError('trusted renderer identity mismatch')
            if mode=='read':
                capture=json.loads(json.dumps(record.get('capture',{})))
                for patch in capture.get('patches',[]):
                    if 'source_capture' in patch or 'source_file' in patch:raise ValueError('internal capture routing fields cannot be supplied')
                    if 'source' not in patch:continue
                    source=patch.pop('source')
                    if set(source)-{'request_id','record_id','key','scale'} or not {'request_id','record_id','key'}<=set(source):raise ValueError('invalid source capture reference')
                    prior=self.ledger.db.execute('SELECT kind,state,payload FROM requests WHERE id=?',(source['request_id'],)).fetchone()
                    if not prior or prior[:2]!=('read','complete'):raise ValueError('patch source must be a completed charged read')
                    payload=json.loads(prior[2]);values=[r for r in payload['raw'] if r['id']==source['record_id']]
                    if payload.get('split') not in ('development','explore') or len(values)!=1 or values[0].get('query_id',values[0]['id'])!=qid or source['key'] not in values[0].get('capture',{}):raise ValueError('patch source query/capture mismatch')
                    if values[0].get('forced_text')!=row['forced_text']:raise ValueError('patch source canonical class differs')
                    if 'values' in patch:raise ValueError('choose captured source or explicit values')
                    patch['source_capture']=dict(source,file=values[0]['capture_file'])
                row=dict(row,capture=capture)
                if params.get('records') is not None:row=dict(row,id=qid+'@'+str(index),query_id=qid)
            rows.append(row)
        validate_requests(rows,'read' if mode=='read' else 'generation',split)
        return rows

    def forward(self, kind, params, *, trusted=False):
        self.require_launch()
        if kind not in ('read','explore','baseline','full','trusted'):raise ValueError('unknown forward purpose')
        if kind in ('baseline','full','trusted') and not trusted:raise AdmissionError('trusted scheduler operation')
        profile=params.get('profile','P00');candidate=params.get('candidate',profile if self.contract.get('weights_frozen') else 'base');split=params.get('split','explore')
        if self.contract.get('weights_frozen'):
            if profile!=candidate or profile not in self.contract.get('profile_registry',[]):raise AdmissionError('frozen track requires exact registered profile identity')
        elif kind in ('explore','full','trusted') and profile!='P00':raise AdmissionError('edited R3 generation is zero-shot only')
        if kind=='full' and (split!='development' or candidate not in (self.ledger.get('shortlist') or [])):raise AdmissionError('full development restricted to frozen finalists')
        if kind=='explore' and split!='explore':raise AdmissionError('candidate exploration restricted to512')
        if kind=='trusted':
            eligible={self.ledger.get('selected')}
            if self.contract.get('weights_frozen'):eligible.update(self.contract.get('trusted_ablation_profiles',[]))
            if split not in ('holdout','benchmark') or candidate not in eligible:raise AdmissionError('only frozen primary or registered fixed-weight ablation on trusted split')
        if kind=='baseline' and (split!='explore' or candidate!='base' or profile not in ('P00','P01','P02','P03','B04')):raise AdmissionError('baseline on frozen explore only')
        if candidate!='base' and not self.ledger.db.execute('SELECT 1 FROM candidates WHERE id=?',(candidate,)).fetchone():raise ValueError('unregistered candidate')
        if self.contract.get('weights_frozen') and candidate=='P00' and kind in ('explore','full') and getattr(self,'reuse_source',None):
            from wsp03_final import reuse_generation
            return reuse_generation(self,self.reuse_source,kind=kind,candidate='P00',profile='P00',split=split,source_candidate=self.reuse_source.ledger.get('selected'),source_profile='P00')
        forward_started=time.monotonic()
        rows=self.requests('read' if kind=='read' else 'generation',params,trusted=trusted)
        rid=uuid.uuid4().hex;identity={'candidate':candidate,'profile':profile,'split':split,'requests_sha256':digest(rows),'query_forward_exposures':len(rows)}
        if kind=='read':identity['conditions']=[{'query_id':row.get('query_id',row['id']),'references':row.get('references',[]),'capture':row.get('capture',{})} for row in rows]
        if kind in ('explore','baseline','full'):
            prior=self.ledger.db.execute("SELECT 1 FROM requests WHERE kind=? AND json_extract(payload,'$.candidate')=? AND json_extract(payload,'$.profile')=? AND json_extract(payload,'$.split')=?",(kind,candidate,profile,split)).fetchone()
            if prior:raise AdmissionError('candidate/profile/split already attempted; no search by retry')
        if kind=='trusted':
            old=self.ledger.db.execute("SELECT 1 FROM requests WHERE (kind='trusted' OR json_extract(payload,'$.reused_kind')='trusted') AND json_extract(payload,'$.split')=? AND json_extract(payload,'$.profile')=?",(split,profile)).fetchone()
            if old:raise AdmissionError('trusted split already attempted')
        self.ledger.admit(kind,rid,identity)
        actual_candidate=self.contract.get('fixed_worker_candidate',candidate) if self.contract.get('weights_frozen') else candidate
        result=self.worker(kind,rows,actual_candidate,split,rid)
        result['request_id']=rid
        complete=result.get('complete') is True and result.get('ids')==[r['id'] for r in rows] and result.get('requests_sha256')==digest(rows)
        if complete and kind!='read' and result.get('raw') and all('wall_seconds' in row for row in result['raw']):
            queries=[row['wall_seconds'] for row in result['raw']]
            native=result.get('native_job_wall_seconds',sum(queries))
            preparation=max(0,time.monotonic()-forward_started-native)
            self.ledger.observe_generation_timing(query_seconds=queries,setup_seconds=max(0,native-sum(queries)),preparation_seconds=preparation,evidence={'request_id':rid,'includes_reference_selection_and_view_copy':True})
        if complete and kind=='baseline':result['metrics']=self.grader(result,split)
        self.ledger.finish(rid,dict(result,complete=complete,**identity))
        if not complete:raise ValueError('partial/mismatched worker result cannot be ranked')
        if kind in ('explore','full'):
            metrics=self.grader(result,split)
            if self.contract.get('weights_frozen'):metrics['mean_reference_count']=sum(len(r.get('references',[])) for r in rows)/len(rows)
            self.ledger.result(candidate,kind,metrics)
            if not trusted:return {'complete':True,'candidate':candidate,'split':split,'metrics':metrics,'request_id':rid}
        if kind=='read' and not trusted:
            visible=('id','query_id','references','correct_class','teacher_forced','class_span','forced_target_tokens','diagnostic_only','capture','capture_file')
            return {'complete':True,'request_id':rid,'records':[{key:row[key] for key in visible if key in row} for row in result['raw']]}
        return result
