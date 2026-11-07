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
from wsp03_state import AdmissionError
from wsp03_worker import digest,validate_requests


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
        if not self.approved:raise AdmissionError('scientific launch remains gated until matching preflight approval')

    def dispatch(self, action, params):
        self.require_launch()
        if action=='status':return self.ledger.status()
        if action=='python':return self.python(params['code'])
        if action=='read':return self.forward('read',params)
        if action=='explore':return self.forward('explore',params)
        raise AdmissionError('editor action not exposed')

    def python(self, code):
        self.require_launch();rid=uuid.uuid4().hex
        self.ledger.admit('python',rid,{'code_sha256':digest(code)})
        # An uncertain transport failure deliberately leaves the debit running.
        response=self.kernel.call({'op':'python','code':code})
        self.ledger.finish(rid,{'complete':response.get('ok') is True,'response':response})
        for path in sorted(self.edit_root.glob('*/manifest.json')):
            manifest=json.loads(path.read_text());cid=manifest['id']
            if self.ledger.db.execute('SELECT 1 FROM candidates WHERE id=?',(cid,)).fetchone():continue
            proof=self.kernel.call({'op':'verify','candidate':cid})
            if proof.get('ok') is not True:raise ValueError('candidate replay proof failed')
            self.ledger.candidate(cid,manifest)
        return response

    def requests(self, mode, params, *, trusted=False):
        split=params.get('split','explore');ids=params.get('ids',[])
        if split not in self.members or not ids or len(set(ids))!=len(ids) or not set(ids)<=self.members[split]:raise ValueError('invalid split membership')
        if mode=='read':
            if split not in ('development','explore') or len(ids)>32:raise AdmissionError('bounded dev-only read')
        elif set(ids)!=self.members[split]:raise ValueError('generation must cover exact frozen membership')
        if split in ('holdout','benchmark') and not trusted:raise AdmissionError('trusted split inaccessible to editor')
        references=params.get('references',{})
        frozen_refs=self.contract.get('generation_references')
        if mode!='read' and frozen_refs is not None:
            expected={qid:frozen_refs.get(qid,[]) for qid in ids}
            if references and references!=expected:raise ValueError('generation references differ from frozen policy')
            references=expected
        if set(references)-set(ids):raise ValueError('unrelated reference query')
        pool=self.members['development'] if split!='benchmark' else set(self.contract['benchmark_reference_ids'])
        rows=[]
        for qid in ids:
            refs=references.get(qid,[])
            if len(refs)!=len(set(refs)) or not set(refs)<=pool or len(refs)>self.contract['max_references']:raise ValueError('reference pool/count violation')
            qjoint=self.contract['joints'][qid]
            if any(self.contract['joints'][ref]==qjoint for ref in refs):raise ValueError('same-joint reference excluded')
            row=self.request_factory(qid,refs,mode)
            if row['id']!=qid or row.get('references',[])!=refs:raise ValueError('trusted renderer identity mismatch')
            if mode=='read':row=dict(row,capture=params.get('capture',{}))
            rows.append(row)
        validate_requests(rows,'read' if mode=='read' else 'generation',split)
        return rows

    def forward(self, kind, params, *, trusted=False):
        self.require_launch()
        if kind not in ('read','explore','baseline','full','trusted'):raise ValueError('unknown forward purpose')
        if kind in ('baseline','full','trusted') and not trusted:raise AdmissionError('trusted scheduler operation')
        candidate=params.get('candidate','base');split=params.get('split','explore')
        if kind=='full' and (split!='development' or candidate not in (self.ledger.get('shortlist') or [])):raise AdmissionError('full development restricted to frozen finalists')
        if kind=='explore' and split!='explore':raise AdmissionError('candidate exploration restricted to512')
        if kind=='trusted' and (split not in ('holdout','benchmark') or candidate!=self.ledger.get('selected')):raise AdmissionError('only frozen candidate on trusted split')
        if kind=='baseline' and (split!='explore' or candidate!='base'):raise AdmissionError('baseline on frozen explore only')
        if candidate!='base' and not self.ledger.db.execute('SELECT 1 FROM candidates WHERE id=?',(candidate,)).fetchone():raise ValueError('unregistered candidate')
        rows=self.requests('read' if kind=='read' else 'generation',params,trusted=trusted)
        rid=uuid.uuid4().hex;identity={'candidate':candidate,'split':split,'requests_sha256':digest(rows)}
        if kind=='trusted':
            old=self.ledger.db.execute("SELECT 1 FROM requests WHERE kind='trusted' AND json_extract(payload,'$.split')=?",(split,)).fetchone()
            if old:raise AdmissionError('trusted split already attempted')
        self.ledger.admit(kind,rid,identity)
        result=self.worker(kind,rows,candidate,split,rid)
        complete=result.get('complete') is True and result.get('ids')==[r['id'] for r in rows] and result.get('requests_sha256')==digest(rows)
        self.ledger.finish(rid,dict(result,complete=complete,**identity))
        if not complete:raise ValueError('partial/mismatched worker result cannot be ranked')
        if kind in ('explore','full'):
            metrics=self.grader(result,split)
            self.ledger.result(candidate,kind,metrics)
        return result
