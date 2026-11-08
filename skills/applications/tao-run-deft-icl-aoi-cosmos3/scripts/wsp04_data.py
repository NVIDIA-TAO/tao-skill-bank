# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit train/validation views and unchanged nearest-example rendering."""
import hashlib, importlib.util, json
from pathlib import Path
from wsp03_worker import digest

class Data:
    def __init__(self, root='/development', context='/context'):
        self.root,self.context=Path(root),Path(context)
        self.rows={r['id']:r for r in self.jsonl(self.root/'source_rows.jsonl')}
        self.mining=json.loads((self.root/'mining-ids.json').read_text())
        self.proxy=json.loads((self.root/'proxy-ids.json').read_text())
        if len(self.mining)!=3642 or len(self.proxy)!=74 or set(self.mining)&set(self.proxy):
            raise ValueError('WSP04 requires original disjoint Mining3642/Proxy74 identities')
        self.embeddings={r['id']:r['embedding'] for r in self.jsonl(self.root/'embeddings.jsonl')}
        self.prompt=(self.context/'target-prompt.txt').read_text()
        self.instruction=(self.context/'icl-instruction.txt').read_text()
        spec=importlib.util.spec_from_file_location('frozen_retrieval',self.context/'retrieval.py')
        self.retrieval=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.retrieval)
        self.refs={}
    @staticmethod
    def jsonl(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    def nearest(self,qid):
        if qid not in self.refs:
            refs=self.retrieval.allowed_references(self.rows[qid],[self.rows[q] for q in self.mining])
            refs.sort(key=lambda r:(-self.retrieval.cosine(self.embeddings[qid],self.embeddings[r['id']]),r['id']))
            if not refs:raise ValueError('no eligible Mining reference')
            self.refs[qid]=refs[0]['id']
        return self.refs[qid]
    def request(self,qid,icl=False):
        if qid not in self.rows:raise ValueError('query not in explicit train/validation view')
        refs=[self.nearest(qid)] if icl else []
        path=self.rows[qid]['view_path']
        if icl:
            messages=self.retrieval.messages({'parameters':{'instruction':self.instruction}},path,[self.rows[r] for r in refs],self.prompt)
        else:messages=[{'role':'user','content':[{'type':'image','image':path},{'type':'text','text':self.prompt}]}]
        return {'id':qid,'messages':messages,'references':refs,'policy_sha256':digest({'profile':'P01' if icl else 'P00','instruction':self.instruction,'prompt':self.prompt})}
    def gold(self,ids):return [{'id':q,'labels':self.rows[q]['labels']} for q in ids]
