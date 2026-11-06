# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transactional suite journal, immutable answers, budgets and freeze records."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from tht_contract import json_hash, sha256

class SuiteStore:
    allowed_arms = 'ABCDEF'

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root/'suite.sqlite', timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, time REAL NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS stages(name TEXT PRIMARY KEY, status TEXT NOT NULL, result TEXT);
          CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY, stage TEXT NOT NULL, started REAL NOT NULL, ended REAL, status TEXT NOT NULL, job_id TEXT, detail TEXT);
          CREATE TABLE IF NOT EXISTS answers(arm TEXT NOT NULL, phase TEXT NOT NULL, candidate TEXT NOT NULL, query_id TEXT NOT NULL, payload TEXT NOT NULL, sha256 TEXT NOT NULL, PRIMARY KEY(arm,phase,candidate,query_id));
          CREATE TABLE IF NOT EXISTS artifacts(name TEXT PRIMARY KEY, path TEXT NOT NULL, sha256 TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS budgets(name TEXT PRIMARY KEY, started REAL NOT NULL, wall_cap REAL NOT NULL, tokens_cap INTEGER NOT NULL, tokens_used INTEGER NOT NULL DEFAULT 0);
          CREATE TABLE IF NOT EXISTS usage(invocation TEXT PRIMARY KEY, budget TEXT, payload TEXT NOT NULL);
        ''')

    @contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def event(self, kind, payload):
        self.db.execute('INSERT INTO events(time,kind,payload) VALUES(?,?,?)', (time.time(),kind,json.dumps(payload,sort_keys=True)))

    def get(self, key, default=None):
        row=self.db.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone()
        return json.loads(row['value']) if row else default

    def seal(self, key, value):
        existing=self.get(key)
        if existing is not None:
            if existing != value:
                raise ValueError(f'immutable {key} cannot be changed on resume')
            return
        self.db.execute('INSERT INTO meta VALUES(?,?)',(key,json.dumps(value,sort_keys=True)))

    def initialize(self, config, approval):
        if not approval or approval['preflight_sha256'] != json_hash(config):
            raise ValueError('one explicit approval must bind the effective preflight')
        with self.transaction():
            self.seal('config',config)
            self.seal('approval',approval)
            self.seal('protocol','tht-six-arm-v1')

    def stage(self,name):
        row=self.db.execute('SELECT * FROM stages WHERE name=?',(name,)).fetchone()
        return dict(row) if row else {'name':name,'status':'pending','result':None}

    def begin(self, stage, dependencies=(), *, max_attempts=2):
        with self.transaction():
            for parent in dependencies:
                if self.stage(parent)['status'] != 'complete':
                    raise ValueError(f'{stage} requires completed {parent}')
            if self.stage(stage)['status'] == 'complete':
                raise ValueError('completed stage is immutable')
            rows=self.db.execute('SELECT * FROM attempts WHERE stage=? ORDER BY id',(stage,)).fetchall()
            if rows and rows[-1]['status'] in ('pending','running'):
                raise ValueError('reconcile the live attempt using backend status before resubmitting')
            if len(rows) >= max_attempts:
                raise ValueError('bounded recovery exhausted')
            if rows and rows[-1]['status'] != 'infrastructure_error':
                raise ValueError('only infrastructure failures may be retried')
            cur=self.db.execute('INSERT INTO attempts(stage,started,status) VALUES(?,?,?)',(stage,time.time(),'pending'))
            self.db.execute('INSERT INTO stages VALUES(?,?,NULL) ON CONFLICT(name) DO UPDATE SET status=excluded.status',(stage,'running'))
            self.event('attempt_started',{'attempt':cur.lastrowid,'stage':stage})
            return cur.lastrowid

    def attach_job(self, attempt, job_id):
        with self.transaction():
            row=self.db.execute('SELECT job_id,status FROM attempts WHERE id=?',(attempt,)).fetchone()
            if not row or row['status'] not in ('pending','running') or row['job_id'] not in (None,job_id):
                raise ValueError('cannot replace attempt job handle')
            self.db.execute('UPDATE attempts SET job_id=?,status=? WHERE id=?',(job_id,'running',attempt))

    def finish(self, attempt, status, detail):
        if status not in ('complete','infrastructure_error','program_error','budget_exhausted'):
            raise ValueError('unknown terminal attempt status')
        with self.transaction():
            row=self.db.execute('SELECT * FROM attempts WHERE id=?',(attempt,)).fetchone()
            if not row or row['status'] not in ('pending','running'):
                raise ValueError('attempt already terminal or missing')
            self.db.execute('UPDATE attempts SET status=?,ended=?,detail=? WHERE id=?',(status,time.time(),json.dumps(detail,sort_keys=True),attempt))
            self.db.execute('UPDATE stages SET status=?,result=? WHERE name=?',('complete' if status=='complete' else 'incomplete',json.dumps(detail,sort_keys=True),row['stage']))
            self.event('attempt_finished',{'attempt':attempt,'status':status,'detail':detail})

    def record_answer(self, arm, phase, candidate, answer):
        if arm not in self.allowed_arms or len(arm)!=1 or phase not in ('proxy','benchmark','smoke'):
            raise ValueError('unknown answer role')
        if phase=='benchmark':
            self.verify_freeze()
        if not isinstance(answer.get('raw_answer'),str) or not answer.get('completed'):
            raise ValueError('only completed raw responses enter immutable answers')
        # INVALID and length-limited answers are completed too. No parser here.
        payload=json.dumps(answer,sort_keys=True)
        with self.transaction():
            key=(arm,phase,candidate,answer['id'])
            prior=self.db.execute('SELECT payload FROM answers WHERE arm=? AND phase=? AND candidate=? AND query_id=?',key).fetchone()
            if prior:
                if prior['payload'] != payload:
                    raise ValueError('cannot overwrite or retry a completed answer')
                return
            self.db.execute('INSERT INTO answers VALUES(?,?,?,?,?,?)',(*key,payload,json_hash(answer)))

    def answers(self,arm,phase,candidate):
        return [json.loads(r['payload']) for r in self.db.execute('SELECT payload FROM answers WHERE arm=? AND phase=? AND candidate=? ORDER BY query_id',(arm,phase,candidate))]

    def artifact(self,name,path):
        path=Path(path).resolve(strict=True)
        value=(str(path),sha256(path))
        row=self.db.execute('SELECT path,sha256 FROM artifacts WHERE name=?',(name,)).fetchone()
        if row:
            if tuple(row)!=value:
                raise ValueError(f'artifact changed: {name}')
        else:
            self.db.execute('INSERT INTO artifacts VALUES(?,?,?)',(name,*value))
        return dict(path=value[0],sha256=value[1])

    def start_budget(self,name,wall_seconds,tokens):
        if wall_seconds<=0 or tokens<=0:
            raise ValueError('positive frozen budgets required')
        with self.transaction():
            row=self.db.execute('SELECT * FROM budgets WHERE name=?',(name,)).fetchone()
            if row:
                if row['wall_cap']!=wall_seconds or row['tokens_cap']!=tokens:
                    raise ValueError('resume cannot reset or expand budget')
            else:
                self.db.execute('INSERT INTO budgets(name,started,wall_cap,tokens_cap) VALUES(?,?,?,?)',(name,time.time(),wall_seconds,tokens))

    def remaining(self,name):
        row=self.db.execute('SELECT * FROM budgets WHERE name=?',(name,)).fetchone()
        if not row: raise ValueError('budget not started')
        return {'wall_seconds':max(0,row['wall_cap']-(time.time()-row['started'])), 'tokens':max(0,row['tokens_cap']-row['tokens_used'])}

    def record_usage(self,invocation,payload,*,budget=None):
        with self.transaction():
            previous=self.db.execute('SELECT * FROM usage WHERE invocation=?',(invocation,)).fetchone()
            serialized=json.dumps(payload,sort_keys=True)
            if previous:
                if previous['payload'] != serialized or previous['budget'] != budget:
                    raise ValueError('invocation usage is immutable')
                return
            if budget:
                if payload.get('input_tokens') is None or payload.get('output_tokens') is None:
                    raise ValueError('unknown agent usage blocks further budgeted search')
                if not self.db.execute('SELECT 1 FROM budgets WHERE name=?',(budget,)).fetchone():
                    raise ValueError('missing budget')
                self.db.execute('UPDATE budgets SET tokens_used=tokens_used+? WHERE name=?',(payload['input_tokens']+payload['output_tokens'],budget))
            self.db.execute('INSERT INTO usage VALUES(?,?,?)',(invocation,budget,serialized))

    def freeze(self, settings):
        if set(settings) != set('ABCDEF'):
            raise ValueError('all six settings must freeze together')
        if settings['D']['checkpoint'] != settings['E']['checkpoint']:
            raise ValueError('E must use exactly the selected D checkpoint')
        if any(settings[a]['base_fingerprint'] != settings['B']['base_fingerprint'] for a in 'CDEF'):
            raise ValueError('B-F must share the same base lineage')
        for a in 'ACDEF':
            if self.stage(a+'/select')['status']!='complete':
                raise ValueError(f'{a} selection is incomplete')
        payload={'settings':settings,'sha256':json_hash(settings)}
        with self.transaction():
            self.seal('freeze',payload)
            self.event('all_arms_frozen',payload)
        return payload

    def verify_freeze(self):
        freeze=self.get('freeze')
        if not freeze or json_hash(freeze['settings'])!=freeze['sha256']:
            raise ValueError('Benchmark is forbidden before all-arm freeze')
        for row in self.db.execute('SELECT * FROM artifacts'):
            if sha256(row['path'])!=row['sha256']:
                raise ValueError(f"sealed artifact changed: {row['name']}")
        for value in freeze['settings'].values():
            manifest=value.get('model_manifest')
            if manifest:
                for file in json.loads(Path(manifest['path']).read_text()):
                    stat=Path(file['path']).stat()
                    if stat.st_size!=file['size'] or stat.st_mtime_ns!=file['mtime_ns']:
                        raise ValueError('frozen model file changed')
        return freeze

    def completed_benchmark(self,expected_ids):
        self.verify_freeze()
        for arm in 'ABCDEF':
            rows=self.answers(arm,'benchmark','final')
            if {r['id'] for r in rows} != set(expected_ids) or len(rows)!=len(expected_ids):
                return False
        return True
