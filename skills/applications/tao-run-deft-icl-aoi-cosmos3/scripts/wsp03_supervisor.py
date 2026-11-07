# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted single-response Codex supervisor; editor tools never inherit this process.

Only exec's stdout event pipe supplies usage. No session files or sandbox-written
counters are consulted. The configured output reservation is an audited upper
bound, not a claim that this CLI sends a max_output_tokens API field.
"""
from __future__ import annotations
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import uuid


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def usage_values(value):
    names=('input_tokens','cached_input_tokens','output_tokens')
    if not isinstance(value,dict) or any(type(value.get(n)) is not int or value[n]<0 for n in names):
        raise ValueError('missing or malformed native usage')
    if value['cached_input_tokens']>value['input_tokens']:raise ValueError('cached input is a subset')
    return {n:value[n] for n in names}


COMPATIBILITY_NOTICE='Code Mode is unavailable because code-mode host is disabled. Code mode will fail closed; enable `features.code_mode_host` and install `codex-code-mode-host`.'


def parse_events(events):
    if not isinstance(events,list) or not events:raise ValueError('native stream missing')
    for key in ('thread.started','turn.started','turn.completed'):
        if sum(e.get('type')==key for e in events)!=1:raise ValueError('exactly one native response required: '+key)
    if events[-1].get('type')!='turn.completed':raise ValueError('native terminal usage must be final')
    text=[];notices=[]
    for event in events:
        kind=event.get('type')
        if kind in ('thread.started','turn.started','turn.completed'):continue
        if kind not in ('item.started','item.updated','item.completed'):raise ValueError('native error or unsupported event')
        item=event.get('item',{});item_type=item.get('type')
        if item_type not in ('agent_message','reasoning'):raise ValueError('built-in tool or unsupported item was emitted')
        if item_type=='agent_message' and kind=='item.completed':text.append(item['text'])
    if len(text)!=1:raise ValueError('one completed assistant response required')
    return {'text':text[0],'usage':usage_values(events[-1].get('usage')),'thread_id':events[0].get('thread_id'),'notices':notices}


BLOCKS={'python','read','forward','save_edit','freeze','handoff','tool'}

def parse_blocks(text,*,require_handoff=False):
    matches=list(re.finditer(r'^```([a-z_.]+)\s*\n(.*?)^```\s*$',text,re.M|re.S))
    if text.count('```')!=2*len(matches):raise ValueError('malformed or nested block protocol')
    blocks=[{'name':m[1],'body':m[2].strip()} for m in matches]
    if any(b['name'] not in BLOCKS for b in blocks):raise ValueError('tool block outside allowlist')
    if require_handoff and sum(b['name']=='handoff' for b in blocks)!=1:raise ValueError('fortieth round requires exactly one handoff')
    if sum(b['name']=='handoff' for b in blocks)>1:raise ValueError('ambiguous handoff')
    if sum(b['name']=='freeze' for b in blocks)>1:raise ValueError('ambiguous freeze')
    if any(b['name']=='freeze' for b in blocks[:-1]):raise ValueError('freeze must be terminal; no subsequent operations or handoff')
    return blocks


class RoundBook:
    """Owner-only O_APPEND journal with durable single-round admission and replay."""
    def __init__(self,path,*,soft_limit=4500000,hard_limit=5000000):
        self.path=Path(path);self.path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        if self.path.is_symlink():raise ValueError('ledger symlink forbidden')
        self.soft=soft_limit;self.hard=hard_limit
        fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_APPEND|os.O_NOFOLLOW,0o600);os.close(fd)
        st=self.path.stat()
        if st.st_uid!=os.getuid() or st.st_mode&0o077:raise ValueError('ledger must be supervisor-owned and private')
        with self.transaction():
            if not self.rows:self.append('contract',{'soft':soft_limit,'hard':hard_limit})
            elif self.rows[0]['data']!={'soft':soft_limit,'hard':hard_limit}:raise ValueError('cannot reset frozen token ceilings')
    @contextmanager
    def transaction(self):
        with self.path.open('r+',encoding='utf-8') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            self.rows=[];previous='0'*64
            for line in lock:
                row=json.loads(line);value={k:row[k] for k in ('seq','prev','kind','data','time')}
                if row['seq']!=len(self.rows)+1 or row['prev']!=previous or row['sha']!=sha(canonical(value).encode()):raise ValueError('usage journal chain corrupted')
                self.rows.append(row);previous=row['sha']
            yield
    def append(self,kind,data):
        value={'seq':len(self.rows)+1,'prev':self.rows[-1]['sha'] if self.rows else '0'*64,'kind':kind,'data':data,'time':time.time()}
        row=dict(value,sha=sha(canonical(value).encode()))
        fd=os.open(self.path,os.O_WRONLY|os.O_APPEND|os.O_NOFOLLOW)
        try:os.write(fd,(canonical(row)+'\n').encode());os.fsync(fd)
        finally:os.close(fd)
        self.rows.append(row)
    def _status(self):
        totals={'input_tokens':0,'cached_input_tokens':0,'output_tokens':0};admitted={};completed={};unknown=[];not_dispatched=set()
        for row in self.rows:
            d=row['data']
            if row['kind']=='admit':admitted[d['id']]=d
            elif row['kind']=='complete':
                completed[d['id']]=d
                for name in totals:totals[name]+=d['usage'][name]
            elif row['kind']=='interrupt':unknown.append(d['id'])
            elif row['kind']=='not_dispatched':not_dispatched.add(d['id'])
        used=totals['input_tokens']+totals['output_tokens']
        return {**totals,'used':used,'admitted':admitted,'completed':completed,'unresolved':sorted(set(admitted)-set(completed)-not_dispatched),'unknown':unknown,'provider_usage_complete':not(set(admitted)-set(completed)-not_dispatched),'budget_overrun':used>self.hard,'soft_stopped':used>=self.soft,'closed':any(r['kind']=='closed' for r in self.rows),'rounds':len(admitted)}
    def status(self):
        with self.transaction():return self._status()
    def not_dispatched(self,identity,reason):
        with self.transaction():
            if identity not in self._status()['unresolved']:raise ValueError('no pending admission to cancel')
            self.append('not_dispatched',{'id':identity,'reason':reason,'provider_requests':0})

    def admit(self,identity,input_bound,output_bound):
        if any(type(n) is not int or n<0 for n in (input_bound,output_bound)):raise ValueError('integer nonnegative bounds required')
        with self.transaction():
            s=self._status()
            if s['closed'] or identity in s['admitted'] or s['unresolved']:raise RuntimeError('round already admitted or unresolved provider work')
            if s['used']+input_bound+output_bound>self.soft:raise RuntimeError('round upper bound exceeds soft admission limit')
            self.append('admit',{'id':identity,'input_bound':input_bound,'output_bound':output_bound})
    def complete(self,identity,usage,*,events_sha256):
        usage=usage_values(usage)
        with self.transaction():
            s=self._status()
            if identity not in s['unresolved'] or identity in s['unknown']:raise RuntimeError('not a live provider round')
            self.append('complete',{'id':identity,'usage':usage,'events_sha256':events_sha256})
            bounds=s['admitted'][identity]
            if usage['input_tokens']>bounds['input_bound'] or usage['output_tokens']>bounds['output_bound']:
                self.append('closed',{'reason':'actual usage exceeded admitted upper bound','id':identity})
    def close(self,reason):
        with self.transaction():self.append('closed',{'reason':reason})
    def interrupt(self,identity,reason):
        with self.transaction():
            if identity not in self._status()['unresolved']:raise RuntimeError('not an unresolved provider round')
            self.append('interrupt',{'id':identity,'reason':reason,'total_usage':'unknown'})


def dispatch_blocks(service,blocks):
    feedback=[];handoff=None;freeze=False
    for block in blocks:
        name,body=block['name'],block['body']
        if name=='handoff':handoff=body;continue
        if name=='freeze':freeze=True;continue
        if name=='python':action,params='python',{'code':body}
        else:
            value=json.loads(body)
            if name=='tool':
                if set(value)!={'action','parameters'}:raise ValueError('tool envelope requires action,parameters')
                action,params=value['action'],value['parameters']
            else:action,params={'read':'activation.read','forward':'eval.dev','save_edit':'edit.commit'}[name],value
        # This call validates fixed schemas and dispatches only to the isolated
        # numeric kernel or trusted native model worker; never host eval/exec.
        try:result=service.dispatch(action,params);feedback.append({'action':action,'result':result})
        except Exception as error:feedback.append({'action':action,'error_type':type(error).__name__,'error':str(error)})
    return feedback,handoff,freeze


def validate_request_shape(body,model,effort):
    if body.get('model')!=model or body.get('reasoning',{}).get('effort')!=effort:raise ValueError('actual model/effort changed')
    def visit(value):
        if isinstance(value,dict):
            if value.get('type')=='additional_tools' or value.get('tools') or value.get('type') in ('function_call','custom_tool_call'):raise ValueError('advertised or called tool in actual request')
            for item in value.values():visit(item)
        elif isinstance(value,list):
            for item in value:visit(item)
    visit(body)
    return {'no_tools':True,'serialized_bytes':len(canonical(body).encode()),'model':model,'effort':effort}


def write_codex_config(home,*,source_config,bundled_catalog,model,effort,neutral_instructions,fixture_provider=None,production_provider=None):
    """Freeze upstream no-tools policy plus required catalog-era tool switches.

    A supported client provider alias preserves the reviewed official endpoint
    and native auth while disabling retries. The reserved built-in provider ID
    cannot be overridden by this CLI. Local fake routes are fixture-only.
    """
    import copy,tomllib
    home=Path(home);home.mkdir(mode=0o700,parents=True,exist_ok=True)
    source=Path(source_config).read_text();parsed=tomllib.loads(source)
    if parsed.get('web_search')!='disabled' or any(v is not False for v in parsed['features'].values()):raise ValueError('upstream no-tool policy required')
    entry=copy.deepcopy(next(m for m in bundled_catalog['models'] if m['slug']==model))
    catalog_changes={'tool_mode':'none','experimental_supported_tools':[],'apply_patch_tool_type':None,'model_messages':None}
    entry.update(catalog_changes)
    catalog_path=home/'model-catalog.json';catalog_path.write_text(canonical({'models':[entry]})+'\n')
    instructions=home/'instructions_neutral.md';instructions.write_text(neutral_instructions)
    config=source.replace('model = '+json.dumps(parsed['model']),'model = '+json.dumps(model)).replace('model_reasoning_effort = '+json.dumps(parsed['model_reasoning_effort']),'model_reasoning_effort = '+json.dumps(effort)).replace('<codex_home>',str(home))
    config='model_context_window=272000\nmodel_auto_compact_token_limit=250000\nmodel_catalog_json='+json.dumps(str(catalog_path))+'\nproject_doc_max_bytes=0\ninclude_environment_context=false\ninclude_apps_instructions=false\ninclude_collaboration_mode_instructions=false\n'+config
    config+='\n[tools]\nexperimental_request_user_input={enabled=false}\nupdate_plan={enabled=false}\n'
    # Additional feature overrides belong inside the existing table.
    config=config.replace('[features]','[features]\ndaemon_auto_start=false\nenable_request_compression=false\nunbounded_connection_retries=false')
    if fixture_provider:
        config='model_provider="fixture"\n'+config+'\n[model_providers.fixture]\nname="Local synthetic response fixture"\nbase_url='+json.dumps(fixture_provider)+'\nwire_api="responses"\nrequires_openai_auth=false\nrequest_max_retries=0\nstream_max_retries=0\nsupports_websockets=false\n'
    else:
        if not isinstance(production_provider,dict) or set(production_provider)!={'base_url','name'}:raise ValueError('reviewed production provider endpoint/name required')
        if not production_provider['base_url'].startswith('https://'):raise ValueError('production requires reviewed HTTPS endpoint')
        config='model_provider="wsp-supervisor"\n'+config+'\n[model_providers.wsp-supervisor]\nname='+json.dumps(production_provider['name'])+'\nbase_url='+json.dumps(production_provider['base_url'])+'\nwire_api="responses"\nrequires_openai_auth=true\nrequest_max_retries=0\nstream_max_retries=0\nsupports_websockets=false\n'
    path=home/'config.toml';path.write_text(config)
    receipt={'source_sha256':sha(source.encode()),'config_sha256':sha(path.read_bytes()),'catalog_sha256':sha(catalog_path.read_bytes()),'instructions_sha256':sha(instructions.read_bytes()),'model':model,'effort':effort,'catalog_changes':catalog_changes,'fixture_provider':bool(fixture_provider),'production_provider':production_provider,'client_alias':'wsp-supervisor' if not fixture_provider else 'fixture','transport':'HTTP Responses, retries0'}
    (home/'runtime-binding.json').write_text(canonical(receipt)+'\n')
    return receipt


class Supervisor:
    def __init__(self,*,binary,codex_home,workdir,output,config_sha256,model,effort,
                 output_reservation=128000,output_bound_evidence,context_overhead_bytes=32768,soft_limit=4500000,hard_limit=5000000,relay_fixture_upstream=None,guard=None,expected_binary_sha256=None,expected_bound_files=None):
        self.binary=Path(binary).resolve(strict=True);self.home=Path(codex_home).resolve(strict=True)
        self.workdir=Path(workdir).resolve(strict=True);self.output=Path(output).resolve();self.output.mkdir(mode=0o700,parents=True,exist_ok=True)
        for path in (self.home,self.output):
            st=path.stat()
            if st.st_uid!=os.getuid() or st.st_mode&0o077:raise ValueError('supervisor namespace must be owner-only')
        config=self.home/'config.toml'
        if sha(config.read_bytes())!=config_sha256:raise ValueError('approved runtime config changed')
        if not output_bound_evidence or output_reservation<=0:raise ValueError('audited model output bound required')
        import tomllib
        from wsp03_relay import OFFICIAL
        native_config=tomllib.loads(config.read_text());self.provider=native_config['model_provider']
        provider=native_config['model_providers'][self.provider]
        if relay_fixture_upstream is None and (provider.get('base_url')!=OFFICIAL or not provider.get('requires_openai_auth') or provider.get('request_max_retries')!=0 or provider.get('stream_max_retries')!=0):raise ValueError('reviewed native production route required')
        self.relay_fixture_upstream=relay_fixture_upstream;self.guard=guard
        self.bound_files={name:sha((self.home/name).read_bytes()) for name in ('config.toml','model-catalog.json','instructions_neutral.md')}
        self.binary_hash=sha(self.binary.read_bytes())
        if relay_fixture_upstream is None:
            if self.binary_hash!=expected_binary_sha256 or not expected_bound_files or any(self.bound_files.get(name)!=value for name,value in expected_bound_files.items()) or set(expected_bound_files)!={'model-catalog.json','instructions_neutral.md'}:raise ValueError('binary/catalog/instructions differ from approved expected bytes')
        self.config_hash=config_sha256;self.model=model;self.effort=effort;self.output_reservation=output_reservation;self.bound_evidence=output_bound_evidence
        self.overhead=context_overhead_bytes;self.book=RoundBook(self.output/'usage.jsonl',soft_limit=soft_limit,hard_limit=hard_limit)
    def round(self,prompt,*,timeout=1800,ledger=None):
        if self.guard:self.guard()
        if any(sha((self.home/name).read_bytes())!=expected for name,expected in self.bound_files.items()) or sha(self.binary.read_bytes())!=self.binary_hash:raise ValueError('frozen binary/config/context changed')
        rid=uuid.uuid4().hex;input_bound=len(prompt.encode())+self.overhead
        if input_bound>200000:raise RuntimeError('context byte bound exceeds safe single-response window; no native compaction allowed')
        self.book.admit(rid,input_bound,self.output_reservation)
        if ledger is not None:
            try:ledger.admit('editor',rid,{'input_upper_bound':input_bound,'output_upper_bound':self.output_reservation})
            except BaseException as error:
                self.book.not_dispatched(rid,type(error).__name__);self.book.close('pre-provider admission rejected');raise
        directory=self.output/rid;directory.mkdir(mode=0o700)
        command=[str(self.binary),'exec','--strict-config','--json','--ignore-rules','--skip-git-repo-check','--color','never','-m',self.model,'-c','model_reasoning_effort='+json.dumps(self.effort),'-C',str(self.workdir),'-']
        # No inherited API keys, proxy variables, plugins, user shell config or
        # editor-auth paths. Native Codex authentication uses private CODEX_HOME.
        env={k:v for k,v in os.environ.items() if k in ('PATH','HOME','LANG','SSL_CERT_FILE','SSL_CERT_DIR')};env['CODEX_HOME']=str(self.home)
        from wsp03_relay import OneResponseRelay,OFFICIAL
        relay=OneResponseRelay(model=self.model,effort=self.effort,input_bound=input_bound,deadline=time.monotonic()+timeout,upstream=self.relay_fixture_upstream or OFFICIAL,fixture=self.relay_fixture_upstream is not None).start()
        command[2:2]=['-c','model_providers.'+self.provider+'.base_url='+json.dumps(relay.base_url)]
        manifest={'relay_upstream':self.relay_fixture_upstream or OFFICIAL,'relay_enforces_one_upstream_request':True,'command':command,'binary_sha256':sha(self.binary.read_bytes()),'config_sha256':self.config_hash,'input_sha256':sha(prompt.encode()),'input_upper_bound':input_bound,'bound_mode':'full_current_prompt_utf8_bytes_plus_pinned_native_envelope','fallback_reason':'fresh stateless exec repacks history/counters; incremental token equivalence is not proven, including at handoff','output_reservation':self.output_reservation,'output_bound_evidence':self.bound_evidence,'uid':os.getuid(),'created_at':time.time()}
        (directory/'invocation.json').write_text(canonical(manifest)+'\n')
        import threading
        process=None;guard_stop=threading.Event();guard_failures=[];guard_thread=None
        def watch_guard():
            while not guard_stop.wait(.1):
                try:self.guard()
                except BaseException as error:
                    guard_failures.append(type(error).__name__)
                    if process and process.poll() is None:
                        try:os.killpg(process.pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                        limit=time.monotonic()+5
                        while process.poll() is None and time.monotonic()<limit:time.sleep(.05)
                        if process.poll() is None:
                            try:os.killpg(process.pid,signal.SIGKILL)
                            except ProcessLookupError:pass
                    return
        try:
            with (directory/'stdout.jsonl').open('wb') as stdout,(directory/'stderr.txt').open('wb') as stderr:
                process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
                if self.guard:
                    guard_thread=threading.Thread(target=watch_guard,daemon=True);guard_thread.start()
                process.communicate(prompt.encode(),timeout=timeout)
            raw=(directory/'stdout.jsonl').read_bytes();events=[json.loads(line) for line in raw.splitlines()]
            parsed=parse_events(events)
            if process.returncode:raise ValueError('native exec exit failure')
            if relay.forwarded!=1 or relay.seen!=1:raise ValueError('native request count violated one-response boundary')
            self.book.complete(rid,parsed['usage'],events_sha256=sha(raw))
            if ledger is not None:
                u=parsed['usage'];ledger.usage(rid,u['input_tokens'],u['output_tokens'],u['cached_input_tokens'])
            (directory/'receipt.json').write_text(canonical({'complete':True,'id':rid,'usage':parsed['usage'],'events_sha256':sha(raw),'thread_id':parsed['thread_id'],'notices':parsed['notices']})+'\n')
            if self.book.status()['closed']:
                if ledger is not None:
                    with ledger.tx():ledger.set('editor_closed','provider_bound_violation')
                raise RuntimeError('actual usage violated the admitted bound; charged and closed')
            return parsed
        except BaseException as error:
            if process and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
            observed=self.book.status()
            if rid in observed['unresolved']:
                self.book.interrupt(rid,type(error).__name__+': '+str(error))
            else:self.book.close('post-completion failure: '+type(error).__name__)
            if ledger is not None:
                state=ledger.db.execute('SELECT state FROM requests WHERE id=?',(rid,)).fetchone()
                if state and state[0]=='running':ledger.finish(rid,{'complete':False,'provider_usage':'unknown','error_type':type(error).__name__})
                with ledger.tx():
                    reason='unknown_provider_usage' if rid in observed['unresolved'] else 'post_completed_response_failure'
                    ledger.set('editor_closed',reason);ledger.event('editor_stopped',{'id':rid,'reason':reason,'known_usage_preserved':rid not in observed['unresolved']})
            raise
        finally:
            guard_stop.set()
            if guard_thread:guard_thread.join(timeout=2)
            record=relay.close();record['guard_failures']=guard_failures
            (directory/'relay-receipt.json').write_text(canonical(record)+'\n')
