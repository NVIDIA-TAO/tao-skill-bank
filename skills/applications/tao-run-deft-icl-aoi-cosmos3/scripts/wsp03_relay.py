# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted one-upstream-request Responses relay; never logs credential values.

Native exec may internally react to an invalid server tool call. This relay
rejects any second request BEFORE external dispatch, independent of client retry
semantics. Usage still comes solely from the supervisor-owned exec JSON stream.
"""
from __future__ import annotations
import hashlib,http.client,http.server,json,secrets,socket,ssl,threading,time
from urllib.parse import urlsplit

OFFICIAL='https://chatgpt.com/backend-api/codex'
HOP={'connection','proxy-connection','keep-alive','transfer-encoding','te','trailer','upgrade','host','content-length'}

class OneResponseRelay:
    def __init__(self,*,model,effort,input_bound,deadline,upstream=OFFICIAL,fixture=False):
        parsed=urlsplit(upstream)
        if not fixture and upstream!=OFFICIAL:raise ValueError('production upstream must be exact pinned official HTTPS route')
        if fixture and (parsed.scheme!='http' or parsed.hostname!='127.0.0.1'):raise ValueError('fixture upstream must be loopback HTTP')
        if parsed.query or parsed.fragment or parsed.username or parsed.password:raise ValueError('invalid upstream route')
        self.upstream,self.parsed,self.fixture=upstream,parsed,fixture;self.model,self.effort=model,effort
        self.input_bound,self.deadline=input_bound,deadline;self.nonce=secrets.token_hex(24);self.lock=threading.Lock();self.seen=0;self.forwarded=0;self.events=[];self.connection=None;self.clients=set()
        relay=self
        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'
            def setup(self):
                super().setup()
                self.connection.settimeout(max(.1,relay.deadline-time.monotonic()))
                with relay.lock:relay.clients.add(self.connection)
            def finish(self):
                try:super().finish()
                finally:
                    with relay.lock:relay.clients.discard(self.connection)
            def log_message(self,*args):pass
            def do_GET(self):self.reject(405,'method not permitted')
            def reject(self,status,message):
                body=message.encode();self.send_response(status);self.send_header('Content-Length',str(len(body)));self.send_header('Connection','close');self.end_headers();self.wfile.write(body);self.close_connection=True
            def do_POST(self):
                with relay.lock:
                    relay.seen+=1;ordinal=relay.seen
                    if ordinal>1:
                        relay.events.append({'kind':'blocked_additional_request','ordinal':ordinal,'time':time.time()});self.reject(409,'one response request per admitted round');return
                if self.path!='/'+relay.nonce+'/responses':relay.events.append({'kind':'blocked_path'});self.reject(403,'route not permitted');return
                try:
                    self.connection.settimeout(max(.1,relay.deadline-time.monotonic()))
                    length=int(self.headers.get('Content-Length','-1'))
                    if not 0<length<=4*1024**2 or self.headers.get('Content-Encoding') or self.headers.get('Transfer-Encoding'):raise ValueError('bounded uncompressed JSON request required')
                    raw=self.rfile.read(length)
                    if len(raw)!=length:raise ValueError('incomplete request body')
                    body=json.loads(raw)
                    from wsp03_supervisor import validate_request_shape
                    validate_request_shape(body,relay.model,relay.effort)
                    def strings(value):
                        if isinstance(value,str):return len(value.encode())
                        if isinstance(value,dict):return sum(len(k.encode())+strings(v) for k,v in value.items())
                        if isinstance(value,list):return sum(strings(v) for v in value)
                        return 16
                    text_bound=strings(body)
                    if text_bound>relay.input_bound:raise ValueError('actual native request text exceeds admitted input bound')
                    remaining=relay.deadline-time.monotonic()
                    if remaining<=0:raise TimeoutError('round deadline exhausted before upstream dispatch')
                    cls=http.client.HTTPConnection if relay.fixture else http.client.HTTPSConnection
                    kwargs={} if relay.fixture else {'context':ssl.create_default_context()}
                    conn=cls(relay.parsed.hostname,relay.parsed.port,timeout=remaining,**kwargs);relay.connection=conn
                    headers={k:v for k,v in self.headers.items() if k.lower() not in HOP}
                    headers['Content-Length']=str(len(raw));headers['Connection']='close'
                    with relay.lock:relay.forwarded+=1
                    relay.events.append({'kind':'upstream_dispatch','request_sha256':hashlib.sha256(raw).hexdigest(),'request_bytes':len(raw),'model_visible_upper_bound_bytes':text_bound,'header_names':sorted(headers),'time':time.time()})
                    conn.request('POST',relay.parsed.path.rstrip('/')+'/responses',body=raw,headers=headers)
                    response=conn.getresponse()
                    # Neither upstream nor downstream may follow redirects.
                    if 300<=response.status<400:
                        relay.events.append({'kind':'blocked_upstream_redirect','status':response.status,'time':time.time()})
                        self.reject(502,'upstream redirect forbidden');return
                    self.send_response(response.status)
                    for key,value in response.getheaders():
                        if key.lower() not in HOP|{'location'}:self.send_header(key,value)
                    self.send_header('Connection','close');self.end_headers();self.close_connection=True
                    copied=0
                    while True:
                        if time.monotonic()>relay.deadline:raise TimeoutError('upstream response exceeded round deadline')
                        if conn.sock:conn.sock.settimeout(max(.1,relay.deadline-time.monotonic()))
                        data=response.read1(65536)
                        if not data:break
                        self.wfile.write(data);self.wfile.flush();copied+=len(data)
                    relay.events.append({'kind':'upstream_finished','status':response.status,'response_bytes':copied,'time':time.time()})
                except BaseException as error:
                    # Error type only: exceptions can include sensitive headers/URLs.
                    relay.events.append({'kind':'relay_error','error_type':type(error).__name__,'time':time.time()})
                    try:self.reject(502,'trusted response relay stopped')
                    except (OSError,ValueError):pass
                finally:
                    if relay.connection:relay.connection.close()
        self.server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);self.server.daemon_threads=True
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
    @property
    def base_url(self):return 'http://127.0.0.1:'+str(self.server.server_port)+'/'+self.nonce
    def start(self):self.thread.start();return self
    def close(self):
        if self.connection:self.connection.close()
        with self.lock:clients=list(self.clients)
        for client in clients:
            try:client.shutdown(socket.SHUT_RDWR);client.close()
            except OSError:pass
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)
        return {'upstream':self.upstream,'fixture':self.fixture,'native_requests_seen':self.seen,'upstream_requests_dispatched':self.forwarded,'events':self.events,'auth_values_recorded':False,'usage_source':'native exec JSON only; relay is admission enforcement, not replacement accounting'}
