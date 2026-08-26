#!/usr/bin/env python3
"""Measuring proxy: client -> :8001 -> oMLX :8000. Logs TTFT, decode tok/s per request to bench.jsonl."""
import json, time, sys, os, http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
UP=("127.0.0.1",8000); LOG=os.path.expanduser("~/Documents/omlx_bench/bench.jsonl")
TAG=os.environ.get("BENCH_TAG","")
class H(BaseHTTPRequestHandler):
    protocol_version="HTTP/1.1"
    def log_message(self,*a): pass
    def _fwd(self):
        n=int(self.headers.get("Content-Length") or 0); body=self.rfile.read(n) if n else b""
        t0=time.perf_counter(); rec={"tag":(open(os.path.expanduser("~/Documents/omlx_bench/TAG")).read().strip() if os.path.exists(os.path.expanduser("~/Documents/omlx_bench/TAG")) else ""),"path":self.path,"method":self.command,"ts":time.time()}
        stream=False
        if body and self.path.startswith("/v1/chat/completions"):
            try:
                j=json.loads(body); rec["model"]=j.get("model"); stream=bool(j.get("stream"))
                rec["n_msgs"]=len(j.get("messages",[])); rec["prompt_chars"]=len(body)
                if stream: j.setdefault("stream_options",{})["include_usage"]=True; body=json.dumps(j).encode()
            except Exception: pass
        hdr={k:v for k,v in self.headers.items() if k.lower() not in("host","content-length","accept-encoding")}
        hdr["Content-Length"]=str(len(body)); hdr["Host"]="127.0.0.1:8000"
        c=http.client.HTTPConnection(*UP,timeout=3600); c.request(self.command,self.path,body=body,headers=hdr); r=c.getresponse()
        self.send_response(r.status)
        for k,v in r.getheaders():
            if k.lower() not in("transfer-encoding","content-length","connection"): self.send_header(k,v)
        ctype=r.getheader("Content-Type","")
        chunked=stream or "event-stream" in ctype
        if chunked: self.send_header("Transfer-Encoding","chunked")
        else: self.send_header("Content-Length","0") if False else None
        data=b""; tfirst=None; tlast=None; ntok=0; usage=None; buf=b""
        if chunked:
            self.end_headers()
            while True:
                ch=r.read1(65536) if hasattr(r,"read1") else r.read(65536)
                if not ch: break
                self.wfile.write(b"%x\r\n%s\r\n"%(len(ch),ch)); self.wfile.flush()
                buf+=ch
                while b"\n" in buf:
                    line,buf=buf.split(b"\n",1); line=line.strip()
                    if not line.startswith(b"data:"): continue
                    p=line[5:].strip()
                    if p==b"[DONE]": continue
                    try: ev=json.loads(p)
                    except Exception: continue
                    u=ev.get("usage")
                    if u: usage=u
                    for chc in ev.get("choices",[]):
                        d=chc.get("delta",{})
                        if d.get("content") or d.get("reasoning_content") or d.get("reasoning") or d.get("tool_calls"):
                            now=time.perf_counter()
                            if tfirst is None: tfirst=now
                            tlast=now; ntok+=1
            self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
        else:
            data=r.read(); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data)
            tfirst=tlast=time.perf_counter()
            try: usage=json.loads(data).get("usage")
            except Exception: pass
        t1=time.perf_counter()
        rec.update({"status":r.status,"stream":chunked,"total_s":round(t1-t0,3),
            "ttft_s":round(tfirst-t0,3) if tfirst else None,"chunks":ntok,"usage":usage})
        if usage and usage.get("completion_tokens") and tfirst and tlast and tlast>tfirst:
            rec["decode_tps"]=round((usage["completion_tokens"]-1)/(tlast-tfirst),1)
            rec["prompt_tokens"]=usage.get("prompt_tokens")
            pt=usage.get("prompt_tokens_details") or {}; rec["cached_tokens"]=pt.get("cached_tokens")
            if rec["prompt_tokens"] and rec["ttft_s"]: rec["prefill_tps"]=round(rec["prompt_tokens"]/rec["ttft_s"],1)
        if self.path.startswith("/v1/chat/completions"):
            with open(LOG,"a") as f: f.write(json.dumps(rec)+"\n")
            print(json.dumps({k:rec.get(k) for k in("tag","model","prompt_tokens","cached_tokens","ttft_s","decode_tps","chunks")}),flush=True)
    do_GET=do_POST=_fwd
ThreadingHTTPServer.daemon_threads=True
ThreadingHTTPServer(("127.0.0.1",8001),H).serve_forever()
