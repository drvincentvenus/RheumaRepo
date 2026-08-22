import json, time, urllib.request, concurrent.futures as cf
EP="http://127.0.0.1:8000/v1/chat/completions"
DOC=open("/workspace/grappa_source.md").read()
SYS="You are a rheumatology assistant. Answer ONLY from the GRAPPA 2021 treatment recommendations paper below. Be precise and cite the relevant domain.\n\n=== PAPER ===\n"+DOC
QS=["What does GRAPPA recommend for peripheral arthritis as first-line?","Summarize the recommendations for axial disease in PsA.","What is recommended for enthesitis?","What is recommended for dactylitis?","How should psoriasis (skin) be treated per these recommendations?","What do the recommendations say about nail disease?","What comorbidities are flagged and how do they alter treatment choice?","Describe the GRADE methodology used and the strength of recommendations."]
def one(i):
    body=json.dumps({"model":"qwen3.8-27b-uncensored","max_tokens":400,"temperature":0.6,"stream":True,"stream_options":{"include_usage":True},"messages":[{"role":"system","content":SYS},{"role":"user","content":QS[i%8]}]}).encode()
    t0=time.time(); ttft=None; usage=None
    with urllib.request.urlopen(urllib.request.Request(EP,data=body,headers={"content-type":"application/json"}),timeout=600) as r:
        for line in r:
            line=line.decode().strip()
            if not line.startswith("data: ") or line=="data: [DONE]": continue
            d=json.loads(line[6:])
            if d.get("usage"): usage=d["usage"]
            ch=d.get("choices")
            if ch and ttft is None:
                delta=ch[0].get("delta") or {}
                if any(v for k,v in delta.items() if k!="role"): ttft=time.time()-t0
    dt=time.time()-t0; ct=usage["completion_tokens"]
    return ct, ttft, ct/(dt-ttft)
ct,t,r=one(0); print(f"solo grounded: {ct} tok | TTFT {t:.2f}s | decode {r:.1f} tok/s")
t0=time.time()
with cf.ThreadPoolExecutor(max_workers=8) as ex: res=list(ex.map(one,range(8)))
wall=time.time()-t0; agg=sum(x[0] for x in res)/wall
print(f"8 grounded agents: aggregate {agg:.0f} tok/s | per-agent " + ", ".join(f"{x[2]:.0f}" for x in res) + " | TTFT " + ", ".join(f"{x[1]:.2f}s" for x in res))
