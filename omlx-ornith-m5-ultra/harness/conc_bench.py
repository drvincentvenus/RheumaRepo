import json,random,time,threading,urllib.request,sys,statistics as st
random.seed(11)
TOPICS=["psoriatic arthritis and cardiovascular risk","axial spondyloarthritis MRI sacroiliitis criteria","giant cell arteritis tocilizumab tapering","systemic sclerosis interstitial lung disease nintedanib","gout urate-lowering therapy treat-to-target","lupus nephritis belimumab voclosporin","rheumatoid arthritis JAK inhibitor cardiovascular safety","fibromyalgia dysautonomia COMPASS-31","VEXAS syndrome UBA1 diagnosis","ANCA vasculitis avacopan induction","polymyalgia rheumatica sarilumab","Sjogren disease lymphoma risk","adult-onset Still disease IL-1 blockade","Behcet uveitis adalimumab","osteoarthritis knee intra-articular therapies","IgG4-related disease rituximab"]
words=open('/usr/share/dict/words').read().split()
def mk(i):
    filler=" ".join(random.choice(words) for _ in range(900))  # unique ~1.5-2k tok filler = cold prefill
    return f"Case notes reference id {i}: {filler}\n\nIgnoring the noise above, write a concise 250-word evidence summary for a rheumatologist on {TOPICS[i%len(TOPICS)]}, citing guideline bodies by name."
def one(model,i,out):
    body={"model":model,"stream":True,"max_tokens":300,"temperature":0.7,"stream_options":{"include_usage":True},"messages":[{"role":"user","content":mk(i)}]}
    req=urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t0=time.perf_counter(); tf=None; u=None
    for line in urllib.request.urlopen(req):
        if not line.startswith(b"data:"): continue
        p=line[5:].strip()
        if p==b"[DONE]": break
        try: ev=json.loads(p)
        except: continue
        if ev.get("usage"): u=ev["usage"]
        if tf is None and ev.get("choices") and ev["choices"][0].get("delta",{}).get("content"): tf=time.perf_counter()
    t1=time.perf_counter(); out.append(dict(i=i,ttft=tf-t0 if tf else None,total=t1-t0,pt=u["prompt_tokens"],ct=u["completion_tokens"],cached=u["prompt_tokens_details"]["cached_tokens"],srv_gen=u.get("generation_tokens_per_second"),srv_ttft=u.get("time_to_first_token"),gen_dur=u.get("generation_duration"),pp_dur=u.get("prompt_eval_duration"),pp_tps=u.get("prompt_tokens_per_second")))
res=[]
for model in sys.argv[1:]:
    # warm the model
    one(model,999,[])
    for n in [1,2,4,8]:
        out=[]; th=[threading.Thread(target=one,args=(model,n*100+k,out)) for k in range(n)]
        t0=time.perf_counter(); [t.start() for t in th]; [t.join() for t in th]; wall=time.perf_counter()-t0
        ct=sum(o["ct"] for o in out); pt=sum(o["pt"] for o in out)
        # aggregate decode: completion tokens / (wall - max ttft)  ; per-stream from server
        r=dict(model=model,n=n,wall=round(wall,2),prompt_tok_each=round(pt/n),out_tok_total=ct,srv_ttft_med=round(st.median(o["srv_ttft"] for o in out),2),srv_ttft_max=round(max(o["srv_ttft"] for o in out),2),per_stream_gen_med=round(st.median(o["srv_gen"] for o in out),1),agg_decode_sum=round(sum(o["srv_gen"] for o in out),1),gen_dur_med=round(st.median(o["gen_dur"] for o in out),2),pp_tps_med=round(st.median(o["pp_tps"] for o in out)),agg_throughput_incl_prefill=round(ct/wall,1))
        print(json.dumps(r),flush=True); res.append(r)
json.dump(res,open("conc_results.json","w"),indent=1)
