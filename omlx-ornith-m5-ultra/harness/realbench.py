import json,time,threading,urllib.request,sys,statistics as st
PAPER=open("paper_grappa2021.txt").read()
SYS="You are an assistant to a rheumatologist. Answer only from the document provided. Cite the section or table you rely on. If the document does not answer, say so."
QS=["Which comorbidities does the review recommend screening for in psoriatic arthritis, and how often?",
"What does the review say about cardiovascular risk assessment and which risk calculators or adjustments are suggested?",
"Summarise the evidence on obesity and weight loss and their effect on treatment response in PsA.",
"What is recommended regarding depression and anxiety screening in PsA patients?",
"What does the document say about liver disease and hepatotoxicity monitoring with methotrexate in PsA?",
"Summarise the recommendations on uveitis and inflammatory bowel disease as PsA-associated conditions.",
"What does the review say about infection risk and vaccination in PsA patients on biologics?",
"List the comorbidities with the strongest evidence of association with PsA, with the effect sizes given."]
def one(model,i,unique,out,maxtok=400):
    pre=f"[Case file {i}-{time.time_ns()}]\n" if unique else ""
    msgs=[{"role":"system","content":SYS},{"role":"user","content":pre+"DOCUMENT:\n"+PAPER+"\n\nQUESTION: "+QS[i%8]}]
    body={"model":model,"stream":True,"max_tokens":maxtok,"temperature":0.3,"stream_options":{"include_usage":True},"messages":msgs}
    req=urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t0=time.perf_counter(); u=None; tvis=None
    for line in urllib.request.urlopen(req):
        if not line.startswith(b"data:"): continue
        p=line[5:].strip()
        if p==b"[DONE]": break
        try: ev=json.loads(p)
        except: continue
        if ev.get("usage"): u=ev["usage"]
        if tvis is None and ev.get("choices") and ev["choices"][0].get("delta",{}).get("content"): tvis=time.perf_counter()
    wall=time.perf_counter()-t0
    out.append(dict(i=i,pt=u["prompt_tokens"],cached=u["prompt_tokens_details"]["cached_tokens"],ct=u["completion_tokens"],ttft=u["time_to_first_token"],gen=u["generation_tokens_per_second"],pp=u["prompt_tokens_per_second"],t_visible=round(tvis-t0,2) if tvis else None,wall=round(wall,2)))
res=[]
def run(model,n,unique,label):
    out=[]; th=[threading.Thread(target=one,args=(model,k,unique,out)) for k in range(n)]
    t0=time.perf_counter(); [t.start() for t in th]; [t.join() for t in th]; wall=time.perf_counter()-t0
    ct=sum(o["ct"] for o in out)
    r=dict(model=model,scenario=label,n=n,prompt_tok=round(st.median(o["pt"] for o in out)),cached_med=round(st.median(o["cached"] for o in out)),ttft_med=round(st.median(o["ttft"] for o in out),2),ttft_max=round(max(o["ttft"] for o in out),2),
           t_visible_med=st.median(o["t_visible"] for o in out if o["t_visible"]) if any(o["t_visible"] for o in out) else None,gen_med=round(st.median(o["gen"] for o in out),1),gen_sum=round(sum(o["gen"] for o in out),1),e2e=round(ct/wall,1),wall=round(wall,2),out_tok=ct,pp_med=round(st.median(o["pp"] for o in out)),per=out)
    print(json.dumps({k:v for k,v in r.items() if k!="per"}),flush=True); res.append(r)
for model in sys.argv[1:]:
    levels=[1,2,4,8] if model=="ornith-4bit" else [1,8]
    # Scenario B first (all cold, unique prefix), then A (shared paper; first call warms it)
    for n in levels: run(model,n,True,"B_unique_cold")
    run(model,1,False,"A_shared_warmup")
    for n in levels: run(model,n,False,"A_shared_paper")
json.dump(res,open("realbench_results.json","w"),indent=1)
