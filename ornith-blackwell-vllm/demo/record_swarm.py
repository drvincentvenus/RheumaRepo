#!/usr/bin/env python3
"""v2: grounded GRAPPA swarm recorder. Snapshot during streaming, render after (true rates)."""
import json, time, threading, urllib.request, os, copy
from PIL import Image, ImageDraw, ImageFont

EP="http://127.0.0.1:8000/v1/chat/completions"
OUT="/workspace/frames2"; os.makedirs(OUT,exist_ok=True)
for f in os.listdir(OUT): os.remove(os.path.join(OUT,f))
SRC=open("SOURCE_DOC.md").read()

DOMAINS=["PERIPHERAL ARTHRITIS","AXIAL DISEASE","ENTHESITIS","DACTYLITIS",
         "SKIN PSORIASIS","NAIL PSORIASIS","UVEITIS","IBD"]
def prompt(d):
    return (f"SOURCE DOCUMENT (FULL TEXT of the GRAPPA 2021 updated treatment "
            f"recommendations paper, Coates et al. Nat Rev Rheumatol 2022, PMC9244095):\n---\n{SRC}\n---\n"
            f"TASK: Extract and clearly structure everything in the SOURCE relevant to the domain: {d}. "
            f"Use ONLY the source. Where the source does not give domain-specific detail, write "
            f"'not detailed in source' - do NOT add outside knowledge. Bullet points, concise.")

state=[{"text":"","tok":0,"t_first":None,"done":False,"rate":0.0,"ttft":None} for _ in range(8)]
lock=threading.Lock(); snaps=[]

def agent(i):
    body=json.dumps({"model":"ornith-1.5","max_tokens":850,"temperature":0.3,"stream":True,
        "chat_template_kwargs":{"enable_thinking":False},
        "messages":[{"role":"user","content":prompt(DOMAINS[i])}]}).encode()
    req=urllib.request.Request(EP,data=body,headers={"content-type":"application/json"})
    t_req=time.time()
    with urllib.request.urlopen(req,timeout=600) as r:
        for line in r:
            line=line.decode("utf-8","ignore").strip()
            if not line.startswith("data: ") or line=="data: [DONE]":
                if line=="data: [DONE]": break
                continue
            try: c=json.loads(line[6:]).get("choices",[{}])[0].get("delta",{}).get("content") or ""
            except: continue
            if c:
                now=time.time()
                with lock:
                    s=state[i]
                    if s["t_first"] is None: s["t_first"]=now; s["ttft"]=now-t_req
                    s["text"]+=c; s["tok"]+=1
                    el=now-s["t_first"]
                    if el>0.2: s["rate"]=s["tok"]/el
    with lock: state[i]["done"]=True   # rate stays frozen at last computed value

threads=[threading.Thread(target=agent,args=(i,),daemon=True) for i in range(8)]
t0=time.time()
for t in threads: t.start()
# SNAPSHOT loop (cheap): no rendering while streaming
while any(not s["done"] for s in state):
    with lock: snaps.append((time.time()-t0, copy.deepcopy(state)))
    time.sleep(0.15)
    if time.time()-t0>300: break
with lock: snaps.append((time.time()-t0, copy.deepcopy(state)))
for _ in range(18): snaps.append((snaps[-1][0], copy.deepcopy(state)))  # hold
stream_wall=snaps[-1][0]
print(f"streamed {sum(s['tok'] for s in state)} chunks in {stream_wall:.1f}s; rendering {len(snaps)} frames...")

W,H=1920,1080; COLS,ROWS=4,2; HDR=84; PW,PH=W//COLS,(H-HDR)//ROWS
BG=(13,14,18);PANEL=(22,24,30);BORDER=(45,50,62);TXT=(210,214,222)
DIM=(120,126,138);ACC=(80,250,123);TITLE=(139,180,250)
def font(sz):
    try: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",sz)
    except: return ImageFont.load_default()
def fontb(sz):
    try: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",sz)
    except: return font(sz)
F_HDR=fontb(30);F_SUB=font(17);F_ID=fontb(15);F_TXT=font(12);F_ST=fontb(16)
CHARS=int(PW/7.3); LINES=int((PH-76)/15)
def wrap_tail(t):
    out=[]
    for ln in t.split("\n"):
        while len(ln)>CHARS: out.append(ln[:CHARS]); ln=ln[CHARS:]
        out.append(ln)
    return out[-LINES:]
for n,(el,snap) in enumerate(snaps):
    img=Image.new("RGB",(W,H),BG); dr=ImageDraw.Draw(img)
    tot=sum(s["tok"] for s in snap); agg=sum(s["rate"] for s in snap if s["rate"]>0)
    dr.text((28,12),"ORNITH-1.5 NVFP4 · vLLM TP2 · 2×RTX 5090 · MTP+CUDA graphs",font=F_HDR,fill=TITLE)
    dr.text((28,52),"8 agents · each grounded on the FULL GRAPPA 2021 paper (16.5k tok) · 262k ctx each",font=F_SUB,fill=DIM)
    dr.text((W-540,32),f"{agg:6.0f} tok/s combined  {tot:6d} tok  {min(el,stream_wall):5.1f}s",font=F_ST,fill=ACC)
    for i,s in enumerate(snap):
        x=(i%COLS)*PW; y=HDR+(i//COLS)*PH
        dr.rectangle([x+6,y+6,x+PW-6,y+PH-6],fill=PANEL,outline=BORDER)
        st="DONE" if s["done"] else ("STREAMING" if s["t_first"] else "PREFILL")
        col=DIM if s["done"] else ACC
        dr.text((x+18,y+14),DOMAINS[i],font=F_ID,fill=TITLE)
        ttft=f"  TTFT {s['ttft']:.2f}s" if s.get("ttft") else ""
        dr.text((x+18,y+PH-28),f"{st}  {s['tok']:4d} tok  {s['rate']:5.0f} tok/s{ttft}",font=F_ID,fill=col)
        for j,ln in enumerate(wrap_tail(s["text"])):
            dr.text((x+18,y+38+j*15),ln,font=F_TXT,fill=TXT)
    img.save(f"{OUT}/f{n:05d}.png")
fps=len(snaps)/max(stream_wall+2.7,1)
print(f"RENDER_DONE frames={len(snaps)} fps={fps:.2f}")
