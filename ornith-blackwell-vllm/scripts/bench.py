#!/usr/bin/env python3
"""Warm concurrent-throughput benchmark. Run ON the serving box, never through a tunnel."""
import json, time, urllib.request, concurrent.futures as cf
EP = "http://127.0.0.1:8000/v1/chat/completions"
def one(i):
    body = json.dumps({"model": "ornith-1.5", "max_tokens": 300, "temperature": 0.6,
        "messages": [{"role": "user", "content": f"Explain concept #{i}: distributed consensus, in 250 words."}]}).encode()
    t0 = time.time()
    d = json.load(urllib.request.urlopen(urllib.request.Request(EP, data=body,
        headers={"content-type": "application/json"}), timeout=300))
    dt = time.time() - t0
    return d.get("usage", {}).get("completion_tokens", 0), dt
one(0); one(1)  # warmup (first request pays flashinfer autotune)
tok, dt = one(2); print(f"1 agent (warm):  {tok/dt:.1f} tok/s")
for N in (2, 4, 8):
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=N) as ex:
        r = list(ex.map(one, range(N)))
    wall = time.time() - t0
    print(f"{N} agents: {sum(x[0] for x in r)/wall:.0f} tok/s aggregate, {sum(x[0]/x[1] for x in r)/N:.1f} each")
