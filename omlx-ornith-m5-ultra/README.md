# Ornith 1.5 on Apple Silicon: measured on an M4 Max, projected for an M5 Ultra 512 GB

Vincenzo Venerito, 26 August 2026. The M4 Max results are measured. The M5 Ultra results are projections based on those measurements, Apple's published M5 scaling data and the assumptions in section 3. Nobody outside Apple has an M5 Ultra yet (ships 22 September; the 512 GB configuration arrives late October). We will run the same harness on real hardware and replace this table.

## 1. Why a rheumatologist should care

Running a model locally can keep patient text on the machine when the full stack is configured and audited accordingly. That includes local-only inference and tools, disabled telemetry and crash reporting, no cloud fallbacks, controlled logs and session files, disk encryption, audited remote administration, and a private network for a multi-node cluster. Local hardware creates the option. The audit verifies the result. Clinical usability depends on two measures of speed plus available capacity:

- Prefill: the time required to read the prompt before the first token appears (time to first token, TTFT). Matrix-multiply throughput dominates this phase, so it follows GPU compute.
- Decode: the rate at which subsequent tokens arrive (tok/s). Every generated token re-reads the active weights, so single-stream decode follows memory bandwidth.
- Concurrency: the number of streams the machine can serve simultaneously (several screeners on a systematic review, several note summarisers) before per-stream performance becomes too slow.

These regimes dominate performance. Kernel efficiency, expert dispatch in MoE models and KV-cache traffic also affect the results. Apple's MLX team documents this split on the M5: prefill 3.3x to 4.1x faster than M4, generation 1.19x to 1.27x on 28% more bandwidth (base chips; Apple's Mac Studio chart puts the M5 Ultra at 4.1x the M3 Ultra on prompt processing), tested with a 4,096-token prompt on models including Qwen 30B-A3B 4-bit, a similarly sized sparse MoE (Ornith's hybrid full and linear attention and its expert kernels may scale differently).

For a client that shows the model's reasoning as it streams, TTFT has the greatest effect on whether a local machine feels like an API. For a client that hides reasoning, or for a thinking model in general, the measure that matters is time to the first visible answer word, which is TTFT plus the thinking time (section 2d). Well-provisioned cloud endpoints usually begin streaming within a second for short or cached prompts. Users notice the initial pause. Decode above about 30 tok/s already exceeds typical reading speed. A local model at 100 tok/s with a 10-second startup still feels slow. At 60 tok/s with a one-second startup, the same model feels like the cloud. The M5's prefill gain therefore has more clinical relevance than its bandwidth gain. Prefix caching should be configured first: in the paired real-document run of section 2d, the same 12.2k-token paper took 13.5 s to first token cold and 2.5 s once cached (the 12 s and 0.9 s in sections 2a and 2b are separate synthetic measurements).

There is another distinction. Ornith reasons before producing its answer, while server TTFT records the first reasoning token. Thinking proceeds at decode speed, so at 8 streams (27 tok/s each) a few hundred reasoning tokens add over ten seconds before the answer starts; the synthetic runs of sections 2a to 2c did not log time to first visible token; the real-document run of section 2d does, and there it was 11.5 s for one cold agent and 37 s at 8 agents on a cached paper. Clients that stream reasoning (Hermes and OpenCode do) show activity at TTFT. A client that hides reasoning leaves the user waiting for TTFT plus thinking time. Decode speed therefore affects perceived latency in the projection whenever thinking is enabled.

Reference point, the cloud APIs clinicians use today: at a 1k-token prompt (client-side, April 2026) Claude Sonnet 4.6 answers in 0.74 s at 104 tok/s, Claude Opus 4.7 0.85 s at 78, GPT-5.5 1.12 s at 92, Gemini 3 Pro 0.93 s at 84, Gemini 3 Flash 0.42 s at 200; with extended thinking the first answer token takes 8.4 s (GPT-5.5 Pro) to 28 s (Claude) to 52 s (Gemini Deep Think). Artificial Analysis (August 2026) lists Claude Sonnet 5 at 87.5 tok/s and Gemini 3.6 Flash at 210 tok/s with 17 s to first answer token in reasoning mode. Longer prompts add prefill time and provider prompt caching cuts TTFT by 50 to 85% on repeat use.

## 2. What we measured

Hardware: MacBook Pro, Apple M4 Max, 40-core GPU, 128 GB, 546 GB/s. macOS 26.5.2.
Software: oMLX 0.6.3rc2 on MLX 0.32.0, default settings (memory guard balanced, paged SSD prefix cache on, max 8 concurrent requests, chunked prefill off). Clients: Hermes Agent 0.20.5, OpenCode 1.18.21.
Model: Ornith 1.5 35B-A3B (Qwen3.5-MoE architecture: 40 layers, 10 full-attention and 30 linear-attention, 256 experts, 8 active per token, about 3B active parameters).
Two files from the LM Studio folder: `ornith-4bit` = PocketAiHub/Ornith-1.5-35B-A3B-Abliterated-MLX-4bit (20.4 GB on disk, 19.0 GiB) and `ornith-6bit` = ornith-ai/Ornith-1.5-35B-A3B-MLX-6bit (28.2 GB). They come from different publishers, and one is abliterated. Any difference combines quantisation with provenance, preventing a clean 4-bit versus 6-bit comparison.
Workload note: none of these runs used the GRAPPA-grounded ingestion of the earlier `ornith-blackwell-vllm/` benchmark. The agent sessions ran a fizzbuzz tool task and a 300-word MoE explanation; the cold-prefill runs used random dictionary words so nothing could hit the prefix cache; the concurrency runs used rheumatology topics padded with random words. These are synthetic throughput workloads, chosen to isolate prefill and decode. Section 2d adds a real-document run (the GRAPPA 2021 comorbidities review as context) so the synthetic and clinical workloads can be read side by side; it is still not the GRAPPA-grounded 8-agent protocol of that folder.

Metrics: TTFT and tok/s are oMLX's per-request `time_to_first_token` and `generation_tokens_per_second`, cross-checked with a client-side proxy (decode rates agree within 2%, TTFT within 0.15 s). TTFT measures time to the first generated token, including thinking tokens. It does not measure time to the first visible word. Exclusions in section 2a: OpenCode's concurrent title-generation calls (prompt under 1,000 tokens) and two cold-cache calls on the 6-bit model (the Hermes first call of the session, server TTFT 18.8 s after a 14.5 s model load, 33.7 s as seen by the client, and one OpenCode turn with an empty cache, TTFT 12.9 s). Raw logs: `bench.jsonl`, `conc_results.json`; harness: `run_bench.sh`, `conc_bench.py` (prompt generator seeded, `random.seed(11)`, from `/usr/share/dict/words`), `proxy.py`. Launch: `omlx serve --model-dir models --port 8000`. Sampling: agent runs use each client's defaults; cold prefill runs temperature 0; concurrency runs temperature 0.7, max_tokens 300. Concurrency and real-document runs are driven by the harness scripts directly against oMLX, not through Hermes or OpenCode, which were used for the single-stream sessions of 2a only. Agent runs 3 reps; cold runs 2 to 3 reps, except the 2.5k rows, which are the single-stream result of the concurrency run, and the 17k row, whose two requests are recorded in `logs/omlx.log` only; concurrency 1 run per level. In `bench.jsonl` the cold tags `8k` and `17k` denote the 14k and 30k prompts (the random-word filler tokenised heavier than planned). The public repository linked from the post contains everything.

### 2a. Real agent sessions, single stream, warm prefix cache (3 reps each)

| Model | Client | System prompt | TTFT | Decode tok/s (outputs of 100+ tokens) |
|---|---|---|---|---|
| 4-bit | Hermes | 17k tok | 0.85 to 0.95 s | 102 to 106 |
| 4-bit | OpenCode | 7.6k tok | 2.0 to 2.3 s | 58 to 69 (OpenCode runs a concurrent title-generation call that shares decode) |
| 6-bit | Hermes | 17k tok | 1.3 to 1.9 s | 46 to 58 |
| 6-bit | OpenCode | 7.6k tok | 2.7 to 2.9 s | 66 to 69 |

Tasks: a tool-calling coding task (write and run fizzbuzz.py, 2 to 4 turns) and a 300-word explanation without tools.

### 2b. Cold prefill, no cache hit, greedy, 200 output tokens (reps as noted in Metrics)

| Model | Prompt tokens | TTFT | Prefill tok/s | Decode tok/s |
|---|---|---|---|---|
| 4-bit | 2.5k | 0.9 s | 2,700 | 103 |
| 4-bit | 14k | 9.2 to 10.0 s | 1,430 to 1,560 | 107 to 110 |
| 4-bit | 17k | 11.7 to 12.2 s | 1,400 to 1,460 | 105 to 108 |
| 4-bit | 30k | 24.5 to 25.6 s | 1,170 to 1,230 | 98 to 102 |
| 6-bit | 2.5k | 0.9 s | 2,750 | 87 |
| 6-bit | 14k | 11.3 to 12.7 s | 1,130 to 1,260 | 88 to 98 |
| 6-bit | 30k | 30.9 to 36.1 s | 830 to 970 | 68 to 76 |

### 2c. Concurrent streams, 4-bit, one unique 2.5k-token prompt per stream, 300 output tokens

| Streams | TTFT median (max) | Per-stream decode, median | Sum of per-request decode rates | End-to-end tokens per wall-clock second (incl. prefill) |
|---|---|---|---|---|
| 1 | 0.9 s | 103 tok/s | 103 tok/s | 78 |
| 2 | 1.6 s | 66 | 133 | 98 |
| 4 | 2.6 (2.7) s | 45 | 173 | 130 |
| 8 | 4.8 (5.3) s | 27 | 203 | 147 |

Per-stream is the median across streams. The sum column adds each request's server-reported decode rate and may overstate sustained aggregate throughput when streams overlap imperfectly. The last column is the strict measure (all generated tokens divided by total wall-clock time). One run per concurrency level, temperature 0.7. 6-bit at 8 streams: TTFT 4.8 s, 21 tok/s per stream, 168 sum of rates, 125 end-to-end.

Interpretation: in this synthetic 2.5k-prompt run, aggregate decode roughly doubles between 1 and 8 streams by either measure, which shows the benefit of batching (the real-document run in 2d behaves differently: end-to-end throughput falls from 35 to 27 tok/s cold and rises only from 58 to 72 cached). Each agent falls below 30 tok/s at 8. TTFT increases roughly with stream count, consistent with serialised prefill (oMLX has chunked prefill off by default). We did not trace the scheduler, so the mechanism remains unconfirmed.

KV cache: only the 10 full-attention layers grow with context (2 KV heads x 256 dim x 2 tensors x 2 bytes = 2,048 bytes per layer per token), giving 20,480 bytes per token per stream of raw FP16 KV payload. Eight streams at the model's 262k limit hold about 43 GB of raw payload. Paged-cache metadata, fragmentation and runtime buffers increase this requirement, so budget roughly 50 to 60 GB. Linear-attention layers keep a fixed-size state per stream (32 value heads x 128 x 128 x 4 bytes, about 2 MB per layer, 63 MB per stream for 30 layers), which is negligible beside the KV payload.

### 2d. Real document workload: a paper plus a clinical question

The synthetic runs above isolate prefill and decode. This run is what a clinic would do: the GRAPPA 2021 comorbidities review (Campanholo et al., J Rheumatol 2023; 6,416 words, 12.2k tokens as extracted) in context, a system prompt restricting the model to the document, and one of eight clinical questions per agent (screening intervals, cardiovascular risk tools, obesity and treatment response, depression screening, liver monitoring, uveitis and IBD, infection and vaccination, effect sizes). 400 output tokens, temperature 0.3, thinking on. Two scenarios: B, cache-defeated: every agent gets the same paper preceded by a unique case-file line, so nothing hits the prefix cache and each prompt is prefilled in full; this is a proxy for agents each reading a different paper, not a measurement of different papers; A, all agents share the same paper, already in the prefix cache (a department working on one guideline). Script `realbench.py`, raw `realbench_results.json`. One run per cell.

| Ornith 1.5 4-bit, 12.2k-token context (B = same paper, cache defeated, proxy for distinct documents) | Agents | TTFT median (max) | First visible answer word, median | Decode per agent | End-to-end tok/s |
|---|---|---|---|---|---|
| B: distinct context, cold | 1 | 7.8 s | 11.5 s | 107 | 35 |
| B: distinct context, cold | 2 | 14.8 (21.3) s | 23 s | 67 | 32 |
| B: distinct context, cold | 4 | 36 (47) s | 45 s | 45 | 30 |
| B: distinct context, cold | 8 | 98 (98) s | 110 s | 20 | 27 |
| A: shared paper, 10,240 tokens cached | 1 | 2.5 s | 6.9 s | 91 | 58 |
| A: shared paper, cached | 2 | 4.5 (5.5) s | 12 s | 51 | 60 |
| A: shared paper, cached | 4 | 7.5 (13.2) s | 20 s | 28 | 67 |
| A: shared paper, cached | 8 | 15.6 (27.8) s | 37 s | 16 | 72 |

6-bit: cold single 10.5 s TTFT, 15.6 s to first word, 80 tok/s; 8 agents cold 74 s median TTFT (max 136 s), 87 s to first word, 24 tok/s per agent, sum of per-request rates 246 tok/s but end-to-end only 22.5 tok/s (the sum is higher than the 4-bit's 190 because its streams overlapped less, which is exactly why the sum is not a throughput measure; the strict figure, 22.5 against 26.9, ranks them the other way, and the two files differ in publisher and post-training, so this says nothing about bit width); 8 agents shared 14.2 s (max 28.3), 37 s to first word, 15 tok/s per agent, 73 tok/s end-to-end.

Reading. Prefill is the wall. The timings are consistent with oMLX prefilling one request at a time under its default configuration (chunked prefill off; the scheduler was not traced), so eight 12k-token prompts appear to queue: the last agent waits 98 s for its first token and 119 s for its first visible word (medians across the eight: 98 s and 110 s). With the paper cached the remaining 1.9k tokens per agent still cost 15 to 28 s at 8 agents, and thinking at 16 tok/s adds another 20 s before the answer starts. On a 7,000 euro M4 Max, eight agents each reading a full document in parallel is a demonstration, not a working experience. Even the single agent needs 11.5 s to the first visible word on a cold paper; 6.9 s on a cached one.

## 3. Projection to M5 Ultra (512 GB, 1.2 TB/s, 80-core GPU with Neural Accelerators)

Assumptions:

A1. Single-stream decode scales with memory bandwidth, with a haircut for small active weights. Bandwidth ratio 1,200 / 546 = 2.2x. Apple's M5 vs M4 data show generation gains of 0.93 to 0.99 of the bandwidth ratio on one die. The llama.cpp cross-chip table (Llama 7B Q4_0, tg128) gives the Ultra over Max decode gain as 1.11x (M3 Ultra 80-core over M4 Max 40-core, across generations) to 1.39x (M3 Ultra over M3 Max, same generation), well under the bandwidth ratio for a small model, and M5 Max over M4 Max as 1.44x. Chaining 1.44x with 1.11x to 1.39x gives 1.6x to 2.0x; the bandwidth ratio gives 2.2x. Projected factor 1.6x to 2.2x.

A2. Prefill scales with GPU compute, and Apple has now published Ultra-to-Ultra numbers. The Mac Studio page's LLM prompt processing chart (time to first token, 8K-token prompt, 14B model at 4-bit, LM Studio) shows M5 Ultra at 9.8x and M3 Ultra at 2.4x an M1 Ultra, so M5 Ultra is 4.1x the M3 Ultra. The M4 Max is not on that chart, so we link it two independent ways. Route 1: the llama.cpp cross-chip table (Llama 7B Q4_0, pp512) has the M3 Ultra 80-core at 1,471 tok/s and the M4 Max 40-core at 886 tok/s, a 1.66x gap; 1.66 x 4.1 = 6.8x. Route 2: Apple's M5 Max chart puts the M4 Max at 2.8x an M1 Max, the llama.cpp table puts the M1 Ultra at 1.94x the M1 Max, and 9.8 x 1.94 / 2.8 = 6.8x. Both routes give 6.8x. The single-die M5 over M4 gain (3.8x from Apple's chart, 3.6x in llama.cpp) is consistent with it once the core count doubles from 40 to 80 with about 1.8x scaling. We carry 6x to 7.5x to cover software differences (LM Studio and llama.cpp versus oMLX) and model differences (dense 7B and 14B versus a hybrid-attention MoE); 6.8x is the central value.

A3. Multi-stream decode scales like single-stream decode (same 1.6x to 2.2x). This is a hypothesis without direct measurement because we did not profile bandwidth versus compute utilisation at 8 streams. Partly compute-bound batched decode could benefit more from the Neural Accelerators. Dominant scheduling overhead could produce weaker scaling. Sensitivity: at 1.3x the 8-agent aggregate would be about 265 tok/s.

A4. Multi-stream TTFT. oMLX reports prompt evaluation as essentially all of TTFT within display rounding (2.5k tokens at 2,700 tok/s is 0.93 s against a 0.91 s TTFT for the same request), indicating little fixed overhead for one request. We did not trace the 8-request queue. Model: T' = U + (4.8 - U) / S, where U is unscaled queueing and overhead and S the prefill factor from A2 (6 to 7.5). With U = 0: 0.6 to 0.8 s. With U = 1 s: 1.5 to 1.6 s. With U = 2 s: 2.4 to 2.5 s. Limiting case, no improvement at all: 4.8 s. We report 0.6 to 1.6 s (U up to 1 s) as the projection and include the remainder as sensitivity.

A5. Same software stack, with no allowance for MLX or oMLX improvements. This assumes MLX exposes the Neural Accelerators on the Ultra as it does on M5 Max. Apple's MLX research measurements (the 3.3x to 4.1x prefill figures) cover a single die; the Mac Studio chart used in A2 does cover the Ultra.

| Ornith 1.5 4-bit | M4 Max measured | M5 Ultra projected | Factor (assumption) |
|---|---|---|---|
| Single-stream decode, 2.5k prompt | 103 tok/s | 165 to 227 tok/s | 1.6 to 2.2x (A1) |
| Cold TTFT, 14k prompt | 9.6 s | 1.3 to 1.6 s | 6 to 7.5x (A2); the high end is the conservative value for a latency |
| Cold TTFT, 17k prompt (Hermes system prompt size) | 12 s | 1.6 to 2.0 s | 6 to 7.5x (A2) |
| 8 agents, per-stream decode | 27 tok/s | 43 to 60 tok/s | 1.6 to 2.2x (A3) |
| 8 agents, sum of per-request decode rates | 203 tok/s | 325 to 447 tok/s | 1.6 to 2.2x (A3) |
| 8 agents, end-to-end tokens per wall-clock second | 147 tok/s | 235 to 323 tok/s | 1.6 to 2.2x (A3); the figure to compare with cluster service throughput |
| 8 agents, TTFT (2.5k prompt each), projected median | 4.8 s | 0.6 to 1.6 s | A4; unmeasured queueing could leave it anywhere up to 4.8 s |
| Real workload, 1 agent, cold 12.2k paper, TTFT | 7.8 s | 1.0 to 1.3 s | 6 to 7.5x (A2); M5 Max at 3.6 to 3.8x: 2.0 to 2.2 s |
| Real workload, 8 agents, cache-defeated (proxy for distinct papers), TTFT median | 98 s | 13 to 16 s | 6 to 7.5x (A2), prefill queue assumed to shrink with prefill speed; M5 Max: 26 to 27 s |
| Real workload, 8 agents, shared cached paper, TTFT median | 15.6 s | 2.1 to 2.6 s | 6 to 7.5x (A2); M5 Max: 4.1 to 4.3 s |
| Real workload, 8 agents, decode per agent | 16 to 20 tok/s | 25 to 43 tok/s | 1.6 to 2.2x (A3); M5 Max at 1.1 to 1.45x: 17 to 28 |
| Memory after weights (nominal) | 107.6 GB | 491.6 GB | 512 minus 20.4 GB, before OS and runtime buffers |

The M5 Max column uses Apple's chart ratio for M5 Max over M4 Max (10.7 / 2.8 = 3.8x; llama.cpp 3.6x) for prefill and 1.1x (bandwidth 614 / 546) to 1.45x (llama.cpp tg128) for decode. Reading of the real-workload rows: an M5 Max would bring eight cache-defeated agents (the proxy for distinct papers) from 98 s to about 26 s, arguably still unusable; an M5 Ultra to 13 to 16 s, usable for batch work, not for a conversation. Shared-document work (one guideline, eight questions) becomes practical on either: 4 s on the M5 Max, 2 to 3 s on the M5 Ultra. Part of the 98 s is software: vLLM on a DGX Spark batches prefill across requests, while oMLX's default behaviour looks serial; if oMLX gains batched or chunked prefill the queue shrinks independently of hardware.

6-bit, applying the same factors to its 87 tok/s single-stream baseline at 2.5k prompt: 139 to 191 tok/s single stream; 269 to 370 tok/s sum of per-request rates at 8 agents from the measured 168.

The 512 GB also expands model capacity. The example that matters today is Qwen3.8-Flash-Next, released on 26 August 2026: 125B parameters, 6B active per token, plus a 51B n-gram embedding table and a 4B MTP head, 262k native context, licence qwen-community-1.0. Its FP8 checkpoint is 173 GiB (BF16 335 GiB), so it does not fit a 128 GB machine at FP8; a 4-bit build is estimated at about 82 GiB (58 GiB main weights plus 24 GiB n-gram table, NVIDIA forum estimate), and community NVFP4 (RadixArk) and GGUF (unsloth) builds are already on Hugging Face. In binary units the 256 GB Mac Studio has 238 GiB, so the FP8 checkpoint leaves about 65 GiB before OS and runtime allocations; the 512 GB one has 477 GiB and leaves about 304 GiB, enough for Ornith and a full set of KV caches beside it. The 4B MTP head is additional to the 125B and adds about 2 GiB at 4-bit on top of the 82 GiB estimate. No MLX conversion existed on release day. The layout (12 blocks of three Gated DeltaNet layers and one Qwen Sparse Attention layer, the same family as Qwen3.5 and Ornith 1.5, which MLX already runs) makes support plausible, but until mlx-lm handles the n-gram embedding, the gated residuals and the new attention we give no tok/s figure. DeepSeek V4 Flash (284B, 13B active, about 167 GB mixed FP4/FP8) would also fit in memory, but there is no MLX support for it at present.

## 4. If you would rather have CUDA: 4x DGX Spark

DGX Spark: 128 GB unified memory, 273 GB/s, about 1 PFLOP sparse FP4, $4,699 US list since February 2026 (launched at $3,999), about 6,000 euro in Italy. Four provide 512 GB aggregate across four nodes, not one pool, for about $18,800, roughly 24,000 euro. Four independent boxes need only ordinary networking; one model spanning nodes uses the 200 GbE ConnectX-7 links, two nodes back to back and a switch beyond two.

The Spark's weakness is per-node bandwidth: 273 GB/s is half our M4 Max and under a quarter of the M5 Ultra. Its strength is batching: CUDA plus vLLM or SGLang serve many streams from one copy of the weights, so aggregate throughput keeps climbing long after a single stream has stopped getting faster. Single-stream numbers understate what a Spark does for an agent swarm.

Ornith 1.5 has been measured on one Spark, so this comparison is measured on both sides.

| Ornith 1.5 35B-A3B | M4 Max, oMLX, 4-bit (ours) | 1x DGX Spark, vLLM 0.27.1, NVFP4+FP8, no MTP (NVIDIA forum) | 1x DGX Spark, vLLM with in-checkpoint MTP, NVFP4 (MiaAI-Lab recipe) |
|---|---|---|---|
| Single stream decode | 103 tok/s | 68 tok/s | 86 tok/s |
| Single stream TTFT | 0.9 s at 2.5k prompt; 12 s at 17k cold | 64 to 181 ms at 300-token prompts; 3.0 s for a 22K-token prompt (about 7.4k tok/s prefill) | 122 ms (prompt length not stated) |
| Aggregate at 8 streams | 203 summed, 147 end-to-end | 178 at 4 streams; 385 at 16 | 440 at 24 |
| TTFT at high concurrency | 4.8 s at 8 streams | 229 ms at 16 streams | sub-second throughout |
| Memory used | 20.4 GB weights | 114 of 121 GiB (weights 20.4, FP8 KV for 256k 69.3, CUDA graphs 4.5) | about 22 GB weights |

The forum concurrency scan used 300-token prompts and the MiaAI-Lab prompt length is not stated, so the short-prompt TTFT values are not comparable with our 2.5k-token ones. The forum's 22K-token prefill in 3.0 s is comparable with our cold 17k in 12 s: on prefill one Spark today is about 5x our M4 Max, in the same region as the M5 Ultra projection (6 to 7.5x). The decode columns are closer to comparable but still differ in engine, quantisation, MTP use and, in the aggregate rows, concurrency (16 or 24 streams against our 8). The Spark aggregate figures come from vLLM's serving benchmark and are most likely output tokens per wall-clock second, the same accounting as our strict 147 tok/s. Reading: on one stream the M4 Max is faster (103 vs 68 to 86). At 16 streams one Spark delivers 385 tok/s, which is 1.9x our summed 203 or 2.6x our strict 147 at 8 streams, and sits inside the M5 Ultra 8-agent projection (325 to 447 summed, 235 to 323 strict).

Four Sparks as four independent Ornith servers therefore project to about 1,540 tok/s aggregate at 64 agents (4 x 385), assuming each box keeps its single-box result. Against the M5 Ultra 8-agent projection that is 3.4 to 4.7x the summed figure (325 to 447) and 4.8 to 6.6x the strict figure (235 to 323), at 64 agents against 8. The trade is one process against four, and 24 tok/s per agent instead of 43 to 60.

Isolated results on larger MoE models point the same way, although they are single data points on different models and concurrency levels, not a scaling curve. On one Spark, gpt-oss-120B (MXFP4, 5B active) runs 33.5 tok/s single stream and 863 tok/s aggregate at 256 streams (Dendro Logic, vLLM 26.03). DeepSeek V4 Flash across two Sparks runs 61 tok/s single stream and 261 tok/s aggregate at 16 streams with DSpark speculative decoding (LLMRequirements), and 49 to 54 tok/s single stream across four (TP=4, expert parallel over RDMA). Per stream these are modest; as a shared service for a department they are API-class throughput from a shelf of small boxes.

What the Mac keeps: a faster single stream for the model that fits, one process, one memory pool, no fabric to configure, and, on the M5 Ultra, the projected TTFT gain from the Neural Accelerators. What the Spark cluster keeps: CUDA and the vLLM/SGLang ecosystem (DeepSeek V4 Flash and Qwen3.8-Flash-Next run there today, with NVFP4 checkpoints already published), better scaling with concurrency, and a proven two- and four-node path for models over 128 GB.

## 5. The harsh reality

If the M5 Max delivers the prefill Apple declares, clinical researchers are among the luckiest users of this generation: their workload is long documents and short answers, which prefill compute speeds up most, with thinking time as the remaining limit. The same is true if NVIDIA promptly releases an updated DGX Spark with more bandwidth and memory. Either way the buyer should wait for third-party benchmarks of the exact workload in section 2d before spending.

Three things this document cannot settle. First, nobody has benchmarked an M5 Ultra; every number in section 3 is a projection and will be replaced. Second, MLX is younger and less optimised than CUDA for serving many streams: on the same Ornith model one Spark with vLLM batches to 385 tok/s at 16 agents while oMLX on our M4 Max reaches 203 summed (147 strict) at 8, with per-agent speed falling faster. Part of the Ultra projection therefore depends on oMLX and MLX catching up on batched decode, which A5 does not assume. Third, at four boxes against one the practical difference for a department is probably smaller than either camp claims.

Where the next round is decided is clustering. macOS 26.2 added RDMA over Thunderbolt 5, and Apple states that four Mac Studios deliver up to 3x the inference of one; MLX distributed uses it for tensor parallelism. Two to four M5 Ultras at 512 GB give 1 to 2 TB of unified memory, enough for Kimi-class trillion-parameter models that no single box holds. How fast they will run is unknown; no benchmarks exist and claims that it will beat a Spark cluster are opinions for now. At 80,000 euro or more for four boxes the question changes from engineering to funding: a Horizon grant, or a department head who owns the clinic.

## 5b. Caveats

- Every M5 Ultra number is a projection. The harness will be re-run on real hardware.
- The Spark figures are other people's measurements with other engines and quantisations; the concurrency TTFT values were taken at 300-token prompts.
- M5 Ultra pricing in Europe: base 96 GB from $5,499 US; the 256 GB configuration is about 12,000 euro; the 512 GB configuration is not orderable until late October and we project 20,000 euro or more. Four Sparks are about 24,000 euro. Figures will be corrected when Apple publishes the 512 GB configurator.
- A2's 6.8x rests on Apple's LM Studio chart and the llama.cpp community table, both dense-model tests; a hybrid-attention MoE under oMLX may scale differently, which is why the range is 6x to 7.5x.
- The two Ornith files differ in publisher and post-training, not only in bit width.
- Two aggregate measures are given; the strict one (147 tok/s end-to-end at 8 streams) is the one to compare against cluster-level service throughput.

## Sources

- MacRumors, 26 Aug 2026, New Mac Studio can be clustered together (RDMA over Thunderbolt 5; Apple: four systems up to 3x faster AI inference than one). https://www.macrumors.com/2026/08/26/new-mac-studio-can-be-clustered-together/
- Apple Developer, WWDC26 session 233, Explore distributed inference and training with MLX (JACCL backend, RDMA over Thunderbolt 5 from macOS 26.2). https://developer.apple.com/videos/play/wwdc2026/233/
- Digital Applied, AI model latency benchmarks 2026 (TTFT and TPS, 1,024-token input, client-side, 23 Apr 2026). https://www.digitalapplied.com/blog/ai-model-latency-benchmarks-2026-ttft-throughput
- Artificial Analysis, model comparison (output speed, time to first answer token). https://artificialanalysis.ai/models
- Apple newsroom, 25 Aug 2026, Mac Studio with M5 Max and M5 Ultra: 1.2 TB/s, up to 512 GB, up to 80-core GPU, M5 Ultra from $5,499. https://www.apple.com/newsroom/2026/08/apple-introduces-new-mac-studio-with-m5-max-and-m5-ultra/
- Apple, Mac Studio product page, performance chart 'LLM prompt processing': M5 Ultra 9.8x and M3 Ultra 2.4x vs M1 Ultra; M5 Max 10.7x and M4 Max 2.8x vs M1 Max; footnote: time to first token, 8K-token prompt, 14B model, 4-bit, LM Studio 0.4.19. https://www.apple.com/mac-studio/
- llama.cpp discussion #4167, Apple Silicon performance table (Llama 7B Q4_0): pp512 M1 Max 32-core 530, M1 Ultra 64-core 1,030, M3 Ultra 80-core 1,471, M4 Max 40-core 886, M5 Max 3,220 tok/s; tg128 M3 Max 66, M4 Max 83, M3 Ultra 80-core 92, M5 Max 120 tok/s. https://github.com/ggml-org/llama.cpp/discussions/4167
- Apple ML Research, Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU: prefill 3.33x to 4.06x and generation 1.19x to 1.27x vs M4, 4,096-token prompt, models including Qwen 30B-A3B 4-bit. https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- InsiderLLM, M4 Max and M3 Ultra for local LLMs: M3 Ultra 40 to 50% faster than M4 Max, 70B at 25 to 30 vs 15 to 18 tok/s. https://insiderllm.com/guides/m4-max-ultra-local-llms-apple-silicon/
- LMSYS, NVIDIA DGX Spark in-depth review: 128 GB, 273 GB/s, 1 PFLOP sparse FP4, gpt-oss-20B 49.7 tok/s decode. https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/
- NVIDIA marketplace, DGX Spark, and price reporting ($3,999 launch, $4,699 from Feb 2026). https://marketplace.nvidia.com/en-us/enterprise/personal-ai-supercomputers/dgx-spark/ ; https://intuitionlabs.ai/articles/nvidia-dgx-spark-review
- vLLM recipes, Qwen/Qwen3.8-Flash-Next: 125B + 51B n-gram table, 6B active, 262,144 native context, FP8 172.78 GiB, BF16 335.28 GiB, Gated DeltaNet plus Qwen Sparse Attention. https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next
- NVIDIA Developer Forums, DGX Spark (GB10) Ornith-1.5-35B-A3B: vLLM 0.27.1, official NVFP4+FP8, 68 tok/s single, 178 at 4, 385 at 16 streams (300-token prompts), TTFT 64 to 229 ms, 22K-token prompt prefilled in 3.0 s, 114 GiB used. https://forums.developer.nvidia.com/t/dgx-spark-gb10-ornith-1-5-35b-a3b-68-70-tok-s-16-agent-385-tok-s/380731
- MiaAI-Lab, Ornith-1.5-35B-A3B on one DGX Spark: NVFP4, vLLM with in-checkpoint MTP, 86.3 tok/s single, 440.3 at 24 streams, TTFT 122 ms. https://github.com/MiaAI-Lab/Ornith-1.5-35B-A3B-DGX-Spark
- Dendro Logic, DGX Spark concurrency benchmark: gpt-oss-120B MXFP4 33.5 tok/s single, 123 at 8, 863 at 256 streams; Nemotron 49B NVFP4 695 at 256. https://dendro-logic.com/engineering/nvidia-dgx-spark-concurrency-benchmark/
- LLMRequirements, DeepSeek V4 Flash on 2x DGX Spark: 61 tok/s single, 261 at 16 concurrent (DSpark); 4 nodes TP=4 49 to 54 single. https://llmrequirements.com/news/2026-06-30-deepseek-v4-flash-dual-dgx-spark
- Hugging Face, RadixArk/Qwen3.8-Flash-Next-NVFP4 and unsloth/Qwen3.8-Flash-Next-GGUF (community builds). https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4 ; https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF
- Qwen/Qwen3.8-Flash-Next model card: 125B, 6B active, 51B n-gram embedding, 4B MTP, 262,144 context, qwen-community-1.0. https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- Route179, DeepSeek-V4-Flash on two DGX Sparks: about 167 GB of weights, mixed FP4/FP8, vLLM TP=2, 50 to 60 tok/s. https://route179.dev/2026/07/28/deepseek-v4-flash-dual-dgx-spark-eks-hybrid/
- StorageReview, DGX Spark cluster review (2-node PP vs TP, 200 Gb/s fabric). https://www.storagereview.com/review/nvidia-dgx-spark-cluster-review-distributed-inference-on-dell-gigabyte-and-hp
- Exxact, building a 4x DGX Spark cluster (switch and cabling). https://www.exxactcorp.com/blog/deep-learning/what-you-need-to-build-a-4x-nvidia-dgx-spark-cluster-switch-cabling-power