# ⚡ Qwen3.8-27B-Uncensored NVFP4 on 1x RTX PRO 6000 · vLLM + MTP + fp8 KV

[![vLLM](https://img.shields.io/badge/vLLM-0.27+-blue)](https://github.com/vllm-project/vllm)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76b900)](https://developer.nvidia.com/cuda-toolkit)
[![GPU](https://img.shields.io/badge/RTX_PRO_6000_96GB-Blackwell_sm__120-76b900)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**372 tokens/s aggregate across 8 concurrent agents each grounded on a full 16.5k-token
paper, sub-150 ms TTFT on short prompts, 131k context — on a single GPU.**

Companion recipe to [`ornith-blackwell-vllm`](../ornith-blackwell-vllm): same Blackwell
toolchain (CUDA 13.0, flashinfer, vLLM), single-GPU serving so none of the NCCL/TP
workarounds are needed. The toolchain is not rebuilt — it is rsync-cloned from an
already-working pod over the datacenter network at ~15 MB/s (`scripts/clone_toolchain.sh`),
which took **15m10s for the 21 GB payload** (venv + CUDA tar + kernel caches).

## 📊 The benchmark: GRAPPA-grounded swarm (22 Aug 2026, on-pod)

The reference benchmark for every recipe in this repo: N concurrent agents, each grounded
on the **full GRAPPA 2021 treatment recommendations paper (16.5k tokens)** as a
prefix-cached system prompt, answering domain questions with 400-token completions
(`scripts/bench_usage.py`).

| Concurrent agents | Aggregate | Per agent | TTFT |
|:---:|:---:|:---:|:---:|
| 1 | 99.6 tok/s | 99.6 tok/s | 0.44 s |
| **8** | **372 tok/s** | 59–83 tok/s | 1.4–3.0 s |

Priming the 16.5k prefix costs 2.3 s once; every later agent reuses the cached KV.

Single-stream latency, for reference:

| Prompt | TTFT | Decode | Notes |
|---|---:|---:|---|
| 67 tok | 148 ms | 74.5 tok/s | 800 tok completion |
| 61 tok | 123 ms | 84.4 tok/s | 800 tok completion |
| ~9k tok | 813 ms | 76.8 tok/s | ≈ 11k tok/s prefill, cold prefix |
During decode the GPU sits at 93% utilization / 317 W with 92.7 GB VRAM resident
(weights + fp8 KV pool at `--gpu-memory-utilization 0.92`).

Boot times: **first boot 12m45s** (torch.compile ran fresh: 156 s, plus flashinfer
autotune; artifacts persist to the network volume) — **subsequent boots ~6 min**.

## ⚠️ Benchmark gotcha: MTP breaks chunk-counting

With `"method":"mtp"` speculative decoding, vLLM packs multiple accepted tokens into a
single SSE chunk. A bench that counts stream chunks under-reports throughput by ~2.5x here
(we first measured a bogus 147 tok/s aggregate / 36 tok/s solo). Always measure with
`stream_options: {"include_usage": true}` and compute rates from `usage.completion_tokens`
— that is what `scripts/bench_usage.py` does.

## 🚀 Quickstart

```bash
# one-time: clone the working toolchain from an existing pod on the same network volume region
bash scripts/clone_toolchain.sh   # venv + cuda-13.0.tar + debs + kernel caches, ~15 min
# download the NVFP4 model to /workspace/qwen-nvfp4 (hf download, ~24 GB, ~40 s on datacenter link)

bash scripts/start_qwen.sh        # unpack CUDA 13, link caches, launch tmux session "serve"
# health in ~6 min (first ever boot ~13 min), then:
curl -s http://127.0.0.1:8000/v1/models
python3 scripts/bench_usage.py    # solo + 8-agent grounded benchmark
```

`serve_qwen.sh` flags of record: TP1, 131k `--max-model-len`, `--kv-cache-dtype fp8`,
MTP `num_speculative_tokens: 2`, `--enable-prefix-caching`,
`--max-num-batched-tokens 8192`, `--tool-call-parser qwen3_xml --reasoning-parser qwen3`
(the XML parser matters: with `hermes`, Qwen3-family tool calls leak into content).

## 📝 Notes

- RunPod's container layer is ephemeral: `start_qwen.sh` re-unpacks the CUDA toolkit tar
  and re-links `/root/.cache/{flashinfer,vllm}` to the network volume on every boot.
- The first request after boot is warmup-contaminated (flashinfer autotune) — benchmark warm,
  and re-run the whole suite once before trusting numbers.
- The transformers `Qwen3VLVideoProcessorInitKwargs` docstring "ERROR" lines at startup
  are harmless.
