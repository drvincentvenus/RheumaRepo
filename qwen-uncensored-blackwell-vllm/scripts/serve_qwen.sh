#!/usr/bin/env bash
# vLLM single-GPU (RTX PRO 6000 96GB) - Qwen3.8-27B-Uncensored NVFP4 + MTP d2 + graphs + fp8 KV.
set -euo pipefail
source /workspace/vllm-env/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

exec vllm serve /workspace/qwen-nvfp4 \
  --served-model-name qwen3.8-27b-uncensored \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --max-num-seqs 8 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.92 \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":2}" \
  --enable-prefix-caching \
  --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000
