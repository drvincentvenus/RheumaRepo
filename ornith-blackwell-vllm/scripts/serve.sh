#!/usr/bin/env bash
# vLLM TP2 + MTP speculative decoding + CUDA graphs + fp8 KV + prefix caching on 2x RTX 5090.
# NCCL_CUMEM_ENABLE=1 is what makes CUDA graph capture work with TP on no-P2P GPUs.
set -euo pipefail
source /workspace/vllm-env/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export NCCL_P2P_DISABLE=1
export NCCL_CUMEM_ENABLE=1               # <- the graph-capture deadlock fix
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

exec vllm serve /workspace/ornith-nvfp4 \
  --served-model-name ornith-1.5 \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.92 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --enable-prefix-caching \
  --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000
