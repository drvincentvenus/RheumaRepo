#!/usr/bin/env bash
set -uo pipefail
[ -d /usr/local/cuda-13.0 ] || tar -C /usr/local -xf /workspace/cuda-13.0.tar
cp -n /workspace/debs/ninja.bin /usr/bin/ninja 2>/dev/null; chmod +x /usr/bin/ninja 2>/dev/null
apt-get install -y -qq tmux >/dev/null 2>&1 || true
mkdir -p /root/.cache
rm -rf /root/.cache/flashinfer /root/.cache/vllm
ln -sfn /workspace/cache/flashinfer /root/.cache/flashinfer
ln -sfn /workspace/cache/vllm /root/.cache/vllm
tmux kill-session -t serve 2>/dev/null
tmux new-session -d -s serve "bash /workspace/serve_qwen.sh > /workspace/serve_qwen.log 2>&1"
echo "serving qwen; health in ~6 min"
