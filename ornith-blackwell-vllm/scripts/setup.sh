#!/usr/bin/env bash
# One-shot setup for Blackwell (sm_120) vLLM serving. Re-run after every pod restart
# (the container layer is ephemeral on RunPod; only the volume disk persists).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "== apt: fix NVIDIA repo Signed-By conflict, add cuda-keyring =="
rm -f /etc/apt/sources.list.d/cuda*.list
echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
  > /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list
curl -sL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb >/dev/null
apt-get update -qq

echo "== CUDA 13.0 toolkit (nvcc>=12.9 + ALL headers: flashinfer JIT needs curand etc) + ninja =="
apt-get install -y -qq ninja-build cuda-toolkit-13-0 tmux
ls /usr/local/cuda-13.0/include/curand_kernel.h >/dev/null || { echo "curand header missing"; exit 1; }

echo "== vLLM in a uv venv =="
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }
[ -d /workspace/vllm-env ] || uv venv --python 3.12 /workspace/vllm-env
source /workspace/vllm-env/bin/activate
uv pip install -q vllm 'huggingface_hub[cli]'
python -c "import vllm; print('vLLM', vllm.__version__)"

echo "== model =="
[ -f /workspace/ornith-nvfp4/config.json ] || \
  hf download ornith-ai/Ornith-1.5-35B-A3B-NVFP4 --local-dir /workspace/ornith-nvfp4
echo "SETUP OK"
