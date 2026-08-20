#!/usr/bin/env bash
# Run ONCE while the pod is healthy: snapshot the ephemeral container layer to /workspace
# so a stop/start needs no network and no recompilation.
set -euo pipefail
mkdir -p /workspace/cache /workspace/debs
tar -C /usr/local -cf /workspace/cuda-13.0.tar cuda-13.0
cp /usr/bin/ninja /workspace/debs/ninja.bin
cp -rn /root/.cache/flashinfer /workspace/cache/ 2>/dev/null || true
cp -rn /root/.cache/vllm /workspace/cache/ 2>/dev/null || true
cat > /workspace/start.sh <<'BOOT'
#!/usr/bin/env bash
set -uo pipefail
[ -d /usr/local/cuda-13.0 ] || tar -C /usr/local -xf /workspace/cuda-13.0.tar
cp -n /workspace/debs/ninja.bin /usr/bin/ninja 2>/dev/null; chmod +x /usr/bin/ninja
apt-get install -y -qq tmux >/dev/null 2>&1 || true
mkdir -p /root/.cache
rm -rf /root/.cache/flashinfer /root/.cache/vllm
ln -sfn /workspace/cache/flashinfer /root/.cache/flashinfer
ln -sfn /workspace/cache/vllm /root/.cache/vllm
tmux kill-session -t serve 2>/dev/null
tmux new-session -d -s serve "bash /path/to/serve.sh > /workspace/serve.log 2>&1"
echo "serving; health in ~4-6 min"
BOOT
chmod +x /workspace/start.sh
echo "Persisted. After any pod restart, run: bash /workspace/start.sh"
