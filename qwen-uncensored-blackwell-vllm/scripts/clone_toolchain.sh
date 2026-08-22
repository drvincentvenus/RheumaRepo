#!/usr/bin/env bash
set -uo pipefail
OLD_IP=<SOURCE_POD_IP>; OLD_PORT=<SOURCE_POD_SSH_PORT>
SSHOPT="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -p $OLD_PORT -i /root/.ssh/id_ed25519"
echo "[clone] starting $(date)"
rsync -a --info=progress2 -e "ssh $SSHOPT" \
  root@$OLD_IP:/workspace/vllm-env \
  root@$OLD_IP:/workspace/cuda-13.0.tar \
  root@$OLD_IP:/workspace/debs \
  root@$OLD_IP:/workspace/cache \
  root@$OLD_IP:/workspace/setup.sh \
  /workspace/ 2>&1 | tail -2
echo "[clone] done $(date); du -sh:"
du -sh /workspace/vllm-env /workspace/cache /workspace/cuda-13.0.tar 2>/dev/null
