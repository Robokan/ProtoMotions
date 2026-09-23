#!/bin/bash
# Launch any ProtoMotions entrypoint on the GB10 (ARM64, Isaac Sim 6 / Isaac Lab 3).
#
# Sets the two environment variables this box requires — WITHOUT them Isaac Sim
# either wedges silently or dies on an interactive prompt (see GB10_SETUP.md).
#
#   ./run_go2.sh protomotions/inference_agent.py --checkpoint ... --headless
#   ./run_go2.sh protomotions/train_agent.py --robot-name go2 ...
set -euo pipefail

# Kit on aarch64 needs libgomp loaded first; otherwise the process hangs at 0% CPU forever.
export LD_PRELOAD="${LD_PRELOAD:-}:/lib/aarch64-linux-gnu/libgomp.so.1"
# Non-interactive EULA; otherwise "Unable to bootstrap inner kit kernel: EOF when reading a line".
export OMNI_KIT_ACCEPT_EULA=YES

PY=/home/evaughan/sparkpack/.venv-isaacsim6/bin/python
cd "$(dirname "$0")"

# Stale carb shm from a previous kill -9 causes an immediate exit-139 segfault.
# Safe to clear when no Kit process is running.
if ! pgrep -x python >/dev/null 2>&1; then
    rm -f /dev/shm/carb-RStringInternals-* \
          /dev/shm/sem.carb-RStringInternals-* \
          /dev/shm/sem.carbonite-sharedmemory 2>/dev/null || true
fi

exec "$PY" "$@"
