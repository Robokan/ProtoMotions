#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Start (or attach to) a persistent ProtoMotions container using the
# canonical Spark image (protomotions:spark, updated via Dockerfile.spark).
# Environment mirrors ../run_protomotions.sh — host IsaacLab on PYTHONPATH,
# X11 + GPU graphics for windowed viewers, Blackwell JIT workarounds — but
# keeps the container running so training jobs survive shell exits.
#
# Usage:
#   scripts/start_battle_container.sh              # start or attach
#   NAME=my-run scripts/start_battle_container.sh  # a second container
#
# Inside the container (Isaac Sim python):
#   python protomotions/inference_agent.py ...
set -euo pipefail

IMAGE="protomotions:spark"
NAME="${NAME:-battle}"
DISPLAY_NUM="${DISPLAY:-:1}"

ISAACLAB_PATH="/workspace/sparkpack/IsaacLab"
PROTOMOTIONS_PATH="/workspace/sparkpack/ProtoMotions"
PYTHONPATH_FULL="${PROTOMOTIONS_PATH}:${ISAACLAB_PATH}/source:${ISAACLAB_PATH}/source/isaaclab:${ISAACLAB_PATH}/source/isaaclab_tasks:${ISAACLAB_PATH}/source/isaaclab_rl:${ISAACLAB_PATH}/source/isaaclab_mimic"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Image $IMAGE not found. Build the update layer (from the repo root):"
    echo "  docker build -f Dockerfile.spark -t protomotions:spark ."
    exit 1
fi

if docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
    echo "Container '$NAME' is running. Attaching..."
    exec docker exec -it "$NAME" bash
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true

# Blackwell (compute capability 12.x) needs CUDA arch pinning + no Dynamo
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
EXTRA_ENV_ARGS=()
if [[ "$COMPUTE_CAP" == 12.* ]]; then
    EXTRA_ENV_ARGS=(-e PYTORCH_JIT=0 -e TORCH_CUDA_ARCH_LIST=12.0 -e TORCHDYNAMO_DISABLE=1)
fi

xhost +local:docker >/dev/null 2>&1 || true

docker run -d --name "$NAME" --gpus all --network=host \
    -e DISPLAY="$DISPLAY_NUM" \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES \
    -e PYTHONPATH="$PYTHONPATH_FULL" \
    "${EXTRA_ENV_ARGS[@]}" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$HOME/sparkpack/isaac-sim-data:/root/.nvidia-omniverse" \
    -v "$HOME/sparkpack:/workspace/sparkpack" \
    --workdir /workspace/sparkpack/ProtoMotions \
    --entrypoint bash \
    "$IMAGE" -c "rm -rf /workspace/IsaacLab 2>/dev/null || true; tail -f /dev/null" >/dev/null

# `python` wrapper: the image's default user (isaac-sim) can't write
# /usr/local/bin, and a plain symlink breaks python.sh's self-relative paths.
docker exec -u root "$NAME" bash -c \
    'printf "#!/bin/bash\nexec /isaac-sim/python.sh \"\$@\"\n" > /usr/local/bin/python && chmod 755 /usr/local/bin/python'

echo "Container '$NAME' started. Attaching..."
exec docker exec -it "$NAME" bash
