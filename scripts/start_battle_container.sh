#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Start (or attach to) the ProtoMotions battle container: deps pre-baked in
# the protomotions:battle image, X11 + GPU graphics wired for windowed
# viewers, Isaac caches mounted so kit starts warm.
#
# Usage:
#   scripts/start_battle_container.sh              # start or attach
#   NAME=my-run scripts/start_battle_container.sh  # a second container
#
# Inside the container:
#   /workspace/isaaclab/isaaclab.sh -p protomotions/inference_agent.py ...
set -euo pipefail

IMAGE="protomotions:battle"
NAME="${NAME:-battle}"
DISPLAY_NUM="${DISPLAY:-:1}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Image $IMAGE not found. Build it first (from the repo root):"
    echo "  docker build -f Dockerfile.battle -t protomotions:battle ."
    exit 1
fi

if docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
    echo "Container '$NAME' is running. Attaching..."
    exec docker exec -it "$NAME" bash
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true

mkdir -p ~/docker/isaac-sim/cache/{kit,ov,pip,glcache,computecache}
xhost +local:docker >/dev/null 2>&1 || true

docker run -d --name "$NAME" --gpus all --network=host --entrypoint bash \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e DISPLAY="$DISPLAY_NUM" \
    -e PYTORCH_JIT=0 -e TORCHDYNAMO_DISABLE=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$HOME/.Xauthority:/root/.Xauthority" \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v "$HOME/sparkpack:/workspace/sparkpack" \
    --workdir /workspace/sparkpack/ProtoMotions \
    "$IMAGE" -c "tail -f /dev/null" >/dev/null

echo "Container '$NAME' started. Attaching..."
exec docker exec -it "$NAME" bash
