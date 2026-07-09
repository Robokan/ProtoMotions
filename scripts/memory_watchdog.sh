#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Unified-memory watchdog for training on DGX Spark (GB10).
#
# The GB10's 128GB is shared between CPU and GPU. When a training run
# exhausts it, the NVIDIA driver deadlocks (nvidia-modeset blocks on a lock
# held by the training process) and freezes the whole machine — a killed
# training run resumes from its checkpoint, a starved driver requires a hard
# reboot. This watchdog kills train_agent.py inside the given container when
# host MemAvailable drops below the threshold.
#
# Usage:  scripts/memory_watchdog.sh [container=battle-shakedown] [min_gb=10]
set -euo pipefail

CONTAINER="${1:-battle-shakedown}"
MIN_GB="${2:-10}"

echo "watchdog: killing train_agent.py in '$CONTAINER' if MemAvailable < ${MIN_GB}GB"
while true; do
    avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    avail_gb=$((avail_kb / 1024 / 1024))
    if [ "$avail_gb" -lt "$MIN_GB" ]; then
        pids=$(docker exec "$CONTAINER" pgrep -f train_agent.py 2>/dev/null | tr '\n' ' ' || true)
        if [ -n "$pids" ]; then
            echo "watchdog: MemAvailable=${avail_gb}GB < ${MIN_GB}GB — killing training ($pids)"
            docker exec "$CONTAINER" bash -c "kill -9 $pids" || true
            echo "watchdog: training killed; resume from its last checkpoint"
            exit 1
        fi
    fi
    sleep 10
done
