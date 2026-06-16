#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Train the Go2 motion tracker (IsaacLab, headless) with the support terrain.
# For a sim2real-deployable tracker instead, swap the experiment for
# examples/experiments/mimic/quadruped_bm_deploy.py and drop the terrain overrides.
#
# Env overrides:
#   GPU=2            pin to a GPU            NUM_ENVS=8192   BATCH_SIZE=32768
# Usage:
#   GPU=2 scripts/train_go2_tracker.sh
#   scripts/train_go2_tracker.sh --checkpoint results/go2_tracker/last.ckpt   # resume
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Activate the IsaacLab uv env
source "$REPO/../.venv-isaacsim5/bin/activate"

export OMNI_KIT_ACCEPT_EULA=YES
[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES="$GPU"

python protomotions/train_agent.py \
    --robot-name go2 --simulator isaaclab \
    --experiment-path examples/experiments/mimic/quadruped_mlp.py \
    --experiment-name go2_tracker \
    --motion-file data/motions/go2/go2_full.pt \
    --num-envs "${NUM_ENVS:-8192}" --batch-size "${BATCH_SIZE:-32768}" \
    --overrides terrain.motion_support_manifest=data/motions/go2/support_manifest.yaml \
                terrain.motion_support_motion_lib=data/motions/go2/go2_full.pt \
    "$@"
