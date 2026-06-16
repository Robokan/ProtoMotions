#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Train the ANYmal-D motion tracker (IsaacLab, headless) on the uniqueness-weighted
# split lib + support terrain (the climb sub-clips spawn on terrain, the flat
# sub-clips on flat ground).
# For a sim2real-deployable tracker instead, swap the experiment for
# examples/experiments/mimic/quadruped_bm_deploy.py and drop the terrain overrides.
#
# Env overrides:
#   GPU=2            pin to a GPU            NUM_ENVS=12288   BATCH_SIZE=49152
# Usage:
#   GPU=2 scripts/train_anymal_tracker.sh
#   scripts/train_anymal_tracker.sh --checkpoint results/anymal_tracker/last.ckpt   # resume
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Activate the IsaacLab uv env
source "$REPO/../.venv-isaacsim5/bin/activate"

export OMNI_KIT_ACCEPT_EULA=YES
[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES="$GPU"

python protomotions/train_agent.py \
    --robot-name anymal_d --simulator isaaclab \
    --experiment-path examples/experiments/mimic/quadruped_mlp.py \
    --experiment-name anymal_tracker \
    --motion-file data/motions/anymal_d/anymal_d_split.pt \
    --num-envs "${NUM_ENVS:-12288}" --batch-size "${BATCH_SIZE:-49152}" \
    --overrides terrain.motion_support_manifest=data/motions/anymal_d/support_manifest_split.yaml \
                terrain.motion_support_motion_lib=data/motions/anymal_d/anymal_d_split.pt \
    "$@"
