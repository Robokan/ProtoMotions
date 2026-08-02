#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Train the dm_control dog motion tracker (Newton, headless). The dog is a
# sim-only skeletal model -- no terrain / deployable variant.
#
# Env overrides:
#   GPU=2            pin to a GPU            NUM_ENVS=4096   BATCH_SIZE=16384
# Usage:
#   GPU=2 scripts/train_dog_tracker.sh
#   scripts/train_dog_tracker.sh --checkpoint results/dog_tracker/last.ckpt   # resume
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Activate the Newton uv env
source "$REPO/../.venv-protomotions-newton/bin/activate"

[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES="$GPU"

python protomotions/train_agent.py \
    --robot-name dog_v2 --simulator newton \
    --experiment-path examples/experiments/mimic/quadruped_mlp.py \
    --experiment-name dog_tracker \
    --motion-file data/motions/dog_v2/dog_full.pt \
    --num-envs "${NUM_ENVS:-4096}" --batch-size "${BATCH_SIZE:-16384}" \
    "$@"
