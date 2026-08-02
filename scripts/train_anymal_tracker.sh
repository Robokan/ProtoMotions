#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Train the ANYmal-D motion tracker (IsaacLab, headless) on the flat
# uniqueness-weighted lib (anymal_d_flat.pt, no support terrain). Matches the
# anymal_flat_v1 run; re-run to resume it.
#
# Resumes automatically from results/$EXP/last.ckpt if it exists (training a
# fresh run otherwise). Reusing an experiment name restores its saved config.
#
# Env overrides:
#   GPU=2                  pin to a GPU       NUM_ENVS=16384   BATCH_SIZE=65536
#   EXP=anymal_flat_v1     experiment name    NO_RESUME=1   force a fresh run
# Usage:
#   GPU=2 scripts/train_anymal_tracker.sh                       # start or resume
#   GPU=2 NO_RESUME=1 scripts/train_anymal_tracker.sh           # force fresh
#   GPU=2 scripts/train_anymal_tracker.sh --checkpoint <path>   # explicit resume
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Activate the IsaacLab uv env
source "$REPO/../.venv-isaacsim5/bin/activate"

export OMNI_KIT_ACCEPT_EULA=YES
[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES="$GPU"

EXP="${EXP:-anymal_flat_v1}"

# Auto-resume from the latest checkpoint unless the caller forced a fresh run or
# passed their own --checkpoint.
RESUME=""
if [ -z "${NO_RESUME:-}" ] && [[ "$*" != *--checkpoint* ]] \
   && [ -f "results/$EXP/last.ckpt" ]; then
    RESUME="--checkpoint results/$EXP/last.ckpt"
    echo "Resuming from results/$EXP/last.ckpt"
fi

python protomotions/train_agent.py \
    --robot-name anymal_d --simulator isaaclab \
    --experiment-path examples/experiments/mimic/quadruped_mlp.py \
    --experiment-name "$EXP" \
    --motion-file data/motions/anymal_d/anymal_d_flat.pt \
    --num-envs "${NUM_ENVS:-16384}" --batch-size "${BATCH_SIZE:-65536}" \
    $RESUME "$@"
