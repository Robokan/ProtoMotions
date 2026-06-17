#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Train the ANYmal-D MaskedMimic policy (IsaacLab, headless) by DISTILLING the
# flat motion tracker (the anymal_flat_v1 "expert"). MaskedMimic learns to
# reproduce the expert's actions under randomly-masked partial conditioning, so
# it requires a trained tracker checkpoint -- it is NOT trained from scratch.
#
# The MaskedMimic experiment (masked_mimic/transformer.py) is robot-agnostic: it
# reads the conditionable bodies from anymal_d's trackable_bodies_subset.
#
# MaskedMimic is heavier per-env than the MLP tracker (transformer student +
# frozen expert), so the env count is lower than the tracker's: 4096 fits a 24 GB
# card; 8192 OOMs. Bump NUM_ENVS on a bigger GPU.
#
# Env overrides:
#   GPU=1                       pin to a GPU       NUM_ENVS=4096  BATCH_SIZE=16384
#   EXP=anymal_masked_mimic_v1  experiment name    NO_RESUME=1   force a fresh run
#   EXPERT=results/anymal_flat_v1/last.ckpt        expert tracker checkpoint
# Usage:
#   GPU=1 scripts/train_anymal_masked_mimic.sh                   # start or resume
#   GPU=1 NO_RESUME=1 scripts/train_anymal_masked_mimic.sh       # force fresh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Activate the IsaacLab uv env
source "$REPO/../.venv-isaacsim5/bin/activate"

export OMNI_KIT_ACCEPT_EULA=YES
# Reduce CUDA fragmentation (the distillation rollout + optimize alloc pattern).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES="$GPU"

EXP="${EXP:-anymal_masked_mimic_v1}"
EXPERT="${EXPERT:-results/anymal_flat_v1/last.ckpt}"

if [ ! -f "$EXPERT" ]; then
    echo "ERROR: expert tracker checkpoint not found: $EXPERT" >&2
    echo "Train the tracker first (scripts/train_anymal_tracker.sh)." >&2
    exit 1
fi

# Auto-resume from the latest checkpoint unless forced fresh or an explicit
# --checkpoint was passed.
RESUME=""
if [ -z "${NO_RESUME:-}" ] && [[ "$*" != *--checkpoint* ]] \
   && [ -f "results/$EXP/last.ckpt" ]; then
    RESUME="--checkpoint results/$EXP/last.ckpt"
    echo "Resuming from results/$EXP/last.ckpt"
fi

python protomotions/train_agent.py \
    --robot-name anymal_d --simulator isaaclab \
    --experiment-path examples/experiments/masked_mimic/transformer.py \
    --experiment-name "$EXP" \
    --motion-file data/motions/anymal_d/anymal_d_flat.pt \
    --expert-model-path "$EXPERT" \
    --num-envs "${NUM_ENVS:-2048}" --batch-size "${BATCH_SIZE:-8192}" \
    $RESUME "$@"
