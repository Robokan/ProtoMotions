#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Combat SFT (SOMA_GPC_COMBAT_PLAN Phase 4): bias the GPC prior toward
# fighting on the combat-only library, conditioned on the same virtual
# opponent obs the battle RLFT uses.
#
# Env overrides:
#   GPU=0                     pin a GPU     NUM_ENVS=1024  BATCH_SIZE=1024
#   EXP=soma_sft_combat       experiment name
#   PRIOR=results/soma_gpc_prior/last.ckpt
#   MOTIONS=data/soma_combat_only.pt
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES="$GPU"

EXP="${EXP:-soma_sft_combat}"
PRIOR="${PRIOR:-results/soma_gpc_prior/last.ckpt}"
MOTIONS="${MOTIONS:-data/soma_combat_only.pt}"

RESUME_ARGS=()
if [ -z "${NO_RESUME:-}" ] && [ -f "results/$EXP/last.ckpt" ]; then
    RESUME_ARGS+=(--checkpoint "results/$EXP/last.ckpt")
fi

python protomotions/train_agent.py \
    --robot-name soma23 --simulator isaaclab --headless \
    --experiment-path examples/experiments/gpc/sft_combat_prior_peft.py \
    --experiment-name "$EXP" \
    --motion-file "$MOTIONS" \
    --prior-checkpoint "$PRIOR" \
    --num-envs "${NUM_ENVS:-1024}" --batch-size "${BATCH_SIZE:-1024}" \
    "${RESUME_ARGS[@]}" \
    "$@"
