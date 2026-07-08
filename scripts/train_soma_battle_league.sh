#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Train the SOMA battle league (SOMA_GPC_COMBAT_PLAN Phase 6): PPO over the
# frozen GPC prior's tokens against PFSP-sampled league snapshots, warm-started
# from the combat SFT adapter.
#
# NUM_ENVS is the TOTAL env count = 2x parallel matches.
#
# Env overrides:
#   GPU=0                        pin a GPU        NUM_ENVS=1024  BATCH_SIZE=4096
#   EXP=soma_battle_league       experiment name  NO_RESUME=1    force fresh
#   PRIOR=results/soma_gpc_prior/last.ckpt        frozen prior checkpoint
#   SFT=results/soma_sft_combat/last.ckpt         SFT warm-start
#   MOTIONS=data/soma_combat_only.pt              combat motion library
#   ROLE=main                                    or main_exploiter
#   OPP_DIR=                                     league dir (exploiter role)
# Usage:
#   GPU=0 scripts/train_soma_battle_league.sh                 # start or resume
#   GPU=1 ROLE=main_exploiter OPP_DIR=results/soma_battle_league/league \
#       EXP=soma_battle_exploiter scripts/train_soma_battle_league.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES="$GPU"

EXP="${EXP:-soma_battle_league}"
PRIOR="${PRIOR:-results/soma_gpc_prior/last.ckpt}"
SFT="${SFT:-results/soma_sft_combat/last.ckpt}"
MOTIONS="${MOTIONS:-data/soma_combat_only.pt}"
ROLE="${ROLE:-main}"

RESUME_ARGS=()
if [ -z "${NO_RESUME:-}" ] && [ -f "results/$EXP/last.ckpt" ]; then
    RESUME_ARGS+=(--checkpoint "results/$EXP/last.ckpt")
elif [ -f "$SFT" ]; then
    RESUME_ARGS+=(--checkpoint "$SFT")
fi

ROLE_ARGS=(--league-role "$ROLE")
if [ "$ROLE" = "main_exploiter" ]; then
    : "${OPP_DIR:?main_exploiter requires OPP_DIR (the main run's league dir)}"
    ROLE_ARGS+=(--league-opponent-dir "$OPP_DIR" --peft-sampling-mode nucleus)
fi

python protomotions/train_agent.py \
    --robot-name soma23 --simulator isaaclab --headless \
    --experiment-path examples/experiments/battle/battle_league_prior_peft.py \
    --experiment-name "$EXP" \
    --motion-file "$MOTIONS" \
    --prior-checkpoint "$PRIOR" \
    --num-envs "${NUM_ENVS:-1024}" --batch-size "${BATCH_SIZE:-4096}" \
    "${ROLE_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    "$@"
