#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Battle evaluation tournament (SOMA_GPC_COMBAT_PLAN Phase 7).
#
# Round-robin ladder over the league (default):
#   scripts/run_soma_battle_tournament.sh
# Exhibition match (viewer on):
#   MODE=exhibition A=league/policy_5.ckpt B=league/policy_9.ckpt \
#       scripts/run_soma_battle_tournament.sh
# Regression gate:
#   MODE=gate A=<candidate> B=<previous> scripts/run_soma_battle_tournament.sh
#
# Env overrides:
#   EXP=soma_battle_league   league run name     GPU=0        pin a GPU
#   MATCHES=32               matches per pairing NUM_ENVS=128
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES="$GPU"

EXP="${EXP:-soma_battle_league}"
MODE="${MODE:-ladder}"
MATCHES="${MATCHES:-32}"
RESOLVED="results/$EXP/resolved_configs_inference.pt"

case "$MODE" in
ladder)
    python protomotions/battle_tournament.py \
        --resolved-configs "$RESOLVED" \
        --adapters "results/$EXP/league/" \
        --matches-per-pairing "$MATCHES" \
        --num-envs "${NUM_ENVS:-128}" --headless \
        --output "results/$EXP/tournament_report.json" "$@"
    ;;
exhibition)
    : "${A:?exhibition requires A=<ckpt>}" "${B:?exhibition requires B=<ckpt>}"
    python protomotions/battle_tournament.py \
        --resolved-configs "$RESOLVED" \
        --exhibition "$A" "$B" \
        --matches-per-pairing "${MATCHES:-4}" \
        --num-envs "${NUM_ENVS:-2}" --deterministic "$@"
    ;;
gate)
    : "${A:?gate requires A=<candidate>}" "${B:?gate requires B=<previous>}"
    python protomotions/battle_tournament.py \
        --resolved-configs "$RESOLVED" \
        --gate "$A" --gate-against "$B" \
        --matches-per-pairing "${MATCHES:-64}" \
        --num-envs "${NUM_ENVS:-128}" --headless "$@"
    ;;
*)
    echo "Unknown MODE=$MODE (ladder|exhibition|gate)" >&2
    exit 1
    ;;
esac
