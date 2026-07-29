#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Record a battle video from the command line.
#
# Records N whole bouts (KO / ring-out / timeout each) of one league snapshot
# vs another into a single mp4 you can open in any player or browser. Uses the
# real IsaacSim render (see battle_tournament.py --record), headless — no
# display needed. Defaults to the latest snapshot vs a random other snapshot
# from the pool; pass A/B checkpoints to override.
#
# Records with the trained prior_constraint decode and stochastic sampling
# (i.e. NOT --deterministic and NOT the nucleus fast-sampling shortcut) so the
# fighters reproduce the combat behavior they were trained with. Greedy/nucleus
# decode collapses toward generic locomotion (they just walk and bump).
#
# Run from the ProtoMotions root inside the training container:
#   scripts/record_fight.sh [bouts] [run_name] [A_ckpt] [B_ckpt]
#
#   scripts/record_fight.sh 3
#   scripts/record_fight.sh 5 soma_battle_league_v3
#   scripts/record_fight.sh 1 soma_battle_league_v3 path/to/policy_9.ckpt path/to/policy_3.ckpt
set -euo pipefail

# Isaac Sim on this box needs the EULA acknowledged non-interactively;
# harmless once already accepted.
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

BOUTS="${1:-3}"
RUN="${2:-soma_battle_league_v6}"
RESOLVED="results/${RUN}/resolved_configs_inference.pt"
OUT_DIR="output/fight_videos"

mapfile -t SNAPS < <(ls -1tr results/"${RUN}"/lightning_logs/*/league/policy_*.ckpt 2>/dev/null || true)
if [ "${#SNAPS[@]}" -eq 0 ]; then
    echo "record_fight: no league snapshots under results/${RUN}/lightning_logs/*/league/" >&2
    exit 1
fi

# Ensure the output dir exists and is writable by whatever container user
# runs this (the canonical `battle` container runs as isaac-sim, not root).
mkdir -p "$OUT_DIR" 2>/dev/null || true
chmod 777 "$OUT_DIR" 2>/dev/null || true

# Default matchup: latest snapshot vs a RANDOM other snapshot from the pool.
# Falls back to itself if there's only one snapshot so far. Override A/B with
# args 3 and 4.
A="${3:-${SNAPS[-1]}}"                        # latest
if [ -n "${4:-}" ]; then
    B="$4"
elif [ "${#SNAPS[@]}" -ge 2 ]; then
    # random index in [0, N-2] — any snapshot except the latest (index N-1)
    B="${SNAPS[$(( RANDOM % (${#SNAPS[@]} - 1) ))]}"
else
    B="${SNAPS[0]}"
fi
stamp="$(date +%Y%m%d-%H%M%S)"
OUT="${OUT_DIR}/fight_$(basename "$A" .ckpt)_vs_$(basename "$B" .ckpt)_${BOUTS}bout_${stamp}.mp4"

echo "record_fight: ${BOUTS} bout(s), $(basename "$A" .ckpt) vs $(basename "$B" .ckpt) -> ${OUT}"
python protomotions/battle_tournament.py \
    --resolved-configs "$RESOLVED" \
    --exhibition "$A" "$B" \
    --record "$OUT" --bouts "$BOUTS" \
    --overlay-ambient "${AMBIENT:-600}" \
    --num-envs 2 --no-fast-sampling

echo "record_fight: done -> ${OUT}"
