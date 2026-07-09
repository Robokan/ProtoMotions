#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# SEED -> combat-weighted prior pipeline (plan Phases 1 + 3), end to end:
#   1. convert curated BVH staging tiers to .motion (parallel ranks)
#   2. package into a MotionLib
#   3. apply combat/adjacent group sampling weights
#   4. launch GPC prior training
#
# Run inside the training container. Expects staging dirs from
# data/scripts/curate_seed_combat_subset.py.
#
# Env overrides:
#   STAGING=/workspace/sparkpack/bones-seed/staging
#   MOTIONS=/workspace/sparkpack/bones-seed/motions
#   OUT_LIB=data/soma_seed_curated.pt
#   RANKS=8            parallel conversion processes
#   COMBAT_FRAC=0.25   ADJACENT_FRAC=0.25
#   NUM_ENVS=1024      BATCH_SIZE=1024
#   EXP=soma_gpc_prior
#   SKIP_TRAIN=1       stop after building the weighted library
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${PY:-/workspace/isaaclab/isaaclab.sh -p}"
STAGING="${STAGING:-/workspace/sparkpack/bones-seed/staging}"
MOTIONS="${MOTIONS:-/workspace/sparkpack/bones-seed/motions}"
OUT_LIB="${OUT_LIB:-data/soma_seed_curated.pt}"
RANKS="${RANKS:-8}"
COMBAT_FRAC="${COMBAT_FRAC:-0.25}"
ADJACENT_FRAC="${ADJACENT_FRAC:-0.25}"
EXP="${EXP:-soma_gpc_prior}"
TRACKER="${TRACKER:-data/pretrained_models/motion_tracker/soma_bones_fsq/last.ckpt}"

echo "=== [1/4] Converting BVH -> .motion ($RANKS ranks) ==="
mkdir -p "$MOTIONS"
pids=()
for rank in $(seq 0 $((RANKS - 1))); do
    $PY data/scripts/convert_soma23_bvh_to_proto.py \
        --input-dir "$STAGING" --output-dir "$MOTIONS" \
        --input-fps 120 --output-fps 30 \
        --num-rank "$RANKS" --slurm-rank "$rank" \
        > "$MOTIONS/convert_rank${rank}.log" 2>&1 &
    pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
[ "$fail" -eq 0 ] || { echo "A conversion rank failed; see $MOTIONS/convert_rank*.log"; exit 1; }
echo "Converted: $(find "$MOTIONS" -name '*.motion' | wc -l) clips"

echo "=== [2/4] Packaging MotionLib ==="
$PY protomotions/components/motion_lib.py \
    --motion-path "$MOTIONS/" --output-file "$OUT_LIB" --device cpu

echo "=== [3/4] Applying combat group weights ==="
$PY data/scripts/apply_group_weights.py \
    --motion-lib "$OUT_LIB" \
    --group "/combat/=$COMBAT_FRAC" --group "/adjacent/=$ADJACENT_FRAC"

if [ -n "${SKIP_TRAIN:-}" ]; then
    echo "SKIP_TRAIN set - stopping after library build: $OUT_LIB"
    exit 0
fi

echo "=== [4/4] Launching GPC prior training ($EXP) ==="
RESUME_ARGS=()
if [ -f "results/$EXP/last.ckpt" ]; then
    RESUME_ARGS+=(--checkpoint "results/$EXP/last.ckpt")
fi
$PY protomotions/train_agent.py \
    --robot-name soma23 --simulator isaaclab --headless \
    --motion-file "$OUT_LIB" \
    --experiment-path examples/experiments/gpc/prior.py \
    --tracker-checkpoint "$TRACKER" \
    --num-envs "${NUM_ENVS:-1024}" --batch-size "${BATCH_SIZE:-1024}" \
    --training-max-steps "${TRAIN_STEPS:-100000000}" \
    --experiment-name "$EXP" \
    "${RESUME_ARGS[@]}"
