#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# SOMA_GPC_COMBAT_PLAN Phase 1 driver: build the combat + BONES-SEED SOMA
# dataset. Runs whichever steps have their inputs available and prints
# what is still missing.
#
# Inputs (set env vars or edit the defaults):
#   SEED_BVH_DIR    BONES-SEED "SOMA Uniform" BVH directory
#                   (huggingface.co/datasets/bones-studio/seed)
#   COMBAT_BVH_DIR  Combat clips retargeted to the 77-joint SOMA BVH
#                   convention (DCC retarget of the IsaacLabASE
#                   reallusion_combat/combat sources -- manual step, see
#                   SOMA_GPC_COMBAT_PLAN.md 1b)
#   MOTIONS_ROOT    Output root for converted .motion clips
#
# Outputs:
#   data/soma_combat_seed.pt   combined library (prior training, Phase 3)
#   data/soma_combat_only.pt   combat-only library (SFT + tracker eval)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

SEED_BVH_DIR="${SEED_BVH_DIR:-}"
COMBAT_BVH_DIR="${COMBAT_BVH_DIR:-}"
MOTIONS_ROOT="${MOTIONS_ROOT:-data/motions/soma_combat}"

missing=0

# --- 1a. BONES-SEED base (breadth) -----------------------------------------
if [ -n "$SEED_BVH_DIR" ] && [ -d "$SEED_BVH_DIR" ]; then
    echo "[1a] Converting BONES-SEED BVH -> proto (.motion) ..."
    python data/scripts/convert_soma23_bvh_to_proto.py \
        --input-dir "$SEED_BVH_DIR" \
        --output-dir "$MOTIONS_ROOT/seed" \
        --input-fps 120 --output-fps 30
else
    echo "[1a] SKIP: SEED_BVH_DIR not set or missing."
    echo "     Download the SOMA Uniform BVH variant of BONES-SEED:"
    echo "     https://huggingface.co/datasets/bones-studio/seed"
    echo "     (see docs/source/getting_started/seed_bvh_preparation.rst)"
    missing=1
fi

# --- 1b. Combat mocap -> SOMA ----------------------------------------------
if [ -n "$COMBAT_BVH_DIR" ] && [ -d "$COMBAT_BVH_DIR" ]; then
    echo "[1b] Converting combat BVH -> proto (.motion) ..."
    python data/scripts/convert_soma23_bvh_to_proto.py \
        --input-dir "$COMBAT_BVH_DIR" \
        --output-dir "$MOTIONS_ROOT/combat" \
        --input-fps 120 --output-fps 30
else
    echo "[1b] SKIP: COMBAT_BVH_DIR not set or missing."
    echo "     Manual step: retarget the combat clips (strikes, blocks,"
    echo "     dodges, footwork, knockdowns, get-ups) to the 77-joint SOMA"
    echo "     BVH convention in a DCC tool. Sources:"
    echo "     ../IsaacLabASE/source/IsaacLabASE/ase/poselib/data/animations/amp/combat/"
    echo "     ../IsaacLabASE/source/IsaacLabASE/ase/poselib/data/animations/amp/reallusion_combat/"
    echo "     (clip .npy files are gitignored there -- regenerate or copy"
    echo "     them from the machine that produced them)"
    missing=1
fi

# --- 1c. Package MotionLibs --------------------------------------------------
if [ -d "$MOTIONS_ROOT/seed" ] || [ -d "$MOTIONS_ROOT/combat" ]; then
    echo "[1c] Packaging MotionLib libraries ..."
    if [ -d "$MOTIONS_ROOT/combat" ]; then
        python protomotions/components/motion_lib.py \
            --motion-path "$MOTIONS_ROOT/combat/" \
            --output-file data/soma_combat_only.pt --device cpu
        echo "     -> data/soma_combat_only.pt"
    fi
    if [ -d "$MOTIONS_ROOT/seed" ] && [ -d "$MOTIONS_ROOT/combat" ]; then
        python protomotions/components/motion_lib.py \
            --motion-path "$MOTIONS_ROOT/" \
            --output-file data/soma_combat_seed.pt --device cpu
        echo "     -> data/soma_combat_seed.pt"
    fi
fi

if [ "$missing" -eq 1 ]; then
    echo ""
    echo "Some inputs are missing (see SKIP notes above)."
    echo "Milestone check before training: inspect the retargeted clips with"
    echo "  python examples/motion_libs_visualizer.py --robot soma23"
    exit 1
fi
echo "Phase 1 dataset build complete."
