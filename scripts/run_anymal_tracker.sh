#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Run the trained ANYmal-D motion tracker in IsaacLab with a SINGLE env (windowed).
# The support terrain is reconstructed automatically from the checkpoint's saved
# inference config (no overrides needed here).
#
# Usage:
#   scripts/run_anymal_tracker.sh                                   # default checkpoint
#   CKPT=results/anymal_split_terrain_v1/last.ckpt scripts/run_anymal_tracker.sh
#   scripts/run_anymal_tracker.sh --full-eval                       # extra args pass through
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Activate the IsaacLab uv env
source "$REPO/../.venv-isaacsim5/bin/activate"

export OMNI_KIT_ACCEPT_EULA=YES
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"

CKPT="${CKPT:-results/anymal_flat_v1/last.ckpt}"

python protomotions/inference_agent.py \
    --checkpoint "$CKPT" \
    --motion-file data/motions/anymal_d/anymal_d_flat.pt \
    --simulator isaaclab \
    --num-envs 1 \
    "$@"
