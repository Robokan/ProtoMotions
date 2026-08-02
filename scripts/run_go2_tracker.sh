#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Run the trained Go2 motion tracker in IsaacLab with a SINGLE env (windowed).
#
# Usage:
#   scripts/run_go2_tracker.sh                 # default checkpoint
#   CKPT=results/foo/last.ckpt scripts/run_go2_tracker.sh
#   scripts/run_go2_tracker.sh --full-eval     # extra args pass through
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Activate the IsaacLab uv env
source "$REPO/../.venv-isaacsim5/bin/activate"

export OMNI_KIT_ACCEPT_EULA=YES
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"

CKPT="${CKPT:-results/go2_tracker/last.ckpt}"

python protomotions/inference_agent.py \
    --checkpoint "$CKPT" \
    --motion-file data/motions/go2/go2_full.pt \
    --simulator isaaclab \
    --num-envs 1 \
    "$@"
