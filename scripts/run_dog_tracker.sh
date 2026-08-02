#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Run the trained dm_control dog motion tracker in Newton with a SINGLE env (windowed).
# The dog is a sim-only skeletal model (not deployable).
#
# Usage:
#   scripts/run_dog_tracker.sh
#   CKPT=results/foo/last.ckpt scripts/run_dog_tracker.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Activate the Newton uv env
source "$REPO/../.venv-protomotions-newton/bin/activate"

export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"

CKPT="${CKPT:-results/dog_tracker/last.ckpt}"

python protomotions/inference_agent.py \
    --checkpoint "$CKPT" \
    --motion-file data/motions/dog_v2/dog_full.pt \
    --simulator newton \
    --num-envs 1 \
    "$@"
