#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Run the trained ANYmal-D MaskedMimic policy in IsaacLab with a SINGLE env
# (windowed). MaskedMimic is the masked/generative controller distilled from the
# flat tracker (see scripts/train_anymal_masked_mimic.sh). The policy config is
# reconstructed automatically from the checkpoint's saved inference config.
#
# Usage:
#   scripts/run_anymal_masked_mimic.sh                                  # default ckpt
#   CKPT=results/anymal_masked_mimic_v1/last.ckpt scripts/run_anymal_masked_mimic.sh
#   GPU=0 scripts/run_anymal_masked_mimic.sh --full-eval                # extra args pass through
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Activate the IsaacLab uv env
source "$REPO/../.venv-isaacsim5/bin/activate"

export OMNI_KIT_ACCEPT_EULA=YES
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"
[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES="$GPU"

CKPT="${CKPT:-results/anymal_masked_mimic_v1/last.ckpt}"

python protomotions/inference_agent.py \
    --checkpoint "$CKPT" \
    --motion-file data/motions/anymal_d/anymal_d_flat.pt \
    --simulator isaaclab \
    --num-envs 1 \
    "$@"
