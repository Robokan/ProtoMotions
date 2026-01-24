#!/bin/bash
# Run SMPL MaskedMimic demo
#
# NOTE: The pretrained SMPL MaskedMimic was trained on IsaacLab (not IsaacGym)
# so we use the IsaacLab environment here.

set -e

# Assume this script is run from the ProtoMotions repo root

# Activate IsaacLab environment
source ../IsaacLab/env_isaaclab/bin/activate
source ../IsaacLab/_isaac_sim/setup_conda_env.sh

# Default values
CHECKPOINT="${1:-data/pretrained_models/masked_mimic/smpl/last.ckpt}"
MOTION_FILE="${2:-examples/data/smpl_humanoid_sit_armchair.motion}"
NUM_ENVS="${3:-1}"
CONTROL_HZ="${4:-30}"

# Convert control Hz to target update interval (assumes 50 Hz sim by default)
SIM_HZ=50
TARGET_UPDATE_INTERVAL=$(
  CONTROL_HZ="$CONTROL_HZ" SIM_HZ="$SIM_HZ" python - <<'PY'
import math
import os
control_hz = float(os.environ.get("CONTROL_HZ", "30"))
sim_hz = float(os.environ.get("SIM_HZ", "50"))
if control_hz <= 0:
    print(1)
else:
    print(max(1, int(round(sim_hz / control_hz))))
PY
)

echo "SMPL MaskedMimic Demo (IsaacLab)"
echo "================================"
echo "Checkpoint: $CHECKPOINT"
echo "Motion file: $MOTION_FILE"
echo "Num envs: $NUM_ENVS"
echo "Control Hz: $CONTROL_HZ"
echo "Sparse Keypoints: Pelvis, L_Hand, R_Hand, L_Ankle, R_Ankle"
echo "MaskedMimic fills in: Head, Spine, Arms, Legs, etc."
echo ""
echo "Controls:"
echo "  Q - Quit"
echo "  R - Reset environments"
echo "  J - Push humanoid (test robustness)"
echo ""

python protomotions/inference_agent.py \
    --checkpoint "$CHECKPOINT" \
    --motion-file "$MOTION_FILE" \
    --num-envs "$NUM_ENVS" \
    --simulator isaaclab \
    --target-update-interval "$TARGET_UPDATE_INTERVAL"
