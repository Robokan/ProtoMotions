#!/bin/bash
# Demo: MaskedMimic with Complex Terrain + Sparse Keypoint Control
# Uses procedural terrain (slopes, stairs, obstacles)
# Only provides pelvis, hands, and feet positions at 5 Hz
# MaskedMimic "inpaints" the full body motion

set -e

# Assume this script is run from the ProtoMotions repo root

# Activate IsaacLab environment
source ../IsaacLab/env_isaaclab/bin/activate
source ../IsaacLab/_isaac_sim/setup_conda_env.sh

# Configuration - checkpoints are in ProtoMotions/results
CHECKPOINT="${1:-results/masked_mimic_g1_phuma_terrain/last.ckpt}"
MOTION_FILE="${2:-data/motions/g1_random_subset_tiny.pt}"
NUM_ENVS="${3:-1}"
CONTROL_HZ="${4:-50.0}"  # 50=every step, 5=sparse like GR00T

# Convert control Hz to target update interval (assumes 50 Hz sim by default)
SIM_HZ=50
TARGET_UPDATE_INTERVAL=$(
  CONTROL_HZ="$CONTROL_HZ" SIM_HZ="$SIM_HZ" python - <<'PY'
import os
control_hz = float(os.environ.get("CONTROL_HZ", "50"))
sim_hz = float(os.environ.get("SIM_HZ", "50"))
if control_hz <= 0:
    print(1)
else:
    print(max(1, int(round(sim_hz / control_hz))))
PY
)

echo "============================================"
echo "ProtoMotions - MaskedMimic TERRAIN Demo"
echo "============================================"
echo "Checkpoint: $CHECKPOINT"
echo "Motion file: $MOTION_FILE"
echo "Num envs: $NUM_ENVS"
echo "Control rate: $CONTROL_HZ Hz"
echo ""
echo "Terrain types:"
echo "  - 20% smooth slopes"
echo "  - 10% rough slopes"
echo "  - 10% stairs up, 10% stairs down"
echo "  - 5% discrete obstacles"
echo "  - 45% flat ground"
echo ""
echo "Sparse Keypoints (5 bodies):"
echo "  - torso_link (pelvis)"
echo "  - left_rubber_hand, right_rubber_hand"
echo "  - left_ankle_roll_link, right_ankle_roll_link"
echo ""
echo "MaskedMimic inpaints the full body motion on terrain!"
echo ""
echo "Camera Controls (keyboard):"
echo "  TAB/= : Next robot  |  - : Previous robot"
echo "  T : Toggle tracking  |  F : Toggle following"
echo "  LEFT/RIGHT : Rotate  |  PAGE_UP/DOWN : Distance"
echo "  UP/DOWN : Camera height  |  R : Reset animation"
echo "============================================"
echo ""

python protomotions/inference_agent.py \
    --checkpoint "$CHECKPOINT" \
    --motion-file "$MOTION_FILE" \
    --num-envs "$NUM_ENVS" \
    --simulator isaaclab \
    --target-update-interval "$TARGET_UPDATE_INTERVAL" \
    --overrides "terrain.height_multiplier=0.25"
