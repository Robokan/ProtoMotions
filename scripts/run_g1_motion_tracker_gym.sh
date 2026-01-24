#!/bin/bash
# Run ProtoMotions demo with IsaacGym simulator
# Uses the pretrained G1-AMASS motion tracker checkpoint
#
# REQUIRES: isaacgym conda environment activated before running
#   conda activate isaacgym

set -e

# Assume this script is run from the ProtoMotions repo root

# Set library path for IsaacGym (required for libpython3.8.so)
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH"

# Default values
CHECKPOINT="${1:-data/pretrained_models/motion_tracker/g1-amass/last.ckpt}"
MOTION_FILE="${2:-data/motions/g1_random_subset_tiny.pt}"
NUM_ENVS="${3:-1}"
TARGET_UPDATE_INTERVAL="${4:-1}"

echo "Running ProtoMotions Demo (IsaacGym)"
echo "====================================="
echo "Checkpoint: $CHECKPOINT"
echo "Motion file: $MOTION_FILE"
echo "Num envs: $NUM_ENVS"
echo "Target update interval: $TARGET_UPDATE_INTERVAL (1=50Hz, 5=10Hz)"
echo ""
echo "Controls:"
echo "  ESC/Q - Quit"
echo "  R - Reset environments"
echo "  J - Push robot (test robustness)"
echo "  V - Toggle viewer"
echo ""

python protomotions/inference_agent.py \
    --checkpoint "$CHECKPOINT" \
    --motion-file "$MOTION_FILE" \
    --simulator isaacgym \
    --num-envs "$NUM_ENVS" \
    --target-update-interval "$TARGET_UPDATE_INTERVAL"
