#!/bin/bash
# Run ProtoMotions demo for SMPL humanoid with IsaacLab
# Uses the SMPL motion tracker for IsaacLab
#
# This script demonstrates motion tracking for the SMPL humanoid model,
# which is a digital human model (as opposed to robot models like G1/H1).
#
# Prerequisites:
# 1. Create SMPL motion library (see below)
# 2. Train or download a pretrained SMPL checkpoint
#
# To create the motion library:
#   python scripts/create_smpl_motion_lib.py --output data/motions/smpl_demo_tiny.pt
#
# To train a SMPL model:
#   ./scripts/train_smpl.sh
#
# Or download pretrained models from ProtoMotions releases.

set -e  # Exit on error

# Assume this script is run from the ProtoMotions repo root

# Activate IsaacLab environment
source ../IsaacLab/env_isaaclab/bin/activate
source ../IsaacLab/_isaac_sim/setup_conda_env.sh

# Default values
# For checkpoint: use pretrained SMPL model
# Options:
#   - data/pretrained_models/motion_tracker/smpl/last.ckpt (pretrained)
#   - results/smpl_mimic_train/last.ckpt (trained with train_smpl.sh)
CHECKPOINT="${1:-data/pretrained_models/motion_tracker/smpl/last.ckpt}"

# Motion file: SMPL motion library
# Option 1: Use the sample motion (single motion)
# Option 2: Create full motion lib with AMASS data
MOTION_FILE="${2:-examples/data/smpl_humanoid_sit_armchair.motion}"

NUM_ENVS="${3:-1}"
TARGET_UPDATE_INTERVAL="${4:-1}"  # 1=50Hz, 5=10Hz, 10=5Hz
SIMULATOR="${5:-isaaclab}"

echo "Running ProtoMotions SMPL Demo"
echo "=============================="
echo "Checkpoint: $CHECKPOINT"
echo "Motion file: $MOTION_FILE"
echo "Num envs: $NUM_ENVS"
echo "Target update interval: $TARGET_UPDATE_INTERVAL (1=50Hz, 5=10Hz)"
echo "Simulator: $SIMULATOR"
echo ""
echo "Controls:"
echo "  Q - Quit"
echo "  R - Reset environments"
echo "  J - Push humanoid (test robustness)"
echo "  L - Start/stop video recording"
echo "  O - Toggle camera view"
echo ""

# Check if motion file exists
if [ ! -f "$MOTION_FILE" ]; then
    echo "Warning: Motion file not found: $MOTION_FILE"
    echo ""
    echo "Available options:"
    echo "  1. Use sample motion: examples/data/smpl_humanoid_sit_armchair.motion"
    echo "  2. Download AMASS and convert with:"
    echo "     python data/scripts/convert_amass_to_motionlib.py <amass_dir> <output_dir>"
    exit 1
fi

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Warning: Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "To train a model, run:"
    echo "  ./scripts/train_smpl.sh"
    echo ""
    echo "Or download pretrained models from ProtoMotions."
    exit 1
fi

python protomotions/inference_agent.py \
    --checkpoint "$CHECKPOINT" \
    --motion-file "$MOTION_FILE" \
    --simulator "$SIMULATOR" \
    --num-envs "$NUM_ENVS" \
    --target-update-interval "$TARGET_UPDATE_INTERVAL"
