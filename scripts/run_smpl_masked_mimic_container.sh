#!/bin/bash
# Run MaskedMimic SMPL demo in container
# Works from either ProtoMotions or originals/ProtoMotions
set -e

# Detect which ProtoMotions directory we're in
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTOMOTIONS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Using ProtoMotions at: $PROTOMOTIONS_DIR"

# Set PYTHONPATH to use this ProtoMotions
ISAACLAB_PATH="$PROTOMOTIONS_DIR/../IsaacLab"
export PYTHONPATH="${PROTOMOTIONS_DIR}:${ISAACLAB_PATH}/source:${ISAACLAB_PATH}/source/isaaclab:${ISAACLAB_PATH}/source/isaaclab_tasks:${ISAACLAB_PATH}/source/isaaclab_rl:${ISAACLAB_PATH}/source/isaaclab_mimic"

# Disable torch dynamo (Triton not available on ARM/aarch64)
export TORCHDYNAMO_DISABLE=1

cd "$PROTOMOTIONS_DIR"

echo ""
echo "Running MaskedMimic SMPL Demo"
echo "============================="
echo "Checkpoint: data/pretrained_models/masked_mimic/smpl/last.ckpt"
echo "Using sample motion: examples/data/smpl_humanoid_sit_armchair.motion"
echo ""
echo "Controls:"
echo "  Q - Quit"
echo "  R - Reset environments"
echo "  J - Push robot (test robustness)"
echo "  O - Toggle camera view"
echo "  M - Toggle markers on/off"
echo ""
echo "Options (pass as arguments):"
echo "  --sequential-targets     Step through motion sequentially (like GR00T)"
echo "  --target-hz 5.0          Target update rate in Hz (default: 5 Hz)"
echo "  --target-lookahead 1.0   How far ahead in motion to place targets (default: 1.0s)"
echo ""
echo "Example: --sequential-targets --target-hz 5 --target-lookahead 0.5"
echo "  (update 5x/sec, each target 0.5s ahead in motion)"
echo ""

# Run inference with sample SMPL motion
/isaac-sim/python.sh protomotions/inference_agent.py \
    --checkpoint data/pretrained_models/masked_mimic/smpl/last.ckpt \
    --motion-file examples/data/smpl_humanoid_sit_armchair.motion \
    --simulator isaaclab \
    --num-envs 1 \
    "$@"
