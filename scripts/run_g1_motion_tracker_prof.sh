#!/bin/bash
# Run ProtoMotions demo with py-spy profiling
# Supports two modes:
#   top    - Live terminal view (like htop for Python)
#   record - Save flame graph SVG for detailed analysis

set -e  # Exit on error

# Assume this script is run from the ProtoMotions repo root

# Parse mode (first arg)
MODE="${1:-both}"
shift 2>/dev/null || true

# Default values - use fine-tuned IsaacLab checkpoint
CHECKPOINT="${1:-results/g1_isaaclab_finetune/last.ckpt}"
MOTION_FILE="${2:-data/motions/g1_random_subset_tiny.pt}"
NUM_ENVS="${3:-1}"
OUTPUT_FILE="${4:-profile_$(date +%Y%m%d_%H%M%S).svg}"

# Activate IsaacLab environment
source ../IsaacLab/env_isaaclab/bin/activate
source ../IsaacLab/_isaac_sim/setup_conda_env.sh

echo "Running ProtoMotions Demo with py-spy Profiling"
echo "================================================"
echo "Mode: $MODE"
echo "Checkpoint: $CHECKPOINT"
echo "Motion file: $MOTION_FILE"
echo "Num envs: $NUM_ENVS"
if [[ "$MODE" == "record" || "$MODE" == "both" ]]; then
    echo "Profile output: $OUTPUT_FILE"
fi
echo ""
echo "Controls:"
echo "  Q - Quit"
echo "  R - Reset environments"
echo "  J - Push robot (test robustness)"
echo "  L - Start/stop video recording"
echo "  O - Toggle camera view"
echo ""

# Check if py-spy is installed
if ! command -v py-spy &> /dev/null; then
    echo "py-spy not found. Installing..."
    pip install py-spy
fi

# Get the full path to python in the current environment
PYTHON_PATH=$(which python)
echo "Using Python: $PYTHON_PATH"

# Disable ptrace restrictions so py-spy can run without sudo
echo "Temporarily disabling ptrace restrictions..."
echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope > /dev/null

if [[ "$MODE" == "top" ]]; then
    # Live terminal view
    echo ""
    echo "=== LIVE PROFILING (press Ctrl+C to stop) ==="
    echo ""
    py-spy top \
        --subprocesses \
        --rate 10 \
        -- "$PYTHON_PATH" protomotions/inference_agent.py \
            --checkpoint "$CHECKPOINT" \
            --motion-file "$MOTION_FILE" \
            --simulator isaaclab \
            --num-envs "$NUM_ENVS"

elif [[ "$MODE" == "record" ]]; then
    # Record to flame graph
    py-spy record \
        --output "$OUTPUT_FILE" \
        --subprocesses \
        --rate 10 \
        -- "$PYTHON_PATH" protomotions/inference_agent.py \
            --checkpoint "$CHECKPOINT" \
            --motion-file "$MOTION_FILE" \
            --simulator isaaclab \
            --num-envs "$NUM_ENVS"
    
    echo ""
    echo "Profile saved to: $OUTPUT_FILE"
    echo "Open the SVG file in a browser to view the flame graph."

elif [[ "$MODE" == "both" ]]; then
    # Run the process, attach py-spy top, and record in background
    echo ""
    echo "Starting simulation..."
    "$PYTHON_PATH" protomotions/inference_agent.py \
        --checkpoint "$CHECKPOINT" \
        --motion-file "$MOTION_FILE" \
        --simulator isaaclab \
        --num-envs "$NUM_ENVS" &
    PID=$!
    
    sleep 3  # Wait for process to start
    
    echo "Attaching py-spy (recording to $OUTPUT_FILE)..."
    echo "=== LIVE PROFILING (close simulation window to stop) ==="
    echo ""
    
    # Record in background
    py-spy record --pid $PID --output "$OUTPUT_FILE" --subprocesses --rate 10 &
    RECORD_PID=$!
    
    # Show live top view
    py-spy top --pid $PID --subprocesses --rate 5 || true
    
    # Wait for processes to finish
    wait $PID 2>/dev/null || true
    wait $RECORD_PID 2>/dev/null || true
    
    echo ""
    echo "Profile saved to: $OUTPUT_FILE"
    echo "Open the SVG file in a browser to view the flame graph."

else
    echo "Usage: $0 [mode] [checkpoint] [motion_file] [num_envs] [output_file]"
    echo ""
    echo "Modes:"
    echo "  top    - Live terminal view (like htop for Python)"
    echo "  record - Save flame graph SVG"
    echo "  both   - Live view + save flame graph (default)"
    exit 1
fi

# Re-enable ptrace restrictions
echo "Re-enabling ptrace restrictions..."
echo 1 | sudo tee /proc/sys/kernel/yama/ptrace_scope > /dev/null
