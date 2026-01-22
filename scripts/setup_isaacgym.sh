#!/bin/bash
# Setup script for IsaacGym environment
# Usage: source scripts/setup_isaacgym.sh
#
# This script must be SOURCED (not executed) to work properly:
#   source scripts/setup_isaacgym.sh
#   OR
#   . scripts/setup_isaacgym.sh

# Check if script is being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced, not executed."
    echo "Usage: source scripts/setup_isaacgym.sh"
    exit 1
fi

# Configuration - adjust these paths if needed
CONDA_PATH="/home/bizon/anaconda3"
ISAACGYM_ENV="isaacgym"  # Name of your IsaacGym conda environment
PROTOMOTIONS_DIR="/home/bizon/sparkpack/ProtoMotions"
ISAACGYM_PATH="/home/bizon/eric/isaacgym/python"

echo "Setting up IsaacGym environment..."

# Deactivate any existing virtual environments
if [[ -n "${VIRTUAL_ENV}" ]]; then
    echo "Deactivating virtual environment: ${VIRTUAL_ENV}"
    deactivate 2>/dev/null || true
fi

# Initialize conda
if [[ -f "${CONDA_PATH}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
else
    echo "ERROR: Conda not found at ${CONDA_PATH}"
    return 1
fi

# Deactivate any conda environments first
conda deactivate 2>/dev/null || true

# Activate IsaacGym environment
echo "Activating conda environment: ${ISAACGYM_ENV}"
CONDA_NO_PLUGINS=true conda activate "${ISAACGYM_ENV}" 2>/dev/null
if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to activate conda environment '${ISAACGYM_ENV}'"
    echo "Available environments:"
    ls "${CONDA_PATH}/envs/" 2>/dev/null
    return 1
fi

# Set LD_LIBRARY_PATH for IsaacGym (including bindings directory)
ISAACGYM_BINDINGS="${ISAACGYM_PATH}/isaacgym/_bindings/linux-x86_64"
export LD_LIBRARY_PATH="${ISAACGYM_BINDINGS}:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
echo "Set LD_LIBRARY_PATH to include IsaacGym bindings"

# Add IsaacGym and ProtoMotions to PYTHONPATH
export PYTHONPATH="${ISAACGYM_PATH}:${PROTOMOTIONS_DIR}:${PYTHONPATH}"
echo "Added IsaacGym to PYTHONPATH: ${ISAACGYM_PATH}"

# Navigate to ProtoMotions directory
cd "${PROTOMOTIONS_DIR}"
echo "Changed to: ${PROTOMOTIONS_DIR}"

# Verify IsaacGym is available
python -c "import isaacgym; print('IsaacGym loaded successfully')" 2>/dev/null
if [[ $? -ne 0 ]]; then
    echo "WARNING: IsaacGym module not found. Make sure it's installed in this environment."
fi

echo ""
echo "=== IsaacGym Environment Ready ==="
echo "Python: $(which python)"
echo "Conda env: ${CONDA_DEFAULT_ENV}"
echo ""
echo "Example commands:"
echo "  python examples/motion_libs_visualizer.py --motion_files data/motions/adam_lite_test.pt --robot adam_lite --simulator isaacgym"
echo "  python protomotions/train_agent.py --robot-name adam_lite --simulator isaacgym --experiment-path examples/experiments/mimic/mlp.py --motion-file data/motions/adam_lite_test.pt"
echo ""
