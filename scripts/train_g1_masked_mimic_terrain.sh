#!/bin/bash
# Train MaskedMimic for G1 with Complex Terrain using IsaacLab
# Trains from scratch with procedural terrain (slopes, stairs, obstacles)
# Supports multi-GPU training with sharded motion library (full PHUMA dataset)
#
# Usage: ./train_masked_mimic_g1_terrain.sh [NUM_GPUS] [NUM_ENVS] [BATCH_SIZE] [HEIGHT_MULTIPLIER]
#
# Curriculum training (recommended):
#   Stage 1: ./train_masked_mimic_g1_terrain.sh 3 2048 1024 0.1   # Nearly flat (stairs 0.5-2cm)
#   Stage 2: ./train_masked_mimic_g1_terrain.sh 3 2048 1024 0.5   # Gentle terrain (stairs 2.5-11cm)
#   Stage 3: ./train_masked_mimic_g1_terrain.sh 3 2048 1024 1.0   # Full terrain (stairs 5-22cm)
#
# height_multiplier scales ALL terrain heights:
#   1.0 = normal (stairs 5-22cm, slopes 40%)
#   0.5 = half height (stairs 2.5-11cm, slopes 20%)
#   0.1 = nearly flat (stairs 0.5-2cm, slopes 4%)

set -e  # Exit on error

# Configuration - adjust paths for your setup
export PROTOMOTIONS_PATH="${PROTOMOTIONS_PATH:-/home/bizon/sparkpack/ProtoMotions}"
cd "$PROTOMOTIONS_PATH"

# Activate IsaacLab environment
source /home/bizon/sparkpack/IsaacLab/env_isaaclab/bin/activate
source /home/bizon/sparkpack/IsaacLab/_isaac_sim/setup_conda_env.sh

# Memory optimization for fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Suppress verbose Isaac Sim/Omniverse output  
export OMNI_LOG_LEVEL=warning              # Only show warnings and errors (not extension loading)
export CARB_LOG_LEVEL=warning              # Carbonite logging  
export PYTHONWARNINGS="ignore::DeprecationWarning"  # Suppress deprecation warnings
export OMNI_KIT_ACCEPT_EULA=YES            # Skip EULA prompt
export LOGLEVEL=WARNING                    # Python logging (suppresses DEBUG:AutoNode)

# NCCL stability for multi-GPU training (prevents watchdog timeout on checkpoint resume)
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800      # 30 min instead of 8 min default
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1          # Better error handling
export TORCH_NCCL_BLOCKING_WAIT=1                 # Avoid async race conditions

# Configuration - optimized for 3x RTX 4090 with full PHUMA dataset (~75K motions)
EXPERIMENT_NAME="masked_mimic_g1_phuma_terrain"
NUM_GPUS="${1:-3}"       # 3 GPUs for sharded training
NUM_ENVS="${2:-2048}"    # Per GPU (~18GB each)
BATCH_SIZE="${3:-1024}"  # Per GPU batch size

# Terrain height multiplier - scales all terrain heights for curriculum learning
# Values: 1.0 (normal), 0.5 (half height), 0.25 (gentle default), 0.1 (nearly flat)
# At 0.1: stairs are 0.5-2cm instead of 5-22cm
HEIGHT_MULTIPLIER="${4:-0.25}"  # Default: gentle terrain

# Motion file sharding for multi-GPU
# Use "slurmrank" placeholder - ProtoMotions replaces it with GPU rank (0, 1, 2)
# So g1_phuma_slurmrank.pt becomes g1_phuma_0.pt, g1_phuma_1.pt, g1_phuma_2.pt
# Created with: python scripts/split_motion_lib_for_multigpu.py data/motions/g1_phuma_full.pt data/motions/g1_phuma_slurmrank.pt --num-shards 3
MOTION_FILE="data/motions/g1_phuma_slurmrank.pt"

echo "============================================"
echo "Training MaskedMimic G1 with Complex Terrain"
echo "============================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "Motion file: $MOTION_FILE"
echo "  (Each GPU loads: g1_phuma_0.pt, g1_phuma_1.pt, g1_phuma_2.pt)"
echo "Num GPUs: $NUM_GPUS"
echo "Num envs per GPU: $NUM_ENVS"
echo "Batch size: $BATCH_SIZE"
echo ""
CHECKPOINT_PATH="results/$EXPERIMENT_NAME/last.ckpt"
if [ -f "$CHECKPOINT_PATH" ]; then
    echo "Terrain height_multiplier: (from checkpoint)"
else
    echo "Terrain height_multiplier: $HEIGHT_MULTIPLIER"
    if (( $(echo "$HEIGHT_MULTIPLIER < 0.2" | bc -l) )); then
        echo "  (Nearly flat - curriculum stage 1: stairs 0.5-2cm)"
    elif (( $(echo "$HEIGHT_MULTIPLIER < 0.6" | bc -l) )); then
        echo "  (Gentle terrain - curriculum stage 2: stairs 2.5-11cm)"
    else
        echo "  (Full terrain - curriculum stage 3: stairs 5-22cm)"
    fi
fi
echo ""
echo "Terrain types (when vertical_scale=0.005):"
echo "  - 20% smooth slopes (up to 40% grade)"
echo "  - 10% rough slopes"  
echo "  - 10% stairs up (5-22cm steps)"
echo "  - 10% stairs down"
echo "  - 5% discrete obstacles"
echo "  - 45% flat ground"
echo ""
echo "Training will run until ~3000+ epochs (5B max steps)"
echo "============================================"
echo ""

# Expert model path - pretrained motion tracker provides teacher actions
# MaskedMimic learns to imitate the expert while handling sparse inputs
EXPERT_MODEL="results/g1_isaaclab_finetune_phuma/last.ckpt"

echo "Expert model: $EXPERT_MODEL"
echo ""

# Check if shards exist
if [ ! -f "data/motions/g1_phuma_0.pt" ] || [ ! -f "data/motions/g1_phuma_1.pt" ] || [ ! -f "data/motions/g1_phuma_2.pt" ]; then
    echo "ERROR: Motion shards not found!"
    echo "Create them with:"
    echo "  python scripts/split_motion_lib_for_multigpu.py data/motions/g1_phuma_full.pt data/motions/g1_phuma_slurmrank.pt --num-shards 3"
    exit 1
fi
echo "Motion shards found: g1_phuma_0.pt, g1_phuma_1.pt, g1_phuma_2.pt"

# Run training with complex terrain experiment
# Key change: --experiment-path points to transformer_complex_terrain.py
# This uses ComplexTerrainConfig instead of flat TerrainConfig
# ProtoMotions auto-resumes from results/$EXPERIMENT_NAME/last.ckpt if it exists
OVERRIDES=(
    "agent.expert_model_path=$EXPERT_MODEL"
    "agent.evaluator.robot_state_metrics_device=cpu"
    "agent.training_max_steps=5000000000"
)
if [ ! -f "$CHECKPOINT_PATH" ]; then
    OVERRIDES+=("terrain.height_multiplier=$HEIGHT_MULTIPLIER")
fi

python protomotions/train_agent.py \
    --robot-name g1 \
    --simulator isaaclab \
    --experiment-path examples/experiments/masked_mimic/transformer_complex_terrain.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --motion-file "$MOTION_FILE" \
    --num-envs "$NUM_ENVS" \
    --batch-size "$BATCH_SIZE" \
    --ngpu "$NUM_GPUS" \
    --headless \
    --training-max-steps 5000000000 \
    --overrides "${OVERRIDES[@]}"

echo ""
echo "============================================"
echo "Training complete!"
echo "Checkpoint saved to: results/$EXPERIMENT_NAME/last.ckpt"
echo "============================================"
