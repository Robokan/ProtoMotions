#!/bin/bash
# Train MaskedMimic for G1 using IsaacLab
# Starts from the fine-tuned motion tracker checkpoint
# Supports multi-GPU training with sharded motion library (full PHUMA dataset)

set -e  # Exit on error

# Assume this script is run from the ProtoMotions repo root

# Activate IsaacLab environment
source ../IsaacLab/env_isaaclab/bin/activate
source ../IsaacLab/_isaac_sim/setup_conda_env.sh

# Memory optimization for fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL stability for multi-GPU training (prevents watchdog timeout on checkpoint resume)
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800      # 30 min instead of 8 min default
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1          # Better error handling
export TORCH_NCCL_BLOCKING_WAIT=1                 # Avoid async race conditions

# Configuration - optimized for 3x RTX 4090 with full PHUMA dataset (~75K motions)
EXPERIMENT_NAME="masked_mimic_g1_phuma"
NUM_GPUS="${1:-3}"       # 3 GPUs for sharded training
NUM_ENVS="${2:-2048}"    # Per GPU (~18GB each)
BATCH_SIZE="${3:-1024}"  # Per GPU batch size

# Motion file sharding for multi-GPU
# Use "slurmrank" placeholder - ProtoMotions replaces it with GPU rank (0, 1, 2)
# So g1_phuma_slurmrank.pt becomes g1_phuma_0.pt, g1_phuma_1.pt, g1_phuma_2.pt
# Created with: python scripts/split_motion_lib_for_multigpu.py data/motions/g1_phuma_full.pt data/motions/g1_phuma_slurmrank.pt --num-shards 3
MOTION_FILE="data/motions/g1_phuma_slurmrank.pt"

echo "============================================"
echo "Training MaskedMimic G1 for IsaacLab"
echo "============================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "Motion file: $MOTION_FILE"
echo "  (Each GPU loads: g1_phuma_0.pt, g1_phuma_1.pt, g1_phuma_2.pt)"
echo "Num GPUs: $NUM_GPUS"
echo "Num envs per GPU: $NUM_ENVS"
echo "Batch size: $BATCH_SIZE"
echo ""
echo "Training with expert distillation from fine-tuned motion tracker"
echo "This will take ~18-24 hours on 3x RTX 4090s (~3000 epochs)"
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

# Run training
# MaskedMimic uses teacher-student training with a pretrained expert
# The motion file will be automatically sharded: each GPU loads g1_phuma_$LOCAL_RANK.pt
python protomotions/train_agent.py \
    --robot-name g1 \
    --simulator isaaclab \
    --experiment-path examples/experiments/masked_mimic/transformer.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --motion-file "$MOTION_FILE" \
    --num-envs "$NUM_ENVS" \
    --batch-size "$BATCH_SIZE" \
    --ngpu "$NUM_GPUS" \
    --headless \
    --training-max-steps 18500000 \
    --overrides "agent.expert_model_path=$EXPERT_MODEL" \
                "agent.evaluator.robot_state_metrics_device=cpu"

echo ""
echo "============================================"
echo "Training complete!"
echo "Checkpoint saved to: results/$EXPERIMENT_NAME/last.ckpt"
echo "============================================"
