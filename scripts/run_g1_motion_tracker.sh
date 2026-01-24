#!/bin/bash
# Run ProtoMotions demo with IsaacLab
# Uses the fine-tuned G1 motion tracker checkpoint for IsaacLab
#
# Usage:
#   ./run_g1_motion_tracker.sh                                    # Default demo
#   ./run_g1_motion_tracker.sh --scene /path/to/warehouse.usd    # With custom scene
#   ./run_g1_motion_tracker.sh --checkpoint results/my_model/last.ckpt

set -e  # Exit on error

# Assume this script is run from the ProtoMotions repo root

# Activate IsaacLab environment
source ../IsaacLab/env_isaaclab/bin/activate
source ../IsaacLab/_isaac_sim/setup_conda_env.sh

# Default values
CHECKPOINT="results/g1_isaaclab_finetune_phuma/last.ckpt"
#MOTION_FILE="data/motions/g1_phuma_full.pt"
MOTION_FILE="data/motions/g1_random_subset_tiny.pt"
NUM_ENVS="1"
TARGET_UPDATE_INTERVAL="1"  # 1=50Hz, 5=10Hz, 10=5Hz
SCENE_USD="omniverse://localhost/Library/assets/IsaacLab/SparkPack/warehouse.usd"
SCENE_OFFSET="0 0 0"

# Randomization options
RANDOMIZE_LIGHTS=""
RANDOMIZE_PER_FRAME=""
# Objects to randomize (translate/rotate) - these are floor items safe to move
RANDOMIZE_OBJECTS="pallets warehousepile drumtruck"
OBJECT_POS_RANGE="-0.5 0.5"
OBJECT_ROT_RANGE="-30 30"
HIDE_OBJECTS_PROB="0.0"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint|-c)
            CHECKPOINT="$2"
            shift 2
            ;;
        --motion-file|-m)
            MOTION_FILE="$2"
            shift 2
            ;;
        --num-envs|-n)
            NUM_ENVS="$2"
            shift 2
            ;;
        --target-update-interval|-t)
            TARGET_UPDATE_INTERVAL="$2"
            shift 2
            ;;
        --scene|-s)
            SCENE_USD="$2"
            shift 2
            ;;
        --scene-offset)
            SCENE_OFFSET="$2"
            shift 2
            ;;
        --randomize-lights)
            RANDOMIZE_LIGHTS="true"
            shift
            ;;
        --randomize-per-frame)
            RANDOMIZE_PER_FRAME="true"
            shift
            ;;
        --randomize-objects)
            RANDOMIZE_OBJECTS="$2"
            shift 2
            ;;
        --object-pos-range)
            OBJECT_POS_RANGE="$2"
            shift 2
            ;;
        --object-rot-range)
            OBJECT_ROT_RANGE="$2"
            shift 2
            ;;
        --hide-objects-prob)
            HIDE_OBJECTS_PROB="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --checkpoint, -c PATH      Path to model checkpoint"
            echo "  --motion-file, -m PATH     Path to motion library"
            echo "  --num-envs, -n NUM         Number of parallel environments"
            echo "  --target-update-interval   1=50Hz, 5=10Hz, 10=5Hz"
            echo "  --scene, -s PATH           Path to USD scene file (warehouse, etc.)"
            echo "  --scene-offset X,Y,Z       Scene position offset (e.g., '0,0,0')"
            echo ""
            echo "Randomization:"
            echo "  --randomize-lights         Randomize existing scene lights"
            echo "  --randomize-per-frame      Randomize every frame (vs per-episode)"
            echo "  --randomize-objects 'a b'  Object names to randomize (e.g., 'forklift pallet')"
            echo "  --object-pos-range 'X Y'   Position range (default: '-0.5 0.5')"
            echo "  --object-rot-range 'X Y'   Rotation range degrees (default: '-30 30')"
            echo "  --hide-objects-prob P      Probability to hide objects (0-1)"
            echo ""
            echo "  --help, -h                 Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Running ProtoMotions Demo"
echo "========================="
echo "Checkpoint: $CHECKPOINT"
echo "Motion file: $MOTION_FILE"
echo "Num envs: $NUM_ENVS"
echo "Target update interval: $TARGET_UPDATE_INTERVAL (1=50Hz, 5=10Hz)"
if [ -n "$SCENE_USD" ]; then
    echo "Scene USD: $SCENE_USD"
    echo "Scene offset: $SCENE_OFFSET"
fi
if [ -n "$RANDOMIZE_LIGHTS" ] || [ -n "$RANDOMIZE_OBJECTS" ]; then
    echo "Randomization: ENABLED"
    if [ -n "$RANDOMIZE_LIGHTS" ]; then
        echo "  Lights: randomizing existing scene lights"
    fi
    if [ -n "$RANDOMIZE_OBJECTS" ]; then
        echo "  Objects: $RANDOMIZE_OBJECTS (pos: $OBJECT_POS_RANGE, rot: $OBJECT_ROT_RANGE)"
    fi
    if [ -n "$RANDOMIZE_PER_FRAME" ]; then
        echo "  Mode: per-frame"
    else
        echo "  Mode: per-episode"
    fi
fi
echo ""
echo "Controls:"
echo "  Q - Quit"
echo "  R - Reset environments"
echo "  J - Push robot (test robustness)"
echo "  L - Start/stop video recording (Isaac Sim MovieCapture)"
echo "  O - Toggle camera view"
echo "  M - Toggle markers on/off"
echo ""

# Build command
CMD="python protomotions/inference_agent.py \
    --checkpoint $CHECKPOINT \
    --motion-file $MOTION_FILE \
    --simulator isaaclab \
    --num-envs $NUM_ENVS \
    --target-update-interval $TARGET_UPDATE_INTERVAL"

# Add scene if specified
if [ -n "$SCENE_USD" ]; then
    CMD="$CMD --scene-usd $SCENE_USD --scene-offset '$SCENE_OFFSET'"
fi

# Add randomization options
if [ -n "$RANDOMIZE_LIGHTS" ] || [ -n "$RANDOMIZE_OBJECTS" ]; then
    if [ -n "$RANDOMIZE_LIGHTS" ]; then
        CMD="$CMD --randomize-lights"
    fi
    CMD="$CMD --object-pos-range '$OBJECT_POS_RANGE'"
    CMD="$CMD --object-rot-range '$OBJECT_ROT_RANGE'"
    CMD="$CMD --hide-objects-prob $HIDE_OBJECTS_PROB"
    
    if [ -n "$RANDOMIZE_PER_FRAME" ]; then
        CMD="$CMD --randomize-per-frame"
    fi
    if [ -n "$RANDOMIZE_OBJECTS" ]; then
        CMD="$CMD --randomize-objects $RANDOMIZE_OBJECTS"
    fi
fi

eval $CMD
