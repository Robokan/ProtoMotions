#!/usr/bin/env bash
# Convert GMR T800 npz -> .motion, pack MotionLib, quality-scan.
# Run inside the battle container from the ProtoMotions repo root.
set -euo pipefail
cd /workspace/sparkpack/ProtoMotions

NPZ_DIR=data/motions/gmr_t800_npz
MOT_DIR=data/motions/gmr_t800
LIB=data/t800_tracker_stage1.pt
QUALITY=data/t800_stage1_quality

echo "[1/3] convert npz -> motion"
python data/scripts/convert_gmr_pkl_to_t800.py \
  --input-dir "$NPZ_DIR" --output-dir "$MOT_DIR"

echo "[2/3] pack MotionLib"
python -m protomotions.components.motion_lib \
  --motion-path "$MOT_DIR" \
  --output-file "$LIB" \
  --device cpu

echo "[3/3] quality scan"
python data/scripts/scan_motion_lib_quality.py \
  --lib "$LIB" \
  --mjcf protomotions/data/assets/mjcf/t800.xml \
  --out "$QUALITY"

echo "done. bad list: ${QUALITY}_bad.txt"
