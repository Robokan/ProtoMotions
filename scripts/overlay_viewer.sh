#!/bin/bash
# Skinned-overlay validation viewer (SKINNED_OVERLAY_PLAN). Always launches
# from the repo root with the right env — immune to shell cwd drift.
set -e
cd /home/bizon/sparkpack/ProtoMotions
source /home/bizon/sparkpack/.venv-isaacsim5/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export DISPLAY=${DISPLAY:-:1} XAUTHORITY=${XAUTHORITY:-/home/bizon/.Xauthority}
exec python examples/motion_libs_visualizer.py \
  --motion_files data/soma_combat_viewer.pt \
  --robot soma23 --simulator isaaclab \
  --overlay-character /home/bizon/sparkpack/ProtoMotions/protomotions/data/assets/overlay/construction_worker.usd \
  "$@"
