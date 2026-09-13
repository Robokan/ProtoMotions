#!/usr/bin/env bash
# One-shot: resume the atlas HLC battle league on GPU 0.
# Scheduled via systemd-run for 2026-07-30 00:00 (Eric, 2026-07-29).
# Safe to re-run: refuses if the league is already up.
set -u
cd /home/bizon/sparkpack/ProtoMotions

if pgrep -f "experiment-name atlas_ase_battle_hlc_v[1]" > /dev/null; then
    echo "$(date): league already running — not launching again" \
        >> results/atlas_ase_battle_hlc_v1_scheduler.log
    exit 0
fi

echo "$(date): launching league resume" \
    >> results/atlas_ase_battle_hlc_v1_scheduler.log

PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 \
setsid /home/bizon/sparkpack/.venv-isaacsim5/bin/python protomotions/train_agent.py \
  --robot-name atlas --simulator isaaclab --headless \
  --motion-file data/atlas_pretrain_corpus_v6.pt \
  --experiment-path examples/experiments/ase/battle_league_ase_hlc.py \
  --llc-checkpoint results/atlas_ase_pretrain_v6/last.ckpt \
  --num-envs 8192 --batch-size 16384 --training-max-steps 200000000000000 \
  --experiment-name atlas_ase_battle_hlc_v1 \
  > results/atlas_ase_battle_hlc_v1.log 2>&1 &

echo "$(date): launched pid $!" >> results/atlas_ase_battle_hlc_v1_scheduler.log
