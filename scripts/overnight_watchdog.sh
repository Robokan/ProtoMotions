#!/usr/bin/env bash
# Overnight trainer watchdog (2026-07-30): if a trainer dies, resume it
# (same-name relaunch continues from last.ckpt). Max 3 restarts each.
# Emits one line per event (consumed by the session monitor).
cd /home/bizon/sparkpack/ProtoMotions

PY=/home/bizon/sparkpack/.venv-isaacsim5/bin/python
declare -A restarts

launch_league() {
    setsid env PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 \
      $PY protomotions/train_agent.py \
      --robot-name atlas --simulator isaaclab --headless \
      --motion-file data/atlas_pretrain_corpus_v6.pt \
      --experiment-path examples/experiments/ase/battle_league_ase_hlc.py \
      --llc-checkpoint results/atlas_ase_pretrain_v6/last.ckpt \
      --num-envs 8192 --batch-size 16384 --training-max-steps 200000000000000 \
      --experiment-name atlas_ase_battle_hlc_v1 \
      >> results/atlas_ase_battle_hlc_v1_resume.log 2>&1 &
}

launch_pretrain() {
    setsid env PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=1 \
      $PY protomotions/train_agent.py \
      --robot-name atlas --simulator isaaclab --headless \
      --motion-file data/atlas_pretrain_corpus_v6.pt \
      --experiment-path examples/experiments/ase/mlp.py \
      --num-envs 4096 --batch-size 8192 --training-max-steps 10000000000000 \
      --experiment-name atlas_ase_pretrain_v6 \
      >> results/atlas_ase_pretrain_v6_resume.log 2>&1 &
}

alive() { pgrep -f "experiment-name $1\$" > /dev/null || pgrep -f "experiment-name $1 " > /dev/null; }

while true; do
    for run in atlas_ase_battle_hlc_v1 atlas_ase_pretrain_v6; do
        if ! alive "$run"; then
            n=${restarts[$run]:-0}
            if [ "$n" -ge 3 ]; then
                echo "$(date +%H:%M) $run DEAD and restart limit reached (3) — leaving down"
                sleep 3600
                continue
            fi
            echo "$(date +%H:%M) $run died — waiting 120s then resuming (restart $((n+1))/3)"
            sleep 120
            # stagger: never boot while the sibling booted <5 min ago
            if [ -n "$LAST_BOOT" ] && [ $(( $(date +%s) - LAST_BOOT )) -lt 300 ]; then
                sleep 300
            fi
            if [ "$run" = "atlas_ase_battle_hlc_v1" ]; then launch_league; else launch_pretrain; fi
            LAST_BOOT=$(date +%s)
            restarts[$run]=$((n+1))
            echo "$(date +%H:%M) $run relaunched"
        fi
    done
    sleep 60
done
