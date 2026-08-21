#!/bin/bash
# Battle-league watchdog: survives Claude sessions, laptops, and weekends.
#
# Every 15 min: if the battle trainer is not running, relaunch it (RESUME
# picks up last.ckpt + frozen configs, so a relaunch is always safe).
# Every ~3 h: append a progress report to results/battle_watchdog.log so
# there is an on-disk record of the approach/strike/win trends even when
# nobody is watching.
#
# Start:  setsid nohup scripts/battle_watchdog.sh > /dev/null 2>&1 &
# Stop:   pkill -f battle_watchdog.sh
#
# NOTE (memory: scheduled-launch-rules): plain setsid nohup from a shell,
# NEVER inside a systemd-run unit (cgroup reaping kills the children).

REPO=~/sparkpack/ProtoMotions
RUN=atlas_ase_battle_hlc_v4
LOG=$REPO/results/battle_watchdog.log
PY5=~/sparkpack/.venv-isaacsim5/bin/python
PY6=~/sparkpack/.venv-isaacsim6/bin/python

cd "$REPO" || exit 1
echo "$(date '+%F %T') watchdog started (pid $$)" >> "$LOG"

LAST_REPORT=0
while true; do
    # --- keep the trainer alive -----------------------------------------
    if ! pgrep -f "experiment-name $RUN" > /dev/null; then
        echo "$(date '+%F %T') trainer not running -- relaunching (auto-resume)" >> "$LOG"
        tail -3 "results/$RUN.log" | tr '\r' '\n' | tail -2 >> "$LOG"
        PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup \
        "$PY5" protomotions/train_agent.py \
            --robot-name atlas --simulator isaaclab --headless \
            --motion-file data/atlas_pretrain_corpus_v17.pt \
            --experiment-path examples/experiments/ase/battle_league_ase_hlc.py \
            --llc-checkpoint results/atlas_v17_physx/last.ckpt \
            --num-envs 8192 --batch-size 16384 \
            --training-max-steps 10000000000000 \
            --experiment-name "$RUN" >> "results/$RUN.log" 2>&1 &
        sleep 300  # give Isaac time to boot before re-checking
    fi

    # --- 3-hourly progress report ---------------------------------------
    NOW=$(date +%s)
    if [ $((NOW - LAST_REPORT)) -ge 10800 ]; then
        LAST_REPORT=$NOW
        {
            echo "===== $(date '+%F %T') ====="
            "$PY6" scripts/check_battle_progress.py --run "$RUN" 2>/dev/null | tail -14
        } >> "$LOG"
    fi

    sleep 900
done
