#!/usr/bin/env bash
# Eric, 2026-08-13: "When you are ready to start training Atlas just stop
# the utahraptor on GPU1 and warm start it there."
#
# Waits for the v13 chain to log CORPUS DONE (the launch step behind it is
# fused: the warm-start ckpt was renamed aside so the chain aborts cleanly
# instead of OOM-ing into the occupied GPU). Then: restore ckpt -> stop the
# utahraptor on GPU 1 ONLY -> wait for VRAM -> warm-start atlas v13 ->
# verify epochs actually advance before declaring success.
set -u
cd /home/bizon/sparkpack/ProtoMotions
LOG=results/atlas_v13_launch.log
PY=/home/bizon/sparkpack/.venv-isaacsim5/bin/python
W=/tmp/claude-1000/-home-bizon-sparkpack/2869939e-7c3b-46e1-998b-1fb4b17b134c/scratchpad/v13tail
say() { echo "$(date '+%F %T'): $*" >> "$LOG"; }

say "waiting for CORPUS DONE from the build chain"
while ! grep -q "CORPUS DONE" results/atlas_v13_build.log; do
  if grep -qE "ABORT: (convert|foot fix|leg optimizer|arm optimizer|trim|corpus|validation)" results/atlas_v13_build.log; then
    say "STOP: chain aborted before corpus -- not launching"; exit 1
  fi
  sleep 30
done
say "corpus confirmed"

# let the fused chain hit its abort and exit before we touch anything
sleep 20
mv results/atlas_ase_pretrain_v12_tuned/last.ckpt.fused_pending_gpu \
   results/atlas_ase_pretrain_v12_tuned/last.ckpt || { say "STOP: fuse restore failed"; exit 1; }
say "warm-start checkpoint restored"

# the utahraptor on GPU 1, by device uuid -> pid (never trust grep self-matches)
GPU1_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 1)
PID=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v u="$GPU1_UUID" '$2==u {print $1}' | head -1)
if [ -n "$PID" ] && ps -o args= -p "$PID" | grep -q utahraptor; then
  say "stopping utahraptor on GPU1: pid $PID"
  kill -9 "$PID"
else
  say "note: no utahraptor found on GPU1 (pid='$PID') -- proceeding"
fi
for i in $(seq 1 60); do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
  [ "$U" -lt 3000 ] && break
  sleep 5
done
say "GPU1 at $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1)"

cp results/atlas_ase_pretrain_v12_tuned/last.ckpt $W/warmstart.ckpt
PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
setsid $PY protomotions/train_agent.py \
  --robot-name atlas --simulator isaaclab --headless \
  --motion-file data/atlas_pretrain_corpus_v13.pt \
  --experiment-path examples/experiments/ase/mlp_template_tuned.py \
  --num-envs 4096 --batch-size 8192 --training-max-steps 10000000000000 \
  --experiment-name atlas_ase_pretrain_v13 \
  --checkpoint $W/warmstart.ckpt \
  > results/atlas_ase_pretrain_v13.log 2>&1 < /dev/null &
say "launched pid $!"

# verify by training output, not by the launcher's word
for i in $(seq 1 120); do
  if grep -qE "Epoch [0-9]+, training" results/atlas_ase_pretrain_v13.log 2>/dev/null; then
    say "VERIFIED: training is stepping ($(grep -oE 'Epoch [0-9]+' results/atlas_ase_pretrain_v13.log | tail -1))"
    exit 0
  fi
  if grep -qiE "Traceback|out of memory" results/atlas_ase_pretrain_v13.log 2>/dev/null; then
    say "LAUNCH FAILED -- see results/atlas_ase_pretrain_v13.log"; exit 1
  fi
  sleep 15
done
say "TIMEOUT: no epoch after 30 min -- check manually"
