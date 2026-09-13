#!/usr/bin/env bash
# When v14 (passing-space corpus) validates: stop the v13 trainer on GPU 1,
# warm-start atlas_ase_pretrain_v14 from its freshest checkpoint. Same
# discriminator, same tuned trainer -- the corpus is the only variable
# (Eric 2026-08-14: AMP's disc is fine as designed; the data was the fault).
set -u
cd /home/bizon/sparkpack/ProtoMotions
LOG=results/atlas_v14_launch.log
PY=/home/bizon/sparkpack/.venv-isaacsim5/bin/python
say() { echo "$(date '+%F %T'): $*" >> "$LOG"; }

say "waiting for V14 DONE"
while ! grep -q "V14 DONE" results/atlas_v14_build.log 2>/dev/null; do
  grep -q "ABORT" results/atlas_v14_build.log 2>/dev/null && { say "STOP: v14 build aborted"; exit 1; }
  sleep 60
done
say "v14 corpus validated"

GPU1_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 1)
PID=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v u="$GPU1_UUID" '$2==u {print $1}' | head -1)
if [ -n "$PID" ] && ps -o args= -p "$PID" | grep -q atlas_ase_pretrain_v13; then
  say "stopping v13 trainer pid $PID"
  kill -9 "$PID"
fi
for i in $(seq 1 60); do
  [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)" -lt 3000 ] && break
  sleep 5
done
say "GPU1 free"

PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
setsid $PY protomotions/train_agent.py \
  --robot-name atlas --simulator isaaclab --headless \
  --motion-file data/atlas_pretrain_corpus_v14.pt \
  --experiment-path examples/experiments/ase/mlp_template_tuned.py \
  --num-envs 4096 --batch-size 8192 --training-max-steps 10000000000000 \
  --experiment-name atlas_ase_pretrain_v14 \
  --checkpoint results/atlas_ase_pretrain_v13/last.ckpt \
  > results/atlas_ase_pretrain_v14.log 2>&1 < /dev/null &
say "launched pid $!"
for i in $(seq 1 120); do
  grep -qE "Epoch [0-9]+, training" results/atlas_ase_pretrain_v14.log 2>/dev/null && { say "VERIFIED stepping"; exit 0; }
  grep -qiE "Traceback|out of memory" results/atlas_ase_pretrain_v14.log 2>/dev/null && { say "LAUNCH FAILED"; exit 1; }
  sleep 15
done
say "TIMEOUT waiting for first epoch"
