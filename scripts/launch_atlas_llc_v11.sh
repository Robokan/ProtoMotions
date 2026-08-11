#!/usr/bin/env bash
# Wait for the clip-level foot fix (fix_foot_clips.py) to finish, rebuild
# data/atlas_pretrain_corpus_v11.pt from the corrected clips so velocities
# are consistent, then start the ASE LLC pretrain on GPU 1.
#
# Rebuilding is REQUIRED, not cosmetic: motion_lib copies velocity fields
# straight from the .motion files (gvs <- rigid_body_vel, dvs <- dof_vel),
# so the corpus must be packaged AFTER the clips are corrected or it would
# carry velocities describing the old ankle angles — which the AMP
# discriminator reads.
#
# Written 2026-08-10 to run unattended overnight. Safe to re-run.
set -u
cd /home/bizon/sparkpack/ProtoMotions
LOG=results/atlas_ase_pretrain_v11_launcher.log
CLIPS=data/motions/atlas_v11
CORPUS=data/atlas_pretrain_corpus_v11.pt
PY=/home/bizon/sparkpack/.venv-isaacsim5/bin/python

say() { echo "$(date '+%F %T'): $*" >> "$LOG"; }

if pgrep -f "experiment-name atlas_ase_pretrain_v11" > /dev/null; then
    say "training already running — not launching again"; exit 0
fi

say "waiting for fix_foot_clips to finish"
while pgrep -f "fix_foot_clips" > /dev/null; do sleep 30; done

if ! grep -q "^written to" /tmp/footclips_v11.log 2>/dev/null; then
    say "ABORT: clip fix did not report a successful write"
    tail -20 /tmp/footclips_v11.log >> "$LOG" 2>&1
    exit 1
fi
say "clip fix finished: $(grep -c 'lift ' /tmp/footclips_v11.log) clips corrected"

# Rebuild the corpus FROM THE CORRECTED CLIPS (velocities now consistent).
say "rebuilding $CORPUS from $CLIPS"
$PY -m protomotions.components.motion_lib \
    --motion-path "$CLIPS" --output-file "$CORPUS" --device cpu >> "$LOG" 2>&1
if [ $? -ne 0 ]; then say "ABORT: corpus rebuild failed"; exit 1; fi

# Validate before spending a night on it.
$PY - >> "$LOG" 2>&1 <<'PY'
import sys, torch
d = torch.load("data/atlas_pretrain_corpus_v11.pt", weights_only=False,
               map_location="cpu")
n = len(d["motion_lengths"])
files = [str(x).split("/")[-1] for x in d["motion_files"]]
falls = [f for f in files if "fall" in f.lower()]
bad = [k for k in ("gts", "grs", "dps", "gvs", "gavs", "dvs")
       if not torch.isfinite(d[k]).all()]
print(f"corpus: {n} clips, {d['gts'].shape[0]} frames, "
      f"{float(sum(d['motion_lengths'])):.1f} s")
print(f"fall clips: {len(falls)}  non-finite: {bad or 'none'}")
print(f"max |dof_vel| {float(d['dvs'].abs().max()):.1f} rad/s   "
      f"max |ang_vel| {float(d['gavs'].abs().max()):.1f} rad/s")
if n != 135 or falls or bad:
    sys.exit(f"ABORT: unexpected corpus (clips={n}, falls={len(falls)}, bad={bad})")
PY
if [ $? -ne 0 ]; then say "ABORT: corpus validation failed"; exit 1; fi

say "launching ASE LLC pretrain on GPU 1"
PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
setsid $PY protomotions/train_agent.py \
  --robot-name atlas --simulator isaaclab --headless \
  --motion-file "$CORPUS" \
  --experiment-path examples/experiments/ase/mlp.py \
  --num-envs 4096 --batch-size 8192 --training-max-steps 10000000000000 \
  --experiment-name atlas_ase_pretrain_v11 \
  > results/atlas_ase_pretrain_v11.log 2>&1 < /dev/null &

say "launched pid $!"
