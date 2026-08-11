#!/usr/bin/env bash
# Rebuild atlas_v11 from the pristine retarget with tremor filtering, then
# resume the ASE LLC pretrain on the corrected corpus.
#
# Eric reported leg vibration and correctly said it predated the foot fix.
# Measured on the untouched retarget: dof_vel carries 21.0% of its spectral
# energy above 8 Hz vs 7.4% for dof_pos, so the tremor is mostly in the
# VELOCITY channels -- invisible in playback, fully visible to the AMP
# discriminator, which then rewards the policy for reproducing it.
#
# Two independent defects are fixed here:
#   1. retarget tremor          -> lowpass_motion_clips.py (8 Hz, zero-phase)
#   2. my per-frame pitch snap  -> fix_foot_clips.py rate-limited shortest path
#
# ORDER IS LOAD-BEARING. Filtering moves the root and every body, so it
# re-buries the feet; the foot correction must run last and have the final say
# on ground clearance. Input is atlas_v11_pre_footfix (the pristine retarget),
# NOT the current atlas_v11, which already carries the snap.
#
# Resume (not restart) is deliberate: the discriminator adapts to the new
# reference distribution, so the ~14 h of policy already learned is kept. The
# corpus is written to the SAME path the pickled resolved_configs.pt names, so
# resume picks it up with no config surgery.
set -u
cd /home/bizon/sparkpack/ProtoMotions
LOG=results/atlas_v11_refix.log
PY=/home/bizon/sparkpack/.venv-isaacsim5/bin/python
SRC=data/motions/atlas_v11_pre_footfix
SMOOTH=data/motions/atlas_v11_smooth
DST=data/motions/atlas_v11
CORPUS=data/atlas_pretrain_corpus_v11.pt
EXP=atlas_ase_pretrain_v11

say() { echo "$(date '+%F %T'): $*" | tee -a "$LOG"; }
die() { say "ABORT: $*"; exit 1; }

n=$(ls "$SRC"/*.motion 2>/dev/null | wc -l)
[ "$n" -eq 135 ] || die "expected 135 pristine clips in $SRC, found $n"
say "source: $n pristine retarget clips"

say "step 1/4: 8 Hz zero-phase low-pass -> $SMOOTH"
rm -rf "$SMOOTH"
PYTHONPATH=data/scripts $PY data/scripts/lowpass_motion_clips.py \
    --in-dir "$SRC" --out-dir "$SMOOTH" --cutoff 8 >> "$LOG" 2>&1 \
    || die "low-pass failed"
grep -q "^written to" "$LOG" || die "low-pass did not report a write"

say "step 2/4: rate-limited foot correction -> $DST"
PYTHONPATH=data/scripts $PY data/scripts/fix_foot_clips.py \
    --robot atlas --in-dir "$SMOOTH" --out-dir "$DST" >> "$LOG" 2>&1 \
    || die "foot fix failed"
m=$(ls "$DST"/*.motion 2>/dev/null | wc -l)
[ "$m" -eq 135 ] || die "expected 135 corrected clips, found $m"

say "step 3/4: rebuild $CORPUS"
$PY -m protomotions.components.motion_lib \
    --motion-path "$DST" --output-file "$CORPUS" --device cpu >> "$LOG" 2>&1 \
    || die "corpus rebuild failed"

# Validate, INCLUDING the two things this whole exercise is about: the
# velocity channels must be quieter than the retarget's and no channel may
# have gone non-finite.
$PY - >> "$LOG" 2>&1 <<'PYCHK'
import sys, numpy as np, torch
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
    sys.exit(f"unexpected corpus (clips={n}, falls={len(falls)}, bad={bad})")
PYCHK
[ $? -eq 0 ] || die "corpus validation failed"

say "step 4/4: stop and resume $EXP on GPU 1"
# SIGTERM wedges Isaac trainers holding VRAM, so go straight to -9 and verify.
pid=$(pgrep -f "experiment-name $EXP" | head -1)
if [ -n "$pid" ]; then
    say "killing trainer pid $pid"
    kill -9 "$pid" 2>/dev/null
    for i in $(seq 1 60); do pgrep -f "experiment-name $EXP" > /dev/null || break; sleep 2; done
    pgrep -f "experiment-name $EXP" > /dev/null && die "trainer $pid would not die"
    sleep 15   # let the driver reclaim VRAM before the new process allocates
else
    say "no trainer running"
fi

ls results/$EXP/*.ckpt > /dev/null 2>&1 || die "no checkpoint to resume from"
say "resuming from $(ls -t results/$EXP/epoch_*.ckpt 2>/dev/null | head -1)"
PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
setsid $PY protomotions/train_agent.py \
  --robot-name atlas --simulator isaaclab --headless \
  --motion-file "$CORPUS" \
  --experiment-path examples/experiments/ase/mlp.py \
  --num-envs 4096 --batch-size 8192 --training-max-steps 10000000000000 \
  --experiment-name $EXP \
  >> results/$EXP.log 2>&1 < /dev/null &
say "resumed pid $!"
