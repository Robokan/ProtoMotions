#!/usr/bin/env bash
# Rebuild atlas_v11 from the pristine retarget with the foot correction only,
# then resume the ASE LLC pretrain on the corrected corpus.
#
# Fixes exactly ONE defect: the per-frame ankle-pitch snap that fix_foot_clips
# used to introduce (110 deg inside a single frame, 3300 deg/s), now a
# rate-limited shortest path over the delta grid.
#
# NO OUTPUT LOW-PASS. An earlier version of this script also ran
# lowpass_motion_clips.py, which was wrong twice over: Eric vetoed filtering
# GMR output on 2026-07-24, and the upstream filter is ALREADY APPLIED
# (lowpass_bvh.py / convert_manny_npy_to_soma --lowpass-hz 8, landed for v9 and
# inherited by v10/v11 -- v11 source clips measure 8.61% of dof_pos energy
# above 8 Hz vs 8.55% for the known-filtered f8 clips). Filtering again would
# have been a second pass over already-filtered data. The residual jitter Eric
# sees is long-standing and belongs upstream in BVH emission, not here.
#
# Input is atlas_v11_pre_footfix (the pristine retarget), NOT the current
# atlas_v11, which is a MIX of old-snap and double-filtered clips from the
# aborted run.
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
DST=data/motions/atlas_v11
CORPUS=data/atlas_pretrain_corpus_v11.pt
EXP=atlas_ase_pretrain_v11

say() { echo "$(date '+%F %T'): $*" | tee -a "$LOG"; }
die() { say "ABORT: $*"; exit 1; }

n=$(ls "$SRC"/*.motion 2>/dev/null | wc -l)
[ "$n" -eq 135 ] || die "expected 135 pristine clips in $SRC, found $n"
say "source: $n pristine retarget clips"

say "step 1/3: rate-limited foot correction, pristine -> $DST"
rm -f "$DST"/*.motion
PYTHONPATH=data/scripts $PY data/scripts/fix_foot_clips.py \
    --robot atlas --in-dir "$SRC" --out-dir "$DST" >> "$LOG" 2>&1 \
    || die "foot fix failed"
m=$(ls "$DST"/*.motion 2>/dev/null | wc -l)
[ "$m" -eq 135 ] || die "expected 135 corrected clips, found $m"

say "step 2/3: rebuild $CORPUS"
$PY -m protomotions.components.motion_lib \
    --motion-path "$DST" --output-file "$CORPUS" --device cpu >> "$LOG" 2>&1 \
    || die "corpus rebuild failed"

# Validate before spending GPU time: clip count, no fall clips, nothing
# non-finite, and velocity magnitudes still inside the actuator envelope.
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

say "step 3/3: stop and resume $EXP on GPU 1"
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
