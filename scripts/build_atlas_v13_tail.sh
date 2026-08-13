#!/usr/bin/env bash
# v13 pipeline TAIL: runs from the arm1test retarget (forearm+shoulder
# orientation weights, GMR 02f3072) through corpus build. Split from
# build_atlas_v13.sh because that script was superseded mid-run when the
# shoulder experiment won -- and because editing a bash script while bash
# is executing it resumes at a stale byte offset (the reason the first run
# was killed rather than patched live).
set -u
cd /home/bizon/sparkpack/ProtoMotions
LOG=results/atlas_v13_build.log
PY=/home/bizon/sparkpack/.venv-isaacsim5/bin/python
SRC=/tmp/claude-1000/-home-bizon-sparkpack/2869939e-7c3b-46e1-998b-1fb4b17b134c/scratchpad/arm1test/npz
W=/tmp/claude-1000/-home-bizon-sparkpack/2869939e-7c3b-46e1-998b-1fb4b17b134c/scratchpad/v13tail
say() { echo "$(date '+%F %T'): $*" >> "$LOG"; }
rm -rf "$W"; mkdir -p "$W"
say "TAIL: from $SRC ($(ls $SRC/*.npz | wc -l) npz)"

$PY data/scripts/convert_gmr_pkl_to_proto.py --input-dir $SRC \
  --output-dir $W/motion --clamp-dof-vel 0.95 --force-remake >> "$LOG" 2>&1 \
  || { say "ABORT: convert"; exit 1; }
say "convert: $(ls $W/motion/*.motion | wc -l)"

$PY data/scripts/fix_foot_clips.py --robot atlas --in-dir $W/motion \
  --out-dir $W/footfix >> "$LOG" 2>&1 || { say "ABORT: foot fix"; exit 1; }
say "foot fix: $(ls $W/footfix/*.motion | wc -l)"

$PY data/scripts/optimize_leg_trajectories.py --robot atlas --in-dir $W/footfix \
  --out-dir $W/opt --w-pen 400000 --w-foot 400000 --maxiter 200 >> "$LOG" 2>&1 \
  || { say "ABORT: trajectory optimizer"; exit 1; }
say "optimizer: $(ls $W/opt/*.motion | wc -l)"

rm -rf data/motions/atlas_v13
$PY data/scripts/trim_motion_collisions.py --robot atlas --in-dir $W/opt \
  --out-dir data/motions/atlas_v13 --depth-cm 1 --blip-frames 1 >> "$LOG" 2>&1 \
  || { say "ABORT: trim"; exit 1; }
say "trim: $(ls data/motions/atlas_v13/*.motion | wc -l) clips"

$PY - >> "$LOG" 2>&1 <<'PYEOF'
import glob, os, yaml
fs=sorted(glob.glob("data/motions/atlas_v13/*.motion"))
yaml.safe_dump({"motions":[{"file":f"motions/atlas_v13/{os.path.basename(f)}","weight":1.0} for f in fs]},
               open("data/atlas_v13_recipe.yaml","w"),sort_keys=False)
print(f"recipe: {len(fs)} clips")
PYEOF
$PY -m protomotions.components.motion_lib --motion-path data/atlas_v13_recipe.yaml \
  --output-file data/atlas_pretrain_corpus_v13.pt --device cpu >> "$LOG" 2>&1 \
  || { say "ABORT: corpus"; exit 1; }
$PY - >> "$LOG" 2>&1 <<'PYEOF'
import sys, torch, numpy as np
sys.path.insert(0,".")
from protomotions.robot_configs.factory import robot_config
ci=robot_config("atlas").control.control_info; dn=list(ci.keys())
lim=np.array([getattr(ci[k],"velocity_limit",np.inf) for k in dn])
d=torch.load("data/atlas_pretrain_corpus_v13.pt",weights_only=False,map_location="cpu")
bad=[k for k in ("gts","grs","dps","gvs","gavs","dvs") if not torch.isfinite(d[k]).all()]
v=d["dvs"].abs().numpy()
print(f"v13: {len(d['motion_lengths'])} clips, {d['gts'].shape[0]} frames, {float(sum(d['motion_lengths'])):.1f}s")
print(f"  non-finite {bad or 'none'} | over-limit dof_vel {(v>lim[None,:]).sum()}")
assert not bad
PYEOF
[ $? -ne 0 ] && { say "ABORT: validation"; exit 1; }
say "DONE: data/atlas_pretrain_corpus_v13.pt"
