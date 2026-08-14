#!/usr/bin/env bash
# v14 = v13 + passing space (Eric 2026-08-14): gap-triggered leg optimizer,
# 2 cm clearance target, then corpus. No re-retarget, no re-trim (no new
# pops; collisions only shrink).
set -u
cd /home/bizon/sparkpack/ProtoMotions
LOG=results/atlas_v14_build.log
PY=/home/bizon/sparkpack/.venv-isaacsim5/bin/python
say() { echo "$(date '+%F %T'): $*" >> "$LOG"; }
rm -rf data/motions/atlas_v14
$PY data/scripts/optimize_leg_trajectories.py --robot atlas --mode legs \
  --in-dir data/motions/atlas_v13 --out-dir data/motions/atlas_v14 \
  --gap-trigger --trigger-cm 1.5 --clearance-cm 2.0 --margin-cm 4.0 \
  --w-pen 400000 --w-foot 400000 --maxiter 400 >> "$LOG" 2>&1 \
  || { say "ABORT: optimizer"; exit 1; }
say "optimized: $(ls data/motions/atlas_v14/*.motion | wc -l)"
$PY - >> "$LOG" 2>&1 <<'PYEOF'
import glob, os, yaml
fs=sorted(glob.glob("data/motions/atlas_v14/*.motion"))
yaml.safe_dump({"motions":[{"file":f"motions/atlas_v14/{os.path.basename(f)}","weight":1.0} for f in fs]},
               open("data/atlas_v14_recipe.yaml","w"),sort_keys=False)
print(f"recipe: {len(fs)}")
PYEOF
$PY -m protomotions.components.motion_lib --motion-path data/atlas_v14_recipe.yaml \
  --output-file data/atlas_pretrain_corpus_v14.pt --device cpu >> "$LOG" 2>&1 \
  || { say "ABORT: corpus"; exit 1; }
$PY - >> "$LOG" 2>&1 <<'PYEOF'
import sys, torch, numpy as np
sys.path.insert(0,".")
from protomotions.robot_configs.factory import robot_config
ci=robot_config("atlas").control.control_info; dn=list(ci.keys())
lim=np.array([getattr(ci[k],"velocity_limit",np.inf) for k in dn])
d=torch.load("data/atlas_pretrain_corpus_v14.pt",weights_only=False,map_location="cpu")
bad=[k for k in ("gts","grs","dps","gvs","gavs","dvs") if not torch.isfinite(d[k]).all()]
v=d["dvs"].abs().numpy()
print(f"v14: {len(d['motion_lengths'])} clips, {d['gts'].shape[0]} frames")
print(f"  non-finite {bad or 'none'} | over-limit {(v>lim[None,:]).sum()}")
assert not bad
PYEOF
[ $? -ne 0 ] && { say "ABORT: validation"; exit 1; }
say "V14 DONE"
