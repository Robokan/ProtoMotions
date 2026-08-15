#!/usr/bin/env bash
# Tiered corpus rebuild (Eric, 2026-08-15): natural clips as base, cheapest
# adequate fix per clip -- keep raw / constant widen / sibling-swap-trim-drop.
set -u
cd /home/bizon/sparkpack/ProtoMotions
LOG=results/atlas_v15_build.log
PY=/home/bizon/sparkpack/.venv-isaacsim5/bin/python
say() { echo "$(date '+%F %T'): $*" >> "$LOG"; }
say "tiered rebuild starting (base: atlas_v12, natural)"
rm -rf data/motions/atlas_v15
$PY data/scripts/build_tiered_corpus.py >> "$LOG" 2>&1 || { say "ABORT: tiering"; exit 1; }
$PY - >> "$LOG" 2>&1 <<'PYEOF'
import glob, os, yaml
fs=sorted(glob.glob("data/motions/atlas_v15/*.motion"))
yaml.safe_dump({"motions":[{"file":f"motions/atlas_v15/{os.path.basename(f)}","weight":1.0} for f in fs]},
               open("data/atlas_v15_recipe.yaml","w"),sort_keys=False)
print(f"recipe: {len(fs)} clips")
PYEOF
$PY -m protomotions.components.motion_lib --motion-path data/atlas_v15_recipe.yaml \
  --output-file data/atlas_pretrain_corpus_v15.pt --device cpu >> "$LOG" 2>&1 || { say "ABORT: corpus"; exit 1; }
$PY - >> "$LOG" 2>&1 <<'PYEOF'
import sys, torch, numpy as np
sys.path.insert(0,".")
from protomotions.robot_configs.factory import robot_config
ci=robot_config("atlas").control.control_info; dn=list(ci.keys())
lim=np.array([getattr(ci[k],"velocity_limit",np.inf) for k in dn])
d=torch.load("data/atlas_pretrain_corpus_v15.pt",weights_only=False,map_location="cpu")
bad=[k for k in ("gts","grs","dps","gvs","gavs","dvs") if not torch.isfinite(d[k]).all()]
v=d["dvs"].abs().numpy()
print(f"v15: {len(d['motion_lengths'])} clips, {d['gts'].shape[0]} frames, {float(sum(d['motion_lengths'])):.1f}s")
print(f"  non-finite {bad or 'none'} | over-limit {(v>lim[None,:]).sum()}")
assert not bad
PYEOF
if [ $? -ne 0 ]; then say "ABORT: validation"; exit 1; fi
say "V15 DONE"
