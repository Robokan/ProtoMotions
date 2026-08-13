#!/usr/bin/env bash
# Build atlas corpus v13: full re-retarget with the GMR arm-flip fix
# (Arm3 rot_w=2, aa599c2) and leg collision avoidance, then the standard
# post chain. Every stage validates before the next runs.
#
#   v12 stems -> source BVHs -> retarget (10 shards)
#     -> convert --clamp-dof-vel 0.95
#     -> fix_foot_clips (foot-ground correction)
#     -> trim_motion_collisions --depth-cm 1 --blip-frames 1
#        (upgraded pop logic b893a00: island / tail-drop / plateau)
#     -> recipe + corpus + validation
#
# Written 2026-08-13. Safe to re-run; stages are idempotent via rm -rf.
set -u
cd /home/bizon/sparkpack/ProtoMotions
LOG=results/atlas_v13_build.log
GMR=/home/bizon/sparkpack/GMR_Grab
PY=/home/bizon/sparkpack/.venv-isaacsim5/bin/python
GPY=$GMR/.venv/bin/python
W=/tmp/claude-1000/-home-bizon-sparkpack/2869939e-7c3b-46e1-998b-1fb4b17b134c/scratchpad/v13
say() { echo "$(date '+%F %T'): $*" >> "$LOG"; }
rm -rf "$W"; mkdir -p "$W"

# 1. source BVH list from v12 stems (strip __pN trim suffixes, dedup)
$PY - >> "$LOG" 2>&1 <<'PYEOF'
import glob, os, re
stems=set()
for p in glob.glob("data/motions/atlas_v12/*.motion"):
    stems.add(re.sub(r"__p\d+$","",os.path.basename(p)[:-7]))
found=[]; missing=[]
for s in sorted(stems):
    for d in ("atlas_seed_bvh_f8","atlas_combat_bvh_f8"):
        p=os.path.expanduser(f"~/sparkpack/output/{d}/{s}.bvh")
        if os.path.exists(p): found.append(p); break
    else: missing.append(s)
W="/tmp/claude-1000/-home-bizon-sparkpack/2869939e-7c3b-46e1-998b-1fb4b17b134c/scratchpad/v13"
open(f"{W}/bvh.list","w").write("\n".join(found)+"\n")
print(f"bvh list: {len(found)} found, missing: {missing or 'none'}")
assert not missing, "missing BVHs would silently shrink the corpus"
PYEOF
[ $? -ne 0 ] && { say "ABORT: bvh list"; exit 1; }
say "step 1 ok: $(wc -l < $W/bvh.list) source BVHs"

# 2. retarget, 10 shards
cd "$GMR"
for i in $(seq 0 9); do
  nohup $GPY scripts/retarget_headless.py --bvh_list $W/bvh.list --shard $i/10 \
    --out_dir $W/npz --robot atlas_fists --force > $W/shard_$i.log 2>&1 &
done
wait
cd /home/bizon/sparkpack/ProtoMotions
N=$(ls $W/npz/*.npz 2>/dev/null | wc -l)
say "step 2: retargeted $N clips"
[ "$N" -ne "$(wc -l < $W/bvh.list)" ] && { say "ABORT: retarget count mismatch"; exit 1; }

# 3. convert with velocity clamp
$PY data/scripts/convert_gmr_pkl_to_proto.py --input-dir $W/npz \
  --output-dir $W/motion --clamp-dof-vel 0.95 --force-remake >> "$LOG" 2>&1
[ $? -ne 0 ] && { say "ABORT: convert"; exit 1; }
say "step 3: converted $(ls $W/motion/*.motion | wc -l)"

# 4. foot-ground correction
$PY data/scripts/fix_foot_clips.py --robot atlas --in-dir $W/motion \
  --out-dir $W/footfix >> "$LOG" 2>&1
[ $? -ne 0 ] && { say "ABORT: foot fix"; exit 1; }
say "step 4: foot-fixed $(ls $W/footfix/*.motion | wc -l)"

# 5. trim collisions >1cm and pop damage (tail-aware)
rm -rf data/motions/atlas_v13
$PY data/scripts/trim_motion_collisions.py --robot atlas --in-dir $W/footfix \
  --out-dir data/motions/atlas_v13 --depth-cm 1 --blip-frames 1 >> "$LOG" 2>&1
[ $? -ne 0 ] && { say "ABORT: trim"; exit 1; }
say "step 5: trimmed -> $(ls data/motions/atlas_v13/*.motion | wc -l) clips"

# 6. recipe + corpus + validation
$PY - >> "$LOG" 2>&1 <<'PYEOF'
import glob, os, yaml
fs=sorted(glob.glob("data/motions/atlas_v13/*.motion"))
yaml.safe_dump({"motions":[{"file":f"motions/atlas_v13/{os.path.basename(f)}","weight":1.0} for f in fs]},
               open("data/atlas_v13_recipe.yaml","w"),sort_keys=False)
print(f"recipe: {len(fs)} clips")
PYEOF
$PY -m protomotions.components.motion_lib --motion-path data/atlas_v13_recipe.yaml \
  --output-file data/atlas_pretrain_corpus_v13.pt --device cpu >> "$LOG" 2>&1
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
[ $? -ne 0 ] && { say "ABORT: corpus validation"; exit 1; }
say "DONE: data/atlas_pretrain_corpus_v13.pt"
