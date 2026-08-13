#!/usr/bin/env bash
# Final v13 chain (Eric, 2026-08-13 evening): wait for the running foot fix,
# then legs optimizer -> arms optimizer -> pop trim -> corpus v13 -> validate
# -> STOP atlas_ase_pretrain_v12_tuned -> warm-start atlas_ase_pretrain_v13
# on GPU 1 with the tuned trainer.
set -u
cd /home/bizon/sparkpack/ProtoMotions
LOG=results/atlas_v13_build.log
PY=/home/bizon/sparkpack/.venv-isaacsim5/bin/python
W=/tmp/claude-1000/-home-bizon-sparkpack/2869939e-7c3b-46e1-998b-1fb4b17b134c/scratchpad/v13tail
say() { echo "$(date '+%F %T'): $*" >> "$LOG"; }

say "CHAIN: waiting for foot fix"
while pgrep -f "fix_foot_clips" > /dev/null; do sleep 30; done
N=$(ls $W/footfix/*.motion 2>/dev/null | wc -l)
[ "$N" -ne 131 ] && { say "ABORT: foot fix produced $N/131"; exit 1; }
say "foot fix done: $N"

$PY data/scripts/optimize_leg_trajectories.py --robot atlas --mode legs \
  --in-dir $W/footfix --out-dir $W/optlegs \
  --w-pen 400000 --w-foot 400000 --maxiter 200 >> "$LOG" 2>&1 \
  || { say "ABORT: leg optimizer"; exit 1; }
say "legs optimized: $(ls $W/optlegs/*.motion | wc -l)"

$PY data/scripts/optimize_leg_trajectories.py --robot atlas --mode arms \
  --in-dir $W/optlegs --out-dir $W/optarms \
  --w-pen 400000 --maxiter 200 >> "$LOG" 2>&1 \
  || { say "ABORT: arm optimizer"; exit 1; }
say "arms optimized: $(ls $W/optarms/*.motion | wc -l)"

rm -rf data/motions/atlas_v13
$PY data/scripts/trim_motion_collisions.py --robot atlas --in-dir $W/optarms \
  --out-dir data/motions/atlas_v13 --depth-cm 1 --blip-frames 1 >> "$LOG" 2>&1 \
  || { say "ABORT: trim"; exit 1; }
say "trimmed: $(ls data/motions/atlas_v13/*.motion | wc -l) clips"

$PY - >> "$LOG" 2>&1 <<'PYEOF'
import glob, os, yaml
fs=sorted(glob.glob("data/motions/atlas_v13/*.motion"))
yaml.safe_dump({"motions":[{"file":f"motions/atlas_v13/{os.path.basename(f)}","weight":1.0} for f in fs]},
               open("data/atlas_v13_recipe.yaml","w"),sort_keys=False)
print(f"recipe: {len(fs)} clips")
PYEOF
$PY -m protomotions.components.motion_lib --motion-path data/atlas_v13_recipe.yaml \
  --output-file data/atlas_pretrain_corpus_v13.pt --device cpu >> "$LOG" 2>&1 \
  || { say "ABORT: corpus build"; exit 1; }
$PY - >> "$LOG" 2>&1 <<'PYEOF'
import sys, torch, numpy as np
sys.path.insert(0,".")
from protomotions.robot_configs.factory import robot_config
ci=robot_config("atlas").control.control_info; dn=list(ci.keys())
lim=np.array([getattr(ci[k],"velocity_limit",np.inf) for k in dn])
d=torch.load("data/atlas_pretrain_corpus_v13.pt",weights_only=False,map_location="cpu")
bad=[k for k in ("gts","grs","dps","gvs","gavs","dvs") if not torch.isfinite(d[k]).all()]
v=d["dvs"].abs().numpy()
n=len(d["motion_lengths"])
print(f"v13: {n} clips, {d['gts'].shape[0]} frames, {float(sum(d['motion_lengths'])):.1f}s")
print(f"  non-finite {bad or 'none'} | over-limit dof_vel {(v>lim[None,:]).sum()}")
assert not bad and n > 100
PYEOF
[ $? -ne 0 ] && { say "ABORT: validation"; exit 1; }
say "CORPUS DONE: data/atlas_pretrain_corpus_v13.pt"

# --- restart training (Eric: "once you have ... rebuilt the corpus restart
# the training"). Warm start from the tuned run's weights; NEW experiment
# name so train_agent takes the warm-start path and re-resolves configs
# against the v13 motion file instead of resuming the pickled v12 path.
OLD=$(ps -eo pid,args | grep "[t]rain_agent.py" | grep atlas_ase_pretrain_v12_tuned | awk '{print $1}' | head -1)
if [ -n "$OLD" ]; then
  say "stopping v12_tuned trainer pid $OLD"
  kill -9 $OLD
  for i in $(seq 1 30); do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    [ "$U" -lt 3000 ] && break
    sleep 5
  done
  say "GPU1 released: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1)"
fi
cp results/atlas_ase_pretrain_v12_tuned/last.ckpt $W/warmstart.ckpt \
  || { say "ABORT: no warm-start checkpoint"; exit 1; }
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
say "TRAINING LAUNCHED pid $!"
