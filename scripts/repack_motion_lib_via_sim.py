# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Rebuild a packaged motion library's global tensors through the simulator.

Why: a retargeted corpus stores global body positions/rotations (gts/grs)
computed by the RETARGET pipeline's forward kinematics. If that pipeline's
link frames differ from the simulator's USD model -- as they do for ANYmal,
whose corpus rotations disagree with sim link frames by up to a full sign
flip on the hind hips while positions match to millimeters -- then every
AMP/ASE discriminator comparison of "agent (sim frames) vs expert (retarget
frames)" is separable by representation alone, and no policy can ever fool
it. scripts/check_disc_obs_parity.py measures exactly this.

Fix: keep the corpus's actual content (root trajectory + dof trajectories,
which drive the robot and render correctly) and re-derive everything else
from the simulator itself. Frame by frame: teleport the sim robot to the
corpus root pose + dof pose, read back all link transforms, store those as
the new gts/grs. Linear/angular velocities are recomputed by finite
difference (per motion, at the corpus dt), which is how retarget pipelines
derive them anyway. After this, expert data and agent rollouts share one
kinematic model BY CONSTRUCTION.

    CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES \
    ~/sparkpack/.venv-isaacsim6/bin/python scripts/repack_motion_lib_via_sim.py \
        --robot-name anymal_d \
        --in-file data/anymal_walk30_amp.pt \
        --out-file data/anymal_walk30_amp_simframes.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

parser = argparse.ArgumentParser()
parser.add_argument("--robot-name", required=True)
parser.add_argument("--in-file", required=True)
parser.add_argument("--out-file", required=True)
parser.add_argument("--experiment-path", default="examples/experiments/amp/mlp.py")
parser.add_argument("--simulator", default="isaaclab")
parser.add_argument("--physics", default="physx")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--no-ground-feet",
    action="store_true",
    help="Skip the per-clip vertical shift that grounds the feet. By default "
    "each clip is lowered (or raised) by a constant so its lowest foot point "
    "over the whole clip matches the foot height of the robot's default "
    "STANDING state -- the retargeted ANYmal clips float visibly above the "
    "ground, which is itself discriminator-separable (expert root/foot "
    "heights say 'hovering', a physical robot's say 'grounded').",
)
args = parser.parse_args()
args.headless = True
args.scenes_file = None
args.overrides = None
args.env_spacing = 2.0
args.extra_args = []
args.experiment_name = "repack_via_sim"
args.motion_file = args.in_file

import torch  # noqa: E402

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

src = torch.load(args.in_file, map_location="cpu", weights_only=False)
for key in ("gts", "grs", "dps", "length_starts", "motion_num_frames", "motion_dt"):
    assert key in src, f"{args.in_file} lacks '{key}' -- not a packaged motion lib"
total_frames = src["gts"].shape[0]
num_motions = len(src["motion_num_frames"])
print(f"repacking {args.in_file}: {num_motions} motions, {total_frames} frames")

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher({"headless": True, "device": str(device)})
simulation_app = app_launcher.app

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("experiment_module", args.experiment_path)
experiment_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_module)

from protomotions.utils.config_builder import build_standard_configs  # noqa: E402

configs = build_standard_configs(
    args=args,
    terrain_config_fn=experiment_module.terrain_config,
    scene_lib_config_fn=experiment_module.scene_lib_config,
    motion_lib_config_fn=experiment_module.motion_lib_config,
    env_config_fn=experiment_module.env_config,
    configure_robot_and_simulator_fn=getattr(
        experiment_module, "configure_robot_and_simulator", None
    ),
    agent_config_fn=None,
)
robot_config = configs["robot"]
simulator_config = configs["simulator"]
env_config = configs["env"]

# Bare env: we teleport manually; nothing else may touch the state.
env_config.control_components = {}
env_config.termination_components = {}
env_config.reward_components = {}
env_config.observation_components = {}
env_config.show_terrain_markers = False

if hasattr(simulator_config, "projectile"):
    simulator_config.projectile.num_projectiles = 0

from protomotions.simulator.base_simulator.utils import (  # noqa: E402
    convert_friction_for_simulator,
)

terrain_config, simulator_config = convert_friction_for_simulator(
    configs["terrain"], simulator_config
)

from protomotions.utils.component_builder import build_all_components  # noqa: E402

components = build_all_components(
    terrain_config=terrain_config,
    scene_lib_config=configs["scene_lib"],
    motion_lib_config=configs["motion_lib"],
    simulator_config=simulator_config,
    robot_config=robot_config,
    device=device,
    simulation_app=simulation_app,
)

from protomotions.envs.base_env.env import BaseEnv  # noqa: E402
from protomotions.simulator.base_simulator.simulator_state import (  # noqa: E402
    ResetState,
    StateConversion,
)

env = BaseEnv(
    config=env_config,
    robot_config=robot_config,
    device=device,
    terrain=components["terrain"],
    scene_lib=components["scene_lib"],
    motion_lib=components["motion_lib"],
    simulator=components["simulator"],
)
env.reset()
sim = env.simulator
env_ids = torch.arange(env.num_envs, dtype=torch.long, device=device)
actions = torch.zeros(env.num_envs, robot_config.number_of_actions, device=device)

gts_src = src["gts"].to(device)
grs_src = src["grs"].to(device)
dps_src = src["dps"].to(device)
dvs_src = src.get("dvs")
dvs_src = dvs_src.to(device) if dvs_src is not None else torch.zeros_like(dps_src)

num_bodies = gts_src.shape[1]
new_gts = torch.zeros_like(gts_src)
new_grs = torch.zeros_like(grs_src)
new_gvs_sim = torch.zeros_like(gts_src)
new_gavs_sim = torch.zeros_like(gts_src)
gvs_src = src["gvs"].to(device)
gavs_src = src["gavs"].to(device)

# Reference "feet touching" height: teleport to the robot's DEFAULT standing
# state (feet on the ground by design) and read where the foot link origins
# sit. This is the target for the per-clip grounding shift -- measured from
# the sim, not guessed from a collision radius.
names = robot_config.kinematic_info.body_names
foot_ids = [i for i, n in enumerate(names) if "FOOT" in n.upper() or "foot" in n]
ref_foot_z = None
if foot_ids and not args.no_ground_feet:
    default_state = sim.get_default_robot_reset_state()
    sim.reset_envs(default_state, None, env_ids)
    standing = sim.get_bodies_state(env_ids)
    # Height of foot origins above the ground under this env (origin offset
    # cancels: use feet relative to root, plus the default root height).
    root_z = standing.rigid_body_pos[0, 0, 2]
    feet_z = standing.rigid_body_pos[0, foot_ids, 2]
    default_root_h = default_state.root_pos[0, 2]
    ref_foot_z = float((feet_z - root_z).min() + default_root_h)
    print(f"feet: {[names[i] for i in foot_ids]}")
    print(f"reference standing foot-origin height: {ref_foot_z:.4f} m")

# Teleport -> physics step -> readback. The physics step is what propagates
# the write into the readable link transforms (verified by the parity test:
# post-teleport readback matched corpus positions to ~2mm). The drift it adds
# is one control step from the EXACT teleported state and shows up at the
# ~1e-2 level, an order below the 0.35-2.0 frame mismatch being repaired.
for f in range(total_frames):
    # Teleport WITH the corpus root/dof velocities: the sim then reports every
    # link's velocity in ITS convention -- IsaacLab's body_lin_vel_w is the
    # COM velocity (both Lab 2 and Lab 3), while retarget corpora store
    # link-origin finite differences. v_com = v_link + w x r_offset, which on
    # ANYmal's offset-COM thighs/feet measured 0.3-0.5 m/s of systematic
    # disagreement against a 1.16 m/s walk -- 51 discriminator channels
    # separable by convention alone. Recording the readback makes expert
    # velocities agree with agent rollouts by construction, like the frames.
    reset_state = ResetState(
        state_conversion=StateConversion.COMMON,
        root_pos=gts_src[f, 0].unsqueeze(0).expand(env.num_envs, -1).clone(),
        root_rot=grs_src[f, 0].unsqueeze(0).expand(env.num_envs, -1).clone(),
        root_vel=gvs_src[f, 0].unsqueeze(0).expand(env.num_envs, -1).clone(),
        root_ang_vel=gavs_src[f, 0].unsqueeze(0).expand(env.num_envs, -1).clone(),
        dof_pos=dps_src[f].unsqueeze(0).expand(env.num_envs, -1).clone(),
        dof_vel=dvs_src[f].unsqueeze(0).expand(env.num_envs, -1).clone(),
    )
    sim.reset_envs(reset_state, None, env_ids)
    body_state = sim.get_bodies_state(env_ids)
    pos = body_state.rigid_body_pos[0]
    rot = body_state.rigid_body_rot[0]
    # Remove env-origin/terrain offset: pin the readback root to the corpus root.
    delta = pos[0] - gts_src[f, 0]
    new_gts[f] = pos - delta
    new_grs[f] = rot
    new_gvs_sim[f] = body_state.rigid_body_vel[0]
    new_gavs_sim[f] = body_state.rigid_body_ang_vel[0]
    if f % 500 == 0:
        print(f"  frame {f}/{total_frames}", flush=True)

# Ground the feet: per clip, one constant vertical shift so the lowest foot
# point over the whole clip sits at the standing reference. A constant shift
# is a rigid transform -- gait dynamics, rotations, and (finite-difference)
# velocities are untouched.
if ref_foot_z is not None:
    print("\nper-clip grounding shift:")
    for m in range(num_motions):
        s, e = int(src["length_starts"][m]), int(src["length_starts"][m] + src["motion_num_frames"][m])
        min_foot = float(new_gts[s:e][:, foot_ids, 2].min())
        dz = min_foot - ref_foot_z
        new_gts[s:e, :, 2] -= dz
        print(f"  motion {m} ({src['motion_files'][m]}): lowered by {dz:+.4f} m")

# Velocities come from the sim readback (COM convention, matching agent
# rollouts). Grounding is a constant per-clip shift, so it does not alter them.
new_gvs = new_gvs_sim
new_gavs = new_gavs_sim

out = dict(src)
out["gts"] = new_gts.cpu()
out["grs"] = new_grs.cpu()
out["gvs"] = new_gvs.cpu()
out["gavs"] = new_gavs.cpu()
torch.save(out, args.out_file)

# Report how much actually changed -- this is the size of the bug.
rot_delta = (grs_src.cpu() - new_grs.cpu()).abs()
pos_delta = (gts_src.cpu() - new_gts.cpu()).abs()
print("\n================ REPACK COMPLETE ================")
print(f"wrote {args.out_file}")
print(f"pos: mean |delta| {pos_delta.mean():.5f}  max {pos_delta.max():.5f}")
print(f"rot: mean |delta| {rot_delta.mean():.5f}  max {rot_delta.max():.5f}")
per_body_rot = rot_delta.mean(dim=(0, 2))
names = robot_config.kinematic_info.body_names
print("per-body rotation change (quat units):")
for b in range(num_bodies):
    flag = "  <-- was wrong in the corpus" if per_body_rot[b] > 0.05 else ""
    print(f"  {names[b]:<22s} {per_body_rot[b]:.4f}{flag}")

simulation_app.close()
