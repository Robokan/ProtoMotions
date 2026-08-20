# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Numerically compare the AMP discriminator's two observation branches.

The discriminator judges "agent vs expert", but the two sides are produced by
DIFFERENT code paths:

  agent  : sim state history -> compute_historical_max_coords_from_state
  expert : motion library    -> compute_historical_max_coords_from_motion_lib

If those disagree representationally (body order, rotation convention, frame,
height reference), every agent sample is stamped fake regardless of the
motion, the disc separates trivially forever, and the policy flails while the
disc looks "healthy". ANYmal AMP shows exactly that signature (logit gap ~13
vs working Atlas ~5), so this script settles it directly:

Drive the simulator KINEMATICALLY along a reference clip (KinematicReplay sets
sim state to the mocap pose every step), so the sim state IS the expert
motion. Then the agent-branch obs computed from that state must match the
expert-branch obs computed from the motion library at the same times, up to
interpolation error. A large per-channel mismatch is the bug, and which
channels disagree says where it lives.

    CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES \
    ~/sparkpack/.venv-isaacsim6/bin/python scripts/check_disc_obs_parity.py \
        --robot-name anymal_d --motion-file data/anymal_walk30_amp.pt \
        --experiment-path examples/experiments/amp/mlp.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

parser = argparse.ArgumentParser()
parser.add_argument("--robot-name", default="anymal_d")
parser.add_argument("--motion-file", default="data/anymal_walk30_amp.pt")
parser.add_argument("--experiment-path", default="examples/experiments/amp/mlp.py")
parser.add_argument("--simulator", default="isaaclab")
parser.add_argument("--physics", default="physx")
parser.add_argument("--motion-id", type=int, default=1, help="clip to replay")
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--warmup", type=int, default=40, help="steps to fill the 32-deep history")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
args.headless = True
args.scenes_file = None
args.overrides = None
args.env_spacing = 2.0
args.extra_args = []
args.experiment_name = "disc_obs_parity_check"

import torch  # noqa: E402

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Isaac app must exist before isaaclab imports.
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

# Kinematic replay: sim state = mocap pose each step. Keep the observation
# components -- they are the object under test (playback example strips them).
from protomotions.envs.control.kinematic_replay_control import (  # noqa: E402
    KinematicReplayControlConfig,
)

env_config.control_components = {"kinematic_replay": KinematicReplayControlConfig()}
env_config.termination_components = {}
env_config.reward_components = {}
env_config.show_terrain_markers = False
# Pin every env to the chosen clip; no random resampling mid-test.
env_config.motion_manager.subset_method = [args.motion_id]
env_config.motion_manager.init_start_prob = 1.0  # start at t=0, deterministic

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

# Two patches to make replay usable for THIS comparison:
# 1. Stock KinematicReplayControl zeroes velocities on teleport -- fine for
#    eyeballing poses, fatal here: expert obs carry real mocap velocities, so
#    the velocity channels would "mismatch" by construction. Keep them.
# 2. It calls motion_manager.get_done_tracks(), which only the mimic manager
#    has; give the plain AMP manager an equivalent.
from protomotions.envs.control import kinematic_replay_control as _krc  # noqa: E402
from protomotions.simulator.base_simulator.simulator_state import ResetState  # noqa: E402


def _replay_step_keep_velocities(self):
    all_env_ids = torch.arange(self.env.num_envs, dtype=torch.long, device=self.env.device)
    if self.env.consume_reset_request():
        self.env.motion_manager.sample_motions(all_env_ids)
        self.env.motion_manager.motion_times[:] = 0.0
    sync_dt = self.env.simulator.decimation * 1.0 / self.env.simulator.config.sim.fps
    mm = self.env.motion_manager
    mm.motion_times += sync_dt
    done = mm.motion_times >= self.env.motion_lib.motion_lengths[mm.motion_ids]
    if bool(done.any()):
        ids = torch.where(done)[0]
        mm.sample_motions(ids)
        mm.motion_times[ids] = 0.0
    ref_state = self.env.motion_lib.get_motion_state(mm.motion_ids, mm.motion_times)
    ref_reset_state = ResetState.from_robot_state(ref_state)  # velocities intact
    offset = self.env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
        ref_reset_state.root_pos[:, None, :], all_env_ids
    ).squeeze(1)
    ref_reset_state.root_pos += offset
    ref_object_state = self.env.scene_lib.get_scene_pose(
        all_env_ids, mm.motion_times, self.env.config.ref_object_respawn_offset
    )
    self.env.simulator.reset_envs(ref_reset_state, ref_object_state, all_env_ids)
    self.env.progress_buf[all_env_ids] = 0
    self.env.reset_buf[all_env_ids] = 0
    self.env.terminate_buf[all_env_ids] = 0


_krc.KinematicReplayControl.step = _replay_step_keep_velocities

from protomotions.envs.base_env.env import BaseEnv  # noqa: E402

env = BaseEnv(
    config=env_config,
    robot_config=robot_config,
    device=device,
    terrain=components["terrain"],
    scene_lib=components["scene_lib"],
    motion_lib=components["motion_lib"],
    simulator=components["simulator"],
)

obs, _ = env.reset()
OBS_KEY = "historical_max_coords_obs"
assert OBS_KEY in obs, f"{OBS_KEY} not in obs: {list(obs.keys())}"

comp = env_config.observation_components[OBS_KEY]
static = dict(comp.static_params)
print(f"obs component static params: {static}", flush=True)

from protomotions.envs.obs.humanoid_historical import (  # noqa: E402
    compute_historical_max_coords_from_motion_lib,
)

actions = torch.zeros(env.num_envs, robot_config.number_of_actions, device=device)

# per-frame obs layout (root_h + per-body blocks), for localizing mismatches
num_bodies = len(robot_config.kinematic_info.body_names)
step_list = static.get("history_steps") or list(
    range(1, env_config.num_state_history_steps + 1)
)

max_err = None
sum_err = None
count = 0
worst = (0.0, -1, -1)  # (err, flat_channel, step)

for step in range(args.steps):
    obs, _, dones, _, _ = env.step(actions)
    if step < args.warmup:
        continue

    agent_obs = obs[OBS_KEY]
    ids = env.motion_manager.motion_ids.clone()
    times = env.motion_manager.motion_times.clone()

    expert_obs = compute_historical_max_coords_from_motion_lib(
        motion_lib=env.motion_lib,
        motion_ids=ids,
        motion_times=times,
        num_state_history_steps=env_config.num_state_history_steps,
        dt=env.dt,
        local_obs=static.get("local_obs", True),
        root_height_obs=static.get("root_height_obs", True),
        observe_contacts=static.get("observe_contacts", False),
        history_steps=static.get("history_steps"),
        body_ids=static.get("body_ids"),
    )

    err = (agent_obs - expert_obs).abs()
    if max_err is None:
        max_err = err.max(dim=0).values
        sum_err = err.sum(dim=0)
    else:
        max_err = torch.maximum(max_err, err.max(dim=0).values)
        sum_err += err.sum(dim=0)
    count += agent_obs.shape[0]
    e, c = err.max().item(), int(err.argmax().item() % err.shape[1])
    if e > worst[0]:
        worst = (e, c, step)

mean_err = sum_err / count
obs_dim_per_step = agent_obs.shape[1] // len(step_list)

print("\n================ DISC OBS PARITY ================", flush=True)
print(f"frames compared : {count} (steps {args.warmup}..{args.steps})")
print(f"obs dim         : {agent_obs.shape[1]} = {len(step_list)} steps x {obs_dim_per_step}")
print(f"overall mean |err| : {mean_err.mean().item():.6f}")
print(f"overall max  |err| : {max_err.max().item():.6f}")

# Break down by history step
print("\nper-history-step mean |err| (older steps tolerate more interpolation error):")
for i, s in enumerate(step_list):
    blk = mean_err[i * obs_dim_per_step : (i + 1) * obs_dim_per_step]
    print(f"  step -{s:2d}: mean {blk.mean().item():.6f}  max {blk.max().item():.6f}")

# Break down the FIRST history step by the actual layout of
# compute_humanoid_max_coords_observations:
#   [ root_h(1) | pos: bodies 1..B-1 (3 each) | rot: bodies 0..B-1 (6 each)
#     | vel: bodies 0..B-1 (3 each) | angvel: bodies 0..B-1 (3 each) ]
first = mean_err[:obs_dim_per_step]
names = robot_config.kinematic_info.body_names
B = num_bodies
expected = 1 + (B - 1) * 3 + B * 6 + B * 3 + B * 3
print(f"\nfirst-step channel profile (dim {obs_dim_per_step}, layout expects {expected}):")
print("  root height: mean |err| %.6f" % first[0].item())
if expected == obs_dim_per_step:
    o = 1
    blocks = [("pos", 3, list(range(1, B))), ("rot6d", 6, list(range(B))),
              ("vel", 3, list(range(B))), ("angvel", 3, list(range(B)))]
    for label, width, bodies in blocks:
        print(f"  -- {label} --")
        for b in bodies:
            blk = first[o : o + width]
            flag = "  <-- MISMATCH" if blk.mean().item() > 0.05 else ""
            print("     %-22s mean %.4f  max %.4f%s" % (names[b], blk.mean().item(), blk.max().item(), flag))
            o += width
else:
    top = torch.topk(first, k=min(12, first.numel()))
    for v, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"    channel {i}: mean |err| {v:.6f}")

print("\nVERDICT:", flush=True)
if max_err.max().item() < 0.05:
    print("  MATCH -- branches agree to interpolation error; the disc obs")
    print("  pipeline is NOT the bug. Next suspects: bounds penalty, mini-epochs.")
elif mean_err.mean().item() > 0.2:
    print("  SYSTEMATIC MISMATCH -- the two branches disagree on most channels.")
    print("  The discriminator can separate agent/expert by representation alone.")
else:
    print("  PARTIAL MISMATCH -- localized channels disagree (see profile above).")
    print("  Those channels identify the offending convention.")

simulation_app.close()
