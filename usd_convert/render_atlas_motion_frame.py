"""Render frames of a motion-lib clip posed on the Atlas robot (verification)."""
import argparse

pa = argparse.ArgumentParser()
pa.add_argument("--lib", default="data/atlas_drunken.pt")
pa.add_argument("--motion", type=int, default=0)
pa.add_argument("--frames", type=int, nargs="+", default=[10, 40])
pa.add_argument("--out", default="output/atlas_motion_frame")
args = pa.parse_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch
AppLauncher = import_simulator_before_torch("isaaclab")
app = AppLauncher({"headless": True, "enable_cameras": True, "device": "cuda:0"}).app

import numpy as np
import torch
import imageio
from protomotions.robot_configs.factory import robot_config
from protomotions.simulator.factory import simulator_config as sim_cfg_fn
from protomotions.components.terrains.config import TerrainConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLib, MotionLibConfig
from protomotions.utils.component_builder import build_all_components
from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator

rc = robot_config("atlas")
sc = sim_cfg_fn("isaaclab", rc, True, 1, "atlas_pose_check")
sc.headless = False
tc, sc = convert_friction_for_simulator(TerrainConfig(), sc)
components = build_all_components(
    terrain_config=tc, scene_lib_config=SceneLibConfig(scene_file=None),
    motion_lib_config=MotionLibConfig(motion_file=args.lib), simulator_config=sc,
    robot_config=rc, device=torch.device("cuda:0"), save_dir=None,
    simulation_app=app,
)
sim = components["simulator"]
ml = components["motion_lib"]
sim._initialize_with_markers({})
sim.render()
for _ in range(120):
    app.update()

for fr in args.frames:
    mid = torch.tensor([args.motion], device="cuda:0", dtype=torch.long)
    t = torch.tensor([fr / 30.0], device="cuda:0")
    st = ml.get_motion_state(mid, t)
    cur = sim.get_robot_state()
    cur.dof_pos = st.dof_pos.to("cuda:0")
    cur.dof_vel[:] = 0
    cur.rigid_body_pos[:, 0] = st.rigid_body_pos[:, 0].to("cuda:0")
    cur.rigid_body_rot[:, 0] = st.rigid_body_rot[:, 0].to("cuda:0")
    cur.rigid_body_vel[:] = 0
    cur.rigid_body_ang_vel[:] = 0
    sim.reset_envs(cur, env_ids=torch.tensor([0], device="cuda:0"))
    root = sim.get_robot_state().root_pos[0].cpu().numpy()
    sim._perspective_view.set_camera_view(
        (root + np.array([2.0, -2.0, 0.5])).tolist(), root.tolist()
    )
    for _ in range(4):
        app.update()
    frame = sim.grab_rgb_frame()
    out = f"{args.out}_{fr}.png"
    imageio.imwrite(out, frame)
    print("WROTE", out, flush=True)

import os
os._exit(0)
