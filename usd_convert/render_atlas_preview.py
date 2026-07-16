"""Render one offscreen frame of the Atlas robot (texture verification)."""
from protomotions.utils.simulator_imports import import_simulator_before_torch
AppLauncher = import_simulator_before_torch("isaaclab")
app = AppLauncher({"headless": True, "enable_cameras": True, "device": "cuda:0"}).app

import torch
import numpy as np
from protomotions.robot_configs.factory import robot_config
from protomotions.simulator.factory import simulator_config as sim_cfg_fn
from protomotions.components.terrains.config import TerrainConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig
from protomotions.utils.component_builder import build_all_components
from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator

rc = robot_config("atlas")
sc = sim_cfg_fn("isaaclab", rc, True, 1, "atlas_render_test")
sc.headless = False  # offscreen render pipeline active
tc, sc = convert_friction_for_simulator(TerrainConfig(), sc)

components = build_all_components(
    terrain_config=tc, scene_lib_config=SceneLibConfig(scene_file=None),
    motion_lib_config=MotionLibConfig(motion_file=None), simulator_config=sc,
    robot_config=rc, device=torch.device("cuda:0"), save_dir=None,
    simulation_app=app,
)
sim = components["simulator"]
sim._initialize_with_markers({})
sim._camera_target = {"env": 0, "element": 0}
sim.render()
for _ in range(150):  # let RTX stream textures in
    app.update()
# robot fell during warmup (gravity runs while streaming textures):
# teleport it back to the spawn pose before framing the shot
st0 = sim.get_default_robot_reset_state() if hasattr(sim, "get_default_robot_reset_state") else None
if st0 is not None:
    sim.reset_envs(st0, env_ids=torch.tensor([0], device="cuda:0"))
# aim AFTER all sim.render() calls (render() re-follows its own target)
root = sim.get_robot_state().root_pos[0].cpu().numpy()
sim._perspective_view.set_camera_view(
    (root + np.array([2.2, -2.2, 0.4])).tolist(), root.tolist()
)
for _ in range(4):
    app.update()
frame = sim.grab_rgb_frame()
if frame is None:
    for _ in range(30):
        app.update()
    frame = sim.grab_rgb_frame()
import imageio
imageio.imwrite("output/atlas_textured.png", frame)
print("WROTE output/atlas_textured.png", frame.shape, "mean rgb:",
      np.asarray(frame).reshape(-1, frame.shape[-1]).mean(0)[:3].round(1), flush=True)
import os
os._exit(0)
