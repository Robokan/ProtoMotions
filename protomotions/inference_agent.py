# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Test trained agents and visualize their behavior.

This script loads trained checkpoints and runs agents in the simulation environment
for inference, visualization, and analysis. It supports interactive controls,
video recording, and motion playback.

Motion Playback
---------------

For kinematic motion playback (no physics simulation)::

    PYTHON_PATH protomotions/inference_agent.py \\
        --config-name play_motion \\
        +robot=smpl \\
        +simulator=isaacgym \\
        +motion_file=data/motions/walk.motion

Inference Config System
------------------------

Inference loads frozen configs from resolved_configs_inference.pt and applies inference-specific overrides.

Override Priority:

1. CLI overrides (--overrides) - Highest (runtime control)
2. Experiment inference overrides (apply_inference_overrides) - High (experiment-specific inference settings)
3. Frozen configs from resolved_configs.pt - Lowest (exact training configs)

Note: configure_robot_and_simulator() is NOT called during inference (already baked into frozen configs).

Keyboard Controls
-----------------

During inference, these controls are available:

- **J**: Apply random forces to test robustness
- **R**: Reset all environments
- **O**: Toggle camera view
- **L**: Start/stop video recording
- **Q**: Quit

Example
-------
>>> # Test with custom settings
>>> # PYTHON_PATH protomotions/inference_agent.py \\
>>> #     +robot=smpl \\
>>> #     +simulator=isaacgym \\
>>> #     +checkpoint=results/tracker/last.ckpt \\
>>> #     motion_file=data/motions/test.pt \\
>>> #     num_envs=16
"""


def create_parser():
    """Create and configure the argument parser for inference."""
    parser = argparse.ArgumentParser(
        description="Test trained reinforcement learning agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint file to test"
    )
    # Optional arguments
    parser.add_argument(
        "--full-eval",
        action="store_true",
        default=False,
        help="Run full evaluation instead of simple inference",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run simulation in headless mode",
    )
    parser.add_argument(
        "--simulator",
        type=str,
        required=True,
        help="Simulator to use (e.g., 'isaacgym', 'isaaclab', 'newton', 'genesis')",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of parallel environments to run"
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        required=False,
        default=None,
        help="Path to motion file for inference. If not provided, will use the motion file from the checkpoint.",
    )
    parser.add_argument(
        "--scenes-file", type=str, default=None, help="Path to scenes file (optional)"
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Config overrides in format key=value (e.g., env.max_episode_length=5000 simulator.headless=True)",
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=1,
        help="How often to update motion targets (1=every step, 5=every 5 steps). Higher values simulate lower-frequency command updates like GR00T at 5-10 Hz.",
    )
    parser.add_argument(
        "--scene-usd",
        type=str,
        default=None,
        help="Path to a USD scene file to load into the simulation (local or omniverse:// path)",
    )
    parser.add_argument(
        "--scene-offset",
        type=str,
        default="0 0 0",
        help="XYZ offset for the scene (e.g., '0 0 -0.1')",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=None,
        help="Automatically record video for specified seconds then exit",
    )
    
    # Domain randomization options (Replicator-based)
    parser.add_argument(
        "--randomize-lights",
        action="store_true",
        default=False,
        help="Randomize intensity and color of existing lights in the scene",
    )
    parser.add_argument(
        "--randomize-per-frame",
        action="store_true",
        default=False,
        help="Randomize every frame (vs per-episode reset)",
    )
    parser.add_argument(
        "--num-lights",
        type=int,
        default=3,
        help="Number of random lights to add for randomization",
    )
    parser.add_argument(
        "--randomize-objects",
        type=str,
        nargs="*",
        default=None,
        help="List of object names to randomize (e.g., --randomize-objects forklift pallet_01 crate_02)",
    )
    parser.add_argument(
        "--object-pos-range",
        type=str,
        default="-0.5 0.5",
        help="Position randomization range (min max)",
    )
    parser.add_argument(
        "--object-rot-range",
        type=str,
        default="-30 30",
        help="Rotation randomization range in degrees (min max)",
    )
    parser.add_argument(
        "--hide-objects-prob",
        type=float,
        default=0.0,
        help="Probability to hide each object (0-1)",
    )

    return parser


# Parse arguments first (argparse is safe, doesn't import torch)
import argparse  # noqa: E402

parser = create_parser()
args, unknown_args = parser.parse_known_args()

# Import simulator before torch - isaacgym/isaaclab must be imported before torch
# This also returns AppLauncher if using isaaclab, None otherwise
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

# Now safe to import everything else including torch
import logging  # noqa: E402
from pathlib import Path  # noqa: E402
import torch  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402
from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

log = logging.getLogger(__name__)


def find_objects_by_name(stage, root_path, object_names):
    """Find objects in the scene by exact name match.
    
    Args:
        stage: USD stage
        root_path: Root path to search under (e.g., /World/CustomScene)
        object_names: List of object names to find (case-insensitive partial match)
    
    Returns:
        List of matching prim paths
    """
    from pxr import UsdGeom
    
    if not object_names:
        return []
    
    # Convert to lowercase for matching
    names_lower = [n.lower() for n in object_names]
    objects = []
    
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return objects
    
    def _find_matching(prim):
        name = prim.GetName().lower()
        # Check if any of the specified names is in this prim's name
        for target_name in names_lower:
            if target_name in name and prim.IsA(UsdGeom.Xformable):
                objects.append(prim.GetPath().pathString)
                log.info(f"  Found: {prim.GetPath().pathString}")
                break
        for child in prim.GetChildren():
            _find_matching(child)
    
    for child in root_prim.GetAllChildren():
        _find_matching(child)
    
    log.info(f"Found {len(objects)} objects matching names: {object_names}")
    return objects


def setup_scene_randomization(stage, scene_root_path, args):
    """Set up Replicator-based scene and lighting randomization.
    
    Returns a function that triggers randomization when called.
    """
    import numpy as np
    from pxr import UsdGeom, Gf, UsdLux
    
    log.info("Setting up scene randomization with Replicator...")
    
    # Parse ranges
    pos_range = [float(x) for x in args.object_pos_range.split()]
    rot_range = [float(x) for x in args.object_rot_range.split()]
    
    # Find specific objects to randomize (if any specified)
    scene_objects = []
    original_positions = {}  # Store original positions
    if scene_root_path and args.randomize_objects:
        scene_objects = find_objects_by_name(stage, scene_root_path, args.randomize_objects)
        
        for obj_path in scene_objects:
            prim = stage.GetPrimAtPath(obj_path)
            if prim.IsValid():
                xform = UsdGeom.Xformable(prim)
                # Get local transform 
                local_transform = xform.GetLocalTransformation()
                local_pos = local_transform.ExtractTranslation()
                
                # Log existing ops for debugging
                ops = xform.GetOrderedXformOps()
                op_types = [str(op.GetOpType()) for op in ops]
                log.info(f"  {prim.GetName()}: pos={local_pos}, ops={op_types}")
                
                original_positions[obj_path] = local_pos
    
    # Find ALL existing lights in the scene (if light randomization is enabled)
    existing_lights = []
    original_intensities = {}
    if args.randomize_lights:
        light_types = ["RectLight", "DiskLight", "SphereLight", "CylinderLight", 
                       "DistantLight", "DomeLight", "PortalLight", "PluginLight"]
        for prim in stage.Traverse():
            if prim.GetTypeName() in light_types:
                existing_lights.append(prim.GetPath().pathString)
                log.info(f"  Found light: {prim.GetPath()} ({prim.GetTypeName()})")
        
        # Store original intensities so we can randomize relative to them
        for light_path in existing_lights:
            light_prim = stage.GetPrimAtPath(light_path)
            if light_prim.IsValid():
                intensity_attr = light_prim.GetAttribute("inputs:intensity")
                if intensity_attr and intensity_attr.Get():
                    original_intensities[light_path] = intensity_attr.Get()
        
        log.info(f"Found {len(existing_lights)} lights in scene, {len(original_intensities)} with intensity attrs")
    
    def randomize():
        """Randomize lighting and objects."""
        # Randomize lights (if enabled)
        if args.randomize_lights:
            for light_path in existing_lights:
                light_prim = stage.GetPrimAtPath(light_path)
                if not light_prim.IsValid():
                    continue
                
                # Get original intensity or use default
                orig_intensity = original_intensities.get(light_path, 1000.0)
                
                # Randomize intensity (50% to 150% of original)
                intensity_multiplier = np.random.uniform(0.5, 1.5)
                new_intensity = orig_intensity * intensity_multiplier
                
                # Set intensity via attribute
                intensity_attr = light_prim.GetAttribute("inputs:intensity")
                if intensity_attr:
                    intensity_attr.Set(new_intensity)
                
                # Randomize color temperature (warm to cool)
                color_temp = np.random.uniform(3500, 7500)
                if color_temp < 5500:
                    r = 1.0
                    g = 0.9 + 0.1 * (color_temp - 3500) / 2000
                    b = 0.8 + 0.2 * (color_temp - 3500) / 2000
                else:
                    r = 0.9 + 0.1 * (7500 - color_temp) / 2000
                    g = 0.95
                    b = 1.0
                
                color_attr = light_prim.GetAttribute("inputs:color")
                if color_attr:
                    color_attr.Set(Gf.Vec3f(r, g, b))
        
        # Randomize objects (only those explicitly specified)
        if scene_objects:
            for obj_path in scene_objects:
                prim = stage.GetPrimAtPath(obj_path)
                if not prim.IsValid():
                    continue
                
                # Random visibility
                if args.hide_objects_prob > 0:
                    imageable = UsdGeom.Imageable(prim)
                    if np.random.random() < args.hide_objects_prob:
                        imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                    else:
                        imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
                
                # Get original world position
                orig_pos = original_positions.get(obj_path)
                if orig_pos is None:
                    continue
                
                # Calculate random offsets
                offset_x = np.random.uniform(pos_range[0], pos_range[1])
                offset_y = np.random.uniform(pos_range[0], pos_range[1])
                rot_z_offset = np.random.uniform(rot_range[0], rot_range[1])
                
                # Find existing ops
                xform = UsdGeom.Xformable(prim)
                ops = xform.GetOrderedXformOps()
                
                translate_op = None
                rotate_op = None
                matrix_op = None
                for op in ops:
                    op_type = op.GetOpType()
                    if op_type == UsdGeom.XformOp.TypeTranslate and translate_op is None:
                        translate_op = op
                    elif op_type == UsdGeom.XformOp.TypeRotateZ and rotate_op is None:
                        rotate_op = op
                    elif op_type == UsdGeom.XformOp.TypeRotateXYZ and rotate_op is None:
                        rotate_op = op  # Will handle differently
                    elif op_type == UsdGeom.XformOp.TypeTransform and matrix_op is None:
                        matrix_op = op
                
                moved = False
                
                # If there's a translate op, modify it directly
                if translate_op:
                    current = translate_op.Get()
                    if current:
                        new_val = Gf.Vec3d(
                            orig_pos[0] + offset_x,
                            orig_pos[1] + offset_y,
                            current[2]
                        )
                        translate_op.Set(new_val)
                        moved = True
                
                # If there's a matrix op, modify the translation in the matrix
                elif matrix_op:
                    current_matrix = matrix_op.Get()
                    if current_matrix:
                        # Create new matrix with offset translation
                        new_matrix = Gf.Matrix4d(current_matrix)
                        current_trans = new_matrix.ExtractTranslation()
                        new_matrix.SetTranslateOnly(Gf.Vec3d(
                            orig_pos[0] + offset_x,
                            orig_pos[1] + offset_y,
                            current_trans[2]
                        ))
                        matrix_op.Set(new_matrix)
                        moved = True
                
                if moved:
                    log.debug(f"  Moved {prim.GetName()} by ({offset_x:.2f}, {offset_y:.2f})")
                
                # Handle rotation
                if rotate_op:
                    op_type = rotate_op.GetOpType()
                    if op_type == UsdGeom.XformOp.TypeRotateZ:
                        rotate_op.Set(rot_z_offset)
                    elif op_type == UsdGeom.XformOp.TypeRotateXYZ:
                        current_rot = rotate_op.Get() or Gf.Vec3f(0, 0, 0)
                        rotate_op.Set(Gf.Vec3f(current_rot[0], current_rot[1], rot_z_offset))
                elif matrix_op:
                    # For matrix ops, apply rotation to the matrix
                    current_matrix = matrix_op.Get()
                    if current_matrix:
                        rot_matrix = Gf.Matrix4d()
                        rot_matrix.SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), rot_z_offset))
                        new_matrix = rot_matrix * current_matrix
                        matrix_op.Set(new_matrix)
                else:
                    # No rotate op exists - add one
                    new_rotate_op = xform.AddRotateZOp()
                    new_rotate_op.Set(rot_z_offset)
                    log.info(f"  Added RotateZ op to {prim.GetName()}")
    
    log.info("Scene randomization ready")
    return randomize


# def tmp_enable_domain_randomization(robot_cfg, simulator_cfg, env_cfg):
#     """Temporary function to enable domain randomization for testing.

#     TODO: find a better way for sophisticated tmp inference overrides beyond CLI.
#     """
#     from protomotions.simulator.base_simulator.config import (
#         # FrictionDomainRandomizationConfig,
#         CenterOfMassDomainRandomizationConfig,
#         DomainRandomizationConfig,
#     )

#     # env_cfg.terrain.sim_config.static_friction = 0.01
#     # env_cfg.terrain.sim_config.dynamic_friction = 0.01

#     simulator_cfg.domain_randomization = DomainRandomizationConfig(
#         # Uncomment to enable action noise and friction randomization:
#         # action_noise=ActionNoiseDomainRandomizationConfig(
#         #     action_noise_range=(-0.01, 0.01),
#         #     dof_names=[".*"],
#         #     dof_indices=None
#         # ),
#         # friction=FrictionDomainRandomizationConfig(
#         #     num_buckets=64,
#         #     static_friction_range=(0.0, 1.0),
#         #     dynamic_friction_range=(0.0, 1.0),
#         #     restitution_range=(0.0, 0.0),
#         #     body_names=[".*"],
#         #     body_indices=None
#         # ),
#     )
#     log.info("Enabled domain randomization for testing")


def main():
    # Re-use the parser and args from module level
    global parser, args
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)

    # Load frozen configs from resolved_configs.pt (exact reproducibility)
    resolved_configs_path = checkpoint.parent / "resolved_configs_inference.pt"
    assert (
        resolved_configs_path.exists()
    ), f"Could not find resolved configs at {resolved_configs_path}"

    log.info(f"Loading resolved configs from {resolved_configs_path}")
    resolved_configs = torch.load(
        resolved_configs_path, map_location="cpu", weights_only=False
    )

    robot_config = resolved_configs["robot"]
    simulator_config = resolved_configs["simulator"]
    terrain_config = resolved_configs.get("terrain")
    scene_lib_config = resolved_configs["scene_lib"]
    motion_lib_config = resolved_configs["motion_lib"]
    env_config = resolved_configs["env"]
    agent_config = resolved_configs["agent"]

    # Check if we need to switch simulators
    # Extract simulator name from current config's _target_
    current_simulator = simulator_config._target_.split(
        "."
    )[
        -3
    ]  # e.g., "isaacgym" from "protomotions.simulator.isaacgym.simulator.IsaacGymSimulator"

    if args.simulator != current_simulator:
        log.info(
            f"Switching simulator from '{current_simulator}' (training) to '{args.simulator}' (inference)"
        )
        from protomotions.simulator.factory import update_simulator_config_for_test

        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator=args.simulator,
            robot_config=robot_config,
        )
    # Apply backward compatibility fixes for old checkpoints
    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    # # Temporary: Enable domain randomization for testing (uncomment to use)
    # tmp_enable_domain_randomization(robot_config, simulator_config, env_config)

    # Apply CLI runtime overrides
    if args.num_envs is not None:
        log.info(f"CLI override: num_envs = {args.num_envs}")
        simulator_config.num_envs = args.num_envs

    if args.motion_file is not None:
        log.info(f"CLI override: motion_file = {args.motion_file}")
        motion_lib_config.motion_file = args.motion_file  # Always present

    if args.scenes_file is not None:
        log.info(f"CLI override: scenes_file = {args.scenes_file}")
        scene_lib_config.scene_file = args.scenes_file  # Always present

    if args.headless is not None:
        log.info(f"CLI override: headless = {args.headless}")
        simulator_config.headless = args.headless

    # Parse and apply general CLI overrides
    from protomotions.utils.config_utils import (
        parse_cli_overrides,
        apply_config_overrides,
    )

    cli_overrides = parse_cli_overrides(args.overrides) if args.overrides else None

    if cli_overrides:
        apply_config_overrides(
            cli_overrides,
            env_config,
            simulator_config,
            robot_config,
            agent_config,
            terrain_config,
            motion_lib_config,
            scene_lib_config,
        )

    # Create fabric config for inference (simplified)
    fabric_config = FabricConfig(
        devices=1,
        num_nodes=1,
        loggers=[],  # No loggers needed for inference
        callbacks=[],  # No callbacks needed for inference
    )
    fabric: Fabric = Fabric(**fabric_config.to_dict())
    fabric.launch()

    # Setup IsaacLab simulation_app if using IsaacLab simulator
    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {"headless": args.headless, "device": str(fabric.device)}
        app_launcher = AppLauncher(app_launcher_flags)
        simulator_extra_params["simulation_app"] = app_launcher.app

    # Create components (terrain, scene_lib, motion_lib, simulator)
    from protomotions.utils.component_builder import build_all_components

    save_dir_for_weights = (
        getattr(env_config, "save_dir", None)
        if hasattr(env_config, "save_dir")
        else None
    )
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=save_dir_for_weights,
        **simulator_extra_params,  # simulation_app for IsaacLab
    )

    terrain = components["terrain"]
    scene_lib = components["scene_lib"]
    motion_lib = components["motion_lib"]
    simulator = components["simulator"]

    # Create env (auto-initializes simulator)
    from protomotions.envs.base_env.env import BaseEnv

    EnvClass = get_class(env_config._target_)
    env: BaseEnv = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=terrain,
        scene_lib=scene_lib,
        motion_lib=motion_lib,
        simulator=simulator,
    )

    # Enable inference mode for faster demos (skips reward computation)
    env.inference_mode = True
    
    # Set target update interval (simulates lower-frequency commands like GR00T)
    env.inference_target_update_interval = args.target_update_interval

    # Store scene args for loading after reset (need robot position)
    _scene_usd = args.scene_usd if args.simulator == "isaaclab" else None
    _scene_offset = [float(x) for x in args.scene_offset.split()] if args.scene_offset else [0, 0, 0]
    if len(_scene_offset) != 3:
        _scene_offset = [0.0, 0.0, 0.0]
    if args.target_update_interval > 1:
        print(f"Target update interval: every {args.target_update_interval} steps (simulating ~{50//args.target_update_interval} Hz commands)")

    # Determine root_dir for agent based on checkpoint path
    agent_kwargs = {}
    checkpoint_path = Path(args.checkpoint)
    agent_kwargs["root_dir"] = checkpoint_path.parent

    # Create agent
    from protomotions.agents.base_agent.agent import BaseAgent

    # agent_config.evaluator.eval_metric_keys = [
    #     "gt_err",
    #     "gr_err_degrees",
    #     "pow_rew",
    #     "gt_left_foot_contact",
    #     "gt_right_foot_contact",
    #     "pred_left_foot_contact",
    #     "pred_right_foot_contact"
    # ]
    AgentClass = get_class(agent_config._target_)
    agent: BaseAgent = AgentClass(
        config=agent_config, env=env, fabric=fabric, **agent_kwargs
    )

    agent.setup()
    agent.load(args.checkpoint, load_env=False)

    # Load custom USD scene centered on robot position
    if _scene_usd:
        import omni.usd
        from pxr import Gf, UsdGeom
        
        stage = omni.usd.get_context().get_stage()
        scene_prim_path = "/World/CustomScene"
        
        # Create xform and add scene as reference
        scene_prim = stage.DefinePrim(scene_prim_path, "Xform")
        scene_prim.GetReferences().AddReference(_scene_usd)
        
        # Create xformable with translate op
        xformable = UsdGeom.Xformable(scene_prim)
        xformable.ClearXformOpOrder()
        translate_op = xformable.AddTranslateOp()
        
        # Hide default terrain visuals (keep physics for collision)
        terrain_paths = ["/World/ground", "/World/ground/terrain", "/World/ground/terrain/mesh"]
        for path in terrain_paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                imageable = UsdGeom.Imageable(prim)
                imageable.MakeInvisible()
                print(f"Hidden: {path}")
        
        print(f"Loaded scene: {_scene_usd}")
        
        # Set up scene randomization if lights or objects randomization is enabled
        randomize_fn = None
        if args.randomize_lights or args.randomize_objects:
            randomize_fn = setup_scene_randomization(stage, scene_prim_path, args)
        
        # Wrap env.reset to update scene position only when env 0 resets
        _original_reset = env.reset

        def _reset_with_scene_update(env_ids=None):
            result = _original_reset(env_ids)
            # Only update scene if env 0 is being reset (None means all envs)
            env0_reset = env_ids is None or (hasattr(env_ids, '__len__') and 0 in env_ids) or (hasattr(env_ids, 'numel') and (env_ids == 0).any())
            if env0_reset:
                robot_state = env.simulator._get_simulator_root_state(env_ids=None)
                robot_pos = robot_state.root_pos[0].cpu().numpy()
                final_offset = Gf.Vec3d(
                    float(robot_pos[0] + _scene_offset[0]),
                    float(robot_pos[1] + _scene_offset[1]),
                    float(_scene_offset[2])
                )
                translate_op.Set(final_offset)
                # Randomize on episode reset (unless per-frame)
                if randomize_fn and not args.randomize_per_frame:
                    randomize_fn()
            return result
        env.reset = _reset_with_scene_update
        
        # For per-frame randomization, wrap env.step
        if randomize_fn and args.randomize_per_frame:
            _original_step = env.step
            def _step_with_randomize(action):
                randomize_fn()
                return _original_step(action)
            env.step = _step_with_randomize
            print("Per-frame randomization enabled")

    # Auto-record for specified duration using Replicator rgb capture
    if args.record_seconds:
        import os
        import time
        import imageio
        from datetime import datetime
        
        if args.headless:
            print("WARNING: Recording requires non-headless mode. Use --enable_cameras for headless recording.")
        
        print(f"Recording in {args.record_seconds}-second chunks (press Q to quit)...")
        os.makedirs("output/videos", exist_ok=True)
        
        agent.eval()
        done_indices = None
        video_count = 0
        fps = int(1.0 / env.simulator.dt) if hasattr(env.simulator, 'dt') else 50
        
        while env.simulator.is_simulation_running():
            video_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = f"output/videos/demo_{timestamp}.mp4"
            print(f"Recording video #{video_count} at {fps} FPS...")
            
            writer = imageio.get_writer(video_path, fps=fps, codec='libx264', quality=8)
            
            start_time = time.time()
            while time.time() - start_time < args.record_seconds:
                if not env.simulator.is_simulation_running():
                    break
                obs, _ = env.reset(done_indices)
                obs = agent.add_agent_info_to_obs(obs)
                obs_td = agent.obs_dict_to_tensordict(obs)
                model_outs = agent.model(obs_td)
                actions = model_outs.get("mean_action", model_outs.get("action"))
                obs, rewards, dones, terminated, extras = env.step(actions)
                
                # Capture frame using Replicator
                frame = env.simulator.render(mode="rgb_array")
                if frame is not None and frame.size > 0:
                    writer.append_data(frame)
                
                done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            
            writer.close()
            print(f"Saved: {video_path}")
        
        print(f"Recording ended. Saved {video_count} videos to output/videos/")
        import os
        os._exit(0)
    elif args.full_eval:
        agent.evaluator.eval_count = 0
        agent.evaluator.evaluate()
    else:
        # collect_metrics=False for better performance (avoids GPU sync every frame)
        agent.evaluator.simple_test_policy(collect_metrics=False)


if __name__ == "__main__":
    main()
