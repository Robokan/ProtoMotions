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
# =============================================================================
# IMPORTANT: ENGINE-INDEPENDENT CODE
# =============================================================================
# This file MUST remain simulator/engine agnostic. Do NOT add any imports or
# code specific to IsaacLab, IsaacGym, or any other simulation engine.
#
# All engine-specific functionality should be abstracted through:
#   - env.simulator.*  (simulator abstraction layer)
#   - env.*            (environment abstraction)
#
# Examples of what NOT to do:
#   - import omni.kit.app
#   - import isaaclab.*
#   - import isaacgym
#
# If you need engine-specific behavior, add it to the appropriate simulator
# class in protomotions/simulator/<engine>/simulator.py
# =============================================================================
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
- **M**: Toggle markers on/off
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
        "--enable_cameras",
        action="store_true",
        default=False,
        help="Enable camera rendering (required for Camera sensor)",
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
        "--sequential-targets",
        action="store_true",
        default=False,
        help="Step through motion sequentially instead of random sampling. Use with --target-hz to set rate.",
    )
    parser.add_argument(
        "--target-hz",
        type=float,
        default=5.0,
        help="Target update rate in Hz when using --sequential-targets (default: 5.0 Hz)",
    )
    parser.add_argument(
        "--target-lookahead",
        type=float,
        default=1.0,
        help="How far ahead in motion (seconds) to place targets (default: 1.0s). Independent of update rate.",
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
        default="-2 -2 0",
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
    
    # Data collection options
    parser.add_argument(
        "--collect-data",
        action="store_true",
        default=False,
        help="Enable egocentric data collection mode",
    )
    parser.add_argument(
        "--collect-output-dir",
        type=str,
        default="/tmp/egocentric_data",
        help="Output directory for collected data",
    )
    parser.add_argument(
        "--collect-resolution",
        type=int,
        default=224,
        help="Camera resolution for data collection",
    )
    parser.add_argument(
        "--collect-max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to collect",
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

    # Create fabric config for inference (simplified, using SingleDeviceStrategy to avoid NCCL issues)
    # SingleDeviceStrategy works on all GPUs and avoids distributed training overhead for inference
    from lightning.fabric.strategies import SingleDeviceStrategy
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    fabric_config = FabricConfig(
        devices=1,
        num_nodes=1,
        strategy=SingleDeviceStrategy(device=device),
        loggers=[],  # No loggers needed for inference
        callbacks=[],  # No callbacks needed for inference
    )
    fabric: Fabric = Fabric(**fabric_config.to_dict())
    fabric.launch()

    # Setup IsaacLab simulation_app if using IsaacLab simulator
    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {
            "headless": args.headless,
            "device": str(fabric.device),
            "enable_cameras": getattr(args, 'enable_cameras', False),
        }
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
    
    # Set sequential target mode (step through motion at fixed rate instead of random sampling)
    env.inference_sequential_targets = args.sequential_targets
    env.inference_target_hz = args.target_hz
    env.inference_target_lookahead = args.target_lookahead
    if args.sequential_targets:
        print(f"Sequential targets enabled: {args.target_hz} Hz update rate, {args.target_lookahead}s lookahead")

    # Store scene args for loading after reset (need robot position)
    _scene_usd = args.scene_usd if args.simulator == "isaaclab" else None
    _scene_offset = [float(x) for x in args.scene_offset.split()] if args.scene_offset else [0, 0, 0]
    if len(_scene_offset) != 3:
        _scene_offset = [0.0, 0.0, 0.0]
    if args.target_update_interval > 1 and not args.sequential_targets:
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

    # Load custom scene centered on robot position (via simulator API)
    if _scene_usd:
        # Load the scene using simulator's scene loading API
        env.simulator.load_scene(_scene_usd, tuple(_scene_offset))

        # Set up scene randomization if requested
        randomize_fn = None
        if args.randomize_lights or args.randomize_objects:
            pos_range = tuple(float(x) for x in args.object_pos_range.split())
            rot_range = tuple(float(x) for x in args.object_rot_range.split())
            randomize_fn = env.simulator.setup_scene_randomization(
                randomize_lights=args.randomize_lights,
                randomize_objects=args.randomize_objects,
                object_pos_range=pos_range,
                object_rot_range=rot_range,
                hide_objects_prob=args.hide_objects_prob,
            )

        # Wrap env.reset to update scene position and apply randomization
        _original_reset = env.reset

        def _reset_with_scene_update(env_ids=None, **kwargs):
            result = _original_reset(env_ids, **kwargs)
            # Only update scene if env 0 is being reset
            env0_reset = (
                env_ids is None
                or (hasattr(env_ids, "__len__") and 0 in env_ids)
                or (hasattr(env_ids, "numel") and (env_ids == 0).any())
            )
            if env0_reset:
                robot_state = env.simulator._get_simulator_root_state(env_ids=None)
                robot_pos = robot_state.root_pos[0].cpu().numpy()
                env.simulator.update_scene_position(robot_pos)
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
    elif args.collect_data:
        # Egocentric data collection mode (via simulator API)
        import os as os_module
        from PIL import Image

        print("Starting egocentric data collection...")
        print(f"Output: {args.collect_output_dir}")
        print(f"Resolution: {args.collect_resolution}x{args.collect_resolution}")

        # Compute target frame counts from motion library (full length at 50Hz)
        target_frame_counts = {}
        sim_dt = env.simulator.dt  # Usually 0.02 (50Hz)
        for i in range(env.motion_lib.num_motions()):
            motion_length = env.motion_lib.motion_lengths[i].item()
            # Full motion length at simulation rate
            target_frame_counts[i] = int(motion_length / sim_dt)
        print(f"Using full motion lengths at {1.0/sim_dt:.0f}Hz")
        print(f"  {len(target_frame_counts)} motions, {min(target_frame_counts.values())}-{max(target_frame_counts.values())} frames each")

        # Set up egocentric camera via simulator API
        resolution = (args.collect_resolution, args.collect_resolution)
        env.simulator.setup_egocentric_camera(
            camera_name="EgoCamera", resolution=resolution
        )

        # Hide markers (red target spheres) for clean video
        env.simulator._show_markers = False
        print("Markers hidden for data collection")

        # Set up lighting randomization if enabled (via simulator API)
        randomize_fn = None
        if args.randomize_lights:
            randomize_fn = env.simulator.setup_scene_randomization(
                randomize_lights=True,
            )
            print("Lighting randomization enabled")

        # Create output directory
        os_module.makedirs(args.collect_output_dir, exist_ok=True)

        # Run simulation loop - sequential collection with exact frame counts
        agent.eval()
        import numpy as np
        max_episodes = args.collect_max_episodes or len(target_frame_counts) or 200
        
        print(f"Collecting {max_episodes} episodes sequentially...")
        print("Press Q or close window to stop")

        try:
            for episode_idx in range(max_episodes):
                if not env.simulator.is_simulation_running():
                    break
                
                motion_idx = episode_idx  # Sequential: episode 0 = motion 0, etc.
                target_frames = target_frame_counts.get(motion_idx, 100)
                
                # Setup episode directory
                episode_dir = os_module.path.join(
                    args.collect_output_dir, f"episode_{episode_idx + 1:05d}"
                )
                os_module.makedirs(
                    os_module.path.join(episode_dir, "rgb"), exist_ok=True
                )
                
                print(f"Episode {episode_idx + 1}: motion {motion_idx} "
                      f"({target_frames} frames)...")
                
                # Force this specific motion
                all_env_ids = torch.arange(
                    env.num_envs, device=env.device, dtype=torch.long
                )
                forced_ids = torch.full(
                    (env.num_envs,), motion_idx, 
                    device=env.device, dtype=torch.long
                )
                env.motion_manager.sample_motions(all_env_ids, new_motion_ids=forced_ids)
                env.motion_manager.motion_times[:] = 0.0
                
                # Reset with forced motion
                obs, _ = env.reset(all_env_ids, disable_motion_resample=True)
                obs = agent.add_agent_info_to_obs(obs)
                
                if not env.simulator.is_simulation_running():
                    break
                
                if randomize_fn and not args.randomize_per_frame:
                    randomize_fn()
                
                # Collect exactly target_frames for this episode
                frames, poses, imu_data = [], [], []
                prev_lin_vel = None
                
                for frame_idx in range(target_frames):
                    if not env.simulator.is_simulation_running():
                        break
                    
                    # ===== Capture state BEFORE action =====
                    bodies_state = env.simulator.get_bodies_state(env_ids=None)
                    dof_state = env.simulator.get_dof_state(env_ids=None)
                    
                    # Capture egocentric frame
                    rgb = env.simulator.capture_egocentric_frame()
                    if rgb is not None:
                        path = os_module.path.join(
                            episode_dir, "rgb", f"frame_{frame_idx:05d}.png"
                        )
                        Image.fromarray(rgb).save(path)
                        frames.append(f"rgb/frame_{frame_idx:05d}.png")
                    
                    # ===== Get motion library reference (ground truth) =====
                    ref_state = env.motion_lib.get_motion_state(
                        env.motion_manager.motion_ids,
                        env.motion_manager.motion_times,
                    )
                    ref_dof = ref_state.dof_pos[0].detach().cpu().numpy()
                    
                    # ===== Compute action =====
                    obs_td = agent.obs_dict_to_tensordict(obs)
                    model_outs = agent.model(obs_td)
                    actions = model_outs.get("mean_action", model_outs.get("action"))
                    
                    # ===== Step physics =====
                    obs, rewards, dones, terminated, extras = env.step(actions)
                    obs = agent.add_agent_info_to_obs(obs)
                    
                    if randomize_fn and args.randomize_per_frame:
                        randomize_fn()
                    
                    # ===== Save pose data =====
                    # dof_positions: motion library reference (for GR00T action)
                    # dof_positions_actual: where robot ended up (for GR00T state)
                    # root_rotation: pelvis orientation from motion lib (for FK)
                    ref_root_rot = ref_state.rigid_body_rot[0, 0].detach().cpu().numpy()  # (4,) XYZW
                    poses.append({
                        "body_positions": bodies_state.rigid_body_pos[0]
                            .cpu().numpy().tolist(),
                        "body_rotations": bodies_state.rigid_body_rot[0]
                            .cpu().numpy().tolist(),
                        "dof_positions": ref_dof.tolist(),
                        "dof_positions_actual": dof_state.dof_pos[0]
                            .cpu().numpy().tolist(),
                        "root_rotation": ref_root_rot.tolist(),  # Pelvis rotation for FK
                        "frame_idx": frame_idx,
                    })
                    
                    # ===== Save IMU data =====
                    pelvis_quat = bodies_state.rigid_body_rot[0, 0].cpu().numpy()
                    ang_vel = bodies_state.rigid_body_ang_vel[0, 0].cpu().numpy()
                    lin_vel = bodies_state.rigid_body_vel[0, 0].cpu().numpy()
                    
                    if prev_lin_vel is not None:
                        lin_accel = (lin_vel - prev_lin_vel) / env.simulator.dt
                    else:
                        lin_accel = np.array([0.0, 0.0, 9.81])
                    prev_lin_vel = lin_vel.copy()
                    
                    imu_data.append({
                        'orientation': pelvis_quat.tolist(),
                        'angular_velocity': ang_vel.tolist(),
                        'linear_acceleration': lin_accel.tolist(),
                        'frame_idx': frame_idx,
                    })
                
                # Save episode data
                episode_data = {
                    "episode_idx": episode_idx + 1,
                    "motion_idx": motion_idx,
                    "num_frames": len(frames),
                    "frames": frames,
                    "poses": poses,
                    "imu": imu_data,
                }
                torch.save(episode_data, 
                           os_module.path.join(episode_dir, "episode_data.pt"))
                print(f"  Saved {len(frames)} frames")

        except KeyboardInterrupt:
            print("\nInterrupted")

        print(f"\nDone! Saved to {args.collect_output_dir}")
        
        # Force exit like simple_test_policy
        import os as os_exit
        os_exit._exit(0)
    elif args.full_eval:
        agent.evaluator.eval_count = 0
        agent.evaluator.evaluate()
    else:
        # collect_metrics=False for better performance (avoids GPU sync every frame)
        agent.evaluator.simple_test_policy(collect_metrics=False)


if __name__ == "__main__":
    main()
