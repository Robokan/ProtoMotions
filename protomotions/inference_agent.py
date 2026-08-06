# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
- **W/A/S/D**: Move target when running with ``--command-source target=keyboard``

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
        "--experiment-path",
        type=str,
        default=None,
        help="Experiment .py file whose apply_inference_overrides hook to "
        "apply on top of the frozen training configs",
    )
    parser.add_argument(
        "--overlay-character",
        nargs="+",
        default=None,
        help="Rigged character USD(s) to skin fighters with (UsdSkel). With "
        "multiple paths they cycle across envs (e.g. two fighters get two "
        "different characters). IsaacLab viewer only.",
    )
    parser.add_argument(
        "--overlay-skeleton",
        default="cc",
        choices=["cc", "ue", "identity"],
        help="Overlay rig family: 'cc' (Reallusion), 'ue' (Epic UE5, e.g. "
        "red samurai), or 'identity' (fbx2robot creatures, whose character "
        "USD shares the robot's own skeleton). Auto-selected for raptor/tiger.",
    )
    parser.add_argument(
        "--overlay-fists",
        action="store_true",
        help="Curl overlay characters' fingers into fists.",
    )
    parser.add_argument(
        "--overlay-hide-robot",
        action="store_true",
        help="Hide the robots' capsule bodies (show only the skinned "
        "characters).",
    )
    parser.add_argument(
        "--overlay-ambient",
        type=float,
        default=50.0,
        help="Ambient fill lighting intensity when overlays are active: a "
        "dome light plus a soft opposite-side fill so shadows aren't pure "
        "black (default 50; 0 disables).",
    )
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
        "--env-spacing",
        "--env_spacing",
        type=float,
        default=None,
        help=(
            "Max random spawn distance in meters from the walkable-terrain "
            "center (keeps random sampling, but clamps how far apart robots "
            "can be). Also sets IsaacLab cloner spacing."
        ),
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
        "--command-source",
        nargs="*",
        default=[],
        help=(
            "Override task command sources for inference, e.g. "
            "target=keyboard. A bare value applies to the single target "
            "control component."
        ),
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
#     """Example for quick inference-only config experiments.
#
#     Keep this commented out unless you are doing a local smoke test and need a
#     richer temporary override than the CLI can express. For reusable behavior,
#     put the override in an experiment file's apply_inference_overrides hook.
#     """
#     from protomotions.simulator.base_simulator.config import (
#         # FrictionDomainRandomizationConfig,
#         CenterOfMassDomainRandomizationConfig,
#         DomainRandomizationConfig,
#     )
#
#     # env_cfg.terrain.sim_config.static_friction = 0.01
#     # env_cfg.terrain.sim_config.dynamic_friction = 0.01
#
#     simulator_cfg.domain_randomization = DomainRandomizationConfig(
#         # Uncomment to enable action noise and friction randomization:
#         # action_noise=ActionNoiseDomainRandomizationConfig(
#         #     action_noise_range=(-0.01, 0.01),
#         #     dof_names=[".*"],
#         #     dof_indices=None,
#         # ),
#         # friction=FrictionDomainRandomizationConfig(
#         #     num_buckets=64,
#         #     static_friction_range=(0.0, 1.0),
#         #     dynamic_friction_range=(0.0, 1.0),
#         #     restitution_range=(0.0, 0.0),
#         #     body_names=[".*"],
#         #     body_indices=None,
#         # ),
#     )
#     log.info("Enabled domain randomization for testing")
#

def apply_command_source_overrides(env_config, command_source_specs):
    """Apply inference-only task command source overrides."""
    if len(command_source_specs) == 0:
        return

    from protomotions.envs.control.target_control import (
        KeyboardTargetCommandSourceConfig,
        RandomTargetCommandSourceConfig,
        TargetControlConfig,
    )

    control_components = env_config.control_components
    for spec in command_source_specs:
        if "=" in spec:
            component_name, source_name = spec.split("=", 1)
        else:
            target_components = [
                name
                for name, component in control_components.items()
                if isinstance(component, TargetControlConfig)
            ]
            if len(target_components) != 1:
                raise ValueError(
                    "Bare --command-source values require exactly one "
                    "TargetControlConfig component"
                )
            component_name = target_components[0]
            source_name = spec

        if component_name not in control_components:
            raise ValueError(
                f"Cannot override command source for unknown control component "
                f"'{component_name}'"
            )

        component_config = control_components[component_name]
        if not isinstance(component_config, TargetControlConfig):
            raise ValueError(
                f"Command source override '{component_name}={source_name}' only "
                "supports TargetControlConfig components"
            )

        source_name = source_name.lower()
        if source_name in ("keyboard", "manual", "user", "user-control"):
            component_config.command_source = KeyboardTargetCommandSourceConfig()
        elif source_name in ("random", "training"):
            component_config.command_source = RandomTargetCommandSourceConfig()
        else:
            raise ValueError(
                f"Unsupported command source '{source_name}' for component "
                f"'{component_name}'"
            )


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

    # fbx2robot creatures ship a character USD built on the robot's own
    # skeleton, so skin them by default (same as motion_libs_visualizer):
    # key 5 toggles the robot body, key 6 the mesh.
    _creature = next(
        (c for c in ("raptor", "tiger")
         if c in type(robot_config).__name__.lower()), None)
    if _creature:
        if not args.overlay_character:
            _cusd = Path(f"protomotions/data/assets/overlay/{_creature}.usd")
            if _cusd.exists():
                args.overlay_character = [str(_cusd)]
                log.info("overlay: auto-skinning %s (keys 5=robot, 6=mesh)",
                         _creature)
            else:
                log.warning("overlay: %s not found, capsules only", _cusd)
        # A creature's character USD is built on the robot's own skeleton,
        # so identity is right whether the USD came from the auto-default or
        # from explicit --overlay-character paths (e.g. passing the purple
        # and black raptors to colour the two fighters differently).
        if args.overlay_skeleton == "cc":      # i.e. left at the default
            args.overlay_skeleton = "identity"
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
    # # Temporary: Enable domain randomization for local inference testing.
    # # Prefer --overrides or apply_inference_overrides for reusable changes.
    # tmp_enable_domain_randomization(robot_config, simulator_config, env_config)

    # from protomotions.robot_configs.base import ControlType
    # robot_config.control.control_type = ControlType.PROPORTIONAL

    # Apply CLI runtime overrides
    if args.num_envs is not None:
        log.info(f"CLI override: num_envs = {args.num_envs}")
        simulator_config.num_envs = args.num_envs

    if args.env_spacing is not None:
        log.info(f"CLI override: env_spacing = {args.env_spacing}")
        simulator_config.env_spacing = args.env_spacing
        env_config.env_spacing = args.env_spacing

    if args.motion_file is not None:
        log.info(f"CLI override: motion_file = {args.motion_file}")
        motion_lib_config.motion_file = args.motion_file  # Always present

    if args.scenes_file is not None:
        # Normalise "None"/"null" strings to actual None (disable scenes)
        scenes_file = (
            None if args.scenes_file.lower() in ("none", "null") else args.scenes_file
        )
        log.info(f"CLI override: scenes_file = {scenes_file}")
        scene_lib_config.scene_file = scenes_file
        if scenes_file is None:
            scene_lib_config.asset_root = None
        # Recompute asset_root from the new scene file path (experiment's
        # asset_root may point to a different machine, e.g. lustre vs local)
        elif scene_lib_config.asset_root is not None:
            import os

            scene_lib_config.asset_root = os.path.dirname(
                os.path.dirname(os.path.abspath(scenes_file))
            )

    if args.headless is not None:
        log.info(f"CLI override: headless = {args.headless}")
        simulator_config.headless = args.headless

    # Apply the experiment's apply_inference_overrides hook (documented in the
    # precedence list above but historically never invoked: frozen configs
    # carry the hook's state from *training time*, so repo-side changes to the
    # hook never reached existing checkpoints). --experiment-path names the
    # experiment file; if omitted, checkpoints trained before the hook was
    # wired in keep their frozen behavior.
    if args.experiment_path is not None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "experiment_module", args.experiment_path
        )
        experiment = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(experiment)
        if hasattr(experiment, "apply_inference_overrides"):
            log.info(
                f"Applying apply_inference_overrides from {args.experiment_path}"
            )
            experiment.apply_inference_overrides(
                robot_config,
                simulator_config,
                env_config,
                agent_config,
                terrain_config,
                motion_lib_config,
                scene_lib_config,
                args,
            )

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

    if args.command_source:
        log.info(f"CLI override: command_source = {args.command_source}")
        apply_command_source_overrides(env_config, args.command_source)

    # Create fabric config for inference (simplified)
    # MuJoCo is CPU-only, so force CPU accelerator
    accelerator = "cpu" if args.simulator == "mujoco" else "gpu"
    fabric_config = FabricConfig(
        accelerator=accelerator,
        devices=1,
        num_nodes=1,
        loggers=[],  # No loggers needed for inference
        callbacks=[],  # No callbacks needed for inference
    )
    fabric: Fabric = Fabric(**fabric_config.as_kwargs())
    fabric.launch()

    # Setup IsaacLab simulation_app if using IsaacLab simulator
    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {"headless": args.headless, "device": str(fabric.device)}
        app_launcher = AppLauncher(app_launcher_flags)
        simulator_extra_params["simulation_app"] = app_launcher.app

    # Convert friction for simulator compatibility
    from protomotions.simulator.base_simulator.utils import (
        convert_friction_for_simulator,
    )

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    # Create components
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

    # Demo/viewer: skip per-step reward computation and raw-state logging
    # (only needed for training) so the inference frame rate isn't
    # bottlenecked by training-only work.
    env.inference_mode = True

    # Skinned character overlays: one per env, synced from live sim state
    # after every simulator step (get_bodies_state returns common xyzw —
    # the layout SkinnedOverlay expects). Battle exhibitions: pass two
    # character USDs to skin the two fighters differently.
    if args.overlay_character and args.simulator == "isaaclab" \
            and not args.headless:
        try:
            import numpy as np
            import omni.usd
            from protomotions.simulator.isaaclab.overlay import SkinnedOverlay
            from protomotions.simulator.isaaclab.overlay_map import (
                SOMA23_TO_CC, SOMA23_REST_REL, SOMA23_TPOSE_POS,
                SOMA23_PARENT, SOMA23_TO_UE, UE_REST_REL, SOMA23_PARENT_UE,
            )

            # Absolute character paths: a relative layer reference makes the
            # asset's own relative texture paths (@./textures/...@) resolve
            # against the wrong anchor -> untextured (black) characters.
            # --overlay-character takes a PATH, not a character name. A bare
            # name like "raptor" silently resolves to a non-existent file in
            # the cwd, suppresses the creature auto-skin, and the viewer comes
            # up with no mesh and no explanation. Fail loudly instead.
            _missing = [p for p in args.overlay_character
                        if not Path(p).expanduser().exists()]
            if _missing:
                _hint = ", ".join(
                    str(q) for q in sorted(
                        Path("protomotions/data/assets/overlay").glob("*.usd")))
                raise FileNotFoundError(
                    f"--overlay-character path(s) not found: {_missing}. "
                    f"Pass a path to a .usd, not a character name. "
                    f"Available: {_hint}")
            args.overlay_character = [
                str(Path(p).expanduser().resolve())
                for p in args.overlay_character
            ]
            _bn = list(robot_config.kinematic_info.body_names)
            if args.overlay_skeleton == "identity":
                # fbx2robot creatures (raptor/tiger): the character USD IS
                # the robot's skeleton, so the joint map is the identity and
                # the parent chain comes from kinematic_info.
                _ki = robot_config.kinematic_info
                _map = {b: b for b in _bn}
                _rel = None
                _par = {
                    b: (_bn[pi] if pi >= 0 else None)
                    for b, pi in zip(_bn, list(_ki.parent_indices))
                }
                _tpose = None
            elif args.overlay_skeleton == "ue":
                _map, _rel, _par = SOMA23_TO_UE, UE_REST_REL, SOMA23_PARENT_UE
                _tpose = SOMA23_TPOSE_POS
            else:
                _map, _rel, _par = SOMA23_TO_CC, SOMA23_REST_REL, SOMA23_PARENT
                _tpose = SOMA23_TPOSE_POS
            _stage = omni.usd.get_context().get_stage()
            if args.overlay_ambient > 0:
                from pxr import UsdLux, Gf as _Gf, UsdGeom as _UsdGeom
                _dome = UsdLux.DomeLight.Define(
                    _stage, "/World/overlayDomeLight")
                _dome.GetIntensityAttr().Set(args.overlay_ambient)
                # Soft fill from the opposite side of the sun (-Y, high
                # angle) at ~40% key intensity so shaded sides read.
                _fill = UsdLux.DistantLight.Define(
                    _stage, "/World/overlayFillLight")
                _fill.GetIntensityAttr().Set(0.4 * args.overlay_ambient)
                _fill.GetAngleAttr().Set(5.0)
                _xf = _UsdGeom.Xformable(_fill.GetPrim())
                _xf.AddRotateXYZOp().Set(_Gf.Vec3f(-130.0, 0.0, 0.0))
                log.info("overlay: dome %.0f + fill %.0f",
                         args.overlay_ambient, 0.4 * args.overlay_ambient)
                # Ring lights: sphere lights around the arena aimed inward,
                # like arena floods — kills the hard single-sun shadows on
                # the fighters from every camera angle.
                _nring = 4

                def _ring_centers(st):
                    # Battle envs know their arena centers exactly (one
                    # arena per match, shared by the env pair); otherwise
                    # every env gets its own ring at its root position.
                    bc = getattr(env, "battle_control", None)
                    z = float(st.rigid_body_pos[:, 0, 2].mean())
                    if bc is not None and hasattr(bc, "arena_centers"):
                        cs = bc.arena_centers.unique(dim=0).cpu().numpy()
                        return [(float(c[0]), float(c[1]), z) for c in cs]
                    roots = st.rigid_body_pos[:, 0, :].cpu().numpy()
                    return [(float(r[0]), float(r[1]), float(r[2]))
                            for r in roots]

                def _place_ring(st):
                    # Fighters' true positions exist only after the first
                    # step (init-time state predates terrain placement).
                    centers = _ring_centers(st)
                    for ci, c in enumerate(centers):
                        for k in range(_nring):
                            _sl = UsdLux.SphereLight.Define(
                                _stage,
                                f"/World/overlayRing{ci}Light{k}")
                            _sl.GetRadiusAttr().Set(1.0)
                            # Sphere falloff ~ (radius/dist)^2: a 1 m light
                            # ~10 m out needs ~1e6 intensity to rival the
                            # sun. 6000x ambient = 3M at the default 500.
                            _sl.GetIntensityAttr().Set(
                                6000.0 * args.overlay_ambient)
                            _sl.GetNormalizeAttr().Set(True)
                            ang = 2.0 * np.pi * k / _nring + 0.7854
                            _UsdGeom.Xformable(
                                _sl.GetPrim()).AddTranslateOp().Set(
                                _Gf.Vec3d(
                                    c[0] + 9.0 * np.cos(ang),
                                    c[1] + 9.0 * np.sin(ang),
                                    c[2] + 5.0))
                    print(f"[overlay] ring lights: {len(centers)} arena(s) "
                          f"x {_nring} lights", flush=True)
            _n = simulator_config.num_envs
            _overlays = []
            for i in range(_n):
                _overlays.append(SkinnedOverlay(
                    stage=_stage,
                    character_usd=args.overlay_character[
                        i % len(args.overlay_character)],
                    prim_root=f"/World/overlay{i}",
                    body_names=_bn,
                    body_rest_rot_wxyz=np.zeros((len(_bn), 4)),
                    joint_map=_map,
                    root_only=True,
                    drive_bodies=[
                        b for b in _map
                        if b not in ("Hips", "Hip", "Pelvis", "RigPelvis")
                    ],
                    rest_rel=_rel,
                    tpose_pos=_tpose,
                    body_parents=_par,
                    fists=args.overlay_fists,
                ))
                if args.overlay_hide_robot:
                    _overlays[-1].set_capsules_visible(
                        f"/World/envs/env_{i}/Robot", False)
            log.info("Skinned overlays active on %d envs", _n)

            # Hook the ACTUAL frame draw. IsaacLabSimulator._physics_step
            # calls self._sim.render() inside its decimation loop, and only
            # afterwards does step() call simulator.render() (which just
            # moves the camera). Syncing anywhere later wrote the skin one
            # frame behind the robot -- visible as the mesh trailing in fast
            # motion. _scene.update() also runs after that draw, so refresh
            # the read buffers here before sampling the pose.
            _ring_placed = [False]
            _sim_ctx = env.simulator._sim
            _orig_sim_render = _sim_ctx.render

            def _sim_render_with_overlays(*a, **kw):
                try:
                    env.simulator._scene.update(
                        dt=_sim_ctx.get_physics_dt())
                    st = env.simulator.get_bodies_state()
                    if not _ring_placed[0] and args.overlay_ambient > 0:
                        _place_ring(st)
                        _ring_placed[0] = True
                    for i, ov in enumerate(_overlays):
                        try:
                            ov.sync(st.rigid_body_pos[i], st.rigid_body_rot[i])
                        except Exception:
                            pass
                except Exception:
                    pass
                return _orig_sim_render(*a, **kw)

            _sim_ctx.render = _sim_render_with_overlays

            # A reset (R key) rewrites the robot pose outside the step/render
            # cycle, and a windowed Kit app redraws continuously — so the
            # viewport showed the teleported robot against a stale skin until
            # the next render. reset_envs' state read-back is already fresh
            # (verified: root AND limb transforms update immediately), so
            # sync right there.
            _orig_reset = env.simulator.reset_envs

            def _reset_with_overlays(*a, **kw):
                out = _orig_reset(*a, **kw)
                try:
                    st = env.simulator.get_bodies_state()
                    for i, ov in enumerate(_overlays):
                        try:
                            ov.sync(st.rigid_body_pos[i], st.rigid_body_rot[i])
                        except Exception:
                            pass
                except Exception:
                    pass
                return out

            env.simulator.reset_envs = _reset_with_overlays

            # Same viewer keys as motion_libs_visualizer: 5 toggles the
            # robot's collision/visual body, 6 toggles the skinned mesh.
            # One tracked flag per layer, applied identically to every env.
            # Reading each prim's own visibility and inverting it made the
            # envs drift out of phase the moment any two differed (and an
            # ancestor-hidden prim reports 'invisible' while its own attr is
            # unset, so the flip was a no-op for it) -- pressing 5 with
            # several raptors then hid some and showed others.
            _hidden = {"robot": False, "mesh": False}

            def _set_layer(kind, paths):
                from pxr import UsdGeom as _UG
                _hidden[kind] = not _hidden[kind]
                token = "invisible" if _hidden[kind] else "inherited"
                missing = 0
                for path in paths:
                    prim = _stage.GetPrimAtPath(path)
                    if not prim or not prim.IsValid():
                        missing += 1
                        continue
                    _UG.Imageable(prim).GetVisibilityAttr().Set(token)
                state = "hidden" if _hidden[kind] else "shown"
                note = f" ({missing} prim(s) missing)" if missing else ""
                print(f"[viewer] {kind} {state} on {len(paths) - missing} "
                      f"env(s){note}", flush=True)

            def _toggle_robots():
                _set_layer("robot",
                           [f"/World/envs/env_{i}/Robot" for i in range(_n)])

            def _toggle_meshes():
                _set_layer("mesh", [f"/World/overlay{i}" for i in range(_n)])

            try:
                env.simulator._register_custom_user_interface_keys(
                    {"5": _toggle_robots, "6": _toggle_meshes})
                log.info("overlay keys: 5 = robot on/off, 6 = mesh on/off")
            except Exception as exc:  # noqa: BLE001
                log.warning("could not register overlay keys: %s", exc)
        except Exception:
            import traceback
            log.error("overlay setup failed — continuing without skins:")
            traceback.print_exc()

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
    agent.load(args.checkpoint, load_env=False, load_training_state=False)
    headless = getattr(env.simulator, "headless", True)
    ui = getattr(env.simulator, "user_interface", None)
    if not headless and ui is not None:
        help_text = ui.help_text()
        if help_text:
            log.info("Viewer keybinds:\n%s", help_text)

    try:
        if args.full_eval:
            agent.evaluator.eval_count = 0
            evaluation_log, evaluated_score, num_eval_items = (
                agent.evaluator.evaluate()
            )

            # Print evaluation metrics
            print("\n" + "=" * 60)
            print("EVALUATION RESULTS")
            print("=" * 60)
            for key, value in sorted(evaluation_log.items()):
                print(f"  {key}: {value:.6f}")
            print(f"  Items Evaluated: {num_eval_items}")
            print("=" * 60)
            if evaluated_score is not None:
                print(f"  Overall Score: {evaluated_score:.6f}")
            print("=" * 60 + "\n")
        else:
            agent.evaluator.simple_test_policy(collect_metrics=True)
    finally:
        # Ensure simulator viewer is properly closed (prevents hangs)
        if hasattr(env.simulator, "shutdown"):
            env.simulator.shutdown()


if __name__ == "__main__":
    main()
