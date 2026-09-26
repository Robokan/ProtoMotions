# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Chase the red ball: a task for the trained MaskedMimic policy.

A red ball is thrown 2-8 m away. The dog's job is to get its torso within two
feet of it (0.6096 m, measured in the ground plane -- the ball sits ON the
floor while the torso rides ~0.34 m above it, so a 3-D distance would spend
half the budget on height the robot cannot remove). Catch it and the ball is
immediately re-thrown somewhere else, so the chase never ends.

Nothing is trained here. The ball's position becomes MaskedMimic base-link
targets DIRECTLY -- "put my torso here, by t+dt_k" along the line to the ball
-- and the EXISTING MaskedMimic checkpoint walks the dog there. So this runs
today:

    python protomotions/inference_agent.py \\
        --checkpoint results/go2_masked_mimic_v4/last.ckpt \\
        --experiment-path examples/experiments/masked_mimic/chase_ball.py \\
        --simulator isaaclab --physics physx --num-envs 4

No velocity command anywhere. MaskedMimic is natively conditioned on
poses-at-times, so turning a goal into a velocity and re-integrating it back
into positions is a lossy round trip through a representation that cannot
express "two feet from the ball" at all. Going straight to positions also
removes every pursuit gain: the waypoint ladder saturates at the aim point,
so the approach slows by construction rather than by tuning.

That matters for what this is FOR. Positions-at-times is exactly the format a
VLA emits, so running this produces (what the robot sees, where the ball is)
-> (MaskedMimic targets) pairs -- the supervision a VLA needs to learn to emit
those targets itself. Swap this component for the VLA later and nothing else
in the task changes.

The observation and reward for that learned version are already wired
(target_obs_factory / target_reward_factory read ctx.target), they are simply
unused while a script is doing the driving.
"""

import argparse
import os

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.envs.base_env.config import EnvConfig


_DEFAULTS = {
    "success_radius": 0.6096,   # two feet
    "throw_min": 2.0,
    "throw_max": 8.0,
    "report_every": 250,
    "moving_ball": False,
    "ball_speed_min": 0.5,
    "ball_speed_max": 2.0,
    "ball_turn_mean_sec": 5.0,
    "unprivileged": False,
    # Roughly the go2's forward camera. Total cone width, yaw only.
    "fov_deg": 120.0,
    "sight_range": 0.0,
    "search_turn_deg": 180.0,
    "search_range": 1.5,
    # The VLA will see frames at this rate; the search sweeps slowly enough
    # that the scene moves only search_deg_per_frame between two of them.
    "vla_hz": 10.0,
    "search_deg_per_frame": 10.0,
    "horizon_sec": None,
    "camera": False,
    "camera_res": 224,
    "camera_probe_every": 50,
    "record": False,
    "record_dir": "output/datasets/go2_chase",
    "record_fps": 10.0,
    "record_episodes": 40,
    "record_episode_steps": 200,
    "record_task": "chase the red ball",
}


def _arg(args, name):
    return getattr(args, name, _DEFAULTS[name])


def _steering():
    """The velocity-command harness this task steers through."""
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "steering.py")
    spec = importlib.util.spec_from_file_location("masked_mimic_steering", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    _steering().additional_experiment_arguments(parser)
    parser.add_argument(
        "--success-radius", type=float, default=_DEFAULTS["success_radius"],
        help="Torso-to-ball distance that counts as a catch (m, ground plane).")
    parser.add_argument(
        "--throw-min", type=float, default=_DEFAULTS["throw_min"],
        help="Closest the ball is ever thrown (m).")
    parser.add_argument(
        "--throw-max", type=float, default=_DEFAULTS["throw_max"],
        help="Furthest the ball is ever thrown (m).")
    parser.add_argument(
        "--moving-ball", action="store_true", default=_DEFAULTS["moving_ball"],
        help="The ball moves: random direction and speed per throw, occasional "
             "direction changes. The dog aims at where the ball WILL be when "
             "its deadline arrives, not where it is.")
    parser.add_argument(
        "--ball-speed-min", type=float, default=_DEFAULTS["ball_speed_min"],
        help="Slowest ball speed (m/s) when --moving-ball.")
    parser.add_argument(
        "--ball-speed-max", type=float, default=_DEFAULTS["ball_speed_max"],
        help="Fastest ball speed (m/s) when --moving-ball.")
    parser.add_argument(
        "--ball-turn-mean-sec", type=float, default=_DEFAULTS["ball_turn_mean_sec"],
        help="Mean seconds between random direction changes of a moving ball "
             "(Poisson). Each change makes the dog re-plan. 0 disables.")
    parser.add_argument(
        "--unprivileged", action="store_true", default=_DEFAULTS["unprivileged"],
        help="Take the ball's true position away: the dog only knows where it "
             "is while it is inside the forward camera cone (--fov-deg). Out "
             "of frame it believes the ball is behind it and turns to look. "
             "This is the observation a VLA will have.")
    parser.add_argument(
        "--fov-deg", type=float, default=_DEFAULTS["fov_deg"],
        help="Total width of the forward cone, degrees (--unprivileged only).")
    parser.add_argument(
        "--sight-range", type=float, default=_DEFAULTS["sight_range"],
        help="How far the ball can be recognised (m). 0 = unlimited.")
    parser.add_argument(
        "--search-turn-deg", type=float, default=_DEFAULTS["search_turn_deg"],
        help="Where the dog believes an unseen ball is, in degrees off its "
             "current heading. 180 = directly behind. Below 180 the search "
             "sweeps consistently toward the side the ball was last seen.")
    parser.add_argument(
        "--search-range", type=float, default=_DEFAULTS["search_range"],
        help="How far away that believed ball sits (m). Small keeps the "
             "search leg mostly a pivot.")
    parser.add_argument(
        "--vla-hz", type=float, default=_DEFAULTS["vla_hz"],
        help="Frame rate of whatever will be doing the seeing. Sets how slowly "
             "the dog sweeps while searching, so the ball cannot cross the "
             "camera between two frames.")
    parser.add_argument(
        "--search-deg-per-frame", type=float, default=_DEFAULTS["search_deg_per_frame"],
        help="How far the view may rotate between two of those frames. Search "
             "yaw rate = this x --vla-hz, capped at the robot's top yaw.")
    parser.add_argument(
        "--camera", action="store_true", default=_DEFAULTS["camera"],
        help="Mount the go2's forward camera and render it. This is the image "
             "a VLA would be shown; --camera-probe-every saves some to look "
             "at. Costs render time, so it is off by default.")
    parser.add_argument(
        "--camera-res", type=int, default=_DEFAULTS["camera_res"],
        help="Square camera resolution in pixels.")
    parser.add_argument(
        "--camera-probe-every", type=int, default=_DEFAULTS["camera_probe_every"],
        help="Save a frame from env 0 every N control steps (0 = never). The "
             "probe stops after a dozen frames.")
    parser.add_argument(
        "--record", action="store_true", default=_DEFAULTS["record"],
        help="Write a LeRobot dataset of (camera, proprioception) -> "
             "(egocentric target, seconds left) pairs. Implies --camera. Run "
             "headless: with a viewer open the conditioned-target markers "
             "render into the camera and the label ends up drawn on the "
             "image.")
    parser.add_argument(
        "--record-dir", type=str, default=_DEFAULTS["record_dir"],
        help="Where the dataset goes.")
    parser.add_argument(
        "--record-fps", type=float, default=_DEFAULTS["record_fps"],
        help="Sampling rate, i.e. the rate the student will run at. Rounded "
             "to a divisor of the control rate.")
    parser.add_argument(
        "--record-episodes", type=int, default=_DEFAULTS["record_episodes"],
        help="Stop after this many episodes.")
    parser.add_argument(
        "--record-episode-steps", type=int,
        default=_DEFAULTS["record_episode_steps"],
        help="Frames per episode. The chase never ends, so this is a "
             "bookkeeping unit; a reset cuts an episode short.")
    parser.add_argument(
        "--record-task", type=str, default=_DEFAULTS["record_task"],
        help="The language prompt stored with every frame.")
    parser.add_argument(
        "--horizon-sec", type=float, default=None,
        help="Lead time of the farthest conditioned target, i.e. how long the "
             "dog is given to reach the ball. Shorter = more urgent. There is "
             "no speed cap; this is the only thing that sets the pace.")


def terrain_config(args):
    return _steering().terrain_config(args)


def scene_lib_config(args):
    return _steering().scene_lib_config(args)


def motion_lib_config(args):
    return _steering().motion_lib_config(args)


def agent_config(robot_config: RobotConfig, env_config: EnvConfig, args):
    return _steering().agent_config(robot_config, env_config, args)


def _install_chase(cfg: EnvConfig, args: argparse.Namespace) -> None:
    """Put the ball in the scene and point the pursuit controller at it."""
    from protomotions.envs.control.ball_chase import (
        MaskedMimicGoalControlConfig,
        ball_chase_target_config,
    )
    from protomotions.envs.component_factories import (
        target_obs_factory,
        target_reward_factory,
    )

    # Reuse the steering harness only for the parts that are about MaskedMimic
    # rather than about velocity: conditioned-body layout, target height,
    # episode length, dropped clip-tracking terminations.
    steering = _steering()
    steering._install_steering(cfg, args)
    trained = cfg.control_components["masked_mimic"]

    # Order matters: the ball moves first, then the targets that lead to it are
    # built. ControlManager steps components in dict order.
    cfg.control_components = {
        "ball": ball_chase_target_config(
            success_radius=_arg(args, "success_radius"),
            throw_min=_arg(args, "throw_min"),
            throw_max=_arg(args, "throw_max"),
            moving=_arg(args, "moving_ball"),
            ball_speed_min=_arg(args, "ball_speed_min"),
            ball_speed_max=_arg(args, "ball_speed_max"),
            ball_turn_mean_sec=_arg(args, "ball_turn_mean_sec"),
        ),
        "masked_mimic": MaskedMimicGoalControlConfig(
            num_masked_future_steps=trained.num_masked_future_steps,
            future_steps=trained.future_steps,
            bootstrap_on_episode_end=trained.bootstrap_on_episode_end,
            horizon_sec=(_arg(args, "horizon_sec") or trained.horizon_sec),
            height_mode=trained.height_mode,
            condition_rotation=trained.condition_rotation,
            report_every_steps=trained.report_every_steps,
            target_component="ball",
            # Aim AT the ball; two feet is the success TEST, not the
            # destination. Aiming at the boundary parks the dog outside it.
            stop_distance=0.0,
            # 0 = privileged: the ball's position is handed over even when it
            # is behind the robot. --unprivileged replaces that with sight.
            fov_deg=(
                _arg(args, "fov_deg") if _arg(args, "unprivileged") else 0.0
            ),
            sight_range_m=_arg(args, "sight_range"),
            search_turn_deg=_arg(args, "search_turn_deg"),
            search_range_m=_arg(args, "search_range"),
            vla_hz=_arg(args, "vla_hz"),
            search_deg_per_frame=_arg(args, "search_deg_per_frame"),
        ),
    }

    from protomotions.envs.control.speed_probe import RootSpeedProbeConfig

    cfg.control_components["speed_probe"] = RootSpeedProbeConfig(label="chase")

    if _arg(args, "record"):
        from protomotions.envs.control.lerobot_recorder import LeRobotRecorderConfig

        cfg.control_components["recorder"] = LeRobotRecorderConfig(
            root=_arg(args, "record_dir"),
            fps=_arg(args, "record_fps"),
            episode_steps=_arg(args, "record_episode_steps"),
            max_episodes=_arg(args, "record_episodes"),
            task=_arg(args, "record_task"),
            robot_type=getattr(args, "robot_name", "go2"),
        )

    if _arg(args, "camera") and _arg(args, "camera_probe_every") > 0:
        from protomotions.envs.control.camera_probe import CameraProbeConfig

        cfg.control_components["camera_probe"] = CameraProbeConfig(
            every_steps=_arg(args, "camera_probe_every")
        )

    # Wired but unused while this component does the driving: these are what a
    # learned high-level policy (or a VLA fine-tune) would train against. The
    # steering reward is dropped with the steering command it scored.
    cfg.observation_components["target_obs"] = target_obs_factory()
    cfg.reward_components = {"target_rew": target_reward_factory()}


def _install_camera(simulator_cfg, args: argparse.Namespace) -> None:
    """Give the dog the camera it would actually be looking through.

    The field of view is shared deliberately: the demonstrator's sight gate
    (--fov-deg) and the lens are the same number, so what the expert is
    allowed to know matches what the student will be shown. Letting them
    drift apart would teach the student to find a ball that never appears in
    its frame.
    """
    if simulator_cfg is None or not (_arg(args, "camera") or _arg(args, "record")):
        return
    from protomotions.robot_configs.go2 import go2_front_camera

    res = _arg(args, "camera_res")
    simulator_cfg.onboard_cameras = {
        "front_camera": go2_front_camera(
            width=res, height=res, fov_deg=_arg(args, "fov_deg")
        )
    }


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args: argparse.Namespace):
    """Training-path hook (config_builder calls this). The inference path
    goes through apply_inference_overrides below, which does the same."""
    _install_camera(simulator_cfg, args)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    cfg = _steering()._transformer().env_config(robot_cfg, args)
    _install_chase(cfg, args)
    return cfg


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg,
    agent_cfg,
    terrain_cfg,
    motion_lib_cfg,
    scene_lib_cfg,
    args: argparse.Namespace,
):
    """inference_agent.py builds configs from the checkpoint pickle and calls
    only this hook, so the task has to be installed here, not in env_config."""
    _install_camera(simulator_cfg, args)
    if env_cfg is None:
        return
    _install_chase(env_cfg, args)
    env_cfg.max_episode_length = 100000
