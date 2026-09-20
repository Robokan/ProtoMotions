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
    "cruise_speed": 1.6,
    "report_every": 250,
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
        "--cruise-speed", type=float, default=_DEFAULTS["cruise_speed"],
        help="Forward command when the ball is far and dead ahead (m/s).")


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
        ),
        "masked_mimic": MaskedMimicGoalControlConfig(
            num_masked_future_steps=trained.num_masked_future_steps,
            future_steps=trained.future_steps,
            bootstrap_on_episode_end=trained.bootstrap_on_episode_end,
            horizon_sec=trained.horizon_sec,
            height_mode=trained.height_mode,
            condition_rotation=trained.condition_rotation,
            report_every_steps=trained.report_every_steps,
            target_component="ball",
            # Aim AT the ball; two feet is the success TEST, not the
            # destination. Aiming at the boundary parks the dog outside it.
            stop_distance=0.0,
            max_speed=_arg(args, "cruise_speed"),
        ),
    }

    # Wired but unused while this component does the driving: these are what a
    # learned high-level policy (or a VLA fine-tune) would train against. The
    # steering reward is dropped with the steering command it scored.
    cfg.observation_components["target_obs"] = target_obs_factory()
    cfg.reward_components = {"target_rew": target_reward_factory()}


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
    if env_cfg is None:
        return
    _install_chase(env_cfg, args)
    env_cfg.max_episode_length = 100000
