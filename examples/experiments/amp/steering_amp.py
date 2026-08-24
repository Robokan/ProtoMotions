# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Single-network AMP steering: one policy, command in, joints out.

The AMP-direct variant of the game-controller task (IsaacLabASE:
rl_games_amp_flat_game_controller_cfg.yaml -- "EDV this is how to get the
task reward to work combined with the amp reward!!!"): no ASE stage, no
latents, no frozen LLC. ONE network sees the robot state plus the velocity
command and directly outputs joint targets, trained on the WHOLE corpus with
the AMP style reward and the steering task reward combined (original mix:
style 1.0 effective vs task 0.5).

Everything steering (command random walk, spawn-velocity seeding, markers,
gamepad teleop) is the same SteeringCommandControl component the ASE HLC
variant uses -- the difference is purely who consumes the command: the policy
itself instead of a latent-emitting HLC.

    python protomotions/train_agent.py --robot-name anymal_d \\
        --simulator isaaclab --physics physx --headless \\
        --motion-file data/motions/anymal_d/anymal_d_flat_simframes.pt \\
        --experiment-path examples/experiments/amp/steering_amp.py \\
        --disc-term-threshold 0.02 \\
        --num-envs 4096 --batch-size 16384 \\
        --experiment-name anymal_amp_steering_v1
"""

import argparse
import importlib.util
from pathlib import Path

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.envs.base_env.config import EnvConfig

# Delegate everything not steering-specific to the plain AMP experiment
# (the proven fair-test recipe), mlp_template_tuned-style.
_spec = importlib.util.spec_from_file_location(
    "_amp_base", str(Path(__file__).parent / "mlp.py")
)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

terrain_config = _base.terrain_config
scene_lib_config = _base.scene_lib_config
motion_lib_config = _base.motion_lib_config
configure_robot_and_simulator = getattr(
    _base, "configure_robot_and_simulator", None
)

HISTORY_STEPS = 8


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    _base.additional_experiment_arguments(parser)
    parser.add_argument(
        "--forward-vel-min", type=float, default=-1.0,
        help="Backward command bound (m/s, negative).")
    parser.add_argument(
        "--forward-vel-max", type=float, default=2.0,
        help="Forward command bound (m/s). Default 2.0 -- walking-corpus "
             "scale; the ASE HLC default of 4.0 assumes running clips.")
    parser.add_argument(
        "--turn-vel-max", type=float, default=2.0,
        help="Yaw-rate command bound (rad/s, symmetric).")
    parser.add_argument(
        "--side-vel-max", type=float, default=1.0,
        help="Lateral command bound (m/s, symmetric).")
    parser.add_argument(
        "--no-terminations", action="store_true",
        help="Strip every termination component (episodes end on timeout "
             "only). Pair with --disc-term-threshold 0 to disable the style "
             "kill too.")
    parser.add_argument(
        "--task-reward-w", type=float, default=0.5,
        help="Steering task reward weight (original AMP game controller: "
             "0.5 task vs 1.0 effective style).")


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Base AMP env + the steering command component, obs and reward."""
    from protomotions.envs.steering.command import (
        SteeringCommandControlConfig,
        steering_command_obs_factory,
        steering_command_reward_factory,
    )

    cfg: EnvConfig = _base.env_config(robot_cfg, args)

    cfg.control_components = dict(cfg.control_components or {})
    cfg.control_components["steering_cmd"] = SteeringCommandControlConfig(
        forward_vel_min=args.forward_vel_min,
        forward_vel_max=args.forward_vel_max,
        turn_vel_max=args.turn_vel_max,
        side_vel_max=args.side_vel_max,
    )
    cfg.observation_components["task_obs"] = steering_command_obs_factory()
    cfg.reward_components = dict(cfg.reward_components or {})
    cfg.reward_components["steering_command_rew"] = (
        steering_command_reward_factory(
            forward_vel_min=args.forward_vel_min,
            forward_vel_max=args.forward_vel_max,
            turn_vel_max=args.turn_vel_max,
            side_vel_max=args.side_vel_max,
            weight=args.task_reward_w,
        )
    )
    if getattr(args, "no_terminations", False):
        cfg.termination_components = {}
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    """Base AMP agent with the command in the policy's (and critics') input.

    The ONE network consumes [robot state, command]; the discriminator keeps
    judging pure motion (no command channels), exactly like the original --
    style is about HOW it moves, the task reward about WHERE.
    """
    cfg = _base.agent_config(robot_cfg, env_cfg, args)

    def _add_task_obs(module_cfg):
        for attr in ("in_keys",):
            keys = getattr(module_cfg, attr, None)
            if isinstance(keys, list) and "task_obs" not in keys and \
                    "max_coords_obs" in keys:
                keys.append("task_obs")
        for sub in getattr(module_cfg, "modules", None) or []:
            _add_task_obs(sub)

    _add_task_obs(cfg.model.actor)
    _add_task_obs(cfg.model.critic)
    if getattr(cfg.model, "disc_critic", None) is not None:
        _add_task_obs(cfg.model.disc_critic)
    # The model container validates every submodule key against its own
    # top-level in_keys list.
    if "task_obs" not in cfg.model.in_keys:
        cfg.model.in_keys.append("task_obs")

    # Reward mix (IsaacLabASE game controller: task_reward_w 0.5,
    # disc_reward_w 0.5). The base AMP experiment ships agent
    # task_reward_w=0.1 -- a multiplier sized for tiny TIE-BREAK rewards
    # (energy preference), not a real task. Left at 0.1 it stacks with the
    # env-side steering weight (0.5) to an effective 0.05 vs style 1.0 --
    # a 20:1 style domination that trained dead-flat for 36k epochs on
    # anymal (2026-08-22). Agent multiplier 1.0 makes the env weight the
    # single source of truth: effective task = --task-reward-w (0.5).
    cfg.task_reward_w = 1.0
    cfg.amp_parameters.discriminator_reward_w = 0.5
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
    """Drive-around viewing, same rig as the ASE HLC variant: long episode,
    gamepad auto-detect, markers on the selected robot."""
    _base.apply_inference_overrides(
        robot_cfg, simulator_cfg, env_cfg, agent_cfg,
        terrain_cfg, motion_lib_cfg, scene_lib_cfg, args,
    )
    if env_cfg is not None:
        env_cfg.max_episode_length = 100000

        import glob as _glob
        import logging as _logging

        _log = _logging.getLogger(__name__)
        cmd = env_cfg.control_components.get("steering_cmd")
        if cmd is not None and not getattr(args, "command_source", None):
            pads = _glob.glob("/dev/input/js*")
            if pads:
                cmd.command_source = "gamepad"
                _log.info("steering: gamepad detected (%s) -- teleop ON", pads[0])
            else:
                cmd.command_source = None
                _log.info("steering: no gamepad -- random command generator")
