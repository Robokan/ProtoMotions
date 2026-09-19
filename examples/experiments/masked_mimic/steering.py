# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Drive a trained MaskedMimic policy with a velocity command.

Same idea as examples/experiments/ase/steering_ase_hlc.py -- forward speed,
yaw rate and lateral speed in the robot's heading frame, the same green
direction arrow and spinning turn dial under the robot -- but there is no
high-level policy to train here. MaskedMimic already takes sparse future body
poses, so the command is turned straight into a base-link target: where the
robot would be in 0.2-1.0 s if it held the commanded velocity, re-anchored on
the live root pose every step (protomotions/envs/steering/masked_mimic_command.py).

Every other body is masked out, so the policy picks its own gait.

The command generator is the one the ASE HLC trains against, so it draws the
same random walk over forward/backward, left/right turns and strafing: leave
it running and it exercises the whole command box on its own. Plug a gamepad
in before launching and it drives the camera-followed robot instead.

Reward is the ASE steering task reward, scored on the SAME command, so the
number is directly comparable between the two stacks. The component also
prints mean |command - achieved| per channel every few hundred steps.

INFERENCE ONLY. The observations and the model come from the checkpoint
unchanged, so an existing MaskedMimic checkpoint loads as-is:

    python protomotions/inference_agent.py \\
        --checkpoint results/go2_masked_mimic_v4/last.ckpt \\
        --experiment-path examples/experiments/masked_mimic/steering.py \\
        --simulator isaaclab --physics physx --num-envs 4

inference_agent.py builds every config from the checkpoint's pickled
resolved_configs and only calls apply_inference_overrides(), so it never sees
this file's argparse flags. Retune from the command line with --overrides:

    --overrides env.control_components.steering_cmd.forward_vel_max=3.0 \\
                env.control_components.masked_mimic.horizon_sec=0.6
"""

import argparse
import os

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.envs.base_env.config import EnvConfig


# Also the defaults used at inference, where argparse never runs.
_DEFAULTS = {
    "forward_vel_min": -1.0,
    # 2.5 m/s is the fastest thing the go2 corpus actually contains: the
    # quickest clip averages 2.60 m/s and only 4.6% of the sampling weight
    # sits above 2.0. The ASE steering default of 4.0 would spend most of its
    # range commanding a gait that was never demonstrated -- and for
    # MaskedMimic it also puts the 1 s target 4 m ahead, far outside any
    # lead distance it was distilled on.
    "forward_vel_max": 2.5,
    # |yaw| above 2.0 rad/s is 2.2% of corpus time; 1.5 keeps the stick inside
    # what is actually shown while still covering brisk turns.
    "turn_vel_max": 1.5,
    # 0.3 m/s, not 1.0: |lateral| above 0.3 is 12% of go2 corpus time and
    # above 0.5 is 3.1% (44 s). A stick that commands 1.0 is asking for a gait
    # that was never demonstrated, and distillation cannot invent one.
    "side_vel_max": 0.3,
    "command_hold_steps_min": 125,
    "command_hold_steps_max": 175,
    "target_horizon_sec": 1.0,
    "height_mode": "current",
    "target_root_height": None,
    "no_target_rotation": False,
    "report_every": 250,
}


def _arg(args, name):
    return getattr(args, name, _DEFAULTS[name])


def _transformer():
    """Load the MaskedMimic training experiment this one evaluates."""
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transformer.py")
    spec = importlib.util.spec_from_file_location("masked_mimic_transformer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    _transformer().additional_experiment_arguments(parser)
    parser.add_argument(
        "--forward-vel-min", type=float, default=_DEFAULTS["forward_vel_min"],
        help="Backward command bound (m/s, negative).")
    parser.add_argument(
        "--forward-vel-max", type=float, default=_DEFAULTS["forward_vel_max"],
        help="Forward command bound (m/s). The go2 corpus tops out at a "
             "2.60 m/s clip mean, so much above 2.5 is undemonstrated.")
    parser.add_argument(
        "--turn-vel-max", type=float, default=_DEFAULTS["turn_vel_max"],
        help="Yaw-rate command bound (rad/s, symmetric).")
    parser.add_argument(
        "--side-vel-max", type=float, default=_DEFAULTS["side_vel_max"],
        help="Lateral command bound (m/s, symmetric).")
    parser.add_argument(
        "--command-hold-steps-min", type=int,
        default=_DEFAULTS["command_hold_steps_min"],
        help="Shortest interval between command redraws (env steps).")
    parser.add_argument(
        "--command-hold-steps-max", type=int,
        default=_DEFAULTS["command_hold_steps_max"],
        help="Longest interval between command redraws (env steps).")
    parser.add_argument(
        "--target-horizon-sec", type=float,
        default=_DEFAULTS["target_horizon_sec"],
        help="Lead time of the farthest conditioned base-link target. The "
             "targets are spread evenly over (0, horizon].")
    parser.add_argument(
        "--height-mode", choices=["current", "corpus", "fixed"],
        default=_DEFAULTS["height_mode"],
        help="Where the commanded base-link height comes from. 'current' "
             "targets the height the robot is already at, so the z channel "
             "carries no instruction and the command is purely planar "
             "(the mask cannot drop z on its own -- one bit covers the whole "
             "translation vector). 'corpus' commands the height the corpus "
             "carries at that speed. 'fixed' uses --target-root-height.")
    parser.add_argument(
        "--target-root-height", type=float,
        default=_DEFAULTS["target_root_height"],
        help="Commanded base-link height above terrain (m). Defaults to the "
             "robot's measured standing height.")
    parser.add_argument(
        "--no-target-rotation", action="store_true",
        help="Condition the base-link position only, leaving its yaw free.")
    parser.add_argument(
        "--report-every", type=int, default=_DEFAULTS["report_every"],
        help="Print mean |command - achieved| velocity every N env steps. "
             "0 silences the readout.")


def terrain_config(args: argparse.Namespace):
    return _transformer().terrain_config(args)


def scene_lib_config(args: argparse.Namespace):
    return _transformer().scene_lib_config(args)


def motion_lib_config(args: argparse.Namespace):
    return _transformer().motion_lib_config(args)


def _install_steering(cfg: EnvConfig, args: argparse.Namespace) -> None:
    """Swap the clip conditioning for a velocity command, in place.

    Observations are left exactly as trained -- the MaskedMimic prior reads
    masked_mimic_target_poses/masks/times plus the robot's own state, and all
    of those keep their widths, so the checkpoint loads unchanged. Only the
    control components change, plus the two things that assume a clip is being
    tracked.
    """
    from protomotions.envs.steering.command import (
        SteeringCommandControlConfig,
        steering_command_reward_factory,
    )
    from protomotions.envs.steering.masked_mimic_command import (
        MaskedMimicSteeringControlConfig,
    )

    trained = cfg.control_components.get("masked_mimic")
    if trained is None:
        trained = next(
            component
            for component in cfg.control_components.values()
            if "masked_mimic" in component._target_.lower()
        )

    forward_vel_min = _arg(args, "forward_vel_min")
    forward_vel_max = _arg(args, "forward_vel_max")
    turn_vel_max = _arg(args, "turn_vel_max")
    side_vel_max = _arg(args, "side_vel_max")

    cfg.control_components = {
        "steering_cmd": SteeringCommandControlConfig(
            # Project commands into the achievable region before the policy
            # sees them, so gamepad corners map onto something demonstrated
            # rather than onto sprint-and-strafe (19 s of the whole corpus).
            shape_commands=True,
            forward_vel_min=forward_vel_min,
            forward_vel_max=forward_vel_max,
            turn_vel_max=turn_vel_max,
            side_vel_max=side_vel_max,
            heading_change_steps_min=_arg(args, "command_hold_steps_min"),
            heading_change_steps_max=_arg(args, "command_hold_steps_max"),
        ),
        "masked_mimic": MaskedMimicSteeringControlConfig(
            # Inherited from the checkpoint's own env so the conditioned-body
            # layout and the observation widths stay identical.
            num_masked_future_steps=trained.num_masked_future_steps,
            future_steps=trained.future_steps,
            bootstrap_on_episode_end=trained.bootstrap_on_episode_end,
            horizon_sec=_arg(args, "target_horizon_sec"),
            height_mode=_arg(args, "height_mode"),
            target_root_height=_arg(args, "target_root_height"),
            condition_rotation=not _arg(args, "no_target_rotation"),
            report_every_steps=_arg(args, "report_every"),
        ),
    }

    # Nothing is tracking a clip any more: the tracking-error termination
    # would fire within a second, and the tracking rewards would score a
    # reference the robot is not following. Score the command instead -- this
    # is the SAME reward the ASE steering HLC is trained and judged on.
    cfg.termination_components = {}
    cfg.reward_components = {
        "steering_command_rew": steering_command_reward_factory(
            forward_vel_min=forward_vel_min,
            forward_vel_max=forward_vel_max,
            turn_vel_max=turn_vel_max,
            side_vel_max=side_vel_max,
            # The command is already projected at the source; projecting again
            # in the reward would compound (the safe-velocity cap is not
            # idempotent) and score a different command than was issued.
            pre_shaped=True,
        ),
    }

    # Spawn mid-gait, not at the clip start. The checkpoint's INFERENCE
    # pickle -- which is what inference_agent.py loads -- carries
    # init_start_prob = 1.0 from the mimic eval overrides, so every robot was
    # being dropped at t=0 of its clip, which for nearly every clip is a
    # standstill. The command is then seeded from that standstill and the run
    # opens with everything at zero no matter what the seeding does
    # (Eric, 2026-09-19). 0.0 = always a random time in the clip, so the robot
    # starts already trotting and the command starts at the speed it is
    # actually moving, which is the whole point of the spawn seeding.
    if getattr(cfg, "motion_manager", None) is not None:
        cfg.motion_manager.init_start_prob = 0.0

    # Drive around until the viewer is closed.
    cfg.max_episode_length = 100000


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    cfg: EnvConfig = _transformer().env_config(robot_cfg, args)
    _install_steering(cfg, args)
    return cfg


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
):
    return _transformer().agent_config(robot_config, env_config, args)


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
    """Install the steering task on the checkpoint's frozen env config.

    This is the ONLY hook inference_agent.py calls on an experiment file --
    env_config() above is for train_agent.py -- so the swap has to happen
    here, not there.

    Deliberately does NOT call the mimic eval overrides transformer.py uses:
    those pin each env to a clip and end the episode with it, which is the
    opposite of what this experiment is for.
    """
    if env_cfg is None:
        return

    _install_steering(env_cfg, args)

    # Gamepad auto-detect, same rule as the ASE steering experiment: a pad
    # plugged in at launch drives the camera-followed robot, otherwise the
    # random generator keeps every env and the markers still read out what is
    # commanded. Explicit override wins:
    #   --command-source steering_cmd=gamepad  (or =random)
    import logging as _logging

    from protomotions.envs.steering.gamepad import find_gamepad

    _log = _logging.getLogger(__name__)
    cmd = env_cfg.control_components.get("steering_cmd")
    if cmd is not None and not getattr(args, "command_source", None):
        # Probe the device rather than trusting that a js* node exists: this
        # machine's js0 is an "ASRock LED Controller" whose axes rest pinned
        # at full scale.
        pad_dev, pad_name = find_gamepad()
        if pad_dev:
            cmd.command_source = "gamepad"
            _log.info(
                "steering: gamepad detected (%s: %s) -- teleop ON",
                pad_dev, pad_name,
            )
        else:
            cmd.command_source = None
            _log.info(
                "steering: no gamepad detected -- random command generator "
                "drives (plug a pad in and relaunch, or force with "
                "--command-source steering_cmd=gamepad)"
            )
