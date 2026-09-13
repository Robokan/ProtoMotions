# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""AMP backwards locomotion: style from real mocap, task reward to sustain it.

The go2 corpus has no sustained backing-up gait. What it has is 42 fragments of
0.35-0.93s (`go2_backwards_real.pt`, 25.2s total) -- single backward steps, cut
from clips where the source dog genuinely reversed while upright and standing.

That is enough for a discriminator to learn what a backward step LOOKS like,
but there is no example anywhere of continuing one, so style alone can produce a
convincing step followed by a stall. This experiment adds a heading-relative
backward-velocity term to supply the "keep going" signal the reference data
cannot.

The alternative -- the 8 `*_walk_backwards` clips -- is time-reversed forward
walking: physically impossible (a footfall's deceleration becomes an
acceleration away from the ground), so the discriminator gap can never close.
Those are deliberately NOT used here.

    train_agent.py --experiment-path examples/experiments/amp/backwards_amp.py \
        --motion-file data/motions/go2/go2_backwards_real.pt ...
"""

import argparse

from examples.experiments.amp.mlp import (  # noqa: F401  (loader re-exports)
    terrain_config,
    scene_lib_config,
    motion_lib_config,
    apply_inference_overrides,
)
from examples.experiments.amp import mlp as _base
from protomotions.agents.amp.config import AMPAgentConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.component_factories import backward_velocity_rew_factory

    cfg = _base.env_config(robot_cfg, args)

    # Heading-relative, so turning around and walking forwards earns nothing.
    # Saturating ramp rather than a Gaussian: a Gaussian wide enough to accept
    # the corpus spread (-0.13..-0.43 m/s) also pays ~0.78 for standing still.
    cfg.reward_components["backward_velocity"] = backward_velocity_rew_factory(
        weight=getattr(args, "backward_weight", 1.0),
        target_speed=getattr(args, "backward_target_speed", 0.25),
        lateral_penalty_w=0.1,
    )
    return cfg


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> AMPAgentConfig:
    """Task-driven AMP weighting: 50% task, 50% style (Peng et al. 2021).

    amp/mlp.py sets task_reward_w = 0.1 because in a style-ONLY run the task
    channel holds nothing but the pow_rew regularizer, and 0.1 keeps that from
    drowning the discriminator. This experiment puts a REAL task reward
    (backward_velocity) into that same channel, so inheriting 0.1 silently
    weighted the objective 1:10 against the task -- the two prior runs both
    stalled after ~1s and then coasted, which is what that ratio buys.
    (Sep-2 run was 0.5/1.0 = 1:2; go2_amp_backwards_v2 was 0.1/1.0 = 1:10.)

    Note the weights multiply ADVANTAGES, not raw rewards -- task at
    ppo/agent.py:780, discriminator at amp/component.py:725 -- and each stream
    is normalized separately beforehand, so this is 1:1 on the gradient rather
    than on the reward blend the paper describes.
    """
    cfg = _base.agent_config(robot_config, env_config, args)
    cfg.task_reward_w = getattr(args, "task_reward_w", 0.5)
    cfg.amp_parameters.discriminator_reward_w = getattr(
        args, "style_reward_w", 0.5
    )
    return cfg


def additional_experiment_arguments(parser: argparse.ArgumentParser) -> None:
    # Delegate FIRST so this file keeps --no-fall-termination,
    # --disc-term-threshold and the rest of amp/mlp.py's flags. Defining the
    # hook without delegating silently drops them and train_agent.py rejects
    # the launch as "unrecognized arguments".
    _base.additional_experiment_arguments(parser)
    parser.add_argument(
        "--backward-weight", type=float, default=1.0,
        help="weight of the backward-velocity task reward",
    )
    parser.add_argument(
        "--task-reward-w", type=float, default=0.5,
        help="AMP task-reward weight (paper default 0.5 for task-driven AMP)",
    )
    parser.add_argument(
        "--style-reward-w", type=float, default=0.5,
        help="AMP discriminator-reward weight (paper default 0.5)",
    )
    parser.add_argument(
        "--backward-target-speed", type=float, default=0.25,
        help="backward speed (m/s, positive) at which the term saturates. "
             "Corpus fragments run 0.13-0.43 m/s, median 0.21.",
    )
