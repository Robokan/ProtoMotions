# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Omnidirectional velocity-command steering over a frozen ASE LLC.

Stage 2 of the two-stage recipe: an ASE pretrain (examples/experiments/ase/
mlp.py) learns the skill space, then this high-level policy emits skill
latents that make the frozen low-level controller track a game-controller
command -- forward velocity, yaw rate, lateral velocity, in the robot's
heading frame. Ported from IsaacLabASE's game_controller task; the command
walk, target shaping and reward live in protomotions/envs/steering/command.py.

Robot-agnostic (root state only): built for anymal_d and go2, works for any
robot with an ASE pretrain.

    python protomotions/train_agent.py --robot-name anymal_d \\
        --simulator isaaclab --headless \\
        --motion-file data/motions/anymal_d/anymal_d_flat.pt \\
        --experiment-path examples/experiments/ase/steering_ase_hlc.py \\
        --llc-checkpoint results/anymal_ase_getup_v1/last.ckpt \\
        --num-envs 4096 --batch-size 8192 \\
        --experiment-name anymal_steering_hlc_v1
"""

import argparse

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.envs.base_env.config import EnvConfig

LATENT_DIM = 64  # must match the LLC pretrain (ase/mlp.py)
HISTORY_STEPS = 8  # must match the LLC pretrain's discriminator window


def additional_experiment_arguments(parser):
    parser.add_argument(
        "--llc-checkpoint", type=str, required=True,
        help="ASE pretrain checkpoint providing the FROZEN low-level "
             "controller (and its discriminator, for the style reward).")
    parser.add_argument(
        "--llc-steps", type=int, default=5,
        help="Sim steps per high-level decision (IsaacLabASE llc_steps).")
    parser.add_argument(
        "--forward-vel-min", type=float, default=-1.0,
        help="Backward command bound (m/s, negative).")
    parser.add_argument(
        "--forward-vel-max", type=float, default=4.0,
        help="Forward command bound (m/s).")
    parser.add_argument(
        "--turn-vel-max", type=float, default=2.0,
        help="Yaw-rate command bound (rad/s, symmetric).")
    parser.add_argument(
        "--side-vel-max", type=float, default=1.0,
        help="Lateral command bound (m/s, symmetric).")
    parser.add_argument(
        "--task-reward-w", type=float, default=0.9,
        help="Command-tracking weight in the HLC reward mix.")
    parser.add_argument(
        "--latent-bank", type=str, default=None,
        help="Skill-latent bank from protomotions/agents/ase/latent_bank.py. "
             "With --button-skills, a held button pipes that skill's latent "
             "straight into the frozen LLC (press B to sit).")
    parser.add_argument(
        "--button-skills", type=str, default="",
        help="Comma-separated skill names, one per button, in button order "
             "(e.g. 'sit,jump').")
    parser.add_argument(
        "--disc-reward-w", type=float, default=0.1,
        help="Style weight from the FROZEN LLC discriminator (0 disables).")


def terrain_config(args: argparse.Namespace):
    from protomotions.components.terrains.config import TerrainConfig

    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    from protomotions.components.scene_lib import SceneLibConfig

    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    from protomotions.components.motion_lib import MotionLibConfig

    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.action import make_pd_action_config
    from protomotions.envs.component_factories import (
        historical_max_coords_obs_factory,
        max_coords_obs_factory,
    )
    from protomotions.envs.steering.command import (
        SteeringCommandControlConfig,
        steering_command_obs_factory,
        steering_command_reward_factory,
    )

    control_components = {
        "steering_cmd": SteeringCommandControlConfig(
            num_buttons=len([s for s in args.button_skills.split(",") if s]),
            forward_vel_min=args.forward_vel_min,
            forward_vel_max=args.forward_vel_max,
            turn_vel_max=args.turn_vel_max,
            side_vel_max=args.side_vel_max,
        ),
    }

    observation_components = {
        # Self-state: MUST match the LLC pretrain's max_coords_obs settings
        # (ase/mlp.py: local_obs, root_height, no contacts) -- the frozen LLC
        # consumes this observation verbatim.
        "max_coords_obs": max_coords_obs_factory(
            local_obs=True, root_height_obs=True, observe_contacts=False
        ),
        "task_obs": steering_command_obs_factory(),
        # Motion-history window for the frozen discriminator's style reward.
        "historical_max_coords_obs": historical_max_coords_obs_factory(
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
            history_steps=HISTORY_STEPS,
        ),
    }

    reward_components = {
        "steering_command_rew": steering_command_reward_factory(
            forward_vel_min=args.forward_vel_min,
            forward_vel_max=args.forward_vel_max,
            turn_vel_max=args.turn_vel_max,
            side_vel_max=args.side_vel_max,
        ),
    }

    return EnvConfig(
        max_episode_length=300,
        num_state_history_steps=HISTORY_STEPS,
        control_components=control_components,
        observation_components=observation_components,
        reward_components=reward_components,
        action_config=make_pd_action_config(robot_cfg),
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
):
    from protomotions.agents.ase.hlc_agent import ASEHLCAgentConfig, HLCParams
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.common.config import (
        MLPLayerConfig,
        MLPWithConcatConfig,
        ModuleContainerConfig,
        ModuleOperationForwardConfig,
        ObsProcessorConfig,
        PretrainedModelConfig,
    )
    from protomotions.agents.ppo.config import (
        AdaptiveLRConfig,
        PPOActorConfig,
        PPOModelConfig,
    )

    hlc_in_keys = ["max_coords_obs", "task_obs"]

    obs_processors = [
        ObsProcessorConfig(
            in_keys=["max_coords_obs"],
            out_keys=["max_coords_obs_flattened"],
            normalize_obs=True,
            norm_clamp_value=5,
            module_operations=[ModuleOperationForwardConfig()],
        ),
        ObsProcessorConfig(
            in_keys=["task_obs"],
            out_keys=["task_obs_flattened"],
            normalize_obs=True,
            norm_clamp_value=5,
            module_operations=[ModuleOperationForwardConfig()],
        ),
    ]

    # The trunk's 64-dim output IS the skill latent (projected to the
    # hypersphere inside the agent before the LLC sees it).
    actor_config = PPOActorConfig(
        mu_key="actor_trunk_out",
        num_out=LATENT_DIM,
        actor_logstd=-2.3,
        in_keys=hlc_in_keys,
        mu_model=ModuleContainerConfig(
            in_keys=hlc_in_keys,
            out_keys=["actor_trunk_out"],
            models=[
                ModuleContainerConfig(
                    in_keys=hlc_in_keys,
                    out_keys=["max_coords_obs_flattened", "task_obs_flattened"],
                    models=list(obs_processors),
                ),
                MLPWithConcatConfig(
                    in_keys=["max_coords_obs_flattened", "task_obs_flattened"],
                    out_keys=["actor_trunk_out"],
                    num_out=LATENT_DIM,
                    layers=[
                        MLPLayerConfig(units=1024, activation="relu"),
                        MLPLayerConfig(units=512, activation="relu"),
                    ],
                ),
            ],
        ),
    )

    critic_config = ModuleContainerConfig(
        in_keys=hlc_in_keys,
        out_keys=["value"],
        models=list(obs_processors)
        + [
            MLPWithConcatConfig(
                in_keys=["max_coords_obs_flattened", "task_obs_flattened"],
                out_keys=["value"],
                num_out=1,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu"),
                    MLPLayerConfig(units=512, activation="relu"),
                ],
            ),
        ],
    )

    pretrained_modules = {
        "llc": PretrainedModelConfig(
            checkpoint_path=args.llc_checkpoint,
            module_path="actor",
        ),
    }
    if args.disc_reward_w > 0:
        pretrained_modules["llc_disc"] = PretrainedModelConfig(
            checkpoint_path=args.llc_checkpoint,
            module_path="discriminator",
        )

    return ASEHLCAgentConfig(
        pretrained_modules=pretrained_modules,
        model=PPOModelConfig(
            in_keys=hlc_in_keys,
            actor=actor_config,
            critic=critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
        ),
        hlc=HLCParams(
            latent_dim=LATENT_DIM,
            llc_deterministic=True,
            decision_interval=args.llc_steps,
            task_reward_w=args.task_reward_w,
            disc_reward_w=args.disc_reward_w,
            latent_bank_path=args.latent_bank,
            button_skills=[s for s in args.button_skills.split(",") if s],
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        e_clip=0.2,
        gamma=0.95,
        tau=0.95,
        num_steps=32,
        num_mini_epochs=6,
        entropy_coef=0.005,
        bounds_loss_coef=10.0,
        adaptive_lr=AdaptiveLRConfig(
            enabled=True, desired_kl=0.003, min_lr=1e-5, max_lr=9e-5
        ),
        normalize_rewards=True,
        gradient_clip_val=25.0,
        save_last_checkpoint_every=10,
    )


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
    """Drive-around viewing: one long episode, commands keep resampling."""
    if env_cfg is not None:
        env_cfg.max_episode_length = 100000
