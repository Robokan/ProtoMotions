# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""ASE self-play battle league — the ASE arm of the ASE-vs-GPC baseline.

Same battle arena, rules, and rewards as the GPC league
(examples/experiments/battle/battle_league_prior_peft.py), but the fighter is
a classic ASE policy: PPO conditioned on a skill latent, with an AMP-style
discriminator on the motion corpus as the naturalness anchor (ASE's analog of
the GPC prior constraint). Self-play only: the league pool holds full-weight
snapshots of this same architecture (see agents/league/ase_agent.py).

Robot-generic: soma23 / t800 / atlas via per-robot battle body tables
(protomotions/envs/battle/robot_tables.py).

Usage (pretrain first, then league — warm start via --checkpoint):

    # 1) ASE pretraining on the curated corpus (style + MI only):
    python protomotions/train_agent.py \
        --robot-name t800 --simulator isaaclab --headless \
        --motion-file data/t800_prior_corpus.pt \
        --experiment-path examples/experiments/ase/mlp.py \
        --num-envs 4096 --batch-size 8192 --training-max-steps 500000000 \
        --experiment-name t800_ase_pretrain_v1

    # 2) Battle league, warm-started from the pretrain:
    python protomotions/train_agent.py \
        --robot-name t800 --simulator isaaclab --headless \
        --motion-file data/t800_prior_corpus.pt \
        --experiment-path examples/experiments/ase/battle_league_ase.py \
        --num-envs 256 --batch-size 512 --training-max-steps 200000000 \
        --experiment-name t800_ase_battle_v1 \
        --checkpoint results/t800_ase_pretrain_v1/last.ckpt
"""

import argparse

from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig

# Arena geometry + terrain grid: identical to the GPC league by construction.
from examples.experiments.battle.battle_league_prior_peft import (
    ARENA_SIZE,
    ARENA_SPACING,
    terrain_config,  # noqa: F401  (re-exported for the experiment loader)
    scene_lib_config,  # noqa: F401
    motion_lib_config,  # noqa: F401
)

HISTORY_STEPS = 8  # discriminator reference window (matches ase/mlp.py)


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--dense-reward-scale",
        type=float,
        default=1.0,
        help="Scale on all dense battle reward terms (win/lose unaffected).",
    )


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    # Contact sensors on strike/damage bodies (semantic names resolve
    # per robot). Same list as the GPC battle league.
    robot_cfg.update_fields(
        contact_bodies=[
            "all_left_foot_bodies",
            "all_right_foot_bodies",
            "all_left_hand_bodies",
            "all_right_hand_bodies",
            "head_body_name",
            "torso_body_name",
        ],
    )
    if hasattr(simulator_cfg, "filter_env_collisions"):
        simulator_cfg.filter_env_collisions = False


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.action import make_pd_action_config
    from protomotions.envs.battle.control import BattleControlConfig
    from protomotions.envs.battle.hit_state import HitStateConfig
    from protomotions.envs.battle.factories import (
        battle_task_obs_factory,
        default_battle_reward_components,
    )
    from protomotions.envs.battle.robot_tables import battle_table_kwargs
    from protomotions.envs.component_factories import (
        historical_max_coords_obs_factory,
        max_coords_obs_factory,
    )
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig

    observation_components = {
        # Self-state for the policy (ASE-style local obs).
        "max_coords_obs": max_coords_obs_factory(
            local_obs=True, root_height_obs=True, observe_contacts=False
        ),
        # Discriminator window (StateHistoryBuffer).
        "historical_max_coords_obs": historical_max_coords_obs_factory(
            local_obs=True, root_height_obs=True, observe_contacts=False
        ),
        # Opponent/fight state.
        "task_obs": battle_task_obs_factory(),
    }

    dense = getattr(args, "dense_reward_scale", 1.0)

    # KEEP IN SYNC with battle_league_prior_peft.py — the rules must be
    # identical for the ASE-vs-GPC comparison to mean anything.
    battle_control = BattleControlConfig(
        arena_size=ARENA_SIZE,
        arena_spacing=ARENA_SPACING,
        raw_health_damage=True,
        damage_to_health=0.005,
        max_hp_per_hit=0.25,
        hit_state=HitStateConfig(strike_min_speed=0.0, ke_reward_ref=5.0),
        stun_gates_ko=True,
        **battle_table_kwargs(robot_cfg, args.robot_name),
    )

    cfg = EnvConfig(
        ref_contact_smooth_window=7,
        num_state_history_steps=HISTORY_STEPS,
        max_episode_length=750,  # 15 s at 50 Hz, same round length as GPC league
        reset_grace_period=5,
        ref_respawn_offset=0.05,
        control_components={"battle": battle_control},
        observation_components=observation_components,
        reward_components=default_battle_reward_components(dense_scale=dense),
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
        ),
    )
    cfg._target_ = "protomotions.envs.battle.env.BattleEnv"
    return cfg


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
):
    from protomotions.agents.amp.config import AMPParametersConfig
    from protomotions.agents.ase.config import (
        ASEDiscriminatorEncoderConfig,
        ASEModelConfig,
        ASEParametersConfig,
    )
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.common.config import (
        MLPLayerConfig,
        MLPWithConcatConfig,
        ModuleContainerConfig,
        ModuleOperationForwardConfig,
        ObsProcessorConfig,
    )
    from protomotions.agents.league.agent import LeagueParams
    from protomotions.agents.league.ase_agent import LeagueASEAgentConfig
    from protomotions.agents.ppo.config import PPOActorConfig
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs import compute_historical_max_coords_from_motion_lib

    ase_parameters = ASEParametersConfig(
        conditional_discriminator=False,
        latent_dim=64,
        mi_reward_w=0.5,
        mi_hypersphere_reward_shift=True,
        diversity_bonus=0.01,
        latent_uniformity_weight=0.0,
    )

    policy_in_keys = ["max_coords_obs", "task_obs", "latents"]

    actor_config = PPOActorConfig(
        mu_key="actor_trunk_out",
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=policy_in_keys,
        mu_model=ModuleContainerConfig(
            in_keys=policy_in_keys,
            out_keys=["actor_trunk_out"],
            models=[
                ModuleContainerConfig(
                    in_keys=policy_in_keys,
                    out_keys=[
                        "max_coords_obs_flattened",
                        "task_obs_flattened",
                        "latents_processed",
                    ],
                    models=[
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
                        MLPWithConcatConfig(
                            in_keys=["latents"],
                            out_keys=["latents_processed"],
                            num_out=ase_parameters.latent_dim,
                            layers=[
                                MLPLayerConfig(units=512, activation="relu"),
                                MLPLayerConfig(units=256, activation="relu"),
                            ],
                        ),
                    ],
                ),
                MLPWithConcatConfig(
                    in_keys=[
                        "max_coords_obs_flattened",
                        "task_obs_flattened",
                        "latents_processed",
                    ],
                    out_keys=["actor_trunk_out"],
                    num_out=robot_config.number_of_actions,
                    layers=[
                        MLPLayerConfig(units=1024, activation="relu"),
                        MLPLayerConfig(units=1024, activation="relu"),
                        MLPLayerConfig(units=512, activation="relu"),
                    ],
                ),
            ],
        ),
    )

    critic_config = ModuleContainerConfig(
        in_keys=policy_in_keys,
        out_keys=["value"],
        models=[
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
            MLPWithConcatConfig(
                in_keys=["max_coords_obs_flattened", "task_obs_flattened", "latents"],
                out_keys=["value"],
                num_out=1,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu"),
                    MLPLayerConfig(units=1024, activation="relu"),
                    MLPLayerConfig(units=512, activation="relu"),
                ],
            ),
        ],
    )

    # Mirrors ase/mlp.py exactly (unconditional discriminator variant).
    from protomotions.agents.common.config import (
        ModuleOperationSphereProjectionConfig,
    )

    discriminator_encoder_config = ASEDiscriminatorEncoderConfig(
        encoder_out_size=ase_parameters.latent_dim,
        in_keys=["historical_max_coords_obs", "latents"],
        out_keys=["disc_logits", "mi_enc_output"],
        models=[
            MLPWithConcatConfig(
                in_keys=["historical_max_coords_obs"],
                normalize_obs=True,
                norm_clamp_value=5,
                out_keys=["trunk_features"],
                num_out=512,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu"),
                    MLPLayerConfig(units=1024, activation="relu"),
                ],
            ),
            ModuleContainerConfig(
                in_keys=["trunk_features"],
                out_keys=["disc_logits", "mi_enc_output"],
                models=[
                    MLPWithConcatConfig(
                        in_keys=["trunk_features"],
                        out_keys=["disc_logits"],
                        num_out=1,
                        layers=[
                            MLPLayerConfig(units=512, activation="relu"),
                            MLPLayerConfig(units=256, activation="relu"),
                        ],
                    ),
                    MLPWithConcatConfig(
                        in_keys=["trunk_features"],
                        out_keys=["mi_enc_output"],
                        num_out=ase_parameters.latent_dim,
                        layers=[],
                        module_operations=[
                            ModuleOperationForwardConfig(),
                            ModuleOperationSphereProjectionConfig(),
                        ],
                    ),
                ],
            ),
        ],
    )

    disc_critic_config = ModuleContainerConfig(
        in_keys=["historical_max_coords_obs"],
        out_keys=["disc_value"],
        models=[
            MLPWithConcatConfig(
                in_keys=["historical_max_coords_obs"],
                out_keys=["disc_value"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=1,
                layers=[
                    MLPLayerConfig(units=512, activation="relu"),
                    MLPLayerConfig(units=256, activation="relu"),
                ],
            )
        ],
    )

    mi_critic_config = ModuleContainerConfig(
        in_keys=["historical_max_coords_obs"],
        out_keys=["mi_value"],
        models=[
            MLPWithConcatConfig(
                in_keys=["historical_max_coords_obs"],
                out_keys=["mi_value"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=1,
                layers=[
                    MLPLayerConfig(units=512, activation="relu"),
                    MLPLayerConfig(units=256, activation="relu"),
                ],
            )
        ],
    )

    reference_obs_components = {
        "historical_max_coords_obs": MdpComponent(
            compute_func=compute_historical_max_coords_from_motion_lib,
            dynamic_vars={},
            static_params={
                "history_steps": HISTORY_STEPS,
                "local_obs": True,
                "root_height_obs": True,
            },
        ),
    }

    return LeagueASEAgentConfig(
        model=ASEModelConfig(
            in_keys=["max_coords_obs", "historical_max_coords_obs", "task_obs", "latents"],
            out_keys=[
                "action", "mean_action", "neglogp", "value",
                "disc_logits", "mi_enc_output", "disc_value", "mi_value",
            ],
            actor=actor_config,
            critic=critic_config,
            discriminator=discriminator_encoder_config,
            disc_critic=disc_critic_config,
            mi_critic=mi_critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
            discriminator_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
            disc_critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
            mi_critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        ),
        reference_obs_components=reference_obs_components,
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        # Battle rewards drive the fight; the corpus discriminator is the
        # naturalness anchor (ASE's analog of the GPC prior constraint).
        task_reward_w=1.0,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        amp_parameters=AMPParametersConfig(
            discriminator_reward_w=0.3,
            discriminator_reward_threshold=0.0,  # never terminate fights on style
            # Must not exceed the PPO minibatch (expert TDs are built at this
            # size from minibatch-sized tensors; default 4096 crashes when
            # batch_size < 4096).
            discriminator_batch_size=args.batch_size,
        ),
        ase_parameters=ase_parameters,
        league=LeagueParams(role="main"),
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
    # Exhibition-style viewing: one long match, viewer on (same as GPC league).
    env_cfg.max_episode_length = 100000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0
