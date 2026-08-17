# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.ase.config import ASEAgentConfig
import argparse
import re

# Historical steps for discriminator reference observations
HISTORY_STEPS = 8


def _disc_body_ids(robot_cfg: RobotConfig):
    """Body indices the ASE discriminator/encoder sees, or None for all.

    Same restriction as examples/experiments/amp/mlp.py -- see the long note
    there. Short version: the discriminator should judge GAIT, and a body
    whose motion the policy cannot reproduce lets it win on jitter instead.
    On the raptor, showing it all 71 bodies (including 36 digit segments at
    ~1.4 g driven by 6 N.m actuators) made agent and reference trivially
    separable; the policy learned to stand and nothing more. Restricting the
    discriminator is what made AMP walk, and ASE has exactly the same
    failure mode because its encoder reads the same historical obs.

    The POLICY still senses every body (max_coords_obs is untouched).
    """
    subset = getattr(robot_cfg, "disc_bodies_subset", None)
    if not subset:
        return None
    names = list(robot_cfg.kinematic_info.body_names)
    missing = [b for b in subset if b not in names]
    if missing:
        raise ValueError(
            f"disc_bodies_subset names bodies that do not exist on "
            f"{type(robot_cfg).__name__}: {missing}")
    # The list is EXPLICIT -- digit tips included by name. It used to append
    # them with a regex for the raptor's Index/Middle/Ring3, which silently
    # matched nothing on the tiger (whose tips are Digit<n>2), leaving every
    # tiger toe unjudged. A body the discriminator cannot see is one the
    # policy is free to exploit: hiding the raptor's digits taught the first
    # walker to plant its fingertips for free support.
    return sorted({0} | {names.index(b) for b in subset})


def terrain_config(args: argparse.Namespace):
    """Build terrain configuration."""
    from protomotions.components.terrains.config import TerrainConfig

    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    """Build scene library configuration."""
    from protomotions.components.scene_lib import SceneLibConfig

    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    """Build motion library configuration."""
    from protomotions.components.motion_lib import MotionLibConfig

    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Build environment configuration (training defaults).
    
    Uses factory functions from protomotions.envs.component_factories for common components.
    """
    from protomotions.envs.motion_manager.config import MotionManagerConfig
    from protomotions.envs.action import make_pd_action_config
    from protomotions.envs.component_factories import (
        max_coords_obs_factory,
        historical_max_coords_obs_factory,
        pow_rew_factory,
    )

    # Observation components configuration
    observation_components = {
        # Humanoid self-observations (current state)
        "max_coords_obs": max_coords_obs_factory(
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
        ),
        # Historical observations for AMP/ASE discriminator (from StateHistoryBuffer)
        "historical_max_coords_obs": historical_max_coords_obs_factory(
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
            history_steps=HISTORY_STEPS,
            body_ids=_disc_body_ids(robot_cfg),
        ),
    }

    # Mechanical power penalty (|τ·q̇|): nudges get-ups / thrashing toward
    # cheaper motions. Needs a non-zero agent task_reward_w to affect PPO.
    reward_components = {
        "pow_rew": pow_rew_factory(weight=-1e-5, min_value=-0.5),
    }

    env_config: EnvConfig = EnvConfig(
        max_episode_length=300,  # Training default (eval override applied automatically)
        num_state_history_steps=HISTORY_STEPS,  # Historical obs for AMP/ASE discriminator
        observation_components=observation_components,
        reward_components=reward_components,
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MotionManagerConfig(
            init_start_prob=0.5  # Bias agent to start at the beginning of the motion to prevent getting stuck in a local-minima (standing still).
        ),
    )

    return env_config


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> ASEAgentConfig:
    from protomotions.agents.common.config import (
        ModuleContainerConfig,
        ObsProcessorConfig,
        MLPWithConcatConfig,
        MLPLayerConfig,
        ModuleOperationForwardConfig,
        ModuleOperationSphereProjectionConfig,
    )
    from protomotions.agents.ppo.config import PPOActorConfig
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.ase.config import (
        ASEParametersConfig,
        ASEDiscriminatorEncoderConfig,
    )
    from protomotions.agents.amp.config import AMPParametersConfig
    from protomotions.agents.ase.config import ASEModelConfig
    from protomotions.envs.obs import compute_historical_max_coords_from_motion_lib
    from protomotions.envs.mdp_component import MdpComponent

    conditional_discriminator = False

    ase_parameters = ASEParametersConfig(
        conditional_discriminator=conditional_discriminator,
        latent_dim=64,
        mi_reward_w=0.5,
        mi_hypersphere_reward_shift=True,
        diversity_bonus=0.01 if not conditional_discriminator else 0.0,
        latent_uniformity_weight=0.0 if not conditional_discriminator else 0.01,
    )

    actor_config = PPOActorConfig(
        mu_key="actor_trunk_out",
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=["max_coords_obs", "latents"],
        mu_model=ModuleContainerConfig(
            in_keys=["max_coords_obs", "latents"],
            out_keys=["actor_trunk_out"],
            models=[
                ModuleContainerConfig(
                    in_keys=["max_coords_obs", "latents"],
                    out_keys=["max_coords_obs_flattened", "latents_processed"],
                    models=[
                        ObsProcessorConfig(
                            in_keys=["max_coords_obs"],
                            out_keys=["max_coords_obs_flattened"],
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
                    in_keys=["max_coords_obs_flattened", "latents_processed"],
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
        in_keys=["max_coords_obs", "latents"],
        out_keys=["value"],
        models=[
            ObsProcessorConfig(
                in_keys=["max_coords_obs"],
                out_keys=["max_coords_obs_flattened"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            MLPWithConcatConfig(
                in_keys=["max_coords_obs_flattened", "latents"],
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

    # Build discriminator keys based on conditional flag
    disc_head_in_keys = ["trunk_features"]
    if conditional_discriminator:
        disc_head_in_keys.append("latents")

    from protomotions.agents.common.config import ModuleContainerConfig

    discriminator_encoder_config = (
        ASEDiscriminatorEncoderConfig(  # This is a sequential module config
            encoder_out_size=ase_parameters.latent_dim,
            in_keys=["historical_max_coords_obs"]
            + (["latents"] if not conditional_discriminator else []),
            out_keys=["disc_logits", "mi_enc_output"],
            models=[  # Models in the sequential module
                # Trunk: process historical_max_coords_obs_factory to features
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
                # Multi-output: discriminator + MI encoder from trunk features
                ModuleContainerConfig(
                    in_keys=disc_head_in_keys,
                    out_keys=["disc_logits", "mi_enc_output"],
                    models=[
                        # Discriminator head
                        MLPWithConcatConfig(
                            in_keys=disc_head_in_keys,
                            out_keys=["disc_logits"],
                            num_out=1,
                            layers=[
                                MLPLayerConfig(units=512, activation="relu"),
                                MLPLayerConfig(units=256, activation="relu"),
                            ],
                        ),
                        # MI Encoder head with sphere projection
                        MLPWithConcatConfig(
                            in_keys=["trunk_features"],
                            out_keys=["mi_enc_output"],
                            num_out=ase_parameters.latent_dim,
                            layers=[],  # Single projection layer
                            module_operations=[
                                ModuleOperationForwardConfig(),
                                ModuleOperationSphereProjectionConfig(),
                            ],
                        ),
                    ],
                ),
            ],
        )
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

    # Reference observation components for discriminator expert samples
    # Agent injects motion_lib/motion_ids/motion_times/dt at runtime (not in EnvContext)
    reference_obs_components = {
        "historical_max_coords_obs": MdpComponent(
            compute_func=compute_historical_max_coords_from_motion_lib,
            dynamic_vars={},  # All parameters injected by agent
            static_params={
                "history_steps": HISTORY_STEPS,
                "local_obs": True,
                "root_height_obs": True,
                # must match the agent-side restriction above, or the
                # discriminator compares different-length vectors
                "body_ids": _disc_body_ids(robot_config),
            },
        ),
    }

    agent_config: ASEAgentConfig = ASEAgentConfig(
        model=ASEModelConfig(
            in_keys=["max_coords_obs", "historical_max_coords_obs", "latents"],
            out_keys=[
                "action",
                "mean_action",
                "neglogp",
                "value",
                "disc_logits",
                "mi_enc_output",
                "disc_value",
                "mi_value",
            ],
            actor=actor_config,
            critic=critic_config,
            discriminator=discriminator_encoder_config,
            disc_critic=disc_critic_config,
            mi_critic=mi_critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
            discriminator_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam", lr=1e-4
            ),
            disc_critic_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam", lr=1e-4
            ),
            mi_critic_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam", lr=1e-4
            ),
        ),
        reference_obs_components=reference_obs_components,
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        # Small weight so pow_rew regularizes without drowning disc/MI style.
        task_reward_w=0.1,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        amp_parameters=AMPParametersConfig(
            discriminator_reward_w=0.5,
            discriminator_reward_threshold=0.05,  # Training default (eval override in apply_inference_overrides if needed)
        ),
        ase_parameters=ase_parameters,
    )
    # Warm-start controls (same semantics as examples/experiments/amp/mlp.py):
    # freeze the EMA obs normalizer so the inherited policy's inputs stay
    # calibrated, and allow disabling the disc-reward kill, which guillotines
    # converged walkers whose disc reward floor sits under any threshold.
    agent_config.freeze_actor_obs_norm = bool(
        getattr(args, "freeze_actor_obs_norm", False)
    )
    if getattr(args, "disc_term_threshold", None) is not None:
        agent_config.amp_parameters.discriminator_reward_threshold = float(
            args.disc_term_threshold
        )
    agent_config.amp_parameters.discriminator_termination_decay_epochs = int(
        getattr(args, "disc_term_decay_epochs", 0) or 0
    )
    return agent_config


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
    """Apply evaluation-specific overrides."""
    # Reuse the amp apply_inference_overrides function
    from protomotions.utils.config_utils import (
        import_experiment_relative_eval_overrides,
    )

    apply_inference_overrides_fn = import_experiment_relative_eval_overrides(
        "../amp/mlp.py"
    )
    apply_inference_overrides_fn(robot_cfg, simulator_cfg, env_cfg, agent_cfg, terrain_cfg, motion_lib_cfg, scene_lib_cfg, args)

    # ASE viewing default (per Eric): spawn from a random time inside a random
    # clip on every reset, not the clip's first frame (AMP's override above
    # sets 1.0 = always initial pose; 0.0 = always random time).
    if env_cfg is not None and hasattr(env_cfg, "motion_manager"):
        if hasattr(env_cfg.motion_manager, "init_start_prob"):
            env_cfg.motion_manager.init_start_prob = 0.0
        if hasattr(env_cfg.motion_manager, "resample_on_reset"):
            env_cfg.motion_manager.resample_on_reset = True


def additional_experiment_arguments(parser):
    parser.add_argument(
        "--freeze-actor-obs-norm", action="store_true",
        help="Pin the warm-started policy's input normalization (the EMA "
             "normalizer otherwise re-centers within epochs and collapses "
             "the inherited behavior while the weights stay intact).")
    parser.add_argument(
        "--disc-term-decay-epochs", type=int, default=0,
        help="Decay --disc-term-threshold linearly to 0 over N epochs, then "
             "no style kills at all (0 = constant for the whole run). Lethal "
             "early so standing still cannot pay, off before it can "
             "guillotine a converged walker.")
    parser.add_argument(
        "--disc-term-threshold", type=float, default=None,
        help="Override discriminator_reward_threshold (0 disables the style "
             "kill entirely -- required for warm starts from a CONVERGED "
             "disc, whose reward floor sits under any useful threshold).")
