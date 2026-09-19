# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployable MaskedMimic: the prior and trunk see only what a real robot can.

Same student as transformer.py -- a VAE with a privileged encoder and a
learned prior -- but the two modules that actually run on hardware are cut
over to proprioception:

    prior + trunk :  noisy reduced coords (joint pos/vel, IMU orientation,
                     IMU gyro) with NO root height and NO root linear
                     velocity, plus a reduced-coords history window.
    encoder       :  UNCHANGED, still privileged max coords. It is training
                     only -- forward_inference() calls prior -> trunk and
                     never touches it -- so the extra information never
                     reaches the robot.

The expert is unchanged too, and does NOT need to be deployable: distillation
runs it in sim to produce target actions, and apply_inference_overrides drops
it. A privileged teacher is a better teacher.

The masked_mimic_target_* observations stay as they are. They are the COMMAND,
constructed rather than sensed -- on hardware you author the target poses and
get the current body poses from forward kinematics on the joint encoders plus
IMU orientation.

Sim2real follows quadruped_bm_deploy.py: observation noise, action noise,
friction/COM randomisation and pushes, so the prior learns to work through a
sensor model rather than an oracle.

    python protomotions/train_agent.py --robot-name go2 \\
        --simulator isaaclab --physics physx --headless \\
        --motion-file data/motions/go2/go2_flat_mirrored_balanced.pt \\
        --experiment-path examples/experiments/masked_mimic/transformer_deploy.py \\
        --expert-model-path results/go2_tracker_v1/last.ckpt \\
        --num-envs 2048 --batch-size 8192 \\
        --experiment-name go2_masked_mimic_deploy_v1

Drives with the velocity-command harness unchanged
(examples/experiments/masked_mimic/steering.py): that only swaps control
components and rewards, which is independent of the observation set.
"""

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.supervised.masked_mimic_config import MaskedMimicSupervisedAgentConfig
import argparse


# Global configuration for masked mimic
NUM_FUTURE_STEPS = 5
TOTAL_STORED_HISTORICAL_STEPS = 5  # How many historical steps we save
NUM_HISTORICAL_CONDITIONED_STEPS = 5  # From those, how many do we sub-sample


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    """Add MaskedMimic-specific CLI arguments."""
    parser.add_argument(
        "--expert-model-path",
        type=str,
        default=None,
        help="Path to expert model checkpoint for distillation training"
    )


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
    
    Uses MdpComponent-based component configuration with explicit context bindings:
        MdpComponent(compute_func=compute_fn, dynamic_vars={...}, static_params={...})
    """
    import torch
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.control.masked_mimic_control import MaskedMimicControlConfig
    from protomotions.envs.component_factories import (
        max_coords_obs_factory,
        reduced_coords_obs_factory,
        historical_reduced_coords_obs_factory,
        historical_max_coords_obs_factory,
        previous_actions_factory,
        mimic_target_poses_max_coords_factory,
        mimic_tracking_rewards_factory,
        action_smoothness_factory,
        tracking_error_term_factory,
    )
    
    # Import compute kernels for masked_mimic-specific observations
    from protomotions.envs.obs import (
        compute_historical_poses_with_time,
        compute_target_poses_only,
        compute_target_masks_only,
        compute_target_time_offsets,
    )
    from protomotions.envs.action import make_pd_action_config
    from protomotions.envs.obs import to_float

    control_components = {
        "masked_mimic": MaskedMimicControlConfig(
            num_masked_future_steps=NUM_FUTURE_STEPS,
            future_steps=1,  # Might be increased if expert requires more
            bootstrap_on_episode_end=True,
            time_alpha=2.0,
            time_beta=5.0,
            repeat_mask_probability=0.8,
            force_max_conditioned_bodies_prob=0.1,
            force_small_num_conditioned_bodies_prob=0.1,
            visible_target_pose_prob=0.8
        ),
    }

    # Compute conditionable body IDs from robot config
    conditionable_body_ids = torch.tensor([
        robot_cfg.kinematic_info.body_names.index(name)
        for name in robot_cfg.trackable_bodies_subset
    ], dtype=torch.long)

    # Observation components
    observation_components = {
        # --- Deployable proprioception: what the prior and trunk consume. ---
        # NO root height and NO root linear velocity -- the two quantities a
        # real quadruped cannot measure. Noisy variants so the policy learns
        # through the sensor model; with domain randomisation stripped at
        # inference these return clean values at the same width.
        "noisy_reduced_coords_obs": reduced_coords_obs_factory(
            use_noisy=True,
            root_height_obs=False,
            root_vel_obs=False,
        ),
        "historical_reduced_coords_obs": historical_reduced_coords_obs_factory(
            use_noisy=True,
        ),
        # --- Privileged: encoder (training only) and the tracking rewards. ---
        "max_coords_obs": max_coords_obs_factory(
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
        ),
        "historical_max_coords_obs": historical_max_coords_obs_factory(
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
        ),
        "previous_actions": previous_actions_factory(history_steps=1),
        "mimic_target_poses": mimic_target_poses_max_coords_factory(
            with_velocities=True,
            with_relative=True,
        ),
        "masked_mimic_target_poses": MdpComponent(
            compute_func=compute_target_poses_only,
            dynamic_vars={
                "current_state_body_pos": EnvContext.current.rigid_body_pos,
                "current_state_body_rot": EnvContext.current.rigid_body_rot,
                "masked_mimic_ref_pos": EnvContext.masked_mimic.ref_pos,
                "masked_mimic_ref_rot": EnvContext.masked_mimic.ref_rot,
                "masked_mimic_target_bodies_masks": EnvContext.masked_mimic.target_bodies_masks,
            },
            static_params={"conditionable_body_ids": conditionable_body_ids, "include_root_relative": True},
        ),
        "masked_mimic_target_masks": MdpComponent(
            compute_func=compute_target_masks_only,
            dynamic_vars={
                "masked_mimic_target_bodies_masks": EnvContext.masked_mimic.target_bodies_masks,
            },
            static_params={"conditionable_body_ids": conditionable_body_ids},
        ),
        "masked_mimic_target_times": MdpComponent(
            compute_func=compute_target_time_offsets,
            dynamic_vars={
                "masked_mimic_time_offsets": EnvContext.masked_mimic.time_offsets,
            },
        ),
        # historical_pose_obs (privileged max-coords history) is deliberately
        # GONE: the prior reads historical_reduced_coords_obs instead, nothing
        # else consumed it, and computing a full-body history every step for
        # an observation no module reads is pure cost -- and an invitation to
        # wire it back into the deployed path by mistake.
        "masked_mimic_target_poses_masks": MdpComponent(
            compute_func=to_float,
            dynamic_vars={
                "x": EnvContext.masked_mimic.target_poses_masks,
            },
        ),
        "masked_mimic_target_bodies_masks": MdpComponent(
            compute_func=to_float,
            dynamic_vars={
                "x": EnvContext.masked_mimic.target_bodies_masks,
            },
        ),
    }

    expert_model_path = getattr(args, 'expert_model_path', None)
    if expert_model_path:
        from protomotions.agents.supervised.expert_utils import (
            get_expert_observation_components,
        )
        from protomotions.utils.config_utils import (
            load_resolved_configs_from_checkpoint,
        )

        expert_configs = load_resolved_configs_from_checkpoint(expert_model_path)
        expert_env_config = expert_configs["env"]
        expert_agent_config = expert_configs["agent"]
        
        expert_history_steps = getattr(expert_env_config, 'num_state_history_steps', 0)
        assert TOTAL_STORED_HISTORICAL_STEPS >= expert_history_steps, (
            f"Insufficient history: current={TOTAL_STORED_HISTORICAL_STEPS}, expert requires={expert_history_steps}"
        )
        
        if hasattr(expert_env_config, 'control_components') and expert_env_config.control_components:
            for ctrl_cfg in expert_env_config.control_components.values():
                expert_num_future = getattr(ctrl_cfg, 'future_steps', None)
                if expert_num_future is not None:
                    masked_mimic_cfg = control_components["masked_mimic"]
                    if masked_mimic_cfg.future_steps < expert_num_future:
                        masked_mimic_cfg.future_steps = expert_num_future
        
        expert_obs_components = get_expert_observation_components(
            expert_env_config,
            expert_agent_config,
            existing_obs_keys=list(observation_components.keys()),
        )
        observation_components.update(expert_obs_components)

    # Termination components configuration
    termination_components = {
        "tracking_error": tracking_error_term_factory(threshold=0.25),
    }

    # Reward components
    reward_components = {
        **mimic_tracking_rewards_factory(
            gt_weight=0.5,
            gr_weight=0.3,
            gt_coef=-100.0,
            gr_coef=-5.0,
        ),
        "action_smoothness": action_smoothness_factory(weight=-0.02),
    }

    env_config: EnvConfig = EnvConfig(
        max_episode_length=1000,
        num_state_history_steps=TOTAL_STORED_HISTORICAL_STEPS,  # Historical obs for masked mimic prior
        # Component-based configuration
        control_components=control_components,
        observation_components=observation_components,
        termination_components=termination_components,
        reward_components=reward_components,
        action_config=make_pd_action_config(robot_cfg),
        # Motion manager configuration
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
        ),
    )

    return env_config


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> MaskedMimicSupervisedAgentConfig:
    from protomotions.agents.supervised.masked_mimic_config import (
        MaskedMimicModelConfig,
        MaskedMimicVAEConfig,
        VAENoiseType,
        KLDScheduleConfig,
    )
    from protomotions.agents.common.config import (
        ObsProcessorConfig,
        MLPLayerConfig,
        ModuleContainerConfig,
        MLPWithConcatConfig,
        TransformerConfig,
        ModuleOperationReshapeConfig,
        ModuleOperationForwardConfig,
    )
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import MimicEvaluatorConfig
    from protomotions.envs.component_factories import gt_error_factory, gr_error_factory, max_joint_error_factory

    transformer_token_size = 512
    transformer_encoder_widths = 256
    vae_latent_dim = 64

    # Encoder: normalizes inputs → MLP trunk → mu/logvar heads (flat structure)
    encoder_config = ModuleContainerConfig(
        in_keys=["max_coords_obs", "mimic_target_poses", "masked_mimic_target_poses", "masked_mimic_target_bodies_masks", "masked_mimic_target_times", "masked_mimic_target_poses_masks"],
        out_keys=["encoder_mu", "encoder_logvar"],
        models=[
            # Normalizers (parallel - no dependencies)
            ObsProcessorConfig(
                in_keys=["max_coords_obs"],
                out_keys=["max_coords_obs_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["mimic_target_poses"],
                out_keys=["mimic_target_poses_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_poses"],
                out_keys=["masked_mimic_target_poses_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_times"],
                out_keys=["masked_mimic_target_times_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            # Trunk MLP (depends on normalizers)
            MLPWithConcatConfig(
                in_keys=["max_coords_obs_norm", "mimic_target_poses_norm", "masked_mimic_target_poses_norm", "masked_mimic_target_bodies_masks", "masked_mimic_target_times_norm", "masked_mimic_target_poses_masks"],
                out_keys=["encoder_trunk_out"],
                num_out=512,
                layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(5)],
                output_activation="relu",
            ),
            # Output heads (parallel - both depend on trunk)
            MLPWithConcatConfig(
                in_keys=["encoder_trunk_out"],
                out_keys=["encoder_mu"],
                num_out=vae_latent_dim,
                layers=[MLPLayerConfig(units=256, activation="relu"), MLPLayerConfig(units=128, activation="relu")],
            ),
            MLPWithConcatConfig(
                in_keys=["encoder_trunk_out"],
                out_keys=["encoder_logvar"],
                num_out=vae_latent_dim,
                layers=[MLPLayerConfig(units=256, activation="relu"), MLPLayerConfig(units=128, activation="relu")],
            ),
        ],
    )

    # Prior: reshape inputs → encode to tokens → transformer → mu/logvar heads (flat structure)
    prior_config = ModuleContainerConfig(
        in_keys=["noisy_reduced_coords_obs", "masked_mimic_target_poses", "masked_mimic_target_masks", "masked_mimic_target_times", "masked_mimic_target_poses_masks", "historical_reduced_coords_obs"],
        out_keys=["prior_mu", "prior_logvar"],
        models=[
            # Reshape and normalize inputs (parallel)
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_poses"],
                out_keys=["target_poses_seq"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[
                    ModuleOperationReshapeConfig(new_shape=["batch_size", NUM_FUTURE_STEPS, -1]),
                    ModuleOperationForwardConfig(),
                ],
            ),
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_masks"],
                out_keys=["target_masks_seq"],
                normalize_obs=False,
                module_operations=[ModuleOperationReshapeConfig(new_shape=["batch_size", NUM_FUTURE_STEPS, -1])],
            ),
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_times"],
                out_keys=["target_times_seq"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[
                    ModuleOperationReshapeConfig(new_shape=["batch_size", NUM_FUTURE_STEPS, -1]),
                    ModuleOperationForwardConfig(),
                ],
            ),
            ObsProcessorConfig(
                in_keys=["historical_reduced_coords_obs"],
                out_keys=["historical_pose_obs_seq"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[
                    ModuleOperationReshapeConfig(new_shape=["batch_size", NUM_HISTORICAL_CONDITIONED_STEPS, -1]),
                    ModuleOperationForwardConfig(),
                ],
            ),
            # Token encoders (depend on reshaped inputs)
            MLPWithConcatConfig(
                in_keys=["noisy_reduced_coords_obs"],
                out_keys=["current_state_token"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=transformer_token_size,
                layers=[MLPLayerConfig(units=transformer_encoder_widths, activation="relu") for _ in range(2)],
                module_operations=[
                    ModuleOperationReshapeConfig(new_shape=["batch_size", 1, -1]),
                    ModuleOperationForwardConfig(),
                ],
            ),
            MLPWithConcatConfig(
                in_keys=["target_poses_seq", "target_masks_seq", "target_times_seq"],
                out_keys=["masked_mimic_target_poses_token"],
                normalize_obs=False,
                num_out=transformer_token_size,
                layers=[MLPLayerConfig(units=transformer_encoder_widths, activation="relu") for _ in range(2)],
                module_operations=[
                    ModuleOperationReshapeConfig(new_shape=["batch_size", NUM_FUTURE_STEPS, -1]),
                    ModuleOperationForwardConfig(),
                ],
            ),
            MLPWithConcatConfig(
                in_keys=["historical_pose_obs_seq"],
                out_keys=["historical_pose_obs_token"],
                normalize_obs=False,
                num_out=transformer_token_size,
                layers=[MLPLayerConfig(units=transformer_encoder_widths, activation="relu") for _ in range(2)],
                module_operations=[
                    ModuleOperationReshapeConfig(new_shape=["batch_size", NUM_HISTORICAL_CONDITIONED_STEPS, -1]),
                    ModuleOperationForwardConfig(),
                ],
            ),
            # Transformer (depends on tokens)
            TransformerConfig(
                in_keys=["current_state_token", "masked_mimic_target_poses_token", "historical_pose_obs_token", "masked_mimic_target_poses_masks"],
                out_keys=["transformer_out"],
                transformer_token_size=transformer_token_size,
                latent_dim=transformer_token_size,
                input_and_mask_mapping={"masked_mimic_target_poses_token": "masked_mimic_target_poses_masks"},
                output_activation="relu",
            ),
            # Output heads (parallel - both depend on transformer)
            MLPWithConcatConfig(
                in_keys=["transformer_out"],
                out_keys=["prior_mu"],
                num_out=vae_latent_dim,
                layers=[MLPLayerConfig(units=256, activation="relu"), MLPLayerConfig(units=128, activation="relu")],
            ),
            MLPWithConcatConfig(
                in_keys=["transformer_out"],
                out_keys=["prior_logvar"],
                num_out=vae_latent_dim,
                layers=[MLPLayerConfig(units=256, activation="relu"), MLPLayerConfig(units=128, activation="relu")],
            ),
        ],
    )

    # Trunk: normalize inputs → MLP → actions (flat structure)
    trunk_config = ModuleContainerConfig(
        in_keys=["noisy_reduced_coords_obs", "previous_actions", "vae_latent"],
        out_keys=["actor_trunk_out"],
        models=[
            # Normalizers (parallel)
            ObsProcessorConfig(
                in_keys=["noisy_reduced_coords_obs"],
                out_keys=["reduced_coords_obs_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["previous_actions"],
                out_keys=["previous_actions_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            # Output MLP (depends on normalizers + vae_latent)
            MLPWithConcatConfig(
                in_keys=["reduced_coords_obs_norm", "previous_actions_norm", "vae_latent"],
                out_keys=["actor_trunk_out"],
                num_out=robot_config.number_of_actions,
                layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(3)],
            ),
        ],
    )

    # Main model configuration
    model_config = MaskedMimicModelConfig(
        encoder=encoder_config,
        prior=prior_config,
        trunk=trunk_config,
        vae=MaskedMimicVAEConfig(
            vae_latent_dim=vae_latent_dim,
            vae_noise_type=VAENoiseType.NORMAL,
            kld_schedule=KLDScheduleConfig(start_epoch=500, end_epoch=2000),
        ),
        optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
    )

    evaluator_config = MimicEvaluatorConfig(
        evaluation_components={
            "gt_error": gt_error_factory(threshold=0.25),
            "gr_error": gr_error_factory(),
            "max_joint_error": max_joint_error_factory(),
        },
    )

    # Get expert model path from args (set via --expert-model-path CLI argument)
    expert_model_path = getattr(args, 'expert_model_path', None)
    
    # Agent configuration
    agent_config = MaskedMimicSupervisedAgentConfig(
        model=model_config,
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        num_mini_epochs=6,
        evaluator=evaluator_config,
        expert_model_path=expert_model_path,
    )
    return agent_config


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Sim2real noise and domain randomisation, lifted from quadruped_bm_deploy.

    observation_noise is what makes the noisy_* observations noisy at all: the
    state-history buffer only allocates noisy storage when
    simulator.domain_randomization.observation_noise is set (env.py:281), so
    without this the deployable path would train on an oracle wearing a
    sensor-model label.
    """
    from protomotions.simulator.base_simulator.config import (
        ActionNoiseDomainRandomizationConfig,
        CenterOfMassDomainRandomizationConfig,
        DomainRandomizationConfig,
        FrictionDomainRandomizationConfig,
        PushDomainRandomizationConfig,
    )
    from protomotions.robot_configs.base import RobotNoiseConfig

    robot_cfg.reset_noise = RobotNoiseConfig(
        dof_pos_noise=0.1,
        root_pos_noise=[0.05, 0.05, 0.01],
        root_rot_noise=[0.1, 0.1, 0.2],
        root_vel_noise=[0.1, 0.1, 0.05],
        root_ang_vel_noise=[0.1, 0.1, 0.1],
    )

    simulator_cfg.domain_randomization = DomainRandomizationConfig(
        action_noise=ActionNoiseDomainRandomizationConfig(
            action_noise_range=(-0.025, 0.025), dof_names=[".*"], dof_indices=None
        ),
        friction=FrictionDomainRandomizationConfig(
            num_buckets=64,
            static_friction_range=(0.3, 1.6),
            dynamic_friction_range=(0.3, 1.2),
            restitution_range=(0.0, 0.5),
            body_names=[".*"],
            body_indices=None,
        ),
        center_of_mass=CenterOfMassDomainRandomizationConfig(
            com_range={"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            body_names=robot_cfg.common_naming_to_robot_body_names["torso_body_name"],
            body_indices=None,
        ),
        observation_noise=RobotNoiseConfig(
            dof_pos_noise=0.01,
            dof_vel_noise=0.5,
            anchor_ang_vel_noise=0.2,
            anchor_rot_noise=0.05,
        ),
        push=PushDomainRandomizationConfig(
            push_interval_range=(1.0, 3.0),
            max_linear_velocity=(0.5, 0.5, 0.2),
            max_angular_velocity=(0.52, 0.52, 0.78),
        ),
    )


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg: EnvConfig,
    agent_cfg: MaskedMimicSupervisedAgentConfig,
    terrain_cfg,
    motion_lib_cfg,
    scene_lib_cfg,
    args: argparse.Namespace,
):
    """Apply evaluation-specific overrides."""
    # Reuse the mimic apply_inference_overrides function from mimic.mlp
    from protomotions.utils.config_utils import (
        import_experiment_relative_eval_overrides,
    )

    apply_inference_overrides_fn = import_experiment_relative_eval_overrides(
        "../mimic/mlp.py"
    )
    apply_inference_overrides_fn(robot_cfg, simulator_cfg, env_cfg, agent_cfg, terrain_cfg, motion_lib_cfg, scene_lib_cfg, args)

    if agent_cfg is not None and hasattr(agent_cfg, "expert_model_path"):
        expert_model_path = agent_cfg.expert_model_path
        
        # Remove expert observation components
        if expert_model_path is not None and env_cfg is not None:
            if hasattr(env_cfg, "observation_components") and env_cfg.observation_components is not None:
                from protomotions.agents.supervised.expert_utils import (
                    get_expert_observation_keys,
                )
                from protomotions.utils.config_utils import (
                    load_resolved_configs_from_checkpoint,
                )

                expert_configs = load_resolved_configs_from_checkpoint(expert_model_path)
                expert_obs_keys = get_expert_observation_keys(expert_configs["env"], expert_configs["agent"])
                for key in expert_obs_keys:
                    if key in env_cfg.observation_components:
                        del env_cfg.observation_components[key]
        
        agent_cfg.expert_model_path = None
