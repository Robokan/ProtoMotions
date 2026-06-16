# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""BeyondMimic deployable tracker for quadrupeds (Go2, ANYmal-D).

The quadruped counterpart of ``mlp_bm_l2c2.py`` (the G1 "bones deploy" config),
intended for robots that run on real hardware. Identical architecture; the only
robot-specific value is the fall (anchor-height) threshold, which is scaled by
the robot's nominal standing height so it works for both the small Go2
(~0.34 m base) and the larger ANYmal-D (~0.6 m base).

Deployable by construction:
- ACTOR sees only ``reduced_coords`` proprioception (joint pos/vel, projected
  gravity, local angular velocity) with ``root_height_obs=False`` and
  ``root_vel_obs=False`` -- i.e. NO root height and NO root linear velocity,
  the two quantities a real quadruped cannot measure directly -- plus the
  reference as a root-relative reduced-coords future trajectory and the previous
  processed actions. All actor obs have a noisy (sensor-model) variant.
- CRITIC is privileged (full max-coords + clean state) -- asymmetric
  actor-critic; the critic's extra information never reaches the deployed policy.
- L2C2 (Kobayashi 2022) penalises MSE(mu_noisy, mu_clean) for robustness to
  sensor noise. Domain randomisation (friction, COM, action noise, pushes) and
  observation noise are on for sim2real.

Export to ONNX (obs computation baked in) with
``deployment/export_bm_tracker_onnx.py``.

The dm_control dog is sim-only (not deployable) and intentionally NOT a target
of this config -- train it with ``quadruped_mlp.py`` instead.
"""

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import (
    SimulatorConfig,
    ActionNoiseDomainRandomizationConfig,
    FrictionDomainRandomizationConfig,
    CenterOfMassDomainRandomizationConfig,
    RobotNoiseConfig,
    PushDomainRandomizationConfig,
    DomainRandomizationConfig,
)
from protomotions.components.terrains.config import (
    TerrainConfig,
    TerrainSimConfig,
    CombineMode,
)
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig
import argparse


def _fall_threshold(robot_cfg: RobotConfig) -> float:
    """Anchor height below which the episode terminates (a fall).

    Scaled to the robot's nominal standing height so the same config fits both
    Go2 (~0.34 m -> ~0.19) and ANYmal-D (~0.6 m -> ~0.33).
    """
    return round(0.55 * float(robot_cfg.default_root_height), 3)


def terrain_config(args: argparse.Namespace):
    """Build terrain configuration."""
    terrain_cfg = TerrainConfig(
        sim_config=TerrainSimConfig(
            static_friction=1,
            dynamic_friction=1,
            restitution=0.0,
            combine_mode=CombineMode.MULTIPLY,
        )
    )
    return terrain_cfg


def scene_lib_config(args: argparse.Namespace):
    """Build scene library configuration."""
    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    """Build motion library configuration."""
    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Build environment configuration with future target poses and L2C2 clean obs.

    Actor sees reduced coords + multi-horizon target poses (no XY offset, no root
    height, no root velocity). Additionally provides clean (noise-free)
    counterparts for L2C2 regularization. Rewards use BeyondMimic-style
    heading-invariant relative body tracking plus global anchor tracking.
    """
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.component_factories import (
        reduced_coords_obs_factory,
        mimic_target_poses_reduced_coords_factory,
        max_coords_obs_factory,
        mimic_target_poses_max_coords_factory,
        previous_actions_factory,
        action_smoothness_factory,
        global_anchor_ori_rew_factory,
        relative_body_pos_rew_factory,
        relative_body_ori_rew_factory,
        global_body_lin_vel_rew_factory,
        global_body_ang_vel_rew_factory,
        anchor_height_error_term_factory,
    )
    from protomotions.envs.rewards import compute_soft_pos_limit_rew
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.action import make_bm_pd_action_config

    # Control components configuration
    control_components = {
        "mimic": MimicControlConfig(
            bootstrap_on_episode_end=True,
            future_steps=[1, 2, 4, 8],
        )
    }

    # Observation components
    observation_components = {
        # Actor observations (noisy) — reduced coords proprioception. NO root
        # height / NO root linear velocity: the real robot cannot measure them.
        "noisy_reduced_coords_obs": reduced_coords_obs_factory(
            use_noisy=True,
            root_height_obs=False,
            root_vel_obs=False,
        ),
        # Actor target poses — reduced coords, no XY offset, multi-horizon
        "noisy_mimic_reduced_coords_target_poses": mimic_target_poses_reduced_coords_factory(
            use_noisy=True,
            include_dof_vel=True,
            include_xy_offset=False,
        ),
        # Clean counterparts for L2C2
        "clean_reduced_coords_obs": reduced_coords_obs_factory(
            use_noisy=False,
            root_height_obs=False,
            root_vel_obs=False,
        ),
        "clean_mimic_reduced_coords_target_poses": mimic_target_poses_reduced_coords_factory(
            use_noisy=False,
            include_dof_vel=True,
            include_xy_offset=False,
        ),
        # Critic observations (clean, privileged) — full max coords
        "max_coords_obs": max_coords_obs_factory(
            use_noisy=False,
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
        ),
        "mimic_max_coords_target_poses": mimic_target_poses_max_coords_factory(
            use_noisy=False,
            with_velocities=True,
            with_relative=True,
        ),
        # Common observations (processed actions after tanh/clamp)
        "historical_previous_processed_actions": previous_actions_factory(
            history_steps=1, processed=True
        ),
    }

    # Termination components
    termination_components = {
        "fall": anchor_height_error_term_factory(threshold=_fall_threshold(robot_cfg)),
    }

    # Reward components
    reward_components = {
        # Global anchor (root) orientation
        "global_anchor_ori": global_anchor_ori_rew_factory(weight=0.5, sigma=0.4),
        # Relative body position and orientation (region-weighted)
        "relative_body_pos": relative_body_pos_rew_factory(
            weight=1.0,
            sigma=0.3,
            use_region_weights=True,
        ),
        "relative_body_ori": relative_body_ori_rew_factory(
            weight=1.0,
            sigma=0.4,
            use_region_weights=True,
        ),
        # Global body velocities (region-weighted)
        "body_lin_vel": global_body_lin_vel_rew_factory(
            weight=1.0,
            sigma=1.0,
            use_region_weights=True,
        ),
        "body_ang_vel": global_body_ang_vel_rew_factory(
            weight=1.0,
            sigma=3.14,
            use_region_weights=True,
        ),
        "action_rate": action_smoothness_factory(weight=-0.1),
        "limits_dof_pos": MdpComponent(
            compute_func=compute_soft_pos_limit_rew,
            dynamic_vars={
                "dof_pos": EnvContext.current.dof_pos,
            },
            static_params={
                "weight": -10.0,
                "dof_limits_lower": robot_cfg.kinematic_info.dof_limits_lower,
                "dof_limits_upper": robot_cfg.kinematic_info.dof_limits_upper,
            },
        ),
    }

    return EnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        num_state_history_steps=1,
        control_components=control_components,
        observation_components=observation_components,
        termination_components=termination_components,
        reward_components=reward_components,
        action_config=make_bm_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
            realign_motion_with_humanoid_on_each_step=False,
        ),
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
    """Build agent configuration with L2C2 regularization."""
    from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
    from protomotions.agents.ppo.config import (
        PPOActorConfig,
        PPOModelConfig,
        AdaptiveLRConfig,
        AdvantageNormalizationConfig,
        L2C2Config,
    )
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import (
        MimicEvaluatorConfig,
        MotionWeightsRulesConfig,
    )
    from protomotions.envs.component_factories import (
        anchor_ori_metric_factory,
        relative_body_pos_metric_factory,
        anchor_height_error_metric_factory,
        gt_error_factory,
        gr_error_factory,
        max_joint_error_factory,
    )

    # Actor configuration — obs normalization ON, learnable std from -2.9
    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        learnable_std=True,
        in_keys=[
            "noisy_reduced_coords_obs",
            "noisy_mimic_reduced_coords_target_poses",
            "historical_previous_processed_actions",
        ],
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=[
                "noisy_reduced_coords_obs",
                "noisy_mimic_reduced_coords_target_poses",
                "historical_previous_processed_actions",
            ],
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
        ),
    )

    # Critic configuration — obs normalization ON
    critic_config = MLPWithConcatConfig(
        in_keys=[
            "max_coords_obs",
            "mimic_max_coords_target_poses",
            "historical_previous_processed_actions",
        ],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5.0,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )

    agent_config: PPOAgentConfig = PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=[
                # Noisy observations for actor
                "noisy_reduced_coords_obs",
                "noisy_mimic_reduced_coords_target_poses",
                # Clean observations for L2C2
                "clean_reduced_coords_obs",
                "clean_mimic_reduced_coords_target_poses",
                # Clean observations for critic
                "max_coords_obs",
                "mimic_max_coords_target_poses",
                # Shared observations (processed actions after tanh/clamp)
                "historical_previous_processed_actions",
            ],
            out_keys=["action", "mean_action", "neglogp", "value"],
            actor=actor_config,
            critic=critic_config,
            actor_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam", lr=2e-5, betas=(0.95, 0.99)
            ),
            critic_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam", lr=1e-4, betas=(0.95, 0.99)
            ),
        ),
        normalize_rewards=False,
        adaptive_lr=AdaptiveLRConfig(enabled=False),
        batch_size=args.batch_size,
        num_mini_epochs=2,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        l2c2=L2C2Config(
            enabled=True,
            lambda_l2c2=1.0,
            obs_pairs={
                "noisy_reduced_coords_obs": "clean_reduced_coords_obs",
                "noisy_mimic_reduced_coords_target_poses": "clean_mimic_reduced_coords_target_poses",
            },
        ),
        evaluator=MimicEvaluatorConfig(
            evaluation_components={
                "anchor_ori": anchor_ori_metric_factory(),
                "relative_body_pos": relative_body_pos_metric_factory(),
                "anchor_height_error": anchor_height_error_metric_factory(
                    threshold=_fall_threshold(robot_config)
                ),
                "gt_error": gt_error_factory(),
                "gr_error": gr_error_factory(),
                "max_joint_error": max_joint_error_factory(),
            },
            motion_weights_rules=MotionWeightsRulesConfig(
                motion_weights_update_success_discount=0.999,
                motion_weights_update_failure_discount=0,
            ),
        ),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True
        ),
    )
    return agent_config


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Configure robot and simulator for this experiment.

    Uses the robot config's default anchor (base_link for Go2, base for
    ANYmal-D). Adds reset noise and sim2real domain randomization.
    """
    robot_cfg.update_fields(
        contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"]
    )

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
    env_cfg,
    agent_cfg,
    terrain_cfg: TerrainConfig,
    motion_lib_cfg: MotionLibConfig,
    scene_lib_cfg: SceneLibConfig,
    args: argparse.Namespace,
):
    """Apply inference overrides.

    Removes clean L2C2 obs (not needed at inference since noise is disabled).
    No XY offset was used in training, so no special zeroing needed.
    """
    from protomotions.envs.component_factories import (
        reduced_coords_obs_factory,
        mimic_target_poses_reduced_coords_factory,
    )

    # Disable all termination components for inference
    if hasattr(env_cfg, "termination_components") and env_cfg.termination_components:
        env_cfg.termination_components = {}

    env_cfg.max_episode_length = 1000000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0

    terrain_cfg.sim_config = TerrainSimConfig(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        combine_mode=CombineMode.AVERAGE,
    )
    simulator_cfg.domain_randomization = None

    # Swap noisy observations for clean (no XY offset)
    env_cfg.observation_components["noisy_reduced_coords_obs"] = (
        reduced_coords_obs_factory(
            use_noisy=False,
            root_height_obs=False,
            root_vel_obs=False,
        )
    )
    env_cfg.observation_components["noisy_mimic_reduced_coords_target_poses"] = (
        mimic_target_poses_reduced_coords_factory(
            use_noisy=False,
            include_dof_vel=True,
            include_xy_offset=False,
        )
    )

    # Remove clean obs (not needed at inference — noise is disabled)
    for key in [
        "clean_reduced_coords_obs",
        "clean_mimic_reduced_coords_target_poses",
    ]:
        env_cfg.observation_components.pop(key, None)
