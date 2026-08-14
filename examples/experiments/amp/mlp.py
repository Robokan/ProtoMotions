# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.amp.config import AMPAgentConfig
import argparse


# Dilated history steps for temporal context (used by actor and discriminator)
HISTORY_STEPS = [1, 2, 3, 4, 8, 16, 32]


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


def _disc_body_ids(robot_cfg: RobotConfig):
    """Body indices the AMP discriminator sees, or None for all bodies.

    Uses robot_cfg.disc_bodies_subset when the robot defines one.
    NOT trackable_bodies_subset -- that means tracking targets and is
    sized for a different job (t800 lists six bodies), so sharing it
    would leave the discriminator judging almost nothing. The root is forced
    in (it supplies the heading frame and root height) and order follows the
    simulator's body ordering.
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

def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Build environment configuration (training defaults).
    
    Uses MdpComponent-based component configuration with explicit context bindings:
        MdpComponent(compute_func=compute_fn, dynamic_vars={...}, static_params={...})
    """
    from protomotions.envs.component_factories import (
        max_coords_obs_factory,
        historical_max_coords_obs_factory,
        pow_efficiency_bonus_factory,
        fall_termination_factory,
    )
    from protomotions.envs.motion_manager.config import MotionManagerConfig
    from protomotions.envs.action import make_pd_action_config

    # Observation components configuration
    observation_components = {
        # Humanoid self-observations (current state)
        "max_coords_obs": max_coords_obs_factory(
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
        ),
        # Historical observations for AMP discriminator (from StateHistoryBuffer).
        # Restricted to disc_bodies_subset when the robot defines one:
        # the discriminator should judge GAIT, and a body whose motion cannot
        # be reproduced lets it win on jitter instead. The raptor's 36 digit
        # segments (1.4 g, 6 N.m actuators, constantly hit by ground contact)
        # made agent and reference trivially separable, so agent_acc sat at
        # ~0.93 and style reward collapsed while the policy learned nothing
        # about walking. The POLICY still senses every body (max_coords_obs
        # above is untouched), so toes still inform balance.
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
        # BONUS form, not penalty: the additive penalty made net per-step
        # reward ~0 once the disc converged (0.104 style vs -0.097 penalty)
        # and the policy learned to fall on purpose after the grace period.
        # Kept at all (unlike pure AMP) because raptor-family references are
        # keyframed animation, not mocap -- energy plausibility is not
        # embedded in the imitation target. coef*exp(-P/10kW); 10kW ~ the
        # measured mean |tau.qd| of the walk run.
        "pow_rew": pow_efficiency_bonus_factory(coef=0.02, power_scale=10000.0),
    }

    env_config: EnvConfig = EnvConfig(
        max_episode_length=300,  # Training default (eval override applied automatically)
        num_state_history_steps=max(HISTORY_STEPS),  # Store enough history for max dilation
        observation_components=observation_components,
        reward_components=reward_components,
        action_config=make_pd_action_config(robot_cfg),
        # Fall termination: falling ends the episode, and since every
        # per-step reward is >= 0, ending early strictly loses -- AMP's
        # survival economics over real 300-step horizons. PER-RUN KNOB
        # (Eric): height+contact termination is only valid for UPRIGHT
        # corpora; a tumbling/rolling reference puts the torso on the floor
        # deliberately and this would execute it mid-roll. For those runs
        # pass --no-fall-termination and let the annealed disc kill (which
        # scores rolls as in-reference) be the early guard.
        termination_components=(
            {} if getattr(args, "no_fall_termination", False) else
            {"fall": fall_termination_factory(termination_height=0.25)}
        ),
        motion_manager=MotionManagerConfig(
            init_start_prob=0.5  # Bias agent to start at the beginning of the motion to prevent getting stuck in a local-minima (standing still).
        ),
    )

    return env_config


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> AMPAgentConfig:
    from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig, ModuleContainerConfig
    from protomotions.agents.ppo.config import PPOActorConfig
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.amp.config import (
        AMPModelConfig,
        DiscriminatorConfig,
        AMPParametersConfig,
    )
    from protomotions.envs.obs import compute_historical_max_coords_from_motion_lib

    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=["max_coords_obs", "historical_max_coords_obs"],
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=["max_coords_obs", "historical_max_coords_obs"],
            out_keys=["actor_trunk_out"],
            normalize_obs=True,
            norm_clamp_value=5,
            num_out=robot_config.number_of_actions,
            layers=[
                MLPLayerConfig(units=512, activation="relu"),
                MLPLayerConfig(units=256, activation="relu"),
            ],
        ),
    )

    critic_config = MLPWithConcatConfig(
        in_keys=["max_coords_obs", "historical_max_coords_obs"],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[
            MLPLayerConfig(units=512, activation="relu"),
            MLPLayerConfig(units=256, activation="relu"),
        ],
    )

    discriminator_config = DiscriminatorConfig(
        in_keys=["historical_max_coords_obs"],
        out_keys=["disc_logits"],
        models=[
            MLPWithConcatConfig(
                in_keys=["historical_max_coords_obs"],
                out_keys=["disc_logits"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=1,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu"),
                    MLPLayerConfig(units=512, activation="relu"),
                ],
            )
        ],
    )

    disc_critic_config = ModuleContainerConfig(
        in_keys=["max_coords_obs", "historical_max_coords_obs"],
        out_keys=["disc_value"],
        models=[
            MLPWithConcatConfig(
                in_keys=["max_coords_obs", "historical_max_coords_obs"],
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

    # Reference observation components for discriminator expert data
    # Agent injects motion_lib/motion_ids/motion_times/dt at runtime (not in EnvContext)
    from protomotions.envs.mdp_component import MdpComponent
    
    reference_obs_components = {
        "historical_max_coords_obs": MdpComponent(
            compute_func=compute_historical_max_coords_from_motion_lib,
            dynamic_vars={},  # All parameters injected by agent
            static_params={"history_steps": HISTORY_STEPS,
                           "body_ids": _disc_body_ids(robot_config)},
        ),
    }

    agent_config: AMPAgentConfig = AMPAgentConfig(
        model=AMPModelConfig(
            in_keys=["max_coords_obs", "historical_max_coords_obs"],
            out_keys=["action", "mean_action", "neglogp", "value", "disc_logits", "disc_value"],
            actor=actor_config,
            critic=critic_config,
            discriminator=discriminator_config,
            disc_critic=disc_critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
            discriminator_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam", lr=1e-4
            ),
        ),
        reference_obs_components=reference_obs_components,
        batch_size=args.batch_size,
        # Small weight so pow_rew regularizes without drowning the disc style signal.
        task_reward_w=0.1,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        amp_parameters=AMPParametersConfig(
            # Disc-kill ANNEALED to zero (Eric, 2026-08-14): potent early --
            # cut hopeless flailing rollouts instead of simulating them to
            # the cap -- and impotent by epoch 2000, so the converged disc
            # (reward floor ~0.006, below any useful fixed threshold) can
            # never again guillotine healthy episodes as it did (every
            # raptor episode dead at ~35 steps; the 300-vs-75 episode-length
            # experiments never had a chance to matter). For FLOOR-CONTENT
            # corpora (rolls/getups) this annealed kill is also the only
            # viable early guard, since height-based fall termination would
            # execute a mid-roll robot (Eric).
            # NOTE for warm starts from a CONVERGED disc: the early anneal
            # window kills at the old rate until it decays -- use
            # --no-fall-termination/-style overrides or threshold 0 for
            # those runs.
            discriminator_reward_threshold=0.02,
            discriminator_termination_anneal_epochs=2000,
        ),
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
    # For AMP: disable discriminator reward during evaluation
    if agent_cfg is not None and hasattr(agent_cfg, "amp_parameters"):
        agent_cfg.amp_parameters.discriminator_reward_threshold = 0.0

    if env_cfg is not None:
        # Keep the training horizon when --amp-disc-term is on so timeouts
        # still reset envs in the viewer (disc kills alone are not enough).
        if hasattr(env_cfg, "max_episode_length") and not getattr(
            args, "amp_disc_term", False
        ):
            env_cfg.max_episode_length = 1000000
        if hasattr(env_cfg, "motion_manager"):
            if hasattr(env_cfg.motion_manager, "init_start_prob"):
                env_cfg.motion_manager.init_start_prob = 1.0


def additional_experiment_arguments(parser):
    parser.add_argument(
        "--no-fall-termination", action="store_true",
        help="Disable height+contact fall termination (tumbling/rolling "
             "corpora put the torso on the floor deliberately).")
