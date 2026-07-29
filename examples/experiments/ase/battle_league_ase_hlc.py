# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Paper-faithful ASE battle league: frozen LLC + high-level latent policy.

Two-stage structure per the ASE paper (Peng 2022), architecturally parallel
to the GPC league (frozen prior + trainable adapters):

    Stage 1 (exists): examples/experiments/ase/mlp.py pretrains the
        latent-conditioned low-level controller (LLC). It can keep training
        independently and be swapped under an existing HLC later.
    Stage 2 (this file): a small PPO high-level controller (HLC) whose 64-dim
        action is the LLC's skill latent. The LLC is loaded FROZEN from
        --llc-checkpoint; league snapshots carry only HLC weights.

Same battle arena, rules, and rewards as the GPC league and the full-weight
ASE league (battle_league_ase.py). No AMP discriminator / MI objective here:
style is guaranteed by the frozen LLC, exactly like the GPC league's frozen
prior. Robot-generic via the per-robot battle body tables.

Usage (LLC pretrain first, then this league — HLC trains from scratch):

    python protomotions/train_agent.py \
        --robot-name atlas --simulator isaaclab --headless \
        --motion-file data/atlas_pretrain_corpus_v6.pt \
        --experiment-path examples/experiments/ase/battle_league_ase_hlc.py \
        --llc-checkpoint results/atlas_ase_pretrain_v6/last.ckpt \
        --num-envs 256 --batch-size 512 --training-max-steps 200000000 \
        --experiment-name atlas_ase_battle_hlc_v1
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

LATENT_DIM = 64  # must match the LLC pretrain (ase/mlp.py)


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--llc-checkpoint",
        required=True,
        help="ASE pretrain checkpoint whose actor becomes the frozen LLC.",
    )
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
    from protomotions.envs.component_factories import max_coords_obs_factory
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig

    observation_components = {
        # Self-state: MUST match the LLC pretrain's max_coords_obs settings
        # (ase/mlp.py: local_obs, root_height, no contacts) — the frozen LLC
        # consumes this observation verbatim.
        "max_coords_obs": max_coords_obs_factory(
            local_obs=True, root_height_obs=True, observe_contacts=False
        ),
        # Opponent/fight state for the HLC.
        "task_obs": battle_task_obs_factory(),
        # No historical obs: stage 2 has no discriminator.
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
        hit_state=HitStateConfig(strike_min_speed=1.5, ke_reward_ref=5.0),
        stun_gates_ko=True,
        **battle_table_kwargs(robot_cfg, args.robot_name),
    )

    cfg = EnvConfig(
        ref_contact_smooth_window=7,
        num_state_history_steps=1,
        max_episode_length=750,  # 15 s at 50 Hz, same round length as GPC league
        reset_grace_period=5,
        ref_respawn_offset=0.05,
        control_components={"battle": battle_control},
        observation_components=observation_components,
        reward_components=default_battle_reward_components(dense_scale=dense),
        action_config=make_pd_action_config(robot_cfg),
        # Reference-state init from corpus clips (spawn mid-fight postures).
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
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.common.config import (
        MLPLayerConfig,
        MLPWithConcatConfig,
        ModuleContainerConfig,
        ModuleOperationForwardConfig,
        ObsProcessorConfig,
        PretrainedModelConfig,
    )
    from protomotions.agents.evaluators.config import EvaluatorConfig
    from protomotions.agents.league.agent import LeagueParams
    from protomotions.agents.league.ase_hlc_agent import (
        HLCParams,
        LeagueASEHLCAgentConfig,
    )
    from protomotions.agents.ppo.config import PPOActorConfig, PPOModelConfig

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

    # Small HLC trunk (paper-style): its 64-dim output IS the skill latent
    # (projected to the hypersphere inside the agent before the LLC sees it).
    actor_config = PPOActorConfig(
        mu_key="actor_trunk_out",
        num_out=LATENT_DIM,
        # Wider exploration than joint-space policies: the latent is
        # direction-coded (unit sphere), so std ~0.2/dim perturbs the chosen
        # skill without drowning it.
        actor_logstd=-1.6,
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
                        MLPLayerConfig(units=512, activation="relu"),
                        MLPLayerConfig(units=512, activation="relu"),
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
                    MLPLayerConfig(units=1024, activation="relu"),
                    MLPLayerConfig(units=512, activation="relu"),
                ],
            ),
        ],
    )

    return LeagueASEHLCAgentConfig(
        # The frozen low-level controller: the pretrain checkpoint's actor.
        pretrained_modules={
            "llc": PretrainedModelConfig(
                checkpoint_path=args.llc_checkpoint,
                module_path="actor",
            ),
        },
        model=PPOModelConfig(
            in_keys=hlc_in_keys,
            actor=actor_config,
            critic=critic_config,
            # HLC trains from scratch — faster LRs than the warm-started
            # joint-space policies.
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=5e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        ),
        hlc=HLCParams(latent_dim=LATENT_DIM, llc_deterministic=True),
        league=LeagueParams(role="main"),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        e_clip=0.2,
        tau=0.95,
        num_steps=64,
        num_mini_epochs=2,
        normalize_rewards=True,
        gradient_clip_val=25.0,
        save_last_checkpoint_every=10,
        # The mimic evaluator is meaningless for battles; league metrics are
        # logged every epoch and the tournament evaluator covers eval.
        evaluator=EvaluatorConfig(eval_metrics_every=None),
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
