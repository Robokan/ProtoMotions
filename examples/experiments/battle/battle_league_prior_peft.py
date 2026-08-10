# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""League self-play battle RLFT for two SOMAs (SOMA_GPC_COMBAT_PLAN Phases 5-6).

PPO over the frozen GPC prior's token logits with DoRA adapters, where the
environment is a paired two-SOMA battle arena and opponents are PFSP-sampled
league snapshots. Warm-start from the combat SFT checkpoint.

``--num-envs`` is the TOTAL env count = 2x the number of parallel matches
(env i fights env i + N).

Usage:
    python protomotions/train_agent.py \
        --robot-name soma23 --simulator isaaclab \
        --experiment-path examples/experiments/battle/battle_league_prior_peft.py \
        --motion-file data/soma_combat_only.pt \
        --prior-checkpoint results/soma_gpc_prior/last.ckpt \
        --checkpoint results/soma_sft_combat/last.ckpt \
        --num-envs 1024 --batch-size 4096 \
        --experiment-name soma_battle_league

    # Main-exploiter seat (finds the main agent's holes; looser prior budget):
    ... --experiment-name soma_battle_exploiter \
        --league-role main_exploiter \
        --league-opponent-dir results/soma_battle_league/league \
        --peft-sampling-mode nucleus
"""

import argparse
import math

from examples.experiments.gpc.prior_context import (
    add_peft_sampling_mode_argument,
    nearest_surface_obs_params,
    peft_sampling_mode_kwargs,
)
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig

ARENA_SIZE = 7.0  # IsaacLabASE borderline_space
ARENA_SPACING = 16.0  # keeps neighboring matches out of contact range


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--shared-pool-dir",
        default=None,
        help="Shared league pool directory (MULTI_ROBOT_LEAGUE_PLAN Phase 1).",
    )
    parser.add_argument("--prior-checkpoint", required=True)
    parser.add_argument(
        "--league-role",
        default="main",
        choices=["main", "main_exploiter"],
        help="League seat: 'main' grows the league; 'main_exploiter' trains "
        "only against the main run's latest snapshot.",
    )
    parser.add_argument(
        "--league-opponent-dir",
        default=None,
        help="For --league-role main_exploiter: the main run's league dir.",
    )
    parser.add_argument(
        "--dense-reward-scale",
        type=float,
        default=1.0,
        help="Scale on all dense battle reward terms (anneal toward 0 as the "
        "league matures; the sparse win/lose signal is unaffected).",
    )
    add_peft_sampling_mode_argument(parser)


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    # Contact sensors on every strike/damage body so the hit integrator sees
    # net forces there (feet stay included for ground contact bookkeeping).
    #
    # The robot's DAMAGE TABLE is appended explicitly: a damage body with
    # no sensor reads a constant 0 N and can never score, which had
    # silently happened to the pelvis row on every robot and to SOMA's
    # Spine1/Spine2 liver-shot rows. Deriving from the same table
    # BattleControlConfig consumes makes the invariant structural.
    from protomotions.envs.battle.robot_tables import battle_table_kwargs
    _damage_rows = battle_table_kwargs(robot_cfg, args.robot_name).get(
        "damage_body_names",
        ["Head", "Chest", "Spine2", "Spine1", "Hips"],  # BattleControlConfig default
    )
    robot_cfg.update_fields(
        contact_bodies=[
            "all_left_foot_bodies",
            "all_right_foot_bodies",
            "all_left_hand_bodies",
            "all_right_hand_bodies",
            "head_body_name",
            "torso_body_name",
        ] + list(_damage_rows),
    )
    robot_cfg.contact_bodies = list(dict.fromkeys(robot_cfg.contact_bodies))
    # Match partners share an arena: envs must be allowed to collide.
    if hasattr(simulator_cfg, "filter_env_collisions"):
        simulator_cfg.filter_env_collisions = False


def terrain_config(args: argparse.Namespace):
    from protomotions.components.terrains.config import TerrainConfig

    # One flat field large enough for the arena grid.
    num_matches = max(args.num_envs // 2, 1)
    grid = math.ceil(math.sqrt(num_matches))
    required = grid * ARENA_SPACING + 2 * ARENA_SPACING
    tiles = max(1, math.ceil(required / 20.0))
    return TerrainConfig(
        map_length=20.0,
        map_width=20.0,
        num_levels=tiles,
        num_terrains=tiles,
        terrain_proportions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    )


def scene_lib_config(args: argparse.Namespace):
    from protomotions.components.scene_lib import SceneLibConfig

    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    from protomotions.components.motion_lib import MotionLibConfig

    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.action import make_pd_action_config
    from protomotions.envs.battle.control import BattleControlConfig
    from protomotions.envs.battle.hit_state import HitStateConfig
    from protomotions.envs.battle.factories import (
        battle_task_obs_factory,
        default_battle_reward_components,
    )
    from protomotions.envs.component_factories import (
        max_coords_obs_factory,
        nearest_surface_obs_factory,
        previous_actions_factory,
    )
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig

    observation_components = {
        "max_coords_obs": max_coords_obs_factory(),
        "previous_actions": previous_actions_factory(),
        "task_obs": battle_task_obs_factory(),
        "nearest_surface": nearest_surface_obs_factory(
            **nearest_surface_obs_params(robot_cfg),
        ),
    }

    dense = getattr(args, "dense_reward_scale", 1.0)

    cfg = EnvConfig(
        ref_contact_smooth_window=7,
        num_state_history_steps=1,
        max_episode_length=750,  # 15 s at 50 Hz (IsaacLabASE round length)
        reset_grace_period=5,
        ref_respawn_offset=0.05,
        control_components={
            # Kinetic-energy damage model (see HitStateConfig/BattleControlConfig):
            # HP loss = damage_to_health (HP/joule) x 0.5 m_limb v_impact^2 x
            # region mult, ONCE per contact event, zero below strike_min_speed,
            # capped at max_hp_per_hit. Pushes/leans/grinds physically cannot
            # score; legs out-damage hands via their larger mass (0.5 m v^2).
            # Calibrated 2026-07-17: probe showed the v4 champion's contacts
            # all arrive < 1.02 m/s (grinds, never strikes), so the 2.5 m/s
            # gate zeroes its entire repertoire; constants above target ~5
            # clean head punches (19 J) or ~4 head kicks (38 J, capped) to KO.
            "battle": BattleControlConfig(
                arena_size=ARENA_SIZE,
                arena_spacing=ARENA_SPACING,
                raw_health_damage=True,
                damage_to_health=0.005,  # HP/joule: hand@6m/s~19J -> ~19% head hit
                max_hp_per_hit=0.25,
                # Speed gate at 0 (was 2.5): probe showed champion contacts
                # arrive <~1 m/s, so 2.5 zeroed all HP/stun. KE∝v² still
                # makes true soft contacts tiny; force_on=20N still required.
                hit_state=HitStateConfig(strike_min_speed=1.5, ke_reward_ref=5.0),  # 1.5: above guard-press chip (the gate-0 era bred crouch-turtles), below a real jab
                # Concussion-gated knockouts (enabled 2026-07-17 with the KE
                # model): a downed fighter is KO'd only while stun > 0.4.
                # Stun deposits stun_gain*KE/stun_raw_energy_ref, head-weighted.
                # With strike_min_speed=0, any contact event can deposit stun;
                # trips/pushes are still small via KE∝v².
                stun_gates_ko=True,
            ),
        },
        observation_components=observation_components,
        reward_components=default_battle_reward_components(dense_scale=dense),
        action_config=make_pd_action_config(robot_cfg),
        # Reference-state init from combat clips (spawn mid-fight postures)
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
        MLPWithConcatConfig,
        MLPLayerConfig,
        PretrainedModelConfig,
    )
    from protomotions.agents.evaluators.config import EvaluatorConfig
    from protomotions.agents.league.agent import (
        LeagueDiscretePriorPEFTRLFTAgentConfig,
        LeagueParams,
    )
    from protomotions.agents.peft.prior_config import (
        DiscretePriorPEFTConfig,
        DiscretePriorPEFTActorConfig,
        DiscretePriorPEFTRLFTModelConfig,
    )

    sampling_kwargs = peft_sampling_mode_kwargs(args)
    role = getattr(args, "league_role", "main")

    return LeagueDiscretePriorPEFTRLFTAgentConfig(
        pretrained_modules={
            "prior": PretrainedModelConfig(
                checkpoint_path=args.prior_checkpoint,
                module_path="",
            ),
        },
        e_clip=0.2,
        tau=0.95,
        league=LeagueParams(
            robot_name=args.robot_name,
            role=role,
            exploiter_opponent_dir=getattr(args, "league_opponent_dir", None),
            shared_pool_dir=getattr(args, "shared_pool_dir", None),
        ),
        model=DiscretePriorPEFTRLFTModelConfig(
            actor=DiscretePriorPEFTActorConfig(
                in_keys=["task_obs"],
                peft=DiscretePriorPEFTConfig(
                    peft_type="dora",
                    rank=32,
                    alpha=64,
                    temperature=1.0,
                    **sampling_kwargs,
                    film_input_norm=True,
                ),
            ),
            critic=MLPWithConcatConfig(
                in_keys=["max_coords_obs", "task_obs"],
                out_keys=["value"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=1,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu") for _ in range(4)
                ],
            ),
            actor_optimizer=OptimizerConfig(_target_="torch.optim.AdamW", lr=1e-4),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.AdamW", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        num_steps=64,
        num_mini_epochs=2,
        normalize_rewards=True,
        gradient_clip_val=25.0,
        save_last_checkpoint_every=10,
        # The mimic evaluator is meaningless for battles; league metrics are
        # logged every epoch and the tournament evaluator (Phase 7) covers eval.
        evaluator=EvaluatorConfig(eval_metrics_every=None),
    )


def apply_inference_overrides(
    robot_cfg,
    simulator_cfg,
    env_cfg,
    agent_cfg,
    terrain_cfg,
    motion_lib_cfg,
    scene_lib_cfg,
    args,
):
    # Exhibition-style viewing: one long match, viewer on.
    env_cfg.max_episode_length = 100000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0
