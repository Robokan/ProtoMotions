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
        --num-envs 4096 --batch-size 16384 --training-max-steps 200000000 \
        --experiment-name atlas_ase_battle_hlc_v1

Hyperparameters are ported from Eric's WORKING IsaacLabASE HRL battle league
(~/eric/IsaacLabExtensionTemplate .../battle/config/sword_and_shield/agents/
rl_games_hrl_cfg.yaml, trained at 4096 matches): HLC [1024, 512] logstd -2.3,
gamma 0.95, horizon 32, 6 mini-epochs, bounds_loss 10, entropy 0.005,
adaptive-KL LR (2e-5 -> max 9e-5, kl 0.003), llc_steps=5 decision cadence,
reward mix 0.9 task + 0.1 frozen-LLC-discriminator style anchor, league pool
16 gated at 80% win rate over 2048 games. NOTE --num-envs counts characters
(2/match): 4096 = 2048 matches (Template parity would be 8192 — VRAM probe
first). training_max_steps counts HLC decisions x envs; sim steps = 5x.
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
HISTORY_STEPS = 8  # frozen-discriminator reference window (matches ase/mlp.py)


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
    parser.add_argument(
        "--shared-pool-dir",
        default=None,
        help="Shared league pool directory (MULTI_ROBOT_LEAGUE_PLAN Phase 1): "
        "publish snapshots here and ingest compatible snapshots from other "
        "concurrent runs.",
    )
    parser.add_argument(
        "--opponent-llc-checkpoint",
        type=str,
        default=None,
        help="The OPPONENT robot's frozen LLC pretrain checkpoint "
        "(required with --opponent-robot for a real cross-morphology "
        "league; without it the opponent block is mechanically driven "
        "by the ego brain).",
    )
    parser.add_argument(
        "--opponent-robot",
        default=None,
        help="Robot name for the opponent block (MULTI_ROBOT_LEAGUE_PLAN "
        "Phase 3): spawns a second articulation per env via "
        "MultiRobotIsaacLabSimulator. Same-robot values exercise the "
        "two-entity scene (validation rung 1).",
    )


BATTLE_CONTACT_BODIES = [
    "all_left_foot_bodies",
    "all_right_foot_bodies",
    "all_left_hand_bodies",
    "all_right_hand_bodies",
    "head_body_name",
    "torso_body_name",
]


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    # Contact sensors on strike/damage bodies (semantic names resolve
    # per robot). Same list as the GPC battle league.
    robot_cfg.update_fields(contact_bodies=BATTLE_CONTACT_BODIES)
    if hasattr(simulator_cfg, "filter_env_collisions"):
        simulator_cfg.filter_env_collisions = False

    if getattr(args, "opponent_robot", None):
        # Phase 3: the opponent block hosts its own morphology through the
        # two-articulation simulator.
        from protomotions.robot_configs.factory import robot_config as build_robot
        opp_cfg = build_robot(args.opponent_robot)
        opp_cfg.update_fields(contact_bodies=BATTLE_CONTACT_BODIES)
        simulator_cfg.opponent_robot_config = opp_cfg
        simulator_cfg._target_ = (
            "protomotions.simulator.isaaclab.multi_robot_simulator."
            "MultiRobotIsaacLabSimulator"
        )


def _opponent_robot_config(args: argparse.Namespace):
    from protomotions.robot_configs.factory import robot_config as build_robot

    return build_robot(args.opponent_robot)


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
        # Self-state: MUST match the LLC pretrain's max_coords_obs settings
        # (ase/mlp.py: local_obs, root_height, no contacts) — the frozen LLC
        # consumes this observation verbatim.
        "max_coords_obs": max_coords_obs_factory(
            local_obs=True, root_height_obs=True, observe_contacts=False
        ),
        # Opponent/fight state for the HLC.
        "task_obs": battle_task_obs_factory(),
        # Motion-history window for the FROZEN pretrain discriminator (style
        # anchor only — no discriminator training in stage 2).
        "historical_max_coords_obs": historical_max_coords_obs_factory(
            local_obs=True, root_height_obs=True, observe_contacts=False
        ),
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
    if getattr(args, "opponent_robot", None):
        # Phase 3: the opponent block's body tables come from ITS robot.
        from protomotions.robot_configs.factory import robot_config as build_robot
        opp_cfg = build_robot(args.opponent_robot)
        battle_control.opponent_tables = dict(
            battle_table_kwargs(opp_cfg, args.opponent_robot),
            body_names=list(opp_cfg.kinematic_info.body_names),
            default_root_height=opp_cfg.default_root_height,
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

    # HLC trunk: Template's [1024, 512] with fixed logstd -2.3. Its 64-dim
    # output IS the skill latent (projected to the hypersphere inside the
    # agent before the LLC sees it).
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

    from protomotions.agents.ppo.config import AdaptiveLRConfig

    return LeagueASEHLCAgentConfig(
        # Frozen stage-1 modules from the pretrain checkpoint: the LLC actor
        # and its discriminator (style anchor for the 0.1 reward mix).
        pretrained_modules=dict(
            {
                "llc": PretrainedModelConfig(
                    checkpoint_path=args.llc_checkpoint,
                    module_path="actor",
                ),
                "llc_disc": PretrainedModelConfig(
                    checkpoint_path=args.llc_checkpoint,
                    module_path="discriminator",
                ),
            },
            **(
                {
                    "opp_llc": PretrainedModelConfig(
                        checkpoint_path=args.opponent_llc_checkpoint,
                        module_path="actor",
                    )
                }
                if getattr(args, "opponent_llc_checkpoint", None)
                else {}
            ),
        ),
        opponent_robot_name=getattr(args, "opponent_robot", None),
        opponent_robot_config=(
            _opponent_robot_config(args)
            if getattr(args, "opponent_llc_checkpoint", None)
            else None
        ),
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
            decision_interval=5,  # Template llc_steps: one z per 5 sim steps
            task_reward_w=0.9,
            disc_reward_w=0.1,
        ),
        # Template league: pool 16, gated over 2048 games. Gate at 0.7 (the
        # SOMA league's value — Eric 2026-07-30; the Template's 0.8 never
        # fired in 595 epochs under PFSP + LLC hot-reloads).
        league=LeagueParams(
            robot_name=args.robot_name,
            role="main",
            max_members=16,
            gate_win_rate=0.7,
            gate_min_games=2048,
            shared_pool_dir=getattr(args, "shared_pool_dir", None),
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
