# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Distill a GPC battle champion into a fast feedforward MLP student.

WHY: the league policy acts in token space — per control step it runs 8
sequential autoregressive prior forwards, which is ~50-100 ms at the batch
size of a single match (GPU badly underutilized; see the perf notes in the
battle README). A distilled MLP maps observations straight to joint targets
in one forward (sub-ms): fast enough for a smooth viewer AND the real
deployment path (ONNX -> TensorRT/fp8 on Jetson Thor, running at control
rate with room to spare).

HOW (DAgger-style, the framework's built-in supervised-distillation path):
- Teacher (expert) = a trained league adapter over the frozen GPC prior,
  loaded via SupervisedAgentConfig.expert_model_path. It labels each visited
  state with its continuous action.
- Student = a plain MLP over the same battle observations
  (proprioception `max_coords_obs` + opponent `task_obs`), output = num_dofs
  joint PD targets.
- rollout_actor = STUDENT: the student drives the rollout (so it sees its own
  state distribution), the teacher labels — this fixes the covariate shift
  that pure behavior cloning suffers.
- Loss = MSE(student_action, expert_action) — the framework's default
  SupervisionLossConfig.

STATUS: SCAFFOLD — the env/agent/loss wiring below follows the working
battle-league and SFT experiments, but this has NOT yet been run end to end.
It needs one GPU validation pass (and a mature league champion as the
teacher). The single block to finalize against the live SupervisedAgent
student contract is the student model config in ``agent_config`` (marked
TODO): confirm the model class and its action output key match
``loss.prediction_key``.

Run (after the league produces a champion):
    python protomotions/train_agent.py \\
        --robot-name soma23 --simulator isaaclab --headless \\
        --motion-file data/soma_combat_viewer.pt \\
        --experiment-path examples/experiments/battle/distill_battle_mlp.py \\
        --expert results/soma_battle_league_v3 \\
        --num-envs 512 --experiment-name soma_battle_distill
"""

import argparse

from protomotions.robot_configs.base import RobotConfig
from protomotions.envs.base_env.config import EnvConfig

# Arena geometry mirrors the league experiment so the student trains in the
# identical fight setup the teacher was trained in.
ARENA_SIZE = 7.0
ARENA_SPACING = 20.0


def additional_experiment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--expert",
        required=True,
        help="League run dir (with resolved_configs.pt + last.ckpt) whose "
        "adapter policy is the distillation teacher.",
    )
    parser.add_argument("--dense-reward-scale", type=float, default=1.0)


def terrain_config(args: argparse.Namespace):
    from protomotions.components.terrains.config import TerrainConfig

    import math

    tiles = max(1, math.ceil(math.sqrt(max(args.num_envs // 2, 1))))
    return TerrainConfig(
        map_length=20.0,
        map_width=20.0,
        num_levels=tiles,
        num_terrains=tiles,
        terrain_proportions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    )


def scene_lib_config(args: argparse.Namespace):
    from protomotions.components.scene_lib import SceneLibConfig

    return SceneLibConfig(scene_file=getattr(args, "scenes_file", None))


def motion_lib_config(args: argparse.Namespace):
    from protomotions.components.motion_lib import MotionLibConfig

    return MotionLibConfig(motion_file=args.motion_file)


def _nearest_surface_params(robot_cfg):
    from protomotions.envs.component_factories import nearest_surface_obs_params

    return nearest_surface_obs_params(robot_cfg)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Identical battle env to the league run — same obs, same arena, same
    fight rules — so the student distills in-distribution. Rewards are
    irrelevant here (supervised loss drives learning) but kept for the env's
    match bookkeeping / telemetry."""
    from protomotions.envs.action import make_pd_action_config
    from protomotions.envs.battle.control import BattleControlConfig
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
            **_nearest_surface_params(robot_cfg)
        ),
    }

    cfg = EnvConfig(
        ref_contact_smooth_window=7,
        num_state_history_steps=1,
        max_episode_length=750,
        reset_grace_period=5,
        ref_respawn_offset=0.05,
        control_components={
            "battle": BattleControlConfig(
                arena_size=ARENA_SIZE,
                arena_spacing=ARENA_SPACING,
                fall_init_prob=0.0,
            ),
        },
        observation_components=observation_components,
        reward_components=default_battle_reward_components(
            dense_scale=getattr(args, "dense_reward_scale", 1.0)
        ),
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2, resample_on_reset=True
        ),
    )
    cfg._target_ = "protomotions.envs.battle.env.BattleEnv"
    return cfg


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
):
    from protomotions.agents.supervised.config import (
        SupervisedAgentConfig,
        RolloutActor,
    )
    from protomotions.agents.common.supervision import (
        SupervisionLossConfig,
        SupervisionLossType,
    )
    from protomotions.agents.common.config import (
        MLPWithConcatConfig,
        MLPLayerConfig,
    )

    # Student obs = proprioception + opponent info (the teacher's inputs minus
    # the prior's token context — the student regresses the final action).
    student_in_keys = ["max_coords_obs", "task_obs", "previous_actions"]

    # TODO(validation): confirm this student model class + its action output
    # key against the live SupervisedAgent student contract during the first
    # GPU run. The MLPWithConcat actor below mirrors the deploy `mlp.py`
    # student; its output key must equal loss.prediction_key ("action").
    student_model = MLPWithConcatConfig(
        in_keys=student_in_keys,
        num_out=robot_config.kinematic_info.num_dofs,
        layers=[
            MLPLayerConfig(units=1024, activation="relu"),
            MLPLayerConfig(units=1024, activation="relu"),
            MLPLayerConfig(units=512, activation="relu"),
        ],
        out_keys=["action"],
        normalize_obs=True,
    )

    return SupervisedAgentConfig(
        model=student_model,
        expert_model_path=args.expert,
        rollout_actor=RolloutActor.STUDENT,  # DAgger: student explores, expert labels
        loss=SupervisionLossConfig(
            loss_type=SupervisionLossType.MSE,
            prediction_key="action",
            target_key="expert_actions",
            log_prefix="distill",
        ),
        num_steps=32,
        batch_size=args.batch_size if hasattr(args, "batch_size") else 16384,
    )
