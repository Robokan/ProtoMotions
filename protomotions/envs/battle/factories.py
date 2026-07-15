# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""MdpComponent factories for the battle task.

These bind the pure battle kernels to ``EnvContext.battle`` paths so
experiment configs can register them in ``observation_components`` /
``reward_components`` exactly like the stock steering/target tasks.
"""

from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.battle.obs import compute_battle_task_obs
from protomotions.envs.battle.rewards import (
    compute_arena_boundary_penalty,
    compute_facing_delta_reward,
    compute_facing_passthrough,
    compute_facing_reward,
    compute_hit_reward,
    compute_hit_taken_penalty,
    compute_idle_penalty,
    compute_range_reward,
    compute_strike_diversity_bonus,
    compute_win_reward,
)


def battle_task_obs_factory(
    pos_clamp: float = 14.0, vel_clamp: float = 50.0
) -> MdpComponent:
    """Opponent + fight-state observation (PEFT ``task_obs``)."""
    return MdpComponent(
        compute_func=compute_battle_task_obs,
        dynamic_vars={
            "root_pos": EnvContext.current.root_pos,
            "root_rot": EnvContext.current.root_rot,
            "opp_root_pos": EnvContext.battle.opp_root_pos,
            "opp_root_rot": EnvContext.battle.opp_root_rot,
            "opp_root_vel": EnvContext.battle.opp_root_vel,
            "opp_root_ang_vel": EnvContext.battle.opp_root_ang_vel,
            "opp_key_body_pos": EnvContext.battle.opp_key_body_pos,
            "opp_key_body_vel": EnvContext.battle.opp_key_body_vel,
            "health": EnvContext.battle.health,
            "opp_health": EnvContext.battle.opp_health,
            "downed": EnvContext.battle.downed,
            "opp_downed": EnvContext.battle.opp_downed,
            "round_time_left": EnvContext.battle.round_time_left,
            "arena_center": EnvContext.battle.arena_center,
            "arena_half_size": EnvContext.battle.arena_half_size,
        },
        static_params={"pos_clamp": pos_clamp, "vel_clamp": vel_clamp},
    )


def battle_win_reward_factory(weight: float = 100.0) -> MdpComponent:
    """Sparse zero-sum outcome reward (the core signal)."""
    return MdpComponent(
        compute_func=compute_win_reward,
        dynamic_vars={"win_signal": EnvContext.battle.win_signal},
        static_params={"weight": weight},
    )


def battle_hit_reward_factory(weight: float = 30.0) -> MdpComponent:
    """Hit energy landed (dense bootstrap term; anneal as league matures)."""
    return MdpComponent(
        compute_func=compute_hit_reward,
        dynamic_vars={"hit_energy_dealt": EnvContext.battle.hit_energy_dealt},
        static_params={"weight": weight},
    )


def battle_hit_taken_penalty_factory(weight: float = -20.0) -> MdpComponent:
    """Hit energy absorbed (weight is negative)."""
    return MdpComponent(
        compute_func=compute_hit_taken_penalty,
        dynamic_vars={"hit_energy_taken": EnvContext.battle.hit_energy_taken},
        static_params={"weight": weight},
    )


def battle_strike_diversity_factory(weight: float = 90.0) -> MdpComponent:
    """Kickboxing diversity: pays for damage from the under-used limb group."""
    return MdpComponent(
        compute_func=compute_strike_diversity_bonus,
        dynamic_vars={
            "strike_diversity_bonus": EnvContext.battle.strike_diversity_bonus
        },
        static_params={"weight": weight},
    )


def battle_facing_reward_factory(weight: float = 2.0) -> MdpComponent:
    """Absolute gaze reward (IsaacLabASE reward_face_w=2.0): pays for looking
    at the opponent, using the pre-computed control-side facing (gaze axis
    already corrected to SOMA's body-frame -y).

    Not farmable in practice: the approach reward (weight 4.0, velocity toward
    the opponent) dominates and the idle penalty punishes standing, so
    circling-while-staring loses to closing-and-hitting. The potential-based
    variant (compute_facing_delta_reward) removed engagement pressure entirely
    and produced fighters that never oriented — reverted.
    """
    return MdpComponent(
        compute_func=compute_facing_passthrough,
        dynamic_vars={"facing": EnvContext.battle.facing},
        static_params={"weight": weight},
    )


def battle_range_reward_factory(
    weight: float = 4.0,
    desired_range: float = 1.0,
    back_away_distance: float = 3.0,
) -> MdpComponent:
    return MdpComponent(
        compute_func=compute_range_reward,
        dynamic_vars={
            "root_pos": EnvContext.current.root_pos,
            "root_vel": EnvContext.current.root_vel,
            "opp_root_pos": EnvContext.battle.opp_root_pos,
            "opp_downed": EnvContext.battle.opp_downed,
        },
        static_params={
            "weight": weight,
            "desired_range": desired_range,
            "back_away_distance": back_away_distance,
        },
    )


def battle_idle_penalty_factory(weight: float = 1.0) -> MdpComponent:
    return MdpComponent(
        compute_func=compute_idle_penalty,
        dynamic_vars={"idle_time": EnvContext.battle.idle_time},
        static_params={"weight": weight},
    )


def battle_boundary_penalty_factory(
    weight: float = 2.0, margin: float = 1.0
) -> MdpComponent:
    return MdpComponent(
        compute_func=compute_arena_boundary_penalty,
        dynamic_vars={
            "root_pos": EnvContext.current.root_pos,
            "arena_center": EnvContext.battle.arena_center,
            "arena_half_size": EnvContext.battle.arena_half_size,
        },
        static_params={"weight": weight, "margin": margin},
    )


def default_battle_reward_components(dense_scale: float = 1.0) -> dict:
    """The simple kickboxing reward: look at each other and make hits.

    Core: sparse win/lose + gaze facing + hit exchange + limb diversity
    (punches AND kicks — the diversity stream pays for damage from the
    under-used limb group). Negative shaping only steers away from things
    we don't want: stalling and edge-hugging. Ring-outs are handled by
    rule (points decision), not reward. ``dense_scale`` anneals every
    dense term toward 0 as the league matures; the win term stays.
    """
    return {
        "battle_win": battle_win_reward_factory(weight=100.0),
        # Approach (velocity toward opponent, gated by range) is the
        # engagement gradient — weight 4.0 per IsaacLabASE, double facing so
        # closing dominates circling. Its absence in the weekend run left
        # fighters with no reason to move together (facing stuck at 0.5).
        "battle_approach": battle_range_reward_factory(weight=4.0 * dense_scale),
        "battle_facing": battle_facing_reward_factory(weight=2.0 * dense_scale),
        "battle_hit": battle_hit_reward_factory(weight=30.0 * dense_scale),
        "battle_hit_taken": battle_hit_taken_penalty_factory(
            weight=-20.0 * dense_scale
        ),
        # Diversity bonus back to 30: bumping it to 90 (v3, epochs 860-1479)
        # gave only a transient ~30-epoch improvement that decayed back past
        # the ~34:1 baseline — cranking the bonus can't beat the structural
        # incentive to punch. The real fix is per-limb raw-energy weighting
        # (BattleControlConfig.strike_group_multipliers, legs 2.0 vs hands 1.0),
        # which makes kicks genuinely out-score punches at the source.
        "battle_strike_diversity": battle_strike_diversity_factory(
            weight=30.0 * dense_scale
        ),
        "battle_idle": battle_idle_penalty_factory(weight=1.0 * dense_scale),
        "battle_boundary": battle_boundary_penalty_factory(weight=1.0 * dense_scale),
    }


__all__ = [
    "battle_task_obs_factory",
    "battle_win_reward_factory",
    "battle_hit_reward_factory",
    "battle_hit_taken_penalty_factory",
    "battle_strike_diversity_factory",
    "battle_facing_reward_factory",
    "battle_range_reward_factory",
    "battle_idle_penalty_factory",
    "battle_boundary_penalty_factory",
    "default_battle_reward_components",
]
