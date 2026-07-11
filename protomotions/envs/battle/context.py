# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Typed context view for two-character battle tasks.

Populated by :class:`protomotions.envs.battle.control.BattleControl` via
``populate_context`` and consumed by the battle observation / reward /
termination kernels through ``EnvContext.battle.<field>`` paths.

Pairing convention: with ``2N`` environments, env ``i`` fights env
``(i + N) % 2N``. All tensors here are laid out per-env (length ``2N``) with
the opponent data already permuted into each env's row, so kernels never need
to know the pairing.
"""

from typing import Optional

from torch import Tensor

from protomotions.envs.context_paths import FieldPath


class BattleContext:
    """View of the battle task state for observation/reward/termination kernels.

    All fields are FieldPath descriptors for dual class/instance access.
    """

    # Opponent root state (already permuted into each env's row)
    opp_root_pos: Tensor = FieldPath()  # [2N, 3]
    opp_root_rot: Tensor = FieldPath()  # [2N, 4] (w-last)
    opp_root_vel: Tensor = FieldPath()  # [2N, 3]
    opp_root_ang_vel: Tensor = FieldPath()  # [2N, 3]

    # Opponent key bodies (hands/feet/head by default), world frame
    opp_key_body_pos: Tensor = FieldPath()  # [2N, K, 3]
    opp_key_body_vel: Tensor = FieldPath()  # [2N, K, 3]

    # Head poses (world frame) for gaze-based rewards: the facing reward is
    # head-looks-at-opponent-head, not root orientation (root-facing rewards
    # a squared pelvis while the fighter stares elsewhere).
    head_pos: Tensor = FieldPath()  # [2N, 3] ego head
    head_rot: Tensor = FieldPath()  # [2N, 4] ego head (w-last)
    opp_head_pos: Tensor = FieldPath()  # [2N, 3] opponent head

    # Fight state scalars
    health: Tensor = FieldPath()  # [2N] in [0, 1]
    opp_health: Tensor = FieldPath()  # [2N] in [0, 1]
    downed: Tensor = FieldPath()  # [2N] normalized down-timer in [0, 1]
    opp_downed: Tensor = FieldPath()  # [2N]
    round_time_left: Tensor = FieldPath()  # [2N] in [0, 1]
    idle_time: Tensor = FieldPath()  # [2N] seconds-equivalent idle accumulator

    # Per-step hit accounting (already region/velocity gated, log-normalized)
    hit_energy_dealt: Tensor = FieldPath()  # [2N] this step
    hit_energy_taken: Tensor = FieldPath()  # [2N] this step
    # Growth of the lesser strike-group cumulative (hands vs legs): pays for
    # damage from the under-used limb group (kickboxing diversity)
    strike_diversity_bonus: Tensor = FieldPath()  # [2N] this step

    # Gaze quality in [0, 1] and its per-step change (potential-based facing)
    facing: Tensor = FieldPath()  # [2N]
    facing_delta: Tensor = FieldPath()  # [2N]

    # Match outcome, stamped on the step the match ends (else zero)
    win_signal: Tensor = FieldPath()  # [2N] +1 win / -1 loss / 0 otherwise
    match_ended: Tensor = FieldPath()  # [2N] bool

    # Arena
    arena_center: Tensor = FieldPath()  # [2N, 2] world XY of this match's arena
    arena_half_size: float = FieldPath()  # scalar, half side length in meters

    def __init__(
        self,
        opp_root_pos: Tensor,
        opp_root_rot: Tensor,
        opp_root_vel: Tensor,
        opp_root_ang_vel: Tensor,
        opp_key_body_pos: Tensor,
        opp_key_body_vel: Tensor,
        head_pos: Tensor,
        head_rot: Tensor,
        opp_head_pos: Tensor,
        health: Tensor,
        opp_health: Tensor,
        downed: Tensor,
        opp_downed: Tensor,
        round_time_left: Tensor,
        idle_time: Tensor,
        hit_energy_dealt: Tensor,
        hit_energy_taken: Tensor,
        strike_diversity_bonus: Tensor,
        facing: Tensor,
        facing_delta: Tensor,
        win_signal: Tensor,
        match_ended: Tensor,
        arena_center: Tensor,
        arena_half_size: float,
    ):
        self.opp_root_pos = opp_root_pos
        self.opp_root_rot = opp_root_rot
        self.opp_root_vel = opp_root_vel
        self.opp_root_ang_vel = opp_root_ang_vel
        self.opp_key_body_pos = opp_key_body_pos
        self.opp_key_body_vel = opp_key_body_vel
        self.head_pos = head_pos
        self.head_rot = head_rot
        self.opp_head_pos = opp_head_pos
        self.health = health
        self.opp_health = opp_health
        self.downed = downed
        self.opp_downed = opp_downed
        self.round_time_left = round_time_left
        self.idle_time = idle_time
        self.hit_energy_dealt = hit_energy_dealt
        self.hit_energy_taken = hit_energy_taken
        self.strike_diversity_bonus = strike_diversity_bonus
        self.facing = facing
        self.facing_delta = facing_delta
        self.win_signal = win_signal
        self.match_ended = match_ended
        self.arena_center = arena_center
        self.arena_half_size = arena_half_size


__all__ = ["BattleContext"]
