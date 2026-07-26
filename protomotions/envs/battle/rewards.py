# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Reward kernels for battle tasks.

Kept deliberately thin per the combat plan: naturalness comes from
prior-constrained sampling, not an AMP/style reward. The zero-sum core is the
sparse win/lose signal; the dense terms exist to bootstrap early training and
should be annealed toward zero as the league matures (AlphaStar lesson: dense
shaping helps bootstrap but caps strategy diversity if left on).

Constants follow IsaacLabASE's tuned battle values where applicable.
"""

import torch
from torch import Tensor

from protomotions.utils import rotations


def compute_win_reward(win_signal: Tensor) -> Tensor:
    """Sparse zero-sum outcome reward: +1 win, -1 loss, 0 otherwise.

    ``win_signal`` is nonzero only on the step the match ends.
    """
    return win_signal


def compute_hit_reward(hit_energy_dealt: Tensor) -> Tensor:
    """Log-normalized hit energy landed on the opponent this step."""
    return hit_energy_dealt


def compute_hit_taken_penalty(hit_energy_taken: Tensor) -> Tensor:
    """Log-normalized hit energy absorbed this step (weight negatively)."""
    return hit_energy_taken


def compute_strike_diversity_bonus(strike_diversity_bonus: Tensor) -> Tensor:
    """Growth of the lesser strike-group cumulative (hands vs legs).

    Kickboxing shaping: each new unit of damage from the under-used limb
    group pays extra, so specializing in punching-only or kicking-only
    leaves reward on the table.
    """
    return strike_diversity_bonus


def compute_facing_reward(
    head_pos: Tensor,
    head_rot: Tensor,
    opp_head_pos: Tensor,
    forward_axis: tuple = (0.0, -1.0, 0.0),
) -> Tensor:
    """Gaze quality: is the head looking at the opponent's head? In [0, 1].

    Head-based, matching IsaacLabASE: the direction is head-to-head and the
    gaze vector is the head's forward axis —
    facing = (dot(head_forward, to_opponent_head) + 1) / 2.

    ``forward_axis`` is the gaze direction in the head's LOCAL frame. For the
    SOMA (SMPL-family) skeleton the face points along body-frame -y — NOT +x
    as `calc_heading` assumes. Using +x here trained fighters to point their
    ear at the opponent.
    """
    to_opp = torch.nn.functional.normalize(opp_head_pos - head_pos, dim=-1)

    forward = torch.zeros_like(head_pos)
    forward[..., 0] = forward_axis[0]
    forward[..., 1] = forward_axis[1]
    forward[..., 2] = forward_axis[2]
    gaze = rotations.quat_rotate(head_rot, forward, True)
    gaze = torch.nn.functional.normalize(gaze, dim=-1)

    dot = (gaze * to_opp).sum(dim=-1)
    return (dot + 1.0) * 0.5


def compute_facing_passthrough(facing: Tensor) -> Tensor:
    """Absolute gaze quality in [0, 1], computed control-side with the
    corrected SOMA gaze axis. Passthrough so the reward and the telemetry
    read the identical value."""
    return facing


def compute_facing_delta_reward(facing_delta: Tensor) -> Tensor:
    """Potential-based facing: reward the CHANGE in gaze quality.

    Turning toward the opponent pays; holding a stare pays ~zero, so the
    stare-farming equilibrium (passive circling for dense facing reward)
    cannot exist. The accumulated reward over any trajectory telescopes to
    facing_end - facing_start, bounded by 1.
    """
    return facing_delta


def compute_range_reward(
    root_pos: Tensor,
    root_vel: Tensor,
    opp_root_pos: Tensor,
    opp_downed: Tensor,
    desired_range: float = 1.0,
    weak_gain: float = 0.2,
    back_away_distance: float = 3.0,
    min_closing_speed: float = 0.5,
) -> Tensor:
    """Approach/range shaping (IsaacLabASE ``r_close``, satisficing variant).

    Outside ``desired_range``: full reward for ANY closing speed at or above
    ``min_closing_speed`` — the reward saturates, so a cautious stalk and an
    explosive blitz earn the same. Below the floor it ramps linearly (keeps a
    learning gradient toward moving in). The original linear-in-speed version
    baked "faster approach is better" into the strategy space; approach pace
    is a tactic the league should discover, not a constant we prescribe.
    Inside: heavily attenuated (don't reward crowding).
    Opponent down: reward holding ``back_away_distance`` instead of piling on.
    """
    delta_xy = opp_root_pos[..., :2] - root_pos[..., :2]
    dist = torch.norm(delta_xy, dim=-1)
    u = torch.nn.functional.normalize(delta_xy, dim=-1)
    toward_speed = (root_vel[..., :2] * u).sum(dim=-1).clamp_min(0.0)
    closing = (toward_speed / max(min_closing_speed, 1e-6)).clamp(0.0, 1.0)

    outside = closing
    x = (dist / desired_range).clamp(0.0, 1.0)
    inside = weak_gain * x * x * closing
    r = torch.where(dist > desired_range, outside, inside)

    # Opponent on the ground: back off to a respectful distance
    back_off = torch.exp(-3.0 * (dist - back_away_distance).abs())
    opp_down = opp_downed > 0.0
    return torch.where(opp_down, back_off, r)


def compute_idle_penalty(idle_time: Tensor) -> Tensor:
    """Stalling penalty in [-1, 0] (IsaacLabASE: engages past idle 0.5)."""
    idle = idle_time.clamp(0.0, 1.0)
    return torch.where(idle > 0.5, -(idle - 0.5) * 2.0, torch.zeros_like(idle))


def compute_arena_boundary_penalty(
    root_pos: Tensor,
    arena_center: Tensor,
    arena_half_size: float,
    margin: float = 1.0,
) -> Tensor:
    """Penalty ramping from 0 to -1 over the last ``margin`` meters to the wall."""
    offset = (root_pos[..., :2] - arena_center).abs().max(dim=-1).values
    overflow = (offset - (arena_half_size - margin)).clamp_min(0.0) / max(margin, 1e-6)
    return -overflow.clamp(0.0, 1.0)


__all__ = [
    "compute_win_reward",
    "compute_hit_reward",
    "compute_hit_taken_penalty",
    "compute_strike_diversity_bonus",
    "compute_facing_reward",
    "compute_facing_passthrough",
    "compute_facing_delta_reward",
    "compute_range_reward",
    "compute_idle_penalty",
    "compute_arena_boundary_penalty",
]


def compute_kick_attempt_bonus(kick_attempt_bonus: Tensor) -> Tensor:
    """Kick-attempt shaping: 1.0 per foot that crossed the kick height this
    step (armed + under the per-episode cap; see BattleControl). Teaches the
    league to try kicks at all — the punch meta otherwise prunes them before
    the KE damage economics (legs out-hit hands) can reward them."""
    return kick_attempt_bonus
