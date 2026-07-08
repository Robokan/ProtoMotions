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


def compute_facing_reward(
    root_pos: Tensor,
    root_rot: Tensor,
    opp_root_pos: Tensor,
) -> Tensor:
    """Reward for facing the opponent, in [0, 1].

    IsaacLabASE: facing = (dot(facing_dir, to_opponent) + 1) / 2.
    """
    to_opp = opp_root_pos - root_pos
    to_opp_xy = torch.nn.functional.normalize(to_opp[..., :2], dim=-1)

    facing_dir3d = torch.zeros_like(root_pos)
    facing_dir3d[..., 0] = 1.0
    facing = rotations.quat_rotate(root_rot, facing_dir3d, True)
    facing_xy = torch.nn.functional.normalize(facing[..., :2], dim=-1)

    dot = (facing_xy * to_opp_xy).sum(dim=-1)
    return (dot + 1.0) * 0.5


def compute_range_reward(
    root_pos: Tensor,
    root_vel: Tensor,
    opp_root_pos: Tensor,
    opp_downed: Tensor,
    desired_range: float = 1.0,
    weak_gain: float = 0.2,
    back_away_distance: float = 3.0,
) -> Tensor:
    """Approach/range shaping (IsaacLabASE ``r_close``).

    Outside ``desired_range``: reward closing speed toward the opponent.
    Inside: heavily attenuated closing reward (don't reward crowding).
    Opponent down: reward holding ``back_away_distance`` instead of piling on.
    """
    delta_xy = opp_root_pos[..., :2] - root_pos[..., :2]
    dist = torch.norm(delta_xy, dim=-1)
    u = torch.nn.functional.normalize(delta_xy, dim=-1)
    toward_speed = (root_vel[..., :2] * u).sum(dim=-1).clamp_min(0.0)

    outside = toward_speed
    x = (dist / desired_range).clamp(0.0, 1.0)
    inside = weak_gain * x * x * toward_speed
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
    "compute_facing_reward",
    "compute_range_reward",
    "compute_idle_penalty",
    "compute_arena_boundary_penalty",
]
