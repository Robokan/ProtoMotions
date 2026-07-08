# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Opponent observation kernel for battle tasks.

Everything opponent-derived is expressed in the ego heading frame and
explicitly clamped — ragdolling opponents produce velocity spikes that
destabilize value functions (an expensive IsaacLabASE lesson: their obs
builders clamp positions to ±2x the arena border and velocities to ±50 m/s).
"""

import torch
from torch import Tensor

from protomotions.utils import rotations


def compute_battle_task_obs(
    root_pos: Tensor,
    root_rot: Tensor,
    opp_root_pos: Tensor,
    opp_root_rot: Tensor,
    opp_root_vel: Tensor,
    opp_root_ang_vel: Tensor,
    opp_key_body_pos: Tensor,
    opp_key_body_vel: Tensor,
    health: Tensor,
    opp_health: Tensor,
    downed: Tensor,
    opp_downed: Tensor,
    round_time_left: Tensor,
    arena_center: Tensor,
    arena_half_size: float,
    pos_clamp: float = 14.0,
    vel_clamp: float = 50.0,
) -> Tensor:
    """Compute the battle task observation in the ego heading frame.

    Args:
        root_pos: Ego root positions [2N, 3].
        root_rot: Ego root orientations [2N, 4] (w-last).
        opp_root_pos: Opponent root positions [2N, 3].
        opp_root_rot: Opponent root orientations [2N, 4].
        opp_root_vel: Opponent root linear velocities [2N, 3].
        opp_root_ang_vel: Opponent root angular velocities [2N, 3].
        opp_key_body_pos: Opponent key body positions [2N, K, 3].
        opp_key_body_vel: Opponent key body velocities [2N, K, 3].
        health: Ego health [2N] in [0, 1].
        opp_health: Opponent health [2N].
        downed: Ego normalized down-timer [2N].
        opp_downed: Opponent normalized down-timer [2N].
        round_time_left: Normalized round time remaining [2N].
        arena_center: Arena center XY per env [2N, 2].
        arena_half_size: Arena half side length (meters).
        pos_clamp: Clamp for opponent-relative positions (±2x arena border).
        vel_clamp: Clamp for all velocity terms.

    Returns:
        Task observation [2N, 15 + 6K + 5]:
        [opp_root_local(3), opp_rot_tan_norm(6), opp_vel_local(3),
         opp_ang_vel_local(3), opp_key_pos_local(3K), opp_key_vel_local(3K),
         arena_offset_local(2), health(1), opp_health(1), downed(1),
         opp_downed(1), time_left(1)].
    """
    num_envs, num_keys = opp_key_body_pos.shape[0], opp_key_body_pos.shape[1]
    heading_inv = rotations.calc_heading_quat_inv(root_rot, True)

    # Opponent root, relative to ego root, in ego heading frame
    rel_root = opp_root_pos - root_pos
    local_opp_root = rotations.quat_rotate(heading_inv, rel_root, True)
    local_opp_root = local_opp_root.clamp(-pos_clamp, pos_clamp)

    # Opponent orientation as tan-norm in ego heading frame
    local_opp_rot = rotations.quat_mul(heading_inv, opp_root_rot, True)
    opp_rot_tan_norm = rotations.quat_to_tan_norm(local_opp_rot, True)

    local_opp_vel = rotations.quat_rotate(heading_inv, opp_root_vel, True).clamp(
        -vel_clamp, vel_clamp
    )
    local_opp_ang_vel = rotations.quat_rotate(
        heading_inv, opp_root_ang_vel, True
    ).clamp(-vel_clamp, vel_clamp)

    # Opponent key bodies relative to ego root in heading frame
    heading_inv_exp = heading_inv.unsqueeze(1).expand(-1, num_keys, 4).reshape(-1, 4)
    rel_keys = (opp_key_body_pos - root_pos.unsqueeze(1)).reshape(-1, 3)
    local_keys = rotations.quat_rotate(heading_inv_exp, rel_keys, True)
    local_keys = local_keys.clamp(-pos_clamp, pos_clamp).reshape(num_envs, -1)

    key_vels = opp_key_body_vel.reshape(-1, 3)
    local_key_vels = rotations.quat_rotate(heading_inv_exp, key_vels, True)
    local_key_vels = local_key_vels.clamp(-vel_clamp, vel_clamp).reshape(num_envs, -1)

    # Arena awareness: vector from ego to arena center, heading frame,
    # normalized by the arena half size
    arena_delta = torch.cat(
        [
            arena_center - root_pos[:, :2],
            torch.zeros_like(root_pos[:, 0:1]),
        ],
        dim=-1,
    )
    local_arena = rotations.quat_rotate(heading_inv, arena_delta, True)[:, :2]
    local_arena = local_arena / max(arena_half_size, 1e-6)
    local_arena = local_arena.clamp(-2.0, 2.0)

    obs = torch.cat(
        [
            local_opp_root,
            opp_rot_tan_norm,
            local_opp_vel,
            local_opp_ang_vel,
            local_keys,
            local_key_vels,
            local_arena,
            health.unsqueeze(-1),
            opp_health.unsqueeze(-1),
            downed.unsqueeze(-1),
            opp_downed.unsqueeze(-1),
            round_time_left.unsqueeze(-1),
        ],
        dim=-1,
    )
    return obs


__all__ = ["compute_battle_task_obs"]
