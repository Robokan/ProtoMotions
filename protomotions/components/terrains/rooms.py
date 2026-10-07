# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Where each env's room goes: a grid of centres on the walkable terrain.

Shared by the simulator (which places a room at each centre) and the env
(which spawns each robot at its own room's centre), so the two cannot
disagree about where env i lives. See SimulatorConfig.rooms.
"""

from __future__ import annotations

import math

import torch


def room_centers(terrain, num_envs: int, spacing: float, half_extent) -> torch.Tensor:
    """[num_envs, 2] world xy of each env's room centre.

    A near-square grid, spacing apart, centred on the middle of the walkable
    terrain. Raises if a room would hang off the walkable area -- a robot
    there would walk off the ground the policy stands on.
    """
    xs, ys = terrain.walkable_x_coords, terrain.walkable_y_coords
    cx, cy = float(xs.mean()), float(ys.mean())
    cols = max(int(math.ceil(math.sqrt(num_envs))), 1)
    rows = int(math.ceil(num_envs / cols))
    index = torch.arange(num_envs, dtype=torch.float32)
    col = index % cols
    row = torch.div(index, cols, rounding_mode="floor")
    centers = torch.stack(
        [
            cx + (col - (cols - 1) / 2.0) * spacing,
            cy + (row - (rows - 1) / 2.0) * spacing,
        ],
        dim=-1,
    )
    hx, hy = float(half_extent[0]), float(half_extent[1])
    lo_x, hi_x = float(xs.min()), float(xs.max())
    lo_y, hi_y = float(ys.min()), float(ys.max())
    outside = (
        (centers[:, 0] - hx < lo_x) | (centers[:, 0] + hx > hi_x)
        | (centers[:, 1] - hy < lo_y) | (centers[:, 1] + hy > hi_y)
    )
    if bool(outside.any()):
        raise ValueError(
            f"{int(outside.sum())} of {num_envs} rooms ({cols}x{rows} grid, "
            f"{spacing:g} m apart) do not fit on the walkable terrain "
            f"x[{lo_x:.0f}, {hi_x:.0f}] y[{lo_y:.0f}, {hi_y:.0f}]. Use fewer "
            "envs or a smaller rooms.spacing."
        )
    return centers
