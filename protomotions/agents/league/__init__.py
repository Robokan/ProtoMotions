# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""League self-play training for battle tasks (SOMA_GPC_COMBAT_PLAN Phase 6)."""

from protomotions.agents.league.elo import elo_update, expected_score
from protomotions.agents.league.pfsp import (
    DRAW_WEIGHT,
    MemberStats,
    PFSPPool,
    PoolMember,
)

__all__ = [
    "DRAW_WEIGHT",
    "MemberStats",
    "PFSPPool",
    "PoolMember",
    "elo_update",
    "expected_score",
]
