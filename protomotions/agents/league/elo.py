# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Minimal two-player Elo with draw support.

Standard Elo update; a draw scores 0.5 for both sides. Used for league
monitoring and evaluation seeding — PFSP opponent sampling stays win-rate
driven (plan §6b.4).
"""

from typing import Tuple


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_update(
    rating_a: float,
    rating_b: float,
    score_a: float,
    k: float = 32.0,
) -> Tuple[float, float]:
    """Update both ratings from one match.

    Args:
        rating_a: Player A's rating.
        rating_b: Player B's rating.
        score_a: A's result — 1.0 win, 0.5 draw, 0.0 loss.
        k: Elo K-factor.

    Returns:
        Tuple of updated (rating_a, rating_b).
    """
    if not 0.0 <= score_a <= 1.0:
        raise ValueError(f"score_a must be in [0, 1], got {score_a}")
    e_a = expected_score(rating_a, rating_b)
    delta = k * (score_a - e_a)
    return rating_a + delta, rating_b - delta


__all__ = ["expected_score", "elo_update"]
