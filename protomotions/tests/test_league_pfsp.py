# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the league layer: PFSP statistics/pool and Elo."""

import pytest

from protomotions.agents.league.elo import elo_update, expected_score
from protomotions.agents.league.pfsp import (
    DRAW_WEIGHT,
    MemberStats,
    PFSPPool,
)


# ---------- statistics ------------------------------------------------------


def test_member_stats_consistent_draw_weighting():
    """One draw constant everywhere (fix for the 0.25/0.5 inconsistency)."""
    stats = MemberStats()
    stats.update(wins=10, losses=10, draws=20)
    total = 10 + 10 + 20
    expected = (10 + DRAW_WEIGHT * 20) / total
    assert stats.win_rate(min_games=1) == pytest.approx(expected, rel=1e-6)

    conservative = stats.conservative_score(min_games=1)
    assert conservative == pytest.approx((10 + DRAW_WEIGHT * 20 + 1) / (total + 2))


def test_member_stats_below_min_games_returns_half():
    stats = MemberStats()
    stats.update(wins=2, losses=0, draws=0)
    assert stats.win_rate(min_games=10) == 0.5
    assert stats.conservative_score(min_games=10) == 0.5


def test_member_stats_ema_decay_tracks_current_agent():
    """Old results decay: a long losing streak then wins shifts the rate."""
    stats = MemberStats(half_life_matches=8.0)
    for _ in range(20):
        stats.update(wins=0, losses=1, draws=0)
    early = stats.win_rate(min_games=1)
    for _ in range(20):
        stats.update(wins=1, losses=0, draws=0)
    late = stats.win_rate(min_games=1)
    assert early < 0.1
    assert late > 0.8, "with an 8-match half-life, recent wins must dominate"


def test_decisive_ratio():
    stats = MemberStats()
    stats.update(wins=1, losses=1, draws=18)
    assert stats.decisive_ratio() == pytest.approx(0.1)


# ---------- pool: sampling ---------------------------------------------------


def _pool(**kwargs):
    defaults = dict(max_members=4, weighting="linear", min_games=1.0, seed=7)
    defaults.update(kwargs)
    return PFSPPool(**defaults)


def test_pfsp_prefers_opponents_agent_loses_to():
    pool = _pool()
    weak = pool.add("weak.ckpt")  # agent always beats it
    strong = pool.add("strong.ckpt")  # agent always loses to it
    for _ in range(30):
        pool.record_result(weak.member_id, wins=1, losses=0, draws=0)
        pool.record_result(strong.member_id, wins=0, losses=1, draws=0)

    counts = {weak.member_id: 0, strong.member_id: 0}
    for _ in range(500):
        counts[pool.sample().member_id] += 1
    assert counts[strong.member_id] > counts[weak.member_id] * 2


def test_pfsp_variance_weighting_prefers_even_matches():
    pool = _pool(weighting="variance")
    even = pool.add("even.ckpt")
    lopsided = pool.add("lopsided.ckpt")
    for _ in range(30):
        pool.record_result(even.member_id, wins=1, losses=1, draws=0)
        pool.record_result(lopsided.member_id, wins=2, losses=0, draws=0)

    counts = {even.member_id: 0, lopsided.member_id: 0}
    for _ in range(500):
        counts[pool.sample().member_id] += 1
    assert counts[even.member_id] > counts[lopsided.member_id]


def test_pfsp_sample_empty_pool_returns_none():
    assert _pool().sample() is None


def test_indecisive_members_excluded_until_fallback():
    pool = _pool(min_decisive_ratio=0.2)
    drawish = pool.add("drawish.ckpt")
    normal = pool.add("normal.ckpt")
    for _ in range(20):
        pool.record_result(drawish.member_id, wins=0, losses=0, draws=1)
        pool.record_result(normal.member_id, wins=1, losses=1, draws=0)
    counts = {drawish.member_id: 0, normal.member_id: 0}
    for _ in range(200):
        counts[pool.sample().member_id] += 1
    assert counts[drawish.member_id] == 0, "stalemate-only members are filtered"


# ---------- pool: eviction ----------------------------------------------------


def test_eviction_protects_earliest_and_highest_rated():
    pool = _pool(max_members=3)
    first = pool.add("first.ckpt", rating=1000)
    mid = pool.add("mid.ckpt", rating=1000)
    champ = pool.add("champ.ckpt", rating=1500)

    # Give mid the lowest usage, then overflow the pool
    first.usage_ema = 0.5
    mid.usage_ema = 0.0
    champ.usage_ema = 0.4
    pool.add("new.ckpt")

    ids = set(pool.members)
    assert first.member_id in ids, "earliest snapshot is the anti-cycling canary"
    assert champ.member_id in ids, "highest-rated member is protected"
    assert mid.member_id not in ids, "least-used unprotected member is evicted"


def test_pool_never_exceeds_max_members():
    pool = _pool(max_members=3)
    for i in range(10):
        pool.add(f"p{i}.ckpt")
    assert len(pool.members) == 3


def test_average_win_rate_gate_signal():
    pool = _pool()
    a = pool.add("a.ckpt")
    b = pool.add("b.ckpt")
    for _ in range(20):
        pool.record_result(a.member_id, wins=1, losses=0, draws=0)
        pool.record_result(b.member_id, wins=1, losses=0, draws=0)
    assert pool.average_win_rate() > 0.9


# ---------- Elo ---------------------------------------------------------------


def test_elo_symmetric_zero_sum():
    a, b = elo_update(1000.0, 1000.0, score_a=1.0)
    assert a + b == pytest.approx(2000.0)
    assert a > 1000.0 > b


def test_elo_draw_moves_toward_underdog():
    a, b = elo_update(1200.0, 1000.0, score_a=0.5)
    assert a < 1200.0 and b > 1000.0


def test_elo_expected_score_monotonic():
    assert expected_score(1200, 1000) > 0.5 > expected_score(1000, 1200)
    assert expected_score(1000, 1000) == pytest.approx(0.5)


def test_elo_rejects_invalid_score():
    with pytest.raises(ValueError):
        elo_update(1000, 1000, score_a=1.5)
