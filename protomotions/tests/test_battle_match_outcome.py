# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Payout rules for how a battle match ends.

Regression cover for the ring-out exploit found 2026-09-02: leaving the ring
resolved the match on a health-points decision AND collected the early-finish
multiplier, so "land one hit, then walk out" paid up to 2x the win reward at no
risk. Seven days of training converged on it -- 93% of matches ended
out-of-bounds, KOs at 0.0003, opponent health at 98.9%.
"""

import torch

from protomotions.envs.battle.control import resolve_match_outcome

EPS = 0.02
DRAW = -0.25
SCALE = 1.0


def outcome(
    *,
    i_lose=False,
    they_lose=False,
    oob=False,
    oob_self=None,
    my_health=1.0,
    their_health=1.0,
    time_left=0.9,
    ends=True,
):
    t = lambda v: torch.tensor([v])
    return float(
        resolve_match_outcome(
            loses_now=t(i_lose),
            loses_now_partner=t(they_lose),
            oob_end=t(oob),
            oob_self=t(oob if oob_self is None else oob_self),
            ends=t(ends),
            health=t(float(my_health)),
            health_partner=t(float(their_health)),
            time_left=t(float(time_left)),
            points_decision_eps=EPS,
            draw_signal=DRAW,
            early_finish_win_scale=SCALE,
        )[0]
    )


def test_walking_out_while_ahead_is_a_loss():
    """THE EXPLOIT: ahead on health, step out early. Must be a full loss."""
    assert outcome(oob=True, my_health=1.0, their_health=0.90, time_left=0.95) == -1.0


def test_walking_out_while_behind_is_a_loss():
    """Leaving must not be an escape hatch from a losing position."""
    assert outcome(oob=True, my_health=0.5, their_health=1.0, time_left=0.95) == -1.0


def test_walking_out_is_worse_than_a_draw():
    """0 payout would beat draw_signal and make leaving attractive."""
    ring_out = outcome(oob=True, my_health=1.0, their_health=1.0)
    draw = outcome(my_health=1.0, their_health=1.0)
    assert ring_out < draw, (ring_out, draw)


def test_opponent_leaving_pays_me_nothing():
    """Shoving them out must not become a strategy of its own."""
    assert outcome(oob=True, oob_self=False, my_health=1.0, their_health=0.5) == 0.0


def test_ko_win_still_earns_early_finish_bonus():
    """A real finish must stay the most valuable outcome."""
    got = outcome(they_lose=True, time_left=1.0)
    assert got == 2.0, got  # 1.0 * (1 + 1.0 * 1.0)


def test_ko_win_late_earns_less_than_early():
    assert outcome(they_lose=True, time_left=0.1) < outcome(
        they_lose=True, time_left=0.9
    )


def test_points_win_is_not_scaled_by_early_finish():
    """Stopping the clock while ahead must not be amplified."""
    early = outcome(my_health=1.0, their_health=0.5, time_left=0.99)
    late = outcome(my_health=1.0, their_health=0.5, time_left=0.0)
    assert early == late == 1.0, (early, late)


def test_ko_beats_walking_out_and_beats_points():
    """Ordering is what steers behaviour: KO > points > no contest."""
    ko = outcome(they_lose=True, time_left=0.9)
    points = outcome(my_health=1.0, their_health=0.5, time_left=0.9)
    ring_out = outcome(oob=True, my_health=1.0, their_health=0.5, time_left=0.9)
    assert ko > points > 0 > ring_out, (ko, points, ring_out)


def test_ko_loss_is_never_amplified():
    assert outcome(i_lose=True, time_left=1.0) == -1.0


def test_draw_pays_draw_signal_unscaled():
    assert outcome(my_health=1.0, their_health=1.0, time_left=1.0) == DRAW


def test_simultaneous_loss_is_a_draw():
    assert outcome(i_lose=True, they_lose=True, time_left=0.9) == DRAW


def test_ko_takes_precedence_over_oob():
    """A fighter cannot dodge a loss by leaving the ring on the same step.

    The caller builds oob_end with ``& ~loses_now & ~loses_now[partner]``, so a
    KO on the same step clears it; verify the payout honours that.
    """
    assert outcome(i_lose=True, oob=False, time_left=0.9) == -1.0


def test_no_payout_before_the_match_ends():
    assert outcome(ends=False, my_health=1.0, their_health=0.5) == 0.0
