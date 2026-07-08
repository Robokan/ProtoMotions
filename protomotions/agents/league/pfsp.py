# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""PFSP (Prioritized Fictitious Self-Play) statistics and opponent pool.

Port of IsaacLabASE's ``pfsp_player_pool.py`` statistics layer with the
design flaws identified in SOMA_GPC_COMBAT_PLAN.md §6b fixed:

1. *Dead snapshot gate* — the pool itself doesn't gate additions; the league
   agent enforces a hard staleness cap (see ``LeagueConfig``).
2. *FIFO eviction* → evict the least informative member (lowest trailing PFSP
   sampling weight), always retaining a protected set: the earliest snapshot
   (anti-cycling canary) and the highest-rated member.
3. *Inconsistent draw weighting* (0.25 / 0.5 mixed) → one constant
   ``DRAW_WEIGHT = 0.5`` (standard game-theoretic value) everywhere.
4. Win-rate-only ratings → members carry an Elo rating (see ``elo.py``),
   updated online by the league agent; PFSP sampling stays driven by win rate
   vs the *current* agent (as AlphaStar did).

Statistics are EMA-decayed with a configurable half-life so they track the
current agent, not its whole history.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# One draw weight, one place (fix for IsaacLabASE's 0.25 vs 0.5 inconsistency)
DRAW_WEIGHT = 0.5


@dataclass
class MemberStats:
    """EMA-decayed win/loss/draw counters for one pool member.

    Counters answer "how does the *current* agent fare against this member?"
    — ``wins`` counts the current agent's wins.
    """

    half_life_matches: float = 256.0
    games: float = 0.0
    wins: float = 0.0
    losses: float = 0.0
    draws: float = 0.0

    def update(self, wins: int, losses: int, draws: int) -> None:
        count = float(wins + losses + draws)
        if count <= 0:
            return
        decay = (0.5 ** (1.0 / self.half_life_matches)) ** count
        self.games = self.games * decay + count
        self.wins = self.wins * decay + float(wins)
        self.losses = self.losses * decay + float(losses)
        self.draws = self.draws * decay + float(draws)

    def win_rate(self, min_games: float = 10.0) -> float:
        """Current agent's win rate vs this member (draws = DRAW_WEIGHT)."""
        total = self.wins + self.losses + self.draws
        if total < min_games:
            return 0.5
        return (self.wins + DRAW_WEIGHT * self.draws) / max(total, 1.0)

    def conservative_score(self, min_games: float = 10.0) -> float:
        """Beta(1,1)-smoothed win rate (draws = DRAW_WEIGHT pseudo-wins)."""
        total = self.wins + self.losses + self.draws
        if total < min_games:
            return 0.5
        pseudo_wins = self.wins + DRAW_WEIGHT * self.draws
        return (pseudo_wins + 1.0) / (total + 2.0)

    def draw_rate(self) -> float:
        total = self.wins + self.losses + self.draws
        if total <= 0:
            return 0.0
        return self.draws / total

    def decisive_ratio(self) -> float:
        total = self.wins + self.losses + self.draws
        if total <= 0:
            return 0.0
        return (self.wins + self.losses) / total

    def reset(self) -> None:
        self.games = self.wins = self.losses = self.draws = 0.0


@dataclass
class PoolMember:
    """One league member: an adapter snapshot plus its bookkeeping."""

    member_id: int
    checkpoint_path: str
    label: str = ""
    stats: MemberStats = field(default_factory=MemberStats)
    rating: float = 1000.0
    # Trailing EMA of this member's PFSP sampling weight (eviction signal)
    usage_ema: float = 0.0
    creation_order: int = 0


PFSP_WEIGHTINGS = {
    "variance": lambda x: x * (1.0 - x),  # focus on evenly-matched opponents
    "linear": lambda x: 1.0 - x,  # focus on opponents we lose to
    "squared": lambda x: (1.0 - x) ** 2,
}


class PFSPPool:
    """League opponent pool with PFSP sampling and informed eviction."""

    def __init__(
        self,
        max_members: int,
        weighting: str = "linear",
        min_games: float = 10.0,
        min_decisive_ratio: float = 0.1,
        usage_ema_coef: float = 0.99,
        seed: Optional[int] = None,
    ):
        if weighting not in PFSP_WEIGHTINGS:
            raise ValueError(
                f"Unknown PFSP weighting '{weighting}'; options: {list(PFSP_WEIGHTINGS)}"
            )
        self.max_members = max_members
        self.weighting = weighting
        self.min_games = min_games
        self.min_decisive_ratio = min_decisive_ratio
        self.usage_ema_coef = usage_ema_coef
        self.members: Dict[int, PoolMember] = {}
        self._creation_counter = 0
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Membership
    # ------------------------------------------------------------------
    def add(self, checkpoint_path: str, label: str = "", rating: float = 1000.0) -> PoolMember:
        """Add a snapshot; evicts the least informative member if full."""
        if len(self.members) >= self.max_members:
            self._evict_one()
        member = PoolMember(
            member_id=self._creation_counter,
            checkpoint_path=checkpoint_path,
            label=label or f"policy_{self._creation_counter}",
            rating=rating,
            creation_order=self._creation_counter,
        )
        self.members[member.member_id] = member
        self._creation_counter += 1
        return member

    def _protected_ids(self) -> set:
        """Earliest snapshot (anti-cycling canary) + highest-rated member."""
        if not self.members:
            return set()
        earliest = min(self.members.values(), key=lambda m: m.creation_order)
        strongest = max(self.members.values(), key=lambda m: m.rating)
        return {earliest.member_id, strongest.member_id}

    def _evict_one(self) -> None:
        protected = self._protected_ids()
        candidates = [m for m in self.members.values() if m.member_id not in protected]
        if not candidates:
            # Degenerate small pool: fall back to evicting the oldest
            victim = min(self.members.values(), key=lambda m: m.creation_order)
        else:
            victim = min(candidates, key=lambda m: m.usage_ema)
        del self.members[victim.member_id]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample(self, weighting: Optional[str] = None) -> Optional[PoolMember]:
        """PFSP-sample one member (None if the pool is empty).

        Candidates need ``min_games`` recorded and a decisive-outcome ratio of
        at least ``min_decisive_ratio``; when nothing qualifies yet, sampling
        is uniform over all members.
        """
        if not self.members:
            return None
        weight_func = PFSP_WEIGHTINGS[weighting or self.weighting]

        candidates = [
            m
            for m in self.members.values()
            if (m.stats.wins + m.stats.losses + m.stats.draws) >= self.min_games
            and m.stats.decisive_ratio() >= self.min_decisive_ratio
        ]
        if not candidates:
            candidates = list(self.members.values())

        scores = [m.stats.conservative_score(self.min_games) for m in candidates]
        weights = [max(0.0, float(weight_func(s))) for s in scores]
        total = sum(weights)
        if total <= 0.0:
            chosen = self._rng.choice(candidates)
        else:
            chosen = self._rng.choices(candidates, weights=weights, k=1)[0]

        # Track trailing usage for informed eviction
        for m in self.members.values():
            hit = 1.0 if m.member_id == chosen.member_id else 0.0
            m.usage_ema = self.usage_ema_coef * m.usage_ema + (
                1.0 - self.usage_ema_coef
            ) * hit
        return chosen

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def record_result(self, member_id: int, wins: int, losses: int, draws: int) -> None:
        member = self.members.get(member_id)
        if member is not None:
            member.stats.update(wins, losses, draws)

    def average_win_rate(self) -> float:
        """Current agent's mean win rate over the pool (gate signal)."""
        if not self.members:
            return 0.0
        return sum(m.stats.win_rate(self.min_games) for m in self.members.values()) / len(
            self.members
        )

    def reset_stats(self) -> None:
        for m in self.members.values():
            m.stats.reset()

    def summary(self) -> List[dict]:
        return [
            {
                "id": m.member_id,
                "label": m.label,
                "win_rate_vs_agent": round(m.stats.win_rate(self.min_games), 3),
                "draw_rate": round(m.stats.draw_rate(), 3),
                "games": round(m.stats.games, 1),
                "rating": round(m.rating, 1),
                "usage": round(m.usage_ema, 4),
            }
            for m in sorted(self.members.values(), key=lambda m: -m.rating)
        ]


__all__ = ["DRAW_WEIGHT", "MemberStats", "PoolMember", "PFSPPool", "PFSP_WEIGHTINGS"]
