# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Battle evaluation tournament (SOMA_GPC_COMBAT_PLAN Phase 7).

Round-robin (and single-pairing) evaluation over slim adapter checkpoints on
a paired BattleEnv, decoupled from training-time statistics:

- All parallel matches run one pairing at a time (adapter A on the ego half,
  adapter B on the opponent half) until the requested match count completes,
  with randomized spawns per match.
- Outcomes update Elo ratings; results aggregate into a ladder table and a
  head-to-head matrix (the head-to-head matrix is what exposes
  non-transitivity — the interesting result).
- ``regression_gate`` is the clean-statistics admission check the plan
  requires before a snapshot enters the league: the candidate must beat the
  previous snapshot at ``threshold`` (default 55%) over a fixed match count.
"""

import itertools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from protomotions.agents.league.elo import elo_update
from protomotions.agents.league.pfsp import DRAW_WEIGHT

log = logging.getLogger(__name__)


@dataclass
class PairingResult:
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0

    @property
    def matches(self) -> int:
        return self.wins_a + self.wins_b + self.draws

    def score_a(self) -> float:
        if self.matches == 0:
            return 0.5
        return (self.wins_a + DRAW_WEIGHT * self.draws) / self.matches


@dataclass
class TournamentReport:
    adapters: List[str] = field(default_factory=list)
    ratings: Dict[str, float] = field(default_factory=dict)
    head_to_head: Dict[str, Dict[str, dict]] = field(default_factory=dict)

    def ladder(self) -> List[Tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda kv: -kv[1])

    def to_json(self, path: str) -> None:
        payload = {
            "ladder": [
                {"adapter": name, "elo": round(rating, 1)}
                for name, rating in self.ladder()
            ],
            "head_to_head": self.head_to_head,
        }
        Path(path).write_text(json.dumps(payload, indent=2))
        log.info("Tournament report written to %s", path)


class BattleTournament:
    """Drives adapter-vs-adapter matches on a league agent's environment.

    Requires a fully-built league agent (see ``battle_tournament.py``); uses
    its model for the ego side, its opponent lanes for the other side, and
    its SelfPlayEnvAdapter for match accounting.
    """

    def __init__(self, agent, deterministic: bool = False):
        self.agent = agent
        self.env = agent.env  # SelfPlayEnvAdapter
        self.device = agent.device
        self.deterministic = deterministic

        # Take over the adapter's callbacks for evaluation
        self._outcomes: List[int] = []  # +1 A wins, -1 B wins, 0 draw
        self.env.set_match_end_callback(self._record)
        self.env.set_opponent_policy(self._opponent_policy)
        self._opponent_model = None

        # Make sure the league agent never engages its own training league
        agent._league_initialized = True

    # ------------------------------------------------------------------
    def _record(self, ego_ids, win, lose, draw) -> None:
        for i in range(len(ego_ids)):
            if win[i] > 0.5:
                self._outcomes.append(1)
            elif lose[i] > 0.5:
                self._outcomes.append(-1)
            else:
                self._outcomes.append(0)

    def _opponent_policy(self, opp_obs):
        obs_td = self.agent._opponent_obs_td(opp_obs)
        with torch.no_grad():
            out = self._opponent_model(obs_td)
        key = "mean_action" if self.deterministic and "mean_action" in out else "action"
        return out[key]

    def _load_ego(self, adapter_path: str) -> None:
        self.agent.load_adapter_checkpoint(adapter_path)

    def _load_opponent(self, adapter_path: str) -> None:
        if self.agent._lanes is None:
            self.agent._build_lanes()
        lanes = self.agent._lanes
        state = torch.load(adapter_path, map_location=self.device, weights_only=False)
        adapter_state = state["model"] if "model" in state else state
        adapter_state = {k: v.to(self.device) for k, v in adapter_state.items()}
        lanes.lane_member = [None] * lanes.num_lanes
        lanes.assign(0, adapter_state)
        self._opponent_model = lanes.lanes[lanes._lane_of(0)]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_pairing(
        self, adapter_a: str, adapter_b: str, matches: int, max_steps: int = 200000
    ) -> PairingResult:
        """Run ``matches`` A-vs-B matches across all parallel arenas."""
        self._load_ego(adapter_a)
        self._load_opponent(adapter_b)
        self._outcomes = []

        obs, _ = self.env.reset()
        model = self.agent.model
        model.eval()

        steps = 0
        while len(self._outcomes) < matches and steps < max_steps:
            obs = self.agent.add_agent_info_to_obs(obs)
            obs_td = self.agent.obs_dict_to_tensordict(obs)
            out = model(obs_td)
            key = (
                "mean_action"
                if self.deterministic and "mean_action" in out
                else "action"
            )
            obs, _, dones, _, _ = self.env.step(out[key])
            done_ids = dones.nonzero(as_tuple=False).flatten()
            if len(done_ids) > 0:
                obs, _ = self.env.reset(done_ids)
            steps += 1

        outcomes = self._outcomes[:matches]
        result = PairingResult(
            wins_a=sum(1 for o in outcomes if o == 1),
            wins_b=sum(1 for o in outcomes if o == -1),
            draws=sum(1 for o in outcomes if o == 0),
        )
        log.info(
            "%s vs %s: %d-%d-%d (W-L-D over %d matches)",
            Path(adapter_a).stem,
            Path(adapter_b).stem,
            result.wins_a,
            result.wins_b,
            result.draws,
            result.matches,
        )
        return result

    def run_round_robin(
        self,
        adapters: List[str],
        matches_per_pairing: int,
        elo_k: float = 32.0,
        initial_rating: float = 1000.0,
    ) -> TournamentReport:
        """Full round-robin over the adapter list, both colors per pairing."""
        names = [Path(a).stem for a in adapters]
        report = TournamentReport(adapters=list(adapters))
        report.ratings = {n: initial_rating for n in names}
        report.head_to_head = {n: {} for n in names}

        for (i, a), (j, b) in itertools.combinations(enumerate(adapters), 2):
            half = max(1, matches_per_pairing // 2)
            res_ab = self.run_pairing(a, b, half)
            res_ba = self.run_pairing(b, a, matches_per_pairing - half)
            combined = PairingResult(
                wins_a=res_ab.wins_a + res_ba.wins_b,
                wins_b=res_ab.wins_b + res_ba.wins_a,
                draws=res_ab.draws + res_ba.draws,
            )
            na, nb = names[i], names[j]
            report.head_to_head[na][nb] = {
                "wins": combined.wins_a,
                "losses": combined.wins_b,
                "draws": combined.draws,
            }
            report.head_to_head[nb][na] = {
                "wins": combined.wins_b,
                "losses": combined.wins_a,
                "draws": combined.draws,
            }
            # Sequential Elo updates, one per match
            for outcome in (
                [1] * combined.wins_a + [-1] * combined.wins_b + [0] * combined.draws
            ):
                score = 1.0 if outcome == 1 else (0.0 if outcome == -1 else 0.5)
                report.ratings[na], report.ratings[nb] = elo_update(
                    report.ratings[na], report.ratings[nb], score, k=elo_k
                )

        for name, rating in report.ladder():
            log.info("  %-40s Elo %.0f", name, rating)
        return report

    def regression_gate(
        self,
        candidate: str,
        previous: str,
        matches: int = 64,
        threshold: float = 0.55,
    ) -> bool:
        """Admission check: candidate must score >= threshold vs previous.

        Uses clean evaluation matches, decoupling the gate from noisy
        training-time counters (a correctness fix over IsaacLabASE).
        """
        half = max(1, matches // 2)
        res_fwd = self.run_pairing(candidate, previous, half)
        res_rev = self.run_pairing(previous, candidate, matches - half)
        combined = PairingResult(
            wins_a=res_fwd.wins_a + res_rev.wins_b,
            wins_b=res_fwd.wins_b + res_rev.wins_a,
            draws=res_fwd.draws + res_rev.draws,
        )
        score = combined.score_a()
        passed = score >= threshold
        log.info(
            "Regression gate: %s vs %s score %.3f (threshold %.2f) -> %s",
            Path(candidate).stem,
            Path(previous).stem,
            score,
            threshold,
            "PASS" if passed else "FAIL",
        )
        return passed


__all__ = ["BattleTournament", "PairingResult", "TournamentReport"]
