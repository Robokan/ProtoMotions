# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""League self-play trainer for battle RLFT (SOMA_GPC_COMBAT_PLAN Phase 6).

Extends the discrete-prior PEFT RLFT agent (PPO over token logits, KL to the
SFT anchor, prior-constrained nucleus sampling) with tournament self-play:

- The agent trains on the ego half of a paired :class:`BattleEnv` through
  :class:`SelfPlayEnvAdapter`; opponents run frozen adapter snapshots through
  :class:`OpponentLanes` (shared frozen prior, per-lane adapters).
- PFSP opponent sampling over EMA win rates, per-match resampling on episode
  end (the agent fights the whole league simultaneously across the batch).
- Snapshot gating on pool-average win rate PLUS a hard staleness cap — the
  fix for IsaacLabASE's dead ``force_add`` gate, which let league growth
  silently stall when the agent plateaued.
- Informed pool eviction with a protected set (earliest snapshot + highest
  Elo) instead of FIFO.
- Online Elo ratings for monitoring/eval seeding; sampling stays win-rate
  driven (AlphaStar convention).
- League persistence: snapshots are slim adapter checkpoints in
  ``league_dir``; restarts rebuild the pool from disk.
- ``force_symmetric_inference`` debug switch: opponents run the live training
  policy — with identical policies the win rate must be ~50%, which is the
  fastest way to catch env asymmetries.
- Exploiter role (``role="main_exploiter"``): trains only against the most
  recent snapshot of a main run's league, refreshed periodically.
"""

import copy
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import Tensor

from protomotions.agents.league.elo import elo_update
from protomotions.agents.league.lanes import OpponentLanes
from protomotions.agents.league.pfsp import PFSPPool, PoolMember
from protomotions.agents.league.self_play_env import SelfPlayEnvAdapter
from protomotions.agents.peft.prior_agent import DiscretePriorPEFTRLFTAgent
from protomotions.agents.peft.prior_config import DiscretePriorPEFTRLFTAgentConfig

log = logging.getLogger(__name__)


@dataclass
class LeagueParams:
    """League/self-play hyperparameters."""

    # Pool
    max_members: int = 32
    half_life_matches: float = 256.0
    min_games: float = 10.0
    min_decisive_ratio: float = 0.1
    # PFSP weighting schedule: start uniform-ish pressure, focus on ~50%
    # opponents once the league has substance (plan §6c).
    pfsp_weighting: str = "linear"
    pfsp_weighting_mature: str = "variance"
    mature_after_members: int = 8

    # Snapshot gating
    gate_win_rate: float = 0.7
    gate_min_games: int = 200  # raw finished matches since the last snapshot
    staleness_epochs: int = 50  # hard cap: force a snapshot regardless of gate

    # Opponent serving
    num_lanes: int = 4
    share_frozen_base: bool = True

    # Ratings
    elo_k: float = 32.0
    initial_rating: float = 1000.0

    # Roles
    role: str = "main"  # "main" | "main_exploiter"
    exploiter_opponent_dir: Optional[str] = None
    exploiter_refresh_epochs: int = 5

    # Debug
    force_symmetric_inference: bool = False


@dataclass
class LeagueDiscretePriorPEFTRLFTAgentConfig(DiscretePriorPEFTRLFTAgentConfig):
    _target_: str = "protomotions.agents.league.agent.LeagueDiscretePriorPEFTRLFTAgent"
    league: LeagueParams = field(default_factory=LeagueParams)


class LeagueDiscretePriorPEFTRLFTAgent(DiscretePriorPEFTRLFTAgent):
    """RLFT agent whose environment is one side of a self-play league."""

    config: LeagueDiscretePriorPEFTRLFTAgentConfig

    def __init__(self, fabric, env, config, root_dir=None):
        adapter = SelfPlayEnvAdapter(env)
        super().__init__(fabric, adapter, config, root_dir=root_dir)

        league_cfg: LeagueParams = config.league
        self.league_cfg = league_cfg

        self.league_dir = Path(self._resolve_root_dir(root_dir)) / "league"
        self.pool = PFSPPool(
            max_members=league_cfg.max_members,
            weighting=league_cfg.pfsp_weighting,
            min_games=league_cfg.min_games,
            min_decisive_ratio=league_cfg.min_decisive_ratio,
        )

        num_matches = adapter.num_matches
        # member id assigned per ego env (-1 = unassigned)
        self.env_member = torch.full(
            (num_matches,), -1, dtype=torch.long, device=self.device
        )
        self.agent_rating = league_cfg.initial_rating
        self.games_since_snapshot = 0
        self.last_snapshot_epoch = 0
        self._snapshot_counter = 0
        self._league_initialized = False
        self._lanes: Optional[OpponentLanes] = None
        self._adapter_cache: Dict[int, Dict[str, Tensor]] = {}
        self._exploiter_ckpt_path: Optional[str] = None
        self._exploiter_last_refresh_epoch = -(10**9)

        adapter.set_opponent_policy(self._opponent_policy)
        adapter.set_match_end_callback(self._on_matches_ended)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def _resolve_root_dir(self, root_dir) -> Path:
        if root_dir is not None:
            return Path(root_dir)
        # Fall back to the fabric logger directory, mirroring BaseAgent.save
        logger_dir = getattr(getattr(self.fabric, "logger", None), "log_dir", None)
        return Path(logger_dir) if logger_dir else Path(".")

    # ------------------------------------------------------------------
    # League lifecycle
    # ------------------------------------------------------------------
    def _ensure_league_initialized(self) -> None:
        if self._league_initialized:
            return
        self._league_initialized = True
        self.league_dir.mkdir(parents=True, exist_ok=True)

        if self.league_cfg.role == "main_exploiter":
            self._refresh_exploiter_opponent(force=True)
        else:
            if not self.pool.members:  # not already restored from a checkpoint
                self._restore_league_from_disk()
            if not self.pool.members:
                # Seed the league with the warm-start (SFT) adapter so the
                # first opponents exist (IsaacLabASE seeded with init_op_model).
                self._take_snapshot(reason="seed")

        self._build_lanes()
        all_matches = torch.arange(
            self.env_member.shape[0], device=self.device, dtype=torch.long
        )
        self._resample_opponents(all_matches)

    def _restore_league_from_disk(self) -> None:
        """Rebuild the pool from league_dir snapshots (most recent first)."""
        snapshots = sorted(
            self.league_dir.glob("policy_*.ckpt"), key=lambda p: p.stat().st_mtime
        )
        if not snapshots:
            return
        keep = snapshots[-self.league_cfg.max_members :]
        for path in keep:
            rating = self.league_cfg.initial_rating
            try:
                meta = torch.load(path, map_location="cpu", weights_only=False)
                rating = float(meta.get("rating", rating))
            except Exception as exc:  # noqa: BLE001 - resumability over strictness
                log.warning("Could not read league snapshot metadata %s: %s", path, exc)
            self.pool.add(str(path), label=path.stem, rating=rating)
        self._snapshot_counter = len(snapshots)
        log.info(
            "Restored league: %d/%d snapshots from %s",
            len(keep),
            len(snapshots),
            self.league_dir,
        )

    def _unwrapped_model(self):
        return getattr(self.model, "module", self.model)

    def _build_lanes(self) -> None:
        base_model = self._unwrapped_model()

        def factory():
            return copy.deepcopy(base_model)

        self._lanes = OpponentLanes(
            model_factory=factory,
            num_lanes=self.league_cfg.num_lanes,
            share_frozen_base_with=(
                base_model if self.league_cfg.share_frozen_base else None
            ),
        )

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------
    def _adapter_state_cpu(self) -> Dict[str, Tensor]:
        actor = self._unwrapped_model()._actor
        return {
            key: value.detach().cpu().clone()
            for key, value in actor.adapter_state_dict().items()
        }

    def _take_snapshot(self, reason: str) -> PoolMember:
        path = self.league_dir / f"policy_{self._snapshot_counter}.ckpt"
        torch.save(
            {
                "model": self._adapter_state_cpu(),
                "epoch": self.current_epoch,
                "rating": self.agent_rating,
                "reason": reason,
                "time": time.time(),
            },
            path,
        )
        member = self.pool.add(str(path), label=path.stem, rating=self.agent_rating)
        self._snapshot_counter += 1
        self.games_since_snapshot = 0
        self.last_snapshot_epoch = self.current_epoch
        # Stats track the current agent; a new snapshot changes what "current"
        # means for future comparisons, so reset per-opponent counters.
        self.pool.reset_stats()

        # Weighting schedule
        if len(self.pool.members) >= self.league_cfg.mature_after_members:
            self.pool.weighting = self.league_cfg.pfsp_weighting_mature

        log.info(
            "League snapshot %s (%s): pool=%d members, agent Elo=%.0f",
            path.name,
            reason,
            len(self.pool.members),
            self.agent_rating,
        )
        return member

    def _load_member_adapter(self, member_id: int) -> Dict[str, Tensor]:
        cached = self._adapter_cache.get(member_id)
        if cached is not None:
            return cached
        member = self.pool.members[member_id]
        state = torch.load(
            member.checkpoint_path, map_location=self.device, weights_only=False
        )
        adapter_state = state["model"] if "model" in state else state
        adapter_state = {k: v.to(self.device) for k, v in adapter_state.items()}
        self._adapter_cache[member_id] = adapter_state
        return adapter_state

    # ------------------------------------------------------------------
    # Opponent sampling / serving
    # ------------------------------------------------------------------
    def _live_members(self, excluding_envs: Optional[Tensor] = None) -> set:
        assigned = self.env_member
        if excluding_envs is not None:
            mask = torch.ones_like(assigned, dtype=torch.bool)
            mask[excluding_envs] = False
            assigned = assigned[mask]
        return {int(m) for m in assigned.unique().tolist() if m >= 0}

    def _sample_member_capped(self, live: set) -> Optional[PoolMember]:
        """PFSP-sample, keeping distinct live members within lane capacity."""
        member = self.pool.sample()
        if member is None:
            return None
        if member.member_id in live or len(live) < self.league_cfg.num_lanes:
            return member
        # Lanes full: restrict to the already-live subset (uniform-ish).
        live_members = [self.pool.members[m] for m in live if m in self.pool.members]
        if not live_members:
            return member
        scores = [m.stats.conservative_score(self.pool.min_games) for m in live_members]
        from protomotions.agents.league.pfsp import PFSP_WEIGHTINGS

        weight_func = PFSP_WEIGHTINGS[self.pool.weighting]
        weights = [max(0.0, float(weight_func(s))) for s in scores]
        total = sum(weights)
        if total <= 0:
            return live_members[0]
        import random

        return random.choices(live_members, weights=weights, k=1)[0]

    def _resample_opponents(self, ego_ids: Tensor) -> None:
        if self.league_cfg.role == "main_exploiter":
            self.env_member[ego_ids] = 0
            return
        if not self.pool.members:
            return
        for env in ego_ids.tolist():
            live = self._live_members(
                excluding_envs=torch.tensor([env], device=self.device)
            )
            member = self._sample_member_capped(live)
            if member is None:
                continue
            self._lanes.assign(
                member.member_id,
                self._load_member_adapter(member.member_id),
                in_use=live,
            )
            self.env_member[env] = member.member_id

    def _opponent_obs_td(self, opp_obs: Dict[str, Tensor]):
        obs = self.add_agent_info_to_obs(opp_obs)
        return self.obs_dict_to_tensordict(obs)

    def _opponent_policy(self, opp_obs: Dict[str, Tensor]) -> Tensor:
        self._ensure_league_initialized()
        obs_td = self._opponent_obs_td(opp_obs)

        use_self = (
            self.league_cfg.force_symmetric_inference
            or (self.league_cfg.role != "main_exploiter" and not self.pool.members)
        )
        if use_self:
            with torch.no_grad():
                out = self.model(obs_td)
            return out["action"]

        # (env.get_action_size() references a stale simulator attr upstream)
        action_dim = self.env.robot_config.number_of_actions
        return self._lanes.act(obs_td, self.env_member, action_dim)

    # ------------------------------------------------------------------
    # Match accounting
    # ------------------------------------------------------------------
    def _on_matches_ended(
        self, ego_ids: Tensor, win: Tensor, lose: Tensor, draw: Tensor
    ) -> None:
        if not self._league_initialized:
            return
        self.games_since_snapshot += len(ego_ids)

        if self.league_cfg.role != "main_exploiter" and self.pool.members:
            # Aggregate per member, update stats + Elo
            for i, env in enumerate(ego_ids.tolist()):
                member_id = int(self.env_member[env])
                member = self.pool.members.get(member_id)
                if member is None:
                    continue
                w, l, d = int(win[i]), int(lose[i]), int(draw[i])
                self.pool.record_result(member_id, wins=w, losses=l, draws=d)
                score = 1.0 if w else (0.5 if d else 0.0)
                self.agent_rating, member.rating = elo_update(
                    self.agent_rating, member.rating, score, k=self.league_cfg.elo_k
                )

        self._resample_opponents(ego_ids)

    # ------------------------------------------------------------------
    # Exploiter role
    # ------------------------------------------------------------------
    def _refresh_exploiter_opponent(self, force: bool = False) -> None:
        cfg = self.league_cfg
        if cfg.exploiter_opponent_dir is None:
            raise ValueError(
                "role='main_exploiter' requires league.exploiter_opponent_dir "
                "(the main run's league directory)"
            )
        if (
            not force
            and self.current_epoch - self._exploiter_last_refresh_epoch
            < cfg.exploiter_refresh_epochs
        ):
            return
        self._exploiter_last_refresh_epoch = self.current_epoch
        snapshots = sorted(
            Path(cfg.exploiter_opponent_dir).glob("policy_*.ckpt"),
            key=lambda p: p.stat().st_mtime,
        )
        if not snapshots:
            raise FileNotFoundError(
                f"No league snapshots in {cfg.exploiter_opponent_dir}"
            )
        latest = str(snapshots[-1])
        if latest != self._exploiter_ckpt_path:
            self._exploiter_ckpt_path = latest
            if not self.pool.members:
                self.pool.add(latest, label=Path(latest).stem)
            else:
                member = self.pool.members[
                    next(iter(self.pool.members))
                ]
                member.checkpoint_path = latest
                member.label = Path(latest).stem
            self._adapter_cache.clear()
            if self._lanes is not None:
                member_id = next(iter(self.pool.members))
                self._lanes.lane_member = [None] * self._lanes.num_lanes
                self._lanes.assign(member_id, self._load_member_adapter(member_id))
            log.info("Exploiter target refreshed: %s", latest)

    # ------------------------------------------------------------------
    # Epoch hook: gating, staleness, logging
    # ------------------------------------------------------------------
    def post_epoch_logging(self, training_log_dict: dict):
        if self._league_initialized:
            cfg = self.league_cfg
            if cfg.role == "main_exploiter":
                self._refresh_exploiter_opponent()
            else:
                pool_avg = self.pool.average_win_rate()
                gated = (
                    self.games_since_snapshot >= cfg.gate_min_games
                    and pool_avg >= cfg.gate_win_rate
                )
                stale = (
                    self.current_epoch - self.last_snapshot_epoch
                    >= cfg.staleness_epochs
                )
                if gated:
                    self._take_snapshot(reason="gate")
                elif stale:
                    # Staleness cap: if the agent plateaus below the gate the
                    # league must still grow (fix for the dead force_add gate).
                    self._take_snapshot(reason="staleness")

                training_log_dict["league/pool_size"] = float(len(self.pool.members))
                training_log_dict["league/pool_avg_win_rate"] = pool_avg
                training_log_dict["league/agent_elo"] = self.agent_rating
                training_log_dict["league/games_since_snapshot"] = float(
                    self.games_since_snapshot
                )
                ratings = [m.rating for m in self.pool.members.values()]
                if ratings:
                    training_log_dict["league/top_member_elo"] = max(ratings)

        super().post_epoch_logging(training_log_dict)

    # ------------------------------------------------------------------
    # Persistence of league bookkeeping inside the training checkpoint
    # ------------------------------------------------------------------
    def get_state_dict(self, state_dict):
        state_dict = super().get_state_dict(state_dict)
        state_dict["league"] = {
            "agent_rating": self.agent_rating,
            "snapshot_counter": self._snapshot_counter,
            "last_snapshot_epoch": self.last_snapshot_epoch,
            "games_since_snapshot": self.games_since_snapshot,
            "members": {
                mid: {
                    "checkpoint_path": m.checkpoint_path,
                    "label": m.label,
                    "rating": m.rating,
                    "creation_order": m.creation_order,
                    "stats": vars(m.stats).copy(),
                }
                for mid, m in self.pool.members.items()
            },
        }
        return state_dict

    def load_parameters(self, state_dict, load_training_state: bool = True):
        super().load_parameters(state_dict, load_training_state=load_training_state)
        league = state_dict.get("league")
        if league is None or not load_training_state:
            return
        self.agent_rating = league.get("agent_rating", self.agent_rating)
        self._snapshot_counter = league.get("snapshot_counter", 0)
        self.last_snapshot_epoch = league.get("last_snapshot_epoch", 0)
        self.games_since_snapshot = league.get("games_since_snapshot", 0)
        members = league.get("members", {})
        for mid, m in members.items():
            if not Path(m["checkpoint_path"]).exists():
                log.warning("League snapshot missing on resume: %s", m["checkpoint_path"])
                continue
            member = self.pool.add(
                m["checkpoint_path"], label=m["label"], rating=m["rating"]
            )
            member.creation_order = m.get("creation_order", member.creation_order)
            for key, value in m.get("stats", {}).items():
                setattr(member.stats, key, value)
        if self.pool.members:
            self._league_initialized = False  # lanes rebuilt lazily with pool


__all__ = [
    "LeagueParams",
    "LeagueDiscretePriorPEFTRLFTAgentConfig",
    "LeagueDiscretePriorPEFTRLFTAgent",
]
