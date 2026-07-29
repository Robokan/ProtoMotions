# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Self-play league orchestration for full-model-snapshot agents.

Extracted verbatim from agents/league/ase_agent.py so the ASE full-weight
league and the paper-faithful ASE HLC league (frozen low-level controller +
high-level latent policy) share one implementation. The PEFT league
(agents/league/agent.py) remains separate on purpose — its snapshots are
adapter slices and the running SOMA league's code path stays untouched.

Snapshots are full ``model.state_dict()`` payloads, drop-in compatible with
the PEFT snapshot format ({"model": ..., "epoch", "rating", "reason",
"time"}) plus provenance keys ("architecture": cls.snapshot_architecture,
"robot"). Whatever lives OUTSIDE the model (e.g. a frozen shared LLC) is by
construction excluded from snapshots and lanes.

Subclass hooks:
    - ``snapshot_architecture``: provenance tag written into snapshots.
    - ``_opponent_action_dim()``: width of the lane output ("action" key).
    - ``_opponent_obs_td(opp_obs)``: build the opponent observation
      TensorDict (override to inject per-opponent state, e.g. ASE latents).
    - ``_pre_opponent_policy()``: per-step hook before opponent inference.
    - ``_on_opponents_resampled(ego_ids)``: hook after opponent resampling.

The mixin must precede the algorithm class in the MRO, e.g.
``class LeagueASEAgent(FullModelLeagueMixin, ASE)``. The subclass __init__
wraps its env in a SelfPlayEnvAdapter, calls ``_init_league`` after
``super().__init__``, and wires ``adapter.set_opponent_policy`` /
``set_match_end_callback``.

The exploiter role is not supported (single-architecture self-play only).
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import Tensor

from protomotions.agents.league.agent import LeagueParams
from protomotions.agents.league.lanes import OpponentLanes
from protomotions.agents.league.elo import elo_update
from protomotions.agents.league.pfsp import PFSPPool, PoolMember

log = logging.getLogger(__name__)


class FullModelLeagueMixin:
    """Self-play league around an agent whose snapshots are full model weights."""

    # Tournament/duck-typing hint: snapshots carry full model weights.
    league_snapshot_kind = "full_model"
    # Provenance tag stored in every snapshot payload.
    snapshot_architecture = "unknown"

    # ------------------------------------------------------------------
    # Initialization (call from subclass __init__ after super().__init__)
    # ------------------------------------------------------------------
    def _init_league(self, adapter, league_cfg: LeagueParams, root_dir) -> None:
        if league_cfg.role != "main":
            raise ValueError(
                f"{type(self).__name__} supports role='main' only (self-play "
                f"vs own snapshots); got role={league_cfg.role!r}"
            )
        self.league_cfg = league_cfg

        self.league_dir = Path(self._resolve_root_dir(root_dir)) / "league"
        self.pool = PFSPPool(
            max_members=league_cfg.max_members,
            weighting=league_cfg.pfsp_weighting,
            min_games=league_cfg.min_games,
            min_decisive_ratio=league_cfg.min_decisive_ratio,
        )

        num_matches = adapter.num_matches
        self.env_member = torch.full(
            (num_matches,), -1, dtype=torch.long, device=self.device
        )
        self.agent_rating = league_cfg.initial_rating
        self.games_since_snapshot = 0
        self.last_snapshot_epoch = 0
        self._snapshot_counter = 0
        self._league_initialized = False
        self._lanes: Optional[OpponentLanes] = None
        self._snapshot_cache: Dict[int, Dict[str, Tensor]] = {}

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def _resolve_root_dir(self, root_dir) -> Path:
        if root_dir is not None:
            return Path(root_dir)
        logger_dir = getattr(getattr(self.fabric, "logger", None), "log_dir", None)
        return Path(logger_dir) if logger_dir else Path(".")

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    def _opponent_action_dim(self) -> int:
        """Width of the lane models' "action" output."""
        return self.env.robot_config.number_of_actions

    def _pre_opponent_policy(self) -> None:
        """Per-step hook before opponent inference (e.g. advance latents)."""

    def _on_opponents_resampled(self, ego_ids: Tensor) -> None:
        """Hook after ``ego_ids`` got new opponents (e.g. resample latents)."""

    def _opponent_obs_td(self, opp_obs: Dict[str, Tensor]):
        obs = self.add_agent_info_to_obs(opp_obs)
        return self.obs_dict_to_tensordict(obs)

    # ------------------------------------------------------------------
    # League lifecycle
    # ------------------------------------------------------------------
    def _ensure_league_initialized(self) -> None:
        if self._league_initialized:
            return
        self._league_initialized = True
        self.league_dir.mkdir(parents=True, exist_ok=True)

        if not self.pool.members:  # not already restored from a checkpoint
            self._restore_league_from_disk()
        if not self.pool.members:
            # Seed with the warm-start weights so first opponents exist.
            self._take_snapshot(reason="seed")

        self._build_lanes()
        all_matches = torch.arange(
            self.env_member.shape[0], device=self.device, dtype=torch.long
        )
        self._resample_opponents(all_matches)

    def _restore_league_from_disk(self) -> None:
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
            len(keep), len(snapshots), self.league_dir,
        )

    def _unwrapped_model(self):
        return getattr(self.model, "module", self.model)

    def _build_lanes(self) -> None:
        base_model = self._unwrapped_model()

        def factory():
            return copy.deepcopy(base_model)

        def assign_full_state(model, state: Dict[str, Tensor]) -> None:
            model.load_state_dict(state, strict=True)

        self._lanes = OpponentLanes(
            model_factory=factory,
            num_lanes=self.league_cfg.num_lanes,
            share_frozen_base_with=None,  # full-weight lanes (small models)
            assign_fn=assign_full_state,
        )

    # ------------------------------------------------------------------
    # Snapshots: full model weights
    # ------------------------------------------------------------------
    def _full_state_cpu(self) -> Dict[str, Tensor]:
        model = self._unwrapped_model()
        return {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    def _take_snapshot(self, reason: str) -> PoolMember:
        path = self.league_dir / f"policy_{self._snapshot_counter}.ckpt"
        torch.save(
            {
                "model": self._full_state_cpu(),
                "epoch": self.current_epoch,
                "rating": self.agent_rating,
                "reason": reason,
                "time": time.time(),
                "architecture": self.snapshot_architecture,
                "robot": getattr(self.env.robot_config, "robot_type", None)
                or getattr(self.env.robot_config, "name", "unknown"),
            },
            path,
        )
        member = self.pool.add(str(path), label=path.stem, rating=self.agent_rating)
        self._snapshot_counter += 1
        self.games_since_snapshot = 0
        self.last_snapshot_epoch = self.current_epoch
        self.pool.reset_stats()

        if len(self.pool.members) >= self.league_cfg.mature_after_members:
            self.pool.weighting = self.league_cfg.pfsp_weighting_mature

        log.info(
            "%s league snapshot %s (%s): pool=%d members, agent Elo=%.0f",
            self.snapshot_architecture, path.name, reason,
            len(self.pool.members), self.agent_rating,
        )
        return member

    def _load_member_snapshot(self, member_id: int) -> Dict[str, Tensor]:
        cached = self._snapshot_cache.get(member_id)
        if cached is not None:
            return cached
        member = self.pool.members[member_id]
        state = torch.load(
            member.checkpoint_path, map_location=self.device, weights_only=False
        )
        payload = state["model"] if "model" in state else state
        payload = {k: v.to(self.device) for k, v in payload.items()}
        self._snapshot_cache[member_id] = payload
        return payload

    # Tournament compatibility: full-weights "adapter" load into the ego.
    def load_adapter_checkpoint(self, checkpoint_path: str) -> None:
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        payload = state["model"] if "model" in state else state
        self._unwrapped_model().load_state_dict(
            {k: v.to(self.device) for k, v in payload.items()}, strict=True
        )
        log.info("Loaded league policy weights from %s", checkpoint_path)

    # ------------------------------------------------------------------
    # Opponent sampling / serving (mirrors the PEFT league)
    # ------------------------------------------------------------------
    def _live_members(self, excluding_envs: Optional[Tensor] = None) -> set:
        assigned = self.env_member
        if excluding_envs is not None:
            mask = torch.ones_like(assigned, dtype=torch.bool)
            mask[excluding_envs] = False
            assigned = assigned[mask]
        return {int(m) for m in assigned.unique().tolist() if m >= 0}

    def _sample_member_capped(self, live: set) -> Optional[PoolMember]:
        member = self.pool.sample()
        if member is None:
            return None
        if member.member_id in live or len(live) < self.league_cfg.num_lanes:
            return member
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
        if not self.pool.members:
            return
        ego_list = ego_ids.tolist()
        if not ego_list:
            return
        # One GPU->CPU sync for the whole batch; the live-member set is then
        # maintained incrementally in Python. (The old per-env
        # env_member.unique().tolist() was ~18% of collection wall-clock at
        # 4096 concurrent matches — a sync per ended match.)
        assignments = self.env_member.tolist()
        counts: Dict[int, int] = {}
        for m in assignments:
            if m >= 0:
                counts[m] = counts.get(m, 0) + 1
        for env in ego_list:
            old = assignments[env]
            if old >= 0:
                counts[old] -= 1
                if counts[old] <= 0:
                    del counts[old]
            live = set(counts.keys())
            member = self._sample_member_capped(live)
            if member is None:
                assignments[env] = old  # unchanged
                if old >= 0:
                    counts[old] = counts.get(old, 0) + 1
                continue
            self._lanes.assign(
                member.member_id,
                self._load_member_snapshot(member.member_id),
                in_use=live,
            )
            assignments[env] = member.member_id
            counts[member.member_id] = counts.get(member.member_id, 0) + 1
        self.env_member = torch.tensor(
            assignments, dtype=torch.long, device=self.device
        )
        self._on_opponents_resampled(ego_ids)

    def _opponent_policy(self, opp_obs: Dict[str, Tensor]) -> Tensor:
        self._ensure_league_initialized()
        self._pre_opponent_policy()
        obs_td = self._opponent_obs_td(opp_obs)

        if self.league_cfg.force_symmetric_inference or not self.pool.members:
            with torch.no_grad():
                out = self.model(obs_td)
            return out["action"]

        return self._lanes.act(obs_td, self.env_member, self._opponent_action_dim())

    # ------------------------------------------------------------------
    # Match accounting
    # ------------------------------------------------------------------
    def _on_matches_ended(
        self, ego_ids: Tensor, win: Tensor, lose: Tensor, draw: Tensor
    ) -> None:
        if not self._league_initialized:
            return
        self.games_since_snapshot += len(ego_ids)

        if self.pool.members:
            # Batch every GPU->CPU transfer: per-element int(tensor[i]) was a
            # device sync per ended match.
            member_ids = self.env_member[ego_ids].tolist()
            wins = win.tolist()
            loses = lose.tolist()
            draws = draw.tolist()
            for member_id, w, l, d in zip(member_ids, wins, loses, draws):
                member = self.pool.members.get(int(member_id))
                if member is None:
                    continue
                w, l, d = int(w), int(l), int(d)
                self.pool.record_result(int(member_id), wins=w, losses=l, draws=d)
                score = 1.0 if w else (0.5 if d else 0.0)
                self.agent_rating, member.rating = elo_update(
                    self.agent_rating, member.rating, score, k=self.league_cfg.elo_k
                )

        self._resample_opponents(ego_ids)

    # ------------------------------------------------------------------
    # Epoch hook: gating, staleness, logging
    # ------------------------------------------------------------------
    def post_epoch_logging(self, training_log_dict: dict):
        if self._league_initialized:
            cfg = self.league_cfg
            pool_avg = self.pool.average_win_rate()
            gated = (
                self.games_since_snapshot >= cfg.gate_min_games
                and pool_avg >= cfg.gate_win_rate
            )
            stale = (
                self.current_epoch - self.last_snapshot_epoch >= cfg.staleness_epochs
            )
            if gated:
                self._take_snapshot(reason="gate")
            elif stale:
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
    # Persistence inside the training checkpoint (mirrors PEFT league)
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
        for mid, m in league.get("members", {}).items():
            if not Path(m["checkpoint_path"]).exists():
                log.warning(
                    "League snapshot missing on resume: %s", m["checkpoint_path"]
                )
                continue
            member = self.pool.add(
                m["checkpoint_path"], label=m["label"], rating=m["rating"]
            )
            member.creation_order = m.get("creation_order", member.creation_order)
            for key, value in m.get("stats", {}).items():
                setattr(member.stats, key, value)
        if self.pool.members:
            self._league_initialized = False  # lanes rebuilt lazily with pool


__all__ = ["FullModelLeagueMixin"]
