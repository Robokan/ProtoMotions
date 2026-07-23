# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""ASE self-play league agent (battle tournament seat for the ASE baseline).

Mirrors :class:`LeagueDiscretePriorPEFTRLFTAgent` (agents/league/agent.py)
with two structural differences:

- **Snapshots are full model state dicts**, not PEFT adapter slices: ASE
  policies are small MLPs with no shared frozen base, so each league snapshot
  is self-contained. Snapshot files stay drop-in compatible with the PEFT
  format ({"model": ..., "epoch", "rating", "reason", "time"}) plus
  provenance keys ("architecture": "ase", "robot").
- **Opponent latents**: ASE policies condition on a skill latent z. The ego
  agent manages its own per-env latents (ASE.setup / update_latents); this
  class keeps a SECOND latent buffer for the opponent half, resampled on the
  same step cadence and whenever an env's opponent is resampled.

Tournament compatibility (BattleTournament): exposes the same surface the
PEFT league agent does — ``load_adapter_checkpoint`` (here: full-weights
load), ``_lanes`` built with a full-state ``assign_fn``, ``_opponent_obs_td``
(injects opponent latents), ``_unwrapped_model``. ASE-vs-ASE exhibitions and
round-robins run unchanged. Known simplification: during tournaments the
skill latents are fixed per bout (the tournament loop does not advance latent
step trackers); fights are still fully functional.

The exploiter role is not supported (single-architecture self-play only).

TODO: unify with agents/league/agent.py via a shared mixin once the PEFT
league is between runs — kept separate for now so the running SOMA league's
code path is untouched.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import Tensor

from protomotions.agents.ase.agent import ASE
from protomotions.agents.ase.config import ASEAgentConfig
from protomotions.agents.league.agent import LeagueParams
from protomotions.agents.league.lanes import OpponentLanes
from protomotions.agents.league.elo import elo_update
from protomotions.agents.league.pfsp import PFSPPool, PoolMember
from protomotions.agents.league.self_play_env import SelfPlayEnvAdapter
from protomotions.agents.utils.step_tracker import StepTracker

log = logging.getLogger(__name__)


@dataclass
class LeagueASEAgentConfig(ASEAgentConfig):
    """ASE agent config with a self-play league around it."""

    _target_: str = "protomotions.agents.league.ase_agent.LeagueASEAgent"
    league: LeagueParams = field(default_factory=LeagueParams)


class LeagueASEAgent(ASE):
    """ASE agent whose environment is one side of a self-play league."""

    config: LeagueASEAgentConfig

    # Tournament/duck-typing hint: snapshots carry full model weights.
    league_snapshot_kind = "full_model"

    def __init__(self, fabric, env, config, root_dir=None):
        adapter = SelfPlayEnvAdapter(env)
        super().__init__(fabric, adapter, config, root_dir=root_dir)

        league_cfg: LeagueParams = config.league
        if league_cfg.role != "main":
            raise ValueError(
                "LeagueASEAgent supports role='main' only (self-play vs own "
                f"snapshots); got role={league_cfg.role!r}"
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

        # Opponent skill latents (ego latents live on the ASE base class).
        self._opp_latents: Optional[Tensor] = None
        self._opp_latent_steps: Optional[StepTracker] = None

        adapter.set_opponent_policy(self._opponent_policy)
        adapter.set_match_end_callback(self._on_matches_ended)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def _resolve_root_dir(self, root_dir) -> Path:
        if root_dir is not None:
            return Path(root_dir)
        logger_dir = getattr(getattr(self.fabric, "logger", None), "log_dir", None)
        return Path(logger_dir) if logger_dir else Path(".")

    # ------------------------------------------------------------------
    # Setup: opponent latent buffers sized to the ego half
    # ------------------------------------------------------------------
    def setup(self):
        super().setup()
        n = self.env_member.shape[0]
        dim = self.config.ase_parameters.latent_dim
        self._opp_latents = torch.zeros((n, dim), dtype=torch.float, device=self.device)
        self._opp_latent_steps = StepTracker(
            n,
            min_steps=self.config.ase_parameters.latent_steps_min,
            max_steps=self.config.ase_parameters.latent_steps_max,
            device=self.device,
        )
        self._reset_opp_latents()

    def _reset_opp_latents(self, env_ids: Optional[Tensor] = None) -> None:
        if self._opp_latents is None:
            return
        if env_ids is None:
            env_ids = torch.arange(self._opp_latents.shape[0], device=self.device)
        if env_ids.numel() == 0:
            return
        self._opp_latents[env_ids] = self.sample_latents(len(env_ids))
        self._opp_latent_steps.reset_steps(env_ids)

    def _update_opp_latents(self) -> None:
        self._opp_latent_steps.advance()
        done = self._opp_latent_steps.done_indices()
        if done.numel() > 0:
            self._reset_opp_latents(done)

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
                "architecture": "ase",
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
            "ASE league snapshot %s (%s): pool=%d members, agent Elo=%.0f",
            path.name, reason, len(self.pool.members), self.agent_rating,
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
        log.info("Loaded ASE policy weights from %s", checkpoint_path)

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
        for env in ego_ids.tolist():
            live = self._live_members(
                excluding_envs=torch.tensor([env], device=self.device)
            )
            member = self._sample_member_capped(live)
            if member is None:
                continue
            self._lanes.assign(
                member.member_id,
                self._load_member_snapshot(member.member_id),
                in_use=live,
            )
            self.env_member[env] = member.member_id
        # New opponent, new skill: resample those envs' opponent latents.
        self._reset_opp_latents(ego_ids)

    def _opponent_obs_td(self, opp_obs: Dict[str, Tensor]):
        obs = self.add_agent_info_to_obs(opp_obs)
        # add_agent_info_to_obs injected the EGO latents; opponents get theirs.
        if self._opp_latents is not None:
            obs["latents"] = self._opp_latents.clone()
        return self.obs_dict_to_tensordict(obs)

    def _opponent_policy(self, opp_obs: Dict[str, Tensor]) -> Tensor:
        self._ensure_league_initialized()
        self._update_opp_latents()
        obs_td = self._opponent_obs_td(opp_obs)

        if self.league_cfg.force_symmetric_inference or not self.pool.members:
            with torch.no_grad():
                out = self.model(obs_td)
            return out["action"]

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

        if self.pool.members:
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


__all__ = ["LeagueASEAgentConfig", "LeagueASEAgent"]
