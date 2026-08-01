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
from protomotions.agents.league import pool_io

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

        # Shared-pool mode (Phase 1): publish to and scan a common directory.
        # getattr: resumed runs deserialize pre-Phase-1 LeagueParams pickles.
        shared = getattr(league_cfg, "shared_pool_dir", None)
        self._shared_pool = shared is not None
        self.league_dir = (
            Path(shared) if shared
            else Path(self._resolve_root_dir(root_dir)) / "league"
        )
        self.run_id = pool_io.new_run_id()  # overwritten on checkpoint resume
        self._own_fingerprint_cache: Optional[str] = None
        self._known_snapshot_paths: set = set()
        self._last_rescan_epoch = -(10**9)
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

    def _snapshot_extra_meta(self) -> dict:
        """Subclass hook: extra provenance keys for snapshots (e.g. the HLC
        league pins its frozen LLC checkpoint here)."""
        return {}

    # ------------------------------------------------------------------
    # Provenance identity of this run
    # ------------------------------------------------------------------
    def _own_robot(self) -> str:
        return (
            getattr(self.league_cfg, "robot_name", None)
            or getattr(self.env.robot_config, "robot_type", None)
            or getattr(self.env.robot_config, "name", None)
            or "unknown"
        )

    def _own_fingerprint(self) -> str:
        if self._own_fingerprint_cache is None:
            self._own_fingerprint_cache = pool_io.state_fingerprint(
                self._unwrapped_model().state_dict()
            )
        return self._own_fingerprint_cache

    def _pool_identity(self) -> dict:
        """Provenance a snapshot must carry to be HOSTED in this run's lanes.

        Default: this run's own robot/architecture/model shapes. A
        cross-morphology league (Phase 3) overrides this — its opponent
        block is a different robot, so the pool hosts THAT robot's
        families and never its own."""
        return {
            "robot": self._own_robot(),
            "architecture": self.snapshot_architecture,
            "fingerprint": self._own_fingerprint(),
        }

    def _host_own_snapshots(self) -> bool:
        """Whether this run's own snapshots join its own opponent pool.
        False in cross-morphology leagues (own snapshots are still
        PUBLISHED for other runs; they just can't fight themselves)."""
        return True

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
        if (
            not any(m.family == "" for m in self.pool.members.values())
            and self._host_own_snapshots()
        ):
            # Seed with the warm-start weights so first OWN opponents exist —
            # a shared pool may already hold other runs' snapshots, but own-
            # family gating needs an own seed to measure against. A
            # cross-morphology league cannot seed itself: until the opponent
            # robot's runs publish, the symmetric fallback serves the live
            # ego policy through the opponent body's LLC.
            self._take_snapshot(reason="seed")

        self._build_lanes()
        all_matches = torch.arange(
            self.env_member.shape[0], device=self.device, dtype=torch.long
        )
        self._resample_opponents(all_matches)

    def _family_quota(self) -> int:
        quota = getattr(self.league_cfg, "family_quota", None)
        return quota if quota else max(4, self.league_cfg.max_members // 3)

    def _classify_snapshot(self, path: Path, meta: dict) -> Optional[str]:
        """Family for a compatible snapshot; None if it must not be hosted.

        Own family is "" — in a private league dir every compatible snapshot
        is own lineage (legacy behavior); in a shared pool only files stamped
        with this run's id are own, the rest join per-family quotas.
        """
        try:
            pool_io.check_compatible(
                meta,
                accept_foreign_rules_era=getattr(
                    self.league_cfg, "accept_foreign_rules_era", False
                ),
                path=path.name,
                **self._pool_identity(),
            )
        except pool_io.SnapshotIncompatible as exc:
            log.info("League pool: not hosting %s (%s)", path.name, exc)
            return None
        if not self._shared_pool:
            return ""
        rid = meta.get("run_id") or pool_io.snapshot_run_id(path)
        return "" if rid == self.run_id else pool_io.family_key(meta)

    def _gate_win_rate_signal(self) -> float:
        """Win rate the snapshot gate compares against gate_win_rate.
        Own-family when this run fights its own lineage; the whole pool in
        a cross-morphology league (every member is the opponent robot)."""
        if self._host_own_snapshots():
            return self.pool.average_win_rate(family="")
        return self.pool.average_win_rate()

    def _restore_league_from_disk(self) -> None:
        entries = []
        for path in self.league_dir.glob("policy_*.ckpt"):
            meta = pool_io.load_snapshot_meta(path)
            if meta is None:
                log.warning("Could not read league snapshot %s", path)
                continue
            family = self._classify_snapshot(path, meta)
            if family is None:
                continue
            entries.append((meta.get("time") or path.stat().st_mtime, path, meta, family))
        if not entries:
            return
        entries.sort(key=lambda e: e[0])  # embedded time, not st_mtime

        quota = self._family_quota()
        kept_per_family: Dict[str, int] = {}
        for _, path, meta, family in reversed(entries):  # newest first
            cap = self.league_cfg.max_members if family == "" else quota
            if kept_per_family.get(family, 0) >= cap:
                continue
            kept_per_family[family] = kept_per_family.get(family, 0) + 1
            rating = float(meta.get("rating", self.league_cfg.initial_rating))
            self.pool.add(str(path), label=path.stem, rating=rating, family=family)
            self._known_snapshot_paths.add(str(path))
        self._snapshot_counter = sum(1 for e in entries if e[3] == "")
        log.info(
            "Restored league: %d snapshots (%d families) from %s",
            len(self.pool.members), len(kept_per_family), self.league_dir,
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
        path = self.league_dir / f"policy_{self.run_id}_{self._snapshot_counter}.ckpt"
        state = self._full_state_cpu()
        payload = pool_io.build_snapshot_meta(
            run_id=self.run_id,
            robot=self._own_robot(),
            architecture=self.snapshot_architecture,
            fingerprint=pool_io.state_fingerprint(state),
            action_dim=self._opponent_action_dim(),
            epoch=self.current_epoch,
            rating=self.agent_rating,
            reason=reason,
            extra=self._snapshot_extra_meta(),
        )
        payload["model"] = state
        pool_io.atomic_save(payload, path)
        member = None
        if self._host_own_snapshots():
            member = self.pool.add(
                str(path), label=path.stem, rating=self.agent_rating, family=""
            )
        self._known_snapshot_paths.add(str(path))
        self._snapshot_counter += 1
        self.games_since_snapshot = 0
        self.last_snapshot_epoch = self.current_epoch
        # Stats answer "how does the CURRENT agent fare" — a new own snapshot
        # changes that reference point. Foreign-family stats survive (their
        # reference point is the same current agent; wiping them re-triggers
        # min_games warmup on every own snapshot — plan Phase 1).
        self.pool.reset_stats(family="")
        self._prune_snapshot_cache()

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
        if isinstance(state, dict) and "model" in state:
            # Last-ditch Phase 0 guard: never assign silently-garbage weights.
            pool_io.check_compatible(
                state,
                accept_foreign_rules_era=getattr(
                    self.league_cfg, "accept_foreign_rules_era", False
                ),
                path=member.checkpoint_path,
                **self._pool_identity(),
            )
        payload = state["model"] if "model" in state else state
        payload = {k: v.to(self.device) for k, v in payload.items()}
        self._snapshot_cache[member_id] = payload
        return payload

    def _prune_snapshot_cache(self) -> None:
        """Drop cached weights for evicted members (member ids only grow)."""
        self._snapshot_cache = {
            k: v for k, v in self._snapshot_cache.items() if k in self.pool.members
        }

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
    # Shared-pool re-scan (Phase 1): ingest other runs' snapshots mid-run
    # ------------------------------------------------------------------
    def _rescan_shared_pool(self) -> None:
        rescan_epochs = getattr(self.league_cfg, "pool_rescan_epochs", 10)
        if self.current_epoch - self._last_rescan_epoch < rescan_epochs:
            return
        self._last_rescan_epoch = self.current_epoch
        quota = self._family_quota()
        for path in sorted(self.league_dir.glob("policy_*.ckpt")):
            key = str(path)
            if key in self._known_snapshot_paths:
                continue
            rid = pool_io.snapshot_run_id(path)
            if rid == self.run_id:
                self._known_snapshot_paths.add(key)
                continue  # own writes join the pool at snapshot time
            meta = pool_io.load_snapshot_meta(path)
            if meta is None:
                continue  # possibly mid-write by another run; retry next scan
            self._known_snapshot_paths.add(key)
            family = self._classify_snapshot(path, meta)
            if family is None or family == "":
                continue
            if self.pool.family_size(family) >= quota:
                # Rolling window per family: newest replaces that family's
                # oldest (informed eviction handles the global cap).
                oldest = min(
                    (m for m in self.pool.members.values() if m.family == family),
                    key=lambda m: m.creation_order,
                )
                del self.pool.members[oldest.member_id]
            self.pool.add(
                key,
                label=path.stem,
                rating=float(meta.get("rating", self.league_cfg.initial_rating)),
                family=family,
            )
            log.info(
                "Shared pool: ingested foreign snapshot %s (family %s, pool=%d)",
                path.name, family, len(self.pool.members),
            )
        self._prune_snapshot_cache()
        # A pool that JUST became non-empty flips _opponent_policy from the
        # symmetric/mechanical fallback to lane serving — matches created
        # before ingestion still hold member -1 and would crash lanes.act.
        # Assign them now (mid-match opponent swap, one-time event).
        if self.pool.members and self._lanes is not None:
            unassigned = (self.env_member < 0).nonzero(as_tuple=False).flatten()
            if len(unassigned) > 0:
                self._resample_opponents(unassigned)

    # ------------------------------------------------------------------
    # Epoch hook: gating, staleness, logging
    # ------------------------------------------------------------------
    def post_epoch_logging(self, training_log_dict: dict):
        if self._league_initialized:
            cfg = self.league_cfg
            if self._shared_pool:
                self._rescan_shared_pool()
            # Gate on OWN-family win rate only: in a shared pool, robot A's
            # league growth must not be gated on beating run B (Phase 1).
            own_avg = self._gate_win_rate_signal()
            pool_avg = self.pool.average_win_rate()
            gated = (
                self.games_since_snapshot >= cfg.gate_min_games
                and own_avg >= cfg.gate_win_rate
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
            training_log_dict["league/own_family_win_rate"] = own_avg
            training_log_dict["league/agent_elo"] = self.agent_rating
            training_log_dict["league/games_since_snapshot"] = float(
                self.games_since_snapshot
            )
            foreign = sum(1 for m in self.pool.members.values() if m.family != "")
            if foreign:
                training_log_dict["league/foreign_members"] = float(foreign)
                foreign_rates = [
                    m.stats.win_rate(self.pool.min_games)
                    for m in self.pool.members.values() if m.family != ""
                ]
                training_log_dict["league/foreign_avg_win_rate"] = sum(
                    foreign_rates
                ) / len(foreign_rates)
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
            "run_id": self.run_id,
            "snapshot_counter": self._snapshot_counter,
            "last_snapshot_epoch": self.last_snapshot_epoch,
            "games_since_snapshot": self.games_since_snapshot,
            "members": {
                mid: {
                    "checkpoint_path": m.checkpoint_path,
                    "label": m.label,
                    "rating": m.rating,
                    "creation_order": m.creation_order,
                    "family": m.family,
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
        self.run_id = league.get("run_id", self.run_id)
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
                m["checkpoint_path"], label=m["label"], rating=m["rating"],
                family=m.get("family", ""),
            )
            member.creation_order = m.get("creation_order", member.creation_order)
            for key, value in m.get("stats", {}).items():
                setattr(member.stats, key, value)
            self._known_snapshot_paths.add(m["checkpoint_path"])
        if self.pool.members:
            self._league_initialized = False  # lanes rebuilt lazily with pool


__all__ = ["FullModelLeagueMixin"]
