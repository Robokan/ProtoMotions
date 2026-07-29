# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""ASE self-play league agent (battle tournament seat for the ASE baseline).

League orchestration (pool, snapshots, lanes, Elo, checkpoint persistence)
lives in :class:`FullModelLeagueMixin` (full_model_league.py), shared with
the paper-faithful HLC league (ase_hlc_agent.py). This class adds the
ASE-specific parts:

- **Snapshots are full model state dicts**, not PEFT adapter slices: ASE
  policies are small MLPs with no shared frozen base, so each league snapshot
  is self-contained (provenance tag "architecture": "ase").
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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import Tensor

from protomotions.agents.ase.agent import ASE
from protomotions.agents.ase.config import ASEAgentConfig
from protomotions.agents.league.agent import LeagueParams
from protomotions.agents.league.full_model_league import FullModelLeagueMixin
from protomotions.agents.league.self_play_env import SelfPlayEnvAdapter
from protomotions.agents.utils.step_tracker import StepTracker

log = logging.getLogger(__name__)


@dataclass
class LeagueASEAgentConfig(ASEAgentConfig):
    """ASE agent config with a self-play league around it."""

    _target_: str = "protomotions.agents.league.ase_agent.LeagueASEAgent"
    league: LeagueParams = field(default_factory=LeagueParams)


class LeagueASEAgent(FullModelLeagueMixin, ASE):
    """ASE agent whose environment is one side of a self-play league."""

    config: LeagueASEAgentConfig

    snapshot_architecture = "ase"

    def __init__(self, fabric, env, config, root_dir=None):
        adapter = SelfPlayEnvAdapter(env)
        super().__init__(fabric, adapter, config, root_dir=root_dir)
        self._init_league(adapter, config.league, root_dir)

        # Opponent skill latents (ego latents live on the ASE base class).
        self._opp_latents: Optional[Tensor] = None
        self._opp_latent_steps: Optional[StepTracker] = None

        adapter.set_opponent_policy(self._opponent_policy)
        adapter.set_match_end_callback(self._on_matches_ended)

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
    # League mixin hooks: weave opponent latents into the shared flow
    # ------------------------------------------------------------------
    def _on_opponents_resampled(self, ego_ids: Tensor) -> None:
        # New opponent, new skill: resample those envs' opponent latents.
        self._reset_opp_latents(ego_ids)

    def _pre_opponent_policy(self) -> None:
        self._update_opp_latents()

    def _opponent_obs_td(self, opp_obs: Dict[str, Tensor]):
        obs = self.add_agent_info_to_obs(opp_obs)
        # add_agent_info_to_obs injected the EGO latents; opponents get theirs.
        if self._opp_latents is not None:
            obs["latents"] = self._opp_latents.clone()
        return self.obs_dict_to_tensordict(obs)


__all__ = ["LeagueASEAgentConfig", "LeagueASEAgent"]
