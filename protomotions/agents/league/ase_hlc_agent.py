# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Paper-faithful ASE battle league: frozen LLC + trainable high-level policy.

Two-level control exactly as in the ASE paper (Peng et al. 2022), mirroring
the GPC league's frozen-prior structure:

- **LLC (stage 1, frozen)**: the latent-conditioned low-level controller from
  an ``examples/experiments/ase/mlp.py`` pretrain. Loaded via
  ``pretrained_modules["llc"]`` (module_path="actor"), eval mode,
  requires_grad False, NO optimizer params. It lives on the agent — NOT on
  the trainable model — so league snapshots, opponent lanes, and training
  checkpoints contain only HLC weights, and a deeper/longer-trained LLC can
  be swapped under an existing HLC via ``--llc-checkpoint``.
- **HLC (stage 2, trained here)**: a small PPO actor-critic over battle task
  obs + self obs whose 64-dim continuous action IS the skill latent z. Before
  the LLC consumes z it is projected onto the unit hypersphere
  (``F.normalize``, the ``ASE.sample_latents`` convention).
- **Action flow**: each control step, HLC(obs) -> z; LLC(z, max_coords_obs)
  -> joint actions. Both halves of every match run through the same frozen
  LLC (:class:`HLCSelfPlayEnvAdapter` translates ego and opponent latents).
  PPO trains purely in latent-action space; the env adapter owns the
  translation, so the training loop and tournament code stay unchanged.
- **League**: :class:`FullModelLeagueMixin` machinery; snapshots are the
  full (small) HLC model state ("architecture": "ase_hlc"). No AMP
  discriminator or MI objective in stage 2 — style is guaranteed by the
  frozen LLC, matching both the paper and the GPC league.

Tournament note: BattleTournament's ``action_hold`` holds the HLC *latent*
between decode frames while the LLC still runs every step with fresh
proprioception — which is precisely the paper's slow-HLC/fast-LLC split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict

import torch
from tensordict import TensorDict
from torch import Tensor

from protomotions.agents.fine_tuning.agent import FineTuningAgent
from protomotions.agents.fine_tuning.config import FineTuningAgentConfig
from protomotions.agents.league.agent import LeagueParams
from protomotions.agents.league.full_model_league import FullModelLeagueMixin
from protomotions.agents.league.self_play_env import SelfPlayEnvAdapter
from protomotions.agents.ppo.agent import PPO

log = logging.getLogger(__name__)


class HLCSelfPlayEnvAdapter(SelfPlayEnvAdapter):
    """Self-play adapter whose policies emit skill latents, not joint actions.

    Both the ego action passed to :meth:`step` and the opponent policy's
    output are latent vectors; a translator callback (the agent's frozen LLC)
    maps (obs, latents) -> joint actions for each half before the inner
    BattleEnv steps.
    """

    def __init__(self, env):
        super().__init__(env)
        self._latent_translator = None

    def set_latent_translator(self, translator) -> None:
        """translator(obs: dict[str, Tensor], latents: Tensor) -> joint actions."""
        self._latent_translator = translator

    def step(self, latent_action: Tensor):
        if self._opponent_policy is None:
            raise RuntimeError(
                "HLCSelfPlayEnvAdapter.step called before set_opponent_policy()"
            )
        if self._latent_translator is None:
            raise RuntimeError(
                "HLCSelfPlayEnvAdapter.step called before set_latent_translator()"
            )
        with torch.no_grad():
            ego_action = self._latent_translator(self.get_obs(), latent_action)
            opp_obs = self.opponent_obs()
            opp_latents = self._opponent_policy(opp_obs)
            opp_action = self._latent_translator(opp_obs, opp_latents)
        return self._step_full(torch.cat([ego_action, opp_action], dim=0))


@dataclass
class HLCParams:
    """High-level controller parameters."""

    latent_dim: int = 64
    llc_deterministic: bool = True  # feed the LLC's mean action to the sim


@dataclass
class LeagueASEHLCAgentConfig(FineTuningAgentConfig):
    """PPO-over-latents agent config with a self-play league around it."""

    _target_: str = "protomotions.agents.league.ase_hlc_agent.LeagueASEHLCAgent"
    # ``model`` stays the parent's BaseModelConfig field; experiments must
    # pass a PPOModelConfig (PPOActorConfig has no default mu_key, so it
    # cannot serve as a dataclass default factory).
    league: LeagueParams = field(default_factory=LeagueParams)
    hlc: HLCParams = field(default_factory=HLCParams)


class LeagueASEHLCAgent(FullModelLeagueMixin, FineTuningAgent):
    """High-level latent policy trained by PPO self-play over a frozen LLC."""

    config: LeagueASEHLCAgentConfig

    snapshot_architecture = "ase_hlc"

    def __init__(self, fabric, env, config, root_dir=None):
        adapter = HLCSelfPlayEnvAdapter(env)
        super().__init__(fabric, adapter, config, root_dir=root_dir)
        self._init_league(adapter, config.league, root_dir)
        self._llc = None  # frozen PPOActor, captured in _post_create_model_hook

        adapter.set_latent_translator(self._latents_to_joint_actions)
        adapter.set_opponent_policy(self._opponent_policy)
        adapter.set_match_end_callback(self._on_matches_ended)

    # ------------------------------------------------------------------
    # Model: plain PPO actor-critic (the HLC). The LLC stays OUTSIDE the
    # model so state_dict/snapshots/lanes carry HLC weights only.
    # ------------------------------------------------------------------
    def create_model(self):
        return PPO.create_model(self)

    def _post_create_model_hook(self) -> None:
        llc = self.pretrained.get("llc")
        if llc is None:
            raise ValueError(
                f"{type(self).__name__} requires config.pretrained_modules['llc'] "
                "(the frozen low-level controller from an ASE pretrain, "
                "module_path='actor')."
            )
        self._llc = llc  # frozen + eval via PretrainedModelConfig(freeze=True)

    # ------------------------------------------------------------------
    # Latent -> joint action translation through the frozen LLC
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _latents_to_joint_actions(
        self, obs: Dict[str, Tensor], latents: Tensor
    ) -> Tensor:
        # Project onto the unit hypersphere — the latent-space convention the
        # LLC was trained under (ASE.sample_latents).
        z = torch.nn.functional.normalize(latents, dim=-1)
        td = TensorDict(
            {"max_coords_obs": obs["max_coords_obs"], "latents": z},
            batch_size=z.shape[0],
        )
        td = self._llc(td)
        key = "mean_action" if self.config.hlc.llc_deterministic else "action"
        return td[key]

    # ------------------------------------------------------------------
    # League mixin hooks
    # ------------------------------------------------------------------
    def _opponent_action_dim(self) -> int:
        # Lanes serve HLC replicas: their "action" output is the latent.
        return self.config.hlc.latent_dim


__all__ = [
    "HLCSelfPlayEnvAdapter",
    "HLCParams",
    "LeagueASEHLCAgentConfig",
    "LeagueASEHLCAgent",
]
